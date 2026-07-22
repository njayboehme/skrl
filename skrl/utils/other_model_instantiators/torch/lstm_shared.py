from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# define the model
class LSTMShared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        structure: list[str] = ["LSTMGaussian", "LSTMDeterministic"],
        roles: list[str] = [],
        parameters: list[dict[str, Any]] = [],
        single_forward_pass: bool = True,
        return_source: bool = False,
        # num_envs=1,
        # num_layers=1,
        # hidden_size=64,
        # sequence_length=1,
        # observation_space,
        # state_space,
        # action_space,
        # device,
        # clip_actions=False,
        # clip_mean_actions=False,
        # clip_log_std=True,
        # min_log_std=-20,
        # max_log_std=2,
        # reduction="sum",
        # num_envs=1,
        # num_layers=1,
        # hidden_size=64,
        # sequence_length=10,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        if structure[0] == 'LSTMGaussian':
            params_0 = parameters[0]
            GaussianMixin.__init__(
                self,
                clip_actions=params_0.get('clip_actions', False),
                clip_mean_actions=params_0.get('clip_mean_actions', False),
                clip_log_std=params_0.get('clip_log_std', True),
                min_log_std=params_0.get('min_log_std', -20.0),
                max_log_std=params_0.get('max_log_std', 2.0),
                reduction=params_0.get('reduction', 'sum'),
                role="policy",
            )
        if structure[1] == 'LSTMDeterministic':
            params_1 = parameters[1]
            DeterministicMixin.__init__(self, clip_actions=params_1.get('clip_actions', False), role="value")

        self.num_envs = parameters[-1]
        network_params = parameters[0]['network'][0]
        self.num_layers = network_params.get('num_layers', 1)
        self.hidden_size = network_params.get('hidden_size', 64)  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = network_params.get('sequence_length', 1)

        self.lstm = nn.LSTM(
            input_size=self.num_observations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # (batch, sequence, features)
        )

        if not single_forward_pass:
            self._shared_output = None
            self._rnn_states = None
        
        self.action_chunk_size = params_0.get('action_chunk_size', 1)
        if network_params.get('action_pred_type', None) is not None:
            self.action_hids = nn.Parameter(torch.full(size=(self.action_chunk_size, network_params.get('hidden_size', 64)), fill_value=0.0, dtype=torch.float32), requires_grad=True)
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions * self.action_chunk_size)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions * self.action_chunk_size,), fill_value=0.0, dtype=torch.float32), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden states (D ∗ num_layers, N, Hout)
                    (self.num_layers, self.num_envs, self.hidden_size),  # cell states  (D ∗ num_layers, N, Hcell)
                ],
            }
        }

    def act(self, inputs, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)

    def compute(self, inputs, role):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        if inputs.get('rnn', None) is None:
            d = self.get_specification()
            layers = d['rnn']['sizes'][0][0]
            hid_size = d['rnn']['sizes'][0][-1]
            # (# layers x B x L)
            hidden_states = torch.zeros((layers, observations.shape[0], hid_size), device=self.device)
            cell_states = torch.zeros((layers, observations.shape[0], hid_size), device=self.device)
            seq_len = 1
        else:
            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]
            seq_len = self.sequence_length

        # training
        if role == 'policy': 
            if self.training:
                rnn_input = observations.view(
                    -1, seq_len, observations.shape[-1]
                )  # (N, L, Hin): N=batch_size, L=sequence_length
                # Add the action states
                if hasattr(self, 'action_hids'):
                    rnn_input = torch.cat((rnn_input, self.action_hids.expand(rnn_input.shape[0], self.action_hids.shape[0], self.action_hids.shape[1])), dim=1)
                hidden_states = hidden_states.view(
                    self.num_layers, -1, seq_len, hidden_states.shape[-1]
                )  # (D * num_layers, N, L, Hout)
                cell_states = cell_states.view(
                    self.num_layers, -1, seq_len, cell_states.shape[-1]
                )  # (D * num_layers, N, L, Hcell)
                # get the hidden/cell states corresponding to the initial sequence
                hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
                cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

                # reset the RNN state in the middle of a sequence
                if terminated is not None and torch.any(terminated):
                    rnn_outputs = []
                    terminated = terminated.view(-1, seq_len)
                    indexes = (
                        [0]
                        + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                        + [seq_len]
                    )

                    for i in range(len(indexes) - 1):
                        i0, i1 = indexes[i], indexes[i + 1]
                        rnn_output, (hidden_states, cell_states) = self.lstm(
                            rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                        )
                        hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                        cell_states[:, (terminated[:, i1 - 1]), :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_states = (hidden_states, cell_states)
                    rnn_output = torch.cat(rnn_outputs, dim=1)
                # no need to reset the RNN state in the sequence
                else:
                    rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
            # rollout
            else:
                rnn_input = observations.view(-1, 1, observations.shape[-1])  # (N, L, Hin): N=num_envs, L=1
                # Add the action states
                if hasattr(self, 'action_hids'):
                    rnn_input = torch.cat((rnn_input, self.action_hids.expand(rnn_input.shape[0], self.action_hids.shape[0], self.action_hids.shape[1])), dim=1)
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

            # flatten the RNN output
            rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)
            self._shared_output = rnn_output
            self._rnn_states = rnn_states
            output = self.policy_layer(rnn_output)
            return output, {"log_std": self.log_std_parameter, "rnn": [rnn_states[0], rnn_states[1]]}
        elif role == 'value':
            if self._shared_output is None:
                # critic models are only used during training
                rnn_input = observations.view(
                    -1, seq_len, observations.shape[-1]
                )  # (N, L, Hin): N=batch_size, L=sequence_length

                hidden_states = hidden_states.view(
                    self.num_layers, -1, seq_len, hidden_states.shape[-1]
                )  # (D * num_layers, N, L, Hout)
                cell_states = cell_states.view(
                    self.num_layers, -1, seq_len, cell_states.shape[-1]
                )  # (D * num_layers, N, L, Hcell)
                # get the hidden/cell states corresponding to the initial sequence
                sequence_index = 1 if role == "target_critic" else 0  # target networks act on the next state of the environment
                hidden_states = hidden_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hout)
                cell_states = cell_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hcell)

                # reset the RNN state in the middle of a sequence
                if terminated is not None and torch.any(terminated):
                    rnn_outputs = []
                    terminated = terminated.view(-1, seq_len)
                    indexes = (
                        [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [seq_len]
                    )

                    for i in range(len(indexes) - 1):
                        i0, i1 = indexes[i], indexes[i + 1]
                        rnn_output, (hidden_states, cell_states) = self.lstm(
                            rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                        )
                        hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                        cell_states[:, (terminated[:, i1 - 1]), :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_states = (hidden_states, cell_states)
                    rnn_output = torch.cat(rnn_outputs, dim=1)
                # no need to reset the RNN state in the sequence
                else:
                    rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

                # flatten the RNN output
                rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)
                shared_output = rnn_output
            else:
                shared_output = self._shared_output
                rnn_states = self._rnn_states
            
            self._shared_output = None
            self._rnn_states = None
            # return self.value_layer(torch.cat([shared_output, inputs["taken_actions"]], dim=1)), {
            #             "rnn": [rnn_states[0], rnn_states[1]]
            #         }
            return self.value_layer(shared_output), {
                            "rnn": [rnn_states[0], rnn_states[1]]
                        }

