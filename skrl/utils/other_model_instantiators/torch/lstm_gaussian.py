import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class LSTMGaussian(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        num_envs=1,
        num_layers=1,
        hidden_size=64,
        sequence_length=10,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

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

        if network_params.get('action_pred_type', None) is not None:
            self.action_hids = nn.Parameter(torch.full(size=(network_params['num_pred_acts'], network_params.get('hidden_size', 64)), fill_value=0.0, dtype=torch.float32), requires_grad=True)
        
        self.net = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Tanh(),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions * network_params.get('num_pred_acts', 1)))

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

    def compute(self, inputs, role):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = observations.view(
                -1, self.sequence_length, observations.shape[-1]
            )  # (N, L, Hin): N=batch_size, L=sequence_length
            if hasattr(self, 'action_hids'):
                rnn_input = torch.cat((rnn_input, self.action_hids.expand(rnn_input.shape[0], self.action_hids.shape[0], self.action_hids.shape[1])), dim=1)
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(
                self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
            )  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
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
            if hasattr(self, 'action_hids'):
                rnn_input = torch.cat((rnn_input, self.action_hids.expand(rnn_input.shape[0], self.action_hids.shape[0], self.action_hids.shape[1])), dim=1)
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {"log_std": self.log_std_parameter, "rnn": [rnn_states[0], rnn_states[1]]}
