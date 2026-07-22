import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin


# define the model
class LSTMDeterministic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
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
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=self.num_observations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # (batch, sequence, features)
        )

        self.net = nn.Sequential(
            nn.Linear(self.hidden_size + self.num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

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

        # critic models are only used during training
        rnn_input = observations.view(
            -1, self.sequence_length, observations.shape[-1]
        )  # (N, L, Hin): N=batch_size, L=sequence_length

        hidden_states = hidden_states.view(
            self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
        )  # (D * num_layers, N, L, Hout)
        cell_states = cell_states.view(
            self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
        )  # (D * num_layers, N, L, Hcell)
        # get the hidden/cell states corresponding to the initial sequence
        sequence_index = 1 if role == "target_critic" else 0  # target networks act on the next state of the environment
        hidden_states = hidden_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hout)
        cell_states = cell_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hcell)

        # reset the RNN state in the middle of a sequence
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            indexes = (
                [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
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

        return self.net(torch.cat([rnn_output, inputs["taken_actions"]], dim=1)), {
            "rnn": [rnn_states[0], rnn_states[1]]
        }
