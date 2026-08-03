from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

class MLPShared(GaussianMixin,DeterministicMixin, Model):
    def __init__(self, observation_space: gymnasium.Space | None = None,
                state_space: gymnasium.Space | None = None,
                action_space: gymnasium.Space | None = None,
                device: str | torch.device | None = None,
                structure: list[str] = ["MLPGaussianMixin", "DeterministicMixin"],
                roles: list[str] = [],
                parameters: list[dict[str, Any]] = [],
                single_forward_pass: bool = True,
                return_source: bool = False,):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        if structure[0] == 'MLPGaussian':
            params_0 = parameters[0]
            GaussianMixin.__init__(
                self,
                clip_actions=params_0.get('clip_actions', False),
                clip_mean_actions=params_0.get('clip_mean_actions', False),
                clip_log_std=params_0.get('clip_log_std', True),
                min_log_std=params_0.get('min_log_std', -20.0),
                max_log_std=params_0.get('max_log_std', 0.0),
                reduction="sum",
                role="policy",
            )
        if structure[1] == 'DeterministicMixin':
            params_1 = parameters[1]
            DeterministicMixin.__init__(self, clip_actions=params_1.get('clip_actions', False), role="value")
        net = params_0.get('network', {})[0]
        act = net.get('activations', 'elu')
        self.net_container = nn.Sequential()
        for layer_size in net.get('layers', []):
            self.net_container.append(nn.LazyLinear(layer_size))
            if act == 'elu':
                a = nn.ELU()
            elif act == 'relu':
                a = nn.ReLU()
            self.net_container.append(a)

        if params_0.get('residual', None) is not None:
            residual_params = params_0.get('residual')
            self.residual = nn.Sequential()
            for layer_size in residual_params['network'][0].get('layers', []):
                self.residual.append(nn.LazyLinear(layer_size))
                if act == 'elu':
                    a = nn.ELU()
                elif act == 'relu':
                    a = nn.ReLU()
                self.net_container.append(a)
            self.residual.append(nn.LazyLinear(out_features=self.num_actions))
            self.residual_log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=residual_params.get('initial_log_std', 0.0), dtype=torch.float32), requires_grad=True)

        self.action_chunk_size = params_0.get('action_chunk_size', 1)
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions * self.action_chunk_size)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions * self.action_chunk_size,), fill_value=params_0.get('initial_log_std', 0.0), dtype=torch.float32), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

    def act(self, inputs, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        elif role == "residual":
            return GaussianMixin.act(self, inputs, role=role)
    
    def compute(self, inputs, role=""):
        if role == "policy":
            observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
            states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            net = self.net_container(observations)
            self._shared_output = net
            output = self.policy_layer(net)
            return output, {"log_std": self.log_std_parameter}
        elif role == "value":
            if self._shared_output is None:
                observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
                states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                net = self.net_container(observations)
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}
        elif role == 'residual':
            observations = inputs.get("observations")
            output = self.residual(observations)
            # # TODO: For a shared output, do I run the observation through the base network?
            # # TODO: ^I don't think I need to because if self_shared_output is None then the value network will just use the shared backbone.
            # net = self.net_container(observations[:, :-self.num_actions])
            # self._shared_output = net
            return output, {"log_std": self.residual_log_std_parameter}
