from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

class MLPGaussian(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        role="",
        action_chunk_size=1,
        initial_log_std: float = 0,
        fixed_log_std: bool = False,
        network: list[dict[str, Any]] = [],
        output: str | list[str] = "",
        return_source: bool = False,
        residual=None,
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
            role=role,
        )

        self.action_chunk_size = action_chunk_size

        net = network[0]
        act = net.get('activations', 'elu')
        self.net_container = nn.Sequential()
        for layer_size in net.get('layers', []):
            self.net_container.append(nn.LazyLinear(layer_size))
            if act == 'elu':
                a = nn.ELU()
            elif act == 'relu':
                a = nn.ReLU()
            self.net_container.append(a)
        self.net_container.append(nn.LazyLinear(self.num_actions * action_chunk_size))

        self.log_std_parameter = nn.Parameter(
                torch.full(size=(self.num_actions * action_chunk_size,), fill_value=initial_log_std, dtype=torch.float32), requires_grad=True
            )

        if residual is not None:
            residual_params = residual['network'][0]
            act = residual_params.get('activations', 'elu')
            self.residual = nn.Sequential()
            for layer_size in residual_params.get('layers', []):
                self.residual.append(nn.LazyLinear(layer_size))
                if act == 'elu':
                    a = nn.ELU()
                elif act == 'relu':
                    a = nn.ReLU()
                self.residual.append(a)
            self.residual.append(nn.LazyLinear(out_features=self.num_actions))
            self.residual_action_scale = 1.0
            if residual.get('action_scale', None) is not None:
                self.residual.append(nn.Tanh())
                scale = residual['action_scale']
                if isinstance(scale, (int, float)):
                    self.residual_action_scale = scale
                elif scale == 'learn_vector':
                    self.residual_action_scale = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0, dtype=torch.float32), requires_grad=True)
                elif scale == 'learn_scale':
                    self.residual_action_scale = nn.Parameter(torch.full(size=(1,), fill_value=0.0, dtype=torch.float32), requires_grad=True)
            self.residual_log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=residual.get('initial_log_std', 0.0), dtype=torch.float32), requires_grad=True)

    def compute(self, inputs, role=""):
        if role == 'policy':
            observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
            states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            output = self.net_container(observations)
            return output, {"log_std": self.log_std_parameter}
        else:
            output = self.residual(inputs.get("observations")) * self.residual_action_scale
            return output, {"log_std": self.residual_log_std_parameter}