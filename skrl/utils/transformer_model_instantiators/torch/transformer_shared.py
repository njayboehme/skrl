from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import Model  # noqa
from skrl.models.torch import (  # noqa
    DeterministicMixin,
    GaussianMixin,
)
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa
from skrl.utils.transformer_utils.torch import TransformerNetwork
from skrl.utils.transformer_utils.torch.utils import get_num_units


class TransformerShared(GaussianMixin,DeterministicMixin, Model):
    def __init__(self,*,
                 observation_space: gymnasium.Space | None = None,
                 state_space: gymnasium.Space | None = None,
                 action_space: gymnasium.Space | None = None,
                 device: str | torch.device | None = None,
                 structure: list[str] = ["GaussianMixin", "DeterministicMixin"],
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
        if structure[0] == 'TransformerGaussian':
            GaussianMixin.__init__(
                self,
                clip_actions=False,
                clip_mean_actions=False,
                clip_log_std=True,
                min_log_std=-20.0,
                max_log_std=2.0,
                reduction="sum",
                role="policy",
            )
        if structure[1] == 'TransformerDeterministic':
            DeterministicMixin.__init__(self, clip_actions=False, role="value")

        # self.net_container = nn.Sequential(
        #     nn.LazyLinear(out_features=256),
        #     nn.ELU(),
        #     nn.LazyLinear(out_features=128),
        #     nn.ELU(),
        #     nn.LazyLinear(out_features=64),
        #     nn.ELU(),
        # )
        model_params = parameters[0]['network'][0]['model_params']
        # model_params = network[0]['model_params']
        inp_size = get_num_units(parameters[0]['network'][0]['input'], self.num_observations, self.num_states, self.num_actions)
        self.net_container = TransformerNetwork(inp_size, model_params)
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0, dtype=torch.float32), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

        if not single_forward_pass:
            self._shared_output = None

    def act(self, inputs, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
    
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