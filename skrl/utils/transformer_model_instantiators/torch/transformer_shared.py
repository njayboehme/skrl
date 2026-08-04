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
                 structure: list[str] = ["TransformerGaussian", "TransformerDeterministic"],
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
        if structure[1] == 'TransformerDeterministic':
            params_1 = parameters[1]
            DeterministicMixin.__init__(self, clip_actions=params_1.get('clip_actions', False), role="value")

        model_params = parameters[0]['network'][0]
        model_params['action_chunk_size'] = params_0.get('action_chunk_size', 1)
        model_params['action_pred_type'] = params_0.get('action_pred_type', None)
        self.action_chunk_size = model_params.get('action_chunk_size', 1)
        self.action_pred_type = model_params.get('action_pred_type', None)
        out_chunk = self.action_chunk_size if not self.action_pred_type else 1
        inp_size = get_num_units(model_params['input'], self.num_observations, self.num_states, self.num_actions)
        self.net_container = TransformerNetwork(inp_size, model_params, shared=True)
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions * out_chunk)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions * self.action_chunk_size,), fill_value=0.0, dtype=torch.float32), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

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
            output = self.policy_layer(net)
            # If there is a different method than just a larger policy layer
            if self.action_pred_type:
                output = output.flatten(start_dim=1, end_dim=-1)
            return output, {"log_std": self.log_std_parameter}
        elif role == "value":
            state_token = self.net_container.get_state_token()
            if state_token is None:
                observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
                states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                net = self.net_container(observations)
                shared_output = net
            else:
                shared_output = state_token
            output = self.value_layer(shared_output)
            return output, {}