from __future__ import annotations

from typing import Any, Literal, Union

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import GaussianMixin  # noqa
from skrl.models.torch import Model
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa
from skrl.utils.transformer_utils.torch import TransformerNetwork
from skrl.utils.transformer_utils.torch.utils import get_num_units

class TransformerGaussian(GaussianMixin, Model):
    def __init__(self,
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
                 initial_log_std: float = 0,
                 fixed_log_std: bool = False,
                 network: list[dict[str, Any]] = [],
                 output: str | list[str] = "",
                 return_source: bool = False,
                #  model_params={}, # This includes pooling_method, tokenization_method, and groups (if used for the tokenization)
                #  pooling_method: Literal['mean', 'max', 'first', 'CLS', 'attn_mean'] = 'mean',
                #  tokenization_method: Literal['all', 'single', 'bin', 'groups',] = 'all',
                #  groups: Union[list[int], None] = None
                 ):
        '''
        tokenization_method: 'all' puts the entire state into a single token, 'single' puts each input into a token, 'bin' uses predefined bins to create tokens for each input, 'groups' groups parts of the input together into a single token
        '''
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
        self.model_params = network[0]
        # self.inp_type = network[0]['input']
        inp_size = get_num_units(self.model_params['input'], self.num_observations, self.num_states, self.num_actions)
        out_size = get_num_units(output, self.num_observations, self.num_states, self.num_actions)
        self.net = TransformerNetwork(inp_size, self.model_params)
        self.output_layer = nn.Linear(self.model_params['d_model'], out_size * self.model_params.get('num_pred_acts', 1))

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std), dtype=torch.float32), requires_grad=not fixed_log_std
        )
    
    def compute(self, inputs, role=""):
        if self.model_params['input'] == 'OBSERVATIONS':
            inp = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        elif self.model_params['input'] == 'STATES':
            inp = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        elif self.model_params['input'] == 'ACTIONS':
            inp = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net(inp)
        output = self.output_layer(output)
        return output, {"log_std": self.log_std_parameter}