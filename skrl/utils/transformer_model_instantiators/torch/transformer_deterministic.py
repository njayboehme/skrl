from __future__ import annotations

from typing import Any

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import DeterministicMixin  # noqa
from skrl.models.torch import Model
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa
from skrl.utils.transformer_utils.torch import TransformerNetwork
from skrl.utils.transformer_utils.torch.utils import get_num_units


def deterministic_model(
    *,
    observation_space: gymnasium.Space | None = None,
    state_space: gymnasium.Space | None = None,
    action_space: gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    clip_actions: bool = False,
    network: list[dict[str, Any]] = [],
    output: str | list[str] = "",
    return_source: bool = False,
) -> Model | str:
    """Instantiate a :class:`~skrl.models.torch.deterministic.DeterministicMixin`-based model.

    :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
    :param state_space: State space. The ``num_states`` property will contain the size of the space.
    :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
    :param device: Data allocation and computation device. If not specified, the default device will be used.
    :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
    :param network: Network definition.
    :param output: Output expression.
    :param return_source: Whether to return the source string containing the model class used to
        instantiate the model rather than the model instance.

    :return: Deterministic model instance or definition source (if ``return_source`` is True).
    """
    # parse model definition
    containers, output = generate_containers(network, output, embed_output=True, indent=1)

    # network definitions
    networks = []
    forward: list[str] = []
    for container in containers:
        networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # process output
    if output["modules"]:
        networks.append(f'self.output_layer = {output["modules"][0]}')
        forward.append(f'output = self.output_layer({container["name"]})')
    if output["output"]:
        forward.append(f'output = {output["output"]}')
    else:
        forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # build substitutions and indent content
    networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

    template = f"""class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device=None, clip_actions=False, role=""):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role=role)

        {networks}

    def compute(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return output, {{}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["DeterministicModel"](
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
    )


class TransformerDeterministic(DeterministicMixin, Model):
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
        DeterministicMixin.__init__(
            self,
            clip_actions=clip_actions,
            role=role,
        )

        model_params = network[0]['model_params']
        self.inp_type = network[0]['input']
        inp_size = get_num_units(self.inp_type, self.num_observations, self.num_states, self.num_actions)
        out_size = get_num_units(output, self.num_observations, self.num_states, self.num_actions)
        self.net = TransformerNetwork(inp_size, out_size, model_params)

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std), dtype=torch.float32), requires_grad=not fixed_log_std
        )
    
    def compute(self, inputs, role=""):
        if self.inp_type == 'OBSERVATIONS':
            inp = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        elif self.inp_type == 'STATES':
            inp = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        elif self.inp_type == 'ACTIONS':
            inp = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net(inp)
        return output, {"log_std": self.log_std_parameter}
