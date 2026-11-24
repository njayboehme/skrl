from __future__ import annotations

from typing import Any

import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .mlp_gaussian_chunked import ChunkedGaussianMixin, build_mlp, build_residual


class MLPShared(ChunkedGaussianMixin, DeterministicMixin, Model):
    """Shared-backbone MLP: chunked Gaussian policy head + deterministic value head (+ optional residual head).

    When the value role is called right after the policy role *on the same inputs*, the backbone output of the
    policy call is reused (single forward pass). The cache is consumed by the value call; call
    :meth:`reset_shared_cache` before a value call on different inputs.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        structure: list[str] = ["MLPGaussian", "DeterministicMixin"],
        roles: list[str] = [],
        parameters: list[dict[str, Any]] = [],
        single_forward_pass: bool = True,
        return_source: bool = False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        if len(structure) != 2 or not structure[0].lower().startswith("mlpgaussian") or structure[1].lower() != "deterministicmixin":
            raise ValueError(f"MLPShared expects structure ['MLPGaussian', 'DeterministicMixin'], got {structure}")
        params_0, params_1 = parameters[0], parameters[1]
        if params_0.get("reduction", "sum") != "sum":
            raise ValueError("Chunked Gaussian policies require reduction='sum'")

        GaussianMixin.__init__(
            self,
            clip_actions=params_0.get("clip_actions", False),
            clip_mean_actions=params_0.get("clip_mean_actions", False),
            clip_log_std=params_0.get("clip_log_std", True),
            min_log_std=params_0.get("min_log_std", -20.0),
            max_log_std=params_0.get("max_log_std", 0.0),
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=params_1.get("clip_actions", False), role="value")

        self.action_chunk_size = params_0.get("action_chunk_size", 1)
        self._tile_action_limits(self.action_chunk_size)

        net = params_0.get("network", [{}])[0]
        self.net_container = build_mlp(net.get("layers", []), net.get("activations", "elu"))

        self.residual = None
        if params_0.get("residual", None) is not None:
            self.residual, self.residual_action_scale, self.residual_log_std_parameter = build_residual(
                params_0["residual"], self.num_actions
            )

        self.policy_layer = nn.LazyLinear(out_features=self.num_actions * self.action_chunk_size)
        self.log_std_parameter = nn.Parameter(
            torch.full(
                size=(self.num_actions * self.action_chunk_size,),
                fill_value=params_0.get("initial_log_std", 0.0),
                dtype=torch.float32,
            ),
            requires_grad=not params_0.get("fixed_log_std", False),
        )
        self.value_layer = nn.LazyLinear(out_features=1)

        self._shared_output = None

    def reset_shared_cache(self) -> None:
        self._shared_output = None

    def init_state_dict(self, inputs: dict[str, Any] = {}, *, role: str = "") -> None:
        super().init_state_dict(inputs, role=role)
        self._init_residual_state_dict()
        self._shared_output = None

    def act(self, inputs, *, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        elif role == "residual":
            return self._residual_act(inputs, role)
        raise ValueError(f"Unknown role: {role}")

    def compute(self, inputs, role=""):
        if role == "policy":
            observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
            net = self.net_container(observations)
            self._shared_output = net
            output = self.policy_layer(net)
            return output, {"log_std": self.log_std_parameter}
        elif role == "value":
            if self._shared_output is None:
                observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
                shared_output = self.net_container(observations)
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}
        elif role == "residual":
            output = self.residual(inputs.get("observations")) * self.residual_action_scale
            return output, {"log_std": self.residual_log_std_parameter}
        raise ValueError(f"Unknown role: {role}")
