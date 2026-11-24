from __future__ import annotations

from typing import Any

import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(layers: list[int], activation: str = "elu") -> nn.Sequential:
    """Hidden MLP layers (LazyLinear + activation)."""
    net = nn.Sequential()
    for layer_size in layers:
        net.append(nn.LazyLinear(layer_size))
        net.append(_activation(activation))
    return net


def build_residual(residual: dict[str, Any], num_actions: int) -> tuple[nn.Sequential, float | nn.Parameter, nn.Parameter]:
    """Build the residual (per-step correction) head.

    The residual takes ``cat(observations, base_action)`` and outputs a single-step action correction.

    :return: (network, action scale, log std parameter)
    """
    net_cfg = residual["network"][0]
    net = build_mlp(net_cfg.get("layers", []), net_cfg.get("activations", "elu"))
    net.append(nn.LazyLinear(out_features=num_actions))
    action_scale = 1.0
    scale = residual.get("action_scale", None)
    if scale is not None:
        net.append(nn.Tanh())
        if isinstance(scale, (int, float)):
            action_scale = float(scale)
        elif scale == "learn_vector":
            action_scale = nn.Parameter(torch.zeros(num_actions, dtype=torch.float32), requires_grad=True)
        elif scale == "learn_scale":
            action_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        else:
            raise ValueError(f"Unsupported residual action_scale: {scale}")
    log_std = nn.Parameter(
        torch.full(size=(num_actions,), fill_value=residual.get("initial_log_std", 0.0), dtype=torch.float32),
        requires_grad=True,
    )
    return net, action_scale, log_std


class ChunkedGaussianMixin(GaussianMixin):
    """Gaussian mixin helpers shared by the chunked policies.

    - The policy outputs ``action_chunk_size`` actions flattened step-major: ``[a_0, a_1, ..., a_{C-1}]``,
      so action-space limits are tiled ``C`` times for clipping.
    - The ``"residual"`` role outputs a single-step correction, so it is never clipped to the action limits.
    """

    def _tile_action_limits(self, chunk_size: int) -> None:
        if chunk_size > 1:
            if isinstance(self._g_min_actions, torch.Tensor):
                self._g_min_actions = self._g_min_actions.repeat(chunk_size)
            if isinstance(self._g_max_actions, torch.Tensor):
                self._g_max_actions = self._g_max_actions.repeat(chunk_size)

    def _residual_act(self, inputs: dict[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
        clip_actions, clip_mean_actions = self._g_clip_actions, self._g_clip_mean_actions
        self._g_clip_actions, self._g_clip_mean_actions = False, False
        try:
            return GaussianMixin.act(self, inputs, role=role)
        finally:
            self._g_clip_actions, self._g_clip_mean_actions = clip_actions, clip_mean_actions

    def _init_residual_state_dict(self) -> None:
        """Materialize the residual's lazy modules (``init_state_dict`` only runs the policy role)."""
        if getattr(self, "residual", None) is None:
            return
        with torch.no_grad():
            obs_size = self.num_observations
            dummy = torch.zeros(1, obs_size + self.num_actions, device=self.device)
            self.residual.to(self.device)
            self.residual(dummy)


class MLPGaussian(ChunkedGaussianMixin, Model):
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
        if reduction != "sum":
            # PPO_CHUNKED relies on the chunk log-probability being the sum over all (step, action) dimensions
            raise ValueError("Chunked Gaussian policies require reduction='sum'")
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
        self._tile_action_limits(action_chunk_size)

        net = network[0]
        self.net_container = build_mlp(net.get("layers", []), net.get("activations", "elu"))
        self.net_container.append(nn.LazyLinear(self.num_actions * action_chunk_size))

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions * action_chunk_size,), fill_value=initial_log_std, dtype=torch.float32),
            requires_grad=not fixed_log_std,
        )

        self.residual = None
        if residual is not None:
            self.residual, self.residual_action_scale, self.residual_log_std_parameter = build_residual(
                residual, self.num_actions
            )

    def init_state_dict(self, inputs: dict[str, Any] = {}, *, role: str = "") -> None:
        super().init_state_dict(inputs, role=role)
        self._init_residual_state_dict()

    def act(self, inputs, *, role=""):
        if role == "residual":
            return self._residual_act(inputs, role)
        return GaussianMixin.act(self, inputs, role=role)

    def compute(self, inputs, role=""):
        if role == "residual":
            output = self.residual(inputs.get("observations")) * self.residual_action_scale
            return output, {"log_std": self.residual_log_std_parameter}
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        output = self.net_container(observations)
        return output, {"log_std": self.log_std_parameter}
