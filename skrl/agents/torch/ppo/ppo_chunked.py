from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory, RandomChunkedMemory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer
from skrl.utils.spaces.torch import compute_space_size

from .ppo_chunked_cfg import PPO_CHUNKED_CFG


# Storage layout
# --------------
# Everything is stored per environment step: tensors have shape (rollouts, num_envs, size).
# Each env executes its own chunk of ``C`` actions. A new chunk is predicted for an env when
#   - its current chunk is exhausted (C actions executed),
#   - its episode ended at the previous step (terminated or truncated), so the remainder of the old chunk is
#     discarded and the new episode starts with a fresh chunk,
#   - a new rollout starts (so no chunk spans two rollouts / updates).
# The step at which a chunk is predicted is flagged in ``chunk_starts`` and holds the whole chunk (``actions``,
# shape C * A) and the per-step log-probabilities of the chunk (``log_prob``, shape C). The number of actions of
# the chunk actually executed (``L <= C``) is recovered from ``chunk_starts`` at update time.


def compute_chunk_lengths(chunk_starts: torch.Tensor) -> torch.Tensor:
    """Number of executed steps of the chunk starting at each step (0 where no chunk starts).

    A chunk runs until (excluded) the next chunk start of the same env, or until the end of the rollout.

    :param chunk_starts: Chunk start flags, shape ``(T, N, 1)``.

    :return: Chunk lengths, shape ``(T, N, 1)``, dtype long.
    """
    T = chunk_starts.shape[0]
    lengths = torch.zeros(chunk_starts.shape, dtype=torch.long, device=chunk_starts.device)
    next_start = torch.full(chunk_starts.shape[1:], T, dtype=torch.long, device=chunk_starts.device)
    for t in reversed(range(T)):
        lengths[t] = torch.where(chunk_starts[t], next_start - t, 0)
        next_start = torch.where(chunk_starts[t], t, next_start)
    return lengths


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    chunk_starts: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
    compute_micro: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute chunk-level (macro) and step-level (micro) Generalized Advantage Estimates.

    Macro (semi-MDP) advantage of a chunk started at ``t`` and executed for ``L`` steps::

        R_t   = sum_{k<L} gamma^k r_{t+k}
        A_t   = R_t + gamma^L * not_done_{t+L-1} * (V(s_{t+L}) + lambda * A_{t+L}) - V(s_t)

    where ``A_{t+L}`` is the advantage of the env's next chunk (0 at the end of the rollout, in which case
    ``V(s_{t+L})`` is ``last_values``). Since an episode end always starts a new chunk, ``R_t`` never contains
    rewards from the following episode. ``L == C`` and aligned chunks reproduce the fixed-grid formulation.

    All inputs have shape ``(T, N, 1)`` except ``last_values`` ``(N, 1)``.

    :return: macro returns, macro advantages (0 where no chunk starts), micro returns, micro advantages.
        Advantages are NOT normalized.
    """
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not().to(values.dtype)
    chunk_starts = chunk_starts.bool()
    T = rewards.shape[0]

    # a chunk segment ends at step t if the env starts a new chunk at t + 1 or if t is the last rollout step
    segment_end = torch.ones_like(chunk_starts)
    segment_end[:-1] = chunk_starts[1:]

    macro_advantages = torch.zeros_like(values)
    micro_advantages = torch.zeros_like(values) if compute_micro else None
    segment_return = torch.zeros_like(last_values)  # discounted (bootstrapped) reward-to-go of the current chunk
    next_macro_advantage = torch.zeros_like(last_values)  # macro advantage of the chunk starting at t + 1
    micro_advantage = torch.zeros_like(last_values)

    for t in reversed(range(T)):
        next_values = values[t + 1] if t < T - 1 else last_values
        # macro
        tail = torch.where(
            segment_end[t], next_values + lambda_coefficient * next_macro_advantage, segment_return
        )
        segment_return = rewards[t] + discount_factor * not_done[t] * tail
        macro_advantages[t] = torch.where(chunk_starts[t], segment_return - values[t], 0.0)
        next_macro_advantage = torch.where(chunk_starts[t], macro_advantages[t], next_macro_advantage)
        # micro
        if compute_micro:
            micro_advantage = (
                rewards[t]
                - values[t]
                + discount_factor * not_done[t] * (next_values + lambda_coefficient * micro_advantage)
            )
            micro_advantages[t] = micro_advantage

    macro_returns = macro_advantages + values
    micro_returns = micro_advantages + values if compute_micro else None
    return macro_returns, macro_advantages, micro_returns, micro_advantages


class PPO_CHUNKED(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: PPO_CHUNKED_CFG | dict = {},
    ) -> None:
        """Proximal Policy Optimization (PPO) with action chunking.

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: PPO_CHUNKED_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=PPO_CHUNKED_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # action chunking
        self.use_residual = getattr(self.policy, "residual", None) is not None
        self.residual_training_step = self.cfg.residual_begin_training or 0
        self.action_size = compute_space_size(self.action_space)
        self.chunk_size = self.cfg.action_chunk_size
        self.use_all_states = bool(self.use_residual or self.cfg.use_all_states)

        self.ratio_clip_chunk_scaling = (self.cfg.ratio_clip_chunk_scaling or "none").lower()
        if self.ratio_clip_chunk_scaling not in ["none", "linear", "exponential", "per_step"]:
            raise ValueError(
                f"Unknown ratio_clip_chunk_scaling: {self.cfg.ratio_clip_chunk_scaling} "
                "(use None, 'linear', 'exponential' or 'per_step')"
            )
        self.per_step_ratio = self.ratio_clip_chunk_scaling == "per_step"
        if self.ratio_clip_chunk_scaling == "linear" and self.cfg.ratio_clip * self.chunk_size >= 1:
            logger.warning(
                f"ratio_clip_chunk_scaling='linear': lower clip bound 1 - {self.chunk_size} * {self.cfg.ratio_clip} <= 0, "
                "so decreasing the probability of negative-advantage chunks is never clipped. Consider 'exponential'"
            )

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            # - optimizers
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self.cfg.learning_rate[0]
                )
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate schedulers
            self.scheduler = self.cfg.learning_rate_scheduler[0]
            if self.scheduler is not None:
                scheduler_kwargs = dict(self.cfg.learning_rate_scheduler_kwargs[0])
                # the chunk KL grows ~linearly with the number of actions per chunk: scale the target accordingly.
                # With per-step ratios the per-action KL is used, so the threshold is kept as given
                if self.scheduler is KLAdaptiveLR and not self.per_step_ratio:
                    scheduler_kwargs["kl_threshold"] = scheduler_kwargs.get("kl_threshold", 0.008) * self.chunk_size
                self.scheduler = self.cfg.learning_rate_scheduler[0](self.optimizer, **scheduler_kwargs)

        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            if self.memory.memory_size != self.cfg.rollouts:
                raise ValueError(
                    f"PPO_CHUNKED stores one entry per environment step: memory_size ({self.memory.memory_size}) "
                    f"must equal rollouts ({self.cfg.rollouts}). Use 'memory_size: -1' in the runner config"
                )
            if self.cfg.rollouts < self.chunk_size:
                logger.warning(f"rollouts ({self.cfg.rollouts}) < action_chunk_size ({self.chunk_size})")
            A, C = self.action_size, self.chunk_size
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=A * C, dtype=torch.float32)  # chunk (at chunk starts)
            self.memory.create_tensor(name="log_prob", size=C, dtype=torch.float32)  # per-step log-prob of the chunk
            self.memory.create_tensor(name="chunk_starts", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            if self.use_residual:
                self.memory.create_tensor(name="base_actions", size=A, dtype=torch.float32)
                self.memory.create_tensor(name="micro_actions", size=A, dtype=torch.float32)
                self.memory.create_tensor(name="micro_log_prob", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="micro_advantages", size=1, dtype=torch.float32)

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_values = None
        self._rollout = 0

        # per-env chunk state
        self._chunk_step = None  # (N,) index of the action of the current chunk to execute next
        self._current_action_chunk = None  # (N, C * A)
        self._current_mean_chunk = None  # (N, C * A)
        self._current_step_log_prob = None  # (N, C)
        self._current_chunk_start = None  # (N,) bool: a chunk was predicted at the current step

        # residual policy
        self._current_base_actions = None
        self._current_action_micro = None
        self._current_log_prob_micro = None

    # ------------------------------------------------------------------ helpers

    def _per_step_log_prob(self, distribution: torch.distributions.Normal, chunk: torch.Tensor) -> torch.Tensor:
        """Log-probability of each action of a chunk: ``(B, C * A) -> (B, C)``."""
        log_prob = distribution.log_prob(chunk)
        return log_prob.view(*log_prob.shape[:-1], self.chunk_size, self.action_size).sum(dim=-1)

    def _select_step(self, chunk: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Action at (per-env) position ``step`` of each env's chunk: ``(N, C * A), (N,) -> (N, A)``."""
        chunk = chunk.view(chunk.shape[0], self.chunk_size, self.action_size)
        return chunk[torch.arange(chunk.shape[0], device=chunk.device), step]

    def _reset_shared_cache(self) -> None:
        """Invalidate the backbone output cached by shared models (before a value call on new inputs)."""
        for model in (self.policy, self.value):
            if model is None:
                continue
            if hasattr(model, "reset_shared_cache"):
                model.reset_shared_cache()
            elif hasattr(model, "_shared_output"):
                model._shared_output = None

    def _ratio_clip_bounds(self, lengths: torch.Tensor) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Clip range of the chunk ratio given the number of executed actions per chunk ``(B,)``."""
        eps = self.cfg.ratio_clip
        if self.ratio_clip_chunk_scaling in ["none", "per_step"]:
            return 1.0 - eps, 1.0 + eps
        L = lengths.unsqueeze(-1).to(torch.float32)
        if self.ratio_clip_chunk_scaling == "linear":
            return 1.0 - eps * L, 1.0 + eps * L
        return (1.0 - eps) ** L, (1.0 + eps) ** L

    def _residual_active(self, timestep: int) -> bool:
        return self.use_residual and (timestep >= self.residual_training_step or not self.training)

    # ------------------------------------------------------------------ interaction

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # sample random actions
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        num_envs = observations.shape[0]
        if self._chunk_step is None or self._chunk_step.shape[0] != num_envs:
            self._chunk_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            self._current_action_chunk = None
        # every rollout starts with a fresh chunk for all envs so that no chunk spans two updates
        if self.training and self._rollout % self.cfg.rollouts == 0:
            self._chunk_step.zero_()

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            # predict a new chunk for the envs whose chunk is exhausted or whose episode just ended
            need_new = self._chunk_step == 0
            policy_called = bool(need_new.any())
            if policy_called:
                chunk, outputs = self.policy.act(inputs, role="policy")
                step_log_prob = self._per_step_log_prob(self.policy.distribution(role="policy"), chunk)
                mean_chunk = outputs["mean_actions"]
                if self._current_action_chunk is None or bool(need_new.all()):
                    self._current_action_chunk = chunk
                    self._current_mean_chunk = mean_chunk
                    self._current_step_log_prob = step_log_prob
                else:
                    mask = need_new.unsqueeze(-1)
                    self._current_action_chunk = torch.where(mask, chunk, self._current_action_chunk)
                    self._current_mean_chunk = torch.where(mask, mean_chunk, self._current_mean_chunk)
                    self._current_step_log_prob = torch.where(mask, step_log_prob, self._current_step_log_prob)
            self._current_chunk_start = need_new

            actions = self._select_step(self._current_action_chunk, self._chunk_step)
            self._current_base_actions = actions

            # residual (per-step) correction
            residual_active = self._residual_active(timestep)
            if residual_active:
                residual_actions, residual_outputs = self.policy.act(
                    {"observations": torch.cat((inputs["observations"], actions), dim=-1)}, role="residual"
                )
                self._current_log_prob_micro = residual_outputs["log_prob"]
                self._current_action_micro = residual_actions
                actions = actions + residual_actions

            # compute values (every step: needed to bootstrap chunks that start at any step)
            if self.training:
                if not policy_called:
                    self._reset_shared_cache()
                values, _ = self.value.act(inputs, role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)
                return actions, {}

            # evaluation: deterministic actions (mean chunk action + mean residual for that mean action)
            mean_actions = self._select_step(self._current_mean_chunk, self._chunk_step)
            if residual_active:
                _, residual_mean_outputs = self.policy.act(
                    {"observations": torch.cat((inputs["observations"], mean_actions), dim=-1)}, role="residual"
                )
                mean_actions = mean_actions + residual_mean_outputs["mean_actions"]
        return actions, {"mean_actions": mean_actions}

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.training and self._chunk_step is not None:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap and truncated.any():
                with torch.no_grad():
                    inputs = {
                        "observations": self._observation_preprocessor(next_observations),
                        "states": self._state_preprocessor(next_states),
                    }
                    self._reset_shared_cache()
                    next_values, _ = self.value.act(inputs, role="value")
                    next_values = self._value_preprocessor(next_values, inverse=True)
                rewards = rewards + self.cfg.discount_factor * next_values * truncated

            tensors = dict(
                observations=observations,
                states=states,
                actions=self._current_action_chunk,
                log_prob=self._current_step_log_prob,
                chunk_starts=self._current_chunk_start.unsqueeze(-1),
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                values=self._current_values,
            )
            if self.use_residual and timestep >= self.residual_training_step:
                tensors.update(
                    base_actions=self._current_base_actions,
                    micro_actions=self._current_action_micro,
                    micro_log_prob=self._current_log_prob_micro,
                )
            if isinstance(self.memory, RandomChunkedMemory):
                self.memory.add_samples(inc_memory_index=True, **tensors)
            else:
                self.memory.add_samples(**tensors)

        # advance each env's position in its chunk. Envs whose chunk is exhausted or whose episode just ended
        # (the env has been reset: the rest of the old chunk is discarded) predict a new chunk at the next step
        if self._chunk_step is not None:
            done = (terminated | truncated).view(-1).to(self._chunk_step.device)
            self._chunk_step += 1
            self._chunk_step[(self._chunk_step >= self.chunk_size) | done] = 0

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def explained_variance(self, y_pred, y_true):
        var_y = torch.var(y_true)
        return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

    # ------------------------------------------------------------------ update

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        C, A = self.chunk_size, self.action_size

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self._reset_shared_cache()
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        chunk_starts = self.memory.get_tensor_by_name("chunk_starts")
        chunk_lengths = compute_chunk_lengths(chunk_starts)
        macro_returns, macro_advantages, micro_returns, micro_advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            chunk_starts=chunk_starts,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
            compute_micro=self.use_all_states,
        )

        start_mask = chunk_starts.squeeze(-1)  # (T, N)
        # normalize the chunk advantages (over chunk starts only)
        start_advantages = macro_advantages[start_mask]
        macro_advantages[start_mask] = (start_advantages - start_advantages.mean()) / (start_advantages.std() + 1e-8)

        # value targets: every step (micro returns) or chunk starts only (macro returns)
        if self.use_all_states:
            value_mask = torch.ones_like(start_mask)
            returns = micro_returns
        else:
            value_mask = start_mask
            returns = macro_returns
        explained_variance = self.explained_variance(values[value_mask], returns[value_mask])
        explained_variance = explained_variance.item() if isinstance(explained_variance, torch.Tensor) else explained_variance

        processed_values = values.clone()
        processed_values[value_mask] = self._value_preprocessor(values[value_mask], train=True)
        processed_returns = returns.clone()
        processed_returns[value_mask] = self._value_preprocessor(returns[value_mask], train=True)
        self.memory.set_tensor_by_name("values", processed_values)
        self.memory.set_tensor_by_name("returns", processed_returns)
        self.memory.set_tensor_by_name("advantages", macro_advantages)

        # the residual is trained once a full rollout has been collected with it acting
        train_residual = self.use_residual and (timestep - self.cfg.rollouts + 1 >= self.residual_training_step)
        if train_residual:
            micro_advantages = (micro_advantages - micro_advantages.mean()) / (micro_advantages.std() + 1e-8)
            self.memory.set_tensor_by_name("micro_advantages", micro_advantages)

        # flat (t * num_envs + env) indices into memory.tensors_view
        views = self.memory.tensors_view
        chunk_indexes = start_mask.reshape(-1).nonzero().squeeze(-1)
        value_indexes = value_mask.reshape(-1).nonzero().squeeze(-1)
        flat_lengths = chunk_lengths.view(-1)
        step_range = torch.arange(C, device=flat_lengths.device)

        def get(name, indexes):
            return views[name][indexes] if name in views else None

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_avg_kl = 0
        num_updates = 0
        policy_distribution = None

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            chunk_permutation = torch.randperm(chunk_indexes.numel(), device=chunk_indexes.device)
            chunk_batches = chunk_indexes[chunk_permutation].tensor_split(self.cfg.mini_batches)
            if self.use_all_states:
                value_permutation = torch.randperm(value_indexes.numel(), device=value_indexes.device)
                value_batches = value_indexes[value_permutation].tensor_split(self.cfg.mini_batches)
            else:
                value_batches = chunk_batches  # same rows: shared models can reuse the policy's backbone output

            # mini-batches loop
            for chunk_batch, value_batch in zip(chunk_batches, value_batches):
                if chunk_batch.numel() == 0:
                    continue

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    if self.use_all_states:
                        value_inputs = {
                            "observations": self._observation_preprocessor(
                                get("observations", value_batch), train=not epoch
                            ),
                            "states": self._state_preprocessor(get("states", value_batch), train=not epoch),
                        }
                        chunk_inputs = {
                            "observations": self._observation_preprocessor(get("observations", chunk_batch)),
                            "states": self._state_preprocessor(get("states", chunk_batch)),
                        }
                    else:
                        chunk_inputs = {
                            "observations": self._observation_preprocessor(
                                get("observations", chunk_batch), train=not epoch
                            ),
                            "states": self._state_preprocessor(get("states", chunk_batch), train=not epoch),
                        }
                        value_inputs = chunk_inputs

                    # chunk policy: log-probability of the executed part of each chunk
                    sampled_chunks = get("actions", chunk_batch)
                    self.policy.act({**chunk_inputs, "taken_actions": sampled_chunks}, role="policy")
                    policy_distribution = self.policy.distribution(role="policy")
                    if self.use_all_states:
                        self._reset_shared_cache()

                    lengths = flat_lengths[chunk_batch]
                    executed = (step_range < lengths.unsqueeze(-1)).to(torch.float32)  # (B, C)
                    num_executed = executed.sum()
                    # per-action log-ratios of the executed actions (0 for the non-executed ones)
                    step_log_ratio = (
                        self._per_step_log_prob(policy_distribution, sampled_chunks) - get("log_prob", chunk_batch)
                    ) * executed
                    log_ratio = step_log_ratio.sum(dim=-1, keepdim=True)  # chunk log-ratio

                    # compute approximate KL divergence (per action for per-step ratios, per chunk otherwise)
                    with torch.no_grad():
                        if self.per_step_ratio:
                            kl = ((torch.exp(step_log_ratio) - 1) - step_log_ratio) * executed
                            kl_divergence = kl.sum() / num_executed
                        else:
                            kl_divergence = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    # compute entropy loss (executed actions only)
                    if self.cfg.entropy_loss_scale:
                        entropy = policy_distribution.entropy()
                        entropy = (entropy.view(-1, C, A).sum(dim=-1) * executed).sum(dim=-1)
                        entropy_loss = -self.cfg.entropy_loss_scale * entropy.mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    sampled_advantages = get("advantages", chunk_batch)
                    lower, upper = self._ratio_clip_bounds(lengths)
                    if self.per_step_ratio:
                        # each executed action is clipped individually; the chunk advantage is shared by its actions
                        ratio = torch.exp(step_log_ratio)  # (B, C)
                        surrogate = sampled_advantages * ratio
                        surrogate_clipped = sampled_advantages * torch.clamp(ratio, min=lower, max=upper)
                        policy_loss = -(torch.min(surrogate, surrogate_clipped) * executed).sum() / num_executed
                    else:
                        ratio = torch.exp(log_ratio)  # (B, 1)
                        surrogate = sampled_advantages * ratio
                        surrogate_clipped = sampled_advantages * torch.clamp(ratio, min=lower, max=upper)
                        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # residual (per-step) policy loss
                    if train_residual:
                        _, micro_outputs = self.policy.act(
                            {
                                "observations": torch.cat(
                                    (value_inputs["observations"], get("base_actions", value_batch)), dim=-1
                                ),
                                "taken_actions": get("micro_actions", value_batch),
                            },
                            role="residual",
                        )
                        sampled_micro_advantages = get("micro_advantages", value_batch)
                        micro_ratio = torch.exp(micro_outputs["log_prob"] - get("micro_log_prob", value_batch))
                        micro_surrogate = sampled_micro_advantages * micro_ratio
                        micro_surrogate_clipped = sampled_micro_advantages * torch.clip(
                            micro_ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                        )
                        policy_loss = policy_loss - torch.min(micro_surrogate, micro_surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = self.value.act(value_inputs, role="value")
                    sampled_values = get("values", value_batch)
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(get("returns", value_batch), predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self.cfg.grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                num_updates += 1
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if kl_divergences:
                cumulative_avg_kl += torch.tensor(kl_divergences, device=self.device).mean().item()
            # update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        num_updates = max(num_updates, 1)
        self.track_data("Loss / Policy loss", cumulative_policy_loss / num_updates)
        self.track_data("Loss / Value loss", cumulative_value_loss / num_updates)
        self.track_data("Loss / Explained Variance", explained_variance)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / num_updates)

        if policy_distribution is not None:
            self.track_data("Policy / Standard deviation", policy_distribution.stddev.mean().item())
        self.track_data("Policy / Mean executed chunk length", flat_lengths[chunk_indexes].float().mean().item())
        self.track_data("Learning / Avg KL", cumulative_avg_kl / self.cfg.learning_epochs)
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
