from __future__ import annotations

from typing import Any

import copy
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


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
    chunk_size: int = 1,
    discount_vector: torch.Tensor = None,
    calculate_micro_return: bool = False
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param truncated: Signals to indicate that episodes have been truncated.
    :param values: Values obtained by the agent.
    :param last_values: Last values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.
    :param time_limit_bootstrap: Whether to use time-limit (truncation) bootstrapping.

    :return: Generalized Advantage Estimator.
    """
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    # num_envs x (num_rollouts * chunk_size) x 1
    macro_rewards_trans = rewards.transpose(0, 1)
    macro_rewards_reshaped = macro_rewards_trans.reshape(macro_rewards_trans.shape[0], -1, chunk_size)
    macro_rewards_fin = torch.bmm(macro_rewards_reshaped, discount_vector)
    macro_rewards = macro_rewards_fin.transpose(0, 1)
    macro_discount_factor = discount_factor**chunk_size

    macro_advantage = 0
    macro_advantages = torch.zeros_like(macro_rewards)
    if calculate_micro_return:
        # reshape rewards to be memory_size * chunk_size x num_envs x 1
        micro_advantage = 0
        micro_advantages = torch.zeros_like(rewards)
    
    memory_size = rewards.shape[0]
    # advantages computation
    for i in reversed(range(memory_size)):
        if calculate_micro_return:
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            micro_advantage = (
                rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * micro_advantage)
            )
            micro_advantages[i] = micro_advantage

            if i % chunk_size == 0:
                next_macro_values = values[i + chunk_size] if i < memory_size - chunk_size else last_values
                # Multiply across timesteps (dim 0)
                not_done_chunk = torch.prod(not_done[i:i+chunk_size], dim=0)
                macro_advantage = (
                    macro_rewards[i//chunk_size] - values[i] + macro_discount_factor * not_done_chunk * (next_macro_values + lambda_coefficient * macro_advantage)
                )
                macro_advantages[i//chunk_size] = macro_advantage
        else:
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            macro_advantage = (
                macro_rewards[i] - values[i] + macro_discount_factor * not_done[i] * (next_values + lambda_coefficient * macro_advantage)
            )
            macro_advantages[i] = macro_advantage

    # returns computation
    macro_returns = macro_advantages + values if not calculate_micro_return else macro_advantages + values[::chunk_size]
    # normalize advantages
    macro_advantages = (macro_advantages - macro_advantages.mean()) / (macro_advantages.std() + 1e-8)
    
    if calculate_micro_return:
        micro_returns = micro_advantages + values
        micro_advantages = (micro_advantages - micro_advantages.mean()) / (micro_advantages.std() + 1e-8)
    else:
        micro_returns = None
        micro_advantages = None

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

        # Action Chunking variables
        self.use_residual = hasattr(self.policy, 'residual')
        self.action_size = compute_space_size(self.action_space)
        self.chunk_size = self.cfg.action_chunk_size
        self.use_all_states = True if self.use_residual or self.cfg.use_all_states else False

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
                self.scheduler = self.cfg.learning_rate_scheduler[0](
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )

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
            # if isinstance(self.memory, RandomChunkedMemory)
            scale = 1 if not self.use_all_states else self.chunk_size
            # Scaled
            self.memory.create_tensor(name="observations", size=self.observation_space, scale=scale, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, scale=scale, dtype=torch.float32)

            self.memory.create_tensor(name="actions", size=compute_space_size(self.action_space) * self.chunk_size, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=self.chunk_size if scale == 1 else 1, scale=scale, dtype=torch.float32)

            # Scaled
            self.memory.create_tensor(name="terminated", size=1, scale=scale, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, scale=scale, dtype=torch.bool)

            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)

            # Scaled
            self.memory.create_tensor(name="values", size=1, scale=scale, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, scale=scale, dtype=torch.float32)

            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            micro_sizes = [None, None, None] if not self.use_residual else [self.action_space, 1, 1]
            # For residual
            self.memory.create_tensor(name="micro_actions", size=micro_sizes[0], scale=scale, dtype=torch.float32) 
            self.memory.create_tensor(name="micro_log_prob", size=micro_sizes[1], scale=scale, dtype=torch.float32)
            self.memory.create_tensor(name="micro_advantages", size=micro_sizes[2], scale=scale, dtype=torch.float32)
            # For using all states
            self.memory.create_tensor(name="micro_returns", size=None if not self.use_all_states else 1, scale=scale, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages", "micro_actions", "micro_log_prob", "micro_returns", "micro_advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._current_values = None
        self._rollout = 0
        self._outputs = None

        # Create temporary variables for residual policy
        self._current_action_micro = None
        self._current_log_prob_micro = None

        # For storing chunk stuff
        num_envs = self.memory.tensors['terminated'].shape[1]
        self._current_action_chunk = None
        self._executing_action_chunk = None
        self._stale_action_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if not self.use_all_states:
            self._chunk_observations = None
            self._chunk_states = None
            self._current_rewards = torch.empty(num_envs, self.chunk_size, dtype=torch.float32)
            self._terminated = torch.empty(num_envs, self.chunk_size, dtype=torch.bool)
            self._truncated = torch.empty(num_envs, self.chunk_size, dtype=torch.bool)
        # num_rollouts x action_chunk_size x 1
        # self._discount_vector = self.cfg.discount_factor ** torch.arange(self.chunk_size, device=self.device).repeat(self.memory.memory_size, 1).unsqueeze(-1)
        # num_envs x chunk_size x 1
        self._discount_vector = self.cfg.discount_factor ** torch.arange(self.chunk_size, device=self.device).repeat(num_envs, 1).unsqueeze(-1)

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
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            chunk_ind = self._rollout % self.chunk_size
            if chunk_ind == 0:
                actions, self._outputs = self.policy.act(inputs, role="policy")
                self._current_log_prob = self._outputs["log_prob"]
                self._current_action_chunk = actions
                # self._executing_action_chunk = actions.clone()
                self._stale_action_mask.zero_()
            # Don't zero out actions
            # else:
            #     if self._stale_action_mask.any():
            #         reset_envs = self._stale_action_mask.nonzero(as_tuple=True)[0]
            #         with torch.no_grad():
            #             new_actions, _ = self.policy.act(inputs, role='policy')

            #         start_idx = chunk_ind * self.action_size
            #         self._executing_action_chunk[reset_envs, start_idx:] = new_actions[reset_envs, start_idx:]

            # actions = self._executing_action_chunk[:, chunk_ind*self.action_size:(chunk_ind+1)*self.action_size]
            actions = self._current_action_chunk[:, chunk_ind*self.action_size:(chunk_ind+1)*self.action_size]
            if self.use_residual:
                residual_actions, outputs = self.policy.act(
                    {"observations":torch.cat((inputs['observations'], actions), dim=-1)}, 
                    role="residual"
                )
                self._current_log_prob_micro = outputs['log_prob']
                self._current_action_micro = residual_actions
                actions = actions + residual_actions

            # Zero out actions
            # # If an environment has ended, make sure no actions are taken
            env_actions = actions.clone()
            env_actions[self._stale_action_mask] = 0.0

            # compute values
            if self.training and (self.use_all_states or chunk_ind == 0):
                values, _ = self.value.act(inputs, role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)

            # When training, outputs isn't used during rollouts
            if self.training:
                outputs = None
            else:
                outputs = copy.deepcopy(self._outputs)
                outputs['mean_actions'] = self._outputs['mean_actions'][:, chunk_ind*self.action_size:(chunk_ind+1)*self.action_size]
                self._rollout += 1
        return actions, outputs

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

        if self.training:
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
                    next_values, _ = self.value.act(inputs, role="value")
                    next_values = self._value_preprocessor(next_values, inverse=True)

                rewards[..., -1:] += (self.cfg.discount_factor**rewards.shape[-1]) * next_values * truncated

            # storage transition in memory
            chunk_ind = self._rollout % self.chunk_size
            self._stale_action_mask |= (terminated | truncated).squeeze(-1)
            if self.use_all_states:
                # If using all states, add in the step by step data (observations, states, rewards, values, terminated, truncated)
                residuals = {} if not self.use_residual else {'micro_actions':self._current_action_micro, 'micro_log_prob':self._current_log_prob_micro}
                self.memory.add_samples(
                    observations=observations,
                    states=states,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    values=self._current_values,
                    **residuals
                )
                # Add in other data when it is the correct time (actions, log_prob)
                if chunk_ind == 0:
                    self.memory.add_samples(
                        inc_memory_index=True,
                        actions=self._current_action_chunk,
                        log_prob=self._current_log_prob,
                    )
            else:
                self._terminated[:, chunk_ind] = terminated.squeeze(-1)
                self._truncated[:, chunk_ind] = truncated.squeeze(-1)
                self._current_rewards[:, chunk_ind] = rewards.squeeze(-1)
                if chunk_ind == 0:
                    self._chunk_observations = observations
                    self._chunk_states = states
                # If the policy is about to predict again, add to the samples
                if chunk_ind == self.chunk_size - 1:
                    self.memory.add_samples(
                        inc_memory_index=True,
                        observations=self._chunk_observations,
                        states=self._chunk_states,
                        actions=self._current_action_chunk,
                        rewards=self._current_rewards,
                        terminated=torch.any(self._terminated, dim=-1, keepdim=True),
                        truncated=torch.any(self._truncated, dim=-1, keepdim=True),
                        log_prob=self._current_log_prob,
                        values=self._current_values,
                    )

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

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        # memory holds stuff as shape [num_rollout_steps, batch, size]
        values = self.memory.get_tensor_by_name("values")
        returns, advantages, micro_returns, micro_advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
            chunk_size=self.chunk_size,
            discount_vector=self._discount_vector,
            calculate_micro_return=self.use_all_states
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        if not self.use_all_states:
            self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
            explained_variance = self.explained_variance(self.memory.get_tensor_by_name("values"), self.memory.get_tensor_by_name("returns")).item()
        else:
            self.memory.set_tensor_by_name("micro_returns", self._value_preprocessor(micro_returns, train=True))
            explained_variance = self.explained_variance(self.memory.get_tensor_by_name("values"), self.memory.get_tensor_by_name("micro_returns")).item()
        self.memory.set_tensor_by_name("advantages", advantages)

        if self.use_residual:
            self.memory.set_tensor_by_name("micro_advantages", micro_advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_avg_kl = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_micro_actions,
                sampled_micro_log_prob,
                sampled_micro_returns,
                sampled_micro_advantages,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches, chunk_size=1 if not self.use_all_states else self.chunk_size,
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                    }
                    if not self.use_all_states:
                        _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    else:
                        _, outputs = self.policy.act({'observations':inputs['observations'][::self.chunk_size], "taken_actions": sampled_actions}, role="policy")
                        if hasattr(self.policy, '_shared_output'):
                            self.policy._shared_output = None
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = self.value.act(inputs, role="value")

                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns if not self.use_all_states else sampled_micro_returns, predicted_values)

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
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
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
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
        )
        self.track_data("Loss / Value loss", cumulative_value_loss / (self.cfg.learning_epochs * self.cfg.mini_batches))
        self.track_data("Loss / Explained Variance", explained_variance)
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        self.track_data("Learning / Avg KL", cumulative_avg_kl / self.cfg.learning_epochs)
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
