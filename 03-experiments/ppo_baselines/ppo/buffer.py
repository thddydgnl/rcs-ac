"""Rollout buffer with GAE support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class DictRolloutBuffer:
    """Collect on-policy transitions from dict observations."""

    def __init__(self, rollout_steps: int, obs_keys: list[str], obs_shapes: dict[str, tuple[int, ...]], action_dim: int):
        self.rollout_steps = rollout_steps
        self.obs_keys = obs_keys
        self.obs_shapes = obs_shapes
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.full = False
        self.observations = {
            key: np.zeros((self.rollout_steps, *self.obs_shapes[key]), dtype=np.float32)
            for key in self.obs_keys
        }
        self.actions = np.zeros((self.rollout_steps, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.rollout_steps, dtype=np.float32)
        self.dones = np.zeros(self.rollout_steps, dtype=np.float32)
        self.values = np.zeros(self.rollout_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.rollout_steps, dtype=np.float32)
        self.advantages = np.zeros(self.rollout_steps, dtype=np.float32)
        self.returns = np.zeros(self.rollout_steps, dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("Rollout buffer overflow.")
        for key in self.obs_keys:
            self.observations[key][self.ptr] = obs[key]
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1
        self.full = self.ptr == self.rollout_steps

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_minibatches(self, minibatch_size: int, device: torch.device):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        advantages = self.advantages[: self.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start_idx in range(0, self.ptr, minibatch_size):
            batch_idx = indices[start_idx : start_idx + minibatch_size]
            obs_batch = {
                key: torch.as_tensor(self.observations[key][batch_idx], dtype=torch.float32, device=device)
                for key in self.obs_keys
            }
            yield RolloutBatch(
                observations=obs_batch,
                actions=torch.as_tensor(self.actions[batch_idx], dtype=torch.float32, device=device),
                log_probs=torch.as_tensor(self.log_probs[batch_idx], dtype=torch.float32, device=device),
                returns=torch.as_tensor(self.returns[batch_idx], dtype=torch.float32, device=device),
                advantages=torch.as_tensor(advantages[batch_idx], dtype=torch.float32, device=device),
                values=torch.as_tensor(self.values[batch_idx], dtype=torch.float32, device=device),
            )

