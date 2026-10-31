from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, gamma: float, gae_lambda: float) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity,), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.values = np.zeros((self.capacity,), dtype=np.float32)
        self.advantages = np.zeros((self.capacity,), dtype=np.float32)
        self.returns = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("RolloutBuffer is full. Call reset() after update().")

        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = float(log_prob)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = float(value)
        self.ptr += 1
        self.full = self.ptr == self.capacity

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        last_advantage = 0.0
        for index in reversed(range(self.ptr)):
            if index == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - self.dones[index + 1]
                next_value = self.values[index + 1]

            delta = self.rewards[index] + self.gamma * next_value * next_non_terminal - self.values[index]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[index] = last_advantage

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_training_batch(self, device: torch.device) -> RolloutBatch:
        if self.ptr == 0:
            raise RuntimeError("RolloutBuffer is empty.")

        advantages = self.advantages[: self.ptr].copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutBatch(
            observations=torch.as_tensor(self.observations[: self.ptr], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[: self.ptr], dtype=torch.float32, device=device),
            log_probs=torch.as_tensor(self.log_probs[: self.ptr], dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
            returns=torch.as_tensor(self.returns[: self.ptr], dtype=torch.float32, device=device),
            values=torch.as_tensor(self.values[: self.ptr], dtype=torch.float32, device=device),
        )
