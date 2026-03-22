"""Sequence-aware rollout buffer for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RecurrentSequenceBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_starts: torch.Tensor
    valid_mask: torch.Tensor
    initial_hidden: torch.Tensor
    extras: dict[str, torch.Tensor]


class RecurrentRolloutBuffer:
    """Stores rollout data and yields sequence batches without timestep shuffle."""

    def __init__(
        self,
        rollout_steps: int,
        sequence_length: int,
        obs_shapes: dict[str, tuple[int, ...]],
        action_dim: int,
        hidden_dim: int,
        obs_keys: list[str],
        extra_specs: dict[str, tuple[int, ...]] | None = None,
    ):
        self.rollout_steps = rollout_steps
        self.sequence_length = sequence_length
        self.obs_shapes = obs_shapes
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.obs_keys = obs_keys
        self.extra_specs = extra_specs or {}
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.observations = {
            key: np.zeros((self.rollout_steps, *self.obs_shapes[key]), dtype=np.float32)
            for key in self.obs_keys
        }
        self.actions = np.zeros((self.rollout_steps, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.rollout_steps, dtype=np.float32)
        self.dones = np.zeros(self.rollout_steps, dtype=np.float32)
        self.episode_starts = np.zeros(self.rollout_steps, dtype=np.float32)
        self.values = np.zeros(self.rollout_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.rollout_steps, dtype=np.float32)
        self.hidden_states = np.zeros((self.rollout_steps, self.hidden_dim), dtype=np.float32)
        self.advantages = np.zeros(self.rollout_steps, dtype=np.float32)
        self.returns = np.zeros(self.rollout_steps, dtype=np.float32)
        self.extras = {
            key: np.zeros((self.rollout_steps, *shape), dtype=np.float32)
            for key, shape in self.extra_specs.items()
        }

    def masked_ratio(self) -> float:
        if self.ptr == 0:
            return 0.0
        num_sequences = int(np.ceil(self.ptr / self.sequence_length))
        total_slots = num_sequences * self.sequence_length
        masked_slots = total_slots - self.ptr
        return float(masked_slots / max(total_slots, 1))

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        episode_start: bool,
        value: float,
        log_prob: float,
        hidden_state: np.ndarray,
        extras: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("Recurrent rollout buffer overflow.")
        for key in self.obs_keys:
            self.observations[key][self.ptr] = obs[key]
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.episode_starts[self.ptr] = float(episode_start)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.hidden_states[self.ptr] = hidden_state
        if extras:
            for key, value in extras.items():
                self.extras[key][self.ptr] = value
        self.ptr += 1

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

    def _sequence_starts(self) -> list[int]:
        return list(range(0, self.ptr, self.sequence_length))

    def get_sequence_minibatches(
        self,
        minibatch_num_sequences: int,
        device: torch.device,
    ):
        advantages = self.advantages[: self.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        starts = self._sequence_starts()
        np.random.shuffle(starts)

        for batch_start in range(0, len(starts), minibatch_num_sequences):
            batch_seq_starts = starts[batch_start : batch_start + minibatch_num_sequences]
            batch_size = len(batch_seq_starts)
            obs_batch = {
                key: np.zeros((batch_size, self.sequence_length, *self.obs_shapes[key]), dtype=np.float32)
                for key in self.obs_keys
            }
            actions = np.zeros((batch_size, self.sequence_length, self.action_dim), dtype=np.float32)
            old_log_probs = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
            returns = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
            advs = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
            episode_starts = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
            valid_mask = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
            initial_hidden = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            extras = {
                key: np.zeros((batch_size, self.sequence_length, *shape), dtype=np.float32)
                for key, shape in self.extra_specs.items()
            }

            for seq_idx, start in enumerate(batch_seq_starts):
                end = min(start + self.sequence_length, self.ptr)
                valid_len = end - start
                for key in self.obs_keys:
                    obs_batch[key][seq_idx, :valid_len] = self.observations[key][start:end]
                actions[seq_idx, :valid_len] = self.actions[start:end]
                old_log_probs[seq_idx, :valid_len] = self.log_probs[start:end]
                returns[seq_idx, :valid_len] = self.returns[start:end]
                advs[seq_idx, :valid_len] = advantages[start:end]
                episode_starts[seq_idx, :valid_len] = self.episode_starts[start:end]
                valid_mask[seq_idx, :valid_len] = 1.0
                initial_hidden[seq_idx] = self.hidden_states[start]
                for key in self.extra_specs:
                    extras[key][seq_idx, :valid_len] = self.extras[key][start:end]

            yield RecurrentSequenceBatch(
                observations={
                    key: torch.as_tensor(value, dtype=torch.float32, device=device)
                    for key, value in obs_batch.items()
                },
                actions=torch.as_tensor(actions, dtype=torch.float32, device=device),
                old_log_probs=torch.as_tensor(old_log_probs, dtype=torch.float32, device=device),
                returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
                advantages=torch.as_tensor(advs, dtype=torch.float32, device=device),
                episode_starts=torch.as_tensor(episode_starts, dtype=torch.float32, device=device),
                valid_mask=torch.as_tensor(valid_mask, dtype=torch.float32, device=device),
                initial_hidden=torch.as_tensor(initial_hidden, dtype=torch.float32, device=device),
                extras={
                    key: torch.as_tensor(value, dtype=torch.float32, device=device)
                    for key, value in extras.items()
                },
            )
