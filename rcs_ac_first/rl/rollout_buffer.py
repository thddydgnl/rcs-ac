from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    auxiliary_observations: torch.Tensor
    actions: torch.Tensor
    hidden_states: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    aux_lane_offset_targets: torch.Tensor
    aux_heading_error_targets: torch.Tensor
    aux_road_mask_targets: torch.Tensor
    aux_lane_edge_targets: torch.Tensor
    aux_mask_targets_available: torch.Tensor
    reliability_targets: torch.Tensor
    risk_targets: torch.Tensor
    corruption_levels: torch.Tensor
    dropout_masks: torch.Tensor
    route_progress: torch.Tensor
    speed_kmh: torch.Tensor


class RolloutBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
        aux_mask_shape: tuple[int, int, int] = (1, 45, 80),
        auxiliary_dim: int = 30,
        hidden_state_shape: tuple[int, int] = (1, 256),
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.auxiliary_dim = int(auxiliary_dim)
        if self.auxiliary_dim <= 0:
            raise ValueError(f"auxiliary_dim must be > 0, got {auxiliary_dim!r}")
        self.aux_mask_shape = tuple(int(dim) for dim in aux_mask_shape)
        if len(self.aux_mask_shape) != 3 or any(dim <= 0 for dim in self.aux_mask_shape):
            raise ValueError(f"aux_mask_shape must be 3 positive ints, got {aux_mask_shape!r}")
        self.hidden_state_shape = tuple(int(dim) for dim in hidden_state_shape)
        if len(self.hidden_state_shape) != 2 or any(dim <= 0 for dim in self.hidden_state_shape):
            raise ValueError(f"hidden_state_shape must be 2 positive ints, got {hidden_state_shape!r}")
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.auxiliary_observations = np.zeros((self.capacity, self.auxiliary_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.hidden_states = np.zeros((self.capacity,) + self.hidden_state_shape, dtype=np.float32)
        self.log_probs = np.zeros((self.capacity,), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.values = np.zeros((self.capacity,), dtype=np.float32)
        self.advantages = np.zeros((self.capacity,), dtype=np.float32)
        self.returns = np.zeros((self.capacity,), dtype=np.float32)
        self.aux_lane_offset_targets = np.zeros((self.capacity,), dtype=np.float32)
        self.aux_heading_error_targets = np.zeros((self.capacity,), dtype=np.float32)
        self.aux_road_mask_targets = np.zeros((self.capacity,) + self.aux_mask_shape, dtype=np.float32)
        self.aux_lane_edge_targets = np.zeros((self.capacity,) + self.aux_mask_shape, dtype=np.float32)
        self.aux_mask_targets_available = np.zeros((self.capacity,), dtype=np.float32)
        self.reliability_targets = np.ones((self.capacity, 4), dtype=np.float32)
        self.risk_targets = np.zeros((self.capacity,), dtype=np.float32)
        self.corruption_levels = np.zeros((self.capacity,), dtype=np.float32)
        self.dropout_masks = np.zeros((self.capacity, 4), dtype=np.float32)
        self.route_progress = np.zeros((self.capacity,), dtype=np.float32)
        self.speed_kmh = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        *,
        observation: np.ndarray,
        auxiliary_observation: np.ndarray | None = None,
        action: np.ndarray,
        hidden_state_before_action: np.ndarray | torch.Tensor | None = None,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        aux_lane_offset_target: float = 0.0,
        aux_heading_error_target: float = 0.0,
        aux_road_mask_target: np.ndarray | None = None,
        aux_lane_edge_target: np.ndarray | None = None,
        aux_mask_targets_available: bool | float = False,
        reliability_target: np.ndarray | None = None,
        risk_target: float = 0.0,
        corruption_level: float = 0.0,
        dropout_mask: np.ndarray | None = None,
        route_progress: float = 0.0,
        speed_kmh: float = 0.0,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("RolloutBuffer is full. Call reset() after update().")
        self.observations[self.ptr] = np.asarray(observation, dtype=np.float32).reshape(self.obs_dim)
        self.auxiliary_observations[self.ptr] = self._coerce_auxiliary_observation(auxiliary_observation)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.hidden_states[self.ptr] = self._coerce_hidden_state(hidden_state_before_action)
        self.log_probs[self.ptr] = float(log_prob)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = float(value)
        self.aux_lane_offset_targets[self.ptr] = float(aux_lane_offset_target)
        self.aux_heading_error_targets[self.ptr] = float(aux_heading_error_target)
        self.aux_road_mask_targets[self.ptr] = self._coerce_mask_target(aux_road_mask_target)
        self.aux_lane_edge_targets[self.ptr] = self._coerce_mask_target(aux_lane_edge_target)
        self.aux_mask_targets_available[self.ptr] = float(bool(aux_mask_targets_available))
        self.reliability_targets[self.ptr] = self._coerce_reliability_target(reliability_target)
        self.risk_targets[self.ptr] = float(np.clip(float(risk_target), 0.0, 1.0))
        self.corruption_levels[self.ptr] = float(np.clip(float(corruption_level), 0.0, 1.0))
        self.dropout_masks[self.ptr] = self._coerce_dropout_mask(dropout_mask)
        self.route_progress[self.ptr] = float(route_progress)
        self.speed_kmh[self.ptr] = float(speed_kmh)
        self.ptr += 1

    def compute_returns_and_advantages(self, *, last_value: float, last_done: bool) -> None:
        last_advantage = 0.0
        for index in reversed(range(self.ptr)):
            if index == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - float(self.dones[index + 1])
                next_value = float(self.values[index + 1])
            delta = self.rewards[index] + self.gamma * next_value * next_non_terminal - self.values[index]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[index] = last_advantage
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_training_batch(self, device: torch.device) -> RolloutBatch:
        if self.ptr == 0:
            raise RuntimeError("RolloutBuffer is empty.")
        advantages = self.advantages[: self.ptr].copy()
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        else:
            advantages = advantages * 0.0
        return RolloutBatch(
            observations=torch.as_tensor(self.observations[: self.ptr], dtype=torch.float32, device=device),
            auxiliary_observations=torch.as_tensor(
                self.auxiliary_observations[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(self.actions[: self.ptr], dtype=torch.float32, device=device),
            hidden_states=torch.as_tensor(self.hidden_states[: self.ptr], dtype=torch.float32, device=device),
            log_probs=torch.as_tensor(self.log_probs[: self.ptr], dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
            returns=torch.as_tensor(self.returns[: self.ptr], dtype=torch.float32, device=device),
            values=torch.as_tensor(self.values[: self.ptr], dtype=torch.float32, device=device),
            aux_lane_offset_targets=torch.as_tensor(
                self.aux_lane_offset_targets[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            aux_heading_error_targets=torch.as_tensor(
                self.aux_heading_error_targets[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            aux_road_mask_targets=torch.as_tensor(
                self.aux_road_mask_targets[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            aux_lane_edge_targets=torch.as_tensor(
                self.aux_lane_edge_targets[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            aux_mask_targets_available=torch.as_tensor(
                self.aux_mask_targets_available[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            reliability_targets=torch.as_tensor(
                self.reliability_targets[: self.ptr],
                dtype=torch.float32,
                device=device,
            ),
            risk_targets=torch.as_tensor(self.risk_targets[: self.ptr], dtype=torch.float32, device=device),
            corruption_levels=torch.as_tensor(self.corruption_levels[: self.ptr], dtype=torch.float32, device=device),
            dropout_masks=torch.as_tensor(self.dropout_masks[: self.ptr], dtype=torch.float32, device=device),
            route_progress=torch.as_tensor(self.route_progress[: self.ptr], dtype=torch.float32, device=device),
            speed_kmh=torch.as_tensor(self.speed_kmh[: self.ptr], dtype=torch.float32, device=device),
        )

    def _coerce_hidden_state(self, value: np.ndarray | torch.Tensor | None) -> np.ndarray:
        if value is None:
            return np.zeros(self.hidden_state_shape, dtype=np.float32)
        if isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
        if tuple(array.shape) != self.hidden_state_shape:
            try:
                array = array.reshape(self.hidden_state_shape)
            except ValueError as exc:
                raise ValueError(
                    "hidden state shape mismatch: "
                    f"expected {self.hidden_state_shape}, got {tuple(array.shape)}"
                ) from exc
        if not np.all(np.isfinite(array)):
            raise ValueError("hidden state contains NaN or inf values")
        return array.astype(np.float32, copy=False)

    def _coerce_auxiliary_observation(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros((self.auxiliary_dim,), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(array.shape[0]) != int(self.auxiliary_dim):
            raise ValueError(f"auxiliary observation dim mismatch: expected {self.auxiliary_dim}, got {array.shape[0]}")
        if not np.all(np.isfinite(array)):
            raise ValueError("auxiliary observation contains NaN or inf values")
        return array.astype(np.float32, copy=False)

    def _coerce_reliability_target(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.ones((4,), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(array.shape[0]) != 4:
            raise ValueError(f"reliability target dim mismatch: expected 4, got {array.shape[0]}")
        if not np.all(np.isfinite(array)):
            raise ValueError("reliability target contains NaN or inf values")
        return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)

    def _coerce_dropout_mask(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros((4,), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(array.shape[0]) != 4:
            raise ValueError(f"dropout mask dim mismatch: expected 4, got {array.shape[0]}")
        if not np.all(np.isfinite(array)):
            raise ValueError("dropout mask contains NaN or inf values")
        return (array >= 0.5).astype(np.float32, copy=False)

    def _coerce_mask_target(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros(self.aux_mask_shape, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if tuple(array.shape) != self.aux_mask_shape:
            try:
                array = array.reshape(self.aux_mask_shape)
            except ValueError as exc:
                raise ValueError(
                    f"aux mask target shape mismatch: expected {self.aux_mask_shape}, got {tuple(array.shape)}"
                ) from exc
        if not np.all(np.isfinite(array)):
            raise ValueError("aux mask target contains NaN or inf values")
        return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
