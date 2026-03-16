from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardConfig:
    progress_weight: float = 5.0
    lane_weight: float = 0.1
    collision_penalty: float = 10.0
    offroad_penalty: float = 5.0
    lane_invasion_penalty: float = 1.0
    steer_change_weight: float = 0.05
    throttle_change_weight: float = 0.02
    brake_change_weight: float = 0.02
    goal_bonus: float = 20.0
    stuck_penalty: float = 3.0
    progress_delta_clip: float = 1.0


class RewardCalculator:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute_reward(
        self,
        prev_state: dict[str, Any] | None,
        current_state: dict[str, Any],
        info: dict[str, Any],
        prev_control: np.ndarray,
        current_control: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        prev_progress = 0.0
        if prev_state is not None:
            prev_info = prev_state.get("info", {})
            prev_progress = float(prev_info.get("route_progress", 0.0))
        current_progress = float(info.get("route_progress", 0.0))

        progress_delta = float(
            np.clip(
                current_progress - prev_progress,
                -self.config.progress_delta_clip,
                self.config.progress_delta_clip,
            )
        )
        progress_reward = self.config.progress_weight * progress_delta

        lane_offset = abs(float(info.get("lane_offset", 0.0)))
        lane_reward = self.config.lane_weight * max(0.0, 1.0 - lane_offset)

        collision_penalty = self.config.collision_penalty if bool(info.get("collision", False)) else 0.0
        offroad_penalty = self.config.offroad_penalty if bool(info.get("offroad", False)) else 0.0
        lane_invasion_penalty = self.config.lane_invasion_penalty if bool(info.get("lane_invasion", False)) else 0.0
        goal_bonus = self.config.goal_bonus if bool(info.get("reached_goal", False)) else 0.0
        stuck_penalty = self.config.stuck_penalty if bool(info.get("stuck", False)) else 0.0

        control_delta = np.abs(current_control.astype(np.float32) - prev_control.astype(np.float32))
        control_cost = (
            float(control_delta[0]) * self.config.steer_change_weight
            + float(control_delta[1]) * self.config.throttle_change_weight
            + float(control_delta[2]) * self.config.brake_change_weight
        )

        reward = (
            progress_reward
            + lane_reward
            - collision_penalty
            - offroad_penalty
            - lane_invasion_penalty
            - control_cost
            + goal_bonus
            - stuck_penalty
        )

        breakdown = {
            "progress_reward": float(progress_reward),
            "lane_reward": float(lane_reward),
            "collision_penalty": float(collision_penalty),
            "offroad_penalty": float(offroad_penalty),
            "lane_invasion_penalty": float(lane_invasion_penalty),
            "control_cost": float(control_cost),
            "goal_bonus": float(goal_bonus),
            "stuck_penalty": float(stuck_penalty),
        }
        return float(reward), breakdown
