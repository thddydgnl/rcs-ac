from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardConfig:
    progress_weight: float = 5.0
    progress_scale: float = 1.0
    dense_progress_distance_weight: float = 0.0
    dense_progress_positive_only: bool = True
    lane_weight: float = 0.1
    lane_offset_penalty_weight: float = 0.0
    heading_error_penalty_weight: float = 0.0
    heading_recovery_reward_weight: float = 0.0
    speed_alignment_weight: float = 0.0
    speed_alignment_lane_coeff: float = 0.0
    speed_alignment_heading_coeff: float = 0.0
    speed_reward_weight: float = 0.0
    speed_reward_target_kmh: float = 8.0
    curve_heading_error_threshold: float = 0.0
    curve_speed_penalty_weight: float = 0.0
    curve_lane_offset_threshold: float = 0.0
    lane_offset_speed_penalty_weight: float = 0.0
    collision_penalty: float = 10.0
    offroad_penalty: float = 5.0
    lane_invasion_penalty: float = 1.0
    steer_change_weight: float = 0.05
    throttle_change_weight: float = 0.02
    brake_change_weight: float = 0.02
    high_speed_steer_change_weight: float = 0.0
    high_speed_threshold_kmh: float = 0.0
    control_cost_scale: float = 1.0
    goal_bonus: float = 20.0
    stuck_penalty: float = 3.0
    stuck_penalty_scale: float = 1.0
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
        prev_target_distance = 0.0
        prev_heading_error = 0.0
        if prev_state is not None:
            prev_info = prev_state.get("info", {})
            prev_progress = float(prev_info.get("route_progress", 0.0))
            prev_target_distance = float(
                prev_info.get(
                    "target_distance",
                    prev_info.get("final_target_distance", prev_info.get("route_progress", 0.0)),
                )
            )
            prev_heading_error = abs(float(prev_info.get("heading_error", 0.0)))
        current_progress = float(info.get("route_progress", 0.0))
        current_target_distance = float(
            info.get(
                "target_distance",
                info.get("final_target_distance", info.get("route_progress", 0.0)),
            )
        )

        progress_delta = float(
            np.clip(
                current_progress - prev_progress,
                -self.config.progress_delta_clip,
                self.config.progress_delta_clip,
            )
        )
        progress_reward = self.config.progress_weight * progress_delta
        scaled_progress_reward = self.config.progress_scale * progress_reward

        dense_progress_delta = prev_target_distance - current_target_distance
        if self.config.dense_progress_positive_only:
            dense_progress_delta = max(0.0, dense_progress_delta)
        dense_progress_reward = self.config.dense_progress_distance_weight * dense_progress_delta

        lane_offset = abs(float(info.get("lane_offset", 0.0)))
        lane_reward = self.config.lane_weight * max(0.0, 1.0 - lane_offset)
        speed_kmh = max(0.0, float(info.get("speed_kmh", 0.0)))
        heading_error = abs(float(info.get("heading_error", 0.0)))
        heading_recovery_delta = max(0.0, prev_heading_error - heading_error)
        lane_offset_penalty = self.config.lane_offset_penalty_weight * lane_offset
        heading_error_penalty = self.config.heading_error_penalty_weight * heading_error
        heading_recovery_reward = self.config.heading_recovery_reward_weight * heading_recovery_delta
        alignment_factor = max(
            0.0,
            1.0
            - self.config.speed_alignment_lane_coeff * lane_offset
            - self.config.speed_alignment_heading_coeff * heading_error,
        )
        speed_reward = self.config.speed_reward_weight * min(
            speed_kmh / max(self.config.speed_reward_target_kmh, 1e-6),
            1.0,
        )
        speed_alignment_bonus = self.config.speed_alignment_weight * alignment_factor * speed_reward
        curve_overspeed_penalty = 0.0
        if heading_error > self.config.curve_heading_error_threshold > 0.0:
            curve_overspeed_penalty += self.config.curve_speed_penalty_weight * speed_kmh
        if lane_offset > self.config.curve_lane_offset_threshold > 0.0:
            curve_overspeed_penalty += self.config.lane_offset_speed_penalty_weight * speed_kmh

        collision_penalty = self.config.collision_penalty if bool(info.get("collision", False)) else 0.0
        offroad_penalty = self.config.offroad_penalty if bool(info.get("offroad", False)) else 0.0
        lane_invasion_penalty = self.config.lane_invasion_penalty if bool(info.get("lane_invasion", False)) else 0.0
        goal_bonus = self.config.goal_bonus if bool(info.get("reached_goal", False)) else 0.0
        stuck_penalty = self.config.stuck_penalty if bool(info.get("stuck", False)) else 0.0
        scaled_stuck_penalty = self.config.stuck_penalty_scale * stuck_penalty

        control_delta = np.abs(current_control.astype(np.float32) - prev_control.astype(np.float32))
        control_cost = (
            float(control_delta[0]) * self.config.steer_change_weight
            + float(control_delta[1]) * self.config.throttle_change_weight
            + float(control_delta[2]) * self.config.brake_change_weight
        )
        high_speed_steer_penalty = 0.0
        if speed_kmh >= self.config.high_speed_threshold_kmh > 0.0:
            high_speed_steer_penalty = self.config.high_speed_steer_change_weight * float(control_delta[0])
        scaled_control_cost = self.config.control_cost_scale * control_cost

        reward = (
            scaled_progress_reward
            + dense_progress_reward
            + lane_reward
            + heading_recovery_reward
            + speed_reward
            + speed_alignment_bonus
            - curve_overspeed_penalty
            - lane_offset_penalty
            - heading_error_penalty
            - high_speed_steer_penalty
            - collision_penalty
            - offroad_penalty
            - lane_invasion_penalty
            - scaled_control_cost
            + goal_bonus
            - scaled_stuck_penalty
        )

        breakdown = {
            "progress_reward": float(scaled_progress_reward),
            "dense_progress_reward": float(dense_progress_reward),
            "lane_reward": float(lane_reward),
            "lane_offset_penalty": float(lane_offset_penalty),
            "heading_error_penalty": float(heading_error_penalty),
            "heading_recovery_reward": float(heading_recovery_reward),
            "speed_reward": float(speed_reward),
            "speed_alignment_factor": float(alignment_factor),
            "speed_alignment_bonus": float(speed_alignment_bonus),
            "curve_overspeed_penalty": float(curve_overspeed_penalty),
            "collision_penalty": float(collision_penalty),
            "offroad_penalty": float(offroad_penalty),
            "lane_invasion_penalty": float(lane_invasion_penalty),
            "control_cost": float(scaled_control_cost),
            "high_speed_steer_penalty": float(high_speed_steer_penalty),
            "goal_bonus": float(goal_bonus),
            "stuck_penalty": float(scaled_stuck_penalty),
        }
        return float(reward), breakdown
