"""Simple multimodal 2D navigation toy environment for RCS-AC-style research.

This environment is intentionally minimal and is designed to validate:
1. multimodal observation plumbing,
2. clean extension points for reliability / consistency / risk-aware fusion,
3. modality-specific corruption handling.

The implementation follows the Gymnasium API and keeps all observation
components shape-stable and normalized as much as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ============================================================================
# Top-level constants
# ============================================================================

DT = 0.1
MAP_SIZE = 10.0
HALF_MAP = MAP_SIZE / 2.0
AGENT_RADIUS = 0.20
GOAL_RADIUS = 0.35
OBSTACLE_RADIUS = 0.45
NUM_OBSTACLES = 4
MAX_EPISODE_STEPS = 200

MAX_SPEED = 1.5
MAX_ACCEL = 1.0
MAX_TURN_RATE = np.pi / 2.0
ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)

VISION_NUM_SECTORS = 8
VISION_FOV = np.pi
VISION_RANGE = 3.0

DISTANCE_NUM_RAYS = 9
DISTANCE_FOV = np.pi
DISTANCE_RANGE = 4.0

ROUTE_INCLUDE_WAYPOINT = True

CORRUPTION_NOISE_STD = 0.15
QUALITY_DEGRADATION_ON_CORRUPTION = 0.5
RISK_BASELINE = 0.05
RISK_CORRUPTION_INCREMENT = 0.20

SEED_OBSTACLE_MARGIN = 1.2
MIN_GOAL_AGENT_DISTANCE = 2.0
MIN_ENTITY_SEPARATION = 1.4
MAX_RESET_SAMPLING_TRIES = 500

EPS = 1e-6
FLOAT32_MIN = np.float32(-1.0e6)
FLOAT32_MAX = np.float32(1.0e6)

PROGRESS_REWARD_SCALE = 1.0
PROGRESS_REWARD_CLIP = 0.20
REWARD_GOAL = 2.0
REWARD_COLLISION = -2.0
REWARD_OUT_OF_BOUNDS = -2.0
ACTION_CHANGE_PENALTY_SCALE = 0.02
ACTION_CHANGE_PENALTY_CLIP = 0.05
IDLE_SPEED_THRESHOLD = 0.05
IDLE_PROGRESS_THRESHOLD = 0.002
IDLE_PENALTY = -0.02
REWARD_TOTAL_CLIP = 3.0


@dataclass
class CircleObstacle:
    """Simple circular obstacle."""

    x: float
    y: float
    radius: float


class RCSToyNavEnv(gym.Env):
    """2D goal-reaching environment with modality-separated observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        render_mode: str | None = None,
        corruption_config: dict[str, bool] | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.corruption_config = corruption_config or {}

        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(VISION_NUM_SECTORS,),
                    dtype=np.float32,
                ),
                "distance": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(DISTANCE_NUM_RAYS,),
                    dtype=np.float32,
                ),
                "ego": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(5,),
                    dtype=np.float32,
                ),
                "route": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(5 if ROUTE_INCLUDE_WAYPOINT else 3,),
                    dtype=np.float32,
                ),
                "aux": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(8,),
                    dtype=np.float32,
                ),
            }
        )

        self.np_random: np.random.Generator | None = None
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

        # These flags are placeholders for future corruption pipelines.
        self.vision_corrupted = False
        self.distance_corrupted = False
        self.ego_corrupted = False
        self.route_corrupted = False

        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_heading = 0.0
        self.agent_speed = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles: list[CircleObstacle] = []
        self.prev_goal_distance = 0.0
        self.last_quality_stats: dict[str, float] = {}
        self.last_risk_stats: dict[str, float] = {}

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.vision_corrupted = False
        self.distance_corrupted = False
        self.ego_corrupted = False
        self.route_corrupted = False

        self._sample_scene()
        self._apply_corruption_config()
        self.prev_goal_distance = self._goal_distance()

        obs = self._get_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = self._sanitize_action(action)
        accel_cmd = float(action[0])
        turn_cmd = float(action[1])

        self.agent_speed = self._clip(
            self.agent_speed + accel_cmd * MAX_ACCEL * DT,
            0.0,
            MAX_SPEED,
        )
        self.agent_heading = self._wrap_angle(
            self.agent_heading + turn_cmd * MAX_TURN_RATE * DT
        )

        dx = self.agent_speed * np.cos(self.agent_heading) * DT
        dy = self.agent_speed * np.sin(self.agent_heading) * DT
        self.agent_x = self._finite_or_zero(self.agent_x + dx)
        self.agent_y = self._finite_or_zero(self.agent_y + dy)

        self.step_count += 1

        goal_distance = self._goal_distance()
        progress_delta = self.prev_goal_distance - goal_distance

        reached_goal = self._check_goal(goal_distance)
        collided = self._check_collision()
        out_of_bounds = self._check_out_of_bounds()
        timeout = self.step_count >= MAX_EPISODE_STEPS

        terminated = reached_goal or collided or out_of_bounds
        truncated = timeout and not terminated

        reward, reward_components = self._compute_reward(
            action=action,
            goal_distance=goal_distance,
            progress_delta=progress_delta,
            reached_goal=reached_goal,
            collided=collided,
            out_of_bounds=out_of_bounds,
        )

        self.prev_action = action.astype(np.float32, copy=True)
        self.prev_goal_distance = goal_distance

        obs = self._get_obs()
        info = self._build_info()
        info.update(
            {
                "goal_reached": reached_goal,
                "collision": collided,
                "out_of_bounds": out_of_bounds,
                "timeout": timeout,
                "goal_distance": float(goal_distance),
                "progress_delta": float(progress_delta),
            }
        )
        info.update(reward_components)

        return obs, float(self._finite_or_zero(reward)), terminated, truncated, info

    # ---------------------------------------------------------------------
    # Observation builders
    # ---------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build a shape-stable multimodal observation dict."""
        obs = {
            "vision": self._build_vision_proxy(),
            "distance": self._build_distance_proxy(),
            "ego": self._build_ego_proxy(),
            "route": self._build_route_proxy(),
            "aux": self._build_aux_state(),
        }

        # Final pass: every returned tensor is float32 and finite.
        for key, value in obs.items():
            obs[key] = self._sanitize_vector(value, self.observation_space[key].shape[0])
        return obs

    def set_corruption_mode(self, corruption_mode: str) -> None:
        """Convenience helper for Stage 0 validation and later corruption studies."""
        if corruption_mode == "clean":
            self.corruption_config = {}
        elif corruption_mode == "corrupted":
            self.corruption_config = {
                "vision": True,
                "distance": True,
                "ego": True,
                "route": True,
            }
        else:
            raise ValueError(f"Unsupported corruption_mode: {corruption_mode}")

    def _build_vision_proxy(self) -> np.ndarray:
        """Sector-based occupancy proxy in the forward field of view.

        Each bin stores the maximum obstacle occupancy score in that sector:
        0 -> free, 1 -> near obstacle center / high occupancy.
        """
        features = np.zeros(VISION_NUM_SECTORS, dtype=np.float32)
        sector_edges = np.linspace(
            -VISION_FOV / 2.0, VISION_FOV / 2.0, VISION_NUM_SECTORS + 1
        )

        for obstacle in self.obstacles:
            rel = np.array([obstacle.x - self.agent_x, obstacle.y - self.agent_y])
            distance = self._safe_norm(rel)
            if distance > VISION_RANGE + obstacle.radius:
                continue

            rel_angle = self._relative_angle(np.arctan2(rel[1], rel[0]))
            if rel_angle < -VISION_FOV / 2.0 or rel_angle > VISION_FOV / 2.0:
                continue

            sector_idx = np.searchsorted(sector_edges, rel_angle, side="right") - 1
            sector_idx = int(np.clip(sector_idx, 0, VISION_NUM_SECTORS - 1))

            edge_distance = max(distance - obstacle.radius, 0.0)
            occupancy = 1.0 - self._clip(edge_distance / VISION_RANGE, 0.0, 1.0)
            features[sector_idx] = max(features[sector_idx], occupancy)

        features = self._apply_modality_corruption(
            features,
            is_corrupted=self.vision_corrupted,
            low=0.0,
            high=1.0,
        )
        return self._clip_array(features, 0.0, 1.0)

    def _build_distance_proxy(self) -> np.ndarray:
        """Ray-based normalized distance readings in the forward field of view.

        1.0 means no obstacle encountered within range.
        Smaller values indicate closer obstacles.
        """
        features = np.ones(DISTANCE_NUM_RAYS, dtype=np.float32)
        ray_offsets = np.linspace(
            -DISTANCE_FOV / 2.0, DISTANCE_FOV / 2.0, DISTANCE_NUM_RAYS
        )

        for ray_idx, offset in enumerate(ray_offsets):
            ray_heading = self.agent_heading + offset
            ray_dir = np.array([np.cos(ray_heading), np.sin(ray_heading)], dtype=np.float32)
            min_distance = DISTANCE_RANGE

            for obstacle in self.obstacles:
                hit_distance = self._ray_circle_intersection_distance(
                    origin=np.array([self.agent_x, self.agent_y], dtype=np.float32),
                    direction=ray_dir,
                    center=np.array([obstacle.x, obstacle.y], dtype=np.float32),
                    radius=obstacle.radius,
                    max_range=DISTANCE_RANGE,
                )
                if hit_distance is not None:
                    min_distance = min(min_distance, hit_distance)

            features[ray_idx] = self._clip(min_distance / DISTANCE_RANGE, 0.0, 1.0)

        features = self._apply_modality_corruption(
            features,
            is_corrupted=self.distance_corrupted,
            low=0.0,
            high=1.0,
        )
        return self._clip_array(features, 0.0, 1.0)

    def _build_ego_proxy(self) -> np.ndarray:
        """Ego-centric features for policy-side control."""
        speed_norm = self._clip(self.agent_speed / MAX_SPEED, 0.0, 1.0)
        features = np.array(
            [
                speed_norm,
                np.sin(self.agent_heading),
                np.cos(self.agent_heading),
                self.prev_action[0],
                self.prev_action[1],
            ],
            dtype=np.float32,
        )
        features = self._apply_modality_corruption(
            features,
            is_corrupted=self.ego_corrupted,
            low=-1.0,
            high=1.0,
        )
        return self._clip_array(features, -1.0, 1.0)

    def _build_route_proxy(self) -> np.ndarray:
        """Goal-relative navigation guidance features."""
        goal_vector = self.goal - np.array([self.agent_x, self.agent_y], dtype=np.float32)
        goal_distance = self._safe_norm(goal_vector)
        goal_bearing = np.arctan2(goal_vector[1], goal_vector[0])
        rel_goal_angle = self._relative_angle(goal_bearing)

        goal_distance_norm = self._clip(
            goal_distance / (np.sqrt(2.0) * MAP_SIZE),
            0.0,
            1.0,
        )

        features = [
            np.sin(rel_goal_angle),
            np.cos(rel_goal_angle),
            goal_distance_norm,
        ]

        if ROUTE_INCLUDE_WAYPOINT:
            # Placeholder waypoint proxy: same as the immediate goal direction.
            # This keeps the shape future-proof for route-planner expansion.
            features.extend(
                [
                    np.sin(rel_goal_angle),
                    np.cos(rel_goal_angle),
                ]
            )

        route_features = np.asarray(features, dtype=np.float32)
        route_features = self._apply_modality_corruption(
            route_features,
            is_corrupted=self.route_corrupted,
            low=-1.0,
            high=1.0,
        )
        return self._clip_array(route_features, -1.0, 1.0)

    def _build_aux_state(self) -> np.ndarray:
        """Auxiliary state for reliability / risk / corruption experiments.

        This is intentionally separate from the four policy input modalities.
        """
        goal_vector = self.goal - np.array([self.agent_x, self.agent_y], dtype=np.float32)
        goal_distance = self._safe_norm(goal_vector)
        goal_bearing = np.arctan2(goal_vector[1], goal_vector[0])
        heading_error = self._relative_angle(goal_bearing)

        obstacle_proximity = 1.0 - np.min(self._build_distance_proxy())
        self.last_quality_stats = self._compute_quality_stats()
        self.last_risk_stats = self._compute_risk_stats(obstacle_proximity, goal_distance)

        features = np.array(
            [
                self.last_quality_stats["vision_quality"],
                self.last_quality_stats["distance_quality"],
                1.0 if self.ego_corrupted else 0.0,
                1.0 if self.route_corrupted else 0.0,
                self._clip(self.agent_speed / MAX_SPEED, 0.0, 1.0),
                self._clip(heading_error / np.pi, -1.0, 1.0),
                self.last_risk_stats["obstacle_risk"],
                self._clip(goal_distance / (np.sqrt(2.0) * MAP_SIZE), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return self._clip_array(features, -1.0, 1.0)

    # ---------------------------------------------------------------------
    # Scene generation
    # ---------------------------------------------------------------------

    def _sample_scene(self) -> None:
        """Sample agent, goal, and obstacle configuration with separation rules."""
        assert self.np_random is not None, "Random generator must be initialized in reset()."

        for _ in range(MAX_RESET_SAMPLING_TRIES):
            agent_xy = self._sample_point(margin=SEED_OBSTACLE_MARGIN)
            goal_xy = self._sample_point(margin=SEED_OBSTACLE_MARGIN)

            if self._safe_norm(goal_xy - agent_xy) < MIN_GOAL_AGENT_DISTANCE:
                continue

            obstacles: list[CircleObstacle] = []
            valid = True
            for _ob_idx in range(NUM_OBSTACLES):
                placed = False
                for _ in range(MAX_RESET_SAMPLING_TRIES):
                    obs_xy = self._sample_point(margin=SEED_OBSTACLE_MARGIN)
                    if self._safe_norm(obs_xy - agent_xy) < MIN_ENTITY_SEPARATION:
                        continue
                    if self._safe_norm(obs_xy - goal_xy) < MIN_ENTITY_SEPARATION:
                        continue
                    if any(
                        self._safe_norm(obs_xy - np.array([o.x, o.y])) < MIN_ENTITY_SEPARATION
                        for o in obstacles
                    ):
                        continue
                    obstacles.append(
                        CircleObstacle(
                            x=float(obs_xy[0]),
                            y=float(obs_xy[1]),
                            radius=OBSTACLE_RADIUS,
                        )
                    )
                    placed = True
                    break

                if not placed:
                    valid = False
                    break

            if valid:
                self.agent_x = float(agent_xy[0])
                self.agent_y = float(agent_xy[1])
                self.agent_heading = float(self.np_random.uniform(-np.pi, np.pi))
                self.agent_speed = 0.0
                self.goal = goal_xy.astype(np.float32)
                self.obstacles = obstacles
                return

        raise RuntimeError("Failed to sample a valid RCSToyNavEnv scene.")

    # ---------------------------------------------------------------------
    # Info / termination helpers
    # ---------------------------------------------------------------------

    def _build_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "true_state": {
                "x": float(self.agent_x),
                "y": float(self.agent_y),
                "heading": float(self.agent_heading),
                "speed": float(self.agent_speed),
            },
            "goal": self.goal.astype(np.float32, copy=True),
            "num_obstacles": len(self.obstacles),
            "corruption_flags": {
                "vision_corrupted": bool(self.vision_corrupted),
                "distance_corrupted": bool(self.distance_corrupted),
                "ego_corrupted": bool(self.ego_corrupted),
                "route_corrupted": bool(self.route_corrupted),
            },
            "quality_stats": dict(self.last_quality_stats),
            "risk_stats": dict(self.last_risk_stats),
        }

    def _goal_distance(self) -> float:
        return float(
            self._safe_norm(self.goal - np.array([self.agent_x, self.agent_y], dtype=np.float32))
        )

    def _check_goal(self, goal_distance: float | None = None) -> bool:
        if goal_distance is None:
            goal_distance = self._goal_distance()
        return bool(goal_distance <= GOAL_RADIUS)

    def _check_collision(self) -> bool:
        for obstacle in self.obstacles:
            distance = self._safe_norm(
                np.array([obstacle.x - self.agent_x, obstacle.y - self.agent_y], dtype=np.float32)
            )
            if distance <= AGENT_RADIUS + obstacle.radius:
                return True
        return False

    def _check_out_of_bounds(self) -> bool:
        return (
            self.agent_x < -HALF_MAP
            or self.agent_x > HALF_MAP
            or self.agent_y < -HALF_MAP
            or self.agent_y > HALF_MAP
        )

    def _compute_progress_reward(self, progress_delta: float) -> float:
        raw_reward = PROGRESS_REWARD_SCALE * progress_delta
        return self._clip(raw_reward, -PROGRESS_REWARD_CLIP, PROGRESS_REWARD_CLIP)

    def _compute_action_change_penalty(self, action: np.ndarray) -> float:
        action_delta = float(self._safe_norm(action - self.prev_action))
        raw_penalty = ACTION_CHANGE_PENALTY_SCALE * action_delta
        return -self._clip(raw_penalty, 0.0, ACTION_CHANGE_PENALTY_CLIP)

    def _compute_idle_penalty(self, progress_delta: float) -> float:
        is_idle = self.agent_speed < IDLE_SPEED_THRESHOLD and abs(progress_delta) < IDLE_PROGRESS_THRESHOLD
        return float(IDLE_PENALTY if is_idle else 0.0)

    def _compute_reward(
        self,
        action: np.ndarray,
        goal_distance: float,
        progress_delta: float,
        reached_goal: bool,
        collided: bool,
        out_of_bounds: bool,
    ) -> tuple[float, dict[str, float]]:
        progress_reward = self._compute_progress_reward(progress_delta)
        goal_reward = float(REWARD_GOAL if reached_goal else 0.0)
        collision_penalty = float(REWARD_COLLISION if collided else 0.0)
        out_of_bounds_penalty = float(REWARD_OUT_OF_BOUNDS if out_of_bounds else 0.0)
        action_change_penalty = self._compute_action_change_penalty(action)
        idle_penalty = self._compute_idle_penalty(progress_delta)

        reward_total = (
            progress_reward
            + goal_reward
            + collision_penalty
            + out_of_bounds_penalty
            + action_change_penalty
            + idle_penalty
        )
        reward_total = self._clip(reward_total, -REWARD_TOTAL_CLIP, REWARD_TOTAL_CLIP)
        reward_total = self._finite_or_zero(reward_total)

        reward_components = {
            "reward/progress": float(progress_reward),
            "reward/goal": float(goal_reward),
            "reward/collision": float(collision_penalty),
            "reward/out_of_bounds": float(out_of_bounds_penalty),
            "reward/action_change": float(action_change_penalty),
            "reward/idle": float(idle_penalty),
            "reward/total": float(reward_total),
            "metric/goal_distance": float(goal_distance),
            "metric/progress_delta": float(progress_delta),
        }
        return float(reward_total), reward_components

    def _apply_corruption_config(self) -> None:
        """Turn corruption flags on/off from the provided config."""
        self.vision_corrupted = bool(self.corruption_config.get("vision", False))
        self.distance_corrupted = bool(self.corruption_config.get("distance", False))
        self.ego_corrupted = bool(self.corruption_config.get("ego", False))
        self.route_corrupted = bool(self.corruption_config.get("route", False))

    def _apply_modality_corruption(
        self,
        features: np.ndarray,
        is_corrupted: bool,
        low: float,
        high: float,
    ) -> np.ndarray:
        """Simple additive corruption placeholder that preserves shape stability."""
        if not is_corrupted:
            return features.astype(np.float32)
        assert self.np_random is not None
        noise = self.np_random.normal(loc=0.0, scale=CORRUPTION_NOISE_STD, size=features.shape)
        corrupted = features + noise.astype(np.float32)
        return np.clip(corrupted, low, high).astype(np.float32)

    def _compute_quality_stats(self) -> dict[str, float]:
        """Basic placeholder quality signals for Stage 0 logging."""
        vision_quality = 1.0 - (QUALITY_DEGRADATION_ON_CORRUPTION if self.vision_corrupted else 0.0)
        distance_quality = 1.0 - (QUALITY_DEGRADATION_ON_CORRUPTION if self.distance_corrupted else 0.0)
        ego_quality = 1.0 - (QUALITY_DEGRADATION_ON_CORRUPTION if self.ego_corrupted else 0.0)
        route_quality = 1.0 - (QUALITY_DEGRADATION_ON_CORRUPTION if self.route_corrupted else 0.0)
        return {
            "vision_quality": float(np.clip(vision_quality, 0.0, 1.0)),
            "distance_quality": float(np.clip(distance_quality, 0.0, 1.0)),
            "ego_quality": float(np.clip(ego_quality, 0.0, 1.0)),
            "route_quality": float(np.clip(route_quality, 0.0, 1.0)),
        }

    def _compute_risk_stats(
        self,
        obstacle_proximity: float,
        goal_distance: float,
    ) -> dict[str, float]:
        """Basic placeholder risk signals for Stage 0 logging."""
        corruption_count = sum(
            [
                int(self.vision_corrupted),
                int(self.distance_corrupted),
                int(self.ego_corrupted),
                int(self.route_corrupted),
            ]
        )
        normalized_goal_distance = self._clip(goal_distance / (np.sqrt(2.0) * MAP_SIZE), 0.0, 1.0)
        corruption_risk = self._clip(
            RISK_BASELINE + corruption_count * RISK_CORRUPTION_INCREMENT,
            0.0,
            1.0,
        )
        obstacle_risk = self._clip(obstacle_proximity, 0.0, 1.0)
        goal_risk = normalized_goal_distance
        fused_risk = self._clip(
            0.5 * obstacle_risk + 0.3 * corruption_risk + 0.2 * goal_risk,
            0.0,
            1.0,
        )
        return {
            "obstacle_risk": float(obstacle_risk),
            "corruption_risk": float(corruption_risk),
            "goal_risk": float(goal_risk),
            "fused_risk": float(fused_risk),
        }

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------

    def _sample_point(self, margin: float) -> np.ndarray:
        assert self.np_random is not None
        low = -HALF_MAP + margin
        high = HALF_MAP - margin
        point = self.np_random.uniform(low=low, high=high, size=(2,))
        return point.astype(np.float32)

    def _relative_angle(self, world_angle: float) -> float:
        return self._wrap_angle(world_angle - self.agent_heading)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _ray_circle_intersection_distance(
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
        max_range: float,
    ) -> float | None:
        """Return nearest forward ray-circle hit distance, or None if no hit."""
        oc = origin - center
        a = float(np.dot(direction, direction))
        b = float(2.0 * np.dot(direction, oc))
        c = float(np.dot(oc, oc) - radius * radius)
        discriminant = b * b - 4.0 * a * c

        if discriminant < 0.0:
            return None

        sqrt_disc = np.sqrt(max(discriminant, 0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a + EPS)
        t2 = (-b + sqrt_disc) / (2.0 * a + EPS)

        candidates = [t for t in (t1, t2) if 0.0 <= t <= max_range]
        if not candidates:
            return None
        return float(min(candidates))

    # ---------------------------------------------------------------------
    # Safety / numeric helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _safe_norm(vec: np.ndarray) -> float:
        return float(np.sqrt(np.maximum(np.sum(np.square(vec)), 0.0)))

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return float(np.clip(value, low, high))

    @staticmethod
    def _clip_array(array: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.clip(array, low, high).astype(np.float32)

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        return float(value) if np.isfinite(value) else 0.0

    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        """Ensure actions are always finite float32 vectors with correct shape."""
        action = np.asarray(action, dtype=np.float32).reshape(2,)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)

    @staticmethod
    def _sanitize_vector(array: np.ndarray, expected_dim: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32).reshape(expected_dim,)
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=FLOAT32_MAX,
            neginf=FLOAT32_MIN,
        )
        return array.astype(np.float32)


if __name__ == "__main__":
    env = RCSToyNavEnv()
    obs, info = env.reset(seed=0)
    print("Reset observation keys:", obs.keys())
    print("Shapes:", {k: v.shape for k, v in obs.items()})

    for step_idx in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step_idx} reward={reward:.3f} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            break
