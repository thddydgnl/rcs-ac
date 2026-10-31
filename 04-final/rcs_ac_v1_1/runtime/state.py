from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class V11EgoState:
    speed_kmh: float
    yaw_rate_rad_s: float
    acceleration_mps2: float = 0.0
    previous_steer: float = 0.0
    previous_throttle: float = 0.0
    previous_brake: float = 0.0
    previous_longitudinal: float = 0.0
    route_progress: float = 0.0
    distance_to_next_turn_m: float = 0.0

    @property
    def effective_previous_longitudinal(self) -> float:
        if np.isfinite(float(self.previous_longitudinal)) and abs(float(self.previous_longitudinal)) > 1.0e-6:
            return float(self.previous_longitudinal)
        return float(self.previous_throttle) - float(self.previous_brake)


@dataclass(frozen=True)
class V11RouteContext:
    route_points_ego: np.ndarray
    target_point_ego: tuple[float, float] | None
    maneuver: str
    route_progress: float = 0.0
    distance_to_next_turn_m: float = 0.0
    target_speed_kmh: float = 0.0
    curvature: float = 0.0
    heading_error_rad: float = 0.0
    available: bool = True


@dataclass(frozen=True)
class V11SensorPacket:
    rgb_front: np.ndarray | None
    lidar_points_ego: np.ndarray | None
    camera_latency_ms: float = 0.0
    lidar_latency_ms: float = 0.0
    route_latency_ms: float = 0.0
    weather_severity_norm: float = 0.0
    corruption_severity_norm: float = 0.0
    camera_dropout: bool = False
    lidar_dropout: bool = False
    route_dropout: bool = False


def as_route_points(route_context: V11RouteContext | None) -> np.ndarray:
    if route_context is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.asarray(route_context.route_points_ego, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return points


def is_route_context_valid(route_context: V11RouteContext | None) -> bool:
    if route_context is None or not bool(route_context.available):
        return False
    points = as_route_points(route_context)
    if points.shape[0] < 2 or not np.isfinite(points).all():
        return False
    return bool(np.any(points[:, 0] > 0.0))
