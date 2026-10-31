from __future__ import annotations

import math

import numpy as np

from rcs_ac_v1_1.runtime.state import V11EgoState, V11RouteContext, is_route_context_valid


def build_ego_scalar(
    ego_state: V11EgoState,
    *,
    max_speed_kmh: float = 40.0,
    max_abs_yaw_rate_rad_s: float = 1.5,
    max_distance_to_turn_m: float = 100.0,
) -> np.ndarray:
    values = np.asarray(
        [
            _clip(float(ego_state.speed_kmh) / float(max_speed_kmh), 0.0, 1.0),
            _clip(float(ego_state.yaw_rate_rad_s) / float(max_abs_yaw_rate_rad_s), -1.0, 1.0),
            _clip(float(ego_state.previous_steer), -1.0, 1.0),
            _clip(float(ego_state.effective_previous_longitudinal), -1.0, 1.0),
            _clip(float(ego_state.route_progress), 0.0, 1.0),
            _clip(float(ego_state.distance_to_next_turn_m) / float(max_distance_to_turn_m), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("ego_scalar contains NaN or inf")
    return values


def build_route_scalar(
    route_context: V11RouteContext | None,
    *,
    max_target_x_m: float = 40.0,
    max_abs_target_y_m: float = 12.0,
    max_abs_curvature: float = 0.5,
    max_distance_to_turn_m: float = 100.0,
) -> np.ndarray:
    if not is_route_context_valid(route_context):
        values = np.zeros(10, dtype=np.float32)
        values[3] = 1.0
        return values
    assert route_context is not None
    target = route_context.target_point_ego
    if target is None:
        points = np.asarray(route_context.route_points_ego, dtype=np.float32)
        forward = points[points[:, 0] >= 0.0]
        if forward.size:
            idx = int(np.argmin(np.abs(forward[:, 0] - 6.0)))
            target_x, target_y = float(forward[idx, 0]), float(forward[idx, 1])
        else:
            target_x, target_y = 0.0, 0.0
    else:
        target_x, target_y = float(target[0]), float(target[1])
    maneuver = str(route_context.maneuver).lower()
    heading = float(route_context.heading_error_rad)
    values = np.asarray(
        [
            _clip(target_x / float(max_target_x_m), 0.0, 1.0),
            _clip(target_y / float(max_abs_target_y_m), -1.0, 1.0),
            math.sin(heading),
            math.cos(heading),
            _clip(float(route_context.curvature) / float(max_abs_curvature), -1.0, 1.0),
            _clip(float(route_context.distance_to_next_turn_m) / float(max_distance_to_turn_m), 0.0, 1.0),
            1.0 if maneuver == "left" else 0.0,
            1.0 if maneuver == "right" else 0.0,
            1.0 if maneuver in {"straight", "lane_follow"} else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("route_scalar contains NaN or inf")
    return values


def _clip(value: float, low: float, high: float) -> float:
    if not np.isfinite(float(value)):
        return 0.0
    return float(np.clip(float(value), float(low), float(high)))
