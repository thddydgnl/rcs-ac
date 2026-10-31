from __future__ import annotations

import math
from typing import Any

import numpy as np

from rcs_ac_v1_1.runtime.state import V11RouteContext


def world_xy_to_ego_xy(
    world_xy: np.ndarray,
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
) -> np.ndarray:
    points = np.asarray(world_xy, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    dx = points[:, 0].astype(np.float32) - np.float32(ego_x)
    dy = points[:, 1].astype(np.float32) - np.float32(ego_y)
    yaw = float(ego_yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_ego = cos_yaw * dx + sin_yaw * dy
    y_right = -sin_yaw * dx + cos_yaw * dy
    y_left = -y_right
    result = np.stack([x_ego, y_left], axis=1).astype(np.float32, copy=False)
    return result[np.isfinite(result).all(axis=1)]


def build_route_context_from_info(
    info: dict[str, Any],
    *,
    default_target_speed_kmh: float = 12.0,
) -> tuple[V11RouteContext | None, dict[str, float | int | str]]:
    route_points = _extract_route_points_ego(info)
    target = _extract_target_point(info, route_points)
    maneuver = _extract_maneuver(info, route_points)
    debug: dict[str, float | int | str] = {
        "adapter_route_point_count": int(route_points.shape[0]),
        "adapter_route_available": int(route_points.shape[0] >= 2 and np.any(route_points[:, 0] > 0.0)),
        "adapter_target_x": float(target[0]) if target is not None else 0.0,
        "adapter_target_y": float(target[1]) if target is not None else 0.0,
        "adapter_maneuver": str(maneuver),
    }
    if int(debug["adapter_route_available"]) <= 0:
        return None, debug
    curvature = _estimate_curvature(route_points)
    heading_error = _extract_heading_error_rad(info, default=_estimate_heading_error(route_points))
    distance_to_next_turn = _extract_float(
        info,
        ("distance_to_next_turn_m", "target_distance", "min_target_distance"),
        default=_distance_to_turn(route_points),
    )
    context = V11RouteContext(
        route_points_ego=route_points,
        target_point_ego=target,
        maneuver=maneuver,
        route_progress=_extract_float(info, ("route_progress",), default=0.0),
        distance_to_next_turn_m=distance_to_next_turn,
        target_speed_kmh=_extract_float(
            info,
            ("target_speed_kmh", "target_speed", "route_target_speed_kmh"),
            default=default_target_speed_kmh,
        ),
        curvature=curvature,
        heading_error_rad=heading_error,
        available=True,
    )
    debug["adapter_route_curvature"] = float(curvature)
    debug["adapter_route_heading_error_rad"] = float(heading_error)
    debug["adapter_distance_to_next_turn_m"] = float(distance_to_next_turn)
    return context, debug


def build_route_context_from_obs_info(
    obs: dict[str, Any],
    info: dict[str, Any],
    *,
    default_target_speed_kmh: float = 12.0,
) -> tuple[V11RouteContext | None, dict[str, float | int | str]]:
    merged = dict(info)
    if not any(
        key in merged
        for key in (
            "route_points_ego",
            "route_waypoints_ego",
            "route_lookahead_points_ego",
            "route_local_points",
            "route_points_local",
        )
    ):
        route_vector_points = _extract_points_from_route_vector(obs.get("route"))
        if route_vector_points.shape[0] >= 2:
            merged["route_points_ego"] = route_vector_points
            merged["adapter_route_source"] = "obs_route_vector_lookahead"
    if not any(
        key in merged
        for key in (
            "route_points_ego",
            "route_waypoints_ego",
            "route_lookahead_points_ego",
            "route_local_points",
            "route_points_local",
            "target_x_local",
            "target_y_local",
            "route_target_x",
            "route_target_y",
        )
    ):
        route_vector_target = _extract_target_from_route_vector(obs.get("route"))
        if route_vector_target is not None:
            merged["target_x_local"] = float(route_vector_target[0])
            merged["target_y_local"] = float(route_vector_target[1])
            merged["adapter_route_source"] = "obs_route_vector"
    context, debug = build_route_context_from_info(merged, default_target_speed_kmh=default_target_speed_kmh)
    if "adapter_route_source" not in debug:
        debug["adapter_route_source"] = str(merged.get("adapter_route_source", "info"))
    return context, debug


def _extract_route_points_ego(info: dict[str, Any]) -> np.ndarray:
    for key in (
        "route_points_ego",
        "route_waypoints_ego",
        "route_lookahead_points_ego",
        "route_local_points",
        "route_points_local",
    ):
        if key in info:
            points = _coerce_points(info[key])
            if points.shape[0] >= 2:
                return points
    target = _extract_target_point(info, np.zeros((0, 2), dtype=np.float32))
    if target is None:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[0.0, 0.0], [float(target[0]), float(target[1])]], dtype=np.float32)


def _extract_target_from_route_vector(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    x = float(arr[0])
    y = float(arr[1])
    if abs(x) <= 1.0 and abs(y) <= 1.0 and arr.size >= 4:
        # Some shared_core route vectors store normalized target coordinates first.
        x *= 40.0
        y *= 12.0
    if x <= 0.0:
        return None
    return x, y


def _extract_points_from_route_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr).all():
        return np.zeros((0, 2), dtype=np.float32)
    points: list[tuple[float, float]] = []
    target = _extract_target_from_route_vector(arr[:2])
    if target is not None:
        points.append(target)
    for idx in (2, 4):
        if arr.size <= idx + 1:
            continue
        distance = float(arr[idx])
        heading = float(arr[idx + 1])
        if distance <= 0.0 or not np.isfinite(distance) or not np.isfinite(heading):
            continue
        points.append((distance * math.cos(heading), distance * math.sin(heading)))
    if len(points) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    result = np.asarray(points, dtype=np.float32)
    result = result[np.isfinite(result).all(axis=1)]
    result = result[result[:, 0] > 0.0]
    if result.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return result


def _coerce_points(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(value, str):
        return np.zeros((0, 2), dtype=np.float32)
    points = np.asarray(value, dtype=np.float32)
    if points.ndim == 1:
        if points.size < 2 or points.size % 2:
            return np.zeros((0, 2), dtype=np.float32)
        points = points.reshape(-1, 2)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    points = points[:, :2]
    return points[np.isfinite(points).all(axis=1)].astype(np.float32, copy=False)


def _extract_target_point(info: dict[str, Any], route_points: np.ndarray) -> tuple[float, float] | None:
    for x_key, y_key in (
        ("route_target_x", "route_target_y"),
        ("target_x_local", "target_y_local"),
        ("route_target_local_x", "route_target_local_y"),
        ("target_x", "target_y"),
    ):
        if x_key in info and y_key in info:
            x = _safe_float(info.get(x_key), default=np.nan)
            y = _safe_float(info.get(y_key), default=np.nan)
            if np.isfinite(x) and np.isfinite(y) and x > 0.0:
                return float(x), float(y)
    if route_points.shape[0] == 0:
        return None
    forward = route_points[route_points[:, 0] >= 0.0]
    if forward.shape[0] == 0:
        return None
    idx = int(np.argmin(np.abs(forward[:, 0] - 7.0)))
    return float(forward[idx, 0]), float(forward[idx, 1])


def _extract_maneuver(info: dict[str, Any], route_points: np.ndarray) -> str:
    raw = str(info.get("maneuver_command", info.get("maneuver", ""))).strip().lower()
    if raw in {"left", "right", "straight", "lane_follow"}:
        return raw
    if route_points.shape[0] >= 2:
        y_mean = float(np.mean(route_points[min(1, route_points.shape[0] - 1) :, 1]))
        if y_mean > 1.0:
            return "left"
        if y_mean < -1.0:
            return "right"
    return "straight"


def _estimate_curvature(route_points: np.ndarray) -> float:
    if route_points.shape[0] < 3:
        return 0.0
    p0, p1, p2 = route_points[0], route_points[route_points.shape[0] // 2], route_points[-1]
    a = float(np.linalg.norm(p1 - p0))
    b = float(np.linalg.norm(p2 - p1))
    c = float(np.linalg.norm(p2 - p0))
    denom = max(a * b * c, 1.0e-6)
    area2 = abs(float(np.cross(p1 - p0, p2 - p0)))
    return float(np.clip(2.0 * area2 / denom, -1.0, 1.0))


def _estimate_heading_error(route_points: np.ndarray) -> float:
    forward = route_points[route_points[:, 0] > 0.5]
    if forward.shape[0] == 0:
        return 0.0
    target = forward[0]
    return float(math.atan2(float(target[1]), float(target[0])))


def _distance_to_turn(route_points: np.ndarray, *, lateral_threshold_m: float = 1.0) -> float:
    if route_points.shape[0] < 2:
        return 0.0
    candidates = route_points[np.abs(route_points[:, 1]) >= float(lateral_threshold_m)]
    if candidates.shape[0] == 0:
        return 100.0
    return float(max(candidates[0, 0], 0.0))


def _extract_float(info: dict[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    for key in keys:
        if key in info:
            value = _safe_float(info.get(key), default=np.nan)
            if np.isfinite(value):
                return float(value)
    return float(default)


def _safe_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _extract_heading_error_rad(info: dict[str, Any], *, default: float) -> float:
    for key in ("heading_error_rad", "route_heading_error_rad"):
        value = _safe_float(info.get(key), default=np.nan)
        if np.isfinite(value):
            return float(value)
    for key in ("heading_error", "route_heading_error"):
        value = _safe_float(info.get(key), default=np.nan)
        if np.isfinite(value):
            return math.radians(float(value))
    return float(default)
