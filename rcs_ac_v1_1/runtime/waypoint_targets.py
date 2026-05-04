from __future__ import annotations

import numpy as np


def build_distance_based_expert_waypoints(
    expert_path_ego: np.ndarray | None,
    *,
    distances_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0),
    fallback_route_ego: np.ndarray | None = None,
    min_path_length_m: float = 4.0,
    allow_extrapolation: bool = True,
    max_extrapolation_m: float = 4.0,
) -> tuple[np.ndarray, float]:
    path = _select_valid_path(expert_path_ego, min_path_length_m)
    if path is None:
        path = _select_valid_path(fallback_route_ego, min_path_length_m)
    if path is None:
        return np.zeros((len(distances_m), 2), dtype=np.float32), np.float32(0.0)
    waypoints = _interpolate_distances(
        path,
        distances_m=distances_m,
        allow_extrapolation=allow_extrapolation,
        max_extrapolation_m=max_extrapolation_m,
    )
    available = validate_waypoint_target(waypoints)
    if available <= 0.0:
        return np.zeros((len(distances_m), 2), dtype=np.float32), np.float32(0.0)
    return waypoints.astype(np.float32, copy=False), np.float32(1.0)


def validate_waypoint_target(
    waypoints: np.ndarray,
    *,
    min_gap_m: float = 0.25,
    max_abs_lateral_m: float = 8.0,
) -> float:
    arr = np.asarray(waypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        return np.float32(0.0)
    if not np.any(arr[:, 0] > 0.0) or arr[0, 0] <= 0.0:
        return np.float32(0.0)
    if arr.shape[0] > 1:
        segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        if not np.all(segment_lengths >= float(min_gap_m)):
            return np.float32(0.0)
    if not np.all(np.abs(arr[:, 1]) <= float(max_abs_lateral_m)):
        return np.float32(0.0)
    return np.float32(1.0)


def build_expert_target_speed(
    *,
    expert_speed_kmh: float | None,
    route_target_speed_kmh: float | None,
    observed_speed_kmh: float | None = None,
    default_speed_kmh: float = 12.0,
    min_speed_kmh: float = 0.0,
    max_speed_kmh: float = 28.0,
) -> np.ndarray:
    value = default_speed_kmh
    if expert_speed_kmh is not None and np.isfinite(float(expert_speed_kmh)) and float(expert_speed_kmh) > 0.0:
        value = float(expert_speed_kmh)
    elif route_target_speed_kmh is not None and np.isfinite(float(route_target_speed_kmh)) and float(route_target_speed_kmh) > 0.0:
        value = float(route_target_speed_kmh)
    elif observed_speed_kmh is not None and np.isfinite(float(observed_speed_kmh)) and float(observed_speed_kmh) > 0.0:
        value = float(observed_speed_kmh)
    return np.asarray([np.clip(value, float(min_speed_kmh), float(max_speed_kmh))], dtype=np.float32)


def _select_valid_path(path: np.ndarray | None, min_path_length_m: float) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[arr[:, 0] >= -0.5]
    if arr.shape[0] < 2 or np.all(arr[:, 0] <= 0.0):
        return None
    anchored = np.vstack([np.zeros((1, 2), dtype=np.float32), arr])
    length = float(np.linalg.norm(np.diff(anchored, axis=0), axis=1).sum())
    if length < float(min_path_length_m):
        return None
    return arr


def _interpolate_distances(
    path: np.ndarray,
    *,
    distances_m: tuple[float, ...],
    allow_extrapolation: bool,
    max_extrapolation_m: float,
) -> np.ndarray:
    anchored = np.vstack([np.zeros((1, 2), dtype=np.float32), path])
    segments = np.diff(anchored, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    result: list[np.ndarray] = []
    for distance in distances_m:
        d = float(distance)
        if d <= total:
            idx = int(np.searchsorted(cumulative, d, side="right") - 1)
            idx = min(idx, len(segments) - 1)
            seg_len = max(float(lengths[idx]), 1.0e-6)
            alpha = (d - float(cumulative[idx])) / seg_len
            result.append(anchored[idx] + alpha * segments[idx])
        else:
            missing = d - total
            if not allow_extrapolation or missing > float(max_extrapolation_m):
                return np.zeros((len(distances_m), 2), dtype=np.float32)
            last = segments[-1]
            norm = max(float(np.linalg.norm(last)), 1.0e-6)
            result.append(anchored[-1] + (last / norm) * missing)
    return np.asarray(result, dtype=np.float32)
