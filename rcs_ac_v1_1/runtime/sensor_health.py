from __future__ import annotations

import math

import numpy as np

from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, DEFAULT_BEV_SPEC, ego_xy_to_bev_row_col
from rcs_ac_v1_1.runtime.state import V11EgoState, V11RouteContext, as_route_points, is_route_context_valid

SENSOR_HEALTH_DIM = 15


def build_sensor_health(
    *,
    rgb_front: np.ndarray | None,
    lidar_points_ego: np.ndarray | None,
    route_context: V11RouteContext | None,
    ego_state: V11EgoState | None,
    camera_latency_ms: float = 0.0,
    lidar_latency_ms: float = 0.0,
    route_latency_ms: float = 0.0,
    weather_severity_norm: float = 0.0,
    corruption_severity_norm: float = 0.0,
    camera_dropout: bool = False,
    lidar_dropout: bool = False,
    route_dropout: bool = False,
    spec: BEVSpec = DEFAULT_BEV_SPEC,
) -> np.ndarray:
    camera_available, camera_quality = _camera_health(rgb_front, camera_dropout)
    lidar_available, lidar_quality = _lidar_health(lidar_points_ego, lidar_dropout, spec)
    route_available, route_quality = _route_health(route_context, route_dropout)
    ego_speed_available, ego_imu_available, ego_quality = _ego_health(ego_state)
    health = np.asarray(
        [
            camera_available,
            camera_quality,
            _latency_norm(camera_latency_ms),
            lidar_available,
            lidar_quality,
            _latency_norm(lidar_latency_ms),
            route_available,
            route_quality,
            _latency_norm(route_latency_ms),
            ego_speed_available,
            ego_imu_available,
            ego_quality,
            _clip01(weather_severity_norm),
            _clip01(corruption_severity_norm),
            1.0,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(health).all():
        raise ValueError("sensor_health contains NaN or inf")
    return np.clip(health, 0.0, 1.0).astype(np.float32, copy=False)


def _camera_health(rgb_front: np.ndarray | None, dropout: bool) -> tuple[float, float]:
    if rgb_front is None or bool(dropout):
        return 0.0, 0.0
    image = np.asarray(rgb_front)
    if image.size == 0:
        return 0.0, 0.0
    finite = np.isfinite(image)
    finite_ratio = float(finite.mean())
    if finite_ratio <= 0.99:
        return 0.0, 0.0
    image_f = image.astype(np.float32, copy=False)
    if image_f.max(initial=0.0) > 1.5:
        image_f = image_f / 255.0
    image_f = np.clip(image_f, 0.0, 1.0)
    if image_f.ndim == 3 and image_f.shape[0] in (1, 3):
        luma = image_f.mean(axis=0)
    elif image_f.ndim == 3 and image_f.shape[-1] in (1, 3):
        luma = image_f.mean(axis=-1)
    elif image_f.ndim == 2:
        luma = image_f
    else:
        return 0.0, 0.0
    mean_luma = float(np.mean(luma))
    std_luma = float(np.std(luma))
    brightness_score = 1.0 - min(abs(mean_luma - 0.5) / 0.5, 1.0)
    contrast_score = _clip01(std_luma / 0.25)
    gx = float(np.abs(np.diff(luma, axis=-1)).mean()) if luma.shape[-1] > 1 else 0.0
    gy = float(np.abs(np.diff(luma, axis=-2)).mean()) if luma.shape[-2] > 1 else 0.0
    edge_score = _clip01((gx + gy) / 0.10)
    quality = finite_ratio * float(np.mean([brightness_score, contrast_score, edge_score]))
    return 1.0, _clip01(quality)


def _lidar_health(
    lidar_points_ego: np.ndarray | None,
    dropout: bool,
    spec: BEVSpec,
    *,
    min_lidar_points: int = 32,
    expected_lidar_points: int = 4096,
) -> tuple[float, float]:
    if lidar_points_ego is None or bool(dropout):
        return 0.0, 0.0
    points = np.asarray(lidar_points_ego, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        return 0.0, 0.0
    finite_mask = np.isfinite(points[:, :3]).all(axis=1)
    valid = points[finite_mask, :3]
    valid_count = int(valid.shape[0])
    if valid_count < int(min_lidar_points):
        return 0.0, 0.0
    finite_ratio = float(finite_mask.mean())
    inside = 0
    for x_m, y_m, _ in valid:
        _, _, ok = ego_xy_to_bev_row_col(float(x_m), float(y_m), spec)
        inside += int(ok)
    count_score = _clip01(math.log1p(valid_count) / math.log1p(float(expected_lidar_points)))
    range_score = _clip01(inside / max(valid_count, 1))
    height_score = 1.0 if float(np.std(valid[:, 2])) > 1.0e-4 else 0.5
    quality = finite_ratio * float(np.mean([count_score, range_score, height_score]))
    return 1.0, _clip01(quality)


def _route_health(route_context: V11RouteContext | None, dropout: bool) -> tuple[float, float]:
    if bool(dropout) or not is_route_context_valid(route_context):
        return 0.0, 0.0
    assert route_context is not None
    points = as_route_points(route_context)
    segment = np.diff(points, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    continuity = float(np.mean((lengths >= 0.1) & (lengths <= 8.0))) if lengths.size else 0.0
    forward = float(np.mean(points[:, 0] >= -1.0))
    total_length = float(lengths.sum())
    length_score = _clip01(total_length / 20.0)
    heading_score = 1.0 - _clip01(abs(float(route_context.heading_error_rad)) / math.pi)
    quality = float(np.mean([continuity, forward, length_score, heading_score]))
    return 1.0, _clip01(quality)


def _ego_health(ego_state: V11EgoState | None) -> tuple[float, float, float]:
    if ego_state is None:
        return 0.0, 0.0, 0.0
    speed_ok = float(np.isfinite(float(ego_state.speed_kmh)))
    imu_ok = float(np.isfinite(float(ego_state.yaw_rate_rad_s)) and np.isfinite(float(ego_state.acceleration_mps2)))
    quality = 1.0 if speed_ok and imu_ok else (0.5 if speed_ok or imu_ok else 0.0)
    return speed_ok, imu_ok, quality


def _latency_norm(latency_ms: float, max_latency_ms: float = 500.0) -> float:
    return _clip01(float(latency_ms) / float(max_latency_ms))


def _clip01(value: float) -> float:
    if not np.isfinite(float(value)):
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))
