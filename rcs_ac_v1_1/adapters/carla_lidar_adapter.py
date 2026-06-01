from __future__ import annotations

from typing import Any

import numpy as np

from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, ego_xy_to_bev_row_col


def extract_lidar_points_ego(obs: dict[str, Any], info: dict[str, Any] | None = None) -> tuple[np.ndarray | None, dict[str, float | int]]:
    info = info or {}
    for source in (obs, info):
        for key in (
            "lidar_points_ego",
            "lidar_points",
            "lidar_point_cloud",
            "lidar_raw",
            "lidar",
        ):
            if key in source:
                if _looks_like_legacy_bev(source[key]):
                    return None, {
                        "adapter_lidar_point_count": 0,
                        "adapter_lidar_finite_ratio": 1.0,
                        "adapter_lidar_coord_checked": 0,
                        "adapter_lidar_legacy_bev_detected": 1,
                    }
                points = coerce_lidar_points(source[key])
                if points is not None:
                    return points, _debug_lidar(points)
    return None, {
        "adapter_lidar_point_count": 0,
        "adapter_lidar_finite_ratio": 0.0,
        "adapter_lidar_coord_checked": 0,
        "adapter_lidar_legacy_bev_detected": 0,
    }


def coerce_lidar_points(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        arr = np.frombuffer(value, dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size % 4 == 0:
            arr = arr.reshape(-1, 4)[:, :3]
        elif arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        else:
            return None
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    points = arr[:, :3].astype(np.float32, copy=False)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return None
    return points[finite]


def _looks_like_legacy_bev(value: Any) -> bool:
    if value is None or isinstance(value, bytes):
        return False
    arr = np.asarray(value)
    return bool(arr.ndim == 3 and arr.shape[0] <= 5 and arr.shape[1] >= 16 and arr.shape[2] >= 16)


def lidar_bev_debug(points: np.ndarray | None, spec: BEVSpec) -> dict[str, float | int]:
    if points is None or points.size == 0:
        return {
            "lidar_point_count": 0,
            "lidar_finite_ratio": 0.0,
            "lidar_inside_bev_ratio": 0.0,
        }
    arr = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(arr[:, :3]).all(axis=1)
    valid = arr[finite, :3]
    inside = 0
    for x_m, y_m, _ in valid:
        _, _, ok = ego_xy_to_bev_row_col(float(x_m), float(y_m), spec)
        inside += int(ok)
    return {
        "lidar_point_count": int(arr.shape[0]),
        "lidar_finite_ratio": float(finite.mean()) if finite.size else 0.0,
        "lidar_inside_bev_ratio": float(inside / max(valid.shape[0], 1)),
    }


def _debug_lidar(points: np.ndarray) -> dict[str, float | int]:
    return {
        "adapter_lidar_point_count": int(points.shape[0]),
        "adapter_lidar_finite_ratio": float(np.isfinite(points[:, :3]).all(axis=1).mean()) if points.size else 0.0,
        "adapter_lidar_coord_checked": 1,
        "adapter_lidar_legacy_bev_detected": 0,
    }
