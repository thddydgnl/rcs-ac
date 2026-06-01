from __future__ import annotations

import math

import numpy as np

from rcs_ac_v1_1.runtime.bev_rasterizer import make_empty_bev
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, ego_xy_to_bev_row_col


def build_lidar_bev(
    lidar_points_ego: np.ndarray | None,
    spec: BEVSpec,
    *,
    min_z_m: float = -2.5,
    max_z_m: float = 3.0,
    max_density_per_cell: int = 8,
    ray_step_m: float | None = None,
) -> np.ndarray:
    bev = make_empty_bev(5, spec)
    if lidar_points_ego is None:
        bev[2] = 1.0
        return bev
    points = np.asarray(lidar_points_ego, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        bev[2] = 1.0
        return bev

    counts = np.zeros(spec.shape_hw, dtype=np.int32)
    max_height = np.full(spec.shape_hw, -np.inf, dtype=np.float32)
    valid_points: list[tuple[float, float, float, int, int]] = []
    for point in points[:, :3]:
        if not np.isfinite(point).all():
            continue
        x_m, y_m, z_m = float(point[0]), float(point[1]), float(point[2])
        if z_m < float(min_z_m) or z_m > float(max_z_m):
            continue
        row, col, valid = ego_xy_to_bev_row_col(x_m, y_m, spec)
        if not valid:
            continue
        valid_points.append((x_m, y_m, z_m, row, col))
        counts[row, col] += 1
        max_height[row, col] = max(max_height[row, col], z_m)
        bev[0, row, col] = 1.0

    if not valid_points:
        bev[2] = 1.0
        return bev.astype(np.float32, copy=False)

    step = float(ray_step_m) if ray_step_m is not None else float(spec.resolution_m)
    step = max(step, 1.0e-3)
    for x_m, y_m, _, occ_row, occ_col in valid_points:
        length = math.hypot(x_m, y_m)
        samples = max(1, int(math.floor(length / step)))
        for idx in range(samples):
            alpha = idx / max(samples, 1)
            row, col, valid = ego_xy_to_bev_row_col(alpha * x_m, alpha * y_m, spec)
            if valid and (row != occ_row or col != occ_col):
                bev[1, row, col] = 1.0

    bev[1][bev[0] > 0.0] = 0.0
    occupied = counts > 0
    bev[3][occupied] = np.clip((max_height[occupied] - float(min_z_m)) / (float(max_z_m) - float(min_z_m)), 0.0, 1.0)
    bev[4][occupied] = np.clip(np.log1p(counts[occupied]) / np.log1p(float(max_density_per_cell)), 0.0, 1.0)
    observed = np.maximum(bev[0], bev[1])
    bev[2] = 1.0 - observed
    return np.clip(bev, 0.0, 1.0).astype(np.float32, copy=False)
