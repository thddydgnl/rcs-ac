from __future__ import annotations

import numpy as np

from rcs_ac_v1_1.runtime.bev_rasterizer import draw_disk, draw_polyline, make_empty_bev
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, bev_row_col_to_ego_xy, ego_xy_to_bev_row_col
from rcs_ac_v1_1.runtime.state import V11EgoState


def build_ego_bev(
    ego_state: V11EgoState,
    spec: BEVSpec,
    *,
    past_positions_ego: np.ndarray | None = None,
    vehicle_length_m: float = 4.7,
    vehicle_width_m: float = 1.9,
) -> np.ndarray:
    bev = make_empty_bev(3, spec)
    half_l = float(vehicle_length_m) / 2.0
    half_w = float(vehicle_width_m) / 2.0
    for row in range(int(spec.height)):
        for col in range(int(spec.width)):
            x_m, y_m = bev_row_col_to_ego_xy(row, col, spec)
            if -half_l <= x_m <= half_l and -half_w <= y_m <= half_w:
                bev[0, row, col] = 1.0
    points = _valid_points(past_positions_ego)
    if points.shape[0] > 1:
        for idx, point in enumerate(points):
            value = 0.2 + 0.8 * (idx / max(points.shape[0] - 1, 1))
            row, col, valid = ego_xy_to_bev_row_col(float(point[0]), float(point[1]), spec)
            if valid:
                draw_disk(bev[1], row, col, 1, value)
        draw_polyline(bev[1], points, spec, radius_cells=1, value=1.0)
    else:
        row, col, valid = ego_xy_to_bev_row_col(0.0, 0.0, spec)
        if valid:
            draw_disk(bev[1], row, col, 1, 1.0)
    steer_value = float(np.clip(float(ego_state.previous_steer), -1.0, 1.0))
    if abs(steer_value) > 0.0:
        bev[2][bev[0] > 0.0] = steer_value
    return bev.astype(np.float32, copy=False)


def _valid_points(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[np.isfinite(arr).all(axis=1)]
