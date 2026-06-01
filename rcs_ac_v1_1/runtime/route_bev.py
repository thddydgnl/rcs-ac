from __future__ import annotations

import math

import numpy as np

from rcs_ac_v1_1.runtime.bev_rasterizer import draw_disk, draw_gaussian, draw_polyline, make_empty_bev
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, ego_xy_to_bev_row_col
from rcs_ac_v1_1.runtime.state import V11RouteContext, as_route_points, is_route_context_valid


def build_route_bev(
    route_context: V11RouteContext | None,
    spec: BEVSpec,
    *,
    path_radius_m: float = 0.75,
    target_sigma_m: float = 1.0,
    turn_radius_m: float = 2.0,
    max_speed_kmh: float = 40.0,
) -> np.ndarray:
    bev = make_empty_bev(4, spec)
    if not is_route_context_valid(route_context):
        return bev
    assert route_context is not None
    points = as_route_points(route_context)
    radius_cells = max(1, int(math.ceil(float(path_radius_m) / float(spec.resolution_m))))
    draw_polyline(bev[0], points, spec, radius_cells=radius_cells, value=1.0)

    target = _select_target_point(route_context, points)
    if target is not None:
        draw_gaussian(bev[1], float(target[0]), float(target[1]), spec, sigma_m=target_sigma_m, peak=1.0)
        turn_value = _turn_value(route_context.maneuver)
        if abs(turn_value) > 0.0:
            row, col, valid = ego_xy_to_bev_row_col(float(target[0]), float(target[1]), spec)
            if valid:
                draw_disk(
                    bev[2],
                    row,
                    col,
                    max(1, int(math.ceil(float(turn_radius_m) / float(spec.resolution_m)))),
                    turn_value,
                )

    speed_value = float(np.clip(float(route_context.target_speed_kmh) / float(max_speed_kmh), 0.0, 1.0))
    if speed_value > 0.0:
        speed_path = np.zeros(spec.shape_hw, dtype=np.float32)
        draw_polyline(speed_path, points, spec, radius_cells=radius_cells, value=speed_value)
        bev[3] = np.maximum(bev[3], speed_path)
    bev[0] = np.clip(bev[0], 0.0, 1.0)
    bev[1] = np.clip(bev[1], 0.0, 1.0)
    bev[3] = np.clip(bev[3], 0.0, 1.0)
    return bev.astype(np.float32, copy=False)


def _select_target_point(route_context: V11RouteContext, points: np.ndarray) -> tuple[float, float] | None:
    if route_context.target_point_ego is not None:
        return float(route_context.target_point_ego[0]), float(route_context.target_point_ego[1])
    forward = points[points[:, 0] >= 0.0]
    if forward.shape[0] == 0:
        return None
    idx = int(np.argmin(np.abs(forward[:, 0] - 7.0)))
    return float(forward[idx, 0]), float(forward[idx, 1])


def _turn_value(maneuver: str) -> float:
    value = str(maneuver).lower()
    if value == "left":
        return 1.0
    if value == "right":
        return -1.0
    return 0.0
