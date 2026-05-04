from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rcs_ac_v1_1.runtime.bev_rasterizer import cell_center_grid, make_empty_bev
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec
from rcs_ac_v1_1.runtime.route_bev import build_route_bev
from rcs_ac_v1_1.runtime.state import V11RouteContext, is_route_context_valid


class V11MapAdapter(Protocol):
    def is_drivable_ego_xy(self, x_m: float, y_m: float) -> bool: ...

    def is_lane_marking_ego_xy(self, x_m: float, y_m: float) -> bool: ...


@dataclass(frozen=True)
class V11ActorBox:
    center_x_m: float
    center_y_m: float
    length_m: float
    width_m: float
    yaw_rad: float = 0.0
    actor_type: str = "vehicle"


def build_gt_bev_target(
    *,
    spec: BEVSpec,
    map_adapter: V11MapAdapter | None,
    actors_ego: list[V11ActorBox] | None,
    route_context: V11RouteContext | None,
) -> tuple[np.ndarray, float]:
    target = make_empty_bev(5, spec)
    available = 1.0 if map_adapter is not None else 0.0
    xs, ys = cell_center_grid(spec)
    if map_adapter is not None:
        for row in range(int(spec.height)):
            for col in range(int(spec.width)):
                x_m, y_m = float(xs[row, col]), float(ys[row, col])
                target[0, row, col] = 1.0 if map_adapter.is_drivable_ego_xy(x_m, y_m) else 0.0
                target[1, row, col] = 1.0 if map_adapter.is_lane_marking_ego_xy(x_m, y_m) else 0.0
    if actors_ego:
        for actor in actors_ego:
            _rasterize_actor_box(target[2], actor, xs, ys)
    target[3] = target[0] * (1.0 - target[2])
    if is_route_context_valid(route_context):
        target[4] = build_route_bev(route_context, spec)[0]
    return np.clip(target, 0.0, 1.0).astype(np.float32, copy=False), np.float32(available)


def _rasterize_actor_box(channel: np.ndarray, actor: V11ActorBox, xs: np.ndarray, ys: np.ndarray) -> None:
    cos_yaw = float(np.cos(-float(actor.yaw_rad)))
    sin_yaw = float(np.sin(-float(actor.yaw_rad)))
    dx = xs - float(actor.center_x_m)
    dy = ys - float(actor.center_y_m)
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    mask = (np.abs(local_x) <= float(actor.length_m) / 2.0) & (np.abs(local_y) <= float(actor.width_m) / 2.0)
    channel[mask] = 1.0
