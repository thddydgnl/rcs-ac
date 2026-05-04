from __future__ import annotations

import math
from typing import Any

import numpy as np

from rcs_ac_v1_1.adapters.carla_route_adapter import world_xy_to_ego_xy
from rcs_ac_v1_1.runtime.gt_bev_targets import V11ActorBox


class CarlaMapAdapter:
    def __init__(self, carla_map: Any, *, ego_x: float, ego_y: float, ego_yaw_rad: float) -> None:
        self.carla_map = carla_map
        self.ego_x = float(ego_x)
        self.ego_y = float(ego_y)
        self.ego_yaw_rad = float(ego_yaw_rad)

    def is_drivable_ego_xy(self, x_m: float, y_m: float) -> bool:
        waypoint = self._waypoint_for_ego_xy(x_m, y_m)
        if waypoint is None:
            return False
        lane_type = str(getattr(waypoint, "lane_type", "")).lower()
        return "driving" in lane_type or lane_type == "1"

    def is_lane_marking_ego_xy(self, x_m: float, y_m: float) -> bool:
        waypoint = self._waypoint_for_ego_xy(x_m, y_m)
        if waypoint is None:
            return False
        lane_width = float(getattr(waypoint, "lane_width", 3.5) or 3.5)
        transform = getattr(waypoint, "transform", None)
        if transform is None:
            return False
        loc = getattr(transform, "location", None)
        rot = getattr(transform, "rotation", None)
        if loc is None or rot is None:
            return False
        center_ego = world_xy_to_ego_xy(
            np.asarray([[float(loc.x), float(loc.y)]], dtype=np.float32),
            ego_x=self.ego_x,
            ego_y=self.ego_y,
            ego_yaw_rad=self.ego_yaw_rad,
        )
        if center_ego.shape[0] != 1:
            return False
        yaw = math.radians(float(getattr(rot, "yaw", 0.0)))
        relative_yaw = yaw - self.ego_yaw_rad
        dx = float(x_m) - float(center_ego[0, 0])
        dy = float(y_m) - float(center_ego[0, 1])
        lateral = -math.sin(relative_yaw) * dx + math.cos(relative_yaw) * dy
        distance_to_edge = abs(abs(lateral) - lane_width / 2.0)
        return bool(distance_to_edge <= 0.25)

    def _waypoint_for_ego_xy(self, x_m: float, y_m: float) -> Any | None:
        if self.carla_map is None:
            return None
        world_x, world_y = ego_xy_to_world_xy(float(x_m), float(y_m), self.ego_x, self.ego_y, self.ego_yaw_rad)
        try:
            import carla  # type: ignore

            location = carla.Location(x=float(world_x), y=float(world_y), z=0.0)
            return self.carla_map.get_waypoint(location, project_to_road=False)
        except Exception:
            return None


def ego_xy_to_world_xy(x_m: float, y_left_m: float, ego_x: float, ego_y: float, ego_yaw_rad: float) -> tuple[float, float]:
    y_right = -float(y_left_m)
    cos_yaw = math.cos(float(ego_yaw_rad))
    sin_yaw = math.sin(float(ego_yaw_rad))
    dx = cos_yaw * float(x_m) - sin_yaw * y_right
    dy = sin_yaw * float(x_m) + cos_yaw * y_right
    return float(ego_x + dx), float(ego_y + dy)


def collect_v11_actor_boxes(world: Any, ego_vehicle: Any) -> list[V11ActorBox]:
    if world is None or ego_vehicle is None or not hasattr(world, "get_actors"):
        return []
    ego_transform = ego_vehicle.get_transform()
    ego_loc = ego_transform.location
    ego_yaw_rad = math.radians(float(ego_transform.rotation.yaw))
    boxes: list[V11ActorBox] = []
    actors = world.get_actors()
    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue
        type_id = str(getattr(actor, "type_id", ""))
        if not any(token in type_id for token in ("vehicle", "walker", "static")):
            continue
        transform = actor.get_transform()
        loc = transform.location
        center = world_xy_to_ego_xy(
            np.asarray([[float(loc.x), float(loc.y)]], dtype=np.float32),
            ego_x=float(ego_loc.x),
            ego_y=float(ego_loc.y),
            ego_yaw_rad=ego_yaw_rad,
        )
        if center.shape[0] != 1:
            continue
        extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
        length = float(getattr(extent, "x", 2.0)) * 2.0 if extent is not None else 4.0
        width = float(getattr(extent, "y", 0.9)) * 2.0 if extent is not None else 1.8
        yaw_rad = math.radians(float(transform.rotation.yaw)) - ego_yaw_rad
        boxes.append(
            V11ActorBox(
                center_x_m=float(center[0, 0]),
                center_y_m=float(center[0, 1]),
                length_m=max(length, 0.1),
                width_m=max(width, 0.1),
                yaw_rad=yaw_rad,
                actor_type=type_id,
            )
        )
    return boxes
