"""CARLA simulator-state privileged observations for Stage 2.

Raw2Drive's privileged stream must be built from simulator privileged state,
not from raw policy inputs such as RGB, LiDAR grids, or route vectors. This
module therefore uses CARLA map, actor, ego vehicle, and route-manager state to
construct a pinned 43-channel semantic BEV plus an 8-field ego vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch


SEMANTIC43_SCHEMA_VERSION = "carla_simulator_semantic43_v1"
EGO8_SCHEMA_VERSION = "carla_simulator_ego8_v1"
PRIVILEGED_OBSERVATION_SOURCE = "carla_simulator_state"
SEMANTIC43_CHANNEL_NAMES: tuple[str, ...] = (
    "drivable_road",
    "lane_marking",
    "route_path",
    "route_target",
    "ego_vehicle",
    "vehicle",
    "pedestrian",
    "static_obstacle",
    "traffic_light",
    "traffic_sign",
    "road_shoulder",
    "sidewalk",
    "parking",
    "other_lane",
    "opposite_lane",
    "junction",
    "stop_region",
    "yield_region",
    "speed_limit_region",
    "crosswalk",
    "curb",
    "guard_rail",
    "bridge",
    "tunnel",
    "building",
    "wall",
    "fence",
    "pole",
    "vegetation",
    "terrain",
    "sky_projected_empty",
    "dynamic_vehicle",
    "dynamic_walker",
    "dynamic_bike",
    "emergency_vehicle",
    "route_left_turn",
    "route_right_turn",
    "route_straight",
    "route_speed_hint",
    "ego_footprint_tail",
    "actor_near_field",
    "actor_far_field",
    "unknown_privileged_reserved",
)
SEMANTIC43_CHANNEL_NAMES_SHA256 = hashlib.sha256("\n".join(SEMANTIC43_CHANNEL_NAMES).encode("utf-8")).hexdigest()
EGO8_FIELD_NAMES: tuple[str, ...] = (
    "speed_kmh_norm40",
    "route_progress",
    "lane_offset_norm5",
    "heading_error_deg_norm180",
    "steer",
    "throttle",
    "brake",
    "target_distance_norm50",
)


@dataclass(frozen=True)
class SimulatorStatePrivilegedObservation:
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BEVGridSpec:
    height: int = 16
    width: int = 16
    front_m: float = 40.0
    rear_m: float = 8.0
    left_m: float = 24.0
    right_m: float = 24.0

    @property
    def x_resolution_m(self) -> float:
        return (float(self.front_m) + float(self.rear_m)) / float(self.height)

    @property
    def y_resolution_m(self) -> float:
        return (float(self.left_m) + float(self.right_m)) / float(self.width)


def build_carla_simulator_semantic43_observation(
    *,
    raw_env: Any,
    expected_bev_shape: tuple[int, ...],
    expected_ego_shape: tuple[int, ...],
) -> SimulatorStatePrivilegedObservation:
    if len(expected_bev_shape) != 5 or tuple(expected_bev_shape[:2]) != (1, 1):
        raise ValueError("expected BEV shape must be [1,1,C,H,W]")
    if len(expected_ego_shape) != 3 or tuple(expected_ego_shape[:2]) != (1, 1):
        raise ValueError("expected ego shape must be [1,1,D]")
    if int(expected_bev_shape[2]) != len(SEMANTIC43_CHANNEL_NAMES):
        raise ValueError("Stage 2 privileged BEV must use exactly 43 semantic channels")
    if int(expected_ego_shape[2]) != len(EGO8_FIELD_NAMES):
        raise ValueError("Stage 2 privileged ego must use the pinned 8-field schema")

    world = getattr(raw_env, "world", None)
    vehicle = getattr(raw_env, "vehicle", None)
    route_manager = getattr(raw_env, "route_manager", None)
    carla_map = getattr(raw_env, "map", None)
    if carla_map is None and world is not None and hasattr(world, "get_map"):
        carla_map = world.get_map()
    if world is None or vehicle is None or carla_map is None:
        raise ValueError("raw_env must expose CARLA world, vehicle, and map for privileged simulator-state BEV")

    transform = vehicle.get_transform()
    loc = transform.location
    ego_x = float(loc.x)
    ego_y = float(loc.y)
    ego_yaw_rad = math.radians(float(transform.rotation.yaw))
    spec = BEVGridSpec(height=int(expected_bev_shape[3]), width=int(expected_bev_shape[4]))
    bev = np.zeros((len(SEMANTIC43_CHANNEL_NAMES), spec.height, spec.width), dtype=np.float32)

    _rasterize_map(bev, carla_map=carla_map, ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad, spec=spec)
    route_points = _route_points_from_manager(route_manager, ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad)
    _rasterize_route(bev, route_points=route_points, route_manager=route_manager, spec=spec)
    _rasterize_ego(bev, spec=spec)
    _rasterize_actors(bev, world=world, ego_vehicle=vehicle, ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad, spec=spec)

    ego_vector = _build_ego8(raw_env=raw_env, vehicle=vehicle, route_manager=route_manager)
    bev_tensor = torch.from_numpy(bev.reshape(expected_bev_shape)).float()
    ego_tensor = torch.from_numpy(ego_vector.reshape(expected_ego_shape)).float()
    if not torch.isfinite(bev_tensor).all() or not torch.isfinite(ego_tensor).all():
        raise ValueError("simulator-state privileged observation contains non-finite values")

    nonzero_channels = [int(index) for index in range(bev.shape[0]) if bool(np.any(bev[index] > 1.0e-6))]
    channel_nonzero_counts = {
        SEMANTIC43_CHANNEL_NAMES[index]: int(np.count_nonzero(bev[index] > 1.0e-6))
        for index in range(bev.shape[0])
    }
    critical_channel_names = (
        "drivable_road",
        "lane_marking",
        "route_path",
        "route_target",
        "ego_vehicle",
        "vehicle",
        "pedestrian",
        "static_obstacle",
        "dynamic_vehicle",
        "actor_near_field",
    )
    empty_critical_channel_names = [
        name for name in critical_channel_names if channel_nonzero_counts.get(name, 0) == 0
    ]
    if not np.any(bev[0] > 0.0):
        raise ValueError("simulator-state privileged BEV has empty drivable_road channel")
    if not np.any(bev[4] > 0.0):
        raise ValueError("simulator-state privileged BEV has empty ego_vehicle channel")
    diagnostics = {
        "privileged_observation_schema": SEMANTIC43_SCHEMA_VERSION,
        "privileged_observation_source": PRIVILEGED_OBSERVATION_SOURCE,
        "privileged_observation_clean_gate_eligible": True,
        "privileged_bev_shape": list(bev_tensor.shape),
        "privileged_ego_shape": list(ego_tensor.shape),
        "privileged_bev_height": int(spec.height),
        "privileged_bev_width": int(spec.width),
        "privileged_bev_grid_spec": {
            "height": int(spec.height),
            "width": int(spec.width),
            "front_m": float(spec.front_m),
            "rear_m": float(spec.rear_m),
            "left_m": float(spec.left_m),
            "right_m": float(spec.right_m),
            "x_resolution_m": float(spec.x_resolution_m),
            "y_resolution_m": float(spec.y_resolution_m),
        },
        "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
        "privileged_ego_schema": EGO8_SCHEMA_VERSION,
        "privileged_ego_fields": list(EGO8_FIELD_NAMES),
        "semantic_nonzero_channels": nonzero_channels,
        "semantic_nonzero_channel_names": [SEMANTIC43_CHANNEL_NAMES[index] for index in nonzero_channels],
        "semantic_channel_nonzero_counts": channel_nonzero_counts,
        "semantic_empty_critical_channel_names": empty_critical_channel_names,
        "semantic_route_available": bool(np.any(bev[2] > 0.0)),
        "semantic_actor_available": bool(np.any(bev[5:10] > 0.0) or np.any(bev[31:35] > 0.0)),
        "semantic_lane_available": bool(np.any(bev[1] > 0.0)),
        "semantic_obstacle_available": bool(
            np.any(bev[5:10] > 0.0) or np.any(bev[31:35] > 0.0) or np.any(bev[40:42] > 0.0)
        ),
    }
    return SimulatorStatePrivilegedObservation(
        privileged_bev=bev_tensor,
        privileged_ego=ego_tensor,
        diagnostics=diagnostics,
    )


def _rasterize_map(
    bev: np.ndarray,
    *,
    carla_map: Any,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
    spec: BEVGridSpec,
) -> None:
    for row in range(spec.height):
        for col in range(spec.width):
            x_m, y_m = _grid_to_ego_xy(row, col, spec)
            waypoint = _waypoint_for_ego_xy(carla_map, x_m, y_m, ego_x, ego_y, ego_yaw_rad)
            if waypoint is None:
                continue
            lane_type = str(getattr(waypoint, "lane_type", "")).lower()
            if "driving" in lane_type or lane_type == "1":
                bev[0, row, col] = 1.0
            if "sidewalk" in lane_type:
                bev[11, row, col] = 1.0
            if "shoulder" in lane_type:
                bev[10, row, col] = 1.0
            if "parking" in lane_type:
                bev[12, row, col] = 1.0
            if _is_lane_marking_cell(waypoint, x_m, y_m, ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad):
                bev[1, row, col] = 1.0
            if bool(getattr(waypoint, "is_junction", False)):
                bev[15, row, col] = 1.0


def _rasterize_route(
    bev: np.ndarray,
    *,
    route_points: np.ndarray,
    route_manager: Any,
    spec: BEVGridSpec,
) -> None:
    if route_points.shape[0] == 0:
        return
    _draw_polyline(bev[2], route_points, spec, radius_cells=1, value=1.0)
    target = route_points[min(1, route_points.shape[0] - 1)]
    target_row, target_col, valid = _ego_xy_to_grid(float(target[0]), float(target[1]), spec)
    if valid:
        _draw_disk(bev[3], target_row, target_col, radius_cells=1, value=1.0)
    maneuver = ""
    if route_manager is not None and hasattr(route_manager, "get_maneuver_command"):
        maneuver = str(route_manager.get_maneuver_command()).lower()
    if maneuver == "left":
        bev[35] = np.maximum(bev[35], bev[2])
    elif maneuver == "right":
        bev[36] = np.maximum(bev[36], bev[2])
    else:
        bev[37] = np.maximum(bev[37], bev[2])


def _rasterize_ego(bev: np.ndarray, *, spec: BEVGridSpec) -> None:
    _rasterize_box(
        bev[4],
        center_x_m=0.0,
        center_y_m=0.0,
        length_m=4.7,
        width_m=1.9,
        yaw_rad=0.0,
        spec=spec,
    )
    ego_row, ego_col, ego_valid = _ego_xy_to_grid(0.0, 0.0, spec)
    if ego_valid:
        _draw_disk(bev[4], ego_row, ego_col, radius_cells=1, value=1.0)
    tail_row, tail_col, tail_valid = _ego_xy_to_grid(-2.0, 0.0, spec)
    if tail_valid:
        _draw_disk(bev[39], tail_row, tail_col, radius_cells=1, value=1.0)


def _rasterize_actors(
    bev: np.ndarray,
    *,
    world: Any,
    ego_vehicle: Any,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
    spec: BEVGridSpec,
) -> None:
    if world is None or not hasattr(world, "get_actors"):
        return
    ego_id = getattr(ego_vehicle, "id", None)
    for actor in world.get_actors():
        if getattr(actor, "id", None) == ego_id:
            continue
        type_id = str(getattr(actor, "type_id", "")).lower()
        transform = actor.get_transform()
        loc = transform.location
        center = _world_xy_to_ego_xy(float(loc.x), float(loc.y), ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad)
        if not _inside_extent(float(center[0]), float(center[1]), spec):
            continue
        extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
        length = max(float(getattr(extent, "x", 2.0)) * 2.0 if extent is not None else 4.0, 0.25)
        width = max(float(getattr(extent, "y", 0.9)) * 2.0 if extent is not None else 1.0, 0.25)
        yaw_rad = math.radians(float(transform.rotation.yaw)) - ego_yaw_rad
        channel = _actor_channel(type_id)
        _rasterize_box(
            bev[channel],
            center_x_m=float(center[0]),
            center_y_m=float(center[1]),
            length_m=length,
            width_m=width,
            yaw_rad=yaw_rad,
            spec=spec,
        )
        near_far_channel = 40 if math.hypot(float(center[0]), float(center[1])) < 15.0 else 41
        actor_row, actor_col, actor_valid = _ego_xy_to_grid(float(center[0]), float(center[1]), spec)
        if actor_valid:
            _draw_disk(bev[near_far_channel], actor_row, actor_col, radius_cells=1, value=1.0)


def _actor_channel(type_id: str) -> int:
    if "traffic_light" in type_id:
        return 8
    if "traffic" in type_id and "sign" in type_id:
        return 9
    if "walker" in type_id or "pedestrian" in type_id:
        return 6
    if "vehicle" in type_id:
        if "bike" in type_id or "bicycle" in type_id or "motorcycle" in type_id:
            return 33
        return 5
    return 7


def _build_ego8(*, raw_env: Any, vehicle: Any, route_manager: Any) -> np.ndarray:
    transform = vehicle.get_transform()
    control = vehicle.get_control() if hasattr(vehicle, "get_control") else getattr(raw_env, "prev_control", None)
    velocity = vehicle.get_velocity() if hasattr(vehicle, "get_velocity") else None
    speed_kmh = _speed_kmh_from_velocity(velocity)
    route_progress = 0.0
    lane_offset = 0.0
    heading_error = 0.0
    target_distance = 0.0
    if route_manager is not None:
        try:
            metrics = route_manager.build_metrics(transform)
            route_progress = float(getattr(metrics, "route_progress", 0.0))
            lane_offset = float(getattr(metrics, "lane_offset", 0.0))
            heading_error = float(getattr(metrics, "heading_error", 0.0))
        except Exception:
            pass
        try:
            target_waypoint = route_manager.get_current_target_waypoint()
            target_distance = float(transform.location.distance(target_waypoint.transform.location))
        except Exception:
            target_distance = 0.0
    return np.asarray(
        [
            _clip_norm(speed_kmh, 40.0, signed=False),
            float(np.clip(route_progress, 0.0, 1.0)),
            _clip_norm(lane_offset, 5.0, signed=True),
            _clip_norm(heading_error, 180.0, signed=True),
            float(np.clip(float(getattr(control, "steer", 0.0)), -1.0, 1.0)),
            float(np.clip(float(getattr(control, "throttle", 0.0)), 0.0, 1.0)),
            float(np.clip(float(getattr(control, "brake", 0.0)), 0.0, 1.0)),
            _clip_norm(target_distance, 50.0, signed=False),
        ],
        dtype=np.float32,
    )


def _route_points_from_manager(route_manager: Any, *, ego_x: float, ego_y: float, ego_yaw_rad: float) -> np.ndarray:
    if route_manager is None:
        return np.zeros((0, 2), dtype=np.float32)
    route = list(getattr(route_manager, "route", []) or [])
    target_index = int(getattr(route_manager, "target_index", 0) or 0)
    selected = route[target_index : min(len(route), target_index + 12)]
    points: list[tuple[float, float]] = []
    for waypoint in selected:
        transform = getattr(waypoint, "transform", None)
        loc = getattr(transform, "location", None)
        if loc is None:
            continue
        point = _world_xy_to_ego_xy(float(loc.x), float(loc.y), ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad)
        points.append((float(point[0]), float(point[1])))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _waypoint_for_ego_xy(carla_map: Any, x_m: float, y_m: float, ego_x: float, ego_y: float, ego_yaw_rad: float) -> Any | None:
    world_x, world_y = _ego_xy_to_world_xy(x_m, y_m, ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad)
    try:
        import carla  # type: ignore

        location = carla.Location(x=float(world_x), y=float(world_y), z=0.0)
    except Exception:
        location = type("Location", (), {"x": float(world_x), "y": float(world_y), "z": 0.0})()
    try:
        return carla_map.get_waypoint(location, project_to_road=False)
    except TypeError:
        return carla_map.get_waypoint(location)
    except Exception:
        return None


def _is_lane_marking_cell(waypoint: Any, x_m: float, y_m: float, *, ego_x: float, ego_y: float, ego_yaw_rad: float) -> bool:
    lane_width = float(getattr(waypoint, "lane_width", 3.5) or 3.5)
    transform = getattr(waypoint, "transform", None)
    loc = getattr(transform, "location", None)
    rot = getattr(transform, "rotation", None)
    if loc is None or rot is None:
        return False
    center = _world_xy_to_ego_xy(float(loc.x), float(loc.y), ego_x=ego_x, ego_y=ego_y, ego_yaw_rad=ego_yaw_rad)
    relative_yaw = math.radians(float(getattr(rot, "yaw", 0.0))) - ego_yaw_rad
    dx = float(x_m) - float(center[0])
    dy = float(y_m) - float(center[1])
    lateral = -math.sin(relative_yaw) * dx + math.cos(relative_yaw) * dy
    return bool(abs(abs(lateral) - lane_width / 2.0) <= 0.35)


def _rasterize_box(
    channel: np.ndarray,
    *,
    center_x_m: float,
    center_y_m: float,
    length_m: float,
    width_m: float,
    yaw_rad: float,
    spec: BEVGridSpec,
) -> None:
    for row in range(spec.height):
        for col in range(spec.width):
            x_m, y_m = _grid_to_ego_xy(row, col, spec)
            dx = x_m - float(center_x_m)
            dy = y_m - float(center_y_m)
            cos_yaw = math.cos(-float(yaw_rad))
            sin_yaw = math.sin(-float(yaw_rad))
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            if abs(local_x) <= float(length_m) / 2.0 and abs(local_y) <= float(width_m) / 2.0:
                channel[row, col] = 1.0


def _draw_polyline(channel: np.ndarray, points: np.ndarray, spec: BEVGridSpec, *, radius_cells: int, value: float) -> None:
    if points.shape[0] == 1:
        row, col, valid = _ego_xy_to_grid(float(points[0, 0]), float(points[0, 1]), spec)
        if valid:
            _draw_disk(channel, row, col, radius_cells=radius_cells, value=value)
        return
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(math.ceil(length / max(spec.x_resolution_m * 0.5, 1.0e-3))))
        for idx in range(steps + 1):
            point = start + (end - start) * (idx / steps)
            row, col, valid = _ego_xy_to_grid(float(point[0]), float(point[1]), spec)
            if valid:
                _draw_disk(channel, row, col, radius_cells=radius_cells, value=value)


def _draw_disk(channel: np.ndarray, row: int, col: int, *, radius_cells: int, value: float) -> None:
    for rr in range(max(0, row - radius_cells), min(channel.shape[0], row + radius_cells + 1)):
        for cc in range(max(0, col - radius_cells), min(channel.shape[1], col + radius_cells + 1)):
            if (rr - row) ** 2 + (cc - col) ** 2 <= radius_cells**2:
                channel[rr, cc] = max(float(channel[rr, cc]), float(value))


def _grid_to_ego_xy(row: int, col: int, spec: BEVGridSpec) -> tuple[float, float]:
    x_m = float(spec.front_m) - (int(row) + 0.5) * spec.x_resolution_m
    y_m = float(spec.left_m) - (int(col) + 0.5) * spec.y_resolution_m
    return x_m, y_m


def _ego_xy_to_grid(x_m: float, y_m: float, spec: BEVGridSpec) -> tuple[int, int, bool]:
    row_float = (float(spec.front_m) - float(x_m)) / spec.x_resolution_m
    col_float = (float(spec.left_m) - float(y_m)) / spec.y_resolution_m
    valid = 0.0 <= row_float < spec.height and 0.0 <= col_float < spec.width
    row = min(max(int(math.floor(row_float)), 0), spec.height - 1)
    col = min(max(int(math.floor(col_float)), 0), spec.width - 1)
    return row, col, bool(valid)


def _inside_extent(x_m: float, y_m: float, spec: BEVGridSpec) -> bool:
    return -float(spec.rear_m) <= float(x_m) <= float(spec.front_m) and -float(spec.right_m) <= float(y_m) <= float(spec.left_m)


def _world_xy_to_ego_xy(world_x: float, world_y: float, *, ego_x: float, ego_y: float, ego_yaw_rad: float) -> tuple[float, float]:
    dx = float(world_x) - float(ego_x)
    dy = float(world_y) - float(ego_y)
    cos_yaw = math.cos(float(ego_yaw_rad))
    sin_yaw = math.sin(float(ego_yaw_rad))
    x_ego = cos_yaw * dx + sin_yaw * dy
    y_right = -sin_yaw * dx + cos_yaw * dy
    return float(x_ego), float(-y_right)


def _ego_xy_to_world_xy(x_m: float, y_left_m: float, *, ego_x: float, ego_y: float, ego_yaw_rad: float) -> tuple[float, float]:
    y_right = -float(y_left_m)
    cos_yaw = math.cos(float(ego_yaw_rad))
    sin_yaw = math.sin(float(ego_yaw_rad))
    dx = cos_yaw * float(x_m) - sin_yaw * y_right
    dy = sin_yaw * float(x_m) + cos_yaw * y_right
    return float(ego_x + dx), float(ego_y + dy)


def _speed_kmh_from_velocity(velocity: Any) -> float:
    if velocity is None:
        return 0.0
    speed_mps = math.sqrt(
        float(getattr(velocity, "x", 0.0)) ** 2
        + float(getattr(velocity, "y", 0.0)) ** 2
        + float(getattr(velocity, "z", 0.0)) ** 2
    )
    return speed_mps * 3.6


def _clip_norm(value: Any, scale: float, *, signed: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if not np.isfinite(number):
        number = 0.0
    lower = -1.0 if signed else 0.0
    return float(np.clip(number / float(scale), lower, 1.0))
