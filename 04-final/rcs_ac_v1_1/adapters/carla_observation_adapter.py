from __future__ import annotations

import math
from typing import Any

import numpy as np

from rcs_ac_v1_1.adapters.carla_lidar_adapter import extract_lidar_points_ego, lidar_bev_debug
from rcs_ac_v1_1.adapters.carla_route_adapter import build_route_context_from_obs_info
from rcs_ac_v1_1.runtime.config import V11RuntimeConfig
from rcs_ac_v1_1.runtime.pipeline import build_flat_policy_observation, build_policy_components
from rcs_ac_v1_1.runtime.state import V11EgoState, V11RouteContext, V11SensorPacket


def build_v11_state_from_carla_step(
    *,
    obs: dict[str, Any],
    info: dict[str, Any],
    world: Any = None,
    ego_vehicle: Any = None,
    route_source: Any = None,
    previous_control: dict[str, float] | None = None,
    runtime_config: V11RuntimeConfig,
) -> tuple[V11SensorPacket, V11EgoState, V11RouteContext | None, dict[str, Any]]:
    del world, route_source
    rgb = _extract_rgb(obs, info)
    lidar_points, lidar_debug = extract_lidar_points_ego(obs, info)
    route_context, route_debug = build_route_context_from_obs_info(obs, info)
    ego_state = _build_ego_state(info=info, ego_vehicle=ego_vehicle, previous_control=previous_control, route_context=route_context)
    packet = V11SensorPacket(
        rgb_front=rgb,
        lidar_points_ego=lidar_points,
        camera_latency_ms=_extract_float(info, ("camera_latency_ms", "rgb_latency_ms"), default=0.0),
        lidar_latency_ms=_extract_float(info, ("lidar_latency_ms",), default=0.0),
        route_latency_ms=_extract_float(info, ("route_latency_ms",), default=0.0),
        weather_severity_norm=_extract_float(info, ("weather_severity_norm",), default=0.0),
        corruption_severity_norm=_extract_float(info, ("corruption_severity_norm", "corruption_level"), default=0.0),
        camera_dropout=_extract_bool(info, ("camera_dropout", "drop_rgb"), default=rgb is None),
        lidar_dropout=_extract_bool(info, ("lidar_dropout", "drop_lidar"), default=lidar_points is None),
        route_dropout=_extract_bool(info, ("route_dropout", "drop_route"), default=route_context is None),
    )
    components = build_policy_components(
        runtime_config=runtime_config,
        sensor_packet=packet,
        ego_state=ego_state,
        route_context=route_context,
    )
    obs_flat = build_flat_policy_observation(
        runtime_config=runtime_config,
        sensor_packet=packet,
        ego_state=ego_state,
        route_context=route_context,
    )
    debug: dict[str, Any] = {
        "adapter_rgb_shape": "none" if rgb is None else "x".join(str(v) for v in np.asarray(rgb).shape),
        "adapter_obs_flat_dim": int(obs_flat.shape[0]),
        "adapter_camera_dropout": int(packet.camera_dropout),
        "adapter_lidar_dropout": int(packet.lidar_dropout),
        "adapter_route_dropout": int(packet.route_dropout),
        "adapter_obs_finite": int(np.isfinite(obs_flat).all()),
        "rgb_mean": float(np.mean(components["rgb"])),
        "lidar_occupied_sum": float(np.sum(components["lidar_bev"][0])),
        "route_bev_path_sum": float(np.sum(components["route_bev"][0])),
        "ego_bev_footprint_sum": float(np.sum(components["ego_bev"][0])),
    }
    debug.update(lidar_debug)
    debug.update(lidar_bev_debug(lidar_points, runtime_config.bev_spec))
    debug.update(route_debug)
    return packet, ego_state, route_context, debug


def build_v11_flat_obs_from_carla_step(
    *,
    obs: dict[str, Any],
    info: dict[str, Any],
    runtime_config: V11RuntimeConfig,
    previous_control: dict[str, float] | None = None,
    ego_vehicle: Any = None,
) -> tuple[np.ndarray, V11SensorPacket, V11EgoState, V11RouteContext | None, dict[str, Any]]:
    packet, ego_state, route_context, debug = build_v11_state_from_carla_step(
        obs=obs,
        info=info,
        ego_vehicle=ego_vehicle,
        previous_control=previous_control,
        runtime_config=runtime_config,
    )
    obs_flat = build_flat_policy_observation(
        runtime_config=runtime_config,
        sensor_packet=packet,
        ego_state=ego_state,
        route_context=route_context,
    )
    return obs_flat, packet, ego_state, route_context, debug


def _build_ego_state(
    *,
    info: dict[str, Any],
    ego_vehicle: Any,
    previous_control: dict[str, float] | None,
    route_context: V11RouteContext | None,
) -> V11EgoState:
    speed_kmh = _extract_float(info, ("speed_kmh",), default=np.nan)
    if not np.isfinite(speed_kmh):
        speed_mps = _extract_float(info, ("speed", "speed_mps"), default=np.nan)
        speed_kmh = speed_mps * 3.6 if np.isfinite(speed_mps) else _vehicle_speed_kmh(ego_vehicle)
    yaw_rate = _extract_float(info, ("yaw_rate_rad_s", "yaw_rate"), default=np.nan)
    if not np.isfinite(yaw_rate):
        yaw_rate = _vehicle_yaw_rate_rad_s(ego_vehicle)
    acceleration = _extract_float(info, ("acceleration_mps2",), default=np.nan)
    if not np.isfinite(acceleration):
        acceleration = _vehicle_acceleration_mps2(ego_vehicle)
    previous_control = previous_control or {}
    distance_to_turn = 0.0 if route_context is None else float(route_context.distance_to_next_turn_m)
    return V11EgoState(
        speed_kmh=float(speed_kmh if np.isfinite(speed_kmh) else 0.0),
        yaw_rate_rad_s=float(yaw_rate if np.isfinite(yaw_rate) else 0.0),
        acceleration_mps2=float(acceleration if np.isfinite(acceleration) else 0.0),
        previous_steer=float(previous_control.get("steer", _extract_float(info, ("applied_steer", "smoothed_steer", "steer"), default=0.0))),
        previous_throttle=float(
            previous_control.get("throttle", _extract_float(info, ("applied_throttle", "throttle"), default=0.0))
        ),
        previous_brake=float(previous_control.get("brake", _extract_float(info, ("applied_brake", "brake"), default=0.0))),
        previous_longitudinal=float(
            previous_control.get(
                "longitudinal",
                _extract_float(info, ("applied_longitudinal", "longitudinal", "control_longitudinal"), default=0.0),
            )
        ),
        route_progress=_extract_float(info, ("route_progress",), default=0.0),
        distance_to_next_turn_m=distance_to_turn,
    )


def _extract_rgb(obs: dict[str, Any], info: dict[str, Any]) -> np.ndarray | None:
    for source in (obs, info):
        for key in ("rgb", "rgb_front", "camera_front", "image", "front_rgb"):
            if key in source:
                arr = np.asarray(source[key], dtype=np.float32)
                if arr.size > 0:
                    return arr
    return None


def _vehicle_speed_kmh(ego_vehicle: Any) -> float:
    if ego_vehicle is None or not hasattr(ego_vehicle, "get_velocity"):
        return 0.0
    velocity = ego_vehicle.get_velocity()
    return float(math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2) * 3.6)


def _vehicle_yaw_rate_rad_s(ego_vehicle: Any) -> float:
    if ego_vehicle is None or not hasattr(ego_vehicle, "get_angular_velocity"):
        return 0.0
    angular = ego_vehicle.get_angular_velocity()
    value = float(getattr(angular, "z", 0.0))
    return math.radians(value) if abs(value) > 2.0 * math.pi else value


def _vehicle_acceleration_mps2(ego_vehicle: Any) -> float:
    if ego_vehicle is None or not hasattr(ego_vehicle, "get_acceleration"):
        return 0.0
    acc = ego_vehicle.get_acceleration()
    return float(math.sqrt(float(acc.x) ** 2 + float(acc.y) ** 2 + float(acc.z) ** 2))


def _extract_float(info: dict[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    for key in keys:
        if key not in info:
            continue
        try:
            value = float(info[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return float(value)
    return float(default)


def _extract_bool(info: dict[str, Any], keys: tuple[str, ...], *, default: bool) -> bool:
    for key in keys:
        if key in info:
            return bool(info[key])
    return bool(default)
