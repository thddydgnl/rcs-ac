from __future__ import annotations

import numpy as np

from rcs_ac_v1_1.runtime.config import V11RuntimeConfig
from rcs_ac_v1_1.runtime.ego_bev import build_ego_bev
from rcs_ac_v1_1.runtime.lidar_bev import build_lidar_bev
from rcs_ac_v1_1.runtime.observation import flatten_observation_dict
from rcs_ac_v1_1.runtime.rgb_bev import build_rgb_bev_from_prediction, build_rgb_bev_unavailable
from rcs_ac_v1_1.runtime.route_bev import build_route_bev
from rcs_ac_v1_1.runtime.scalars import build_ego_scalar, build_route_scalar
from rcs_ac_v1_1.runtime.sensor_health import build_sensor_health
from rcs_ac_v1_1.runtime.state import V11EgoState, V11RouteContext, V11SensorPacket


def build_policy_components(
    *,
    runtime_config: V11RuntimeConfig,
    sensor_packet: V11SensorPacket,
    ego_state: V11EgoState,
    route_context: V11RouteContext | None,
    rgb_bev_prediction: np.ndarray | None = None,
    past_positions_ego: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    spec = runtime_config.bev_spec
    if rgb_bev_prediction is None:
        rgb_bev = build_rgb_bev_unavailable(spec)
    else:
        rgb_bev = build_rgb_bev_from_prediction(rgb_bev_prediction, spec)
    components = {
        "rgb": _normalize_rgb(sensor_packet.rgb_front, runtime_config.observation_spec.modal("rgb").shape),
        "lidar_bev": build_lidar_bev(sensor_packet.lidar_points_ego, spec),
        "route_bev": build_route_bev(route_context, spec),
        "ego_bev": build_ego_bev(ego_state, spec, past_positions_ego=past_positions_ego),
        "ego_scalar": build_ego_scalar(ego_state),
        "route_scalar": build_route_scalar(route_context),
        "sensor_health": build_sensor_health(
            rgb_front=sensor_packet.rgb_front,
            lidar_points_ego=sensor_packet.lidar_points_ego,
            route_context=route_context,
            ego_state=ego_state,
            camera_latency_ms=sensor_packet.camera_latency_ms,
            lidar_latency_ms=sensor_packet.lidar_latency_ms,
            route_latency_ms=sensor_packet.route_latency_ms,
            weather_severity_norm=sensor_packet.weather_severity_norm,
            corruption_severity_norm=sensor_packet.corruption_severity_norm,
            camera_dropout=sensor_packet.camera_dropout,
            lidar_dropout=sensor_packet.lidar_dropout,
            route_dropout=sensor_packet.route_dropout,
            spec=spec,
        ),
    }
    components["rgb_bev"] = rgb_bev
    _validate_policy_components(components, runtime_config)
    return components


def build_flat_policy_observation(
    *,
    runtime_config: V11RuntimeConfig,
    sensor_packet: V11SensorPacket,
    ego_state: V11EgoState,
    route_context: V11RouteContext | None,
    rgb_bev_prediction: np.ndarray | None = None,
    past_positions_ego: np.ndarray | None = None,
) -> np.ndarray:
    components = build_policy_components(
        runtime_config=runtime_config,
        sensor_packet=sensor_packet,
        ego_state=ego_state,
        route_context=route_context,
        rgb_bev_prediction=rgb_bev_prediction,
        past_positions_ego=past_positions_ego,
    )
    policy_components = {key: value for key, value in components.items() if key != "rgb_bev"}
    return flatten_observation_dict(policy_components, runtime_config.observation_spec)


def _normalize_rgb(rgb_front: np.ndarray | None, expected_shape: tuple[int, ...]) -> np.ndarray:
    if rgb_front is None:
        return np.zeros(expected_shape, dtype=np.float32)
    rgb = np.asarray(rgb_front, dtype=np.float32)
    if rgb.ndim == 3 and rgb.shape[-1] in (1, 3):
        rgb = np.moveaxis(rgb, -1, 0)
    if rgb.max(initial=0.0) > 1.5:
        rgb = rgb / 255.0
    if tuple(rgb.shape) != tuple(expected_shape):
        raise ValueError(f"rgb expected shape {expected_shape}, got {tuple(rgb.shape)}")
    if not np.isfinite(rgb).all():
        raise ValueError("rgb contains NaN or inf")
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def _validate_policy_components(components: dict[str, np.ndarray], runtime_config: V11RuntimeConfig) -> None:
    for key in runtime_config.observation_spec.keys:
        expected = runtime_config.observation_spec.modal(key).shape
        actual = tuple(np.asarray(components[key]).shape)
        if actual != tuple(expected):
            raise ValueError(f"component {key!r} expected shape {expected}, got {actual}")
    rgb_bev_shape = runtime_config.bev_spec.channel_shape(4)
    if tuple(np.asarray(components["rgb_bev"]).shape) != tuple(rgb_bev_shape):
        raise ValueError(f"component 'rgb_bev' expected shape {rgb_bev_shape}, got {components['rgb_bev'].shape}")
