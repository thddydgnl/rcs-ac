from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rcs_ac_v1_1.models.bev_policy import RCSACV11BEVPolicy, V11BEVPolicyConfig
from rcs_ac_v1_1.runtime.bev_spec import BEVChannelSpec, BEVSpec
from rcs_ac_v1_1.runtime.observation import (
    V11PolicyObservationSpec,
    V11TargetSpec,
    build_v11_policy_observation_spec,
    build_v11_target_spec,
)
from rcs_ac_v1_1.runtime.pid_controller import PurePursuitControllerConfig, V11SpeedAdaptivePurePursuitController


@dataclass(frozen=True)
class V11RuntimeConfig:
    version: str
    description: str
    bev_spec: BEVSpec
    bev_channels: BEVChannelSpec
    policy_config: V11BEVPolicyConfig
    controller_config: PurePursuitControllerConfig
    observation_spec: V11PolicyObservationSpec
    target_spec: V11TargetSpec
    raw: dict[str, Any]

    def build_policy(self) -> RCSACV11BEVPolicy:
        return RCSACV11BEVPolicy(self.policy_config)

    def build_controller(self) -> V11SpeedAdaptivePurePursuitController:
        return V11SpeedAdaptivePurePursuitController(self.controller_config)


def load_v11_runtime_config(path: str | Path) -> V11RuntimeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"v1.1 config must be a mapping, got {type(raw).__name__}")
    return build_v11_runtime_config(raw)


def build_v11_runtime_config(raw: dict[str, Any]) -> V11RuntimeConfig:
    _require_mapping(raw, "model")
    _require_mapping(raw, "bev_spec")
    _require_mapping(raw, "controller")
    _require_mapping(raw, "policy_inputs")
    _require_mapping(raw, "target_only")

    model = raw["model"]
    bev_spec = BEVSpec(
        height=_as_int(raw["bev_spec"], "height"),
        width=_as_int(raw["bev_spec"], "width"),
        resolution_m=_as_float(raw["bev_spec"], "resolution_m"),
        front_m=_as_float(raw["bev_spec"], "front_m"),
        rear_m=_as_float(raw["bev_spec"], "rear_m"),
        left_m=_as_float(raw["bev_spec"], "left_m"),
        right_m=_as_float(raw["bev_spec"], "right_m"),
        strict_extent_check=bool(raw["bev_spec"].get("strict_extent_check", True)),
        extent_tolerance_m=float(raw["bev_spec"].get("extent_tolerance_m", 1.0e-4)),
    )
    _assert_model_bev_consistency(model, bev_spec)

    channels = _build_bev_channel_spec(model)
    policy_config = V11BEVPolicyConfig(
        rgb_shape=_as_shape(model, "rgb_shape", ndim=3),
        bev_spec=bev_spec,
        bev_channels=channels,
        ego_scalar_dim=_as_int(model, "ego_scalar_dim"),
        route_scalar_dim=_as_int(model, "route_scalar_dim"),
        sensor_health_dim=_as_int(model, "sensor_health_dim"),
        feature_dim=_as_int(model, "feature_dim"),
        fusion_dim=_as_int(model, "fusion_dim"),
        waypoint_count=_as_int(model, "waypoint_count"),
        waypoint_mode=str(model.get("waypoint_mode", "distance")),
        waypoint_distances_m=tuple(float(v) for v in _require_sequence(model, "waypoint_distances_m")),
        waypoint_gap_residual_scale_m=_as_float(model, "waypoint_gap_residual_scale_m"),
        waypoint_min_gap_m=_as_float(model, "waypoint_min_gap_m"),
        waypoint_max_lateral_m=_as_float(model, "waypoint_max_lateral_m"),
        waypoint_parameterization=str(model.get("waypoint_parameterization", "monotonic_bounded")),
        min_target_speed_kmh=_as_float(model, "min_target_speed_kmh"),
        max_target_speed_kmh=_as_float(model, "max_target_speed_kmh"),
        use_bev_prediction_head=bool(model.get("use_bev_prediction_head", True)),
    )
    controller_config = _build_controller_config(raw["controller"])
    observation_spec = build_v11_policy_observation_spec(
        rgb_shape=policy_config.rgb_shape,
        bev_spec=bev_spec,
        bev_channels=channels,
        ego_scalar_dim=policy_config.ego_scalar_dim,
        route_scalar_dim=policy_config.route_scalar_dim,
        sensor_health_dim=policy_config.sensor_health_dim,
    )
    target_spec = build_v11_target_spec(
        bev_spec=bev_spec,
        bev_channels=channels,
        waypoint_count=policy_config.waypoint_count,
    )
    _assert_declared_shapes(raw, observation_spec, target_spec)
    return V11RuntimeConfig(
        version=str(raw.get("version", "")),
        description=str(raw.get("description", "")),
        bev_spec=bev_spec,
        bev_channels=channels,
        policy_config=policy_config,
        controller_config=controller_config,
        observation_spec=observation_spec,
        target_spec=target_spec,
        raw=raw,
    )


def _build_bev_channel_spec(model: dict[str, Any]) -> BEVChannelSpec:
    channels = _require_mapping(model, "bev_channels")
    return BEVChannelSpec(
        rgb_channels=_as_int(channels, "rgb"),
        lidar_channels=_as_int(channels, "lidar"),
        route_channels=_as_int(channels, "route"),
        ego_channels=_as_int(channels, "ego"),
        target_channels=_as_int(channels, "target"),
    )


def _build_controller_config(controller: dict[str, Any]) -> PurePursuitControllerConfig:
    controller_type = str(controller.get("type", ""))
    if controller_type != "speed_adaptive_pure_pursuit":
        raise ValueError(f"controller.type must be speed_adaptive_pure_pursuit, got {controller_type!r}")
    defaults = PurePursuitControllerConfig()
    values = {
        field: controller.get(field, getattr(defaults, field))
        for field in PurePursuitControllerConfig.__dataclass_fields__
    }
    return PurePursuitControllerConfig(**values)


def _assert_model_bev_consistency(model: dict[str, Any], bev_spec: BEVSpec) -> None:
    checks: tuple[tuple[str, Any], ...] = (
        ("bev_height", bev_spec.height),
        ("bev_width", bev_spec.width),
        ("bev_resolution_m", bev_spec.resolution_m),
        ("bev_front_m", bev_spec.front_m),
        ("bev_rear_m", bev_spec.rear_m),
        ("bev_left_m", bev_spec.left_m),
        ("bev_right_m", bev_spec.right_m),
    )
    for key, expected in checks:
        if key not in model:
            raise KeyError(f"model.{key} is required")
        actual = model[key]
        if isinstance(expected, int):
            if int(actual) != int(expected):
                raise ValueError(f"model.{key}={actual!r} does not match bev_spec value {expected!r}")
        elif abs(float(actual) - float(expected)) > float(bev_spec.extent_tolerance_m):
            raise ValueError(f"model.{key}={actual!r} does not match bev_spec value {expected!r}")
    if bool(model.get("bev_strict_extent_check", True)) != bool(bev_spec.strict_extent_check):
        raise ValueError("model.bev_strict_extent_check does not match bev_spec.strict_extent_check")


def _assert_declared_shapes(
    raw: dict[str, Any],
    observation_spec: V11PolicyObservationSpec,
    target_spec: V11TargetSpec,
) -> None:
    policy_inputs = raw["policy_inputs"]
    target_only = raw["target_only"]
    for key in ("rgb", "lidar_bev", "route_bev", "ego_bev"):
        declared = _as_shape(policy_inputs, key, ndim=3)
        expected = observation_spec.modal(key).shape
        if declared != expected:
            raise ValueError(f"policy_inputs.{key}={declared!r} does not match expected {expected!r}")
    if int(policy_inputs.get("route_scalar_dim")) != observation_spec.modal("route_scalar").shape[0]:
        raise ValueError("policy_inputs.route_scalar_dim does not match observation spec")
    if int(policy_inputs.get("sensor_health_dim")) != observation_spec.modal("sensor_health").shape[0]:
        raise ValueError("policy_inputs.sensor_health_dim does not match observation spec")

    gt_shape = _as_shape(target_only, "gt_bev_target", ndim=3)
    if gt_shape != target_spec.gt_bev_shape:
        raise ValueError(f"target_only.gt_bev_target={gt_shape!r} does not match expected {target_spec.gt_bev_shape!r}")
    waypoint_shape = _as_shape(target_only, "expert_waypoints", ndim=2)
    if waypoint_shape != target_spec.expert_waypoint_shape:
        raise ValueError(
            f"target_only.expert_waypoints={waypoint_shape!r} does not match expected {target_spec.expert_waypoint_shape!r}"
        )
    speed_shape = _as_shape(target_only, "expert_target_speed", ndim=1)
    if speed_shape != target_spec.expert_target_speed_shape:
        raise ValueError(
            f"target_only.expert_target_speed={speed_shape!r} does not match expected {target_spec.expert_target_speed_shape!r}"
        )


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"{key} must be a mapping")
    return value


def _require_sequence(mapping: dict[str, Any], key: str) -> list[Any] | tuple[Any, ...]:
    value = mapping.get(key)
    if not isinstance(value, (list, tuple)):
        raise KeyError(f"{key} must be a sequence")
    return value


def _as_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise KeyError(f"{key} is required")
    return int(mapping[key])


def _as_float(mapping: dict[str, Any], key: str) -> float:
    if key not in mapping:
        raise KeyError(f"{key} is required")
    return float(mapping[key])


def _as_shape(mapping: dict[str, Any], key: str, *, ndim: int) -> tuple[int, ...]:
    values = _require_sequence(mapping, key)
    shape = tuple(int(v) for v in values)
    if len(shape) != int(ndim) or any(dim <= 0 for dim in shape):
        raise ValueError(f"{key} must be a positive {ndim}D shape, got {values!r}")
    return shape
