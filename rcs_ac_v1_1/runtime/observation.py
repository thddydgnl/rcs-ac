from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rcs_ac_v1_1.runtime.bev_spec import BEVChannelSpec, BEVSpec, DEFAULT_BEV_CHANNELS, DEFAULT_BEV_SPEC

DEFAULT_RGB_SHAPE: tuple[int, int, int] = (3, 90, 160)
DEFAULT_ROUTE_SCALAR_DIM = 10
DEFAULT_EGO_SCALAR_DIM = 6
DEFAULT_SENSOR_HEALTH_DIM = 15
DEFAULT_WAYPOINT_COUNT = 4

POLICY_MODALITY_ORDER: tuple[str, ...] = (
    "rgb",
    "lidar_bev",
    "route_bev",
    "ego_bev",
    "ego_scalar",
    "route_scalar",
    "sensor_health",
)


@dataclass(frozen=True)
class ObservationSliceSpec:
    key: str
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def size(self) -> int:
        return int(self.stop - self.start)

    @property
    def slice(self) -> slice:
        return slice(int(self.start), int(self.stop))


@dataclass(frozen=True)
class V11PolicyObservationSpec:
    modalities: tuple[ObservationSliceSpec, ...]
    total_dim: int
    bev_spec: BEVSpec
    bev_channels: BEVChannelSpec

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(modality.key for modality in self.modalities)

    def modal(self, key: str) -> ObservationSliceSpec:
        for modality in self.modalities:
            if modality.key == key:
                return modality
        raise KeyError(f"Unknown v1.1 policy observation key: {key}")


@dataclass(frozen=True)
class V11TargetSpec:
    gt_bev_shape: tuple[int, int, int]
    expert_waypoint_shape: tuple[int, int]
    expert_target_speed_shape: tuple[int, ...] = (1,)


def _shape_tuple(shape: tuple[int, ...] | list[int], *, name: str, ndim: int) -> tuple[int, ...]:
    values = tuple(int(dim) for dim in shape)
    if len(values) != ndim or any(dim <= 0 for dim in values):
        raise ValueError(f"{name} must be a positive {ndim}D shape, got {shape!r}")
    return values


def build_v11_policy_observation_spec(
    *,
    rgb_shape: tuple[int, int, int] | list[int] = DEFAULT_RGB_SHAPE,
    bev_spec: BEVSpec = DEFAULT_BEV_SPEC,
    bev_channels: BEVChannelSpec = DEFAULT_BEV_CHANNELS,
    ego_scalar_dim: int = DEFAULT_EGO_SCALAR_DIM,
    route_scalar_dim: int = DEFAULT_ROUTE_SCALAR_DIM,
    sensor_health_dim: int = DEFAULT_SENSOR_HEALTH_DIM,
) -> V11PolicyObservationSpec:
    for name, value in (
        ("ego_scalar_dim", ego_scalar_dim),
        ("route_scalar_dim", route_scalar_dim),
        ("sensor_health_dim", sensor_health_dim),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    modality_shapes: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("rgb", _shape_tuple(rgb_shape, name="rgb_shape", ndim=3)),
        ("lidar_bev", bev_spec.channel_shape(bev_channels.lidar_channels)),
        ("route_bev", bev_spec.channel_shape(bev_channels.route_channels)),
        ("ego_bev", bev_spec.channel_shape(bev_channels.ego_channels)),
        ("ego_scalar", (int(ego_scalar_dim),)),
        ("route_scalar", (int(route_scalar_dim),)),
        ("sensor_health", (int(sensor_health_dim),)),
    )

    start = 0
    modalities: list[ObservationSliceSpec] = []
    for key, shape in modality_shapes:
        size = int(np.prod(shape, dtype=np.int64))
        stop = start + size
        modalities.append(ObservationSliceSpec(key=key, shape=shape, start=start, stop=stop))
        start = stop
    return V11PolicyObservationSpec(
        modalities=tuple(modalities),
        total_dim=int(start),
        bev_spec=bev_spec,
        bev_channels=bev_channels,
    )


def build_v11_target_spec(
    *,
    bev_spec: BEVSpec = DEFAULT_BEV_SPEC,
    bev_channels: BEVChannelSpec = DEFAULT_BEV_CHANNELS,
    waypoint_count: int = DEFAULT_WAYPOINT_COUNT,
) -> V11TargetSpec:
    if int(waypoint_count) <= 0:
        raise ValueError("waypoint_count must be > 0")
    return V11TargetSpec(
        gt_bev_shape=bev_spec.channel_shape(bev_channels.target_channels),
        expert_waypoint_shape=(int(waypoint_count), 2),
    )


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def split_flat_observation(flat_observation: Any, spec: V11PolicyObservationSpec) -> dict[str, torch.Tensor]:
    flat = _as_float_tensor(flat_observation)
    if flat.ndim == 1:
        flat = flat.unsqueeze(0)
    if flat.ndim != 2:
        raise ValueError(f"Expected flat observation [B, obs_dim], got {tuple(flat.shape)}")
    if int(flat.shape[-1]) != int(spec.total_dim):
        raise ValueError(f"Expected flat observation dim {spec.total_dim}, got {int(flat.shape[-1])}")
    if not torch.isfinite(flat).all():
        raise ValueError("Flat observation contains NaN or inf values.")

    batch_size = int(flat.shape[0])
    result: dict[str, torch.Tensor] = {}
    for modality in spec.modalities:
        values = flat[:, modality.start : modality.stop]
        result[modality.key] = values.reshape((batch_size,) + tuple(modality.shape))
    return result


def validate_observation_dict(observation: dict[str, Any], spec: V11PolicyObservationSpec) -> dict[str, np.ndarray]:
    missing = [key for key in spec.keys if key not in observation]
    if missing:
        raise KeyError(f"Observation is missing v1.1 policy keys: {missing}")

    validated: dict[str, np.ndarray] = {}
    for modality in spec.modalities:
        value = np.asarray(observation[modality.key], dtype=np.float32)
        if tuple(value.shape) != tuple(modality.shape):
            raise ValueError(
                f"Observation key '{modality.key}' expected shape {modality.shape}, got {tuple(value.shape)}"
            )
        if value.size == 0:
            raise ValueError(f"Observation key '{modality.key}' is empty.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Observation key '{modality.key}' contains NaN or inf values.")
        validated[modality.key] = value
    return validated


def flatten_observation_dict(observation: dict[str, Any], spec: V11PolicyObservationSpec) -> np.ndarray:
    validated = validate_observation_dict(observation, spec)
    return np.concatenate(
        [validated[modality.key].reshape(-1) for modality in spec.modalities],
        axis=0,
    ).astype(np.float32, copy=False)


def summarize_v11_policy_observation_spec(spec: V11PolicyObservationSpec) -> dict[str, dict[str, Any]]:
    return {
        modality.key: {
            "shape": tuple(modality.shape),
            "start": int(modality.start),
            "stop": int(modality.stop),
            "size": int(modality.size),
        }
        for modality in spec.modalities
    }
