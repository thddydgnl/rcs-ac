from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

QUALITY_STATS_FIELDS: tuple[str, ...] = (
    "rgb_brightness",
    "rgb_contrast",
    "rgb_entropy",
    "rgb_blur_proxy",
    "rgb_dropout_flag",
    "lidar_point_count",
    "lidar_valid_ratio",
    "lidar_sparsity",
    "lidar_drop_ratio",
    "ego_jerk_magnitude",
    "ego_jerk_anomaly",
    "ego_state_mismatch_flag",
    "route_continuity",
    "route_heading_mismatch",
    "route_availability_flag",
)
RISK_STATS_FIELDS: tuple[str, ...] = (
    "speed_mps",
    "lane_offset",
    "heading_error",
    "curve_proxy",
    "target_distance",
    "weather_severity",
)
ROUTE_COMMITMENT_STATS_FIELDS: tuple[str, ...] = (
    "route_progress",
    "remaining_route_ratio",
    "target_distance_ratio",
    "route_heading_mismatch_norm",
    "route_continuity_score",
)
SENSOR_DROPOUT_KEYS: tuple[str, ...] = ("rgb", "lidar", "ego", "route")
AUXILIARY_OBSERVATION_ORDER: tuple[str, ...] = (
    "quality_stats",
    "risk_stats",
    "route_commitment_stats",
    "sensor_dropout_mask",
)

DEFAULT_QUALITY_STATS_DIM = len(QUALITY_STATS_FIELDS)
DEFAULT_RISK_STATS_DIM = len(RISK_STATS_FIELDS)
DEFAULT_ROUTE_COMMITMENT_STATS_DIM = len(ROUTE_COMMITMENT_STATS_FIELDS)
DEFAULT_SENSOR_DROPOUT_DIM = len(SENSOR_DROPOUT_KEYS)
DEFAULT_AUXILIARY_DIM = (
    DEFAULT_QUALITY_STATS_DIM
    + DEFAULT_RISK_STATS_DIM
    + DEFAULT_ROUTE_COMMITMENT_STATS_DIM
    + DEFAULT_SENSOR_DROPOUT_DIM
)


@dataclass(frozen=True)
class AuxiliarySliceSpec:
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
class AuxiliaryObservationSpec:
    slices: tuple[AuxiliarySliceSpec, ...]
    total_dim: int

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.slices)

    def item(self, key: str) -> AuxiliarySliceSpec:
        for spec in self.slices:
            if spec.key == key:
                return spec
        raise KeyError(f"Unknown auxiliary observation key: {key}")


def build_auxiliary_observation_spec(
    *,
    quality_stats_dim: int = DEFAULT_QUALITY_STATS_DIM,
    risk_stats_dim: int = DEFAULT_RISK_STATS_DIM,
    route_commitment_stats_dim: int = DEFAULT_ROUTE_COMMITMENT_STATS_DIM,
    sensor_dropout_dim: int = DEFAULT_SENSOR_DROPOUT_DIM,
) -> AuxiliaryObservationSpec:
    dims = {
        "quality_stats": int(quality_stats_dim),
        "risk_stats": int(risk_stats_dim),
        "route_commitment_stats": int(route_commitment_stats_dim),
        "sensor_dropout_mask": int(sensor_dropout_dim),
    }
    for key, value in dims.items():
        if value <= 0:
            raise ValueError(f"{key} dim must be > 0, got {value}")

    start = 0
    slices: list[AuxiliarySliceSpec] = []
    for key in AUXILIARY_OBSERVATION_ORDER:
        stop = start + dims[key]
        slices.append(AuxiliarySliceSpec(key=key, shape=(dims[key],), start=start, stop=stop))
        start = stop
    return AuxiliaryObservationSpec(slices=tuple(slices), total_dim=int(start))


def default_auxiliary_observation_dict(spec: AuxiliaryObservationSpec | None = None) -> dict[str, np.ndarray]:
    spec = spec or build_auxiliary_observation_spec()
    result = {item.key: np.zeros(item.shape, dtype=np.float32) for item in spec.slices}
    if int(spec.item("quality_stats").size) >= DEFAULT_QUALITY_STATS_DIM:
        quality = result["quality_stats"]
        quality[QUALITY_STATS_FIELDS.index("route_availability_flag")] = 1.0
        quality[QUALITY_STATS_FIELDS.index("route_continuity")] = 1.0
    if int(spec.item("route_commitment_stats").size) >= DEFAULT_ROUTE_COMMITMENT_STATS_DIM:
        route = result["route_commitment_stats"]
        route[ROUTE_COMMITMENT_STATS_FIELDS.index("remaining_route_ratio")] = 1.0
        route[ROUTE_COMMITMENT_STATS_FIELDS.index("route_continuity_score")] = 1.0
    return result


def validate_auxiliary_observation_dict(
    observation: dict[str, Any],
    spec: AuxiliaryObservationSpec,
    *,
    allow_missing: bool = True,
) -> dict[str, np.ndarray]:
    fallback = default_auxiliary_observation_dict(spec)
    validated: dict[str, np.ndarray] = {}
    for item in spec.slices:
        if item.key not in observation:
            if not allow_missing:
                raise KeyError(f"Auxiliary observation missing key: {item.key}")
            validated[item.key] = fallback[item.key]
            continue
        value = np.asarray(observation[item.key], dtype=np.float32)
        if tuple(value.shape) != item.shape:
            try:
                value = value.reshape(item.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Auxiliary key '{item.key}' expected shape {item.shape}, got {tuple(value.shape)}"
                ) from exc
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Auxiliary key '{item.key}' contains NaN or inf values.")
        validated[item.key] = value.astype(np.float32, copy=False)
    return validated


def flatten_auxiliary_observation_dict(
    observation: dict[str, Any],
    spec: AuxiliaryObservationSpec | None = None,
) -> np.ndarray:
    spec = spec or build_auxiliary_observation_spec()
    validated = validate_auxiliary_observation_dict(observation, spec, allow_missing=True)
    return np.concatenate([validated[item.key].reshape(-1) for item in spec.slices], axis=0).astype(
        np.float32,
        copy=False,
    )


def split_flat_auxiliary_observation(
    flat_auxiliary: Any,
    spec: AuxiliaryObservationSpec | None = None,
) -> dict[str, torch.Tensor]:
    spec = spec or build_auxiliary_observation_spec()
    if isinstance(flat_auxiliary, torch.Tensor):
        flat = flat_auxiliary.to(dtype=torch.float32)
    else:
        flat = torch.as_tensor(flat_auxiliary, dtype=torch.float32)
    if flat.ndim == 1:
        flat = flat.unsqueeze(0)
    if flat.ndim != 2:
        raise ValueError(f"Expected flat auxiliary observation [B, aux_dim], got {tuple(flat.shape)}")
    if int(flat.shape[-1]) != int(spec.total_dim):
        raise ValueError(f"Expected auxiliary dim {spec.total_dim}, got {int(flat.shape[-1])}")
    if not torch.isfinite(flat).all():
        raise ValueError("Flat auxiliary observation contains NaN or inf values.")
    return {item.key: flat[:, item.start : item.stop].reshape((int(flat.shape[0]),) + item.shape) for item in spec.slices}


def default_flat_auxiliary(
    batch_size: int,
    *,
    spec: AuxiliaryObservationSpec | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    spec = spec or build_auxiliary_observation_spec()
    base = flatten_auxiliary_observation_dict(default_auxiliary_observation_dict(spec), spec)
    tensor = torch.as_tensor(base, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0).repeat(int(batch_size), 1)
