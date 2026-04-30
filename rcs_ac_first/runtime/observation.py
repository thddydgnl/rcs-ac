from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

DEFAULT_RGB_SHAPE: tuple[int, int, int] = (3, 90, 160)
DEFAULT_LIDAR_SHAPE: tuple[int, int, int] = (1, 64, 64)
DEFAULT_EGO_DIM = 5
DEFAULT_ROUTE_DIM = 10
POLICY_MODALITY_ORDER: tuple[str, ...] = ("rgb", "lidar", "ego", "route")


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
class PolicyObservationSpec:
    modalities: tuple[ObservationSliceSpec, ...]
    total_dim: int

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(modality.key for modality in self.modalities)

    @property
    def slices(self) -> dict[str, slice]:
        return {modality.key: modality.slice for modality in self.modalities}

    def modal(self, key: str) -> ObservationSliceSpec:
        for modality in self.modalities:
            if modality.key == key:
                return modality
        raise KeyError(f"Unknown policy observation key: {key}")


def _shape_tuple(shape: tuple[int, ...] | list[int], *, name: str, ndim: int) -> tuple[int, ...]:
    values = tuple(int(dim) for dim in shape)
    if len(values) != ndim or any(dim <= 0 for dim in values):
        raise ValueError(f"{name} must be a positive {ndim}D shape, got {shape!r}")
    return values


def build_policy_observation_spec(
    rgb_shape: tuple[int, int, int] | list[int] = DEFAULT_RGB_SHAPE,
    lidar_shape: tuple[int, int, int] | list[int] = DEFAULT_LIDAR_SHAPE,
    ego_dim: int = DEFAULT_EGO_DIM,
    route_dim: int = DEFAULT_ROUTE_DIM,
) -> PolicyObservationSpec:
    modality_shapes: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("rgb", _shape_tuple(rgb_shape, name="rgb_shape", ndim=3)),
        ("lidar", _shape_tuple(lidar_shape, name="lidar_shape", ndim=3)),
        ("ego", (int(ego_dim),)),
        ("route", (int(route_dim),)),
    )
    if int(ego_dim) <= 0:
        raise ValueError(f"ego_dim must be > 0, got {ego_dim}")
    if int(route_dim) <= 0:
        raise ValueError(f"route_dim must be > 0, got {route_dim}")

    start = 0
    modalities: list[ObservationSliceSpec] = []
    for key, shape in modality_shapes:
        size = int(np.prod(shape, dtype=np.int64))
        stop = start + size
        modalities.append(ObservationSliceSpec(key=key, shape=shape, start=start, stop=stop))
        start = stop
    return PolicyObservationSpec(modalities=tuple(modalities), total_dim=int(start))


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def split_flat_observation(
    flat_observation: Any,
    spec: PolicyObservationSpec,
) -> dict[str, torch.Tensor]:
    flat = _as_float_tensor(flat_observation)
    if flat.ndim == 1:
        flat = flat.unsqueeze(0)
    if flat.ndim != 2:
        raise ValueError(f"Expected flat observation with shape [B, obs_dim], got {tuple(flat.shape)}")
    if int(flat.shape[-1]) != int(spec.total_dim):
        raise ValueError(
            f"Expected flat observation dim {spec.total_dim}, got {int(flat.shape[-1])}"
        )
    if not torch.isfinite(flat).all():
        raise ValueError("Flat observation contains NaN or inf values.")

    batch_size = int(flat.shape[0])
    split: dict[str, torch.Tensor] = {}
    for modality in spec.modalities:
        values = flat[:, modality.start : modality.stop]
        split[modality.key] = values.reshape((batch_size,) + tuple(modality.shape))
    return split


def validate_observation_dict(
    observation: dict[str, Any],
    spec: PolicyObservationSpec,
) -> dict[str, np.ndarray]:
    missing = [key for key in spec.keys if key not in observation]
    if missing:
        raise KeyError(f"Observation is missing policy keys: {missing}")

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


def flatten_observation_dict(
    observation: dict[str, Any],
    spec: PolicyObservationSpec,
) -> np.ndarray:
    validated = validate_observation_dict(observation, spec)
    return np.concatenate(
        [validated[modality.key].reshape(-1) for modality in spec.modalities],
        axis=0,
    ).astype(np.float32, copy=False)


def summarize_policy_observation_spec(spec: PolicyObservationSpec) -> dict[str, dict[str, Any]]:
    return {
        modality.key: {
            "shape": tuple(modality.shape),
            "start": int(modality.start),
            "stop": int(modality.stop),
            "size": int(modality.size),
        }
        for modality in spec.modalities
    }
