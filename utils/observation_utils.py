from __future__ import annotations

from typing import Any

import numpy as np

REQUIRED_MODAL_KEYS = ("rgb", "lidar", "ego", "route")


def _as_float32_array(value: Any, key: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Observation key '{key}' is None.")
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Observation key '{key}' is empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Observation key '{key}' contains NaN or inf values.")
    return array


def validate_observation_dict(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_MODAL_KEYS if key not in observation]
    if missing:
        raise KeyError(f"Observation is missing required modality keys: {missing}")

    validated = {key: _as_float32_array(observation[key], key) for key in REQUIRED_MODAL_KEYS}

    if validated["rgb"].ndim != 3:
        raise ValueError(f"Expected 'rgb' observation with 3 dims, got shape {validated['rgb'].shape}")
    if validated["lidar"].ndim != 3:
        raise ValueError(f"Expected 'lidar' observation with 3 dims, got shape {validated['lidar'].shape}")
    if validated["ego"].ndim != 1:
        raise ValueError(f"Expected 'ego' observation with 1 dim, got shape {validated['ego'].shape}")
    if validated["route"].ndim != 1:
        raise ValueError(f"Expected 'route' observation with 1 dim, got shape {validated['route'].shape}")
    return validated


def flatten_multimodal_observation(observation: dict[str, Any]) -> np.ndarray:
    validated = validate_observation_dict(observation)
    return np.concatenate(
        [
            validated["rgb"].reshape(-1),
            validated["lidar"].reshape(-1),
            validated["ego"].reshape(-1),
            validated["route"].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def summarize_observation_shapes(observation: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    validated = validate_observation_dict(observation)
    return {key: tuple(value.shape) for key, value in validated.items()}
