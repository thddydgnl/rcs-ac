from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rcs_ac_first.runtime.auxiliary import (
    QUALITY_STATS_FIELDS,
    RISK_STATS_FIELDS,
    ROUTE_COMMITMENT_STATS_FIELDS,
    SENSOR_DROPOUT_KEYS,
    AuxiliaryObservationSpec,
    build_auxiliary_observation_spec,
    flatten_auxiliary_observation_dict,
    split_flat_auxiliary_observation,
)

MODALITY_ORDER: tuple[str, ...] = ("rgb", "lidar", "ego", "route")


def build_reliability_targets_from_info(
    info: dict[str, Any] | None,
    auxiliary_observation: np.ndarray | None = None,
    *,
    spec: AuxiliaryObservationSpec | None = None,
) -> np.ndarray:
    spec = spec or build_auxiliary_observation_spec()
    info = dict(info or {})
    aux_dict = _split_numpy_auxiliary(auxiliary_observation, spec) if auxiliary_observation is not None else {}
    quality = aux_dict.get("quality_stats", np.zeros((spec.item("quality_stats").size,), dtype=np.float32))
    dropout = aux_dict.get("sensor_dropout_mask", np.zeros((len(MODALITY_ORDER),), dtype=np.float32)).copy()
    for index, key in enumerate(MODALITY_ORDER):
        dropout[index] = max(float(dropout[index]), float(info.get(f"drop_{key}", 0.0)))

    target = 1.0 - 0.95 * np.clip(dropout[: len(MODALITY_ORDER)], 0.0, 1.0)
    if quality.size >= len(QUALITY_STATS_FIELDS):
        target[0] = min(target[0], 1.0 - 0.95 * float(quality[QUALITY_STATS_FIELDS.index("rgb_dropout_flag")] >= 0.5))
        lidar_drop = float(quality[QUALITY_STATS_FIELDS.index("lidar_drop_ratio")])
        lidar_sparse = float(quality[QUALITY_STATS_FIELDS.index("lidar_sparsity")])
        target[1] = min(target[1], 1.0 - 0.75 * np.clip(max(lidar_drop, lidar_sparse - 0.95), 0.0, 1.0))
        ego_flag = max(
            float(quality[QUALITY_STATS_FIELDS.index("ego_jerk_anomaly")] >= 0.5),
            float(quality[QUALITY_STATS_FIELDS.index("ego_state_mismatch_flag")] >= 0.5),
        )
        target[2] = min(target[2], 1.0 - 0.75 * ego_flag)
        route_mismatch = abs(float(quality[QUALITY_STATS_FIELDS.index("route_heading_mismatch")]))
        target[3] = min(target[3], 1.0 - 0.60 * np.clip(route_mismatch / 45.0, 0.0, 1.0))
    return np.clip(target, 0.05, 1.0).astype(np.float32, copy=False)


def build_risk_target_from_auxiliary(
    auxiliary_observation: np.ndarray | None,
    *,
    spec: AuxiliaryObservationSpec | None = None,
) -> float:
    spec = spec or build_auxiliary_observation_spec()
    aux_dict = _split_numpy_auxiliary(auxiliary_observation, spec) if auxiliary_observation is not None else {}
    risk = aux_dict.get("risk_stats", np.zeros((spec.item("risk_stats").size,), dtype=np.float32))
    route = aux_dict.get(
        "route_commitment_stats",
        np.zeros((spec.item("route_commitment_stats").size,), dtype=np.float32),
    )
    dropout = aux_dict.get("sensor_dropout_mask", np.zeros((len(MODALITY_ORDER),), dtype=np.float32))
    return float(_risk_formula_numpy(risk, route, dropout))


def build_phase2_targets_from_info(
    info: dict[str, Any] | None,
    auxiliary_observation: np.ndarray | None,
    *,
    spec: AuxiliaryObservationSpec | None = None,
) -> tuple[np.ndarray, float, float]:
    spec = spec or build_auxiliary_observation_spec()
    reliability = build_reliability_targets_from_info(info, auxiliary_observation, spec=spec)
    risk = build_risk_target_from_auxiliary(auxiliary_observation, spec=spec)
    corruption_level = float((info or {}).get("corruption_level", 0.0))
    return reliability, risk, corruption_level


def reliability_prior_from_auxiliary_tensor(
    auxiliary: torch.Tensor,
    *,
    spec: AuxiliaryObservationSpec | None = None,
    minimum: float = 0.02,
) -> torch.Tensor:
    spec = spec or build_auxiliary_observation_spec()
    split = split_flat_auxiliary_observation(auxiliary, spec)
    quality = split["quality_stats"]
    dropout = split["sensor_dropout_mask"].clamp(0.0, 1.0)
    target = 1.0 - 0.95 * dropout
    if quality.shape[-1] >= len(QUALITY_STATS_FIELDS):
        rgb_drop = (quality[:, QUALITY_STATS_FIELDS.index("rgb_dropout_flag")] >= 0.5).to(dtype=target.dtype)
        target[:, 0] = torch.minimum(target[:, 0], 1.0 - 0.95 * rgb_drop)
        lidar_drop = quality[:, QUALITY_STATS_FIELDS.index("lidar_drop_ratio")].clamp(0.0, 1.0)
        lidar_sparse = (quality[:, QUALITY_STATS_FIELDS.index("lidar_sparsity")] - 0.95).clamp(0.0, 1.0)
        target[:, 1] = torch.minimum(target[:, 1], 1.0 - 0.75 * torch.maximum(lidar_drop, lidar_sparse))
        ego_flag = torch.maximum(
            (quality[:, QUALITY_STATS_FIELDS.index("ego_jerk_anomaly")] >= 0.5).to(dtype=target.dtype),
            (quality[:, QUALITY_STATS_FIELDS.index("ego_state_mismatch_flag")] >= 0.5).to(dtype=target.dtype),
        )
        target[:, 2] = torch.minimum(target[:, 2], 1.0 - 0.75 * ego_flag)
        route_mismatch = quality[:, QUALITY_STATS_FIELDS.index("route_heading_mismatch")].abs()
        target[:, 3] = torch.minimum(target[:, 3], 1.0 - 0.60 * (route_mismatch / 45.0).clamp(0.0, 1.0))
    return target.clamp(float(minimum), 1.0)


def risk_prior_from_auxiliary_tensor(
    auxiliary: torch.Tensor,
    *,
    spec: AuxiliaryObservationSpec | None = None,
) -> torch.Tensor:
    spec = spec or build_auxiliary_observation_spec()
    split = split_flat_auxiliary_observation(auxiliary, spec)
    return _risk_formula_torch(split["risk_stats"], split["route_commitment_stats"], split["sensor_dropout_mask"])


def _split_numpy_auxiliary(
    auxiliary_observation: np.ndarray | None,
    spec: AuxiliaryObservationSpec,
) -> dict[str, np.ndarray]:
    if auxiliary_observation is None:
        return {}
    flat = np.asarray(auxiliary_observation, dtype=np.float32).reshape(-1)
    if flat.size != int(spec.total_dim):
        flat = flatten_auxiliary_observation_dict({}, spec)
    result: dict[str, np.ndarray] = {}
    for item in spec.slices:
        result[item.key] = flat[item.start : item.stop].reshape(item.shape)
    return result


def _risk_formula_numpy(risk: np.ndarray, route: np.ndarray, dropout: np.ndarray) -> np.float32:
    values = np.zeros((len(RISK_STATS_FIELDS),), dtype=np.float32)
    values[: min(values.size, risk.size)] = risk[: min(values.size, risk.size)]
    route_values = np.zeros((len(ROUTE_COMMITMENT_STATS_FIELDS),), dtype=np.float32)
    route_values[: min(route_values.size, route.size)] = route[: min(route_values.size, route.size)]
    dropout_level = float(np.clip(dropout, 0.0, 1.0).max()) if dropout.size else 0.0
    risk_value = (
        0.18 * np.clip(abs(values[0]) / 15.0, 0.0, 1.0)
        + 0.20 * np.clip(abs(values[1]) / 2.0, 0.0, 1.0)
        + 0.18 * np.clip(abs(values[2]) / 45.0, 0.0, 1.0)
        + 0.10 * np.clip(abs(values[3]) / 45.0, 0.0, 1.0)
        + 0.07 * np.clip(values[5], 0.0, 1.0)
        + 0.17 * dropout_level
        + 0.10 * np.clip(route_values[3], 0.0, 1.0)
    )
    return np.float32(np.clip(risk_value, 0.0, 1.0))


def _risk_formula_torch(risk: torch.Tensor, route: torch.Tensor, dropout: torch.Tensor) -> torch.Tensor:
    speed = (risk[:, 0].abs() / 15.0).clamp(0.0, 1.0)
    lane = (risk[:, 1].abs() / 2.0).clamp(0.0, 1.0)
    heading = (risk[:, 2].abs() / 45.0).clamp(0.0, 1.0)
    curve = (risk[:, 3].abs() / 45.0).clamp(0.0, 1.0)
    weather = risk[:, 5].clamp(0.0, 1.0)
    dropout_level = dropout.clamp(0.0, 1.0).amax(dim=-1)
    route_mismatch = route[:, 3].clamp(0.0, 1.0)
    return (
        0.18 * speed
        + 0.20 * lane
        + 0.18 * heading
        + 0.10 * curve
        + 0.07 * weather
        + 0.17 * dropout_level
        + 0.10 * route_mismatch
    ).clamp(0.0, 1.0)
