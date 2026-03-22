"""Shared modality-specific corruption helpers for training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from rcs_ac_v1.model import MODALITY_NAMES
from stage1_warmup.evaluate import aux_to_reliability_targets


def normalize_pair(sin_val: float, cos_val: float) -> tuple[float, float]:
    norm = max(np.sqrt(sin_val * sin_val + cos_val * cos_val), 1e-6)
    return float(sin_val / norm), float(cos_val / norm)


def default_corruption_params(corruption_type: str, corruption_strength: float) -> dict[str, float]:
    strength = float(corruption_strength)
    params: dict[str, float] = {}
    if corruption_type == "vision":
        params["noise_std"] = strength
        params["dropout_prob"] = min(0.10, strength)
    elif corruption_type == "distance":
        params["noise_std"] = 0.8 * strength
        params["dropout_prob"] = min(0.10, strength)
    elif corruption_type == "ego":
        params["noise_std"] = 0.7 * strength
        params["bias"] = 0.5 * strength
    elif corruption_type == "route":
        params["angle_noise_std"] = 1.5 * strength
        params["distance_bias"] = 0.8 * strength
    return params


def resolve_corruption_spec(
    corruption_type: str,
    corruption_strength: float,
    corruption_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = {
        "corruption_type": str(corruption_type),
        "corruption_strength": float(corruption_strength),
    }
    resolved.update(default_corruption_params(corruption_type, corruption_strength))
    if corruption_params:
        for key, value in corruption_params.items():
            if key not in {"corruption_type", "corruption_strength", "probe_name"} and value is not None:
                resolved[key] = float(value)
    return resolved


def apply_shared_corruption(
    obs: dict[str, np.ndarray],
    corruption_type: str,
    corruption_strength: float,
    rng: np.random.Generator,
    corruption_params: dict[str, Any] | None = None,
    probe_name: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Apply corruption with a single shared implementation for train/eval paths."""
    out = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
    severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
    resolved = resolve_corruption_spec(corruption_type, corruption_strength, corruption_params)
    metadata = {
        "requested_probe_name": probe_name or resolved["corruption_type"],
        "resolved_corruption_type": resolved["corruption_type"],
        "resolved_corruption_strength": resolved["corruption_strength"],
        "corruption_applied_flag": int(
            resolved["corruption_type"] != "clean" and resolved["corruption_strength"] > 0.0
        ),
        "resolved_corruption_params": {
            key: value
            for key, value in resolved.items()
            if key not in {"corruption_type", "corruption_strength"}
        },
    }

    if resolved["corruption_type"] == "clean" or resolved["corruption_strength"] <= 0.0:
        return out, severity, metadata

    if resolved["corruption_type"] == "vision":
        x = out["vision"]
        x = x + rng.normal(0.0, resolved.get("noise_std", 0.0), size=x.shape).astype(np.float32)
        dropout_prob = float(resolved.get("dropout_prob", 0.0))
        if dropout_prob > 0.0:
            mask = (rng.random(x.shape) > dropout_prob).astype(np.float32)
            x = x * mask
        out["vision"] = np.clip(x, 0.0, 1.0).astype(np.float32)
        severity[MODALITY_NAMES.index("vision")] = float(resolved["corruption_strength"])
    elif resolved["corruption_type"] == "distance":
        x = out["distance"]
        x = x + rng.normal(0.0, resolved.get("noise_std", 0.0), size=x.shape).astype(np.float32)
        dropout_prob = float(resolved.get("dropout_prob", 0.0))
        if dropout_prob > 0.0:
            mask = (rng.random(x.shape) > dropout_prob).astype(np.float32)
            x = x * mask
        out["distance"] = np.clip(x, 0.0, 1.0).astype(np.float32)
        severity[MODALITY_NAMES.index("distance")] = float(resolved["corruption_strength"])
    elif resolved["corruption_type"] == "ego":
        x = out["ego"]
        x = x + rng.normal(0.0, resolved.get("noise_std", 0.0), size=x.shape).astype(np.float32)
        bias = float(resolved.get("bias", 0.0))
        if bias != 0.0:
            x[0] = np.clip(x[0] + bias, -1.0, 1.0)
            x[3] = np.clip(x[3] + 0.5 * bias, -1.0, 1.0)
            x[4] = np.clip(x[4] - 0.5 * bias, -1.0, 1.0)
        out["ego"] = np.clip(x, -1.0, 1.0).astype(np.float32)
        severity[MODALITY_NAMES.index("ego")] = float(resolved["corruption_strength"])
    elif resolved["corruption_type"] == "route":
        x = out["route"].copy()
        rel_angle = float(np.arctan2(x[0], np.clip(x[1], -1.0, 1.0)))
        rel_angle += float(rng.normal(0.0, resolved.get("angle_noise_std", 0.0)))
        x[0] = np.sin(rel_angle)
        x[1] = np.cos(rel_angle)
        x[2] = np.clip(x[2] + float(resolved.get("distance_bias", 0.0)), 0.0, 1.0)
        if x.shape[0] >= 5:
            s, c = normalize_pair(
                float(x[3] + 0.5 * np.sin(rel_angle) * resolved["corruption_strength"]),
                float(x[4] + 0.5 * np.cos(rel_angle) * resolved["corruption_strength"]),
            )
            x[3] = s
            x[4] = c
        out["route"] = np.clip(x, -1.0, 1.0).astype(np.float32)
        severity[MODALITY_NAMES.index("route")] = float(resolved["corruption_strength"])
    else:
        raise ValueError(f"Unsupported corruption_type: {resolved['corruption_type']}")

    return out, severity, metadata


def apply_targeted_corruption(
    obs: dict[str, np.ndarray],
    corruption_type: str,
    corruption_strength: float,
    rng: np.random.Generator,
    corruption_params: dict[str, Any] | None = None,
    probe_name: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    corrupted, severity, _metadata = apply_shared_corruption(
        obs=obs,
        corruption_type=corruption_type,
        corruption_strength=corruption_strength,
        rng=rng,
        corruption_params=corruption_params,
        probe_name=probe_name,
    )
    return corrupted, severity


def build_reliability_targets(
    aux_obs: np.ndarray,
    severity: np.ndarray,
    use_direct_corruption_targets: bool,
    direct_target_drop_scale: float,
) -> np.ndarray:
    targets = aux_to_reliability_targets(aux_obs)
    if use_direct_corruption_targets:
        drop = np.clip(severity * direct_target_drop_scale, 0.0, 0.35).astype(np.float32)
        targets = np.clip(targets - drop, 0.0, 1.0)
    return targets.astype(np.float32)
