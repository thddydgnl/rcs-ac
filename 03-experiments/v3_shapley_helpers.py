"""Helpers for Version 3 counterfactual Shapley alignment runs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rcs_ac_v1.model import MODALITY_NAMES
from stage2_transition_helpers import (
    Stage2TransitionTrainer,
    _probe_logging_fields,
    aggregate_stage2_results,
    build_model_from_candidate,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    ensure_dir,
    infer_experiment_mode,
    is_calibration_only,
    latest_checkpoint,
    load_yaml,
    make_probe_rng,
    obs_to_tensor,
    save_yaml,
    stable_probe_offset,
    validate_v3_like_config,
    _resolve_seed_dir_from_root,
    build_reliability_targets,
)

_NUM_MODALITIES = len(MODALITY_NAMES)
_ALL_SUBSET_MASKS = tuple(range(1 << _NUM_MODALITIES))
_FACTORIAL = [math.factorial(i) for i in range(_NUM_MODALITIES + 1)]
_SHAPLEY_WEIGHTS = {
    subset_size: (_FACTORIAL[subset_size] * _FACTORIAL[_NUM_MODALITIES - subset_size - 1]) / _FACTORIAL[_NUM_MODALITIES]
    for subset_size in range(_NUM_MODALITIES)
}
_TRAIN_SHAPLEY_REFERENCE_MODES = {"clean_full", "current_state"}
_TRAIN_SHAPLEY_COUNTERFACTUAL_BASELINES = {"zero"}
_TRAIN_SHAPLEY_UTILITY_ALIASES = {
    "clean_fused_l2": "clean_fused_l2",
    "fused_feature_l2_to_full": "clean_fused_l2",
    "target_logit_clean_l1": "target_logit_clean_l1",
}

_GATE_LOSS_KEY_ALIASES = {
    "clean_gate_suppression_loss_coef": "clean_gate_cap_loss_coef",
    "mild_gate_activation_loss_coef": "mild_gate_floor_loss_coef",
    "gate_contrast_loss_coef": "gate_gap_margin_loss_coef",
    "mild_gate_activation_margin": "mild_gate_min",
    "gate_contrast_margin": "gate_gap_margin",
}


def canonicalize_gate_loss_keys(losses_cfg: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(losses_cfg)
    for old_key, new_key in _GATE_LOSS_KEY_ALIASES.items():
        if old_key not in canonical:
            continue
        if new_key in canonical:
            if canonical[new_key] != canonical[old_key]:
                warnings.warn(
                    f"Ignoring deprecated gate loss key {old_key} because canonical {new_key} is already set.",
                    stacklevel=2,
                )
            continue
        warnings.warn(
            f"Deprecated gate loss key {old_key} detected; mapping to canonical key {new_key}.",
            stacklevel=2,
        )
        canonical[new_key] = canonical[old_key]
    return canonical


def build_candidate_gate_thresholds(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    base_losses = canonicalize_gate_loss_keys(config["losses"])
    for candidate_name, candidate_spec in config["candidates"].items():
        candidate_losses = dict(base_losses)
        candidate_losses.update(candidate_spec.get("loss_overrides", {}))
        candidate_losses = canonicalize_gate_loss_keys(candidate_losses)
        thresholds[candidate_name] = {
            "clean_gate_max": float(candidate_losses.get("clean_gate_max", 0.02)),
            "mild_gate_min": float(candidate_losses.get("mild_gate_min", 0.08)),
            "gate_gap_margin": float(candidate_losses.get("gate_gap_margin", 0.05)),
            "clean_rescue_gate_max": float(candidate_losses.get("clean_rescue_gate_max", 0.10)),
            "clean_selector_max": float(candidate_losses.get("clean_selector_max", 0.10)),
        }
    return thresholds


def resolve_experiment_output_root(output_root: str | Path, smoke: bool) -> Path:
    base = Path(output_root)
    return ensure_dir(base.with_name(f"{base.name}_smoke") if smoke else base)


def compute_paired_selective_gate_losses(
    clean_gate: torch.Tensor,
    mild_gate: torch.Tensor,
    *,
    clean_gate_max: float,
    mild_gate_min: float,
    gate_gap_margin: float,
    residual_clean: torch.Tensor | None = None,
    residual_mild: torch.Tensor | None = None,
    residual_margin: float | None = None,
) -> dict[str, torch.Tensor]:
    device = clean_gate.device
    losses = {
        "clean_gate_cap": F.relu(clean_gate - torch.as_tensor(clean_gate_max, device=device)),
        "mild_gate_floor": F.relu(torch.as_tensor(mild_gate_min, device=device) - mild_gate),
        "gate_gap_margin": F.relu(torch.as_tensor(gate_gap_margin, device=device) - (mild_gate - clean_gate)),
    }
    if residual_clean is not None and residual_mild is not None and residual_margin is not None:
        losses["residual_gap_margin"] = F.relu(
            torch.as_tensor(residual_margin, device=device) - (residual_mild - residual_clean)
        )
    return losses


def _binary_auc_from_scores(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    neg = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    pos = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if neg.size == 0 or pos.size == 0:
        return 0.5
    scores = np.concatenate([neg, pos], axis=0)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(scores.size, dtype=np.float64) + 1.0
    pos_ranks = ranks[neg.size :]
    u_value = pos_ranks.sum() - (pos.size * (pos.size + 1) / 2.0)
    return float(u_value / max(pos.size * neg.size, 1))


def compute_linear_probe_auc(
    clean_features: np.ndarray,
    mild_features: np.ndarray,
    *,
    drop_slices: Sequence[tuple[int, int]] | None = None,
) -> float:
    clean = np.asarray(clean_features, dtype=np.float64)
    mild = np.asarray(mild_features, dtype=np.float64)
    if clean.ndim != 2 or mild.ndim != 2 or clean.shape[1] != mild.shape[1]:
        return 0.5
    if drop_slices:
        keep_mask = np.ones(clean.shape[1], dtype=bool)
        for start, end in drop_slices:
            keep_mask[int(start) : int(end)] = False
        if not np.any(keep_mask):
            return 0.5
        clean = clean[:, keep_mask]
        mild = mild[:, keep_mask]
    direction = mild.mean(axis=0) - clean.mean(axis=0)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1.0e-12:
        return 0.5
    clean_scores = clean @ direction
    mild_scores = mild @ direction
    return _binary_auc_from_scores(clean_scores, mild_scores)


def compute_gate_feature_separability_audit(
    clean_features: np.ndarray,
    mild_features: np.ndarray,
    feature_slices: dict[str, tuple[int, int]] | None,
) -> dict[str, float]:
    clean = np.asarray(clean_features, dtype=np.float64)
    mild = np.asarray(mild_features, dtype=np.float64)
    if clean.ndim != 2 or mild.ndim != 2 or clean.shape != mild.shape or clean.size == 0:
        return {
            "clean_vs_mild_gate_feature_auc": 0.5,
            "clean_vs_mild_gate_feature_delta_norm": 0.0,
            "source_gate_feature_onoff_auc_delta": 0.0,
            "teacher_delta_feature_onoff_auc_delta": 0.0,
        }
    full_auc = compute_linear_probe_auc(clean, mild)
    delta_norm = float(np.linalg.norm(mild - clean, axis=1).mean())
    feature_slices = dict(feature_slices or {})
    source_drop = [feature_slices["source_gate_scores"]] if "source_gate_scores" in feature_slices else []
    teacher_drop = [
        feature_slices[key]
        for key in ("clean_teacher_delta", "sign_teacher_delta")
        if key in feature_slices
    ]
    source_off_auc = compute_linear_probe_auc(clean, mild, drop_slices=source_drop) if source_drop else full_auc
    teacher_off_auc = compute_linear_probe_auc(clean, mild, drop_slices=teacher_drop) if teacher_drop else full_auc
    return {
        "clean_vs_mild_gate_feature_auc": float(full_auc),
        "clean_vs_mild_gate_feature_delta_norm": delta_norm,
        "source_gate_feature_onoff_auc_delta": float(full_auc - source_off_auc),
        "teacher_delta_feature_onoff_auc_delta": float(full_auc - teacher_off_auc),
    }


def build_gate_feature_context(
    *,
    clean_teacher_outputs: dict[str, torch.Tensor],
    sign_teacher_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "clean_teacher_scores": clean_teacher_outputs["attention_scores"],
        "sign_teacher_scores": sign_teacher_outputs["attention_scores"],
        "source_gate_scores": sign_teacher_outputs["attention_bias_gates"],
        "clean_teacher_adapter_delta": clean_teacher_outputs["attention_score_adapter_delta"],
        "sign_teacher_adapter_delta": sign_teacher_outputs["attention_score_adapter_delta"],
    }


def compute_residual_logit_target_loss(
    *,
    clean_residual_logit: torch.Tensor,
    mild_residual_logit: torch.Tensor,
    source_clean_gate: torch.Tensor,
    source_mild_gate: torch.Tensor,
    residual_alpha: torch.Tensor,
    clean_gate_max: float,
    mild_gate_min: float,
    clip: float | None = None,
    loss_type: str = "huber",
) -> torch.Tensor:
    alpha = torch.clamp(residual_alpha.detach(), min=1.0e-4)
    source_clean_logit = torch.logit(source_clean_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    source_mild_logit = torch.logit(source_mild_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    clean_target = (torch.logit(torch.as_tensor(clean_gate_max, device=clean_residual_logit.device)) - source_clean_logit) / alpha
    mild_target = (torch.logit(torch.as_tensor(mild_gate_min, device=mild_residual_logit.device)) - source_mild_logit) / alpha
    if clip is not None and clip > 0.0:
        clean_target = torch.clamp(clean_target, min=-float(clip), max=float(clip))
        mild_target = torch.clamp(mild_target, min=-float(clip), max=float(clip))
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_residual_logit - clean_target).pow(2))
        mild_loss = torch.mean((mild_residual_logit - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_residual_logit, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_residual_logit, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_final_gate_target_loss(
    *,
    clean_gate: torch.Tensor,
    mild_gate: torch.Tensor,
    clean_gate_max: float,
    mild_gate_min: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    clean_target = torch.as_tensor(clean_gate_max, device=clean_gate.device, dtype=clean_gate.dtype)
    mild_target = torch.as_tensor(mild_gate_min, device=mild_gate.device, dtype=mild_gate.dtype)
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_gate - clean_target).pow(2))
        mild_loss = torch.mean((mild_gate - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_gate, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_gate, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_relative_gap_anchor_loss(
    *,
    current_relative_gap: torch.Tensor,
    teacher_relative_gap: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    if str(loss_type) == "huber":
        return F.huber_loss(current_relative_gap, teacher_relative_gap, reduction="mean")
    return torch.mean((current_relative_gap - teacher_relative_gap).pow(2))


def compute_relative_gap_margin_loss(
    *,
    clean_relative_gap: torch.Tensor,
    mild_relative_gap: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    clean_mean = torch.mean(clean_relative_gap)
    mild_mean = torch.mean(mild_relative_gap)
    return torch.relu(
        torch.as_tensor(margin, device=clean_relative_gap.device, dtype=clean_relative_gap.dtype)
        - (clean_mean - mild_mean)
    )


def compute_score_adapter_anchor_loss(
    *,
    current_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    if str(loss_type) == "huber":
        return F.huber_loss(current_delta, teacher_delta, reduction="mean")
    return torch.mean((current_delta - teacher_delta).pow(2))


def compute_adapter_delta_margin_loss(
    *,
    clean_delta: torch.Tensor,
    mild_delta: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    clean_norm = torch.norm(clean_delta, dim=-1).mean()
    mild_norm = torch.norm(mild_delta, dim=-1).mean()
    return torch.relu(
        torch.as_tensor(margin, device=clean_delta.device, dtype=clean_delta.dtype)
        - (mild_norm - clean_norm)
    )


def compute_adapter_gate_target_loss(
    *,
    clean_adapter_gate_logit: torch.Tensor,
    mild_adapter_gate_logit: torch.Tensor,
    source_clean_gate: torch.Tensor,
    source_mild_gate: torch.Tensor,
    clean_gate_max: float,
    mild_gate_min: float,
    target_fraction: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    target_fraction_tensor = torch.as_tensor(
        target_fraction,
        device=clean_adapter_gate_logit.device,
        dtype=clean_adapter_gate_logit.dtype,
    )
    clean_needed = torch.logit(
        torch.clamp(torch.as_tensor(clean_gate_max, device=clean_adapter_gate_logit.device, dtype=clean_adapter_gate_logit.dtype), 1.0e-4, 1.0 - 1.0e-4)
    ) - torch.logit(source_clean_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    mild_needed = torch.logit(
        torch.clamp(torch.as_tensor(mild_gate_min, device=mild_adapter_gate_logit.device, dtype=mild_adapter_gate_logit.dtype), 1.0e-4, 1.0 - 1.0e-4)
    ) - torch.logit(source_mild_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    clean_target = target_fraction_tensor * clean_needed
    mild_target = target_fraction_tensor * mild_needed
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_adapter_gate_logit - clean_target).pow(2))
        mild_loss = torch.mean((mild_adapter_gate_logit - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_adapter_gate_logit, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_adapter_gate_logit, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_adapter_gate_margin_loss(
    *,
    clean_adapter_gate_logit: torch.Tensor,
    mild_adapter_gate_logit: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    clean_mean = torch.mean(clean_adapter_gate_logit)
    mild_mean = torch.mean(mild_adapter_gate_logit)
    return torch.relu(
        torch.as_tensor(margin, device=clean_adapter_gate_logit.device, dtype=clean_adapter_gate_logit.dtype)
        - (mild_mean - clean_mean)
    )


def compute_projection_norm_floor_loss(
    *,
    projection: torch.Tensor,
    min_norm: float,
) -> torch.Tensor:
    projection_norm = torch.norm(projection, dim=-1).mean()
    return torch.relu(
        torch.as_tensor(min_norm, device=projection.device, dtype=projection.dtype) - projection_norm
    )


def compute_threshold_shift_target_loss(
    *,
    clean_threshold_shift_logit: torch.Tensor,
    mild_threshold_shift_logit: torch.Tensor,
    source_clean_gate: torch.Tensor,
    source_mild_gate: torch.Tensor,
    pre_shift_delta_clean: torch.Tensor,
    pre_shift_delta_mild: torch.Tensor,
    clean_gate_max: float,
    mild_gate_min: float,
    target_fraction: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    target_fraction_tensor = torch.as_tensor(
        target_fraction,
        device=clean_threshold_shift_logit.device,
        dtype=clean_threshold_shift_logit.dtype,
    )
    source_clean_logit = torch.logit(source_clean_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    source_mild_logit = torch.logit(source_mild_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    clean_target = target_fraction_tensor * (
        torch.logit(torch.as_tensor(clean_gate_max, device=clean_threshold_shift_logit.device, dtype=clean_threshold_shift_logit.dtype))
        - (source_clean_logit + pre_shift_delta_clean)
    )
    mild_target = target_fraction_tensor * (
        torch.logit(torch.as_tensor(mild_gate_min, device=mild_threshold_shift_logit.device, dtype=mild_threshold_shift_logit.dtype))
        - (source_mild_logit + pre_shift_delta_mild)
    )
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_threshold_shift_logit - clean_target).pow(2))
        mild_loss = torch.mean((mild_threshold_shift_logit - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_threshold_shift_logit, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_threshold_shift_logit, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_threshold_shift_margin_loss(
    *,
    clean_threshold_shift_logit: torch.Tensor,
    mild_threshold_shift_logit: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    clean_mean = torch.mean(clean_threshold_shift_logit)
    mild_mean = torch.mean(mild_threshold_shift_logit)
    return torch.relu(
        torch.as_tensor(margin, device=clean_threshold_shift_logit.device, dtype=clean_threshold_shift_logit.dtype)
        - (mild_mean - clean_mean)
    )


def compute_threshold_shift_polarity_loss(
    *,
    clean_threshold_shift_logit: torch.Tensor,
    mild_threshold_shift_logit: torch.Tensor,
    clean_shift_slack: float,
    mild_shift_slack: float,
) -> torch.Tensor:
    clean_term = torch.relu(
        torch.mean(clean_threshold_shift_logit)
        + torch.as_tensor(clean_shift_slack, device=clean_threshold_shift_logit.device, dtype=clean_threshold_shift_logit.dtype)
    )
    mild_term = torch.relu(
        torch.as_tensor(mild_shift_slack, device=mild_threshold_shift_logit.device, dtype=mild_threshold_shift_logit.dtype)
        - torch.mean(mild_threshold_shift_logit)
    )
    return clean_term + mild_term


def compute_common_cancel_target_loss(
    *,
    clean_common_cancel_logit: torch.Tensor,
    mild_common_cancel_logit: torch.Tensor,
    source_clean_gate: torch.Tensor,
    source_mild_gate: torch.Tensor,
    pre_shift_delta_clean: torch.Tensor,
    pre_shift_delta_mild: torch.Tensor,
    anchor_gate: float,
    clean_target_fraction: float,
    mild_target_fraction: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    source_clean_logit = torch.logit(source_clean_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    source_mild_logit = torch.logit(source_mild_gate.clamp(1.0e-4, 1.0 - 1.0e-4))
    anchor_logit_clean = torch.logit(
        torch.as_tensor(anchor_gate, device=clean_common_cancel_logit.device, dtype=clean_common_cancel_logit.dtype)
    )
    anchor_logit_mild = torch.logit(
        torch.as_tensor(anchor_gate, device=mild_common_cancel_logit.device, dtype=mild_common_cancel_logit.dtype)
    )
    clean_target = torch.as_tensor(
        clean_target_fraction,
        device=clean_common_cancel_logit.device,
        dtype=clean_common_cancel_logit.dtype,
    ) * (anchor_logit_clean - (source_clean_logit + pre_shift_delta_clean))
    mild_target = torch.as_tensor(
        mild_target_fraction,
        device=mild_common_cancel_logit.device,
        dtype=mild_common_cancel_logit.dtype,
    ) * (anchor_logit_mild - (source_mild_logit + pre_shift_delta_mild))
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_common_cancel_logit - clean_target).pow(2))
        mild_loss = torch.mean((mild_common_cancel_logit - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_common_cancel_logit, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_common_cancel_logit, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_contrast_shift_target_loss(
    *,
    clean_contrast_shift_logit: torch.Tensor,
    mild_contrast_shift_logit: torch.Tensor,
    post_cancel_clean_logit: torch.Tensor,
    post_cancel_mild_logit: torch.Tensor,
    clean_gate_max: float,
    mild_gate_min: float,
    target_fraction: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    target_fraction_tensor = torch.as_tensor(
        target_fraction,
        device=clean_contrast_shift_logit.device,
        dtype=clean_contrast_shift_logit.dtype,
    )
    clean_target = target_fraction_tensor * (
        torch.logit(
            torch.as_tensor(clean_gate_max, device=clean_contrast_shift_logit.device, dtype=clean_contrast_shift_logit.dtype)
        )
        - post_cancel_clean_logit
    )
    mild_target = target_fraction_tensor * (
        torch.logit(
            torch.as_tensor(mild_gate_min, device=mild_contrast_shift_logit.device, dtype=mild_contrast_shift_logit.dtype)
        )
        - post_cancel_mild_logit
    )
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_contrast_shift_logit - clean_target).pow(2))
        mild_loss = torch.mean((mild_contrast_shift_logit - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_contrast_shift_logit, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_contrast_shift_logit, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_common_cancel_overreach_loss(
    *,
    post_cancel_clean_logit: torch.Tensor,
    post_cancel_mild_logit: torch.Tensor,
    floor_gate: float,
) -> torch.Tensor:
    floor_clean = torch.logit(
        torch.as_tensor(floor_gate, device=post_cancel_clean_logit.device, dtype=post_cancel_clean_logit.dtype)
    )
    floor_mild = torch.logit(
        torch.as_tensor(floor_gate, device=post_cancel_mild_logit.device, dtype=post_cancel_mild_logit.dtype)
    )
    clean_loss = torch.relu(floor_clean - post_cancel_clean_logit)
    mild_loss = torch.relu(floor_mild - post_cancel_mild_logit)
    return 0.5 * (torch.mean(clean_loss) + torch.mean(mild_loss))


def compute_common_cancel_excess_loss(
    *,
    clean_common_cancel_logit: torch.Tensor,
    mild_common_cancel_logit: torch.Tensor,
    excess_clean: torch.Tensor,
    excess_mild: torch.Tensor,
    clean_target_fraction: float,
    mild_target_fraction: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    clean_target = torch.as_tensor(
        clean_target_fraction,
        device=clean_common_cancel_logit.device,
        dtype=clean_common_cancel_logit.dtype,
    ) * excess_clean
    mild_target = torch.as_tensor(
        mild_target_fraction,
        device=mild_common_cancel_logit.device,
        dtype=mild_common_cancel_logit.dtype,
    ) * excess_mild
    clean_pred = -clean_common_cancel_logit
    mild_pred = -mild_common_cancel_logit
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_pred - clean_target).pow(2))
        mild_loss = torch.mean((mild_pred - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_pred, clean_target, reduction="mean")
        mild_loss = F.huber_loss(mild_pred, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_mild_rescue_target_loss(
    *,
    clean_mild_rescue_logit: torch.Tensor,
    mild_mild_rescue_logit: torch.Tensor,
    post_cancel_mild_logit: torch.Tensor,
    mild_gate_min: float,
    target_fraction: float,
    loss_type: str = "huber",
) -> torch.Tensor:
    zero_target = torch.zeros_like(clean_mild_rescue_logit)
    target_fraction_tensor = torch.as_tensor(
        target_fraction,
        device=mild_mild_rescue_logit.device,
        dtype=mild_mild_rescue_logit.dtype,
    )
    mild_target = target_fraction_tensor * (
        torch.logit(torch.as_tensor(mild_gate_min, device=mild_mild_rescue_logit.device, dtype=mild_mild_rescue_logit.dtype))
        - post_cancel_mild_logit
    )
    if str(loss_type) == "mse":
        clean_loss = torch.mean((clean_mild_rescue_logit - zero_target).pow(2))
        mild_loss = torch.mean((mild_mild_rescue_logit - mild_target).pow(2))
    else:
        clean_loss = F.huber_loss(clean_mild_rescue_logit, zero_target, reduction="mean")
        mild_loss = F.huber_loss(mild_mild_rescue_logit, mild_target, reduction="mean")
    return 0.5 * (clean_loss + mild_loss)


def compute_clean_rescue_zero_loss(
    *,
    clean_mild_rescue_logit: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(clean_mild_rescue_logit.pow(2))


def compute_mild_rescue_gate_bce_loss(
    *,
    clean_mild_rescue_gate: torch.Tensor,
    mild_mild_rescue_gate: torch.Tensor,
) -> torch.Tensor:
    clean_targets = torch.zeros_like(clean_mild_rescue_gate)
    mild_targets = torch.ones_like(mild_mild_rescue_gate)
    clean_loss = F.binary_cross_entropy(clean_mild_rescue_gate, clean_targets)
    mild_loss = F.binary_cross_entropy(mild_mild_rescue_gate, mild_targets)
    return 0.5 * (clean_loss + mild_loss)


def compute_clean_rescue_gate_ceiling_loss(
    *,
    clean_mild_rescue_gate: torch.Tensor,
    max_gate: float,
) -> torch.Tensor:
    return torch.mean(
        torch.relu(
            clean_mild_rescue_gate
            - torch.as_tensor(max_gate, device=clean_mild_rescue_gate.device, dtype=clean_mild_rescue_gate.dtype)
        )
    )


def compute_cancel_gate_floor_loss(
    *,
    clean_common_cancel_prob: torch.Tensor,
    clean_excess: torch.Tensor,
    trigger: float,
    min_prob: float = 0.80,
) -> torch.Tensor:
    trigger_tensor = torch.as_tensor(
        trigger,
        device=clean_common_cancel_prob.device,
        dtype=clean_common_cancel_prob.dtype,
    )
    min_prob_tensor = torch.as_tensor(
        min_prob,
        device=clean_common_cancel_prob.device,
        dtype=clean_common_cancel_prob.dtype,
    )
    active_mask = (clean_excess > trigger_tensor).to(clean_common_cancel_prob.dtype)
    penalties = torch.relu(min_prob_tensor - clean_common_cancel_prob) * active_mask
    denom = torch.clamp(active_mask.sum(), min=1.0)
    return penalties.sum() / denom


def compute_selector_pair_margin_loss(
    *,
    clean_selector_prob: torch.Tensor,
    mild_selector_prob: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    return torch.mean(
        torch.relu(
            torch.as_tensor(
                margin,
                device=clean_selector_prob.device,
                dtype=clean_selector_prob.dtype,
            )
            - (mild_selector_prob - clean_selector_prob)
        )
    )


def _metric_tolerance_for_key(config: dict[str, Any], key: str) -> float:
    per_key = dict(config.get("teacher_reproduction_tolerances", {}) or {})
    if key in per_key:
        return float(per_key[key])
    return float(config.get("teacher_reproduction_tolerance_default", 1.0e-4))


def _load_reference_metric_row(path: str | Path, candidate_name: str) -> dict[str, Any]:
    ref_df = pd.read_csv(path)
    matched = ref_df[ref_df["candidate_name"] == candidate_name]
    if matched.empty:
        raise RuntimeError(f"Teacher reproduction reference row not found for {candidate_name} in {path}")
    return matched.iloc[0].to_dict()


def validate_teacher_reproduction_metrics(
    observed: dict[str, Any],
    reference: dict[str, Any],
    *,
    keys: list[str],
    config: dict[str, Any],
) -> None:
    mismatches: list[str] = []
    for key in keys:
        expected = float(reference.get(key, 0.0))
        actual = float(observed.get(key, 0.0))
        if abs(actual - expected) > _metric_tolerance_for_key(config, key):
            mismatches.append(f"{key}: expected={expected:.6f}, actual={actual:.6f}")
    if mismatches:
        raise RuntimeError(
            "Teacher reproduction check failed:\n" + "\n".join(mismatches)
        )


def validate_run_completeness(
    df: pd.DataFrame,
    *,
    expected_candidates: list[str],
    expected_seeds: list[int],
    expected_probe_names: list[str],
) -> None:
    actual_candidates = sorted(df["candidate_name"].dropna().unique().tolist())
    if sorted(expected_candidates) != actual_candidates:
        raise RuntimeError(
            f"Incomplete candidate set: expected {sorted(expected_candidates)}, actual {actual_candidates}"
        )
    missing: list[str] = []
    for candidate_name in expected_candidates:
        sub = df[df["candidate_name"] == candidate_name]
        actual_seeds = sorted(int(seed) for seed in sub["seed"].dropna().unique().tolist())
        if actual_seeds != sorted(expected_seeds):
            missing.append(f"{candidate_name}: expected seeds {sorted(expected_seeds)}, actual {actual_seeds}")
            continue
        for seed in expected_seeds:
            seed_sub = sub[sub["seed"] == seed]
            actual_probes = sorted(str(name) for name in seed_sub["corruption_type"].dropna().unique().tolist())
            if actual_probes != sorted(expected_probe_names):
                missing.append(
                    f"{candidate_name}/seed_{seed}: expected probes {sorted(expected_probe_names)}, actual {actual_probes}"
                )
    if missing:
        raise RuntimeError("Incomplete eval rows detected:\n" + "\n".join(missing))


def select_gate_teacher_preserve_indices(disable_vision_final_gate_teacher_preserve: bool) -> list[int]:
    return [1, 2, 3] if disable_vision_final_gate_teacher_preserve else [0, 1, 2, 3]


def _load_reference_eval_rows(
    reference_root: str | Path,
    *,
    seeds: Sequence[int],
    filename: str,
    candidate_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(reference_root)
    for seed in seeds:
        path = root / f"seed_{int(seed)}" / filename
        if not path.exists():
            raise FileNotFoundError(f"Teacher reproduction reference eval rows missing: {path}")
        loaded_rows = json.loads(path.read_text(encoding="utf-8"))
        for row in loaded_rows:
            normalized = dict(row)
            normalized["candidate_name"] = candidate_name
            normalized["seed"] = int(seed)
            rows.append(normalized)
    return rows


def compute_teacher_reproduction_metrics_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise RuntimeError("Teacher reproduction metrics require non-empty rows")
    normal = df[df["corruption_type"] == "normal_eval"].mean(numeric_only=True)
    mild_vision = df[df["corruption_type"] == "mild_vision"].mean(numeric_only=True)
    if normal.empty or mild_vision.empty:
        raise RuntimeError("Teacher reproduction metrics require normal_eval and mild_vision rows")
    vision_attention_delta = float(normal.get("attention_mean_vision", 0.0) - mild_vision.get("attention_mean_vision", 0.0))
    vision_targetlogit_delta = float(
        normal.get("targetlogit_shapley_mean_vision", 0.0) - mild_vision.get("targetlogit_shapley_mean_vision", 0.0)
    )
    normal_vision_score = float(normal.get("attention_score_mean_vision", 0.0))
    mild_vision_score = float(mild_vision.get("attention_score_mean_vision", 0.0))
    normal_route_score = float(normal.get("attention_score_mean_route", 0.0))
    mild_route_score = float(mild_vision.get("attention_score_mean_route", 0.0))
    normal_distance_score = float(normal.get("attention_score_mean_distance", 0.0))
    mild_distance_score = float(mild_vision.get("attention_score_mean_distance", 0.0))
    return {
        "vision_attention_targetlogit_sign_match_flag": _raw_sign_match(vision_attention_delta, vision_targetlogit_delta),
        "vision_attention_relative_score_clean_minus_mild": float(normal_vision_score - mild_vision_score),
        "vision_attention_clean_minus_mild": vision_attention_delta,
        "vision_route_score_gap_clean_minus_mild": float(
            (normal_vision_score - normal_route_score) - (mild_vision_score - mild_route_score)
        ),
        "vision_distance_score_gap_clean_minus_mild": float(
            (normal_vision_score - normal_distance_score) - (mild_vision_score - mild_distance_score)
        ),
        "vision_gate_clean_mean": float(normal.get("attention_bias_gate_mean_vision", 0.0)),
        "vision_gate_mild_mean": float(mild_vision.get("attention_bias_gate_mean_vision", 0.0)),
    }


def _build_v3_7f_teacher_check_frames(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    gate_metric_fn: Callable[[pd.DataFrame, dict[str, dict[str, float]]], pd.DataFrame],
    score_builder_fn: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    degradation_df = compute_degradation_metrics(df)
    consistency_df = compute_consistency_metrics(df)
    tiebreak_df = compute_tiebreak_metrics(df)
    targetlogit_df = build_v3_1b_targetlogit_summary(df)
    gate_df = gate_metric_fn(df, candidate_thresholds=build_candidate_gate_thresholds(config))
    score_df = score_builder_fn(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=str(config["baseline_candidate"]),
        tiebreak_weights=config["tiebreak_weights"],
    )
    return score_df, gate_df


def run_teacher_reproduction_preflight(
    current_rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    score_builder_fn: Callable[..., pd.DataFrame],
    gate_metric_fn: Callable[[pd.DataFrame, dict[str, dict[str, float]]], pd.DataFrame],
) -> None:
    if not bool(config.get("teacher_reproduction_check", False)):
        return
    current_candidate = str(config.get("source_gate_teacher_candidate", "reference_v3_7c_medium_locked"))
    current_df = pd.DataFrame(list(current_rows))
    if current_df.empty:
        raise RuntimeError("Teacher reproduction check requires non-empty rows")
    current_rows_only = current_df[current_df["candidate_name"] == current_candidate].to_dict(orient="records")
    if not current_rows_only:
        raise RuntimeError("Teacher reproduction check missing source candidate rows")
    observed = compute_teacher_reproduction_metrics_from_rows(current_rows_only)

    reference_candidate_name = str(config.get("teacher_reference_candidate_name", "v3_7c_gated_bias_medium"))
    if config.get("teacher_reference_eval_rows_root"):
        source_seeds = sorted(int(seed) for seed in current_df[current_df["candidate_name"] == current_candidate]["seed"].dropna().unique().tolist())
        reference_rows = _load_reference_eval_rows(
            config["teacher_reference_eval_rows_root"],
            seeds=source_seeds,
            filename=str(config.get("teacher_reference_eval_rows_filename", "v3_7c_pilot_eval_rows.json")),
            candidate_name=current_candidate,
        )
        reference = compute_teacher_reproduction_metrics_from_rows(reference_rows)
    else:
        reference = _load_reference_metric_row(Path(config["teacher_reference_summary_path"]), reference_candidate_name)
        gate_summary_path = config.get("teacher_reference_gate_summary_path")
        if gate_summary_path:
            reference = {
                **reference,
                **_load_reference_metric_row(Path(gate_summary_path), reference_candidate_name),
            }
    validate_teacher_reproduction_metrics(
        observed=observed,
        reference=reference,
        keys=[
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_relative_score_clean_minus_mild",
            "vision_attention_clean_minus_mild",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "vision_gate_clean_mean",
            "vision_gate_mild_mean",
        ],
        config=config,
    )


def resolve_shapley_reference_obs(
    reference_mode: str,
    state_obs: dict[str, torch.Tensor],
    clean_reference_obs: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if reference_mode == "clean_full":
        return clean_reference_obs if clean_reference_obs is not None else state_obs
    if reference_mode == "current_state":
        return state_obs
    raise ValueError(f"Unsupported shapley_reference_mode: {reference_mode}")


def normalize_shapley_utility(utility_mode: str) -> str:
    if utility_mode not in _TRAIN_SHAPLEY_UTILITY_ALIASES:
        raise ValueError(f"Unsupported shapley_utility: {utility_mode}")
    return _TRAIN_SHAPLEY_UTILITY_ALIASES[utility_mode]


def build_shapley_compute_kwargs(
    losses_cfg: dict[str, Any],
    *,
    rng_seed: int | None = None,
    utility_mode: str | None = None,
    target_modality_index: int | None = None,
    return_raw: bool = True,
) -> dict[str, Any]:
    return {
        "temperature": float(losses_cfg.get("shapley_temperature", 0.25)),
        "utility_mode": normalize_shapley_utility(str(utility_mode or losses_cfg.get("shapley_utility", "clean_fused_l2"))),
        "return_raw": return_raw,
        "target_modality_index": target_modality_index,
        "reference_mode": str(losses_cfg.get("shapley_reference_mode", "clean_full")),
        "counterfactual_baseline": str(losses_cfg.get("shapley_counterfactual_baseline", "zero")),
        "estimator": str(losses_cfg.get("shapley_estimator", "exact")),
        "num_subsets": losses_cfg.get("shapley_num_subsets"),
        "rng_seed": rng_seed,
    }


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(float(temperature), 1e-6)
    shifted = scaled - float(np.max(scaled))
    exps = np.exp(shifted)
    return exps / np.clip(np.sum(exps), a_min=1e-8, a_max=None)


def _spearman_rank_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_rank = pd.Series(lhs).rank(method="average").to_numpy(dtype=np.float64)
    rhs_rank = pd.Series(rhs).rank(method="average").to_numpy(dtype=np.float64)
    if float(lhs_rank.std()) < 1e-8 or float(rhs_rank.std()) < 1e-8:
        return 0.0
    corr = np.corrcoef(lhs_rank, rhs_rank)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def _clone_state_obs_from_sequence(
    obs_seq: dict[str, torch.Tensor],
    batch_idx: int,
    time_idx: int,
) -> dict[str, torch.Tensor]:
    return {
        name: obs_seq[name][batch_idx, time_idx].unsqueeze(0)
        for name in MODALITY_NAMES
    }


def _clone_clean_reference_obs_from_sequence(
    extras: dict[str, torch.Tensor],
    batch_idx: int,
    time_idx: int,
) -> dict[str, torch.Tensor]:
    return {
        name: extras[f"clean_{name}"][batch_idx, time_idx].unsqueeze(0)
        for name in MODALITY_NAMES
    }


def _build_subset_obs(
    state_obs: dict[str, torch.Tensor],
    subset_mask: int,
) -> dict[str, torch.Tensor]:
    subset_obs: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(MODALITY_NAMES):
        if subset_mask & (1 << idx):
            subset_obs[name] = state_obs[name]
        else:
            subset_obs[name] = torch.zeros_like(state_obs[name])
    return subset_obs


def _compute_exact_counterfactual_shapley_torch(
    model: torch.nn.Module,
    state_obs: dict[str, torch.Tensor],
    clean_reference_obs: dict[str, torch.Tensor] | None = None,
    temperature: float = 0.25,
    utility_mode: str = "clean_fused_l2",
    target_modality_index: int | None = None,
    estimator: str = "exact",
    num_subsets: int | None = None,
    rng_seed: int | None = None,
    reference_mode: str = "clean_full",
    counterfactual_baseline: str = "zero",
    clean_teacher_model: torch.nn.Module | None = None,
    sign_teacher_model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if counterfactual_baseline != "zero":
        raise ValueError(f"Unsupported shapley_counterfactual_baseline: {counterfactual_baseline}")
    utility_mode = normalize_shapley_utility(utility_mode)
    reference_obs = resolve_shapley_reference_obs(reference_mode, state_obs=state_obs, clean_reference_obs=clean_reference_obs)

    def _forward_reliability_with_optional_gate_context(
        obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        gate_mode = str(getattr(getattr(model, "fusion_module", None), "gate_mode", "absolute_v1"))
        if gate_mode not in {"source_residual_v1", "source_residual_logit_v2"}:
            return model.forward_reliability(obs)
        if clean_teacher_model is None or sign_teacher_model is None:
            raise ValueError(f"{gate_mode} Shapley requires clean_teacher_model and sign_teacher_model")
        clean_teacher_outputs = clean_teacher_model.forward_reliability(obs)
        sign_teacher_outputs = sign_teacher_model.forward_reliability(obs)
        return model.forward_reliability(
            obs,
            gate_feature_context=build_gate_feature_context(
                clean_teacher_outputs=clean_teacher_outputs,
                sign_teacher_outputs=sign_teacher_outputs,
            ),
        )

    reference_outputs = _forward_reliability_with_optional_gate_context(reference_obs)
    fused_reference = reference_outputs["fused_feature"]
    reference_logits = reference_outputs["reliability_logits"]
    utilities: dict[int, torch.Tensor] = {}

    for subset_mask in _ALL_SUBSET_MASKS:
        subset_obs = _build_subset_obs(state_obs, subset_mask)
        subset_outputs = _forward_reliability_with_optional_gate_context(subset_obs)
        if utility_mode == "clean_fused_l2":
            diff = fused_reference - subset_outputs["fused_feature"]
            utilities[subset_mask] = -torch.mean(diff.pow(2))
        elif utility_mode == "target_logit_clean_l1":
            if target_modality_index is None:
                raise ValueError("target_modality_index is required for target_logit_clean_l1")
            logit_diff = torch.abs(
                reference_logits[..., target_modality_index] - subset_outputs["reliability_logits"][..., target_modality_index]
            )
            utilities[subset_mask] = -torch.mean(logit_diff)
        else:
            raise ValueError(f"Unsupported Shapley utility mode: {utility_mode}")

    base_dtype = fused_reference.dtype
    base_device = fused_reference.device
    shapley_values = torch.zeros(_NUM_MODALITIES, dtype=base_dtype, device=base_device)
    if estimator == "exact":
        subset_masks = _ALL_SUBSET_MASKS
    elif estimator == "sampled":
        candidate_masks = list(_ALL_SUBSET_MASKS)
        if num_subsets is None or num_subsets <= 0:
            raise ValueError("shapley_num_subsets must be positive when shapley_estimator='sampled'")
        generator = np.random.default_rng(0 if rng_seed is None else int(rng_seed))
        chosen = generator.choice(candidate_masks, size=min(len(candidate_masks), int(num_subsets)), replace=False)
        subset_masks = tuple(sorted(int(mask) for mask in np.atleast_1d(chosen)))
    else:
        raise ValueError(f"Unsupported shapley_estimator: {estimator}")

    for modality_idx in range(_NUM_MODALITIES):
        total = torch.zeros((), dtype=base_dtype, device=base_device)
        bit = 1 << modality_idx
        used = 0
        for subset_mask in subset_masks:
            if subset_mask & bit:
                continue
            subset_size = int(bin(subset_mask).count("1"))
            weight = _SHAPLEY_WEIGHTS[subset_size]
            marginal = utilities[subset_mask | bit] - utilities[subset_mask]
            total = total + float(weight) * marginal
            used += 1
        if estimator == "sampled" and used > 0:
            total = total * (float(1 << (_NUM_MODALITIES - 1)) / float(max(used, 1)))
        shapley_values[modality_idx] = total

    shapley_target = torch.softmax(shapley_values / max(float(temperature), 1e-6), dim=-1)
    return shapley_values, shapley_target


@torch.no_grad()
def compute_exact_counterfactual_shapley(
    model: torch.nn.Module,
    state_obs: dict[str, torch.Tensor],
    clean_reference_obs: dict[str, torch.Tensor] | None = None,
    temperature: float = 0.25,
    utility_mode: str = "clean_fused_l2",
    target_modality_index: int | None = None,
    return_raw: bool = True,
    reference_mode: str = "clean_full",
    counterfactual_baseline: str = "zero",
    estimator: str = "exact",
    num_subsets: int | None = None,
    rng_seed: int | None = None,
    clean_teacher_model: torch.nn.Module | None = None,
    sign_teacher_model: torch.nn.Module | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute counterfactual Shapley values.

    When ``return_raw`` is True, returns ``(raw_values, softmax_target)``.
    When ``return_raw`` is False, returns only the softmax target.
    """
    shapley_values, shapley_target = _compute_exact_counterfactual_shapley_torch(
        model=model,
        state_obs=state_obs,
        clean_reference_obs=clean_reference_obs,
        temperature=temperature,
        utility_mode=utility_mode,
        target_modality_index=target_modality_index,
        reference_mode=reference_mode,
        counterfactual_baseline=counterfactual_baseline,
        estimator=estimator,
        num_subsets=num_subsets,
        rng_seed=rng_seed,
        clean_teacher_model=clean_teacher_model,
        sign_teacher_model=sign_teacher_model,
    )
    shapley_values_np = shapley_values.detach().cpu().numpy().astype(np.float32)
    shapley_target_np = shapley_target.detach().cpu().numpy().astype(np.float32)
    if not return_raw:
        return shapley_target_np
    return shapley_values_np, shapley_target_np


def _softmax_numpy_from_raw(values: np.ndarray, temperature: float) -> np.ndarray:
    return _softmax(values.astype(np.float64), temperature=float(temperature)).astype(np.float32)


def _raw_sign_match(delta_attention: float, delta_raw_shapley: float) -> float:
    return float(delta_attention > 0.0 and delta_raw_shapley > 0.0)


class Stage3ShapleyTrainer(Stage2TransitionTrainer):
    """Stage 2 transition trainer extended with Shapley alignment."""

    def _sample_shapley_positions(
        self,
        valid_mask: torch.Tensor,
        corrupted_flag: torch.Tensor,
        target_mask: torch.Tensor,
        max_states: int,
    ) -> list[tuple[int, int]]:
        eligible = valid_mask > 0.0
        single_target = target_mask.sum(dim=-1) == 1
        clean_mask = corrupted_flag.squeeze(-1) <= 0.0
        mild_single_mask = (corrupted_flag.squeeze(-1) > 0.0) & single_target
        eligible = eligible & (clean_mask | mild_single_mask)
        positions = torch.nonzero(eligible, as_tuple=False)
        if positions.numel() == 0:
            return []
        selected = positions[:max_states]
        return [(int(item[0].item()), int(item[1].item())) for item in selected]

    def _compute_shapley_alignment_loss(
        self,
        batch,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses_cfg = self.raw_config["losses"]
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 4))
        positions = self._sample_shapley_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            max_states=num_states,
        )
        if not positions:
            return torch.zeros((), device=self.device)

        kl_terms = []
        for batch_idx, time_idx in positions:
            state_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
            clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
            shapley_target = compute_exact_counterfactual_shapley(
                self.model,
                state_obs,
                clean_reference_obs=clean_reference_obs,
                **build_shapley_compute_kwargs(losses_cfg, return_raw=False),
            )
            target_tensor = torch.as_tensor(shapley_target, dtype=torch.float32, device=self.device).detach()
            attention = outputs["attention_weights"][batch_idx, time_idx].clamp_min(1e-8)
            kl = torch.sum(attention * (torch.log(attention) - torch.log(target_tensor.clamp_min(1e-8))))
            kl_terms.append(kl)
        return torch.stack(kl_terms).mean()

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/reliability_total": 0.0,
            "loss/consistency_loss": 0.0,
            "loss/consistency_gap_mean": 0.0,
            "loss/vision_logit_retention": 0.0,
            "loss/ego_logit_retention": 0.0,
            "loss/shapley_alignment": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        vision_retention_coef = float(losses_cfg.get("vision_logit_retention_loss_coef", 0.0))
        ego_retention_coef = float(losses_cfg.get("ego_logit_retention_loss_coef", 0.0))
        vision_retention_margin = float(losses_cfg.get("vision_logit_retention_margin", 0.0))
        ego_retention_margin = float(losses_cfg.get("ego_logit_retention_margin", 0.0))
        shapley_loss_coef = float(losses_cfg.get("shapley_loss_coef", 0.0))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                valid_mask = batch.valid_mask
                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                ratio = torch.exp(outputs["log_probs"] - batch.old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.base_cfg.ppo.clip_eps, 1.0 + self.base_cfg.ppo.clip_eps)
                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((outputs["values"] - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (outputs["entropies"] * valid_mask).sum() / valid_count
                expanded_mask = valid_mask.unsqueeze(-1)
                reliability_loss = (
                    ((outputs["reliability_scores"] - batch.extras["reliability_targets"]) ** 2) * expanded_mask
                ).sum() / expanded_mask.sum().clamp(min=1.0)
                corrupted_mask = batch.extras["corrupted_flag"].squeeze(-1) * valid_mask
                action_diff = torch.abs(outputs["action_means"] - clean_outputs["action_means"]).mean(dim=-1)
                consistency_gap = (action_diff * corrupted_mask).sum() / corrupted_mask.sum().clamp(min=1.0)
                consistency_loss = (
                    ((outputs["action_means"] - clean_outputs["action_means"]) ** 2).mean(dim=-1) * corrupted_mask
                ).sum() / corrupted_mask.sum().clamp(min=1.0)
                target_mask = batch.extras["target_mask"]
                single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
                vision_target_mask = single_target_mask * target_mask[..., 0]
                ego_target_mask = single_target_mask * target_mask[..., 2]
                vision_logit_gap = clean_outputs["reliability_logits"][..., 0] - outputs["reliability_logits"][..., 0]
                ego_logit_gap = clean_outputs["reliability_logits"][..., 2] - outputs["reliability_logits"][..., 2]
                vision_retention_loss = (
                    (F.relu(vision_retention_margin - vision_logit_gap) * vision_target_mask).sum()
                    / vision_target_mask.sum().clamp(min=1.0)
                )
                ego_retention_loss = (
                    (F.relu(ego_retention_margin - ego_logit_gap) * ego_target_mask).sum()
                    / ego_target_mask.sum().clamp(min=1.0)
                )
                shapley_alignment_loss = self._compute_shapley_alignment_loss(batch, outputs)

                total_loss = (
                    float(losses_cfg["actor_loss_coef"]) * actor_loss
                    + float(losses_cfg["value_loss_coef"]) * critic_loss
                    - float(losses_cfg["entropy_coef"]) * entropy
                    + float(losses_cfg["reliability_loss_coef"]) * reliability_loss
                    + float(losses_cfg["consistency_loss_coef"]) * consistency_loss
                    + vision_retention_coef * vision_retention_loss
                    + ego_retention_coef * ego_retention_loss
                    + shapley_loss_coef * shapley_alignment_loss
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()
                metrics["loss/actor"] += float(actor_loss.item())
                metrics["loss/critic"] += float(critic_loss.item())
                metrics["loss/entropy"] += float(entropy.item())
                metrics["loss/reliability_total"] += float(reliability_loss.item())
                metrics["loss/consistency_loss"] += float(consistency_loss.item())
                metrics["loss/consistency_gap_mean"] += float(consistency_gap.item())
                metrics["loss/vision_logit_retention"] += float(vision_retention_loss.item())
                metrics["loss/ego_logit_retention"] += float(ego_retention_loss.item())
                metrics["loss/shapley_alignment"] += float(shapley_alignment_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    @torch.no_grad()
    def run_eval_suite(self, num_eval_episodes: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        env = self.eval_env
        eval_probes = self.raw_config["eval_probes"]
        losses_cfg = self.raw_config["losses"]

        for probe in eval_probes:
            probe_name = str(probe["name"])
            returns = []
            successes = []
            collisions = []
            timeouts = []
            progress_values = []
            attention_values = []
            reliability_values = []
            vision_logit_values = []
            ego_logit_values = []
            reference_vision_logit_values = []
            reference_ego_logit_values = []
            action_diffs = []
            reference_action_diffs = []
            shapley_target_values = []
            shapley_kls = []
            shapley_rank_corrs = []
            rng = make_probe_rng(self.seed, 19000, probe_name)

            for episode_idx in range(num_eval_episodes):
                obs, info = env.reset(seed=self.seed * 100 + episode_idx)
                del info
                hidden = self.model.initial_hidden(batch_size=1, device=self.device)
                episode_start = torch.ones(1, device=self.device)
                episode_return = 0.0
                episode_progress = []
                while True:
                    clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
                    if probe_name == "normal_eval":
                        policy_obs = clean_obs
                    elif probe["kind"] == "single":
                        policy_obs, _severity = self._apply_single_probe(obs, probe, rng, probe_name)
                    else:
                        policy_obs, _severity = self._apply_mixed_probe(obs, probe, rng, probe_name)

                    reference_clean_output = self._reference_step(
                        obs_dict={key: torch.as_tensor(clean_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0) for key in MODALITY_NAMES},
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    clean_path_output = self.model.step(
                        obs_dict={key: torch.as_tensor(clean_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0) for key in MODALITY_NAMES},
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    corrupt_output = self.model.step(
                        obs_dict={key: torch.as_tensor(policy_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0) for key in MODALITY_NAMES},
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    state_obs = {
                        key: torch.as_tensor(policy_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }
                    shapley_values, shapley_target = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs,
                        clean_reference_obs={
                            key: torch.as_tensor(clean_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                            for key in MODALITY_NAMES
                        },
                        **build_shapley_compute_kwargs(losses_cfg, return_raw=True, rng_seed=self.seed * 100000 + episode_idx),
                    )
                    attention = corrupt_output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    shapley_target_values.append(shapley_target)
                    shapley_kls.append(
                        float(np.sum(attention * (np.log(np.clip(attention, 1e-8, None)) - np.log(np.clip(shapley_target, 1e-8, None)))))
                    )
                    shapley_rank_corrs.append(_spearman_rank_corr(attention.astype(np.float64), shapley_values.astype(np.float64)))

                    action = corrupt_output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    reliability = corrupt_output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    logits = corrupt_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    current_clean_logits = clean_path_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    reference_clean_logits = reference_clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(current_clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(current_clean_logits[2]), float(logits[2])))
                    reference_vision_logit_values.append(float(reference_clean_logits[0]))
                    reference_ego_logit_values.append(float(reference_clean_logits[2]))
                    action_diffs.append(
                        float(torch.mean(torch.abs(clean_path_output.action - corrupt_output.action)).cpu().item())
                    )
                    reference_action_diffs.append(
                        float(torch.mean(torch.abs(reference_clean_output.action - corrupt_output.action)).cpu().item())
                    )
                    hidden = corrupt_output.hidden_state
                    episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                    hidden = self.model.reset_hidden(hidden, episode_start)
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(float(info["goal_reached"]))
                        collisions.append(float(info["collision"]))
                        timeouts.append(float(info["timeout"]))
                        progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                        break

            att = np.stack(attention_values, axis=0)
            rel = np.stack(reliability_values, axis=0)
            shapley_arr = np.stack(shapley_target_values, axis=0)
            vision_logit_arr = np.asarray(vision_logit_values, dtype=np.float32)
            ego_logit_arr = np.asarray(ego_logit_values, dtype=np.float32)
            reference_vision_logit_arr = np.asarray(reference_vision_logit_values, dtype=np.float32)
            reference_ego_logit_arr = np.asarray(reference_ego_logit_values, dtype=np.float32)
            row = {
                "candidate_name": self.candidate_name,
                "seed": self.seed,
                "env_mode": "normal" if probe_name == "normal_eval" else "mild",
                "corruption_type": probe_name,
                "train_step": int(self.raw_config["train"]["total_steps"]),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "consistency_loss": float(np.mean(np.square(action_diffs))),
                "consistency_gap_mean": float(np.mean(action_diffs)),
                "same_state_action_diff_clean_vs_corrupt": float(np.mean(action_diffs)),
                "reference_same_state_action_diff_clean_vs_corrupt": float(np.mean(reference_action_diffs)),
                "consistency_gain_score": 0.0,
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(
                    att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)
                ),
                "logit_mean_vision_clean": float(vision_logit_arr[:, 0].mean()),
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "reference_logit_mean_vision_clean": float(reference_vision_logit_arr.mean()),
                "reference_logit_mean_ego_clean": float(reference_ego_logit_arr.mean()),
                "current_clean_logit_mean_vision": float(vision_logit_arr[:, 0].mean()),
                "current_clean_logit_mean_ego": float(ego_logit_arr[:, 0].mean()),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
            }
            row.update(_probe_logging_fields(probe_name, probe))
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
                row[f"shapley_mean_{name}"] = float(shapley_arr[:, idx].mean())
            rows.append(row)
        return rows

    def _apply_single_probe(
        self,
        obs: dict[str, np.ndarray],
        probe: dict[str, Any],
        rng: np.random.Generator,
        probe_name: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        from stage2_transition_helpers import _apply_single_corruption

        return _apply_single_corruption(
            obs=obs,
            corruption_type=str(probe["modality"]),
            corruption_strength=float(probe["strength"]),
            rng=rng,
            probe_name=probe_name,
        )

    def _apply_mixed_probe(
        self,
        obs: dict[str, np.ndarray],
        probe: dict[str, Any],
        rng: np.random.Generator,
        probe_name: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        from stage2_transition_helpers import _apply_mixed_corruption

        return _apply_mixed_corruption(
            obs=obs,
            corruption_types=list(probe["modalities"]),
            corruption_strengths=list(probe["strengths"]),
            rng=rng,
            probe_name=probe_name,
        )


class Stage31ShapleyTrainer(Stage3ShapleyTrainer):
    """V3.1 trainer with clean-referenced Shapley and targeted raw drop loss."""

    def _sample_target_positions(
        self,
        valid_mask: torch.Tensor,
        corrupted_flag: torch.Tensor,
        target_mask: torch.Tensor,
        target_index: int,
        max_states: int,
    ) -> list[tuple[int, int]]:
        eligible = (
            (valid_mask > 0.0)
            & (corrupted_flag.squeeze(-1) > 0.0)
            & (target_mask.sum(dim=-1) == 1)
            & (target_mask[..., target_index] > 0.0)
        )
        positions = torch.nonzero(eligible, as_tuple=False)
        if positions.numel() == 0:
            return []
        selected = positions[:max_states]
        return [(int(item[0].item()), int(item[1].item())) for item in selected]

    def _compute_shapley_alignment_loss(
        self,
        batch,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses_cfg = self.raw_config["losses"]
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 4))
        temperature = float(losses_cfg.get("shapley_temperature", 0.25))
        positions = self._sample_shapley_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            max_states=num_states,
        )
        if not positions:
            return torch.zeros((), device=self.device)

        kl_terms = []
        for batch_idx, time_idx in positions:
            state_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
            clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
            _raw_values, shapley_target = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=state_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
            )
            target_tensor = shapley_target.detach().clamp_min(1e-8)
            attention = outputs["attention_weights"][batch_idx, time_idx].clamp_min(1e-8)
            kl_terms.append(torch.sum(attention * (torch.log(attention) - torch.log(target_tensor))))
        return torch.stack(kl_terms).mean()

    def _compute_targeted_shapley_drop_loss(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses_cfg = self.raw_config["losses"]
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 4))
        temperature = float(losses_cfg.get("shapley_temperature", 0.25))
        vision_margin = float(losses_cfg.get("vision_shapley_drop_margin", 0.001))
        ego_margin = float(losses_cfg.get("ego_shapley_drop_margin", 0.001))

        vision_terms: list[torch.Tensor] = []
        ego_terms: list[torch.Tensor] = []

        for batch_idx, time_idx in self._sample_target_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            target_index=0,
            max_states=num_states,
        ):
            clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
            clean_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=clean_reference_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
            )
            corrupt_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
            corrupt_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=corrupt_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
            )
            vision_terms.append(F.relu(torch.as_tensor(vision_margin, device=self.device) - (clean_raw[0] - corrupt_raw[0])))

        for batch_idx, time_idx in self._sample_target_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            target_index=2,
            max_states=num_states,
        ):
            clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
            clean_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=clean_reference_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
            )
            corrupt_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
            corrupt_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=corrupt_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
            )
            ego_terms.append(F.relu(torch.as_tensor(ego_margin, device=self.device) - (clean_raw[2] - corrupt_raw[2])))

        vision_loss = torch.stack(vision_terms).mean() if vision_terms else torch.zeros((), device=self.device)
        ego_loss = torch.stack(ego_terms).mean() if ego_terms else torch.zeros((), device=self.device)
        return vision_loss, ego_loss

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/reliability_total": 0.0,
            "loss/consistency_loss": 0.0,
            "loss/consistency_gap_mean": 0.0,
            "loss/vision_logit_retention": 0.0,
            "loss/ego_logit_retention": 0.0,
            "loss/shapley_alignment_kl": 0.0,
            "loss/vision_shapley_drop": 0.0,
            "loss/ego_shapley_drop": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        vision_retention_coef = float(losses_cfg.get("vision_logit_retention_loss_coef", 0.0))
        ego_retention_coef = float(losses_cfg.get("ego_logit_retention_loss_coef", 0.0))
        vision_retention_margin = float(losses_cfg.get("vision_logit_retention_margin", 0.0))
        ego_retention_margin = float(losses_cfg.get("ego_logit_retention_margin", 0.0))
        shapley_alignment_coef = float(losses_cfg.get("shapley_alignment_loss_coef", losses_cfg.get("shapley_loss_coef", 0.0)))
        vision_shapley_drop_coef = float(losses_cfg.get("vision_shapley_drop_loss_coef", 0.0))
        ego_shapley_drop_coef = float(losses_cfg.get("ego_shapley_drop_loss_coef", 0.0))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                valid_mask = batch.valid_mask
                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                ratio = torch.exp(outputs["log_probs"] - batch.old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.base_cfg.ppo.clip_eps, 1.0 + self.base_cfg.ppo.clip_eps)
                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((outputs["values"] - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (outputs["entropies"] * valid_mask).sum() / valid_count
                expanded_mask = valid_mask.unsqueeze(-1)
                reliability_loss = (
                    ((outputs["reliability_scores"] - batch.extras["reliability_targets"]) ** 2) * expanded_mask
                ).sum() / expanded_mask.sum().clamp(min=1.0)
                corrupted_mask = batch.extras["corrupted_flag"].squeeze(-1) * valid_mask
                action_diff = torch.abs(outputs["action_means"] - clean_outputs["action_means"]).mean(dim=-1)
                consistency_gap = (action_diff * corrupted_mask).sum() / corrupted_mask.sum().clamp(min=1.0)
                consistency_loss = (
                    ((outputs["action_means"] - clean_outputs["action_means"]) ** 2).mean(dim=-1) * corrupted_mask
                ).sum() / corrupted_mask.sum().clamp(min=1.0)
                target_mask = batch.extras["target_mask"]
                single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
                vision_target_mask = single_target_mask * target_mask[..., 0]
                ego_target_mask = single_target_mask * target_mask[..., 2]
                vision_logit_gap = clean_outputs["reliability_logits"][..., 0] - outputs["reliability_logits"][..., 0]
                ego_logit_gap = clean_outputs["reliability_logits"][..., 2] - outputs["reliability_logits"][..., 2]
                vision_retention_loss = (
                    (F.relu(vision_retention_margin - vision_logit_gap) * vision_target_mask).sum()
                    / vision_target_mask.sum().clamp(min=1.0)
                )
                ego_retention_loss = (
                    (F.relu(ego_retention_margin - ego_logit_gap) * ego_target_mask).sum()
                    / ego_target_mask.sum().clamp(min=1.0)
                )
                shapley_alignment_loss = self._compute_shapley_alignment_loss(batch, outputs)
                vision_shapley_drop_loss, ego_shapley_drop_loss = self._compute_targeted_shapley_drop_loss(batch)

                total_loss = (
                    float(losses_cfg["actor_loss_coef"]) * actor_loss
                    + float(losses_cfg["value_loss_coef"]) * critic_loss
                    - float(losses_cfg["entropy_coef"]) * entropy
                    + float(losses_cfg["reliability_loss_coef"]) * reliability_loss
                    + float(losses_cfg["consistency_loss_coef"]) * consistency_loss
                    + vision_retention_coef * vision_retention_loss
                    + ego_retention_coef * ego_retention_loss
                    + shapley_alignment_coef * shapley_alignment_loss
                    + vision_shapley_drop_coef * vision_shapley_drop_loss
                    + ego_shapley_drop_coef * ego_shapley_drop_loss
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()
                metrics["loss/actor"] += float(actor_loss.item())
                metrics["loss/critic"] += float(critic_loss.item())
                metrics["loss/entropy"] += float(entropy.item())
                metrics["loss/reliability_total"] += float(reliability_loss.item())
                metrics["loss/consistency_loss"] += float(consistency_loss.item())
                metrics["loss/consistency_gap_mean"] += float(consistency_gap.item())
                metrics["loss/vision_logit_retention"] += float(vision_retention_loss.item())
                metrics["loss/ego_logit_retention"] += float(ego_retention_loss.item())
                metrics["loss/shapley_alignment_kl"] += float(shapley_alignment_loss.item())
                metrics["loss/vision_shapley_drop"] += float(vision_shapley_drop_loss.item())
                metrics["loss/ego_shapley_drop"] += float(ego_shapley_drop_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    @torch.no_grad()
    def run_eval_suite(self, num_eval_episodes: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        env = self.eval_env
        eval_probes = self.raw_config["eval_probes"]
        losses_cfg = self.raw_config["losses"]

        for probe in eval_probes:
            probe_name = str(probe["name"])
            returns = []
            successes = []
            collisions = []
            timeouts = []
            progress_values = []
            attention_values = []
            reliability_values = []
            vision_logit_values = []
            ego_logit_values = []
            action_diffs = []
            raw_shapley_values = []
            shapley_target_values = []
            shapley_kls = []
            shapley_rank_corrs = []
            rng = make_probe_rng(self.seed, 29000, probe_name)

            for episode_idx in range(num_eval_episodes):
                obs, info = env.reset(seed=self.seed * 100 + episode_idx)
                del info
                hidden = self.model.initial_hidden(batch_size=1, device=self.device)
                episode_start = torch.ones(1, device=self.device)
                episode_return = 0.0
                episode_progress = []
                while True:
                    clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
                    if probe_name == "normal_eval":
                        policy_obs = clean_obs
                    elif probe["kind"] == "single":
                        policy_obs, _severity = self._apply_single_probe(obs, probe, rng, probe_name)
                    else:
                        policy_obs, _severity = self._apply_mixed_probe(obs, probe, rng, probe_name)

                    clean_obs_t = {
                        key: torch.as_tensor(clean_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }
                    corrupt_obs_t = {
                        key: torch.as_tensor(policy_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }

                    reference_clean_output = self._reference_step(
                        obs_dict=clean_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    clean_path_output = self.model.step(
                        obs_dict=clean_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    corrupt_output = self.model.step(
                        obs_dict=corrupt_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    raw_shapley, shapley_target = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        **build_shapley_compute_kwargs(losses_cfg, return_raw=True, rng_seed=self.seed * 100000 + episode_idx),
                    )
                    attention = corrupt_output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    raw_shapley_values.append(raw_shapley)
                    shapley_target_values.append(shapley_target)
                    shapley_kls.append(
                        float(np.sum(attention * (np.log(np.clip(attention, 1e-8, None)) - np.log(np.clip(shapley_target, 1e-8, None)))))
                    )
                    shapley_rank_corrs.append(_spearman_rank_corr(attention.astype(np.float64), raw_shapley.astype(np.float64)))

                    action = corrupt_output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    reliability = corrupt_output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    logits = corrupt_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    clean_logits = clean_path_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    reference_clean_logits = reference_clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(clean_logits[2]), float(logits[2])))
                    reference_vision_logit_values.append(float(reference_clean_logits[0]))
                    reference_ego_logit_values.append(float(reference_clean_logits[2]))
                    action_diffs.append(float(torch.mean(torch.abs(clean_path_output.action - corrupt_output.action)).cpu().item()))
                    reference_action_diffs.append(
                        float(torch.mean(torch.abs(reference_clean_output.action - corrupt_output.action)).cpu().item())
                    )
                    hidden = corrupt_output.hidden_state
                    episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                    hidden = self.model.reset_hidden(hidden, episode_start)
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(float(info["goal_reached"]))
                        collisions.append(float(info["collision"]))
                        timeouts.append(float(info["timeout"]))
                        progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                        break

            att = np.stack(attention_values, axis=0)
            rel = np.stack(reliability_values, axis=0)
            raw_shapley_arr = np.stack(raw_shapley_values, axis=0)
            shapley_target_arr = np.stack(shapley_target_values, axis=0)
            vision_logit_arr = np.asarray(vision_logit_values, dtype=np.float32)
            ego_logit_arr = np.asarray(ego_logit_values, dtype=np.float32)
            row = {
                "candidate_name": self.candidate_name,
                "seed": self.seed,
                "env_mode": "normal" if probe_name == "normal_eval" else "mild",
                "corruption_type": probe_name,
                "train_step": int(self.raw_config["train"]["total_steps"]),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "consistency_loss": float(np.mean(np.square(action_diffs))),
                "consistency_gap_mean": float(np.mean(action_diffs)),
                "same_state_action_diff_clean_vs_corrupt": float(np.mean(action_diffs)),
                "reference_same_state_action_diff_clean_vs_corrupt": float(np.mean(reference_action_diffs)),
                "consistency_gain_score": 0.0,
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(
                    att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)
                ),
                "logit_mean_vision_clean": float(vision_logit_arr[:, 0].mean()),
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "reference_logit_mean_vision_clean": float(reference_vision_logit_arr.mean()),
                "reference_logit_mean_ego_clean": float(reference_ego_logit_arr.mean()),
                "current_clean_logit_mean_vision": float(vision_logit_arr[:, 0].mean()),
                "current_clean_logit_mean_ego": float(ego_logit_arr[:, 0].mean()),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
            }
            row.update(_probe_logging_fields(probe_name, probe))
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
                row[f"raw_shapley_mean_{name}"] = float(raw_shapley_arr[:, idx].mean())
                row[f"shapley_target_mean_{name}"] = float(shapley_target_arr[:, idx].mean())
            rows.append(row)
        return rows


class Stage31bShapleyTrainer(Stage31ShapleyTrainer):
    """V3.1b trainer with target-logit Shapley drop for vision/ego directionality."""

    def _sample_shapley_positions(
        self,
        valid_mask: torch.Tensor,
        corrupted_flag: torch.Tensor,
        target_mask: torch.Tensor,
        max_states: int,
    ) -> list[tuple[int, int]]:
        eligible = valid_mask > 0.0
        single_target = target_mask.sum(dim=-1) == 1
        target_focus = (target_mask[..., 0] > 0.0) | (target_mask[..., 2] > 0.0)
        clean_mask = corrupted_flag.squeeze(-1) <= 0.0
        mild_target_mask = (corrupted_flag.squeeze(-1) > 0.0) & single_target & target_focus
        eligible = eligible & (clean_mask | mild_target_mask)
        positions = torch.nonzero(eligible, as_tuple=False)
        if positions.numel() == 0:
            return []
        selected = positions[:max_states]
        return [(int(item[0].item()), int(item[1].item())) for item in selected]

    def _compute_targetlogit_shapley_drop_loss(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses_cfg = self.raw_config["losses"]
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 4))
        temperature = float(losses_cfg.get("shapley_temperature", 0.25))
        vision_margin = float(losses_cfg.get("vision_targetlogit_drop_margin", 0.002))
        ego_margin = float(losses_cfg.get("ego_targetlogit_drop_margin", 0.001))

        vision_terms: list[torch.Tensor] = []
        ego_terms: list[torch.Tensor] = []

        for batch_idx, time_idx in self._sample_target_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            target_index=0,
            max_states=num_states,
        ):
            clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
            clean_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=clean_reference_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
                utility_mode="target_logit_clean_l1",
                target_modality_index=0,
            )
            corrupt_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
            corrupt_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=corrupt_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
                utility_mode="target_logit_clean_l1",
                target_modality_index=0,
            )
            vision_terms.append(F.relu(torch.as_tensor(vision_margin, device=self.device) - (clean_raw[0] - corrupt_raw[0])))

        for batch_idx, time_idx in self._sample_target_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            target_index=2,
            max_states=num_states,
        ):
            clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
            clean_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=clean_reference_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
                utility_mode="target_logit_clean_l1",
                target_modality_index=2,
            )
            corrupt_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
            corrupt_raw, _ = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=corrupt_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
                utility_mode="target_logit_clean_l1",
                target_modality_index=2,
            )
            ego_terms.append(F.relu(torch.as_tensor(ego_margin, device=self.device) - (clean_raw[2] - corrupt_raw[2])))

        vision_loss = torch.stack(vision_terms).mean() if vision_terms else torch.zeros((), device=self.device)
        ego_loss = torch.stack(ego_terms).mean() if ego_terms else torch.zeros((), device=self.device)
        return vision_loss, ego_loss

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/reliability_total": 0.0,
            "loss/consistency_loss": 0.0,
            "loss/consistency_gap_mean": 0.0,
            "loss/vision_logit_retention": 0.0,
            "loss/ego_logit_retention": 0.0,
            "loss/shapley_alignment_kl": 0.0,
            "loss/vision_targetlogit_shapley_drop": 0.0,
            "loss/ego_targetlogit_shapley_drop": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        vision_retention_coef = float(losses_cfg.get("vision_logit_retention_loss_coef", 0.0))
        ego_retention_coef = float(losses_cfg.get("ego_logit_retention_loss_coef", 0.0))
        vision_retention_margin = float(losses_cfg.get("vision_logit_retention_margin", 0.0))
        ego_retention_margin = float(losses_cfg.get("ego_logit_retention_margin", 0.0))
        shapley_alignment_coef = float(losses_cfg.get("shapley_alignment_loss_coef", losses_cfg.get("shapley_loss_coef", 0.0)))
        vision_targetlogit_coef = float(losses_cfg.get("vision_targetlogit_shapley_drop_loss_coef", 0.0))
        ego_targetlogit_coef = float(losses_cfg.get("ego_targetlogit_shapley_drop_loss_coef", 0.0))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                valid_mask = batch.valid_mask
                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                ratio = torch.exp(outputs["log_probs"] - batch.old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.base_cfg.ppo.clip_eps, 1.0 + self.base_cfg.ppo.clip_eps)
                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((outputs["values"] - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (outputs["entropies"] * valid_mask).sum() / valid_count
                expanded_mask = valid_mask.unsqueeze(-1)
                reliability_loss = (
                    ((outputs["reliability_scores"] - batch.extras["reliability_targets"]) ** 2) * expanded_mask
                ).sum() / expanded_mask.sum().clamp(min=1.0)
                corrupted_mask = batch.extras["corrupted_flag"].squeeze(-1) * valid_mask
                action_diff = torch.abs(outputs["action_means"] - clean_outputs["action_means"]).mean(dim=-1)
                consistency_gap = (action_diff * corrupted_mask).sum() / corrupted_mask.sum().clamp(min=1.0)
                consistency_loss = (
                    ((outputs["action_means"] - clean_outputs["action_means"]) ** 2).mean(dim=-1) * corrupted_mask
                ).sum() / corrupted_mask.sum().clamp(min=1.0)
                target_mask = batch.extras["target_mask"]
                single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
                vision_target_mask = single_target_mask * target_mask[..., 0]
                ego_target_mask = single_target_mask * target_mask[..., 2]
                vision_logit_gap = clean_outputs["reliability_logits"][..., 0] - outputs["reliability_logits"][..., 0]
                ego_logit_gap = clean_outputs["reliability_logits"][..., 2] - outputs["reliability_logits"][..., 2]
                vision_retention_loss = (
                    (F.relu(vision_retention_margin - vision_logit_gap) * vision_target_mask).sum()
                    / vision_target_mask.sum().clamp(min=1.0)
                )
                ego_retention_loss = (
                    (F.relu(ego_retention_margin - ego_logit_gap) * ego_target_mask).sum()
                    / ego_target_mask.sum().clamp(min=1.0)
                )
                shapley_alignment_loss = self._compute_shapley_alignment_loss(batch, outputs)
                vision_targetlogit_loss, ego_targetlogit_loss = self._compute_targetlogit_shapley_drop_loss(batch)

                total_loss = (
                    float(losses_cfg["actor_loss_coef"]) * actor_loss
                    + float(losses_cfg["value_loss_coef"]) * critic_loss
                    - float(losses_cfg["entropy_coef"]) * entropy
                    + float(losses_cfg["reliability_loss_coef"]) * reliability_loss
                    + float(losses_cfg["consistency_loss_coef"]) * consistency_loss
                    + vision_retention_coef * vision_retention_loss
                    + ego_retention_coef * ego_retention_loss
                    + shapley_alignment_coef * shapley_alignment_loss
                    + vision_targetlogit_coef * vision_targetlogit_loss
                    + ego_targetlogit_coef * ego_targetlogit_loss
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()
                metrics["loss/actor"] += float(actor_loss.item())
                metrics["loss/critic"] += float(critic_loss.item())
                metrics["loss/entropy"] += float(entropy.item())
                metrics["loss/reliability_total"] += float(reliability_loss.item())
                metrics["loss/consistency_loss"] += float(consistency_loss.item())
                metrics["loss/consistency_gap_mean"] += float(consistency_gap.item())
                metrics["loss/vision_logit_retention"] += float(vision_retention_loss.item())
                metrics["loss/ego_logit_retention"] += float(ego_retention_loss.item())
                metrics["loss/shapley_alignment_kl"] += float(shapley_alignment_loss.item())
                metrics["loss/vision_targetlogit_shapley_drop"] += float(vision_targetlogit_loss.item())
                metrics["loss/ego_targetlogit_shapley_drop"] += float(ego_targetlogit_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    @torch.no_grad()
    def run_eval_suite(self, num_eval_episodes: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        env = self.eval_env
        eval_probes = self.raw_config["eval_probes"]
        temperature = float(self.raw_config["losses"].get("shapley_temperature", 0.25))

        for probe in eval_probes:
            probe_name = str(probe["name"])
            returns = []
            successes = []
            collisions = []
            timeouts = []
            progress_values = []
            attention_values = []
            reliability_values = []
            vision_logit_values = []
            ego_logit_values = []
            action_diffs = []
            raw_shapley_values = []
            shapley_target_values = []
            shapley_kls = []
            shapley_rank_corrs = []
            vision_targetlogit_values = []
            ego_targetlogit_values = []
            losses_cfg = self.raw_config["losses"]
            rng = make_probe_rng(self.seed, 39000, probe_name)

            for episode_idx in range(num_eval_episodes):
                obs, info = env.reset(seed=self.seed * 100 + episode_idx)
                del info
                hidden = self.model.initial_hidden(batch_size=1, device=self.device)
                episode_start = torch.ones(1, device=self.device)
                episode_return = 0.0
                episode_progress = []
                while True:
                    clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
                    if probe_name == "normal_eval":
                        policy_obs = clean_obs
                    elif probe["kind"] == "single":
                        policy_obs, _severity = self._apply_single_probe(obs, probe, rng, probe_name)
                    else:
                        policy_obs, _severity = self._apply_mixed_probe(obs, probe, rng, probe_name)

                    clean_obs_t = {
                        key: torch.as_tensor(clean_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }
                    corrupt_obs_t = {
                        key: torch.as_tensor(policy_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }

                    reference_clean_output = self._reference_step(
                        obs_dict=clean_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    clean_path_output = self.model.step(
                        obs_dict=clean_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    corrupt_output = self.model.step(
                        obs_dict=corrupt_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    raw_shapley, shapley_target = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        utility_mode="clean_fused_l2",
                        **build_shapley_compute_kwargs(losses_cfg, return_raw=True, rng_seed=self.seed * 100000 + episode_idx, utility_mode="clean_fused_l2"),
                    )
                    vision_targetlogit_shapley, _ = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        **build_shapley_compute_kwargs(
                            losses_cfg,
                            return_raw=True,
                            rng_seed=self.seed * 100000 + episode_idx + 1,
                            utility_mode="target_logit_clean_l1",
                            target_modality_index=0,
                        ),
                    )
                    ego_targetlogit_shapley, _ = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        **build_shapley_compute_kwargs(
                            losses_cfg,
                            return_raw=True,
                            rng_seed=self.seed * 100000 + episode_idx + 2,
                            utility_mode="target_logit_clean_l1",
                            target_modality_index=2,
                        ),
                    )
                    attention = corrupt_output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    raw_shapley_values.append(raw_shapley)
                    shapley_target_values.append(shapley_target)
                    vision_targetlogit_values.append(float(vision_targetlogit_shapley[0]))
                    ego_targetlogit_values.append(float(ego_targetlogit_shapley[2]))
                    shapley_kls.append(
                        float(np.sum(attention * (np.log(np.clip(attention, 1e-8, None)) - np.log(np.clip(shapley_target, 1e-8, None)))))
                    )
                    shapley_rank_corrs.append(_spearman_rank_corr(attention.astype(np.float64), raw_shapley.astype(np.float64)))

                    action = corrupt_output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    reliability = corrupt_output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    logits = corrupt_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    clean_logits = clean_path_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    reference_clean_logits = reference_clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(clean_logits[2]), float(logits[2])))
                    reference_vision_logit_values.append(float(reference_clean_logits[0]))
                    reference_ego_logit_values.append(float(reference_clean_logits[2]))
                    action_diffs.append(float(torch.mean(torch.abs(clean_path_output.action - corrupt_output.action)).cpu().item()))
                    reference_action_diffs.append(
                        float(torch.mean(torch.abs(reference_clean_output.action - corrupt_output.action)).cpu().item())
                    )
                    hidden = corrupt_output.hidden_state
                    episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                    hidden = self.model.reset_hidden(hidden, episode_start)
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(float(info["goal_reached"]))
                        collisions.append(float(info["collision"]))
                        timeouts.append(float(info["timeout"]))
                        progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                        break

            att = np.stack(attention_values, axis=0)
            rel = np.stack(reliability_values, axis=0)
            raw_shapley_arr = np.stack(raw_shapley_values, axis=0)
            shapley_target_arr = np.stack(shapley_target_values, axis=0)
            vision_targetlogit_arr = np.asarray(vision_targetlogit_values, dtype=np.float32)
            ego_targetlogit_arr = np.asarray(ego_targetlogit_values, dtype=np.float32)
            vision_logit_arr = np.asarray(vision_logit_values, dtype=np.float32)
            ego_logit_arr = np.asarray(ego_logit_values, dtype=np.float32)
            reference_vision_logit_arr = np.asarray(reference_vision_logit_values, dtype=np.float32)
            reference_ego_logit_arr = np.asarray(reference_ego_logit_values, dtype=np.float32)
            row = {
                "candidate_name": self.candidate_name,
                "seed": self.seed,
                "env_mode": "normal" if probe_name == "normal_eval" else "mild",
                "corruption_type": probe_name,
                "train_step": int(self.raw_config["train"]["total_steps"]),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "consistency_loss": float(np.mean(np.square(action_diffs))),
                "consistency_gap_mean": float(np.mean(action_diffs)),
                "same_state_action_diff_clean_vs_corrupt": float(np.mean(action_diffs)),
                "reference_same_state_action_diff_clean_vs_corrupt": float(np.mean(reference_action_diffs)),
                "consistency_gain_score": 0.0,
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(
                    att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)
                ),
                "logit_mean_vision_clean": float(vision_logit_arr[:, 0].mean()),
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "reference_logit_mean_vision_clean": float(reference_vision_logit_arr.mean()),
                "reference_logit_mean_ego_clean": float(reference_ego_logit_arr.mean()),
                "current_clean_logit_mean_vision": float(vision_logit_arr[:, 0].mean()),
                "current_clean_logit_mean_ego": float(ego_logit_arr[:, 0].mean()),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
                "targetlogit_shapley_mean_vision": float(vision_targetlogit_arr.mean()),
                "targetlogit_shapley_mean_ego": float(ego_targetlogit_arr.mean()),
            }
            row.update(_probe_logging_fields(probe_name, probe))
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
                row[f"raw_shapley_mean_{name}"] = float(raw_shapley_arr[:, idx].mean())
                row[f"shapley_target_mean_{name}"] = float(shapley_target_arr[:, idx].mean())
            rows.append(row)
        return rows


class Stage31cShapleyTrainer(Stage31bShapleyTrainer):
    """V3.1c trainer with explicit vision attention sign-match recovery."""

    def _compute_vision_attention_drop_loss(
        self,
        batch,
        outputs: dict[str, torch.Tensor],
        clean_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses_cfg = self.raw_config["losses"]
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 4))
        margin = float(losses_cfg.get("vision_attention_drop_margin", 0.002))
        terms: list[torch.Tensor] = []

        for batch_idx, time_idx in self._sample_target_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            target_index=0,
            max_states=num_states,
        ):
            clean_attention = clean_outputs["attention_weights"][batch_idx, time_idx, 0]
            corrupt_attention = outputs["attention_weights"][batch_idx, time_idx, 0]
            terms.append(
                F.relu(torch.as_tensor(margin, device=self.device) - (clean_attention - corrupt_attention))
            )

        if not terms:
            return torch.zeros((), device=self.device)
        return torch.stack(terms).mean()

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/reliability_total": 0.0,
            "loss/consistency_loss": 0.0,
            "loss/consistency_gap_mean": 0.0,
            "loss/vision_logit_retention": 0.0,
            "loss/ego_logit_retention": 0.0,
            "loss/shapley_alignment_kl": 0.0,
            "loss/vision_targetlogit_shapley_drop": 0.0,
            "loss/ego_targetlogit_shapley_drop": 0.0,
            "loss/vision_attention_drop": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        vision_retention_coef = float(losses_cfg.get("vision_logit_retention_loss_coef", 0.0))
        ego_retention_coef = float(losses_cfg.get("ego_logit_retention_loss_coef", 0.0))
        vision_retention_margin = float(losses_cfg.get("vision_logit_retention_margin", 0.0))
        ego_retention_margin = float(losses_cfg.get("ego_logit_retention_margin", 0.0))
        shapley_alignment_coef = float(losses_cfg.get("shapley_alignment_loss_coef", losses_cfg.get("shapley_loss_coef", 0.0)))
        vision_targetlogit_coef = float(losses_cfg.get("vision_targetlogit_shapley_drop_loss_coef", 0.0))
        ego_targetlogit_coef = float(losses_cfg.get("ego_targetlogit_shapley_drop_loss_coef", 0.0))
        vision_attention_coef = float(losses_cfg.get("vision_attention_drop_loss_coef", 0.0))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                valid_mask = batch.valid_mask
                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                ratio = torch.exp(outputs["log_probs"] - batch.old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.base_cfg.ppo.clip_eps, 1.0 + self.base_cfg.ppo.clip_eps)
                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((outputs["values"] - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (outputs["entropies"] * valid_mask).sum() / valid_count
                expanded_mask = valid_mask.unsqueeze(-1)
                reliability_loss = (
                    ((outputs["reliability_scores"] - batch.extras["reliability_targets"]) ** 2) * expanded_mask
                ).sum() / expanded_mask.sum().clamp(min=1.0)
                corrupted_mask = batch.extras["corrupted_flag"].squeeze(-1) * valid_mask
                action_diff = torch.abs(outputs["action_means"] - clean_outputs["action_means"]).mean(dim=-1)
                consistency_gap = (action_diff * corrupted_mask).sum() / corrupted_mask.sum().clamp(min=1.0)
                consistency_loss = (
                    ((outputs["action_means"] - clean_outputs["action_means"]) ** 2).mean(dim=-1) * corrupted_mask
                ).sum() / corrupted_mask.sum().clamp(min=1.0)
                target_mask = batch.extras["target_mask"]
                single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
                vision_target_mask = single_target_mask * target_mask[..., 0]
                ego_target_mask = single_target_mask * target_mask[..., 2]
                vision_logit_gap = clean_outputs["reliability_logits"][..., 0] - outputs["reliability_logits"][..., 0]
                ego_logit_gap = clean_outputs["reliability_logits"][..., 2] - outputs["reliability_logits"][..., 2]
                vision_retention_loss = (
                    (F.relu(vision_retention_margin - vision_logit_gap) * vision_target_mask).sum()
                    / vision_target_mask.sum().clamp(min=1.0)
                )
                ego_retention_loss = (
                    (F.relu(ego_retention_margin - ego_logit_gap) * ego_target_mask).sum()
                    / ego_target_mask.sum().clamp(min=1.0)
                )
                shapley_alignment_loss = self._compute_shapley_alignment_loss(batch, outputs)
                vision_targetlogit_loss, ego_targetlogit_loss = self._compute_targetlogit_shapley_drop_loss(batch)
                vision_attention_loss = self._compute_vision_attention_drop_loss(
                    batch=batch,
                    outputs=outputs,
                    clean_outputs=clean_outputs,
                )

                total_loss = (
                    float(losses_cfg["actor_loss_coef"]) * actor_loss
                    + float(losses_cfg["value_loss_coef"]) * critic_loss
                    - float(losses_cfg["entropy_coef"]) * entropy
                    + float(losses_cfg["reliability_loss_coef"]) * reliability_loss
                    + float(losses_cfg["consistency_loss_coef"]) * consistency_loss
                    + vision_retention_coef * vision_retention_loss
                    + ego_retention_coef * ego_retention_loss
                    + shapley_alignment_coef * shapley_alignment_loss
                    + vision_targetlogit_coef * vision_targetlogit_loss
                    + ego_targetlogit_coef * ego_targetlogit_loss
                    + vision_attention_coef * vision_attention_loss
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()
                metrics["loss/actor"] += float(actor_loss.item())
                metrics["loss/critic"] += float(critic_loss.item())
                metrics["loss/entropy"] += float(entropy.item())
                metrics["loss/reliability_total"] += float(reliability_loss.item())
                metrics["loss/consistency_loss"] += float(consistency_loss.item())
                metrics["loss/consistency_gap_mean"] += float(consistency_gap.item())
                metrics["loss/vision_logit_retention"] += float(vision_retention_loss.item())
                metrics["loss/ego_logit_retention"] += float(ego_retention_loss.item())
                metrics["loss/shapley_alignment_kl"] += float(shapley_alignment_loss.item())
                metrics["loss/vision_targetlogit_shapley_drop"] += float(vision_targetlogit_loss.item())
                metrics["loss/ego_targetlogit_shapley_drop"] += float(ego_targetlogit_loss.item())
                metrics["loss/vision_attention_drop"] += float(vision_attention_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage32ShapleyTrainer(Stage31cShapleyTrainer):
    """V3.2 trainer with corruption-aware hybrid attention teachers."""

    def _sample_shapley_positions(
        self,
        valid_mask: torch.Tensor,
        corrupted_flag: torch.Tensor,
        target_mask: torch.Tensor,
        max_states: int,
    ) -> list[tuple[int, int]]:
        eligible = valid_mask > 0.0
        single_target = target_mask.sum(dim=-1) == 1
        clean_mask = eligible & (corrupted_flag.squeeze(-1) <= 0.0)
        mild_vision_mask = eligible & (corrupted_flag.squeeze(-1) > 0.0) & single_target & (target_mask[..., 0] > 0.0)
        mild_ego_mask = eligible & (corrupted_flag.squeeze(-1) > 0.0) & single_target & (target_mask[..., 2] > 0.0)

        def _positions(mask: torch.Tensor) -> list[tuple[int, int]]:
            pos = torch.nonzero(mask, as_tuple=False)
            return [(int(item[0].item()), int(item[1].item())) for item in pos]

        vision_positions = _positions(mild_vision_mask)
        ego_positions = _positions(mild_ego_mask)
        clean_positions = _positions(clean_mask)
        selected: list[tuple[int, int]] = []

        buckets = [
            vision_positions,
            vision_positions,
            vision_positions,
            vision_positions,
            ego_positions,
            ego_positions,
            clean_positions,
            clean_positions,
        ]
        for bucket in buckets:
            if len(selected) >= max_states:
                break
            if not bucket:
                continue
            candidate = bucket.pop(0)
            if candidate not in selected:
                selected.append(candidate)
        for bucket in (vision_positions, ego_positions, clean_positions):
            for candidate in bucket:
                if len(selected) >= max_states:
                    break
                if candidate not in selected:
                    selected.append(candidate)
        return selected[:max_states]

    def _build_attention_teacher(
        self,
        batch,
        batch_idx: int,
        time_idx: int,
        temperature: float,
    ) -> tuple[torch.Tensor, str]:
        losses_cfg = self.raw_config["losses"]
        alpha = float(losses_cfg.get("mild_target_teacher_mix_alpha", 0.0))
        clean_reference_obs = _clone_clean_reference_obs_from_sequence(batch.extras, batch_idx, time_idx)
        state_obs = _clone_state_obs_from_sequence(batch.observations, batch_idx, time_idx)
        fused_raw, fused_target = _compute_exact_counterfactual_shapley_torch(
            model=self.model,
            state_obs=state_obs,
            clean_reference_obs=clean_reference_obs,
            temperature=temperature,
            utility_mode="clean_fused_l2",
        )
        del fused_raw
        corrupted = float(batch.extras["corrupted_flag"][batch_idx, time_idx, 0].item()) > 0.0
        target_vec = batch.extras["target_mask"][batch_idx, time_idx]
        single_target = bool(target_vec.sum().item() == 1.0)
        if not corrupted or not single_target:
            return fused_target, "clean_fused_l2"
        if float(target_vec[0].item()) > 0.0:
            vision_raw, vision_target = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=state_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
                utility_mode="target_logit_clean_l1",
                target_modality_index=0,
            )
            del vision_raw
            teacher = (1.0 - alpha) * fused_target + alpha * vision_target
            teacher = teacher / teacher.sum().clamp(min=1e-8)
            source = "hybrid_vision" if alpha < 0.999 else "targetlogit_vision"
            return teacher, source
        if float(target_vec[2].item()) > 0.0:
            ego_raw, ego_target = _compute_exact_counterfactual_shapley_torch(
                model=self.model,
                state_obs=state_obs,
                clean_reference_obs=clean_reference_obs,
                temperature=temperature,
                utility_mode="target_logit_clean_l1",
                target_modality_index=2,
            )
            del ego_raw
            teacher = (1.0 - alpha) * fused_target + alpha * ego_target
            teacher = teacher / teacher.sum().clamp(min=1e-8)
            source = "hybrid_ego" if alpha < 0.999 else "targetlogit_ego"
            return teacher, source
        return fused_target, "clean_fused_l2"

    def _compute_shapley_alignment_loss(
        self,
        batch,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses_cfg = self.raw_config["losses"]
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 4))
        temperature = float(losses_cfg.get("shapley_temperature", 0.25))
        positions = self._sample_shapley_positions(
            valid_mask=batch.valid_mask,
            corrupted_flag=batch.extras["corrupted_flag"],
            target_mask=batch.extras["target_mask"],
            max_states=num_states,
        )
        if not positions:
            return torch.zeros((), device=self.device)

        kl_terms = []
        for batch_idx, time_idx in positions:
            teacher_target, _source = self._build_attention_teacher(
                batch=batch,
                batch_idx=batch_idx,
                time_idx=time_idx,
                temperature=temperature,
            )
            target_tensor = teacher_target.detach().clamp_min(1e-8)
            attention = outputs["attention_weights"][batch_idx, time_idx].clamp_min(1e-8)
            kl_terms.append(torch.sum(attention * (torch.log(attention) - torch.log(target_tensor))))
        return torch.stack(kl_terms).mean()

    @torch.no_grad()
    def run_eval_suite(self, num_eval_episodes: int) -> list[dict[str, Any]]:
        rows = super().run_eval_suite(num_eval_episodes=num_eval_episodes)
        alpha = float(self.raw_config["losses"].get("mild_target_teacher_mix_alpha", 0.0))
        for row in rows:
            probe_name = str(row["corruption_type"])
            if probe_name == "normal_eval":
                row["attention_teacher_source"] = "clean_fused_l2"
            elif probe_name == "mild_vision":
                row["attention_teacher_source"] = "hybrid_vision" if alpha < 0.999 else "targetlogit_vision"
            elif probe_name == "mild_ego":
                row["attention_teacher_source"] = "hybrid_ego" if alpha < 0.999 else "targetlogit_ego"
            else:
                row["attention_teacher_source"] = "clean_fused_l2"
        return rows


def build_v3_candidate_config(config: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    candidate_cfg = copy.deepcopy(config)
    candidate_spec = candidate_cfg["candidates"][candidate_name]
    if "reference_model_mode" not in candidate_cfg:
        candidate_cfg["reference_model_mode"] = "source_checkpoint" if candidate_cfg.get("source_checkpoint_root") else "frozen_copy"
    if "reference_checkpoint_root" not in candidate_cfg and candidate_spec.get("checkpoint_root"):
        candidate_cfg["reference_checkpoint_root"] = candidate_spec["checkpoint_root"]
    for key, value in candidate_spec.get("loss_overrides", {}).items():
        candidate_cfg["losses"][key] = value
    candidate_cfg["losses"] = canonicalize_gate_loss_keys(candidate_cfg["losses"])
    for key, value in candidate_spec.get("train_overrides", {}).items():
        candidate_cfg["train"][key] = value
    for key, value in candidate_spec.get("train_corruption_overrides", {}).items():
        candidate_cfg["train_corruption"][key] = value
    return candidate_cfg


def extract_loss_overrides_from_snapshot(snapshot_path: Path | None) -> dict[str, Any]:
    if snapshot_path is None or not snapshot_path.exists():
        return {}
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if isinstance(snapshot.get("losses"), dict):
        return dict(snapshot["losses"])
    nested_config = snapshot.get("config")
    if isinstance(nested_config, dict) and isinstance(nested_config.get("losses"), dict):
        return dict(nested_config["losses"])
    return {}


def merge_teacher_loss_overrides(
    snapshot_overrides: dict[str, Any],
    runtime_overrides: dict[str, Any] | None,
    *,
    snapshot_precedence: bool = True,
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    runtime = dict(runtime_overrides or {})
    merged = dict(runtime)
    diffs: dict[str, tuple[Any, Any]] = {}
    if snapshot_precedence:
        for key, value in snapshot_overrides.items():
            if key in runtime and runtime[key] != value:
                diffs[key] = (runtime[key], value)
            merged[key] = value
    else:
        merged.update(snapshot_overrides)
    return merged, diffs


def apply_attention_loss_overrides_to_model(model: torch.nn.Module, losses_cfg: dict[str, Any] | None) -> None:
    if not losses_cfg or not hasattr(model, "fusion_module"):
        return
    losses_cfg = canonicalize_gate_loss_keys(dict(losses_cfg))
    fusion = model.fusion_module
    fusion.score_adapter_residual_scale = float(
        losses_cfg.get("score_adapter_residual_scale", getattr(fusion, "score_adapter_residual_scale", 0.0))
    )
    fusion.relative_gap_residual_scale = float(
        losses_cfg.get("relative_gap_residual_scale", getattr(fusion, "relative_gap_residual_scale", 0.0))
    )
    fusion.reliability_bias_residual_scale = float(
        losses_cfg.get("reliability_bias_residual_scale", getattr(fusion, "reliability_bias_residual_scale", 0.0))
    )
    fusion.reliability_bias_gate_temperature = float(
        losses_cfg.get(
            "reliability_bias_gate_temperature",
            getattr(fusion, "reliability_bias_gate_temperature", 1.0),
        )
    )
    fusion.reliability_bias_gate_threshold = float(
        losses_cfg.get(
            "reliability_bias_gate_threshold",
            getattr(fusion, "reliability_bias_gate_threshold", 0.0),
        )
    )
    fusion.cancel_temperature = float(
        losses_cfg.get("cancel_temperature", getattr(fusion, "cancel_temperature", 1.0))
    )
    fusion.selector_temperature = float(
        losses_cfg.get("selector_temperature", getattr(fusion, "selector_temperature", 1.0))
    )
    fusion.rescue_selector_floor = min(
        max(float(losses_cfg.get("rescue_selector_floor", getattr(fusion, "rescue_selector_floor", 0.0))), 0.0),
        0.999,
    )
    fusion.clean_gate_max = float(losses_cfg.get("clean_gate_max", getattr(fusion, "clean_gate_max", 0.02)))
    fusion.mild_gate_min = float(losses_cfg.get("mild_gate_min", getattr(fusion, "mild_gate_min", 0.08)))
    fusion.gate_gap_margin = float(losses_cfg.get("gate_gap_margin", getattr(fusion, "gate_gap_margin", 0.05)))
    fusion.gate_residual_scale = float(losses_cfg.get("gate_residual_scale", getattr(fusion, "gate_residual_scale", 0.0)))
    fusion.gate_residual_alpha_mode = str(
        losses_cfg.get("residual_alpha_mode", getattr(fusion, "gate_residual_alpha_mode", "fixed"))
    )
    fusion.gate_residual_alpha_max = float(
        losses_cfg.get("residual_alpha_max", getattr(fusion, "gate_residual_alpha_max", 8.0))
    )
    fusion.gate_residual_alpha_l2_coef = float(
        losses_cfg.get("residual_alpha_l2_coef", getattr(fusion, "gate_residual_alpha_l2_coef", 0.0))
    )
    alpha_init = losses_cfg.get("residual_alpha_init", getattr(fusion, "gate_residual_alpha", 1.0))
    alpha_tensor = torch.as_tensor(alpha_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_residual_alpha", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_residual_alpha.copy_(alpha_tensor.to(device=fusion.gate_residual_alpha.device))
    else:
        fusion.gate_residual_alpha = alpha_tensor if alpha_tensor.ndim > 0 else float(alpha_tensor.item())
    fusion.gate_logit_offset_mode = str(
        losses_cfg.get("gate_logit_offset_mode", getattr(fusion, "gate_logit_offset_mode", "fixed"))
    )
    fusion.gate_logit_offset_max_abs = float(
        losses_cfg.get("gate_logit_offset_max_abs", getattr(fusion, "gate_logit_offset_max_abs", 2.0))
    )
    fusion.gate_logit_offset_l2_coef = float(
        losses_cfg.get("gate_logit_offset_l2_coef", getattr(fusion, "gate_logit_offset_l2_coef", 0.0))
    )
    offset_init = losses_cfg.get("gate_logit_offset_init", getattr(fusion, "gate_logit_offset", 0.0))
    offset_tensor = torch.as_tensor(offset_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_logit_offset", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_logit_offset.copy_(offset_tensor.to(device=fusion.gate_logit_offset.device))
    else:
        fusion.gate_logit_offset = offset_tensor if offset_tensor.ndim > 0 else float(offset_tensor.item())
    fusion.adapter_gate_alpha_mode = str(
        losses_cfg.get("adapter_gate_alpha_mode", getattr(fusion, "adapter_gate_alpha_mode", "fixed"))
    )
    fusion.adapter_gate_alpha_max = float(
        losses_cfg.get("adapter_gate_alpha_max", getattr(fusion, "adapter_gate_alpha_max", 8.0))
    )
    fusion.adapter_gate_alpha_l2_coef = float(
        losses_cfg.get("adapter_gate_alpha_l2_coef", getattr(fusion, "adapter_gate_alpha_l2_coef", 0.0))
    )
    adapter_alpha_init = losses_cfg.get("adapter_gate_alpha_init", getattr(fusion, "adapter_gate_alpha", 0.0))
    adapter_alpha_tensor = torch.as_tensor(adapter_alpha_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "adapter_gate_alpha", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.adapter_gate_alpha.copy_(adapter_alpha_tensor.to(device=fusion.adapter_gate_alpha.device))
    else:
        fusion.adapter_gate_alpha = (
            adapter_alpha_tensor if adapter_alpha_tensor.ndim > 0 else float(adapter_alpha_tensor.item())
        )
    fusion.gate_output_scale_mode = str(
        losses_cfg.get("gate_output_scale_mode", getattr(fusion, "gate_output_scale_mode", "fixed"))
    )
    fusion.gate_output_scale_max = float(
        losses_cfg.get("gate_output_scale_max", getattr(fusion, "gate_output_scale_max", 16.0))
    )
    fusion.gate_output_scale_l2_coef = float(
        losses_cfg.get("gate_output_scale_l2_coef", getattr(fusion, "gate_output_scale_l2_coef", 0.0))
    )
    gate_output_scale_init = losses_cfg.get("gate_output_scale_init", getattr(fusion, "gate_output_scale", 1.0))
    gate_output_scale_tensor = torch.as_tensor(gate_output_scale_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_output_scale", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_output_scale.copy_(gate_output_scale_tensor.to(device=fusion.gate_output_scale.device))
    else:
        fusion.gate_output_scale = (
            gate_output_scale_tensor if gate_output_scale_tensor.ndim > 0 else float(gate_output_scale_tensor.item())
        )
    fusion.gate_threshold_shift_alpha_mode = str(
        losses_cfg.get("gate_threshold_shift_alpha_mode", getattr(fusion, "gate_threshold_shift_alpha_mode", "fixed"))
    )
    fusion.gate_threshold_shift_alpha_max = float(
        losses_cfg.get("gate_threshold_shift_alpha_max", getattr(fusion, "gate_threshold_shift_alpha_max", 8.0))
    )
    fusion.gate_threshold_shift_alpha_l2_coef = float(
        losses_cfg.get("gate_threshold_shift_alpha_l2_coef", getattr(fusion, "gate_threshold_shift_alpha_l2_coef", 0.0))
    )
    gate_threshold_shift_alpha_init = losses_cfg.get(
        "gate_threshold_shift_alpha_init",
        getattr(fusion, "gate_threshold_shift_alpha", 0.0),
    )
    gate_threshold_shift_alpha_tensor = torch.as_tensor(gate_threshold_shift_alpha_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_threshold_shift_alpha", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_threshold_shift_alpha.copy_(
                gate_threshold_shift_alpha_tensor.to(device=fusion.gate_threshold_shift_alpha.device)
            )
    else:
        fusion.gate_threshold_shift_alpha = (
            gate_threshold_shift_alpha_tensor
            if gate_threshold_shift_alpha_tensor.ndim > 0
            else float(gate_threshold_shift_alpha_tensor.item())
        )
    fusion.gate_source_logit_bias_mode = str(
        losses_cfg.get("gate_source_logit_bias_mode", getattr(fusion, "gate_source_logit_bias_mode", "fixed"))
    )
    fusion.gate_source_logit_bias_max_abs = float(
        losses_cfg.get("gate_source_logit_bias_max_abs", getattr(fusion, "gate_source_logit_bias_max_abs", 2.0))
    )
    fusion.gate_source_logit_bias_l2_coef = float(
        losses_cfg.get("gate_source_logit_bias_l2_coef", getattr(fusion, "gate_source_logit_bias_l2_coef", 0.0))
    )
    gate_source_logit_bias_init = losses_cfg.get(
        "gate_source_logit_bias_init",
        getattr(fusion, "gate_source_logit_bias", 0.0),
    )
    gate_source_logit_bias_tensor = torch.as_tensor(gate_source_logit_bias_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_source_logit_bias", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_source_logit_bias.copy_(
                gate_source_logit_bias_tensor.to(device=fusion.gate_source_logit_bias.device)
            )
    else:
        fusion.gate_source_logit_bias = (
            gate_source_logit_bias_tensor
            if gate_source_logit_bias_tensor.ndim > 0
            else float(gate_source_logit_bias_tensor.item())
        )
    fusion.gate_common_cancel_alpha_mode = str(
        losses_cfg.get("gate_common_cancel_alpha_mode", getattr(fusion, "gate_common_cancel_alpha_mode", "fixed"))
    )
    fusion.gate_common_cancel_alpha_max = float(
        losses_cfg.get("gate_common_cancel_alpha_max", getattr(fusion, "gate_common_cancel_alpha_max", 8.0))
    )
    fusion.gate_common_cancel_alpha_l2_coef = float(
        losses_cfg.get("gate_common_cancel_alpha_l2_coef", getattr(fusion, "gate_common_cancel_alpha_l2_coef", 0.0))
    )
    gate_common_cancel_alpha_init = losses_cfg.get(
        "gate_common_cancel_alpha_init",
        getattr(fusion, "gate_common_cancel_alpha", 0.0),
    )
    gate_common_cancel_alpha_tensor = torch.as_tensor(gate_common_cancel_alpha_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_common_cancel_alpha", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_common_cancel_alpha.copy_(
                gate_common_cancel_alpha_tensor.to(device=fusion.gate_common_cancel_alpha.device)
            )
    else:
        fusion.gate_common_cancel_alpha = (
            gate_common_cancel_alpha_tensor
            if gate_common_cancel_alpha_tensor.ndim > 0
            else float(gate_common_cancel_alpha_tensor.item())
        )
    fusion.common_cancel_anchor_gate = float(
        losses_cfg.get("common_cancel_anchor_gate", getattr(fusion, "common_cancel_anchor_gate", 0.05))
    )
    fusion.gate_contrast_shift_alpha_mode = str(
        losses_cfg.get("gate_contrast_shift_alpha_mode", getattr(fusion, "gate_contrast_shift_alpha_mode", "fixed"))
    )
    fusion.gate_contrast_shift_alpha_max = float(
        losses_cfg.get("gate_contrast_shift_alpha_max", getattr(fusion, "gate_contrast_shift_alpha_max", 8.0))
    )
    fusion.gate_contrast_shift_alpha_l2_coef = float(
        losses_cfg.get("gate_contrast_shift_alpha_l2_coef", getattr(fusion, "gate_contrast_shift_alpha_l2_coef", 0.0))
    )
    gate_contrast_shift_alpha_init = losses_cfg.get(
        "gate_contrast_shift_alpha_init",
        getattr(fusion, "gate_contrast_shift_alpha", 0.0),
    )
    gate_contrast_shift_alpha_tensor = torch.as_tensor(gate_contrast_shift_alpha_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_contrast_shift_alpha", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_contrast_shift_alpha.copy_(
                gate_contrast_shift_alpha_tensor.to(device=fusion.gate_contrast_shift_alpha.device)
            )
    else:
        fusion.gate_contrast_shift_alpha = (
            gate_contrast_shift_alpha_tensor
            if gate_contrast_shift_alpha_tensor.ndim > 0
            else float(gate_contrast_shift_alpha_tensor.item())
        )
    fusion.gate_mild_rescue_alpha_mode = str(
        losses_cfg.get("gate_mild_rescue_alpha_mode", getattr(fusion, "gate_mild_rescue_alpha_mode", "fixed"))
    )
    fusion.gate_mild_rescue_alpha_max = float(
        losses_cfg.get("gate_mild_rescue_alpha_max", getattr(fusion, "gate_mild_rescue_alpha_max", 8.0))
    )
    fusion.gate_mild_rescue_alpha_l2_coef = float(
        losses_cfg.get("gate_mild_rescue_alpha_l2_coef", getattr(fusion, "gate_mild_rescue_alpha_l2_coef", 0.0))
    )
    gate_mild_rescue_alpha_init = losses_cfg.get(
        "gate_mild_rescue_alpha_init",
        getattr(fusion, "gate_mild_rescue_alpha", 0.0),
    )
    gate_mild_rescue_alpha_tensor = torch.as_tensor(gate_mild_rescue_alpha_init, dtype=torch.float32)
    if isinstance(getattr(fusion, "gate_mild_rescue_alpha", None), torch.nn.Parameter):
        with torch.no_grad():
            fusion.gate_mild_rescue_alpha.copy_(
                gate_mild_rescue_alpha_tensor.to(device=fusion.gate_mild_rescue_alpha.device)
            )
    else:
        fusion.gate_mild_rescue_alpha = (
            gate_mild_rescue_alpha_tensor
            if gate_mild_rescue_alpha_tensor.ndim > 0
            else float(gate_mild_rescue_alpha_tensor.item())
        )
    fusion.use_relative_score_head = bool(
        losses_cfg.get("use_relative_score_head", getattr(fusion, "use_relative_score_head", False))
    )
    fusion.use_reliability_bias_head = bool(
        losses_cfg.get("use_reliability_bias_head", getattr(fusion, "use_reliability_bias_head", False))
    )
    fusion.use_reliability_bias_gate = bool(
        losses_cfg.get("use_reliability_bias_gate", getattr(fusion, "use_reliability_bias_gate", False))
    )
    fusion.relative_gap_mode = str(losses_cfg.get("relative_gap_mode", getattr(fusion, "relative_gap_mode", "vision_anchor")))
    fusion.gate_mode = str(losses_cfg.get("gate_mode", getattr(fusion, "gate_mode", "absolute_v1")))
    fusion.gate_input_mode = str(losses_cfg.get("gate_input_mode", getattr(fusion, "gate_input_mode", "legacy")))
    fusion.adapter_gate_input_mode = str(
        losses_cfg.get("adapter_gate_input_mode", getattr(fusion, "adapter_gate_input_mode", "legacy"))
    )
    fusion.gate_use_abs_bias_feature = bool(
        losses_cfg.get("gate_use_abs_bias_feature", getattr(fusion, "gate_use_abs_bias_feature", False))
    )
    fusion.gate_use_score_delta_feature = bool(
        losses_cfg.get("gate_use_score_delta_feature", getattr(fusion, "gate_use_score_delta_feature", False))
    )
    fusion.gate_use_teacher_delta_feature = bool(
        losses_cfg.get("gate_use_teacher_delta_feature", getattr(fusion, "gate_use_teacher_delta_feature", False))
    )
    fusion.gate_use_source_gate_feature = bool(
        losses_cfg.get("gate_use_source_gate_feature", getattr(fusion, "gate_use_source_gate_feature", False))
    )
    fusion.gate_use_adapter_delta_feature = bool(
        losses_cfg.get("gate_use_adapter_delta_feature", getattr(fusion, "gate_use_adapter_delta_feature", False))
    )
    fusion.gate_use_adapter_norm_feature = bool(
        losses_cfg.get("gate_use_adapter_norm_feature", getattr(fusion, "gate_use_adapter_norm_feature", False))
    )
    fusion.gate_use_adapter_teacher_delta_feature = bool(
        losses_cfg.get(
            "gate_use_adapter_teacher_delta_feature",
            getattr(fusion, "gate_use_adapter_teacher_delta_feature", False),
        )
    )
    fusion.gate_feature_stopgrad_on_bias = bool(
        losses_cfg.get("gate_feature_stopgrad_on_bias", getattr(fusion, "gate_feature_stopgrad_on_bias", False))
    )
    fusion.gate_feature_stopgrad_on_source_gate = bool(
        losses_cfg.get(
            "gate_feature_stopgrad_on_source_gate",
            getattr(fusion, "gate_feature_stopgrad_on_source_gate", False),
        )
    )
    fusion.gate_feature_stopgrad_on_adapter_delta = bool(
        losses_cfg.get(
            "gate_feature_stopgrad_on_adapter_delta",
            getattr(fusion, "gate_feature_stopgrad_on_adapter_delta", False),
        )
    )


def configure_trainable_gate_residual_alpha(
    model: torch.nn.Module,
    losses_cfg: dict[str, Any],
) -> None:
    if not hasattr(model, "fusion_module"):
        return
    fusion = model.fusion_module
    alpha_mode = str(losses_cfg.get("residual_alpha_mode", getattr(fusion, "gate_residual_alpha_mode", "fixed")))
    alpha_init = torch.as_tensor(
        losses_cfg.get("residual_alpha_init", getattr(fusion, "gate_residual_alpha", 1.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_residual_alpha_mode = alpha_mode
    fusion.gate_residual_alpha_max = float(
        losses_cfg.get("residual_alpha_max", getattr(fusion, "gate_residual_alpha_max", 8.0))
    )
    fusion.gate_residual_alpha_l2_coef = float(
        losses_cfg.get("residual_alpha_l2_coef", getattr(fusion, "gate_residual_alpha_l2_coef", 0.0))
    )
    if alpha_mode == "learnable":
        parameter = getattr(fusion, "gate_residual_alpha", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(alpha_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_residual_alpha"):
                delattr(fusion, "gate_residual_alpha")
            fusion.register_parameter(
                "gate_residual_alpha",
                torch.nn.Parameter(alpha_init.clone()),
            )
    else:
        existing = getattr(fusion, "gate_residual_alpha", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_residual_alpha"]
        fusion.gate_residual_alpha = alpha_init.clone() if alpha_init.ndim > 0 else float(alpha_init.item())


def configure_trainable_gate_logit_offset(
    model: torch.nn.Module,
    losses_cfg: dict[str, Any],
) -> None:
    if not hasattr(model, "fusion_module"):
        return
    fusion = model.fusion_module
    offset_mode = str(losses_cfg.get("gate_logit_offset_mode", getattr(fusion, "gate_logit_offset_mode", "fixed")))
    offset_init = torch.as_tensor(
        losses_cfg.get("gate_logit_offset_init", getattr(fusion, "gate_logit_offset", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_logit_offset_mode = offset_mode
    fusion.gate_logit_offset_max_abs = float(
        losses_cfg.get("gate_logit_offset_max_abs", getattr(fusion, "gate_logit_offset_max_abs", 2.0))
    )
    fusion.gate_logit_offset_l2_coef = float(
        losses_cfg.get("gate_logit_offset_l2_coef", getattr(fusion, "gate_logit_offset_l2_coef", 0.0))
    )
    if offset_mode == "learnable":
        parameter = getattr(fusion, "gate_logit_offset", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(offset_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_logit_offset"):
                delattr(fusion, "gate_logit_offset")
            fusion.register_parameter(
                "gate_logit_offset",
                torch.nn.Parameter(offset_init.clone()),
            )
    else:
        existing = getattr(fusion, "gate_logit_offset", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_logit_offset"]
        fusion.gate_logit_offset = offset_init.clone() if offset_init.ndim > 0 else float(offset_init.item())


def configure_trainable_gate_actuation_parameters(
    model: torch.nn.Module,
    losses_cfg: dict[str, Any],
) -> None:
    configure_trainable_gate_residual_alpha(model, losses_cfg)
    configure_trainable_gate_logit_offset(model, losses_cfg)
    if not hasattr(model, "fusion_module"):
        return
    fusion = model.fusion_module
    alpha_mode = str(losses_cfg.get("adapter_gate_alpha_mode", getattr(fusion, "adapter_gate_alpha_mode", "fixed")))
    alpha_init = torch.as_tensor(
        losses_cfg.get("adapter_gate_alpha_init", getattr(fusion, "adapter_gate_alpha", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.adapter_gate_alpha_mode = alpha_mode
    fusion.adapter_gate_alpha_max = float(
        losses_cfg.get("adapter_gate_alpha_max", getattr(fusion, "adapter_gate_alpha_max", 8.0))
    )
    fusion.adapter_gate_alpha_l2_coef = float(
        losses_cfg.get("adapter_gate_alpha_l2_coef", getattr(fusion, "adapter_gate_alpha_l2_coef", 0.0))
    )
    if alpha_mode == "learnable":
        parameter = getattr(fusion, "adapter_gate_alpha", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(alpha_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "adapter_gate_alpha"):
                delattr(fusion, "adapter_gate_alpha")
            fusion.register_parameter("adapter_gate_alpha", torch.nn.Parameter(alpha_init.clone()))
    else:
        existing = getattr(fusion, "adapter_gate_alpha", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["adapter_gate_alpha"]
        fusion.adapter_gate_alpha = alpha_init.clone() if alpha_init.ndim > 0 else float(alpha_init.item())
    scale_mode = str(losses_cfg.get("gate_output_scale_mode", getattr(fusion, "gate_output_scale_mode", "fixed")))
    scale_init = torch.as_tensor(
        losses_cfg.get("gate_output_scale_init", getattr(fusion, "gate_output_scale", 1.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_output_scale_mode = scale_mode
    fusion.gate_output_scale_max = float(
        losses_cfg.get("gate_output_scale_max", getattr(fusion, "gate_output_scale_max", 16.0))
    )
    fusion.gate_output_scale_l2_coef = float(
        losses_cfg.get("gate_output_scale_l2_coef", getattr(fusion, "gate_output_scale_l2_coef", 0.0))
    )
    if scale_mode == "learnable":
        parameter = getattr(fusion, "gate_output_scale", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(scale_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_output_scale"):
                delattr(fusion, "gate_output_scale")
            fusion.register_parameter("gate_output_scale", torch.nn.Parameter(scale_init.clone()))
    else:
        existing = getattr(fusion, "gate_output_scale", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_output_scale"]
        fusion.gate_output_scale = scale_init.clone() if scale_init.ndim > 0 else float(scale_init.item())
    threshold_shift_mode = str(
        losses_cfg.get("gate_threshold_shift_alpha_mode", getattr(fusion, "gate_threshold_shift_alpha_mode", "fixed"))
    )
    threshold_shift_init = torch.as_tensor(
        losses_cfg.get("gate_threshold_shift_alpha_init", getattr(fusion, "gate_threshold_shift_alpha", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_threshold_shift_alpha_mode = threshold_shift_mode
    fusion.gate_threshold_shift_alpha_max = float(
        losses_cfg.get("gate_threshold_shift_alpha_max", getattr(fusion, "gate_threshold_shift_alpha_max", 8.0))
    )
    fusion.gate_threshold_shift_alpha_l2_coef = float(
        losses_cfg.get("gate_threshold_shift_alpha_l2_coef", getattr(fusion, "gate_threshold_shift_alpha_l2_coef", 0.0))
    )
    if threshold_shift_mode == "learnable":
        parameter = getattr(fusion, "gate_threshold_shift_alpha", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(threshold_shift_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_threshold_shift_alpha"):
                delattr(fusion, "gate_threshold_shift_alpha")
            fusion.register_parameter("gate_threshold_shift_alpha", torch.nn.Parameter(threshold_shift_init.clone()))
    else:
        existing = getattr(fusion, "gate_threshold_shift_alpha", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_threshold_shift_alpha"]
        fusion.gate_threshold_shift_alpha = (
            threshold_shift_init.clone() if threshold_shift_init.ndim > 0 else float(threshold_shift_init.item())
        )
    source_logit_bias_mode = str(
        losses_cfg.get("gate_source_logit_bias_mode", getattr(fusion, "gate_source_logit_bias_mode", "fixed"))
    )
    source_logit_bias_init = torch.as_tensor(
        losses_cfg.get("gate_source_logit_bias_init", getattr(fusion, "gate_source_logit_bias", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_source_logit_bias_mode = source_logit_bias_mode
    fusion.gate_source_logit_bias_max_abs = float(
        losses_cfg.get("gate_source_logit_bias_max_abs", getattr(fusion, "gate_source_logit_bias_max_abs", 2.0))
    )
    fusion.gate_source_logit_bias_l2_coef = float(
        losses_cfg.get("gate_source_logit_bias_l2_coef", getattr(fusion, "gate_source_logit_bias_l2_coef", 0.0))
    )
    if source_logit_bias_mode == "learnable":
        parameter = getattr(fusion, "gate_source_logit_bias", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(source_logit_bias_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_source_logit_bias"):
                delattr(fusion, "gate_source_logit_bias")
            fusion.register_parameter("gate_source_logit_bias", torch.nn.Parameter(source_logit_bias_init.clone()))
    else:
        existing = getattr(fusion, "gate_source_logit_bias", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_source_logit_bias"]
        fusion.gate_source_logit_bias = (
            source_logit_bias_init.clone() if source_logit_bias_init.ndim > 0 else float(source_logit_bias_init.item())
        )
    common_cancel_mode = str(
        losses_cfg.get("gate_common_cancel_alpha_mode", getattr(fusion, "gate_common_cancel_alpha_mode", "fixed"))
    )
    common_cancel_init = torch.as_tensor(
        losses_cfg.get("gate_common_cancel_alpha_init", getattr(fusion, "gate_common_cancel_alpha", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_common_cancel_alpha_mode = common_cancel_mode
    fusion.gate_common_cancel_alpha_max = float(
        losses_cfg.get("gate_common_cancel_alpha_max", getattr(fusion, "gate_common_cancel_alpha_max", 8.0))
    )
    fusion.gate_common_cancel_alpha_l2_coef = float(
        losses_cfg.get("gate_common_cancel_alpha_l2_coef", getattr(fusion, "gate_common_cancel_alpha_l2_coef", 0.0))
    )
    if common_cancel_mode == "learnable":
        parameter = getattr(fusion, "gate_common_cancel_alpha", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(common_cancel_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_common_cancel_alpha"):
                delattr(fusion, "gate_common_cancel_alpha")
            fusion.register_parameter("gate_common_cancel_alpha", torch.nn.Parameter(common_cancel_init.clone()))
    else:
        existing = getattr(fusion, "gate_common_cancel_alpha", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_common_cancel_alpha"]
        fusion.gate_common_cancel_alpha = (
            common_cancel_init.clone() if common_cancel_init.ndim > 0 else float(common_cancel_init.item())
        )
    fusion.common_cancel_anchor_gate = float(
        losses_cfg.get("common_cancel_anchor_gate", getattr(fusion, "common_cancel_anchor_gate", 0.05))
    )
    contrast_shift_mode = str(
        losses_cfg.get("gate_contrast_shift_alpha_mode", getattr(fusion, "gate_contrast_shift_alpha_mode", "fixed"))
    )
    contrast_shift_init = torch.as_tensor(
        losses_cfg.get("gate_contrast_shift_alpha_init", getattr(fusion, "gate_contrast_shift_alpha", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_contrast_shift_alpha_mode = contrast_shift_mode
    fusion.gate_contrast_shift_alpha_max = float(
        losses_cfg.get("gate_contrast_shift_alpha_max", getattr(fusion, "gate_contrast_shift_alpha_max", 8.0))
    )
    fusion.gate_contrast_shift_alpha_l2_coef = float(
        losses_cfg.get("gate_contrast_shift_alpha_l2_coef", getattr(fusion, "gate_contrast_shift_alpha_l2_coef", 0.0))
    )
    if contrast_shift_mode == "learnable":
        parameter = getattr(fusion, "gate_contrast_shift_alpha", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(contrast_shift_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_contrast_shift_alpha"):
                delattr(fusion, "gate_contrast_shift_alpha")
            fusion.register_parameter("gate_contrast_shift_alpha", torch.nn.Parameter(contrast_shift_init.clone()))
    else:
        existing = getattr(fusion, "gate_contrast_shift_alpha", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_contrast_shift_alpha"]
        fusion.gate_contrast_shift_alpha = (
            contrast_shift_init.clone() if contrast_shift_init.ndim > 0 else float(contrast_shift_init.item())
        )
    mild_rescue_mode = str(
        losses_cfg.get("gate_mild_rescue_alpha_mode", getattr(fusion, "gate_mild_rescue_alpha_mode", "fixed"))
    )
    mild_rescue_init = torch.as_tensor(
        losses_cfg.get("gate_mild_rescue_alpha_init", getattr(fusion, "gate_mild_rescue_alpha", 0.0)),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    fusion.gate_mild_rescue_alpha_mode = mild_rescue_mode
    fusion.gate_mild_rescue_alpha_max = float(
        losses_cfg.get("gate_mild_rescue_alpha_max", getattr(fusion, "gate_mild_rescue_alpha_max", 8.0))
    )
    fusion.gate_mild_rescue_alpha_l2_coef = float(
        losses_cfg.get("gate_mild_rescue_alpha_l2_coef", getattr(fusion, "gate_mild_rescue_alpha_l2_coef", 0.0))
    )
    if mild_rescue_mode == "learnable":
        parameter = getattr(fusion, "gate_mild_rescue_alpha", None)
        if isinstance(parameter, torch.nn.Parameter):
            with torch.no_grad():
                parameter.copy_(mild_rescue_init.to(device=parameter.device))
        else:
            if hasattr(fusion, "gate_mild_rescue_alpha"):
                delattr(fusion, "gate_mild_rescue_alpha")
            fusion.register_parameter("gate_mild_rescue_alpha", torch.nn.Parameter(mild_rescue_init.clone()))
    else:
        existing = getattr(fusion, "gate_mild_rescue_alpha", None)
        if isinstance(existing, torch.nn.Parameter):
            del fusion._parameters["gate_mild_rescue_alpha"]
        fusion.gate_mild_rescue_alpha = (
            mild_rescue_init.clone() if mild_rescue_init.ndim > 0 else float(mild_rescue_init.item())
        )


def normalize_tiebreak_weights(tiebreak_weights: dict[str, float] | None) -> dict[str, float]:
    weights = dict(tiebreak_weights or {})
    for key in [
        "adapter_activity",
        "alignment_kl_improvement",
        "alignment_rank_corr_improvement",
        "attention_sign_match",
        "clean_action_anchor",
        "clean_score_anchor",
        "ego_delta",
        "ego_logit_delta",
        "ego_raw_shapley_delta",
        "ego_shapley_delta",
        "vision_sign_match",
        "vision_attention_delta",
        "vision_attention_relative_score_delta",
        "vision_attention_score_delta",
        "vision_bias_delta",
        "vision_delta",
        "vision_route_gap_delta",
        "vision_distance_gap_delta",
        "vision_gate_delta",
        "vision_logit_delta",
        "vision_raw_shapley_delta",
        "vision_shapley_delta",
        "normal_success",
        "mild_success",
        "return_degradation",
        "bias_activity",
        "consistency_gain",
        "success_rate",
        "degradation",
        "vision_targetlogit_delta",
        "relative_head_activity",
        "saturation_penalty",
        "side_effect_penalty",
    ]:
        weights.setdefault(key, 0.0)
    return weights


def restore_legacy_absolute_gate_head_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    prefix = "fusion_module.reliability_bias_gate_head."
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    required = [
        f"{prefix}0.weight",
        f"{prefix}0.bias",
        f"{prefix}1.weight",
        f"{prefix}1.bias",
        f"{prefix}3.weight",
        f"{prefix}3.bias",
    ]
    if not all(key in state_dict for key in required):
        return
    layernorm_weight = state_dict[f"{prefix}0.weight"]
    linear1_weight = state_dict[f"{prefix}1.weight"]
    linear1_bias = state_dict[f"{prefix}1.bias"]
    linear2_weight = state_dict[f"{prefix}3.weight"]
    linear2_bias = state_dict[f"{prefix}3.bias"]
    in_dim = int(layernorm_weight.shape[0])
    hidden_dim = int(linear1_weight.shape[0])
    out_dim = int(linear2_weight.shape[0])
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(in_dim),
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        head[0].weight.copy_(layernorm_weight.to(head[0].weight.device))
        head[0].bias.copy_(state_dict[f"{prefix}0.bias"].to(head[0].bias.device))
        head[1].weight.copy_(linear1_weight.to(head[1].weight.device))
        head[1].bias.copy_(linear1_bias.to(head[1].bias.device))
        head[3].weight.copy_(linear2_weight.to(head[3].weight.device))
        head[3].bias.copy_(linear2_bias.to(head[3].bias.device))
    model.fusion_module.reliability_bias_gate_head = head


def load_frozen_teacher_model(
    config: dict[str, Any],
    checkpoint_root: str,
    source_experiment_name: str,
    device: torch.device,
    seed: int,
    loss_overrides: dict[str, Any] | None = None,
) -> torch.nn.Module:
    seed_dir, source_seed = _resolve_seed_dir_from_root(Path(checkpoint_root), seed)
    checkpoint_path = latest_checkpoint(seed_dir)
    snapshot_path = seed_dir / "config_snapshot.json"
    candidate_snapshot_path = seed_dir / "candidate_config_snapshot.json"
    effective_snapshot_path = candidate_snapshot_path if candidate_snapshot_path.exists() else snapshot_path
    _cfg, model = build_model_from_candidate(
        base_stage1_config_path=str(config["base_stage1_config_path"]),
        checkpoint_path=checkpoint_path,
        candidate_snapshot_path=effective_snapshot_path if effective_snapshot_path.exists() else None,
        device=device,
        seed=source_seed,
    )
    snapshot_overrides = extract_loss_overrides_from_snapshot(effective_snapshot_path)
    merged_overrides, override_diffs = merge_teacher_loss_overrides(
        snapshot_overrides,
        loss_overrides,
        snapshot_precedence=bool(config.get("teacher_snapshot_precedence", True)),
    )
    if override_diffs:
        diff_lines = ", ".join(
            f"{key}: runtime={runtime!r} -> snapshot={snapshot!r}"
            for key, (runtime, snapshot) in sorted(override_diffs.items())
        )
        warnings.warn(
            f"Teacher override conflict resolved in favor of snapshot for {checkpoint_path.name}: {diff_lines}",
            stacklevel=2,
        )
    apply_attention_loss_overrides_to_model(model, merged_overrides)
    fusion = model.fusion_module
    if (
        bool(getattr(fusion, "use_reliability_bias_gate", False))
        and str(getattr(fusion, "gate_mode", "absolute_v1")) == "absolute_v1"
        and str(getattr(fusion, "gate_input_mode", "legacy")) == "legacy"
    ):
        restore_legacy_absolute_gate_head_from_checkpoint(model, checkpoint_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def save_v3_handoff_snapshot(config: dict[str, Any], summary_dir: Path, stem: str) -> dict[str, Any]:
    snapshot = {
        "experiment_name": config["experiment_name"],
        "source_experiment_name": config.get("source_experiment_name", "stage2_full"),
        "source_checkpoint_root": config.get("source_checkpoint_root", ""),
        "candidate_order": list(config.get("candidate_order", config["candidates"].keys())),
        "seeds": list(config["seeds"]),
        "baseline_candidate": config["baseline_candidate"],
        "losses": config["losses"],
        "train_corruption": config["train_corruption"],
        "eval_probes": config["eval_probes"],
    }
    (summary_dir / f"{stem}_handoff_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return snapshot


def resolve_teacher_loss_overrides(
    config: dict[str, Any],
    explicit_key: str,
    candidate_name: str | None = None,
) -> dict[str, Any]:
    explicit = config.get(explicit_key)
    if isinstance(explicit, dict):
        return dict(explicit)
    if candidate_name is None:
        return {}
    candidate_spec = config.get("candidates", {}).get(candidate_name, {})
    return dict(candidate_spec.get("loss_overrides", {}))


def run_v3_candidate(
    candidate_name: str,
    config: dict[str, Any],
    output_root: Path,
    metrics_filename: str,
    smoke: bool = False,
    trainer_cls: type[Stage3ShapleyTrainer] = Stage3ShapleyTrainer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = list(config["seeds"][:1] if smoke else config["seeds"])
    eval_episodes = int(config["num_eval_episodes_smoke"] if smoke else config["num_eval_episodes"])
    candidate_cfg = build_v3_candidate_config(config, candidate_name)
    candidate_spec = candidate_cfg["candidates"][candidate_name]
    force_rerun = bool(smoke) or bool(config.get("force_rerun_existing_eval_rows", False))

    for seed in seeds:
        run_dir = ensure_dir(output_root / candidate_name / f"seed_{seed}")
        summary_path = run_dir / metrics_filename
        if summary_path.exists() and not force_rerun:
            rows.extend(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        trainer = trainer_cls(config=candidate_cfg, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        if not bool(candidate_spec.get("eval_only", False)):
            trainer.train()
        else:
            torch.save({"model_state_dict": trainer.model.state_dict()}, run_dir / "checkpoint_final.pt")
            trainer.logger.save_status(
                {
                    "finished_flag": 1,
                    "nan_detected": 0,
                    "crashed_flag": 0,
                    "wall_clock_time": 0.0,
                    "total_updates": 0,
                }
            )
        eval_rows = trainer.run_eval_suite(num_eval_episodes=eval_episodes)
        summary_path.write_text(json.dumps(eval_rows, ensure_ascii=True, indent=2), encoding="utf-8")
        rows.extend(eval_rows)
    return rows


def compute_alignment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        mild_ego = sub[sub["corruption_type"] == "mild_ego"].mean(numeric_only=True)
        row = {
            "candidate_name": candidate_name,
            "attention_shapley_kl": float(sub["attention_shapley_kl"].mean()),
            "attention_shapley_rank_corr": float(sub["attention_shapley_rank_corr"].mean()),
            "vision_shapley_clean_minus_mild": float(normal["shapley_mean_vision"] - mild_vision["shapley_mean_vision"]),
            "ego_shapley_clean_minus_mild": float(normal["shapley_mean_ego"] - mild_ego["shapley_mean_ego"]),
        }
        for name in MODALITY_NAMES:
            row[f"shapley_mean_{name}"] = float(sub[f"shapley_mean_{name}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_v3_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["normal_success_rate"] < 0.7:
        return "FAIL"
    if summary["normal_to_mild_success_drop"] > 0.12 and summary["degradation_score"] < 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["normal_success_rate"] >= 0.75
        and summary["normal_to_mild_success_drop"] <= 0.10
        and summary["consistency_gain_score"] > 0.0
        and (summary["vision_reliability_clean_minus_mild"] > 0.0 or summary["ego_reliability_clean_minus_mild"] > 0.0)
        and summary["attention_shapley_kl_improvement"] > 0.0
        and summary["attention_shapley_rank_corr_improvement"] > 0.0
        and (summary["vision_shapley_clean_minus_mild"] > 0.0 or summary["ego_shapley_clean_minus_mild"] > 0.0)
        and summary["saturation_warning_flag"] == 0.0
        and summary["route_distance_side_effect_flag"] == 0.0
    ):
        return "PASS"
    return "WARN"


def build_v3_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    baseline_alignment = alignment_df[alignment_df["candidate_name"] == baseline_candidate]
    if baseline_alignment.empty:
        raise ValueError(f"Baseline candidate missing from alignment summary: {baseline_candidate}")
    baseline_alignment_row = baseline_alignment.iloc[0]
    final_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        align = alignment_df[alignment_df["candidate_name"] == candidate_name].iloc[0]
        summary = {
            "candidate_name": candidate_name,
            "normal_success_rate": float(normal["eval_success_rate"]),
            "normal_return": float(normal["eval_return"]),
            "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
            "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
            "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
            "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
            "consistency_loss": float(cons["consistency_loss"]),
            "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
            "consistency_gain_score": float(cons["consistency_gain_score"]),
            "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
            "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
            "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
            "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
            "saturation_warning_flag": float(tie["saturation_warning_flag"]),
            "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
            "attention_shapley_kl": float(align["attention_shapley_kl"]),
            "attention_shapley_rank_corr": float(align["attention_shapley_rank_corr"]),
            "vision_shapley_clean_minus_mild": float(align["vision_shapley_clean_minus_mild"]),
            "ego_shapley_clean_minus_mild": float(align["ego_shapley_clean_minus_mild"]),
        }
        for name in MODALITY_NAMES:
            summary[f"shapley_mean_{name}"] = float(align[f"shapley_mean_{name}"])
        summary["attention_shapley_kl_improvement"] = float(
            baseline_alignment_row["attention_shapley_kl"] - summary["attention_shapley_kl"]
        )
        summary["attention_shapley_rank_corr_improvement"] = float(
            summary["attention_shapley_rank_corr"] - baseline_alignment_row["attention_shapley_rank_corr"]
        )
        summary["degradation_score"] = -(
            2.0 * summary["normal_to_mild_success_drop"]
            + 1.0 * summary["normal_to_mild_return_drop"]
            + 1.0 * summary["normal_to_mild_progress_drop"]
        )
        summary["tie_break_score"] = (
            float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["vision_delta"]) * summary["vision_reliability_clean_minus_mild"]
            + float(tiebreak_weights["ego_delta"]) * summary["ego_reliability_clean_minus_mild"]
            + float(tiebreak_weights["vision_logit_delta"]) * summary["vision_logit_clean_minus_mild"]
            + float(tiebreak_weights["ego_logit_delta"]) * summary["ego_logit_clean_minus_mild"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["alignment_kl_improvement"]) * summary["attention_shapley_kl_improvement"]
            + float(tiebreak_weights["alignment_rank_corr_improvement"]) * summary["attention_shapley_rank_corr_improvement"]
            + float(tiebreak_weights["vision_shapley_delta"]) * summary["vision_shapley_clean_minus_mild"]
            + float(tiebreak_weights["ego_shapley_delta"]) * summary["ego_shapley_clean_minus_mild"]
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_v3_pass_fail(
            summary=summary,
            baseline_summary=baseline_alignment_row.to_dict(),
            is_baseline=candidate_name == baseline_candidate,
        )
        final_rows.append(summary)

    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "normal_success_rate",
            "degradation_score",
            "attention_shapley_kl_improvement",
            "attention_shapley_rank_corr_improvement",
            "vision_shapley_clean_minus_mild",
            "ego_shapley_clean_minus_mild",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def choose_v3_recommendations(score_df: pd.DataFrame, baseline_candidate: str) -> tuple[str, str]:
    pass_rows = score_df[(score_df["pass_fail"] == "PASS") & (score_df["candidate_name"] != baseline_candidate)]
    if not pass_rows.empty:
        recommended = str(pass_rows.iloc[0]["candidate_name"])
        remaining = score_df[score_df["candidate_name"] != recommended]
        secondary = str(remaining.iloc[0]["candidate_name"]) if not remaining.empty else recommended
        return recommended, secondary
    recommended = str(score_df.iloc[0]["candidate_name"])
    remaining = score_df[score_df["candidate_name"] != recommended]
    secondary = str(remaining.iloc[0]["candidate_name"]) if not remaining.empty else recommended
    return recommended, secondary


def generate_v3_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Alignment Summary",
        "",
        alignment_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_v3_winner(score_summary_path: str | Path, fallback_candidate: str) -> str:
    score_summary = Path(score_summary_path)
    if not score_summary.exists():
        return fallback_candidate
    df = pd.read_csv(score_summary)
    if df.empty or "candidate_name" not in df.columns:
        return fallback_candidate
    return str(df.iloc[0]["candidate_name"])


def maybe_override_checkpoint_root(
    config: dict[str, Any],
    candidate_name: str,
    inherited_output_root: str | None,
    source_experiment_name: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if not inherited_output_root:
        return updated
    candidate_spec = updated["candidates"][candidate_name]
    if bool(candidate_spec.get("eval_only", False)):
        return updated
    candidate_root = Path(inherited_output_root) / candidate_name
    if candidate_root.exists():
        candidate_spec["checkpoint_root"] = str(candidate_root)
        candidate_spec["source_experiment_name"] = source_experiment_name
    return updated


def compute_v3_1_alignment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        mild_ego = sub[sub["corruption_type"] == "mild_ego"].mean(numeric_only=True)
        vision_attention_delta = float(normal["attention_mean_vision"] - mild_vision["attention_mean_vision"])
        ego_attention_delta = float(normal["attention_mean_ego"] - mild_ego["attention_mean_ego"])
        vision_raw_delta = float(normal["raw_shapley_mean_vision"] - mild_vision["raw_shapley_mean_vision"])
        ego_raw_delta = float(normal["raw_shapley_mean_ego"] - mild_ego["raw_shapley_mean_ego"])
        row = {
            "candidate_name": candidate_name,
            "attention_shapley_kl": float(sub["attention_shapley_kl"].mean()),
            "attention_shapley_rank_corr": float(sub["attention_shapley_rank_corr"].mean()),
            "vision_raw_shapley_clean_minus_mild": vision_raw_delta,
            "ego_raw_shapley_clean_minus_mild": ego_raw_delta,
            "vision_attention_clean_minus_mild": vision_attention_delta,
            "ego_attention_clean_minus_mild": ego_attention_delta,
            "vision_attention_shapley_sign_match_flag": _raw_sign_match(vision_attention_delta, vision_raw_delta),
            "ego_attention_shapley_sign_match_flag": _raw_sign_match(ego_attention_delta, ego_raw_delta),
        }
        for name in MODALITY_NAMES:
            row[f"raw_shapley_mean_{name}"] = float(sub[f"raw_shapley_mean_{name}"].mean())
            row[f"shapley_target_mean_{name}"] = float(sub[f"shapley_target_mean_{name}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_v3_1_raw_shapley_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["candidate_name"]
    for prefix in ("raw_shapley_mean", "shapley_target_mean"):
        for name in MODALITY_NAMES:
            columns.append(f"{prefix}_{name}")
    columns.extend(
        [
            "vision_raw_shapley_clean_minus_mild",
            "ego_raw_shapley_clean_minus_mild",
            "vision_attention_clean_minus_mild",
            "ego_attention_clean_minus_mild",
            "vision_attention_shapley_sign_match_flag",
            "ego_attention_shapley_sign_match_flag",
        ]
    )
    available = [column for column in columns if column in df.columns]
    return df[available].copy()


def evaluate_v3_1_pass_fail(summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["normal_success_rate"] < 0.7:
        return "FAIL"
    if summary["normal_to_mild_success_drop"] > 0.12 and summary["degradation_score"] < 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["normal_success_rate"] >= 0.75
        and summary["normal_to_mild_success_drop"] <= 0.10
        and summary["consistency_gain_score"] > 0.0
        and (summary["vision_reliability_clean_minus_mild"] > 0.0 or summary["ego_reliability_clean_minus_mild"] > 0.0)
        and (summary["vision_raw_shapley_clean_minus_mild"] > 0.0 or summary["ego_raw_shapley_clean_minus_mild"] > 0.0)
        and summary["attention_shapley_kl_improvement"] > 0.0
        and summary["attention_shapley_rank_corr_improvement"] > 0.0
        and (
            summary["vision_attention_shapley_sign_match_flag"] > 0.0
            or summary["ego_attention_shapley_sign_match_flag"] > 0.0
        )
        and summary["saturation_warning_flag"] == 0.0
        and summary["route_distance_side_effect_flag"] == 0.0
    ):
        return "PASS"
    return "WARN"


def build_v3_1_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    baseline_alignment = alignment_df[alignment_df["candidate_name"] == baseline_candidate]
    if baseline_alignment.empty:
        raise ValueError(f"Baseline candidate missing from V3.1 alignment summary: {baseline_candidate}")
    baseline_alignment_row = baseline_alignment.iloc[0]
    final_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        align = alignment_df[alignment_df["candidate_name"] == candidate_name].iloc[0]
        summary = {
            "candidate_name": candidate_name,
            "normal_success_rate": float(normal["eval_success_rate"]),
            "normal_return": float(normal["eval_return"]),
            "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
            "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
            "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
            "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
            "consistency_loss": float(cons["consistency_loss"]),
            "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
            "consistency_gain_score": float(cons["consistency_gain_score"]),
            "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
            "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
            "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
            "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
            "saturation_warning_flag": float(tie["saturation_warning_flag"]),
            "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
            "attention_shapley_kl": float(align["attention_shapley_kl"]),
            "attention_shapley_rank_corr": float(align["attention_shapley_rank_corr"]),
            "vision_raw_shapley_clean_minus_mild": float(align["vision_raw_shapley_clean_minus_mild"]),
            "ego_raw_shapley_clean_minus_mild": float(align["ego_raw_shapley_clean_minus_mild"]),
            "vision_attention_clean_minus_mild": float(align["vision_attention_clean_minus_mild"]),
            "ego_attention_clean_minus_mild": float(align["ego_attention_clean_minus_mild"]),
            "vision_attention_shapley_sign_match_flag": float(align["vision_attention_shapley_sign_match_flag"]),
            "ego_attention_shapley_sign_match_flag": float(align["ego_attention_shapley_sign_match_flag"]),
        }
        for name in MODALITY_NAMES:
            summary[f"raw_shapley_mean_{name}"] = float(align[f"raw_shapley_mean_{name}"])
            summary[f"shapley_target_mean_{name}"] = float(align[f"shapley_target_mean_{name}"])
        summary["attention_shapley_kl_improvement"] = float(
            baseline_alignment_row["attention_shapley_kl"] - summary["attention_shapley_kl"]
        )
        summary["attention_shapley_rank_corr_improvement"] = float(
            summary["attention_shapley_rank_corr"] - baseline_alignment_row["attention_shapley_rank_corr"]
        )
        summary["degradation_score"] = -(
            2.0 * summary["normal_to_mild_success_drop"]
            + 1.0 * summary["normal_to_mild_return_drop"]
            + 1.0 * summary["normal_to_mild_progress_drop"]
        )
        summary["tie_break_score"] = (
            float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["vision_delta"]) * summary["vision_reliability_clean_minus_mild"]
            + float(tiebreak_weights["ego_delta"]) * summary["ego_reliability_clean_minus_mild"]
            + float(tiebreak_weights["vision_logit_delta"]) * summary["vision_logit_clean_minus_mild"]
            + float(tiebreak_weights["ego_logit_delta"]) * summary["ego_logit_clean_minus_mild"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["alignment_kl_improvement"]) * summary["attention_shapley_kl_improvement"]
            + float(tiebreak_weights["alignment_rank_corr_improvement"]) * summary["attention_shapley_rank_corr_improvement"]
            + float(tiebreak_weights["vision_raw_shapley_delta"]) * summary["vision_raw_shapley_clean_minus_mild"]
            + float(tiebreak_weights["ego_raw_shapley_delta"]) * summary["ego_raw_shapley_clean_minus_mild"]
            + float(tiebreak_weights["attention_sign_match"]) * (
                summary["vision_attention_shapley_sign_match_flag"] + summary["ego_attention_shapley_sign_match_flag"]
            )
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_v3_1_pass_fail(
            summary=summary,
            is_baseline=(candidate_name == baseline_candidate),
        )
        final_rows.append(summary)

    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "normal_success_rate",
            "vision_raw_shapley_clean_minus_mild",
            "ego_raw_shapley_clean_minus_mild",
            "attention_shapley_kl_improvement",
            "attention_shapley_rank_corr_improvement",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_1_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Alignment Summary",
        "",
        alignment_df.to_markdown(index=False),
        "",
        "## Raw Shapley Summary",
        "",
        raw_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_1b_targetlogit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        mild_ego = sub[sub["corruption_type"] == "mild_ego"].mean(numeric_only=True)
        vision_attention_delta = float(normal["attention_mean_vision"] - mild_vision["attention_mean_vision"])
        ego_attention_delta = float(normal["attention_mean_ego"] - mild_ego["attention_mean_ego"])
        vision_targetlogit_delta = float(normal["targetlogit_shapley_mean_vision"] - mild_vision["targetlogit_shapley_mean_vision"])
        ego_targetlogit_delta = float(normal["targetlogit_shapley_mean_ego"] - mild_ego["targetlogit_shapley_mean_ego"])
        rows.append(
            {
                "candidate_name": candidate_name,
                "vision_targetlogit_shapley_clean_minus_mild": vision_targetlogit_delta,
                "ego_targetlogit_shapley_clean_minus_mild": ego_targetlogit_delta,
                "vision_attention_targetlogit_sign_match_flag": _raw_sign_match(vision_attention_delta, vision_targetlogit_delta),
                "ego_attention_targetlogit_sign_match_flag": _raw_sign_match(ego_attention_delta, ego_targetlogit_delta),
                "targetlogit_shapley_mean_vision": float(sub["targetlogit_shapley_mean_vision"].mean()),
                "targetlogit_shapley_mean_ego": float(sub["targetlogit_shapley_mean_ego"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_v3_1c_attention_sign_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        mild_ego = sub[sub["corruption_type"] == "mild_ego"].mean(numeric_only=True)
        vision_attention_delta = float(normal["attention_mean_vision"] - mild_vision["attention_mean_vision"])
        ego_attention_delta = float(normal["attention_mean_ego"] - mild_ego["attention_mean_ego"])
        vision_targetlogit_delta = float(normal["targetlogit_shapley_mean_vision"] - mild_vision["targetlogit_shapley_mean_vision"])
        ego_targetlogit_delta = float(normal["targetlogit_shapley_mean_ego"] - mild_ego["targetlogit_shapley_mean_ego"])
        rows.append(
            {
                "candidate_name": candidate_name,
                "vision_attention_clean_minus_mild": vision_attention_delta,
                "ego_attention_clean_minus_mild": ego_attention_delta,
                "vision_targetlogit_shapley_clean_minus_mild": vision_targetlogit_delta,
                "ego_targetlogit_shapley_clean_minus_mild": ego_targetlogit_delta,
                "vision_attention_targetlogit_sign_match_flag": _raw_sign_match(vision_attention_delta, vision_targetlogit_delta),
                "ego_attention_targetlogit_sign_match_flag": _raw_sign_match(ego_attention_delta, ego_targetlogit_delta),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_1b_pass_fail(summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["normal_success_rate"] < 0.70:
        return "FAIL"
    if summary["route_distance_side_effect_flag"] > 0.0 or summary["saturation_warning_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["normal_success_rate"] >= 0.80
        and summary["normal_to_mild_success_drop"] <= 0.05
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_attention_targetlogit_sign_match_flag"] > 0.0
        and summary["consistency_gain_score"] > 0.0
        and summary["saturation_warning_flag"] == 0.0
        and summary["route_distance_side_effect_flag"] == 0.0
    ):
        return "PASS"
    return "WARN"


def build_v3_1b_targetlogit_summary(df: pd.DataFrame) -> pd.DataFrame:
    return compute_v3_1b_targetlogit_metrics(df)


def build_v3_1b_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    baseline_alignment = alignment_df[alignment_df["candidate_name"] == baseline_candidate]
    baseline_targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == baseline_candidate]
    if baseline_alignment.empty or baseline_targetlogit.empty:
        raise ValueError(f"Baseline candidate missing from V3.1b summaries: {baseline_candidate}")
    baseline_alignment_row = baseline_alignment.iloc[0]
    baseline_targetlogit_row = baseline_targetlogit.iloc[0]
    final_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        align = alignment_df[alignment_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        summary = {
            "candidate_name": candidate_name,
            "normal_success_rate": float(normal["eval_success_rate"]),
            "normal_return": float(normal["eval_return"]),
            "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
            "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
            "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
            "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
            "consistency_loss": float(cons["consistency_loss"]),
            "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
            "consistency_gain_score": float(cons["consistency_gain_score"]),
            "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
            "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
            "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
            "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
            "saturation_warning_flag": float(tie["saturation_warning_flag"]),
            "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
            "attention_shapley_kl": float(align["attention_shapley_kl"]),
            "attention_shapley_rank_corr": float(align["attention_shapley_rank_corr"]),
            "vision_raw_shapley_clean_minus_mild": float(align["vision_raw_shapley_clean_minus_mild"]),
            "ego_raw_shapley_clean_minus_mild": float(align["ego_raw_shapley_clean_minus_mild"]),
            "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
            "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
            "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
            "ego_attention_targetlogit_sign_match_flag": float(targetlogit["ego_attention_targetlogit_sign_match_flag"]),
            "targetlogit_shapley_mean_vision": float(targetlogit["targetlogit_shapley_mean_vision"]),
            "targetlogit_shapley_mean_ego": float(targetlogit["targetlogit_shapley_mean_ego"]),
        }
        summary["attention_shapley_kl_improvement"] = float(
            baseline_alignment_row["attention_shapley_kl"] - summary["attention_shapley_kl"]
        )
        summary["attention_shapley_rank_corr_improvement"] = float(
            summary["attention_shapley_rank_corr"] - baseline_alignment_row["attention_shapley_rank_corr"]
        )
        summary["vision_targetlogit_shapley_improvement"] = float(
            summary["vision_targetlogit_shapley_clean_minus_mild"]
            - baseline_targetlogit_row["vision_targetlogit_shapley_clean_minus_mild"]
        )
        summary["degradation_score"] = -(
            2.0 * summary["normal_to_mild_success_drop"]
            + 1.0 * summary["normal_to_mild_return_drop"]
            + 1.0 * summary["normal_to_mild_progress_drop"]
        )
        summary["tie_break_score"] = (
            float(tiebreak_weights["vision_targetlogit_delta"]) * summary["vision_targetlogit_shapley_clean_minus_mild"]
            + float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["vision_delta"]) * summary["vision_reliability_clean_minus_mild"]
            + float(tiebreak_weights["vision_logit_delta"]) * summary["vision_logit_clean_minus_mild"]
            + float(tiebreak_weights["alignment_kl_improvement"]) * summary["attention_shapley_kl_improvement"]
            + float(tiebreak_weights["alignment_rank_corr_improvement"]) * summary["attention_shapley_rank_corr_improvement"]
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_v3_1b_pass_fail(
            summary=summary,
            is_baseline=(candidate_name == baseline_candidate),
        )
        final_rows.append(summary)

    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_targetlogit_shapley_clean_minus_mild",
            "vision_attention_targetlogit_sign_match_flag",
            "normal_success_rate",
            "normal_to_mild_success_drop",
            "consistency_gain_score",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, True, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_1b_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Alignment Summary",
        "",
        alignment_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_v3_1c_pass_fail(summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["normal_success_rate"] < 0.70:
        return "FAIL"
    if summary["route_distance_side_effect_flag"] > 0.0 or summary["saturation_warning_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["normal_success_rate"] >= 0.80
        and summary["normal_to_mild_success_drop"] <= 0.05
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_attention_targetlogit_sign_match_flag"] > 0.0
        and summary["consistency_gain_score"] > 0.0
        and summary["saturation_warning_flag"] == 0.0
        and summary["route_distance_side_effect_flag"] == 0.0
    ):
        return "PASS"
    return "WARN"


def build_v3_1c_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    attention_sign_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    final_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        align = alignment_df[alignment_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        attn_sign = attention_sign_df[attention_sign_df["candidate_name"] == candidate_name].iloc[0]
        summary = {
            "candidate_name": candidate_name,
            "normal_success_rate": float(normal["eval_success_rate"]),
            "normal_return": float(normal["eval_return"]),
            "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
            "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
            "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
            "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
            "consistency_loss": float(cons["consistency_loss"]),
            "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
            "consistency_gain_score": float(cons["consistency_gain_score"]),
            "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
            "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
            "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
            "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
            "saturation_warning_flag": float(tie["saturation_warning_flag"]),
            "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
            "attention_shapley_kl": float(align["attention_shapley_kl"]),
            "attention_shapley_rank_corr": float(align["attention_shapley_rank_corr"]),
            "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
            "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
            "vision_attention_clean_minus_mild": float(attn_sign["vision_attention_clean_minus_mild"]),
            "ego_attention_clean_minus_mild": float(attn_sign["ego_attention_clean_minus_mild"]),
            "vision_attention_targetlogit_sign_match_flag": float(attn_sign["vision_attention_targetlogit_sign_match_flag"]),
            "ego_attention_targetlogit_sign_match_flag": float(attn_sign["ego_attention_targetlogit_sign_match_flag"]),
        }
        summary["degradation_score"] = -(
            2.0 * summary["normal_to_mild_success_drop"]
            + 1.0 * summary["normal_to_mild_return_drop"]
            + 1.0 * summary["normal_to_mild_progress_drop"]
        )
        summary["tie_break_score"] = (
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_targetlogit_delta"]) * summary["vision_targetlogit_shapley_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_v3_1c_pass_fail(
            summary=summary,
            is_baseline=(candidate_name == baseline_candidate),
        )
        final_rows.append(summary)

    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_targetlogit_shapley_clean_minus_mild",
            "normal_success_rate",
            "normal_to_mild_success_drop",
            "consistency_gain_score",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, True, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_1c_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    attention_sign_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Attention Sign Summary",
        "",
        attention_sign_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Alignment Summary",
        "",
        alignment_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_v3_2_pass_fail(summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["normal_success_rate"] < 0.75:
        return "FAIL"
    if summary["route_distance_side_effect_flag"] > 0.0 or summary["saturation_warning_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["normal_success_rate"] >= 0.85
        and summary["normal_to_mild_success_drop"] <= 0.03
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_attention_targetlogit_sign_match_flag"] > 0.0
        and summary["consistency_gain_score"] > 0.0
        and summary["saturation_warning_flag"] == 0.0
        and summary["route_distance_side_effect_flag"] == 0.0
    ):
        return "PASS"
    return "WARN"


def build_v3_2_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    attention_sign_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    final_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        align = alignment_df[alignment_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        attn_sign = attention_sign_df[attention_sign_df["candidate_name"] == candidate_name].iloc[0]
        summary = {
            "candidate_name": candidate_name,
            "normal_success_rate": float(normal["eval_success_rate"]),
            "normal_return": float(normal["eval_return"]),
            "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
            "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
            "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
            "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
            "consistency_loss": float(cons["consistency_loss"]),
            "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
            "consistency_gain_score": float(cons["consistency_gain_score"]),
            "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
            "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
            "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
            "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
            "saturation_warning_flag": float(tie["saturation_warning_flag"]),
            "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
            "attention_shapley_kl": float(align["attention_shapley_kl"]),
            "attention_shapley_rank_corr": float(align["attention_shapley_rank_corr"]),
            "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
            "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
            "vision_attention_clean_minus_mild": float(attn_sign["vision_attention_clean_minus_mild"]),
            "ego_attention_clean_minus_mild": float(attn_sign["ego_attention_clean_minus_mild"]),
            "vision_attention_targetlogit_sign_match_flag": float(attn_sign["vision_attention_targetlogit_sign_match_flag"]),
            "ego_attention_targetlogit_sign_match_flag": float(attn_sign["ego_attention_targetlogit_sign_match_flag"]),
        }
        summary["degradation_score"] = -(
            2.0 * summary["normal_to_mild_success_drop"]
            + 1.0 * summary["normal_to_mild_return_drop"]
            + 1.0 * summary["normal_to_mild_progress_drop"]
        )
        summary["tie_break_score"] = (
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_targetlogit_delta"]) * summary["vision_targetlogit_shapley_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_v3_2_pass_fail(
            summary=summary,
            is_baseline=(candidate_name == baseline_candidate),
        )
        final_rows.append(summary)

    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_targetlogit_shapley_clean_minus_mild",
            "normal_success_rate",
            "normal_to_mild_success_drop",
            "consistency_gain_score",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, True, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_2_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    attention_sign_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Attention Sign Summary",
        "",
        attention_sign_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Alignment Summary",
        "",
        alignment_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Stage33AttentionScoreTrainer(Stage31bShapleyTrainer):
    """V3.3 trainer for attention score-only calibration."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        self.model.freeze_attention_score_calibration_only()
        self.model.fusion_module.score_adapter_residual_scale = float(config["losses"].get("score_adapter_residual_scale", 0.0))
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def _sample_attention_score_positions(
        self,
        valid_mask: torch.Tensor,
        corrupted_flag: torch.Tensor,
        target_mask: torch.Tensor,
        max_states: int,
    ) -> list[tuple[int, int]]:
        eligible = valid_mask > 0.0
        single_target = target_mask.sum(dim=-1) == 1
        mild_vision_mask = eligible & (corrupted_flag.squeeze(-1) > 0.0) & single_target & (target_mask[..., 0] > 0.0)
        positions = torch.nonzero(mild_vision_mask, as_tuple=False)
        if positions.numel() == 0:
            return []
        selected = positions[:max_states]
        return [(int(item[0].item()), int(item[1].item())) for item in selected]

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_attention_score_margin": 0.0,
            "loss/vision_attention_weight_margin": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        weight_coef = float(losses_cfg.get("vision_attention_weight_margin_loss_coef", 0.0))
        score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        weight_margin = float(losses_cfg.get("vision_attention_weight_margin", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                score_terms: list[torch.Tensor] = []
                weight_terms: list[torch.Tensor] = []
                for batch_idx, time_idx in positions:
                    clean_scores = clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = outputs["attention_scores"][batch_idx, time_idx]
                    clean_attention = clean_outputs["attention_weights"][batch_idx, time_idx]
                    corrupt_attention = outputs["attention_weights"][batch_idx, time_idx]

                    clean_rel_vision = clean_scores[0] - clean_scores[1:].mean()
                    corrupt_rel_vision = corrupt_scores[0] - corrupt_scores[1:].mean()
                    score_terms.append(
                        F.relu(torch.as_tensor(score_margin, device=self.device) - (clean_rel_vision - corrupt_rel_vision))
                    )
                    weight_terms.append(
                        F.relu(
                            torch.as_tensor(weight_margin, device=self.device)
                            - (clean_attention[0] - corrupt_attention[0])
                        )
                    )

                score_loss = torch.stack(score_terms).mean() if score_terms else torch.zeros((), device=self.device)
                weight_loss = torch.stack(weight_terms).mean() if weight_terms else torch.zeros((), device=self.device)
                total_loss = score_coef * score_loss + weight_coef * weight_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_attention_score_margin"] += float(score_loss.item())
                metrics["loss/vision_attention_weight_margin"] += float(weight_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    @torch.no_grad()
    def run_eval_suite(self, num_eval_episodes: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        env = self.eval_env
        eval_probes = self.raw_config["eval_probes"]
        temperature = float(self.raw_config["losses"].get("shapley_temperature", 0.25))

        for probe in eval_probes:
            probe_name = str(probe["name"])
            returns = []
            successes = []
            collisions = []
            timeouts = []
            progress_values = []
            attention_values = []
            attention_score_values = []
            adapter_delta_norm_values = []
            relative_gap_values = []
            relative_gap_norm_values = []
            bias_values = []
            bias_norm_values = []
            bias_abs_values = []
            gate_values = []
            source_gate_values = []
            gate_residual_values = []
            gate_residual_logit_values = []
            gate_adapter_logit_values = []
            gate_delta_logit_values = []
            gate_scaled_delta_logit_values = []
            gate_threshold_shift_logit_values = []
            gate_post_shift_logit_values = []
            gate_post_shift_excess_values = []
            gate_common_cancel_gate_values = []
            gate_common_cancel_logit_values = []
            gate_post_cancel_logit_values = []
            gate_contrast_shift_logit_values = []
            gate_mild_rescue_gate_values = []
            gate_selector_active_values = []
            gate_mild_rescue_value_values = []
            gate_mild_rescue_logit_values = []
            adapter_gate_hidden_norm_values = []
            gate_reconstruction_error_values = []
            gate_norm_values = []
            residual_alpha_values = []
            adapter_gate_alpha_values = []
            gate_logit_offset_values = []
            gate_output_scale_values = []
            gate_threshold_shift_alpha_values = []
            gate_source_logit_bias_values = []
            gate_common_cancel_alpha_values = []
            gate_contrast_shift_alpha_values = []
            gate_mild_rescue_alpha_values = []
            gate_cancel_temperature_values = []
            gate_selector_temperature_values = []
            gate_rescue_selector_floor_values = []
            gate_mild_gate_min_values = []
            reliability_values = []
            vision_logit_values = []
            ego_logit_values = []
            action_diffs = []
            reference_vision_logit_values = []
            reference_ego_logit_values = []
            reference_action_diffs = []
            clean_score_anchor_errors = []
            clean_action_anchor_errors = []
            source_sign_score_anchor_errors = []
            source_clean_gate_preserve_errors = []
            clean_gate_input_values = []
            corrupt_gate_input_values = []
            gate_input_feature_slices: dict[str, tuple[int, int]] | None = None
            raw_shapley_values = []
            shapley_target_values = []
            shapley_kls = []
            shapley_rank_corrs = []
            vision_targetlogit_values = []
            ego_targetlogit_values = []
            losses_cfg = self.raw_config["losses"]
            rng = make_probe_rng(self.seed, 49000, probe_name)

            for episode_idx in range(num_eval_episodes):
                obs, info = env.reset(seed=self.seed * 100 + episode_idx)
                del info
                hidden = self.model.initial_hidden(batch_size=1, device=self.device)
                clean_hidden = self.model.initial_hidden(batch_size=1, device=self.device)
                teacher_hidden = None
                sign_teacher_hidden_clean = None
                sign_teacher_hidden_corrupt = None
                clean_teacher_hidden = None
                dual_teacher_mode = hasattr(self, "sign_teacher_model") and hasattr(self, "clean_teacher_model")
                needs_source_gate_context = (
                    str(getattr(self.model.fusion_module, "gate_mode", "absolute_v1"))
                    in {"source_residual_v1", "source_residual_logit_v2"}
                )
                if hasattr(self, "teacher_model") and getattr(self, "teacher_model") is not None:
                    teacher_hidden = self.teacher_model.initial_hidden(batch_size=1, device=self.device)
                if dual_teacher_mode:
                    sign_teacher_hidden_clean = self.sign_teacher_model.initial_hidden(batch_size=1, device=self.device)
                    sign_teacher_hidden_corrupt = self.sign_teacher_model.initial_hidden(batch_size=1, device=self.device)
                    clean_teacher_hidden = self.clean_teacher_model.initial_hidden(batch_size=1, device=self.device)
                episode_start = torch.ones(1, device=self.device)
                clean_episode_start = torch.ones(1, device=self.device)
                teacher_episode_start = torch.ones(1, device=self.device) if teacher_hidden is not None else None
                sign_teacher_episode_start_clean = (
                    torch.ones(1, device=self.device) if sign_teacher_hidden_clean is not None else None
                )
                sign_teacher_episode_start_corrupt = (
                    torch.ones(1, device=self.device) if sign_teacher_hidden_corrupt is not None else None
                )
                clean_teacher_episode_start = (
                    torch.ones(1, device=self.device) if clean_teacher_hidden is not None else None
                )
                episode_return = 0.0
                episode_progress = []
                while True:
                    clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
                    if probe_name == "normal_eval":
                        policy_obs = clean_obs
                    elif probe["kind"] == "single":
                        policy_obs, _severity = self._apply_single_probe(obs, probe, rng, probe_name)
                    else:
                        policy_obs, _severity = self._apply_mixed_probe(obs, probe, rng, probe_name)

                    clean_obs_t = {
                        key: torch.as_tensor(clean_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }
                    corrupt_obs_t = {
                        key: torch.as_tensor(policy_obs[key], dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key in MODALITY_NAMES
                    }

                    reference_clean_output = self._reference_step(
                        obs_dict=clean_obs_t,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    clean_path_output = self._reference_step(
                        obs_dict=clean_obs_t,
                        hidden_state=clean_hidden,
                        episode_start=clean_episode_start,
                        deterministic=True,
                    )
                    teacher_clean_output = None
                    sign_teacher_clean_output = None
                    sign_teacher_corrupt_output = None
                    if teacher_hidden is not None and teacher_episode_start is not None:
                        teacher_clean_output = self.teacher_model.step(
                            obs_dict=clean_obs_t,
                            hidden_state=teacher_hidden,
                            episode_start=teacher_episode_start,
                            deterministic=True,
                        )
                    if dual_teacher_mode and sign_teacher_episode_start_clean is not None and sign_teacher_episode_start_corrupt is not None and clean_teacher_episode_start is not None:
                        sign_teacher_clean_output = self.sign_teacher_model.step(
                            obs_dict=clean_obs_t,
                            hidden_state=sign_teacher_hidden_clean,
                            episode_start=sign_teacher_episode_start_clean,
                            deterministic=True,
                        )
                        sign_teacher_corrupt_output = self.sign_teacher_model.step(
                            obs_dict=corrupt_obs_t,
                            hidden_state=sign_teacher_hidden_corrupt,
                            episode_start=sign_teacher_episode_start_corrupt,
                            deterministic=True,
                        )
                        teacher_clean_output = self.clean_teacher_model.step(
                            obs_dict=clean_obs_t,
                            hidden_state=clean_teacher_hidden,
                            episode_start=clean_teacher_episode_start,
                            deterministic=True,
                        )
                        clean_path_output = self.model.step(
                            obs_dict=clean_obs_t,
                            hidden_state=clean_hidden,
                            episode_start=clean_episode_start,
                            deterministic=True,
                            gate_feature_context=build_gate_feature_context(
                                clean_teacher_outputs=teacher_clean_output.aux_outputs,
                                sign_teacher_outputs=sign_teacher_clean_output.aux_outputs,
                            ),
                        )
                        corrupt_output = self.model.step(
                            obs_dict=corrupt_obs_t,
                            hidden_state=hidden,
                            episode_start=episode_start,
                            deterministic=True,
                            gate_feature_context=build_gate_feature_context(
                                clean_teacher_outputs=teacher_clean_output.aux_outputs,
                                sign_teacher_outputs=sign_teacher_corrupt_output.aux_outputs,
                            ),
                        )
                    else:
                        if needs_source_gate_context:
                            raise ValueError("source_residual_v1 evaluation requires dual teacher source gate context")
                        clean_path_output = self.model.step(
                            obs_dict=clean_obs_t,
                            hidden_state=clean_hidden,
                            episode_start=clean_episode_start,
                            deterministic=True,
                        )
                        corrupt_output = self.model.step(
                            obs_dict=corrupt_obs_t,
                            hidden_state=hidden,
                            episode_start=episode_start,
                            deterministic=True,
                        )
                    raw_shapley, shapley_target = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        clean_teacher_model=self.clean_teacher_model if dual_teacher_mode else None,
                        sign_teacher_model=self.sign_teacher_model if dual_teacher_mode else None,
                        **build_shapley_compute_kwargs(
                            losses_cfg,
                            return_raw=True,
                            rng_seed=self.seed * 100000 + episode_idx,
                            utility_mode="clean_fused_l2",
                        ),
                    )
                    vision_targetlogit_shapley, _ = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        clean_teacher_model=self.clean_teacher_model if dual_teacher_mode else None,
                        sign_teacher_model=self.sign_teacher_model if dual_teacher_mode else None,
                        **build_shapley_compute_kwargs(
                            losses_cfg,
                            return_raw=True,
                            rng_seed=self.seed * 100000 + episode_idx + 1,
                            utility_mode="target_logit_clean_l1",
                            target_modality_index=0,
                        ),
                    )
                    ego_targetlogit_shapley, _ = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        clean_teacher_model=self.clean_teacher_model if dual_teacher_mode else None,
                        sign_teacher_model=self.sign_teacher_model if dual_teacher_mode else None,
                        **build_shapley_compute_kwargs(
                            losses_cfg,
                            return_raw=True,
                            rng_seed=self.seed * 100000 + episode_idx + 2,
                            utility_mode="target_logit_clean_l1",
                            target_modality_index=2,
                        ),
                    )
                    attention = corrupt_output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    attention_scores = corrupt_output.aux_outputs["attention_scores"].squeeze(0).detach().cpu().numpy()
                    adapter_delta_norms = (
                        corrupt_output.aux_outputs["attention_score_adapter_delta_norms"].squeeze(0).detach().cpu().numpy()
                    )
                    relative_gap_scores = (
                        corrupt_output.aux_outputs["attention_relative_gap_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    relative_gap_norm = float(
                        corrupt_output.aux_outputs["attention_relative_gap_norm"].squeeze(0).detach().cpu().item()
                    )
                    bias_scores = (
                        corrupt_output.aux_outputs["attention_bias_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    gate_scores = (
                        corrupt_output.aux_outputs["attention_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    source_gate_scores = (
                        corrupt_output.aux_outputs["attention_source_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    gate_residual_scores = (
                        corrupt_output.aux_outputs["attention_bias_gate_residuals"].squeeze(0).detach().cpu().numpy()
                    )
                    gate_residual_logits = (
                        corrupt_output.aux_outputs["attention_bias_gate_residual_logits"].squeeze(0).detach().cpu().numpy()
                    )
                    gate_adapter_logits = (
                        corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"].squeeze(0).detach().cpu().numpy()
                    )
                    gate_delta_logits = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_delta_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_scaled_delta_logits = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_scaled_delta_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_threshold_shift_logits = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_threshold_shift_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_post_shift_logits = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_post_shift_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_post_shift_excess = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_post_shift_excess",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_common_cancel_gates = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_common_cancel_gates",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    adapter_gate_hidden_norms = (
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_adapter_hidden_norms",
                            torch.zeros_like(
                                corrupt_output.aux_outputs["attention_bias_gate_adapter_logits"],
                            ),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_reconstruction_error = float(
                        corrupt_output.aux_outputs["attention_gate_reconstruction_error"].squeeze(0).detach().cpu().item()
                    )
                    residual_alpha = float(corrupt_output.aux_outputs["attention_residual_alpha"].squeeze(0).detach().cpu().item())
                    adapter_gate_alpha = float(
                        corrupt_output.aux_outputs.get("attention_adapter_gate_alpha", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_logit_offset = float(
                        corrupt_output.aux_outputs.get("attention_gate_logit_offset", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_output_scale = float(
                        corrupt_output.aux_outputs.get("attention_gate_output_scale", torch.ones(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_threshold_shift_alpha = float(
                        corrupt_output.aux_outputs.get("attention_gate_threshold_shift_alpha", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_source_logit_bias = float(
                        corrupt_output.aux_outputs.get("attention_gate_source_logit_bias", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_common_cancel_alpha = float(
                        corrupt_output.aux_outputs.get("attention_gate_common_cancel_alpha", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_contrast_shift_alpha = float(
                        corrupt_output.aux_outputs.get("attention_gate_contrast_shift_alpha", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_mild_rescue_alpha = float(
                        corrupt_output.aux_outputs.get("attention_gate_mild_rescue_alpha", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_cancel_temperature = float(
                        corrupt_output.aux_outputs.get("attention_gate_cancel_temperature", torch.ones(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_selector_temperature = float(
                        corrupt_output.aux_outputs.get("attention_gate_selector_temperature", torch.ones(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_rescue_selector_floor = float(
                        corrupt_output.aux_outputs.get("attention_gate_rescue_selector_floor", torch.zeros(1, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    gate_mild_gate_min = float(
                        corrupt_output.aux_outputs.get("attention_gate_mild_gate_min", torch.full((1,), 0.08, device=self.device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    bias_norm = float(
                        corrupt_output.aux_outputs["attention_bias_norm"].squeeze(0).detach().cpu().item()
                    )
                    gate_norm = float(
                        corrupt_output.aux_outputs["attention_bias_gate_norm"].squeeze(0).detach().cpu().item()
                    )
                    raw_shapley_values.append(raw_shapley)
                    shapley_target_values.append(shapley_target)
                    vision_targetlogit_values.append(float(vision_targetlogit_shapley[0]))
                    ego_targetlogit_values.append(float(ego_targetlogit_shapley[2]))
                    shapley_kls.append(
                        float(np.sum(attention * (np.log(np.clip(attention, 1e-8, None)) - np.log(np.clip(shapley_target, 1e-8, None)))))
                    )
                    shapley_rank_corrs.append(_spearman_rank_corr(attention.astype(np.float64), raw_shapley.astype(np.float64)))

                    action = corrupt_output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    reliability = corrupt_output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    logits = corrupt_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    current_clean_logits = clean_path_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    reference_clean_logits = reference_clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    attention_score_values.append(attention_scores)
                    adapter_delta_norm_values.append(adapter_delta_norms)
                    relative_gap_values.append(relative_gap_scores)
                    relative_gap_norm_values.append(relative_gap_norm)
                    bias_values.append(bias_scores)
                    bias_norm_values.append(bias_norm)
                    bias_abs_values.append(np.abs(bias_scores))
                    gate_values.append(gate_scores)
                    source_gate_values.append(source_gate_scores)
                    gate_residual_values.append(gate_residual_scores)
                    gate_residual_logit_values.append(gate_residual_logits)
                    gate_adapter_logit_values.append(gate_adapter_logits)
                    gate_delta_logit_values.append(gate_delta_logits)
                    gate_scaled_delta_logit_values.append(gate_scaled_delta_logits)
                    gate_threshold_shift_logit_values.append(gate_threshold_shift_logits)
                    gate_post_shift_logit_values.append(gate_post_shift_logits)
                    gate_post_shift_excess_values.append(gate_post_shift_excess)
                    gate_common_cancel_gate_values.append(gate_common_cancel_gates)
                    gate_common_cancel_logit_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_common_cancel_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_post_cancel_logit_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_post_cancel_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_contrast_shift_logit_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_contrast_shift_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_mild_rescue_gate_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_mild_rescue_gates",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_selector_active_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_selector_actives",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_mild_rescue_value_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_mild_rescue_values",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gate_mild_rescue_logit_values.append(
                        corrupt_output.aux_outputs.get(
                            "attention_bias_gate_mild_rescue_logits",
                            torch.zeros_like(corrupt_output.aux_outputs["attention_bias_gates"]),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    adapter_gate_hidden_norm_values.append(adapter_gate_hidden_norms)
                    gate_reconstruction_error_values.append(gate_reconstruction_error)
                    gate_norm_values.append(gate_norm)
                    residual_alpha_values.append(residual_alpha)
                    adapter_gate_alpha_values.append(adapter_gate_alpha)
                    gate_logit_offset_values.append(gate_logit_offset)
                    gate_output_scale_values.append(gate_output_scale)
                    gate_threshold_shift_alpha_values.append(gate_threshold_shift_alpha)
                    gate_source_logit_bias_values.append(gate_source_logit_bias)
                    gate_common_cancel_alpha_values.append(gate_common_cancel_alpha)
                    gate_contrast_shift_alpha_values.append(gate_contrast_shift_alpha)
                    gate_mild_rescue_alpha_values.append(gate_mild_rescue_alpha)
                    gate_cancel_temperature_values.append(gate_cancel_temperature)
                    gate_selector_temperature_values.append(gate_selector_temperature)
                    gate_rescue_selector_floor_values.append(gate_rescue_selector_floor)
                    gate_mild_gate_min_values.append(gate_mild_gate_min)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(current_clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(current_clean_logits[2]), float(logits[2])))
                    reference_vision_logit_values.append(float(reference_clean_logits[0]))
                    reference_ego_logit_values.append(float(reference_clean_logits[2]))
                    action_diffs.append(float(torch.mean(torch.abs(clean_path_output.action - corrupt_output.action)).cpu().item()))
                    reference_action_diffs.append(
                        float(torch.mean(torch.abs(reference_clean_output.action - corrupt_output.action)).cpu().item())
                    )
                    if "attention_bias_gate_inputs" in clean_path_output.aux_outputs and "attention_bias_gate_inputs" in corrupt_output.aux_outputs:
                        clean_gate_inputs = clean_path_output.aux_outputs["attention_bias_gate_inputs"].squeeze(0).detach().cpu().numpy()
                        corrupt_gate_inputs = corrupt_output.aux_outputs["attention_bias_gate_inputs"].squeeze(0).detach().cpu().numpy()
                        clean_gate_input_values.append(clean_gate_inputs[0])
                        corrupt_gate_input_values.append(corrupt_gate_inputs[0])
                        gate_input_feature_slices = dict(
                            corrupt_output.aux_outputs.get("attention_bias_gate_input_slices", gate_input_feature_slices or {})
                        )
                    if teacher_clean_output is not None:
                        clean_score_anchor_errors.append(
                            float(
                                torch.mean(
                                    (clean_path_output.aux_outputs["attention_scores"] - teacher_clean_output.aux_outputs["attention_scores"]).pow(2)
                                ).cpu().item()
                            )
                        )
                        clean_action_anchor_errors.append(
                            float(
                                torch.mean(
                                    (clean_path_output.aux_outputs["action_mean"] - teacher_clean_output.aux_outputs["action_mean"]).pow(2)
                                ).cpu().item()
                            )
                        )
                    if sign_teacher_corrupt_output is not None:
                        source_sign_score_anchor_errors.append(
                            float(
                                torch.mean(
                                    (
                                        corrupt_output.aux_outputs["attention_scores"]
                                        - sign_teacher_corrupt_output.aux_outputs["attention_scores"]
                                    ).pow(2)
                                ).cpu().item()
                            )
                        )
                    if sign_teacher_clean_output is not None:
                        source_clean_gate_preserve_errors.append(
                            float(
                                torch.mean(
                                    (
                                        clean_path_output.aux_outputs["attention_bias_gates"]
                                        - sign_teacher_clean_output.aux_outputs["attention_bias_gates"]
                                    ).pow(2)
                                ).cpu().item()
                            )
                        )
                    hidden = corrupt_output.hidden_state
                    clean_hidden = clean_path_output.hidden_state
                    if teacher_clean_output is not None and hasattr(self, "teacher_model"):
                        teacher_hidden = teacher_clean_output.hidden_state
                    if sign_teacher_clean_output is not None:
                        sign_teacher_hidden_clean = sign_teacher_clean_output.hidden_state
                    if sign_teacher_corrupt_output is not None:
                        sign_teacher_hidden_corrupt = sign_teacher_corrupt_output.hidden_state
                    if dual_teacher_mode and teacher_clean_output is not None:
                        clean_teacher_hidden = teacher_clean_output.hidden_state
                    episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                    hidden = self.model.reset_hidden(hidden, episode_start)
                    clean_episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                    clean_hidden = self.model.reset_hidden(clean_hidden, clean_episode_start)
                    if teacher_hidden is not None:
                        teacher_episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                        teacher_hidden = self.teacher_model.reset_hidden(teacher_hidden, teacher_episode_start)
                    if sign_teacher_hidden_clean is not None:
                        sign_teacher_episode_start_clean = torch.tensor([float(terminated or truncated)], device=self.device)
                        sign_teacher_hidden_clean = self.sign_teacher_model.reset_hidden(
                            sign_teacher_hidden_clean, sign_teacher_episode_start_clean
                        )
                    if sign_teacher_hidden_corrupt is not None:
                        sign_teacher_episode_start_corrupt = torch.tensor([float(terminated or truncated)], device=self.device)
                        sign_teacher_hidden_corrupt = self.sign_teacher_model.reset_hidden(
                            sign_teacher_hidden_corrupt, sign_teacher_episode_start_corrupt
                        )
                    if clean_teacher_hidden is not None:
                        clean_teacher_episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                        clean_teacher_hidden = self.clean_teacher_model.reset_hidden(
                            clean_teacher_hidden, clean_teacher_episode_start
                        )
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(float(info["goal_reached"]))
                        collisions.append(float(info["collision"]))
                        timeouts.append(float(info["timeout"]))
                        progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                        break

            att = np.stack(attention_values, axis=0)
            att_scores = np.stack(attention_score_values, axis=0)
            adapter_delta_arr = np.stack(adapter_delta_norm_values, axis=0)
            relative_gap_arr = np.stack(relative_gap_values, axis=0)
            relative_gap_norm_arr = np.asarray(relative_gap_norm_values, dtype=np.float32)
            bias_arr = np.stack(bias_values, axis=0)
            bias_norm_arr = np.asarray(bias_norm_values, dtype=np.float32)
            bias_abs_arr = np.stack(bias_abs_values, axis=0)
            gate_arr = np.stack(gate_values, axis=0)
            source_gate_arr = np.stack(source_gate_values, axis=0)
            gate_residual_arr = np.stack(gate_residual_values, axis=0)
            gate_residual_logit_arr = np.stack(gate_residual_logit_values, axis=0)
            gate_adapter_logit_arr = np.stack(gate_adapter_logit_values, axis=0)
            gate_delta_logit_arr = np.stack(gate_delta_logit_values, axis=0)
            gate_scaled_delta_logit_arr = np.stack(gate_scaled_delta_logit_values, axis=0)
            gate_threshold_shift_logit_arr = np.stack(gate_threshold_shift_logit_values, axis=0)
            gate_post_shift_logit_arr = np.stack(gate_post_shift_logit_values, axis=0)
            gate_post_shift_excess_arr = np.stack(gate_post_shift_excess_values, axis=0)
            gate_common_cancel_gate_arr = np.stack(gate_common_cancel_gate_values, axis=0)
            gate_common_cancel_logit_arr = np.stack(gate_common_cancel_logit_values, axis=0)
            gate_post_cancel_logit_arr = np.stack(gate_post_cancel_logit_values, axis=0)
            gate_contrast_shift_logit_arr = np.stack(gate_contrast_shift_logit_values, axis=0)
            gate_mild_rescue_gate_arr = np.stack(gate_mild_rescue_gate_values, axis=0)
            gate_selector_active_arr = np.stack(gate_selector_active_values, axis=0)
            gate_mild_rescue_value_arr = np.stack(gate_mild_rescue_value_values, axis=0)
            gate_mild_rescue_logit_arr = np.stack(gate_mild_rescue_logit_values, axis=0)
            adapter_gate_hidden_norm_arr = np.stack(adapter_gate_hidden_norm_values, axis=0)
            gate_reconstruction_error_arr = np.asarray(gate_reconstruction_error_values, dtype=np.float32)
            gate_norm_arr = np.asarray(gate_norm_values, dtype=np.float32)
            rel = np.stack(reliability_values, axis=0)
            raw_shapley_arr = np.stack(raw_shapley_values, axis=0)
            shapley_target_arr = np.stack(shapley_target_values, axis=0)
            vision_targetlogit_arr = np.asarray(vision_targetlogit_values, dtype=np.float32)
            ego_targetlogit_arr = np.asarray(ego_targetlogit_values, dtype=np.float32)
            vision_logit_arr = np.asarray(vision_logit_values, dtype=np.float32)
            ego_logit_arr = np.asarray(ego_logit_values, dtype=np.float32)
            reference_vision_logit_arr = np.asarray(reference_vision_logit_values, dtype=np.float32)
            reference_ego_logit_arr = np.asarray(reference_ego_logit_values, dtype=np.float32)
            residual_alpha_arr = np.asarray(residual_alpha_values, dtype=np.float32)
            adapter_gate_alpha_arr = np.asarray(adapter_gate_alpha_values, dtype=np.float32)
            gate_logit_offset_arr = np.asarray(gate_logit_offset_values, dtype=np.float32)
            gate_output_scale_arr = np.asarray(gate_output_scale_values, dtype=np.float32)
            gate_threshold_shift_alpha_arr = np.asarray(gate_threshold_shift_alpha_values, dtype=np.float32)
            gate_source_logit_bias_arr = np.asarray(gate_source_logit_bias_values, dtype=np.float32)
            gate_common_cancel_alpha_arr = np.asarray(gate_common_cancel_alpha_values, dtype=np.float32)
            gate_contrast_shift_alpha_arr = np.asarray(gate_contrast_shift_alpha_values, dtype=np.float32)
            gate_mild_rescue_alpha_arr = np.asarray(gate_mild_rescue_alpha_values, dtype=np.float32)
            gate_cancel_temperature_arr = np.asarray(gate_cancel_temperature_values, dtype=np.float32)
            gate_selector_temperature_arr = np.asarray(gate_selector_temperature_values, dtype=np.float32)
            gate_rescue_selector_floor_arr = np.asarray(gate_rescue_selector_floor_values, dtype=np.float32)
            gate_mild_gate_min_arr = np.asarray(gate_mild_gate_min_values, dtype=np.float32)
            feature_audit = (
                compute_gate_feature_separability_audit(
                    np.stack(clean_gate_input_values, axis=0),
                    np.stack(corrupt_gate_input_values, axis=0),
                    gate_input_feature_slices,
                )
                if clean_gate_input_values and corrupt_gate_input_values and "vision" in probe_name
                else {
                    "clean_vs_mild_gate_feature_auc": 0.5,
                    "clean_vs_mild_gate_feature_delta_norm": 0.0,
                    "source_gate_feature_onoff_auc_delta": 0.0,
                    "teacher_delta_feature_onoff_auc_delta": 0.0,
                }
            )
            row = {
                "candidate_name": self.candidate_name,
                "seed": self.seed,
                "probe_name": probe_name,
                "env_mode": "normal" if probe_name == "normal_eval" else "mild",
                "corruption_type": probe_name,
                "train_step": int(self.raw_config["train"]["total_steps"]),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "consistency_loss": float(np.mean(np.square(action_diffs))),
                "consistency_gap_mean": float(np.mean(action_diffs)),
                "same_state_action_diff_clean_vs_corrupt": float(np.mean(action_diffs)),
                "reference_same_state_action_diff_clean_vs_corrupt": float(np.mean(reference_action_diffs)),
                "consistency_gain_score": 0.0,
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(
                    att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)
                ),
                "logit_mean_vision_clean": float(vision_logit_arr[:, 0].mean()),
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "reference_logit_mean_vision_clean": float(reference_vision_logit_arr.mean()),
                "reference_logit_mean_ego_clean": float(reference_ego_logit_arr.mean()),
                "current_clean_logit_mean_vision": float(vision_logit_arr[:, 0].mean()),
                "current_clean_logit_mean_ego": float(ego_logit_arr[:, 0].mean()),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
                "targetlogit_shapley_mean_vision": float(vision_targetlogit_arr.mean()),
                "targetlogit_shapley_mean_ego": float(ego_targetlogit_arr.mean()),
                "attention_bias_gates": gate_arr.tolist(),
                "attention_source_bias_gates": source_gate_arr.tolist(),
                "attention_bias_gate_residuals": gate_residual_arr.tolist(),
                "attention_bias_gate_residual_logits": gate_residual_logit_arr.tolist(),
                "attention_gate_reconstruction_errors": gate_reconstruction_error_arr.tolist(),
                "gate_reconstruction_error_mean": float(gate_reconstruction_error_arr.mean()) if gate_reconstruction_error_arr.size else 0.0,
                "residual_alpha_mean": float(residual_alpha_arr.mean()) if residual_alpha_arr.size else 0.0,
                "adapter_gate_alpha_mean": float(adapter_gate_alpha_arr.mean()) if adapter_gate_alpha_arr.size else 0.0,
                "gate_logit_offset_mean": float(gate_logit_offset_arr.mean()) if gate_logit_offset_arr.size else 0.0,
                "gate_output_scale_mean": float(gate_output_scale_arr.mean()) if gate_output_scale_arr.size else 1.0,
                "gate_threshold_shift_alpha_mean": float(gate_threshold_shift_alpha_arr.mean()) if gate_threshold_shift_alpha_arr.size else 0.0,
                "gate_source_logit_bias_mean": float(gate_source_logit_bias_arr.mean()) if gate_source_logit_bias_arr.size else 0.0,
                "gate_common_cancel_alpha_mean": float(gate_common_cancel_alpha_arr.mean()) if gate_common_cancel_alpha_arr.size else 0.0,
                "gate_contrast_shift_alpha_mean": float(gate_contrast_shift_alpha_arr.mean()) if gate_contrast_shift_alpha_arr.size else 0.0,
                "gate_mild_rescue_alpha_mean": float(gate_mild_rescue_alpha_arr.mean()) if gate_mild_rescue_alpha_arr.size else 0.0,
                "attention_gate_cancel_temperature": float(gate_cancel_temperature_arr.mean()) if gate_cancel_temperature_arr.size else 1.0,
                "attention_gate_selector_temperature": float(gate_selector_temperature_arr.mean()) if gate_selector_temperature_arr.size else 1.0,
                "attention_gate_rescue_selector_floor": float(gate_rescue_selector_floor_arr.mean()) if gate_rescue_selector_floor_arr.size else 0.0,
                "attention_gate_mild_gate_min": float(gate_mild_gate_min_arr.mean()) if gate_mild_gate_min_arr.size else 0.08,
            }
            row.update(_probe_logging_fields(probe_name, probe))
            row.update(feature_audit)
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"attention_score_mean_{name}"] = float(att_scores[:, idx].mean())
                row[f"attention_score_std_{name}"] = float(att_scores[:, idx].std())
                row[f"attention_score_adapter_delta_norm_mean_{name}"] = float(adapter_delta_arr[:, idx].mean())
                row[f"attention_bias_mean_{name}"] = float(bias_arr[:, idx].mean())
                row[f"attention_source_bias_gate_mean_{name}"] = float(source_gate_arr[:, idx].mean())
                row[f"attention_bias_gate_residual_mean_{name}"] = float(gate_residual_arr[:, idx].mean())
                row[f"attention_bias_gate_residual_logit_mean_{name}"] = float(gate_residual_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_adapter_logit_mean_{name}"] = float(gate_adapter_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_delta_logit_mean_{name}"] = float(gate_delta_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_scaled_delta_logit_mean_{name}"] = float(gate_scaled_delta_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_threshold_shift_logit_mean_{name}"] = float(gate_threshold_shift_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_post_shift_logit_mean_{name}"] = float(gate_post_shift_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_post_shift_excess_mean_{name}"] = float(gate_post_shift_excess_arr[:, idx].mean())
                row[f"attention_bias_gate_common_cancel_gate_mean_{name}"] = float(gate_common_cancel_gate_arr[:, idx].mean())
                row[f"attention_bias_gate_common_cancel_logit_mean_{name}"] = float(gate_common_cancel_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_post_cancel_logit_mean_{name}"] = float(gate_post_cancel_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_contrast_shift_logit_mean_{name}"] = float(gate_contrast_shift_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_mild_rescue_gate_mean_{name}"] = float(gate_mild_rescue_gate_arr[:, idx].mean())
                row[f"attention_bias_gate_selector_prob_mean_{name}"] = float(gate_mild_rescue_gate_arr[:, idx].mean())
                row[f"attention_bias_gate_selector_active_mean_{name}"] = float(gate_selector_active_arr[:, idx].mean())
                row[f"attention_bias_gate_mild_rescue_value_mean_{name}"] = float(gate_mild_rescue_value_arr[:, idx].mean())
                row[f"attention_bias_gate_mild_rescue_logit_mean_{name}"] = float(gate_mild_rescue_logit_arr[:, idx].mean())
                row[f"attention_bias_gate_adapter_hidden_norm_mean_{name}"] = float(
                    adapter_gate_hidden_norm_arr[:, idx].mean()
                )
                row[f"attention_bias_gate_mean_{name}"] = float(gate_arr[:, idx].mean())
                row[f"attention_gate_output_scale_mean_{name}"] = float(gate_output_scale_arr.mean()) if gate_output_scale_arr.size else 1.0
                row[f"attention_gate_threshold_shift_alpha_mean_{name}"] = float(gate_threshold_shift_alpha_arr.mean()) if gate_threshold_shift_alpha_arr.size else 0.0
                row[f"attention_gate_source_logit_bias_mean_{name}"] = float(gate_source_logit_bias_arr.mean()) if gate_source_logit_bias_arr.size else 0.0
                row[f"attention_gate_common_cancel_alpha_mean_{name}"] = float(gate_common_cancel_alpha_arr.mean()) if gate_common_cancel_alpha_arr.size else 0.0
                row[f"attention_gate_contrast_shift_alpha_mean_{name}"] = float(gate_contrast_shift_alpha_arr.mean()) if gate_contrast_shift_alpha_arr.size else 0.0
                row[f"attention_gate_mild_rescue_alpha_mean_{name}"] = float(gate_mild_rescue_alpha_arr.mean()) if gate_mild_rescue_alpha_arr.size else 0.0
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
                row[f"raw_shapley_mean_{name}"] = float(raw_shapley_arr[:, idx].mean())
                row[f"shapley_target_mean_{name}"] = float(shapley_target_arr[:, idx].mean())
            row["attention_relative_gap_score_mean_vision_distance"] = float(relative_gap_arr[:, 0].mean())
            row["attention_relative_gap_score_mean_vision_ego"] = float(relative_gap_arr[:, 1].mean())
            row["attention_relative_gap_score_mean_vision_route"] = float(relative_gap_arr[:, 2].mean())
            row["attention_relative_gap_norm_mean"] = float(relative_gap_norm_arr.mean())
            row["attention_bias_norm_mean"] = float(bias_norm_arr.mean())
            row["attention_bias_gate_norm_mean"] = float(gate_norm_arr.mean())
            row["attention_bias_abs_mean"] = float(bias_abs_arr.mean())
            row["clean_score_anchor_error"] = float(np.mean(clean_score_anchor_errors)) if clean_score_anchor_errors else 0.0
            row["clean_action_anchor_error"] = float(np.mean(clean_action_anchor_errors)) if clean_action_anchor_errors else 0.0
            row["source_sign_score_anchor_error"] = (
                float(np.mean(source_sign_score_anchor_errors)) if source_sign_score_anchor_errors else 0.0
            )
            row["source_clean_gate_preserve_error"] = (
                float(np.mean(source_clean_gate_preserve_errors)) if source_clean_gate_preserve_errors else 0.0
            )
            rows.append(row)
        return rows


def compute_v3_3_attention_score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        normal_other_score_mean = float(
            np.mean(
                [
                    normal["attention_score_mean_distance"],
                    normal["attention_score_mean_ego"],
                    normal["attention_score_mean_route"],
                ]
            )
        )
        mild_other_score_mean = float(
            np.mean(
                [
                    mild_vision["attention_score_mean_distance"],
                    mild_vision["attention_score_mean_ego"],
                    mild_vision["attention_score_mean_route"],
                ]
            )
        )
        vision_attention_delta = float(normal["attention_mean_vision"] - mild_vision["attention_mean_vision"])
        vision_targetlogit_delta = float(
            normal["targetlogit_shapley_mean_vision"] - mild_vision["targetlogit_shapley_mean_vision"]
        )
        rows.append(
            {
                "candidate_name": candidate_name,
                "vision_attention_score_clean_minus_mild": float(
                    normal["attention_score_mean_vision"] - mild_vision["attention_score_mean_vision"]
                ),
                "vision_attention_relative_score_clean_minus_mild": float(
                    (normal["attention_score_mean_vision"] - normal_other_score_mean)
                    - (mild_vision["attention_score_mean_vision"] - mild_other_score_mean)
                ),
                "vision_attention_clean_minus_mild": vision_attention_delta,
                "vision_targetlogit_shapley_clean_minus_mild": vision_targetlogit_delta,
                "vision_attention_targetlogit_sign_match_flag": _raw_sign_match(
                    vision_attention_delta,
                    vision_targetlogit_delta,
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_3_pass_fail(summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["normal_success_rate"] < 0.75:
        return "FAIL"
    if summary["route_distance_side_effect_flag"] > 0.0 or summary["saturation_warning_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["normal_success_rate"] >= 0.85
        and summary["normal_to_mild_success_drop"] <= 0.03
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_attention_targetlogit_sign_match_flag"] > 0.0
        and summary["saturation_warning_flag"] == 0.0
        and summary["route_distance_side_effect_flag"] == 0.0
    ):
        return "PASS"
    return "WARN"


def build_v3_3_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    attention_score_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    final_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        attn_score = attention_score_df[attention_score_df["candidate_name"] == candidate_name].iloc[0]
        summary = {
            "candidate_name": candidate_name,
            "normal_success_rate": float(normal["eval_success_rate"]),
            "normal_return": float(normal["eval_return"]),
            "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
            "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
            "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
            "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
            "consistency_loss": float(cons["consistency_loss"]),
            "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
            "consistency_gain_score": float(cons["consistency_gain_score"]),
            "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
            "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
            "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
            "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
            "saturation_warning_flag": float(tie["saturation_warning_flag"]),
            "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
            "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
            "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
            "vision_attention_score_clean_minus_mild": float(attn_score["vision_attention_score_clean_minus_mild"]),
            "vision_attention_relative_score_clean_minus_mild": float(attn_score["vision_attention_relative_score_clean_minus_mild"]),
            "vision_attention_clean_minus_mild": float(attn_score["vision_attention_clean_minus_mild"]),
            "vision_attention_targetlogit_sign_match_flag": float(attn_score["vision_attention_targetlogit_sign_match_flag"]),
        }
        summary["tie_break_score"] = (
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_relative_score_delta"]) * summary["vision_attention_relative_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_targetlogit_delta"]) * summary["vision_targetlogit_shapley_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_v3_3_pass_fail(
            summary=summary,
            is_baseline=(candidate_name == baseline_candidate),
        )
        final_rows.append(summary)

    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_attention_relative_score_clean_minus_mild",
            "vision_targetlogit_shapley_clean_minus_mild",
            "normal_success_rate",
            "normal_to_mild_success_drop",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_3_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    attention_score_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Attention Score Summary",
        "",
        attention_score_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _center_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores - scores.mean(dim=-1, keepdim=True)


class Stage34RelativeGapTrainer(Stage33AttentionScoreTrainer):
    """V3.4 trainer using centered pairwise relative-gap supervision."""

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_logweight_gap_route": 0.0,
            "loss/vision_logweight_gap_distance": 0.0,
            "loss/vision_reliability_score_coupling": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        logweight_route_coef = float(losses_cfg.get("vision_logweight_gap_route_loss_coef", 0.0))
        logweight_distance_coef = float(losses_cfg.get("vision_logweight_gap_distance_loss_coef", 0.0))
        logweight_margin = float(losses_cfg.get("vision_logweight_gap_margin", 0.0))
        coupling_coef = float(losses_cfg.get("vision_reliability_score_coupling_loss_coef", 0.0))
        coupling_alpha = float(losses_cfg.get("vision_reliability_score_coupling_alpha", 4.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        eps = float(losses_cfg.get("attention_logratio_eps", 1e-6))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                route_logweight_terms: list[torch.Tensor] = []
                distance_logweight_terms: list[torch.Tensor] = []
                coupling_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    clean_scores = _center_scores(clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    clean_attention = clean_outputs["attention_weights"][batch_idx, time_idx].clamp_min(eps)
                    corrupt_attention = outputs["attention_weights"][batch_idx, time_idx].clamp_min(eps)
                    clean_reliability = clean_outputs["reliability_scores"][batch_idx, time_idx].clamp_min(eps)
                    corrupt_reliability = outputs["reliability_scores"][batch_idx, time_idx].clamp_min(eps)

                    g_vr_clean = clean_scores[0] - clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = clean_scores[0] - clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )

                    h_vr_clean = torch.log(clean_attention[0]) - torch.log(clean_attention[3])
                    h_vr_mild = torch.log(corrupt_attention[0]) - torch.log(corrupt_attention[3])
                    h_vd_clean = torch.log(clean_attention[0]) - torch.log(clean_attention[1])
                    h_vd_mild = torch.log(corrupt_attention[0]) - torch.log(corrupt_attention[1])
                    route_logweight_terms.append(
                        F.relu(torch.as_tensor(logweight_margin, device=self.device) - (h_vr_clean - h_vr_mild))
                    )
                    distance_logweight_terms.append(
                        F.relu(torch.as_tensor(logweight_margin, device=self.device) - (h_vd_clean - h_vd_mild))
                    )

                    if coupling_coef > 0.0:
                        delta_r = torch.log(clean_reliability[0]) - torch.log(corrupt_reliability[0])
                        delta_s = clean_scores[0] - corrupt_scores[0]
                        coupling_terms.append(torch.abs(delta_s - coupling_alpha * delta_r))

                    drift_terms.append(clean_scores.pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                route_logweight_loss = (
                    torch.stack(route_logweight_terms).mean() if route_logweight_terms else torch.zeros((), device=self.device)
                )
                distance_logweight_loss = (
                    torch.stack(distance_logweight_terms).mean() if distance_logweight_terms else torch.zeros((), device=self.device)
                )
                coupling_loss = torch.stack(coupling_terms).mean() if coupling_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + logweight_route_coef * route_logweight_loss
                    + logweight_distance_coef * distance_logweight_loss
                    + coupling_coef * coupling_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_logweight_gap_route"] += float(route_logweight_loss.item())
                metrics["loss/vision_logweight_gap_distance"] += float(distance_logweight_loss.item())
                metrics["loss/vision_reliability_score_coupling"] += float(coupling_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage34bHybridTrainer(Stage34RelativeGapTrainer):
    """V3.4b trainer combining relative-gap geometry with direct sign push."""

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_logweight_gap_route": 0.0,
            "loss/vision_logweight_gap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/vision_attention_weight_margin": 0.0,
            "loss/vision_reliability_score_coupling": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        logweight_route_coef = float(losses_cfg.get("vision_logweight_gap_route_loss_coef", 0.0))
        logweight_distance_coef = float(losses_cfg.get("vision_logweight_gap_distance_loss_coef", 0.0))
        logweight_margin = float(losses_cfg.get("vision_logweight_gap_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_weight_coef = float(losses_cfg.get("vision_attention_weight_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        sign_weight_margin = float(losses_cfg.get("vision_attention_weight_margin", 0.0))
        coupling_coef = float(losses_cfg.get("vision_reliability_score_coupling_loss_coef", 0.0))
        coupling_alpha = float(losses_cfg.get("vision_reliability_score_coupling_alpha", 4.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        eps = float(losses_cfg.get("attention_logratio_eps", 1e-6))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                route_logweight_terms: list[torch.Tensor] = []
                distance_logweight_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                sign_weight_terms: list[torch.Tensor] = []
                coupling_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    clean_scores = _center_scores(clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    clean_attention = clean_outputs["attention_weights"][batch_idx, time_idx].clamp_min(eps)
                    corrupt_attention = outputs["attention_weights"][batch_idx, time_idx].clamp_min(eps)
                    clean_reliability = clean_outputs["reliability_scores"][batch_idx, time_idx].clamp_min(eps)
                    corrupt_reliability = outputs["reliability_scores"][batch_idx, time_idx].clamp_min(eps)

                    g_vr_clean = clean_scores[0] - clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = clean_scores[0] - clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )

                    h_vr_clean = torch.log(clean_attention[0]) - torch.log(clean_attention[3])
                    h_vr_mild = torch.log(corrupt_attention[0]) - torch.log(corrupt_attention[3])
                    h_vd_clean = torch.log(clean_attention[0]) - torch.log(clean_attention[1])
                    h_vd_mild = torch.log(corrupt_attention[0]) - torch.log(corrupt_attention[1])
                    route_logweight_terms.append(
                        F.relu(torch.as_tensor(logweight_margin, device=self.device) - (h_vr_clean - h_vr_mild))
                    )
                    distance_logweight_terms.append(
                        F.relu(torch.as_tensor(logweight_margin, device=self.device) - (h_vd_clean - h_vd_mild))
                    )

                    sign_score_terms.append(
                        F.relu(torch.as_tensor(sign_score_margin, device=self.device) - (clean_scores[0] - corrupt_scores[0]))
                    )
                    sign_weight_terms.append(
                        F.relu(torch.as_tensor(sign_weight_margin, device=self.device) - (clean_attention[0] - corrupt_attention[0]))
                    )

                    if coupling_coef > 0.0:
                        delta_r = torch.log(clean_reliability[0]) - torch.log(corrupt_reliability[0])
                        delta_s = clean_scores[0] - corrupt_scores[0]
                        coupling_terms.append(torch.abs(delta_s - coupling_alpha * delta_r))

                    drift_terms.append(clean_scores.pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                route_logweight_loss = (
                    torch.stack(route_logweight_terms).mean() if route_logweight_terms else torch.zeros((), device=self.device)
                )
                distance_logweight_loss = (
                    torch.stack(distance_logweight_terms).mean() if distance_logweight_terms else torch.zeros((), device=self.device)
                )
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                sign_weight_loss = torch.stack(sign_weight_terms).mean() if sign_weight_terms else torch.zeros((), device=self.device)
                coupling_loss = torch.stack(coupling_terms).mean() if coupling_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + logweight_route_coef * route_logweight_loss
                    + logweight_distance_coef * distance_logweight_loss
                    + sign_score_coef * sign_score_loss
                    + sign_weight_coef * sign_weight_loss
                    + coupling_coef * coupling_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_logweight_gap_route"] += float(route_logweight_loss.item())
                metrics["loss/vision_logweight_gap_distance"] += float(distance_logweight_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/vision_attention_weight_margin"] += float(sign_weight_loss.item())
                metrics["loss/vision_reliability_score_coupling"] += float(coupling_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage35AttentionAdapterTrainer(Stage34bHybridTrainer):
    """V3.5 trainer with score adapter plus reduced confirm-ready objectives."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        self.model.freeze_attention_score_adapter_calibration_only()
        self.model.fusion_module.score_adapter_residual_scale = float(config["losses"].get("score_adapter_residual_scale", 0.0))
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    clean_scores = _center_scores(clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])

                    g_vr_clean = clean_scores[0] - clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = clean_scores[0] - clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(torch.as_tensor(sign_score_margin, device=self.device) - (clean_scores[0] - corrupt_scores[0]))
                    )
                    drift_terms.append(clean_scores.pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage36RelativeScoreHeadTrainer(Stage35AttentionAdapterTrainer):
    """V3.6 trainer with relative score head reparameterization."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        self.model.freeze_attention_relative_score_calibration_only()
        self.model.fusion_module.score_adapter_residual_scale = float(config["losses"].get("score_adapter_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_residual_scale = float(config["losses"].get("relative_gap_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_mode = str(config["losses"].get("relative_gap_mode", "vision_anchor"))
        self.model.fusion_module.use_relative_score_head = bool(config["losses"].get("use_relative_score_head", False))
        if self.model.fusion_module.use_relative_score_head:
            self.model.fusion_module.initialize_base_score_head_from_score_net()
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))


class Stage37ReliabilityBiasTrainer(Stage35AttentionAdapterTrainer):
    """V3.7 trainer with reliability-conditioned score bias branch."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        Stage35AttentionAdapterTrainer.__init__(self, config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        self.model.freeze_attention_reliability_bias_calibration_only()
        self.model.fusion_module.score_adapter_residual_scale = float(config["losses"].get("score_adapter_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_residual_scale = float(config["losses"].get("relative_gap_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_mode = str(config["losses"].get("relative_gap_mode", "vision_anchor"))
        self.model.fusion_module.reliability_bias_residual_scale = float(
            config["losses"].get("reliability_bias_residual_scale", 0.0)
        )
        self.model.fusion_module.use_relative_score_head = bool(config["losses"].get("use_relative_score_head", False))
        self.model.fusion_module.use_reliability_bias_head = bool(config["losses"].get("use_reliability_bias_head", False))
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/vision_bias_monotonicity": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        bias_coef = float(losses_cfg.get("vision_bias_monotonicity_loss_coef", 0.0))
        bias_margin = float(losses_cfg.get("vision_bias_monotonicity_margin", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                bias_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    clean_scores = _center_scores(clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    clean_bias = clean_outputs["attention_bias_scores"][batch_idx, time_idx]
                    corrupt_bias = outputs["attention_bias_scores"][batch_idx, time_idx]

                    g_vr_clean = clean_scores[0] - clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = clean_scores[0] - clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(torch.as_tensor(sign_score_margin, device=self.device) - (clean_scores[0] - corrupt_scores[0]))
                    )
                    bias_terms.append(
                        F.relu(torch.as_tensor(bias_margin, device=self.device) - (clean_bias[0] - corrupt_bias[0]))
                    )
                    drift_terms.append(clean_scores.pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                bias_loss = torch.stack(bias_terms).mean() if bias_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + bias_coef * bias_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/vision_bias_monotonicity"] += float(bias_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37bCleanPreservingBiasTrainer(Stage37ReliabilityBiasTrainer):
    """V3.7b trainer that preserves V3.7 sign recovery while restoring clean performance."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        self.teacher_model = self.reference_model

        train_score_adapter = bool(config["losses"].get("train_score_adapter_alongside_bias", False))
        self.model.freeze_attention_clean_preserving_bias_calibration_only(train_score_adapter=train_score_adapter)
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/vision_bias_monotonicity": 0.0,
            "loss/clean_bias_zero": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        bias_coef = float(losses_cfg.get("vision_bias_monotonicity_loss_coef", 0.0))
        bias_margin = float(losses_cfg.get("vision_bias_monotonicity_margin", 0.0))
        clean_bias_coef = float(losses_cfg.get("clean_bias_zero_loss_coef", 0.0))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                with torch.no_grad():
                    teacher_clean_outputs = self.teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                bias_terms: list[torch.Tensor] = []
                clean_bias_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    clean_scores = _center_scores(clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    clean_bias = clean_outputs["attention_bias_scores"][batch_idx, time_idx]
                    corrupt_bias = outputs["attention_bias_scores"][batch_idx, time_idx]
                    teacher_clean_scores = teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    teacher_clean_action_means = teacher_clean_outputs["action_means"][batch_idx, time_idx]
                    clean_action_means = clean_outputs["action_means"][batch_idx, time_idx]

                    g_vr_clean = clean_scores[0] - clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = clean_scores[0] - clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(torch.as_tensor(sign_score_margin, device=self.device) - (clean_scores[0] - corrupt_scores[0]))
                    )
                    bias_terms.append(
                        F.relu(torch.as_tensor(bias_margin, device=self.device) - (clean_bias[0] - corrupt_bias[0]))
                    )
                    clean_bias_terms.append(clean_bias.abs().mean())
                    clean_score_anchor_terms.append(
                        torch.mean((clean_outputs["attention_scores"][batch_idx, time_idx] - teacher_clean_scores).pow(2))
                    )
                    clean_action_anchor_terms.append(torch.mean((clean_action_means - teacher_clean_action_means).pow(2)))
                    drift_terms.append(clean_scores.pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = (
                    torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                )
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                bias_loss = torch.stack(bias_terms).mean() if bias_terms else torch.zeros((), device=self.device)
                clean_bias_zero_loss = (
                    torch.stack(clean_bias_terms).mean() if clean_bias_terms else torch.zeros((), device=self.device)
                )
                clean_score_anchor_loss = (
                    torch.stack(clean_score_anchor_terms).mean()
                    if clean_score_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                clean_action_anchor_loss = (
                    torch.stack(clean_action_anchor_terms).mean()
                    if clean_action_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + bias_coef * bias_loss
                    + clean_bias_coef * clean_bias_zero_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/vision_bias_monotonicity"] += float(bias_loss.item())
                metrics["loss/clean_bias_zero"] += float(clean_bias_zero_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37cDualTeacherGatedBiasTrainer(Stage37bCleanPreservingBiasTrainer):
    """V3.7c trainer with dual-teacher gated reliability bias calibration."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        Stage37ReliabilityBiasTrainer.__init__(self, config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        self.model.freeze_attention_gated_reliability_bias_calibration_only()
        self.model.fusion_module.score_adapter_residual_scale = float(losses_cfg.get("score_adapter_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_residual_scale = float(losses_cfg.get("relative_gap_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_mode = str(losses_cfg.get("relative_gap_mode", "vision_anchor"))
        self.model.fusion_module.reliability_bias_residual_scale = float(
            losses_cfg.get("reliability_bias_residual_scale", 0.0)
        )
        self.model.fusion_module.reliability_bias_gate_temperature = float(
            losses_cfg.get("reliability_bias_gate_temperature", losses_cfg.get("gate_temperature", 1.0))
        )
        self.model.fusion_module.reliability_bias_gate_threshold = float(
            losses_cfg.get("reliability_bias_gate_threshold", losses_cfg.get("gate_threshold", 0.0))
        )
        self.model.fusion_module.use_relative_score_head = bool(losses_cfg.get("use_relative_score_head", False))
        self.model.fusion_module.use_reliability_bias_head = bool(losses_cfg.get("use_reliability_bias_head", False))
        self.model.fusion_module.use_reliability_bias_gate = bool(losses_cfg.get("use_reliability_bias_gate", False))
        self.model.fusion_module.gate_input_mode = str(losses_cfg.get("gate_input_mode", "legacy"))
        self.model.fusion_module.gate_use_abs_bias_feature = bool(losses_cfg.get("gate_use_abs_bias_feature", False))
        self.model.fusion_module.gate_use_score_delta_feature = bool(losses_cfg.get("gate_use_score_delta_feature", False))
        self.model.fusion_module.gate_use_teacher_delta_feature = bool(
            losses_cfg.get("gate_use_teacher_delta_feature", False)
        )
        self.model.fusion_module.gate_feature_stopgrad_on_bias = bool(
            losses_cfg.get("gate_feature_stopgrad_on_bias", False)
        )

        sign_teacher_root = str(config.get("sign_teacher_checkpoint_root", config.get("source_checkpoint_root", "")))
        clean_teacher_root = str(config.get("clean_teacher_checkpoint_root", ""))
        if not sign_teacher_root:
            raise ValueError("V3.7c requires sign_teacher_checkpoint_root or source_checkpoint_root")
        if not clean_teacher_root:
            raise ValueError("V3.7c requires clean_teacher_checkpoint_root")

        self.sign_teacher_model = load_frozen_teacher_model(
            config=config,
            checkpoint_root=sign_teacher_root,
            source_experiment_name=str(config.get("source_experiment_name", "v3_7_reliability_bias_pilot")),
            device=self.device,
            seed=seed,
        )
        self.clean_teacher_model = load_frozen_teacher_model(
            config=config,
            checkpoint_root=clean_teacher_root,
            source_experiment_name=str(config.get("clean_teacher_experiment_name", "v3_6_relative_head_pilot")),
            device=self.device,
            seed=seed,
        )
        self.teacher_model = self.clean_teacher_model
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/vision_bias_monotonicity": 0.0,
            "loss/clean_bias_zero": 0.0,
            "loss/clean_gate_suppression": 0.0,
            "loss/mild_gate_activation": 0.0,
            "loss/gate_contrast": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        bias_coef = float(losses_cfg.get("vision_bias_monotonicity_loss_coef", 0.0))
        bias_margin = float(losses_cfg.get("vision_bias_monotonicity_margin", 0.0))
        clean_bias_coef = float(losses_cfg.get("clean_bias_zero_loss_coef", 0.0))
        clean_gate_coef = float(losses_cfg.get("clean_gate_suppression_loss_coef", 0.0))
        mild_gate_coef = float(losses_cfg.get("mild_gate_activation_loss_coef", 0.0))
        mild_gate_margin = float(losses_cfg.get("mild_gate_activation_margin", 0.0))
        gate_contrast_coef = float(losses_cfg.get("gate_contrast_loss_coef", 0.0))
        gate_contrast_margin = float(losses_cfg.get("gate_contrast_margin", 0.0))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                with torch.no_grad():
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                bias_terms: list[torch.Tensor] = []
                clean_bias_terms: list[torch.Tensor] = []
                clean_gate_terms: list[torch.Tensor] = []
                mild_gate_terms: list[torch.Tensor] = []
                gate_contrast_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    teacher_clean_scores = _center_scores(sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    teacher_clean_bias = sign_teacher_clean_outputs["attention_bias_scores"][batch_idx, time_idx]
                    corrupt_bias = outputs["attention_bias_scores"][batch_idx, time_idx]
                    current_clean_bias = current_clean_outputs["attention_bias_scores"][batch_idx, time_idx]
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]

                    g_vr_clean = teacher_clean_scores[0] - teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = teacher_clean_scores[0] - teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )
                    bias_terms.append(
                        F.relu(torch.as_tensor(bias_margin, device=self.device) - (teacher_clean_bias[0] - corrupt_bias[0]))
                    )
                    clean_bias_terms.append(current_clean_bias.abs().mean())
                    clean_gate_terms.append(current_clean_gate[0])
                    mild_gate_terms.append(
                        F.relu(torch.as_tensor(mild_gate_margin, device=self.device) - corrupt_gate[0])
                    )
                    gate_contrast_terms.append(
                        F.relu(torch.as_tensor(gate_contrast_margin, device=self.device) - (corrupt_gate[0] - current_clean_gate[0]))
                    )
                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(
                        torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2))
                    )
                    drift_terms.append(_center_scores(current_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = (
                    torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                )
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                bias_loss = torch.stack(bias_terms).mean() if bias_terms else torch.zeros((), device=self.device)
                clean_bias_zero_loss = (
                    torch.stack(clean_bias_terms).mean() if clean_bias_terms else torch.zeros((), device=self.device)
                )
                clean_gate_suppression_loss = (
                    torch.stack(clean_gate_terms).mean() if clean_gate_terms else torch.zeros((), device=self.device)
                )
                mild_gate_activation_loss = (
                    torch.stack(mild_gate_terms).mean() if mild_gate_terms else torch.zeros((), device=self.device)
                )
                gate_contrast_loss = (
                    torch.stack(gate_contrast_terms).mean() if gate_contrast_terms else torch.zeros((), device=self.device)
                )
                clean_score_anchor_loss = (
                    torch.stack(clean_score_anchor_terms).mean()
                    if clean_score_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                clean_action_anchor_loss = (
                    torch.stack(clean_action_anchor_terms).mean()
                    if clean_action_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + bias_coef * bias_loss
                    + clean_bias_coef * clean_bias_zero_loss
                    + clean_gate_coef * clean_gate_suppression_loss
                    + mild_gate_coef * mild_gate_activation_loss
                    + gate_contrast_coef * gate_contrast_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/vision_bias_monotonicity"] += float(bias_loss.item())
                metrics["loss/clean_bias_zero"] += float(clean_bias_zero_loss.item())
                metrics["loss/clean_gate_suppression"] += float(clean_gate_suppression_loss.item())
                metrics["loss/mild_gate_activation"] += float(mild_gate_activation_loss.item())
                metrics["loss/gate_contrast"] += float(gate_contrast_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37dSelectiveGateTrainer(Stage37cDualTeacherGatedBiasTrainer):
    """V3.7d trainer with bounded selective gate targets."""

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/vision_bias_monotonicity": 0.0,
            "loss/clean_bias_zero": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        bias_coef = float(losses_cfg.get("vision_bias_monotonicity_loss_coef", 0.0))
        bias_margin = float(losses_cfg.get("vision_bias_monotonicity_margin", 0.0))
        clean_bias_coef = float(losses_cfg.get("clean_bias_zero_loss_coef", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                )
                with torch.no_grad():
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                bias_terms: list[torch.Tensor] = []
                clean_bias_terms: list[torch.Tensor] = []
                clean_gate_cap_terms: list[torch.Tensor] = []
                mild_gate_floor_terms: list[torch.Tensor] = []
                gate_gap_margin_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    teacher_clean_scores = _center_scores(sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx])
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    teacher_clean_bias = sign_teacher_clean_outputs["attention_bias_scores"][batch_idx, time_idx]
                    corrupt_bias = outputs["attention_bias_scores"][batch_idx, time_idx]
                    current_clean_bias = current_clean_outputs["attention_bias_scores"][batch_idx, time_idx]
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]

                    g_vr_clean = teacher_clean_scores[0] - teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = teacher_clean_scores[0] - teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )
                    bias_terms.append(
                        F.relu(torch.as_tensor(bias_margin, device=self.device) - (teacher_clean_bias[0] - corrupt_bias[0]))
                    )
                    clean_bias_terms.append(current_clean_bias.abs().mean())
                    clean_gate_cap_terms.append(
                        F.relu(current_clean_gate[0] - torch.as_tensor(clean_gate_max, device=self.device))
                    )
                    mild_gate_floor_terms.append(
                        F.relu(torch.as_tensor(mild_gate_min, device=self.device) - corrupt_gate[0])
                    )
                    gate_gap_margin_terms.append(
                        F.relu(torch.as_tensor(gate_gap_margin, device=self.device) - (corrupt_gate[0] - current_clean_gate[0]))
                    )
                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(
                        torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2))
                    )
                    drift_terms.append(_center_scores(current_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = (
                    torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                )
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                bias_loss = torch.stack(bias_terms).mean() if bias_terms else torch.zeros((), device=self.device)
                clean_bias_zero_loss = (
                    torch.stack(clean_bias_terms).mean() if clean_bias_terms else torch.zeros((), device=self.device)
                )
                clean_gate_cap_loss = (
                    torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                )
                mild_gate_floor_loss = (
                    torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                )
                gate_gap_margin_loss = (
                    torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                )
                clean_score_anchor_loss = (
                    torch.stack(clean_score_anchor_terms).mean()
                    if clean_score_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                clean_action_anchor_loss = (
                    torch.stack(clean_action_anchor_terms).mean()
                    if clean_action_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + bias_coef * bias_loss
                    + clean_bias_coef * clean_bias_zero_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/vision_bias_monotonicity"] += float(bias_loss.item())
                metrics["loss/clean_bias_zero"] += float(clean_bias_zero_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37eGateOnlyTrainer(Stage37ReliabilityBiasTrainer):
    """V3.7e gate-only selective finetune that preserves V3.7c bias geometry."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        self.model.fusion_module.reliability_bias_gate_head = torch.nn.Sequential(
            torch.nn.LazyLinear(8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 1),
        ).to(self.device)
        self.model.freeze_attention_gate_only_selective_finetune(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", False))
        )
        self.model.fusion_module.score_adapter_residual_scale = float(losses_cfg.get("score_adapter_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_residual_scale = float(losses_cfg.get("relative_gap_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_mode = str(losses_cfg.get("relative_gap_mode", "vision_anchor"))
        self.model.fusion_module.reliability_bias_residual_scale = float(
            losses_cfg.get("reliability_bias_residual_scale", 0.0)
        )
        self.model.fusion_module.reliability_bias_gate_temperature = float(
            losses_cfg.get("reliability_bias_gate_temperature", losses_cfg.get("gate_temperature", 1.0))
        )
        self.model.fusion_module.reliability_bias_gate_threshold = float(
            losses_cfg.get("reliability_bias_gate_threshold", losses_cfg.get("gate_threshold", 0.0))
        )
        self.model.fusion_module.use_relative_score_head = bool(losses_cfg.get("use_relative_score_head", False))
        self.model.fusion_module.use_reliability_bias_head = bool(losses_cfg.get("use_reliability_bias_head", False))
        self.model.fusion_module.use_reliability_bias_gate = bool(losses_cfg.get("use_reliability_bias_gate", False))
        self.model.fusion_module.gate_input_mode = str(losses_cfg.get("gate_input_mode", "legacy"))
        self.model.fusion_module.gate_use_abs_bias_feature = bool(losses_cfg.get("gate_use_abs_bias_feature", False))
        self.model.fusion_module.gate_use_score_delta_feature = bool(losses_cfg.get("gate_use_score_delta_feature", False))
        self.model.fusion_module.gate_use_teacher_delta_feature = bool(
            losses_cfg.get("gate_use_teacher_delta_feature", False)
        )
        self.model.fusion_module.gate_feature_stopgrad_on_bias = bool(
            losses_cfg.get("gate_feature_stopgrad_on_bias", False)
        )

        sign_teacher_root = str(config.get("sign_teacher_checkpoint_root", config.get("source_checkpoint_root", "")))
        clean_teacher_root = str(config.get("clean_teacher_checkpoint_root", ""))
        if not sign_teacher_root:
            raise ValueError("V3.7e requires sign_teacher_checkpoint_root or source_checkpoint_root")
        if not clean_teacher_root:
            raise ValueError("V3.7e requires clean_teacher_checkpoint_root")
        self.sign_teacher_model = load_frozen_teacher_model(
            config=config,
            checkpoint_root=sign_teacher_root,
            source_experiment_name=str(config.get("source_experiment_name", "v3_7c_dual_teacher_gated_bias_pilot")),
            device=self.device,
            seed=seed,
        )
        self.clean_teacher_model = load_frozen_teacher_model(
            config=config,
            checkpoint_root=clean_teacher_root,
            source_experiment_name=str(config.get("clean_teacher_experiment_name", "v3_6_relative_head_pilot")),
            device=self.device,
            seed=seed,
        )
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise ValueError("V3.7e gate-only finetune produced no trainable parameters")
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                clean_gate_cap_terms: list[torch.Tensor] = []
                mild_gate_floor_terms: list[torch.Tensor] = []
                gate_gap_margin_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )
                    clean_gate_cap_terms.append(
                        F.relu(current_clean_gate[0] - torch.as_tensor(clean_gate_max, device=self.device))
                    )
                    mild_gate_floor_terms.append(
                        F.relu(torch.as_tensor(mild_gate_min, device=self.device) - corrupt_gate[0])
                    )
                    gate_gap_margin_terms.append(
                        F.relu(torch.as_tensor(gate_gap_margin, device=self.device) - (corrupt_gate[0] - current_clean_gate[0]))
                    )
                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(
                        torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2))
                    )
                    drift_terms.append(_center_scores(current_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = (
                    torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                )
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = (
                    torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                )
                mild_gate_floor_loss = (
                    torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                )
                gate_gap_margin_loss = (
                    torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                )
                clean_score_anchor_loss = (
                    torch.stack(clean_score_anchor_terms).mean()
                    if clean_score_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                clean_action_anchor_loss = (
                    torch.stack(clean_action_anchor_terms).mean()
                    if clean_action_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37fSourceGateResidualTrainer(Stage37eGateOnlyTrainer):
    """V3.7f trainer preserving source gate operating point with residual selective finetune."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        Stage37ReliabilityBiasTrainer.__init__(self, config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        self.model.fusion_module.reliability_bias_gate_residual_head = torch.nn.Sequential(
            torch.nn.LazyLinear(8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 1),
        ).to(self.device)
        last_layer = self.model.fusion_module.reliability_bias_gate_residual_head[-1]
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.zeros_(last_layer.bias)
        self.model.freeze_attention_source_gate_residual_only(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", False))
        )
        self.model.fusion_module.score_adapter_residual_scale = float(losses_cfg.get("score_adapter_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_residual_scale = float(losses_cfg.get("relative_gap_residual_scale", 0.0))
        self.model.fusion_module.relative_gap_mode = str(losses_cfg.get("relative_gap_mode", "vision_anchor"))
        self.model.fusion_module.reliability_bias_residual_scale = float(
            losses_cfg.get("reliability_bias_residual_scale", 0.0)
        )
        self.model.fusion_module.gate_residual_scale = float(losses_cfg.get("gate_residual_scale", 0.0))
        self.model.fusion_module.gate_mode = str(losses_cfg.get("gate_mode", "source_residual_v1"))
        self.model.fusion_module.use_relative_score_head = bool(losses_cfg.get("use_relative_score_head", False))
        self.model.fusion_module.use_reliability_bias_head = bool(losses_cfg.get("use_reliability_bias_head", False))
        self.model.fusion_module.use_reliability_bias_gate = bool(losses_cfg.get("use_reliability_bias_gate", False))
        self.model.fusion_module.gate_input_mode = str(losses_cfg.get("gate_input_mode", "legacy"))
        self.model.fusion_module.gate_use_abs_bias_feature = bool(losses_cfg.get("gate_use_abs_bias_feature", False))
        self.model.fusion_module.gate_use_score_delta_feature = bool(losses_cfg.get("gate_use_score_delta_feature", False))
        self.model.fusion_module.gate_use_teacher_delta_feature = bool(
            losses_cfg.get("gate_use_teacher_delta_feature", False)
        )
        self.model.fusion_module.gate_use_source_gate_feature = bool(
            losses_cfg.get("gate_use_source_gate_feature", False)
        )
        self.model.fusion_module.gate_feature_stopgrad_on_bias = bool(
            losses_cfg.get("gate_feature_stopgrad_on_bias", False)
        )
        self.model.fusion_module.gate_feature_stopgrad_on_source_gate = bool(
            losses_cfg.get("gate_feature_stopgrad_on_source_gate", False)
        )

        sign_teacher_root = str(config.get("sign_teacher_checkpoint_root", config.get("source_checkpoint_root", "")))
        clean_teacher_root = str(config.get("clean_teacher_checkpoint_root", ""))
        if not sign_teacher_root:
            raise ValueError("V3.7f requires sign_teacher_checkpoint_root or source_checkpoint_root")
        if not clean_teacher_root:
            raise ValueError("V3.7f requires clean_teacher_checkpoint_root")
        sign_teacher_overrides = resolve_teacher_loss_overrides(
            config,
            explicit_key="sign_teacher_loss_overrides",
            candidate_name=str(config.get("source_gate_teacher_candidate", "reference_v3_7c_medium_locked")),
        )
        clean_teacher_overrides = resolve_teacher_loss_overrides(
            config,
            explicit_key="clean_teacher_loss_overrides",
            candidate_name=str(config.get("clean_teacher_candidate", config.get("baseline_candidate", "reference_v3_6_medium_locked"))),
        )
        self.sign_teacher_model = load_frozen_teacher_model(
            config=config,
            checkpoint_root=sign_teacher_root,
            source_experiment_name=str(config.get("source_experiment_name", "v3_7c_dual_teacher_gated_bias_pilot")),
            device=self.device,
            seed=seed,
            loss_overrides=sign_teacher_overrides,
        )
        self.clean_teacher_model = load_frozen_teacher_model(
            config=config,
            checkpoint_root=clean_teacher_root,
            source_experiment_name=str(config.get("clean_teacher_experiment_name", "v3_6_relative_head_pilot")),
            device=self.device,
            seed=seed,
            loss_overrides=clean_teacher_overrides,
        )
        if bool(config["candidates"][candidate_name].get("eval_only", False)) and candidate_name == "reference_v3_7c_medium_locked":
            self.model = self.sign_teacher_model
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def collect_rollout(self, update_idx: int) -> dict[str, float]:
        self._current_update_idx = int(update_idx)
        self.buffer.reset()
        obs, info = self.train_env.reset(seed=self.seed * 10000 + update_idx)
        del info
        hidden = self.model.initial_hidden(batch_size=1, device=self.device)
        sign_teacher_clean_hidden = self.sign_teacher_model.initial_hidden(batch_size=1, device=self.device)
        sign_teacher_corrupt_hidden = self.sign_teacher_model.initial_hidden(batch_size=1, device=self.device)
        clean_teacher_hidden = self.clean_teacher_model.initial_hidden(batch_size=1, device=self.device)
        episode_start = torch.ones(1, device=self.device)
        sign_teacher_episode_start_clean = torch.ones(1, device=self.device)
        sign_teacher_episode_start_corrupt = torch.ones(1, device=self.device)
        clean_teacher_episode_start = torch.ones(1, device=self.device)
        rng = np.random.default_rng(self.seed * 1000 + update_idx)
        episode_returns = []
        progress_values = []
        current_return = 0.0
        current_progress = []
        corruption_counter: dict[str, int] = {}
        single_names = [f"mild_{item['name']}" for item in self.raw_config["train_corruption"].get("single_modality", [])]
        mixed_names = [str(item["name"]) for item in self.raw_config["train_corruption"].get("mixed", [])]

        for _ in range(int(self.raw_config["train"]["rollout_steps"])):
            clean_policy_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
            policy_obs, severity, target_mask, corruption_name = self._sample_train_corruption(obs, rng)
            corruption_counter[corruption_name] = corruption_counter.get(corruption_name, 0) + 1
            reliability_targets = build_reliability_targets(
                aux_obs=obs["aux"],
                severity=severity,
                use_direct_corruption_targets=True,
                direct_target_drop_scale=float(self.raw_config["losses"]["direct_target_drop_scale"]),
            )
            hidden_before = hidden.clone()
            clean_obs_t = obs_to_tensor(clean_policy_obs, self.device)
            corrupt_obs_t = obs_to_tensor(policy_obs, self.device)
            with torch.no_grad():
                clean_teacher_output = self.clean_teacher_model.step(
                    obs_dict=clean_obs_t,
                    hidden_state=clean_teacher_hidden,
                    episode_start=clean_teacher_episode_start,
                    deterministic=False,
                )
                sign_teacher_clean_output = self.sign_teacher_model.step(
                    obs_dict=clean_obs_t,
                    hidden_state=sign_teacher_clean_hidden,
                    episode_start=sign_teacher_episode_start_clean,
                    deterministic=False,
                )
                sign_teacher_corrupt_output = self.sign_teacher_model.step(
                    obs_dict=corrupt_obs_t,
                    hidden_state=sign_teacher_corrupt_hidden,
                    episode_start=sign_teacher_episode_start_corrupt,
                    deterministic=False,
                )
                output = self.model.step(
                    obs_dict=corrupt_obs_t,
                    hidden_state=hidden,
                    episode_start=episode_start,
                    deterministic=False,
                    gate_feature_context={
                        "clean_teacher_scores": clean_teacher_output.aux_outputs["attention_scores"],
                        "sign_teacher_scores": sign_teacher_corrupt_output.aux_outputs["attention_scores"],
                        "source_gate_scores": sign_teacher_corrupt_output.aux_outputs["attention_bias_gates"],
                        "clean_teacher_adapter_delta": clean_teacher_output.aux_outputs.get(
                            "attention_score_adapter_delta",
                            torch.zeros_like(clean_teacher_output.aux_outputs["attention_score_features"]),
                        ),
                        "sign_teacher_adapter_delta": sign_teacher_corrupt_output.aux_outputs.get(
                            "attention_score_adapter_delta",
                            torch.zeros_like(sign_teacher_corrupt_output.aux_outputs["attention_score_features"]),
                        ),
                    },
                )
            action = output.action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated
            self.buffer.add(
                obs={**policy_obs, "aux": np.asarray(obs["aux"], dtype=np.float32).copy()},
                action=action,
                reward=float(reward),
                done=done,
                episode_start=bool(episode_start.item()),
                value=float(output.value.squeeze(0).cpu().item()),
                log_prob=float(output.log_prob.squeeze(0).cpu().item()),
                hidden_state=hidden_before.squeeze(0).cpu().numpy(),
                extras={
                    "reliability_targets": reliability_targets.astype(np.float32),
                    "severity": severity.astype(np.float32),
                    "target_mask": target_mask.astype(np.float32),
                    "corrupted_flag": np.array([float(corruption_name != "clean")], dtype=np.float32),
                    **{f"clean_{key}": clean_policy_obs[key].astype(np.float32) for key in MODALITY_NAMES},
                },
            )
            hidden = output.hidden_state
            sign_teacher_clean_hidden = sign_teacher_clean_output.hidden_state
            sign_teacher_corrupt_hidden = sign_teacher_corrupt_output.hidden_state
            clean_teacher_hidden = clean_teacher_output.hidden_state
            current_return += float(reward)
            current_progress.append(float(info.get("progress_delta", 0.0)))
            obs = next_obs
            episode_start = torch.tensor([float(done)], device=self.device)
            hidden = self.model.reset_hidden(hidden, episode_start)
            sign_teacher_episode_start_clean = torch.tensor([float(done)], device=self.device)
            sign_teacher_clean_hidden = self.sign_teacher_model.reset_hidden(
                sign_teacher_clean_hidden, sign_teacher_episode_start_clean
            )
            sign_teacher_episode_start_corrupt = torch.tensor([float(done)], device=self.device)
            sign_teacher_corrupt_hidden = self.sign_teacher_model.reset_hidden(
                sign_teacher_corrupt_hidden, sign_teacher_episode_start_corrupt
            )
            clean_teacher_episode_start = torch.tensor([float(done)], device=self.device)
            clean_teacher_hidden = self.clean_teacher_model.reset_hidden(
                clean_teacher_hidden, clean_teacher_episode_start
            )
            if done:
                episode_returns.append(current_return)
                progress_values.append(float(np.mean(current_progress)) if current_progress else 0.0)
                current_return = 0.0
                current_progress = []
                obs, info = self.train_env.reset()
                del info

        with torch.no_grad():
            clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
            clean_obs_t = obs_to_tensor(clean_obs, self.device)
            clean_teacher_output = self.clean_teacher_model.step(
                obs_dict=clean_obs_t,
                hidden_state=clean_teacher_hidden,
                episode_start=clean_teacher_episode_start,
                deterministic=False,
            )
            sign_teacher_clean_output = self.sign_teacher_model.step(
                obs_dict=clean_obs_t,
                hidden_state=sign_teacher_clean_hidden,
                episode_start=sign_teacher_episode_start_clean,
                deterministic=False,
            )
            output = self.model.step(
                obs_dict=clean_obs_t,
                hidden_state=hidden,
                episode_start=episode_start,
                deterministic=False,
                gate_feature_context={
                    "clean_teacher_scores": clean_teacher_output.aux_outputs["attention_scores"],
                    "sign_teacher_scores": sign_teacher_clean_output.aux_outputs["attention_scores"],
                    "source_gate_scores": sign_teacher_clean_output.aux_outputs["attention_bias_gates"],
                    "clean_teacher_adapter_delta": clean_teacher_output.aux_outputs.get(
                        "attention_score_adapter_delta",
                        torch.zeros_like(clean_teacher_output.aux_outputs["attention_score_features"]),
                    ),
                    "sign_teacher_adapter_delta": sign_teacher_clean_output.aux_outputs.get(
                        "attention_score_adapter_delta",
                        torch.zeros_like(sign_teacher_clean_output.aux_outputs["attention_score_features"]),
                    ),
                },
            )
        self.buffer.compute_returns_and_advantages(
            last_value=float(output.value.squeeze(0).cpu().item()),
            last_done=bool(episode_start.item()),
            gamma=self.base_cfg.ppo.gamma,
            gae_lambda=self.base_cfg.ppo.gae_lambda,
        )
        total = max(sum(corruption_counter.values()), 1)
        expected_clean_ratio = float(self.raw_config["train_corruption"].get("clean_probability", 0.0))
        expected_mild_ratio = float(
            self.raw_config["train_corruption"].get("mild_single_probability", 0.0)
            + self.raw_config["train_corruption"].get("mild_mixed_probability", 0.0)
        )
        actual_clean_ratio = float(corruption_counter.get("clean", 0) / total)
        return {
            "train/episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "train/avg_progress_to_goal": float(np.mean(progress_values)) if progress_values else 0.0,
            "train/corruption_expected_clean_ratio": expected_clean_ratio,
            "train/corruption_expected_mild_ratio": expected_mild_ratio,
            "train/corruption_clean_ratio": actual_clean_ratio,
            "train/corruption_actual_clean_ratio": actual_clean_ratio,
            "train/corruption_mild_ratio": float(1.0 - actual_clean_ratio),
            "train/corruption_actual_mild_ratio": float(1.0 - actual_clean_ratio),
            "train/corruption_mild_vision_ratio": float(
                (corruption_counter.get("mild_vision", 0) + corruption_counter.get("mild_vision_paired", 0)) / total
            ),
            "train/corruption_paired_mild_vision_ratio": float(corruption_counter.get("mild_vision_paired", 0) / total),
            "train/corruption_mild_ego_ratio": float(corruption_counter.get("mild_ego", 0) / total),
            "train/corruption_mild_distance_ratio": float(corruption_counter.get("mild_distance", 0) / total),
            "train/corruption_mild_route_ratio": float(corruption_counter.get("mild_route", 0) / total),
            "train/corruption_mixed_ratio": float(sum(corruption_counter.get(name, 0) for name in mixed_names) / total),
            "train/corruption_single_ratio": float(
                (sum(corruption_counter.get(name, 0) for name in single_names) + corruption_counter.get("mild_vision_paired", 0)) / total
            ),
        }

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                clean_gate_cap_terms: list[torch.Tensor] = []
                mild_gate_floor_terms: list[torch.Tensor] = []
                gate_gap_margin_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                source_gate_residual_zero_terms: list[torch.Tensor] = []
                source_sign_score_anchor_terms: list[torch.Tensor] = []
                source_clean_gate_preserve_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )
                    clean_gate_cap_terms.append(
                        F.relu(current_clean_gate[0] - torch.as_tensor(clean_gate_max, device=self.device))
                    )
                    mild_gate_floor_terms.append(
                        F.relu(torch.as_tensor(mild_gate_min, device=self.device) - corrupt_gate[0])
                    )
                    gate_gap_margin_terms.append(
                        F.relu(torch.as_tensor(gate_gap_margin, device=self.device) - (corrupt_gate[0] - current_clean_gate[0]))
                    )
                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(
                        torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2))
                    )
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    source_clean_gate_preserve_terms.append(
                        torch.mean((current_clean_gate - source_clean_gate).pow(2))
                    )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = (
                    torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                )
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = (
                    torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                )
                mild_gate_floor_loss = (
                    torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                )
                gate_gap_margin_loss = (
                    torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                )
                clean_score_anchor_loss = (
                    torch.stack(clean_score_anchor_terms).mean()
                    if clean_score_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                clean_action_anchor_loss = (
                    torch.stack(clean_action_anchor_terms).mean()
                    if clean_action_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                source_gate_residual_zero_loss = (
                    torch.stack(source_gate_residual_zero_terms).mean()
                    if source_gate_residual_zero_terms
                    else torch.zeros((), device=self.device)
                )
                source_sign_score_anchor_loss = (
                    torch.stack(source_sign_score_anchor_terms).mean()
                    if source_sign_score_anchor_terms
                    else torch.zeros((), device=self.device)
                )
                source_clean_gate_preserve_loss = (
                    torch.stack(source_clean_gate_preserve_terms).mean()
                    if source_clean_gate_preserve_terms
                    else torch.zeros((), device=self.device)
                )
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f1SourceGateDistilledResidualTrainer(Stage37fSourceGateResidualTrainer):
    """V3.7f1 trainer with explicit source-gate distillation and residual selectivity."""

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_gate_selective_residual": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_gate_selective_residual_coef = float(losses_cfg.get("source_gate_selective_residual_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                clean_gate_cap_terms: list[torch.Tensor] = []
                mild_gate_floor_terms: list[torch.Tensor] = []
                gate_gap_margin_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                source_gate_residual_zero_terms: list[torch.Tensor] = []
                source_gate_distill_terms: list[torch.Tensor] = []
                source_gate_selective_residual_terms: list[torch.Tensor] = []
                source_sign_score_anchor_terms: list[torch.Tensor] = []
                source_clean_gate_preserve_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )
                    clean_gate_cap_terms.append(
                        F.relu(current_clean_gate[0] - torch.as_tensor(clean_gate_max, device=self.device))
                    )
                    mild_gate_floor_terms.append(
                        F.relu(torch.as_tensor(mild_gate_min, device=self.device) - corrupt_gate[0])
                    )
                    gate_gap_margin_terms.append(
                        F.relu(
                            torch.as_tensor(gate_gap_margin, device=self.device)
                            - (corrupt_gate[0] - current_clean_gate[0])
                        )
                    )
                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(
                        torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2))
                    )
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    source_gate_distill_terms.append(torch.mean((current_clean_gate - source_clean_gate).pow(2)))
                    source_gate_distill_terms.append(torch.mean((corrupt_gate - source_corrupt_gate).pow(2)))
                    source_gate_selective_residual_terms.append(
                        F.relu(
                            torch.as_tensor(residual_selective_margin, device=self.device)
                            - (gate_residual_corrupt[0] - gate_residual_clean[0])
                        )
                    )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    source_clean_gate_preserve_terms.append(
                        torch.mean((current_clean_gate - source_clean_gate).pow(2))
                    )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_gate_selective_residual_loss = torch.stack(source_gate_selective_residual_terms).mean() if source_gate_selective_residual_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_gate_selective_residual_coef * source_gate_selective_residual_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_gate_selective_residual"] += float(source_gate_selective_residual_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f2SourceGateResidualLogitSelectiveTrainer(Stage37fSourceGateResidualTrainer):
    """V3.7f2 trainer with teacher-verified logit residual gate and paired selective supervision."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        configure_trainable_gate_residual_alpha(self.model, losses_cfg)
        self.model.freeze_attention_source_gate_residual_only(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", False))
        )
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/residual_gap_margin": 0.0,
            "loss/residual_logit_target": 0.0,
            "loss/residual_alpha_l2": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
            "train/residual_alpha": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        residual_gap_margin_coef = float(losses_cfg.get("residual_gap_margin_loss_coef", gate_gap_margin_coef))
        residual_logit_target_loss_coef = float(losses_cfg.get("residual_logit_target_loss_coef", 0.0))
        residual_logit_target_loss_type = str(losses_cfg.get("residual_logit_target_loss_type", "huber"))
        residual_logit_target_clip = losses_cfg.get("residual_logit_target_clip")
        residual_alpha_l2_coef = float(losses_cfg.get("residual_alpha_l2_coef", 0.0))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        disable_vision_preserve = bool(losses_cfg.get("disable_vision_final_gate_teacher_preserve", False))
        preserve_indices = select_gate_teacher_preserve_indices(disable_vision_preserve)
        paired_supervision = bool(losses_cfg.get("paired_selective_gate_supervision", False))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms: list[torch.Tensor] = []
                distance_gap_terms: list[torch.Tensor] = []
                sign_score_terms: list[torch.Tensor] = []
                clean_gate_cap_terms: list[torch.Tensor] = []
                mild_gate_floor_terms: list[torch.Tensor] = []
                gate_gap_margin_terms: list[torch.Tensor] = []
                residual_gap_terms: list[torch.Tensor] = []
                clean_score_anchor_terms: list[torch.Tensor] = []
                clean_action_anchor_terms: list[torch.Tensor] = []
                source_gate_residual_zero_terms: list[torch.Tensor] = []
                source_gate_distill_terms: list[torch.Tensor] = []
                source_sign_score_anchor_terms: list[torch.Tensor] = []
                source_clean_gate_preserve_terms: list[torch.Tensor] = []
                residual_logit_target_terms: list[torch.Tensor] = []
                drift_terms: list[torch.Tensor] = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_logit_clean = current_clean_outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    gate_residual_logit_corrupt = outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )

                    paired_losses = compute_paired_selective_gate_losses(
                        current_clean_gate[0],
                        corrupt_gate[0],
                        clean_gate_max=clean_gate_max,
                        mild_gate_min=mild_gate_min,
                        gate_gap_margin=gate_gap_margin,
                        residual_clean=gate_residual_clean[0] if paired_supervision else None,
                        residual_mild=gate_residual_corrupt[0] if paired_supervision else None,
                        residual_margin=residual_selective_margin if paired_supervision else None,
                    )
                    clean_gate_cap_terms.append(paired_losses["clean_gate_cap"])
                    mild_gate_floor_terms.append(paired_losses["mild_gate_floor"])
                    gate_gap_margin_terms.append(paired_losses["gate_gap_margin"])
                    if "residual_gap_margin" in paired_losses:
                        residual_gap_terms.append(paired_losses["residual_gap_margin"])

                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(
                        torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2))
                    )
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    if source_gate_distill_coef > 0.0:
                        source_gate_distill_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                        source_gate_distill_terms.append(
                            torch.mean((corrupt_gate[preserve_indices] - source_corrupt_gate[preserve_indices]).pow(2))
                        )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    if source_clean_gate_preserve_coef > 0.0:
                        source_clean_gate_preserve_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                    if residual_logit_target_loss_coef > 0.0:
                        residual_logit_target_terms.append(
                            compute_residual_logit_target_loss(
                                clean_residual_logit=gate_residual_logit_clean[0],
                                mild_residual_logit=gate_residual_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                residual_alpha=self.model.fusion_module.get_gate_residual_alpha_tensor(
                                    device=self.device,
                                    dtype=gate_residual_logit_clean.dtype,
                                ),
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                clip=float(residual_logit_target_clip) if residual_logit_target_clip is not None else None,
                                loss_type=residual_logit_target_loss_type,
                            )
                        )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                residual_gap_margin_loss = torch.stack(residual_gap_terms).mean() if residual_gap_terms else torch.zeros((), device=self.device)
                residual_logit_target_loss = (
                    torch.stack(residual_logit_target_terms).mean() if residual_logit_target_terms else torch.zeros((), device=self.device)
                )
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)
                residual_alpha = self.model.fusion_module.get_gate_residual_alpha_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                residual_alpha_l2_loss = residual_alpha.pow(2).mean()

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + residual_gap_margin_coef * residual_gap_margin_loss
                    + residual_logit_target_loss_coef * residual_logit_target_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + residual_alpha_l2_coef * residual_alpha_l2_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/residual_gap_margin"] += float(residual_gap_margin_loss.item())
                metrics["loss/residual_logit_target"] += float(residual_logit_target_loss.item())
                metrics["loss/residual_alpha_l2"] += float(residual_alpha_l2_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["train/residual_alpha"] += float(residual_alpha.detach().reshape(-1)[0].cpu().item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f23VisionActuatedResidualTrainer(Stage37f2SourceGateResidualLogitSelectiveTrainer):
    """V3.7f2.3 trainer with vision-focused actuation and gate input adapter micro-unfreeze."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        configure_trainable_gate_actuation_parameters(self.model, losses_cfg)
        self.model.freeze_attention_source_gate_residual_only(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(trainable_params, lr=float(config["train"]["learning_rate"]))

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/residual_gap_margin": 0.0,
            "loss/residual_logit_target": 0.0,
            "loss/vision_final_gate_target": 0.0,
            "loss/nonvision_gate_stability": 0.0,
            "loss/residual_alpha_l2": 0.0,
            "loss/gate_logit_offset_l2": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
            "train/residual_alpha": 0.0,
            "train/vision_logit_offset": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        residual_gap_margin_coef = float(losses_cfg.get("residual_gap_margin_loss_coef", gate_gap_margin_coef))
        residual_logit_target_loss_coef = float(losses_cfg.get("residual_logit_target_loss_coef", 0.0))
        residual_logit_target_loss_type = str(losses_cfg.get("residual_logit_target_loss_type", "huber"))
        residual_logit_target_clip = losses_cfg.get("residual_logit_target_clip")
        residual_alpha_l2_coef = float(losses_cfg.get("residual_alpha_l2_coef", 0.0))
        vision_final_gate_target_loss_coef = float(losses_cfg.get("vision_final_gate_target_loss_coef", 0.0))
        nonvision_gate_stability_loss_coef = float(losses_cfg.get("nonvision_gate_stability_loss_coef", 0.0))
        gate_logit_offset_l2_coef = float(losses_cfg.get("gate_logit_offset_l2_coef", 0.0))
        final_gate_target_loss_type = str(losses_cfg.get("final_gate_target_loss_type", "huber"))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        disable_vision_preserve = bool(losses_cfg.get("disable_vision_final_gate_teacher_preserve", False))
        preserve_indices = select_gate_teacher_preserve_indices(disable_vision_preserve)
        paired_supervision = bool(losses_cfg.get("paired_selective_gate_supervision", False))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms = []
                distance_gap_terms = []
                sign_score_terms = []
                clean_gate_cap_terms = []
                mild_gate_floor_terms = []
                gate_gap_margin_terms = []
                residual_gap_terms = []
                residual_logit_target_terms = []
                vision_final_gate_target_terms = []
                nonvision_gate_stability_terms = []
                clean_score_anchor_terms = []
                clean_action_anchor_terms = []
                source_gate_residual_zero_terms = []
                source_gate_distill_terms = []
                source_sign_score_anchor_terms = []
                source_clean_gate_preserve_terms = []
                drift_terms = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_logit_clean = current_clean_outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    gate_residual_logit_corrupt = outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )

                    paired_losses = compute_paired_selective_gate_losses(
                        current_clean_gate[0],
                        corrupt_gate[0],
                        clean_gate_max=clean_gate_max,
                        mild_gate_min=mild_gate_min,
                        gate_gap_margin=gate_gap_margin,
                        residual_clean=gate_residual_clean[0] if paired_supervision else None,
                        residual_mild=gate_residual_corrupt[0] if paired_supervision else None,
                        residual_margin=residual_selective_margin if paired_supervision else None,
                    )
                    clean_gate_cap_terms.append(paired_losses["clean_gate_cap"])
                    mild_gate_floor_terms.append(paired_losses["mild_gate_floor"])
                    gate_gap_margin_terms.append(paired_losses["gate_gap_margin"])
                    if "residual_gap_margin" in paired_losses:
                        residual_gap_terms.append(paired_losses["residual_gap_margin"])

                    if residual_logit_target_loss_coef > 0.0:
                        residual_logit_target_terms.append(
                            compute_residual_logit_target_loss(
                                clean_residual_logit=gate_residual_logit_clean[0],
                                mild_residual_logit=gate_residual_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                residual_alpha=self.model.fusion_module.get_gate_residual_alpha_tensor(
                                    device=self.device,
                                    dtype=gate_residual_logit_clean.dtype,
                                ).reshape(-1)[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                clip=float(residual_logit_target_clip) if residual_logit_target_clip is not None else None,
                                loss_type=residual_logit_target_loss_type,
                            )
                        )
                    if vision_final_gate_target_loss_coef > 0.0:
                        vision_final_gate_target_terms.append(
                            compute_final_gate_target_loss(
                                clean_gate=current_clean_gate[0],
                                mild_gate=corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                loss_type=final_gate_target_loss_type,
                            )
                        )
                    if nonvision_gate_stability_loss_coef > 0.0 and current_clean_gate.shape[0] > 1:
                        nonvision_gate_stability_terms.append(
                            0.5
                            * (
                                torch.mean((current_clean_gate[1:] - source_clean_gate[1:]).pow(2))
                                + torch.mean((corrupt_gate[1:] - source_corrupt_gate[1:]).pow(2))
                            )
                        )

                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    if source_gate_distill_coef > 0.0:
                        source_gate_distill_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                        source_gate_distill_terms.append(
                            torch.mean((corrupt_gate[preserve_indices] - source_corrupt_gate[preserve_indices]).pow(2))
                        )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    if source_clean_gate_preserve_coef > 0.0:
                        source_clean_gate_preserve_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                residual_gap_margin_loss = torch.stack(residual_gap_terms).mean() if residual_gap_terms else torch.zeros((), device=self.device)
                residual_logit_target_loss = torch.stack(residual_logit_target_terms).mean() if residual_logit_target_terms else torch.zeros((), device=self.device)
                vision_final_gate_target_loss = torch.stack(vision_final_gate_target_terms).mean() if vision_final_gate_target_terms else torch.zeros((), device=self.device)
                nonvision_gate_stability_loss = torch.stack(nonvision_gate_stability_terms).mean() if nonvision_gate_stability_terms else torch.zeros((), device=self.device)
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)
                residual_alpha = self.model.fusion_module.get_gate_residual_alpha_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                gate_logit_offset = self.model.fusion_module.get_gate_logit_offset_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                residual_alpha_l2_loss = residual_alpha.pow(2).mean()
                gate_logit_offset_l2_loss = gate_logit_offset.pow(2).mean()

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + residual_gap_margin_coef * residual_gap_margin_loss
                    + residual_logit_target_loss_coef * residual_logit_target_loss
                    + vision_final_gate_target_loss_coef * vision_final_gate_target_loss
                    + nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + residual_alpha_l2_coef * residual_alpha_l2_loss
                    + gate_logit_offset_l2_coef * gate_logit_offset_l2_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/residual_gap_margin"] += float(residual_gap_margin_loss.item())
                metrics["loss/residual_logit_target"] += float(residual_logit_target_loss.item())
                metrics["loss/vision_final_gate_target"] += float(vision_final_gate_target_loss.item())
                metrics["loss/nonvision_gate_stability"] += float(nonvision_gate_stability_loss.item())
                metrics["loss/residual_alpha_l2"] += float(residual_alpha_l2_loss.item())
                metrics["loss/gate_logit_offset_l2"] += float(gate_logit_offset_l2_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["train/residual_alpha"] += float(residual_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_logit_offset"] += float(gate_logit_offset.detach().reshape(-1)[0].cpu().item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f24GateRelativeMicroUnfreezeTrainer(Stage37f23VisionActuatedResidualTrainer):
    """V3.7f2.4 trainer with gate branch plus relative-gap-head micro-unfreeze."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            self._initial_relative_gap_head_params = []
            return
        self._initial_relative_gap_head_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.relative_gap_head.parameters()
        ]
        self.model.freeze_attention_source_gate_relative_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        relative_lr_scale = float(losses_cfg.get("relative_head_lr_scale", 0.5))
        relative_params = list(self.model.fusion_module.relative_gap_head.parameters())
        relative_param_ids = {id(parameter) for parameter in relative_params}
        gate_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in relative_param_ids
        ]
        param_groups = []
        if gate_params:
            param_groups.append({"params": gate_params, "lr": base_lr})
        if relative_params:
            param_groups.append({"params": relative_params, "lr": base_lr * relative_lr_scale})
        if not param_groups:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(param_groups)

    def _relative_head_delta_norm(self) -> torch.Tensor:
        if not self._initial_relative_gap_head_params:
            return torch.zeros((), device=self.device)
        current_params = list(self.model.fusion_module.relative_gap_head.parameters())
        deltas = [
            torch.sum((current - reference.to(device=current.device)).pow(2))
            for current, reference in zip(current_params, self._initial_relative_gap_head_params, strict=True)
        ]
        if not deltas:
            return torch.zeros((), device=self.device)
        return torch.sqrt(torch.stack(deltas).sum())

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/residual_gap_margin": 0.0,
            "loss/residual_logit_target": 0.0,
            "loss/vision_final_gate_target": 0.0,
            "loss/nonvision_gate_stability": 0.0,
            "loss/clean_relative_gap_anchor": 0.0,
            "loss/source_relative_gap_anchor": 0.0,
            "loss/vision_relative_gap_margin": 0.0,
            "loss/nonvision_relative_gap_stability": 0.0,
            "loss/relative_head_weight_l2": 0.0,
            "loss/residual_alpha_l2": 0.0,
            "loss/gate_logit_offset_l2": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
            "train/residual_alpha": 0.0,
            "train/vision_logit_offset": 0.0,
            "train/relative_head_delta_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        residual_gap_margin_coef = float(losses_cfg.get("residual_gap_margin_loss_coef", gate_gap_margin_coef))
        residual_logit_target_loss_coef = float(losses_cfg.get("residual_logit_target_loss_coef", 0.0))
        residual_logit_target_loss_type = str(losses_cfg.get("residual_logit_target_loss_type", "huber"))
        residual_logit_target_clip = losses_cfg.get("residual_logit_target_clip")
        residual_alpha_l2_coef = float(losses_cfg.get("residual_alpha_l2_coef", 0.0))
        vision_final_gate_target_loss_coef = float(losses_cfg.get("vision_final_gate_target_loss_coef", 0.0))
        nonvision_gate_stability_loss_coef = float(losses_cfg.get("nonvision_gate_stability_loss_coef", 0.0))
        gate_logit_offset_l2_coef = float(losses_cfg.get("gate_logit_offset_l2_coef", 0.0))
        final_gate_target_loss_type = str(losses_cfg.get("final_gate_target_loss_type", "huber"))
        clean_relative_gap_anchor_coef = float(losses_cfg.get("clean_relative_gap_anchor_loss_coef", 0.0))
        source_relative_gap_anchor_coef = float(losses_cfg.get("source_relative_gap_anchor_loss_coef", 0.0))
        vision_relative_gap_margin_coef = float(losses_cfg.get("vision_relative_gap_margin_loss_coef", 0.0))
        nonvision_relative_gap_stability_coef = float(losses_cfg.get("nonvision_relative_gap_stability_loss_coef", 0.0))
        relative_head_weight_l2_coef = float(losses_cfg.get("relative_head_weight_l2_coef", 0.0))
        relative_gap_anchor_loss_type = str(losses_cfg.get("relative_gap_anchor_loss_type", "mse"))
        relative_gap_margin = float(losses_cfg.get("vision_relative_gap_margin", gate_gap_margin))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        disable_vision_preserve = bool(losses_cfg.get("disable_vision_final_gate_teacher_preserve", False))
        preserve_indices = select_gate_teacher_preserve_indices(disable_vision_preserve)
        paired_supervision = bool(losses_cfg.get("paired_selective_gate_supervision", False))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms = []
                distance_gap_terms = []
                sign_score_terms = []
                clean_gate_cap_terms = []
                mild_gate_floor_terms = []
                gate_gap_margin_terms = []
                residual_gap_terms = []
                residual_logit_target_terms = []
                vision_final_gate_target_terms = []
                nonvision_gate_stability_terms = []
                clean_relative_gap_anchor_terms = []
                source_relative_gap_anchor_terms = []
                vision_relative_gap_margin_terms = []
                nonvision_relative_gap_stability_terms = []
                clean_score_anchor_terms = []
                clean_action_anchor_terms = []
                source_gate_residual_zero_terms = []
                source_gate_distill_terms = []
                source_sign_score_anchor_terms = []
                source_clean_gate_preserve_terms = []
                drift_terms = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_logit_clean = current_clean_outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    gate_residual_logit_corrupt = outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_relative_gap = current_clean_outputs["attention_relative_gap_scores"][batch_idx, time_idx]
                    corrupt_relative_gap = outputs["attention_relative_gap_scores"][batch_idx, time_idx]
                    clean_teacher_relative_gap = clean_teacher_outputs["attention_relative_gap_scores"][batch_idx, time_idx]
                    source_corrupt_relative_gap = sign_teacher_corrupt_outputs["attention_relative_gap_scores"][batch_idx, time_idx]
                    current_clean_relative_scores_before_bias = current_clean_outputs["attention_relative_scores_before_bias"][batch_idx, time_idx]
                    corrupt_relative_scores_before_bias = outputs["attention_relative_scores_before_bias"][batch_idx, time_idx]
                    source_clean_relative_scores_before_bias = sign_teacher_clean_outputs["attention_relative_scores_before_bias"][batch_idx, time_idx]
                    source_corrupt_relative_scores_before_bias = sign_teacher_corrupt_outputs["attention_relative_scores_before_bias"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )

                    paired_losses = compute_paired_selective_gate_losses(
                        current_clean_gate[0],
                        corrupt_gate[0],
                        clean_gate_max=clean_gate_max,
                        mild_gate_min=mild_gate_min,
                        gate_gap_margin=gate_gap_margin,
                        residual_clean=gate_residual_clean[0] if paired_supervision else None,
                        residual_mild=gate_residual_corrupt[0] if paired_supervision else None,
                        residual_margin=residual_selective_margin if paired_supervision else None,
                    )
                    clean_gate_cap_terms.append(paired_losses["clean_gate_cap"])
                    mild_gate_floor_terms.append(paired_losses["mild_gate_floor"])
                    gate_gap_margin_terms.append(paired_losses["gate_gap_margin"])
                    if "residual_gap_margin" in paired_losses:
                        residual_gap_terms.append(paired_losses["residual_gap_margin"])

                    if residual_logit_target_loss_coef > 0.0:
                        residual_logit_target_terms.append(
                            compute_residual_logit_target_loss(
                                clean_residual_logit=gate_residual_logit_clean[0],
                                mild_residual_logit=gate_residual_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                residual_alpha=self.model.fusion_module.get_gate_residual_alpha_tensor(
                                    device=self.device,
                                    dtype=gate_residual_logit_clean.dtype,
                                ).reshape(-1)[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                clip=float(residual_logit_target_clip) if residual_logit_target_clip is not None else None,
                                loss_type=residual_logit_target_loss_type,
                            )
                        )
                    if vision_final_gate_target_loss_coef > 0.0:
                        vision_final_gate_target_terms.append(
                            compute_final_gate_target_loss(
                                clean_gate=current_clean_gate[0],
                                mild_gate=corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                loss_type=final_gate_target_loss_type,
                            )
                        )
                    if nonvision_gate_stability_loss_coef > 0.0 and current_clean_gate.shape[0] > 1:
                        nonvision_gate_stability_terms.append(
                            0.5
                            * (
                                torch.mean((current_clean_gate[1:] - source_clean_gate[1:]).pow(2))
                                + torch.mean((corrupt_gate[1:] - source_corrupt_gate[1:]).pow(2))
                            )
                        )
                    if clean_relative_gap_anchor_coef > 0.0:
                        clean_relative_gap_anchor_terms.append(
                            compute_relative_gap_anchor_loss(
                                current_relative_gap=current_clean_relative_gap,
                                teacher_relative_gap=clean_teacher_relative_gap,
                                loss_type=relative_gap_anchor_loss_type,
                            )
                        )
                    if source_relative_gap_anchor_coef > 0.0:
                        source_relative_gap_anchor_terms.append(
                            compute_relative_gap_anchor_loss(
                                current_relative_gap=corrupt_relative_gap,
                                teacher_relative_gap=source_corrupt_relative_gap,
                                loss_type=relative_gap_anchor_loss_type,
                            )
                        )
                    if vision_relative_gap_margin_coef > 0.0:
                        vision_relative_gap_margin_terms.append(
                            compute_relative_gap_margin_loss(
                                clean_relative_gap=current_clean_relative_gap,
                                mild_relative_gap=corrupt_relative_gap,
                                margin=relative_gap_margin,
                            )
                        )
                    if nonvision_relative_gap_stability_coef > 0.0 and current_clean_relative_scores_before_bias.shape[0] > 1:
                        nonvision_relative_gap_stability_terms.append(
                            0.5
                            * (
                                torch.mean(
                                    (current_clean_relative_scores_before_bias[1:] - source_clean_relative_scores_before_bias[1:]).pow(2)
                                )
                                + torch.mean(
                                    (corrupt_relative_scores_before_bias[1:] - source_corrupt_relative_scores_before_bias[1:]).pow(2)
                                )
                            )
                        )

                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    if source_gate_distill_coef > 0.0:
                        source_gate_distill_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                        source_gate_distill_terms.append(
                            torch.mean((corrupt_gate[preserve_indices] - source_corrupt_gate[preserve_indices]).pow(2))
                        )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    if source_clean_gate_preserve_coef > 0.0:
                        source_clean_gate_preserve_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                residual_gap_margin_loss = torch.stack(residual_gap_terms).mean() if residual_gap_terms else torch.zeros((), device=self.device)
                residual_logit_target_loss = torch.stack(residual_logit_target_terms).mean() if residual_logit_target_terms else torch.zeros((), device=self.device)
                vision_final_gate_target_loss = torch.stack(vision_final_gate_target_terms).mean() if vision_final_gate_target_terms else torch.zeros((), device=self.device)
                nonvision_gate_stability_loss = torch.stack(nonvision_gate_stability_terms).mean() if nonvision_gate_stability_terms else torch.zeros((), device=self.device)
                clean_relative_gap_anchor_loss = torch.stack(clean_relative_gap_anchor_terms).mean() if clean_relative_gap_anchor_terms else torch.zeros((), device=self.device)
                source_relative_gap_anchor_loss = torch.stack(source_relative_gap_anchor_terms).mean() if source_relative_gap_anchor_terms else torch.zeros((), device=self.device)
                vision_relative_gap_margin_loss = torch.stack(vision_relative_gap_margin_terms).mean() if vision_relative_gap_margin_terms else torch.zeros((), device=self.device)
                nonvision_relative_gap_stability_loss = torch.stack(nonvision_relative_gap_stability_terms).mean() if nonvision_relative_gap_stability_terms else torch.zeros((), device=self.device)
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)
                residual_alpha = self.model.fusion_module.get_gate_residual_alpha_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                gate_logit_offset = self.model.fusion_module.get_gate_logit_offset_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                residual_alpha_l2_loss = residual_alpha.pow(2).mean()
                gate_logit_offset_l2_loss = gate_logit_offset.pow(2).mean()
                relative_head_weight_l2_loss = torch.zeros((), device=self.device)
                if relative_head_weight_l2_coef > 0.0:
                    relative_head_weight_l2_loss = torch.stack(
                        [parameter.pow(2).mean() for parameter in self.model.fusion_module.relative_gap_head.parameters()]
                    ).mean()

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + residual_gap_margin_coef * residual_gap_margin_loss
                    + residual_logit_target_loss_coef * residual_logit_target_loss
                    + vision_final_gate_target_loss_coef * vision_final_gate_target_loss
                    + nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss
                    + clean_relative_gap_anchor_coef * clean_relative_gap_anchor_loss
                    + source_relative_gap_anchor_coef * source_relative_gap_anchor_loss
                    + vision_relative_gap_margin_coef * vision_relative_gap_margin_loss
                    + nonvision_relative_gap_stability_coef * nonvision_relative_gap_stability_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + residual_alpha_l2_coef * residual_alpha_l2_loss
                    + gate_logit_offset_l2_coef * gate_logit_offset_l2_loss
                    + relative_head_weight_l2_coef * relative_head_weight_l2_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/residual_gap_margin"] += float(residual_gap_margin_loss.item())
                metrics["loss/residual_logit_target"] += float(residual_logit_target_loss.item())
                metrics["loss/vision_final_gate_target"] += float(vision_final_gate_target_loss.item())
                metrics["loss/nonvision_gate_stability"] += float(nonvision_gate_stability_loss.item())
                metrics["loss/clean_relative_gap_anchor"] += float(clean_relative_gap_anchor_loss.item())
                metrics["loss/source_relative_gap_anchor"] += float(source_relative_gap_anchor_loss.item())
                metrics["loss/vision_relative_gap_margin"] += float(vision_relative_gap_margin_loss.item())
                metrics["loss/nonvision_relative_gap_stability"] += float(nonvision_relative_gap_stability_loss.item())
                metrics["loss/relative_head_weight_l2"] += float(relative_head_weight_l2_loss.item())
                metrics["loss/residual_alpha_l2"] += float(residual_alpha_l2_loss.item())
                metrics["loss/gate_logit_offset_l2"] += float(gate_logit_offset_l2_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["train/residual_alpha"] += float(residual_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_logit_offset"] += float(gate_logit_offset.detach().reshape(-1)[0].cpu().item())
                metrics["train/relative_head_delta_norm"] += float(self._relative_head_delta_norm().detach().cpu().item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f25GateScoreAdapterMicroUnfreezeTrainer(Stage37f23VisionActuatedResidualTrainer):
    """V3.7f2.5 trainer with gate branch plus score-adapter micro-unfreeze."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            self._initial_score_adapter_params = []
            return
        self._initial_score_adapter_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.score_adapter.parameters()
        ]
        self.model.freeze_attention_source_gate_score_adapter_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        adapter_lr_scale = float(losses_cfg.get("score_adapter_lr_scale", 0.5))
        adapter_params = list(self.model.fusion_module.score_adapter.parameters())
        adapter_param_ids = {id(parameter) for parameter in adapter_params}
        gate_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in adapter_param_ids
        ]
        param_groups = []
        if gate_params:
            param_groups.append({"params": gate_params, "lr": base_lr})
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": base_lr * adapter_lr_scale})
        if not param_groups:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(param_groups)

    def _score_adapter_delta_norm(self) -> torch.Tensor:
        if not self._initial_score_adapter_params:
            return torch.zeros((), device=self.device)
        current_params = list(self.model.fusion_module.score_adapter.parameters())
        deltas = [
            torch.sum((current - reference.to(device=current.device)).pow(2))
            for current, reference in zip(current_params, self._initial_score_adapter_params, strict=True)
        ]
        if not deltas:
            return torch.zeros((), device=self.device)
        return torch.sqrt(torch.stack(deltas).sum())

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/residual_gap_margin": 0.0,
            "loss/residual_logit_target": 0.0,
            "loss/vision_final_gate_target": 0.0,
            "loss/nonvision_gate_stability": 0.0,
            "loss/clean_score_adapter_anchor": 0.0,
            "loss/source_score_adapter_anchor": 0.0,
            "loss/vision_adapter_delta_margin": 0.0,
            "loss/nonvision_adapter_stability": 0.0,
            "loss/score_adapter_weight_l2": 0.0,
            "loss/residual_alpha_l2": 0.0,
            "loss/gate_logit_offset_l2": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
            "train/residual_alpha": 0.0,
            "train/vision_logit_offset": 0.0,
            "train/score_adapter_delta_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        residual_gap_margin_coef = float(losses_cfg.get("residual_gap_margin_loss_coef", gate_gap_margin_coef))
        residual_logit_target_loss_coef = float(losses_cfg.get("residual_logit_target_loss_coef", 0.0))
        residual_logit_target_loss_type = str(losses_cfg.get("residual_logit_target_loss_type", "huber"))
        residual_logit_target_clip = losses_cfg.get("residual_logit_target_clip")
        residual_alpha_l2_coef = float(losses_cfg.get("residual_alpha_l2_coef", 0.0))
        vision_final_gate_target_loss_coef = float(losses_cfg.get("vision_final_gate_target_loss_coef", 0.0))
        nonvision_gate_stability_loss_coef = float(losses_cfg.get("nonvision_gate_stability_loss_coef", 0.0))
        gate_logit_offset_l2_coef = float(losses_cfg.get("gate_logit_offset_l2_coef", 0.0))
        final_gate_target_loss_type = str(losses_cfg.get("final_gate_target_loss_type", "huber"))
        clean_score_adapter_anchor_coef = float(losses_cfg.get("clean_score_adapter_anchor_loss_coef", 0.0))
        source_score_adapter_anchor_coef = float(losses_cfg.get("source_score_adapter_anchor_loss_coef", 0.0))
        vision_adapter_delta_margin_coef = float(losses_cfg.get("vision_adapter_delta_margin_loss_coef", 0.0))
        nonvision_adapter_stability_coef = float(losses_cfg.get("nonvision_adapter_stability_loss_coef", 0.0))
        score_adapter_weight_l2_coef = float(losses_cfg.get("score_adapter_weight_l2_coef", 0.0))
        score_adapter_anchor_loss_type = str(losses_cfg.get("score_adapter_anchor_loss_type", "mse"))
        vision_adapter_delta_margin = float(losses_cfg.get("vision_adapter_delta_margin", 0.05))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        disable_vision_preserve = bool(losses_cfg.get("disable_vision_final_gate_teacher_preserve", False))
        preserve_indices = select_gate_teacher_preserve_indices(disable_vision_preserve)
        paired_supervision = bool(losses_cfg.get("paired_selective_gate_supervision", False))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms = []
                distance_gap_terms = []
                sign_score_terms = []
                clean_gate_cap_terms = []
                mild_gate_floor_terms = []
                gate_gap_margin_terms = []
                residual_gap_terms = []
                residual_logit_target_terms = []
                vision_final_gate_target_terms = []
                nonvision_gate_stability_terms = []
                clean_score_adapter_anchor_terms = []
                source_score_adapter_anchor_terms = []
                vision_adapter_delta_margin_terms = []
                nonvision_adapter_stability_terms = []
                clean_score_anchor_terms = []
                clean_action_anchor_terms = []
                source_gate_residual_zero_terms = []
                source_gate_distill_terms = []
                source_sign_score_anchor_terms = []
                source_clean_gate_preserve_terms = []
                drift_terms = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_logit_clean = current_clean_outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    gate_residual_logit_corrupt = outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_adapter_delta = current_clean_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    corrupt_adapter_delta = outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    clean_teacher_adapter_delta = clean_teacher_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    source_corrupt_adapter_delta = sign_teacher_corrupt_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    source_clean_adapter_delta = sign_teacher_clean_outputs["attention_score_adapter_delta"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )

                    paired_losses = compute_paired_selective_gate_losses(
                        current_clean_gate[0],
                        corrupt_gate[0],
                        clean_gate_max=clean_gate_max,
                        mild_gate_min=mild_gate_min,
                        gate_gap_margin=gate_gap_margin,
                        residual_clean=gate_residual_clean[0] if paired_supervision else None,
                        residual_mild=gate_residual_corrupt[0] if paired_supervision else None,
                        residual_margin=residual_selective_margin if paired_supervision else None,
                    )
                    clean_gate_cap_terms.append(paired_losses["clean_gate_cap"])
                    mild_gate_floor_terms.append(paired_losses["mild_gate_floor"])
                    gate_gap_margin_terms.append(paired_losses["gate_gap_margin"])
                    if "residual_gap_margin" in paired_losses:
                        residual_gap_terms.append(paired_losses["residual_gap_margin"])

                    if residual_logit_target_loss_coef > 0.0:
                        residual_logit_target_terms.append(
                            compute_residual_logit_target_loss(
                                clean_residual_logit=gate_residual_logit_clean[0],
                                mild_residual_logit=gate_residual_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                residual_alpha=self.model.fusion_module.get_gate_residual_alpha_tensor(
                                    device=self.device,
                                    dtype=gate_residual_logit_clean.dtype,
                                ).reshape(-1)[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                clip=float(residual_logit_target_clip) if residual_logit_target_clip is not None else None,
                                loss_type=residual_logit_target_loss_type,
                            )
                        )
                    if vision_final_gate_target_loss_coef > 0.0:
                        vision_final_gate_target_terms.append(
                            compute_final_gate_target_loss(
                                clean_gate=current_clean_gate[0],
                                mild_gate=corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                loss_type=final_gate_target_loss_type,
                            )
                        )
                    if nonvision_gate_stability_loss_coef > 0.0 and current_clean_gate.shape[0] > 1:
                        nonvision_gate_stability_terms.append(
                            0.5
                            * (
                                torch.mean((current_clean_gate[1:] - source_clean_gate[1:]).pow(2))
                                + torch.mean((corrupt_gate[1:] - source_corrupt_gate[1:]).pow(2))
                            )
                        )
                    if clean_score_adapter_anchor_coef > 0.0:
                        clean_score_adapter_anchor_terms.append(
                            compute_score_adapter_anchor_loss(
                                current_delta=current_clean_adapter_delta,
                                teacher_delta=clean_teacher_adapter_delta,
                                loss_type=score_adapter_anchor_loss_type,
                            )
                        )
                    if source_score_adapter_anchor_coef > 0.0:
                        source_score_adapter_anchor_terms.append(
                            compute_score_adapter_anchor_loss(
                                current_delta=corrupt_adapter_delta,
                                teacher_delta=source_corrupt_adapter_delta,
                                loss_type=score_adapter_anchor_loss_type,
                            )
                        )
                    if vision_adapter_delta_margin_coef > 0.0:
                        vision_adapter_delta_margin_terms.append(
                            compute_adapter_delta_margin_loss(
                                clean_delta=current_clean_adapter_delta[0].unsqueeze(0),
                                mild_delta=corrupt_adapter_delta[0].unsqueeze(0),
                                margin=vision_adapter_delta_margin,
                            )
                        )
                    if nonvision_adapter_stability_coef > 0.0 and current_clean_adapter_delta.shape[0] > 1:
                        nonvision_adapter_stability_terms.append(
                            0.5
                            * (
                                torch.mean((current_clean_adapter_delta[1:] - source_clean_adapter_delta[1:]).pow(2))
                                + torch.mean((corrupt_adapter_delta[1:] - source_corrupt_adapter_delta[1:]).pow(2))
                            )
                        )

                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    if source_gate_distill_coef > 0.0:
                        source_gate_distill_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                        source_gate_distill_terms.append(
                            torch.mean((corrupt_gate[preserve_indices] - source_corrupt_gate[preserve_indices]).pow(2))
                        )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    if source_clean_gate_preserve_coef > 0.0:
                        source_clean_gate_preserve_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                residual_gap_margin_loss = torch.stack(residual_gap_terms).mean() if residual_gap_terms else torch.zeros((), device=self.device)
                residual_logit_target_loss = torch.stack(residual_logit_target_terms).mean() if residual_logit_target_terms else torch.zeros((), device=self.device)
                vision_final_gate_target_loss = torch.stack(vision_final_gate_target_terms).mean() if vision_final_gate_target_terms else torch.zeros((), device=self.device)
                nonvision_gate_stability_loss = torch.stack(nonvision_gate_stability_terms).mean() if nonvision_gate_stability_terms else torch.zeros((), device=self.device)
                clean_score_adapter_anchor_loss = torch.stack(clean_score_adapter_anchor_terms).mean() if clean_score_adapter_anchor_terms else torch.zeros((), device=self.device)
                source_score_adapter_anchor_loss = torch.stack(source_score_adapter_anchor_terms).mean() if source_score_adapter_anchor_terms else torch.zeros((), device=self.device)
                vision_adapter_delta_margin_loss = torch.stack(vision_adapter_delta_margin_terms).mean() if vision_adapter_delta_margin_terms else torch.zeros((), device=self.device)
                nonvision_adapter_stability_loss = torch.stack(nonvision_adapter_stability_terms).mean() if nonvision_adapter_stability_terms else torch.zeros((), device=self.device)
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)
                residual_alpha = self.model.fusion_module.get_gate_residual_alpha_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                gate_logit_offset = self.model.fusion_module.get_gate_logit_offset_tensor(
                    device=self.device,
                    dtype=clean_gate_cap_loss.dtype,
                )
                residual_alpha_l2_loss = residual_alpha.pow(2).mean()
                gate_logit_offset_l2_loss = gate_logit_offset.pow(2).mean()
                score_adapter_weight_l2_loss = torch.zeros((), device=self.device)
                if score_adapter_weight_l2_coef > 0.0:
                    score_adapter_weight_l2_loss = torch.stack(
                        [parameter.pow(2).mean() for parameter in self.model.fusion_module.score_adapter.parameters()]
                    ).mean()

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + residual_gap_margin_coef * residual_gap_margin_loss
                    + residual_logit_target_loss_coef * residual_logit_target_loss
                    + vision_final_gate_target_loss_coef * vision_final_gate_target_loss
                    + nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss
                    + clean_score_adapter_anchor_coef * clean_score_adapter_anchor_loss
                    + source_score_adapter_anchor_coef * source_score_adapter_anchor_loss
                    + vision_adapter_delta_margin_coef * vision_adapter_delta_margin_loss
                    + nonvision_adapter_stability_coef * nonvision_adapter_stability_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + residual_alpha_l2_coef * residual_alpha_l2_loss
                    + gate_logit_offset_l2_coef * gate_logit_offset_l2_loss
                    + score_adapter_weight_l2_coef * score_adapter_weight_l2_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/residual_gap_margin"] += float(residual_gap_margin_loss.item())
                metrics["loss/residual_logit_target"] += float(residual_logit_target_loss.item())
                metrics["loss/vision_final_gate_target"] += float(vision_final_gate_target_loss.item())
                metrics["loss/nonvision_gate_stability"] += float(nonvision_gate_stability_loss.item())
                metrics["loss/clean_score_adapter_anchor"] += float(clean_score_adapter_anchor_loss.item())
                metrics["loss/source_score_adapter_anchor"] += float(source_score_adapter_anchor_loss.item())
                metrics["loss/vision_adapter_delta_margin"] += float(vision_adapter_delta_margin_loss.item())
                metrics["loss/nonvision_adapter_stability"] += float(nonvision_adapter_stability_loss.item())
                metrics["loss/score_adapter_weight_l2"] += float(score_adapter_weight_l2_loss.item())
                metrics["loss/residual_alpha_l2"] += float(residual_alpha_l2_loss.item())
                metrics["loss/gate_logit_offset_l2"] += float(gate_logit_offset_l2_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["train/residual_alpha"] += float(residual_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_logit_offset"] += float(gate_logit_offset.detach().reshape(-1)[0].cpu().item())
                metrics["train/score_adapter_delta_norm"] += float(self._score_adapter_delta_norm().detach().cpu().item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f26AdapterGateCouplingTrainer(Stage37f23VisionActuatedResidualTrainer):
    """V3.7f2.6 trainer with explicit adapter-delta to gate coupling."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super().__init__(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        self._train_score_adapter = bool(losses_cfg.get("train_score_adapter", False))
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            self._initial_score_adapter_params = []
            self._initial_adapter_gate_coupler_params = []
            return
        self._initial_score_adapter_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.score_adapter.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self._initial_adapter_gate_coupler_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.reliability_bias_gate_adapter_coupler.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self.model.freeze_attention_source_gate_adapter_coupled_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True)),
            train_score_adapter=self._train_score_adapter,
        )
        base_lr = float(config["train"]["learning_rate"])
        score_adapter_lr_scale = float(losses_cfg.get("score_adapter_lr_scale", 0.25))
        score_adapter_params = list(self.model.fusion_module.score_adapter.parameters()) if self._train_score_adapter else []
        score_adapter_param_ids = {id(parameter) for parameter in score_adapter_params}
        gate_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in score_adapter_param_ids
        ]
        param_groups = []
        if gate_params:
            param_groups.append({"params": gate_params, "lr": base_lr})
        if score_adapter_params:
            param_groups.append({"params": score_adapter_params, "lr": base_lr * score_adapter_lr_scale})
        if not param_groups:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(param_groups)

    def _score_adapter_delta_norm(self) -> torch.Tensor:
        if not self._initial_score_adapter_params:
            return torch.zeros((), device=self.device)
        current_params = [
            parameter
            for parameter in self.model.fusion_module.score_adapter.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        deltas = [
            torch.sum((current - reference.to(device=current.device)).pow(2))
            for current, reference in zip(current_params, self._initial_score_adapter_params, strict=True)
        ]
        if not deltas:
            return torch.zeros((), device=self.device)
        return torch.sqrt(torch.stack(deltas).sum())

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/residual_gap_margin": 0.0,
            "loss/residual_logit_target": 0.0,
            "loss/vision_final_gate_target": 0.0,
            "loss/nonvision_gate_stability": 0.0,
            "loss/vision_adapter_gate_target": 0.0,
            "loss/vision_adapter_gate_margin": 0.0,
            "loss/nonvision_adapter_gate_stability": 0.0,
            "loss/adapter_gate_weight_l2": 0.0,
            "loss/clean_score_adapter_anchor": 0.0,
            "loss/source_score_adapter_anchor": 0.0,
            "loss/score_adapter_weight_l2": 0.0,
            "loss/residual_alpha_l2": 0.0,
            "loss/adapter_gate_alpha_l2": 0.0,
            "loss/gate_logit_offset_l2": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/grad_norm": 0.0,
            "train/residual_alpha": 0.0,
            "train/vision_logit_offset": 0.0,
            "train/adapter_gate_alpha": 0.0,
            "train/score_adapter_delta_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        residual_gap_margin_coef = float(losses_cfg.get("residual_gap_margin_loss_coef", gate_gap_margin_coef))
        residual_logit_target_loss_coef = float(losses_cfg.get("residual_logit_target_loss_coef", 0.0))
        residual_logit_target_loss_type = str(losses_cfg.get("residual_logit_target_loss_type", "huber"))
        residual_logit_target_clip = losses_cfg.get("residual_logit_target_clip")
        residual_alpha_l2_coef = float(losses_cfg.get("residual_alpha_l2_coef", 0.0))
        adapter_gate_alpha_l2_coef = float(losses_cfg.get("adapter_gate_alpha_l2_coef", 0.0))
        vision_final_gate_target_loss_coef = float(losses_cfg.get("vision_final_gate_target_loss_coef", 0.0))
        nonvision_gate_stability_loss_coef = float(losses_cfg.get("nonvision_gate_stability_loss_coef", 0.0))
        gate_logit_offset_l2_coef = float(losses_cfg.get("gate_logit_offset_l2_coef", 0.0))
        final_gate_target_loss_type = str(losses_cfg.get("final_gate_target_loss_type", "huber"))
        vision_adapter_gate_target_loss_coef = float(losses_cfg.get("vision_adapter_gate_target_loss_coef", 0.0))
        vision_adapter_gate_margin_loss_coef = float(losses_cfg.get("vision_adapter_gate_margin_loss_coef", 0.0))
        nonvision_adapter_gate_stability_loss_coef = float(losses_cfg.get("nonvision_adapter_gate_stability_loss_coef", 0.0))
        adapter_gate_weight_l2_coef = float(losses_cfg.get("adapter_gate_weight_l2_coef", 0.0))
        adapter_gate_target_fraction = float(losses_cfg.get("adapter_gate_target_fraction", 0.5))
        adapter_gate_target_loss_type = str(losses_cfg.get("adapter_gate_target_loss_type", "huber"))
        vision_adapter_gate_margin = float(losses_cfg.get("vision_adapter_gate_margin", 0.2))
        clean_score_adapter_anchor_coef = float(losses_cfg.get("clean_score_adapter_anchor_loss_coef", 0.0))
        source_score_adapter_anchor_coef = float(losses_cfg.get("source_score_adapter_anchor_loss_coef", 0.0))
        score_adapter_weight_l2_coef = float(losses_cfg.get("score_adapter_weight_l2_coef", 0.0))
        score_adapter_anchor_loss_type = str(losses_cfg.get("score_adapter_anchor_loss_type", "mse"))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        disable_vision_preserve = bool(losses_cfg.get("disable_vision_final_gate_teacher_preserve", False))
        preserve_indices = select_gate_teacher_preserve_indices(disable_vision_preserve)
        paired_supervision = bool(losses_cfg.get("paired_selective_gate_supervision", False))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )

                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms = []
                distance_gap_terms = []
                sign_score_terms = []
                clean_gate_cap_terms = []
                mild_gate_floor_terms = []
                gate_gap_margin_terms = []
                residual_gap_terms = []
                residual_logit_target_terms = []
                vision_final_gate_target_terms = []
                nonvision_gate_stability_terms = []
                vision_adapter_gate_target_terms = []
                vision_adapter_gate_margin_terms = []
                nonvision_adapter_gate_stability_terms = []
                clean_score_adapter_anchor_terms = []
                source_score_adapter_anchor_terms = []
                clean_score_anchor_terms = []
                clean_action_anchor_terms = []
                source_gate_residual_zero_terms = []
                source_gate_distill_terms = []
                source_sign_score_anchor_terms = []
                source_clean_gate_preserve_terms = []
                drift_terms = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_logit_clean = current_clean_outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    gate_residual_logit_corrupt = outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    adapter_gate_logit_clean = current_clean_outputs["attention_bias_gate_adapter_logits"][batch_idx, time_idx]
                    adapter_gate_logit_corrupt = outputs["attention_bias_gate_adapter_logits"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_adapter_delta = current_clean_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    corrupt_adapter_delta = outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    clean_teacher_adapter_delta = clean_teacher_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    source_corrupt_adapter_delta = sign_teacher_corrupt_outputs["attention_score_adapter_delta"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(
                        F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild))
                    )
                    distance_gap_terms.append(
                        F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild))
                    )
                    sign_score_terms.append(
                        F.relu(
                            torch.as_tensor(sign_score_margin, device=self.device)
                            - (sign_teacher_clean_scores[0] - corrupt_scores[0])
                        )
                    )

                    paired_losses = compute_paired_selective_gate_losses(
                        current_clean_gate[0],
                        corrupt_gate[0],
                        clean_gate_max=clean_gate_max,
                        mild_gate_min=mild_gate_min,
                        gate_gap_margin=gate_gap_margin,
                        residual_clean=gate_residual_clean[0] if paired_supervision else None,
                        residual_mild=gate_residual_corrupt[0] if paired_supervision else None,
                        residual_margin=residual_selective_margin if paired_supervision else None,
                    )
                    clean_gate_cap_terms.append(paired_losses["clean_gate_cap"])
                    mild_gate_floor_terms.append(paired_losses["mild_gate_floor"])
                    gate_gap_margin_terms.append(paired_losses["gate_gap_margin"])
                    if "residual_gap_margin" in paired_losses:
                        residual_gap_terms.append(paired_losses["residual_gap_margin"])

                    if residual_logit_target_loss_coef > 0.0:
                        residual_logit_target_terms.append(
                            compute_residual_logit_target_loss(
                                clean_residual_logit=gate_residual_logit_clean[0],
                                mild_residual_logit=gate_residual_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                residual_alpha=self.model.fusion_module.get_gate_residual_alpha_tensor(
                                    device=self.device,
                                    dtype=gate_residual_logit_clean.dtype,
                                ).reshape(-1)[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                clip=float(residual_logit_target_clip) if residual_logit_target_clip is not None else None,
                                loss_type=residual_logit_target_loss_type,
                            )
                        )
                    if vision_final_gate_target_loss_coef > 0.0:
                        vision_final_gate_target_terms.append(
                            compute_final_gate_target_loss(
                                clean_gate=current_clean_gate[0],
                                mild_gate=corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                loss_type=final_gate_target_loss_type,
                            )
                        )
                    if nonvision_gate_stability_loss_coef > 0.0 and current_clean_gate.shape[0] > 1:
                        nonvision_gate_stability_terms.append(
                            0.5
                            * (
                                torch.mean((current_clean_gate[1:] - source_clean_gate[1:]).pow(2))
                                + torch.mean((corrupt_gate[1:] - source_corrupt_gate[1:]).pow(2))
                            )
                        )
                    if vision_adapter_gate_target_loss_coef > 0.0:
                        vision_adapter_gate_target_terms.append(
                            compute_adapter_gate_target_loss(
                                clean_adapter_gate_logit=adapter_gate_logit_clean[0],
                                mild_adapter_gate_logit=adapter_gate_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                target_fraction=adapter_gate_target_fraction,
                                loss_type=adapter_gate_target_loss_type,
                            )
                        )
                    if vision_adapter_gate_margin_loss_coef > 0.0:
                        vision_adapter_gate_margin_terms.append(
                            compute_adapter_gate_margin_loss(
                                clean_adapter_gate_logit=adapter_gate_logit_clean[0].unsqueeze(0),
                                mild_adapter_gate_logit=adapter_gate_logit_corrupt[0].unsqueeze(0),
                                margin=vision_adapter_gate_margin,
                            )
                        )
                    if nonvision_adapter_gate_stability_loss_coef > 0.0 and adapter_gate_logit_clean.shape[0] > 1:
                        nonvision_adapter_gate_stability_terms.append(
                            0.5
                            * (
                                torch.mean(adapter_gate_logit_clean[1:].pow(2))
                                + torch.mean(adapter_gate_logit_corrupt[1:].pow(2))
                            )
                        )
                    if self._train_score_adapter and clean_score_adapter_anchor_coef > 0.0:
                        clean_score_adapter_anchor_terms.append(
                            compute_score_adapter_anchor_loss(
                                current_delta=current_clean_adapter_delta,
                                teacher_delta=clean_teacher_adapter_delta,
                                loss_type=score_adapter_anchor_loss_type,
                            )
                        )
                    if self._train_score_adapter and source_score_adapter_anchor_coef > 0.0:
                        source_score_adapter_anchor_terms.append(
                            compute_score_adapter_anchor_loss(
                                current_delta=corrupt_adapter_delta,
                                teacher_delta=source_corrupt_adapter_delta,
                                loss_type=score_adapter_anchor_loss_type,
                            )
                        )

                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    if source_gate_distill_coef > 0.0:
                        source_gate_distill_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                        source_gate_distill_terms.append(
                            torch.mean((corrupt_gate[preserve_indices] - source_corrupt_gate[preserve_indices]).pow(2))
                        )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    if source_clean_gate_preserve_coef > 0.0:
                        source_clean_gate_preserve_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                residual_gap_margin_loss = torch.stack(residual_gap_terms).mean() if residual_gap_terms else torch.zeros((), device=self.device)
                residual_logit_target_loss = torch.stack(residual_logit_target_terms).mean() if residual_logit_target_terms else torch.zeros((), device=self.device)
                vision_final_gate_target_loss = torch.stack(vision_final_gate_target_terms).mean() if vision_final_gate_target_terms else torch.zeros((), device=self.device)
                nonvision_gate_stability_loss = torch.stack(nonvision_gate_stability_terms).mean() if nonvision_gate_stability_terms else torch.zeros((), device=self.device)
                vision_adapter_gate_target_loss = torch.stack(vision_adapter_gate_target_terms).mean() if vision_adapter_gate_target_terms else torch.zeros((), device=self.device)
                vision_adapter_gate_margin_loss = torch.stack(vision_adapter_gate_margin_terms).mean() if vision_adapter_gate_margin_terms else torch.zeros((), device=self.device)
                nonvision_adapter_gate_stability_loss = torch.stack(nonvision_adapter_gate_stability_terms).mean() if nonvision_adapter_gate_stability_terms else torch.zeros((), device=self.device)
                clean_score_adapter_anchor_loss = torch.stack(clean_score_adapter_anchor_terms).mean() if clean_score_adapter_anchor_terms else torch.zeros((), device=self.device)
                source_score_adapter_anchor_loss = torch.stack(source_score_adapter_anchor_terms).mean() if source_score_adapter_anchor_terms else torch.zeros((), device=self.device)
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)
                residual_alpha = self.model.fusion_module.get_gate_residual_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                adapter_gate_alpha = self.model.fusion_module.get_adapter_gate_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_logit_offset = self.model.fusion_module.get_gate_logit_offset_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                residual_alpha_l2_loss = residual_alpha.pow(2).mean()
                adapter_gate_alpha_l2_loss = adapter_gate_alpha.pow(2).mean()
                gate_logit_offset_l2_loss = gate_logit_offset.pow(2).mean()
                adapter_gate_weight_l2_loss = torch.stack(
                    [parameter.pow(2).mean() for parameter in self.model.fusion_module.reliability_bias_gate_adapter_coupler.parameters()]
                ).mean()
                score_adapter_weight_l2_loss = torch.zeros((), device=self.device)
                if self._train_score_adapter and score_adapter_weight_l2_coef > 0.0:
                    score_adapter_weight_l2_loss = torch.stack(
                        [parameter.pow(2).mean() for parameter in self.model.fusion_module.score_adapter.parameters()]
                    ).mean()

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + residual_gap_margin_coef * residual_gap_margin_loss
                    + residual_logit_target_loss_coef * residual_logit_target_loss
                    + vision_final_gate_target_loss_coef * vision_final_gate_target_loss
                    + nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss
                    + vision_adapter_gate_target_loss_coef * vision_adapter_gate_target_loss
                    + vision_adapter_gate_margin_loss_coef * vision_adapter_gate_margin_loss
                    + nonvision_adapter_gate_stability_loss_coef * nonvision_adapter_gate_stability_loss
                    + clean_score_adapter_anchor_coef * clean_score_adapter_anchor_loss
                    + source_score_adapter_anchor_coef * source_score_adapter_anchor_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + residual_alpha_l2_coef * residual_alpha_l2_loss
                    + adapter_gate_alpha_l2_coef * adapter_gate_alpha_l2_loss
                    + gate_logit_offset_l2_coef * gate_logit_offset_l2_loss
                    + adapter_gate_weight_l2_coef * adapter_gate_weight_l2_loss
                    + score_adapter_weight_l2_coef * score_adapter_weight_l2_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/residual_gap_margin"] += float(residual_gap_margin_loss.item())
                metrics["loss/residual_logit_target"] += float(residual_logit_target_loss.item())
                metrics["loss/vision_final_gate_target"] += float(vision_final_gate_target_loss.item())
                metrics["loss/nonvision_gate_stability"] += float(nonvision_gate_stability_loss.item())
                metrics["loss/vision_adapter_gate_target"] += float(vision_adapter_gate_target_loss.item())
                metrics["loss/vision_adapter_gate_margin"] += float(vision_adapter_gate_margin_loss.item())
                metrics["loss/nonvision_adapter_gate_stability"] += float(nonvision_adapter_gate_stability_loss.item())
                metrics["loss/adapter_gate_weight_l2"] += float(adapter_gate_weight_l2_loss.item())
                metrics["loss/clean_score_adapter_anchor"] += float(clean_score_adapter_anchor_loss.item())
                metrics["loss/source_score_adapter_anchor"] += float(source_score_adapter_anchor_loss.item())
                metrics["loss/score_adapter_weight_l2"] += float(score_adapter_weight_l2_loss.item())
                metrics["loss/residual_alpha_l2"] += float(residual_alpha_l2_loss.item())
                metrics["loss/adapter_gate_alpha_l2"] += float(adapter_gate_alpha_l2_loss.item())
                metrics["loss/gate_logit_offset_l2"] += float(gate_logit_offset_l2_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["train/residual_alpha"] += float(residual_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_logit_offset"] += float(gate_logit_offset.detach().reshape(-1)[0].cpu().item())
                metrics["train/adapter_gate_alpha"] += float(adapter_gate_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/score_adapter_delta_norm"] += float(self._score_adapter_delta_norm().detach().cpu().item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f27AdapterGateReadoutTrainer(Stage37f26AdapterGateCouplingTrainer):
    """V3.7f2.7 trainer with dedicated adapter readout adapter."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f26AdapterGateCouplingTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        self._train_score_adapter = bool(losses_cfg.get("train_score_adapter", False))
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            self._initial_score_adapter_params = []
            self._initial_adapter_gate_coupler_params = []
            self._initial_adapter_gate_input_adapter_params = []
            return
        self._initial_score_adapter_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.score_adapter.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self._initial_adapter_gate_coupler_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.reliability_bias_gate_adapter_coupler.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self._initial_adapter_gate_input_adapter_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.reliability_bias_gate_adapter_input_adapter.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self.model.freeze_attention_source_gate_adapter_readout_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True)),
            train_score_adapter=self._train_score_adapter,
        )
        base_lr = float(config["train"]["learning_rate"])
        score_adapter_lr_scale = float(losses_cfg.get("score_adapter_lr_scale", 0.20))
        score_adapter_params = list(self.model.fusion_module.score_adapter.parameters()) if self._train_score_adapter else []
        score_adapter_param_ids = {id(parameter) for parameter in score_adapter_params}
        gate_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in score_adapter_param_ids
        ]
        param_groups = []
        if gate_params:
            param_groups.append({"params": gate_params, "lr": base_lr})
        if score_adapter_params:
            param_groups.append({"params": score_adapter_params, "lr": base_lr * score_adapter_lr_scale})
        if not param_groups:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(param_groups)

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/vision_pairgap_route": 0.0,
            "loss/vision_pairgap_distance": 0.0,
            "loss/vision_attention_score_margin": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/residual_gap_margin": 0.0,
            "loss/residual_logit_target": 0.0,
            "loss/vision_final_gate_target": 0.0,
            "loss/nonvision_gate_stability": 0.0,
            "loss/vision_adapter_readout_target": 0.0,
            "loss/vision_adapter_readout_margin": 0.0,
            "loss/adapter_gate_projection_norm_floor": 0.0,
            "loss/nonvision_adapter_readout_stability": 0.0,
            "loss/adapter_gate_input_adapter_weight_l2": 0.0,
            "loss/adapter_gate_weight_l2": 0.0,
            "loss/clean_score_adapter_anchor": 0.0,
            "loss/source_score_adapter_anchor": 0.0,
            "loss/score_adapter_weight_l2": 0.0,
            "loss/residual_alpha_l2": 0.0,
            "loss/adapter_gate_alpha_l2": 0.0,
            "loss/gate_logit_offset_l2": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_sign_score_anchor": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/attention_score_drift": 0.0,
            "loss/vision_final_delta_logit_target": 0.0,
            "loss/vision_final_delta_margin": 0.0,
            "loss/nonvision_final_delta_stability": 0.0,
            "loss/gate_output_scale_l2": 0.0,
            "loss/vision_threshold_shift_target": 0.0,
            "loss/vision_threshold_shift_margin": 0.0,
            "loss/vision_threshold_shift_polarity": 0.0,
            "loss/nonvision_threshold_shift_stability": 0.0,
            "loss/threshold_shift_head_weight_l2": 0.0,
            "loss/gate_threshold_shift_alpha_l2": 0.0,
            "loss/gate_source_logit_bias_l2": 0.0,
            "loss/vision_common_cancel_target": 0.0,
            "loss/vision_contrast_shift_target": 0.0,
            "loss/vision_contrast_shift_margin": 0.0,
            "loss/vision_contrast_shift_polarity": 0.0,
            "loss/nonvision_common_cancel_stability": 0.0,
            "loss/nonvision_contrast_shift_stability": 0.0,
            "loss/common_cancel_head_weight_l2": 0.0,
            "loss/contrast_shift_head_weight_l2": 0.0,
            "loss/vision_common_cancel_overreach": 0.0,
            "loss/vision_mild_rescue_target": 0.0,
            "loss/vision_mild_rescue_margin": 0.0,
            "loss/vision_clean_rescue_zero": 0.0,
            "loss/vision_common_cancel_excess": 0.0,
            "loss/vision_mild_rescue_gate_bce": 0.0,
            "loss/vision_mild_rescue_gate_margin": 0.0,
            "loss/vision_mild_rescue_logit_target": 0.0,
            "loss/vision_clean_rescue_gate_ceiling": 0.0,
            "loss/nonvision_mild_rescue_stability": 0.0,
            "loss/mild_rescue_gate_head_weight_l2": 0.0,
            "loss/mild_rescue_value_head_weight_l2": 0.0,
            "loss/mild_evidence_input_adapter_weight_l2": 0.0,
            "loss/mild_value_input_adapter_weight_l2": 0.0,
            "loss/vision_cancel_clean_full_excess": 0.0,
            "loss/vision_cancel_mild_fractional_excess": 0.0,
            "loss/vision_cancel_gate_floor": 0.0,
            "loss/vision_selector_bce": 0.0,
            "loss/vision_selector_pair_margin": 0.0,
            "loss/vision_clean_selector_ceiling": 0.0,
            "loss/nonvision_selector_stability": 0.0,
            "loss/gate_common_cancel_alpha_l2": 0.0,
            "loss/gate_contrast_shift_alpha_l2": 0.0,
            "loss/gate_mild_rescue_alpha_l2": 0.0,
            "loss/grad_norm": 0.0,
            "train/residual_alpha": 0.0,
            "train/vision_logit_offset": 0.0,
            "train/adapter_gate_alpha": 0.0,
            "train/vision_gate_output_scale": 0.0,
            "train/vision_threshold_shift_alpha": 0.0,
            "train/vision_source_logit_bias": 0.0,
            "train/vision_common_cancel_alpha": 0.0,
            "train/vision_contrast_shift_alpha": 0.0,
            "train/vision_mild_rescue_alpha": 0.0,
            "train/score_adapter_delta_norm": 0.0,
            "train/adapter_gate_hidden_norm": 0.0,
        }
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        route_coef = float(losses_cfg.get("vision_pairgap_route_loss_coef", 0.0))
        distance_coef = float(losses_cfg.get("vision_pairgap_distance_loss_coef", 0.0))
        route_margin = float(losses_cfg.get("vision_pairgap_route_margin", 0.0))
        distance_margin = float(losses_cfg.get("vision_pairgap_distance_margin", 0.0))
        sign_score_coef = float(losses_cfg.get("vision_attention_score_margin_loss_coef", 0.0))
        sign_score_margin = float(losses_cfg.get("vision_attention_score_margin", 0.0))
        clean_gate_cap_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_score_anchor_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_coef = float(losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0))
        source_gate_distill_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_sign_score_anchor_coef = float(losses_cfg.get("source_sign_score_anchor_loss_coef", 0.0))
        source_clean_gate_preserve_coef = float(losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0))
        residual_selective_margin = float(losses_cfg.get("residual_selective_margin", 0.01))
        residual_gap_margin_coef = float(losses_cfg.get("residual_gap_margin_loss_coef", gate_gap_margin_coef))
        residual_logit_target_loss_coef = float(losses_cfg.get("residual_logit_target_loss_coef", 0.0))
        residual_logit_target_loss_type = str(losses_cfg.get("residual_logit_target_loss_type", "huber"))
        residual_logit_target_clip = losses_cfg.get("residual_logit_target_clip")
        residual_alpha_l2_coef = float(losses_cfg.get("residual_alpha_l2_coef", 0.0))
        adapter_gate_alpha_l2_coef = float(losses_cfg.get("adapter_gate_alpha_l2_coef", 0.0))
        vision_final_gate_target_loss_coef = float(losses_cfg.get("vision_final_gate_target_loss_coef", 0.0))
        nonvision_gate_stability_loss_coef = float(losses_cfg.get("nonvision_gate_stability_loss_coef", 0.0))
        gate_logit_offset_l2_coef = float(losses_cfg.get("gate_logit_offset_l2_coef", 0.0))
        final_gate_target_loss_type = str(losses_cfg.get("final_gate_target_loss_type", "huber"))
        vision_adapter_readout_target_loss_coef = float(losses_cfg.get("vision_adapter_readout_target_loss_coef", 0.0))
        vision_adapter_readout_margin_loss_coef = float(losses_cfg.get("vision_adapter_readout_margin_loss_coef", 0.0))
        adapter_gate_projection_norm_floor_loss_coef = float(losses_cfg.get("adapter_gate_projection_norm_floor_loss_coef", 0.0))
        nonvision_adapter_readout_stability_loss_coef = float(losses_cfg.get("nonvision_adapter_readout_stability_loss_coef", 0.0))
        adapter_gate_input_adapter_weight_l2_coef = float(losses_cfg.get("adapter_gate_input_adapter_weight_l2_coef", 0.0))
        adapter_gate_weight_l2_coef = float(losses_cfg.get("adapter_gate_weight_l2_coef", 0.0))
        adapter_target_fraction = float(losses_cfg.get("adapter_target_fraction", 0.35))
        vision_adapter_readout_margin = float(losses_cfg.get("vision_adapter_readout_margin", 0.06))
        vision_adapter_projection_norm_min = float(losses_cfg.get("vision_adapter_projection_norm_min", 0.5))
        clean_score_adapter_anchor_coef = float(losses_cfg.get("clean_score_adapter_anchor_loss_coef", 0.0))
        source_score_adapter_anchor_coef = float(losses_cfg.get("source_score_adapter_anchor_loss_coef", 0.0))
        score_adapter_weight_l2_coef = float(losses_cfg.get("score_adapter_weight_l2_coef", 0.0))
        score_adapter_anchor_loss_type = str(losses_cfg.get("score_adapter_anchor_loss_type", "mse"))
        vision_final_delta_logit_target_loss_coef = float(losses_cfg.get("vision_final_delta_logit_target_loss_coef", 0.0))
        vision_final_delta_margin_loss_coef = float(losses_cfg.get("vision_final_delta_margin_loss_coef", 0.0))
        nonvision_final_delta_stability_loss_coef = float(losses_cfg.get("nonvision_final_delta_stability_loss_coef", 0.0))
        final_delta_target_fraction = float(losses_cfg.get("final_delta_target_fraction", losses_cfg.get("threshold_shift_target_fraction", 0.35)))
        final_delta_margin = float(losses_cfg.get("final_delta_margin", losses_cfg.get("threshold_shift_margin", 0.08)))
        gate_output_scale_l2_coef = float(losses_cfg.get("gate_output_scale_l2_coef", 0.0))
        vision_threshold_shift_target_loss_coef = float(losses_cfg.get("vision_threshold_shift_target_loss_coef", 0.0))
        vision_threshold_shift_margin_loss_coef = float(losses_cfg.get("vision_threshold_shift_margin_loss_coef", 0.0))
        vision_threshold_shift_polarity_loss_coef = float(losses_cfg.get("vision_threshold_shift_polarity_loss_coef", 0.0))
        nonvision_threshold_shift_stability_loss_coef = float(losses_cfg.get("nonvision_threshold_shift_stability_loss_coef", 0.0))
        threshold_shift_head_weight_l2_coef = float(losses_cfg.get("threshold_shift_head_weight_l2_coef", 0.0))
        threshold_shift_target_fraction = float(losses_cfg.get("threshold_shift_target_fraction", 0.35))
        threshold_shift_margin = float(losses_cfg.get("threshold_shift_margin", 0.08))
        clean_shift_slack = float(losses_cfg.get("clean_shift_slack", 0.02))
        mild_shift_slack = float(losses_cfg.get("mild_shift_slack", 0.02))
        gate_threshold_shift_alpha_l2_coef = float(losses_cfg.get("gate_threshold_shift_alpha_l2_coef", 0.0))
        gate_source_logit_bias_l2_coef = float(losses_cfg.get("gate_source_logit_bias_l2_coef", 0.0))
        vision_common_cancel_target_loss_coef = float(losses_cfg.get("vision_common_cancel_target_loss_coef", 0.0))
        vision_common_cancel_clean_target_fraction = float(losses_cfg.get("vision_common_cancel_clean_target_fraction", 0.55))
        vision_common_cancel_mild_target_fraction = float(losses_cfg.get("vision_common_cancel_mild_target_fraction", 0.45))
        common_cancel_anchor_gate = float(losses_cfg.get("common_cancel_anchor_gate", 0.05))
        vision_contrast_shift_target_loss_coef = float(losses_cfg.get("vision_contrast_shift_target_loss_coef", 0.0))
        vision_contrast_shift_margin_loss_coef = float(losses_cfg.get("vision_contrast_shift_margin_loss_coef", 0.0))
        vision_contrast_shift_polarity_loss_coef = float(losses_cfg.get("vision_contrast_shift_polarity_loss_coef", 0.0))
        nonvision_common_cancel_stability_loss_coef = float(losses_cfg.get("nonvision_common_cancel_stability_loss_coef", 0.0))
        nonvision_contrast_shift_stability_loss_coef = float(losses_cfg.get("nonvision_contrast_shift_stability_loss_coef", 0.0))
        vision_common_cancel_overreach_loss_coef = float(losses_cfg.get("vision_common_cancel_overreach_loss_coef", 0.0))
        vision_mild_rescue_target_loss_coef = float(losses_cfg.get("vision_mild_rescue_target_loss_coef", 0.0))
        vision_mild_rescue_margin_loss_coef = float(losses_cfg.get("vision_mild_rescue_margin_loss_coef", 0.0))
        vision_clean_rescue_zero_loss_coef = float(losses_cfg.get("vision_clean_rescue_zero_loss_coef", 0.0))
        nonvision_mild_rescue_stability_loss_coef = float(losses_cfg.get("nonvision_mild_rescue_stability_loss_coef", 0.0))
        common_cancel_head_weight_l2_coef = float(losses_cfg.get("common_cancel_head_weight_l2_coef", 0.0))
        contrast_shift_head_weight_l2_coef = float(losses_cfg.get("contrast_shift_head_weight_l2_coef", 0.0))
        mild_rescue_gate_head_weight_l2_coef = float(losses_cfg.get("mild_rescue_gate_head_weight_l2_coef", 0.0))
        mild_rescue_value_head_weight_l2_coef = float(losses_cfg.get("mild_rescue_value_head_weight_l2_coef", 0.0))
        gate_common_cancel_alpha_l2_coef = float(losses_cfg.get("gate_common_cancel_alpha_l2_coef", 0.0))
        gate_contrast_shift_alpha_l2_coef = float(losses_cfg.get("gate_contrast_shift_alpha_l2_coef", 0.0))
        gate_mild_rescue_alpha_l2_coef = float(losses_cfg.get("gate_mild_rescue_alpha_l2_coef", 0.0))
        contrast_target_fraction = float(losses_cfg.get("contrast_target_fraction", 0.35))
        contrast_shift_margin = float(losses_cfg.get("contrast_shift_margin", 0.02))
        common_cancel_loss_type = str(losses_cfg.get("common_cancel_target_loss_type", "huber"))
        contrast_shift_loss_type = str(losses_cfg.get("contrast_shift_target_loss_type", "huber"))
        mild_rescue_target_fraction = float(losses_cfg.get("mild_rescue_target_fraction", 0.40))
        mild_rescue_margin = float(losses_cfg.get("mild_rescue_margin", 0.04))
        mild_rescue_target_loss_type = str(losses_cfg.get("mild_rescue_target_loss_type", "huber"))
        post_cancel_floor_gate = float(losses_cfg.get("post_cancel_floor_gate", 0.02))
        vision_common_cancel_excess_loss_coef = float(
            losses_cfg.get("vision_common_cancel_excess_loss_coef", 0.0)
        )
        vision_mild_rescue_gate_bce_loss_coef = float(
            losses_cfg.get("vision_mild_rescue_gate_bce_loss_coef", 0.0)
        )
        vision_mild_rescue_gate_margin_loss_coef = float(
            losses_cfg.get("vision_mild_rescue_gate_margin_loss_coef", 0.0)
        )
        mild_rescue_gate_margin = float(losses_cfg.get("mild_rescue_gate_margin", 0.15))
        vision_mild_rescue_logit_target_loss_coef = float(
            losses_cfg.get("vision_mild_rescue_logit_target_loss_coef", 0.0)
        )
        vision_clean_rescue_gate_ceiling_loss_coef = float(
            losses_cfg.get("vision_clean_rescue_gate_ceiling_loss_coef", 0.0)
        )
        vision_cancel_clean_full_excess_loss_coef = float(
            losses_cfg.get("vision_cancel_clean_full_excess_loss_coef", 0.0)
        )
        vision_cancel_mild_fractional_excess_loss_coef = float(
            losses_cfg.get("vision_cancel_mild_fractional_excess_loss_coef", 0.0)
        )
        vision_cancel_gate_floor_loss_coef = float(
            losses_cfg.get("vision_cancel_gate_floor_loss_coef", 0.0)
        )
        mild_cancel_fraction = float(losses_cfg.get("mild_cancel_fraction", 0.90))
        cancel_gate_floor_trigger = float(losses_cfg.get("cancel_gate_floor_trigger", 0.50))
        vision_selector_bce_loss_coef = float(losses_cfg.get("vision_selector_bce_loss_coef", 0.0))
        vision_selector_pair_margin_loss_coef = float(
            losses_cfg.get("vision_selector_pair_margin_loss_coef", 0.0)
        )
        vision_clean_selector_ceiling_loss_coef = float(
            losses_cfg.get("vision_clean_selector_ceiling_loss_coef", 0.0)
        )
        nonvision_selector_stability_loss_coef = float(
            losses_cfg.get("nonvision_selector_stability_loss_coef", 0.0)
        )
        selector_gap_margin = float(losses_cfg.get("selector_gap_margin", 0.25))
        clean_selector_max = float(losses_cfg.get("clean_selector_max", 0.10))
        mild_evidence_input_adapter_weight_l2_coef = float(
            losses_cfg.get("mild_evidence_input_adapter_weight_l2_coef", 0.0)
        )
        mild_value_input_adapter_weight_l2_coef = float(
            losses_cfg.get("mild_value_input_adapter_weight_l2_coef", 0.0)
        )
        clean_rescue_gate_max = float(losses_cfg.get("clean_rescue_gate_max", 0.10))
        drift_coef = float(losses_cfg.get("attention_score_drift_loss_coef", 0.0))
        num_states = int(losses_cfg.get("shapley_num_states_per_minibatch", 2))
        disable_vision_preserve = bool(losses_cfg.get("disable_vision_final_gate_teacher_preserve", False))
        preserve_indices = select_gate_teacher_preserve_indices(disable_vision_preserve)
        paired_supervision = bool(losses_cfg.get("paired_selective_gate_supervision", False))
        num_updates = 0

        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                with torch.no_grad():
                    sign_teacher_corrupt_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    sign_teacher_clean_outputs = self.sign_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )
                    clean_teacher_outputs = self.clean_teacher_model.forward_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        actions=None,
                        deterministic=False,
                    )

                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_corrupt_outputs,
                    ),
                )
                current_clean_outputs = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=None,
                    deterministic=False,
                    gate_feature_context=build_gate_feature_context(
                        clean_teacher_outputs=clean_teacher_outputs,
                        sign_teacher_outputs=sign_teacher_clean_outputs,
                    ),
                )
                positions = self._sample_attention_score_positions(
                    valid_mask=batch.valid_mask,
                    corrupted_flag=batch.extras["corrupted_flag"],
                    target_mask=batch.extras["target_mask"],
                    max_states=num_states,
                )
                if not positions:
                    continue

                route_gap_terms = []
                distance_gap_terms = []
                sign_score_terms = []
                clean_gate_cap_terms = []
                mild_gate_floor_terms = []
                gate_gap_margin_terms = []
                residual_gap_terms = []
                residual_logit_target_terms = []
                vision_final_gate_target_terms = []
                nonvision_gate_stability_terms = []
                vision_adapter_readout_target_terms = []
                vision_adapter_readout_margin_terms = []
                adapter_gate_projection_norm_floor_terms = []
                nonvision_adapter_readout_stability_terms = []
                vision_final_delta_logit_target_terms = []
                vision_final_delta_margin_terms = []
                nonvision_final_delta_stability_terms = []
                vision_threshold_shift_target_terms = []
                vision_threshold_shift_margin_terms = []
                vision_threshold_shift_polarity_terms = []
                nonvision_threshold_shift_stability_terms = []
                vision_common_cancel_target_terms = []
                vision_contrast_shift_target_terms = []
                vision_contrast_shift_margin_terms = []
                vision_contrast_shift_polarity_terms = []
                nonvision_common_cancel_stability_terms = []
                nonvision_contrast_shift_stability_terms = []
                vision_common_cancel_overreach_terms = []
                vision_mild_rescue_target_terms = []
                vision_mild_rescue_margin_terms = []
                vision_clean_rescue_zero_terms = []
                vision_common_cancel_excess_terms = []
                vision_mild_rescue_gate_bce_terms = []
                vision_mild_rescue_gate_margin_terms = []
                vision_mild_rescue_logit_target_terms = []
                vision_clean_rescue_gate_ceiling_terms = []
                vision_cancel_clean_full_excess_terms = []
                vision_cancel_mild_fractional_excess_terms = []
                vision_cancel_gate_floor_terms = []
                vision_selector_bce_terms = []
                vision_selector_pair_margin_terms = []
                vision_clean_selector_ceiling_terms = []
                nonvision_selector_stability_terms = []
                nonvision_mild_rescue_stability_terms = []
                clean_score_adapter_anchor_terms = []
                source_score_adapter_anchor_terms = []
                clean_score_anchor_terms = []
                clean_action_anchor_terms = []
                source_gate_residual_zero_terms = []
                source_gate_distill_terms = []
                source_sign_score_anchor_terms = []
                source_clean_gate_preserve_terms = []
                drift_terms = []

                for batch_idx, time_idx in positions:
                    sign_teacher_clean_scores = _center_scores(
                        sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    )
                    sign_teacher_corrupt_scores = sign_teacher_corrupt_outputs["attention_scores"][batch_idx, time_idx]
                    clean_teacher_scores = clean_teacher_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_scores = current_clean_outputs["attention_scores"][batch_idx, time_idx]
                    corrupt_scores = _center_scores(outputs["attention_scores"][batch_idx, time_idx])
                    current_clean_gate = current_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    corrupt_gate = outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_clean_gate = sign_teacher_clean_outputs["attention_bias_gates"][batch_idx, time_idx]
                    source_corrupt_gate = sign_teacher_corrupt_outputs["attention_bias_gates"][batch_idx, time_idx]
                    gate_residual_clean = current_clean_outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_corrupt = outputs["attention_bias_gate_residuals"][batch_idx, time_idx]
                    gate_residual_logit_clean = current_clean_outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    gate_residual_logit_corrupt = outputs["attention_bias_gate_residual_logits"][batch_idx, time_idx]
                    adapter_gate_logit_clean = current_clean_outputs["attention_bias_gate_adapter_logits"][batch_idx, time_idx]
                    adapter_gate_logit_corrupt = outputs["attention_bias_gate_adapter_logits"][batch_idx, time_idx]
                    adapter_gate_hidden_clean = current_clean_outputs["attention_bias_gate_adapter_hidden"][batch_idx, time_idx]
                    adapter_gate_hidden_corrupt = outputs["attention_bias_gate_adapter_hidden"][batch_idx, time_idx]
                    gate_scaled_delta_logit_clean = current_clean_outputs["attention_bias_gate_scaled_delta_logits"][batch_idx, time_idx]
                    gate_scaled_delta_logit_corrupt = outputs["attention_bias_gate_scaled_delta_logits"][batch_idx, time_idx]
                    gate_threshold_shift_logit_clean = current_clean_outputs["attention_bias_gate_threshold_shift_logits"][batch_idx, time_idx]
                    gate_threshold_shift_logit_corrupt = outputs["attention_bias_gate_threshold_shift_logits"][batch_idx, time_idx]
                    gate_post_shift_excess_clean = current_clean_outputs["attention_bias_gate_post_shift_excess"][batch_idx, time_idx]
                    gate_post_shift_excess_corrupt = outputs["attention_bias_gate_post_shift_excess"][batch_idx, time_idx]
                    gate_common_cancel_gate_clean = current_clean_outputs["attention_bias_gate_common_cancel_gates"][batch_idx, time_idx]
                    gate_common_cancel_gate_corrupt = outputs["attention_bias_gate_common_cancel_gates"][batch_idx, time_idx]
                    gate_common_cancel_logit_clean = current_clean_outputs["attention_bias_gate_common_cancel_logits"][batch_idx, time_idx]
                    gate_common_cancel_logit_corrupt = outputs["attention_bias_gate_common_cancel_logits"][batch_idx, time_idx]
                    gate_post_cancel_logit_clean = current_clean_outputs["attention_bias_gate_post_cancel_logits"][batch_idx, time_idx]
                    gate_post_cancel_logit_corrupt = outputs["attention_bias_gate_post_cancel_logits"][batch_idx, time_idx]
                    gate_contrast_shift_logit_clean = current_clean_outputs["attention_bias_gate_contrast_shift_logits"][batch_idx, time_idx]
                    gate_contrast_shift_logit_corrupt = outputs["attention_bias_gate_contrast_shift_logits"][batch_idx, time_idx]
                    gate_mild_rescue_gate_clean = current_clean_outputs["attention_bias_gate_mild_rescue_gates"][batch_idx, time_idx]
                    gate_mild_rescue_gate_corrupt = outputs["attention_bias_gate_mild_rescue_gates"][batch_idx, time_idx]
                    gate_selector_active_clean = current_clean_outputs["attention_bias_gate_selector_actives"][batch_idx, time_idx]
                    gate_selector_active_corrupt = outputs["attention_bias_gate_selector_actives"][batch_idx, time_idx]
                    gate_mild_rescue_logit_clean = current_clean_outputs["attention_bias_gate_mild_rescue_logits"][batch_idx, time_idx]
                    gate_mild_rescue_logit_corrupt = outputs["attention_bias_gate_mild_rescue_logits"][batch_idx, time_idx]
                    clean_teacher_action_means = clean_teacher_outputs["action_means"][batch_idx, time_idx]
                    current_clean_action_means = current_clean_outputs["action_means"][batch_idx, time_idx]
                    source_clean_scores = sign_teacher_clean_outputs["attention_scores"][batch_idx, time_idx]
                    current_clean_adapter_delta = current_clean_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    corrupt_adapter_delta = outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    clean_teacher_adapter_delta = clean_teacher_outputs["attention_score_adapter_delta"][batch_idx, time_idx]
                    source_corrupt_adapter_delta = sign_teacher_corrupt_outputs["attention_score_adapter_delta"][batch_idx, time_idx]

                    g_vr_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[3]
                    g_vr_mild = corrupt_scores[0] - corrupt_scores[3]
                    g_vd_clean = sign_teacher_clean_scores[0] - sign_teacher_clean_scores[1]
                    g_vd_mild = corrupt_scores[0] - corrupt_scores[1]
                    route_gap_terms.append(F.relu(torch.as_tensor(route_margin, device=self.device) - (g_vr_clean - g_vr_mild)))
                    distance_gap_terms.append(F.relu(torch.as_tensor(distance_margin, device=self.device) - (g_vd_clean - g_vd_mild)))
                    sign_score_terms.append(
                        F.relu(torch.as_tensor(sign_score_margin, device=self.device) - (sign_teacher_clean_scores[0] - corrupt_scores[0]))
                    )

                    paired_losses = compute_paired_selective_gate_losses(
                        current_clean_gate[0],
                        corrupt_gate[0],
                        clean_gate_max=clean_gate_max,
                        mild_gate_min=mild_gate_min,
                        gate_gap_margin=gate_gap_margin,
                        residual_clean=gate_residual_clean[0] if paired_supervision else None,
                        residual_mild=gate_residual_corrupt[0] if paired_supervision else None,
                        residual_margin=residual_selective_margin if paired_supervision else None,
                    )
                    clean_gate_cap_terms.append(paired_losses["clean_gate_cap"])
                    mild_gate_floor_terms.append(paired_losses["mild_gate_floor"])
                    gate_gap_margin_terms.append(paired_losses["gate_gap_margin"])
                    if "residual_gap_margin" in paired_losses:
                        residual_gap_terms.append(paired_losses["residual_gap_margin"])

                    if residual_logit_target_loss_coef > 0.0:
                        residual_logit_target_terms.append(
                            compute_residual_logit_target_loss(
                                clean_residual_logit=gate_residual_logit_clean[0],
                                mild_residual_logit=gate_residual_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                residual_alpha=self.model.fusion_module.get_gate_residual_alpha_tensor(
                                    device=self.device,
                                    dtype=gate_residual_logit_clean.dtype,
                                ).reshape(-1)[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                clip=float(residual_logit_target_clip) if residual_logit_target_clip is not None else None,
                                loss_type=residual_logit_target_loss_type,
                            )
                        )
                    if vision_final_gate_target_loss_coef > 0.0:
                        vision_final_gate_target_terms.append(
                            compute_final_gate_target_loss(
                                clean_gate=current_clean_gate[0],
                                mild_gate=corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                loss_type=final_gate_target_loss_type,
                            )
                        )
                    if nonvision_gate_stability_loss_coef > 0.0 and current_clean_gate.shape[0] > 1:
                        nonvision_gate_stability_terms.append(
                            0.5 * (
                                torch.mean((current_clean_gate[1:] - source_clean_gate[1:]).pow(2))
                                + torch.mean((corrupt_gate[1:] - source_corrupt_gate[1:]).pow(2))
                            )
                        )
                    if vision_adapter_readout_target_loss_coef > 0.0:
                        vision_adapter_readout_target_terms.append(
                            compute_adapter_gate_target_loss(
                                clean_adapter_gate_logit=adapter_gate_logit_clean[0],
                                mild_adapter_gate_logit=adapter_gate_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                target_fraction=adapter_target_fraction,
                                loss_type="huber",
                            )
                        )
                    if vision_adapter_readout_margin_loss_coef > 0.0:
                        vision_adapter_readout_margin_terms.append(
                            compute_adapter_gate_margin_loss(
                                clean_adapter_gate_logit=adapter_gate_logit_clean[0].unsqueeze(0),
                                mild_adapter_gate_logit=adapter_gate_logit_corrupt[0].unsqueeze(0),
                                margin=vision_adapter_readout_margin,
                            )
                        )
                    if adapter_gate_projection_norm_floor_loss_coef > 0.0:
                        adapter_gate_projection_norm_floor_terms.append(
                            compute_projection_norm_floor_loss(
                                projection=adapter_gate_hidden_corrupt[0].unsqueeze(0),
                                min_norm=vision_adapter_projection_norm_min,
                            )
                        )
                    if nonvision_adapter_readout_stability_loss_coef > 0.0 and adapter_gate_logit_clean.shape[0] > 1:
                        nonvision_adapter_readout_stability_terms.append(
                            0.5 * (
                                torch.mean(adapter_gate_logit_clean[1:].pow(2))
                                + torch.mean(adapter_gate_logit_corrupt[1:].pow(2))
                            )
                        )
                    if vision_final_delta_logit_target_loss_coef > 0.0:
                        vision_final_delta_logit_target_terms.append(
                            compute_threshold_shift_target_loss(
                                clean_threshold_shift_logit=gate_scaled_delta_logit_clean[0],
                                mild_threshold_shift_logit=gate_scaled_delta_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                pre_shift_delta_clean=torch.zeros_like(gate_scaled_delta_logit_clean[0]),
                                pre_shift_delta_mild=torch.zeros_like(gate_scaled_delta_logit_corrupt[0]),
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                target_fraction=final_delta_target_fraction,
                                loss_type="huber",
                            )
                        )
                    if vision_final_delta_margin_loss_coef > 0.0:
                        vision_final_delta_margin_terms.append(
                            compute_threshold_shift_margin_loss(
                                clean_threshold_shift_logit=gate_scaled_delta_logit_clean[0].unsqueeze(0),
                                mild_threshold_shift_logit=gate_scaled_delta_logit_corrupt[0].unsqueeze(0),
                                margin=final_delta_margin,
                            )
                        )
                    if nonvision_final_delta_stability_loss_coef > 0.0 and gate_scaled_delta_logit_clean.shape[0] > 1:
                        nonvision_final_delta_stability_terms.append(
                            0.5 * (
                                torch.mean(gate_scaled_delta_logit_clean[1:].pow(2))
                                + torch.mean(gate_scaled_delta_logit_corrupt[1:].pow(2))
                            )
                        )
                    if vision_threshold_shift_target_loss_coef > 0.0:
                        vision_threshold_shift_target_terms.append(
                            compute_threshold_shift_target_loss(
                                clean_threshold_shift_logit=gate_threshold_shift_logit_clean[0],
                                mild_threshold_shift_logit=gate_threshold_shift_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                pre_shift_delta_clean=gate_scaled_delta_logit_clean[0],
                                pre_shift_delta_mild=gate_scaled_delta_logit_corrupt[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                target_fraction=threshold_shift_target_fraction,
                                loss_type="huber",
                            )
                        )
                    if vision_threshold_shift_margin_loss_coef > 0.0:
                        vision_threshold_shift_margin_terms.append(
                            compute_threshold_shift_margin_loss(
                                clean_threshold_shift_logit=gate_threshold_shift_logit_clean[0].unsqueeze(0),
                                mild_threshold_shift_logit=gate_threshold_shift_logit_corrupt[0].unsqueeze(0),
                                margin=threshold_shift_margin,
                            )
                        )
                    if vision_threshold_shift_polarity_loss_coef > 0.0:
                        vision_threshold_shift_polarity_terms.append(
                            compute_threshold_shift_polarity_loss(
                                clean_threshold_shift_logit=gate_threshold_shift_logit_clean[0].unsqueeze(0),
                                mild_threshold_shift_logit=gate_threshold_shift_logit_corrupt[0].unsqueeze(0),
                                clean_shift_slack=clean_shift_slack,
                                mild_shift_slack=mild_shift_slack,
                            )
                        )
                    if nonvision_threshold_shift_stability_loss_coef > 0.0 and gate_threshold_shift_logit_clean.shape[0] > 1:
                        nonvision_threshold_shift_stability_terms.append(
                            0.5 * (
                                torch.mean(gate_threshold_shift_logit_clean[1:].pow(2))
                                + torch.mean(gate_threshold_shift_logit_corrupt[1:].pow(2))
                            )
                        )
                    if vision_common_cancel_target_loss_coef > 0.0:
                        vision_common_cancel_target_terms.append(
                            compute_common_cancel_target_loss(
                                clean_common_cancel_logit=gate_common_cancel_logit_clean[0],
                                mild_common_cancel_logit=gate_common_cancel_logit_corrupt[0],
                                source_clean_gate=source_clean_gate[0],
                                source_mild_gate=source_corrupt_gate[0],
                                pre_shift_delta_clean=gate_scaled_delta_logit_clean[0] + gate_threshold_shift_logit_clean[0],
                                pre_shift_delta_mild=gate_scaled_delta_logit_corrupt[0] + gate_threshold_shift_logit_corrupt[0],
                                anchor_gate=common_cancel_anchor_gate,
                                clean_target_fraction=vision_common_cancel_clean_target_fraction,
                                mild_target_fraction=vision_common_cancel_mild_target_fraction,
                                loss_type=common_cancel_loss_type,
                            )
                        )
                    if vision_common_cancel_overreach_loss_coef > 0.0:
                        vision_common_cancel_overreach_terms.append(
                            compute_common_cancel_overreach_loss(
                                post_cancel_clean_logit=gate_post_cancel_logit_clean[0].unsqueeze(0),
                                post_cancel_mild_logit=gate_post_cancel_logit_corrupt[0].unsqueeze(0),
                                floor_gate=post_cancel_floor_gate,
                            )
                        )
                    if vision_common_cancel_excess_loss_coef > 0.0:
                        vision_common_cancel_excess_terms.append(
                            compute_common_cancel_excess_loss(
                                clean_common_cancel_logit=gate_common_cancel_logit_clean[0].unsqueeze(0),
                                mild_common_cancel_logit=gate_common_cancel_logit_corrupt[0].unsqueeze(0),
                                excess_clean=gate_post_shift_excess_clean[0].unsqueeze(0),
                                excess_mild=gate_post_shift_excess_corrupt[0].unsqueeze(0),
                                clean_target_fraction=vision_common_cancel_clean_target_fraction,
                                mild_target_fraction=vision_common_cancel_mild_target_fraction,
                                loss_type=common_cancel_loss_type,
                            )
                        )
                    if vision_cancel_clean_full_excess_loss_coef > 0.0:
                        vision_cancel_clean_full_excess_terms.append(
                            F.huber_loss(
                                (-gate_common_cancel_logit_clean[0]).unsqueeze(0),
                                gate_post_shift_excess_clean[0].unsqueeze(0),
                                reduction="mean",
                            )
                        )
                    if vision_cancel_mild_fractional_excess_loss_coef > 0.0:
                        target = torch.as_tensor(
                            mild_cancel_fraction,
                            device=self.device,
                            dtype=gate_post_shift_excess_corrupt.dtype,
                        ) * gate_post_shift_excess_corrupt[0].unsqueeze(0)
                        vision_cancel_mild_fractional_excess_terms.append(
                            F.huber_loss(
                                (-gate_common_cancel_logit_corrupt[0]).unsqueeze(0),
                                target,
                                reduction="mean",
                            )
                        )
                    if vision_cancel_gate_floor_loss_coef > 0.0:
                        vision_cancel_gate_floor_terms.append(
                            compute_cancel_gate_floor_loss(
                                clean_common_cancel_prob=gate_common_cancel_gate_clean[0].unsqueeze(0),
                                clean_excess=gate_post_shift_excess_clean[0].unsqueeze(0),
                                trigger=cancel_gate_floor_trigger,
                            )
                        )
                    if nonvision_common_cancel_stability_loss_coef > 0.0 and gate_common_cancel_logit_clean.shape[0] > 1:
                        nonvision_common_cancel_stability_terms.append(
                            0.5 * (
                                torch.mean(gate_common_cancel_logit_clean[1:].pow(2))
                                + torch.mean(gate_common_cancel_logit_corrupt[1:].pow(2))
                            )
                        )
                    if vision_mild_rescue_gate_bce_loss_coef > 0.0:
                        vision_mild_rescue_gate_bce_terms.append(
                            compute_mild_rescue_gate_bce_loss(
                                clean_mild_rescue_gate=gate_mild_rescue_gate_clean[0].unsqueeze(0),
                                mild_mild_rescue_gate=gate_mild_rescue_gate_corrupt[0].unsqueeze(0),
                            )
                        )
                    if vision_selector_bce_loss_coef > 0.0:
                        vision_selector_bce_terms.append(
                            compute_mild_rescue_gate_bce_loss(
                                clean_mild_rescue_gate=gate_mild_rescue_gate_clean[0].unsqueeze(0),
                                mild_mild_rescue_gate=gate_mild_rescue_gate_corrupt[0].unsqueeze(0),
                            )
                        )
                    if vision_mild_rescue_gate_margin_loss_coef > 0.0:
                        vision_mild_rescue_gate_margin_terms.append(
                            compute_adapter_gate_margin_loss(
                                clean_adapter_gate_logit=gate_mild_rescue_gate_clean[0].unsqueeze(0),
                                mild_adapter_gate_logit=gate_mild_rescue_gate_corrupt[0].unsqueeze(0),
                                margin=mild_rescue_gate_margin,
                            )
                        )
                    if vision_selector_pair_margin_loss_coef > 0.0:
                        vision_selector_pair_margin_terms.append(
                            compute_selector_pair_margin_loss(
                                clean_selector_prob=gate_mild_rescue_gate_clean[0].unsqueeze(0),
                                mild_selector_prob=gate_mild_rescue_gate_corrupt[0].unsqueeze(0),
                                margin=selector_gap_margin,
                            )
                        )
                    if vision_mild_rescue_target_loss_coef > 0.0:
                        vision_mild_rescue_target_terms.append(
                            compute_mild_rescue_target_loss(
                                clean_mild_rescue_logit=gate_mild_rescue_logit_clean[0].unsqueeze(0),
                                mild_mild_rescue_logit=gate_mild_rescue_logit_corrupt[0].unsqueeze(0),
                                post_cancel_mild_logit=gate_post_cancel_logit_corrupt[0].unsqueeze(0),
                                mild_gate_min=mild_gate_min,
                                target_fraction=mild_rescue_target_fraction,
                                loss_type=mild_rescue_target_loss_type,
                            )
                        )
                    if vision_mild_rescue_logit_target_loss_coef > 0.0:
                        vision_mild_rescue_logit_target_terms.append(
                            compute_mild_rescue_target_loss(
                                clean_mild_rescue_logit=gate_mild_rescue_logit_clean[0].unsqueeze(0),
                                mild_mild_rescue_logit=gate_mild_rescue_logit_corrupt[0].unsqueeze(0),
                                post_cancel_mild_logit=gate_post_cancel_logit_corrupt[0].unsqueeze(0),
                                mild_gate_min=mild_gate_min,
                                target_fraction=mild_rescue_target_fraction,
                                loss_type=mild_rescue_target_loss_type,
                            )
                        )
                    if vision_mild_rescue_margin_loss_coef > 0.0:
                        vision_mild_rescue_margin_terms.append(
                            compute_threshold_shift_margin_loss(
                                clean_threshold_shift_logit=gate_mild_rescue_logit_clean[0].unsqueeze(0),
                                mild_threshold_shift_logit=gate_mild_rescue_logit_corrupt[0].unsqueeze(0),
                                margin=mild_rescue_margin,
                            )
                        )
                    if vision_clean_rescue_zero_loss_coef > 0.0:
                        vision_clean_rescue_zero_terms.append(
                            compute_clean_rescue_zero_loss(
                                clean_mild_rescue_logit=gate_mild_rescue_logit_clean[0].unsqueeze(0),
                            )
                        )
                    if vision_clean_rescue_gate_ceiling_loss_coef > 0.0:
                        vision_clean_rescue_gate_ceiling_terms.append(
                            compute_clean_rescue_gate_ceiling_loss(
                                clean_mild_rescue_gate=gate_mild_rescue_gate_clean[0].unsqueeze(0),
                                max_gate=clean_rescue_gate_max,
                            )
                        )
                    if vision_clean_selector_ceiling_loss_coef > 0.0:
                        vision_clean_selector_ceiling_terms.append(
                            compute_clean_rescue_gate_ceiling_loss(
                                clean_mild_rescue_gate=gate_mild_rescue_gate_clean[0].unsqueeze(0),
                                max_gate=clean_selector_max,
                            )
                        )
                    if nonvision_selector_stability_loss_coef > 0.0 and gate_mild_rescue_gate_clean.shape[0] > 1:
                        nonvision_selector_stability_terms.append(
                            0.5 * (
                                torch.mean(gate_mild_rescue_gate_clean[1:].pow(2))
                                + torch.mean(gate_mild_rescue_gate_corrupt[1:].pow(2))
                            )
                        )
                    if nonvision_mild_rescue_stability_loss_coef > 0.0 and gate_mild_rescue_logit_clean.shape[0] > 1:
                        nonvision_mild_rescue_stability_terms.append(
                            0.5 * (
                                torch.mean(gate_mild_rescue_logit_clean[1:].pow(2))
                                + torch.mean(gate_mild_rescue_logit_corrupt[1:].pow(2))
                            )
                        )
                    if vision_contrast_shift_target_loss_coef > 0.0:
                        vision_contrast_shift_target_terms.append(
                            compute_contrast_shift_target_loss(
                                clean_contrast_shift_logit=gate_contrast_shift_logit_clean[0],
                                mild_contrast_shift_logit=gate_contrast_shift_logit_corrupt[0],
                                post_cancel_clean_logit=gate_post_cancel_logit_clean[0],
                                post_cancel_mild_logit=gate_post_cancel_logit_corrupt[0],
                                clean_gate_max=clean_gate_max,
                                mild_gate_min=mild_gate_min,
                                target_fraction=contrast_target_fraction,
                                loss_type=contrast_shift_loss_type,
                            )
                        )
                    if vision_contrast_shift_margin_loss_coef > 0.0:
                        vision_contrast_shift_margin_terms.append(
                            compute_threshold_shift_margin_loss(
                                clean_threshold_shift_logit=gate_contrast_shift_logit_clean[0].unsqueeze(0),
                                mild_threshold_shift_logit=gate_contrast_shift_logit_corrupt[0].unsqueeze(0),
                                margin=contrast_shift_margin,
                            )
                        )
                    if vision_contrast_shift_polarity_loss_coef > 0.0:
                        vision_contrast_shift_polarity_terms.append(
                            compute_threshold_shift_polarity_loss(
                                clean_threshold_shift_logit=gate_contrast_shift_logit_clean[0].unsqueeze(0),
                                mild_threshold_shift_logit=gate_contrast_shift_logit_corrupt[0].unsqueeze(0),
                                clean_shift_slack=clean_shift_slack,
                                mild_shift_slack=mild_shift_slack,
                            )
                        )
                    if nonvision_contrast_shift_stability_loss_coef > 0.0 and gate_contrast_shift_logit_clean.shape[0] > 1:
                        nonvision_contrast_shift_stability_terms.append(
                            0.5 * (
                                torch.mean(gate_contrast_shift_logit_clean[1:].pow(2))
                                + torch.mean(gate_contrast_shift_logit_corrupt[1:].pow(2))
                            )
                        )
                    if self._train_score_adapter and clean_score_adapter_anchor_coef > 0.0:
                        clean_score_adapter_anchor_terms.append(
                            compute_score_adapter_anchor_loss(
                                current_delta=current_clean_adapter_delta,
                                teacher_delta=clean_teacher_adapter_delta,
                                loss_type=score_adapter_anchor_loss_type,
                            )
                        )
                    if self._train_score_adapter and source_score_adapter_anchor_coef > 0.0:
                        source_score_adapter_anchor_terms.append(
                            compute_score_adapter_anchor_loss(
                                current_delta=corrupt_adapter_delta,
                                teacher_delta=source_corrupt_adapter_delta,
                                loss_type=score_adapter_anchor_loss_type,
                            )
                        )
                    clean_score_anchor_terms.append(torch.mean((current_clean_scores - clean_teacher_scores).pow(2)))
                    clean_action_anchor_terms.append(torch.mean((current_clean_action_means - clean_teacher_action_means).pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_clean.pow(2)))
                    source_gate_residual_zero_terms.append(torch.mean(gate_residual_corrupt.pow(2)))
                    if source_gate_distill_coef > 0.0:
                        source_gate_distill_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                        source_gate_distill_terms.append(
                            torch.mean((corrupt_gate[preserve_indices] - source_corrupt_gate[preserve_indices]).pow(2))
                        )
                    source_sign_score_anchor_terms.append(
                        torch.mean((outputs["attention_scores"][batch_idx, time_idx] - sign_teacher_corrupt_scores).pow(2))
                    )
                    if source_clean_gate_preserve_coef > 0.0:
                        source_clean_gate_preserve_terms.append(
                            torch.mean((current_clean_gate[preserve_indices] - source_clean_gate[preserve_indices]).pow(2))
                        )
                    drift_terms.append(_center_scores(current_clean_scores - source_clean_scores).pow(2).mean())
                    drift_terms.append(corrupt_scores.pow(2).mean())

                route_gap_loss = torch.stack(route_gap_terms).mean() if route_gap_terms else torch.zeros((), device=self.device)
                distance_gap_loss = torch.stack(distance_gap_terms).mean() if distance_gap_terms else torch.zeros((), device=self.device)
                sign_score_loss = torch.stack(sign_score_terms).mean() if sign_score_terms else torch.zeros((), device=self.device)
                clean_gate_cap_loss = torch.stack(clean_gate_cap_terms).mean() if clean_gate_cap_terms else torch.zeros((), device=self.device)
                mild_gate_floor_loss = torch.stack(mild_gate_floor_terms).mean() if mild_gate_floor_terms else torch.zeros((), device=self.device)
                gate_gap_margin_loss = torch.stack(gate_gap_margin_terms).mean() if gate_gap_margin_terms else torch.zeros((), device=self.device)
                residual_gap_margin_loss = torch.stack(residual_gap_terms).mean() if residual_gap_terms else torch.zeros((), device=self.device)
                residual_logit_target_loss = torch.stack(residual_logit_target_terms).mean() if residual_logit_target_terms else torch.zeros((), device=self.device)
                vision_final_gate_target_loss = torch.stack(vision_final_gate_target_terms).mean() if vision_final_gate_target_terms else torch.zeros((), device=self.device)
                nonvision_gate_stability_loss = torch.stack(nonvision_gate_stability_terms).mean() if nonvision_gate_stability_terms else torch.zeros((), device=self.device)
                vision_adapter_readout_target_loss = torch.stack(vision_adapter_readout_target_terms).mean() if vision_adapter_readout_target_terms else torch.zeros((), device=self.device)
                vision_adapter_readout_margin_loss = torch.stack(vision_adapter_readout_margin_terms).mean() if vision_adapter_readout_margin_terms else torch.zeros((), device=self.device)
                adapter_gate_projection_norm_floor_loss = torch.stack(adapter_gate_projection_norm_floor_terms).mean() if adapter_gate_projection_norm_floor_terms else torch.zeros((), device=self.device)
                nonvision_adapter_readout_stability_loss = torch.stack(nonvision_adapter_readout_stability_terms).mean() if nonvision_adapter_readout_stability_terms else torch.zeros((), device=self.device)
                vision_final_delta_logit_target_loss = torch.stack(vision_final_delta_logit_target_terms).mean() if vision_final_delta_logit_target_terms else torch.zeros((), device=self.device)
                vision_final_delta_margin_loss = torch.stack(vision_final_delta_margin_terms).mean() if vision_final_delta_margin_terms else torch.zeros((), device=self.device)
                nonvision_final_delta_stability_loss = torch.stack(nonvision_final_delta_stability_terms).mean() if nonvision_final_delta_stability_terms else torch.zeros((), device=self.device)
                vision_threshold_shift_target_loss = torch.stack(vision_threshold_shift_target_terms).mean() if vision_threshold_shift_target_terms else torch.zeros((), device=self.device)
                vision_threshold_shift_margin_loss = torch.stack(vision_threshold_shift_margin_terms).mean() if vision_threshold_shift_margin_terms else torch.zeros((), device=self.device)
                vision_threshold_shift_polarity_loss = torch.stack(vision_threshold_shift_polarity_terms).mean() if vision_threshold_shift_polarity_terms else torch.zeros((), device=self.device)
                nonvision_threshold_shift_stability_loss = torch.stack(nonvision_threshold_shift_stability_terms).mean() if nonvision_threshold_shift_stability_terms else torch.zeros((), device=self.device)
                vision_common_cancel_target_loss = torch.stack(vision_common_cancel_target_terms).mean() if vision_common_cancel_target_terms else torch.zeros((), device=self.device)
                vision_common_cancel_overreach_loss = torch.stack(vision_common_cancel_overreach_terms).mean() if vision_common_cancel_overreach_terms else torch.zeros((), device=self.device)
                vision_mild_rescue_target_loss = torch.stack(vision_mild_rescue_target_terms).mean() if vision_mild_rescue_target_terms else torch.zeros((), device=self.device)
                vision_mild_rescue_margin_loss = torch.stack(vision_mild_rescue_margin_terms).mean() if vision_mild_rescue_margin_terms else torch.zeros((), device=self.device)
                vision_clean_rescue_zero_loss = torch.stack(vision_clean_rescue_zero_terms).mean() if vision_clean_rescue_zero_terms else torch.zeros((), device=self.device)
                vision_common_cancel_excess_loss = torch.stack(vision_common_cancel_excess_terms).mean() if vision_common_cancel_excess_terms else torch.zeros((), device=self.device)
                vision_mild_rescue_gate_bce_loss = torch.stack(vision_mild_rescue_gate_bce_terms).mean() if vision_mild_rescue_gate_bce_terms else torch.zeros((), device=self.device)
                vision_mild_rescue_gate_margin_loss = torch.stack(vision_mild_rescue_gate_margin_terms).mean() if vision_mild_rescue_gate_margin_terms else torch.zeros((), device=self.device)
                vision_mild_rescue_logit_target_loss = torch.stack(vision_mild_rescue_logit_target_terms).mean() if vision_mild_rescue_logit_target_terms else torch.zeros((), device=self.device)
                vision_clean_rescue_gate_ceiling_loss = torch.stack(vision_clean_rescue_gate_ceiling_terms).mean() if vision_clean_rescue_gate_ceiling_terms else torch.zeros((), device=self.device)
                vision_cancel_clean_full_excess_loss = torch.stack(vision_cancel_clean_full_excess_terms).mean() if vision_cancel_clean_full_excess_terms else torch.zeros((), device=self.device)
                vision_cancel_mild_fractional_excess_loss = torch.stack(vision_cancel_mild_fractional_excess_terms).mean() if vision_cancel_mild_fractional_excess_terms else torch.zeros((), device=self.device)
                vision_cancel_gate_floor_loss = torch.stack(vision_cancel_gate_floor_terms).mean() if vision_cancel_gate_floor_terms else torch.zeros((), device=self.device)
                vision_selector_bce_loss = torch.stack(vision_selector_bce_terms).mean() if vision_selector_bce_terms else torch.zeros((), device=self.device)
                vision_selector_pair_margin_loss = torch.stack(vision_selector_pair_margin_terms).mean() if vision_selector_pair_margin_terms else torch.zeros((), device=self.device)
                vision_clean_selector_ceiling_loss = torch.stack(vision_clean_selector_ceiling_terms).mean() if vision_clean_selector_ceiling_terms else torch.zeros((), device=self.device)
                nonvision_selector_stability_loss = torch.stack(nonvision_selector_stability_terms).mean() if nonvision_selector_stability_terms else torch.zeros((), device=self.device)
                nonvision_mild_rescue_stability_loss = torch.stack(nonvision_mild_rescue_stability_terms).mean() if nonvision_mild_rescue_stability_terms else torch.zeros((), device=self.device)
                vision_contrast_shift_target_loss = torch.stack(vision_contrast_shift_target_terms).mean() if vision_contrast_shift_target_terms else torch.zeros((), device=self.device)
                vision_contrast_shift_margin_loss = torch.stack(vision_contrast_shift_margin_terms).mean() if vision_contrast_shift_margin_terms else torch.zeros((), device=self.device)
                vision_contrast_shift_polarity_loss = torch.stack(vision_contrast_shift_polarity_terms).mean() if vision_contrast_shift_polarity_terms else torch.zeros((), device=self.device)
                nonvision_common_cancel_stability_loss = torch.stack(nonvision_common_cancel_stability_terms).mean() if nonvision_common_cancel_stability_terms else torch.zeros((), device=self.device)
                nonvision_contrast_shift_stability_loss = torch.stack(nonvision_contrast_shift_stability_terms).mean() if nonvision_contrast_shift_stability_terms else torch.zeros((), device=self.device)
                clean_score_adapter_anchor_loss = torch.stack(clean_score_adapter_anchor_terms).mean() if clean_score_adapter_anchor_terms else torch.zeros((), device=self.device)
                source_score_adapter_anchor_loss = torch.stack(source_score_adapter_anchor_terms).mean() if source_score_adapter_anchor_terms else torch.zeros((), device=self.device)
                clean_score_anchor_loss = torch.stack(clean_score_anchor_terms).mean() if clean_score_anchor_terms else torch.zeros((), device=self.device)
                clean_action_anchor_loss = torch.stack(clean_action_anchor_terms).mean() if clean_action_anchor_terms else torch.zeros((), device=self.device)
                source_gate_residual_zero_loss = torch.stack(source_gate_residual_zero_terms).mean() if source_gate_residual_zero_terms else torch.zeros((), device=self.device)
                source_gate_distill_loss = torch.stack(source_gate_distill_terms).mean() if source_gate_distill_terms else torch.zeros((), device=self.device)
                source_sign_score_anchor_loss = torch.stack(source_sign_score_anchor_terms).mean() if source_sign_score_anchor_terms else torch.zeros((), device=self.device)
                source_clean_gate_preserve_loss = torch.stack(source_clean_gate_preserve_terms).mean() if source_clean_gate_preserve_terms else torch.zeros((), device=self.device)
                drift_loss = torch.stack(drift_terms).mean() if drift_terms else torch.zeros((), device=self.device)
                residual_alpha = self.model.fusion_module.get_gate_residual_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                adapter_gate_alpha = self.model.fusion_module.get_adapter_gate_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_logit_offset = self.model.fusion_module.get_gate_logit_offset_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_output_scale = self.model.fusion_module.get_gate_output_scale_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_threshold_shift_alpha = self.model.fusion_module.get_gate_threshold_shift_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_source_logit_bias = self.model.fusion_module.get_gate_source_logit_bias_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_common_cancel_alpha = self.model.fusion_module.get_gate_common_cancel_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_contrast_shift_alpha = self.model.fusion_module.get_gate_contrast_shift_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                gate_mild_rescue_alpha = self.model.fusion_module.get_gate_mild_rescue_alpha_tensor(device=self.device, dtype=clean_gate_cap_loss.dtype)
                residual_alpha_l2_loss = residual_alpha.pow(2).mean()
                adapter_gate_alpha_l2_loss = adapter_gate_alpha.pow(2).mean()
                gate_logit_offset_l2_loss = gate_logit_offset.pow(2).mean()
                gate_output_scale_l2_loss = gate_output_scale.pow(2).mean()
                gate_threshold_shift_alpha_l2_loss = gate_threshold_shift_alpha.pow(2).mean()
                gate_source_logit_bias_l2_loss = gate_source_logit_bias.pow(2).mean()
                gate_common_cancel_alpha_l2_loss = gate_common_cancel_alpha.pow(2).mean()
                gate_contrast_shift_alpha_l2_loss = gate_contrast_shift_alpha.pow(2).mean()
                gate_mild_rescue_alpha_l2_loss = gate_mild_rescue_alpha.pow(2).mean()
                adapter_input_adapter_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_adapter_input_adapter.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                adapter_gate_coupler_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_adapter_coupler.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                adapter_gate_input_adapter_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in adapter_input_adapter_params]).mean()
                    if adapter_input_adapter_params
                    else torch.zeros((), device=self.device)
                )
                mild_evidence_input_adapter_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_mild_evidence_input_adapter.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                mild_evidence_input_adapter_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in mild_evidence_input_adapter_params]).mean()
                    if mild_evidence_input_adapter_params
                    else torch.zeros((), device=self.device)
                )
                mild_value_input_adapter_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_mild_value_input_adapter.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                mild_value_input_adapter_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in mild_value_input_adapter_params]).mean()
                    if mild_value_input_adapter_params
                    else torch.zeros((), device=self.device)
                )
                adapter_gate_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in adapter_gate_coupler_params]).mean()
                    if adapter_gate_coupler_params
                    else torch.zeros((), device=self.device)
                )
                threshold_shift_head_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_threshold_shift_head.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                threshold_shift_head_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in threshold_shift_head_params]).mean()
                    if threshold_shift_head_params
                    else torch.zeros((), device=self.device)
                )
                common_cancel_head_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_common_cancel_head.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                common_cancel_head_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in common_cancel_head_params]).mean()
                    if common_cancel_head_params
                    else torch.zeros((), device=self.device)
                )
                contrast_shift_head_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_contrast_shift_head.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                contrast_shift_head_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in contrast_shift_head_params]).mean()
                    if contrast_shift_head_params
                    else torch.zeros((), device=self.device)
                )
                mild_rescue_gate_head_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_mild_rescue_gate_head.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                mild_rescue_gate_head_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in mild_rescue_gate_head_params]).mean()
                    if mild_rescue_gate_head_params
                    else torch.zeros((), device=self.device)
                )
                mild_rescue_value_head_params = [
                    parameter
                    for parameter in self.model.fusion_module.reliability_bias_gate_mild_rescue_value_head.parameters()
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                ]
                mild_rescue_value_head_weight_l2_loss = (
                    torch.stack([parameter.pow(2).mean() for parameter in mild_rescue_value_head_params]).mean()
                    if mild_rescue_value_head_params
                    else torch.zeros((), device=self.device)
                )
                score_adapter_weight_l2_loss = torch.zeros((), device=self.device)
                if self._train_score_adapter and score_adapter_weight_l2_coef > 0.0:
                    score_adapter_weight_l2_loss = torch.stack(
                        [parameter.pow(2).mean() for parameter in self.model.fusion_module.score_adapter.parameters()]
                    ).mean()

                total_loss = (
                    route_coef * route_gap_loss
                    + distance_coef * distance_gap_loss
                    + sign_score_coef * sign_score_loss
                    + clean_gate_cap_coef * clean_gate_cap_loss
                    + mild_gate_floor_coef * mild_gate_floor_loss
                    + gate_gap_margin_coef * gate_gap_margin_loss
                    + residual_gap_margin_coef * residual_gap_margin_loss
                    + residual_logit_target_loss_coef * residual_logit_target_loss
                    + vision_final_gate_target_loss_coef * vision_final_gate_target_loss
                    + nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss
                    + vision_adapter_readout_target_loss_coef * vision_adapter_readout_target_loss
                    + vision_adapter_readout_margin_loss_coef * vision_adapter_readout_margin_loss
                    + adapter_gate_projection_norm_floor_loss_coef * adapter_gate_projection_norm_floor_loss
                    + nonvision_adapter_readout_stability_loss_coef * nonvision_adapter_readout_stability_loss
                    + vision_final_delta_logit_target_loss_coef * vision_final_delta_logit_target_loss
                    + vision_final_delta_margin_loss_coef * vision_final_delta_margin_loss
                    + nonvision_final_delta_stability_loss_coef * nonvision_final_delta_stability_loss
                    + vision_threshold_shift_target_loss_coef * vision_threshold_shift_target_loss
                    + vision_threshold_shift_margin_loss_coef * vision_threshold_shift_margin_loss
                    + vision_threshold_shift_polarity_loss_coef * vision_threshold_shift_polarity_loss
                    + nonvision_threshold_shift_stability_loss_coef * nonvision_threshold_shift_stability_loss
                    + vision_common_cancel_target_loss_coef * vision_common_cancel_target_loss
                    + vision_contrast_shift_target_loss_coef * vision_contrast_shift_target_loss
                    + vision_contrast_shift_margin_loss_coef * vision_contrast_shift_margin_loss
                    + vision_contrast_shift_polarity_loss_coef * vision_contrast_shift_polarity_loss
                    + nonvision_common_cancel_stability_loss_coef * nonvision_common_cancel_stability_loss
                    + nonvision_contrast_shift_stability_loss_coef * nonvision_contrast_shift_stability_loss
                    + clean_score_adapter_anchor_coef * clean_score_adapter_anchor_loss
                    + source_score_adapter_anchor_coef * source_score_adapter_anchor_loss
                    + clean_score_anchor_coef * clean_score_anchor_loss
                    + clean_action_anchor_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_coef * source_gate_residual_zero_loss
                    + source_gate_distill_coef * source_gate_distill_loss
                    + source_sign_score_anchor_coef * source_sign_score_anchor_loss
                    + source_clean_gate_preserve_coef * source_clean_gate_preserve_loss
                    + residual_alpha_l2_coef * residual_alpha_l2_loss
                    + adapter_gate_alpha_l2_coef * adapter_gate_alpha_l2_loss
                    + gate_logit_offset_l2_coef * gate_logit_offset_l2_loss
                    + gate_output_scale_l2_coef * gate_output_scale_l2_loss
                    + gate_threshold_shift_alpha_l2_coef * gate_threshold_shift_alpha_l2_loss
                    + gate_source_logit_bias_l2_coef * gate_source_logit_bias_l2_loss
                    + gate_common_cancel_alpha_l2_coef * gate_common_cancel_alpha_l2_loss
                    + gate_contrast_shift_alpha_l2_coef * gate_contrast_shift_alpha_l2_loss
                    + gate_mild_rescue_alpha_l2_coef * gate_mild_rescue_alpha_l2_loss
                    + adapter_gate_input_adapter_weight_l2_coef * adapter_gate_input_adapter_weight_l2_loss
                    + adapter_gate_weight_l2_coef * adapter_gate_weight_l2_loss
                    + threshold_shift_head_weight_l2_coef * threshold_shift_head_weight_l2_loss
                    + common_cancel_head_weight_l2_coef * common_cancel_head_weight_l2_loss
                    + contrast_shift_head_weight_l2_coef * contrast_shift_head_weight_l2_loss
                    + vision_common_cancel_overreach_loss_coef * vision_common_cancel_overreach_loss
                    + vision_common_cancel_excess_loss_coef * vision_common_cancel_excess_loss
                    + vision_cancel_clean_full_excess_loss_coef * vision_cancel_clean_full_excess_loss
                    + vision_cancel_mild_fractional_excess_loss_coef * vision_cancel_mild_fractional_excess_loss
                    + vision_cancel_gate_floor_loss_coef * vision_cancel_gate_floor_loss
                    + vision_mild_rescue_target_loss_coef * vision_mild_rescue_target_loss
                    + vision_mild_rescue_margin_loss_coef * vision_mild_rescue_margin_loss
                    + vision_clean_rescue_zero_loss_coef * vision_clean_rescue_zero_loss
                    + vision_mild_rescue_gate_bce_loss_coef * vision_mild_rescue_gate_bce_loss
                    + vision_mild_rescue_gate_margin_loss_coef * vision_mild_rescue_gate_margin_loss
                    + vision_mild_rescue_logit_target_loss_coef * vision_mild_rescue_logit_target_loss
                    + vision_clean_rescue_gate_ceiling_loss_coef * vision_clean_rescue_gate_ceiling_loss
                    + vision_selector_bce_loss_coef * vision_selector_bce_loss
                    + vision_selector_pair_margin_loss_coef * vision_selector_pair_margin_loss
                    + vision_clean_selector_ceiling_loss_coef * vision_clean_selector_ceiling_loss
                    + nonvision_selector_stability_loss_coef * nonvision_selector_stability_loss
                    + nonvision_mild_rescue_stability_loss_coef * nonvision_mild_rescue_stability_loss
                    + mild_evidence_input_adapter_weight_l2_coef * mild_evidence_input_adapter_weight_l2_loss
                    + mild_value_input_adapter_weight_l2_coef * mild_value_input_adapter_weight_l2_loss
                    + mild_rescue_gate_head_weight_l2_coef * mild_rescue_gate_head_weight_l2_loss
                    + mild_rescue_value_head_weight_l2_coef * mild_rescue_value_head_weight_l2_loss
                    + score_adapter_weight_l2_coef * score_adapter_weight_l2_loss
                    + drift_coef * drift_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.base_cfg.ppo.max_grad_norm).item())
                self.optimizer.step()

                metrics["loss/vision_pairgap_route"] += float(route_gap_loss.item())
                metrics["loss/vision_pairgap_distance"] += float(distance_gap_loss.item())
                metrics["loss/vision_attention_score_margin"] += float(sign_score_loss.item())
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/residual_gap_margin"] += float(residual_gap_margin_loss.item())
                metrics["loss/residual_logit_target"] += float(residual_logit_target_loss.item())
                metrics["loss/vision_final_gate_target"] += float(vision_final_gate_target_loss.item())
                metrics["loss/nonvision_gate_stability"] += float(nonvision_gate_stability_loss.item())
                metrics["loss/vision_adapter_readout_target"] += float(vision_adapter_readout_target_loss.item())
                metrics["loss/vision_adapter_readout_margin"] += float(vision_adapter_readout_margin_loss.item())
                metrics["loss/adapter_gate_projection_norm_floor"] += float(adapter_gate_projection_norm_floor_loss.item())
                metrics["loss/nonvision_adapter_readout_stability"] += float(nonvision_adapter_readout_stability_loss.item())
                metrics["loss/vision_final_delta_logit_target"] += float(vision_final_delta_logit_target_loss.item())
                metrics["loss/vision_final_delta_margin"] += float(vision_final_delta_margin_loss.item())
                metrics["loss/nonvision_final_delta_stability"] += float(nonvision_final_delta_stability_loss.item())
                metrics["loss/vision_threshold_shift_target"] += float(vision_threshold_shift_target_loss.item())
                metrics["loss/vision_threshold_shift_margin"] += float(vision_threshold_shift_margin_loss.item())
                metrics["loss/vision_threshold_shift_polarity"] += float(vision_threshold_shift_polarity_loss.item())
                metrics["loss/nonvision_threshold_shift_stability"] += float(nonvision_threshold_shift_stability_loss.item())
                metrics["loss/threshold_shift_head_weight_l2"] += float(threshold_shift_head_weight_l2_loss.item())
                metrics["loss/vision_common_cancel_target"] += float(vision_common_cancel_target_loss.item())
                metrics["loss/vision_common_cancel_overreach"] += float(vision_common_cancel_overreach_loss.item())
                metrics["loss/vision_mild_rescue_target"] += float(vision_mild_rescue_target_loss.item())
                metrics["loss/vision_mild_rescue_margin"] += float(vision_mild_rescue_margin_loss.item())
                metrics["loss/vision_clean_rescue_zero"] += float(vision_clean_rescue_zero_loss.item())
                metrics["loss/vision_common_cancel_excess"] += float(vision_common_cancel_excess_loss.item())
                metrics["loss/vision_cancel_clean_full_excess"] += float(vision_cancel_clean_full_excess_loss.item())
                metrics["loss/vision_cancel_mild_fractional_excess"] += float(vision_cancel_mild_fractional_excess_loss.item())
                metrics["loss/vision_cancel_gate_floor"] += float(vision_cancel_gate_floor_loss.item())
                metrics["loss/vision_mild_rescue_gate_bce"] += float(vision_mild_rescue_gate_bce_loss.item())
                metrics["loss/vision_mild_rescue_gate_margin"] += float(vision_mild_rescue_gate_margin_loss.item())
                metrics["loss/vision_mild_rescue_logit_target"] += float(vision_mild_rescue_logit_target_loss.item())
                metrics["loss/vision_clean_rescue_gate_ceiling"] += float(vision_clean_rescue_gate_ceiling_loss.item())
                metrics["loss/vision_selector_bce"] += float(vision_selector_bce_loss.item())
                metrics["loss/vision_selector_pair_margin"] += float(vision_selector_pair_margin_loss.item())
                metrics["loss/vision_clean_selector_ceiling"] += float(vision_clean_selector_ceiling_loss.item())
                metrics["loss/nonvision_selector_stability"] += float(nonvision_selector_stability_loss.item())
                metrics["loss/nonvision_mild_rescue_stability"] += float(nonvision_mild_rescue_stability_loss.item())
                metrics["loss/vision_contrast_shift_target"] += float(vision_contrast_shift_target_loss.item())
                metrics["loss/vision_contrast_shift_margin"] += float(vision_contrast_shift_margin_loss.item())
                metrics["loss/vision_contrast_shift_polarity"] += float(vision_contrast_shift_polarity_loss.item())
                metrics["loss/nonvision_common_cancel_stability"] += float(nonvision_common_cancel_stability_loss.item())
                metrics["loss/nonvision_contrast_shift_stability"] += float(nonvision_contrast_shift_stability_loss.item())
                metrics["loss/common_cancel_head_weight_l2"] += float(common_cancel_head_weight_l2_loss.item())
                metrics["loss/contrast_shift_head_weight_l2"] += float(contrast_shift_head_weight_l2_loss.item())
                metrics["loss/mild_evidence_input_adapter_weight_l2"] += float(mild_evidence_input_adapter_weight_l2_loss.item())
                metrics["loss/mild_value_input_adapter_weight_l2"] += float(mild_value_input_adapter_weight_l2_loss.item())
                metrics["loss/mild_rescue_gate_head_weight_l2"] += float(mild_rescue_gate_head_weight_l2_loss.item())
                metrics["loss/mild_rescue_value_head_weight_l2"] += float(mild_rescue_value_head_weight_l2_loss.item())
                metrics["loss/adapter_gate_input_adapter_weight_l2"] += float(adapter_gate_input_adapter_weight_l2_loss.item())
                metrics["loss/adapter_gate_weight_l2"] += float(adapter_gate_weight_l2_loss.item())
                metrics["loss/clean_score_adapter_anchor"] += float(clean_score_adapter_anchor_loss.item())
                metrics["loss/source_score_adapter_anchor"] += float(source_score_adapter_anchor_loss.item())
                metrics["loss/score_adapter_weight_l2"] += float(score_adapter_weight_l2_loss.item())
                metrics["loss/residual_alpha_l2"] += float(residual_alpha_l2_loss.item())
                metrics["loss/adapter_gate_alpha_l2"] += float(adapter_gate_alpha_l2_loss.item())
                metrics["loss/gate_logit_offset_l2"] += float(gate_logit_offset_l2_loss.item())
                metrics["loss/gate_output_scale_l2"] += float(gate_output_scale_l2_loss.item())
                metrics["loss/gate_threshold_shift_alpha_l2"] += float(gate_threshold_shift_alpha_l2_loss.item())
                metrics["loss/gate_source_logit_bias_l2"] += float(gate_source_logit_bias_l2_loss.item())
                metrics["loss/gate_common_cancel_alpha_l2"] += float(gate_common_cancel_alpha_l2_loss.item())
                metrics["loss/gate_contrast_shift_alpha_l2"] += float(gate_contrast_shift_alpha_l2_loss.item())
                metrics["loss/gate_mild_rescue_alpha_l2"] += float(gate_mild_rescue_alpha_l2_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_sign_score_anchor"] += float(source_sign_score_anchor_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/attention_score_drift"] += float(drift_loss.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["train/residual_alpha"] += float(residual_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_logit_offset"] += float(gate_logit_offset.detach().reshape(-1)[0].cpu().item())
                metrics["train/adapter_gate_alpha"] += float(adapter_gate_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_gate_output_scale"] += float(gate_output_scale.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_threshold_shift_alpha"] += float(gate_threshold_shift_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_source_logit_bias"] += float(gate_source_logit_bias.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_common_cancel_alpha"] += float(gate_common_cancel_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_contrast_shift_alpha"] += float(gate_contrast_shift_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/vision_mild_rescue_alpha"] += float(gate_mild_rescue_alpha.detach().reshape(-1)[0].cpu().item())
                metrics["train/score_adapter_delta_norm"] += float(self._score_adapter_delta_norm().detach().cpu().item())
                metrics["train/adapter_gate_hidden_norm"] += float(
                    outputs["attention_bias_gate_adapter_hidden_norms"][..., 0].detach().mean().cpu().item()
                )
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics


class Stage37f28VisionGateOutputScaleTrainer(Stage37f27AdapterGateReadoutTrainer):
    """V3.7f2.8 trainer with explicit gate output scale."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f26AdapterGateCouplingTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        self._train_score_adapter = bool(losses_cfg.get("train_score_adapter", False))
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            self._initial_score_adapter_params = []
            self._initial_adapter_gate_coupler_params = []
            self._initial_adapter_gate_input_adapter_params = []
            return
        self._initial_score_adapter_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.score_adapter.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self._initial_adapter_gate_coupler_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.reliability_bias_gate_adapter_coupler.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self._initial_adapter_gate_input_adapter_params = [
            parameter.detach().clone()
            for parameter in self.model.fusion_module.reliability_bias_gate_adapter_input_adapter.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        self.model.freeze_attention_source_gate_output_scale_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True)),
            train_score_adapter=self._train_score_adapter,
        )
        base_lr = float(config["train"]["learning_rate"])
        score_adapter_lr_scale = float(losses_cfg.get("score_adapter_lr_scale", 0.15))
        score_adapter_params = list(self.model.fusion_module.score_adapter.parameters()) if self._train_score_adapter else []
        score_adapter_param_ids = {id(parameter) for parameter in score_adapter_params}
        gate_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in score_adapter_param_ids
        ]
        param_groups = []
        if gate_params:
            param_groups.append({"params": gate_params, "lr": base_lr})
        if score_adapter_params:
            param_groups.append({"params": score_adapter_params, "lr": base_lr * score_adapter_lr_scale})
        if not param_groups:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam(param_groups)

    def update_model(self) -> dict[str, float]:
        metrics = super().update_model()
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        gate_output_scale = self.model.fusion_module.get_gate_output_scale_tensor(
            device=self.device,
            dtype=torch.float32,
        )
        metrics["loss/gate_output_scale_l2"] = float(
            float(losses_cfg.get("gate_output_scale_l2_coef", 0.0)) * gate_output_scale.pow(2).mean().detach().cpu().item()
        )
        metrics["train/vision_gate_output_scale"] = float(gate_output_scale.detach().reshape(-1)[0].cpu().item())
        return metrics


class Stage37f29VisionThresholdShiftTrainer(Stage37f28VisionGateOutputScaleTrainer):
    """V3.7f2.9 trainer with conditional vision threshold shift."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f28VisionGateOutputScaleTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        self.model.freeze_attention_source_gate_threshold_shift_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        gate_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not gate_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam([{"params": gate_params, "lr": base_lr}])

    def update_model(self) -> dict[str, float]:
        metrics = super().update_model()
        losses_cfg = canonicalize_gate_loss_keys(self.raw_config["losses"])
        threshold_alpha = self.model.fusion_module.get_gate_threshold_shift_alpha_tensor(
            device=self.device,
            dtype=torch.float32,
        )
        source_logit_bias = self.model.fusion_module.get_gate_source_logit_bias_tensor(
            device=self.device,
            dtype=torch.float32,
        )
        metrics["loss/gate_threshold_shift_alpha_l2"] = float(
            float(losses_cfg.get("gate_threshold_shift_alpha_l2_coef", 0.0))
            * threshold_alpha.pow(2).mean().detach().cpu().item()
        )
        metrics["loss/gate_source_logit_bias_l2"] = float(
            float(losses_cfg.get("gate_source_logit_bias_l2_coef", 0.0))
            * source_logit_bias.pow(2).mean().detach().cpu().item()
        )
        metrics["train/vision_threshold_shift_alpha"] = float(threshold_alpha.detach().reshape(-1)[0].cpu().item())
        metrics["train/vision_source_logit_bias"] = float(source_logit_bias.detach().reshape(-1)[0].cpu().item())
        return metrics


class Stage37f210CancelContrastTrainer(Stage37f29VisionThresholdShiftTrainer):
    """V3.7f2.10 trainer with common-mode cancel and contrastive vision shift."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f28VisionGateOutputScaleTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        self.model.freeze_attention_source_gate_cancel_contrast_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        gate_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not gate_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam([{"params": gate_params, "lr": base_lr}])


class Stage37f211CancelRescueTrainer(Stage37f210CancelContrastTrainer):
    """V3.7f2.11 trainer with negative common cancel and positive mild rescue."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f28VisionGateOutputScaleTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        self.model.freeze_attention_source_gate_cancel_rescue_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        gate_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not gate_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam([{"params": gate_params, "lr": base_lr}])


class Stage37f212ExcessCancelSelectiveRescueTrainer(Stage37f211CancelRescueTrainer):
    """V3.7f2.12 trainer with excess-only cancel and teacher-centered selective rescue."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f28VisionGateOutputScaleTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        self._train_score_adapter = False
        self.model.freeze_attention_source_gate_excess_cancel_selective_rescue_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        gate_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not gate_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam([{"params": gate_params, "lr": base_lr}])


class Stage37f213AnchorCancelSelectorRescueTrainer(Stage37f212ExcessCancelSelectiveRescueTrainer):
    """V3.7f2.13 trainer with anchor cancel and thresholded selector rescue."""

    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        super(Stage37f28VisionGateOutputScaleTrainer, self).__init__(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
        )
        losses_cfg = canonicalize_gate_loss_keys(config["losses"])
        if bool(config["candidates"][candidate_name].get("eval_only", False)):
            return
        self._train_score_adapter = False
        self.model.freeze_attention_source_gate_anchor_cancel_selector_rescue_micro_unfreeze(
            train_gate_input_adapter=bool(losses_cfg.get("train_gate_input_adapter", True))
        )
        base_lr = float(config["train"]["learning_rate"])
        gate_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not gate_params:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros((), device=self.device))], lr=0.0)
        else:
            self.optimizer = torch.optim.Adam([{"params": gate_params, "lr": base_lr}])


def compute_v3_4_relative_gap_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eps = 1e-6
    alpha = 4.0
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)

        normal_scores = np.asarray(
            [
                normal["attention_score_mean_vision"],
                normal["attention_score_mean_distance"],
                normal["attention_score_mean_ego"],
                normal["attention_score_mean_route"],
            ],
            dtype=np.float64,
        )
        mild_scores = np.asarray(
            [
                mild_vision["attention_score_mean_vision"],
                mild_vision["attention_score_mean_distance"],
                mild_vision["attention_score_mean_ego"],
                mild_vision["attention_score_mean_route"],
            ],
            dtype=np.float64,
        )
        normal_centered = normal_scores - normal_scores.mean()
        mild_centered = mild_scores - mild_scores.mean()

        route_gap = (normal_centered[0] - normal_centered[3]) - (mild_centered[0] - mild_centered[3])
        distance_gap = (normal_centered[0] - normal_centered[1]) - (mild_centered[0] - mild_centered[1])

        normal_attention = np.asarray(
            [
                normal["attention_mean_vision"],
                normal["attention_mean_distance"],
                normal["attention_mean_ego"],
                normal["attention_mean_route"],
            ],
            dtype=np.float64,
        )
        mild_attention = np.asarray(
            [
                mild_vision["attention_mean_vision"],
                mild_vision["attention_mean_distance"],
                mild_vision["attention_mean_ego"],
                mild_vision["attention_mean_route"],
            ],
            dtype=np.float64,
        )
        route_logweight_gap = (
            (np.log(normal_attention[0] + eps) - np.log(normal_attention[3] + eps))
            - (np.log(mild_attention[0] + eps) - np.log(mild_attention[3] + eps))
        )
        distance_logweight_gap = (
            (np.log(normal_attention[0] + eps) - np.log(normal_attention[1] + eps))
            - (np.log(mild_attention[0] + eps) - np.log(mild_attention[1] + eps))
        )
        coupling_error = abs(
            (normal_centered[0] - mild_centered[0])
            - alpha
            * (
                np.log(float(normal["reliability_mean_vision"]) + eps)
                - np.log(float(mild_vision["reliability_mean_vision"]) + eps)
            )
        )
        drift_mean = float(np.mean(np.square(normal_centered)) + np.mean(np.square(mild_centered))) / 2.0
        rows.append(
            {
                "candidate_name": candidate_name,
                "vision_attention_score_clean_minus_mild": float(normal_scores[0] - mild_scores[0]),
                "vision_attention_relative_score_clean_minus_mild": float(
                    (normal_scores[0] - np.mean(normal_scores[1:])) - (mild_scores[0] - np.mean(mild_scores[1:]))
                ),
                "vision_attention_clean_minus_mild": float(normal_attention[0] - mild_attention[0]),
                "vision_route_score_gap_clean_minus_mild": float(route_gap),
                "vision_distance_score_gap_clean_minus_mild": float(distance_gap),
                "vision_route_logweight_gap_clean_minus_mild": float(route_logweight_gap),
                "vision_distance_logweight_gap_clean_minus_mild": float(distance_logweight_gap),
                "vision_score_reliability_coupling_error": float(coupling_error),
                "attention_score_center_drift_mean": float(drift_mean),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_4_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"

    performance_guardrails = (
        summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.05
    )
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] > 0.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and performance_guardrails
    ):
        return "PASS"

    near_attention_count = sum(
        [
            summary["vision_attention_score_clean_minus_mild"] > -0.02,
            summary["vision_attention_relative_score_clean_minus_mild"] > -0.02,
            summary["vision_attention_clean_minus_mild"] > -0.002,
        ]
    )
    near_gap_count = sum(
        [
            summary["vision_route_score_gap_clean_minus_mild"] > -0.01,
            summary["vision_distance_score_gap_clean_minus_mild"] > -0.01,
        ]
    )
    if near_attention_count >= 2 and near_gap_count >= 1 and performance_guardrails:
        return "WARN"

    if (
        summary["vision_attention_score_clean_minus_mild"] <= baseline_summary["vision_attention_score_clean_minus_mild"]
        or summary["vision_attention_relative_score_clean_minus_mild"]
        <= baseline_summary["vision_attention_relative_score_clean_minus_mild"]
        or summary["vision_attention_clean_minus_mild"] <= baseline_summary["vision_attention_clean_minus_mild"]
        or summary["vision_route_score_gap_clean_minus_mild"] < -0.03
        or summary["vision_distance_score_gap_clean_minus_mild"] < -0.03
        or summary["normal_success_rate"] < baseline_summary["normal_success_rate"] - 0.03
        or summary["mild_success_rate"] < baseline_summary["mild_success_rate"] - 0.03
    ):
        return "FAIL"
    return "WARN"


def build_v3_4_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    relative_gap_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []

    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        relative_gap = relative_gap_df[relative_gap_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(relative_gap["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(relative_gap["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(relative_gap["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(relative_gap["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(relative_gap["vision_distance_score_gap_clean_minus_mild"]),
                "vision_route_logweight_gap_clean_minus_mild": float(relative_gap["vision_route_logweight_gap_clean_minus_mild"]),
                "vision_distance_logweight_gap_clean_minus_mild": float(relative_gap["vision_distance_logweight_gap_clean_minus_mild"]),
                "vision_score_reliability_coupling_error": float(relative_gap["vision_score_reliability_coupling_error"]),
                "attention_score_center_drift_mean": float(relative_gap["attention_score_center_drift_mean"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.4 summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()

    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_route_gap_delta"]) * summary["vision_route_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_distance_gap_delta"]) * summary["vision_distance_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_relative_score_delta"]) * summary["vision_attention_relative_score_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["degradation"]) * summary["normal_to_mild_success_drop"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        pass_fail_values.append(
            evaluate_v3_4_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values

    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_attention_relative_score_clean_minus_mild",
            "normal_success_rate",
            "normal_to_mild_success_drop",
            "consistency_gain_score",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, False, True, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_4_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    relative_gap_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Relative Gap Summary",
        "",
        relative_gap_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_4b_hybrid_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base_df = compute_v3_4_relative_gap_metrics(df)
    rows = []
    eps = 1e-6
    for _, row in base_df.iterrows():
        candidate_name = str(row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)

        normal_scores = np.asarray(
            [
                normal["attention_score_mean_vision"],
                normal["attention_score_mean_distance"],
                normal["attention_score_mean_ego"],
                normal["attention_score_mean_route"],
            ],
            dtype=np.float64,
        )
        mild_scores = np.asarray(
            [
                mild_vision["attention_score_mean_vision"],
                mild_vision["attention_score_mean_distance"],
                mild_vision["attention_score_mean_ego"],
                mild_vision["attention_score_mean_route"],
            ],
            dtype=np.float64,
        )
        normal_centered = normal_scores - normal_scores.mean()
        mild_centered = mild_scores - mild_scores.mean()
        normal_attention = np.asarray(
            [
                normal["attention_mean_vision"],
                normal["attention_mean_distance"],
                normal["attention_mean_ego"],
                normal["attention_mean_route"],
            ],
            dtype=np.float64,
        )
        mild_attention = np.asarray(
            [
                mild_vision["attention_mean_vision"],
                mild_vision["attention_mean_distance"],
                mild_vision["attention_mean_ego"],
                mild_vision["attention_mean_route"],
            ],
            dtype=np.float64,
        )

        sign_score_margin_satisfied = float((normal_centered[0] - mild_centered[0]) > 0.0)
        sign_weight_margin_satisfied = float((normal_attention[0] - mild_attention[0]) > 0.0)
        route_pairgap_satisfied = float(row["vision_route_score_gap_clean_minus_mild"] > 0.0)
        distance_pairgap_satisfied = float(row["vision_distance_score_gap_clean_minus_mild"] > 0.0)
        hybrid_gain = float(
            (normal_centered[0] - mild_centered[0])
            + (normal_attention[0] - mild_attention[0])
            + row["vision_route_score_gap_clean_minus_mild"]
            + row["vision_distance_score_gap_clean_minus_mild"]
        )
        out = dict(row)
        out["vision_sign_score_margin_satisfied_ratio"] = sign_score_margin_satisfied
        out["vision_sign_weight_margin_satisfied_ratio"] = sign_weight_margin_satisfied
        out["vision_route_pairgap_margin_satisfied_ratio"] = route_pairgap_satisfied
        out["vision_distance_pairgap_margin_satisfied_ratio"] = distance_pairgap_satisfied
        out["hybrid_sign_push_gain_score"] = hybrid_gain
        out["attention_score_center_drift_mean"] = float(
            out["attention_score_center_drift_mean"] + eps * 0.0
        )
        rows.append(out)
    return pd.DataFrame(rows)


def evaluate_v3_4b_pass_fail(summary: dict[str, Any], reference_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"

    performance_guardrails = (
        summary["normal_success_rate"] >= reference_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= reference_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= reference_summary["normal_to_mild_return_drop"] + 0.10
    )
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] > 0.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and performance_guardrails
    ):
        return "PASS"

    if (
        summary["vision_attention_score_clean_minus_mild"] <= -0.05
        or summary["vision_attention_clean_minus_mild"] <= -0.005
        or summary["vision_route_score_gap_clean_minus_mild"] <= -0.04
        or summary["vision_distance_score_gap_clean_minus_mild"] <= -0.04
        or summary["normal_success_rate"] < reference_summary["normal_success_rate"] - 0.03
        or summary["mild_success_rate"] < reference_summary["mild_success_rate"] - 0.03
    ):
        return "FAIL"

    if (
        summary["vision_attention_score_clean_minus_mild"] > -0.03
        and summary["vision_attention_relative_score_clean_minus_mild"] > -0.03
        and summary["vision_attention_clean_minus_mild"] > -0.0025
        and summary["vision_route_score_gap_clean_minus_mild"] > -0.02
        and summary["vision_distance_score_gap_clean_minus_mild"] > -0.02
        and performance_guardrails
    ):
        return "WARN"
    return "FAIL"


def build_v3_4b_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        hybrid = hybrid_df[hybrid_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(hybrid["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(hybrid["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(hybrid["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(hybrid["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(hybrid["vision_distance_score_gap_clean_minus_mild"]),
                "vision_route_logweight_gap_clean_minus_mild": float(hybrid["vision_route_logweight_gap_clean_minus_mild"]),
                "vision_distance_logweight_gap_clean_minus_mild": float(hybrid["vision_distance_logweight_gap_clean_minus_mild"]),
                "vision_score_reliability_coupling_error": float(hybrid["vision_score_reliability_coupling_error"]),
                "attention_score_center_drift_mean": float(hybrid["attention_score_center_drift_mean"]),
                "vision_sign_score_margin_satisfied_ratio": float(hybrid["vision_sign_score_margin_satisfied_ratio"]),
                "vision_sign_weight_margin_satisfied_ratio": float(hybrid["vision_sign_weight_margin_satisfied_ratio"]),
                "vision_route_pairgap_margin_satisfied_ratio": float(hybrid["vision_route_pairgap_margin_satisfied_ratio"]),
                "vision_distance_pairgap_margin_satisfied_ratio": float(hybrid["vision_distance_pairgap_margin_satisfied_ratio"]),
                "hybrid_sign_push_gain_score": float(hybrid["hybrid_sign_push_gain_score"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.4b summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()

    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_route_gap_delta"]) * summary["vision_route_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_distance_gap_delta"]) * summary["vision_distance_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights.get("mild_success", 0.0)) * summary["mild_success_rate"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights.get("drift_penalty", 0.0)) * summary["attention_score_center_drift_mean"]
        )
        pass_fail_values.append(
            evaluate_v3_4b_pass_fail(
                summary=summary,
                reference_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values

    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "normal_success_rate",
            "mild_success_rate",
            "normal_to_mild_return_drop",
            "consistency_gain_score",
            "attention_score_center_drift_mean",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, False, True, False, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_4b_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Hybrid Summary",
        "",
        hybrid_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_5_adapter_metrics(df: pd.DataFrame) -> pd.DataFrame:
    hybrid_df = compute_v3_4b_hybrid_metrics(df)
    rows = []
    for _, hybrid in hybrid_df.iterrows():
        candidate_name = str(hybrid["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        adapter_cols = [f"attention_score_adapter_delta_norm_mean_{name}" for name in MODALITY_NAMES]
        rows.append(
            {
                **hybrid.to_dict(),
                "attention_score_adapter_delta_norm_mean": float(sub[adapter_cols].mean(numeric_only=True).mean()),
                "attention_score_adapter_delta_norm_mean_vision_probe": float(
                    np.mean([float(mild_vision[column]) for column in adapter_cols])
                ),
                "vision_adapter_delta_norm_clean_minus_mild": float(
                    normal["attention_score_adapter_delta_norm_mean_vision"]
                    - mild_vision["attention_score_adapter_delta_norm_mean_vision"]
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_v3_6_relative_head_metrics(df: pd.DataFrame) -> pd.DataFrame:
    adapter_df = compute_v3_5_adapter_metrics(df)
    rows = []
    for _, adapter in adapter_df.iterrows():
        candidate_name = str(adapter["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        normal_gaps = np.asarray(
            [
                normal.get("attention_relative_gap_score_mean_vision_distance", 0.0),
                normal.get("attention_relative_gap_score_mean_vision_ego", 0.0),
                normal.get("attention_relative_gap_score_mean_vision_route", 0.0),
            ],
            dtype=np.float64,
        )
        mild_gaps = np.asarray(
            [
                mild_vision.get("attention_relative_gap_score_mean_vision_distance", 0.0),
                mild_vision.get("attention_relative_gap_score_mean_vision_ego", 0.0),
                mild_vision.get("attention_relative_gap_score_mean_vision_route", 0.0),
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                **adapter.to_dict(),
                "relative_head_gap_mean_vision_distance": float(sub["attention_relative_gap_score_mean_vision_distance"].mean()),
                "relative_head_gap_mean_vision_ego": float(sub["attention_relative_gap_score_mean_vision_ego"].mean()),
                "relative_head_gap_mean_vision_route": float(sub["attention_relative_gap_score_mean_vision_route"].mean()),
                "relative_head_activity_norm_mean": float(sub["attention_relative_gap_norm_mean"].mean()),
                "vision_relative_gap_margin_satisfied_ratio": float(
                    (adapter["vision_route_score_gap_clean_minus_mild"] > 0.0)
                    and (adapter["vision_distance_score_gap_clean_minus_mild"] > 0.0)
                ),
                "relative_head_gap_clean_minus_mild": float((normal_gaps - mild_gaps).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_5_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.10
    ):
        return "PASS"
    return "WARN"


def build_v3_5_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    adapter_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        adapter = adapter_df[adapter_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(adapter["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(adapter["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(adapter["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(adapter["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(adapter["vision_distance_score_gap_clean_minus_mild"]),
                "attention_score_adapter_delta_norm_mean": float(adapter["attention_score_adapter_delta_norm_mean"]),
                "attention_score_adapter_delta_norm_mean_vision_probe": float(adapter["attention_score_adapter_delta_norm_mean_vision_probe"]),
                "vision_adapter_delta_norm_clean_minus_mild": float(adapter["vision_adapter_delta_norm_clean_minus_mild"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.5 summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()

    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_route_gap_delta"]) * summary["vision_route_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_distance_gap_delta"]) * summary["vision_distance_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["mild_success"]) * summary["mild_success_rate"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["adapter_activity"]) * summary["attention_score_adapter_delta_norm_mean"]
        )
        pass_fail_values.append(
            evaluate_v3_5_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "normal_success_rate",
            "mild_success_rate",
            "normal_to_mild_return_drop",
            "consistency_gain_score",
            "attention_score_adapter_delta_norm_mean",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, False, True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_5_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    adapter_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Adapter Summary",
        "",
        adapter_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_v3_6_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.10
    ):
        return "PASS"
    return "WARN"


def build_v3_6_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    relative_head_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        relative_head = relative_head_df[relative_head_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(relative_head["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(relative_head["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(relative_head["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(relative_head["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(relative_head["vision_distance_score_gap_clean_minus_mild"]),
                "relative_head_gap_mean_vision_distance": float(relative_head["relative_head_gap_mean_vision_distance"]),
                "relative_head_gap_mean_vision_ego": float(relative_head["relative_head_gap_mean_vision_ego"]),
                "relative_head_gap_mean_vision_route": float(relative_head["relative_head_gap_mean_vision_route"]),
                "relative_head_activity_norm_mean": float(relative_head["relative_head_activity_norm_mean"]),
                "vision_relative_gap_margin_satisfied_ratio": float(relative_head["vision_relative_gap_margin_satisfied_ratio"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.6 summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()
    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_route_gap_delta"]) * summary["vision_route_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_distance_gap_delta"]) * summary["vision_distance_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["mild_success"]) * summary["mild_success_rate"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["consistency_gain"]) * summary["consistency_gain_score"]
            + float(tiebreak_weights["relative_head_activity"]) * summary["relative_head_activity_norm_mean"]
        )
        pass_fail_values.append(
            evaluate_v3_6_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "normal_success_rate",
            "mild_success_rate",
            "normal_to_mild_return_drop",
            "consistency_gain_score",
            "relative_head_activity_norm_mean",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, False, True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_6_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    relative_head_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    is_calibration = bool(score_df.get("calibration_only_flag", pd.Series(dtype=float)).fillna(0.0).max() > 0.5)
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
    ]
    if is_calibration:
        lines.extend(
            [
                "> Warning: this experiment is tagged as `calibration_only` because RL core losses are disabled.",
                "> Interpret results as score calibration behavior, not full robustness training.",
                "",
            ]
        )
    lines.extend(
        [
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Relative Head Summary",
        "",
        relative_head_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_7_reliability_bias_metrics(df: pd.DataFrame) -> pd.DataFrame:
    relative_head_df = compute_v3_6_relative_head_metrics(df)
    rows = []
    for _, relative_head in relative_head_df.iterrows():
        candidate_name = str(relative_head["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        corr_df = sub[
            [
                "attention_bias_mean_vision",
                "reliability_mean_vision",
                "logit_mean_vision_clean",
            ]
        ].replace([np.inf, -np.inf], np.nan)
        bias_reliability_corr = corr_df["attention_bias_mean_vision"].corr(corr_df["reliability_mean_vision"])
        bias_logit_corr = corr_df["attention_bias_mean_vision"].corr(corr_df["logit_mean_vision_clean"])
        rows.append(
            {
                **relative_head.to_dict(),
                "vision_bias_clean_minus_mild": float(
                    normal["attention_bias_mean_vision"] - mild_vision["attention_bias_mean_vision"]
                ),
                "distance_bias_clean_minus_mild": float(
                    normal["attention_bias_mean_distance"] - mild_vision["attention_bias_mean_distance"]
                ),
                "ego_bias_clean_minus_mild": float(
                    normal["attention_bias_mean_ego"] - mild_vision["attention_bias_mean_ego"]
                ),
                "route_bias_clean_minus_mild": float(
                    normal["attention_bias_mean_route"] - mild_vision["attention_bias_mean_route"]
                ),
                "bias_head_activity_norm_mean": float(sub["attention_bias_norm_mean"].mean()),
                "vision_bias_margin_satisfied_ratio": float(
                    (normal["attention_bias_mean_vision"] - mild_vision["attention_bias_mean_vision"]) > 0.0
                ),
                "vision_bias_to_reliability_corr": float(0.0 if pd.isna(bias_reliability_corr) else bias_reliability_corr),
                "vision_bias_to_logit_corr": float(0.0 if pd.isna(bias_logit_corr) else bias_logit_corr),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_7_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_bias_clean_minus_mild"] > 0.0
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.10
    ):
        return "PASS"
    return "WARN"


def build_v3_7_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        bias = bias_df[bias_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(bias["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(bias["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(bias["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(bias["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(bias["vision_distance_score_gap_clean_minus_mild"]),
                "vision_bias_clean_minus_mild": float(bias["vision_bias_clean_minus_mild"]),
                "distance_bias_clean_minus_mild": float(bias["distance_bias_clean_minus_mild"]),
                "ego_bias_clean_minus_mild": float(bias["ego_bias_clean_minus_mild"]),
                "route_bias_clean_minus_mild": float(bias["route_bias_clean_minus_mild"]),
                "bias_head_activity_norm_mean": float(bias["bias_head_activity_norm_mean"]),
                "vision_bias_margin_satisfied_ratio": float(bias["vision_bias_margin_satisfied_ratio"]),
                "vision_bias_to_reliability_corr": float(bias["vision_bias_to_reliability_corr"]),
                "vision_bias_to_logit_corr": float(bias["vision_bias_to_logit_corr"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.7 summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()
    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_route_gap_delta"]) * summary["vision_route_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_distance_gap_delta"]) * summary["vision_distance_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_bias_delta"]) * summary["vision_bias_clean_minus_mild"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["mild_success"]) * summary["mild_success_rate"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["bias_activity"]) * summary["bias_head_activity_norm_mean"]
        )
        pass_fail_values.append(
            evaluate_v3_7_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "vision_bias_clean_minus_mild",
            "normal_success_rate",
            "mild_success_rate",
            "normal_to_mild_return_drop",
            "bias_head_activity_norm_mean",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, False, False, True, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    is_calibration = bool(score_df.get("calibration_only_flag", pd.Series(dtype=float)).fillna(0.0).max() > 0.5)
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
    ]
    if is_calibration:
        lines.extend(
            [
                "> Warning: this experiment is tagged as `calibration_only` because RL core losses are disabled.",
                "> Interpret results as score calibration behavior, not full robustness training.",
                "",
            ]
        )
    lines.extend(
        [
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Bias Summary",
        "",
        bias_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_7b_clean_preserving_metrics(df: pd.DataFrame) -> pd.DataFrame:
    bias_df = compute_v3_7_reliability_bias_metrics(df)
    rows = []
    for _, bias_row in bias_df.iterrows():
        candidate_name = str(bias_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        clean_bias_abs_mean = float(normal.get("attention_bias_abs_mean", 0.0))
        mild_vision_bias_abs_mean = float(mild_vision.get("attention_bias_abs_mean", 0.0))
        rows.append(
            {
                **bias_row.to_dict(),
                "clean_bias_abs_mean": clean_bias_abs_mean,
                "mild_vision_bias_abs_mean": mild_vision_bias_abs_mean,
                "clean_score_anchor_error": float(normal.get("clean_score_anchor_error", 0.0)),
                "clean_action_anchor_error": float(normal.get("clean_action_anchor_error", 0.0)),
                "clean_bias_suppression_ratio": float(
                    1.0 - (clean_bias_abs_mean / max(mild_vision_bias_abs_mean, 1e-6))
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_7b_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    return evaluate_v3_7_pass_fail(summary=summary, baseline_summary=baseline_summary, is_baseline=is_baseline)


def build_v3_7b_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    clean_anchor_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        clean_anchor = clean_anchor_df[clean_anchor_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(clean_anchor["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(
                    clean_anchor["vision_attention_relative_score_clean_minus_mild"]
                ),
                "vision_attention_clean_minus_mild": float(clean_anchor["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(clean_anchor["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(clean_anchor["vision_distance_score_gap_clean_minus_mild"]),
                "vision_bias_clean_minus_mild": float(clean_anchor["vision_bias_clean_minus_mild"]),
                "distance_bias_clean_minus_mild": float(clean_anchor["distance_bias_clean_minus_mild"]),
                "ego_bias_clean_minus_mild": float(clean_anchor["ego_bias_clean_minus_mild"]),
                "route_bias_clean_minus_mild": float(clean_anchor["route_bias_clean_minus_mild"]),
                "bias_head_activity_norm_mean": float(clean_anchor["bias_head_activity_norm_mean"]),
                "vision_bias_margin_satisfied_ratio": float(clean_anchor["vision_bias_margin_satisfied_ratio"]),
                "vision_bias_to_reliability_corr": float(clean_anchor["vision_bias_to_reliability_corr"]),
                "vision_bias_to_logit_corr": float(clean_anchor["vision_bias_to_logit_corr"]),
                "clean_bias_abs_mean": float(clean_anchor["clean_bias_abs_mean"]),
                "mild_vision_bias_abs_mean": float(clean_anchor["mild_vision_bias_abs_mean"]),
                "clean_score_anchor_error": float(clean_anchor["clean_score_anchor_error"]),
                "clean_action_anchor_error": float(clean_anchor["clean_action_anchor_error"]),
                "clean_bias_suppression_ratio": float(clean_anchor["clean_bias_suppression_ratio"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.7b summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()
    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_route_gap_delta"]) * summary["vision_route_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["vision_distance_gap_delta"]) * summary["vision_distance_score_gap_clean_minus_mild"]
            + float(tiebreak_weights["clean_score_anchor"]) * summary["clean_score_anchor_error"]
            + float(tiebreak_weights["clean_action_anchor"]) * summary["clean_action_anchor_error"]
            + float(tiebreak_weights["bias_activity"]) * summary["bias_head_activity_norm_mean"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
        )
        pass_fail_values.append(
            evaluate_v3_7b_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "normal_success_rate",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_route_score_gap_clean_minus_mild",
            "vision_distance_score_gap_clean_minus_mild",
            "clean_score_anchor_error",
            "clean_action_anchor_error",
            "bias_head_activity_norm_mean",
            "normal_to_mild_return_drop",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, True, True, False, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7b_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    clean_anchor_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    is_calibration = bool(score_df.get("calibration_only_flag", pd.Series(dtype=float)).fillna(0.0).max() > 0.5)
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
    ]
    if is_calibration:
        lines.extend(
            [
                "> Warning: this experiment is tagged as `calibration_only` because RL core losses are disabled.",
                "> Interpret results as score calibration behavior, not full robustness training.",
                "",
            ]
        )
    lines.extend(
        [
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "## Clean Anchor Summary",
        "",
        clean_anchor_df.to_markdown(index=False),
        "",
        "## Target-Logit Shapley Summary",
        "",
        targetlogit_df.to_markdown(index=False),
        "",
        "## Degradation Summary",
        "",
        degradation_df.to_markdown(index=False),
        "",
        "## Tie-break Summary",
        "",
        tiebreak_df.to_markdown(index=False),
        "",
        "## Consistency Summary",
        "",
        consistency_df.to_markdown(index=False),
        "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_7c_dual_teacher_gated_bias_metrics(df: pd.DataFrame) -> pd.DataFrame:
    clean_anchor_df = compute_v3_7b_clean_preserving_metrics(df)
    rows = []
    for _, clean_anchor_row in clean_anchor_df.iterrows():
        candidate_name = str(clean_anchor_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        vision_gate_clean = float(normal.get("attention_bias_gate_mean_vision", 0.0))
        vision_gate_mild = float(mild_vision.get("attention_bias_gate_mean_vision", 0.0))
        distance_gate_clean = float(normal.get("attention_bias_gate_mean_distance", 0.0))
        route_gate_clean = float(normal.get("attention_bias_gate_mean_route", 0.0))
        clean_bias = float(normal.get("attention_bias_abs_mean", 0.0))
        mild_bias = float(mild_vision.get("attention_bias_abs_mean", 0.0))
        rows.append(
            {
                **clean_anchor_row.to_dict(),
                "vision_gate_clean_mean": vision_gate_clean,
                "vision_gate_mild_mean": vision_gate_mild,
                "vision_gate_mild_minus_clean": float(vision_gate_mild - vision_gate_clean),
                "distance_gate_clean_mean": distance_gate_clean,
                "route_gate_clean_mean": route_gate_clean,
                "clean_gate_suppression_ratio": float(1.0 - vision_gate_clean / max(vision_gate_mild, 1e-6)),
                "mild_gate_activation_ratio": float(vision_gate_mild),
                "gate_bias_overlap_score": float(min(vision_gate_mild, mild_bias) - min(vision_gate_clean, clean_bias)),
                "clean_teacher_score_anchor_error": float(normal.get("clean_score_anchor_error", 0.0)),
                "clean_teacher_action_anchor_error": float(normal.get("clean_action_anchor_error", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def compute_v3_7d_selective_gate_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    gate_df = compute_v3_7c_dual_teacher_gated_bias_metrics(df)
    rows = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        thresholds = candidate_thresholds.get(candidate_name, {}) if candidate_thresholds is not None else {}
        clean_gate_cap = float(thresholds.get("clean_gate_max", 0.02))
        mild_gate_floor = float(thresholds.get("mild_gate_min", 0.08))
        gate_gap_margin = float(thresholds.get("gate_gap_margin", 0.05))
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"]
        mild_vision = sub[sub["corruption_type"] == "mild_vision"]
        clean_gate_values = normal["attention_bias_gate_mean_vision"].dropna().to_numpy()
        mild_gate_values = mild_vision["attention_bias_gate_mean_vision"].dropna().to_numpy()
        clean_gate_ratio = float((clean_gate_values <= clean_gate_cap).mean()) if clean_gate_values.size > 0 else 0.0
        mild_gate_ratio = float((mild_gate_values >= mild_gate_floor).mean()) if mild_gate_values.size > 0 else 0.0
        shared_count = min(clean_gate_values.size, mild_gate_values.size)
        if shared_count > 0:
            gap_satisfied_ratio = float(
                ((mild_gate_values[:shared_count] - clean_gate_values[:shared_count]) >= gate_gap_margin).mean()
            )
        else:
            gap_satisfied_ratio = 0.0
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_gate_clean_below_cap_ratio": clean_gate_ratio,
                "vision_gate_mild_above_floor_ratio": mild_gate_ratio,
                "vision_gate_gap_margin_satisfied_ratio": gap_satisfied_ratio,
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_7c_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_gate_mild_minus_clean"] > 0.0
        and summary["clean_gate_suppression_ratio"] > 0.0
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.10
    ):
        return "PASS"
    return "WARN"


def evaluate_v3_7d_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_targetlogit_shapley_clean_minus_mild"] > 0.0
        and summary["vision_reliability_clean_minus_mild"] > 0.0
        and summary["vision_logit_clean_minus_mild"] > 0.0
        and summary["vision_gate_mild_minus_clean"] > 0.03
        and summary["vision_gate_clean_below_cap_ratio"] >= 0.8
        and summary["vision_gate_mild_above_floor_ratio"] >= 0.8
        and summary["vision_gate_gap_margin_satisfied_ratio"] >= 0.7
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.10
    ):
        return "PASS"
    return "WARN"


def build_v3_7c_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        gate = gate_df[gate_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(gate["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(gate["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(gate["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(gate["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(gate["vision_distance_score_gap_clean_minus_mild"]),
                "vision_bias_clean_minus_mild": float(gate["vision_bias_clean_minus_mild"]),
                "distance_bias_clean_minus_mild": float(gate["distance_bias_clean_minus_mild"]),
                "ego_bias_clean_minus_mild": float(gate["ego_bias_clean_minus_mild"]),
                "route_bias_clean_minus_mild": float(gate["route_bias_clean_minus_mild"]),
                "bias_head_activity_norm_mean": float(gate["bias_head_activity_norm_mean"]),
                "clean_bias_abs_mean": float(gate["clean_bias_abs_mean"]),
                "mild_vision_bias_abs_mean": float(gate["mild_vision_bias_abs_mean"]),
                "clean_teacher_score_anchor_error": float(gate["clean_teacher_score_anchor_error"]),
                "clean_teacher_action_anchor_error": float(gate["clean_teacher_action_anchor_error"]),
                "vision_gate_clean_mean": float(gate["vision_gate_clean_mean"]),
                "vision_gate_mild_mean": float(gate["vision_gate_mild_mean"]),
                "vision_gate_mild_minus_clean": float(gate["vision_gate_mild_minus_clean"]),
                "distance_gate_clean_mean": float(gate["distance_gate_clean_mean"]),
                "route_gate_clean_mean": float(gate["route_gate_clean_mean"]),
                "clean_gate_suppression_ratio": float(gate["clean_gate_suppression_ratio"]),
                "mild_gate_activation_ratio": float(gate["mild_gate_activation_ratio"]),
                "gate_bias_overlap_score": float(gate["gate_bias_overlap_score"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.7c summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()
    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_gate_delta"]) * summary["vision_gate_mild_minus_clean"]
            + float(tiebreak_weights["clean_score_anchor"]) * summary["clean_teacher_score_anchor_error"]
            + float(tiebreak_weights["clean_action_anchor"]) * summary["clean_teacher_action_anchor_error"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["bias_activity"]) * summary["bias_head_activity_norm_mean"]
        )
        pass_fail_values.append(
            evaluate_v3_7c_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "normal_success_rate",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_gate_mild_minus_clean",
            "clean_teacher_score_anchor_error",
            "clean_teacher_action_anchor_error",
            "normal_to_mild_return_drop",
            "consistency_gain_score",
            "bias_head_activity_norm_mean",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def build_v3_7d_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    provisional_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
        targetlogit = targetlogit_df[targetlogit_df["candidate_name"] == candidate_name].iloc[0]
        gate = gate_df[gate_df["candidate_name"] == candidate_name].iloc[0]
        provisional_rows.append(
            {
                "candidate_name": candidate_name,
                "normal_success_rate": float(normal["eval_success_rate"]),
                "normal_return": float(normal["eval_return"]),
                "mild_success_rate": float(sub[sub["env_mode"] == "mild"]["eval_success_rate"].mean()),
                "normal_to_mild_success_drop": float(deg["normal_to_mild_success_drop"].mean()),
                "normal_to_mild_return_drop": float(deg["normal_to_mild_return_drop"].mean()),
                "normal_to_mild_progress_drop": float(deg["normal_to_mild_progress_drop"].mean()),
                "consistency_loss": float(cons["consistency_loss"]),
                "same_state_action_diff_clean_vs_corrupt": float(cons["same_state_action_diff_clean_vs_corrupt"]),
                "consistency_gain_score": float(cons["consistency_gain_score"]),
                "vision_reliability_clean_minus_mild": float(tie["vision_reliability_clean_minus_mild"]),
                "ego_reliability_clean_minus_mild": float(tie["ego_reliability_clean_minus_mild"]),
                "vision_logit_clean_minus_mild": float(tie["vision_logit_clean_minus_mild"]),
                "ego_logit_clean_minus_mild": float(tie["ego_logit_clean_minus_mild"]),
                "saturation_warning_flag": float(tie["saturation_warning_flag"]),
                "route_distance_side_effect_flag": float(tie["route_distance_side_effect_flag"]),
                "vision_targetlogit_shapley_clean_minus_mild": float(targetlogit["vision_targetlogit_shapley_clean_minus_mild"]),
                "ego_targetlogit_shapley_clean_minus_mild": float(targetlogit["ego_targetlogit_shapley_clean_minus_mild"]),
                "vision_attention_targetlogit_sign_match_flag": float(targetlogit["vision_attention_targetlogit_sign_match_flag"]),
                "vision_attention_score_clean_minus_mild": float(gate["vision_attention_score_clean_minus_mild"]),
                "vision_attention_relative_score_clean_minus_mild": float(gate["vision_attention_relative_score_clean_minus_mild"]),
                "vision_attention_clean_minus_mild": float(gate["vision_attention_clean_minus_mild"]),
                "vision_route_score_gap_clean_minus_mild": float(gate["vision_route_score_gap_clean_minus_mild"]),
                "vision_distance_score_gap_clean_minus_mild": float(gate["vision_distance_score_gap_clean_minus_mild"]),
                "vision_bias_clean_minus_mild": float(gate["vision_bias_clean_minus_mild"]),
                "distance_bias_clean_minus_mild": float(gate["distance_bias_clean_minus_mild"]),
                "ego_bias_clean_minus_mild": float(gate["ego_bias_clean_minus_mild"]),
                "route_bias_clean_minus_mild": float(gate["route_bias_clean_minus_mild"]),
                "bias_head_activity_norm_mean": float(gate["bias_head_activity_norm_mean"]),
                "clean_bias_abs_mean": float(gate["clean_bias_abs_mean"]),
                "mild_vision_bias_abs_mean": float(gate["mild_vision_bias_abs_mean"]),
                "clean_teacher_score_anchor_error": float(gate["clean_teacher_score_anchor_error"]),
                "clean_teacher_action_anchor_error": float(gate["clean_teacher_action_anchor_error"]),
                "vision_gate_clean_mean": float(gate["vision_gate_clean_mean"]),
                "vision_gate_mild_mean": float(gate["vision_gate_mild_mean"]),
                "vision_gate_mild_minus_clean": float(gate["vision_gate_mild_minus_clean"]),
                "distance_gate_clean_mean": float(gate["distance_gate_clean_mean"]),
                "route_gate_clean_mean": float(gate["route_gate_clean_mean"]),
                "vision_gate_clean_below_cap_ratio": float(gate["vision_gate_clean_below_cap_ratio"]),
                "vision_gate_mild_above_floor_ratio": float(gate["vision_gate_mild_above_floor_ratio"]),
                "vision_gate_gap_margin_satisfied_ratio": float(gate["vision_gate_gap_margin_satisfied_ratio"]),
                "gate_bias_overlap_score": float(gate["gate_bias_overlap_score"]),
            }
        )

    score_df = pd.DataFrame(provisional_rows)
    baseline_summary = score_df[score_df["candidate_name"] == baseline_candidate]
    if baseline_summary.empty:
        raise ValueError(f"Baseline candidate missing from V3.7d summary: {baseline_candidate}")
    baseline_row = baseline_summary.iloc[0].to_dict()
    tie_break_scores = []
    pass_fail_values = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        tie_break_scores.append(
            float(tiebreak_weights["vision_sign_match"]) * summary["vision_attention_targetlogit_sign_match_flag"]
            + float(tiebreak_weights["normal_success"]) * summary["normal_success_rate"]
            + float(tiebreak_weights["vision_attention_delta"]) * summary["vision_attention_clean_minus_mild"]
            + float(tiebreak_weights["vision_attention_score_delta"]) * summary["vision_attention_score_clean_minus_mild"]
            + float(tiebreak_weights["vision_gate_delta"]) * summary["vision_gate_mild_minus_clean"]
            + float(tiebreak_weights["clean_score_anchor"]) * summary["clean_teacher_score_anchor_error"]
            + float(tiebreak_weights["clean_action_anchor"]) * summary["clean_teacher_action_anchor_error"]
            + float(tiebreak_weights["return_degradation"]) * summary["normal_to_mild_return_drop"]
            + float(tiebreak_weights["bias_activity"]) * summary["bias_head_activity_norm_mean"]
        )
        pass_fail_values.append(
            evaluate_v3_7d_pass_fail(
                summary=summary,
                baseline_summary=baseline_row,
                is_baseline=(summary["candidate_name"] == baseline_candidate),
            )
        )
    score_df["tie_break_score"] = tie_break_scores
    score_df["pass_fail"] = pass_fail_values
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "normal_success_rate",
            "vision_attention_clean_minus_mild",
            "vision_attention_score_clean_minus_mild",
            "vision_gate_mild_minus_clean",
            "clean_teacher_score_anchor_error",
            "clean_teacher_action_anchor_error",
            "normal_to_mild_return_drop",
            "consistency_gain_score",
            "bias_head_activity_norm_mean",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7c_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    is_calibration = bool(score_df.get("calibration_only_flag", pd.Series(dtype=float)).fillna(0.0).max() > 0.5)
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
    ]
    if is_calibration:
        lines.extend(
            [
                "> Warning: this experiment is tagged as `calibration_only` because RL core losses are disabled.",
                "> Interpret results as gated score calibration behavior, not full robustness training.",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Summary",
            "",
            score_df.to_markdown(index=False),
            "",
            "## Gate Summary",
            "",
            gate_df.to_markdown(index=False),
            "",
            "## Target-Logit Shapley Summary",
            "",
            targetlogit_df.to_markdown(index=False),
            "",
            "## Degradation Summary",
            "",
            degradation_df.to_markdown(index=False),
            "",
            "## Tie-break Summary",
            "",
            tiebreak_df.to_markdown(index=False),
            "",
            "## Consistency Summary",
            "",
            consistency_df.to_markdown(index=False),
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_v3_7d_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    is_calibration = bool(score_df.get("calibration_only_flag", pd.Series(dtype=float)).fillna(0.0).max() > 0.5)
    lines = [
        f"# {title}",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
    ]
    if is_calibration:
        lines.extend(
            [
                "> Warning: this experiment is tagged as `calibration_only` because RL core losses are disabled.",
                "> Interpret results as selective gate calibration behavior, not full robustness training.",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Summary",
            "",
            score_df.to_markdown(index=False),
            "",
            "## Gate Summary",
            "",
            gate_df.to_markdown(index=False),
            "",
            "## Target-Logit Shapley Summary",
            "",
            targetlogit_df.to_markdown(index=False),
            "",
            "## Degradation Summary",
            "",
            degradation_df.to_markdown(index=False),
            "",
            "## Tie-break Summary",
            "",
            tiebreak_df.to_markdown(index=False),
            "",
            "## Consistency Summary",
            "",
            consistency_df.to_markdown(index=False),
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_v3_7e_gate_only_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    return compute_v3_7d_selective_gate_metrics(df, candidate_thresholds=candidate_thresholds)


def evaluate_v3_7e_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_gate_mild_minus_clean"] > 0.03
        and summary["vision_gate_clean_below_cap_ratio"] >= 0.8
        and summary["vision_gate_mild_above_floor_ratio"] >= 0.8
        and summary["vision_gate_gap_margin_satisfied_ratio"] >= 0.7
        and summary["clean_teacher_score_anchor_error"] <= 1.0e-6
        and summary["clean_teacher_action_anchor_error"] <= 1.0e-6
    ):
        return "PASS"
    return "WARN"


def build_v3_7e_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7d_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    baseline_row = score_df[score_df["candidate_name"] == baseline_candidate].iloc[0].to_dict()
    score_df["pass_fail"] = [
        evaluate_v3_7e_pass_fail(
            summary=row.to_dict(),
            baseline_summary=baseline_row,
            is_baseline=(row["candidate_name"] == baseline_candidate),
        )
        for _, row in score_df.iterrows()
    ]
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "vision_gate_mild_minus_clean",
            "vision_attention_clean_minus_mild",
            "vision_attention_relative_score_clean_minus_mild",
            "normal_success_rate",
            "clean_teacher_score_anchor_error",
            "clean_teacher_action_anchor_error",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7e_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    return generate_v3_7d_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )


def compute_v3_7f_source_gate_residual_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    gate_df = compute_v3_7d_selective_gate_metrics(df, candidate_thresholds=candidate_thresholds)
    rows = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_source_gate_clean_mean": float(normal.get("attention_source_bias_gate_mean_vision", 0.0)),
                "vision_source_gate_mild_mean": float(mild_vision.get("attention_source_bias_gate_mean_vision", 0.0)),
                "vision_gate_residual_clean_mean": float(normal.get("attention_bias_gate_residual_mean_vision", 0.0)),
                "vision_gate_residual_mild_mean": float(mild_vision.get("attention_bias_gate_residual_mean_vision", 0.0)),
                "vision_residual_logit_clean_mean": float(
                    normal.get(
                        "attention_bias_gate_residual_logit_mean_vision",
                        normal.get("attention_bias_gate_residual_mean_vision", 0.0),
                    )
                ),
                "vision_residual_logit_mild_mean": float(
                    mild_vision.get(
                        "attention_bias_gate_residual_logit_mean_vision",
                        mild_vision.get("attention_bias_gate_residual_mean_vision", 0.0),
                    )
                ),
                "vision_residual_logit_mild_minus_clean": float(
                    mild_vision.get(
                        "attention_bias_gate_residual_logit_mean_vision",
                        mild_vision.get("attention_bias_gate_residual_mean_vision", 0.0),
                    )
                    - normal.get(
                        "attention_bias_gate_residual_logit_mean_vision",
                        normal.get("attention_bias_gate_residual_mean_vision", 0.0),
                    )
                ),
                "vision_final_gate_clean_mean": float(normal.get("attention_bias_gate_mean_vision", 0.0)),
                "vision_final_gate_mild_mean": float(mild_vision.get("attention_bias_gate_mean_vision", 0.0)),
                "vision_final_gate_mild_minus_clean": float(
                    mild_vision.get("attention_bias_gate_mean_vision", 0.0)
                    - normal.get("attention_bias_gate_mean_vision", 0.0)
                ),
                "source_sign_score_anchor_error": float(mild_vision.get("source_sign_score_anchor_error", 0.0)),
                "source_clean_gate_preserve_error": float(normal.get("source_clean_gate_preserve_error", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_7f_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_gate_mild_minus_clean"] > 0.03
        and summary["vision_gate_clean_below_cap_ratio"] >= 0.8
        and summary["vision_gate_mild_above_floor_ratio"] >= 0.8
        and summary["vision_gate_gap_margin_satisfied_ratio"] >= 0.7
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["normal_to_mild_return_drop"] <= baseline_summary["normal_to_mild_return_drop"] + 0.10
        and summary["clean_teacher_score_anchor_error"] <= 0.02
        and summary["clean_teacher_action_anchor_error"] <= 0.01
        and summary["source_sign_score_anchor_error"] <= 0.02
        and summary["source_clean_gate_preserve_error"] <= 0.01
    ):
        return "PASS"
    return "WARN"


def build_v3_7f_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7d_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    score_df = score_df.merge(
        gate_df[
            [
                "candidate_name",
                "vision_source_gate_clean_mean",
                "vision_source_gate_mild_mean",
                "vision_gate_residual_clean_mean",
                "vision_gate_residual_mild_mean",
                "vision_residual_logit_clean_mean",
                "vision_residual_logit_mild_mean",
                "vision_residual_logit_mild_minus_clean",
                "vision_final_gate_clean_mean",
                "vision_final_gate_mild_mean",
                "vision_final_gate_mild_minus_clean",
                "source_sign_score_anchor_error",
                "source_clean_gate_preserve_error",
            ]
        ],
        on="candidate_name",
        how="left",
    )
    baseline_row = score_df[score_df["candidate_name"] == baseline_candidate].iloc[0].to_dict()
    score_df["pass_fail"] = [
        evaluate_v3_7f_pass_fail(
            summary=row.to_dict(),
            baseline_summary=baseline_row,
            is_baseline=(row["candidate_name"] == baseline_candidate),
        )
        for _, row in score_df.iterrows()
    ]
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "normal_success_rate",
            "vision_attention_clean_minus_mild",
            "vision_attention_relative_score_clean_minus_mild",
            "vision_gate_mild_minus_clean",
            "source_sign_score_anchor_error",
            "source_clean_gate_preserve_error",
            "normal_to_mild_return_drop",
            "bias_head_activity_norm_mean",
            "consistency_gain_score",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    return generate_v3_7d_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )


def compute_v3_7f1_source_gate_distilled_residual_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    gate_df = compute_v3_7f_source_gate_residual_metrics(df, candidate_thresholds=candidate_thresholds)
    rows = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        rows.append(
            {
                **gate_row.to_dict(),
                "gate_reconstruction_error_mean": float(sub["gate_reconstruction_error_mean"].mean())
                if "gate_reconstruction_error_mean" in sub.columns
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_v3_7f1_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if abs(float(summary.get("vision_source_gate_clean_mean", 0.0))) <= 1e-6 or abs(
        float(summary.get("vision_source_gate_mild_mean", 0.0))
    ) <= 1e-6:
        return "FAIL"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_gate_mild_minus_clean"] > 0.03
        and summary["vision_gate_clean_below_cap_ratio"] >= 0.8
        and summary["vision_gate_mild_above_floor_ratio"] >= 0.8
        and summary["vision_gate_gap_margin_satisfied_ratio"] >= 0.7
        and float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1e-6
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
        and summary["clean_teacher_score_anchor_error"] <= 0.02
        and summary["clean_teacher_action_anchor_error"] <= 0.01
        and summary["source_sign_score_anchor_error"] <= 0.02
        and summary["source_clean_gate_preserve_error"] <= 0.01
    ):
        return "PASS"
    return "WARN"


def build_v3_7f1_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7d_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    score_df = score_df.merge(
        gate_df[
            [
                "candidate_name",
                "vision_source_gate_clean_mean",
                "vision_source_gate_mild_mean",
                "vision_gate_residual_clean_mean",
                "vision_gate_residual_mild_mean",
                "vision_residual_logit_clean_mean",
                "vision_residual_logit_mild_mean",
                "vision_residual_logit_mild_minus_clean",
                "vision_final_gate_clean_mean",
                "vision_final_gate_mild_mean",
                "vision_final_gate_mild_minus_clean",
                "gate_reconstruction_error_mean",
                "source_sign_score_anchor_error",
                "source_clean_gate_preserve_error",
            ]
        ],
        on="candidate_name",
        how="left",
    )
    baseline_row = score_df[score_df["candidate_name"] == baseline_candidate].iloc[0].to_dict()
    score_df["pass_fail"] = [
        evaluate_v3_7f1_pass_fail(
            summary=row.to_dict(),
            baseline_summary=baseline_row,
            is_baseline=(row["candidate_name"] == baseline_candidate),
        )
        for _, row in score_df.iterrows()
    ]
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_attention_targetlogit_sign_match_flag",
            "normal_success_rate",
            "vision_attention_clean_minus_mild",
            "vision_attention_relative_score_clean_minus_mild",
            "vision_gate_mild_minus_clean",
            "source_sign_score_anchor_error",
            "source_clean_gate_preserve_error",
            "gate_reconstruction_error_mean",
            "normal_to_mild_return_drop",
            "bias_head_activity_norm_mean",
            "consistency_gain_score",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, True, True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f1_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7d_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    collapse_rows = score_df[
        (score_df.get("vision_source_gate_mild_mean", 0.0) - score_df.get("vision_source_gate_clean_mean", 0.0)).abs() < 1.0e-6
    ]
    collapse_rows = collapse_rows[
        (collapse_rows.get("vision_residual_logit_mild_minus_clean", 0.0).abs() < 1.0e-3)
        & (collapse_rows.get("vision_final_gate_mild_minus_clean", 0.0).abs() < 1.0e-3)
    ]
    if not collapse_rows.empty:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n## Gate Residual Collapse Warning\n\n")
            for candidate_name in collapse_rows["candidate_name"].tolist():
                handle.write(f"- {candidate_name}: teacher copied but residual dead pattern detected.\n")


def compute_v3_7f2_source_gate_residual_logit_selective_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    return compute_v3_7f1_source_gate_distilled_residual_metrics(df, candidate_thresholds=candidate_thresholds)


def evaluate_v3_7f2_pass_fail(summary: dict[str, Any], baseline_summary: dict[str, Any], is_baseline: bool) -> str:
    if summary["saturation_warning_flag"] > 0.0 or summary["route_distance_side_effect_flag"] > 0.0:
        return "FAIL"
    if is_baseline:
        return "WARN"
    if abs(float(summary.get("vision_source_gate_clean_mean", 0.0))) <= 1e-6 or abs(float(summary.get("vision_source_gate_mild_mean", 0.0))) <= 1e-6:
        return "FAIL"
    if float(summary.get("gate_reconstruction_error_mean", 1.0)) >= 1e-6:
        return "FAIL"
    if (
        summary["vision_attention_targetlogit_sign_match_flag"] == 1.0
        and summary["vision_attention_relative_score_clean_minus_mild"] > 0.0
        and summary["vision_attention_clean_minus_mild"] > 0.0
        and summary["vision_route_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_distance_score_gap_clean_minus_mild"] > 0.0
        and summary["vision_gate_mild_minus_clean"] > 0.03
        and summary["vision_gate_clean_below_cap_ratio"] >= 0.8
        and summary["vision_gate_mild_above_floor_ratio"] >= 0.8
        and summary["vision_gate_gap_margin_satisfied_ratio"] >= 0.7
        and summary["normal_success_rate"] >= baseline_summary["normal_success_rate"] - 0.02
        and summary["mild_success_rate"] >= baseline_summary["mild_success_rate"] - 0.02
    ):
        return "PASS"
    return "WARN"


def build_v3_7f2_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f1_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    baseline_row = score_df[score_df["candidate_name"] == baseline_candidate].iloc[0].to_dict()
    score_df["pass_fail"] = [
        evaluate_v3_7f2_pass_fail(
            summary=row.to_dict(),
            baseline_summary=baseline_row,
            is_baseline=(row["candidate_name"] == baseline_candidate),
        )
        for _, row in score_df.iterrows()
    ]
    return score_df


def generate_v3_7f2_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f1_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )


def compute_v3_7f2_2_activation_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_source_gate_residual_logit_selective_metrics(df, candidate_thresholds=candidate_thresholds)
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        rows.append(
            {
                **gate_row.to_dict(),
                "clean_vs_mild_gate_feature_auc": float(mild_vision.get("clean_vs_mild_gate_feature_auc", 0.5)),
                "clean_vs_mild_gate_feature_delta_norm": float(
                    mild_vision.get("clean_vs_mild_gate_feature_delta_norm", 0.0)
                ),
                "source_gate_feature_onoff_auc_delta": float(
                    mild_vision.get("source_gate_feature_onoff_auc_delta", 0.0)
                ),
                "teacher_delta_feature_onoff_auc_delta": float(
                    mild_vision.get("teacher_delta_feature_onoff_auc_delta", 0.0)
                ),
                "residual_alpha": float(sub.get("residual_alpha_mean", pd.Series(dtype=float)).mean())
                if "residual_alpha_mean" in sub.columns
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_v3_7f2_3_vision_actuated_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_2_activation_metrics(df, candidate_thresholds=candidate_thresholds)
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        nonvision_clean_drift = []
        nonvision_mild_drift = []
        for name in ("distance", "ego", "route"):
            nonvision_clean_drift.append(
                abs(
                    float(normal.get(f"attention_bias_gate_mean_{name}", 0.0))
                    - float(normal.get(f"attention_source_bias_gate_mean_{name}", 0.0))
                )
            )
            nonvision_mild_drift.append(
                abs(
                    float(mild_vision.get(f"attention_bias_gate_mean_{name}", 0.0))
                    - float(mild_vision.get(f"attention_source_bias_gate_mean_{name}", 0.0))
                )
            )
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_logit_offset": float(sub.get("gate_logit_offset_mean", pd.Series(dtype=float)).mean())
                if "gate_logit_offset_mean" in sub.columns
                else 0.0,
                "nonvision_gate_drift_mean": float(np.mean(nonvision_clean_drift + nonvision_mild_drift))
                if (nonvision_clean_drift or nonvision_mild_drift)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_v3_7f2_4_gate_relative_micro_unfreeze_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
    relative_head_activity_cap: float = 0.35,
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_3_vision_actuated_metrics(df, candidate_thresholds=candidate_thresholds)
    relative_df = compute_v3_6_relative_head_metrics(df)
    score_means = (
        df.groupby(["candidate_name", "corruption_type"], as_index=False)
        .mean(numeric_only=True)
    )
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    clean_reference = normal_by_candidate.get(clean_reference_candidate)
    source_normal = normal_by_candidate.get(source_candidate)
    source_mild = mild_by_candidate.get(source_candidate)
    if clean_reference is None or source_normal is None or source_mild is None:
        raise ValueError("V3.7f2.4 metrics require reference_v3_6 and reference_v3_7c normal/mild rows")
    rows: list[dict[str, Any]] = []
    relative_mean_cols = [
        "attention_relative_gap_score_mean_vision_distance",
        "attention_relative_gap_score_mean_vision_ego",
        "attention_relative_gap_score_mean_vision_route",
    ]
    nonvision_relative_cols = [
        "attention_relative_score_before_bias_mean_distance",
        "attention_relative_score_before_bias_mean_ego",
        "attention_relative_score_before_bias_mean_route",
    ]
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        relative_row = relative_df[relative_df["candidate_name"] == candidate_name].iloc[0]
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild_vision = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))

        relative_head_delta_norm_terms = []
        clean_anchor_terms = []
        mild_anchor_terms = []
        nonvision_drift_terms = []
        for column in relative_mean_cols:
            relative_head_delta_norm_terms.append(abs(float(normal.get(column, 0.0)) - float(source_normal.get(column, 0.0))))
            relative_head_delta_norm_terms.append(abs(float(mild_vision.get(column, 0.0)) - float(source_mild.get(column, 0.0))))
            clean_anchor_terms.append(abs(float(normal.get(column, 0.0)) - float(clean_reference.get(column, 0.0))))
            mild_anchor_terms.append(abs(float(mild_vision.get(column, 0.0)) - float(source_mild.get(column, 0.0))))
        for column in nonvision_relative_cols:
            nonvision_drift_terms.append(abs(float(normal.get(column, 0.0)) - float(source_normal.get(column, 0.0))))
            nonvision_drift_terms.append(abs(float(mild_vision.get(column, 0.0)) - float(source_mild.get(column, 0.0))))

        relative_head_delta_norm = float(np.mean(relative_head_delta_norm_terms)) if relative_head_delta_norm_terms else 0.0
        clean_anchor_error = float(np.mean(clean_anchor_terms)) if clean_anchor_terms else 0.0
        mild_anchor_error = float(np.mean(mild_anchor_terms)) if mild_anchor_terms else 0.0
        nonvision_relative_gap_drift_mean = float(np.mean(nonvision_drift_terms)) if nonvision_drift_terms else 0.0
        gate_relative_synergy_gain = float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) + (
            float(relative_row.get("relative_head_gap_clean_minus_mild", 0.0))
            - float(relative_df[relative_df["candidate_name"] == source_candidate]["relative_head_gap_clean_minus_mild"].iloc[0])
        )
        relative_head_cheat_warning_flag = float(
            float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.005
            and (
                relative_head_delta_norm > 0.01
                or float(relative_row.get("relative_head_activity_norm_mean", 0.0)) > float(relative_head_activity_cap)
            )
        )
        rows.append(
            {
                **gate_row.to_dict(),
                "relative_head_activity_norm_mean": float(relative_row.get("relative_head_activity_norm_mean", 0.0)),
                "relative_head_gap_clean_minus_mild": float(relative_row.get("relative_head_gap_clean_minus_mild", 0.0)),
                "relative_head_delta_norm": relative_head_delta_norm,
                "vision_relative_gap_anchor_error_clean": clean_anchor_error,
                "vision_relative_gap_anchor_error_mild": mild_anchor_error,
                "nonvision_relative_gap_drift_mean": nonvision_relative_gap_drift_mean,
                "gate_relative_synergy_gain": gate_relative_synergy_gain,
                "relative_head_cheat_warning_flag": relative_head_cheat_warning_flag,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_2_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    baseline_row = score_df[score_df["candidate_name"] == baseline_candidate].iloc[0].to_dict()
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        rows.append(
            {
                "candidate_name": summary["candidate_name"],
                "distance_to_activation_residual_logit": float(summary.get("vision_residual_logit_mild_minus_clean", 0.0) - 0.01),
                "distance_to_activation_final_gate": float(summary.get("vision_final_gate_mild_minus_clean", 0.0) - 0.01),
                "distance_to_promotion_final_gate": float(summary.get("vision_final_gate_mild_minus_clean", 0.0) - 0.02),
                "distance_to_promotion_mild_floor_ratio": float(summary.get("vision_gate_mild_above_floor_ratio", 0.0) - 0.2),
                "distance_to_promotion_gap_ratio": float(
                    summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0) - 0.1
                ),
                "delta_vs_reference_v3_7c_final_gate_gap": float(
                    summary.get("vision_final_gate_mild_minus_clean", 0.0)
                    - source_row.get("vision_final_gate_mild_minus_clean", 0.0)
                ),
                "delta_vs_reference_v3_7c_relative_gap": float(
                    summary.get("vision_attention_relative_score_clean_minus_mild", 0.0)
                    - source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)
                ),
                "delta_vs_reference_v3_6_normal_success_rate": float(
                    summary.get("normal_success_rate", 0.0) - baseline_row.get("normal_success_rate", 0.0)
                ),
                "delta_vs_reference_v3_6_mild_success_rate": float(
                    summary.get("mild_success_rate", 0.0) - baseline_row.get("mild_success_rate", 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_4_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_2_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_relative_gap_anchor_clean": 0.02 - float(score_row.get("vision_relative_gap_anchor_error_clean", 0.0)),
                "distance_to_relative_gap_anchor_mild": 0.02 - float(score_row.get("vision_relative_gap_anchor_error_mild", 0.0)),
                "distance_to_nonvision_relative_drift_cap": 0.02 - float(score_row.get("nonvision_relative_gap_drift_mean", 0.0)),
                "distance_to_synergy_gain_positive": float(score_row.get("gate_relative_synergy_gain", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_2_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    activation_cols = [
        "candidate_name",
        "vision_source_gate_clean_mean",
        "vision_source_gate_mild_mean",
        "vision_residual_logit_clean_mean",
        "vision_residual_logit_mild_mean",
        "vision_residual_logit_mild_minus_clean",
        "vision_final_gate_clean_mean",
        "vision_final_gate_mild_mean",
        "vision_final_gate_mild_minus_clean",
        "clean_vs_mild_gate_feature_auc",
        "clean_vs_mild_gate_feature_delta_norm",
        "source_gate_feature_onoff_auc_delta",
        "teacher_delta_feature_onoff_auc_delta",
        "residual_alpha",
    ]
    missing_activation_cols = [
        col for col in activation_cols
        if col != "candidate_name" and col in gate_df.columns and col not in score_df.columns
    ]
    if missing_activation_cols:
        score_df = score_df.merge(
            gate_df[["candidate_name", *missing_activation_cols]],
            on="candidate_name",
            how="left",
        )
    baseline_row = score_df[score_df["candidate_name"] == baseline_candidate].iloc[0].to_dict()
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    alpha_init = float(score_df.get("residual_alpha", pd.Series([0.0])).median()) if "residual_alpha" in score_df.columns else 0.0
    alpha_cheat_flags = []
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        alpha_cheat = (
            summary.get("residual_alpha", 0.0) > max(alpha_init * 1.5, 2.5)
            and abs(summary.get("vision_residual_logit_mild_minus_clean", 0.0)) < 0.005
            and abs(summary.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.005
        )
        activation = (
            abs(float(summary.get("vision_source_gate_clean_mean", 0.0))) > 1.0e-6
            and abs(float(summary.get("vision_source_gate_mild_mean", 0.0))) > 1.0e-6
            and float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_residual_logit_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("clean_vs_mild_gate_feature_auc", 0.5)) > 0.65
            and not alpha_cheat
        )
        promotion = (
            activation
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.02
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.2
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.1
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", 0.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0))
            and float(summary.get("vision_attention_clean_minus_mild", 0.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0))
        )
        readiness = (
            promotion
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0))
            > float(source_row.get("vision_final_gate_mild_minus_clean", 0.0))
            and float(summary.get("normal_success_rate", 0.0)) >= float(baseline_row.get("normal_success_rate", 0.0)) - 0.05
        )
        alpha_cheat_flags.append(float(alpha_cheat))
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["alpha_cheat_warning_flag"] = alpha_cheat_flags
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    score_df["pass_fail"] = np.where(
        score_df["promotion_success_flag"] > 0.5,
        "PASS",
        np.where(score_df["activation_debug_success_flag"] > 0.5, "WARN", score_df["pass_fail"]),
    )
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_final_gate_mild_minus_clean",
            "vision_residual_logit_mild_minus_clean",
            "clean_vs_mild_gate_feature_auc",
            "alpha_cheat_warning_flag",
            "vision_attention_relative_score_clean_minus_mild",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_2_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Activation Debug Flags\n\n")
        debug_cols = [
            "candidate_name",
            "activation_debug_success_flag",
            "promotion_success_flag",
            "pass_closing_ready_flag",
            "alpha_cheat_warning_flag",
            "vision_residual_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "clean_vs_mild_gate_feature_auc",
            "residual_alpha",
        ]
        handle.write(score_df[debug_cols].to_markdown(index=False))
        handle.write("\n")


def build_v3_7f2_3_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_2_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [c for c in ["candidate_name", "vision_logit_offset", "nonvision_gate_drift_mean"] if c in gate_df.columns]
    if len(extra_cols) > 1:
        score_df = score_df.merge(gate_df[extra_cols], on="candidate_name", how="left")
    return score_df


def generate_v3_7f2_3_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_2_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Vision Actuation Summary\n\n")
        cols = [
            "candidate_name",
            "vision_logit_offset",
            "nonvision_gate_drift_mean",
            "vision_residual_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "clean_vs_mild_gate_feature_auc",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def build_v3_7f2_4_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_3_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "relative_head_activity_norm_mean",
            "relative_head_gap_clean_minus_mild",
            "relative_head_delta_norm",
            "vision_relative_gap_anchor_error_clean",
            "vision_relative_gap_anchor_error_mild",
            "nonvision_relative_gap_drift_mean",
            "gate_relative_synergy_gain",
            "relative_head_cheat_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.merge(gate_df[extra_cols], on="candidate_name", how="left")

    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_residual_logit_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("clean_vs_mild_gate_feature_auc", 0.5)) > 0.65
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("nonvision_gate_drift_mean", 1.0)) <= 0.02
            and float(summary.get("nonvision_relative_gap_drift_mean", 1.0)) <= 0.02
            and float(summary.get("alpha_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("relative_head_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.02
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.2
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.1
            and float(summary.get("gate_relative_synergy_gain", -1.0)) > 0.0
        )
        readiness = (
            promotion
            and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        )
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    score_df["pass_fail"] = np.where(
        score_df["promotion_success_flag"] > 0.5,
        "PASS",
        np.where(score_df["activation_debug_success_flag"] > 0.5, "WARN", score_df["pass_fail"]),
    )
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_final_gate_mild_minus_clean",
            "vision_residual_logit_mild_minus_clean",
            "gate_relative_synergy_gain",
            "clean_vs_mild_gate_feature_auc",
            "alpha_cheat_warning_flag",
            "relative_head_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_4_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_3_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Relative Head Micro-Unfreeze Summary\n\n")
        cols = [
            "candidate_name",
            "relative_head_delta_norm",
            "relative_head_activity_norm_mean",
            "vision_relative_gap_anchor_error_clean",
            "vision_relative_gap_anchor_error_mild",
            "nonvision_relative_gap_drift_mean",
            "gate_relative_synergy_gain",
            "relative_head_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_5_gate_score_adapter_micro_unfreeze_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
    score_adapter_activity_cap: float = 0.35,
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_3_vision_actuated_metrics(df, candidate_thresholds=candidate_thresholds)
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    clean_reference = normal_by_candidate.get(clean_reference_candidate)
    source_normal = normal_by_candidate.get(source_candidate)
    source_mild = mild_by_candidate.get(source_candidate)
    if clean_reference is None or source_normal is None or source_mild is None:
        raise ValueError("V3.7f2.5 metrics require reference_v3_6 and reference_v3_7c normal/mild rows")

    rows: list[dict[str, Any]] = []
    adapter_cols = [f"attention_score_adapter_delta_norm_mean_{name}" for name in MODALITY_NAMES]
    nonvision_cols = [f"attention_score_adapter_delta_norm_mean_{name}" for name in ("distance", "ego", "route")]
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        sub = df[df["candidate_name"] == candidate_name]
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild_vision = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        clean_norm = float(normal.get("attention_score_adapter_delta_norm_mean_vision", 0.0))
        mild_norm = float(mild_vision.get("attention_score_adapter_delta_norm_mean_vision", 0.0))
        adapter_mean_terms = [float(sub.get(column, pd.Series(dtype=float)).mean()) for column in adapter_cols if column in sub.columns]
        nonvision_drift_terms = []
        for column in nonvision_cols:
            nonvision_drift_terms.append(abs(float(normal.get(column, 0.0)) - float(source_normal.get(column, 0.0))))
            nonvision_drift_terms.append(abs(float(mild_vision.get(column, 0.0)) - float(source_mild.get(column, 0.0))))
        clean_anchor_error = abs(clean_norm - float(clean_reference.get("attention_score_adapter_delta_norm_mean_vision", 0.0)))
        mild_anchor_error = abs(mild_norm - float(source_mild.get("attention_score_adapter_delta_norm_mean_vision", 0.0)))
        score_adapter_delta_mild_minus_clean = mild_norm - clean_norm
        adapter_gate_synergy_gain = float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) + score_adapter_delta_mild_minus_clean
        score_adapter_cheat_warning_flag = float(
            float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.005
            and (
                abs(score_adapter_delta_mild_minus_clean) > 0.01
                or float(np.mean(adapter_mean_terms)) > float(score_adapter_activity_cap)
            )
        )
        rows.append(
            {
                **gate_row.to_dict(),
                "score_adapter_delta_norm_mean": float(np.mean(adapter_mean_terms)) if adapter_mean_terms else 0.0,
                "score_adapter_delta_norm_mean_vision_probe": score_adapter_delta_mild_minus_clean,
                "vision_adapter_delta_margin": score_adapter_delta_mild_minus_clean,
                "score_adapter_delta_norm_clean": clean_norm,
                "score_adapter_delta_norm_mild": mild_norm,
                "score_adapter_delta_mild_minus_clean": score_adapter_delta_mild_minus_clean,
                "score_adapter_delta_norm_nonvision_drift_mean": float(np.mean(nonvision_drift_terms)) if nonvision_drift_terms else 0.0,
                "score_adapter_delta_anchor_error_clean": clean_anchor_error,
                "score_adapter_delta_anchor_error_mild": mild_anchor_error,
                "adapter_gate_synergy_gain": adapter_gate_synergy_gain,
                "score_adapter_cheat_warning_flag": score_adapter_cheat_warning_flag,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_5_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_2_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_adapter_delta_margin_positive": float(score_row.get("vision_adapter_delta_margin", 0.0)),
                "distance_to_adapter_anchor_clean": 0.02 - float(score_row.get("score_adapter_delta_anchor_error_clean", 0.0)),
                "distance_to_adapter_anchor_mild": 0.02 - float(score_row.get("score_adapter_delta_anchor_error_mild", 0.0)),
                "distance_to_nonvision_adapter_drift_cap": 0.02 - float(score_row.get("score_adapter_delta_norm_nonvision_drift_mean", 0.0)),
                "delta_vs_reference_v3_7c_adapter_activity": float(
                    score_row.get("score_adapter_delta_mild_minus_clean", 0.0)
                    - source_row.get("score_adapter_delta_mild_minus_clean", 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_5_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_3_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "score_adapter_delta_norm_mean",
            "score_adapter_delta_norm_mean_vision_probe",
            "vision_adapter_delta_margin",
            "score_adapter_delta_norm_clean",
            "score_adapter_delta_norm_mild",
            "score_adapter_delta_mild_minus_clean",
            "score_adapter_delta_norm_nonvision_drift_mean",
            "score_adapter_delta_anchor_error_clean",
            "score_adapter_delta_anchor_error_mild",
            "adapter_gate_synergy_gain",
            "score_adapter_cheat_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.merge(gate_df[extra_cols], on="candidate_name", how="left")

    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_residual_logit_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("clean_vs_mild_gate_feature_auc", 0.5)) > 0.65
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("nonvision_gate_drift_mean", 1.0)) <= 0.02
            and float(summary.get("score_adapter_delta_norm_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("alpha_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("score_adapter_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.02
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.2
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.1
            and float(summary.get("adapter_gate_synergy_gain", -1.0)) > 0.0
        )
        readiness = (
            promotion
            and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        )
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    score_df["pass_fail"] = np.where(
        score_df["promotion_success_flag"] > 0.5,
        "PASS",
        np.where(score_df["activation_debug_success_flag"] > 0.5, "WARN", score_df["pass_fail"]),
    )
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_final_gate_mild_minus_clean",
            "vision_residual_logit_mild_minus_clean",
            "adapter_gate_synergy_gain",
            "clean_vs_mild_gate_feature_auc",
            "alpha_cheat_warning_flag",
            "score_adapter_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_5_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_3_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Score Adapter Micro-Unfreeze Summary\n\n")
        cols = [
            "candidate_name",
            "score_adapter_delta_norm_mean",
            "score_adapter_delta_norm_clean",
            "score_adapter_delta_norm_mild",
            "score_adapter_delta_mild_minus_clean",
            "score_adapter_delta_anchor_error_clean",
            "score_adapter_delta_anchor_error_mild",
            "score_adapter_delta_norm_nonvision_drift_mean",
            "adapter_gate_synergy_gain",
            "score_adapter_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_6_adapter_gate_coupling_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_5_gate_score_adapter_micro_unfreeze_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    source_normal = normal_by_candidate.get(source_candidate)
    if source_normal is None:
        raise ValueError("V3.7f2.6 metrics require source candidate normal rows")
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        vision_core_clean = float(normal.get("attention_bias_gate_residual_logit_mean_vision", 0.0))
        vision_core_mild = float(mild.get("attention_bias_gate_residual_logit_mean_vision", 0.0))
        vision_adapter_clean = float(normal.get("attention_bias_gate_adapter_logit_mean_vision", 0.0))
        vision_adapter_mild = float(mild.get("attention_bias_gate_adapter_logit_mean_vision", 0.0))
        vision_adapter_gap = vision_adapter_mild - vision_adapter_clean
        nonvision_adapter_drift_terms = []
        for name in ("distance", "ego", "route"):
            nonvision_adapter_drift_terms.append(abs(float(normal.get(f"attention_bias_gate_adapter_logit_mean_{name}", 0.0))))
            nonvision_adapter_drift_terms.append(abs(float(mild.get(f"attention_bias_gate_adapter_logit_mean_{name}", 0.0))))
        final_gap = float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0))
        transfer_efficiency = final_gap / max(1.0e-6, abs(vision_adapter_gap))
        source_adapter_gap = float(
            mild_by_candidate.get(source_candidate, pd.Series(dtype=float)).get("attention_bias_gate_adapter_logit_mean_vision", 0.0)
        ) - float(source_normal.get("attention_bias_gate_adapter_logit_mean_vision", 0.0))
        adapter_path_usage_ratio = abs(float(mild.get("adapter_gate_alpha_mean", 0.0)) * vision_adapter_gap) / max(
            1.0e-6,
            abs(float(mild.get("residual_alpha_mean", 0.0)) * (vision_core_mild - vision_core_clean)),
        )
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_core_residual_logit_clean_mean": vision_core_clean,
                "vision_core_residual_logit_mild_mean": vision_core_mild,
                "vision_adapter_gate_logit_clean_mean": vision_adapter_clean,
                "vision_adapter_gate_logit_mild_mean": vision_adapter_mild,
                "vision_adapter_gate_logit_mild_minus_clean": vision_adapter_gap,
                "adapter_to_gate_transfer_efficiency": transfer_efficiency,
                "adapter_path_usage_ratio": adapter_path_usage_ratio,
                "adapter_gate_nonvision_drift_mean": float(np.mean(nonvision_adapter_drift_terms)) if nonvision_adapter_drift_terms else 0.0,
                "adapter_gate_cheat_warning_flag": float(abs(vision_adapter_gap) > 0.03 and final_gap < 0.002),
                "delta_vs_reference_v3_7c_adapter_gate_activity": vision_adapter_gap - source_adapter_gap,
                "adapter_gate_alpha_mean": float(mild.get("adapter_gate_alpha_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_6_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_5_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_activation_adapter_gate_logit": float(score_row.get("vision_adapter_gate_logit_mild_minus_clean", 0.0)) - 0.20,
                "distance_to_transfer_efficiency_min": float(score_row.get("adapter_to_gate_transfer_efficiency", 0.0)) - 0.03,
                "distance_to_nonvision_adapter_gate_drift_cap": 0.02 - float(score_row.get("adapter_gate_nonvision_drift_mean", 0.0)),
                "delta_vs_reference_v3_7c_adapter_gate_activity": float(
                    score_row.get("delta_vs_reference_v3_7c_adapter_gate_activity", 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_6_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_5_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "vision_core_residual_logit_clean_mean",
            "vision_core_residual_logit_mild_mean",
            "vision_adapter_gate_logit_clean_mean",
            "vision_adapter_gate_logit_mild_mean",
            "vision_adapter_gate_logit_mild_minus_clean",
            "adapter_to_gate_transfer_efficiency",
            "adapter_path_usage_ratio",
            "adapter_gate_cheat_warning_flag",
            "adapter_gate_nonvision_drift_mean",
            "delta_vs_reference_v3_7c_adapter_gate_activity",
            "adapter_gate_alpha_mean",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.merge(gate_df[extra_cols], on="candidate_name", how="left")
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("clean_vs_mild_gate_feature_auc", 0.5)) > 0.65
            and float(summary.get("vision_adapter_gate_logit_mild_minus_clean", 0.0)) > 0.20
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("adapter_to_gate_transfer_efficiency", 0.0)) > 0.03
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("nonvision_gate_drift_mean", 1.0)) <= 0.02
            and float(summary.get("adapter_gate_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("alpha_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("adapter_gate_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.02
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.2
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.1
            and float(summary.get("adapter_to_gate_transfer_efficiency", 0.0)) > 0.05
        )
        readiness = (
            promotion
            and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        )
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_final_gate_mild_minus_clean",
            "vision_adapter_gate_logit_mild_minus_clean",
            "adapter_to_gate_transfer_efficiency",
            "adapter_gate_cheat_warning_flag",
            "alpha_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_6_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_5_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Adapter-to-Gate Coupling Summary\n\n")
        cols = [
            "candidate_name",
            "vision_adapter_gate_logit_clean_mean",
            "vision_adapter_gate_logit_mild_mean",
            "vision_adapter_gate_logit_mild_minus_clean",
            "adapter_to_gate_transfer_efficiency",
            "adapter_path_usage_ratio",
            "adapter_gate_nonvision_drift_mean",
            "adapter_gate_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_7_adapter_readout_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_6_adapter_gate_coupling_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        vision_adapter_delta_norm_clean = float(normal.get("attention_score_adapter_delta_norm_mean_vision", 0.0))
        vision_adapter_delta_norm_mild = float(mild.get("attention_score_adapter_delta_norm_mean_vision", 0.0))
        vision_adapter_hidden_norm_clean = float(normal.get("attention_bias_gate_adapter_hidden_norm_mean_vision", 0.0))
        vision_adapter_hidden_norm_mild = float(mild.get("attention_bias_gate_adapter_hidden_norm_mean_vision", 0.0))
        vision_adapter_gate_logit_gap = float(gate_row.get("vision_adapter_gate_logit_mild_minus_clean", 0.0))
        adapter_delta_gap = vision_adapter_delta_norm_mild - vision_adapter_delta_norm_clean
        nonvision_readout_drift_terms = []
        for name in ("distance", "ego", "route"):
            nonvision_readout_drift_terms.append(abs(float(normal.get(f"attention_bias_gate_adapter_logit_mean_{name}", 0.0))))
            nonvision_readout_drift_terms.append(abs(float(mild.get(f"attention_bias_gate_adapter_logit_mean_{name}", 0.0))))
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_adapter_delta_norm_clean_mean": vision_adapter_delta_norm_clean,
                "vision_adapter_delta_norm_mild_mean": vision_adapter_delta_norm_mild,
                "vision_adapter_delta_norm_mild_minus_clean": adapter_delta_gap,
                "vision_adapter_gate_hidden_norm_clean_mean": vision_adapter_hidden_norm_clean,
                "vision_adapter_gate_hidden_norm_mild_mean": vision_adapter_hidden_norm_mild,
                "vision_adapter_gate_hidden_norm_mild_minus_clean": vision_adapter_hidden_norm_mild - vision_adapter_hidden_norm_clean,
                "vision_adapter_gate_readout_compression_ratio": vision_adapter_gate_logit_gap / max(1.0e-6, float(gate_row.get("score_adapter_delta_mild_minus_clean", 0.0))),
                "adapter_readout_nonvision_drift_mean": float(np.mean(nonvision_readout_drift_terms)) if nonvision_readout_drift_terms else 0.0,
                "adapter_readout_cheat_warning_flag": float(
                    float(gate_row.get("score_adapter_delta_mild_minus_clean", 0.0)) > 0.05 and vision_adapter_gate_logit_gap < 0.01
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_7_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_6_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_adapter_readout_logit": float(score_row.get("vision_adapter_gate_logit_mild_minus_clean", 0.0)) - 0.03,
                "distance_to_adapter_readout_ratio": float(score_row.get("vision_adapter_gate_readout_compression_ratio", 0.0)) - 0.30,
                "distance_to_adapter_hidden_norm_floor": float(score_row.get("vision_adapter_gate_hidden_norm_mild_mean", 0.0)) - 0.5,
                "distance_to_nonvision_adapter_readout_drift_cap": 0.02 - float(score_row.get("adapter_readout_nonvision_drift_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_7_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_6_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "vision_adapter_delta_norm_clean_mean",
            "vision_adapter_delta_norm_mild_mean",
            "vision_adapter_delta_norm_mild_minus_clean",
            "vision_adapter_gate_hidden_norm_clean_mean",
            "vision_adapter_gate_hidden_norm_mild_mean",
            "vision_adapter_gate_hidden_norm_mild_minus_clean",
            "vision_adapter_gate_readout_compression_ratio",
            "adapter_readout_nonvision_drift_mean",
            "adapter_readout_cheat_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.merge(gate_df[extra_cols], on="candidate_name", how="left")
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("clean_vs_mild_gate_feature_auc", 0.5)) > 0.65
            and float(summary.get("score_adapter_delta_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_adapter_gate_logit_mild_minus_clean", 0.0)) > 0.03
            and float(summary.get("vision_adapter_gate_readout_compression_ratio", 0.0)) > 0.30
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.002
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("nonvision_gate_drift_mean", 1.0)) <= 0.02
            and float(summary.get("adapter_readout_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("alpha_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("adapter_readout_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_adapter_gate_logit_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.005
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
            and float(summary.get("adapter_to_gate_transfer_efficiency", 0.0)) > 0.05
        )
        readiness = (
            promotion
            and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        )
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_adapter_gate_logit_mild_minus_clean",
            "vision_adapter_gate_readout_compression_ratio",
            "vision_final_gate_mild_minus_clean",
            "adapter_readout_cheat_warning_flag",
            "alpha_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_7_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_6_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Adapter Readout Summary\n\n")
        cols = [
            "candidate_name",
            "score_adapter_delta_mild_minus_clean",
            "vision_adapter_gate_logit_mild_minus_clean",
            "vision_adapter_gate_readout_compression_ratio",
            "vision_adapter_gate_hidden_norm_mild_mean",
            "vision_final_gate_mild_minus_clean",
            "adapter_readout_nonvision_drift_mean",
            "adapter_readout_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_8_output_scale_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_7_adapter_readout_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        vision_delta_clean = float(normal.get("attention_bias_gate_delta_logit_mean_vision", 0.0))
        vision_delta_mild = float(mild.get("attention_bias_gate_delta_logit_mean_vision", 0.0))
        vision_scaled_delta_clean = float(normal.get("attention_bias_gate_scaled_delta_logit_mean_vision", 0.0))
        vision_scaled_delta_mild = float(mild.get("attention_bias_gate_scaled_delta_logit_mean_vision", 0.0))
        nonvision_scaled_terms = []
        for name in ("distance", "ego", "route"):
            nonvision_scaled_terms.append(abs(float(normal.get(f"attention_bias_gate_scaled_delta_logit_mean_{name}", 0.0))))
            nonvision_scaled_terms.append(abs(float(mild.get(f"attention_bias_gate_scaled_delta_logit_mean_{name}", 0.0))))
        vision_scaled_gap = vision_scaled_delta_mild - vision_scaled_delta_clean
        vision_gate_output_scale = float(mild.get("attention_gate_output_scale_mean_vision", normal.get("attention_gate_output_scale_mean_vision", 1.0)))
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_delta_logit_clean_mean": vision_delta_clean,
                "vision_delta_logit_mild_mean": vision_delta_mild,
                "vision_delta_logit_mild_minus_clean": vision_delta_mild - vision_delta_clean,
                "vision_scaled_delta_logit_clean_mean": vision_scaled_delta_clean,
                "vision_scaled_delta_logit_mild_mean": vision_scaled_delta_mild,
                "vision_scaled_delta_logit_mild_minus_clean": vision_scaled_gap,
                "vision_gate_output_scale": vision_gate_output_scale,
                "output_scale_transfer_efficiency": float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0))
                / max(1.0e-6, abs(vision_scaled_gap)),
                "output_scaled_nonvision_drift_mean": float(np.mean(nonvision_scaled_terms)) if nonvision_scaled_terms else 0.0,
                "output_scale_cheat_warning_flag": float(
                    vision_gate_output_scale > 8.0 and float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.002
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_8_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_7_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_scaled_delta_logit": float(score_row.get("vision_scaled_delta_logit_mild_minus_clean", 0.0)) - 0.08,
                "distance_to_final_gate_gap": float(score_row.get("vision_final_gate_mild_minus_clean", 0.0)) - 0.002,
                "distance_to_output_transfer_efficiency": float(score_row.get("output_scale_transfer_efficiency", 0.0)) - 0.03,
                "distance_to_output_scaled_nonvision_drift_cap": 0.02 - float(score_row.get("output_scaled_nonvision_drift_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_8_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_7_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "vision_delta_logit_clean_mean",
            "vision_delta_logit_mild_mean",
            "vision_delta_logit_mild_minus_clean",
            "vision_scaled_delta_logit_clean_mean",
            "vision_scaled_delta_logit_mild_mean",
            "vision_scaled_delta_logit_mild_minus_clean",
            "vision_gate_output_scale",
            "output_scale_transfer_efficiency",
            "output_scaled_nonvision_drift_mean",
            "output_scale_cheat_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.drop(
            columns=[col for col in extra_cols if col != "candidate_name" and col in score_df.columns],
            errors="ignore",
        ).merge(gate_df[extra_cols], on="candidate_name", how="left")
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("clean_vs_mild_gate_feature_auc", 0.5)) > 0.65
            and float(summary.get("score_adapter_delta_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_adapter_gate_logit_mild_minus_clean", 0.0)) > 0.03
            and float(summary.get("vision_scaled_delta_logit_mild_minus_clean", 0.0)) > 0.08
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.002
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("output_scaled_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("alpha_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("adapter_readout_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("output_scale_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_scaled_delta_logit_mild_minus_clean", 0.0)) > 0.12
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.005
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
            and float(summary.get("output_scale_transfer_efficiency", 0.0)) > 0.05
        )
        readiness = (
            promotion
            and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        )
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_scaled_delta_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "output_scale_transfer_efficiency",
            "output_scale_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_8_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_7_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Output Scale Transfer Summary\n\n")
        cols = [
            "candidate_name",
            "vision_delta_logit_mild_minus_clean",
            "vision_scaled_delta_logit_mild_minus_clean",
            "vision_gate_output_scale",
            "vision_final_gate_mild_minus_clean",
            "output_scale_transfer_efficiency",
            "output_scale_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_9_threshold_shift_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_8_output_scale_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        threshold_clean = float(normal.get("attention_bias_gate_threshold_shift_logit_mean_vision", 0.0))
        threshold_mild = float(mild.get("attention_bias_gate_threshold_shift_logit_mean_vision", 0.0))
        post_clean = float(normal.get("attention_bias_gate_post_shift_logit_mean_vision", 0.0))
        post_mild = float(mild.get("attention_bias_gate_post_shift_logit_mean_vision", 0.0))
        source_bias = float(mild.get("attention_gate_source_logit_bias_mean_vision", normal.get("attention_gate_source_logit_bias_mean_vision", 0.0)))
        nonvision_terms = []
        for name in ("distance", "ego", "route"):
            nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_threshold_shift_logit_mean_{name}", 0.0))))
            nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_threshold_shift_logit_mean_{name}", 0.0))))
        threshold_gap = threshold_mild - threshold_clean
        pre_shift_gap = float(gate_row.get("vision_scaled_delta_logit_mild_minus_clean", 0.0))
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_pre_shift_delta_logit_clean_mean": float(normal.get("attention_bias_gate_scaled_delta_logit_mean_vision", 0.0)),
                "vision_pre_shift_delta_logit_mild_mean": float(mild.get("attention_bias_gate_scaled_delta_logit_mean_vision", 0.0)),
                "vision_pre_shift_delta_logit_mild_minus_clean": pre_shift_gap,
                "vision_threshold_shift_logit_clean_mean": threshold_clean,
                "vision_threshold_shift_logit_mild_mean": threshold_mild,
                "vision_threshold_shift_logit_mild_minus_clean": threshold_gap,
                "vision_post_shift_logit_clean_mean": post_clean,
                "vision_post_shift_logit_mild_mean": post_mild,
                "vision_post_shift_logit_mild_minus_clean": post_mild - post_clean,
                "vision_source_logit_bias": source_bias,
                "threshold_shift_usage_ratio": threshold_gap / max(1.0e-6, abs(pre_shift_gap)),
                "threshold_shift_sign_agreement_flag": float(threshold_mild > threshold_clean),
                "threshold_shift_nonvision_drift_mean": float(np.mean(nonvision_terms)) if nonvision_terms else 0.0,
                "threshold_shift_cheat_warning_flag": float(abs(threshold_gap) > 0.15 and float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.002),
                "source_logit_bias_cheat_warning_flag": float(abs(source_bias) > 0.5 and float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.002),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_9_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_8_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_threshold_shift_logit": float(score_row.get("vision_threshold_shift_logit_mild_minus_clean", 0.0)) - 0.08,
                "distance_to_post_shift_logit": float(score_row.get("vision_post_shift_logit_mild_minus_clean", 0.0)) - 0.03,
                "distance_to_threshold_shift_nonvision_drift_cap": 0.02 - float(score_row.get("threshold_shift_nonvision_drift_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_9_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_8_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "vision_pre_shift_delta_logit_clean_mean",
            "vision_pre_shift_delta_logit_mild_mean",
            "vision_pre_shift_delta_logit_mild_minus_clean",
            "vision_threshold_shift_logit_clean_mean",
            "vision_threshold_shift_logit_mild_mean",
            "vision_threshold_shift_logit_mild_minus_clean",
            "vision_post_shift_logit_clean_mean",
            "vision_post_shift_logit_mild_mean",
            "vision_post_shift_logit_mild_minus_clean",
            "vision_source_logit_bias",
            "threshold_shift_usage_ratio",
            "threshold_shift_sign_agreement_flag",
            "threshold_shift_nonvision_drift_mean",
            "threshold_shift_cheat_warning_flag",
            "source_logit_bias_cheat_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.drop(
            columns=[col for col in extra_cols if col != "candidate_name" and col in score_df.columns],
            errors="ignore",
        ).merge(gate_df[extra_cols], on="candidate_name", how="left")
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("score_adapter_delta_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_threshold_shift_logit_mild_minus_clean", 0.0)) > 0.08
            and float(summary.get("vision_post_shift_logit_mild_minus_clean", 0.0)) > 0.03
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.002
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("threshold_shift_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("threshold_shift_sign_agreement_flag", 0.0)) == 1.0
            and float(summary.get("alpha_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("adapter_readout_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("threshold_shift_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("source_logit_bias_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_threshold_shift_logit_mild_minus_clean", 0.0)) > 0.12
            and float(summary.get("vision_post_shift_logit_mild_minus_clean", 0.0)) > 0.06
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.005
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
        )
        readiness = promotion and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_threshold_shift_logit_mild_minus_clean",
            "vision_post_shift_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "threshold_shift_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_9_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_8_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Threshold Shift Summary\n\n")
        cols = [
            "candidate_name",
            "vision_threshold_shift_logit_mild_minus_clean",
            "vision_post_shift_logit_mild_minus_clean",
            "vision_source_logit_bias",
            "vision_final_gate_mild_minus_clean",
            "threshold_shift_sign_agreement_flag",
            "threshold_shift_cheat_warning_flag",
            "source_logit_bias_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_10_cancel_contrast_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_9_threshold_shift_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        clean_gate_max = float(candidate_thresholds.get(candidate_name, {}).get("clean_gate_max", 0.02))
        common_cancel_clean = float(normal.get("attention_bias_gate_common_cancel_logit_mean_vision", 0.0))
        common_cancel_mild = float(mild.get("attention_bias_gate_common_cancel_logit_mean_vision", 0.0))
        post_cancel_clean = float(normal.get("attention_bias_gate_post_cancel_logit_mean_vision", 0.0))
        post_cancel_mild = float(mild.get("attention_bias_gate_post_cancel_logit_mean_vision", 0.0))
        contrast_clean = float(normal.get("attention_bias_gate_contrast_shift_logit_mean_vision", 0.0))
        contrast_mild = float(mild.get("attention_bias_gate_contrast_shift_logit_mean_vision", 0.0))
        common_nonvision_terms = []
        contrast_nonvision_terms = []
        for name in ("distance", "ego", "route"):
            common_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            common_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            contrast_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_contrast_shift_logit_mean_{name}", 0.0))))
            contrast_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_contrast_shift_logit_mean_{name}", 0.0))))
        vision_gate_clean_mean = float(gate_row.get("vision_gate_clean_mean", 0.0))
        contrast_gap = contrast_mild - contrast_clean
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_common_cancel_logit_clean_mean": common_cancel_clean,
                "vision_common_cancel_logit_mild_mean": common_cancel_mild,
                "vision_common_cancel_logit_mild_minus_clean": common_cancel_mild - common_cancel_clean,
                "vision_post_cancel_logit_clean_mean": post_cancel_clean,
                "vision_post_cancel_logit_mild_mean": post_cancel_mild,
                "vision_post_cancel_logit_mild_minus_clean": post_cancel_mild - post_cancel_clean,
                "vision_contrast_shift_logit_clean_mean": contrast_clean,
                "vision_contrast_shift_logit_mild_mean": contrast_mild,
                "vision_contrast_shift_logit_mild_minus_clean": contrast_gap,
                "vision_clean_common_mode_excess": max(0.0, vision_gate_clean_mean - clean_gate_max),
                "common_cancel_nonvision_drift_mean": float(np.mean(common_nonvision_terms)) if common_nonvision_terms else 0.0,
                "contrast_shift_nonvision_drift_mean": float(np.mean(contrast_nonvision_terms)) if contrast_nonvision_terms else 0.0,
                "common_cancel_cheat_warning_flag": float(abs(common_cancel_clean) > 1.5 and float(gate_row.get("vision_gate_clean_below_cap_ratio", 0.0)) == 0.0),
                "contrast_shift_cheat_warning_flag": float(contrast_gap > 0.05 and float(gate_row.get("vision_final_gate_mild_minus_clean", 0.0)) < 0.002),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_10_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_9_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_clean_gate_mean_cap": 0.15 - float(score_row.get("vision_gate_clean_mean", 0.0)),
                "distance_to_common_cancel_logit": -1.0 - float(score_row.get("vision_common_cancel_logit_clean_mean", 0.0)),
                "distance_to_post_cancel_logit": 0.0 - float(score_row.get("vision_post_cancel_logit_clean_mean", 0.0)),
                "distance_to_contrast_shift_logit": float(score_row.get("vision_contrast_shift_logit_mild_minus_clean", 0.0)) - 0.02,
                "distance_to_common_cancel_nonvision_drift_cap": 0.02 - float(score_row.get("common_cancel_nonvision_drift_mean", 0.0)),
                "distance_to_contrast_shift_nonvision_drift_cap": 0.02 - float(score_row.get("contrast_shift_nonvision_drift_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_10_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_9_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "vision_common_cancel_logit_clean_mean",
            "vision_common_cancel_logit_mild_mean",
            "vision_common_cancel_logit_mild_minus_clean",
            "vision_post_cancel_logit_clean_mean",
            "vision_post_cancel_logit_mild_mean",
            "vision_post_cancel_logit_mild_minus_clean",
            "vision_contrast_shift_logit_clean_mean",
            "vision_contrast_shift_logit_mild_mean",
            "vision_contrast_shift_logit_mild_minus_clean",
            "vision_clean_common_mode_excess",
            "common_cancel_nonvision_drift_mean",
            "contrast_shift_nonvision_drift_mean",
            "common_cancel_cheat_warning_flag",
            "contrast_shift_cheat_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.drop(
            columns=[col for col in extra_cols if col != "candidate_name" and col in score_df.columns],
            errors="ignore",
        ).merge(gate_df[extra_cols], on="candidate_name", how="left")
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_gate_clean_mean", 1.0)) < 0.15
            and float(summary.get("vision_gate_clean_below_cap_ratio", 0.0)) > 0.0
            and float(summary.get("vision_common_cancel_logit_clean_mean", 0.0)) < -1.0
            and float(summary.get("vision_post_cancel_logit_clean_mean", 1.0)) < 0.0
            and float(summary.get("vision_contrast_shift_logit_mild_minus_clean", 0.0)) > 0.02
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.003
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("common_cancel_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("contrast_shift_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("common_cancel_cheat_warning_flag", 0.0)) == 0.0
            and float(summary.get("contrast_shift_cheat_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_gate_clean_below_cap_ratio", 0.0)) > 0.1
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_contrast_shift_logit_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
        )
        readiness = promotion and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_gate_clean_below_cap_ratio",
            "vision_contrast_shift_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "common_cancel_cheat_warning_flag",
            "contrast_shift_cheat_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_10_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_9_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Cancel Contrast Summary\n\n")
        cols = [
            "candidate_name",
            "vision_common_cancel_logit_clean_mean",
            "vision_post_cancel_logit_clean_mean",
            "vision_contrast_shift_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "vision_clean_common_mode_excess",
            "common_cancel_cheat_warning_flag",
            "contrast_shift_cheat_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_11_cancel_rescue_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_10_cancel_contrast_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        normal_rows = df[(df["candidate_name"] == candidate_name) & (df["corruption_type"] == "normal_eval")]
        source_gate_clean = float(normal.get("attention_source_bias_gate_mean_vision", 0.0))
        clean_gate_max = float(candidate_thresholds.get(candidate_name, {}).get("clean_gate_max", 0.02))
        post_cancel_clean = float(normal.get("attention_bias_gate_post_cancel_logit_mean_vision", 0.0))
        rescue_logit_clean = float(normal.get("attention_bias_gate_mild_rescue_logit_mean_vision", 0.0))
        rescue_logit_mild = float(mild.get("attention_bias_gate_mild_rescue_logit_mean_vision", 0.0))
        common_nonvision_terms = []
        rescue_nonvision_terms = []
        for name in ("distance", "ego", "route"):
            common_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            common_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            rescue_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_mild_rescue_logit_mean_{name}", 0.0))))
            rescue_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_mild_rescue_logit_mean_{name}", 0.0))))
        near_source_ratio = 0.0
        if not normal_rows.empty:
            near_source_ratio = float(
                (
                    normal_rows["attention_bias_gate_mean_vision"]
                    <= normal_rows["attention_source_bias_gate_mean_vision"] + 0.015
                ).mean()
            )
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_mild_rescue_gate_clean_mean": float(normal.get("attention_bias_gate_mild_rescue_gate_mean_vision", 0.0)),
                "vision_mild_rescue_gate_mild_mean": float(mild.get("attention_bias_gate_mild_rescue_gate_mean_vision", 0.0)),
                "vision_mild_rescue_value_clean_mean": float(normal.get("attention_bias_gate_mild_rescue_value_mean_vision", 0.0)),
                "vision_mild_rescue_value_mild_mean": float(mild.get("attention_bias_gate_mild_rescue_value_mean_vision", 0.0)),
                "vision_mild_rescue_logit_clean_mean": rescue_logit_clean,
                "vision_mild_rescue_logit_mild_mean": rescue_logit_mild,
                "vision_mild_rescue_logit_mild_minus_clean": rescue_logit_mild - rescue_logit_clean,
                "vision_gate_clean_source_gap_mean": float(gate_row.get("vision_gate_clean_mean", 0.0)) - source_gate_clean,
                "vision_gate_clean_near_source_ratio": near_source_ratio,
                "mild_rescue_nonvision_drift_mean": float(np.mean(rescue_nonvision_terms)) if rescue_nonvision_terms else 0.0,
                "clean_rescue_leak_warning_flag": float(rescue_logit_clean > 0.02),
                "over_cancel_warning_flag": float(post_cancel_clean < np.log(0.01 / 0.99)),
                "vision_clean_common_mode_excess": max(0.0, float(gate_row.get("vision_gate_clean_mean", 0.0)) - clean_gate_max),
                "common_cancel_nonvision_drift_mean": float(np.mean(common_nonvision_terms)) if common_nonvision_terms else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_11_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_10_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    post_cancel_floor_logit = np.log(0.02 / 0.98)
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_clean_near_source_ratio": float(score_row.get("vision_gate_clean_near_source_ratio", 0.0)) - 0.3,
                "distance_to_post_cancel_lower": -2.7 - float(score_row.get("vision_post_cancel_logit_clean_mean", 0.0)),
                "distance_to_post_cancel_floor": float(score_row.get("vision_post_cancel_logit_clean_mean", 0.0)) - post_cancel_floor_logit,
                "distance_to_mild_rescue_gap": float(score_row.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) - 0.03,
                "distance_to_mild_rescue_nonvision_drift_cap": 0.02 - float(score_row.get("mild_rescue_nonvision_drift_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_11_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_10_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    extra_cols = [
        col
        for col in [
            "candidate_name",
            "vision_mild_rescue_gate_clean_mean",
            "vision_mild_rescue_gate_mild_mean",
            "vision_mild_rescue_value_clean_mean",
            "vision_mild_rescue_value_mild_mean",
            "vision_mild_rescue_logit_clean_mean",
            "vision_mild_rescue_logit_mild_mean",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_gate_clean_source_gap_mean",
            "vision_gate_clean_near_source_ratio",
            "mild_rescue_nonvision_drift_mean",
            "clean_rescue_leak_warning_flag",
            "over_cancel_warning_flag",
        ]
        if col in gate_df.columns
    ]
    if len(extra_cols) > 1:
        score_df = score_df.drop(
            columns=[col for col in extra_cols if col != "candidate_name" and col in score_df.columns],
            errors="ignore",
        ).merge(gate_df[extra_cols], on="candidate_name", how="left")
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    post_cancel_floor_logit = np.log(0.02 / 0.98)
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_gate_clean_mean", 1.0)) < 0.08
            and float(summary.get("vision_gate_clean_near_source_ratio", 0.0)) > 0.3
            and float(summary.get("vision_post_cancel_logit_clean_mean", 1.0)) < -2.7
            and float(summary.get("vision_post_cancel_logit_clean_mean", -999.0)) > post_cancel_floor_logit
            and float(summary.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) > 0.03
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.005
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("common_cancel_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("mild_rescue_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("clean_rescue_leak_warning_flag", 0.0)) == 0.0
            and float(summary.get("over_cancel_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_gate_clean_mean", 1.0)) < 0.06
            and float(summary.get("vision_gate_clean_near_source_ratio", 0.0)) > 0.5
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
            and float(summary.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
        )
        readiness = promotion and float(summary.get("normal_success_rate", 0.0)) >= float(source_row.get("normal_success_rate", 0.0)) - 0.03
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_gate_clean_near_source_ratio",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "clean_rescue_leak_warning_flag",
            "over_cancel_warning_flag",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False, True, True, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def generate_v3_7f2_11_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_10_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Cancel Rescue Summary\n\n")
        cols = [
            "candidate_name",
            "vision_gate_clean_mean",
            "vision_gate_clean_near_source_ratio",
            "vision_post_cancel_logit_clean_mean",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "clean_rescue_leak_warning_flag",
            "over_cancel_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_12_excess_cancel_selective_rescue_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_11_cancel_rescue_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    over_cancel_floor = np.log(0.02 / 0.98)
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        rescue_gate_clean = float(normal.get("attention_bias_gate_mild_rescue_gate_mean_vision", 0.0))
        rescue_gate_mild = float(mild.get("attention_bias_gate_mild_rescue_gate_mean_vision", 0.0))
        rescue_logit_clean = float(normal.get("attention_bias_gate_mild_rescue_logit_mean_vision", 0.0))
        rescue_logit_mild = float(mild.get("attention_bias_gate_mild_rescue_logit_mean_vision", 0.0))
        excess_clean = float(normal.get("attention_bias_gate_post_shift_excess_mean_vision", 0.0))
        excess_mild = float(mild.get("attention_bias_gate_post_shift_excess_mean_vision", 0.0))
        cancel_gate_clean = float(normal.get("attention_bias_gate_common_cancel_gate_mean_vision", 0.0))
        cancel_gate_mild = float(mild.get("attention_bias_gate_common_cancel_gate_mean_vision", 0.0))
        common_nonvision_terms = []
        rescue_nonvision_terms = []
        for name in ("distance", "ego", "route"):
            common_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            common_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            rescue_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_mild_rescue_logit_mean_{name}", 0.0))))
            rescue_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_mild_rescue_logit_mean_{name}", 0.0))))
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_post_shift_excess_clean_mean": excess_clean,
                "vision_post_shift_excess_mild_mean": excess_mild,
                "vision_common_cancel_gate_clean_mean": cancel_gate_clean,
                "vision_common_cancel_gate_mild_mean": cancel_gate_mild,
                "vision_common_cancel_gate_mild_minus_clean": cancel_gate_mild - cancel_gate_clean,
                "vision_mild_rescue_gate_clean_mean": rescue_gate_clean,
                "vision_mild_rescue_gate_mild_mean": rescue_gate_mild,
                "vision_mild_rescue_gate_mild_minus_clean": rescue_gate_mild - rescue_gate_clean,
                "vision_mild_rescue_logit_clean_mean": rescue_logit_clean,
                "vision_mild_rescue_logit_mild_mean": rescue_logit_mild,
                "vision_mild_rescue_logit_mild_minus_clean": rescue_logit_mild - rescue_logit_clean,
                "cancel_excess_unused_warning_flag": float(excess_clean > 0.2 and abs(float(normal.get("attention_bias_gate_common_cancel_logit_mean_vision", 0.0))) < 0.2),
                "rescue_constant_mode_warning_flag": float(abs(rescue_gate_mild - rescue_gate_clean) < 0.05 and abs(rescue_logit_mild - rescue_logit_clean) < 0.01),
                "clean_rescue_leak_warning_flag": float(rescue_gate_clean > float(candidate_thresholds.get(candidate_name, {}).get("clean_rescue_gate_max", 0.10))),
                "over_cancel_warning_flag": float(float(normal.get("attention_bias_gate_post_cancel_logit_mean_vision", 0.0)) < over_cancel_floor),
                "common_cancel_nonvision_drift_mean": float(np.mean(common_nonvision_terms)) if common_nonvision_terms else 0.0,
                "mild_rescue_nonvision_drift_mean": float(np.mean(rescue_nonvision_terms)) if rescue_nonvision_terms else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_12_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_11_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_post_shift_excess_clean_zero": 0.05 - float(score_row.get("vision_post_shift_excess_clean_mean", 1.0)),
                "distance_to_common_cancel_gate_activation": float(score_row.get("vision_common_cancel_gate_clean_mean", 0.0)) - 0.10,
                "distance_to_rescue_gate_selectivity": float(score_row.get("vision_mild_rescue_gate_mild_minus_clean", 0.0)) - 0.15,
                "distance_to_clean_rescue_gate_ceiling": float(score_row.get("clean_rescue_gate_max", 0.10)) - float(score_row.get("vision_mild_rescue_gate_clean_mean", 0.0)),
                "distance_to_final_gate_gap": float(score_row.get("vision_final_gate_mild_minus_clean", 0.0)) - 0.005,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_12_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_11_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    keep_cols = [
        "candidate_name",
        "vision_post_shift_excess_clean_mean",
        "vision_post_shift_excess_mild_mean",
        "vision_common_cancel_gate_clean_mean",
        "vision_common_cancel_gate_mild_mean",
        "vision_common_cancel_gate_mild_minus_clean",
        "vision_mild_rescue_gate_clean_mean",
        "vision_mild_rescue_gate_mild_mean",
        "vision_mild_rescue_gate_mild_minus_clean",
        "vision_mild_rescue_logit_clean_mean",
        "vision_mild_rescue_logit_mild_mean",
        "vision_mild_rescue_logit_mild_minus_clean",
        "cancel_excess_unused_warning_flag",
        "rescue_constant_mode_warning_flag",
        "clean_rescue_leak_warning_flag",
        "over_cancel_warning_flag",
    ]
    available = [col for col in keep_cols if col in gate_df.columns]
    if len(available) > 1:
        score_df = score_df.drop(columns=[col for col in available if col != "candidate_name" and col in score_df.columns], errors="ignore").merge(
            gate_df[available],
            on="candidate_name",
            how="left",
        )
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_gate_clean_near_source_ratio", 0.0)) > 0.6
            and float(summary.get("vision_post_shift_excess_clean_mean", 1.0)) < 0.05
            and float(summary.get("vision_mild_rescue_gate_clean_mean", 1.0)) < 0.15
            and float(summary.get("vision_mild_rescue_gate_mild_minus_clean", 0.0)) > 0.15
            and float(summary.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) > 0.03
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.005
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("common_cancel_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("mild_rescue_nonvision_drift_mean", 1.0)) <= 0.02
            and float(summary.get("cancel_excess_unused_warning_flag", 0.0)) == 0.0
            and float(summary.get("rescue_constant_mode_warning_flag", 0.0)) == 0.0
            and float(summary.get("clean_rescue_leak_warning_flag", 0.0)) == 0.0
            and float(summary.get("over_cancel_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_gate_clean_near_source_ratio", 0.0)) > 0.7
            and float(summary.get("vision_mild_rescue_gate_mild_minus_clean", 0.0)) > 0.20
            and float(summary.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
        )
        readiness = promotion
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(readiness))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    score_df = score_df.sort_values(
        [
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_gate_clean_near_source_ratio",
            "vision_mild_rescue_gate_mild_minus_clean",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "clean_rescue_leak_warning_flag",
            "rescue_constant_mode_warning_flag",
            "over_cancel_warning_flag",
            "tie_break_score",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, False],
    ).reset_index(drop=True)
    return score_df


def generate_v3_7f2_12_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_11_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Excess Cancel Selective Rescue Summary\n\n")
        cols = [
            "candidate_name",
            "vision_post_shift_excess_clean_mean",
            "vision_common_cancel_gate_clean_mean",
            "vision_mild_rescue_gate_clean_mean",
            "vision_mild_rescue_gate_mild_minus_clean",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "cancel_excess_unused_warning_flag",
            "rescue_constant_mode_warning_flag",
            "clean_rescue_leak_warning_flag",
            "over_cancel_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")


def compute_v3_7f2_13_anchor_cancel_selector_rescue_metrics(
    df: pd.DataFrame,
    candidate_thresholds: dict[str, dict[str, float]],
    *,
    clean_reference_candidate: str = "reference_v3_6_medium_locked",
    source_candidate: str = "reference_v3_7c_medium_locked",
) -> pd.DataFrame:
    gate_df = compute_v3_7f2_11_cancel_rescue_metrics(
        df,
        candidate_thresholds=candidate_thresholds,
        clean_reference_candidate=clean_reference_candidate,
        source_candidate=source_candidate,
    )
    score_means = df.groupby(["candidate_name", "corruption_type"], as_index=False).mean(numeric_only=True)
    normal_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "normal_eval"].iterrows()
    }
    mild_by_candidate = {
        str(row["candidate_name"]): row
        for _, row in score_means[score_means["corruption_type"] == "mild_vision"].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, gate_row in gate_df.iterrows():
        candidate_name = str(gate_row["candidate_name"])
        normal = normal_by_candidate.get(candidate_name, pd.Series(dtype=float))
        mild = mild_by_candidate.get(candidate_name, pd.Series(dtype=float))
        clean_selector_max = float(candidate_thresholds.get(candidate_name, {}).get("clean_selector_max", 0.10))
        common_nonvision_terms = []
        selector_nonvision_terms = []
        for name in ("distance", "ego", "route"):
            common_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            common_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_common_cancel_logit_mean_{name}", 0.0))))
            selector_nonvision_terms.append(abs(float(normal.get(f"attention_bias_gate_mild_rescue_logit_mean_{name}", 0.0))))
            selector_nonvision_terms.append(abs(float(mild.get(f"attention_bias_gate_mild_rescue_logit_mean_{name}", 0.0))))
        selector_prob_clean = float(normal.get("attention_bias_gate_selector_prob_mean_vision", normal.get("attention_bias_gate_mild_rescue_gate_mean_vision", 0.0)))
        selector_prob_mild = float(mild.get("attention_bias_gate_selector_prob_mean_vision", mild.get("attention_bias_gate_mild_rescue_gate_mean_vision", 0.0)))
        selector_active_clean = float(normal.get("attention_bias_gate_selector_active_mean_vision", 0.0))
        selector_active_mild = float(mild.get("attention_bias_gate_selector_active_mean_vision", 0.0))
        rescue_logit_clean = float(normal.get("attention_bias_gate_mild_rescue_logit_mean_vision", 0.0))
        rescue_logit_mild = float(mild.get("attention_bias_gate_mild_rescue_logit_mean_vision", 0.0))
        excess_clean = float(normal.get("attention_bias_gate_post_shift_excess_mean_vision", 0.0))
        excess_mild = float(mild.get("attention_bias_gate_post_shift_excess_mean_vision", 0.0))
        cancel_prob_clean = float(normal.get("attention_bias_gate_common_cancel_gate_mean_vision", 0.0))
        cancel_prob_mild = float(mild.get("attention_bias_gate_common_cancel_gate_mean_vision", 0.0))
        clean_source_margin = (
            float(gate_row.get("vision_source_gate_clean_mean", float(normal.get("attention_source_bias_gate_mean_vision", 0.0))))
            + 0.015
            - float(gate_row.get("vision_gate_clean_mean", 0.0))
        )
        rows.append(
            {
                **gate_row.to_dict(),
                "vision_common_cancel_prob_clean_mean": cancel_prob_clean,
                "vision_common_cancel_prob_mild_mean": cancel_prob_mild,
                "vision_post_shift_excess_clean_mean": excess_clean,
                "vision_post_shift_excess_mild_mean": excess_mild,
                "vision_post_cancel_logit_clean_mean": float(normal.get("attention_bias_gate_post_cancel_logit_mean_vision", 0.0)),
                "vision_post_cancel_logit_mild_mean": float(mild.get("attention_bias_gate_post_cancel_logit_mean_vision", 0.0)),
                "vision_selector_prob_clean_mean": selector_prob_clean,
                "vision_selector_prob_mild_mean": selector_prob_mild,
                "vision_selector_prob_mild_minus_clean": selector_prob_mild - selector_prob_clean,
                "vision_selector_active_clean_mean": selector_active_clean,
                "vision_selector_active_mild_mean": selector_active_mild,
                "vision_mild_rescue_logit_clean_mean": rescue_logit_clean,
                "vision_mild_rescue_logit_mild_mean": rescue_logit_mild,
                "vision_mild_rescue_logit_mild_minus_clean": rescue_logit_mild - rescue_logit_clean,
                "vision_gate_clean_source_margin": clean_source_margin,
                "attention_gate_cancel_temperature": float(normal.get("attention_gate_cancel_temperature", 1.0)),
                "attention_gate_selector_temperature": float(normal.get("attention_gate_selector_temperature", 1.0)),
                "attention_gate_rescue_selector_floor": float(normal.get("attention_gate_rescue_selector_floor", 0.0)),
                "attention_gate_mild_gate_min": float(normal.get("attention_gate_mild_gate_min", 0.08)),
                "cancel_midrange_stall_warning_flag": float(excess_clean > 1.0 and cancel_prob_clean < 0.75),
                "selector_constant_mode_warning_flag": float(abs(selector_prob_mild - selector_prob_clean) < 0.05),
                "clean_selector_leak_warning_flag": float(selector_prob_clean > clean_selector_max),
                "common_cancel_nonvision_drift_mean": float(np.mean(common_nonvision_terms)) if common_nonvision_terms else 0.0,
                "selector_nonvision_drift_mean": float(np.mean(selector_nonvision_terms)) if selector_nonvision_terms else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_13_pass_distance_dashboard(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> pd.DataFrame:
    dashboard = build_v3_7f2_11_pass_distance_dashboard(
        score_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
    )
    rows = []
    for _, row in dashboard.iterrows():
        candidate_name = str(row["candidate_name"])
        score_row = score_df[score_df["candidate_name"] == candidate_name].iloc[0]
        rows.append(
            {
                **row.to_dict(),
                "distance_to_post_shift_excess_clean_zero": float(score_row.get("vision_post_shift_excess_clean_mean", 0.0)) - 1.0,
                "distance_to_common_cancel_gate_activation": float(score_row.get("vision_common_cancel_prob_clean_mean", 0.0)) - 0.80,
                "distance_to_clean_source_margin": float(score_row.get("vision_gate_clean_source_margin", -1.0)),
                "distance_to_rescue_gate_selectivity": float(score_row.get("vision_selector_prob_mild_minus_clean", 0.0)) - 0.25,
                "distance_to_clean_rescue_gate_ceiling": float(score_row.get("vision_selector_prob_clean_mean", 0.0)) - 0.15,
                "distance_to_final_gate_gap": float(score_row.get("vision_final_gate_mild_minus_clean", 0.0)) - 0.006,
            }
        )
    return pd.DataFrame(rows)


def build_v3_7f2_13_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_candidate: str,
    source_candidate: str,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    score_df = build_v3_7f2_11_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=baseline_candidate,
        source_candidate=source_candidate,
        tiebreak_weights=tiebreak_weights,
    )
    keep_cols = [
        "candidate_name",
        "vision_common_cancel_prob_clean_mean",
        "vision_common_cancel_prob_mild_mean",
        "vision_post_shift_excess_clean_mean",
        "vision_post_shift_excess_mild_mean",
        "vision_post_cancel_logit_clean_mean",
        "vision_post_cancel_logit_mild_mean",
        "vision_selector_prob_clean_mean",
        "vision_selector_prob_mild_mean",
        "vision_selector_prob_mild_minus_clean",
        "vision_selector_active_clean_mean",
        "vision_selector_active_mild_mean",
        "vision_mild_rescue_logit_clean_mean",
        "vision_mild_rescue_logit_mild_mean",
        "vision_mild_rescue_logit_mild_minus_clean",
        "vision_gate_clean_source_margin",
        "attention_gate_cancel_temperature",
        "attention_gate_selector_temperature",
        "attention_gate_rescue_selector_floor",
        "attention_gate_mild_gate_min",
        "cancel_midrange_stall_warning_flag",
        "selector_constant_mode_warning_flag",
        "clean_selector_leak_warning_flag",
        "common_cancel_nonvision_drift_mean",
        "selector_nonvision_drift_mean",
    ]
    score_df = score_df.drop(columns=[col for col in keep_cols if col != "candidate_name" and col in score_df.columns], errors="ignore").merge(
        gate_df[[col for col in keep_cols if col in gate_df.columns]],
        on="candidate_name",
        how="left",
    )
    source_row = score_df[score_df["candidate_name"] == source_candidate].iloc[0].to_dict()
    activation_flags = []
    promotion_flags = []
    readiness_flags = []
    for _, row in score_df.iterrows():
        summary = row.to_dict()
        activation = (
            float(summary.get("gate_reconstruction_error_mean", 1.0)) < 1.0e-6
            and float(summary.get("vision_post_shift_excess_clean_mean", 0.0)) > 1.0
            and float(summary.get("vision_common_cancel_prob_clean_mean", 0.0)) > 0.80
            and float(summary.get("vision_post_cancel_logit_clean_mean", 0.0)) < -2.4
            and float(summary.get("vision_gate_clean_source_margin", -1.0)) > 0.0
            and float(summary.get("vision_selector_prob_clean_mean", 1.0)) < 0.15
            and float(summary.get("vision_selector_prob_mild_minus_clean", 0.0)) > 0.25
            and float(summary.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) > 0.03
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.006
            and float(summary.get("vision_attention_targetlogit_sign_match_flag", 0.0)) == 1.0
            and float(summary.get("vision_attention_relative_score_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_relative_score_clean_minus_mild", 0.0)) - 0.002
            and float(summary.get("vision_attention_clean_minus_mild", -1.0))
            >= float(source_row.get("vision_attention_clean_minus_mild", 0.0)) - 0.001
            and float(summary.get("cancel_midrange_stall_warning_flag", 0.0)) == 0.0
            and float(summary.get("selector_constant_mode_warning_flag", 0.0)) == 0.0
            and float(summary.get("clean_selector_leak_warning_flag", 0.0)) == 0.0
        )
        promotion = (
            activation
            and float(summary.get("vision_gate_clean_source_margin", -1.0)) > 0.0
            and float(summary.get("vision_selector_prob_mild_minus_clean", 0.0)) > 0.30
            and float(summary.get("vision_mild_rescue_logit_mild_minus_clean", 0.0)) > 0.05
            and float(summary.get("vision_final_gate_mild_minus_clean", 0.0)) > 0.01
            and float(summary.get("vision_gate_mild_above_floor_ratio", 0.0)) > 0.0
            and float(summary.get("vision_gate_gap_margin_satisfied_ratio", 0.0)) > 0.0
        )
        activation_flags.append(float(activation))
        promotion_flags.append(float(promotion))
        readiness_flags.append(float(promotion))
    score_df["activation_debug_success_flag"] = activation_flags
    score_df["promotion_success_flag"] = promotion_flags
    score_df["pass_closing_ready_flag"] = readiness_flags
    return score_df.sort_values(
        [
            "promotion_success_flag",
            "activation_debug_success_flag",
            "vision_gate_clean_source_margin",
            "vision_selector_prob_mild_minus_clean",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "clean_selector_leak_warning_flag",
            "selector_constant_mode_warning_flag",
            "cancel_midrange_stall_warning_flag",
            "tie_break_score",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, False],
    ).reset_index(drop=True)


def generate_v3_7f2_13_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    targetlogit_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
    title: str,
) -> None:
    generate_v3_7f2_11_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=output_path,
        recommended=recommended,
        secondary=secondary,
        title=title,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Anchor Cancel Selector Rescue Summary\n\n")
        cols = [
            "candidate_name",
            "vision_common_cancel_prob_clean_mean",
            "vision_post_shift_excess_clean_mean",
            "vision_post_cancel_logit_clean_mean",
            "vision_selector_prob_clean_mean",
            "vision_selector_prob_mild_minus_clean",
            "vision_mild_rescue_logit_mild_minus_clean",
            "vision_final_gate_mild_minus_clean",
            "cancel_midrange_stall_warning_flag",
            "selector_constant_mode_warning_flag",
            "clean_selector_leak_warning_flag",
        ]
        available = [col for col in cols if col in score_df.columns]
        handle.write(score_df[available].to_markdown(index=False))
        handle.write("\n")
