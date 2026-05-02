from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from rcs_ac_first.data.expert_dataset import (
    ExpertDataset,
    SequenceExpertDataset,
    load_expert_npz,
    summarize_expert_arrays,
)
from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.rl.losses import compute_dense_perception_aux_loss
from rcs_ac_first.runtime.auxiliary import RISK_STATS_FIELDS
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.scripts.train import _project_path, _resolve_config_path, _set_seed


BC_LOG_FIELDS: tuple[str, ...] = (
    "epoch",
    "global_step",
    "split",
    "bc_loss",
    "steer_mse",
    "longitudinal_mse",
    "action_l1",
    "bc_sample_weight_mean",
    "bc_sample_weight_min",
    "bc_sample_weight_late_mask_rate",
    "visual_zero_action_l1",
    "visual_zero_steer_delta",
    "visual_zero_longitudinal_delta",
    "visual_zero_margin_loss",
    "visual_zero_steer_margin_loss",
    "visual_zero_corrective_steer_delta",
    "visual_zero_corrective_sign_loss",
    "visual_zero_corrective_mask_rate",
    "target_corrective_delta_loss",
    "target_corrective_delta_l1",
    "target_corrective_delta_sign_rate",
    "visual_zero_target_delta_loss",
    "visual_zero_target_delta_l1",
    "visual_zero_target_delta_sign_rate",
    "target_corrective_delta_mask_rate",
    "visual_center_corrective_sign_loss",
    "visual_center_corrective_sign_rate",
    "visual_center_correction_rate",
    "visual_center_corrective_mask_rate",
    "visual_center_target_consistency_rate",
    "visual_center_target_consistency_mask_rate",
    "visual_center_error_abs_mean",
    "visual_center_signed_delta_mean",
    "route_aware_visual_center_loss",
    "route_aware_visual_center_sign_rate",
    "route_aware_visual_center_correction_rate",
    "route_aware_visual_center_mask_rate",
    "route_aware_visual_center_target_consistency_rate",
    "route_aware_visual_center_target_consistency_mask_rate",
    "route_aware_visual_center_signed_delta_mean",
    "collision_cluster_route_offset_agreement_loss",
    "collision_cluster_route_offset_agreement_sign_rate",
    "collision_cluster_route_offset_agreement_mask_rate",
    "collision_cluster_route_offset_agreement_signed_delta_mean",
    "collision_cluster_route_offset_agreement_corrected_delta_mean",
    "collision_cluster_route_offset_agreement_positive_sign_rate",
    "collision_cluster_route_offset_agreement_negative_sign_rate",
    "collision_cluster_route_offset_agreement_positive_mask_rate",
    "collision_cluster_route_offset_agreement_negative_mask_rate",
    "early_route_offset_agreement_loss",
    "early_route_offset_agreement_sign_rate",
    "early_route_offset_agreement_mask_rate",
    "early_route_offset_agreement_signed_delta_mean",
    "early_route_offset_agreement_corrected_delta_mean",
    "early_route_offset_agreement_positive_sign_rate",
    "early_route_offset_agreement_negative_sign_rate",
    "early_route_offset_agreement_positive_mask_rate",
    "early_route_offset_agreement_negative_mask_rate",
    "early_severe_steer_delta_loss",
    "early_severe_steer_delta_l1",
    "early_severe_steer_sign_loss",
    "early_severe_steer_sign_rate",
    "early_severe_steer_wrong_sign_rate",
    "early_severe_absolute_steer_loss",
    "early_severe_absolute_steer_sign_rate",
    "early_severe_probe_absolute_steer_loss",
    "early_severe_probe_absolute_steer_sign_rate",
    "early_severe_probe_absolute_steer_mask_rate",
    "early_severe_longitudinal_ceiling_loss",
    "early_severe_longitudinal_ceiling_rate",
    "late_recovery_absolute_steer_loss",
    "late_recovery_absolute_steer_sign_rate",
    "late_recovery_absolute_steer_mask_rate",
    "early_center_zero_steer_loss",
    "early_center_zero_steer_abs_mean",
    "early_center_zero_steer_mask_rate",
    "early_severe_steer_mask_rate",
    "early_severe_steer_left_sign_rate",
    "early_severe_steer_right_sign_rate",
    "early_severe_steer_left_mask_rate",
    "early_severe_steer_right_mask_rate",
    "early_severe_steer_signed_delta_mean",
    "early_severe_steer_corrective_delta_mean",
    "route_progress_longitudinal_floor_loss",
    "route_progress_longitudinal_floor_rate",
    "route_progress_longitudinal_mask_rate",
    "route_progress_longitudinal_mean",
    "perception_aux_loss",
    "aux_road_mask_loss",
    "aux_lane_edge_mask_loss",
    "aux_road_mask_mae",
    "aux_lane_edge_mask_mae",
    "aux_road_mask_iou",
    "aux_lane_edge_mask_iou",
    "aux_road_mask_positive_ratio",
    "aux_lane_edge_positive_ratio",
    "aux_mask_available_rate",
    "sequence_valid_rate",
    "sequence_mask_available_valid_rate",
    "sequence_perception_logits_available",
    "policy_std",
    "fusion_rgb",
    "fusion_lidar",
    "fusion_ego",
    "fusion_route",
    "reliability_rgb",
    "reliability_lidar",
    "reliability_ego",
    "reliability_route",
    "risk_scalar",
    "contribution_rgb",
    "contribution_lidar",
    "contribution_ego",
    "contribution_route",
    "compact_visual_features_enabled",
    "compact_visual_feature_dim",
    "compact_visual_feature_l1_mean",
    "road_compact_feature_l1_mean",
    "lane_compact_feature_l1_mean",
    "visual_road_x_mean",
    "visual_lane_x_mean",
    "visual_lane_center_error_mean",
    "compact_visual_actor_input_l1_mean",
    "visual_actor_delta_l1_mean",
)

PERCEPTION_AUX_METRIC_KEYS: tuple[str, ...] = (
    "perception_aux_loss",
    "aux_road_mask_loss",
    "aux_lane_edge_mask_loss",
    "aux_road_mask_mae",
    "aux_lane_edge_mask_mae",
    "aux_road_mask_iou",
    "aux_lane_edge_mask_iou",
    "aux_road_mask_positive_ratio",
    "aux_lane_edge_positive_ratio",
    "aux_mask_available_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior Cloning training for RCS-AC-first.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-strict", default="true")
    return parser.parse_args()


def _bc_raw(config: Any) -> dict[str, Any]:
    raw = config.raw.get("bc") or {}
    if not isinstance(raw, dict):
        raise ValueError("config bc section must be a mapping")
    return dict(raw)


def _device(config: Any, override: str | None) -> torch.device:
    requested = override or config.train.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(str(requested))


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in batch.items()}


def _weighted_action_loss(
    squared: torch.Tensor,
    *,
    steer_loss_weight: float,
    longitudinal_loss_weight: float,
) -> torch.Tensor:
    weights = torch.as_tensor(
        [float(steer_loss_weight), float(longitudinal_loss_weight)],
        dtype=squared.dtype,
        device=squared.device,
    ).clamp_min(0.0)
    weight_sum = weights.sum().clamp_min(1.0)
    return (squared * weights).sum(dim=-1) / weight_sum


def _compute_losses(
    action_mean: torch.Tensor,
    expert_action: torch.Tensor,
    *,
    steer_loss_weight: float = 1.0,
    longitudinal_loss_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    squared = (action_mean - expert_action).pow(2)
    per_sample_loss = _weighted_action_loss(
        squared,
        steer_loss_weight=steer_loss_weight,
        longitudinal_loss_weight=longitudinal_loss_weight,
    )
    if sample_weight is not None:
        weights = sample_weight.to(device=action_mean.device, dtype=action_mean.dtype).reshape_as(per_sample_loss)
        weights = weights.clamp_min(0.0)
        bc_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
        weight_mean = weights.mean()
        weight_min = weights.min() if weights.numel() else action_mean.new_ones(())
    else:
        bc_loss = per_sample_loss.mean()
        weight_mean = action_mean.new_ones(())
        weight_min = action_mean.new_ones(())
    steer_mse = F.mse_loss(action_mean[:, 0], expert_action[:, 0])
    longitudinal_mse = F.mse_loss(action_mean[:, 1], expert_action[:, 1])
    action_l1 = torch.mean(torch.abs(action_mean - expert_action))
    return {
        "bc_loss": bc_loss,
        "steer_mse": steer_mse,
        "longitudinal_mse": longitudinal_mse,
        "action_l1": action_l1,
        "bc_sample_weight_mean": weight_mean,
        "bc_sample_weight_min": weight_min,
        "bc_sample_weight_late_mask_rate": action_mean.new_zeros(()),
    }


def _zero_perception_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {key: zero for key in PERCEPTION_AUX_METRIC_KEYS}


def _compute_frame_perception_losses(
    *,
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    perception_aux_loss_weight: float,
    road_mask_weight: float,
    lane_edge_weight: float,
    lane_edge_pos_weight: float,
) -> dict[str, torch.Tensor]:
    action_mean = outputs.get("action_mean")
    if not isinstance(action_mean, torch.Tensor):
        raise RuntimeError("Model output action_mean is missing.")
    if float(perception_aux_loss_weight) <= 0.0:
        return _zero_perception_losses(action_mean)
    pred_road = outputs.get("pred_road_mask_logits")
    pred_lane = outputs.get("pred_lane_edge_mask_logits")
    if not isinstance(pred_road, torch.Tensor) or not isinstance(pred_lane, torch.Tensor):
        return _zero_perception_losses(action_mean)
    road_target = batch.get("aux_road_mask_target")
    lane_target = batch.get("aux_lane_edge_mask_target")
    if lane_target is None:
        lane_target = batch.get("aux_lane_edge_target")
    available = batch.get("aux_mask_targets_available")
    if road_target is None or lane_target is None or available is None:
        return _zero_perception_losses(action_mean)
    dense_loss, metrics = compute_dense_perception_aux_loss(
        pred_road,
        road_target,
        pred_lane,
        lane_target,
        available,
        road_mask_weight=road_mask_weight,
        lane_edge_weight=lane_edge_weight,
        lane_edge_pos_weight=lane_edge_pos_weight,
    )
    result = {key: action_mean.new_tensor(float(value)) for key, value in metrics.items()}
    result["perception_aux_loss"] = dense_loss
    return result


def _compute_sequence_perception_losses(
    *,
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    perception_aux_loss_weight: float,
    road_mask_weight: float,
    lane_edge_weight: float,
    lane_edge_pos_weight: float,
) -> dict[str, torch.Tensor]:
    action_mean = outputs.get("action_mean_sequence")
    if not isinstance(action_mean, torch.Tensor):
        fallback = outputs.get("action_mean")
        if not isinstance(fallback, torch.Tensor):
            raise RuntimeError("Model sequence output action_mean is missing.")
        return _zero_perception_losses(fallback)
    if float(perception_aux_loss_weight) <= 0.0:
        return _zero_perception_losses(action_mean)
    pred_road = outputs.get("pred_road_mask_logits_sequence")
    pred_lane = outputs.get("pred_lane_edge_mask_logits_sequence")
    if not isinstance(pred_road, torch.Tensor) or not isinstance(pred_lane, torch.Tensor):
        result = _zero_perception_losses(action_mean)
        result["sequence_valid_rate"] = valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).mean()
        result["sequence_mask_available_valid_rate"] = action_mean.new_zeros(())
        result["sequence_perception_logits_available"] = action_mean.new_zeros(())
        return result
    road_target = batch.get("aux_road_mask_target")
    lane_target = batch.get("aux_lane_edge_mask_target")
    if lane_target is None:
        lane_target = batch.get("aux_lane_edge_target")
    available = batch.get("aux_mask_targets_available")
    if road_target is None or lane_target is None or available is None:
        return _zero_perception_losses(action_mean)
    batch_size, sequence_length = int(pred_road.shape[0]), int(pred_road.shape[1])
    flat_valid = valid_mask.to(device=pred_road.device, dtype=pred_road.dtype).reshape(batch_size * sequence_length)
    flat_available = available.to(device=pred_road.device, dtype=pred_road.dtype).reshape(batch_size * sequence_length)
    dense_loss, metrics = compute_dense_perception_aux_loss(
        pred_road.reshape(batch_size * sequence_length, *pred_road.shape[2:]),
        road_target.to(device=pred_road.device, dtype=pred_road.dtype).reshape(batch_size * sequence_length, *pred_road.shape[2:]),
        pred_lane.reshape(batch_size * sequence_length, *pred_lane.shape[2:]),
        lane_target.to(device=pred_lane.device, dtype=pred_lane.dtype).reshape(batch_size * sequence_length, *pred_lane.shape[2:]),
        flat_available * flat_valid,
        road_mask_weight=road_mask_weight,
        lane_edge_weight=lane_edge_weight,
        lane_edge_pos_weight=lane_edge_pos_weight,
    )
    result = {key: action_mean.new_tensor(float(value)) for key, value in metrics.items()}
    result["perception_aux_loss"] = dense_loss
    result["sequence_valid_rate"] = flat_valid.mean()
    valid_count = flat_valid.sum().clamp_min(1.0)
    result["sequence_mask_available_valid_rate"] = (flat_available * flat_valid).sum() / valid_count
    result["sequence_perception_logits_available"] = action_mean.new_ones(())
    return result


def _compute_sequence_losses(
    action_mean: torch.Tensor,
    expert_action: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    action_smooth_loss_weight: float = 0.0,
    steer_loss_weight: float = 1.0,
    longitudinal_loss_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if action_mean.ndim != 3:
        raise ValueError(f"action_mean must have shape [B, L, A], got {tuple(action_mean.shape)}")
    mask = valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    denom = mask.sum().clamp_min(1.0)
    diff = action_mean - expert_action.to(device=action_mean.device, dtype=action_mean.dtype)
    squared = diff.pow(2)
    abs_diff = diff.abs()
    weighted_squared = _weighted_action_loss(
        squared,
        steer_loss_weight=steer_loss_weight,
        longitudinal_loss_weight=longitudinal_loss_weight,
    )
    if sample_weight is not None:
        weights = sample_weight.to(device=action_mean.device, dtype=action_mean.dtype).reshape_as(mask).clamp_min(0.0)
        weighted_mask = mask * weights
    else:
        weights = torch.ones_like(mask)
        weighted_mask = mask
    weighted_denom = weighted_mask.sum().clamp_min(1.0)
    bc_loss = (weighted_squared * weighted_mask).sum() / weighted_denom
    steer_mse = (squared[..., 0] * mask).sum() / denom
    longitudinal_mse = (squared[..., 1] * mask).sum() / denom
    action_l1 = (abs_diff.sum(dim=-1) * mask).sum() / (denom * float(action_mean.shape[-1]))
    if int(action_mean.shape[1]) > 1 and float(action_smooth_loss_weight) > 0.0:
        pair_mask = (mask[:, 1:] * mask[:, :-1]).clamp(0.0, 1.0)
        pair_denom = pair_mask.sum().clamp_min(1.0)
        smooth_loss = ((action_mean[:, 1:] - action_mean[:, :-1]).abs().sum(dim=-1) * pair_mask).sum()
        smooth_loss = smooth_loss / (pair_denom * float(action_mean.shape[-1]))
    else:
        smooth_loss = action_mean.new_zeros(())
    return {
        "bc_loss": bc_loss + float(action_smooth_loss_weight) * smooth_loss,
        "steer_mse": steer_mse,
        "longitudinal_mse": longitudinal_mse,
        "action_l1": action_l1,
        "bc_sample_weight_mean": (weights * mask).sum() / denom,
        "bc_sample_weight_min": torch.where(mask > 0.0, weights, torch.ones_like(weights)).min(),
        "bc_sample_weight_late_mask_rate": action_mean.new_zeros(()),
        "action_smooth_loss": smooth_loss,
    }


def _compute_visual_zero_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    *,
    visual_zero_margin: float = 0.0,
    visual_zero_steer_margin: float = 0.0,
    visual_zero_route_min: float | None = None,
    visual_zero_route_max: float | None = None,
    visual_zero_abs_lane_offset_min: float = 0.0,
    visual_zero_abs_heading_error_min: float = 0.0,
    visual_zero_corrective_steer_margin: float = 0.0,
    route_progress: torch.Tensor | None = None,
    lane_offset: torch.Tensor | None = None,
    heading_error: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if action_mean.shape != zero_action_mean.shape:
        raise ValueError(
            "action_mean and zero_action_mean must have the same shape, "
            f"got {tuple(action_mean.shape)} and {tuple(zero_action_mean.shape)}"
        )
    delta = (action_mean - zero_action_mean).abs()
    if valid_mask is not None:
        mask = valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
        denom = mask.sum().clamp_min(1.0)
        action_l1 = (delta.sum(dim=-1) * mask).sum() / denom
        steer_delta = (delta[..., 0] * mask).sum() / denom
        longitudinal_delta = (delta[..., 1] * mask).sum() / denom
    else:
        action_l1 = delta.sum(dim=-1).mean()
        steer_delta = delta[..., 0].mean()
        longitudinal_delta = delta[..., 1].mean()
    margin = float(visual_zero_margin)
    margin_loss = F.relu(action_l1.new_tensor(margin) - action_l1) if margin > 0.0 else action_l1.new_zeros(())
    steer_margin = float(visual_zero_steer_margin)
    steer_margin_loss = (
        F.relu(steer_delta.new_tensor(steer_margin) - steer_delta) if steer_margin > 0.0 else steer_delta.new_zeros(())
    )
    corrective_mask = torch.ones_like(delta[..., 0], dtype=action_mean.dtype)
    if valid_mask is not None:
        corrective_mask = corrective_mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if route_progress is not None:
        route_tensor = route_progress.to(device=action_mean.device, dtype=action_mean.dtype)
        if visual_zero_route_min is not None:
            corrective_mask = corrective_mask * (route_tensor >= float(visual_zero_route_min)).to(action_mean.dtype)
        if visual_zero_route_max is not None:
            corrective_mask = corrective_mask * (route_tensor <= float(visual_zero_route_max)).to(action_mean.dtype)
    risk_mask: torch.Tensor | None = None
    lane_tensor: torch.Tensor | None = None
    if lane_offset is not None:
        lane_tensor = lane_offset.to(device=action_mean.device, dtype=action_mean.dtype)
        lane_threshold = float(visual_zero_abs_lane_offset_min)
        if lane_threshold > 0.0:
            risk_mask = lane_tensor.abs() >= lane_threshold
    if heading_error is not None:
        heading_tensor = heading_error.to(device=action_mean.device, dtype=action_mean.dtype)
        heading_threshold = float(visual_zero_abs_heading_error_min)
        if heading_threshold > 0.0:
            heading_mask = heading_tensor.abs() >= heading_threshold
            risk_mask = heading_mask if risk_mask is None else (risk_mask | heading_mask)
    if risk_mask is not None:
        corrective_mask = corrective_mask * risk_mask.to(action_mean.dtype)
    corrective_count = corrective_mask.sum()
    corrective_denom = corrective_count.clamp_min(1.0)
    signed_steer_delta = action_mean[..., 0] - zero_action_mean[..., 0]
    if lane_tensor is not None:
        desired_sign = -torch.sign(lane_tensor)
        signed_corrective_delta = signed_steer_delta * desired_sign
    else:
        signed_corrective_delta = signed_steer_delta.abs()
    corrective_steer_delta = (torch.relu(signed_corrective_delta) * corrective_mask).sum() / corrective_denom
    corrective_margin = float(visual_zero_corrective_steer_margin)
    if corrective_margin > 0.0 and float(corrective_count.detach().cpu().item()) > 0.0:
        corrective_sign_loss = F.relu(corrective_steer_delta.new_tensor(corrective_margin) - corrective_steer_delta)
    else:
        corrective_sign_loss = corrective_steer_delta.new_zeros(())
    corrective_mask_rate = corrective_mask.mean() if corrective_mask.numel() else corrective_steer_delta.new_zeros(())
    return {
        "visual_zero_action_l1": action_l1,
        "visual_zero_steer_delta": steer_delta,
        "visual_zero_longitudinal_delta": longitudinal_delta,
        "visual_zero_margin_loss": margin_loss,
        "visual_zero_steer_margin_loss": steer_margin_loss,
        "visual_zero_corrective_steer_delta": corrective_steer_delta,
        "visual_zero_corrective_sign_loss": corrective_sign_loss,
        "visual_zero_corrective_mask_rate": corrective_mask_rate,
    }


def _zero_corrective_delta_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "target_corrective_delta_loss": zero,
        "target_corrective_delta_l1": zero,
        "target_corrective_delta_sign_rate": zero,
        "visual_zero_target_delta_loss": zero,
        "visual_zero_target_delta_l1": zero,
        "visual_zero_target_delta_sign_rate": zero,
        "target_corrective_delta_mask_rate": zero,
    }


def _compute_corrective_delta_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    valid_mask: torch.Tensor | None = None,
    sign_epsilon: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    policy_steer = batch.get("policy_action_steer")
    target_delta = batch.get("target_corrective_delta")
    available = batch.get("target_corrective_delta_available")
    if policy_steer is None or target_delta is None or available is None:
        return _zero_corrective_delta_losses(action_mean)
    policy_steer = policy_steer.to(device=action_mean.device, dtype=action_mean.dtype)
    target_delta = target_delta.to(device=action_mean.device, dtype=action_mean.dtype)
    mask = available.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if tuple(policy_steer.shape) != tuple(action_mean[..., 0].shape):
        policy_steer = policy_steer.reshape_as(action_mean[..., 0])
    if tuple(target_delta.shape) != tuple(action_mean[..., 0].shape):
        target_delta = target_delta.reshape_as(action_mean[..., 0])
    if tuple(mask.shape) != tuple(action_mean[..., 0].shape):
        mask = mask.reshape_as(action_mean[..., 0])
    denom = mask.sum().clamp_min(1.0)
    predicted_delta = action_mean[..., 0] - policy_steer
    visual_delta = action_mean[..., 0] - zero_action_mean[..., 0]
    target_loss = ((predicted_delta - target_delta).pow(2) * mask).sum() / denom
    target_l1 = ((predicted_delta - target_delta).abs() * mask).sum() / denom
    visual_loss = ((visual_delta - target_delta).pow(2) * mask).sum() / denom
    visual_l1 = ((visual_delta - target_delta).abs() * mask).sum() / denom
    sign_mask = mask * (target_delta.abs() > float(sign_epsilon)).to(action_mean.dtype)
    sign_denom = sign_mask.sum().clamp_min(1.0)
    target_sign_rate = ((predicted_delta * target_delta > 0.0).to(action_mean.dtype) * sign_mask).sum() / sign_denom
    visual_sign_rate = ((visual_delta * target_delta > 0.0).to(action_mean.dtype) * sign_mask).sum() / sign_denom
    return {
        "target_corrective_delta_loss": target_loss,
        "target_corrective_delta_l1": target_l1,
        "target_corrective_delta_sign_rate": target_sign_rate,
        "visual_zero_target_delta_loss": visual_loss,
        "visual_zero_target_delta_l1": visual_l1,
        "visual_zero_target_delta_sign_rate": visual_sign_rate,
        "target_corrective_delta_mask_rate": mask.mean() if mask.numel() else target_loss.new_zeros(()),
    }


def _zero_visual_center_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "visual_center_corrective_sign_loss": zero,
        "visual_center_corrective_sign_rate": zero,
        "visual_center_correction_rate": zero,
        "visual_center_corrective_mask_rate": zero,
        "visual_center_error_abs_mean": zero,
        "visual_center_signed_delta_mean": zero,
    }


def _compute_visual_center_corrective_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    *,
    visual_center_error_min: float = 0.0,
    visual_center_corrective_margin: float = 0.0,
    valid_mask: torch.Tensor | None = None,
    sign_epsilon: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    raw_features = outputs.get("rgb_perception_actor_raw_features_sequence")
    if not isinstance(raw_features, torch.Tensor):
        raw_features = outputs.get("rgb_perception_actor_raw_features")
    target_delta = batch.get("target_corrective_delta")
    available = batch.get("target_corrective_delta_available")
    if not isinstance(raw_features, torch.Tensor) or target_delta is None or available is None:
        return _zero_visual_center_losses(action_mean)
    if int(raw_features.shape[-1]) <= 11:
        return _zero_visual_center_losses(action_mean)
    raw_features = raw_features.to(device=action_mean.device, dtype=action_mean.dtype)
    target_delta = target_delta.to(device=action_mean.device, dtype=action_mean.dtype)
    mask = available.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    action_steer = action_mean[..., 0]
    if tuple(raw_features.shape[:-1]) != tuple(action_steer.shape):
        raw_features = raw_features.reshape(*action_steer.shape, int(raw_features.shape[-1]))
    if tuple(target_delta.shape) != tuple(action_steer.shape):
        target_delta = target_delta.reshape_as(action_steer)
    if tuple(mask.shape) != tuple(action_steer.shape):
        mask = mask.reshape_as(action_steer)

    road_x = raw_features[..., 10]
    lane_x = raw_features[..., 11]
    visual_center_error = lane_x - road_x
    target_sign = torch.sign(target_delta)
    visual_delta = action_steer - zero_action_mean[..., 0]
    signed_visual_delta = visual_delta * target_sign
    center_consistent = (-visual_center_error * target_delta) > 0.0
    sign_active = target_delta.abs() > float(sign_epsilon)
    center_active = visual_center_error.abs() >= float(visual_center_error_min)
    pre_consistency_mask = mask * sign_active.to(action_mean.dtype) * center_active.to(action_mean.dtype)
    pre_consistency_denom = pre_consistency_mask.sum().clamp_min(1.0)
    target_consistency_rate = (
        center_consistent.to(action_mean.dtype) * pre_consistency_mask
    ).sum() / pre_consistency_denom
    corrective_mask = pre_consistency_mask * center_consistent.to(action_mean.dtype)
    denom = corrective_mask.sum().clamp_min(1.0)
    margin = float(visual_center_corrective_margin)
    if margin > 0.0:
        sign_loss_values = F.relu(signed_visual_delta.new_tensor(margin) - signed_visual_delta)
        sign_loss = (sign_loss_values * corrective_mask).sum() / denom
    else:
        sign_loss = signed_visual_delta.new_zeros(())
    sign_rate = ((signed_visual_delta > 0.0).to(action_mean.dtype) * corrective_mask).sum() / denom
    correction_rate = ((visual_delta * visual_center_error < 0.0).to(action_mean.dtype) * corrective_mask).sum() / denom
    error_abs_mean = (visual_center_error.abs() * corrective_mask).sum() / denom
    signed_delta_mean = (signed_visual_delta * corrective_mask).sum() / denom
    return {
        "visual_center_corrective_sign_loss": sign_loss,
        "visual_center_corrective_sign_rate": sign_rate,
        "visual_center_correction_rate": correction_rate,
        "visual_center_corrective_mask_rate": corrective_mask.mean() if corrective_mask.numel() else sign_loss.new_zeros(()),
        "visual_center_target_consistency_rate": target_consistency_rate,
        "visual_center_target_consistency_mask_rate": (
            pre_consistency_mask.mean() if pre_consistency_mask.numel() else sign_loss.new_zeros(())
        ),
        "visual_center_error_abs_mean": error_abs_mean,
        "visual_center_signed_delta_mean": signed_delta_mean,
    }


def _zero_route_aware_visual_center_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "route_aware_visual_center_loss": zero,
        "route_aware_visual_center_sign_rate": zero,
        "route_aware_visual_center_correction_rate": zero,
        "route_aware_visual_center_mask_rate": zero,
        "route_aware_visual_center_target_consistency_rate": zero,
        "route_aware_visual_center_target_consistency_mask_rate": zero,
        "route_aware_visual_center_signed_delta_mean": zero,
    }


def _compute_route_aware_visual_center_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    *,
    route_min: float | None = None,
    route_max: float | None = None,
    visual_center_error_min: float = 0.0,
    margin: float = 0.0,
    valid_mask: torch.Tensor | None = None,
    sign_epsilon: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    raw_features = outputs.get("rgb_perception_actor_raw_features_sequence")
    if not isinstance(raw_features, torch.Tensor):
        raw_features = outputs.get("rgb_perception_actor_raw_features")
    target_delta = batch.get("target_corrective_delta")
    available = batch.get("target_corrective_delta_available")
    route_progress = batch.get("route_progress")
    if not isinstance(raw_features, torch.Tensor) or target_delta is None or available is None or route_progress is None:
        return _zero_route_aware_visual_center_losses(action_mean)
    if int(raw_features.shape[-1]) <= 11:
        return _zero_route_aware_visual_center_losses(action_mean)
    raw_features = raw_features.to(device=action_mean.device, dtype=action_mean.dtype)
    target_delta = target_delta.to(device=action_mean.device, dtype=action_mean.dtype)
    route_tensor = route_progress.to(device=action_mean.device, dtype=action_mean.dtype)
    mask = available.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    action_steer = action_mean[..., 0]
    if tuple(raw_features.shape[:-1]) != tuple(action_steer.shape):
        raw_features = raw_features.reshape(*action_steer.shape, int(raw_features.shape[-1]))
    if tuple(target_delta.shape) != tuple(action_steer.shape):
        target_delta = target_delta.reshape_as(action_steer)
    if tuple(route_tensor.shape) != tuple(action_steer.shape):
        route_tensor = route_tensor.reshape_as(action_steer)
    if tuple(mask.shape) != tuple(action_steer.shape):
        mask = mask.reshape_as(action_steer)

    if route_min is not None:
        mask = mask * (route_tensor >= float(route_min)).to(action_mean.dtype)
    if route_max is not None:
        mask = mask * (route_tensor <= float(route_max)).to(action_mean.dtype)
    road_x = raw_features[..., 10]
    lane_x = raw_features[..., 11]
    visual_center_error = lane_x - road_x
    visual_delta = action_steer - zero_action_mean[..., 0]
    target_sign = torch.sign(target_delta)
    signed_visual_delta = visual_delta * target_sign
    center_consistent = (-visual_center_error * target_delta) > 0.0
    sign_active = target_delta.abs() > float(sign_epsilon)
    center_active = visual_center_error.abs() >= float(visual_center_error_min)
    pre_consistency_mask = mask * sign_active.to(action_mean.dtype) * center_active.to(action_mean.dtype)
    pre_consistency_denom = pre_consistency_mask.sum().clamp_min(1.0)
    target_consistency_rate = (
        center_consistent.to(action_mean.dtype) * pre_consistency_mask
    ).sum() / pre_consistency_denom
    route_mask = pre_consistency_mask * center_consistent.to(action_mean.dtype)
    denom = route_mask.sum().clamp_min(1.0)
    if float(margin) > 0.0:
        loss_values = F.relu(signed_visual_delta.new_tensor(float(margin)) - signed_visual_delta)
        loss = (loss_values * route_mask).sum() / denom
    else:
        loss = signed_visual_delta.new_zeros(())
    sign_rate = ((signed_visual_delta > 0.0).to(action_mean.dtype) * route_mask).sum() / denom
    correction_rate = ((visual_delta * visual_center_error < 0.0).to(action_mean.dtype) * route_mask).sum() / denom
    signed_delta_mean = (signed_visual_delta * route_mask).sum() / denom
    return {
        "route_aware_visual_center_loss": loss,
        "route_aware_visual_center_sign_rate": sign_rate,
        "route_aware_visual_center_correction_rate": correction_rate,
        "route_aware_visual_center_mask_rate": route_mask.mean() if route_mask.numel() else loss.new_zeros(()),
        "route_aware_visual_center_target_consistency_rate": target_consistency_rate,
        "route_aware_visual_center_target_consistency_mask_rate": (
            pre_consistency_mask.mean() if pre_consistency_mask.numel() else loss.new_zeros(())
        ),
        "route_aware_visual_center_signed_delta_mean": signed_delta_mean,
    }


def _zero_collision_cluster_route_offset_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "collision_cluster_route_offset_agreement_loss": zero,
        "collision_cluster_route_offset_agreement_sign_rate": zero,
        "collision_cluster_route_offset_agreement_mask_rate": zero,
        "collision_cluster_route_offset_agreement_signed_delta_mean": zero,
        "collision_cluster_route_offset_agreement_corrected_delta_mean": zero,
        "collision_cluster_route_offset_agreement_positive_sign_rate": zero,
        "collision_cluster_route_offset_agreement_negative_sign_rate": zero,
        "collision_cluster_route_offset_agreement_positive_mask_rate": zero,
        "collision_cluster_route_offset_agreement_negative_mask_rate": zero,
    }


_ROUTE_OFFSET_SOURCE_PREFIX = "collision_cluster_route_offset_agreement"
_EARLY_ROUTE_OFFSET_PREFIX = "early_route_offset_agreement"


def _rename_route_offset_losses(
    losses: dict[str, torch.Tensor],
    *,
    target_prefix: str,
) -> dict[str, torch.Tensor]:
    renamed: dict[str, torch.Tensor] = {}
    for key, value in losses.items():
        if key.startswith(_ROUTE_OFFSET_SOURCE_PREFIX):
            suffix = key[len(_ROUTE_OFFSET_SOURCE_PREFIX) :]
            renamed[f"{target_prefix}{suffix}"] = value
    return renamed


def _compute_collision_cluster_route_offset_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    lane_offset_min: float,
    heading_error_max: float,
    margin: float,
    lane_offset_mode: str = "positive",
    heading_error_mode: str = "less_than",
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if route_progress is None or lane_offset is None or heading_error is None:
        return _zero_collision_cluster_route_offset_losses(action_mean)
    action_steer = action_mean[..., 0]
    zero_steer = zero_action_mean[..., 0]
    route_tensor = route_progress.to(device=action_mean.device, dtype=action_mean.dtype)
    lane_tensor = lane_offset.to(device=action_mean.device, dtype=action_mean.dtype)
    heading_tensor = heading_error.to(device=action_mean.device, dtype=action_mean.dtype)
    if tuple(route_tensor.shape) != tuple(action_steer.shape):
        route_tensor = route_tensor.reshape_as(action_steer)
    if tuple(lane_tensor.shape) != tuple(action_steer.shape):
        lane_tensor = lane_tensor.reshape_as(action_steer)
    if tuple(heading_tensor.shape) != tuple(action_steer.shape):
        heading_tensor = heading_tensor.reshape_as(action_steer)

    mask = torch.ones_like(action_steer, dtype=action_mean.dtype)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if route_min is not None:
        mask = mask * (route_tensor >= float(route_min)).to(action_mean.dtype)
    if route_max is not None:
        mask = mask * (route_tensor <= float(route_max)).to(action_mean.dtype)

    lane_mode = str(lane_offset_mode).strip().lower()
    if lane_mode in {"positive", "pos", "right"}:
        lane_active = lane_tensor > float(lane_offset_min)
    elif lane_mode in {"negative", "neg", "left"}:
        lane_active = lane_tensor < -float(lane_offset_min)
    elif lane_mode in {"abs", "absolute", "symmetric", "both"}:
        lane_active = lane_tensor.abs() > float(lane_offset_min)
    else:
        raise ValueError(f"Unsupported collision_cluster_lane_offset_mode: {lane_offset_mode!r}")

    heading_mode = str(heading_error_mode).strip().lower()
    if heading_mode in {"less_than", "negative", "neg"}:
        heading_active = heading_tensor < float(heading_error_max)
    elif heading_mode in {"greater_than", "positive", "pos"}:
        heading_active = heading_tensor > abs(float(heading_error_max))
    elif heading_mode in {"abs", "absolute", "symmetric", "both"}:
        heading_active = heading_tensor.abs() > abs(float(heading_error_max))
    else:
        raise ValueError(f"Unsupported collision_cluster_heading_error_mode: {heading_error_mode!r}")

    mask = mask * lane_active.to(action_mean.dtype)
    mask = mask * heading_active.to(action_mean.dtype)

    signed_steer_delta = action_steer - zero_steer
    desired_sign = -torch.sign(lane_tensor).clamp(min=-1.0, max=1.0)
    signed_corrective_delta = signed_steer_delta * desired_sign
    denom = mask.sum().clamp_min(1.0)
    loss = (F.relu(signed_corrective_delta.new_tensor(float(margin)) - signed_corrective_delta) * mask).sum() / denom
    sign_rate = ((signed_corrective_delta > 0.0).to(action_mean.dtype) * mask).sum() / denom
    signed_delta_mean = (signed_steer_delta * mask).sum() / denom
    corrected_delta_mean = (signed_corrective_delta * mask).sum() / denom
    positive_mask = mask * (lane_tensor > 0.0).to(action_mean.dtype)
    negative_mask = mask * (lane_tensor < 0.0).to(action_mean.dtype)
    positive_denom = positive_mask.sum().clamp_min(1.0)
    negative_denom = negative_mask.sum().clamp_min(1.0)
    positive_sign_rate = ((signed_corrective_delta > 0.0).to(action_mean.dtype) * positive_mask).sum() / positive_denom
    negative_sign_rate = ((signed_corrective_delta > 0.0).to(action_mean.dtype) * negative_mask).sum() / negative_denom
    return {
        "collision_cluster_route_offset_agreement_loss": loss,
        "collision_cluster_route_offset_agreement_sign_rate": sign_rate,
        "collision_cluster_route_offset_agreement_mask_rate": mask.mean() if mask.numel() else loss.new_zeros(()),
        "collision_cluster_route_offset_agreement_signed_delta_mean": signed_delta_mean,
        "collision_cluster_route_offset_agreement_corrected_delta_mean": corrected_delta_mean,
        "collision_cluster_route_offset_agreement_positive_sign_rate": positive_sign_rate,
        "collision_cluster_route_offset_agreement_negative_sign_rate": negative_sign_rate,
        "collision_cluster_route_offset_agreement_positive_mask_rate": (
            positive_mask.mean() if positive_mask.numel() else loss.new_zeros(())
        ),
        "collision_cluster_route_offset_agreement_negative_mask_rate": (
            negative_mask.mean() if negative_mask.numel() else loss.new_zeros(())
        ),
    }


def _compute_early_route_offset_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    lane_offset_min: float,
    heading_error_min: float,
    margin: float,
    lane_offset_mode: str = "negative",
    heading_error_mode: str = "positive",
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    losses = _compute_collision_cluster_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
        route_min=route_min,
        route_max=route_max,
        lane_offset_min=lane_offset_min,
        heading_error_max=heading_error_min,
        margin=margin,
        lane_offset_mode=lane_offset_mode,
        heading_error_mode=heading_error_mode,
        valid_mask=valid_mask,
    )
    return _rename_route_offset_losses(losses, target_prefix=_EARLY_ROUTE_OFFSET_PREFIX)


def _zero_early_severe_steer_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "early_severe_steer_delta_loss": zero,
        "early_severe_steer_delta_l1": zero,
        "early_severe_steer_sign_loss": zero,
        "early_severe_steer_sign_rate": zero,
        "early_severe_steer_wrong_sign_rate": zero,
        "early_severe_absolute_steer_loss": zero,
        "early_severe_absolute_steer_sign_rate": zero,
        "early_severe_probe_absolute_steer_loss": zero,
        "early_severe_probe_absolute_steer_sign_rate": zero,
        "early_severe_probe_absolute_steer_mask_rate": zero,
        "early_severe_longitudinal_ceiling_loss": zero,
        "early_severe_longitudinal_ceiling_rate": zero,
        "late_recovery_absolute_steer_loss": zero,
        "late_recovery_absolute_steer_sign_rate": zero,
        "late_recovery_absolute_steer_mask_rate": zero,
        "early_center_zero_steer_loss": zero,
        "early_center_zero_steer_abs_mean": zero,
        "early_center_zero_steer_mask_rate": zero,
        "early_severe_steer_mask_rate": zero,
        "early_severe_steer_left_sign_rate": zero,
        "early_severe_steer_right_sign_rate": zero,
        "early_severe_steer_left_mask_rate": zero,
        "early_severe_steer_right_mask_rate": zero,
        "early_severe_steer_signed_delta_mean": zero,
        "early_severe_steer_corrective_delta_mean": zero,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _side_balanced_mean(values: torch.Tensor, mask: torch.Tensor, lane_tensor: torch.Tensor) -> torch.Tensor:
    left_mask = mask * (lane_tensor < 0.0).to(values.dtype)
    right_mask = mask * (lane_tensor > 0.0).to(values.dtype)
    parts: list[torch.Tensor] = []
    if float(left_mask.detach().sum().cpu().item()) > 0.0:
        parts.append(_masked_mean(values, left_mask))
    if float(right_mask.detach().sum().cpu().item()) > 0.0:
        parts.append(_masked_mean(values, right_mask))
    if not parts:
        return values.sum() * 0.0
    return torch.stack(parts).mean()


def _compute_early_severe_steer_losses(
    action_mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    margin: float,
    absolute_steer_margin: float = 0.0,
    side_balanced: bool = True,
    alignment_mode: str = "lane",
    valid_mask: torch.Tensor | None = None,
    sign_epsilon: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    policy_steer = batch.get("policy_action_steer")
    target_delta = batch.get("target_corrective_delta")
    available = batch.get("target_corrective_delta_available")
    if (
        policy_steer is None
        or target_delta is None
        or available is None
        or route_progress is None
        or lane_offset is None
        or heading_error is None
    ):
        return _zero_early_severe_steer_losses(action_mean)

    action_steer = action_mean[..., 0]
    policy_tensor = policy_steer.to(device=action_mean.device, dtype=action_mean.dtype)
    target_tensor = target_delta.to(device=action_mean.device, dtype=action_mean.dtype)
    route_tensor = route_progress.to(device=action_mean.device, dtype=action_mean.dtype)
    lane_tensor = lane_offset.to(device=action_mean.device, dtype=action_mean.dtype)
    heading_tensor = heading_error.to(device=action_mean.device, dtype=action_mean.dtype)
    mask = available.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)

    for name, tensor in (
        ("policy_action_steer", policy_tensor),
        ("target_corrective_delta", target_tensor),
        ("route_progress", route_tensor),
        ("lane_offset", lane_tensor),
        ("heading_error", heading_tensor),
        ("target_corrective_delta_available", mask),
    ):
        if tuple(tensor.shape) != tuple(action_steer.shape):
            if name == "policy_action_steer":
                policy_tensor = tensor.reshape_as(action_steer)
            elif name == "target_corrective_delta":
                target_tensor = tensor.reshape_as(action_steer)
            elif name == "route_progress":
                route_tensor = tensor.reshape_as(action_steer)
            elif name == "lane_offset":
                lane_tensor = tensor.reshape_as(action_steer)
            elif name == "heading_error":
                heading_tensor = tensor.reshape_as(action_steer)
            else:
                mask = tensor.reshape_as(action_steer)

    if route_min is not None:
        mask = mask * (route_tensor >= float(route_min)).to(action_mean.dtype)
    if route_max is not None:
        mask = mask * (route_tensor <= float(route_max)).to(action_mean.dtype)
    lane_active = lane_tensor.abs() >= float(abs_lane_offset_min)
    heading_active = heading_tensor.abs() >= float(abs_heading_error_min)
    heading_dominant = (~lane_active) & heading_active
    sign_active = target_tensor.abs() > float(sign_epsilon)
    if str(alignment_mode) == "semantic":
        lane_aligned = (target_tensor * (-torch.sign(lane_tensor))) > 0.0
        heading_aligned = (target_tensor * torch.sign(heading_tensor)) > 0.0
        correction_aligned = (lane_active & lane_aligned) | (heading_dominant & heading_aligned)
        severe_active = lane_active | heading_dominant
    elif str(alignment_mode) == "lane":
        correction_aligned = (target_tensor * (-lane_tensor)) > 0.0
        severe_active = lane_active | heading_active
    else:
        raise ValueError("early severe alignment_mode must be one of: lane, semantic")
    mask = (
        mask
        * severe_active.to(action_mean.dtype)
        * sign_active.to(action_mean.dtype)
        * correction_aligned.to(action_mean.dtype)
    )

    predicted_delta = action_steer - policy_tensor
    delta_error = predicted_delta - target_tensor
    delta_values = F.smooth_l1_loss(predicted_delta, target_tensor, reduction="none")
    l1_values = delta_error.abs()
    signed_delta = predicted_delta * torch.sign(target_tensor)
    margin_values = F.relu(signed_delta.new_tensor(float(margin)) - signed_delta)
    signed_absolute_steer = action_steer * torch.sign(target_tensor)
    absolute_margin_values = F.relu(
        signed_absolute_steer.new_tensor(float(absolute_steer_margin)) - signed_absolute_steer
    )
    if bool(side_balanced):
        delta_loss = _side_balanced_mean(delta_values, mask, lane_tensor)
        l1_loss = _side_balanced_mean(l1_values, mask, lane_tensor)
        sign_loss = _side_balanced_mean(margin_values, mask, lane_tensor)
        absolute_steer_loss = _side_balanced_mean(absolute_margin_values, mask, lane_tensor)
    else:
        delta_loss = _masked_mean(delta_values, mask)
        l1_loss = _masked_mean(l1_values, mask)
        sign_loss = _masked_mean(margin_values, mask)
        absolute_steer_loss = _masked_mean(absolute_margin_values, mask)

    sign_rate = _masked_mean((signed_delta > 0.0).to(action_mean.dtype), mask)
    wrong_sign_rate = _masked_mean((signed_delta <= 0.0).to(action_mean.dtype), mask)
    absolute_steer_sign_rate = _masked_mean((signed_absolute_steer >= 0.0).to(action_mean.dtype), mask)
    left_mask = mask * (lane_tensor < 0.0).to(action_mean.dtype)
    right_mask = mask * (lane_tensor > 0.0).to(action_mean.dtype)
    left_rate = _masked_mean((signed_delta > 0.0).to(action_mean.dtype), left_mask)
    right_rate = _masked_mean((signed_delta > 0.0).to(action_mean.dtype), right_mask)
    signed_delta_mean = _masked_mean(signed_delta, mask)
    corrective_delta_mean = _masked_mean(predicted_delta, mask)

    return {
        "early_severe_steer_delta_loss": delta_loss,
        "early_severe_steer_delta_l1": l1_loss,
        "early_severe_steer_sign_loss": sign_loss,
        "early_severe_steer_sign_rate": sign_rate,
        "early_severe_steer_wrong_sign_rate": wrong_sign_rate,
        "early_severe_absolute_steer_loss": absolute_steer_loss,
        "early_severe_absolute_steer_sign_rate": absolute_steer_sign_rate,
        "early_severe_steer_mask_rate": mask.mean() if mask.numel() else delta_loss.new_zeros(()),
        "early_severe_steer_left_sign_rate": left_rate,
        "early_severe_steer_right_sign_rate": right_rate,
        "early_severe_steer_left_mask_rate": left_mask.mean() if left_mask.numel() else delta_loss.new_zeros(()),
        "early_severe_steer_right_mask_rate": right_mask.mean() if right_mask.numel() else delta_loss.new_zeros(()),
        "early_severe_steer_signed_delta_mean": signed_delta_mean,
        "early_severe_steer_corrective_delta_mean": corrective_delta_mean,
    }


def _zero_early_severe_probe_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "early_severe_probe_absolute_steer_loss": zero,
        "early_severe_probe_absolute_steer_sign_rate": zero,
        "early_severe_probe_absolute_steer_mask_rate": zero,
    }


def _early_severe_recovery_mask(
    *,
    action_reference: torch.Tensor,
    batch: dict[str, torch.Tensor],
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    alignment_mode: str,
    valid_mask: torch.Tensor | None = None,
    sign_epsilon: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    target_delta = batch.get("target_corrective_delta")
    available = batch.get("target_corrective_delta_available")
    if target_delta is None or available is None or route_progress is None or lane_offset is None or heading_error is None:
        return None
    action_steer = action_reference[..., 0]
    target_tensor = target_delta.to(device=action_reference.device, dtype=action_reference.dtype)
    route_tensor = route_progress.to(device=action_reference.device, dtype=action_reference.dtype)
    lane_tensor = lane_offset.to(device=action_reference.device, dtype=action_reference.dtype)
    heading_tensor = heading_error.to(device=action_reference.device, dtype=action_reference.dtype)
    mask = available.to(device=action_reference.device, dtype=action_reference.dtype).clamp(0.0, 1.0)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_reference.device, dtype=action_reference.dtype).clamp(0.0, 1.0)
    for name, tensor in (
        ("target_corrective_delta", target_tensor),
        ("route_progress", route_tensor),
        ("lane_offset", lane_tensor),
        ("heading_error", heading_tensor),
        ("target_corrective_delta_available", mask),
    ):
        if tuple(tensor.shape) != tuple(action_steer.shape):
            if name == "target_corrective_delta":
                target_tensor = tensor.reshape_as(action_steer)
            elif name == "route_progress":
                route_tensor = tensor.reshape_as(action_steer)
            elif name == "lane_offset":
                lane_tensor = tensor.reshape_as(action_steer)
            elif name == "heading_error":
                heading_tensor = tensor.reshape_as(action_steer)
            else:
                mask = tensor.reshape_as(action_steer)

    if route_min is not None:
        mask = mask * (route_tensor >= float(route_min)).to(action_reference.dtype)
    if route_max is not None:
        mask = mask * (route_tensor <= float(route_max)).to(action_reference.dtype)
    lane_active = lane_tensor.abs() >= float(abs_lane_offset_min)
    heading_active = heading_tensor.abs() >= float(abs_heading_error_min)
    heading_dominant = (~lane_active) & heading_active
    sign_active = target_tensor.abs() > float(sign_epsilon)
    if str(alignment_mode) == "semantic":
        lane_aligned = (target_tensor * (-torch.sign(lane_tensor))) > 0.0
        heading_aligned = (target_tensor * torch.sign(heading_tensor)) > 0.0
        correction_aligned = (lane_active & lane_aligned) | (heading_dominant & heading_aligned)
        severe_active = lane_active | heading_dominant
    elif str(alignment_mode) == "lane":
        correction_aligned = (target_tensor * (-lane_tensor)) > 0.0
        severe_active = lane_active | heading_active
    else:
        raise ValueError("early severe alignment_mode must be one of: lane, semantic")
    mask = (
        mask
        * severe_active.to(action_reference.dtype)
        * sign_active.to(action_reference.dtype)
        * correction_aligned.to(action_reference.dtype)
    )
    return mask, target_tensor, lane_tensor


def _risk_field_index(model: RCSACFirstModel, field: str) -> int:
    risk = model.aux_spec.item("risk_stats")
    return int(risk.start + RISK_STATS_FIELDS.index(field))


def _compute_early_severe_probe_absolute_steer_losses(
    *,
    model: RCSACFirstModel,
    batch: dict[str, torch.Tensor],
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    margin: float,
    axes: tuple[str, ...],
    side_balanced: bool = True,
    alignment_mode: str = "lane",
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    obs = batch.get("obs")
    auxiliary = batch.get("auxiliary")
    if not isinstance(obs, torch.Tensor) or not isinstance(auxiliary, torch.Tensor) or obs.ndim != 2:
        reference = obs if isinstance(obs, torch.Tensor) else auxiliary
        if isinstance(reference, torch.Tensor):
            return _zero_early_severe_probe_losses(reference)
        raise ValueError("Batch must include obs or auxiliary tensors.")

    probe_losses: list[torch.Tensor] = []
    probe_rates: list[torch.Tensor] = []
    probe_mask_rates: list[torch.Tensor] = []
    route_start = int(model.obs_spec.modal("route").start)
    for axis in axes:
        probe_obs = obs.clone()
        probe_auxiliary = auxiliary.clone()
        if axis == "lane_offset":
            probe_auxiliary[:, _risk_field_index(model, "lane_offset")] = 0.0
        elif axis == "heading_error":
            probe_auxiliary[:, _risk_field_index(model, "heading_error")] = 0.0
        elif axis == "route_target_y":
            if int(model.config.route_dim) <= 1:
                continue
            probe_obs[:, route_start + 1] = 0.0
        else:
            raise ValueError(f"Unsupported early severe probe axis: {axis!r}")
        outputs = model.forward_policy_outputs(probe_obs, auxiliary=probe_auxiliary, track_diagnostics=False)
        action_mean = outputs.get("action_mean")
        if not isinstance(action_mean, torch.Tensor):
            raise RuntimeError("Model output action_mean is missing.")
        mask_result = _early_severe_recovery_mask(
            action_reference=action_mean,
            batch=batch,
            route_progress=route_progress,
            lane_offset=lane_offset,
            heading_error=heading_error,
            route_min=route_min,
            route_max=route_max,
            abs_lane_offset_min=abs_lane_offset_min,
            abs_heading_error_min=abs_heading_error_min,
            alignment_mode=alignment_mode,
            valid_mask=valid_mask,
        )
        if mask_result is None:
            return _zero_early_severe_probe_losses(action_mean)
        mask, target_tensor, lane_tensor = mask_result
        signed_absolute_steer = action_mean[..., 0] * torch.sign(target_tensor)
        loss_values = F.relu(signed_absolute_steer.new_tensor(float(margin)) - signed_absolute_steer)
        if bool(side_balanced):
            loss = _side_balanced_mean(loss_values, mask, lane_tensor)
        else:
            loss = _masked_mean(loss_values, mask)
        probe_losses.append(loss)
        probe_rates.append(_masked_mean((signed_absolute_steer >= 0.0).to(action_mean.dtype), mask))
        probe_mask_rates.append(mask.mean() if mask.numel() else loss.new_zeros(()))

    if not probe_losses:
        return _zero_early_severe_probe_losses(obs)
    return {
        "early_severe_probe_absolute_steer_loss": torch.stack(probe_losses).mean(),
        "early_severe_probe_absolute_steer_sign_rate": torch.stack(probe_rates).mean(),
        "early_severe_probe_absolute_steer_mask_rate": torch.stack(probe_mask_rates).mean(),
    }


def _zero_early_severe_longitudinal_ceiling_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "early_severe_longitudinal_ceiling_loss": zero,
        "early_severe_longitudinal_ceiling_rate": zero,
    }


def _compute_early_severe_longitudinal_ceiling_losses(
    action_mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    longitudinal_ceiling: float,
    side_balanced: bool = True,
    alignment_mode: str = "lane",
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if int(action_mean.shape[-1]) <= 1:
        return _zero_early_severe_longitudinal_ceiling_losses(action_mean)
    mask_result = _early_severe_recovery_mask(
        action_reference=action_mean,
        batch=batch,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
        route_min=route_min,
        route_max=route_max,
        abs_lane_offset_min=abs_lane_offset_min,
        abs_heading_error_min=abs_heading_error_min,
        alignment_mode=alignment_mode,
        valid_mask=valid_mask,
    )
    if mask_result is None:
        return _zero_early_severe_longitudinal_ceiling_losses(action_mean)
    mask, _, lane_tensor = mask_result
    longitudinal = action_mean[..., 1]
    ceiling = longitudinal.new_tensor(float(longitudinal_ceiling))
    loss_values = F.relu(longitudinal - ceiling)
    if bool(side_balanced):
        loss = _side_balanced_mean(loss_values, mask, lane_tensor)
    else:
        loss = _masked_mean(loss_values, mask)
    ceiling_rate = _masked_mean((longitudinal <= ceiling).to(action_mean.dtype), mask)
    return {
        "early_severe_longitudinal_ceiling_loss": loss,
        "early_severe_longitudinal_ceiling_rate": ceiling_rate,
    }


def _zero_late_recovery_absolute_steer_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "late_recovery_absolute_steer_loss": zero,
        "late_recovery_absolute_steer_sign_rate": zero,
        "late_recovery_absolute_steer_mask_rate": zero,
    }


def _compute_late_recovery_absolute_steer_losses(
    action_mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    absolute_steer_margin: float,
    side_balanced: bool = True,
    alignment_mode: str = "semantic",
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mask_result = _early_severe_recovery_mask(
        action_reference=action_mean,
        batch=batch,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
        route_min=route_min,
        route_max=route_max,
        abs_lane_offset_min=abs_lane_offset_min,
        abs_heading_error_min=abs_heading_error_min,
        alignment_mode=alignment_mode,
        valid_mask=valid_mask,
    )
    if mask_result is None:
        return _zero_late_recovery_absolute_steer_losses(action_mean)
    mask, target_tensor, lane_tensor = mask_result
    signed_absolute_steer = action_mean[..., 0] * torch.sign(target_tensor)
    loss_values = F.relu(signed_absolute_steer.new_tensor(float(absolute_steer_margin)) - signed_absolute_steer)
    if bool(side_balanced):
        loss = _side_balanced_mean(loss_values, mask, lane_tensor)
    else:
        loss = _masked_mean(loss_values, mask)
    sign_rate = _masked_mean((signed_absolute_steer >= 0.0).to(action_mean.dtype), mask)
    return {
        "late_recovery_absolute_steer_loss": loss,
        "late_recovery_absolute_steer_sign_rate": sign_rate,
        "late_recovery_absolute_steer_mask_rate": mask.mean() if mask.numel() else loss.new_zeros(()),
    }


def _late_recovery_bc_sample_weight(
    action_mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    scale: float,
    alignment_mode: str,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    zero = action_mean.sum() * 0.0
    if float(scale) >= 0.999:
        return None, action_mean.new_ones(()), zero
    mask_result = _early_severe_recovery_mask(
        action_reference=action_mean,
        batch=batch,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
        route_min=route_min,
        route_max=route_max,
        abs_lane_offset_min=abs_lane_offset_min,
        abs_heading_error_min=abs_heading_error_min,
        alignment_mode=alignment_mode,
        valid_mask=valid_mask,
    )
    if mask_result is None:
        return None, action_mean.new_ones(()), zero
    mask, _, _ = mask_result
    clamped_scale = max(0.0, min(1.0, float(scale)))
    weights = torch.ones_like(mask) - mask.clamp(0.0, 1.0) * (1.0 - clamped_scale)
    return weights, weights.mean(), mask.mean() if mask.numel() else zero


def _zero_early_center_zero_steer_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "early_center_zero_steer_loss": zero,
        "early_center_zero_steer_abs_mean": zero,
        "early_center_zero_steer_mask_rate": zero,
    }


def _compute_early_center_zero_steer_losses(
    action_mean: torch.Tensor,
    *,
    route_progress: torch.Tensor | None,
    lane_offset: torch.Tensor | None,
    heading_error: torch.Tensor | None,
    route_min: float | None,
    route_max: float | None,
    abs_lane_offset_max: float,
    abs_heading_error_max: float,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if route_progress is None or lane_offset is None or heading_error is None:
        return _zero_early_center_zero_steer_losses(action_mean)
    action_steer = action_mean[..., 0]
    route_tensor = route_progress.to(device=action_mean.device, dtype=action_mean.dtype)
    lane_tensor = lane_offset.to(device=action_mean.device, dtype=action_mean.dtype)
    heading_tensor = heading_error.to(device=action_mean.device, dtype=action_mean.dtype)
    mask = torch.ones_like(action_steer, dtype=action_mean.dtype)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    for name, tensor in (
        ("route_progress", route_tensor),
        ("lane_offset", lane_tensor),
        ("heading_error", heading_tensor),
        ("mask", mask),
    ):
        if tuple(tensor.shape) != tuple(action_steer.shape):
            if name == "route_progress":
                route_tensor = tensor.reshape_as(action_steer)
            elif name == "lane_offset":
                lane_tensor = tensor.reshape_as(action_steer)
            elif name == "heading_error":
                heading_tensor = tensor.reshape_as(action_steer)
            else:
                mask = tensor.reshape_as(action_steer)
    if route_min is not None:
        mask = mask * (route_tensor >= float(route_min)).to(action_mean.dtype)
    if route_max is not None:
        mask = mask * (route_tensor <= float(route_max)).to(action_mean.dtype)
    mask = mask * (lane_tensor.abs() <= float(abs_lane_offset_max)).to(action_mean.dtype)
    mask = mask * (heading_tensor.abs() <= float(abs_heading_error_max)).to(action_mean.dtype)
    abs_steer = action_steer.abs()
    loss = _masked_mean(abs_steer, mask)
    return {
        "early_center_zero_steer_loss": loss,
        "early_center_zero_steer_abs_mean": loss,
        "early_center_zero_steer_mask_rate": mask.mean() if mask.numel() else loss.new_zeros(()),
    }


def _zero_route_progress_longitudinal_losses(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0
    return {
        "route_progress_longitudinal_floor_loss": zero,
        "route_progress_longitudinal_floor_rate": zero,
        "route_progress_longitudinal_mask_rate": zero,
        "route_progress_longitudinal_mean": zero,
    }


def _auxiliary_speed_mps(batch: dict[str, torch.Tensor], action_shape: tuple[int, ...]) -> torch.Tensor | None:
    auxiliary = batch.get("auxiliary")
    if not isinstance(auxiliary, torch.Tensor) or int(auxiliary.shape[-1]) <= 15:
        return None
    speed = auxiliary[..., 15]
    if tuple(speed.shape) != tuple(action_shape):
        speed = speed.reshape(action_shape)
    return speed


def _compute_route_progress_longitudinal_losses(
    action_mean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    route_min: float | None = None,
    route_max: float | None = None,
    speed_max_mps: float | None = None,
    longitudinal_floor: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    route_progress = batch.get("route_progress")
    if route_progress is None:
        return _zero_route_progress_longitudinal_losses(action_mean)
    longitudinal = action_mean[..., 1]
    route_tensor = route_progress.to(device=action_mean.device, dtype=action_mean.dtype)
    if tuple(route_tensor.shape) != tuple(longitudinal.shape):
        route_tensor = route_tensor.reshape_as(longitudinal)
    mask = torch.ones_like(longitudinal, dtype=action_mean.dtype)
    if valid_mask is not None:
        mask = mask * valid_mask.to(device=action_mean.device, dtype=action_mean.dtype).clamp(0.0, 1.0)
    if route_min is not None:
        mask = mask * (route_tensor >= float(route_min)).to(action_mean.dtype)
    if route_max is not None:
        mask = mask * (route_tensor <= float(route_max)).to(action_mean.dtype)
    if speed_max_mps is not None:
        speed = _auxiliary_speed_mps(batch, tuple(longitudinal.shape))
        if speed is not None:
            speed = speed.to(device=action_mean.device, dtype=action_mean.dtype)
            mask = mask * (speed <= float(speed_max_mps)).to(action_mean.dtype)
    denom = mask.sum().clamp_min(1.0)
    floor = float(longitudinal_floor)
    if floor > -1.0:
        floor_loss_values = F.relu(longitudinal.new_tensor(floor) - longitudinal)
        floor_loss = (floor_loss_values * mask).sum() / denom
    else:
        floor_loss = longitudinal.new_zeros(())
    floor_rate = ((longitudinal >= longitudinal.new_tensor(floor)).to(action_mean.dtype) * mask).sum() / denom
    longitudinal_mean = (longitudinal * mask).sum() / denom
    return {
        "route_progress_longitudinal_floor_loss": floor_loss,
        "route_progress_longitudinal_floor_rate": floor_rate,
        "route_progress_longitudinal_mask_rate": mask.mean() if mask.numel() else floor_loss.new_zeros(()),
        "route_progress_longitudinal_mean": longitudinal_mean,
    }


def _forward_sequence_outputs(
    model: RCSACFirstModel,
    batch: dict[str, torch.Tensor],
    *,
    use_auxiliary: bool,
    track_diagnostics: bool = False,
    zero_rgb_perception_actor_features: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    obs = batch["obs"]
    if obs.ndim != 3:
        raise ValueError(f"sequence obs must have shape [B, L, D], got {tuple(obs.shape)}")
    hidden: torch.Tensor | None = None
    actions: list[torch.Tensor] = []
    road_logits: list[torch.Tensor] = []
    lane_logits: list[torch.Tensor] = []
    raw_features: list[torch.Tensor] = []
    last_outputs: dict[str, torch.Tensor | None] | None = None
    sequence_length = int(obs.shape[1])
    for step in range(sequence_length):
        auxiliary = batch["auxiliary"][:, step] if use_auxiliary else None
        outputs = model.forward_policy_outputs(
            obs[:, step],
            auxiliary=auxiliary,
            hidden_state=hidden,
            track_diagnostics=track_diagnostics and step == sequence_length - 1,
            zero_rgb_perception_actor_features=zero_rgb_perception_actor_features,
        )
        action_mean = outputs["action_mean"]
        if not isinstance(action_mean, torch.Tensor):
            raise RuntimeError("Model output action_mean is missing.")
        actions.append(action_mean)
        pred_road = outputs.get("pred_road_mask_logits")
        pred_lane = outputs.get("pred_lane_edge_mask_logits")
        if isinstance(pred_road, torch.Tensor) and isinstance(pred_lane, torch.Tensor):
            road_logits.append(pred_road)
            lane_logits.append(pred_lane)
        raw = outputs.get("rgb_perception_actor_raw_features")
        if isinstance(raw, torch.Tensor):
            raw_features.append(raw)
        next_hidden = outputs["next_hidden_state"]
        hidden = next_hidden if isinstance(next_hidden, torch.Tensor) else None
        last_outputs = outputs
    if last_outputs is None:
        raise RuntimeError("Empty sequence batch.")
    action_sequence = torch.stack(actions, dim=1)
    last_outputs = dict(last_outputs)
    last_outputs["action_mean_sequence"] = action_sequence
    if len(road_logits) == sequence_length and len(lane_logits) == sequence_length:
        last_outputs["pred_road_mask_logits_sequence"] = torch.stack(road_logits, dim=1)
        last_outputs["pred_lane_edge_mask_logits_sequence"] = torch.stack(lane_logits, dim=1)
    if len(raw_features) == sequence_length:
        last_outputs["rgb_perception_actor_raw_features_sequence"] = torch.stack(raw_features, dim=1)
    return action_sequence, last_outputs


def _metric_row(
    *,
    epoch: int,
    global_step: int,
    split: str,
    losses: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor | None],
    model: RCSACFirstModel,
) -> dict[str, float | int | str]:
    fusion = outputs["fusion_weights"]
    reliability = outputs["reliability"]
    contribution = outputs["contribution"]
    risk_scalar = outputs["risk_scalar"]
    if not isinstance(fusion, torch.Tensor) or not isinstance(reliability, torch.Tensor):
        raise RuntimeError("Model outputs missing fusion/reliability tensors.")
    if not isinstance(contribution, torch.Tensor) or not isinstance(risk_scalar, torch.Tensor):
        raise RuntimeError("Model outputs missing contribution/risk tensors.")
    row: dict[str, float | int | str] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "split": split,
        "bc_loss": float(losses["bc_loss"].detach().cpu().item()),
        "steer_mse": float(losses["steer_mse"].detach().cpu().item()),
        "longitudinal_mse": float(losses["longitudinal_mse"].detach().cpu().item()),
        "action_l1": float(losses["action_l1"].detach().cpu().item()),
        "bc_sample_weight_mean": float(
            losses.get("bc_sample_weight_mean", fusion.new_ones(())).detach().cpu().item()
        ),
        "bc_sample_weight_min": float(
            losses.get("bc_sample_weight_min", fusion.new_ones(())).detach().cpu().item()
        ),
        "bc_sample_weight_late_mask_rate": float(
            losses.get("bc_sample_weight_late_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_action_l1": float(losses.get("visual_zero_action_l1", fusion.new_zeros(())).detach().cpu().item()),
        "visual_zero_steer_delta": float(
            losses.get("visual_zero_steer_delta", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_longitudinal_delta": float(
            losses.get("visual_zero_longitudinal_delta", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_margin_loss": float(
            losses.get("visual_zero_margin_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_steer_margin_loss": float(
            losses.get("visual_zero_steer_margin_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_corrective_steer_delta": float(
            losses.get("visual_zero_corrective_steer_delta", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_corrective_sign_loss": float(
            losses.get("visual_zero_corrective_sign_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_corrective_mask_rate": float(
            losses.get("visual_zero_corrective_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "target_corrective_delta_loss": float(
            losses.get("target_corrective_delta_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "target_corrective_delta_l1": float(
            losses.get("target_corrective_delta_l1", fusion.new_zeros(())).detach().cpu().item()
        ),
        "target_corrective_delta_sign_rate": float(
            losses.get("target_corrective_delta_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_target_delta_loss": float(
            losses.get("visual_zero_target_delta_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_target_delta_l1": float(
            losses.get("visual_zero_target_delta_l1", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_zero_target_delta_sign_rate": float(
            losses.get("visual_zero_target_delta_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "target_corrective_delta_mask_rate": float(
            losses.get("target_corrective_delta_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_corrective_sign_loss": float(
            losses.get("visual_center_corrective_sign_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_corrective_sign_rate": float(
            losses.get("visual_center_corrective_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_correction_rate": float(
            losses.get("visual_center_correction_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_corrective_mask_rate": float(
            losses.get("visual_center_corrective_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_target_consistency_rate": float(
            losses.get("visual_center_target_consistency_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_target_consistency_mask_rate": float(
            losses.get("visual_center_target_consistency_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_error_abs_mean": float(
            losses.get("visual_center_error_abs_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "visual_center_signed_delta_mean": float(
            losses.get("visual_center_signed_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_loss": float(
            losses.get("route_aware_visual_center_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_sign_rate": float(
            losses.get("route_aware_visual_center_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_correction_rate": float(
            losses.get("route_aware_visual_center_correction_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_mask_rate": float(
            losses.get("route_aware_visual_center_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_target_consistency_rate": float(
            losses.get("route_aware_visual_center_target_consistency_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_target_consistency_mask_rate": float(
            losses.get("route_aware_visual_center_target_consistency_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_aware_visual_center_signed_delta_mean": float(
            losses.get("route_aware_visual_center_signed_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_loss": float(
            losses.get("collision_cluster_route_offset_agreement_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_sign_rate": float(
            losses.get("collision_cluster_route_offset_agreement_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_mask_rate": float(
            losses.get("collision_cluster_route_offset_agreement_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_signed_delta_mean": float(
            losses.get("collision_cluster_route_offset_agreement_signed_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_corrected_delta_mean": float(
            losses.get("collision_cluster_route_offset_agreement_corrected_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_positive_sign_rate": float(
            losses.get("collision_cluster_route_offset_agreement_positive_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_negative_sign_rate": float(
            losses.get("collision_cluster_route_offset_agreement_negative_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_positive_mask_rate": float(
            losses.get("collision_cluster_route_offset_agreement_positive_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "collision_cluster_route_offset_agreement_negative_mask_rate": float(
            losses.get("collision_cluster_route_offset_agreement_negative_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_loss": float(
            losses.get("early_route_offset_agreement_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_sign_rate": float(
            losses.get("early_route_offset_agreement_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_mask_rate": float(
            losses.get("early_route_offset_agreement_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_signed_delta_mean": float(
            losses.get("early_route_offset_agreement_signed_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_corrected_delta_mean": float(
            losses.get("early_route_offset_agreement_corrected_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_positive_sign_rate": float(
            losses.get("early_route_offset_agreement_positive_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_negative_sign_rate": float(
            losses.get("early_route_offset_agreement_negative_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_positive_mask_rate": float(
            losses.get("early_route_offset_agreement_positive_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_route_offset_agreement_negative_mask_rate": float(
            losses.get("early_route_offset_agreement_negative_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_delta_loss": float(
            losses.get("early_severe_steer_delta_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_delta_l1": float(
            losses.get("early_severe_steer_delta_l1", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_sign_loss": float(
            losses.get("early_severe_steer_sign_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_sign_rate": float(
            losses.get("early_severe_steer_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_wrong_sign_rate": float(
            losses.get("early_severe_steer_wrong_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_absolute_steer_loss": float(
            losses.get("early_severe_absolute_steer_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_absolute_steer_sign_rate": float(
            losses.get("early_severe_absolute_steer_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_probe_absolute_steer_loss": float(
            losses.get("early_severe_probe_absolute_steer_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_probe_absolute_steer_sign_rate": float(
            losses.get("early_severe_probe_absolute_steer_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_probe_absolute_steer_mask_rate": float(
            losses.get("early_severe_probe_absolute_steer_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_longitudinal_ceiling_loss": float(
            losses.get("early_severe_longitudinal_ceiling_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_longitudinal_ceiling_rate": float(
            losses.get("early_severe_longitudinal_ceiling_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "late_recovery_absolute_steer_loss": float(
            losses.get("late_recovery_absolute_steer_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "late_recovery_absolute_steer_sign_rate": float(
            losses.get("late_recovery_absolute_steer_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "late_recovery_absolute_steer_mask_rate": float(
            losses.get("late_recovery_absolute_steer_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_center_zero_steer_loss": float(
            losses.get("early_center_zero_steer_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_center_zero_steer_abs_mean": float(
            losses.get("early_center_zero_steer_abs_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_center_zero_steer_mask_rate": float(
            losses.get("early_center_zero_steer_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_mask_rate": float(
            losses.get("early_severe_steer_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_left_sign_rate": float(
            losses.get("early_severe_steer_left_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_right_sign_rate": float(
            losses.get("early_severe_steer_right_sign_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_left_mask_rate": float(
            losses.get("early_severe_steer_left_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_right_mask_rate": float(
            losses.get("early_severe_steer_right_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_signed_delta_mean": float(
            losses.get("early_severe_steer_signed_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "early_severe_steer_corrective_delta_mean": float(
            losses.get("early_severe_steer_corrective_delta_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_progress_longitudinal_floor_loss": float(
            losses.get("route_progress_longitudinal_floor_loss", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_progress_longitudinal_floor_rate": float(
            losses.get("route_progress_longitudinal_floor_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_progress_longitudinal_mask_rate": float(
            losses.get("route_progress_longitudinal_mask_rate", fusion.new_zeros(())).detach().cpu().item()
        ),
        "route_progress_longitudinal_mean": float(
            losses.get("route_progress_longitudinal_mean", fusion.new_zeros(())).detach().cpu().item()
        ),
        "policy_std": float(model.get_std().detach().mean().cpu().item()),
        "risk_scalar": float(risk_scalar.detach().mean().cpu().item()),
    }
    for key in PERCEPTION_AUX_METRIC_KEYS:
        value = losses.get(key)
        row[key] = 0.0 if value is None else float(value.detach().cpu().item())
    for key in ("sequence_valid_rate", "sequence_mask_available_valid_rate", "sequence_perception_logits_available"):
        value = losses.get(key)
        row[key] = 0.0 if value is None else float(value.detach().cpu().item())
    for index, key in enumerate(("rgb", "lidar", "ego", "route")):
        row[f"fusion_{key}"] = float(fusion[:, index].detach().mean().cpu().item())
        row[f"reliability_{key}"] = float(reliability[:, index].detach().mean().cpu().item())
        row[f"contribution_{key}"] = float(contribution[:, index].detach().mean().cpu().item())
    diagnostics = model.get_diagnostics()
    for key in (
        "compact_visual_features_enabled",
        "compact_visual_feature_dim",
        "compact_visual_feature_l1_mean",
        "road_compact_feature_l1_mean",
        "lane_compact_feature_l1_mean",
        "visual_road_x_mean",
        "visual_lane_x_mean",
        "visual_lane_center_error_mean",
        "compact_visual_actor_input_l1_mean",
        "visual_actor_delta_l1_mean",
    ):
        row[key] = float(diagnostics.get(key, 0.0))
    return row


def _evaluate(
    *,
    model: RCSACFirstModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    use_auxiliary: bool,
    steer_loss_weight: float,
    longitudinal_loss_weight: float,
    perception_aux_loss_weight: float,
    road_mask_weight: float,
    lane_edge_weight: float,
    lane_edge_pos_weight: float,
    visual_zero_margin: float,
    visual_zero_steer_margin: float,
    visual_zero_route_min: float | None,
    visual_zero_route_max: float | None,
    visual_zero_abs_lane_offset_min: float,
    visual_zero_abs_heading_error_min: float,
    visual_zero_corrective_steer_margin: float,
    visual_center_error_min: float,
    visual_center_corrective_margin: float,
    route_aware_visual_center_route_min: float | None,
    route_aware_visual_center_route_max: float | None,
    route_aware_visual_center_margin: float,
    collision_cluster_route_min: float | None,
    collision_cluster_route_max: float | None,
    collision_cluster_lane_offset_min: float,
    collision_cluster_heading_error_max: float,
    collision_cluster_route_offset_margin: float,
    collision_cluster_lane_offset_mode: str,
    collision_cluster_heading_error_mode: str,
    early_route_offset_margin: float,
    early_route_min: float | None,
    early_route_max: float | None,
    early_route_lane_offset_min: float,
    early_route_heading_error_min: float,
    early_route_lane_offset_mode: str,
    early_route_heading_error_mode: str,
    early_severe_route_min: float | None,
    early_severe_route_max: float | None,
    early_severe_abs_lane_offset_min: float,
    early_severe_abs_heading_error_min: float,
    early_severe_steer_margin: float,
    early_severe_absolute_steer_margin: float,
    early_severe_probe_absolute_steer_margin: float,
    early_severe_probe_axes: tuple[str, ...],
    early_severe_longitudinal_ceiling_route_min: float | None,
    early_severe_longitudinal_ceiling_route_max: float | None,
    early_severe_longitudinal_ceiling: float,
    late_recovery_route_min: float | None,
    late_recovery_route_max: float | None,
    late_recovery_abs_lane_offset_min: float,
    late_recovery_abs_heading_error_min: float,
    late_recovery_absolute_steer_margin: float,
    late_recovery_bc_loss_scale: float,
    early_center_zero_steer_route_min: float | None,
    early_center_zero_steer_route_max: float | None,
    early_center_zero_steer_abs_lane_offset_max: float,
    early_center_zero_steer_abs_heading_error_max: float,
    early_severe_side_balanced: bool,
    early_severe_alignment_mode: str,
    route_progress_longitudinal_floor_route_min: float | None,
    route_progress_longitudinal_floor_route_max: float | None,
    route_progress_longitudinal_floor_speed_max_mps: float | None,
    route_progress_longitudinal_floor: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
    model.eval()
    totals = {
        "bc_loss": 0.0,
        "steer_mse": 0.0,
        "longitudinal_mse": 0.0,
        "action_l1": 0.0,
        "bc_sample_weight_mean": 0.0,
        "bc_sample_weight_min": 0.0,
        "bc_sample_weight_late_mask_rate": 0.0,
        "visual_zero_action_l1": 0.0,
        "visual_zero_steer_delta": 0.0,
        "visual_zero_longitudinal_delta": 0.0,
        "visual_zero_margin_loss": 0.0,
        "visual_zero_steer_margin_loss": 0.0,
        "visual_zero_corrective_steer_delta": 0.0,
        "visual_zero_corrective_sign_loss": 0.0,
        "visual_zero_corrective_mask_rate": 0.0,
        "target_corrective_delta_loss": 0.0,
        "target_corrective_delta_l1": 0.0,
        "target_corrective_delta_sign_rate": 0.0,
        "visual_zero_target_delta_loss": 0.0,
        "visual_zero_target_delta_l1": 0.0,
        "visual_zero_target_delta_sign_rate": 0.0,
        "target_corrective_delta_mask_rate": 0.0,
        "visual_center_corrective_sign_loss": 0.0,
        "visual_center_corrective_sign_rate": 0.0,
        "visual_center_correction_rate": 0.0,
        "visual_center_corrective_mask_rate": 0.0,
        "visual_center_target_consistency_rate": 0.0,
        "visual_center_target_consistency_mask_rate": 0.0,
        "visual_center_error_abs_mean": 0.0,
        "visual_center_signed_delta_mean": 0.0,
        "route_aware_visual_center_loss": 0.0,
        "route_aware_visual_center_sign_rate": 0.0,
        "route_aware_visual_center_correction_rate": 0.0,
        "route_aware_visual_center_mask_rate": 0.0,
        "route_aware_visual_center_target_consistency_rate": 0.0,
        "route_aware_visual_center_target_consistency_mask_rate": 0.0,
        "route_aware_visual_center_signed_delta_mean": 0.0,
        "collision_cluster_route_offset_agreement_loss": 0.0,
        "collision_cluster_route_offset_agreement_sign_rate": 0.0,
        "collision_cluster_route_offset_agreement_mask_rate": 0.0,
        "collision_cluster_route_offset_agreement_signed_delta_mean": 0.0,
        "collision_cluster_route_offset_agreement_corrected_delta_mean": 0.0,
        "collision_cluster_route_offset_agreement_positive_sign_rate": 0.0,
        "collision_cluster_route_offset_agreement_negative_sign_rate": 0.0,
        "collision_cluster_route_offset_agreement_positive_mask_rate": 0.0,
        "collision_cluster_route_offset_agreement_negative_mask_rate": 0.0,
        "early_route_offset_agreement_loss": 0.0,
        "early_route_offset_agreement_sign_rate": 0.0,
        "early_route_offset_agreement_mask_rate": 0.0,
        "early_route_offset_agreement_signed_delta_mean": 0.0,
        "early_route_offset_agreement_corrected_delta_mean": 0.0,
        "early_route_offset_agreement_positive_sign_rate": 0.0,
        "early_route_offset_agreement_negative_sign_rate": 0.0,
        "early_route_offset_agreement_positive_mask_rate": 0.0,
        "early_route_offset_agreement_negative_mask_rate": 0.0,
        "early_severe_steer_delta_loss": 0.0,
        "early_severe_steer_delta_l1": 0.0,
        "early_severe_steer_sign_loss": 0.0,
        "early_severe_steer_sign_rate": 0.0,
        "early_severe_steer_wrong_sign_rate": 0.0,
        "early_severe_absolute_steer_loss": 0.0,
        "early_severe_absolute_steer_sign_rate": 0.0,
        "early_severe_probe_absolute_steer_loss": 0.0,
        "early_severe_probe_absolute_steer_sign_rate": 0.0,
        "early_severe_probe_absolute_steer_mask_rate": 0.0,
        "early_severe_longitudinal_ceiling_loss": 0.0,
        "early_severe_longitudinal_ceiling_rate": 0.0,
        "late_recovery_absolute_steer_loss": 0.0,
        "late_recovery_absolute_steer_sign_rate": 0.0,
        "late_recovery_absolute_steer_mask_rate": 0.0,
        "early_center_zero_steer_loss": 0.0,
        "early_center_zero_steer_abs_mean": 0.0,
        "early_center_zero_steer_mask_rate": 0.0,
        "early_severe_steer_mask_rate": 0.0,
        "early_severe_steer_left_sign_rate": 0.0,
        "early_severe_steer_right_sign_rate": 0.0,
        "early_severe_steer_left_mask_rate": 0.0,
        "early_severe_steer_right_mask_rate": 0.0,
        "early_severe_steer_signed_delta_mean": 0.0,
        "early_severe_steer_corrective_delta_mean": 0.0,
        "route_progress_longitudinal_floor_loss": 0.0,
        "route_progress_longitudinal_floor_rate": 0.0,
        "route_progress_longitudinal_mask_rate": 0.0,
        "route_progress_longitudinal_mean": 0.0,
    }
    totals.update({key: 0.0 for key in PERCEPTION_AUX_METRIC_KEYS})
    count = 0
    last_outputs: dict[str, torch.Tensor | None] | None = None
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            if batch["obs"].ndim == 3:
                action_mean, outputs = _forward_sequence_outputs(
                    model,
                    batch,
                    use_auxiliary=use_auxiliary,
                    track_diagnostics=False,
                )
                losses = _compute_sequence_losses(
                    action_mean,
                    batch["expert_action"],
                    batch["valid_mask"],
                    steer_loss_weight=steer_loss_weight,
                    longitudinal_loss_weight=longitudinal_loss_weight,
                    sample_weight=_late_recovery_bc_sample_weight(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=late_recovery_route_min,
                        route_max=late_recovery_route_max,
                        abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                        abs_heading_error_min=late_recovery_abs_heading_error_min,
                        scale=late_recovery_bc_loss_scale,
                        alignment_mode=early_severe_alignment_mode,
                        valid_mask=batch.get("valid_mask"),
                    )[0],
                )
                _, bc_weight_mean, bc_weight_late_mask_rate = _late_recovery_bc_sample_weight(
                    action_mean,
                    batch,
                    route_progress=batch.get("route_progress"),
                    lane_offset=batch.get("lane_offset"),
                    heading_error=batch.get("heading_error"),
                    route_min=late_recovery_route_min,
                    route_max=late_recovery_route_max,
                    abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                    abs_heading_error_min=late_recovery_abs_heading_error_min,
                    scale=late_recovery_bc_loss_scale,
                    alignment_mode=early_severe_alignment_mode,
                    valid_mask=batch.get("valid_mask"),
                )
                losses["bc_sample_weight_mean"] = bc_weight_mean
                losses["bc_sample_weight_late_mask_rate"] = bc_weight_late_mask_rate
                losses.update(
                    _compute_sequence_perception_losses(
                        outputs=outputs,
                        batch=batch,
                        valid_mask=batch["valid_mask"],
                        perception_aux_loss_weight=perception_aux_loss_weight,
                        road_mask_weight=road_mask_weight,
                        lane_edge_weight=lane_edge_weight,
                        lane_edge_pos_weight=lane_edge_pos_weight,
                    )
                )
                zero_action_mean, _ = _forward_sequence_outputs(
                    model,
                    batch,
                    use_auxiliary=use_auxiliary,
                    track_diagnostics=False,
                    zero_rgb_perception_actor_features=True,
                )
                losses.update(
                    _compute_visual_zero_losses(
                        action_mean,
                        zero_action_mean,
                        visual_zero_margin=visual_zero_margin,
                        visual_zero_steer_margin=visual_zero_steer_margin,
                        visual_zero_route_min=visual_zero_route_min,
                        visual_zero_route_max=visual_zero_route_max,
                        visual_zero_abs_lane_offset_min=visual_zero_abs_lane_offset_min,
                        visual_zero_abs_heading_error_min=visual_zero_abs_heading_error_min,
                        visual_zero_corrective_steer_margin=visual_zero_corrective_steer_margin,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        valid_mask=batch["valid_mask"],
                    )
                )
                losses.update(
                    _compute_corrective_delta_losses(
                        action_mean,
                        zero_action_mean,
                        batch,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_visual_center_corrective_losses(
                        action_mean,
                        zero_action_mean,
                        outputs,
                        batch,
                        visual_center_error_min=visual_center_error_min,
                        visual_center_corrective_margin=visual_center_corrective_margin,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_route_aware_visual_center_losses(
                        action_mean,
                        zero_action_mean,
                        outputs,
                        batch,
                        route_min=route_aware_visual_center_route_min,
                        route_max=route_aware_visual_center_route_max,
                        visual_center_error_min=visual_center_error_min,
                        margin=route_aware_visual_center_margin,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_collision_cluster_route_offset_losses(
                        action_mean,
                        zero_action_mean,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=collision_cluster_route_min,
                        route_max=collision_cluster_route_max,
                        lane_offset_min=collision_cluster_lane_offset_min,
                        heading_error_max=collision_cluster_heading_error_max,
                        margin=collision_cluster_route_offset_margin,
                        lane_offset_mode=collision_cluster_lane_offset_mode,
                        heading_error_mode=collision_cluster_heading_error_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_early_route_offset_losses(
                        action_mean,
                        zero_action_mean,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_route_min,
                        route_max=early_route_max,
                        lane_offset_min=early_route_lane_offset_min,
                        heading_error_min=early_route_heading_error_min,
                        margin=early_route_offset_margin,
                        lane_offset_mode=early_route_lane_offset_mode,
                        heading_error_mode=early_route_heading_error_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_early_severe_steer_losses(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_severe_route_min,
                        route_max=early_severe_route_max,
                        abs_lane_offset_min=early_severe_abs_lane_offset_min,
                        abs_heading_error_min=early_severe_abs_heading_error_min,
                        margin=early_severe_steer_margin,
                        absolute_steer_margin=early_severe_absolute_steer_margin,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_early_severe_probe_absolute_steer_losses(
                        model=model,
                        batch=batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_severe_route_min,
                        route_max=early_severe_route_max,
                        abs_lane_offset_min=early_severe_abs_lane_offset_min,
                        abs_heading_error_min=early_severe_abs_heading_error_min,
                        margin=early_severe_probe_absolute_steer_margin,
                        axes=early_severe_probe_axes,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_early_severe_longitudinal_ceiling_losses(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_severe_longitudinal_ceiling_route_min,
                        route_max=early_severe_longitudinal_ceiling_route_max,
                        abs_lane_offset_min=early_severe_abs_lane_offset_min,
                        abs_heading_error_min=early_severe_abs_heading_error_min,
                        longitudinal_ceiling=early_severe_longitudinal_ceiling,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_late_recovery_absolute_steer_losses(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=late_recovery_route_min,
                        route_max=late_recovery_route_max,
                        abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                        abs_heading_error_min=late_recovery_abs_heading_error_min,
                        absolute_steer_margin=late_recovery_absolute_steer_margin,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_early_center_zero_steer_losses(
                        action_mean,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_center_zero_steer_route_min,
                        route_max=early_center_zero_steer_route_max,
                        abs_lane_offset_max=early_center_zero_steer_abs_lane_offset_max,
                        abs_heading_error_max=early_center_zero_steer_abs_heading_error_max,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
                losses.update(
                    _compute_route_progress_longitudinal_losses(
                        action_mean,
                        batch,
                        route_min=route_progress_longitudinal_floor_route_min,
                        route_max=route_progress_longitudinal_floor_route_max,
                        speed_max_mps=route_progress_longitudinal_floor_speed_max_mps,
                        longitudinal_floor=route_progress_longitudinal_floor,
                        valid_mask=batch.get("valid_mask"),
                    )
                )
            else:
                auxiliary = batch["auxiliary"] if use_auxiliary else None
                outputs = model.forward_policy_outputs(
                    batch["obs"],
                    auxiliary=auxiliary,
                    hidden_state=None,
                    track_diagnostics=True,
                )
                action_mean = outputs["action_mean"]
                if not isinstance(action_mean, torch.Tensor):
                    raise RuntimeError("Model output action_mean is missing.")
                bc_weights, bc_weight_mean, bc_weight_late_mask_rate = _late_recovery_bc_sample_weight(
                    action_mean,
                    batch,
                    route_progress=batch.get("route_progress"),
                    lane_offset=batch.get("lane_offset"),
                    heading_error=batch.get("heading_error"),
                    route_min=late_recovery_route_min,
                    route_max=late_recovery_route_max,
                    abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                    abs_heading_error_min=late_recovery_abs_heading_error_min,
                    scale=late_recovery_bc_loss_scale,
                    alignment_mode=early_severe_alignment_mode,
                )
                losses = _compute_losses(
                    action_mean,
                    batch["expert_action"],
                    steer_loss_weight=steer_loss_weight,
                    longitudinal_loss_weight=longitudinal_loss_weight,
                    sample_weight=bc_weights,
                )
                losses["bc_sample_weight_mean"] = bc_weight_mean
                losses["bc_sample_weight_late_mask_rate"] = bc_weight_late_mask_rate
                losses.update(
                    _compute_frame_perception_losses(
                        outputs=outputs,
                        batch=batch,
                        perception_aux_loss_weight=perception_aux_loss_weight,
                        road_mask_weight=road_mask_weight,
                        lane_edge_weight=lane_edge_weight,
                        lane_edge_pos_weight=lane_edge_pos_weight,
                    )
                )
                zero_outputs = model.forward_policy_outputs(
                    batch["obs"],
                    auxiliary=auxiliary,
                    hidden_state=None,
                    track_diagnostics=False,
                    zero_rgb_perception_actor_features=True,
                )
                zero_action_mean = zero_outputs["action_mean"]
                if not isinstance(zero_action_mean, torch.Tensor):
                    raise RuntimeError("Model zero-visual output action_mean is missing.")
                losses.update(
                    _compute_visual_zero_losses(
                        action_mean,
                        zero_action_mean,
                        visual_zero_margin=visual_zero_margin,
                        visual_zero_steer_margin=visual_zero_steer_margin,
                        visual_zero_route_min=visual_zero_route_min,
                        visual_zero_route_max=visual_zero_route_max,
                        visual_zero_abs_lane_offset_min=visual_zero_abs_lane_offset_min,
                        visual_zero_abs_heading_error_min=visual_zero_abs_heading_error_min,
                        visual_zero_corrective_steer_margin=visual_zero_corrective_steer_margin,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                    )
                )
                losses.update(_compute_corrective_delta_losses(action_mean, zero_action_mean, batch))
                losses.update(
                    _compute_visual_center_corrective_losses(
                        action_mean,
                        zero_action_mean,
                        outputs,
                        batch,
                        visual_center_error_min=visual_center_error_min,
                        visual_center_corrective_margin=visual_center_corrective_margin,
                    )
                )
                losses.update(
                    _compute_route_aware_visual_center_losses(
                        action_mean,
                        zero_action_mean,
                        outputs,
                        batch,
                        route_min=route_aware_visual_center_route_min,
                        route_max=route_aware_visual_center_route_max,
                        visual_center_error_min=visual_center_error_min,
                        margin=route_aware_visual_center_margin,
                    )
                )
                losses.update(
                    _compute_collision_cluster_route_offset_losses(
                        action_mean,
                        zero_action_mean,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=collision_cluster_route_min,
                        route_max=collision_cluster_route_max,
                        lane_offset_min=collision_cluster_lane_offset_min,
                        heading_error_max=collision_cluster_heading_error_max,
                        margin=collision_cluster_route_offset_margin,
                        lane_offset_mode=collision_cluster_lane_offset_mode,
                        heading_error_mode=collision_cluster_heading_error_mode,
                    )
                )
                losses.update(
                    _compute_early_route_offset_losses(
                        action_mean,
                        zero_action_mean,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_route_min,
                        route_max=early_route_max,
                        lane_offset_min=early_route_lane_offset_min,
                        heading_error_min=early_route_heading_error_min,
                        margin=early_route_offset_margin,
                        lane_offset_mode=early_route_lane_offset_mode,
                        heading_error_mode=early_route_heading_error_mode,
                    )
                )
                losses.update(
                    _compute_early_severe_steer_losses(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_severe_route_min,
                        route_max=early_severe_route_max,
                        abs_lane_offset_min=early_severe_abs_lane_offset_min,
                        abs_heading_error_min=early_severe_abs_heading_error_min,
                        margin=early_severe_steer_margin,
                        absolute_steer_margin=early_severe_absolute_steer_margin,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                    )
                )
                losses.update(
                    _compute_early_severe_probe_absolute_steer_losses(
                        model=model,
                        batch=batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_severe_route_min,
                        route_max=early_severe_route_max,
                        abs_lane_offset_min=early_severe_abs_lane_offset_min,
                        abs_heading_error_min=early_severe_abs_heading_error_min,
                        margin=early_severe_probe_absolute_steer_margin,
                        axes=early_severe_probe_axes,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                    )
                )
                losses.update(
                    _compute_early_severe_longitudinal_ceiling_losses(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_severe_longitudinal_ceiling_route_min,
                        route_max=early_severe_longitudinal_ceiling_route_max,
                        abs_lane_offset_min=early_severe_abs_lane_offset_min,
                        abs_heading_error_min=early_severe_abs_heading_error_min,
                        longitudinal_ceiling=early_severe_longitudinal_ceiling,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                    )
                )
                losses.update(
                    _compute_late_recovery_absolute_steer_losses(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=late_recovery_route_min,
                        route_max=late_recovery_route_max,
                        abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                        abs_heading_error_min=late_recovery_abs_heading_error_min,
                        absolute_steer_margin=late_recovery_absolute_steer_margin,
                        side_balanced=early_severe_side_balanced,
                        alignment_mode=early_severe_alignment_mode,
                    )
                )
                losses.update(
                    _compute_early_center_zero_steer_losses(
                        action_mean,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=early_center_zero_steer_route_min,
                        route_max=early_center_zero_steer_route_max,
                        abs_lane_offset_max=early_center_zero_steer_abs_lane_offset_max,
                        abs_heading_error_max=early_center_zero_steer_abs_heading_error_max,
                    )
                )
                losses.update(
                    _compute_route_progress_longitudinal_losses(
                        action_mean,
                        batch,
                        route_min=route_progress_longitudinal_floor_route_min,
                        route_max=route_progress_longitudinal_floor_route_max,
                        speed_max_mps=route_progress_longitudinal_floor_speed_max_mps,
                        longitudinal_floor=route_progress_longitudinal_floor,
                    )
                )
            batch_size = int(batch["obs"].shape[0])
            for key, value in losses.items():
                if key not in totals:
                    continue
                totals[key] += float(value.detach().cpu().item()) * batch_size
            count += batch_size
            last_outputs = outputs
    if count <= 0 or last_outputs is None:
        raise RuntimeError("Validation loader produced no batches.")
    return {key: torch.tensor(value / count, dtype=torch.float32, device=device) for key, value in totals.items()}, last_outputs


def _save_checkpoint(
    *,
    path: Path,
    model: RCSACFirstModel,
    optimizer: torch.optim.Optimizer,
    config: Any,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    dataset_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "dataset_path": str(dataset_path),
            "model_type": str(config.train.model_type),
            "model_variant": str(config.train.model_variant),
        },
        path,
    )


def _parse_bool(value: str | bool | None, *, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _load_init_checkpoint(model: RCSACFirstModel, checkpoint_path: Path, *, strict: bool, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BC init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict.")
    if strict:
        model.load_state_dict(state, strict=True)
        print(f"[init] loaded checkpoint={checkpoint_path} strict=True")
        return

    current = model.state_dict()
    loadable = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing = sorted(set(current.keys()) - set(loadable.keys()))
    unexpected = sorted(set(state.keys()) - set(current.keys()))
    model.load_state_dict(loadable, strict=False)
    print(
        f"[init] loaded checkpoint={checkpoint_path} strict=False "
        f"loaded_keys={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}"
    )


def _select_trainable_parameters(
    model: RCSACFirstModel,
    module_names: list[str],
) -> list[torch.nn.Parameter]:
    if not module_names:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    prefixes = tuple(str(name).strip() for name in module_names if str(name).strip())
    if not prefixes:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"bc.trainable_module_names selected no parameters: {module_names}")
    selected_count = sum(int(parameter.numel()) for parameter in selected)
    total_count = sum(int(parameter.numel()) for parameter in model.parameters())
    print(
        "[trainable] "
        f"prefixes={list(prefixes)} selected_params={selected_count} total_params={total_count}"
    )
    return selected


def _parse_trainable_action_indices(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    elif isinstance(raw, int):
        pieces = [raw]
    else:
        pieces = list(raw)
    indices = tuple(sorted({int(item) for item in pieces}))
    if any(index < 0 for index in indices):
        raise ValueError(f"bc.actor_head_output_trainable_indices must be non-negative, got {indices}")
    return indices


def _actor_head_output_row_restorer(
    model: RCSACFirstModel,
    trainable_indices: tuple[int, ...],
) -> Callable[[], None] | None:
    if not trainable_indices:
        return None
    actor_head = getattr(model.recurrent_actor_critic, "actor_head", None)
    output_layer: nn.Linear | None = None
    if isinstance(actor_head, nn.Sequential):
        for module in reversed(actor_head):
            if isinstance(module, nn.Linear):
                output_layer = module
                break
    if output_layer is None:
        raise ValueError("bc.actor_head_output_trainable_indices requires a Linear actor output layer")
    action_dim = int(output_layer.out_features)
    invalid = [index for index in trainable_indices if index >= action_dim]
    if invalid:
        raise ValueError(
            "bc.actor_head_output_trainable_indices contains indices outside "
            f"action_dim={action_dim}: {invalid}"
        )
    trainable_mask = torch.zeros(action_dim, dtype=torch.bool, device=output_layer.weight.device)
    trainable_mask[list(trainable_indices)] = True
    frozen_mask = ~trainable_mask
    if not bool(frozen_mask.any().item()):
        return None
    frozen_weight = output_layer.weight.detach().clone()
    frozen_bias = output_layer.bias.detach().clone() if output_layer.bias is not None else None

    def restore() -> None:
        with torch.no_grad():
            output_layer.weight[frozen_mask] = frozen_weight[frozen_mask].to(output_layer.weight.device)
            if output_layer.bias is not None and frozen_bias is not None:
                output_layer.bias[frozen_mask] = frozen_bias[frozen_mask].to(output_layer.bias.device)

    print(
        "[trainable] actor_head_output_trainable_indices="
        f"{list(trainable_indices)} frozen_indices={torch.nonzero(frozen_mask, as_tuple=False).view(-1).tolist()}"
    )
    return restore


def main() -> int:
    args = parse_args()
    config = load_phase1_config(_resolve_config_path(str(args.config)))
    if args.device is not None:
        config.train.device = str(args.device)
    run_name = args.run_name or config.train.run_name
    _set_seed(config.train.seed)

    bc_raw = _bc_raw(config)
    epochs = int(args.epochs if args.epochs is not None else bc_raw.get("epochs", 5))
    batch_size = int(args.batch_size if args.batch_size is not None else bc_raw.get("batch_size", 32))
    learning_rate = float(args.learning_rate if args.learning_rate is not None else bc_raw.get("learning_rate", 3e-4))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else bc_raw.get("val_fraction", 0.1))
    grad_clip_norm = float(bc_raw.get("grad_clip_norm", 0.5))
    weight_decay = float(bc_raw.get("weight_decay", 1e-5))
    use_auxiliary = bool(bc_raw.get("use_auxiliary", True))
    include_bad = bool(bc_raw.get("include_bad_transitions", True))
    num_workers = int(bc_raw.get("num_workers", 0))
    log_interval = max(1, int(bc_raw.get("log_interval", 20)))
    bc_loss_weight = float(bc_raw.get("bc_loss_weight", 1.0))
    bc_mode = str(bc_raw.get("mode", "frame")).strip().lower()
    sequence_length = int(bc_raw.get("sequence_length", 32))
    sequence_stride = int(bc_raw.get("sequence_stride", max(1, sequence_length // 4)))
    action_smooth_loss_weight = float(bc_raw.get("action_smooth_loss_weight", 0.0))
    steer_loss_weight = float(bc_raw.get("steer_loss_weight", 1.0))
    longitudinal_loss_weight = float(bc_raw.get("longitudinal_loss_weight", 1.0))
    perception_aux_loss_weight = float(bc_raw.get("perception_aux_loss_weight", 0.0))
    visual_zero_margin = float(bc_raw.get("visual_zero_margin", 0.0))
    visual_zero_margin_loss_weight = float(bc_raw.get("visual_zero_margin_loss_weight", 0.0))
    visual_zero_steer_margin = float(bc_raw.get("visual_zero_steer_margin", 0.0))
    visual_zero_steer_margin_loss_weight = float(bc_raw.get("visual_zero_steer_margin_loss_weight", 0.0))
    visual_zero_route_min = bc_raw.get("visual_zero_route_min")
    visual_zero_route_max = bc_raw.get("visual_zero_route_max")
    visual_zero_abs_lane_offset_min = float(bc_raw.get("visual_zero_abs_lane_offset_min", 0.0))
    visual_zero_abs_heading_error_min = float(bc_raw.get("visual_zero_abs_heading_error_min", 0.0))
    visual_zero_corrective_steer_margin = float(bc_raw.get("visual_zero_corrective_steer_margin", 0.0))
    visual_zero_corrective_sign_loss_weight = float(bc_raw.get("visual_zero_corrective_sign_loss_weight", 0.0))
    target_corrective_delta_loss_weight = float(bc_raw.get("target_corrective_delta_loss_weight", 0.0))
    visual_zero_target_delta_loss_weight = float(bc_raw.get("visual_zero_target_delta_loss_weight", 0.0))
    visual_center_corrective_sign_loss_weight = float(bc_raw.get("visual_center_corrective_sign_loss_weight", 0.0))
    visual_center_corrective_margin = float(bc_raw.get("visual_center_corrective_margin", 0.0))
    visual_center_error_min = float(bc_raw.get("visual_center_error_min", 0.0))
    route_aware_visual_center_loss_weight = float(bc_raw.get("route_aware_visual_center_loss_weight", 0.0))
    route_aware_visual_center_route_min = bc_raw.get("route_aware_visual_center_route_min")
    route_aware_visual_center_route_max = bc_raw.get("route_aware_visual_center_route_max")
    route_aware_visual_center_margin = float(bc_raw.get("route_aware_visual_center_margin", 0.0))
    collision_cluster_route_offset_agreement_loss_weight = float(
        bc_raw.get("collision_cluster_route_offset_agreement_loss_weight", 0.0)
    )
    collision_cluster_route_min = bc_raw.get("collision_cluster_route_min")
    collision_cluster_route_max = bc_raw.get("collision_cluster_route_max")
    collision_cluster_lane_offset_min = float(bc_raw.get("collision_cluster_lane_offset_min", 0.0))
    collision_cluster_heading_error_max = float(bc_raw.get("collision_cluster_heading_error_max", 0.0))
    collision_cluster_route_offset_margin = float(bc_raw.get("collision_cluster_route_offset_margin", 0.0))
    collision_cluster_lane_offset_mode = str(bc_raw.get("collision_cluster_lane_offset_mode", "positive"))
    collision_cluster_heading_error_mode = str(bc_raw.get("collision_cluster_heading_error_mode", "less_than"))
    early_route_offset_agreement_loss_weight = float(
        bc_raw.get("early_route_offset_agreement_loss_weight", 0.0)
    )
    early_route_min = bc_raw.get("early_route_min")
    early_route_max = bc_raw.get("early_route_max")
    early_route_lane_offset_min = float(bc_raw.get("early_route_lane_offset_min", 0.0))
    early_route_heading_error_min = float(bc_raw.get("early_route_heading_error_min", 0.0))
    early_route_offset_margin = float(bc_raw.get("early_route_offset_margin", 0.0))
    early_route_lane_offset_mode = str(bc_raw.get("early_route_lane_offset_mode", "negative"))
    early_route_heading_error_mode = str(bc_raw.get("early_route_heading_error_mode", "positive"))
    early_severe_steer_delta_loss_weight = float(bc_raw.get("early_severe_steer_delta_loss_weight", 0.0))
    early_severe_steer_sign_loss_weight = float(bc_raw.get("early_severe_steer_sign_loss_weight", 0.0))
    early_severe_route_min = bc_raw.get("early_severe_route_min", early_route_min)
    early_severe_route_max = bc_raw.get("early_severe_route_max", 0.10)
    early_severe_abs_lane_offset_min = float(bc_raw.get("early_severe_abs_lane_offset_min", 0.55))
    early_severe_abs_heading_error_min = float(bc_raw.get("early_severe_abs_heading_error_min", 3.0))
    early_severe_steer_margin = float(bc_raw.get("early_severe_steer_margin", 0.005))
    early_severe_absolute_steer_loss_weight = float(
        bc_raw.get("early_severe_absolute_steer_loss_weight", 0.0)
    )
    early_severe_absolute_steer_margin = float(bc_raw.get("early_severe_absolute_steer_margin", 0.0))
    early_severe_probe_absolute_steer_loss_weight = float(
        bc_raw.get("early_severe_probe_absolute_steer_loss_weight", 0.0)
    )
    early_severe_probe_absolute_steer_margin = float(
        bc_raw.get("early_severe_probe_absolute_steer_margin", 0.0)
    )
    early_severe_probe_axes_raw = bc_raw.get("early_severe_probe_axes", [])
    if isinstance(early_severe_probe_axes_raw, str):
        early_severe_probe_axes = tuple(
            item.strip() for item in early_severe_probe_axes_raw.split(",") if item.strip()
        )
    else:
        early_severe_probe_axes = tuple(str(item) for item in early_severe_probe_axes_raw)
    early_severe_longitudinal_ceiling_loss_weight = float(
        bc_raw.get("early_severe_longitudinal_ceiling_loss_weight", 0.0)
    )
    early_severe_longitudinal_ceiling_route_min = bc_raw.get(
        "early_severe_longitudinal_ceiling_route_min", early_severe_route_min
    )
    early_severe_longitudinal_ceiling_route_max = bc_raw.get(
        "early_severe_longitudinal_ceiling_route_max", early_severe_route_max
    )
    early_severe_longitudinal_ceiling = float(bc_raw.get("early_severe_longitudinal_ceiling", 1.0))
    late_recovery_absolute_steer_loss_weight = float(bc_raw.get("late_recovery_absolute_steer_loss_weight", 0.0))
    late_recovery_route_min = bc_raw.get("late_recovery_route_min", early_severe_route_min)
    late_recovery_route_max = bc_raw.get("late_recovery_route_max", early_severe_route_max)
    late_recovery_abs_lane_offset_min = float(
        bc_raw.get("late_recovery_abs_lane_offset_min", early_severe_abs_lane_offset_min)
    )
    late_recovery_abs_heading_error_min = float(
        bc_raw.get("late_recovery_abs_heading_error_min", early_severe_abs_heading_error_min)
    )
    late_recovery_absolute_steer_margin = float(bc_raw.get("late_recovery_absolute_steer_margin", 0.0))
    late_recovery_bc_loss_scale = float(bc_raw.get("late_recovery_bc_loss_scale", 1.0))
    early_center_zero_steer_loss_weight = float(bc_raw.get("early_center_zero_steer_loss_weight", 0.0))
    early_center_zero_steer_route_min = bc_raw.get("early_center_zero_steer_route_min", early_route_min)
    early_center_zero_steer_route_max = bc_raw.get("early_center_zero_steer_route_max", early_route_max)
    early_center_zero_steer_abs_lane_offset_max = float(
        bc_raw.get("early_center_zero_steer_abs_lane_offset_max", 0.20)
    )
    early_center_zero_steer_abs_heading_error_max = float(
        bc_raw.get("early_center_zero_steer_abs_heading_error_max", 2.0)
    )
    early_severe_side_balanced = bool(bc_raw.get("early_severe_side_balanced", True))
    early_severe_alignment_mode = str(bc_raw.get("early_severe_alignment_mode", "lane"))
    route_progress_longitudinal_floor_loss_weight = float(
        bc_raw.get("route_progress_longitudinal_floor_loss_weight", 0.0)
    )
    route_progress_longitudinal_floor_route_min = bc_raw.get("route_progress_longitudinal_floor_route_min")
    route_progress_longitudinal_floor_route_max = bc_raw.get("route_progress_longitudinal_floor_route_max")
    route_progress_longitudinal_floor_speed_max_mps = bc_raw.get("route_progress_longitudinal_floor_speed_max_mps")
    route_progress_longitudinal_floor = float(bc_raw.get("route_progress_longitudinal_floor", -1.0))
    road_mask_weight = float(bc_raw.get("road_mask_weight", config.loss.aux_road_mask_loss_weight))
    lane_edge_weight = float(bc_raw.get("lane_edge_weight", config.loss.aux_lane_edge_mask_loss_weight))
    lane_edge_pos_weight = float(bc_raw.get("lane_edge_pos_weight", config.loss.aux_lane_edge_pos_weight))
    trainable_module_names_raw = bc_raw.get("trainable_module_names", [])
    if isinstance(trainable_module_names_raw, str):
        trainable_module_names = [trainable_module_names_raw]
    else:
        trainable_module_names = [str(item) for item in trainable_module_names_raw]
    actor_head_output_trainable_indices = _parse_trainable_action_indices(
        bc_raw.get("actor_head_output_trainable_indices")
    )

    dataset_path = _project_path(args.dataset)
    arrays = load_expert_npz(dataset_path)
    summary = summarize_expert_arrays(arrays)
    print(f"[dataset] {summary.to_dict()}")

    if bc_mode == "sequence":
        train_dataset = SequenceExpertDataset(
            dataset_path,
            split="train",
            val_fraction=val_fraction,
            seed=config.train.seed,
            sequence_length=sequence_length,
            stride=sequence_stride,
            include_bad_transitions=include_bad,
        )
        val_dataset = SequenceExpertDataset(
            dataset_path,
            split="val",
            val_fraction=val_fraction,
            seed=config.train.seed,
            sequence_length=sequence_length,
            stride=sequence_stride,
            include_bad_transitions=include_bad,
        )
    elif bc_mode == "frame":
        train_dataset = ExpertDataset(
            dataset_path,
            split="train",
            val_fraction=val_fraction,
            seed=config.train.seed,
            include_bad_transitions=include_bad,
        )
        val_dataset = ExpertDataset(
            dataset_path,
            split="val",
            val_fraction=val_fraction,
            seed=config.train.seed,
            include_bad_transitions=include_bad,
        )
    else:
        raise ValueError("bc.mode must be 'frame' or 'sequence'")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = _device(config, args.device)
    model = RCSACFirstModel(config.model, config.fusion).to(device)
    if args.init_checkpoint is not None:
        _load_init_checkpoint(
            model,
            _project_path(args.init_checkpoint),
            strict=_parse_bool(args.init_strict, default=True),
            device=device,
        )
    trainable_parameters = _select_trainable_parameters(model, trainable_module_names)
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    restore_frozen_actor_output_rows = _actor_head_output_row_restorer(
        model,
        actor_head_output_trainable_indices,
    )

    logs_dir = _project_path(config.train.logs_dir) / str(run_name)
    checkpoints_dir = _project_path(config.train.checkpoints_dir) / str(run_name)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "bc_training_log.csv"

    global_step = 0
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BC_LOG_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = _batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                if bc_mode == "sequence":
                    action_mean, outputs = _forward_sequence_outputs(
                        model,
                        batch,
                        use_auxiliary=use_auxiliary,
                        track_diagnostics=True,
                    )
                    bc_weights, bc_weight_mean, bc_weight_late_mask_rate = _late_recovery_bc_sample_weight(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=late_recovery_route_min,
                        route_max=late_recovery_route_max,
                        abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                        abs_heading_error_min=late_recovery_abs_heading_error_min,
                        scale=late_recovery_bc_loss_scale,
                        alignment_mode=early_severe_alignment_mode,
                        valid_mask=batch.get("valid_mask"),
                    )
                    losses = _compute_sequence_losses(
                        action_mean,
                        batch["expert_action"],
                        batch["valid_mask"],
                        action_smooth_loss_weight=action_smooth_loss_weight,
                        steer_loss_weight=steer_loss_weight,
                        longitudinal_loss_weight=longitudinal_loss_weight,
                        sample_weight=bc_weights,
                    )
                    losses["bc_sample_weight_mean"] = bc_weight_mean
                    losses["bc_sample_weight_late_mask_rate"] = bc_weight_late_mask_rate
                    losses.update(
                        _compute_sequence_perception_losses(
                            outputs=outputs,
                            batch=batch,
                            valid_mask=batch["valid_mask"],
                            perception_aux_loss_weight=perception_aux_loss_weight,
                            road_mask_weight=road_mask_weight,
                            lane_edge_weight=lane_edge_weight,
                            lane_edge_pos_weight=lane_edge_pos_weight,
                        )
                    )
                    zero_action_mean, _ = _forward_sequence_outputs(
                        model,
                        batch,
                        use_auxiliary=use_auxiliary,
                        track_diagnostics=False,
                        zero_rgb_perception_actor_features=True,
                    )
                    losses.update(
                        _compute_visual_zero_losses(
                            action_mean,
                            zero_action_mean,
                            visual_zero_margin=visual_zero_margin,
                            visual_zero_steer_margin=visual_zero_steer_margin,
                            visual_zero_route_min=visual_zero_route_min,
                            visual_zero_route_max=visual_zero_route_max,
                            visual_zero_abs_lane_offset_min=visual_zero_abs_lane_offset_min,
                            visual_zero_abs_heading_error_min=visual_zero_abs_heading_error_min,
                            visual_zero_corrective_steer_margin=visual_zero_corrective_steer_margin,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            valid_mask=batch["valid_mask"],
                        )
                    )
                    losses.update(
                        _compute_corrective_delta_losses(
                            action_mean,
                            zero_action_mean,
                            batch,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                    losses.update(
                        _compute_visual_center_corrective_losses(
                            action_mean,
                            zero_action_mean,
                            outputs,
                            batch,
                            visual_center_error_min=visual_center_error_min,
                            visual_center_corrective_margin=visual_center_corrective_margin,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                    losses.update(
                        _compute_route_aware_visual_center_losses(
                            action_mean,
                            zero_action_mean,
                            outputs,
                            batch,
                            route_min=route_aware_visual_center_route_min,
                            route_max=route_aware_visual_center_route_max,
                            visual_center_error_min=visual_center_error_min,
                            margin=route_aware_visual_center_margin,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                    losses.update(
                        _compute_collision_cluster_route_offset_losses(
                            action_mean,
                            zero_action_mean,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            route_min=collision_cluster_route_min,
                            route_max=collision_cluster_route_max,
                            lane_offset_min=collision_cluster_lane_offset_min,
                            heading_error_max=collision_cluster_heading_error_max,
                            margin=collision_cluster_route_offset_margin,
                            lane_offset_mode=collision_cluster_lane_offset_mode,
                            heading_error_mode=collision_cluster_heading_error_mode,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                    losses.update(
                        _compute_early_route_offset_losses(
                            action_mean,
                            zero_action_mean,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            route_min=early_route_min,
                            route_max=early_route_max,
                            lane_offset_min=early_route_lane_offset_min,
                            heading_error_min=early_route_heading_error_min,
                            margin=early_route_offset_margin,
                            lane_offset_mode=early_route_lane_offset_mode,
                            heading_error_mode=early_route_heading_error_mode,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                    losses.update(
                        _compute_early_severe_steer_losses(
                            action_mean,
                            batch,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            route_min=early_severe_route_min,
                            route_max=early_severe_route_max,
                            abs_lane_offset_min=early_severe_abs_lane_offset_min,
                            abs_heading_error_min=early_severe_abs_heading_error_min,
                            margin=early_severe_steer_margin,
                            absolute_steer_margin=early_severe_absolute_steer_margin,
                            side_balanced=early_severe_side_balanced,
                            alignment_mode=early_severe_alignment_mode,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                    if early_severe_probe_absolute_steer_loss_weight > 0.0 and early_severe_probe_axes:
                        losses.update(
                            _compute_early_severe_probe_absolute_steer_losses(
                                model=model,
                                batch=batch,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=early_severe_route_min,
                                route_max=early_severe_route_max,
                                abs_lane_offset_min=early_severe_abs_lane_offset_min,
                                abs_heading_error_min=early_severe_abs_heading_error_min,
                                margin=early_severe_probe_absolute_steer_margin,
                                axes=early_severe_probe_axes,
                                side_balanced=early_severe_side_balanced,
                                alignment_mode=early_severe_alignment_mode,
                                valid_mask=batch.get("valid_mask"),
                            )
                        )
                    if early_severe_longitudinal_ceiling_loss_weight > 0.0:
                        losses.update(
                            _compute_early_severe_longitudinal_ceiling_losses(
                                action_mean,
                                batch,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=early_severe_longitudinal_ceiling_route_min,
                                route_max=early_severe_longitudinal_ceiling_route_max,
                                abs_lane_offset_min=early_severe_abs_lane_offset_min,
                                abs_heading_error_min=early_severe_abs_heading_error_min,
                                longitudinal_ceiling=early_severe_longitudinal_ceiling,
                                side_balanced=early_severe_side_balanced,
                                alignment_mode=early_severe_alignment_mode,
                                valid_mask=batch.get("valid_mask"),
                            )
                        )
                    if late_recovery_absolute_steer_loss_weight > 0.0:
                        losses.update(
                            _compute_late_recovery_absolute_steer_losses(
                                action_mean,
                                batch,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=late_recovery_route_min,
                                route_max=late_recovery_route_max,
                                abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                                abs_heading_error_min=late_recovery_abs_heading_error_min,
                                absolute_steer_margin=late_recovery_absolute_steer_margin,
                                side_balanced=early_severe_side_balanced,
                                alignment_mode=early_severe_alignment_mode,
                                valid_mask=batch.get("valid_mask"),
                            )
                        )
                    if early_center_zero_steer_loss_weight > 0.0:
                        losses.update(
                            _compute_early_center_zero_steer_losses(
                                action_mean,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=early_center_zero_steer_route_min,
                                route_max=early_center_zero_steer_route_max,
                                abs_lane_offset_max=early_center_zero_steer_abs_lane_offset_max,
                                abs_heading_error_max=early_center_zero_steer_abs_heading_error_max,
                                valid_mask=batch.get("valid_mask"),
                            )
                        )
                    losses.update(
                        _compute_route_progress_longitudinal_losses(
                            action_mean,
                            batch,
                            route_min=route_progress_longitudinal_floor_route_min,
                            route_max=route_progress_longitudinal_floor_route_max,
                            speed_max_mps=route_progress_longitudinal_floor_speed_max_mps,
                            longitudinal_floor=route_progress_longitudinal_floor,
                            valid_mask=batch.get("valid_mask"),
                        )
                    )
                else:
                    auxiliary = batch["auxiliary"] if use_auxiliary else None
                    outputs = model.forward_policy_outputs(
                        batch["obs"],
                        auxiliary=auxiliary,
                        hidden_state=None,
                        track_diagnostics=True,
                    )
                    action_mean = outputs["action_mean"]
                    if not isinstance(action_mean, torch.Tensor):
                        raise RuntimeError("Model output action_mean is missing.")
                    bc_weights, bc_weight_mean, bc_weight_late_mask_rate = _late_recovery_bc_sample_weight(
                        action_mean,
                        batch,
                        route_progress=batch.get("route_progress"),
                        lane_offset=batch.get("lane_offset"),
                        heading_error=batch.get("heading_error"),
                        route_min=late_recovery_route_min,
                        route_max=late_recovery_route_max,
                        abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                        abs_heading_error_min=late_recovery_abs_heading_error_min,
                        scale=late_recovery_bc_loss_scale,
                        alignment_mode=early_severe_alignment_mode,
                    )
                    losses = _compute_losses(
                        action_mean,
                        batch["expert_action"],
                        steer_loss_weight=steer_loss_weight,
                        longitudinal_loss_weight=longitudinal_loss_weight,
                        sample_weight=bc_weights,
                    )
                    losses["bc_sample_weight_mean"] = bc_weight_mean
                    losses["bc_sample_weight_late_mask_rate"] = bc_weight_late_mask_rate
                    losses.update(
                        _compute_frame_perception_losses(
                            outputs=outputs,
                            batch=batch,
                            perception_aux_loss_weight=perception_aux_loss_weight,
                            road_mask_weight=road_mask_weight,
                            lane_edge_weight=lane_edge_weight,
                            lane_edge_pos_weight=lane_edge_pos_weight,
                        )
                    )
                    zero_outputs = model.forward_policy_outputs(
                        batch["obs"],
                        auxiliary=auxiliary,
                        hidden_state=None,
                        track_diagnostics=False,
                        zero_rgb_perception_actor_features=True,
                    )
                    zero_action_mean = zero_outputs["action_mean"]
                    if not isinstance(zero_action_mean, torch.Tensor):
                        raise RuntimeError("Model zero-visual output action_mean is missing.")
                    losses.update(
                        _compute_visual_zero_losses(
                            action_mean,
                            zero_action_mean,
                            visual_zero_margin=visual_zero_margin,
                            visual_zero_steer_margin=visual_zero_steer_margin,
                            visual_zero_route_min=visual_zero_route_min,
                            visual_zero_route_max=visual_zero_route_max,
                            visual_zero_abs_lane_offset_min=visual_zero_abs_lane_offset_min,
                            visual_zero_abs_heading_error_min=visual_zero_abs_heading_error_min,
                            visual_zero_corrective_steer_margin=visual_zero_corrective_steer_margin,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                        )
                    )
                    losses.update(_compute_corrective_delta_losses(action_mean, zero_action_mean, batch))
                    losses.update(
                        _compute_visual_center_corrective_losses(
                            action_mean,
                            zero_action_mean,
                            outputs,
                            batch,
                            visual_center_error_min=visual_center_error_min,
                            visual_center_corrective_margin=visual_center_corrective_margin,
                        )
                    )
                    losses.update(
                        _compute_route_aware_visual_center_losses(
                            action_mean,
                            zero_action_mean,
                            outputs,
                            batch,
                            route_min=route_aware_visual_center_route_min,
                            route_max=route_aware_visual_center_route_max,
                            visual_center_error_min=visual_center_error_min,
                            margin=route_aware_visual_center_margin,
                        )
                    )
                    losses.update(
                        _compute_collision_cluster_route_offset_losses(
                            action_mean,
                            zero_action_mean,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            route_min=collision_cluster_route_min,
                            route_max=collision_cluster_route_max,
                            lane_offset_min=collision_cluster_lane_offset_min,
                            heading_error_max=collision_cluster_heading_error_max,
                            margin=collision_cluster_route_offset_margin,
                            lane_offset_mode=collision_cluster_lane_offset_mode,
                            heading_error_mode=collision_cluster_heading_error_mode,
                        )
                    )
                    losses.update(
                        _compute_early_route_offset_losses(
                            action_mean,
                            zero_action_mean,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            route_min=early_route_min,
                            route_max=early_route_max,
                            lane_offset_min=early_route_lane_offset_min,
                            heading_error_min=early_route_heading_error_min,
                            margin=early_route_offset_margin,
                            lane_offset_mode=early_route_lane_offset_mode,
                            heading_error_mode=early_route_heading_error_mode,
                        )
                    )
                    losses.update(
                        _compute_early_severe_steer_losses(
                            action_mean,
                            batch,
                            route_progress=batch.get("route_progress"),
                            lane_offset=batch.get("lane_offset"),
                            heading_error=batch.get("heading_error"),
                            route_min=early_severe_route_min,
                            route_max=early_severe_route_max,
                            abs_lane_offset_min=early_severe_abs_lane_offset_min,
                            abs_heading_error_min=early_severe_abs_heading_error_min,
                            margin=early_severe_steer_margin,
                            absolute_steer_margin=early_severe_absolute_steer_margin,
                            side_balanced=early_severe_side_balanced,
                            alignment_mode=early_severe_alignment_mode,
                        )
                    )
                    if early_severe_probe_absolute_steer_loss_weight > 0.0 and early_severe_probe_axes:
                        losses.update(
                            _compute_early_severe_probe_absolute_steer_losses(
                                model=model,
                                batch=batch,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=early_severe_route_min,
                                route_max=early_severe_route_max,
                                abs_lane_offset_min=early_severe_abs_lane_offset_min,
                                abs_heading_error_min=early_severe_abs_heading_error_min,
                                margin=early_severe_probe_absolute_steer_margin,
                                axes=early_severe_probe_axes,
                                side_balanced=early_severe_side_balanced,
                                alignment_mode=early_severe_alignment_mode,
                            )
                        )
                    if early_severe_longitudinal_ceiling_loss_weight > 0.0:
                        losses.update(
                            _compute_early_severe_longitudinal_ceiling_losses(
                                action_mean,
                                batch,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=early_severe_longitudinal_ceiling_route_min,
                                route_max=early_severe_longitudinal_ceiling_route_max,
                                abs_lane_offset_min=early_severe_abs_lane_offset_min,
                                abs_heading_error_min=early_severe_abs_heading_error_min,
                                longitudinal_ceiling=early_severe_longitudinal_ceiling,
                                side_balanced=early_severe_side_balanced,
                                alignment_mode=early_severe_alignment_mode,
                            )
                        )
                    if late_recovery_absolute_steer_loss_weight > 0.0:
                        losses.update(
                            _compute_late_recovery_absolute_steer_losses(
                                action_mean,
                                batch,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=late_recovery_route_min,
                                route_max=late_recovery_route_max,
                                abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                                abs_heading_error_min=late_recovery_abs_heading_error_min,
                                absolute_steer_margin=late_recovery_absolute_steer_margin,
                                side_balanced=early_severe_side_balanced,
                                alignment_mode=early_severe_alignment_mode,
                            )
                        )
                    if early_center_zero_steer_loss_weight > 0.0:
                        losses.update(
                            _compute_early_center_zero_steer_losses(
                                action_mean,
                                route_progress=batch.get("route_progress"),
                                lane_offset=batch.get("lane_offset"),
                                heading_error=batch.get("heading_error"),
                                route_min=early_center_zero_steer_route_min,
                                route_max=early_center_zero_steer_route_max,
                                abs_lane_offset_max=early_center_zero_steer_abs_lane_offset_max,
                                abs_heading_error_max=early_center_zero_steer_abs_heading_error_max,
                            )
                        )
                    losses.update(
                        _compute_route_progress_longitudinal_losses(
                            action_mean,
                            batch,
                            route_min=route_progress_longitudinal_floor_route_min,
                            route_max=route_progress_longitudinal_floor_route_max,
                            speed_max_mps=route_progress_longitudinal_floor_speed_max_mps,
                            longitudinal_floor=route_progress_longitudinal_floor,
                        )
                    )
                total_loss = losses["bc_loss"] * bc_loss_weight
                total_loss = total_loss + losses.get("perception_aux_loss", total_loss.new_zeros(())) * perception_aux_loss_weight
                total_loss = (
                    total_loss
                    + losses.get("visual_zero_margin_loss", total_loss.new_zeros(())) * visual_zero_margin_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("visual_zero_steer_margin_loss", total_loss.new_zeros(()))
                    * visual_zero_steer_margin_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("visual_zero_corrective_sign_loss", total_loss.new_zeros(()))
                    * visual_zero_corrective_sign_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("target_corrective_delta_loss", total_loss.new_zeros(()))
                    * target_corrective_delta_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("visual_zero_target_delta_loss", total_loss.new_zeros(()))
                    * visual_zero_target_delta_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("visual_center_corrective_sign_loss", total_loss.new_zeros(()))
                    * visual_center_corrective_sign_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("route_aware_visual_center_loss", total_loss.new_zeros(()))
                    * route_aware_visual_center_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("collision_cluster_route_offset_agreement_loss", total_loss.new_zeros(()))
                    * collision_cluster_route_offset_agreement_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_route_offset_agreement_loss", total_loss.new_zeros(()))
                    * early_route_offset_agreement_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_severe_steer_delta_loss", total_loss.new_zeros(()))
                    * early_severe_steer_delta_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_severe_steer_sign_loss", total_loss.new_zeros(()))
                    * early_severe_steer_sign_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_severe_absolute_steer_loss", total_loss.new_zeros(()))
                    * early_severe_absolute_steer_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_severe_probe_absolute_steer_loss", total_loss.new_zeros(()))
                    * early_severe_probe_absolute_steer_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_severe_longitudinal_ceiling_loss", total_loss.new_zeros(()))
                    * early_severe_longitudinal_ceiling_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("late_recovery_absolute_steer_loss", total_loss.new_zeros(()))
                    * late_recovery_absolute_steer_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("early_center_zero_steer_loss", total_loss.new_zeros(()))
                    * early_center_zero_steer_loss_weight
                )
                total_loss = (
                    total_loss
                    + losses.get("route_progress_longitudinal_floor_loss", total_loss.new_zeros(()))
                    * route_progress_longitudinal_floor_loss_weight
                )
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Non-finite BC loss at step {global_step}: {float(total_loss.detach().cpu())}")
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
                optimizer.step()
                if restore_frozen_actor_output_rows is not None:
                    restore_frozen_actor_output_rows()
                global_step += 1
                last_train_loss = float(losses["bc_loss"].detach().cpu().item())
                if global_step % log_interval == 0:
                    writer.writerow(
                        _metric_row(
                            epoch=epoch,
                            global_step=global_step,
                            split="train",
                            losses=losses,
                            outputs=outputs,
                            model=model,
                        )
                    )
                    handle.flush()

            val_losses, val_outputs = _evaluate(
                model=model,
                loader=val_loader,
                device=device,
                use_auxiliary=use_auxiliary,
                steer_loss_weight=steer_loss_weight,
                longitudinal_loss_weight=longitudinal_loss_weight,
                perception_aux_loss_weight=perception_aux_loss_weight,
                road_mask_weight=road_mask_weight,
                lane_edge_weight=lane_edge_weight,
                lane_edge_pos_weight=lane_edge_pos_weight,
                visual_zero_margin=visual_zero_margin,
                visual_zero_steer_margin=visual_zero_steer_margin,
                visual_zero_route_min=visual_zero_route_min,
                visual_zero_route_max=visual_zero_route_max,
                visual_zero_abs_lane_offset_min=visual_zero_abs_lane_offset_min,
                visual_zero_abs_heading_error_min=visual_zero_abs_heading_error_min,
                visual_zero_corrective_steer_margin=visual_zero_corrective_steer_margin,
                visual_center_error_min=visual_center_error_min,
                visual_center_corrective_margin=visual_center_corrective_margin,
                route_aware_visual_center_route_min=route_aware_visual_center_route_min,
                route_aware_visual_center_route_max=route_aware_visual_center_route_max,
                route_aware_visual_center_margin=route_aware_visual_center_margin,
                collision_cluster_route_min=collision_cluster_route_min,
                collision_cluster_route_max=collision_cluster_route_max,
                collision_cluster_lane_offset_min=collision_cluster_lane_offset_min,
                collision_cluster_heading_error_max=collision_cluster_heading_error_max,
                collision_cluster_route_offset_margin=collision_cluster_route_offset_margin,
                collision_cluster_lane_offset_mode=collision_cluster_lane_offset_mode,
                collision_cluster_heading_error_mode=collision_cluster_heading_error_mode,
                early_route_offset_margin=early_route_offset_margin,
                early_route_min=early_route_min,
                early_route_max=early_route_max,
                early_route_lane_offset_min=early_route_lane_offset_min,
                early_route_heading_error_min=early_route_heading_error_min,
                early_route_lane_offset_mode=early_route_lane_offset_mode,
                early_route_heading_error_mode=early_route_heading_error_mode,
                early_severe_route_min=early_severe_route_min,
                early_severe_route_max=early_severe_route_max,
                early_severe_abs_lane_offset_min=early_severe_abs_lane_offset_min,
                early_severe_abs_heading_error_min=early_severe_abs_heading_error_min,
                early_severe_steer_margin=early_severe_steer_margin,
                early_severe_absolute_steer_margin=early_severe_absolute_steer_margin,
                early_severe_probe_absolute_steer_margin=early_severe_probe_absolute_steer_margin,
                early_severe_probe_axes=early_severe_probe_axes,
                early_severe_longitudinal_ceiling_route_min=early_severe_longitudinal_ceiling_route_min,
                early_severe_longitudinal_ceiling_route_max=early_severe_longitudinal_ceiling_route_max,
                early_severe_longitudinal_ceiling=early_severe_longitudinal_ceiling,
                late_recovery_route_min=late_recovery_route_min,
                late_recovery_route_max=late_recovery_route_max,
                late_recovery_abs_lane_offset_min=late_recovery_abs_lane_offset_min,
                late_recovery_abs_heading_error_min=late_recovery_abs_heading_error_min,
                late_recovery_absolute_steer_margin=late_recovery_absolute_steer_margin,
                late_recovery_bc_loss_scale=late_recovery_bc_loss_scale,
                early_center_zero_steer_route_min=early_center_zero_steer_route_min,
                early_center_zero_steer_route_max=early_center_zero_steer_route_max,
                early_center_zero_steer_abs_lane_offset_max=early_center_zero_steer_abs_lane_offset_max,
                early_center_zero_steer_abs_heading_error_max=early_center_zero_steer_abs_heading_error_max,
                early_severe_side_balanced=early_severe_side_balanced,
                early_severe_alignment_mode=early_severe_alignment_mode,
                route_progress_longitudinal_floor_route_min=route_progress_longitudinal_floor_route_min,
                route_progress_longitudinal_floor_route_max=route_progress_longitudinal_floor_route_max,
                route_progress_longitudinal_floor_speed_max_mps=route_progress_longitudinal_floor_speed_max_mps,
                route_progress_longitudinal_floor=route_progress_longitudinal_floor,
            )
            last_val_loss = float(val_losses["bc_loss"].detach().cpu().item())
            writer.writerow(
                _metric_row(
                    epoch=epoch,
                    global_step=global_step,
                    split="val",
                    losses=val_losses,
                    outputs=val_outputs,
                    model=model,
                )
            )
            handle.flush()
            checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            _save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                global_step=global_step,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                dataset_path=dataset_path,
            )
            print(f"[epoch] {epoch}/{epochs} train_loss={last_train_loss:.6f} val_loss={last_val_loss:.6f}")

    final_checkpoint = checkpoints_dir / "last.pt"
    _save_checkpoint(
        path=final_checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=epochs,
        global_step=global_step,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        dataset_path=dataset_path,
    )

    reloaded = RCSACFirstModel(config.model, config.fusion).to(device)
    checkpoint = torch.load(final_checkpoint, map_location=device)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    sample = next(iter(val_loader))
    sample = _batch_to_device(sample, device)
    with torch.no_grad():
        if sample["obs"].ndim == 3:
            action_mean, _ = _forward_sequence_outputs(
                reloaded,
                sample,
                use_auxiliary=use_auxiliary,
                track_diagnostics=False,
            )
        else:
            outputs = reloaded.forward_policy_outputs(
                sample["obs"],
                auxiliary=sample["auxiliary"] if use_auxiliary else None,
                hidden_state=None,
                track_diagnostics=False,
            )
            action_mean = outputs["action_mean"]
        if not isinstance(action_mean, torch.Tensor) or tuple(action_mean.shape[-1:]) != (2,):
            raise RuntimeError("Reloaded checkpoint failed dummy action forward.")
        if not torch.isfinite(action_mean).all():
            raise RuntimeError("Reloaded checkpoint produced non-finite actions.")

    print(f"[done] log={log_path} checkpoint={final_checkpoint} val_loss={last_val_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
