from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_ppo_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    ratio = (new_log_probs - old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - float(clip_coef), 1.0 + float(clip_coef)) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def compute_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(values, returns)


def compute_entropy_loss(entropy: torch.Tensor) -> torch.Tensor:
    return -entropy.mean()


def compute_scalar_aux_loss(
    pred_lane_offset: torch.Tensor,
    target_lane_offset: torch.Tensor,
    pred_heading_error: torch.Tensor,
    target_heading_error: torch.Tensor,
    *,
    lane_offset_weight: float = 1.0,
    heading_error_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    lane_loss = F.smooth_l1_loss(pred_lane_offset, target_lane_offset)
    heading_loss = F.smooth_l1_loss(pred_heading_error, target_heading_error)
    total = float(lane_offset_weight) * lane_loss + float(heading_error_weight) * heading_loss
    metrics = {
        "aux_lane_offset_loss": float(lane_loss.detach().cpu().item()),
        "aux_heading_error_loss": float(heading_loss.detach().cpu().item()),
        "aux_lane_offset_mae": float((pred_lane_offset - target_lane_offset).detach().abs().mean().cpu().item()),
        "aux_heading_error_mae": float(
            (pred_heading_error - target_heading_error).detach().abs().mean().cpu().item()
        ),
    }
    return total, metrics


def compute_estimated_scalar_aux_loss(
    estimated_lane_offset: torch.Tensor | None,
    target_lane_offset: torch.Tensor,
    estimated_heading_error: torch.Tensor | None,
    target_heading_error: torch.Tensor,
    *,
    lane_offset_weight: float = 1.0,
    heading_error_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    reference = target_lane_offset
    if isinstance(estimated_lane_offset, torch.Tensor):
        reference = estimated_lane_offset
    elif isinstance(estimated_heading_error, torch.Tensor):
        reference = estimated_heading_error
    zero = reference.sum() * 0.0
    metrics = {
        "estimated_scalar_loss": 0.0,
        "estimated_lane_offset_loss": 0.0,
        "estimated_heading_error_loss": 0.0,
        "estimated_lane_offset_mae": 0.0,
        "estimated_heading_error_mae": 0.0,
        "estimated_lane_offset_sign_agreement": 0.0,
        "estimated_heading_error_sign_agreement": 0.0,
    }
    if estimated_lane_offset is None or estimated_heading_error is None:
        return zero, metrics

    lane = estimated_lane_offset.reshape(-1)
    heading = estimated_heading_error.reshape(-1)
    lane_target = target_lane_offset.to(device=lane.device, dtype=lane.dtype).reshape(-1)
    heading_target = target_heading_error.to(device=heading.device, dtype=heading.dtype).reshape(-1)
    if tuple(lane.shape) != tuple(lane_target.shape):
        raise ValueError(f"estimated lane target shape mismatch: {tuple(lane.shape)} vs {tuple(lane_target.shape)}")
    if tuple(heading.shape) != tuple(heading_target.shape):
        raise ValueError(
            f"estimated heading target shape mismatch: {tuple(heading.shape)} vs {tuple(heading_target.shape)}"
        )

    lane_loss = F.smooth_l1_loss(lane, lane_target)
    heading_loss = F.smooth_l1_loss(heading, heading_target)
    total = float(lane_offset_weight) * lane_loss + float(heading_error_weight) * heading_loss
    lane_sign_mask = (lane.detach().abs() > 1.0e-6) & (lane_target.detach().abs() > 1.0e-6)
    heading_sign_mask = (heading.detach().abs() > 1.0e-6) & (heading_target.detach().abs() > 1.0e-6)
    metrics.update(
        {
            "estimated_scalar_loss": float(total.detach().cpu().item()),
            "estimated_lane_offset_loss": float(lane_loss.detach().cpu().item()),
            "estimated_heading_error_loss": float(heading_loss.detach().cpu().item()),
            "estimated_lane_offset_mae": float((lane - lane_target).detach().abs().mean().cpu().item()),
            "estimated_heading_error_mae": float((heading - heading_target).detach().abs().mean().cpu().item()),
        }
    )
    if bool(lane_sign_mask.any().item()):
        metrics["estimated_lane_offset_sign_agreement"] = float(
            (torch.sign(lane[lane_sign_mask]) == torch.sign(lane_target[lane_sign_mask]))
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .cpu()
            .item()
        )
    if bool(heading_sign_mask.any().item()):
        metrics["estimated_heading_error_sign_agreement"] = float(
            (torch.sign(heading[heading_sign_mask]) == torch.sign(heading_target[heading_sign_mask]))
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .cpu()
            .item()
        )
    return total, metrics


def compute_dense_perception_aux_loss(
    pred_road_mask_logits: torch.Tensor,
    road_mask_target: torch.Tensor,
    pred_lane_edge_mask_logits: torch.Tensor,
    lane_edge_mask_target: torch.Tensor,
    mask_targets_available: torch.Tensor,
    *,
    road_mask_weight: float = 0.25,
    lane_edge_weight: float = 1.0,
    lane_edge_pos_weight: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_dense_aux_shapes(
        pred_road_mask_logits,
        road_mask_target,
        pred_lane_edge_mask_logits,
        lane_edge_mask_target,
        mask_targets_available,
    )
    availability = mask_targets_available.to(
        device=pred_road_mask_logits.device,
        dtype=pred_road_mask_logits.dtype,
    ).reshape(-1)
    valid = availability > 0.5
    available_rate = float(valid.to(dtype=torch.float32).mean().detach().cpu().item())
    metrics = {
        "aux_road_mask_loss": 0.0,
        "aux_lane_edge_mask_loss": 0.0,
        "aux_road_mask_mae": 0.0,
        "aux_lane_edge_mask_mae": 0.0,
        "aux_road_mask_iou": 0.0,
        "aux_lane_edge_mask_iou": 0.0,
        "aux_road_mask_positive_ratio": 0.0,
        "aux_lane_edge_positive_ratio": 0.0,
        "aux_mask_available_rate": available_rate,
    }
    if not bool(valid.any()):
        zero = pred_road_mask_logits.sum() * 0.0 + pred_lane_edge_mask_logits.sum() * 0.0
        return zero, metrics

    road_logits = pred_road_mask_logits[valid]
    lane_logits = pred_lane_edge_mask_logits[valid]
    road_target = road_mask_target.to(device=road_logits.device, dtype=road_logits.dtype)[valid].clamp(0.0, 1.0)
    lane_target = lane_edge_mask_target.to(device=lane_logits.device, dtype=lane_logits.dtype)[valid].clamp(0.0, 1.0)

    road_loss = F.binary_cross_entropy_with_logits(road_logits, road_target)
    lane_pos_weight = torch.as_tensor(
        [float(lane_edge_pos_weight)],
        dtype=lane_logits.dtype,
        device=lane_logits.device,
    )
    lane_loss = F.binary_cross_entropy_with_logits(
        lane_logits,
        lane_target,
        pos_weight=lane_pos_weight,
    )
    total = float(road_mask_weight) * road_loss + float(lane_edge_weight) * lane_loss

    road_prob = torch.sigmoid(road_logits.detach())
    lane_prob = torch.sigmoid(lane_logits.detach())
    metrics.update(
        {
            "aux_road_mask_loss": float(road_loss.detach().cpu().item()),
            "aux_lane_edge_mask_loss": float(lane_loss.detach().cpu().item()),
            "aux_road_mask_mae": float((road_prob - road_target.detach()).abs().mean().cpu().item()),
            "aux_lane_edge_mask_mae": float((lane_prob - lane_target.detach()).abs().mean().cpu().item()),
            "aux_road_mask_iou": _binary_iou(road_prob, road_target.detach()),
            "aux_lane_edge_mask_iou": _binary_iou(lane_prob, lane_target.detach()),
            "aux_road_mask_positive_ratio": float(road_target.detach().mean().cpu().item()),
            "aux_lane_edge_positive_ratio": float(lane_target.detach().mean().cpu().item()),
        }
    )
    return total, metrics


def _validate_dense_aux_shapes(
    pred_road_mask_logits: torch.Tensor,
    road_mask_target: torch.Tensor,
    pred_lane_edge_mask_logits: torch.Tensor,
    lane_edge_mask_target: torch.Tensor,
    mask_targets_available: torch.Tensor,
) -> None:
    expected = tuple(pred_road_mask_logits.shape)
    if pred_road_mask_logits.ndim != 4 or expected[1] != 1:
        raise ValueError(f"pred_road_mask_logits must have shape [B, 1, H, W], got {expected}")
    if tuple(pred_lane_edge_mask_logits.shape) != expected:
        raise ValueError(
            "pred_lane_edge_mask_logits shape mismatch: "
            f"expected {expected}, got {tuple(pred_lane_edge_mask_logits.shape)}"
        )
    if tuple(road_mask_target.shape) != expected:
        raise ValueError(f"road_mask_target shape mismatch: expected {expected}, got {tuple(road_mask_target.shape)}")
    if tuple(lane_edge_mask_target.shape) != expected:
        raise ValueError(
            f"lane_edge_mask_target shape mismatch: expected {expected}, got {tuple(lane_edge_mask_target.shape)}"
        )
    if int(mask_targets_available.reshape(-1).shape[0]) != int(expected[0]):
        raise ValueError(
            "mask_targets_available length mismatch: "
            f"expected {expected[0]}, got {int(mask_targets_available.reshape(-1).shape[0])}"
        )


def _binary_iou(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    pred_mask = probabilities >= 0.5
    target_mask = targets >= 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum().to(dtype=torch.float32)
    union = torch.logical_or(pred_mask, target_mask).sum().to(dtype=torch.float32)
    if float(union.detach().cpu().item()) <= 0.0:
        return 1.0
    return float((intersection / union.clamp_min(1.0)).detach().cpu().item())


def zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def compute_reliability_supervision_loss(
    reliability: torch.Tensor,
    reliability_target: torch.Tensor,
) -> torch.Tensor:
    if tuple(reliability.shape) != tuple(reliability_target.shape):
        raise ValueError(
            f"reliability target shape mismatch: expected {tuple(reliability.shape)}, got {tuple(reliability_target.shape)}"
        )
    return F.mse_loss(reliability, reliability_target.to(device=reliability.device, dtype=reliability.dtype).clamp(0.0, 1.0))


def compute_risk_supervision_loss(
    risk_scalar: torch.Tensor,
    risk_target: torch.Tensor,
) -> torch.Tensor:
    risk = risk_scalar.reshape(-1)
    target = risk_target.to(device=risk.device, dtype=risk.dtype).reshape(-1).clamp(0.0, 1.0)
    if tuple(risk.shape) != tuple(target.shape):
        raise ValueError(f"risk target shape mismatch: expected {tuple(risk.shape)}, got {tuple(target.shape)}")
    return F.mse_loss(risk, target)


def compute_risk_aware_regularization(
    fusion_weights: torch.Tensor,
    reliability: torch.Tensor,
    risk_scalar: torch.Tensor,
) -> torch.Tensor:
    if tuple(fusion_weights.shape) != tuple(reliability.shape):
        raise ValueError(
            f"fusion/reliability shape mismatch: {tuple(fusion_weights.shape)} vs {tuple(reliability.shape)}"
        )
    risk = risk_scalar.reshape(-1, 1).to(device=fusion_weights.device, dtype=fusion_weights.dtype).clamp(0.0, 1.0)
    low_reliability = (1.0 - reliability.detach().to(dtype=fusion_weights.dtype)).clamp(0.0, 1.0)
    return (risk.detach() * low_reliability * fusion_weights).sum(dim=-1).mean()


def compute_reliability_loss(reference: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
    del args, kwargs
    return zero_like_loss(reference)


def compute_risk_regularization(reference: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
    del args, kwargs
    return zero_like_loss(reference)


def compute_shapley_alignment_loss(
    fusion_weights: torch.Tensor,
    contribution: torch.Tensor,
    *,
    detach_target: bool = True,
) -> torch.Tensor:
    if tuple(fusion_weights.shape) != tuple(contribution.shape):
        raise ValueError(
            f"fusion/contribution shape mismatch: {tuple(fusion_weights.shape)} vs {tuple(contribution.shape)}"
        )
    target = contribution.detach() if bool(detach_target) else contribution
    target = target.to(device=fusion_weights.device, dtype=fusion_weights.dtype)
    target = target.clamp_min(1.0e-8)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return F.mse_loss(fusion_weights, target)


def compute_dropout_consistency_loss(
    clean_action_mean: torch.Tensor,
    drop_action_mean: torch.Tensor,
    clean_value: torch.Tensor | None = None,
    drop_value: torch.Tensor | None = None,
    *,
    action_weight: float = 1.0,
    value_weight: float = 0.25,
    detach_clean: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(clean_action_mean.shape) != tuple(drop_action_mean.shape):
        raise ValueError(
            f"action mean shape mismatch: {tuple(clean_action_mean.shape)} vs {tuple(drop_action_mean.shape)}"
        )
    clean_action = clean_action_mean.detach() if bool(detach_clean) else clean_action_mean
    action_loss = F.mse_loss(drop_action_mean, clean_action.to(device=drop_action_mean.device, dtype=drop_action_mean.dtype))
    total = float(action_weight) * action_loss
    metrics = {
        "dropout_action_l1": float((drop_action_mean.detach() - clean_action.detach()).abs().mean().cpu().item()),
        "dropout_action_mse": float(action_loss.detach().cpu().item()),
        "dropout_value_l1": 0.0,
        "dropout_value_mse": 0.0,
    }
    if clean_value is not None and drop_value is not None:
        if tuple(clean_value.shape) != tuple(drop_value.shape):
            raise ValueError(f"value shape mismatch: {tuple(clean_value.shape)} vs {tuple(drop_value.shape)}")
        clean_v = clean_value.detach() if bool(detach_clean) else clean_value
        value_loss = F.mse_loss(drop_value, clean_v.to(device=drop_value.device, dtype=drop_value.dtype))
        total = total + float(value_weight) * value_loss
        metrics["dropout_value_l1"] = float((drop_value.detach() - clean_v.detach()).abs().mean().cpu().item())
        metrics["dropout_value_mse"] = float(value_loss.detach().cpu().item())
    metrics["dropout_consistency_loss"] = float(total.detach().cpu().item())
    return total, metrics
