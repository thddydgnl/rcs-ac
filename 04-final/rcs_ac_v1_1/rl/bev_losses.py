from __future__ import annotations

import torch
import torch.nn.functional as F

BEV_CHANNEL_NAMES: tuple[str, ...] = ("drivable", "lane", "obstacle", "free", "route")
DEFAULT_BEV_LOSS_WEIGHTS: dict[str, float] = {
    "drivable": 1.0,
    "lane": 2.0,
    "obstacle": 3.0,
    "free": 1.0,
    "route": 1.5,
}


def compute_bev_loss(
    bev_logits: torch.Tensor,
    gt_bev_target: torch.Tensor,
    available: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if bev_logits.shape != gt_bev_target.shape or bev_logits.ndim != 4 or bev_logits.shape[1] != 5:
        raise ValueError(f"Expected bev logits/target [B,5,H,W], got {bev_logits.shape} and {gt_bev_target.shape}")
    weights = weights or DEFAULT_BEV_LOSS_WEIGHTS
    mask = available.to(device=bev_logits.device, dtype=bev_logits.dtype).reshape(-1)
    bce = F.binary_cross_entropy_with_logits(bev_logits, gt_bev_target.to(bev_logits), reduction="none")
    channel_loss = bce.mean(dim=(2, 3))
    denom = mask.sum().clamp_min(1.0)
    total = bev_logits.new_tensor(0.0)
    metrics: dict[str, torch.Tensor] = {"bev_target_available_rate": mask.mean()}
    for idx, name in enumerate(BEV_CHANNEL_NAMES):
        per_sample = channel_loss[:, idx]
        loss = (per_sample * mask).sum() / denom
        metrics[f"bev_{name}_loss"] = loss.detach()
        total = total + float(weights.get(name, 1.0)) * loss
        if name in {"drivable", "lane", "obstacle"}:
            metrics[f"bev_{name}_iou"] = _binary_iou(bev_logits[:, idx], gt_bev_target[:, idx].to(bev_logits), mask)
    metrics["bev_total_loss"] = total.detach()
    return total, metrics


def compute_waypoint_speed_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    waypoint_weight: float = 1.0,
    target_speed_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_wp = outputs["waypoints"]
    target_wp = targets["expert_waypoints"].to(pred_wp)
    pred_speed = outputs["target_speed_kmh"]
    target_speed = targets["expert_target_speed"].to(pred_speed)
    mask = targets["waypoint_target_available"].to(pred_wp).reshape(-1)
    if pred_wp.shape != target_wp.shape:
        raise ValueError(f"waypoint shape mismatch: {pred_wp.shape} vs {target_wp.shape}")
    if pred_speed.shape != target_speed.shape:
        raise ValueError(f"target speed shape mismatch: {pred_speed.shape} vs {target_speed.shape}")
    wp_l1 = F.smooth_l1_loss(pred_wp, target_wp, reduction="none").mean(dim=(1, 2))
    denom = mask.sum().clamp_min(1.0)
    waypoint_loss = (wp_l1 * mask).sum() / denom
    speed_loss = F.smooth_l1_loss(pred_speed, target_speed, reduction="mean")
    total = float(waypoint_weight) * waypoint_loss + float(target_speed_weight) * speed_loss
    abs_wp = torch.abs(pred_wp - target_wp)
    heading_error = torch.atan2(pred_wp[:, -1, 1], pred_wp[:, -1, 0]) - torch.atan2(
        target_wp[:, -1, 1], target_wp[:, -1, 0]
    )
    metrics = {
        "waypoint_l1": ((abs_wp.mean(dim=(1, 2)) * mask).sum() / denom).detach(),
        "waypoint_x_l1": ((abs_wp[:, :, 0].mean(dim=1) * mask).sum() / denom).detach(),
        "waypoint_y_l1": ((abs_wp[:, :, 1].mean(dim=1) * mask).sum() / denom).detach(),
        "waypoint_heading_error": ((torch.abs(torch.atan2(torch.sin(heading_error), torch.cos(heading_error))) * mask).sum() / denom).detach(),
        "target_speed_mae": torch.abs(pred_speed - target_speed).mean().detach(),
        "waypoint_target_available_rate": mask.mean().detach(),
        "waypoint_loss": waypoint_loss.detach(),
        "target_speed_loss": speed_loss.detach(),
    }
    return total, metrics


def _binary_iou(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits) > 0.5
    tgt = target > 0.5
    sample_mask = mask.reshape((-1,) + (1,) * (pred.ndim - 1)) > 0.5
    pred = pred & sample_mask
    tgt = tgt & sample_mask
    intersection = (pred & tgt).sum(dtype=torch.float32)
    union = (pred | tgt).sum(dtype=torch.float32)
    return intersection / union.clamp_min(1.0)
