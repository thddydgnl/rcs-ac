from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rcs_ac_first.data.expert_dataset import (
    ExpertDataset,
    SequenceExpertDataset,
    load_expert_npz,
    summarize_expert_arrays,
)
from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.rl.losses import compute_dense_perception_aux_loss
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
    "visual_zero_action_l1",
    "visual_zero_steer_delta",
    "visual_zero_longitudinal_delta",
    "visual_zero_margin_loss",
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
) -> dict[str, torch.Tensor]:
    squared = (action_mean - expert_action).pow(2)
    bc_loss = _weighted_action_loss(
        squared,
        steer_loss_weight=steer_loss_weight,
        longitudinal_loss_weight=longitudinal_loss_weight,
    ).mean()
    steer_mse = F.mse_loss(action_mean[:, 0], expert_action[:, 0])
    longitudinal_mse = F.mse_loss(action_mean[:, 1], expert_action[:, 1])
    action_l1 = torch.mean(torch.abs(action_mean - expert_action))
    return {
        "bc_loss": bc_loss,
        "steer_mse": steer_mse,
        "longitudinal_mse": longitudinal_mse,
        "action_l1": action_l1,
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
    bc_loss = (weighted_squared * mask).sum() / denom
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
        "action_smooth_loss": smooth_loss,
    }


def _compute_visual_zero_losses(
    action_mean: torch.Tensor,
    zero_action_mean: torch.Tensor,
    *,
    visual_zero_margin: float = 0.0,
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
    return {
        "visual_zero_action_l1": action_l1,
        "visual_zero_steer_delta": steer_delta,
        "visual_zero_longitudinal_delta": longitudinal_delta,
        "visual_zero_margin_loss": margin_loss,
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
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
    model.eval()
    totals = {
        "bc_loss": 0.0,
        "steer_mse": 0.0,
        "longitudinal_mse": 0.0,
        "action_l1": 0.0,
        "visual_zero_action_l1": 0.0,
        "visual_zero_steer_delta": 0.0,
        "visual_zero_longitudinal_delta": 0.0,
        "visual_zero_margin_loss": 0.0,
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
                )
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
                        valid_mask=batch["valid_mask"],
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
                losses = _compute_losses(
                    action_mean,
                    batch["expert_action"],
                    steer_loss_weight=steer_loss_weight,
                    longitudinal_loss_weight=longitudinal_loss_weight,
                )
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
    road_mask_weight = float(bc_raw.get("road_mask_weight", config.loss.aux_road_mask_loss_weight))
    lane_edge_weight = float(bc_raw.get("lane_edge_weight", config.loss.aux_lane_edge_mask_loss_weight))
    lane_edge_pos_weight = float(bc_raw.get("lane_edge_pos_weight", config.loss.aux_lane_edge_pos_weight))
    trainable_module_names_raw = bc_raw.get("trainable_module_names", [])
    if isinstance(trainable_module_names_raw, str):
        trainable_module_names = [trainable_module_names_raw]
    else:
        trainable_module_names = [str(item) for item in trainable_module_names_raw]

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
                    losses = _compute_sequence_losses(
                        action_mean,
                        batch["expert_action"],
                        batch["valid_mask"],
                        action_smooth_loss_weight=action_smooth_loss_weight,
                        steer_loss_weight=steer_loss_weight,
                        longitudinal_loss_weight=longitudinal_loss_weight,
                    )
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
                            valid_mask=batch["valid_mask"],
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
                    losses = _compute_losses(
                        action_mean,
                        batch["expert_action"],
                        steer_loss_weight=steer_loss_weight,
                        longitudinal_loss_weight=longitudinal_loss_weight,
                    )
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
                        )
                    )
                total_loss = losses["bc_loss"] * bc_loss_weight
                total_loss = total_loss + losses.get("perception_aux_loss", total_loss.new_zeros(())) * perception_aux_loss_weight
                total_loss = (
                    total_loss
                    + losses.get("visual_zero_margin_loss", total_loss.new_zeros(())) * visual_zero_margin_loss_weight
                )
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Non-finite BC loss at step {global_step}: {float(total_loss.detach().cpu())}")
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_norm)
                optimizer.step()
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
