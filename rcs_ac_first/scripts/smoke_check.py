from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
import numpy as np

from rcs_ac_first.models import RCSACFirstModel
from rcs_ac_first.models.masked_critic import MaskedCritic, MaskedCriticConfig
from rcs_ac_first.rl.distributions import BoundedSquashedNormal
from rcs_ac_first.rl.losses import (
    compute_dense_perception_aux_loss,
    compute_dropout_consistency_loss,
    compute_entropy_loss,
    compute_ppo_actor_loss,
    compute_reliability_supervision_loss,
    compute_risk_aware_regularization,
    compute_risk_supervision_loss,
    compute_scalar_aux_loss,
    compute_shapley_alignment_loss,
    compute_value_loss,
)
from rcs_ac_first.rl.ppo_agent import PPOAgent, PPOConfig
from rcs_ac_first.rl.rollout_buffer import RolloutBuffer
from rcs_ac_first.rl.shapley import ShapleyEstimatorConfig, estimate_exact_shapley_current_critic
from rcs_ac_first.runtime.auxiliary import (
    QUALITY_STATS_FIELDS,
    RISK_STATS_FIELDS,
    SENSOR_DROPOUT_KEYS,
    build_auxiliary_observation_spec,
    default_flat_auxiliary,
)
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.modality_mask import single_modality_drop_masks
from rcs_ac_first.runtime.phase2_targets import build_phase2_targets_from_info


def _default_config_path() -> Path:
    root_config = PROJECT_ROOT / "configs" / "phase1_core_smoke.yaml"
    if root_config.exists():
        return root_config
    return PROJECT_ROOT / "rcs_ac_first" / "configs" / "phase1_core_smoke.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCS-AC-first Phase 1 local smoke checks.")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--only-model", action="store_true")
    parser.add_argument("--only-loss", action="store_true")
    parser.add_argument("--only-checkpoint", action="store_true")
    return parser.parse_args()


def _finite_all(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or inf values.")


def _build_dummy_obs(batch_size: int, obs_dim: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    obs = torch.randn((batch_size, obs_dim), generator=generator, dtype=torch.float32) * 0.05
    return obs.to(device=device)


def _build_dummy_auxiliary(
    batch_size: int,
    device: torch.device,
    *,
    corrupt_modality: str | None = None,
) -> torch.Tensor:
    spec = build_auxiliary_observation_spec()
    aux = default_flat_auxiliary(batch_size, spec=spec, device=device)
    quality = spec.item("quality_stats")
    risk = spec.item("risk_stats")
    dropout = spec.item("sensor_dropout_mask")
    aux[:, risk.start + RISK_STATS_FIELDS.index("speed_mps")] = 8.0
    aux[:, risk.start + RISK_STATS_FIELDS.index("target_distance")] = 12.0
    if corrupt_modality is None:
        return aux
    if corrupt_modality not in SENSOR_DROPOUT_KEYS:
        raise ValueError(f"Unknown corrupt modality {corrupt_modality!r}")
    aux[:, dropout.start + SENSOR_DROPOUT_KEYS.index(corrupt_modality)] = 1.0
    aux[:, risk.start + RISK_STATS_FIELDS.index("lane_offset")] = 1.0
    aux[:, risk.start + RISK_STATS_FIELDS.index("heading_error")] = 25.0
    if corrupt_modality == "rgb":
        aux[:, quality.start + QUALITY_STATS_FIELDS.index("rgb_dropout_flag")] = 1.0
    elif corrupt_modality == "lidar":
        aux[:, quality.start + QUALITY_STATS_FIELDS.index("lidar_drop_ratio")] = 1.0
    elif corrupt_modality == "ego":
        aux[:, quality.start + QUALITY_STATS_FIELDS.index("ego_jerk_anomaly")] = 1.0
    elif corrupt_modality == "route":
        aux[:, quality.start + QUALITY_STATS_FIELDS.index("route_heading_mismatch")] = 35.0
    return aux


def _build_dummy_masks(batch_size: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    road = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=device)
    lane = torch.zeros_like(road)
    road[:, :, height // 2 :, :] = 1.0
    left = torch.linspace(width * 0.35, width * 0.45, height, device=device).round().to(dtype=torch.long)
    right = torch.linspace(width * 0.65, width * 0.55, height, device=device).round().to(dtype=torch.long)
    rows = torch.arange(height, device=device)
    lane[:, :, rows, left.clamp(0, width - 1)] = 1.0
    lane[:, :, rows, right.clamp(0, width - 1)] = 1.0
    return road, lane


def run_model_check(model: RCSACFirstModel, obs: torch.Tensor, auxiliary: torch.Tensor) -> dict[str, object]:
    outputs = model.forward_policy_outputs(obs, auxiliary=auxiliary)
    aux = model.predict_auxiliary_targets(obs, auxiliary=auxiliary)
    action_mean = outputs["action_mean"]
    value = outputs["value"]
    fusion_weights = outputs["fusion_weights"]
    reliability = outputs["reliability"]
    risk_scalar = outputs["risk_scalar"]
    risk_context = outputs["risk_context"]
    contribution = outputs["contribution"]
    visual_features = outputs["rgb_perception_actor_features"]
    lane = aux["pred_lane_offset"]
    heading = aux["pred_heading_error"]
    road_mask_logits = aux["pred_road_mask_logits"]
    lane_edge_logits = aux["pred_lane_edge_mask_logits"]
    assert isinstance(action_mean, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert isinstance(fusion_weights, torch.Tensor)
    assert isinstance(reliability, torch.Tensor)
    assert isinstance(risk_scalar, torch.Tensor)
    assert isinstance(risk_context, torch.Tensor)
    assert isinstance(contribution, torch.Tensor)
    assert isinstance(visual_features, torch.Tensor)
    if tuple(action_mean.shape) != (int(obs.shape[0]), 2):
        raise RuntimeError(f"action_mean shape mismatch: {tuple(action_mean.shape)}")
    if tuple(value.shape) != (int(obs.shape[0]),):
        raise RuntimeError(f"value shape mismatch: {tuple(value.shape)}")
    if tuple(fusion_weights.shape) != (int(obs.shape[0]), 4):
        raise RuntimeError(f"fusion_weights shape mismatch: {tuple(fusion_weights.shape)}")
    if tuple(reliability.shape) != (int(obs.shape[0]), 4):
        raise RuntimeError(f"reliability shape mismatch: {tuple(reliability.shape)}")
    if tuple(risk_scalar.shape) != (int(obs.shape[0]),):
        raise RuntimeError(f"risk_scalar shape mismatch: {tuple(risk_scalar.shape)}")
    if tuple(risk_context.shape) != (int(obs.shape[0]), int(model.config.risk_context_dim)):
        raise RuntimeError(f"risk_context shape mismatch: {tuple(risk_context.shape)}")
    if tuple(contribution.shape) != (int(obs.shape[0]), 4):
        raise RuntimeError(f"contribution shape mismatch: {tuple(contribution.shape)}")
    expected_visual_shape = (int(obs.shape[0]), int(model.config.rgb_perception_actor_feature_dim))
    if tuple(visual_features.shape) != expected_visual_shape:
        raise RuntimeError(
            f"rgb_perception_actor_features shape mismatch: expected {expected_visual_shape}, "
            f"got {tuple(visual_features.shape)}"
        )
    if tuple(lane.shape) != (int(obs.shape[0]),) or tuple(heading.shape) != (int(obs.shape[0]),):
        raise RuntimeError("auxiliary prediction shape mismatch")
    expected_mask_shape = (
        int(obs.shape[0]),
        1,
        int(model.config.aux_mask_height),
        int(model.config.aux_mask_width),
    )
    if tuple(road_mask_logits.shape) != expected_mask_shape:
        raise RuntimeError(f"road mask logits shape mismatch: {tuple(road_mask_logits.shape)}")
    if tuple(lane_edge_logits.shape) != expected_mask_shape:
        raise RuntimeError(f"lane edge logits shape mismatch: {tuple(lane_edge_logits.shape)}")
    _finite_all("action_mean", action_mean)
    _finite_all("value", value)
    _finite_all("fusion_weights", fusion_weights)
    _finite_all("reliability", reliability)
    _finite_all("risk_scalar", risk_scalar)
    _finite_all("risk_context", risk_context)
    _finite_all("contribution", contribution)
    _finite_all("rgb_perception_actor_features", visual_features)
    _finite_all("road_mask_logits", road_mask_logits)
    _finite_all("lane_edge_logits", lane_edge_logits)
    if not torch.allclose(fusion_weights.sum(dim=-1), torch.ones_like(fusion_weights[:, 0]), atol=1.0e-5):
        raise RuntimeError("fusion weights do not sum to 1")
    if not torch.allclose(contribution.sum(dim=-1), torch.ones_like(contribution[:, 0]), atol=1.0e-5):
        raise RuntimeError("contribution rows do not sum to 1")
    visual_delta = 0.0
    if bool(model.config.use_rgb_perception_actor_features):
        zero_outputs = model.forward_policy_outputs(
            obs,
            auxiliary=auxiliary,
            zero_rgb_perception_actor_features=True,
            track_diagnostics=False,
        )
        zero_action = zero_outputs["action_mean"]
        assert isinstance(zero_action, torch.Tensor)
        _finite_all("zero_visual_action_mean", zero_action)
        visual_delta = float((action_mean.detach() - zero_action.detach()).abs().mean().cpu().item())
    return {
        "action_mean_shape": list(action_mean.shape),
        "value_shape": list(value.shape),
        "fusion_weights_shape": list(fusion_weights.shape),
        "auxiliary_shape": list(auxiliary.shape),
        "reliability_shape": list(reliability.shape),
        "risk_scalar_shape": list(risk_scalar.shape),
        "risk_context_shape": list(risk_context.shape),
        "contribution_shape": list(contribution.shape),
        "road_mask_logits_shape": list(road_mask_logits.shape),
        "lane_edge_logits_shape": list(lane_edge_logits.shape),
        "rgb_perception_actor_features_shape": list(visual_features.shape),
        "visual_feature_zero_action_l1_mean": round(float(visual_delta), 8),
        "compact_visual_features_enabled": bool(model.config.use_rgb_perception_actor_features),
        "fusion_weight_sums": [round(float(v), 6) for v in fusion_weights.sum(dim=-1).detach().cpu()],
        "contribution_sums": [round(float(v), 6) for v in contribution.sum(dim=-1).detach().cpu()],
        "std": [round(float(v), 6) for v in model.get_std().detach().cpu()],
    }


def run_loss_check(model: RCSACFirstModel, obs: torch.Tensor, auxiliary: torch.Tensor) -> dict[str, float]:
    outputs = model.forward_policy_outputs(obs, auxiliary=auxiliary)
    action_mean = outputs["action_mean"]
    value = outputs["value"]
    assert isinstance(action_mean, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    dist = BoundedSquashedNormal(action_mean, model.get_std(), [(-1.0, 1.0), (-1.0, 1.0)])
    actions = dist.rsample()
    old_log_probs = dist.log_prob(actions).detach()
    new_log_probs = dist.log_prob(actions)
    advantages = torch.randn_like(old_log_probs)
    returns = value.detach() + torch.randn_like(value) * 0.1
    lane_targets = torch.zeros_like(outputs["pred_lane_offset"])  # type: ignore[arg-type]
    heading_targets = torch.zeros_like(outputs["pred_heading_error"])  # type: ignore[arg-type]
    road_targets, lane_edge_targets = _build_dummy_masks(
        int(obs.shape[0]),
        int(model.config.aux_mask_height),
        int(model.config.aux_mask_width),
        obs.device,
    )
    available = torch.ones((int(obs.shape[0]),), dtype=torch.float32, device=obs.device)
    actor_loss = compute_ppo_actor_loss(new_log_probs, old_log_probs, advantages, 0.2)
    value_loss = compute_value_loss(value, returns)
    entropy_loss = compute_entropy_loss(dist.entropy())
    aux_loss, aux_metrics = compute_scalar_aux_loss(
        outputs["pred_lane_offset"],  # type: ignore[arg-type]
        lane_targets,
        outputs["pred_heading_error"],  # type: ignore[arg-type]
        heading_targets,
    )
    dense_aux_loss, dense_aux_metrics = compute_dense_perception_aux_loss(
        outputs["pred_road_mask_logits"],  # type: ignore[arg-type]
        road_targets,
        outputs["pred_lane_edge_mask_logits"],  # type: ignore[arg-type]
        lane_edge_targets,
        available,
    )
    unavailable_loss, unavailable_metrics = compute_dense_perception_aux_loss(
        outputs["pred_road_mask_logits"],  # type: ignore[arg-type]
        road_targets,
        outputs["pred_lane_edge_mask_logits"],  # type: ignore[arg-type]
        lane_edge_targets,
        torch.zeros_like(available),
    )
    if float(unavailable_loss.detach().cpu().item()) != 0.0:
        raise RuntimeError("dense aux loss must be zero when mask targets are unavailable")
    if unavailable_metrics["aux_mask_available_rate"] != 0.0:
        raise RuntimeError("dense aux availability rate should be zero for unavailable smoke")
    reliability_targets = outputs["reliability"].detach().clone()  # type: ignore[union-attr]
    risk_targets = outputs["risk_scalar"].detach().clone()  # type: ignore[union-attr]
    reliability_loss = compute_reliability_supervision_loss(
        outputs["reliability"],  # type: ignore[arg-type]
        reliability_targets,
    )
    risk_supervision_loss = compute_risk_supervision_loss(
        outputs["risk_scalar"],  # type: ignore[arg-type]
        risk_targets,
    )
    risk_regularization = compute_risk_aware_regularization(
        outputs["fusion_weights"],  # type: ignore[arg-type]
        outputs["reliability"],  # type: ignore[arg-type]
        outputs["risk_scalar"],  # type: ignore[arg-type]
    )
    shapley_alignment_loss = compute_shapley_alignment_loss(
        outputs["fusion_weights"],  # type: ignore[arg-type]
        outputs["contribution"],  # type: ignore[arg-type]
    )
    mask = single_modality_drop_masks(int(obs.shape[0]), device=obs.device, dtype=obs.dtype)
    drop_outputs = model.forward_policy_outputs(
        obs,
        auxiliary=auxiliary,
        contribution_override=outputs["contribution"].detach(),  # type: ignore[union-attr]
        modality_mask=mask,
        track_diagnostics=False,
    )
    dropout_consistency_loss, dropout_metrics = compute_dropout_consistency_loss(
        outputs["action_mean"],  # type: ignore[arg-type]
        drop_outputs["action_mean"],  # type: ignore[arg-type]
        outputs["value"],  # type: ignore[arg-type]
        drop_outputs["value"],  # type: ignore[arg-type]
    )
    total = (
        actor_loss
        + 0.5 * value_loss
        + 0.01 * entropy_loss
        + aux_loss
        + dense_aux_loss
        + reliability_loss
        + risk_supervision_loss
        + 0.1 * risk_regularization
        + shapley_alignment_loss
        + dropout_consistency_loss
    )
    if not torch.isfinite(total):
        raise RuntimeError("total loss is not finite")
    model.zero_grad(set_to_none=True)
    total.backward()
    finite_grad_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("model gradient contains NaN or inf")
        finite_grad_count += 1
    if finite_grad_count == 0:
        raise RuntimeError("loss backward produced no gradients")
    metrics = {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(value_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
        "finite_grad_count": float(finite_grad_count),
        "reliability_loss": float(reliability_loss.detach().cpu().item()),
        "risk_supervision_loss": float(risk_supervision_loss.detach().cpu().item()),
        "risk_regularization": float(risk_regularization.detach().cpu().item()),
        "shapley_alignment_loss": float(shapley_alignment_loss.detach().cpu().item()),
        "dropout_consistency_loss": float(dropout_consistency_loss.detach().cpu().item()),
    }
    metrics.update(aux_metrics)
    metrics.update(dense_aux_metrics)
    metrics.update(dropout_metrics)
    metrics["dense_unavailable_loss"] = float(unavailable_loss.detach().cpu().item())
    return metrics


def run_checkpoint_check(config_path: Path, device_override: str | None) -> dict[str, object]:
    config = load_phase1_config(config_path)
    agent = PPOAgent(PPOConfig.from_phase1_config(config, device_override=device_override))
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "smoke_check"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "last.pt"
    agent.save(str(checkpoint_path), extra_state={"smoke_check": True, "global_step": 0})
    loaded = PPOAgent(PPOConfig.from_phase1_config(config, device_override=device_override))
    checkpoint = loaded.load(str(checkpoint_path), map_location=agent.device)
    obs = _build_dummy_obs(batch_size=2, obs_dim=config.model.obs_dim or 0, device=agent.device)
    outputs = loaded.model.forward_policy_outputs(obs)
    _finite_all("loaded action_mean", outputs["action_mean"])  # type: ignore[arg-type]
    return {
        "checkpoint_path": str(checkpoint_path),
        "extra_state": checkpoint.get("extra_state", {}),
    }


def run_forced_corruption_check(model: RCSACFirstModel, obs: torch.Tensor) -> dict[str, object]:
    if not bool(model.config.enable_reliability):
        return {"skipped": True, "reason": "enable_reliability is false"}
    clean_aux = _build_dummy_auxiliary(int(obs.shape[0]), obs.device)
    rgb_aux = _build_dummy_auxiliary(int(obs.shape[0]), obs.device, corrupt_modality="rgb")
    lidar_aux = _build_dummy_auxiliary(int(obs.shape[0]), obs.device, corrupt_modality="lidar")
    clean = model.forward_policy_outputs(obs, auxiliary=clean_aux, track_diagnostics=False)
    rgb = model.forward_policy_outputs(obs, auxiliary=rgb_aux, track_diagnostics=False)
    lidar = model.forward_policy_outputs(obs, auxiliary=lidar_aux, track_diagnostics=False)
    clean_rel = clean["reliability"]
    rgb_rel = rgb["reliability"]
    lidar_rel = lidar["reliability"]
    clean_weights = clean["fusion_weights"]
    rgb_weights = rgb["fusion_weights"]
    lidar_weights = lidar["fusion_weights"]
    assert isinstance(clean_rel, torch.Tensor)
    assert isinstance(rgb_rel, torch.Tensor)
    assert isinstance(lidar_rel, torch.Tensor)
    assert isinstance(clean_weights, torch.Tensor)
    assert isinstance(rgb_weights, torch.Tensor)
    assert isinstance(lidar_weights, torch.Tensor)
    if float(rgb_rel[:, 0].mean().detach().cpu().item()) >= float(clean_rel[:, 0].mean().detach().cpu().item()):
        raise RuntimeError("RGB corruption did not lower RGB reliability")
    if float(lidar_rel[:, 1].mean().detach().cpu().item()) >= float(clean_rel[:, 1].mean().detach().cpu().item()):
        raise RuntimeError("LiDAR corruption did not lower LiDAR reliability")
    if float(rgb_weights[:, 0].mean().detach().cpu().item()) >= float(clean_weights[:, 0].mean().detach().cpu().item()):
        raise RuntimeError("RGB corruption did not lower RGB fusion weight")
    if float(lidar_weights[:, 1].mean().detach().cpu().item()) >= float(clean_weights[:, 1].mean().detach().cpu().item()):
        raise RuntimeError("LiDAR corruption did not lower LiDAR fusion weight")
    return {
        "clean_reliability_rgb": round(float(clean_rel[:, 0].mean().detach().cpu().item()), 6),
        "corrupt_reliability_rgb": round(float(rgb_rel[:, 0].mean().detach().cpu().item()), 6),
        "clean_fusion_rgb": round(float(clean_weights[:, 0].mean().detach().cpu().item()), 6),
        "corrupt_fusion_rgb": round(float(rgb_weights[:, 0].mean().detach().cpu().item()), 6),
        "clean_reliability_lidar": round(float(clean_rel[:, 1].mean().detach().cpu().item()), 6),
        "corrupt_reliability_lidar": round(float(lidar_rel[:, 1].mean().detach().cpu().item()), 6),
        "clean_fusion_lidar": round(float(clean_weights[:, 1].mean().detach().cpu().item()), 6),
        "corrupt_fusion_lidar": round(float(lidar_weights[:, 1].mean().detach().cpu().item()), 6),
    }


def run_phase3_shapley_check(model: RCSACFirstModel, obs: torch.Tensor, auxiliary: torch.Tensor) -> dict[str, object]:
    if not bool(model.config.enable_shapley):
        return {"skipped": True, "reason": "enable_shapley is false"}
    base = model.forward_policy_outputs(obs, auxiliary=auxiliary, return_shapley_context=True, track_diagnostics=False)
    estimate = estimate_exact_shapley_current_critic(
        model,
        projected_feature_stack=base["projected_feature_stack"],  # type: ignore[arg-type]
        fusion_weights=base["fusion_weights"],  # type: ignore[arg-type]
        hidden_state=None,
        config=ShapleyEstimatorConfig(contribution_temperature=1.0, min_contribution=0.02),
    )
    _finite_all("shapley_raw", estimate.raw)
    _finite_all("shapley_contribution", estimate.contribution)
    if tuple(estimate.contribution.shape) != (int(obs.shape[0]), 4):
        raise RuntimeError(f"Shapley contribution shape mismatch: {tuple(estimate.contribution.shape)}")
    if not torch.allclose(estimate.contribution.sum(dim=-1), torch.ones_like(estimate.contribution[:, 0]), atol=1.0e-5):
        raise RuntimeError("Shapley contribution rows do not sum to 1")
    final = model.forward_policy_outputs(
        obs,
        auxiliary=auxiliary,
        contribution_override=estimate.contribution,
        track_diagnostics=False,
    )
    alignment = compute_shapley_alignment_loss(
        final["fusion_weights"],  # type: ignore[arg-type]
        estimate.contribution,
    )
    mask = single_modality_drop_masks(int(obs.shape[0]), device=obs.device, dtype=obs.dtype)
    drop = model.forward_policy_outputs(
        obs,
        auxiliary=auxiliary,
        contribution_override=estimate.contribution,
        modality_mask=mask,
        track_diagnostics=False,
    )
    consistency, consistency_metrics = compute_dropout_consistency_loss(
        final["action_mean"],  # type: ignore[arg-type]
        drop["action_mean"],  # type: ignore[arg-type]
        final["value"],  # type: ignore[arg-type]
        drop["value"],  # type: ignore[arg-type]
    )
    masked_critic = MaskedCritic(
        MaskedCriticConfig(
            fusion_dim=int(model.config.fusion_dim),
            risk_context_dim=int(model.config.risk_context_dim),
            hidden_dim=int(model.config.critic_hidden_dim),
        )
    ).to(obs.device)
    critic_value = masked_critic(
        final["fusion_weights"].new_zeros((int(obs.shape[0]), int(model.config.fusion_dim))),  # type: ignore[union-attr]
        mask,
        final["reliability"],  # type: ignore[arg-type]
        final["risk_scalar"],  # type: ignore[arg-type]
        final["risk_context"],  # type: ignore[arg-type]
    )
    _finite_all("masked_critic_value", critic_value)
    return {
        "raw_shape": list(estimate.raw.shape),
        "contribution_shape": list(estimate.contribution.shape),
        "contribution_sums": [round(float(v), 6) for v in estimate.contribution.sum(dim=-1).detach().cpu()],
        "subset_count": int(estimate.subset_count),
        "alignment_loss": round(float(alignment.detach().cpu().item()), 8),
        "dropout_consistency_loss": round(float(consistency.detach().cpu().item()), 8),
        "masked_critic_shape": list(critic_value.shape),
        "dropout_action_l1": round(float(consistency_metrics["dropout_action_l1"]), 8),
    }


def run_recurrent_rollout_update_check(config_path: Path, device_override: str | None) -> dict[str, object]:
    config = load_phase1_config(config_path)
    config.train.rollout_steps = 8
    config.train.minibatch_size = 4
    config.train.update_epochs = 1
    agent = PPOAgent(PPOConfig.from_phase1_config(config, device_override=device_override))
    obs_dim = int(config.model.obs_dim or 0)
    buffer = RolloutBuffer(
        capacity=config.train.rollout_steps,
        obs_dim=obs_dim,
        action_dim=config.model.action_dim,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        aux_mask_shape=(1, config.model.aux_mask_height, config.model.aux_mask_width),
        hidden_state_shape=(config.model.recurrent_layers, config.model.recurrent_hidden_dim),
        auxiliary_dim=config.model.auxiliary_dim,
    )
    generator = np.random.default_rng(int(config.train.seed))
    road_mask = np.zeros((1, config.model.aux_mask_height, config.model.aux_mask_width), dtype=np.float32)
    lane_mask = np.zeros_like(road_mask)
    road_mask[:, config.model.aux_mask_height // 2 :, :] = 1.0
    lane_mask[:, :, config.model.aux_mask_width // 2] = 1.0

    hidden_l1_means: list[float] = []
    for index in range(int(config.train.rollout_steps)):
        obs = generator.normal(0.0, 0.05, size=(obs_dim,)).astype(np.float32)
        auxiliary = _build_dummy_auxiliary(1, agent.device).squeeze(0).detach().cpu().numpy().astype(np.float32)
        reliability_target, risk_target, corruption_level = build_phase2_targets_from_info({}, auxiliary)
        action, log_prob, value = agent.act(obs, auxiliary_observation=auxiliary, deterministic=False)
        hidden_before = agent.hidden_state_before_action
        if hidden_before is None:
            raise RuntimeError("PPOAgent.act did not record hidden_state_before_action")
        hidden_l1_means.append(float(hidden_before.detach().abs().mean().cpu().item()))
        buffer.add(
            observation=obs,
            auxiliary_observation=auxiliary,
            action=action,
            hidden_state_before_action=hidden_before,
            log_prob=log_prob,
            reward=0.1,
            done=index == int(config.train.rollout_steps) - 1,
            value=value,
            aux_road_mask_target=road_mask,
            aux_lane_edge_target=lane_mask,
            aux_mask_targets_available=True,
            reliability_target=reliability_target,
            risk_target=risk_target,
            corruption_level=corruption_level,
        )

    if hidden_l1_means[0] != 0.0:
        raise RuntimeError(f"Initial hidden state should be zero, got {hidden_l1_means[0]}")
    if max(hidden_l1_means[1:]) <= 0.0:
        raise RuntimeError("Stored hidden states after the first step should be nonzero")

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)
    batch = buffer.get_training_batch(agent.device)
    if tuple(batch.hidden_states.shape) != (
        int(config.train.rollout_steps),
        int(config.model.recurrent_layers),
        int(config.model.recurrent_hidden_dim),
    ):
        raise RuntimeError(f"hidden state batch shape mismatch: {tuple(batch.hidden_states.shape)}")
    metrics = agent.update(batch)
    required = ("actor_loss", "critic_loss", "entropy", "total_loss", "aux_mask_available_rate")
    if bool(config.model.enable_shapley):
        required = required + ("shapley_alignment_loss", "shapley_subset_count")
    if bool(config.model.enable_dropout_consistency):
        required = required + ("dropout_consistency_loss",)
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"recurrent update metrics missing keys: {missing}")
    for key in required:
        if not np.isfinite(float(metrics[key])):
            raise RuntimeError(f"recurrent update metric {key} is not finite: {metrics[key]!r}")
    return {
        "hidden_state_shape": list(batch.hidden_states.shape),
        "initial_hidden_l1": round(hidden_l1_means[0], 8),
        "max_post_initial_hidden_l1": round(max(hidden_l1_means[1:]), 8),
        "metrics": {key: round(float(metrics[key]), 6) for key in required},
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_phase1_config(config_path)
    device = torch.device(config.train.resolve_device(args.device))
    torch.manual_seed(int(config.train.seed))
    model = RCSACFirstModel(config.model, config.fusion).to(device)
    obs = _build_dummy_obs(batch_size=4, obs_dim=config.model.obs_dim or 0, device=device)
    auxiliary = _build_dummy_auxiliary(batch_size=4, device=device)

    results: dict[str, object] = {
        "config": str(config_path),
        "device": str(device),
        "obs_dim": int(config.model.obs_dim or 0),
    }
    run_all = not (args.only_model or args.only_loss or args.only_checkpoint)
    if run_all or args.only_model:
        results["model"] = run_model_check(model, obs, auxiliary)
    if run_all or args.only_loss:
        results["loss"] = run_loss_check(model, obs, auxiliary)
    if run_all or args.only_checkpoint:
        results["checkpoint"] = run_checkpoint_check(config_path, args.device)
    if run_all:
        results["forced_corruption"] = run_forced_corruption_check(model, obs)
        results["phase3_shapley"] = run_phase3_shapley_check(model, obs, auxiliary)
        results["recurrent_rollout_update"] = run_recurrent_rollout_update_check(config_path, args.device)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
