"""Shared helpers for Stage 2 confirm/full transition runs."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
import yaml

from ppo_baselines.utils.logger import ExperimentLogger
from rcs_ac_v1.model import MODALITY_NAMES
from rcs_ac_v1.recurrent_buffer import RecurrentRolloutBuffer
from rcs_toy_nav_env import RCSToyNavEnv
from run_stage1_targeted_monotonicity_calibration import deep_apply_overrides
from stage1_warmup.config import ExperimentConfig, load_experiment_config
from stage1_warmup.corruption import apply_shared_corruption, build_reliability_targets
from stage1_warmup.model import Stage1WarmupActorCritic


SUPPORTED_REFERENCE_MODEL_MODES = {"frozen_copy", "source_checkpoint"}
SUPPORTED_TEACHER_REFERENCE_MODEL_MODES = {"frozen_copy", "source_checkpoint", "disabled"}
SUPPORTED_RELATIVE_GAP_MODES = {"vision_anchor", "all_pairs"}
SUPPORTED_SHAPLEY_REFERENCE_MODES = {"clean_full", "current_state"}
SUPPORTED_SHAPLEY_COUNTERFACTUAL_BASELINES = {"zero"}
SUPPORTED_SHAPLEY_UTILITIES = {
    "clean_fused_l2",
    "fused_feature_l2_to_full",
    "target_logit_clean_l1",
}
SUPPORTED_SHAPLEY_ESTIMATORS = {"exact", "sampled"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


def stable_probe_offset(probe_name: str, modulus: int = 997) -> int:
    digest = hashlib.sha256(probe_name.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def make_probe_rng(seed: int, base_offset: int, probe_name: str) -> np.random.Generator:
    return np.random.default_rng(int(seed) + int(base_offset) + stable_probe_offset(probe_name))


def resolve_runtime_import_paths() -> dict[str, str]:
    import ppo_baselines.models.fusion as fusion_module
    import rcs_ac_v1.model as rcs_model_module
    import stage1_warmup.model as warmup_model_module

    return {
        "attention_fusion_module": str(Path(fusion_module.__file__).resolve()),
        "rcs_model_module": str(Path(rcs_model_module.__file__).resolve()),
        "warmup_model_module": str(Path(warmup_model_module.__file__).resolve()),
    }


def is_calibration_only(losses_cfg: dict[str, Any]) -> bool:
    rl_core_keys = [
        "actor_loss_coef",
        "value_loss_coef",
        "entropy_coef",
        "reliability_loss_coef",
        "consistency_loss_coef",
        "shapley_loss_coef",
        "shapley_alignment_loss_coef",
        "risk_loss_coef",
    ]
    return all(abs(float(losses_cfg.get(key, 0.0))) <= 1e-12 for key in rl_core_keys)


def infer_experiment_mode(losses_cfg: dict[str, Any]) -> str:
    return "calibration_only" if is_calibration_only(losses_cfg) else "full_training"


def validate_v3_like_config(config: dict[str, Any]) -> None:
    candidates = config.get("candidates", {})
    candidate_order = list(config.get("candidate_order", candidates.keys()))
    baseline_candidate = config.get("baseline_candidate")
    comparison_baseline_candidate = config.get("comparison_baseline_candidate", baseline_candidate)
    evaluation_candidate = config.get("evaluation_candidate")
    losses_cfg = config.get("losses", {})

    if baseline_candidate and baseline_candidate not in candidates:
        raise ValueError(f"baseline_candidate missing from candidates: {baseline_candidate}")
    if baseline_candidate and baseline_candidate not in candidate_order:
        raise ValueError(f"baseline_candidate missing from candidate_order: {baseline_candidate}")
    if comparison_baseline_candidate and comparison_baseline_candidate not in candidates:
        raise ValueError(f"comparison_baseline_candidate missing from candidates: {comparison_baseline_candidate}")
    if evaluation_candidate and evaluation_candidate not in candidates:
        raise ValueError(f"evaluation_candidate missing from candidates: {evaluation_candidate}")
    if evaluation_candidate and comparison_baseline_candidate and evaluation_candidate == comparison_baseline_candidate:
        raise ValueError("evaluation_candidate and comparison_baseline_candidate must be different in confirm configs")

    relative_gap_mode = str(losses_cfg.get("relative_gap_mode", "vision_anchor"))
    if relative_gap_mode not in SUPPORTED_RELATIVE_GAP_MODES:
        raise ValueError(f"Unsupported relative_gap_mode: {relative_gap_mode}")

    reference_mode = str(config.get("reference_model_mode", "frozen_copy"))
    if reference_mode not in SUPPORTED_REFERENCE_MODEL_MODES:
        raise ValueError(f"Unsupported reference_model_mode: {reference_mode}")
    if reference_mode == "source_checkpoint":
        reference_root = config.get("reference_checkpoint_root") or config.get("source_checkpoint_root")
        if not reference_root:
            raise ValueError("reference_model_mode=source_checkpoint requires reference_checkpoint_root or source_checkpoint_root")
        if not Path(reference_root).exists():
            raise FileNotFoundError(f"Reference checkpoint root not found: {reference_root}")

    teacher_reference_mode = str(
        config.get(
            "teacher_reference_model_mode",
            "source_checkpoint" if config.get("teacher_reference_checkpoint_root") else "disabled",
        )
    )
    if teacher_reference_mode not in SUPPORTED_TEACHER_REFERENCE_MODEL_MODES:
        raise ValueError(f"Unsupported teacher_reference_model_mode: {teacher_reference_mode}")
    if teacher_reference_mode == "source_checkpoint":
        teacher_reference_root = config.get("teacher_reference_checkpoint_root")
        if not teacher_reference_root:
            if bool(config.get("teacher_reference_required", False)):
                raise ValueError(
                    "teacher_reference_model_mode=source_checkpoint requires teacher_reference_checkpoint_root"
                )
        elif not Path(str(teacher_reference_root)).exists():
            if bool(config.get("teacher_reference_required", False)):
                raise FileNotFoundError(f"Teacher checkpoint root not found: {teacher_reference_root}")

    shapley_reference_mode = str(losses_cfg.get("shapley_reference_mode", "clean_full"))
    if shapley_reference_mode not in SUPPORTED_SHAPLEY_REFERENCE_MODES:
        raise ValueError(f"Unsupported shapley_reference_mode: {shapley_reference_mode}")
    shapley_counterfactual_baseline = str(losses_cfg.get("shapley_counterfactual_baseline", "zero"))
    if shapley_counterfactual_baseline not in SUPPORTED_SHAPLEY_COUNTERFACTUAL_BASELINES:
        raise ValueError(f"Unsupported shapley_counterfactual_baseline: {shapley_counterfactual_baseline}")
    shapley_utility = str(losses_cfg.get("shapley_utility", "fused_feature_l2_to_full"))
    if shapley_utility not in SUPPORTED_SHAPLEY_UTILITIES:
        raise ValueError(f"Unsupported shapley_utility: {shapley_utility}")
    shapley_estimator = str(losses_cfg.get("shapley_estimator", "exact"))
    if shapley_estimator not in SUPPORTED_SHAPLEY_ESTIMATORS:
        raise ValueError(f"Unsupported shapley_estimator: {shapley_estimator}")

    if is_calibration_only(losses_cfg):
        warnings.warn(
            f"{config.get('experiment_name', 'experiment')} is running in calibration_only mode because RL core losses are zero.",
            stacklevel=2,
        )


def latest_checkpoint(seed_dir: Path) -> Path:
    checkpoint_final = seed_dir / "checkpoint_final.pt"
    if checkpoint_final.exists():
        return checkpoint_final
    checkpoints = sorted(seed_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No checkpoint found in {seed_dir}")


def build_run_cache_metadata(
    config: dict[str, Any],
    candidate_name: str,
    metrics_filename: str,
    mode: str,
    num_eval_episodes: int,
    *,
    step_name: str | None = None,
    upstream_checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = {
        "config": config,
        "candidate_name": candidate_name,
        "metrics_filename": metrics_filename,
        "mode": mode,
        "num_eval_episodes": int(num_eval_episodes),
        "step_name": str(step_name or config.get("experiment_name", "unknown_step")),
        "upstream_checkpoint_root": str(upstream_checkpoint_root or ""),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return {
        "step_name": str(step_name or config.get("experiment_name", "unknown_step")),
        "mode": str(mode),
        "num_eval_episodes": int(num_eval_episodes),
        "metrics_filename": str(metrics_filename),
        "candidate_name": str(candidate_name),
        "upstream_checkpoint_root": str(upstream_checkpoint_root or ""),
        "config_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16],
    }


def should_reuse_run_cache(metadata_path: Path, expected_metadata: dict[str, Any]) -> bool:
    if not metadata_path.exists():
        return False
    try:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return existing_metadata == expected_metadata


def build_model_from_candidate(
    base_stage1_config_path: str,
    checkpoint_path: Path,
    candidate_snapshot_path: Path | None,
    device: torch.device,
    seed: int,
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[ExperimentConfig, Stage1WarmupActorCritic]:
    base_cfg = load_experiment_config(base_stage1_config_path)
    if candidate_snapshot_path and candidate_snapshot_path.exists():
        snapshot = json.loads(candidate_snapshot_path.read_text(encoding="utf-8"))
        if {"seed", "device", "model", "ppo", "warmup"}.issubset(snapshot.keys()):
            cfg = ExperimentConfig(
                seed=snapshot["seed"],
                device=snapshot["device"],
                env=base_cfg.env,
                model=base_cfg.model,
                ppo=base_cfg.ppo,
                warmup=base_cfg.warmup,
                logging=base_cfg.logging,
            )
            deep_apply_overrides(
                cfg,
                {
                    "model": snapshot["model"],
                    "ppo": snapshot["ppo"],
                    "warmup": snapshot["warmup"],
                },
            )
        else:
            cfg = base_cfg
    else:
        cfg = base_cfg
    if runtime_overrides:
        applicable = {
            key: value
            for key, value in runtime_overrides.items()
            if key in {"model", "ppo", "warmup"} and value is not None
        }
        if applicable:
            deep_apply_overrides(cfg, applicable)
    cfg.seed = seed
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        encoder_output_dim=cfg.model.encoder_output_dim,
        attention_hidden_dim=cfg.model.attention_hidden_dim,
        recurrent_hidden_dim=cfg.model.recurrent_hidden_dim,
        attention_temperature=cfg.model.attention_temperature,
        use_feature_layer_norm=cfg.model.use_feature_layer_norm,
        actor_hidden_dims=cfg.model.actor_hidden_dims,
        critic_hidden_dims=cfg.model.critic_hidden_dims,
        reliability_hidden_dims=cfg.model.reliability_hidden_dims,
        reliability_input_mode=cfg.model.reliability_input_mode,
        reliability_raw_adapter_dim=cfg.model.reliability_raw_adapter_dim,
        reliability_ego_residual_enabled=cfg.model.reliability_ego_residual_enabled,
        reliability_ego_raw_adapter_dim=cfg.model.reliability_ego_raw_adapter_dim,
        reliability_ego_residual_input_mode=cfg.model.reliability_ego_residual_input_mode,
        reliability_ego_residual_apply_to_provisional=cfg.model.reliability_ego_residual_apply_to_provisional,
        reliability_local_residual_enabled_modalities=cfg.model.reliability_local_residual_enabled_modalities,
        reliability_local_residual_adapter_dim=cfg.model.reliability_local_residual_adapter_dim,
        reliability_local_residual_input_mode=cfg.model.reliability_local_residual_input_mode,
        reliability_local_residual_apply_to_provisional=cfg.model.reliability_local_residual_apply_to_provisional,
        action_dim=2,
        action_log_std_init=cfg.model.action_log_std_init,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    current = model.state_dict()
    compatible = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key not in current:
            continue
        current_value = current[key]
        if isinstance(current_value, (UninitializedParameter, UninitializedBuffer)):
            compatible[key] = value
            continue
        if tuple(current_value.shape) == tuple(value.shape):
            compatible[key] = value
    current.update(compatible)
    model.load_state_dict(current)
    return cfg, model


def _resolve_seed_dir_from_root(checkpoint_root: Path, seed: int) -> tuple[Path, int]:
    seed_dir = checkpoint_root / f"seed_{seed}"
    source_seed = seed
    if not seed_dir.exists():
        available = sorted(
            int(path.name.split("_")[-1])
            for path in checkpoint_root.glob("seed_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        )
        if not available:
            raise FileNotFoundError(f"No seed directories found in {checkpoint_root}")
        source_seed = available[seed % len(available)]
        seed_dir = checkpoint_root / f"seed_{source_seed}"
    return seed_dir, source_seed


def _resolve_candidate_checkpoint_root(checkpoint_root: Path, candidate_name: str | None) -> Path:
    if not candidate_name:
        return checkpoint_root
    candidate_dir = checkpoint_root / str(candidate_name)
    if candidate_dir.exists() and candidate_dir.is_dir():
        return candidate_dir
    return checkpoint_root


def register_stage2_standard_candidate(config: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    registration = {
        "standard_stage2_candidate": config["standard_stage2_candidate"],
        "standard_candidate_path": config["candidates"][config["standard_stage2_candidate"]]["checkpoint_root"],
        "source_experiment_name": config["candidates"][config["standard_stage2_candidate"]].get(
            "source_experiment_name",
            "stage1_vision_ego_logit_triplet_calibration",
        ),
    }
    return registration


def register_stage2_backup_candidate(config: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    registration = {
        "backup_candidate": config["backup_candidate"],
        "backup_candidate_path": config["candidates"][config["backup_candidate"]]["checkpoint_root"],
        "source_experiment_name": config["candidates"][config["backup_candidate"]].get(
            "source_experiment_name",
            "stage1_vision_ego_logit_triplet_calibration",
        ),
    }
    return registration


def load_stage2_handoff_checkpoint(
    config: dict[str, Any],
    candidate_name: str,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    candidate_spec = config["candidates"][candidate_name]
    checkpoint_root = Path(candidate_spec["checkpoint_root"])
    preferred_seed_dir_names = [f"seed_{seed}_smoke", f"seed_{seed}"] if smoke else [f"seed_{seed}", f"seed_{seed}_smoke"]
    seed_dir = None
    source_seed = seed
    for seed_dir_name in preferred_seed_dir_names:
        candidate_seed_dir = checkpoint_root / seed_dir_name
        if candidate_seed_dir.exists():
            seed_dir = candidate_seed_dir
            break
    if seed_dir is None:
        available_seed_dirs = sorted(
            path
            for path in checkpoint_root.glob("seed_*")
            if path.is_dir() and len(path.name.split("_")) >= 2 and path.name.split("_")[1].isdigit()
        )
        if not available_seed_dirs:
            raise FileNotFoundError(f"No seed directories found in {checkpoint_root}")
        chosen_seed_dir = available_seed_dirs[seed % len(available_seed_dirs)]
        seed_dir = chosen_seed_dir
        source_seed = int(chosen_seed_dir.name.split("_")[1])
    checkpoint_path = latest_checkpoint(seed_dir)
    snapshot_path = seed_dir / "candidate_config_snapshot.json"
    if not snapshot_path.exists():
        snapshot_path = seed_dir / "config_snapshot.json"
    return {
        "candidate_name": candidate_name,
        "source_experiment_name": candidate_spec.get("source_experiment_name", checkpoint_root.name),
        "source_checkpoint_path": str(checkpoint_path),
        "source_snapshot_path": str(snapshot_path) if snapshot_path.exists() else None,
        "seed": seed,
        "source_seed": source_seed,
    }


def save_stage2_handoff_snapshot(config: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    standard = register_stage2_standard_candidate(config, summary_dir)
    backup = register_stage2_backup_candidate(config, summary_dir)
    snapshot = {
        **standard,
        **backup,
        "candidate_order": list(config.get("candidate_order", config["candidates"].keys())),
        "seeds": list(config["seeds"]),
        "enabled_losses": config["losses"],
        "corruption_policy": {
            "train_corruption": config["train_corruption"],
            "eval_probes": config["eval_probes"],
        },
    }
    (summary_dir / "stage2_handoff_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return snapshot


def build_stage2_candidate_effective_config(config: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    def _merge_dict(target: dict[str, Any], overrides: dict[str, Any]) -> None:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _merge_dict(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    effective_config = copy.deepcopy(config)
    candidate_spec = effective_config["candidates"][candidate_name]
    candidate_overrides = candidate_spec.get("overrides")
    if candidate_overrides:
        _merge_dict(effective_config, candidate_overrides)
    return effective_config


def _apply_single_corruption(
    obs: dict[str, np.ndarray],
    corruption_type: str,
    corruption_strength: float,
    rng: np.random.Generator,
    probe_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    corrupted, severity, _metadata = apply_shared_corruption(
        obs=obs,
        corruption_type=corruption_type,
        corruption_strength=corruption_strength,
        rng=rng,
        corruption_params={"corruption_type": corruption_type, "corruption_strength": corruption_strength},
        probe_name=probe_name,
    )
    return corrupted, severity


def _apply_mixed_corruption(
    obs: dict[str, np.ndarray],
    corruption_types: list[str],
    corruption_strengths: list[float],
    rng: np.random.Generator,
    probe_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    out = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
    total_severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
    for idx, corruption_type in enumerate(corruption_types):
        out, severity = _apply_single_corruption(
            {**out, "aux": obs["aux"]},
            corruption_type=corruption_type,
            corruption_strength=float(corruption_strengths[idx]),
            rng=rng,
            probe_name=f"{probe_name}_{corruption_type}",
        )
        total_severity = np.maximum(total_severity, severity)
    return out, total_severity


def _probe_logging_fields(probe_name: str, probe: dict[str, Any]) -> dict[str, Any]:
    probe_kind = "clean" if probe_name == "normal_eval" else str(probe.get("kind", "single"))
    if probe_kind == "mixed":
        strengths = [float(value) for value in probe.get("strengths", [])]
        scalar = float(max(strengths)) if strengths else float("nan")
    else:
        strength = float(probe.get("strength", 0.0))
        strengths = [strength]
        scalar = strength
    return {
        "corruption_kind": probe_kind,
        "corruption_strength": scalar,
        "corruption_strength_scalar": scalar,
        "corruption_strengths_json": json.dumps(strengths, ensure_ascii=True),
    }


def build_stage2_reference_anchored_targets(
    clean_reliability_scores: torch.Tensor,
    severity: torch.Tensor,
    target_mask: torch.Tensor,
    direct_target_drop_scale: float,
    drop_cap: float = 0.35,
) -> torch.Tensor:
    drop = torch.clamp(severity * direct_target_drop_scale, min=0.0, max=drop_cap)
    targets = clean_reliability_scores.detach().clone()
    targets = torch.clamp(targets - target_mask * drop, min=0.0, max=1.0)
    return targets


def compute_stage2_modality_positive_losses(
    reliability_scores: torch.Tensor,
    reliability_logits: torch.Tensor,
    clean_reliability_scores: torch.Tensor,
    clean_reliability_logits: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    modality_name: str,
    positive_margin: float,
    logit_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    modality_idx = MODALITY_NAMES.index(modality_name)
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
    modality_target_mask = single_target_mask * target_mask[..., modality_idx]
    target_scores = reliability_scores[..., modality_idx]
    target_target_scores = clean_reliability_scores[..., modality_idx]
    positive_term = torch.relu(target_scores - (target_target_scores - positive_margin))
    positive_loss = (positive_term * modality_target_mask).sum() / modality_target_mask.sum().clamp(min=1.0)
    target_logits = reliability_logits[..., modality_idx]
    target_target_logits = clean_reliability_logits[..., modality_idx]
    positive_active_ratio = (
        (positive_term > 0).float() * modality_target_mask
    ).sum() / modality_target_mask.sum().clamp(min=1.0)
    logit_positive_term = torch.relu(target_logits - (target_target_logits - logit_margin))
    logit_positive_loss = (
        logit_positive_term * modality_target_mask
    ).sum() / modality_target_mask.sum().clamp(min=1.0)
    logit_positive_active_ratio = (
        (logit_positive_term > 0).float() * modality_target_mask
    ).sum() / modality_target_mask.sum().clamp(min=1.0)
    return positive_loss, logit_positive_loss, positive_active_ratio, logit_positive_active_ratio


def compute_stage2_ego_positive_losses(
    reliability_scores: torch.Tensor,
    reliability_logits: torch.Tensor,
    clean_reliability_scores: torch.Tensor,
    clean_reliability_logits: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    positive_margin: float,
    logit_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return compute_stage2_modality_positive_losses(
        reliability_scores=reliability_scores,
        reliability_logits=reliability_logits,
        clean_reliability_scores=clean_reliability_scores,
        clean_reliability_logits=clean_reliability_logits,
        target_mask=target_mask,
        corrupted_mask=corrupted_mask,
        modality_name="ego",
        positive_margin=positive_margin,
        logit_margin=logit_margin,
    )


def compute_stage2_local_residual_losses(
    local_residual_deltas: torch.Tensor,
    current_clean_local_residual_deltas: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    enabled_modalities: list[str],
    selectivity_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_count = valid_mask.sum().clamp(min=1.0)
    clean_zero_loss = (
        current_clean_local_residual_deltas[..., [MODALITY_NAMES.index(name) for name in enabled_modalities]].abs()
        * valid_mask.unsqueeze(-1)
    ).sum() / (valid_count * max(len(enabled_modalities), 1))
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
    selectivity_loss = local_residual_deltas.new_tensor(0.0)
    active_modalities = 0.0
    for modality_name in enabled_modalities:
        modality_idx = MODALITY_NAMES.index(modality_name)
        modality_target_mask = single_target_mask * target_mask[..., modality_idx]
        if float(modality_target_mask.sum().item()) <= 0.0:
            continue
        clean_delta = current_clean_local_residual_deltas[..., modality_idx]
        corrupt_delta = local_residual_deltas[..., modality_idx]
        modality_loss = (
            F.relu(selectivity_margin - (clean_delta - corrupt_delta)) * modality_target_mask
        ).sum() / modality_target_mask.sum().clamp(min=1.0)
        selectivity_loss = selectivity_loss + modality_loss
        active_modalities += 1.0
    if active_modalities > 0.0:
        selectivity_loss = selectivity_loss / active_modalities
    return clean_zero_loss, selectivity_loss


def compute_stage2_fusion_shift_losses(
    clean_attention_weights: torch.Tensor,
    corrupt_attention_weights: torch.Tensor,
    clean_final_scores: torch.Tensor,
    corrupt_final_scores: torch.Tensor,
    clean_gate_scores: torch.Tensor,
    corrupt_gate_scores: torch.Tensor,
    clean_reliability_scores: torch.Tensor,
    corrupt_reliability_scores: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    target_modality: str,
    gate_down_margin: float,
    attention_down_margin: float,
    final_score_down_margin: float,
    nontarget_rescue_margin: float,
    alignment_margin: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    target_idx = MODALITY_NAMES.index(target_modality)
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask * valid_mask)
    target_only_mask = single_target_mask * target_mask[..., target_idx]
    target_count = target_only_mask.sum().clamp(min=1.0)

    clean_att = clean_attention_weights[..., target_idx]
    corrupt_att = corrupt_attention_weights[..., target_idx]
    clean_score = clean_final_scores[..., target_idx]
    corrupt_score = corrupt_final_scores[..., target_idx]
    clean_gate = clean_gate_scores[..., target_idx]
    corrupt_gate = corrupt_gate_scores[..., target_idx]
    clean_rel = clean_reliability_scores[..., target_idx]
    corrupt_rel = corrupt_reliability_scores[..., target_idx]

    att_delta = clean_att - corrupt_att
    score_delta = clean_score - corrupt_score
    gate_delta = clean_gate - corrupt_gate
    rel_delta = clean_rel - corrupt_rel

    target_gate_down_loss = (
        F.relu(gate_down_margin - gate_delta) * target_only_mask
    ).sum() / target_count

    target_attention_down_loss = (
        F.relu(attention_down_margin - att_delta) * target_only_mask
    ).sum() / target_count
    target_final_score_down_loss = (
        F.relu(final_score_down_margin - score_delta) * target_only_mask
    ).sum() / target_count

    non_target_indices = [idx for idx, name in enumerate(MODALITY_NAMES) if name != target_modality]
    clean_non_target_mean = clean_attention_weights[..., non_target_indices].mean(dim=-1)
    corrupt_non_target_mean = corrupt_attention_weights[..., non_target_indices].mean(dim=-1)
    clean_non_target_score_mean = clean_final_scores[..., non_target_indices].mean(dim=-1)
    corrupt_non_target_score_mean = corrupt_final_scores[..., non_target_indices].mean(dim=-1)
    nontarget_attention_rescue_delta = corrupt_non_target_mean - clean_non_target_mean
    nontarget_score_rescue_delta = corrupt_non_target_score_mean - clean_non_target_score_mean
    nontarget_rescue_loss = 0.5 * (
        (F.relu(nontarget_rescue_margin - nontarget_attention_rescue_delta) * target_only_mask).sum() / target_count
        + (F.relu(nontarget_rescue_margin - nontarget_score_rescue_delta) * target_only_mask).sum() / target_count
    )

    active_alignment = (rel_delta > alignment_margin).float() * target_only_mask
    active_alignment_count = active_alignment.sum().clamp(min=1.0)
    reliability_gate_alignment_loss = (
        F.relu(alignment_margin - gate_delta) * active_alignment
    ).sum() / active_alignment_count
    reliability_attention_alignment_loss = (
        F.relu(alignment_margin - att_delta) * active_alignment
    ).sum() / active_alignment_count
    reliability_gate_alignment_ratio = (
        ((gate_delta > 0.0).float() * active_alignment).sum() / active_alignment_count
    )
    reliability_attention_alignment_ratio = (
        ((att_delta > 0.0).float() * active_alignment).sum() / active_alignment_count
    )
    gate_active_alignment = (gate_delta > alignment_margin).float() * target_only_mask
    gate_active_alignment_count = gate_active_alignment.sum().clamp(min=1.0)
    gate_attention_alignment_loss = (
        F.relu(alignment_margin - att_delta) * gate_active_alignment
    ).sum() / gate_active_alignment_count
    gate_attention_alignment_ratio = (
        ((att_delta > 0.0).float() * gate_active_alignment).sum() / gate_active_alignment_count
    )
    return (
        target_gate_down_loss,
        target_attention_down_loss,
        target_final_score_down_loss,
        nontarget_rescue_loss,
        reliability_gate_alignment_loss,
        gate_attention_alignment_loss,
        reliability_attention_alignment_loss,
        gate_attention_alignment_ratio,
        reliability_gate_alignment_ratio,
        reliability_attention_alignment_ratio,
    )


def select_stage2_gate_teacher_preserve_indices(
    disable_vision_final_gate_teacher_preserve: bool,
) -> list[int]:
    return [1, 2, 3] if disable_vision_final_gate_teacher_preserve else [0, 1, 2, 3]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    weights = torch.ones_like(values) * weights
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


def compute_stage2_teacher_distilled_gate_losses(
    *,
    clean_gate_scores: torch.Tensor,
    corrupt_gate_scores: torch.Tensor,
    clean_teacher_gate_scores: torch.Tensor,
    corrupt_teacher_gate_scores: torch.Tensor,
    clean_gate_residual_logits: torch.Tensor,
    clean_attention_scores_standalone: torch.Tensor,
    reference_clean_attention_scores: torch.Tensor,
    clean_action_means_standalone: torch.Tensor,
    reference_clean_action_means: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    target_modality: str,
    clean_gate_max: float,
    mild_gate_min: float,
    gate_gap_margin: float,
    disable_vision_final_gate_teacher_preserve: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    target_idx = MODALITY_NAMES.index(target_modality)
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask * valid_mask)
    target_only_mask = single_target_mask * target_mask[..., target_idx]
    preserve_indices = select_stage2_gate_teacher_preserve_indices(
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve
    )

    clean_gate = clean_gate_scores[..., target_idx]
    corrupt_gate = corrupt_gate_scores[..., target_idx]
    clean_gate_cap_loss = _masked_mean(
        F.relu(clean_gate - float(clean_gate_max)),
        target_only_mask,
    )
    mild_gate_floor_loss = _masked_mean(
        F.relu(float(mild_gate_min) - corrupt_gate),
        target_only_mask,
    )
    gate_gap_margin_loss = _masked_mean(
        F.relu(float(gate_gap_margin) - (corrupt_gate - clean_gate)),
        target_only_mask,
    )

    clean_score_anchor_loss = _masked_mean(
        torch.abs(clean_attention_scores_standalone - reference_clean_attention_scores),
        valid_mask,
    )
    clean_action_anchor_loss = _masked_mean(
        torch.abs(clean_action_means_standalone - reference_clean_action_means),
        valid_mask,
    )

    if preserve_indices:
        clean_preserve_error = F.huber_loss(
            clean_gate_scores[..., preserve_indices],
            clean_teacher_gate_scores[..., preserve_indices],
            reduction="none",
        ).mean(dim=-1)
        corrupt_preserve_error = F.huber_loss(
            corrupt_gate_scores[..., preserve_indices],
            corrupt_teacher_gate_scores[..., preserve_indices],
            reduction="none",
        ).mean(dim=-1)
        source_gate_distill_loss = 0.5 * (
            _masked_mean(clean_preserve_error, valid_mask)
            + _masked_mean(corrupt_preserve_error, target_only_mask)
        )
        source_clean_gate_preserve_loss = _masked_mean(clean_preserve_error, valid_mask)
        source_gate_residual_zero_loss = _masked_mean(
            torch.abs(clean_gate_residual_logits[..., preserve_indices]).mean(dim=-1),
            valid_mask,
        )
        nonvision_gate_stability_loss = _masked_mean(
            torch.abs(
                corrupt_gate_scores[..., preserve_indices] - clean_gate_scores[..., preserve_indices]
            ).mean(dim=-1),
            target_only_mask,
        )
    else:
        zero = clean_gate_scores.new_tensor(0.0)
        source_gate_distill_loss = zero
        source_clean_gate_preserve_loss = zero
        source_gate_residual_zero_loss = zero
        nonvision_gate_stability_loss = zero

    vision_final_gate_target_loss = 0.5 * (
        _masked_mean(
            F.huber_loss(
                clean_gate_scores[..., target_idx],
                clean_teacher_gate_scores[..., target_idx],
                reduction="none",
            ),
            valid_mask,
        )
        + _masked_mean(
            F.huber_loss(
                corrupt_gate_scores[..., target_idx],
                corrupt_teacher_gate_scores[..., target_idx],
                reduction="none",
            ),
            target_only_mask,
        )
    )

    return (
        clean_gate_cap_loss,
        mild_gate_floor_loss,
        gate_gap_margin_loss,
        source_gate_distill_loss,
        source_clean_gate_preserve_loss,
        clean_score_anchor_loss,
        clean_action_anchor_loss,
        source_gate_residual_zero_loss,
        vision_final_gate_target_loss,
        nonvision_gate_stability_loss,
    )


def compute_attention_bias_gate_effects(
    attention_bias_scores: torch.Tensor,
    attention_bias_gates: torch.Tensor,
    scale: float,
    attention_temperature: float = 1.0,
) -> torch.Tensor:
    return attention_bias_scores * attention_bias_gates * (float(scale) / max(float(attention_temperature), 1.0e-6))


def compute_stage2_fusion_shift_gate_effect_losses(
    *,
    clean_attention_weights: torch.Tensor,
    corrupt_attention_weights: torch.Tensor,
    clean_final_scores: torch.Tensor,
    corrupt_final_scores: torch.Tensor,
    clean_gate_effects: torch.Tensor,
    corrupt_gate_effects: torch.Tensor,
    clean_bias_scores: torch.Tensor,
    corrupt_bias_scores: torch.Tensor,
    clean_reliability_scores: torch.Tensor,
    corrupt_reliability_scores: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    target_modality: str,
    gate_effect_down_margin: float,
    attention_down_margin: float,
    final_score_down_margin: float,
    bias_score_down_margin: float,
    nontarget_rescue_margin: float,
    alignment_margin: float,
    score_alignment_margin: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    target_idx = MODALITY_NAMES.index(target_modality)
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask * valid_mask)
    target_only_mask = single_target_mask * target_mask[..., target_idx]
    target_count = target_only_mask.sum().clamp(min=1.0)

    clean_att = clean_attention_weights[..., target_idx]
    corrupt_att = corrupt_attention_weights[..., target_idx]
    clean_score = clean_final_scores[..., target_idx]
    corrupt_score = corrupt_final_scores[..., target_idx]
    clean_effect = clean_gate_effects[..., target_idx]
    corrupt_effect = corrupt_gate_effects[..., target_idx]
    clean_bias = clean_bias_scores[..., target_idx]
    corrupt_bias = corrupt_bias_scores[..., target_idx]
    clean_rel = clean_reliability_scores[..., target_idx]
    corrupt_rel = corrupt_reliability_scores[..., target_idx]

    effect_delta = clean_effect - corrupt_effect
    att_delta = clean_att - corrupt_att
    score_delta = clean_score - corrupt_score
    bias_delta = clean_bias - corrupt_bias
    rel_delta = clean_rel - corrupt_rel

    target_gate_effect_down_loss = (
        F.relu(float(gate_effect_down_margin) - effect_delta) * target_only_mask
    ).sum() / target_count
    target_attention_down_loss = (
        F.relu(float(attention_down_margin) - att_delta) * target_only_mask
    ).sum() / target_count
    target_final_score_down_loss = (
        F.relu(float(final_score_down_margin) - score_delta) * target_only_mask
    ).sum() / target_count
    target_bias_score_down_loss = (
        F.relu(float(bias_score_down_margin) - bias_delta) * target_only_mask
    ).sum() / target_count

    non_target_indices = [idx for idx, name in enumerate(MODALITY_NAMES) if name != target_modality]
    clean_non_target_mean = clean_attention_weights[..., non_target_indices].mean(dim=-1)
    corrupt_non_target_mean = corrupt_attention_weights[..., non_target_indices].mean(dim=-1)
    clean_non_target_score_mean = clean_final_scores[..., non_target_indices].mean(dim=-1)
    corrupt_non_target_score_mean = corrupt_final_scores[..., non_target_indices].mean(dim=-1)
    nontarget_attention_rescue_delta = corrupt_non_target_mean - clean_non_target_mean
    nontarget_score_rescue_delta = corrupt_non_target_score_mean - clean_non_target_score_mean
    nontarget_rescue_loss = 0.5 * (
        (F.relu(float(nontarget_rescue_margin) - nontarget_attention_rescue_delta) * target_only_mask).sum() / target_count
        + (F.relu(float(nontarget_rescue_margin) - nontarget_score_rescue_delta) * target_only_mask).sum() / target_count
    )

    active_alignment = (rel_delta > float(alignment_margin)).float() * target_only_mask
    active_alignment_count = active_alignment.sum().clamp(min=1.0)
    reliability_gate_effect_alignment_ratio = (
        ((effect_delta > 0.0).float() * active_alignment).sum() / active_alignment_count
    )
    gate_effect_active_alignment = (effect_delta > float(alignment_margin)).float() * target_only_mask
    gate_effect_active_alignment_count = gate_effect_active_alignment.sum().clamp(min=1.0)
    gate_effect_attention_alignment_ratio = (
        ((att_delta > 0.0).float() * gate_effect_active_alignment).sum() / gate_effect_active_alignment_count
    )
    gate_effect_score_alignment_ratio = (
        ((score_delta > 0.0).float() * gate_effect_active_alignment).sum() / gate_effect_active_alignment_count
    )
    gate_effect_score_alignment_loss = (
        F.relu(float(score_alignment_margin) - score_delta) * gate_effect_active_alignment
    ).sum() / gate_effect_active_alignment_count
    return (
        target_gate_effect_down_loss,
        target_attention_down_loss,
        target_final_score_down_loss,
        target_bias_score_down_loss,
        nontarget_rescue_loss,
        reliability_gate_effect_alignment_ratio,
        gate_effect_attention_alignment_ratio,
        gate_effect_score_alignment_ratio,
        gate_effect_score_alignment_loss,
    )


def compute_stage2_fusion_shift_prebias_losses(
    *,
    clean_prebias_scores: torch.Tensor,
    corrupt_prebias_scores: torch.Tensor,
    clean_final_scores: torch.Tensor,
    corrupt_final_scores: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    target_modality: str,
    prebias_score_down_margin: float,
    alignment_margin: float,
    score_alignment_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_idx = MODALITY_NAMES.index(target_modality)
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask * valid_mask)
    target_only_mask = single_target_mask * target_mask[..., target_idx]
    target_count = target_only_mask.sum().clamp(min=1.0)

    clean_prebias = clean_prebias_scores[..., target_idx]
    corrupt_prebias = corrupt_prebias_scores[..., target_idx]
    clean_score = clean_final_scores[..., target_idx]
    corrupt_score = corrupt_final_scores[..., target_idx]

    prebias_delta = clean_prebias - corrupt_prebias
    score_delta = clean_score - corrupt_score

    target_prebias_score_down_loss = (
        F.relu(float(prebias_score_down_margin) - prebias_delta) * target_only_mask
    ).sum() / target_count

    prebias_active_alignment = (prebias_delta > float(alignment_margin)).float() * target_only_mask
    prebias_active_alignment_count = prebias_active_alignment.sum().clamp(min=1.0)
    prebias_score_final_score_alignment_ratio = (
        ((score_delta > 0.0).float() * prebias_active_alignment).sum() / prebias_active_alignment_count
    )
    prebias_score_alignment_loss = (
        F.relu(float(score_alignment_margin) - score_delta) * prebias_active_alignment
    ).sum() / prebias_active_alignment_count
    return (
        target_prebias_score_down_loss,
        prebias_score_final_score_alignment_ratio,
        prebias_score_alignment_loss,
    )


def select_stage2_non_target_preserve_indices(
    *,
    target_modality: str,
    disable_teacher_target_distill: bool,
    disable_vision_final_gate_teacher_preserve: bool,
) -> list[int]:
    if disable_teacher_target_distill:
        return [idx for idx, name in enumerate(MODALITY_NAMES) if name != target_modality]
    return select_stage2_gate_teacher_preserve_indices(
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve
    )


def compute_stage2_teacher_preserve_signal_absmax(
    *,
    clean_teacher_gate_effects: torch.Tensor,
    corrupt_teacher_gate_effects: torch.Tensor,
    target_modality: str,
    disable_teacher_target_distill: bool,
    disable_vision_final_gate_teacher_preserve: bool,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    preserve_indices = select_stage2_non_target_preserve_indices(
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
    )
    if not preserve_indices:
        return clean_teacher_gate_effects.new_tensor(0.0)
    clean_preserve = clean_teacher_gate_effects[..., preserve_indices]
    corrupt_preserve = corrupt_teacher_gate_effects[..., preserve_indices]
    if valid_mask is not None:
        mask = valid_mask.detach().to(device=clean_preserve.device, dtype=clean_preserve.dtype)
        while mask.dim() < clean_preserve.dim():
            mask = mask.unsqueeze(-1)
        active = mask > 0.0
        if not torch.any(active):
            return clean_teacher_gate_effects.new_tensor(0.0)
        clean_preserve = torch.where(active, clean_preserve, torch.zeros_like(clean_preserve))
        corrupt_preserve = torch.where(active, corrupt_preserve, torch.zeros_like(corrupt_preserve))
    stacked_absmax = torch.stack(
        [
            clean_preserve.detach().abs().amax(),
            corrupt_preserve.detach().abs().amax(),
            (clean_preserve - corrupt_preserve).detach().abs().amax(),
        ]
    )
    return stacked_absmax.amax()


def compute_stage2_preserve_signal_absmax(
    *,
    clean_values: torch.Tensor,
    corrupt_values: torch.Tensor,
    target_modality: str,
    disable_teacher_target_distill: bool,
    disable_vision_final_gate_teacher_preserve: bool,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    preserve_indices = select_stage2_non_target_preserve_indices(
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
    )
    if not preserve_indices:
        return clean_values.new_tensor(0.0)
    clean_preserve = clean_values[..., preserve_indices]
    corrupt_preserve = corrupt_values[..., preserve_indices]
    if valid_mask is not None:
        mask = valid_mask.detach().to(device=clean_preserve.device, dtype=clean_preserve.dtype)
        while mask.dim() < clean_preserve.dim():
            mask = mask.unsqueeze(-1)
        active = mask > 0.0
        if not torch.any(active):
            return clean_values.new_tensor(0.0)
        clean_preserve = torch.where(active, clean_preserve, torch.zeros_like(clean_preserve))
        corrupt_preserve = torch.where(active, corrupt_preserve, torch.zeros_like(corrupt_preserve))
    stacked_absmax = torch.stack(
        [
            clean_preserve.detach().abs().amax(),
            corrupt_preserve.detach().abs().amax(),
            (clean_preserve - corrupt_preserve).detach().abs().amax(),
        ]
    )
    return stacked_absmax.amax()


def compute_stage2_teacher_actuation_regime_state(
    *,
    clean_current_gate_effects: torch.Tensor,
    corrupt_current_gate_effects: torch.Tensor,
    clean_teacher_gate_effects: torch.Tensor,
    corrupt_teacher_gate_effects: torch.Tensor,
    clean_current_bias_scores: torch.Tensor,
    corrupt_current_bias_scores: torch.Tensor,
    clean_teacher_bias_scores: torch.Tensor,
    corrupt_teacher_bias_scores: torch.Tensor,
    target_modality: str,
    disable_teacher_target_distill: bool,
    disable_vision_final_gate_teacher_preserve: bool,
    valid_mask: torch.Tensor | None = None,
    ratio_threshold: float = 10.0,
    nonflat_floor: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_gate_effect_absmax = compute_stage2_preserve_signal_absmax(
        clean_values=clean_current_gate_effects,
        corrupt_values=corrupt_current_gate_effects,
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
        valid_mask=valid_mask,
    )
    teacher_gate_effect_absmax = compute_stage2_preserve_signal_absmax(
        clean_values=clean_teacher_gate_effects,
        corrupt_values=corrupt_teacher_gate_effects,
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
        valid_mask=valid_mask,
    )
    current_bias_score_absmax = compute_stage2_preserve_signal_absmax(
        clean_values=clean_current_bias_scores,
        corrupt_values=corrupt_current_bias_scores,
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
        valid_mask=valid_mask,
    )
    teacher_bias_score_absmax = compute_stage2_preserve_signal_absmax(
        clean_values=clean_teacher_bias_scores,
        corrupt_values=corrupt_teacher_bias_scores,
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
        valid_mask=valid_mask,
    )

    def _ratio_mismatch(current_absmax: torch.Tensor, teacher_absmax: torch.Tensor) -> bool:
        current_value = float(current_absmax.detach().item())
        teacher_value = float(teacher_absmax.detach().item())
        high = max(abs(current_value), abs(teacher_value))
        low = min(abs(current_value), abs(teacher_value))
        if high < float(nonflat_floor):
            return False
        if low < float(nonflat_floor):
            return True
        return high / max(low, 1.0e-12) >= float(ratio_threshold)

    mismatch = _ratio_mismatch(current_gate_effect_absmax, teacher_gate_effect_absmax) or _ratio_mismatch(
        current_bias_score_absmax, teacher_bias_score_absmax
    )
    preserve_nonflat = float(teacher_gate_effect_absmax.detach().item()) >= float(nonflat_floor)
    preserve_active = preserve_nonflat and (not mismatch)
    mismatch_tensor = clean_current_gate_effects.new_tensor(float(mismatch))
    preserve_active_tensor = clean_current_gate_effects.new_tensor(float(preserve_active))
    return teacher_gate_effect_absmax, teacher_bias_score_absmax, mismatch_tensor, preserve_active_tensor


def compute_stage2_teacher_gate_effect_losses(
    *,
    clean_gate_effects: torch.Tensor,
    corrupt_gate_effects: torch.Tensor,
    clean_teacher_gate_effects: torch.Tensor,
    corrupt_teacher_gate_effects: torch.Tensor,
    clean_attention_weights_standalone: torch.Tensor,
    anchor_clean_attention_weights: torch.Tensor,
    clean_reliability_scores_standalone: torch.Tensor,
    anchor_clean_reliability_scores: torch.Tensor,
    clean_bias_scores_standalone: torch.Tensor,
    anchor_clean_bias_scores: torch.Tensor,
    clean_prebias_scores_standalone: torch.Tensor,
    anchor_clean_prebias_scores: torch.Tensor,
    target_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    target_modality: str,
    disable_vision_final_gate_teacher_preserve: bool,
    disable_teacher_target_distill: bool,
    teacher_effect_available: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_idx = MODALITY_NAMES.index(target_modality)
    single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask * valid_mask)
    target_only_mask = single_target_mask * target_mask[..., target_idx]
    preserve_indices = select_stage2_non_target_preserve_indices(
        target_modality=target_modality,
        disable_teacher_target_distill=disable_teacher_target_distill,
        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
    )

    if preserve_indices and teacher_effect_available:
        clean_preserve_error = F.huber_loss(
            clean_gate_effects[..., preserve_indices],
            clean_teacher_gate_effects[..., preserve_indices],
            reduction="none",
        ).mean(dim=-1)
        corrupt_preserve_error = F.huber_loss(
            corrupt_gate_effects[..., preserve_indices],
            corrupt_teacher_gate_effects[..., preserve_indices],
            reduction="none",
        ).mean(dim=-1)
        teacher_gate_effect_distill_loss = 0.5 * (
            _masked_mean(clean_preserve_error, valid_mask)
            + _masked_mean(corrupt_preserve_error, target_only_mask)
        )
        teacher_clean_gate_effect_preserve_loss = _masked_mean(clean_preserve_error, valid_mask)
    else:
        zero = clean_gate_effects.new_tensor(0.0)
        teacher_gate_effect_distill_loss = zero
        teacher_clean_gate_effect_preserve_loss = zero

    clean_attention_weight_anchor_loss = _masked_mean(
        torch.abs(clean_attention_weights_standalone - anchor_clean_attention_weights),
        valid_mask,
    )
    clean_reliability_anchor_loss = _masked_mean(
        torch.abs(clean_reliability_scores_standalone - anchor_clean_reliability_scores),
        valid_mask,
    )
    clean_bias_score_anchor_loss = _masked_mean(
        torch.abs(clean_bias_scores_standalone - anchor_clean_bias_scores),
        valid_mask,
    )
    clean_prebias_score_anchor_loss = _masked_mean(
        torch.abs(clean_prebias_scores_standalone - anchor_clean_prebias_scores),
        valid_mask,
    )
    return (
        teacher_gate_effect_distill_loss,
        clean_attention_weight_anchor_loss,
        clean_reliability_anchor_loss,
        clean_bias_score_anchor_loss,
        clean_prebias_score_anchor_loss,
    )


class Stage2TransitionTrainer:
    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path, smoke: bool = False):
        effective_config = build_stage2_candidate_effective_config(config, candidate_name)
        validate_v3_like_config(effective_config)
        self.raw_config = effective_config
        self.candidate_name = candidate_name
        self.seed = seed
        self.run_dir = run_dir
        self.smoke = bool(smoke)
        self.device = torch.device(self.raw_config.get("device", "cpu"))
        self.logger = ExperimentLogger(str(run_dir))

        handoff = load_stage2_handoff_checkpoint(self.raw_config, candidate_name, seed, smoke=smoke)
        self.base_cfg, self.model = build_model_from_candidate(
            base_stage1_config_path=self.raw_config["base_stage1_config_path"],
            checkpoint_path=Path(handoff["source_checkpoint_path"]),
            candidate_snapshot_path=Path(handoff["source_snapshot_path"]) if handoff["source_snapshot_path"] else None,
            device=self.device,
            seed=seed,
            runtime_overrides=self.raw_config,
        )
        self.reference_model_mode = str(self.raw_config.get("reference_model_mode", "frozen_copy"))
        self._apply_standard_fusion_overrides(self.model)
        self.anchor_reference_model = self._build_reference_model(handoff)
        self._apply_standard_fusion_overrides(self.anchor_reference_model)
        self._materialize_gate_actuation_parameters(self.anchor_reference_model, include_fusion_gate=True)
        self.teacher_reference_model = self._build_teacher_reference_model(handoff)
        self.teacher_source = "external" if self.teacher_reference_model is not None else "disabled"
        self.reference_model = self.anchor_reference_model
        self._apply_proof_fusion_gate_overrides(self.model)
        self._materialize_gate_actuation_parameters(self.model, include_fusion_gate=True)

        warmup_cfg = self.base_cfg.warmup
        if str(getattr(warmup_cfg, "attention_finetune_mode", "")).strip():
            self._apply_attention_finetune_freeze(warmup_cfg)
        elif bool(getattr(warmup_cfg, "train_local_reliability_residual_only", False)):
            self.model.freeze_local_reliability_residual_only()
        elif bool(getattr(warmup_cfg, "train_ego_reliability_residual_only", False)):
            self.model.freeze_ego_reliability_residual_only()
        elif bool(getattr(warmup_cfg, "freeze_policy_backbone", False)) or bool(
            getattr(warmup_cfg, "train_reliability_head_only", False)
        ):
            self.model.freeze_policy_backbone()
            self.model.unfreeze_reliability_head()
        self.use_gate_feature_context = bool(str(getattr(warmup_cfg, "attention_finetune_mode", "")).strip())

        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise ValueError("Stage2TransitionTrainer found no trainable parameters for the selected candidate config")
        self.optimizer = torch.optim.Adam(trainable_params, lr=float(self.raw_config["train"]["learning_rate"]))
        self.train_env = RCSToyNavEnv()
        self.eval_env = RCSToyNavEnv()
        self.train_env.set_corruption_mode("clean")
        self.eval_env.set_corruption_mode("clean")
        self.obs_shapes = {key: self.train_env.observation_space[key].shape for key in [*MODALITY_NAMES, "aux"]}
        extra_specs = {
            "reliability_targets": (len(MODALITY_NAMES),),
            "severity": (len(MODALITY_NAMES),),
            "target_mask": (len(MODALITY_NAMES),),
            "corrupted_flag": (1,),
            **{f"clean_{key}": self.obs_shapes[key] for key in MODALITY_NAMES},
        }
        self.buffer = RecurrentRolloutBuffer(
            rollout_steps=int(config["train"]["rollout_steps"]),
            sequence_length=int(config["train"]["sequence_length"]),
            obs_shapes=self.obs_shapes,
            action_dim=2,
            hidden_dim=self.base_cfg.model.recurrent_hidden_dim,
            obs_keys=[*MODALITY_NAMES, "aux"],
            extra_specs=extra_specs,
        )
        (self.run_dir / "candidate_config_snapshot.json").write_text(
            json.dumps(asdict(self.base_cfg), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self.logger.save_config(
            {
                "candidate_name": candidate_name,
                "seed": seed,
                "config": self.raw_config,
                "handoff": handoff,
                "import_paths": resolve_runtime_import_paths(),
                "experiment_mode": infer_experiment_mode(self.raw_config.get("losses", {})),
                "reference_model_mode": self.reference_model_mode,
            }
        )
        self.nan_detected = False

    def _apply_standard_fusion_overrides(self, model: Stage1WarmupActorCritic) -> None:
        fusion = getattr(model, "fusion_module", None)
        if fusion is None:
            return
        losses_cfg = dict(self.raw_config.get("losses", {}))
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
        fusion.gate_gap_margin = float(
            losses_cfg.get("gate_gap_margin", getattr(fusion, "gate_gap_margin", 0.05))
        )
        fusion.common_cancel_anchor_gate = float(
            losses_cfg.get(
                "common_cancel_anchor_gate",
                getattr(fusion, "common_cancel_anchor_gate", 0.05),
            )
        )

    def _apply_proof_fusion_gate_overrides(self, model: Stage1WarmupActorCritic) -> None:
        fusion = getattr(model, "fusion_module", None)
        if fusion is None:
            return
        fusion_gate_cfg = dict(self.raw_config.get("fusion_gate", {}))
        if not fusion_gate_cfg:
            return
        default_gate_overrides = {
            "use_relative_score_head": True,
            "use_reliability_bias_head": True,
            "use_reliability_bias_gate": True,
            "gate_mode": "source_residual_logit_v2",
            "relative_gap_residual_scale": 0.22,
            "reliability_bias_residual_scale": 0.22,
            "gate_residual_scale": 0.05,
        }
        for key, value in default_gate_overrides.items():
            if key not in fusion_gate_cfg:
                setattr(fusion, key, value)
        for key, value in fusion_gate_cfg.items():
            setattr(fusion, key, value)

    def _materialize_gate_actuation_parameters(
        self,
        model: Stage1WarmupActorCritic,
        *,
        include_fusion_gate: bool,
    ) -> None:
        fusion = getattr(model, "fusion_module", None)
        if fusion is None:
            return
        merged_cfg = dict(self.raw_config.get("losses", {}))
        if include_fusion_gate:
            merged_cfg.update(self.raw_config.get("fusion_gate", {}))
        if "gate_residual_alpha_mode" in merged_cfg and "residual_alpha_mode" not in merged_cfg:
            merged_cfg["residual_alpha_mode"] = merged_cfg["gate_residual_alpha_mode"]
        if "gate_residual_alpha_init" in merged_cfg and "residual_alpha_init" not in merged_cfg:
            merged_cfg["residual_alpha_init"] = merged_cfg["gate_residual_alpha_init"]
        if "gate_residual_alpha_max" in merged_cfg and "residual_alpha_max" not in merged_cfg:
            merged_cfg["residual_alpha_max"] = merged_cfg["gate_residual_alpha_max"]
        if "gate_residual_alpha_l2_coef" in merged_cfg and "residual_alpha_l2_coef" not in merged_cfg:
            merged_cfg["residual_alpha_l2_coef"] = merged_cfg["gate_residual_alpha_l2_coef"]
        fusion.score_adapter_residual_scale = float(
            merged_cfg.get(
                "score_adapter_residual_scale",
                getattr(fusion, "score_adapter_residual_scale", 0.0),
            )
        )
        from v3_shapley_helpers import configure_trainable_gate_actuation_parameters

        configure_trainable_gate_actuation_parameters(model, merged_cfg)

    def _apply_attention_finetune_freeze(self, warmup_cfg: Any) -> None:
        mode = str(getattr(warmup_cfg, "attention_finetune_mode", "")).strip()
        train_gate_input_adapter = bool(getattr(warmup_cfg, "train_gate_input_adapter", False))
        if mode == "source_gate_residual_only":
            self.model.freeze_attention_source_gate_residual_only(
                train_gate_input_adapter=train_gate_input_adapter
            )
            return
        if mode == "source_gate_relative_micro_unfreeze":
            self.model.freeze_attention_source_gate_relative_micro_unfreeze(
                train_gate_input_adapter=train_gate_input_adapter
            )
            return
        if mode == "source_gate_threshold_shift_micro_unfreeze":
            self.model.freeze_attention_source_gate_threshold_shift_micro_unfreeze(
                train_gate_input_adapter=train_gate_input_adapter
            )
            return
        if mode == "source_gate_bias_adapter_output_scale_micro_unfreeze":
            self.model.freeze_attention_source_gate_bias_adapter_output_scale_micro_unfreeze(
                train_gate_input_adapter=train_gate_input_adapter
            )
            return
        if mode == "source_gate_threshold_shift_bias_micro_unfreeze":
            self.model.freeze_attention_source_gate_threshold_shift_bias_micro_unfreeze(
                train_gate_input_adapter=train_gate_input_adapter
            )
            return
        raise ValueError(f"Unsupported attention_finetune_mode: {mode}")

    def _build_gate_feature_context(
        self,
        clean_teacher_scores: torch.Tensor,
        source_gate_scores: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        corrupted_mask: torch.Tensor | None = None,
        sign_teacher_scores: torch.Tensor | None = None,
        clean_teacher_adapter_delta: torch.Tensor | None = None,
        sign_teacher_adapter_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        clean_teacher = clean_teacher_scores.detach()
        if sign_teacher_scores is None:
            sign_teacher = clean_teacher.clone()
            teacher_target_shift_margin = float(
                self.raw_config.get("losses", {}).get("teacher_target_shift_margin", 0.0)
            )
            if (
                teacher_target_shift_margin > 0.0
                and target_mask is not None
                and corrupted_mask is not None
            ):
                target = target_mask.detach().to(device=clean_teacher.device, dtype=clean_teacher.dtype)
                corrupted = corrupted_mask.detach().to(device=clean_teacher.device, dtype=clean_teacher.dtype)
                while corrupted.dim() < target.dim():
                    corrupted = corrupted.unsqueeze(-1)
                sign_teacher = sign_teacher - teacher_target_shift_margin * target * corrupted
        else:
            sign_teacher = sign_teacher_scores.detach()
        context = {
            "clean_teacher_scores": clean_teacher,
            "sign_teacher_scores": sign_teacher,
            "source_gate_scores": source_gate_scores.detach(),
        }
        if clean_teacher_adapter_delta is not None and sign_teacher_adapter_delta is not None:
            context["clean_teacher_adapter_delta"] = clean_teacher_adapter_delta.detach()
            context["sign_teacher_adapter_delta"] = sign_teacher_adapter_delta.detach()
        return context

    def _build_step_gate_feature_context(
        self,
        clean_teacher_output: Any,
        target_mask: np.ndarray | torch.Tensor,
        corrupted_flag: float,
        sign_teacher_output: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if not self.use_gate_feature_context:
            return None
        proof_teacher_mode = str(self.raw_config.get("losses", {}).get("proof_teacher_mode", "synthetic_margin"))
        target_mask_tensor = torch.as_tensor(target_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        corrupted_mask_tensor = torch.tensor([float(corrupted_flag)], dtype=torch.float32, device=self.device)
        if proof_teacher_mode == "reference_corrupt":
            sign_teacher = clean_teacher_output if sign_teacher_output is None else sign_teacher_output
            return self._build_gate_feature_context(
                clean_teacher_scores=clean_teacher_output.aux_outputs["attention_relative_scores_before_bias"],
                sign_teacher_scores=sign_teacher.aux_outputs["attention_relative_scores_before_bias"],
                source_gate_scores=sign_teacher.aux_outputs["attention_source_bias_gates"],
                clean_teacher_adapter_delta=clean_teacher_output.aux_outputs["attention_score_adapter_delta"],
                sign_teacher_adapter_delta=sign_teacher.aux_outputs["attention_score_adapter_delta"],
            )
        return self._build_gate_feature_context(
            clean_teacher_scores=clean_teacher_output.aux_outputs["attention_final_scores"],
            source_gate_scores=clean_teacher_output.aux_outputs["attention_source_bias_gates"],
            target_mask=target_mask_tensor,
            corrupted_mask=corrupted_mask_tensor,
        )

    def _build_sequence_gate_feature_context(
        self,
        reference_clean_outputs: dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        corrupted_mask: torch.Tensor,
        sign_teacher_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if not self.use_gate_feature_context:
            return None
        proof_teacher_mode = str(self.raw_config.get("losses", {}).get("proof_teacher_mode", "synthetic_margin"))
        if proof_teacher_mode == "reference_corrupt":
            sign_teacher = reference_clean_outputs if sign_teacher_outputs is None else sign_teacher_outputs
            return self._build_gate_feature_context(
                clean_teacher_scores=reference_clean_outputs["attention_relative_scores_before_bias"],
                sign_teacher_scores=sign_teacher["attention_relative_scores_before_bias"],
                source_gate_scores=sign_teacher["attention_source_bias_gates"],
                clean_teacher_adapter_delta=reference_clean_outputs["attention_score_adapter_delta"],
                sign_teacher_adapter_delta=sign_teacher["attention_score_adapter_delta"],
            )
        return self._build_gate_feature_context(
            clean_teacher_scores=reference_clean_outputs["attention_final_scores"],
            source_gate_scores=reference_clean_outputs["attention_source_bias_gates"],
            target_mask=target_mask,
            corrupted_mask=corrupted_mask,
        )

    def _build_reference_model(self, handoff: dict[str, Any]) -> Stage1WarmupActorCritic:
        if self.reference_model_mode == "frozen_copy":
            reference_model = copy.deepcopy(self.model)
        else:
            reference_root = Path(self.raw_config.get("reference_checkpoint_root") or self.raw_config.get("source_checkpoint_root", ""))
            reference_model = self._build_model_from_checkpoint_root(
                checkpoint_root=reference_root,
                snapshot_path_override=self.raw_config.get("reference_snapshot_path"),
            )
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad = False
        return reference_model

    def _build_model_from_checkpoint_root(
        self,
        *,
        checkpoint_root: Path,
        snapshot_path_override: str | None = None,
    ) -> Stage1WarmupActorCritic:
        seed_dir, _source_seed = _resolve_seed_dir_from_root(checkpoint_root, self.seed)
        checkpoint_path = latest_checkpoint(seed_dir)
        snapshot_path = Path(snapshot_path_override) if snapshot_path_override else seed_dir / "candidate_config_snapshot.json"
        if not snapshot_path.exists():
            snapshot_path = seed_dir / "config_snapshot.json"
        _cfg, model = build_model_from_candidate(
            base_stage1_config_path=self.raw_config["base_stage1_config_path"],
            checkpoint_path=checkpoint_path,
            candidate_snapshot_path=snapshot_path if snapshot_path.exists() else None,
            device=self.device,
            seed=self.seed,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def _build_teacher_reference_model(
        self,
        handoff: dict[str, Any],
    ) -> Stage1WarmupActorCritic | None:
        teacher_mode = str(
            self.raw_config.get(
                "teacher_reference_model_mode",
                "source_checkpoint" if self.raw_config.get("teacher_reference_checkpoint_root") else "disabled",
            )
        )
        if teacher_mode == "disabled":
            return None
        if teacher_mode == "frozen_copy":
            teacher_reference_model = copy.deepcopy(self.anchor_reference_model)
        else:
            teacher_reference_root = self.raw_config.get("teacher_reference_checkpoint_root")
            if not teacher_reference_root:
                if bool(self.raw_config.get("teacher_reference_required", False)):
                    raise ValueError("teacher_reference_checkpoint_root is required when teacher_reference_required=true")
                return None
            teacher_checkpoint_root = _resolve_candidate_checkpoint_root(
                Path(str(teacher_reference_root)),
                self.raw_config.get("teacher_reference_candidate_name"),
            )
            teacher_reference_model = self._build_model_from_checkpoint_root(
                checkpoint_root=teacher_checkpoint_root,
                snapshot_path_override=self.raw_config.get("teacher_reference_snapshot_path"),
            )
        teacher_reference_model.eval()
        for parameter in teacher_reference_model.parameters():
            parameter.requires_grad = False
        self._apply_standard_fusion_overrides(teacher_reference_model)
        if teacher_mode == "frozen_copy":
            self._apply_proof_fusion_gate_overrides(teacher_reference_model)
            self._materialize_gate_actuation_parameters(teacher_reference_model, include_fusion_gate=True)
        return teacher_reference_model

    @torch.no_grad()
    def _forward_reference_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor,
        deterministic: bool = False,
        gate_feature_context: dict[str, torch.Tensor] | None = None,
        reference_model: Stage1WarmupActorCritic | None = None,
    ) -> dict[str, torch.Tensor]:
        model = reference_model or self.anchor_reference_model
        return model.forward_sequence(
            obs_seq=obs_seq,
            episode_starts=episode_starts,
            initial_hidden=initial_hidden,
            actions=None,
            deterministic=deterministic,
            gate_feature_context=gate_feature_context,
        )

    @torch.no_grad()
    def _reference_step(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool = True,
        gate_feature_context: dict[str, torch.Tensor] | None = None,
        reference_model: Stage1WarmupActorCritic | None = None,
    ):
        model = reference_model or self.anchor_reference_model
        return model.step(
            obs_dict=obs_dict,
            hidden_state=hidden_state,
            episode_start=episode_start,
            deterministic=deterministic,
            gate_feature_context=gate_feature_context,
        )

    def _metrics_have_nan(self, metrics: dict[str, float]) -> bool:
        for value in metrics.values():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                return True
        return False

    def _sample_train_corruption(
        self,
        obs: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
        spec = self.raw_config["train_corruption"]
        losses_cfg = self.raw_config.get("losses", {})
        total_updates = max(int(self.raw_config.get("train", {}).get("total_steps", 1)) // max(int(self.raw_config.get("train", {}).get("rollout_steps", 1)), 1), 1)
        update_idx = int(getattr(self, "_current_update_idx", 0))

        def _resolve_train_strength(selected_name: str, fallback_strength: float) -> float:
            schedule = spec.get("train_vision_corruption_schedule", losses_cfg.get("train_vision_corruption_schedule"))
            if selected_name != "vision" or schedule is None:
                return fallback_strength
            if isinstance(schedule, (list, tuple)) and schedule:
                progress = min(max(update_idx / max(total_updates - 1, 1), 0.0), 1.0)
                stage_idx = min(int(progress * len(schedule)), len(schedule) - 1)
                return float(schedule[stage_idx])
            return float(schedule)

        choice = float(rng.random())
        clean_prob = float(spec["clean_probability"])
        mild_single_prob = float(spec["mild_single_probability"])
        if choice < clean_prob:
            severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
            return {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}, severity, severity.copy(), "clean"
        if choice < clean_prob + mild_single_prob:
            single_specs = spec["single_modality"]
            paired_prob = float(losses_cfg.get("paired_clean_mild_vision_prob", spec.get("paired_clean_mild_vision_prob", 0.0)))
            use_paired_vision = bool(losses_cfg.get("paired_selective_gate_supervision", False)) and float(rng.random()) < paired_prob
            if use_paired_vision:
                selected = "vision"
                selected_cfg = next(item for item in single_specs if item["name"] == selected)
                probe_name = "train_paired_mild_vision"
                corruption_name = "mild_vision_paired"
            else:
                names = [item["name"] for item in single_specs]
                weights = np.asarray([float(item.get("weight", 1.0)) for item in single_specs], dtype=np.float64)
                weights = weights / weights.sum()
                selected = str(rng.choice(names, p=weights))
                selected_cfg = next(item for item in single_specs if item["name"] == selected)
                probe_name = f"train_mild_{selected}"
                corruption_name = f"mild_{selected}"
            corrupted, severity = _apply_single_corruption(
                obs=obs,
                corruption_type=selected,
                corruption_strength=_resolve_train_strength(selected, float(selected_cfg["strength"])),
                rng=rng,
                probe_name=probe_name,
            )
            return corrupted, severity, (severity > 0.0).astype(np.float32), corruption_name
        mixed_specs = spec.get("mixed", [])
        if mixed_specs:
            mixed = mixed_specs[int(rng.integers(0, len(mixed_specs)))]
            corrupted, severity = _apply_mixed_corruption(
                obs=obs,
                corruption_types=list(mixed["modalities"]),
                corruption_strengths=list(mixed["strengths"]),
                rng=rng,
                probe_name=str(mixed["name"]),
            )
            return corrupted, severity, (severity > 0.0).astype(np.float32), str(mixed["name"])
        severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
        return {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}, severity, severity.copy(), "clean"

    def collect_rollout(self, update_idx: int) -> dict[str, float]:
        self._current_update_idx = int(update_idx)
        self.buffer.reset()
        obs, info = self.train_env.reset(seed=self.seed * 10000 + update_idx)
        del info
        hidden = self.model.initial_hidden(batch_size=1, device=self.device)
        episode_start = torch.ones(1, device=self.device)
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
            with torch.no_grad():
                gate_feature_context = None
                if self.use_gate_feature_context:
                    reference_clean_output = self._reference_step(
                        obs_dict=obs_to_tensor(clean_policy_obs, self.device),
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    sign_teacher_output = reference_clean_output
                    if (
                        str(self.raw_config.get("losses", {}).get("proof_teacher_mode", "synthetic_margin"))
                        == "reference_corrupt"
                        and corruption_name != "clean"
                    ):
                        sign_teacher_output = self._reference_step(
                            obs_dict=obs_to_tensor(policy_obs, self.device),
                            hidden_state=hidden,
                            episode_start=episode_start,
                            deterministic=True,
                        )
                    gate_feature_context = self._build_step_gate_feature_context(
                        clean_teacher_output=reference_clean_output,
                        target_mask=target_mask,
                        corrupted_flag=float(corruption_name != "clean"),
                        sign_teacher_output=sign_teacher_output,
                    )
                output = self.model.step(
                    obs_dict=obs_to_tensor(policy_obs, self.device),
                    hidden_state=hidden,
                    episode_start=episode_start,
                    deterministic=False,
                    gate_feature_context=gate_feature_context,
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
            current_return += float(reward)
            current_progress.append(float(info.get("progress_delta", 0.0)))
            obs = next_obs
            episode_start = torch.tensor([float(done)], device=self.device)
            hidden = self.model.reset_hidden(hidden, episode_start)
            if done:
                episode_returns.append(current_return)
                progress_values.append(float(np.mean(current_progress)) if current_progress else 0.0)
                current_return = 0.0
                current_progress = []
                obs, info = self.train_env.reset()
                del info

        with torch.no_grad():
            clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
            gate_feature_context = None
            if self.use_gate_feature_context:
                reference_clean_output = self._reference_step(
                    obs_dict=obs_to_tensor(clean_obs, self.device),
                    hidden_state=hidden,
                    episode_start=episode_start,
                    deterministic=True,
                )
                gate_feature_context = self._build_step_gate_feature_context(
                    clean_teacher_output=reference_clean_output,
                    target_mask=np.zeros(len(MODALITY_NAMES), dtype=np.float32),
                    corrupted_flag=0.0,
                    sign_teacher_output=reference_clean_output,
                )
            output = self.model.step(
                obs_dict=obs_to_tensor(clean_obs, self.device),
                hidden_state=hidden,
                episode_start=episode_start,
                deterministic=False,
                gate_feature_context=gate_feature_context,
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
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/reliability_total": 0.0,
            "loss/consistency_loss": 0.0,
            "loss/consistency_gap_mean": 0.0,
            "loss/vision_logit_retention": 0.0,
            "loss/ego_logit_retention": 0.0,
            "loss/vision_target_positive": 0.0,
            "loss/vision_logit_positive": 0.0,
            "loss/ego_target_positive": 0.0,
            "loss/ego_logit_positive": 0.0,
            "loss/local_residual_clean_zero": 0.0,
            "loss/local_residual_selectivity_margin": 0.0,
            "loss/target_gate_down": 0.0,
            "loss/target_attention_down": 0.0,
            "loss/target_final_score_down": 0.0,
            "loss/nontarget_rescue": 0.0,
            "loss/reliability_gate_alignment": 0.0,
            "loss/gate_attention_alignment": 0.0,
            "loss/reliability_attention_alignment": 0.0,
            "loss/clean_gate_cap": 0.0,
            "loss/mild_gate_floor": 0.0,
            "loss/gate_gap_margin": 0.0,
            "loss/source_gate_distill": 0.0,
            "loss/source_clean_gate_preserve": 0.0,
            "loss/clean_score_anchor": 0.0,
            "loss/clean_action_anchor": 0.0,
            "loss/source_gate_residual_zero": 0.0,
            "loss/vision_final_gate_target": 0.0,
            "loss/nonvision_gate_stability": 0.0,
            "loss_weighted/vision_target_positive": 0.0,
            "loss_weighted/vision_logit_positive": 0.0,
            "loss_weighted/ego_target_positive": 0.0,
            "loss_weighted/ego_logit_positive": 0.0,
            "loss_weighted/local_residual_clean_zero": 0.0,
            "loss_weighted/local_residual_selectivity_margin": 0.0,
            "loss_weighted/target_gate_down": 0.0,
            "loss_weighted/target_attention_down": 0.0,
            "loss_weighted/target_final_score_down": 0.0,
            "loss_weighted/nontarget_rescue": 0.0,
            "loss_weighted/reliability_gate_alignment": 0.0,
            "loss_weighted/gate_attention_alignment": 0.0,
            "loss_weighted/reliability_attention_alignment": 0.0,
            "loss_weighted/clean_gate_cap": 0.0,
            "loss_weighted/mild_gate_floor": 0.0,
            "loss_weighted/gate_gap_margin": 0.0,
            "loss_weighted/source_gate_distill": 0.0,
            "loss_weighted/source_clean_gate_preserve": 0.0,
            "loss_weighted/clean_score_anchor": 0.0,
            "loss_weighted/clean_action_anchor": 0.0,
            "loss_weighted/source_gate_residual_zero": 0.0,
            "loss_weighted/vision_final_gate_target": 0.0,
            "loss_weighted/nonvision_gate_stability": 0.0,
            "loss/grad_norm": 0.0,
            "train/vision_clean_anchor_score_mean": 0.0,
            "train/vision_corrupt_score_mean": 0.0,
            "train/vision_target_score_mean": 0.0,
            "train/ego_clean_anchor_score_mean": 0.0,
            "train/ego_corrupt_score_mean": 0.0,
            "train/ego_target_score_mean": 0.0,
            "train/vision_residual_delta_mean": 0.0,
            "train/vision_residual_delta_abs_mean": 0.0,
            "train/ego_residual_delta_mean": 0.0,
            "train/ego_residual_delta_abs_mean": 0.0,
            "train/vision_target_positive_active_ratio": 0.0,
            "train/vision_logit_positive_active_ratio": 0.0,
            "train/ego_target_positive_active_ratio": 0.0,
            "train/ego_logit_positive_active_ratio": 0.0,
            "train/vision_gate_delta_clean_minus_corrupt": 0.0,
            "train/vision_source_gate_base_delta_clean_minus_corrupt": 0.0,
            "train/vision_attention_delta_clean_minus_corrupt": 0.0,
            "train/vision_final_score_delta_clean_minus_corrupt": 0.0,
            "train/vision_gate_attention_alignment_ratio": 0.0,
            "train/vision_reliability_gate_alignment_ratio": 0.0,
            "train/vision_reliability_attention_alignment_ratio": 0.0,
            "train/vision_teacher_gate_delta_clean_minus_corrupt": 0.0,
            "train/clean_score_anchor_error": 0.0,
            "train/clean_action_anchor_error": 0.0,
            "loss/clean_gate_effect_small": 0.0,
            "loss/mild_gate_effect_target": 0.0,
            "loss/gate_effect_gap_margin": 0.0,
            "loss/teacher_gate_effect_distill": 0.0,
            "loss/teacher_clean_gate_effect_preserve": 0.0,
            "loss/clean_attention_weight_anchor": 0.0,
            "loss/clean_reliability_anchor": 0.0,
            "loss/target_gate_effect_down": 0.0,
            "loss/target_bias_score_down": 0.0,
            "loss/gate_effect_score_alignment": 0.0,
            "loss/teacher_nonvision_gate_effect_preserve": 0.0,
            "loss/clean_bias_score_anchor": 0.0,
            "loss/target_prebias_score_down": 0.0,
            "loss/prebias_score_alignment": 0.0,
            "loss/clean_prebias_score_anchor": 0.0,
            "loss_weighted/clean_gate_effect_small": 0.0,
            "loss_weighted/mild_gate_effect_target": 0.0,
            "loss_weighted/gate_effect_gap_margin": 0.0,
            "loss_weighted/teacher_gate_effect_distill": 0.0,
            "loss_weighted/teacher_clean_gate_effect_preserve": 0.0,
            "loss_weighted/clean_attention_weight_anchor": 0.0,
            "loss_weighted/clean_reliability_anchor": 0.0,
            "loss_weighted/target_gate_effect_down": 0.0,
            "loss_weighted/target_bias_score_down": 0.0,
            "loss_weighted/gate_effect_score_alignment": 0.0,
            "loss_weighted/teacher_nonvision_gate_effect_preserve": 0.0,
            "loss_weighted/clean_bias_score_anchor": 0.0,
            "loss_weighted/target_prebias_score_down": 0.0,
            "loss_weighted/prebias_score_alignment": 0.0,
            "loss_weighted/clean_prebias_score_anchor": 0.0,
            "train/vision_gate_effect_delta_clean_minus_corrupt": 0.0,
            "train/vision_teacher_gate_effect_delta_clean_minus_corrupt": 0.0,
            "train/vision_reliability_gate_effect_alignment_ratio": 0.0,
            "train/vision_gate_effect_attention_alignment_ratio": 0.0,
            "train/vision_gate_effect_score_alignment_ratio": 0.0,
            "train/teacher_preserve_signal_absmax": 0.0,
            "train/teacher_effect_nonflat_preserve_flag": 0.0,
            "train/teacher_effect_nonflat_vision_flag": 0.0,
            "train/teacher_actuation_regime_mismatch_flag": 0.0,
            "train/teacher_preserve_active_flag": 0.0,
            "train/teacher_gate_effect_absmax": 0.0,
            "train/teacher_bias_score_absmax": 0.0,
            "train/attention_bias_score_mean_vision_clean": 0.0,
            "train/attention_bias_score_mean_vision_corrupt": 0.0,
            "train/vision_bias_score_delta_clean_minus_corrupt": 0.0,
            "train/vision_prebias_score_delta_clean_minus_corrupt": 0.0,
            "train/vision_prebias_score_final_score_alignment_ratio": 0.0,
            "train/clean_route_attention_anchor_error": 0.0,
            "train/clean_reliability_anchor_error": 0.0,
            "train/clean_bias_score_anchor_error": 0.0,
            "train/clean_prebias_score_anchor_error": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        proof_teacher_mode = str(losses_cfg.get("proof_teacher_mode", "synthetic_margin"))
        disable_vision_final_gate_teacher_preserve = bool(
            losses_cfg.get("disable_vision_final_gate_teacher_preserve", False)
        )
        disable_teacher_target_distill = bool(losses_cfg.get("disable_teacher_target_distill", False))
        reliability_target_mode = str(losses_cfg.get("reliability_target_mode", "legacy_aux"))
        vision_retention_coef = float(losses_cfg.get("vision_logit_retention_loss_coef", 0.0))
        ego_retention_coef = float(losses_cfg.get("ego_logit_retention_loss_coef", 0.0))
        vision_retention_margin = float(losses_cfg.get("vision_logit_retention_margin", 0.0))
        ego_retention_margin = float(losses_cfg.get("ego_logit_retention_margin", 0.0))
        vision_target_positive_coef = float(losses_cfg.get("vision_target_positive_loss_coef", 0.0))
        vision_logit_positive_coef = float(losses_cfg.get("vision_logit_positive_loss_coef", 0.0))
        ego_target_positive_coef = float(losses_cfg.get("ego_target_positive_loss_coef", 0.0))
        ego_logit_positive_coef = float(losses_cfg.get("ego_logit_positive_loss_coef", 0.0))
        local_residual_clean_zero_coef = float(losses_cfg.get("local_residual_clean_zero_loss_coef", 0.0))
        local_residual_selectivity_margin_coef = float(
            losses_cfg.get("local_residual_selectivity_margin_loss_coef", 0.0)
        )
        local_residual_selectivity_margin = float(losses_cfg.get("local_residual_selectivity_margin", 0.002))
        target_gate_down_loss_coef = float(losses_cfg.get("target_gate_down_loss_coef", 0.0))
        target_attention_down_loss_coef = float(losses_cfg.get("target_attention_down_loss_coef", 0.0))
        target_final_score_down_loss_coef = float(losses_cfg.get("target_final_score_down_loss_coef", 0.0))
        nontarget_rescue_loss_coef = float(losses_cfg.get("nontarget_rescue_loss_coef", 0.0))
        reliability_gate_alignment_loss_coef = float(
            losses_cfg.get("reliability_gate_alignment_loss_coef", 0.0)
        )
        gate_attention_alignment_loss_coef = float(losses_cfg.get("gate_attention_alignment_loss_coef", 0.0))
        reliability_attention_alignment_loss_coef = float(
            losses_cfg.get("reliability_attention_alignment_loss_coef", 0.0)
        )
        clean_gate_cap_loss_coef = float(losses_cfg.get("clean_gate_cap_loss_coef", 0.0))
        mild_gate_floor_loss_coef = float(losses_cfg.get("mild_gate_floor_loss_coef", 0.0))
        gate_gap_margin_loss_coef = float(losses_cfg.get("gate_gap_margin_loss_coef", 0.0))
        source_gate_distill_loss_coef = float(losses_cfg.get("source_gate_distill_loss_coef", 0.0))
        source_clean_gate_preserve_loss_coef = float(
            losses_cfg.get("source_clean_gate_preserve_loss_coef", 0.0)
        )
        clean_score_anchor_loss_coef = float(losses_cfg.get("clean_score_anchor_loss_coef", 0.0))
        clean_action_anchor_loss_coef = float(losses_cfg.get("clean_action_anchor_loss_coef", 0.0))
        source_gate_residual_zero_loss_coef = float(
            losses_cfg.get("source_gate_residual_zero_loss_coef", 0.0)
        )
        vision_final_gate_target_loss_coef = float(losses_cfg.get("vision_final_gate_target_loss_coef", 0.0))
        nonvision_gate_stability_loss_coef = float(losses_cfg.get("nonvision_gate_stability_loss_coef", 0.0))
        clean_gate_effect_small_loss_coef = float(losses_cfg.get("clean_gate_effect_small_loss_coef", 0.0))
        mild_gate_effect_target_loss_coef = float(losses_cfg.get("mild_gate_effect_target_loss_coef", 0.0))
        gate_effect_gap_margin_loss_coef = float(losses_cfg.get("gate_effect_gap_margin_loss_coef", 0.0))
        teacher_gate_effect_distill_loss_coef = float(
            losses_cfg.get("teacher_gate_effect_distill_loss_coef", 0.0)
        )
        teacher_clean_gate_effect_preserve_loss_coef = float(
            losses_cfg.get("teacher_clean_gate_effect_preserve_loss_coef", 0.0)
        )
        clean_attention_weight_anchor_loss_coef = float(
            losses_cfg.get("clean_attention_weight_anchor_loss_coef", 0.0)
        )
        clean_reliability_anchor_loss_coef = float(
            losses_cfg.get("clean_reliability_anchor_loss_coef", 0.0)
        )
        target_gate_effect_down_loss_coef = float(losses_cfg.get("target_gate_effect_down_loss_coef", 0.0))
        target_bias_score_down_loss_coef = float(losses_cfg.get("target_bias_score_down_loss_coef", 0.0))
        gate_effect_score_alignment_loss_coef = float(
            losses_cfg.get("gate_effect_score_alignment_loss_coef", 0.0)
        )
        teacher_nonvision_gate_effect_preserve_loss_coef = float(
            losses_cfg.get(
                "teacher_nonvision_gate_effect_preserve_loss_coef",
                teacher_clean_gate_effect_preserve_loss_coef,
            )
        )
        clean_bias_score_anchor_loss_coef = float(losses_cfg.get("clean_bias_score_anchor_loss_coef", 0.0))
        target_prebias_score_down_loss_coef = float(losses_cfg.get("target_prebias_score_down_loss_coef", 0.0))
        prebias_score_alignment_loss_coef = float(losses_cfg.get("prebias_score_alignment_loss_coef", 0.0))
        clean_prebias_score_anchor_loss_coef = float(losses_cfg.get("clean_prebias_score_anchor_loss_coef", 0.0))
        gate_down_margin = float(losses_cfg.get("gate_down_margin", 0.0))
        attention_down_margin = float(losses_cfg.get("attention_down_margin", 0.0))
        final_score_down_margin = float(losses_cfg.get("final_score_down_margin", 0.0))
        nontarget_rescue_margin = float(losses_cfg.get("nontarget_rescue_margin", 0.0))
        alignment_margin = float(losses_cfg.get("alignment_margin", 0.0))
        clean_gate_max = float(losses_cfg.get("clean_gate_max", 0.02))
        mild_gate_min = float(losses_cfg.get("mild_gate_min", 0.08))
        gate_gap_margin = float(losses_cfg.get("gate_gap_margin", 0.05))
        clean_gate_effect_abs_max = float(losses_cfg.get("clean_gate_effect_abs_max", 0.01))
        mild_gate_effect_min_abs = float(losses_cfg.get("mild_gate_effect_min_abs", 0.02))
        gate_effect_gap_margin = float(losses_cfg.get("gate_effect_gap_margin", 0.01))
        gate_effect_down_margin = float(losses_cfg.get("gate_effect_down_margin", gate_effect_gap_margin))
        bias_score_down_margin = float(losses_cfg.get("bias_score_down_margin", 0.01))
        prebias_score_down_margin = float(losses_cfg.get("prebias_score_down_margin", final_score_down_margin))
        score_alignment_margin = float(losses_cfg.get("score_alignment_margin", 0.005))
        fusion_shift_target_modality = str(losses_cfg.get("fusion_shift_target_modality", "vision"))
        target_positive_margin_mild = float(losses_cfg.get("target_positive_margin_mild", 0.0))
        logit_positive_margin_mild = float(losses_cfg.get("logit_positive_margin_mild", 0.0))
        local_residual_enabled_modalities = [
            str(name)
            for name in getattr(self.model, "reliability_local_residual_enabled_modalities", [])
            if str(name) in MODALITY_NAMES
        ]
        num_updates = 0
        for _ in range(self.base_cfg.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=int(self.raw_config["train"]["minibatch_num_sequences"]),
                device=self.device,
            ):
                clean_obs_seq = {key: batch.extras[f"clean_{key}"] for key in MODALITY_NAMES}
                target_mask = batch.extras["target_mask"]
                corrupted_mask = batch.extras["corrupted_flag"].squeeze(-1) * batch.valid_mask
                anchor_reference_clean_outputs = self._forward_reference_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    deterministic=False,
                    reference_model=self.anchor_reference_model,
                )
                anchor_reference_corrupt_outputs = self._forward_reference_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    deterministic=False,
                    reference_model=self.anchor_reference_model,
                )
                zero_corrupted_mask = torch.zeros_like(corrupted_mask)
                corrupt_gate_feature_context = self._build_sequence_gate_feature_context(
                    reference_clean_outputs=anchor_reference_clean_outputs,
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    sign_teacher_outputs=(
                        anchor_reference_corrupt_outputs if proof_teacher_mode == "reference_corrupt" else None
                    ),
                )
                clean_gate_feature_context = self._build_sequence_gate_feature_context(
                    reference_clean_outputs=anchor_reference_clean_outputs,
                    target_mask=target_mask,
                    corrupted_mask=zero_corrupted_mask,
                    sign_teacher_outputs=(
                        anchor_reference_clean_outputs if proof_teacher_mode == "reference_corrupt" else None
                    ),
                )
                current_clean_outputs_standalone = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    deterministic=False,
                    gate_feature_context=None,
                )
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                    gate_feature_context=corrupt_gate_feature_context,
                )
                clean_outputs = anchor_reference_clean_outputs
                current_clean_outputs_teacher = self.model.forward_sequence(
                    obs_seq=clean_obs_seq,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    deterministic=False,
                    gate_feature_context=clean_gate_feature_context,
                )
                valid_mask = batch.valid_mask
                teacher_preserve_signal_absmax = valid_mask.new_tensor(0.0)
                teacher_gate_effect_absmax = valid_mask.new_tensor(0.0)
                teacher_bias_score_absmax = valid_mask.new_tensor(0.0)
                teacher_actuation_regime_mismatch_flag = valid_mask.new_tensor(0.0)
                teacher_preserve_active_flag = valid_mask.new_tensor(0.0)
                teacher_effect_nonflat_preserve_flag = False
                teacher_effect_available = self.teacher_reference_model is not None
                if teacher_effect_available:
                    teacher_reference_clean_outputs = self._forward_reference_sequence(
                        obs_seq=clean_obs_seq,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        deterministic=False,
                        gate_feature_context=clean_gate_feature_context,
                        reference_model=self.teacher_reference_model,
                    )
                    teacher_reference_corrupt_outputs = self._forward_reference_sequence(
                        obs_seq=batch.observations,
                        episode_starts=batch.episode_starts,
                        initial_hidden=batch.initial_hidden,
                        deterministic=False,
                        gate_feature_context=corrupt_gate_feature_context,
                        reference_model=self.teacher_reference_model,
                    )
                    teacher_preserve_signal_absmax = compute_stage2_teacher_preserve_signal_absmax(
                        clean_teacher_gate_effects=teacher_reference_clean_outputs["attention_bias_gate_effects"],
                        corrupt_teacher_gate_effects=teacher_reference_corrupt_outputs["attention_bias_gate_effects"],
                        target_modality=fusion_shift_target_modality,
                        disable_teacher_target_distill=disable_teacher_target_distill,
                        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
                        valid_mask=valid_mask,
                    )
                    teacher_effect_nonflat_preserve_flag = bool(
                        teacher_preserve_signal_absmax.detach().item() >= 1.0e-5
                    )
                    (
                        teacher_gate_effect_absmax,
                        teacher_bias_score_absmax,
                        teacher_actuation_regime_mismatch_flag,
                        teacher_preserve_active_flag,
                    ) = compute_stage2_teacher_actuation_regime_state(
                        clean_current_gate_effects=current_clean_outputs_teacher["attention_bias_gate_effects"],
                        corrupt_current_gate_effects=outputs["attention_bias_gate_effects"],
                        clean_teacher_gate_effects=teacher_reference_clean_outputs["attention_bias_gate_effects"],
                        corrupt_teacher_gate_effects=teacher_reference_corrupt_outputs["attention_bias_gate_effects"],
                        clean_current_bias_scores=current_clean_outputs_teacher["attention_bias_scores"],
                        corrupt_current_bias_scores=outputs["attention_bias_scores"],
                        clean_teacher_bias_scores=teacher_reference_clean_outputs["attention_bias_scores"],
                        corrupt_teacher_bias_scores=teacher_reference_corrupt_outputs["attention_bias_scores"],
                        target_modality=fusion_shift_target_modality,
                        disable_teacher_target_distill=disable_teacher_target_distill,
                        disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
                        valid_mask=valid_mask,
                    )
                    teacher_effect_available = bool(teacher_preserve_active_flag.detach().item() > 0.0)
                else:
                    teacher_reference_clean_outputs = {
                        "attention_bias_gates": torch.zeros_like(
                            current_clean_outputs_teacher["attention_bias_gates"]
                        ),
                        "attention_bias_gate_effects": torch.zeros_like(
                            current_clean_outputs_teacher["attention_bias_gate_effects"]
                        ),
                        "attention_bias_scores": torch.zeros_like(
                            current_clean_outputs_teacher["attention_bias_scores"]
                        ),
                    }
                    teacher_reference_corrupt_outputs = {
                        "attention_bias_gates": torch.zeros_like(outputs["attention_bias_gates"]),
                        "attention_bias_gate_effects": torch.zeros_like(outputs["attention_bias_gate_effects"]),
                        "attention_bias_scores": torch.zeros_like(outputs["attention_bias_scores"]),
                    }
                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                ratio = torch.exp(outputs["log_probs"] - batch.old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.base_cfg.ppo.clip_eps, 1.0 + self.base_cfg.ppo.clip_eps)
                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((outputs["values"] - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (outputs["entropies"] * valid_mask).sum() / valid_count
                expanded_mask = valid_mask.unsqueeze(-1)
                if reliability_target_mode == "reference_anchored":
                    reliability_targets_for_loss = build_stage2_reference_anchored_targets(
                        clean_reliability_scores=clean_outputs["reliability_scores"],
                        severity=batch.extras["severity"],
                        target_mask=batch.extras["target_mask"],
                        direct_target_drop_scale=float(losses_cfg["direct_target_drop_scale"]),
                    )
                else:
                    reliability_targets_for_loss = batch.extras["reliability_targets"]
                reliability_loss = (
                    ((outputs["reliability_scores"] - reliability_targets_for_loss) ** 2) * expanded_mask
                ).sum() / expanded_mask.sum().clamp(min=1.0)
                action_diff = torch.abs(outputs["action_means"] - clean_outputs["action_means"]).mean(dim=-1)
                consistency_gap = (action_diff * corrupted_mask).sum() / corrupted_mask.sum().clamp(min=1.0)
                consistency_loss = (
                    ((outputs["action_means"] - clean_outputs["action_means"]) ** 2).mean(dim=-1) * corrupted_mask
                ).sum() / corrupted_mask.sum().clamp(min=1.0)
                single_target_mask = ((target_mask.sum(dim=-1) == 1).float() * corrupted_mask)
                vision_target_mask = single_target_mask * target_mask[..., 0]
                ego_target_mask = single_target_mask * target_mask[..., 2]
                (
                    vision_positive_loss,
                    vision_logit_positive_loss,
                    vision_positive_active_ratio,
                    vision_logit_positive_active_ratio,
                ) = compute_stage2_modality_positive_losses(
                    reliability_scores=outputs["reliability_scores"],
                    reliability_logits=outputs["reliability_logits"],
                    clean_reliability_scores=clean_outputs["reliability_scores"],
                    clean_reliability_logits=clean_outputs["reliability_logits"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    modality_name="vision",
                    positive_margin=target_positive_margin_mild,
                    logit_margin=logit_positive_margin_mild,
                )
                (
                    ego_positive_loss,
                    ego_logit_positive_loss,
                    ego_positive_active_ratio,
                    ego_logit_positive_active_ratio,
                ) = compute_stage2_ego_positive_losses(
                    reliability_scores=outputs["reliability_scores"],
                    reliability_logits=outputs["reliability_logits"],
                    clean_reliability_scores=clean_outputs["reliability_scores"],
                    clean_reliability_logits=clean_outputs["reliability_logits"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    positive_margin=target_positive_margin_mild,
                    logit_margin=logit_positive_margin_mild,
                )
                (
                    local_residual_clean_zero_loss,
                    local_residual_selectivity_margin_loss,
                ) = compute_stage2_local_residual_losses(
                    local_residual_deltas=outputs["reliability_local_residual_deltas"],
                    current_clean_local_residual_deltas=current_clean_outputs_teacher["reliability_local_residual_deltas"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    valid_mask=valid_mask,
                    enabled_modalities=local_residual_enabled_modalities,
                    selectivity_margin=local_residual_selectivity_margin,
                )
                (
                    target_gate_down_loss,
                    target_attention_down_loss,
                    target_final_score_down_loss,
                    nontarget_rescue_loss,
                    reliability_gate_alignment_loss,
                    gate_attention_alignment_loss,
                    reliability_attention_alignment_loss,
                    gate_attention_alignment_ratio,
                    reliability_gate_alignment_ratio,
                    vision_alignment_ratio,
                ) = compute_stage2_fusion_shift_losses(
                    clean_attention_weights=current_clean_outputs_teacher["attention_weights"],
                    corrupt_attention_weights=outputs["attention_weights"],
                    clean_final_scores=current_clean_outputs_teacher["attention_final_scores"],
                    corrupt_final_scores=outputs["attention_final_scores"],
                    clean_gate_scores=current_clean_outputs_teacher["attention_bias_gates"],
                    corrupt_gate_scores=outputs["attention_bias_gates"],
                    clean_reliability_scores=current_clean_outputs_teacher["reliability_scores"],
                    corrupt_reliability_scores=outputs["reliability_scores"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    valid_mask=valid_mask,
                    target_modality=fusion_shift_target_modality,
                    gate_down_margin=gate_down_margin,
                    attention_down_margin=attention_down_margin,
                    final_score_down_margin=final_score_down_margin,
                    nontarget_rescue_margin=nontarget_rescue_margin,
                    alignment_margin=alignment_margin,
                )
                (
                    clean_gate_cap_loss,
                    mild_gate_floor_loss,
                    gate_gap_margin_loss,
                    source_gate_distill_loss,
                    source_clean_gate_preserve_loss,
                    clean_score_anchor_loss,
                    clean_action_anchor_loss,
                    source_gate_residual_zero_loss,
                    vision_final_gate_target_loss,
                    nonvision_gate_stability_loss,
                ) = compute_stage2_teacher_distilled_gate_losses(
                    clean_gate_scores=current_clean_outputs_teacher["attention_bias_gates"],
                    corrupt_gate_scores=outputs["attention_bias_gates"],
                    clean_teacher_gate_scores=teacher_reference_clean_outputs["attention_bias_gates"],
                    corrupt_teacher_gate_scores=teacher_reference_corrupt_outputs["attention_bias_gates"],
                    clean_gate_residual_logits=current_clean_outputs_teacher["attention_bias_gate_residual_logits"],
                    clean_attention_scores_standalone=current_clean_outputs_standalone["attention_scores"],
                    reference_clean_attention_scores=anchor_reference_clean_outputs["attention_scores"],
                    clean_action_means_standalone=current_clean_outputs_standalone["action_means"],
                    reference_clean_action_means=anchor_reference_clean_outputs["action_means"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    valid_mask=valid_mask,
                    target_modality=fusion_shift_target_modality,
                    clean_gate_max=clean_gate_max,
                    mild_gate_min=mild_gate_min,
                    gate_gap_margin=gate_gap_margin,
                    disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
                )
                (
                    target_gate_effect_down_loss,
                    _effect_attention_down_loss,
                    _effect_final_score_down_loss,
                    target_bias_score_down_loss,
                    _effect_nontarget_rescue_loss,
                    reliability_gate_effect_alignment_ratio,
                    gate_effect_attention_alignment_ratio,
                    gate_effect_score_alignment_ratio,
                    gate_effect_score_alignment_loss,
                ) = compute_stage2_fusion_shift_gate_effect_losses(
                    clean_attention_weights=current_clean_outputs_teacher["attention_weights"],
                    corrupt_attention_weights=outputs["attention_weights"],
                    clean_final_scores=current_clean_outputs_teacher["attention_final_scores"],
                    corrupt_final_scores=outputs["attention_final_scores"],
                    clean_gate_effects=current_clean_outputs_teacher["attention_bias_gate_effects"],
                    corrupt_gate_effects=outputs["attention_bias_gate_effects"],
                    clean_bias_scores=current_clean_outputs_teacher["attention_bias_scores"],
                    corrupt_bias_scores=outputs["attention_bias_scores"],
                    clean_reliability_scores=current_clean_outputs_teacher["reliability_scores"],
                    corrupt_reliability_scores=outputs["reliability_scores"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    valid_mask=valid_mask,
                    target_modality=fusion_shift_target_modality,
                    gate_effect_down_margin=gate_effect_down_margin,
                    attention_down_margin=attention_down_margin,
                    final_score_down_margin=final_score_down_margin,
                    bias_score_down_margin=bias_score_down_margin,
                    nontarget_rescue_margin=nontarget_rescue_margin,
                    alignment_margin=alignment_margin,
                    score_alignment_margin=score_alignment_margin,
                )
                (
                    target_prebias_score_down_loss,
                    prebias_score_final_score_alignment_ratio,
                    prebias_score_alignment_loss,
                ) = compute_stage2_fusion_shift_prebias_losses(
                    clean_prebias_scores=current_clean_outputs_teacher["attention_relative_scores_before_bias"],
                    corrupt_prebias_scores=outputs["attention_relative_scores_before_bias"],
                    clean_final_scores=current_clean_outputs_teacher["attention_final_scores"],
                    corrupt_final_scores=outputs["attention_final_scores"],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    valid_mask=valid_mask,
                    target_modality=fusion_shift_target_modality,
                    prebias_score_down_margin=prebias_score_down_margin,
                    alignment_margin=alignment_margin,
                    score_alignment_margin=score_alignment_margin,
                )
                (
                    teacher_nonvision_gate_effect_preserve_loss,
                    clean_attention_weight_anchor_loss,
                    clean_reliability_anchor_loss,
                    clean_bias_score_anchor_loss,
                    clean_prebias_score_anchor_loss,
                ) = compute_stage2_teacher_gate_effect_losses(
                    clean_gate_effects=current_clean_outputs_teacher["attention_bias_gate_effects"],
                    corrupt_gate_effects=outputs["attention_bias_gate_effects"],
                    clean_teacher_gate_effects=teacher_reference_clean_outputs["attention_bias_gate_effects"],
                    corrupt_teacher_gate_effects=teacher_reference_corrupt_outputs["attention_bias_gate_effects"],
                    clean_attention_weights_standalone=current_clean_outputs_standalone["attention_weights"],
                    anchor_clean_attention_weights=anchor_reference_clean_outputs["attention_weights"],
                    clean_reliability_scores_standalone=current_clean_outputs_standalone["reliability_scores"],
                    anchor_clean_reliability_scores=anchor_reference_clean_outputs["reliability_scores"],
                    clean_bias_scores_standalone=current_clean_outputs_standalone["attention_bias_scores"],
                    anchor_clean_bias_scores=anchor_reference_clean_outputs["attention_bias_scores"],
                    clean_prebias_scores_standalone=current_clean_outputs_standalone[
                        "attention_relative_scores_before_bias"
                    ],
                    anchor_clean_prebias_scores=anchor_reference_clean_outputs[
                        "attention_relative_scores_before_bias"
                    ],
                    target_mask=target_mask,
                    corrupted_mask=corrupted_mask,
                    valid_mask=valid_mask,
                    target_modality=fusion_shift_target_modality,
                    disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
                    disable_teacher_target_distill=disable_teacher_target_distill,
                    teacher_effect_available=teacher_effect_available,
                )
                zero_loss = valid_mask.new_tensor(0.0)
                clean_gate_effect_small_loss = zero_loss
                mild_gate_effect_target_loss = zero_loss
                gate_effect_gap_margin_loss = zero_loss
                teacher_gate_effect_distill_loss = zero_loss
                teacher_clean_gate_effect_preserve_loss = zero_loss
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
                total_loss = (
                    float(losses_cfg["actor_loss_coef"]) * actor_loss
                    + float(losses_cfg["value_loss_coef"]) * critic_loss
                    - float(losses_cfg["entropy_coef"]) * entropy
                    + float(losses_cfg["reliability_loss_coef"]) * reliability_loss
                    + float(losses_cfg["consistency_loss_coef"]) * consistency_loss
                    + vision_retention_coef * vision_retention_loss
                    + ego_retention_coef * ego_retention_loss
                    + vision_target_positive_coef * vision_positive_loss
                    + vision_logit_positive_coef * vision_logit_positive_loss
                    + ego_target_positive_coef * ego_positive_loss
                    + ego_logit_positive_coef * ego_logit_positive_loss
                    + local_residual_clean_zero_coef * local_residual_clean_zero_loss
                    + local_residual_selectivity_margin_coef * local_residual_selectivity_margin_loss
                    + target_gate_down_loss_coef * target_gate_down_loss
                    + target_attention_down_loss_coef * target_attention_down_loss
                    + target_final_score_down_loss_coef * target_final_score_down_loss
                    + nontarget_rescue_loss_coef * nontarget_rescue_loss
                    + reliability_gate_alignment_loss_coef * reliability_gate_alignment_loss
                    + gate_attention_alignment_loss_coef * gate_attention_alignment_loss
                    + reliability_attention_alignment_loss_coef * reliability_attention_alignment_loss
                    + clean_gate_cap_loss_coef * clean_gate_cap_loss
                    + mild_gate_floor_loss_coef * mild_gate_floor_loss
                    + gate_gap_margin_loss_coef * gate_gap_margin_loss
                    + source_gate_distill_loss_coef * source_gate_distill_loss
                    + source_clean_gate_preserve_loss_coef * source_clean_gate_preserve_loss
                    + clean_score_anchor_loss_coef * clean_score_anchor_loss
                    + clean_action_anchor_loss_coef * clean_action_anchor_loss
                    + source_gate_residual_zero_loss_coef * source_gate_residual_zero_loss
                    + vision_final_gate_target_loss_coef * vision_final_gate_target_loss
                    + nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss
                    + clean_gate_effect_small_loss_coef * clean_gate_effect_small_loss
                    + mild_gate_effect_target_loss_coef * mild_gate_effect_target_loss
                    + gate_effect_gap_margin_loss_coef * gate_effect_gap_margin_loss
                    + teacher_gate_effect_distill_loss_coef * teacher_gate_effect_distill_loss
                    + teacher_clean_gate_effect_preserve_loss_coef * teacher_clean_gate_effect_preserve_loss
                    + clean_attention_weight_anchor_loss_coef * clean_attention_weight_anchor_loss
                    + clean_reliability_anchor_loss_coef * clean_reliability_anchor_loss
                    + target_gate_effect_down_loss_coef * target_gate_effect_down_loss
                    + target_bias_score_down_loss_coef * target_bias_score_down_loss
                    + gate_effect_score_alignment_loss_coef * gate_effect_score_alignment_loss
                    + teacher_nonvision_gate_effect_preserve_loss_coef * teacher_nonvision_gate_effect_preserve_loss
                    + clean_bias_score_anchor_loss_coef * clean_bias_score_anchor_loss
                    + target_prebias_score_down_loss_coef * target_prebias_score_down_loss
                    + prebias_score_alignment_loss_coef * prebias_score_alignment_loss
                    + clean_prebias_score_anchor_loss_coef * clean_prebias_score_anchor_loss
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
                metrics["loss/vision_target_positive"] += float(vision_positive_loss.item())
                metrics["loss/vision_logit_positive"] += float(vision_logit_positive_loss.item())
                metrics["loss/ego_target_positive"] += float(ego_positive_loss.item())
                metrics["loss/ego_logit_positive"] += float(ego_logit_positive_loss.item())
                metrics["loss/local_residual_clean_zero"] += float(local_residual_clean_zero_loss.item())
                metrics["loss/local_residual_selectivity_margin"] += float(
                    local_residual_selectivity_margin_loss.item()
                )
                metrics["loss/target_gate_down"] += float(target_gate_down_loss.item())
                metrics["loss/target_attention_down"] += float(target_attention_down_loss.item())
                metrics["loss/target_final_score_down"] += float(target_final_score_down_loss.item())
                metrics["loss/nontarget_rescue"] += float(nontarget_rescue_loss.item())
                metrics["loss/reliability_gate_alignment"] += float(reliability_gate_alignment_loss.item())
                metrics["loss/gate_attention_alignment"] += float(gate_attention_alignment_loss.item())
                metrics["loss/reliability_attention_alignment"] += float(
                    reliability_attention_alignment_loss.item()
                )
                metrics["loss/clean_gate_cap"] += float(clean_gate_cap_loss.item())
                metrics["loss/mild_gate_floor"] += float(mild_gate_floor_loss.item())
                metrics["loss/gate_gap_margin"] += float(gate_gap_margin_loss.item())
                metrics["loss/source_gate_distill"] += float(source_gate_distill_loss.item())
                metrics["loss/source_clean_gate_preserve"] += float(source_clean_gate_preserve_loss.item())
                metrics["loss/clean_score_anchor"] += float(clean_score_anchor_loss.item())
                metrics["loss/clean_action_anchor"] += float(clean_action_anchor_loss.item())
                metrics["loss/source_gate_residual_zero"] += float(source_gate_residual_zero_loss.item())
                metrics["loss/vision_final_gate_target"] += float(vision_final_gate_target_loss.item())
                metrics["loss/nonvision_gate_stability"] += float(nonvision_gate_stability_loss.item())
                metrics["loss/clean_gate_effect_small"] += float(clean_gate_effect_small_loss.item())
                metrics["loss/mild_gate_effect_target"] += float(mild_gate_effect_target_loss.item())
                metrics["loss/gate_effect_gap_margin"] += float(gate_effect_gap_margin_loss.item())
                metrics["loss/teacher_gate_effect_distill"] += float(teacher_gate_effect_distill_loss.item())
                metrics["loss/teacher_clean_gate_effect_preserve"] += float(
                    teacher_clean_gate_effect_preserve_loss.item()
                )
                metrics["loss/clean_attention_weight_anchor"] += float(clean_attention_weight_anchor_loss.item())
                metrics["loss/clean_reliability_anchor"] += float(clean_reliability_anchor_loss.item())
                metrics["loss/target_gate_effect_down"] += float(target_gate_effect_down_loss.item())
                metrics["loss/target_bias_score_down"] += float(target_bias_score_down_loss.item())
                metrics["loss/gate_effect_score_alignment"] += float(gate_effect_score_alignment_loss.item())
                metrics["loss/teacher_nonvision_gate_effect_preserve"] += float(
                    teacher_nonvision_gate_effect_preserve_loss.item()
                )
                metrics["loss/clean_bias_score_anchor"] += float(clean_bias_score_anchor_loss.item())
                metrics["loss/target_prebias_score_down"] += float(target_prebias_score_down_loss.item())
                metrics["loss/prebias_score_alignment"] += float(prebias_score_alignment_loss.item())
                metrics["loss/clean_prebias_score_anchor"] += float(clean_prebias_score_anchor_loss.item())
                metrics["loss_weighted/vision_target_positive"] += float(
                    (vision_target_positive_coef * vision_positive_loss).item()
                )
                metrics["loss_weighted/vision_logit_positive"] += float(
                    (vision_logit_positive_coef * vision_logit_positive_loss).item()
                )
                metrics["loss_weighted/ego_target_positive"] += float(
                    (ego_target_positive_coef * ego_positive_loss).item()
                )
                metrics["loss_weighted/ego_logit_positive"] += float(
                    (ego_logit_positive_coef * ego_logit_positive_loss).item()
                )
                metrics["loss_weighted/local_residual_clean_zero"] += float(
                    (local_residual_clean_zero_coef * local_residual_clean_zero_loss).item()
                )
                metrics["loss_weighted/local_residual_selectivity_margin"] += float(
                    (local_residual_selectivity_margin_coef * local_residual_selectivity_margin_loss).item()
                )
                metrics["loss_weighted/target_gate_down"] += float(
                    (target_gate_down_loss_coef * target_gate_down_loss).item()
                )
                metrics["loss_weighted/target_attention_down"] += float(
                    (target_attention_down_loss_coef * target_attention_down_loss).item()
                )
                metrics["loss_weighted/target_final_score_down"] += float(
                    (target_final_score_down_loss_coef * target_final_score_down_loss).item()
                )
                metrics["loss_weighted/nontarget_rescue"] += float(
                    (nontarget_rescue_loss_coef * nontarget_rescue_loss).item()
                )
                metrics["loss_weighted/reliability_gate_alignment"] += float(
                    (reliability_gate_alignment_loss_coef * reliability_gate_alignment_loss).item()
                )
                metrics["loss_weighted/gate_attention_alignment"] += float(
                    (gate_attention_alignment_loss_coef * gate_attention_alignment_loss).item()
                )
                metrics["loss_weighted/reliability_attention_alignment"] += float(
                    (reliability_attention_alignment_loss_coef * reliability_attention_alignment_loss).item()
                )
                metrics["loss_weighted/clean_gate_cap"] += float(
                    (clean_gate_cap_loss_coef * clean_gate_cap_loss).item()
                )
                metrics["loss_weighted/mild_gate_floor"] += float(
                    (mild_gate_floor_loss_coef * mild_gate_floor_loss).item()
                )
                metrics["loss_weighted/gate_gap_margin"] += float(
                    (gate_gap_margin_loss_coef * gate_gap_margin_loss).item()
                )
                metrics["loss_weighted/source_gate_distill"] += float(
                    (source_gate_distill_loss_coef * source_gate_distill_loss).item()
                )
                metrics["loss_weighted/source_clean_gate_preserve"] += float(
                    (source_clean_gate_preserve_loss_coef * source_clean_gate_preserve_loss).item()
                )
                metrics["loss_weighted/clean_score_anchor"] += float(
                    (clean_score_anchor_loss_coef * clean_score_anchor_loss).item()
                )
                metrics["loss_weighted/clean_action_anchor"] += float(
                    (clean_action_anchor_loss_coef * clean_action_anchor_loss).item()
                )
                metrics["loss_weighted/source_gate_residual_zero"] += float(
                    (source_gate_residual_zero_loss_coef * source_gate_residual_zero_loss).item()
                )
                metrics["loss_weighted/vision_final_gate_target"] += float(
                    (vision_final_gate_target_loss_coef * vision_final_gate_target_loss).item()
                )
                metrics["loss_weighted/nonvision_gate_stability"] += float(
                    (nonvision_gate_stability_loss_coef * nonvision_gate_stability_loss).item()
                )
                metrics["loss_weighted/clean_gate_effect_small"] += float(
                    (clean_gate_effect_small_loss_coef * clean_gate_effect_small_loss).item()
                )
                metrics["loss_weighted/mild_gate_effect_target"] += float(
                    (mild_gate_effect_target_loss_coef * mild_gate_effect_target_loss).item()
                )
                metrics["loss_weighted/gate_effect_gap_margin"] += float(
                    (gate_effect_gap_margin_loss_coef * gate_effect_gap_margin_loss).item()
                )
                metrics["loss_weighted/teacher_gate_effect_distill"] += float(
                    (teacher_gate_effect_distill_loss_coef * teacher_gate_effect_distill_loss).item()
                )
                metrics["loss_weighted/teacher_clean_gate_effect_preserve"] += float(
                    (
                        teacher_clean_gate_effect_preserve_loss_coef
                        * teacher_clean_gate_effect_preserve_loss
                    ).item()
                )
                metrics["loss_weighted/clean_attention_weight_anchor"] += float(
                    (clean_attention_weight_anchor_loss_coef * clean_attention_weight_anchor_loss).item()
                )
                metrics["loss_weighted/clean_reliability_anchor"] += float(
                    (clean_reliability_anchor_loss_coef * clean_reliability_anchor_loss).item()
                )
                metrics["loss_weighted/target_gate_effect_down"] += float(
                    (target_gate_effect_down_loss_coef * target_gate_effect_down_loss).item()
                )
                metrics["loss_weighted/target_bias_score_down"] += float(
                    (target_bias_score_down_loss_coef * target_bias_score_down_loss).item()
                )
                metrics["loss_weighted/gate_effect_score_alignment"] += float(
                    (gate_effect_score_alignment_loss_coef * gate_effect_score_alignment_loss).item()
                )
                metrics["loss_weighted/teacher_nonvision_gate_effect_preserve"] += float(
                    (
                        teacher_nonvision_gate_effect_preserve_loss_coef
                        * teacher_nonvision_gate_effect_preserve_loss
                    ).item()
                )
                metrics["loss_weighted/clean_bias_score_anchor"] += float(
                    (clean_bias_score_anchor_loss_coef * clean_bias_score_anchor_loss).item()
                )
                metrics["loss_weighted/target_prebias_score_down"] += float(
                    (target_prebias_score_down_loss_coef * target_prebias_score_down_loss).item()
                )
                metrics["loss_weighted/prebias_score_alignment"] += float(
                    (prebias_score_alignment_loss_coef * prebias_score_alignment_loss).item()
                )
                metrics["loss_weighted/clean_prebias_score_anchor"] += float(
                    (clean_prebias_score_anchor_loss_coef * clean_prebias_score_anchor_loss).item()
                )
                metrics["loss/grad_norm"] += grad_norm
                local_residual_deltas = outputs["reliability_local_residual_deltas"]
                metrics["train/vision_clean_anchor_score_mean"] += float(
                    (clean_outputs["reliability_scores"][..., 0] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/vision_corrupt_score_mean"] += float(
                    (outputs["reliability_scores"][..., 0] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/vision_target_score_mean"] += float(
                    (reliability_targets_for_loss[..., 0] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/ego_clean_anchor_score_mean"] += float(
                    (clean_outputs["reliability_scores"][..., 2] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/ego_corrupt_score_mean"] += float(
                    (outputs["reliability_scores"][..., 2] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/ego_target_score_mean"] += float(
                    (reliability_targets_for_loss[..., 2] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/vision_residual_delta_mean"] += float(
                    (local_residual_deltas[..., 0] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/vision_residual_delta_abs_mean"] += float(
                    (local_residual_deltas[..., 0].abs() * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/ego_residual_delta_mean"] += float(
                    (local_residual_deltas[..., 2] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/ego_residual_delta_abs_mean"] += float(
                    (local_residual_deltas[..., 2].abs() * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/vision_target_positive_active_ratio"] += float(vision_positive_active_ratio.item())
                metrics["train/vision_logit_positive_active_ratio"] += float(
                    vision_logit_positive_active_ratio.item()
                )
                metrics["train/ego_target_positive_active_ratio"] += float(ego_positive_active_ratio.item())
                metrics["train/ego_logit_positive_active_ratio"] += float(ego_logit_positive_active_ratio.item())
                vision_idx = MODALITY_NAMES.index("vision")
                metrics["train/vision_gate_delta_clean_minus_corrupt"] += float(
                    (((current_clean_outputs_teacher["attention_bias_gates"][..., vision_idx] - outputs["attention_bias_gates"][..., vision_idx]) * valid_mask).sum() / valid_count).item()
                )
                metrics["train/vision_gate_effect_delta_clean_minus_corrupt"] += float(
                    (
                        (
                            current_clean_outputs_teacher["attention_bias_gate_effects"][..., vision_idx]
                            - outputs["attention_bias_gate_effects"][..., vision_idx]
                        )
                        * valid_mask
                    ).sum().item()
                    / valid_count.item()
                )
                metrics["train/vision_source_gate_base_delta_clean_minus_corrupt"] += float(
                    (((current_clean_outputs_teacher["attention_source_bias_gates"][..., vision_idx] - outputs["attention_source_bias_gates"][..., vision_idx]) * valid_mask).sum() / valid_count).item()
                )
                metrics["train/vision_attention_delta_clean_minus_corrupt"] += float(
                    (((current_clean_outputs_teacher["attention_weights"][..., vision_idx] - outputs["attention_weights"][..., vision_idx]) * valid_mask).sum() / valid_count).item()
                )
                metrics["train/vision_final_score_delta_clean_minus_corrupt"] += float(
                    (((current_clean_outputs_teacher["attention_final_scores"][..., vision_idx] - outputs["attention_final_scores"][..., vision_idx]) * valid_mask).sum() / valid_count).item()
                )
                metrics["train/vision_gate_attention_alignment_ratio"] += float(gate_attention_alignment_ratio.item())
                metrics["train/vision_reliability_gate_alignment_ratio"] += float(
                    reliability_gate_alignment_ratio.item()
                )
                metrics["train/vision_reliability_attention_alignment_ratio"] += float(vision_alignment_ratio.item())
                metrics["train/vision_teacher_gate_delta_clean_minus_corrupt"] += float(
                    (((teacher_reference_clean_outputs["attention_bias_gates"][..., vision_idx] - teacher_reference_corrupt_outputs["attention_bias_gates"][..., vision_idx]) * valid_mask).sum() / valid_count).item()
                )
                metrics["train/vision_teacher_gate_effect_delta_clean_minus_corrupt"] += float(
                    (
                        (
                            teacher_reference_clean_outputs["attention_bias_gate_effects"][..., vision_idx]
                            - teacher_reference_corrupt_outputs["attention_bias_gate_effects"][..., vision_idx]
                        )
                        * valid_mask
                    ).sum().item()
                    / valid_count.item()
                )
                metrics["train/vision_reliability_gate_effect_alignment_ratio"] += float(
                    reliability_gate_effect_alignment_ratio.item()
                )
                metrics["train/vision_gate_effect_attention_alignment_ratio"] += float(
                    gate_effect_attention_alignment_ratio.item()
                )
                metrics["train/vision_gate_effect_score_alignment_ratio"] += float(
                    gate_effect_score_alignment_ratio.item()
                )
                metrics["train/teacher_preserve_signal_absmax"] += float(teacher_preserve_signal_absmax.item())
                metrics["train/teacher_effect_nonflat_preserve_flag"] += float(teacher_effect_nonflat_preserve_flag)
                metrics["train/teacher_effect_nonflat_vision_flag"] += float(
                    (
                        (
                            teacher_reference_clean_outputs["attention_bias_gate_effects"][..., vision_idx]
                            - teacher_reference_corrupt_outputs["attention_bias_gate_effects"][..., vision_idx]
                        )
                        .detach()
                        .abs()
                        .amax()
                        .item()
                        >= 1.0e-5
                    )
                )
                metrics["train/teacher_actuation_regime_mismatch_flag"] += float(
                    teacher_actuation_regime_mismatch_flag.item()
                )
                metrics["train/teacher_preserve_active_flag"] += float(teacher_preserve_active_flag.item())
                metrics["train/teacher_gate_effect_absmax"] += float(teacher_gate_effect_absmax.item())
                metrics["train/teacher_bias_score_absmax"] += float(teacher_bias_score_absmax.item())
                metrics["train/attention_bias_score_mean_vision_clean"] += float(
                    (current_clean_outputs_teacher["attention_bias_scores"][..., vision_idx] * valid_mask).sum().item()
                    / valid_count.item()
                )
                metrics["train/attention_bias_score_mean_vision_corrupt"] += float(
                    (outputs["attention_bias_scores"][..., vision_idx] * valid_mask).sum().item() / valid_count.item()
                )
                metrics["train/vision_bias_score_delta_clean_minus_corrupt"] += float(
                    (
                        (
                            current_clean_outputs_teacher["attention_bias_scores"][..., vision_idx]
                            - outputs["attention_bias_scores"][..., vision_idx]
                        )
                        * valid_mask
                    ).sum().item()
                    / valid_count.item()
                )
                metrics["train/vision_prebias_score_delta_clean_minus_corrupt"] += float(
                    (
                        (
                            current_clean_outputs_teacher["attention_relative_scores_before_bias"][..., vision_idx]
                            - outputs["attention_relative_scores_before_bias"][..., vision_idx]
                        )
                        * valid_mask
                    ).sum().item()
                    / valid_count.item()
                )
                metrics["train/vision_prebias_score_final_score_alignment_ratio"] += float(
                    prebias_score_final_score_alignment_ratio.item()
                )
                metrics["train/clean_score_anchor_error"] += float(clean_score_anchor_loss.item())
                metrics["train/clean_action_anchor_error"] += float(clean_action_anchor_loss.item())
                route_idx = MODALITY_NAMES.index("route")
                metrics["train/clean_route_attention_anchor_error"] += float(
                    (
                        (
                            torch.abs(
                                current_clean_outputs_standalone["attention_weights"][..., route_idx]
                                - anchor_reference_clean_outputs["attention_weights"][..., route_idx]
                            )
                            * valid_mask
                        ).sum().item()
                        / valid_count.item()
                    )
                )
                metrics["train/clean_reliability_anchor_error"] += float(clean_reliability_anchor_loss.item())
                metrics["train/clean_bias_score_anchor_error"] += float(clean_bias_score_anchor_loss.item())
                metrics["train/clean_prebias_score_anchor_error"] += float(clean_prebias_score_anchor_loss.item())
                num_updates += 1
        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    @torch.no_grad()
    def run_eval_suite(self, num_eval_episodes: int, run_mode: str = "full") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        env = self.eval_env
        eval_probes = self.raw_config["eval_probes"]
        losses_cfg = self.raw_config.get("losses", {})
        alignment_margin = float(losses_cfg.get("alignment_margin", 0.0))
        proof_teacher_mode = str(losses_cfg.get("proof_teacher_mode", "synthetic_margin"))
        disable_teacher_target_distill = bool(losses_cfg.get("disable_teacher_target_distill", False))
        disable_vision_final_gate_teacher_preserve = bool(
            losses_cfg.get("disable_vision_final_gate_teacher_preserve", False)
        )
        for probe in eval_probes:
            probe_name = str(probe["name"])
            returns = []
            successes = []
            collisions = []
            timeouts = []
            progress_values = []
            proof_clean_attention_values = []
            standalone_clean_attention_values = []
            policy_attention_values = []
            proof_clean_reliability_values = []
            policy_reliability_values = []
            proof_clean_final_score_values = []
            policy_final_score_values = []
            corrupt_final_score_values = []
            proof_clean_prebias_score_values = []
            policy_prebias_score_values = []
            corrupt_prebias_score_values = []
            proof_clean_gate_values = []
            policy_gate_values = []
            corrupt_gate_values = []
            proof_clean_gate_effect_values = []
            policy_gate_effect_values = []
            corrupt_gate_effect_values = []
            proof_clean_bias_score_values = []
            policy_bias_score_values = []
            corrupt_bias_score_values = []
            proof_clean_source_gate_base_values = []
            policy_source_gate_base_values = []
            corrupt_source_gate_base_values = []
            vision_residual_delta_values = []
            provisional_vision_residual_delta_values = []
            ego_residual_delta_values = []
            provisional_ego_residual_delta_values = []
            vision_logit_values = []
            ego_logit_values = []
            reference_vision_logit_values = []
            reference_ego_logit_values = []
            teacher_gate_mean_vision_clean_values = []
            teacher_gate_mean_vision_corrupt_values = []
            teacher_gate_effect_mean_vision_clean_values = []
            teacher_gate_effect_mean_vision_corrupt_values = []
            teacher_gate_effect_clean_values = []
            teacher_gate_effect_corrupt_values = []
            teacher_bias_score_clean_values = []
            teacher_bias_score_corrupt_values = []
            clean_score_anchor_error_values = []
            clean_action_anchor_error_values = []
            clean_route_attention_anchor_error_values = []
            clean_reliability_anchor_error_values = []
            clean_bias_score_anchor_error_values = []
            clean_prebias_score_anchor_error_values = []
            action_diffs = []
            reference_action_diffs = []
            reference_to_current_action_diffs = []
            rng = make_probe_rng(self.seed, 9000, probe_name)
            for episode_idx in range(num_eval_episodes):
                obs, info = env.reset(seed=self.seed * 100 + episode_idx)
                del info
                hidden = self.model.initial_hidden(batch_size=1, device=self.device)
                episode_start = torch.ones(1, device=self.device)
                episode_return = 0.0
                episode_progress = []
                while True:
                    clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
                    target_mask_np = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
                    corrupted_flag = 0.0
                    if probe_name == "normal_eval":
                        policy_obs = clean_obs
                    elif probe["kind"] == "single":
                        policy_obs, _severity = _apply_single_corruption(
                            obs=obs,
                            corruption_type=str(probe["modality"]),
                            corruption_strength=float(probe["strength"]),
                            rng=rng,
                            probe_name=probe_name,
                        )
                        modality_name = str(probe["modality"])
                        if modality_name in MODALITY_NAMES and float(probe["strength"]) > 0.0:
                            target_mask_np[MODALITY_NAMES.index(modality_name)] = 1.0
                            corrupted_flag = 1.0
                    else:
                        policy_obs, _severity = _apply_mixed_corruption(
                            obs=obs,
                            corruption_types=list(probe["modalities"]),
                            corruption_strengths=list(probe["strengths"]),
                            rng=rng,
                            probe_name=probe_name,
                        )
                        target_mask_np = (_severity > 0.0).astype(np.float32)
                        corrupted_flag = float(target_mask_np.sum() > 0.0)
                    clean_obs_tensor = obs_to_tensor(clean_obs, self.device)
                    clean_gate_feature_context = None
                    corrupt_gate_feature_context = None
                    anchor_reference_clean_output = self._reference_step(
                        obs_dict=clean_obs_tensor,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                        reference_model=self.anchor_reference_model,
                    )
                    anchor_reference_corrupt_output = self._reference_step(
                        obs_dict=obs_to_tensor(policy_obs, self.device),
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                        reference_model=self.anchor_reference_model,
                    )
                    if self.use_gate_feature_context:
                        clean_gate_feature_context = self._build_step_gate_feature_context(
                            clean_teacher_output=anchor_reference_clean_output,
                            target_mask=target_mask_np,
                            corrupted_flag=0.0,
                            sign_teacher_output=(
                                anchor_reference_clean_output if proof_teacher_mode == "reference_corrupt" else None
                            ),
                        )
                        corrupt_gate_feature_context = self._build_step_gate_feature_context(
                            clean_teacher_output=anchor_reference_clean_output,
                            target_mask=target_mask_np,
                            corrupted_flag=corrupted_flag,
                            sign_teacher_output=(
                                anchor_reference_corrupt_output if proof_teacher_mode == "reference_corrupt" else None
                            ),
                        )
                    teacher_reference_clean_output = None
                    teacher_reference_corrupt_output = None
                    if self.teacher_reference_model is not None:
                        teacher_reference_clean_output = self._reference_step(
                            obs_dict=clean_obs_tensor,
                            hidden_state=hidden,
                            episode_start=episode_start,
                            deterministic=True,
                            gate_feature_context=clean_gate_feature_context,
                            reference_model=self.teacher_reference_model,
                        )
                        teacher_reference_corrupt_output = self._reference_step(
                            obs_dict=obs_to_tensor(policy_obs, self.device),
                            hidden_state=hidden,
                            episode_start=episode_start,
                            deterministic=True,
                            gate_feature_context=corrupt_gate_feature_context,
                            reference_model=self.teacher_reference_model,
                        )
                    current_clean_output_standalone = self.model.step(
                        obs_dict=clean_obs_tensor,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                        gate_feature_context=None,
                    )
                    current_clean_output_teacher = self.model.step(
                        obs_dict=clean_obs_tensor,
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                        gate_feature_context=clean_gate_feature_context,
                    )
                    corrupt_output = self.model.step(
                        obs_dict=obs_to_tensor(policy_obs, self.device),
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                        gate_feature_context=corrupt_gate_feature_context,
                    )
                    if teacher_reference_clean_output is None or teacher_reference_corrupt_output is None:
                        teacher_reference_clean_output = SimpleNamespace(
                            aux_outputs={
                                "attention_bias_gates": torch.zeros_like(
                                    current_clean_output_teacher.aux_outputs["attention_bias_gates"]
                                ),
                                "attention_bias_gate_effects": torch.zeros_like(
                                    current_clean_output_teacher.aux_outputs["attention_bias_gate_effects"]
                                ),
                                "attention_bias_scores": torch.zeros_like(
                                    current_clean_output_teacher.aux_outputs["attention_bias_scores"]
                                ),
                            }
                        )
                        teacher_reference_corrupt_output = SimpleNamespace(
                            aux_outputs={
                                "attention_bias_gates": torch.zeros_like(
                                    corrupt_output.aux_outputs["attention_bias_gates"]
                                ),
                                "attention_bias_gate_effects": torch.zeros_like(
                                    corrupt_output.aux_outputs["attention_bias_gate_effects"]
                                ),
                                "attention_bias_scores": torch.zeros_like(
                                    corrupt_output.aux_outputs["attention_bias_scores"]
                                ),
                            }
                        )
                    active_policy_output = (
                        current_clean_output_standalone if probe_name == "normal_eval" else corrupt_output
                    )
                    action = active_policy_output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    proof_clean_attention = (
                        current_clean_output_teacher.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    )
                    standalone_clean_attention = (
                        current_clean_output_standalone.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    )
                    policy_attention = (
                        active_policy_output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    )
                    proof_clean_reliability = (
                        current_clean_output_teacher.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    policy_reliability = (
                        active_policy_output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    proof_clean_final_scores = (
                        current_clean_output_teacher.aux_outputs["attention_final_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    policy_final_scores = (
                        active_policy_output.aux_outputs["attention_final_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    corrupt_final_scores = (
                        corrupt_output.aux_outputs["attention_final_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    proof_clean_prebias_scores = (
                        current_clean_output_teacher.aux_outputs["attention_relative_scores_before_bias"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    policy_prebias_scores = (
                        active_policy_output.aux_outputs["attention_relative_scores_before_bias"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    corrupt_prebias_scores = (
                        corrupt_output.aux_outputs["attention_relative_scores_before_bias"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    proof_clean_gates = (
                        current_clean_output_teacher.aux_outputs["attention_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    policy_gates = (
                        active_policy_output.aux_outputs["attention_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    corrupt_gates = (
                        corrupt_output.aux_outputs["attention_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    proof_clean_gate_effects = (
                        current_clean_output_teacher.aux_outputs["attention_bias_gate_effects"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    policy_gate_effects = (
                        active_policy_output.aux_outputs["attention_bias_gate_effects"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    corrupt_gate_effects = (
                        corrupt_output.aux_outputs["attention_bias_gate_effects"].squeeze(0).detach().cpu().numpy()
                    )
                    proof_clean_bias_scores = (
                        current_clean_output_teacher.aux_outputs["attention_bias_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    policy_bias_scores = (
                        active_policy_output.aux_outputs["attention_bias_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    corrupt_bias_scores = (
                        corrupt_output.aux_outputs["attention_bias_scores"].squeeze(0).detach().cpu().numpy()
                    )
                    proof_clean_source_gates = (
                        current_clean_output_teacher.aux_outputs["attention_source_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    policy_source_gates = (
                        active_policy_output.aux_outputs["attention_source_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    corrupt_source_gates = (
                        corrupt_output.aux_outputs["attention_source_bias_gates"].squeeze(0).detach().cpu().numpy()
                    )
                    local_residual_delta = (
                        corrupt_output.aux_outputs.get(
                            "reliability_local_residual_deltas",
                            corrupt_output.aux_outputs.get("reliability_ego_residual_deltas", torch.zeros(1, len(MODALITY_NAMES))),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    provisional_local_residual_delta = (
                        corrupt_output.aux_outputs.get(
                            "provisional_reliability_local_residual_deltas",
                            corrupt_output.aux_outputs.get(
                                "provisional_reliability_ego_residual_deltas",
                                torch.zeros(1, len(MODALITY_NAMES)),
                            ),
                        )
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    logits = corrupt_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    current_clean_logits = (
                        current_clean_output_teacher.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    )
                    reference_clean_logits = anchor_reference_clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    proof_clean_attention_values.append(proof_clean_attention)
                    standalone_clean_attention_values.append(standalone_clean_attention)
                    policy_attention_values.append(policy_attention)
                    proof_clean_reliability_values.append(proof_clean_reliability)
                    policy_reliability_values.append(policy_reliability)
                    proof_clean_final_score_values.append(proof_clean_final_scores)
                    policy_final_score_values.append(policy_final_scores)
                    corrupt_final_score_values.append(corrupt_final_scores)
                    proof_clean_prebias_score_values.append(proof_clean_prebias_scores)
                    policy_prebias_score_values.append(policy_prebias_scores)
                    corrupt_prebias_score_values.append(corrupt_prebias_scores)
                    proof_clean_gate_values.append(proof_clean_gates)
                    policy_gate_values.append(policy_gates)
                    corrupt_gate_values.append(corrupt_gates)
                    proof_clean_gate_effect_values.append(proof_clean_gate_effects)
                    policy_gate_effect_values.append(policy_gate_effects)
                    corrupt_gate_effect_values.append(corrupt_gate_effects)
                    proof_clean_bias_score_values.append(proof_clean_bias_scores)
                    policy_bias_score_values.append(policy_bias_scores)
                    corrupt_bias_score_values.append(corrupt_bias_scores)
                    proof_clean_source_gate_base_values.append(proof_clean_source_gates)
                    policy_source_gate_base_values.append(policy_source_gates)
                    corrupt_source_gate_base_values.append(corrupt_source_gates)
                    vision_residual_delta_values.append(float(local_residual_delta[MODALITY_NAMES.index("vision")]))
                    ego_residual_delta_values.append(float(local_residual_delta[MODALITY_NAMES.index("ego")]))
                    provisional_vision_residual_delta_values.append(
                        float(provisional_local_residual_delta[MODALITY_NAMES.index("vision")])
                    )
                    provisional_ego_residual_delta_values.append(
                        float(provisional_local_residual_delta[MODALITY_NAMES.index("ego")])
                    )
                    vision_logit_values.append((float(current_clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(current_clean_logits[2]), float(logits[2])))
                    reference_vision_logit_values.append(float(reference_clean_logits[0]))
                    reference_ego_logit_values.append(float(reference_clean_logits[2]))
                    teacher_gate_mean_vision_clean_values.append(
                        float(teacher_reference_clean_output.aux_outputs["attention_bias_gates"].squeeze(0)[MODALITY_NAMES.index("vision")].detach().cpu().item())
                    )
                    teacher_gate_mean_vision_corrupt_values.append(
                        float(teacher_reference_corrupt_output.aux_outputs["attention_bias_gates"].squeeze(0)[MODALITY_NAMES.index("vision")].detach().cpu().item())
                    )
                    teacher_gate_effect_mean_vision_clean_values.append(
                        float(teacher_reference_clean_output.aux_outputs["attention_bias_gate_effects"].squeeze(0)[MODALITY_NAMES.index("vision")].detach().cpu().item())
                    )
                    teacher_gate_effect_mean_vision_corrupt_values.append(
                        float(teacher_reference_corrupt_output.aux_outputs["attention_bias_gate_effects"].squeeze(0)[MODALITY_NAMES.index("vision")].detach().cpu().item())
                    )
                    teacher_gate_effect_clean_values.append(
                        teacher_reference_clean_output.aux_outputs["attention_bias_gate_effects"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    teacher_gate_effect_corrupt_values.append(
                        teacher_reference_corrupt_output.aux_outputs["attention_bias_gate_effects"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    teacher_bias_score_clean_values.append(
                        teacher_reference_clean_output.aux_outputs["attention_bias_scores"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    teacher_bias_score_corrupt_values.append(
                        teacher_reference_corrupt_output.aux_outputs["attention_bias_scores"]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    clean_score_anchor_error_values.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    current_clean_output_standalone.aux_outputs["attention_scores"]
                                    - anchor_reference_clean_output.aux_outputs["attention_scores"]
                                )
                            ).cpu().item()
                        )
                    )
                    clean_action_anchor_error_values.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    current_clean_output_standalone.aux_outputs["action_mean"]
                                    - anchor_reference_clean_output.aux_outputs["action_mean"]
                                )
                            ).cpu().item()
                        )
                    )
                    clean_route_attention_anchor_error_values.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    current_clean_output_standalone.aux_outputs["attention_weights"][
                                        :, MODALITY_NAMES.index("route")
                                    ]
                                    - anchor_reference_clean_output.aux_outputs["attention_weights"][
                                        :, MODALITY_NAMES.index("route")
                                    ]
                                )
                            ).cpu().item()
                        )
                    )
                    clean_reliability_anchor_error_values.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    current_clean_output_standalone.aux_outputs["reliability_scores"]
                                    - anchor_reference_clean_output.aux_outputs["reliability_scores"]
                                )
                            ).cpu().item()
                        )
                    )
                    clean_bias_score_anchor_error_values.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    current_clean_output_standalone.aux_outputs["attention_bias_scores"]
                                    - anchor_reference_clean_output.aux_outputs["attention_bias_scores"]
                                )
                            ).cpu().item()
                        )
                    )
                    clean_prebias_score_anchor_error_values.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    current_clean_output_standalone.aux_outputs["attention_relative_scores_before_bias"]
                                    - anchor_reference_clean_output.aux_outputs["attention_relative_scores_before_bias"]
                                )
                            ).cpu().item()
                        )
                    )
                    action_diffs.append(
                        float(torch.mean(torch.abs(current_clean_output_teacher.action - corrupt_output.action)).cpu().item())
                    )
                    reference_action_diffs.append(
                        float(torch.mean(torch.abs(anchor_reference_clean_output.action - anchor_reference_corrupt_output.action)).cpu().item())
                    )
                    reference_to_current_action_diffs.append(
                        float(torch.mean(torch.abs(anchor_reference_clean_output.action - corrupt_output.action)).cpu().item())
                    )
                    hidden = active_policy_output.hidden_state
                    episode_start = torch.tensor([float(terminated or truncated)], device=self.device)
                    hidden = self.model.reset_hidden(hidden, episode_start)
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(float(info["goal_reached"]))
                        collisions.append(float(info["collision"]))
                        timeouts.append(float(info["timeout"]))
                        progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                        break
            proof_clean_att = np.stack(proof_clean_attention_values, axis=0)
            standalone_clean_att = np.stack(standalone_clean_attention_values, axis=0)
            att = np.stack(policy_attention_values, axis=0)
            proof_clean_rel = np.stack(proof_clean_reliability_values, axis=0)
            rel = np.stack(policy_reliability_values, axis=0)
            proof_clean_final = np.stack(proof_clean_final_score_values, axis=0)
            policy_final = np.stack(policy_final_score_values, axis=0)
            corrupt_final = np.stack(corrupt_final_score_values, axis=0)
            proof_clean_prebias = np.stack(proof_clean_prebias_score_values, axis=0)
            policy_prebias = np.stack(policy_prebias_score_values, axis=0)
            corrupt_prebias = np.stack(corrupt_prebias_score_values, axis=0)
            proof_clean_gate = np.stack(proof_clean_gate_values, axis=0)
            policy_gate = np.stack(policy_gate_values, axis=0)
            corrupt_gate = np.stack(corrupt_gate_values, axis=0)
            proof_clean_gate_effect = np.stack(proof_clean_gate_effect_values, axis=0)
            policy_gate_effect = np.stack(policy_gate_effect_values, axis=0)
            corrupt_gate_effect = np.stack(corrupt_gate_effect_values, axis=0)
            proof_clean_bias_scores = np.stack(proof_clean_bias_score_values, axis=0)
            policy_bias_scores = np.stack(policy_bias_score_values, axis=0)
            corrupt_bias_scores = np.stack(corrupt_bias_score_values, axis=0)
            proof_clean_source = np.stack(proof_clean_source_gate_base_values, axis=0)
            policy_source = np.stack(policy_source_gate_base_values, axis=0)
            corrupt_source = np.stack(corrupt_source_gate_base_values, axis=0)
            vision_logit_arr = np.asarray(vision_logit_values, dtype=np.float32)
            ego_logit_arr = np.asarray(ego_logit_values, dtype=np.float32)
            reference_vision_logit_arr = np.asarray(reference_vision_logit_values, dtype=np.float32)
            reference_ego_logit_arr = np.asarray(reference_ego_logit_values, dtype=np.float32)
            target_modality = None
            if probe_name != "normal_eval":
                if probe["kind"] == "single" and str(probe["modality"]) in MODALITY_NAMES:
                    target_modality = str(probe["modality"])
            target_attention_delta = 0.0
            target_final_score_delta = 0.0
            target_gate_delta = 0.0
            target_gate_effect_delta = 0.0
            target_bias_score_delta = 0.0
            target_prebias_score_delta = 0.0
            target_source_gate_delta = 0.0
            target_teacher_gate_delta = 0.0
            target_teacher_gate_effect_delta = 0.0
            target_reliability_delta = 0.0
            reliability_attention_alignment_ratio = 0.0
            reliability_gate_alignment_ratio = 0.0
            reliability_gate_effect_alignment_ratio = 0.0
            gate_attention_alignment_ratio = 0.0
            gate_effect_attention_alignment_ratio = 0.0
            gate_effect_score_alignment_ratio = 0.0
            prebias_score_final_score_alignment_ratio = 0.0
            nontarget_rescue_delta = 0.0
            if target_modality is not None:
                target_idx = MODALITY_NAMES.index(target_modality)
                non_target_indices = [idx for idx, name in enumerate(MODALITY_NAMES) if name != target_modality]
                rel_delta = proof_clean_rel[:, target_idx] - rel[:, target_idx]
                att_delta = proof_clean_att[:, target_idx] - att[:, target_idx]
                final_delta = proof_clean_final[:, target_idx] - corrupt_final[:, target_idx]
                gate_delta = proof_clean_gate[:, target_idx] - corrupt_gate[:, target_idx]
                gate_effect_delta = proof_clean_gate_effect[:, target_idx] - corrupt_gate_effect[:, target_idx]
                bias_score_delta = proof_clean_bias_scores[:, target_idx] - corrupt_bias_scores[:, target_idx]
                prebias_score_delta = proof_clean_prebias[:, target_idx] - corrupt_prebias[:, target_idx]
                source_gate_base_delta = proof_clean_source[:, target_idx] - corrupt_source[:, target_idx]
                teacher_gate_delta = (
                    np.asarray(teacher_gate_mean_vision_clean_values, dtype=np.float32)
                    - np.asarray(teacher_gate_mean_vision_corrupt_values, dtype=np.float32)
                )
                teacher_gate_effect_delta = (
                    np.asarray(teacher_gate_effect_mean_vision_clean_values, dtype=np.float32)
                    - np.asarray(teacher_gate_effect_mean_vision_corrupt_values, dtype=np.float32)
                )
                target_reliability_delta = float(rel_delta.mean())
                target_attention_delta = float(att_delta.mean())
                target_final_score_delta = float(final_delta.mean())
                target_gate_delta = float(gate_delta.mean())
                target_gate_effect_delta = float(gate_effect_delta.mean())
                target_bias_score_delta = float(bias_score_delta.mean())
                target_prebias_score_delta = float(prebias_score_delta.mean())
                target_source_gate_delta = float(source_gate_base_delta.mean())
                target_teacher_gate_delta = float(teacher_gate_delta.mean()) if teacher_gate_delta.size else 0.0
                target_teacher_gate_effect_delta = (
                    float(teacher_gate_effect_delta.mean()) if teacher_gate_effect_delta.size else 0.0
                )
                active_alignment = rel_delta > alignment_margin
                if np.any(active_alignment):
                    reliability_attention_alignment_ratio = float((att_delta[active_alignment] > 0.0).mean())
                    reliability_gate_alignment_ratio = float((gate_delta[active_alignment] > 0.0).mean())
                    reliability_gate_effect_alignment_ratio = float(
                        (gate_effect_delta[active_alignment] > 0.0).mean()
                    )
                gate_active_alignment = gate_delta > alignment_margin
                if np.any(gate_active_alignment):
                    gate_attention_alignment_ratio = float((att_delta[gate_active_alignment] > 0.0).mean())
                gate_effect_active_alignment = gate_effect_delta > alignment_margin
                if np.any(gate_effect_active_alignment):
                    gate_effect_attention_alignment_ratio = float(
                        (att_delta[gate_effect_active_alignment] > 0.0).mean()
                    )
                    gate_effect_score_alignment_ratio = float(
                        (final_delta[gate_effect_active_alignment] > 0.0).mean()
                    )
                prebias_active_alignment = prebias_score_delta > alignment_margin
                if np.any(prebias_active_alignment):
                    prebias_score_final_score_alignment_ratio = float(
                        (final_delta[prebias_active_alignment] > 0.0).mean()
                    )
                clean_non_target_attention = proof_clean_att[:, non_target_indices].mean(axis=1)
                corrupt_non_target_attention = att[:, non_target_indices].mean(axis=1)
                clean_non_target_final = proof_clean_final[:, non_target_indices].mean(axis=1)
                corrupt_non_target_final = corrupt_final[:, non_target_indices].mean(axis=1)
                nontarget_rescue_delta = float(
                    0.5
                    * (
                        (corrupt_non_target_attention - clean_non_target_attention).mean()
                        + (corrupt_non_target_final - clean_non_target_final).mean()
                    )
                )
            teacher_gate_mean_vision_clean = float(np.mean(teacher_gate_mean_vision_clean_values)) if teacher_gate_mean_vision_clean_values else 0.0
            teacher_gate_mean_vision_corrupt = float(np.mean(teacher_gate_mean_vision_corrupt_values)) if teacher_gate_mean_vision_corrupt_values else 0.0
            teacher_gate_effect_mean_vision_clean = float(np.mean(teacher_gate_effect_mean_vision_clean_values)) if teacher_gate_effect_mean_vision_clean_values else 0.0
            teacher_gate_effect_mean_vision_corrupt = float(np.mean(teacher_gate_effect_mean_vision_corrupt_values)) if teacher_gate_effect_mean_vision_corrupt_values else 0.0
            teacher_preserve_signal_absmax = 0.0
            teacher_effect_nonflat_preserve_flag = 0
            teacher_effect_nonflat_vision_flag = 0
            teacher_gate_effect_absmax = 0.0
            teacher_bias_score_absmax = 0.0
            teacher_actuation_regime_mismatch_flag = 0
            teacher_preserve_active_flag = 0
            if (
                self.teacher_reference_model is not None
                and target_modality is not None
                and teacher_gate_effect_clean_values
                and teacher_gate_effect_corrupt_values
            ):
                preserve_indices = select_stage2_non_target_preserve_indices(
                    target_modality=target_modality,
                    disable_teacher_target_distill=disable_teacher_target_distill,
                    disable_vision_final_gate_teacher_preserve=disable_vision_final_gate_teacher_preserve,
                )
                teacher_clean_gate_effect = np.stack(teacher_gate_effect_clean_values, axis=0)
                teacher_corrupt_gate_effect = np.stack(teacher_gate_effect_corrupt_values, axis=0)
                vision_idx = MODALITY_NAMES.index("vision")
                teacher_effect_nonflat_vision_flag = int(
                    np.max(
                        np.abs(
                            teacher_clean_gate_effect[:, vision_idx] - teacher_corrupt_gate_effect[:, vision_idx]
                        )
                    )
                    >= 1.0e-5
                )
                if preserve_indices:
                    teacher_clean_preserve = teacher_clean_gate_effect[:, preserve_indices]
                    teacher_corrupt_preserve = teacher_corrupt_gate_effect[:, preserve_indices]
                    teacher_clean_bias = np.stack(teacher_bias_score_clean_values, axis=0)[:, preserve_indices]
                    teacher_corrupt_bias = np.stack(teacher_bias_score_corrupt_values, axis=0)[:, preserve_indices]
                    current_clean_preserve = proof_clean_gate_effect[:, preserve_indices]
                    current_corrupt_preserve = corrupt_gate_effect[:, preserve_indices]
                    current_clean_bias = proof_clean_bias_scores[:, preserve_indices]
                    current_corrupt_bias = corrupt_bias_scores[:, preserve_indices]
                    teacher_gate_effect_absmax = float(
                        max(
                            float(np.max(np.abs(teacher_clean_preserve))),
                            float(np.max(np.abs(teacher_corrupt_preserve))),
                            float(np.max(np.abs(teacher_clean_preserve - teacher_corrupt_preserve))),
                        )
                    )
                    teacher_bias_score_absmax = float(
                        max(
                            float(np.max(np.abs(teacher_clean_bias))),
                            float(np.max(np.abs(teacher_corrupt_bias))),
                            float(np.max(np.abs(teacher_clean_bias - teacher_corrupt_bias))),
                        )
                    )
                    teacher_preserve_signal_absmax = float(
                        max(
                            float(np.max(np.abs(teacher_clean_preserve))),
                            float(np.max(np.abs(teacher_corrupt_preserve))),
                            float(np.max(np.abs(teacher_clean_preserve - teacher_corrupt_preserve))),
                        )
                    )
                    teacher_effect_nonflat_preserve_flag = int(teacher_preserve_signal_absmax >= 1.0e-5)
                    current_gate_effect_absmax = float(
                        max(
                            float(np.max(np.abs(current_clean_preserve))),
                            float(np.max(np.abs(current_corrupt_preserve))),
                            float(np.max(np.abs(current_clean_preserve - current_corrupt_preserve))),
                        )
                    )
                    current_bias_score_absmax = float(
                        max(
                            float(np.max(np.abs(current_clean_bias))),
                            float(np.max(np.abs(current_corrupt_bias))),
                            float(np.max(np.abs(current_clean_bias - current_corrupt_bias))),
                        )
                    )
                    def _ratio_mismatch(current_absmax: float, teacher_absmax: float) -> bool:
                        high = max(abs(current_absmax), abs(teacher_absmax))
                        low = min(abs(current_absmax), abs(teacher_absmax))
                        if high < 1.0e-5:
                            return False
                        if low < 1.0e-5:
                            return True
                        return high / max(low, 1.0e-12) >= 10.0
                    teacher_actuation_regime_mismatch_flag = int(
                        _ratio_mismatch(current_gate_effect_absmax, teacher_gate_effect_absmax)
                        or _ratio_mismatch(current_bias_score_absmax, teacher_bias_score_absmax)
                    )
                    teacher_preserve_active_flag = int(
                        teacher_effect_nonflat_preserve_flag == 1
                        and teacher_actuation_regime_mismatch_flag == 0
                    )
            clean_score_anchor_error = float(np.mean(clean_score_anchor_error_values)) if clean_score_anchor_error_values else 0.0
            clean_action_anchor_error = float(np.mean(clean_action_anchor_error_values)) if clean_action_anchor_error_values else 0.0
            clean_route_attention_anchor_error = float(np.mean(clean_route_attention_anchor_error_values)) if clean_route_attention_anchor_error_values else 0.0
            clean_reliability_anchor_error = float(np.mean(clean_reliability_anchor_error_values)) if clean_reliability_anchor_error_values else 0.0
            clean_bias_score_anchor_error = float(np.mean(clean_bias_score_anchor_error_values)) if clean_bias_score_anchor_error_values else 0.0
            clean_prebias_score_anchor_error = float(np.mean(clean_prebias_score_anchor_error_values)) if clean_prebias_score_anchor_error_values else 0.0
            row = {
                "candidate_name": self.candidate_name,
                "seed": self.seed,
                "env_mode": "normal" if probe_name == "normal_eval" else "mild",
                "corruption_type": probe_name,
                "num_eval_episodes": int(num_eval_episodes),
                "run_mode": str(run_mode),
                "train_step": int(self.raw_config["train"]["total_steps"]),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "consistency_loss": float(np.mean(np.square(action_diffs))),
                "consistency_gap_mean": float(np.mean(action_diffs)),
                "same_state_action_diff_clean_vs_corrupt": float(np.mean(action_diffs)),
                "same_state_action_diff_reference_clean_vs_corrupt": float(np.mean(reference_action_diffs)),
                "same_state_action_diff_reference_clean_vs_current_corrupt": float(
                    np.mean(reference_to_current_action_diffs)
                ),
                "clean_eval_variant": "standalone",
                "proof_clean_eval_variant": "teacher_conditioned",
                "target_modality_for_probe": target_modality or "none",
                "target_reliability_delta_clean_minus_corrupt": target_reliability_delta,
                "target_gate_delta_clean_minus_corrupt": target_gate_delta,
                "target_gate_effect_delta_clean_minus_corrupt": target_gate_effect_delta,
                "target_bias_score_delta_clean_minus_corrupt": target_bias_score_delta,
                "target_prebias_score_delta_clean_minus_corrupt": target_prebias_score_delta,
                "target_attention_delta_clean_minus_corrupt": target_attention_delta,
                "target_final_score_delta_clean_minus_corrupt": target_final_score_delta,
                "target_source_gate_base_delta_clean_minus_corrupt": target_source_gate_delta,
                "target_source_gate_delta_clean_minus_corrupt": target_source_gate_delta,
                "teacher_gate_mean_vision_clean": teacher_gate_mean_vision_clean,
                "teacher_gate_mean_vision_corrupt": teacher_gate_mean_vision_corrupt,
                "teacher_gate_effect_mean_vision_clean": teacher_gate_effect_mean_vision_clean,
                "teacher_gate_effect_mean_vision_corrupt": teacher_gate_effect_mean_vision_corrupt,
                "teacher_gate_effect_absmax": teacher_gate_effect_absmax,
                "teacher_bias_score_absmax": teacher_bias_score_absmax,
                "target_teacher_gate_delta_clean_minus_corrupt": teacher_gate_mean_vision_clean - teacher_gate_mean_vision_corrupt,
                "target_teacher_gate_effect_delta_clean_minus_corrupt": target_teacher_gate_effect_delta,
                "teacher_preserve_signal_absmax": teacher_preserve_signal_absmax,
                "teacher_effect_nonflat_preserve_flag": int(teacher_effect_nonflat_preserve_flag),
                "teacher_gate_effect_nonflat_preserve_flag": int(teacher_effect_nonflat_preserve_flag),
                "teacher_effect_nonflat_vision_flag": int(teacher_effect_nonflat_vision_flag),
                "teacher_effect_nonflat_flag": int(teacher_effect_nonflat_preserve_flag),
                "teacher_actuation_regime_mismatch_flag": int(teacher_actuation_regime_mismatch_flag),
                "teacher_preserve_active_flag": int(teacher_preserve_active_flag),
                "reliability_attention_alignment_ratio": reliability_attention_alignment_ratio,
                "reliability_gate_alignment_ratio": reliability_gate_alignment_ratio,
                "reliability_gate_effect_alignment_ratio": reliability_gate_effect_alignment_ratio,
                "gate_attention_alignment_ratio": gate_attention_alignment_ratio,
                "gate_effect_attention_alignment_ratio": gate_effect_attention_alignment_ratio,
                "gate_effect_score_alignment_ratio": gate_effect_score_alignment_ratio,
                "prebias_score_final_score_alignment_ratio": prebias_score_final_score_alignment_ratio,
                "nontarget_rescue_delta": nontarget_rescue_delta,
                "consistency_gain_score": 0.0,
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(
                    att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)
                ),
                "attention_mean_vision_clean": float(
                    (proof_clean_att if target_modality is not None else att)[:, MODALITY_NAMES.index("vision")].mean()
                ),
                "attention_mean_vision_corrupt": float(att[:, MODALITY_NAMES.index("vision")].mean()),
                "attention_final_score_mean_vision_clean": float(
                    (proof_clean_final if target_modality is not None else policy_final)[
                        :, MODALITY_NAMES.index("vision")
                    ].mean()
                ),
                "attention_final_score_mean_vision_corrupt": float(
                    corrupt_final[:, MODALITY_NAMES.index("vision")].mean()
                ),
                "attention_prebias_score_mean_vision_clean": float(
                    (proof_clean_prebias if target_modality is not None else policy_prebias)[
                        :, MODALITY_NAMES.index("vision")
                    ].mean()
                ),
                "attention_prebias_score_mean_vision_corrupt": float(
                    corrupt_prebias[:, MODALITY_NAMES.index("vision")].mean()
                ),
                "attention_gate_mean_vision_clean": float(
                    (proof_clean_gate if target_modality is not None else policy_gate)[
                        :, MODALITY_NAMES.index("vision")
                    ].mean()
                ),
                "attention_gate_mean_vision_corrupt": float(corrupt_gate[:, MODALITY_NAMES.index("vision")].mean()),
                "attention_bias_score_mean_vision_clean": float(
                    (proof_clean_bias_scores if target_modality is not None else policy_bias_scores)[
                        :, MODALITY_NAMES.index("vision")
                    ].mean()
                ),
                "attention_bias_score_mean_vision_corrupt": float(
                    corrupt_bias_scores[:, MODALITY_NAMES.index("vision")].mean()
                ),
                "attention_bias_gate_effect_mean_vision_clean": float(
                    (proof_clean_gate_effect if target_modality is not None else policy_gate_effect)[
                        :, MODALITY_NAMES.index("vision")
                    ].mean()
                ),
                "attention_bias_gate_effect_mean_vision_corrupt": float(
                    corrupt_gate_effect[:, MODALITY_NAMES.index("vision")].mean()
                ),
                "source_gate_base_mean_vision_clean": float(
                    (proof_clean_source if target_modality is not None else policy_source)[
                        :, MODALITY_NAMES.index("vision")
                    ].mean()
                ),
                "source_gate_base_mean_vision_corrupt": float(
                    corrupt_source[:, MODALITY_NAMES.index("vision")].mean()
                ),
                "source_gate_mean_vision_clean": float(proof_clean_source[:, MODALITY_NAMES.index("vision")].mean()),
                "source_gate_mean_vision_corrupt": float(corrupt_source[:, MODALITY_NAMES.index("vision")].mean()),
                "clean_score_anchor_error": clean_score_anchor_error,
                "clean_action_anchor_error": clean_action_anchor_error,
                "clean_route_attention_anchor_error": clean_route_attention_anchor_error,
                "clean_reliability_anchor_error": clean_reliability_anchor_error,
                "clean_bias_score_anchor_error": clean_bias_score_anchor_error,
                "clean_prebias_score_anchor_error": clean_prebias_score_anchor_error,
                "teacher_source": self.teacher_source,
                "attention_mean_distance_clean": float(standalone_clean_att[:, MODALITY_NAMES.index("distance")].mean()),
                "attention_mean_route_clean": float(standalone_clean_att[:, MODALITY_NAMES.index("route")].mean()),
                "attention_mean_ego_clean": float(standalone_clean_att[:, MODALITY_NAMES.index("ego")].mean()),
                "logit_mean_vision_clean": float(vision_logit_arr[:, 0].mean()),
                "logit_mean_vision_weak": (
                    float(vision_logit_arr[:, 1].mean())
                    if "vision" in probe_name and "mild" not in probe_name and probe_name != "normal_eval"
                    else np.nan
                ),
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": (
                    float(ego_logit_arr[:, 1].mean())
                    if "ego" in probe_name and "mild" not in probe_name and probe_name != "normal_eval"
                    else np.nan
                ),
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "reference_logit_mean_vision_clean": float(reference_vision_logit_arr.mean()),
                "reference_logit_mean_ego_clean": float(reference_ego_logit_arr.mean()),
                "current_clean_logit_mean_vision": float(vision_logit_arr[:, 0].mean()),
                "current_clean_logit_mean_ego": float(ego_logit_arr[:, 0].mean()),
                "reliability_vision_residual_delta_mean": float(np.mean(vision_residual_delta_values))
                if vision_residual_delta_values
                else 0.0,
                "reliability_vision_residual_delta_std": float(np.std(vision_residual_delta_values))
                if vision_residual_delta_values
                else 0.0,
                "reliability_ego_residual_delta_mean": float(np.mean(ego_residual_delta_values))
                if ego_residual_delta_values
                else 0.0,
                "reliability_ego_residual_delta_std": float(np.std(ego_residual_delta_values))
                if ego_residual_delta_values
                else 0.0,
                "provisional_reliability_vision_residual_delta_mean": float(
                    np.mean(provisional_vision_residual_delta_values)
                )
                if provisional_vision_residual_delta_values
                else 0.0,
                "provisional_reliability_vision_residual_delta_std": float(
                    np.std(provisional_vision_residual_delta_values)
                )
                if provisional_vision_residual_delta_values
                else 0.0,
                "provisional_reliability_ego_residual_delta_mean": float(np.mean(provisional_ego_residual_delta_values))
                if provisional_ego_residual_delta_values
                else 0.0,
                "provisional_reliability_ego_residual_delta_std": float(np.std(provisional_ego_residual_delta_values))
                if provisional_ego_residual_delta_values
                else 0.0,
                "experiment_mode": infer_experiment_mode(self.raw_config.get("losses", {})),
                "calibration_only_flag": float(is_calibration_only(self.raw_config.get("losses", {}))),
            }
            row.update(_probe_logging_fields(probe_name, probe))
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
            rows.append(row)
        return rows

    def train(self) -> None:
        total_steps = int(self.raw_config["train"]["total_steps"])
        rollout_steps = int(self.raw_config["train"]["rollout_steps"])
        num_updates = total_steps // rollout_steps
        start_time = time.time()
        for update_idx in range(1, num_updates + 1):
            rollout_metrics = self.collect_rollout(update_idx)
            loss_metrics = self.update_model()
            self.nan_detected = self.nan_detected or self._metrics_have_nan(rollout_metrics) or self._metrics_have_nan(loss_metrics)
            payload = {
                "update": update_idx,
                "train_step": update_idx * rollout_steps,
                "wall_clock_time": float(time.time() - start_time),
                "finished_flag": 0,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
                **rollout_metrics,
                **loss_metrics,
            }
            self.logger.log_train(payload)
            self.logger.log(payload)
            if update_idx % int(self.raw_config["train"]["eval_interval_updates"]) == 0:
                eval_rows = self.run_eval_suite(
                    num_eval_episodes=int(self.raw_config["num_eval_episodes"]),
                    run_mode="full",
                )
                normal = [row for row in eval_rows if row["corruption_type"] == "normal_eval"][0]
                eval_payload = {
                    "update": update_idx,
                    "train_step": update_idx * rollout_steps,
                    "wall_clock_time": float(time.time() - start_time),
                    "finished_flag": 0,
                    "nan_detected": int(self.nan_detected),
                    "crashed_flag": 0,
                    "eval/return_mean": normal["eval_return"],
                    "eval/success_rate": normal["eval_success_rate"],
                    "eval/collision_rate": normal["eval_collision_rate"],
                    "eval/timeout_rate": normal["eval_timeout_rate"],
                    "eval/avg_progress_to_goal": normal["eval_avg_progress_to_goal"],
                    "eval/consistency_gap_mean": normal["consistency_gap_mean"],
                }
                self.logger.log_eval(eval_payload)
                self.logger.log(eval_payload)
        torch.save({"model_state_dict": self.model.state_dict()}, self.run_dir / "checkpoint_final.pt")
        self.logger.save_status(
            {
                "finished_flag": 1,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
                "wall_clock_time": float(time.time() - start_time),
                "total_updates": num_updates,
            }
        )


def run_stage2_candidate(
    candidate_name: str,
    config: dict[str, Any],
    output_root: Path,
    metrics_filename: str,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = list(config["seeds"][:1] if smoke else config["seeds"])
    eval_episodes = int(config["num_eval_episodes_smoke"] if smoke else config["num_eval_episodes"])
    mode = "smoke" if smoke else "full"
    candidate_spec = dict(config.get("candidates", {}).get(candidate_name, {}))
    upstream_checkpoint_root = str(candidate_spec.get("checkpoint_root", ""))
    for seed in seeds:
        seed_dir_name = f"seed_{seed}_smoke" if smoke else f"seed_{seed}"
        run_dir = ensure_dir(output_root / candidate_name / seed_dir_name)
        summary_path = run_dir / metrics_filename
        metadata_path = run_dir / "run_metadata.json"
        expected_metadata = build_run_cache_metadata(
            config=config,
            candidate_name=candidate_name,
            metrics_filename=metrics_filename,
            mode=mode,
            num_eval_episodes=eval_episodes,
            step_name=str(config.get("experiment_name", "stage2_transition")),
            upstream_checkpoint_root=upstream_checkpoint_root,
        )
        if summary_path.exists() and should_reuse_run_cache(metadata_path, expected_metadata):
            rows.extend(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        trainer = Stage2TransitionTrainer(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
            smoke=smoke,
        )
        trainer.train()
        eval_rows = trainer.run_eval_suite(num_eval_episodes=eval_episodes, run_mode=mode)
        summary_path.write_text(json.dumps(eval_rows, ensure_ascii=True, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(expected_metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        rows.extend(eval_rows)
    candidate_root = ensure_dir(output_root / candidate_name)
    (candidate_root / f"eval_rows_{mode}.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    if not smoke:
        (candidate_root / "eval_rows.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    return rows


def run_stage2_eval_only_candidate(
    candidate_name: str,
    config: dict[str, Any],
    output_root: Path,
    metrics_filename: str,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = list(config["seeds"][:1] if smoke else config["seeds"])
    eval_episodes = int(config["num_eval_episodes_smoke"] if smoke else config["num_eval_episodes"])
    mode = "smoke" if smoke else "full"
    candidate_spec = dict(config.get("candidates", {}).get(candidate_name, {}))
    upstream_checkpoint_root = str(candidate_spec.get("checkpoint_root", ""))
    for seed in seeds:
        seed_dir_name = f"seed_{seed}_smoke" if smoke else f"seed_{seed}"
        run_dir = ensure_dir(output_root / candidate_name / seed_dir_name)
        summary_path = run_dir / metrics_filename
        metadata_path = run_dir / "run_metadata.json"
        expected_metadata = build_run_cache_metadata(
            config=config,
            candidate_name=candidate_name,
            metrics_filename=metrics_filename,
            mode=mode,
            num_eval_episodes=eval_episodes,
            step_name=str(config.get("experiment_name", "stage2_transition_eval_only")),
            upstream_checkpoint_root=upstream_checkpoint_root,
        )
        if summary_path.exists() and should_reuse_run_cache(metadata_path, expected_metadata):
            rows.extend(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        trainer = Stage2TransitionTrainer(
            config=config,
            candidate_name=candidate_name,
            seed=seed,
            run_dir=run_dir,
            smoke=smoke,
        )
        eval_rows = trainer.run_eval_suite(num_eval_episodes=eval_episodes, run_mode=mode)
        summary_path.write_text(json.dumps(eval_rows, ensure_ascii=True, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(expected_metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        rows.extend(eval_rows)
    candidate_root = ensure_dir(output_root / candidate_name)
    (candidate_root / f"eval_rows_{mode}.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    if not smoke:
        (candidate_root / "eval_rows.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    return rows


def compute_degradation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"]
        if normal.empty:
            continue
        normal_row = normal.mean(numeric_only=True)
        for probe_name in sorted([name for name in sub["corruption_type"].unique() if name != "normal_eval"]):
            probe = sub[sub["corruption_type"] == probe_name].mean(numeric_only=True)
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "corruption_type": probe_name,
                    "normal_to_mild_return_drop": float(normal_row["eval_return"] - probe["eval_return"]),
                    "normal_to_mild_success_drop": float(normal_row["eval_success_rate"] - probe["eval_success_rate"]),
                    "normal_to_mild_progress_drop": float(
                        normal_row["eval_avg_progress_to_goal"] - probe["eval_avg_progress_to_goal"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_consistency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        mean_gap = float(sub["same_state_action_diff_clean_vs_corrupt"].mean())
        rows.append(
            {
                "candidate_name": candidate_name,
                "consistency_loss": float(sub["consistency_loss"].mean()),
                "consistency_gap_mean": float(sub["consistency_gap_mean"].mean()),
                "same_state_action_diff_clean_vs_corrupt": mean_gap,
                "consistency_gain_score": float(max(0.0, 0.15 - mean_gap)),
            }
        )
    return pd.DataFrame(rows)


def compute_tiebreak_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        mild_vision = sub[sub["corruption_type"] == "mild_vision"].mean(numeric_only=True)
        mild_ego = sub[sub["corruption_type"] == "mild_ego"].mean(numeric_only=True)
        mild_distance = sub[sub["corruption_type"] == "mild_distance"].mean(numeric_only=True)
        mild_route = sub[sub["corruption_type"] == "mild_route"].mean(numeric_only=True)
        clean_rel_max = max(float(normal.get(f"reliability_mean_{name}", 0.0)) for name in MODALITY_NAMES)
        rows.append(
            {
                "candidate_name": candidate_name,
                "vision_reliability_clean_minus_mild": float(
                    normal["reliability_mean_vision"] - mild_vision["reliability_mean_vision"]
                ),
                "ego_reliability_clean_minus_mild": float(
                    normal["reliability_mean_ego"] - mild_ego["reliability_mean_ego"]
                ),
                "vision_logit_clean_minus_mild": float(
                    normal["logit_mean_vision_clean"] - mild_vision["logit_mean_vision_mild"]
                ),
                "ego_logit_clean_minus_mild": float(normal["logit_mean_ego_clean"] - mild_ego["logit_mean_ego_mild"]),
                "saturation_warning_flag": float(clean_rel_max >= 0.98),
                "route_distance_side_effect_flag": float(
                    (normal["reliability_mean_route"] - mild_route["reliability_mean_route"] < -1e-4)
                    or (normal["reliability_mean_distance"] - mild_distance["reliability_mean_distance"] < -1e-4)
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_stage2_results(rows: list[dict[str, Any]], summary_dir: Path, stem: str) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["candidate_name", "corruption_type", "seed"]).reset_index(drop=True)
    df.to_csv(summary_dir / f"{stem}_metrics.csv", index=False)
    df.to_json(summary_dir / f"{stem}_metrics.json", orient="records", indent=2)
    return df


def evaluate_pass_fail(summary: dict[str, Any]) -> str:
    if summary["normal_success_rate"] < 0.7:
        return "FAIL"
    if summary["normal_to_mild_success_drop"] > 0.12 and summary["degradation_score"] < 0.0:
        return "FAIL"
    if (
        summary["normal_success_rate"] >= 0.75
        and summary["normal_to_mild_success_drop"] <= 0.10
        and summary["consistency_gain_score"] > 0.0
        and (summary["vision_reliability_clean_minus_mild"] > 0.0 or summary["ego_reliability_clean_minus_mild"] > 0.0)
    ):
        return "PASS"
    return "WARN"


def build_candidate_score_summary(
    df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    tiebreak_weights: dict[str, float],
) -> pd.DataFrame:
    final_rows = []
    for candidate_name, sub in df.groupby("candidate_name"):
        normal = sub[sub["corruption_type"] == "normal_eval"].mean(numeric_only=True)
        deg = degradation_df[degradation_df["candidate_name"] == candidate_name]
        tie = tiebreak_df[tiebreak_df["candidate_name"] == candidate_name].iloc[0]
        cons = consistency_df[consistency_df["candidate_name"] == candidate_name].iloc[0]
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
        }
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
            + float(tiebreak_weights["saturation_penalty"]) * summary["saturation_warning_flag"]
            + float(tiebreak_weights["side_effect_penalty"]) * summary["route_distance_side_effect_flag"]
        )
        summary["pass_fail"] = evaluate_pass_fail(summary)
        final_rows.append(summary)
    score_df = pd.DataFrame(final_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "normal_success_rate",
            "degradation_score",
            "vision_reliability_clean_minus_mild",
            "ego_reliability_clean_minus_mild",
            "tie_break_score",
        ],
        ascending=[True, False, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    return score_df


def choose_recommendations(
    score_df: pd.DataFrame,
    standard_candidate: str,
    backup_candidate: str,
) -> tuple[str, str]:
    standard_row = score_df[score_df["candidate_name"] == standard_candidate]
    backup_row = score_df[score_df["candidate_name"] == backup_candidate]
    if not standard_row.empty and standard_row.iloc[0]["pass_fail"] == "PASS":
        recommended = standard_candidate
        if not backup_row.empty:
            secondary = backup_candidate
        else:
            secondary = standard_candidate
        return recommended, secondary
    recommended = score_df.iloc[0]["candidate_name"]
    secondary = score_df.iloc[1]["candidate_name"] if len(score_df) > 1 else recommended
    return recommended, secondary


def generate_stage2_pilot_report(
    score_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    tiebreak_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
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
