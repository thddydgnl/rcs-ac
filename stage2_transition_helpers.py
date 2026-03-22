"""Shared helpers for Stage 2 confirm/full transition runs."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from ppo_baselines.utils.logger import ExperimentLogger
from rcs_ac_v1.model import MODALITY_NAMES
from rcs_ac_v1.recurrent_buffer import RecurrentRolloutBuffer
from rcs_toy_nav_env import RCSToyNavEnv
from run_stage1_targeted_monotonicity_calibration import deep_apply_overrides
from stage1_warmup.config import ExperimentConfig, load_experiment_config
from stage1_warmup.corruption import apply_shared_corruption, build_reliability_targets
from stage1_warmup.model import Stage1WarmupActorCritic


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


def latest_checkpoint(seed_dir: Path) -> Path:
    checkpoint_final = seed_dir / "checkpoint_final.pt"
    if checkpoint_final.exists():
        return checkpoint_final
    checkpoints = sorted(seed_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No checkpoint found in {seed_dir}")


def build_model_from_candidate(
    base_stage1_config_path: str,
    checkpoint_path: Path,
    candidate_snapshot_path: Path | None,
    device: torch.device,
    seed: int,
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
        action_dim=2,
        action_log_std_init=cfg.model.action_log_std_init,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    current.update(compatible)
    model.load_state_dict(current)
    return cfg, model


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
) -> dict[str, Any]:
    candidate_spec = config["candidates"][candidate_name]
    checkpoint_root = Path(candidate_spec["checkpoint_root"])
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


class Stage2TransitionTrainer:
    def __init__(self, config: dict[str, Any], candidate_name: str, seed: int, run_dir: Path):
        self.raw_config = config
        self.candidate_name = candidate_name
        self.seed = seed
        self.run_dir = run_dir
        self.device = torch.device(config.get("device", "cpu"))
        self.logger = ExperimentLogger(str(run_dir))

        handoff = load_stage2_handoff_checkpoint(config, candidate_name, seed)
        self.base_cfg, self.model = build_model_from_candidate(
            base_stage1_config_path=config["base_stage1_config_path"],
            checkpoint_path=Path(handoff["source_checkpoint_path"]),
            candidate_snapshot_path=Path(handoff["source_snapshot_path"]) if handoff["source_snapshot_path"] else None,
            device=self.device,
            seed=seed,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config["train"]["learning_rate"]))
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
        self.logger.save_config({"candidate_name": candidate_name, "seed": seed, "config": config, "handoff": handoff})
        self.nan_detected = False

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
        choice = float(rng.random())
        clean_prob = float(spec["clean_probability"])
        mild_single_prob = float(spec["mild_single_probability"])
        if choice < clean_prob:
            severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
            return {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}, severity, severity.copy(), "clean"
        if choice < clean_prob + mild_single_prob:
            single_specs = spec["single_modality"]
            names = [item["name"] for item in single_specs]
            weights = np.asarray([float(item.get("weight", 1.0)) for item in single_specs], dtype=np.float64)
            weights = weights / weights.sum()
            selected = str(rng.choice(names, p=weights))
            selected_cfg = next(item for item in single_specs if item["name"] == selected)
            corrupted, severity = _apply_single_corruption(
                obs=obs,
                corruption_type=selected,
                corruption_strength=float(selected_cfg["strength"]),
                rng=rng,
                probe_name=f"train_mild_{selected}",
            )
            return corrupted, severity, (severity > 0.0).astype(np.float32), f"mild_{selected}"
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
                output = self.model.step(
                    obs_dict=obs_to_tensor(policy_obs, self.device),
                    hidden_state=hidden,
                    episode_start=episode_start,
                    deterministic=False,
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
            output = self.model.step(
                obs_dict=obs_to_tensor(clean_obs, self.device),
                hidden_state=hidden,
                episode_start=episode_start,
                deterministic=False,
            )
        self.buffer.compute_returns_and_advantages(
            last_value=float(output.value.squeeze(0).cpu().item()),
            last_done=bool(episode_start.item()),
            gamma=self.base_cfg.ppo.gamma,
            gae_lambda=self.base_cfg.ppo.gae_lambda,
        )
        total = max(sum(corruption_counter.values()), 1)
        return {
            "train/episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "train/avg_progress_to_goal": float(np.mean(progress_values)) if progress_values else 0.0,
            "train/corruption_clean_ratio": float(corruption_counter.get("clean", 0) / total),
            "train/corruption_mild_ratio": float(1.0 - corruption_counter.get("clean", 0) / total),
            "train/corruption_mild_vision_ratio": float(corruption_counter.get("mild_vision", 0) / total),
            "train/corruption_mild_ego_ratio": float(corruption_counter.get("mild_ego", 0) / total),
            "train/corruption_mild_distance_ratio": float(corruption_counter.get("mild_distance", 0) / total),
            "train/corruption_mild_route_ratio": float(corruption_counter.get("mild_route", 0) / total),
            "train/corruption_mixed_ratio": float(sum(corruption_counter.get(name, 0) for name in mixed_names) / total),
            "train/corruption_single_ratio": float(sum(corruption_counter.get(name, 0) for name in single_names) / total),
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
            "loss/grad_norm": 0.0,
        }
        losses_cfg = self.raw_config["losses"]
        vision_retention_coef = float(losses_cfg.get("vision_logit_retention_loss_coef", 0.0))
        ego_retention_coef = float(losses_cfg.get("ego_logit_retention_loss_coef", 0.0))
        vision_retention_margin = float(losses_cfg.get("vision_logit_retention_margin", 0.0))
        ego_retention_margin = float(losses_cfg.get("ego_logit_retention_margin", 0.0))
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
                total_loss = (
                    float(losses_cfg["actor_loss_coef"]) * actor_loss
                    + float(losses_cfg["value_loss_coef"]) * critic_loss
                    - float(losses_cfg["entropy_coef"]) * entropy
                    + float(losses_cfg["reliability_loss_coef"]) * reliability_loss
                    + float(losses_cfg["consistency_loss_coef"]) * consistency_loss
                    + vision_retention_coef * vision_retention_loss
                    + ego_retention_coef * ego_retention_loss
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
            rng = np.random.default_rng(self.seed + 9000 + abs(hash(probe_name)) % 997)
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
                        policy_obs, _severity = _apply_single_corruption(
                            obs=obs,
                            corruption_type=str(probe["modality"]),
                            corruption_strength=float(probe["strength"]),
                            rng=rng,
                            probe_name=probe_name,
                        )
                    else:
                        policy_obs, _severity = _apply_mixed_corruption(
                            obs=obs,
                            corruption_types=list(probe["modalities"]),
                            corruption_strengths=list(probe["strengths"]),
                            rng=rng,
                            probe_name=probe_name,
                        )
                    clean_output = self.model.step(
                        obs_dict=obs_to_tensor(clean_obs, self.device),
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    corrupt_output = self.model.step(
                        obs_dict=obs_to_tensor(policy_obs, self.device),
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    action = corrupt_output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    attention = corrupt_output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
                    reliability = corrupt_output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy()
                    logits = corrupt_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    clean_logits = clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(clean_logits[2]), float(logits[2])))
                    action_diffs.append(
                        float(torch.mean(torch.abs(clean_output.action - corrupt_output.action)).cpu().item())
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
            vision_logit_arr = np.asarray(vision_logit_values, dtype=np.float32)
            ego_logit_arr = np.asarray(ego_logit_values, dtype=np.float32)
            row = {
                "candidate_name": self.candidate_name,
                "seed": self.seed,
                "env_mode": "normal" if probe_name == "normal_eval" else "mild",
                "corruption_type": probe_name,
                "corruption_strength": float(probe.get("strength", 0.0)),
                "train_step": int(self.raw_config["train"]["total_steps"]),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "consistency_loss": float(np.mean(np.square(action_diffs))),
                "consistency_gap_mean": float(np.mean(action_diffs)),
                "same_state_action_diff_clean_vs_corrupt": float(np.mean(action_diffs)),
                "consistency_gain_score": 0.0,
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(
                    att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)
                ),
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
            }
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
                eval_rows = self.run_eval_suite(num_eval_episodes=int(self.raw_config["num_eval_episodes"]))
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
    for seed in seeds:
        run_dir = ensure_dir(output_root / candidate_name / f"seed_{seed}")
        summary_path = run_dir / metrics_filename
        if summary_path.exists():
            rows.extend(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        trainer = Stage2TransitionTrainer(config=config, candidate_name=candidate_name, seed=seed, run_dir=run_dir)
        trainer.train()
        eval_rows = trainer.run_eval_suite(num_eval_episodes=eval_episodes)
        summary_path.write_text(json.dumps(eval_rows, ensure_ascii=True, indent=2), encoding="utf-8")
        rows.extend(eval_rows)
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
