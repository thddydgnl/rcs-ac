"""Stage 1 warm-up trainer: v1 backbone + weak reliability loss."""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import torch
from torch import nn

from ppo_baselines.utils.logger import ExperimentLogger
from rcs_ac_v1.model import MODALITY_NAMES
from rcs_ac_v1.recurrent_buffer import RecurrentRolloutBuffer
from rcs_toy_nav_env import RCSToyNavEnv
from stage1_warmup.config import ExperimentConfig
from stage1_warmup.corruption import apply_targeted_corruption, build_reliability_targets
from stage1_warmup.evaluate import aux_to_reliability_targets, evaluate_stage1_policy
from stage1_warmup.model import Stage1WarmupActorCritic


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(config: ExperimentConfig) -> RCSToyNavEnv:
    env = RCSToyNavEnv()
    env.set_corruption_mode(config.env.corruption_mode)
    return env


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


class Stage1WarmupTrainer:
    """Weak reliability warm-up on top of standard recurrent v1."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)

        self.train_env = make_env(config)
        self.eval_env = make_env(config)
        self.obs_shapes = {
            key: self.train_env.observation_space[key].shape
            for key in [*MODALITY_NAMES, "aux"]
        }

        self.model = Stage1WarmupActorCritic(
            encoder_hidden_dims=config.model.encoder_hidden_dims,
            encoder_output_dim=config.model.encoder_output_dim,
            attention_hidden_dim=config.model.attention_hidden_dim,
            recurrent_hidden_dim=config.model.recurrent_hidden_dim,
            attention_temperature=config.model.attention_temperature,
            use_feature_layer_norm=config.model.use_feature_layer_norm,
            actor_hidden_dims=config.model.actor_hidden_dims,
            critic_hidden_dims=config.model.critic_hidden_dims,
            reliability_hidden_dims=config.model.reliability_hidden_dims,
            reliability_input_mode=config.model.reliability_input_mode,
            action_dim=int(self.train_env.action_space.shape[0]),
            action_log_std_init=config.model.action_log_std_init,
        ).to(self.device)
        if config.warmup.init_checkpoint_path:
            self.load_checkpoint(config.warmup.init_checkpoint_path)
        if config.warmup.freeze_policy_backbone or config.warmup.train_reliability_head_only:
            self.model.freeze_policy_backbone()
            self.model.unfreeze_reliability_head()
        trainable_params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=config.ppo.learning_rate)
        self.buffer = RecurrentRolloutBuffer(
            rollout_steps=config.ppo.rollout_steps,
            sequence_length=config.ppo.sequence_length,
            obs_shapes=self.obs_shapes,
            action_dim=int(self.train_env.action_space.shape[0]),
            hidden_dim=config.model.recurrent_hidden_dim,
            obs_keys=[*MODALITY_NAMES, "aux"],
            extra_specs={
                "reliability_targets": (len(MODALITY_NAMES),),
                "severity": (len(MODALITY_NAMES),),
                "target_mask": (len(MODALITY_NAMES),),
                "corrupted_flag": (1,),
            },
        )
        self.logger = ExperimentLogger(config.logging.output_dir)
        self.logger.save_config(asdict(config))
        self.nan_detected = False
        self.reliability_nan_detected = False
        self.hidden_validation = {
            "hidden_zero_init_ok": True,
            "hidden_reset_ok": True,
            "hidden_leakage_detected": False,
            "hidden_shape_mismatch_detected": False,
            "hidden_nan_detected": False,
            "hidden_reset_count": 0,
        }
        self._corruption_schedule = self._build_corruption_schedule()
        self._corruption_schedule_idx = 0

    @staticmethod
    def _metrics_have_nan(metrics: dict[str, float]) -> bool:
        for value in metrics.values():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                return True
        return False

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        current = self.model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        current.update(compatible)
        self.model.load_state_dict(current)

    def _build_corruption_schedule(self) -> list[tuple[str, float]]:
        weak_prob = float(self.config.warmup.weak_corruption_probability)
        mild_prob = float(self.config.warmup.mild_corruption_probability)
        total_prob = weak_prob + mild_prob
        if total_prob <= 0.0:
            return [("clean", 0.0)]

        if self.config.warmup.focus_vision_ego_pairing:
            corruption_entries = [
                ("vision", float(self.config.warmup.weak_corruption_scale)),
                ("vision", float(self.config.warmup.mild_corruption_scale)),
                ("ego", float(self.config.warmup.weak_corruption_scale)),
                ("ego", float(self.config.warmup.mild_corruption_scale)),
                ("vision", float(self.config.warmup.weak_corruption_scale)),
                ("vision", float(self.config.warmup.mild_corruption_scale)),
                ("ego", float(self.config.warmup.weak_corruption_scale)),
                ("ego", float(self.config.warmup.mild_corruption_scale)),
                ("distance", float(self.config.warmup.weak_corruption_scale)),
                ("distance", float(self.config.warmup.mild_corruption_scale)),
                ("route", float(self.config.warmup.weak_corruption_scale)),
                ("route", float(self.config.warmup.mild_corruption_scale)),
            ]
        else:
            corruption_entries = [
                ("vision", float(self.config.warmup.weak_corruption_scale)),
                ("vision", float(self.config.warmup.mild_corruption_scale)),
                ("ego", float(self.config.warmup.weak_corruption_scale)),
                ("ego", float(self.config.warmup.mild_corruption_scale)),
                ("distance", float(self.config.warmup.weak_corruption_scale)),
                ("distance", float(self.config.warmup.mild_corruption_scale)),
                ("route", float(self.config.warmup.weak_corruption_scale)),
                ("route", float(self.config.warmup.mild_corruption_scale)),
            ]
        clean_count = max(1, int(round(len(corruption_entries) * max(1.0 - total_prob, 0.0) / max(total_prob, 1e-6))))
        schedule: list[tuple[str, float]] = []
        clean_slots_remaining = clean_count
        for entry_idx, entry in enumerate(corruption_entries):
            schedule.append(entry)
            remaining_entries = len(corruption_entries) - entry_idx - 1
            slots_here = clean_slots_remaining // max(remaining_entries + 1, 1)
            for _ in range(slots_here):
                schedule.append(("clean", 0.0))
            clean_slots_remaining -= slots_here
        while clean_slots_remaining > 0:
            schedule.append(("clean", 0.0))
            clean_slots_remaining -= 1
        return schedule

    def ensure_weak_mild_pair_batching(self) -> tuple[str, float]:
        if not self._corruption_schedule:
            return "clean", 0.0
        corruption_type, corruption_strength = self._corruption_schedule[self._corruption_schedule_idx % len(self._corruption_schedule)]
        self._corruption_schedule_idx += 1
        return corruption_type, corruption_strength

    def _sample_training_corruption(
        self,
        obs: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, str]:
        weak_prob = float(self.config.warmup.weak_corruption_probability)
        mild_prob = float(self.config.warmup.mild_corruption_probability)
        if weak_prob <= 0.0 and mild_prob <= 0.0:
            clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
            severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
            targets = build_reliability_targets(
                aux_obs=obs["aux"],
                severity=severity,
                use_direct_corruption_targets=self.config.warmup.use_direct_corruption_targets,
                direct_target_drop_scale=self.config.warmup.direct_target_drop_scale,
            )
            return clean_obs, targets, severity, severity.copy(), "clean"

        corruption_type, corruption_strength = self.ensure_weak_mild_pair_batching()
        if corruption_type == "clean" or corruption_strength <= 0.0:
            clean_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
            severity = np.zeros(len(MODALITY_NAMES), dtype=np.float32)
            targets = build_reliability_targets(
                aux_obs=obs["aux"],
                severity=severity,
                use_direct_corruption_targets=self.config.warmup.use_direct_corruption_targets,
                direct_target_drop_scale=self.config.warmup.direct_target_drop_scale,
            )
            return clean_obs, targets, severity, severity.copy(), "clean"
        corrupted_obs, severity = apply_targeted_corruption(
            obs=obs,
            corruption_type=str(corruption_type),
            corruption_strength=corruption_strength,
            rng=rng,
        )
        targets = build_reliability_targets(
            aux_obs=obs["aux"],
            severity=severity,
            use_direct_corruption_targets=self.config.warmup.use_direct_corruption_targets,
            direct_target_drop_scale=self.config.warmup.direct_target_drop_scale,
        )
        target_mask = (severity > 0.0).astype(np.float32)
        return corrupted_obs, targets, severity, target_mask, str(corruption_type)

    def collect_rollout(self, global_step: int) -> dict[str, float]:
        obs, info = self.train_env.reset(seed=self.config.seed + global_step)
        del info
        self.buffer.reset()
        hidden = self.model.initial_hidden(batch_size=1, device=self.device)
        if not torch.allclose(hidden, torch.zeros_like(hidden)):
            self.hidden_validation["hidden_zero_init_ok"] = False
        episode_start_flag = torch.ones(1, device=self.device)

        episode_returns = []
        episode_lengths = []
        success_flags = []
        collision_flags = []
        timeout_flags = []
        progress_values = []
        attention_values = []
        reliability_values = []
        hidden_norm_values = []
        reliability_target_values = []
        corruption_counter = {name: 0 for name in ["clean", *MODALITY_NAMES]}
        corruption_severity_counter = {"clean": 0, "weak": 0, "mild": 0}
        rng = np.random.default_rng(self.config.seed + global_step + 999)

        current_return = 0.0
        current_length = 0
        current_progress = []

        for _ in range(self.config.ppo.rollout_steps):
            if bool(episode_start_flag.item()) and not torch.allclose(hidden, torch.zeros_like(hidden), atol=1e-6):
                self.hidden_validation["hidden_leakage_detected"] = True
            if not torch.all(torch.isfinite(hidden)):
                self.hidden_validation["hidden_nan_detected"] = True

            policy_obs, reliability_targets, severity, target_mask, corruption_type = self._sample_training_corruption(obs, rng)
            corruption_counter[corruption_type] = corruption_counter.get(corruption_type, 0) + 1
            max_severity_np = float(np.max(severity))
            if corruption_type == "clean" or max_severity_np <= 0.0:
                corruption_severity_counter["clean"] += 1
            elif max_severity_np >= float(self.config.warmup.mild_corruption_scale) - 1e-6:
                corruption_severity_counter["mild"] += 1
            else:
                corruption_severity_counter["weak"] += 1
            obs_tensor = obs_to_tensor(policy_obs, self.device)
            hidden_before = hidden.clone()
            try:
                with torch.no_grad():
                    output = self.model.step(
                        obs_dict=obs_tensor,
                        hidden_state=hidden,
                        episode_start=episode_start_flag,
                        deterministic=False,
                    )
            except ValueError:
                self.hidden_validation["hidden_shape_mismatch_detected"] = True
                raise

            action = output.action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            self.buffer.add(
                obs={
                    **{key: policy_obs[key] for key in MODALITY_NAMES},
                    "aux": obs["aux"],
                },
                action=action,
                reward=float(reward),
                done=done,
                episode_start=bool(episode_start_flag.item()),
                value=float(output.value.squeeze(0).cpu().item()),
                log_prob=float(output.log_prob.squeeze(0).cpu().item()),
                hidden_state=hidden_before.squeeze(0).cpu().numpy(),
                extras={
                    "reliability_targets": reliability_targets.astype(np.float32),
                    "severity": severity.astype(np.float32),
                    "target_mask": target_mask.astype(np.float32),
                    "corrupted_flag": np.array([float(corruption_type != "clean")], dtype=np.float32),
                },
            )

            hidden = output.hidden_state
            reliability_scores = output.aux_outputs["reliability_scores"].squeeze(0).cpu().numpy()
            attention_values.append(output.aux_outputs["attention_weights"].squeeze(0).cpu().numpy())
            reliability_values.append(reliability_scores)
            reliability_target_values.append(reliability_targets)
            self.reliability_nan_detected = self.reliability_nan_detected or not np.all(np.isfinite(reliability_scores))
            hidden_norm_values.append(float(output.aux_outputs["hidden_norm"].mean().cpu().item()))

            current_return += float(reward)
            current_length += 1
            current_progress.append(float(info.get("progress_delta", 0.0)))
            obs = next_obs
            episode_start_flag = torch.tensor([float(done)], device=self.device)
            hidden = self.model.reset_hidden(hidden, episode_start_flag)
            if done:
                self.hidden_validation["hidden_reset_count"] += 1
                if not torch.allclose(hidden, torch.zeros_like(hidden), atol=1e-6):
                    self.hidden_validation["hidden_reset_ok"] = False

            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                success_flags.append(float(info["goal_reached"]))
                collision_flags.append(float(info["collision"]))
                timeout_flags.append(float(info["timeout"]))
                progress_values.append(float(np.mean(current_progress)) if current_progress else 0.0)
                current_return = 0.0
                current_length = 0
                current_progress = []
                obs, info = self.train_env.reset()
                del info

        with torch.no_grad():
            next_policy_obs, _next_targets, _next_severity, _next_target_mask, _next_type = self._sample_training_corruption(obs, rng)
            obs_tensor = obs_to_tensor(next_policy_obs, self.device)
            last_output = self.model.step(
                obs_dict=obs_tensor,
                hidden_state=hidden,
                episode_start=episode_start_flag,
                deterministic=False,
            )
        self.buffer.compute_returns_and_advantages(
            last_value=float(last_output.value.squeeze(0).cpu().item()),
            last_done=bool(episode_start_flag.item()),
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
        )

        metrics = {
            "train/episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "train/episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "train/success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            "train/collision_rate": float(np.mean(collision_flags)) if collision_flags else 0.0,
            "train/timeout_rate": float(np.mean(timeout_flags)) if timeout_flags else 0.0,
            "train/avg_progress_to_goal": float(np.mean(progress_values)) if progress_values else 0.0,
            "train/hidden_norm_mean": float(np.mean(hidden_norm_values)) if hidden_norm_values else 0.0,
            "train/hidden_norm_std": float(np.std(hidden_norm_values)) if hidden_norm_values else 0.0,
            "train/hidden_reset_count": float(self.hidden_validation["hidden_reset_count"]),
            "train/sequence_length": float(self.config.ppo.sequence_length),
            "train/masked_timestep_ratio": float(self.buffer.masked_ratio()),
            "train/hidden_zero_init_ok": float(self.hidden_validation["hidden_zero_init_ok"]),
            "train/hidden_reset_ok": float(self.hidden_validation["hidden_reset_ok"]),
            "train/hidden_leakage_detected": float(self.hidden_validation["hidden_leakage_detected"]),
            "train/hidden_shape_mismatch_detected": float(self.hidden_validation["hidden_shape_mismatch_detected"]),
            "train/hidden_nan_detected": float(self.hidden_validation["hidden_nan_detected"]),
            "train/reliability_nan_detected": float(self.reliability_nan_detected),
            "train/weak_corruption_flag": float(self.config.warmup.weak_corruption_probability > 0.0),
            "train/quality_label_if_available": 1.0,
            "train/corruption_clean_ratio": float(corruption_counter["clean"] / max(sum(corruption_counter.values()), 1)),
            "train/corruption_ratio_weak": float(corruption_severity_counter["weak"] / max(sum(corruption_severity_counter.values()), 1)),
            "train/corruption_ratio_mild": float(corruption_severity_counter["mild"] / max(sum(corruption_severity_counter.values()), 1)),
        }
        for name in MODALITY_NAMES:
            metrics[f"train/corruption_ratio_{name}"] = float(corruption_counter[name] / max(sum(corruption_counter.values()), 1))
        if attention_values:
            stacked = np.stack(attention_values, axis=0)
            mean_weights = stacked.mean(axis=0)
            std_weights = stacked.std(axis=0)
            for idx, name in enumerate(MODALITY_NAMES):
                metrics[f"train_attention/mean_{name}"] = float(mean_weights[idx])
                metrics[f"train_attention/std_{name}"] = float(std_weights[idx])
            route_idx = MODALITY_NAMES.index("route")
            other_mean = np.mean(np.delete(mean_weights, route_idx))
            metrics["train_attention/route_dominance_ratio"] = float(mean_weights[route_idx] / max(other_mean, 1e-6))
        if reliability_values:
            stacked = np.stack(reliability_values, axis=0)
            std_scores = stacked.std(axis=0)
            for idx, name in enumerate(MODALITY_NAMES):
                metrics[f"train_reliability/mean_{name}"] = float(stacked[:, idx].mean())
                metrics[f"train_reliability/std_{name}"] = float(std_scores[idx])
        if attention_values and reliability_values:
            att = np.stack(attention_values, axis=0).reshape(-1)
            rel = np.stack(reliability_values, axis=0).reshape(-1)
            if np.std(att) > 1e-8 and np.std(rel) > 1e-8:
                metrics["train_attention_reliability/correlation"] = float(np.corrcoef(att, rel)[0, 1])
            else:
                metrics["train_attention_reliability/correlation"] = 0.0
        if reliability_values and reliability_target_values:
            pred = np.stack(reliability_values, axis=0)
            tgt = np.stack(reliability_target_values, axis=0)
            metrics["train/reliability_mse"] = float(np.mean((pred - tgt) ** 2))
        return metrics

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/grad_norm": 0.0,
            "loss/hidden_norm_mean": 0.0,
            "loss/hidden_norm_std": 0.0,
            "loss/reliability": 0.0,
            "loss/reliability_margin": 0.0,
            "loss/reliability_ranking": 0.0,
            "loss/reliability_monotonicity": 0.0,
            "loss/reliability_directional_monotonicity": 0.0,
            "loss/reliability_vision_positive": 0.0,
            "loss/reliability_ego_positive": 0.0,
            "loss/reliability_vision_logit_positive": 0.0,
            "loss/reliability_ego_logit_positive": 0.0,
        }
        num_updates = 0

        for _ in range(self.config.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=self.config.ppo.minibatch_num_sequences,
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                valid_mask = batch.valid_mask
                log_probs = outputs["log_probs"]
                values = outputs["values"]
                entropies = outputs["entropies"]
                hidden_norms = outputs["hidden_norms"]
                reliability_scores = outputs["reliability_scores"]
                reliability_logits = outputs["reliability_logits"]
                reliability_targets = batch.extras["reliability_targets"]
                severity = batch.extras["severity"]
                target_mask = batch.extras["target_mask"]
                corrupted_flag = batch.extras["corrupted_flag"].squeeze(-1)

                ratio = torch.exp(log_probs - batch.old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_eps,
                    1.0 + self.config.ppo.clip_eps,
                )
                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                expanded_mask = valid_mask.unsqueeze(-1)

                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((values - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (entropies * valid_mask).sum() / valid_count
                reliability_loss = (((reliability_scores - reliability_targets) ** 2) * expanded_mask).sum() / (
                    expanded_mask.sum().clamp(min=1.0)
                )
                corrupted_valid_mask = valid_mask * corrupted_flag
                corrupted_count = corrupted_valid_mask.sum().clamp(min=1.0)

                target_scores = (reliability_scores * target_mask).sum(dim=-1)
                non_target_count = (1.0 - target_mask).sum(dim=-1).clamp(min=1.0)
                non_target_mean = (reliability_scores * (1.0 - target_mask)).sum(dim=-1) / non_target_count
                max_severity = severity.max(dim=-1).values
                target_idx = torch.argmax(target_mask, dim=-1)
                target_weight = torch.ones_like(target_scores)
                margin_multiplier = torch.ones_like(target_scores)
                vision_idx = MODALITY_NAMES.index("vision")
                ego_idx = MODALITY_NAMES.index("ego")
                target_weight = torch.where(
                    target_idx == vision_idx,
                    target_weight * float(self.config.warmup.vision_weight_multiplier),
                    target_weight,
                )
                target_weight = torch.where(
                    target_idx == ego_idx,
                    target_weight * float(self.config.warmup.ego_weight_multiplier),
                    target_weight,
                )
                target_weight = torch.where(
                    (target_idx == vision_idx) | (target_idx == ego_idx),
                    target_weight * float(self.config.warmup.vision_ego_weight_multiplier),
                    target_weight,
                )
                margin_multiplier = torch.where(
                    target_idx == vision_idx,
                    margin_multiplier * float(self.config.warmup.vision_margin_multiplier),
                    margin_multiplier,
                )
                margin_multiplier = torch.where(
                    target_idx == ego_idx,
                    margin_multiplier * float(self.config.warmup.ego_margin_multiplier),
                    margin_multiplier,
                )
                margin_multiplier = torch.where(
                    (target_idx == vision_idx) | (target_idx == ego_idx),
                    margin_multiplier * float(self.config.warmup.vision_ego_margin_multiplier),
                    margin_multiplier,
                )
                target_upper_bound = torch.clamp(
                    reliability_targets.max(dim=-1).values
                    - (
                        self.config.warmup.target_margin_value
                        + self.config.warmup.monotonicity_margin_clean_weak
                    ) * max_severity * margin_multiplier,
                    min=0.0,
                    max=1.0,
                )
                margin_gap = target_scores - target_upper_bound
                target_margin_loss = (
                    torch.relu(margin_gap) * corrupted_valid_mask * target_weight
                ).sum() / (corrupted_valid_mask * target_weight).sum().clamp(min=1.0)

                ranking_margin = self.config.warmup.target_ranking_margin * max_severity * margin_multiplier
                ranking_gap = ranking_margin - (non_target_mean - target_scores)
                target_ranking_loss = (
                    torch.relu(ranking_gap) * corrupted_valid_mask * target_weight
                ).sum() / (corrupted_valid_mask * target_weight).sum().clamp(min=1.0)

                monotonicity_losses = []
                directional_monotonicity_losses = []
                flat_target_scores = target_scores.reshape(-1)
                flat_target_idx = target_idx.reshape(-1)
                flat_corrupted = corrupted_valid_mask.reshape(-1) > 0.0
                flat_severity = max_severity.reshape(-1)
                flat_weight = target_weight.reshape(-1)
                for modality_idx, _modality_name in enumerate(MODALITY_NAMES):
                    modality_mask = flat_corrupted & (flat_target_idx == modality_idx)
                    if int(modality_mask.sum().item()) < 2:
                        continue
                    modality_scores = flat_target_scores[modality_mask]
                    modality_severity = flat_severity[modality_mask]
                    modality_weight = flat_weight[modality_mask]
                    higher_mask = modality_severity.unsqueeze(1) > modality_severity.unsqueeze(0) + 1e-6
                    if not bool(higher_mask.any().item()):
                        continue
                    severity_gap = modality_severity.unsqueeze(1) - modality_severity.unsqueeze(0)
                    pair_margin = (
                        self.config.warmup.monotonicity_margin
                        + self.config.warmup.monotonicity_margin_weak_mild
                    ) * severity_gap
                    score_gap = modality_scores.unsqueeze(1) - modality_scores.unsqueeze(0)
                    pair_loss = torch.relu(pair_margin + score_gap)
                    directional_pair_loss = torch.relu(score_gap)
                    pair_weight = modality_weight.unsqueeze(1).expand_as(pair_loss)
                    weighted_loss = pair_loss * pair_weight * higher_mask.float()
                    weighted_directional_loss = directional_pair_loss * pair_weight * higher_mask.float()
                    denom = (pair_weight * higher_mask.float()).sum().clamp(min=1.0)
                    monotonicity_losses.append(weighted_loss.sum() / denom)
                    directional_monotonicity_losses.append(weighted_directional_loss.sum() / denom)
                if monotonicity_losses:
                    monotonicity_loss = torch.stack(monotonicity_losses).mean()
                else:
                    monotonicity_loss = torch.zeros((), device=self.device)
                if directional_monotonicity_losses:
                    directional_monotonicity_loss = torch.stack(directional_monotonicity_losses).mean()
                else:
                    directional_monotonicity_loss = torch.zeros((), device=self.device)

                weak_threshold = 0.5 * (
                    float(self.config.warmup.weak_corruption_scale) + float(self.config.warmup.mild_corruption_scale)
                )
                is_mild = max_severity >= weak_threshold
                target_target_scores = (reliability_targets * target_mask).sum(dim=-1)
                positive_margin = torch.where(
                    is_mild,
                    torch.full_like(target_scores, float(self.config.warmup.target_positive_margin_mild)),
                    torch.full_like(target_scores, float(self.config.warmup.target_positive_margin_weak)),
                )
                vision_positive_mask = corrupted_valid_mask * (target_idx == vision_idx).float()
                ego_positive_mask = corrupted_valid_mask * (target_idx == ego_idx).float()
                vision_positive_loss = (
                    torch.relu(target_scores - (target_target_scores - positive_margin)) * vision_positive_mask
                ).sum() / vision_positive_mask.sum().clamp(min=1.0)
                ego_positive_loss = (
                    torch.relu(target_scores - (target_target_scores - positive_margin)) * ego_positive_mask
                ).sum() / ego_positive_mask.sum().clamp(min=1.0)

                target_target_logits = torch.logit(
                    torch.clamp(target_target_scores, min=1e-4, max=1.0 - 1e-4),
                    eps=1e-4,
                )
                target_logits = (reliability_logits * target_mask).sum(dim=-1)
                logit_margin = torch.where(
                    is_mild,
                    torch.full_like(target_logits, float(self.config.warmup.logit_positive_margin_mild)),
                    torch.full_like(target_logits, float(self.config.warmup.logit_positive_margin_weak)),
                )
                vision_logit_positive_loss = (
                    torch.relu(target_logits - (target_target_logits - logit_margin)) * vision_positive_mask
                ).sum() / vision_positive_mask.sum().clamp(min=1.0)
                ego_logit_positive_loss = (
                    torch.relu(target_logits - (target_target_logits - logit_margin)) * ego_positive_mask
                ).sum() / ego_positive_mask.sum().clamp(min=1.0)

                loss = (
                    actor_loss
                    + self.config.ppo.value_loss_coef * critic_loss
                    - self.config.ppo.entropy_coef * entropy
                    + self.config.warmup.reliability_loss_coef * reliability_loss
                    + self.config.warmup.target_margin_loss_coef * target_margin_loss
                    + self.config.warmup.target_ranking_loss_coef * target_ranking_loss
                    + self.config.warmup.monotonicity_loss_coef * monotonicity_loss
                    + self.config.warmup.directional_monotonicity_loss_coef * directional_monotonicity_loss
                    + self.config.warmup.vision_target_positive_loss_coef * vision_positive_loss
                    + self.config.warmup.ego_target_positive_loss_coef * ego_positive_loss
                    + self.config.warmup.vision_logit_positive_loss_coef * vision_logit_positive_loss
                    + self.config.warmup.ego_logit_positive_loss_coef * ego_logit_positive_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/actor"] += float(actor_loss.item())
                metrics["loss/critic"] += float(critic_loss.item())
                metrics["loss/entropy"] += float(entropy.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["loss/reliability"] += float(reliability_loss.item())
                metrics["loss/reliability_margin"] += float(target_margin_loss.item())
                metrics["loss/reliability_ranking"] += float(target_ranking_loss.item())
                metrics["loss/reliability_monotonicity"] += float(monotonicity_loss.item())
                metrics["loss/reliability_directional_monotonicity"] += float(directional_monotonicity_loss.item())
                metrics["loss/reliability_vision_positive"] += float(vision_positive_loss.item())
                metrics["loss/reliability_ego_positive"] += float(ego_positive_loss.item())
                metrics["loss/reliability_vision_logit_positive"] += float(vision_logit_positive_loss.item())
                metrics["loss/reliability_ego_logit_positive"] += float(ego_logit_positive_loss.item())
                metrics["loss/hidden_norm_mean"] += float((hidden_norms * valid_mask).sum().item() / valid_count.item())
                centered = (hidden_norms - hidden_norms.mean()) ** 2
                metrics["loss/hidden_norm_std"] += float(torch.sqrt((centered * valid_mask).sum() / valid_count).item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    def train(self) -> None:
        num_updates = self.config.ppo.total_steps // self.config.ppo.rollout_steps
        start_time = time.time()
        for update_idx in range(1, num_updates + 1):
            rollout_metrics = self.collect_rollout(global_step=update_idx * self.config.ppo.rollout_steps)
            loss_metrics = self.update_model()
            self.nan_detected = (
                self.nan_detected
                or self._metrics_have_nan(rollout_metrics)
                or self._metrics_have_nan(loss_metrics)
                or self.reliability_nan_detected
            )

            train_payload = {
                "update": update_idx,
                "train_step": update_idx * self.config.ppo.rollout_steps,
                "wall_clock_time": float(time.time() - start_time),
                "finished_flag": 0,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
            }
            train_payload.update(rollout_metrics)
            train_payload.update(loss_metrics)
            self.logger.log_train(train_payload)
            self.logger.log(train_payload)

            if update_idx % self.config.logging.eval_interval_updates == 0:
                eval_metrics = evaluate_stage1_policy(
                    env=self.eval_env,
                    model=self.model,
                    device=self.device,
                    num_episodes=self.config.ppo.num_eval_episodes,
                )
                self.nan_detected = self.nan_detected or self._metrics_have_nan(eval_metrics)
                eval_payload = {
                    "update": update_idx,
                    "train_step": update_idx * self.config.ppo.rollout_steps,
                    "wall_clock_time": float(time.time() - start_time),
                    "finished_flag": 0,
                    "nan_detected": int(self.nan_detected),
                    "crashed_flag": 0,
                }
                eval_payload.update(eval_metrics)
                self.logger.log_eval(eval_payload)
                self.logger.log(eval_payload)

            if update_idx % self.config.logging.save_interval_updates == 0:
                self.save_checkpoint(update_idx)

        self.logger.save_status(
            {
                "finished_flag": 1,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
                "wall_clock_time": float(time.time() - start_time),
                "total_updates": num_updates,
                "hidden_validation": self.hidden_validation,
                "reliability_nan_detected": int(self.reliability_nan_detected),
            }
        )

    def save_checkpoint(self, update_idx: int) -> None:
        checkpoint_path = self.logger.output_dir / f"checkpoint_{update_idx:05d}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "update_idx": update_idx,
            },
            checkpoint_path,
        )
