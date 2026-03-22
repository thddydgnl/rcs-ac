"""Helpers for Version 3 counterfactual Shapley alignment runs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rcs_ac_v1.model import MODALITY_NAMES
from stage2_transition_helpers import (
    Stage2TransitionTrainer,
    aggregate_stage2_results,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    ensure_dir,
    load_yaml,
    save_yaml,
)

_NUM_MODALITIES = len(MODALITY_NAMES)
_ALL_SUBSET_MASKS = tuple(range(1 << _NUM_MODALITIES))
_FACTORIAL = [math.factorial(i) for i in range(_NUM_MODALITIES + 1)]
_SHAPLEY_WEIGHTS = {
    subset_size: (_FACTORIAL[subset_size] * _FACTORIAL[_NUM_MODALITIES - subset_size - 1]) / _FACTORIAL[_NUM_MODALITIES]
    for subset_size in range(_NUM_MODALITIES)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_obs = clean_reference_obs if clean_reference_obs is not None else state_obs
    reference_outputs = model.forward_reliability(reference_obs)
    fused_reference = reference_outputs["fused_feature"]
    reference_logits = reference_outputs["reliability_logits"]
    utilities: dict[int, torch.Tensor] = {}

    for subset_mask in _ALL_SUBSET_MASKS:
        subset_obs = _build_subset_obs(state_obs, subset_mask)
        subset_outputs = model.forward_reliability(subset_obs)
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
    for modality_idx in range(_NUM_MODALITIES):
        total = torch.zeros((), dtype=base_dtype, device=base_device)
        bit = 1 << modality_idx
        for subset_mask in _ALL_SUBSET_MASKS:
            if subset_mask & bit:
                continue
            subset_size = int(bin(subset_mask).count("1"))
            weight = _SHAPLEY_WEIGHTS[subset_size]
            marginal = utilities[subset_mask | bit] - utilities[subset_mask]
            total = total + float(weight) * marginal
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
) -> tuple[np.ndarray, np.ndarray]:
    shapley_values, shapley_target = _compute_exact_counterfactual_shapley_torch(
        model=model,
        state_obs=state_obs,
        clean_reference_obs=clean_reference_obs,
        temperature=temperature,
        utility_mode=utility_mode,
        target_modality_index=target_modality_index,
    )
    shapley_values_np = shapley_values.detach().cpu().numpy().astype(np.float32)
    shapley_target_np = shapley_target.detach().cpu().numpy().astype(np.float32)
    if not return_raw:
        return shapley_target_np, shapley_target_np
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
            _values, shapley_target = compute_exact_counterfactual_shapley(self.model, state_obs, temperature=temperature)
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
            shapley_target_values = []
            shapley_kls = []
            shapley_rank_corrs = []
            rng = np.random.default_rng(self.seed + 19000 + abs(hash(probe_name)) % 997)

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

                    clean_output = self.model.step(
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
                        temperature=temperature,
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
                    clean_logits = clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(clean_logits[2]), float(logits[2])))
                    action_diffs.append(float(torch.mean(torch.abs(clean_output.action - corrupt_output.action)).cpu().item()))
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
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
            }
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
            rng = np.random.default_rng(self.seed + 29000 + abs(hash(probe_name)) % 997)

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

                    clean_output = self.model.step(
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
                        temperature=temperature,
                        return_raw=True,
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
                    clean_logits = clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(clean_logits[2]), float(logits[2])))
                    action_diffs.append(float(torch.mean(torch.abs(clean_output.action - corrupt_output.action)).cpu().item()))
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
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
            }
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
            rng = np.random.default_rng(self.seed + 39000 + abs(hash(probe_name)) % 997)

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

                    clean_output = self.model.step(
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
                        temperature=temperature,
                        utility_mode="clean_fused_l2",
                        return_raw=True,
                    )
                    vision_targetlogit_shapley, _ = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        temperature=temperature,
                        utility_mode="target_logit_clean_l1",
                        target_modality_index=0,
                        return_raw=True,
                    )
                    ego_targetlogit_shapley, _ = compute_exact_counterfactual_shapley(
                        self.model,
                        state_obs=corrupt_obs_t,
                        clean_reference_obs=clean_obs_t,
                        temperature=temperature,
                        utility_mode="target_logit_clean_l1",
                        target_modality_index=2,
                        return_raw=True,
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
                    clean_logits = clean_output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy()
                    attention_values.append(attention)
                    reliability_values.append(reliability)
                    vision_logit_values.append((float(clean_logits[0]), float(logits[0])))
                    ego_logit_values.append((float(clean_logits[2]), float(logits[2])))
                    action_diffs.append(float(torch.mean(torch.abs(clean_output.action - corrupt_output.action)).cpu().item()))
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
                "logit_mean_vision_weak": np.nan,
                "logit_mean_vision_mild": (
                    float(vision_logit_arr[:, 1].mean()) if "vision" in probe_name and "mild" in probe_name else np.nan
                ),
                "logit_mean_ego_clean": float(ego_logit_arr[:, 0].mean()),
                "logit_mean_ego_weak": np.nan,
                "logit_mean_ego_mild": (
                    float(ego_logit_arr[:, 1].mean()) if "ego" in probe_name and "mild" in probe_name else np.nan
                ),
                "attention_shapley_kl": float(np.mean(shapley_kls)),
                "attention_shapley_rank_corr": float(np.mean(shapley_rank_corrs)),
                "targetlogit_shapley_mean_vision": float(vision_targetlogit_arr.mean()),
                "targetlogit_shapley_mean_ego": float(ego_targetlogit_arr.mean()),
            }
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
                row[f"raw_shapley_mean_{name}"] = float(raw_shapley_arr[:, idx].mean())
                row[f"shapley_target_mean_{name}"] = float(shapley_target_arr[:, idx].mean())
            rows.append(row)
        return rows


def build_v3_candidate_config(config: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    candidate_cfg = copy.deepcopy(config)
    candidate_spec = candidate_cfg["candidates"][candidate_name]
    for key, value in candidate_spec.get("loss_overrides", {}).items():
        candidate_cfg["losses"][key] = value
    for key, value in candidate_spec.get("train_overrides", {}).items():
        candidate_cfg["train"][key] = value
    for key, value in candidate_spec.get("train_corruption_overrides", {}).items():
        candidate_cfg["train_corruption"][key] = value
    return candidate_cfg


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

    for seed in seeds:
        run_dir = ensure_dir(output_root / candidate_name / f"seed_{seed}")
        summary_path = run_dir / metrics_filename
        if summary_path.exists():
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
