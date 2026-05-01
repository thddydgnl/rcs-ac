from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.rl.distributions import BoundedSquashedNormal
from rcs_ac_first.rl.losses import (
    compute_dense_perception_aux_loss,
    compute_entropy_loss,
    compute_dropout_consistency_loss,
    compute_ppo_actor_loss,
    compute_reliability_supervision_loss,
    compute_risk_aware_regularization,
    compute_risk_supervision_loss,
    compute_scalar_aux_loss,
    compute_shapley_alignment_loss,
    compute_value_loss,
)
from rcs_ac_first.rl.rollout_buffer import RolloutBatch
from rcs_ac_first.rl.shapley import ShapleyEstimatorConfig, estimate_exact_shapley_current_critic
from rcs_ac_first.runtime.config import (
    Phase1CoreConfig,
    RCSACFirstDropoutConsistencyConfig,
    RCSACFirstFusionConfig,
    RCSACFirstLossConfig,
    RCSACFirstModelConfig,
    RCSACFirstShapleyConfig,
    RCSACFirstTrainConfig,
)
from rcs_ac_first.runtime.modality_mask import MODALITY_ORDER, single_modality_drop_masks


@dataclass
class PPOConfig:
    model: RCSACFirstModelConfig
    fusion: RCSACFirstFusionConfig
    loss: RCSACFirstLossConfig
    shapley: RCSACFirstShapleyConfig
    dropout_consistency: RCSACFirstDropoutConsistencyConfig
    train: RCSACFirstTrainConfig
    device: str
    action_bounds: list[tuple[float, float]]

    @classmethod
    def from_phase1_config(cls, config: Phase1CoreConfig, *, device_override: str | None = None) -> "PPOConfig":
        return cls(
            model=config.model,
            fusion=config.fusion,
            loss=config.loss,
            shapley=config.shapley,
            dropout_consistency=config.dropout_consistency,
            train=config.train,
            device=config.train.resolve_device(device_override),
            action_bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        )


class PPOAgent:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = RCSACFirstModel(config.model, config.fusion).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.train.learning_rate))
        self.hidden_state: torch.Tensor | None = None
        self.hidden_state_before_action: torch.Tensor | None = None
        self.update_count = 0

    def reset_hidden(self) -> None:
        self.hidden_state = None
        self.hidden_state_before_action = None

    def _distribution(self, action_mean: torch.Tensor) -> BoundedSquashedNormal:
        return BoundedSquashedNormal(
            action_mean=action_mean,
            action_std=self.model.get_std().to(device=action_mean.device, dtype=action_mean.dtype),
            action_bounds=self.config.action_bounds,
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray | torch.Tensor,
        auxiliary_observation: np.ndarray | torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        aux = self._prepare_single_auxiliary(auxiliary_observation)
        hidden_before = self.hidden_state
        if hidden_before is None:
            hidden_before = self._zero_hidden(batch_size=1)
        self.hidden_state_before_action = hidden_before.detach()
        outputs = self._forward_with_phase3_contribution(
            obs,
            aux,
            hidden_before,
            track_diagnostics=True,
        )
        action_mean = outputs["action_mean"]
        value = outputs["value"]
        next_hidden = outputs["next_hidden_state"]
        if isinstance(next_hidden, torch.Tensor):
            self.hidden_state = next_hidden.detach()
        dist = self._distribution(action_mean)
        action_tensor = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action_tensor)
        return (
            action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
            float(log_prob.squeeze(0).detach().cpu().item()),
            float(value.squeeze(0).detach().cpu().item()),  # type: ignore[union-attr]
        )

    @torch.no_grad()
    def estimate_value(
        self,
        observation: np.ndarray | torch.Tensor,
        auxiliary_observation: np.ndarray | torch.Tensor | None = None,
        hidden_state: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        aux = self._prepare_single_auxiliary(auxiliary_observation)
        hidden = self._prepare_single_hidden(hidden_state if hidden_state is not None else self.hidden_state)
        outputs = self._forward_with_phase3_contribution(obs, aux, hidden, track_diagnostics=False)
        value = outputs["value"]
        return float(value.squeeze(0).detach().cpu().item())  # type: ignore[union-attr]

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        auxiliary_observations: torch.Tensor | None,
        actions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        *,
        use_shapley_estimate: bool = True,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]:
        hidden_for_model = self._prepare_batch_hidden(
            hidden_states,
            batch_size=int(observations.shape[0]),
        )
        outputs = self._forward_with_phase3_contribution(
            observations,
            auxiliary_observations,
            hidden_for_model,
            track_diagnostics=True,
            use_shapley_estimate=use_shapley_estimate,
        )
        action_mean = outputs["action_mean"]
        dist = self._distribution(action_mean)  # type: ignore[arg-type]
        return outputs, dist.log_prob(actions), dist.entropy()

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        num_items = int(batch.observations.shape[0])
        indices = torch.arange(num_items, device=self.device)
        minibatch_size = min(int(self.config.train.minibatch_size), num_items)
        totals = _MetricAccumulator()
        use_shapley_this_update = self._shapley_enabled() and (
            int(self.update_count) % int(self.config.shapley.every_updates) == 0
        )

        for _ in range(int(self.config.train.update_epochs)):
            perm = indices[torch.randperm(num_items, device=self.device)]
            for start in range(0, num_items, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                outputs, new_log_probs, entropy = self.evaluate_actions(
                    batch.observations[mb_idx],
                    batch.auxiliary_observations[mb_idx],
                    batch.actions[mb_idx],
                    batch.hidden_states[mb_idx],
                    use_shapley_estimate=use_shapley_this_update,
                )
                outputs["_shapley_estimated"] = torch.as_tensor(
                    float(use_shapley_this_update),
                    dtype=torch.float32,
                    device=self.device,
                )
                value = outputs["value"]

                actor_loss = compute_ppo_actor_loss(
                    new_log_probs,
                    batch.log_probs[mb_idx],
                    batch.advantages[mb_idx],
                    self.config.train.clip_coef,
                )
                critic_loss = compute_value_loss(value, batch.returns[mb_idx])  # type: ignore[arg-type]
                aux_loss, aux_metrics = compute_scalar_aux_loss(
                    outputs["pred_lane_offset"],  # type: ignore[arg-type]
                    batch.aux_lane_offset_targets[mb_idx],
                    outputs["pred_heading_error"],  # type: ignore[arg-type]
                    batch.aux_heading_error_targets[mb_idx],
                    lane_offset_weight=self.config.loss.aux_lane_offset_loss_weight,
                    heading_error_weight=self.config.loss.aux_heading_error_loss_weight,
                )
                dense_aux_loss, dense_aux_metrics = compute_dense_perception_aux_loss(
                    outputs["pred_road_mask_logits"],  # type: ignore[arg-type]
                    batch.aux_road_mask_targets[mb_idx],
                    outputs["pred_lane_edge_mask_logits"],  # type: ignore[arg-type]
                    batch.aux_lane_edge_targets[mb_idx],
                    batch.aux_mask_targets_available[mb_idx],
                    road_mask_weight=self.config.loss.aux_road_mask_loss_weight,
                    lane_edge_weight=self.config.loss.aux_lane_edge_mask_loss_weight,
                    lane_edge_pos_weight=self.config.loss.aux_lane_edge_pos_weight,
                )
                entropy_loss = compute_entropy_loss(entropy)
                reliability_loss = compute_reliability_supervision_loss(
                    outputs["reliability"],  # type: ignore[arg-type]
                    batch.reliability_targets[mb_idx],
                )
                risk_supervision_loss = compute_risk_supervision_loss(
                    outputs["risk_scalar"],  # type: ignore[arg-type]
                    batch.risk_targets[mb_idx],
                )
                risk_regularization = compute_risk_aware_regularization(
                    outputs["fusion_weights"],  # type: ignore[arg-type]
                    outputs["reliability"],  # type: ignore[arg-type]
                    outputs["risk_scalar"],  # type: ignore[arg-type]
                )
                shapley_alignment_loss = self._compute_shapley_alignment_loss(outputs)
                dropout_consistency_loss, dropout_metrics = self._compute_dropout_consistency_loss(
                    outputs,
                    batch,
                    mb_idx,
                )
                total_loss = (
                    actor_loss
                    + self.config.loss.value_loss_weight * critic_loss
                    + self.config.loss.entropy_loss_weight * entropy_loss
                    + aux_loss
                    + dense_aux_loss
                    + self.config.loss.reliability_loss_weight * reliability_loss
                    + self.config.loss.risk_supervision_loss_weight * risk_supervision_loss
                    + self.config.loss.risk_regularization_weight * risk_regularization
                    + self.config.loss.shapley_alignment_loss_weight * shapley_alignment_loss
                    + self.config.loss.dropout_consistency_loss_weight * dropout_consistency_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)
                self.optimizer.step()

                totals.add(
                    {
                        "actor_loss": float(actor_loss.detach().cpu().item()),
                        "critic_loss": float(critic_loss.detach().cpu().item()),
                        "entropy": float(entropy.detach().mean().cpu().item()),
                        "total_loss": float(total_loss.detach().cpu().item()),
                        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
                        **aux_metrics,
                        **dense_aux_metrics,
                        "reliability_loss": float(reliability_loss.detach().cpu().item()),
                        "risk_supervision_loss": float(risk_supervision_loss.detach().cpu().item()),
                        "risk_regularization": float(risk_regularization.detach().cpu().item()),
                        **self._phase2_metrics(outputs, batch, mb_idx),
                        "shapley_estimated": float(use_shapley_this_update),
                        "shapley_alignment_loss": float(shapley_alignment_loss.detach().cpu().item()),
                        **dropout_metrics,
                    }
                )
        self.update_count += 1
        return totals.mean()

    def get_diagnostics(self) -> dict[str, float]:
        return self.model.get_diagnostics()

    def _zero_hidden(self, *, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            (
                int(self.config.model.recurrent_layers),
                int(batch_size),
                int(self.config.model.recurrent_hidden_dim),
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def _prepare_single_hidden(self, hidden_state: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if hidden_state is None:
            return self._zero_hidden(batch_size=1)
        hidden = torch.as_tensor(hidden_state, dtype=torch.float32, device=self.device)
        layers = int(self.config.model.recurrent_layers)
        hidden_dim = int(self.config.model.recurrent_hidden_dim)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        expected = (layers, 1, hidden_dim)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"Expected single hidden state shape {expected}, got {tuple(hidden.shape)}")
        return hidden

    def _prepare_single_auxiliary(self, auxiliary_observation: np.ndarray | torch.Tensor | None) -> torch.Tensor | None:
        if auxiliary_observation is None:
            return None
        auxiliary = torch.as_tensor(auxiliary_observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        expected = int(self.config.model.auxiliary_dim)
        if int(auxiliary.shape[-1]) != expected:
            raise ValueError(f"Expected auxiliary dim {expected}, got {int(auxiliary.shape[-1])}")
        return auxiliary

    def _prepare_batch_hidden(
        self,
        hidden_states: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        if hidden_states is None:
            return None
        hidden = hidden_states.to(device=self.device, dtype=torch.float32)
        layers = int(self.config.model.recurrent_layers)
        hidden_dim = int(self.config.model.recurrent_hidden_dim)
        buffer_shape = (int(batch_size), layers, hidden_dim)
        model_shape = (layers, int(batch_size), hidden_dim)
        if tuple(hidden.shape) == buffer_shape:
            return hidden.permute(1, 0, 2).contiguous()
        if tuple(hidden.shape) == model_shape:
            return hidden.contiguous()
        raise ValueError(
            "Expected hidden states shaped as "
            f"{buffer_shape} or {model_shape}, got {tuple(hidden.shape)}"
        )

    def _forward_with_phase3_contribution(
        self,
        observations: torch.Tensor,
        auxiliary_observations: torch.Tensor | None,
        hidden_state: torch.Tensor | None,
        *,
        track_diagnostics: bool,
        modality_mask: torch.Tensor | None = None,
        contribution_override: torch.Tensor | None = None,
        use_shapley_estimate: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        if contribution_override is not None or not self._shapley_enabled() or not bool(use_shapley_estimate):
            return self.model.forward_policy_outputs(
                observations,
                auxiliary=auxiliary_observations,
                hidden_state=hidden_state,
                contribution_override=contribution_override,
                modality_mask=modality_mask,
                track_diagnostics=track_diagnostics,
            )

        with torch.no_grad():
            base = self.model.forward_policy_outputs(
                observations,
                auxiliary=auxiliary_observations,
                hidden_state=hidden_state,
                return_shapley_context=True,
                track_diagnostics=False,
            )
            contribution, shapley_raw, subset_count = self._estimate_shapley_from_outputs(base, hidden_state)
        outputs = self.model.forward_policy_outputs(
            observations,
            auxiliary=auxiliary_observations,
            hidden_state=hidden_state,
            contribution_override=contribution,
            modality_mask=modality_mask,
            track_diagnostics=track_diagnostics,
        )
        outputs["shapley_raw"] = shapley_raw
        outputs["shapley_subset_count"] = torch.as_tensor(float(subset_count), dtype=torch.float32, device=self.device)
        return outputs

    def _estimate_shapley_from_outputs(
        self,
        outputs: dict[str, torch.Tensor | None],
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        projected = outputs.get("projected_feature_stack")
        weights = outputs.get("fusion_weights")
        if not isinstance(projected, torch.Tensor) or not isinstance(weights, torch.Tensor):
            raise RuntimeError("Shapley estimation requires projected_feature_stack and fusion_weights")
        estimate = estimate_exact_shapley_current_critic(
            self.model,
            projected_feature_stack=projected,
            fusion_weights=weights,
            hidden_state=hidden_state,
            config=ShapleyEstimatorConfig(
                subsets_per_sample=int(self.config.shapley.subsets_per_sample),
                batch_fraction=float(self.config.shapley.batch_fraction),
                contribution_temperature=float(self.config.shapley.contribution_temperature),
                min_contribution=float(self.config.shapley.min_contribution),
                include_empty_subset=bool(self.config.shapley.include_empty_subset),
                include_full_subset=bool(self.config.shapley.include_full_subset),
                detach_contribution_target=bool(self.config.shapley.detach_contribution_target),
            ),
        )
        return estimate.contribution, estimate.raw, estimate.subset_count

    def _shapley_enabled(self) -> bool:
        return bool(self.config.model.enable_shapley and self.config.shapley.enabled)

    def _dropout_consistency_enabled(self) -> bool:
        return bool(self.config.model.enable_dropout_consistency and self.config.dropout_consistency.enabled)

    def _compute_shapley_alignment_loss(self, outputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
        fusion_weights = outputs["fusion_weights"]
        contribution = outputs["contribution"]
        if not isinstance(fusion_weights, torch.Tensor):
            raise RuntimeError("fusion_weights missing from model outputs")
        if not self._shapley_enabled():
            return fusion_weights.new_zeros(())
        estimated = outputs.get("_shapley_estimated")
        if isinstance(estimated, torch.Tensor) and float(estimated.detach().cpu().item()) <= 0.0:
            return fusion_weights.new_zeros(())
        if not isinstance(contribution, torch.Tensor):
            raise RuntimeError("contribution missing from model outputs")
        return compute_shapley_alignment_loss(
            fusion_weights,
            contribution,
            detach_target=bool(self.config.shapley.detach_contribution_target),
        )

    def _compute_dropout_consistency_loss(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: RolloutBatch,
        mb_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        clean_action = outputs["action_mean"]
        clean_value = outputs["value"]
        if not isinstance(clean_action, torch.Tensor) or not isinstance(clean_value, torch.Tensor):
            raise RuntimeError("action/value missing from model outputs")
        zero = clean_action.new_zeros(())
        if not self._dropout_consistency_enabled():
            return zero, {
                "dropout_consistency_loss": 0.0,
                "dropout_action_l1": 0.0,
                "dropout_value_l1": 0.0,
                "dropout_mask_rgb_rate": 0.0,
                "dropout_mask_lidar_rate": 0.0,
                "dropout_mask_ego_rate": 0.0,
                "dropout_mask_route_rate": 0.0,
            }
        batch_size = int(mb_idx.shape[0])
        selected_count = max(1, int(round(batch_size * float(self.config.dropout_consistency.batch_fraction))))
        selected = mb_idx[:selected_count]
        clean_local = clean_action[:selected_count]
        clean_value_local = clean_value[:selected_count]
        hidden = self._prepare_batch_hidden(batch.hidden_states[selected], batch_size=selected_count)
        mask = single_modality_drop_masks(selected_count, device=self.device, dtype=batch.observations.dtype)
        contribution = outputs["contribution"]
        contribution_selected = contribution[:selected_count].detach() if isinstance(contribution, torch.Tensor) else None
        drop_outputs = self.model.forward_policy_outputs(
            batch.observations[selected],
            auxiliary=batch.auxiliary_observations[selected],
            hidden_state=hidden,
            contribution_override=contribution_selected,
            modality_mask=mask,
            track_diagnostics=False,
        )
        loss, metrics = compute_dropout_consistency_loss(
            clean_local,
            drop_outputs["action_mean"],  # type: ignore[arg-type]
            clean_value_local,
            drop_outputs["value"],  # type: ignore[arg-type]
            action_weight=float(self.config.dropout_consistency.action_mean_weight),
            value_weight=float(self.config.dropout_consistency.value_weight),
            detach_clean=bool(self.config.dropout_consistency.detach_clean_policy),
        )
        for index, key in enumerate(MODALITY_ORDER):
            metrics[f"dropout_mask_{key}_rate"] = float((1.0 - mask[:, index]).mean().detach().cpu().item())
        return loss, metrics

    def _phase2_metrics(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: RolloutBatch,
        mb_idx: torch.Tensor,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        reliability = outputs["reliability"]
        fusion_weights = outputs["fusion_weights"]
        risk_scalar = outputs["risk_scalar"]
        if isinstance(reliability, torch.Tensor):
            for index, key in enumerate(("rgb", "lidar", "ego", "route")):
                metrics[f"reliability_{key}"] = float(reliability[:, index].detach().mean().cpu().item())
                metrics[f"reliability_target_{key}"] = float(
                    batch.reliability_targets[mb_idx, index].detach().mean().cpu().item()
                )
        if isinstance(fusion_weights, torch.Tensor):
            for index, key in enumerate(("rgb", "lidar", "ego", "route")):
                metrics[f"fusion_weight_{key}"] = float(fusion_weights[:, index].detach().mean().cpu().item())
        contribution = outputs.get("contribution")
        if isinstance(contribution, torch.Tensor):
            for index, key in enumerate(("rgb", "lidar", "ego", "route")):
                metrics[f"contribution_{key}"] = float(contribution[:, index].detach().mean().cpu().item())
            entropy = -(contribution * contribution.clamp_min(1.0e-8).log()).sum(dim=-1)
            metrics["contribution_entropy"] = float(entropy.detach().mean().cpu().item())
        shapley_raw = outputs.get("shapley_raw")
        if isinstance(shapley_raw, torch.Tensor):
            for index, key in enumerate(("rgb", "lidar", "ego", "route")):
                metrics[f"shapley_raw_{key}"] = float(shapley_raw[:, index].detach().mean().cpu().item())
        subset_count = outputs.get("shapley_subset_count")
        if isinstance(subset_count, torch.Tensor):
            metrics["shapley_subset_count"] = float(subset_count.detach().cpu().item())
        if isinstance(risk_scalar, torch.Tensor):
            metrics["risk_scalar"] = float(risk_scalar.detach().mean().cpu().item())
        metrics["risk_target"] = float(batch.risk_targets[mb_idx].detach().mean().cpu().item())
        metrics["corruption_level"] = float(batch.corruption_levels[mb_idx].detach().mean().cpu().item())
        for index, key in enumerate(("rgb", "lidar", "ego", "route")):
            metrics[f"drop_{key}"] = float(batch.dropout_masks[mb_idx, index].detach().mean().cpu().item())
        return metrics

    def save(self, checkpoint_path: str, extra_state: dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.train.__dict__ | {"model": self.config.model.__dict__},
                "extra_state": dict(extra_state or {}),
            },
            checkpoint_path,
        )

    def load(
        self,
        checkpoint_path: str,
        *,
        load_optimizer: bool = False,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location or self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


class _MetricAccumulator:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def add(self, metrics: dict[str, float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / float(self.count) for key, value in self.totals.items()}
