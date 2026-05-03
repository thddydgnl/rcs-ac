from __future__ import annotations

import torch
from torch import nn

from rcs_ac_first.models.encoders import MultiModalEncoder
from rcs_ac_first.models.fusion_gate import FusionGate, FusionGateConfig, MODALITY_ORDER
from rcs_ac_first.models.masked_critic import MaskedCritic, MaskedCriticConfig
from rcs_ac_first.models.perception_aux import (
    EstimatedScalarHead,
    RGBPerceptionActorFeatureExtractor,
    RGBPerceptionAuxHead,
    build_rgb_perception_raw_features,
)
from rcs_ac_first.models.reliability import RCSACReliabilityModule, ReliabilityConfig
from rcs_ac_first.models.recurrent_actor_critic import (
    RecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from rcs_ac_first.models.risk_context import RiskContextConfig, RiskContextHead
from rcs_ac_first.runtime.auxiliary import (
    RISK_STATS_FIELDS,
    build_auxiliary_observation_spec,
    default_flat_auxiliary,
    split_flat_auxiliary_observation,
)
from rcs_ac_first.runtime.config import RCSACFirstFusionConfig, RCSACFirstModelConfig
from rcs_ac_first.runtime.modality_mask import validate_modality_mask
from rcs_ac_first.runtime.observation import build_policy_observation_spec, split_flat_observation


class RCSACFirstModel(nn.Module):
    def __init__(
        self,
        config: RCSACFirstModelConfig | None = None,
        fusion_config: RCSACFirstFusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RCSACFirstModelConfig()
        self.obs_spec = build_policy_observation_spec(
            rgb_shape=self.config.rgb_shape,
            lidar_shape=self.config.lidar_shape,
            ego_dim=self.config.ego_dim,
            route_dim=self.config.route_dim,
        )
        self.aux_spec = build_auxiliary_observation_spec(
            quality_stats_dim=self.config.quality_stats_dim,
            risk_stats_dim=self.config.risk_stats_dim,
            route_commitment_stats_dim=self.config.route_commitment_stats_dim,
            sensor_dropout_dim=self.config.sensor_dropout_dim,
        )
        self.encoder = MultiModalEncoder(
            rgb_shape=self.config.rgb_shape,
            lidar_shape=self.config.lidar_shape,
            ego_dim=self.config.ego_dim,
            route_dim=self.config.route_dim,
            feature_dim=self.config.feature_dim,
        )
        fusion_cfg = fusion_config or RCSACFirstFusionConfig(
            feature_dim=self.config.feature_dim,
            fusion_dim=self.config.fusion_dim,
        )
        self.fusion_gate = FusionGate(
            FusionGateConfig(
                feature_dim=fusion_cfg.feature_dim,
                fusion_dim=fusion_cfg.fusion_dim,
                hidden_dim=fusion_cfg.hidden_dim,
                temperature=fusion_cfg.temperature,
                route_weight_floor=fusion_cfg.route_weight_floor,
                ego_weight_floor=fusion_cfg.ego_weight_floor,
                contribution_bias_scale=fusion_cfg.contribution_bias_scale,
                risk_context_dim=fusion_cfg.risk_context_dim,
                reliability_logit_scale=fusion_cfg.reliability_logit_scale,
                risk_logit_scale=fusion_cfg.risk_logit_scale,
                use_risk_context=fusion_cfg.use_risk_context,
            )
        )
        self.recurrent_actor_critic = RecurrentActorCritic(
            RecurrentActorCriticConfig(
                input_dim=self.config.fusion_dim,
                hidden_dim=self.config.recurrent_hidden_dim,
                layers=self.config.recurrent_layers,
                action_dim=self.config.action_dim,
                actor_hidden_dim=self.config.actor_hidden_dim,
                critic_hidden_dim=self.config.critic_hidden_dim,
                log_std_init_by_dim=tuple(self.config.log_std_init_by_dim),
                min_std_by_dim=tuple(self.config.min_std_by_dim),
                max_std_by_dim=tuple(self.config.max_std_by_dim),
            )
        )
        self.aux_head = nn.Sequential(
            nn.Linear(int(self.config.fusion_dim), int(self.config.aux_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.aux_hidden_dim), 2),
        )
        self.perception_aux_head = (
            RGBPerceptionAuxHead(
                in_channels=128,
                hidden_channels=self.config.aux_perception_hidden_channels,
                output_size=(self.config.aux_mask_height, self.config.aux_mask_width),
            )
            if self.config.enable_rgb_perception_aux
            else None
        )
        self.rgb_perception_actor_features = (
            RGBPerceptionActorFeatureExtractor(
                output_dim=self.config.rgb_perception_actor_feature_dim,
                hidden_dim=max(32, int(self.config.rgb_perception_actor_feature_dim) * 2),
            )
            if self.config.use_rgb_perception_actor_features
            else None
        )
        self.actor_input_adapter = (
            nn.Linear(
                int(self.config.fusion_dim) + int(self.config.rgb_perception_actor_feature_dim),
                int(self.config.fusion_dim),
            )
            if self.config.use_rgb_perception_actor_features
            else None
        )
        if self.actor_input_adapter is not None:
            self._initialize_actor_input_adapter()
        self.scalar_actor_adapter = (
            nn.Linear(int(self.config.scalar_actor_feature_dim), int(self.config.fusion_dim), bias=False)
            if self.config.use_scalar_actor_features
            else None
        )
        if self.scalar_actor_adapter is not None:
            self._initialize_scalar_actor_adapter()
        self.estimated_scalar_head = (
            EstimatedScalarHead(
                route_dim=int(self.config.route_dim),
                hidden_dim=int(self.config.estimated_scalar_hidden_dim),
            )
            if self.config.use_scalar_actor_features and self.config.scalar_actor_feature_source == "estimated"
            else None
        )
        self.reliability_module = (
            RCSACReliabilityModule(
                ReliabilityConfig(
                    auxiliary_dim=self.config.auxiliary_dim,
                    hidden_dim=self.config.reliability_hidden_dim,
                    minimum=self.config.reliability_min,
                    maximum=self.config.reliability_max,
                ),
                spec=self.aux_spec,
            )
            if self.config.enable_reliability
            else None
        )
        self.risk_context_head = (
            RiskContextHead(
                RiskContextConfig(
                    risk_stats_dim=self.config.risk_stats_dim,
                    route_commitment_dim=self.config.route_commitment_stats_dim,
                    sensor_dropout_dim=self.config.sensor_dropout_dim,
                    reliability_dim=4,
                    hidden_dim=self.config.risk_hidden_dim,
                    context_dim=self.config.risk_context_dim,
                ),
                spec=self.aux_spec,
            )
            if self.config.enable_risk_context
            else None
        )
        self.masked_critic = MaskedCritic(
            MaskedCriticConfig(
                fusion_dim=self.config.fusion_dim,
                risk_context_dim=self.config.risk_context_dim,
                hidden_dim=self.config.critic_hidden_dim,
            )
        )
        self._last_scalar_actor_features: torch.Tensor | None = None
        self._last_estimated_lane_offset: torch.Tensor | None = None
        self._last_estimated_heading_error: torch.Tensor | None = None
        self._last_diagnostics = self._empty_diagnostics()

    def forward_policy_outputs(
        self,
        obs: torch.Tensor,
        *,
        auxiliary: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        contribution_override: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
        return_shapley_context: bool = False,
        track_diagnostics: bool = True,
        apply_route_dropout: bool | None = None,
        zero_rgb_perception_actor_features: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        del apply_route_dropout
        split = split_flat_observation(obs, self.obs_spec)
        encoder_output = self.encoder(split)
        features = {
            "rgb": encoder_output.rgb,
            "lidar": encoder_output.lidar,
            "ego": encoder_output.ego,
            "route": encoder_output.route,
        }
        batch_size = int(encoder_output.rgb.shape[0])
        perception_aux = self._predict_perception_aux(
            encoder_output.rgb_feature_map,
            batch_size=batch_size,
        )
        perception_actor = self._predict_rgb_perception_actor_features(
            perception_aux["pred_road_mask_logits"],
            perception_aux["pred_lane_edge_mask_logits"],
            zero_features=zero_rgb_perception_actor_features,
        )
        auxiliary_tensor = self._prepare_auxiliary(auxiliary, batch_size=batch_size, reference=encoder_output.rgb)
        reliability_outputs = self._predict_reliability(auxiliary_tensor, batch_size=batch_size, reference=encoder_output.rgb)
        reliability = reliability_outputs["reliability"]
        reliability_logits = reliability_outputs["reliability_logits"]
        contribution = self._prepare_contribution(contribution_override, reference=reliability)
        risk_outputs = self._predict_risk(
            auxiliary_tensor,
            reliability=reliability,
            batch_size=batch_size,
            reference=encoder_output.rgb,
        )
        risk_scalar = risk_outputs["risk_scalar"]
        risk_context = risk_outputs["risk_context"]
        risk_logits = risk_outputs["risk_logits"]
        fusion_output = self.fusion_gate(
            features,
            reliability=reliability,
            contribution=contribution,
            risk_scalar=risk_scalar,
            risk_context=risk_context,
            modality_mask=modality_mask,
        )
        actor_input = self._build_actor_input(
            fusion_output.fused,
            perception_actor["compact_features"],
            road_mask_logits=perception_aux["pred_road_mask_logits"],
            lane_edge_mask_logits=perception_aux["pred_lane_edge_mask_logits"],
            auxiliary_tensor=auxiliary_tensor,
            route_input=split["route"],
        )
        ac_output = self.recurrent_actor_critic(actor_input, hidden_state=hidden_state)
        aux = self.aux_head(fusion_output.fused)
        projected_feature_stack = torch.stack([fusion_output.projected_features[key] for key in MODALITY_ORDER], dim=1)
        prepared_mask = validate_modality_mask(
            modality_mask,
            batch_size=batch_size,
            device=projected_feature_stack.device,
            dtype=projected_feature_stack.dtype,
            allow_empty=False,
        )
        outputs: dict[str, torch.Tensor | None] = {
            "action_mean": ac_output.action_mean,
            "steer_mean": ac_output.action_mean[:, 0:1],
            "longitudinal_mean": ac_output.action_mean[:, 1:2],
            "value": ac_output.value,
            "fusion_weights": fusion_output.weights,
            "next_hidden_state": ac_output.next_hidden,
            "pred_lane_offset": aux[:, 0],
            "pred_heading_error": aux[:, 1],
            "pred_road_mask_logits": perception_aux["pred_road_mask_logits"],
            "pred_lane_edge_mask_logits": perception_aux["pred_lane_edge_mask_logits"],
            "rgb_perception_actor_features": perception_actor["compact_features"],
            "rgb_perception_actor_raw_features": perception_actor["raw_features"],
            "estimated_lane_offset": getattr(self, "_last_estimated_lane_offset", None),
            "estimated_heading_error": getattr(self, "_last_estimated_heading_error", None),
            "scalar_actor_feature_source": None,
            "actor_input_feature": actor_input,
            "reliability": reliability,
            "reliability_logits": reliability_logits,
            "risk_scalar": risk_scalar,
            "risk_context": risk_context,
            "risk_logits": risk_logits,
            "contribution": contribution,
            "fusion_logits": fusion_output.logits,
            "projected_feature_stack": projected_feature_stack if return_shapley_context else None,
            "modality_mask": prepared_mask,
            "shapley_raw": None,
        }
        if track_diagnostics:
            self._last_diagnostics = self._build_diagnostics(
                encoder_diagnostics=encoder_output.diagnostics,
                fusion_diagnostics=fusion_output.diagnostics,
                perception_actor_features=perception_actor["compact_features"],
                perception_actor_raw_features=perception_actor["raw_features"],
                actor_input=actor_input,
                fused_feature=fusion_output.fused,
                recurrent_feature=ac_output.recurrent_feature,
                auxiliary_tensor=auxiliary_tensor,
                route_input=split["route"],
            )
        return outputs

    def forward(
        self,
        obs: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_policy_outputs(obs, **kwargs)
        return outputs["action_mean"], outputs["value"]  # type: ignore[return-value]

    def predict_auxiliary_targets(
        self,
        obs: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
        apply_route_dropout: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward_policy_outputs(
            obs,
            auxiliary=auxiliary,
            hidden_state=hidden_state,
            apply_route_dropout=apply_route_dropout,
            track_diagnostics=False,
        )
        return {
            "pred_lane_offset": outputs["pred_lane_offset"],  # type: ignore[dict-item]
            "pred_heading_error": outputs["pred_heading_error"],  # type: ignore[dict-item]
            "pred_road_mask_logits": outputs["pred_road_mask_logits"],  # type: ignore[dict-item]
            "pred_lane_edge_mask_logits": outputs["pred_lane_edge_mask_logits"],  # type: ignore[dict-item]
        }

    def get_std(self) -> torch.Tensor:
        return self.recurrent_actor_critic.get_std()

    def get_diagnostics(self) -> dict[str, float]:
        diagnostics = dict(self._last_diagnostics)
        std = self.get_std()
        diagnostics["policy_std_mean"] = float(std.mean().detach().cpu().item())
        for index in range(int(std.numel())):
            diagnostics[f"policy_std_dim{index}"] = float(std[index].detach().cpu().item())
        log_std = self.recurrent_actor_critic.log_std.detach()
        for index in range(int(log_std.numel())):
            diagnostics[f"log_std_dim{index}"] = float(log_std[index].cpu().item())
        for index, key in enumerate(MODALITY_ORDER):
            fusion_key = f"fusion_weight_{key}"
            diagnostics[f"attention_mean_{key}"] = float(diagnostics.get(fusion_key, 0.0))
        diagnostics["attention_perception_sum"] = float(
            diagnostics.get("fusion_weight_rgb", 0.0) + diagnostics.get("fusion_weight_lidar", 0.0)
        )
        diagnostics["attention_state_sum"] = float(
            diagnostics.get("fusion_weight_ego", 0.0) + diagnostics.get("fusion_weight_route", 0.0)
        )
        diagnostics["route_token_dropout_rate"] = 0.0
        return diagnostics

    def get_stage_b_warmup_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.encoder, self.fusion_gate, self.aux_head]
        if self.perception_aux_head is not None:
            modules.append(self.perception_aux_head)
        if self.rgb_perception_actor_features is not None:
            modules.append(self.rgb_perception_actor_features)
        if self.actor_input_adapter is not None:
            modules.append(self.actor_input_adapter)
        if self.scalar_actor_adapter is not None:
            modules.append(self.scalar_actor_adapter)
        if self.estimated_scalar_head is not None:
            modules.append(self.estimated_scalar_head)
        if self.reliability_module is not None:
            modules.append(self.reliability_module)
        if self.risk_context_head is not None:
            modules.append(self.risk_context_head)
        return modules

    def get_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        zero = self.get_std().new_zeros(())
        return {
            "reliability_loss": zero,
            "risk_regularization": zero,
            "shapley_alignment_loss": zero,
            "dropout_consistency_loss": zero,
        }

    def evaluate_masked_values(
        self,
        projected_feature_stack: torch.Tensor,
        fusion_weights: torch.Tensor,
        modality_masks: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if projected_feature_stack.ndim != 3 or int(projected_feature_stack.shape[1]) != len(MODALITY_ORDER):
            raise ValueError(f"projected_feature_stack must have shape [B, 4, D], got {tuple(projected_feature_stack.shape)}")
        batch_size = int(projected_feature_stack.shape[0])
        if tuple(fusion_weights.shape) != (batch_size, len(MODALITY_ORDER)):
            raise ValueError(f"fusion_weights shape mismatch: {tuple(fusion_weights.shape)}")
        masks = validate_modality_mask(
            modality_masks,
            batch_size=batch_size,
            device=projected_feature_stack.device,
            dtype=projected_feature_stack.dtype,
            allow_empty=True,
        )
        masked_weights = fusion_weights.to(dtype=projected_feature_stack.dtype, device=projected_feature_stack.device) * masks
        weight_sum = masked_weights.sum(dim=-1, keepdim=True)
        non_empty = weight_sum > 0.0
        normalized = torch.where(non_empty, masked_weights / weight_sum.clamp_min(1.0e-8), torch.zeros_like(masked_weights))
        fused = torch.sum(projected_feature_stack * normalized.unsqueeze(-1), dim=1)
        return self.recurrent_actor_critic.evaluate_value(fused, hidden_state=hidden_state)

    def evaluate_masked_critic(
        self,
        projected_feature_stack: torch.Tensor,
        fusion_weights: torch.Tensor,
        modality_masks: torch.Tensor,
        reliability: torch.Tensor,
        risk_scalar: torch.Tensor,
        risk_context: torch.Tensor,
    ) -> torch.Tensor:
        masks = validate_modality_mask(
            modality_masks,
            batch_size=int(projected_feature_stack.shape[0]),
            device=projected_feature_stack.device,
            dtype=projected_feature_stack.dtype,
            allow_empty=True,
        )
        masked_weights = fusion_weights.to(dtype=projected_feature_stack.dtype, device=projected_feature_stack.device) * masks
        normalized = masked_weights / masked_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        fused = torch.sum(projected_feature_stack * normalized.unsqueeze(-1), dim=1)
        return self.masked_critic(fused, masks, reliability, risk_scalar, risk_context)

    def _empty_diagnostics(self) -> dict[str, float]:
        return {
            "policy_std_mean": 0.0,
            "policy_std_dim0": 0.0,
            "policy_std_dim1": 0.0,
            "rgb_input_mean": 0.0,
            "lidar_input_mean": 0.0,
            "rgb_feature_l1_mean": 0.0,
            "rgb_feature_map_l1_mean": 0.0,
            "lidar_feature_l1_mean": 0.0,
            "ego_feature_l1_mean": 0.0,
            "route_feature_l1_mean": 0.0,
            "fused_feature_l1_mean": 0.0,
            "fusion_weight_rgb": 0.25,
            "fusion_weight_lidar": 0.25,
            "fusion_weight_ego": 0.25,
            "fusion_weight_route": 0.25,
            "fusion_entropy": 0.0,
            "fusion_ego_floor_applied_rate": 0.0,
            "fusion_route_floor_applied_rate": 0.0,
            "contribution_rgb": 0.25,
            "contribution_lidar": 0.25,
            "contribution_ego": 0.25,
            "contribution_route": 0.25,
            "contribution_entropy": 0.0,
            "contribution_max": 0.25,
            "contribution_min": 0.25,
            "shapley_raw_rgb": 0.0,
            "shapley_raw_lidar": 0.0,
            "shapley_raw_ego": 0.0,
            "shapley_raw_route": 0.0,
            "reliability_rgb": 1.0,
            "reliability_lidar": 1.0,
            "reliability_ego": 1.0,
            "reliability_route": 1.0,
            "risk_scalar": 0.0,
            "risk_scalar_mean": 0.0,
            "risk_context_l1_mean": 0.0,
            "compact_visual_features_enabled": 0.0,
            "compact_visual_feature_dim": 0.0,
            "compact_visual_feature_l1_mean": 0.0,
            "road_compact_feature_l1_mean": 0.0,
            "lane_compact_feature_l1_mean": 0.0,
            "visual_road_x_mean": 0.0,
            "visual_lane_x_mean": 0.0,
            "visual_lane_center_error_mean": 0.0,
            "compact_visual_actor_input_l1_mean": 0.0,
            "visual_actor_delta_l1_mean": 0.0,
            "scalar_actor_features_enabled": 0.0,
            "scalar_actor_feature_source_estimated": 0.0,
            "scalar_actor_feature_source_oracle_auxiliary": 0.0,
            "scalar_actor_feature_l1_mean": 0.0,
            "scalar_actor_delta_l1_mean": 0.0,
            "estimated_lane_offset_mean": 0.0,
            "estimated_heading_error_mean": 0.0,
            "estimated_lane_offset_mae": 0.0,
            "estimated_heading_error_mae": 0.0,
            "estimated_lane_offset_sign_agreement": 0.0,
            "estimated_heading_error_sign_agreement": 0.0,
            "recurrent_hidden_l1_mean": 0.0,
        }

    def _initialize_actor_input_adapter(self) -> None:
        if self.actor_input_adapter is None:
            return
        with torch.no_grad():
            self.actor_input_adapter.weight.zero_()
            self.actor_input_adapter.bias.zero_()
            fusion_dim = int(self.config.fusion_dim)
            self.actor_input_adapter.weight[:, :fusion_dim].copy_(torch.eye(fusion_dim))
            visual_dim = int(self.config.rgb_perception_actor_feature_dim)
            if visual_dim > 0:
                self.actor_input_adapter.weight[:, fusion_dim:].normal_(
                    mean=0.0,
                    std=float(self.config.rgb_perception_actor_adapter_init_std),
                )

    def _initialize_scalar_actor_adapter(self) -> None:
        if self.scalar_actor_adapter is None:
            return
        with torch.no_grad():
            self.scalar_actor_adapter.weight.normal_(
                mean=0.0,
                std=float(self.config.scalar_actor_adapter_init_std),
            )

    def _prepare_auxiliary(
        self,
        auxiliary: torch.Tensor | None,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if auxiliary is None:
            return default_flat_auxiliary(batch_size, spec=self.aux_spec, device=reference.device).to(dtype=reference.dtype)
        aux = auxiliary.to(device=reference.device, dtype=reference.dtype)
        if aux.ndim == 1:
            aux = aux.unsqueeze(0)
        if tuple(aux.shape) != (int(batch_size), int(self.config.auxiliary_dim)):
            raise ValueError(
                f"Expected auxiliary shape {(int(batch_size), int(self.config.auxiliary_dim))}, got {tuple(aux.shape)}"
            )
        if not torch.isfinite(aux).all():
            raise ValueError("Auxiliary observation contains NaN or inf values.")
        return aux

    def _prepare_contribution(
        self,
        contribution: torch.Tensor | None,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(reference.shape[0])
        if contribution is None:
            return torch.full((batch_size, len(MODALITY_ORDER)), 1.0 / float(len(MODALITY_ORDER)), dtype=reference.dtype, device=reference.device)
        result = contribution.to(device=reference.device, dtype=reference.dtype)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        expected = (batch_size, len(MODALITY_ORDER))
        if tuple(result.shape) != expected:
            raise ValueError(f"contribution_override shape mismatch: expected {expected}, got {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ValueError("contribution_override contains NaN or inf values")
        result = result.clamp_min(1.0e-8)
        return result / result.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _predict_reliability(
        self,
        auxiliary: torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.reliability_module is not None:
            return self.reliability_module(auxiliary)
        reliability = torch.ones((batch_size, 4), dtype=reference.dtype, device=reference.device)
        return {
            "reliability": reliability,
            "reliability_logits": torch.logit(reliability.clamp(1.0e-4, 1.0 - 1.0e-4)),
        }

    def _predict_risk(
        self,
        auxiliary: torch.Tensor,
        *,
        reliability: torch.Tensor,
        batch_size: int,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.risk_context_head is not None:
            return self.risk_context_head(auxiliary, reliability)
        risk_scalar = torch.zeros((batch_size,), dtype=reference.dtype, device=reference.device)
        risk_context = torch.zeros((batch_size, int(self.config.risk_context_dim)), dtype=reference.dtype, device=reference.device)
        return {
            "risk_scalar": risk_scalar,
            "risk_context": risk_context,
            "risk_logits": torch.logit(risk_scalar.clamp(1.0e-4, 1.0 - 1.0e-4)),
        }

    def _predict_perception_aux(
        self,
        rgb_feature_map: torch.Tensor,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if self.perception_aux_head is not None:
            return self.perception_aux_head(rgb_feature_map)
        shape = (
            int(batch_size),
            1,
            int(self.config.aux_mask_height),
            int(self.config.aux_mask_width),
        )
        zeros = rgb_feature_map.new_zeros(shape)
        return {
            "pred_road_mask_logits": zeros,
            "pred_lane_edge_mask_logits": zeros.clone(),
        }

    def _predict_rgb_perception_actor_features(
        self,
        road_mask_logits: torch.Tensor,
        lane_edge_mask_logits: torch.Tensor,
        *,
        zero_features: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(road_mask_logits.shape[0])
        feature_dim = int(self.config.rgb_perception_actor_feature_dim)
        if self.rgb_perception_actor_features is None:
            return {
                "raw_features": road_mask_logits.new_zeros((batch_size, 14)),
                "compact_features": road_mask_logits.new_zeros((batch_size, feature_dim)),
            }
        return self.rgb_perception_actor_features(
            road_mask_logits,
            lane_edge_mask_logits,
            zero_features=zero_features,
        )

    def _build_actor_input(
        self,
        fused_feature: torch.Tensor,
        perception_actor_features: torch.Tensor,
        *,
        road_mask_logits: torch.Tensor | None = None,
        lane_edge_mask_logits: torch.Tensor | None = None,
        auxiliary_tensor: torch.Tensor,
        route_input: torch.Tensor,
    ) -> torch.Tensor:
        if self.actor_input_adapter is None:
            actor_input = fused_feature
        else:
            if tuple(perception_actor_features.shape) != (
                int(fused_feature.shape[0]),
                int(self.config.rgb_perception_actor_feature_dim),
            ):
                raise ValueError(
                    "RGB perception actor feature shape mismatch: "
                    f"expected {(int(fused_feature.shape[0]), int(self.config.rgb_perception_actor_feature_dim))}, "
                    f"got {tuple(perception_actor_features.shape)}"
                )
            adapted_input = torch.cat([fused_feature, perception_actor_features.to(dtype=fused_feature.dtype)], dim=-1)
            actor_input = self.actor_input_adapter(adapted_input)
        if self.scalar_actor_adapter is None:
            return actor_input
        scalar_features = self._scalar_actor_features(
            auxiliary_tensor,
            road_mask_logits=road_mask_logits,
            lane_edge_mask_logits=lane_edge_mask_logits,
            route_input=route_input,
            reference=actor_input,
        )
        return actor_input + self.scalar_actor_adapter(scalar_features)

    def _scalar_actor_features(
        self,
        auxiliary_tensor: torch.Tensor,
        *,
        road_mask_logits: torch.Tensor | None = None,
        lane_edge_mask_logits: torch.Tensor | None = None,
        route_input: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        route = route_input.to(device=reference.device, dtype=reference.dtype)
        if self.config.scalar_actor_feature_source == "estimated":
            if self.estimated_scalar_head is None:
                raise RuntimeError("estimated scalar source requires estimated_scalar_head")
            if road_mask_logits is None or lane_edge_mask_logits is None:
                shape = (
                    int(route.shape[0]),
                    1,
                    int(self.config.aux_mask_height),
                    int(self.config.aux_mask_width),
                )
                road_mask_logits = reference.new_zeros(shape)
                lane_edge_mask_logits = reference.new_zeros(shape)
            visual_geometry = build_rgb_perception_raw_features(road_mask_logits, lane_edge_mask_logits).to(
                device=reference.device,
                dtype=reference.dtype,
            )
            estimated = self.estimated_scalar_head(visual_geometry, route)
            lane_value = estimated[:, 0]
            heading_value = estimated[:, 1]
        else:
            split = split_flat_auxiliary_observation(auxiliary_tensor, self.aux_spec)
            risk = split["risk_stats"].to(device=reference.device, dtype=reference.dtype)
            lane_index = RISK_STATS_FIELDS.index("lane_offset")
            heading_index = RISK_STATS_FIELDS.index("heading_error")
            lane_value = risk[:, lane_index]
            heading_value = risk[:, heading_index]
        self._last_estimated_lane_offset = lane_value if self.config.scalar_actor_feature_source == "estimated" else None
        self._last_estimated_heading_error = heading_value if self.config.scalar_actor_feature_source == "estimated" else None
        lane_offset = lane_value / float(self.config.scalar_actor_lane_offset_scale)
        heading_error = heading_value / float(self.config.scalar_actor_heading_error_scale)
        if int(route.shape[-1]) > 1:
            route_y = route[:, 1] / float(self.config.scalar_actor_route_y_scale)
        else:
            route_y = route.new_zeros((int(route.shape[0]),))
        scalar_features = torch.stack([lane_offset, heading_error, route_y], dim=-1).clamp(-5.0, 5.0)
        self._last_scalar_actor_features = scalar_features
        return scalar_features

    def _build_diagnostics(
        self,
        *,
        encoder_diagnostics: dict[str, float],
        fusion_diagnostics: dict[str, float],
        perception_actor_features: torch.Tensor,
        perception_actor_raw_features: torch.Tensor,
        actor_input: torch.Tensor,
        fused_feature: torch.Tensor,
        recurrent_feature: torch.Tensor,
        auxiliary_tensor: torch.Tensor,
        route_input: torch.Tensor,
    ) -> dict[str, float]:
        diagnostics = self._empty_diagnostics()
        diagnostics.update(encoder_diagnostics)
        diagnostics.update(fusion_diagnostics)
        diagnostics["recurrent_hidden_l1_mean"] = float(
            recurrent_feature.detach().abs().mean().cpu().item()
        )
        diagnostics["compact_visual_features_enabled"] = float(self.config.use_rgb_perception_actor_features)
        diagnostics["compact_visual_feature_dim"] = float(self.config.rgb_perception_actor_feature_dim)
        diagnostics["compact_visual_feature_l1_mean"] = float(
            perception_actor_features.detach().abs().mean().cpu().item()
        )
        if perception_actor_raw_features.numel() > 0:
            road_raw = perception_actor_raw_features[:, [0, 1, 2, 3, 10, 12]]
            lane_raw = perception_actor_raw_features[:, [4, 5, 6, 7, 8, 9, 11, 13]]
            diagnostics["road_compact_feature_l1_mean"] = float(road_raw.detach().abs().mean().cpu().item())
            diagnostics["lane_compact_feature_l1_mean"] = float(lane_raw.detach().abs().mean().cpu().item())
            road_x = perception_actor_raw_features[:, 10]
            lane_x = perception_actor_raw_features[:, 11]
            diagnostics["visual_road_x_mean"] = float(road_x.detach().mean().cpu().item())
            diagnostics["visual_lane_x_mean"] = float(lane_x.detach().mean().cpu().item())
            diagnostics["visual_lane_center_error_mean"] = float((lane_x - road_x).detach().mean().cpu().item())
        diagnostics["compact_visual_actor_input_l1_mean"] = float(
            actor_input.detach().abs().mean().cpu().item()
        )
        diagnostics["visual_actor_delta_l1_mean"] = float(
            (actor_input.detach() - fused_feature.detach()).abs().mean().cpu().item()
        )
        diagnostics["scalar_actor_features_enabled"] = float(self.config.use_scalar_actor_features)
        diagnostics["scalar_actor_feature_source_estimated"] = float(
            self.config.use_scalar_actor_features and self.config.scalar_actor_feature_source == "estimated"
        )
        diagnostics["scalar_actor_feature_source_oracle_auxiliary"] = float(
            self.config.use_scalar_actor_features and self.config.scalar_actor_feature_source == "oracle_auxiliary"
        )
        if self.scalar_actor_adapter is not None:
            diagnostics["scalar_actor_delta_l1_mean"] = float(
                self.scalar_actor_adapter.weight.detach().abs().mean().cpu().item()
            )
            if isinstance(self._last_scalar_actor_features, torch.Tensor):
                diagnostics["scalar_actor_feature_l1_mean"] = float(
                    self._last_scalar_actor_features.detach().abs().mean().cpu().item()
                )
        estimated_lane = getattr(self, "_last_estimated_lane_offset", None)
        estimated_heading = getattr(self, "_last_estimated_heading_error", None)
        if isinstance(estimated_lane, torch.Tensor) and isinstance(estimated_heading, torch.Tensor):
            diagnostics["estimated_lane_offset_mean"] = float(estimated_lane.detach().mean().cpu().item())
            diagnostics["estimated_heading_error_mean"] = float(estimated_heading.detach().mean().cpu().item())
            split = split_flat_auxiliary_observation(auxiliary_tensor, self.aux_spec)
            risk = split["risk_stats"].to(device=estimated_lane.device, dtype=estimated_lane.dtype)
            lane_target = risk[:, RISK_STATS_FIELDS.index("lane_offset")]
            heading_target = risk[:, RISK_STATS_FIELDS.index("heading_error")]
            diagnostics["estimated_lane_offset_mae"] = float((estimated_lane - lane_target).detach().abs().mean().cpu().item())
            diagnostics["estimated_heading_error_mae"] = float(
                (estimated_heading - heading_target).detach().abs().mean().cpu().item()
            )
            lane_mask = (estimated_lane.detach().abs() > 1.0e-6) & (lane_target.detach().abs() > 1.0e-6)
            heading_mask = (estimated_heading.detach().abs() > 1.0e-6) & (heading_target.detach().abs() > 1.0e-6)
            if bool(lane_mask.any().item()):
                diagnostics["estimated_lane_offset_sign_agreement"] = float(
                    (torch.sign(estimated_lane[lane_mask]) == torch.sign(lane_target[lane_mask]))
                    .to(dtype=torch.float32)
                    .mean()
                    .cpu()
                    .item()
                )
            if bool(heading_mask.any().item()):
                diagnostics["estimated_heading_error_sign_agreement"] = float(
                    (torch.sign(estimated_heading[heading_mask]) == torch.sign(heading_target[heading_mask]))
                    .to(dtype=torch.float32)
                    .mean()
                    .cpu()
                    .item()
                )
        return diagnostics
