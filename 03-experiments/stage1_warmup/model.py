"""Stage 1 warm-up recurrent model with weak reliability head."""

from __future__ import annotations

import torch
from torch import nn

from ppo_baselines.models.baselines import OBS_DIMS
from ppo_baselines.models.encoders import build_mlp
from rcs_ac_v1.model import MODALITY_NAMES, RecurrentAttentionActorCritic


class Stage1WarmupActorCritic(RecurrentAttentionActorCritic):
    """RCS-AC v1 backbone plus modality-wise reliability head.

    Policy/fusion stay identical to v1. Reliability is only a weak auxiliary
    prediction head during warm-up.
    """

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        encoder_output_dim: int,
        attention_hidden_dim: int,
        recurrent_hidden_dim: int,
        attention_temperature: float,
        use_feature_layer_norm: bool,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        reliability_hidden_dims: list[int],
        reliability_input_mode: str,
        action_dim: int,
        action_log_std_init: float,
        reliability_raw_adapter_dim: int = 16,
        reliability_ego_residual_enabled: bool = False,
        reliability_ego_raw_adapter_dim: int = 8,
        reliability_ego_residual_input_mode: str = "raw_only",
        reliability_ego_residual_apply_to_provisional: bool = True,
        reliability_local_residual_enabled_modalities: list[str] | None = None,
        reliability_local_residual_adapter_dim: int = 16,
        reliability_local_residual_input_mode: str = "raw_route_fused_v2",
        reliability_local_residual_apply_to_provisional: bool = False,
        append_mean_context: bool = False,
    ):
        super().__init__(
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            attention_hidden_dim=attention_hidden_dim,
            recurrent_hidden_dim=recurrent_hidden_dim,
            append_mean_context=append_mean_context,
            attention_temperature=attention_temperature,
            use_feature_layer_norm=use_feature_layer_norm,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
        self.reliability_input_mode = reliability_input_mode
        self.reliability_raw_adapter_dim = int(reliability_raw_adapter_dim)
        self.reliability_ego_raw_adapter_dim = int(reliability_ego_raw_adapter_dim)
        self.reliability_ego_residual_input_mode = str(reliability_ego_residual_input_mode)
        self.reliability_ego_residual_apply_to_provisional = bool(reliability_ego_residual_apply_to_provisional)
        self._reliability_use_fused = reliability_input_mode in {"own_plus_fused", "own_plus_fused_plus_raw"}
        self._reliability_use_raw = reliability_input_mode == "own_plus_fused_plus_raw"
        reliability_input_dim = encoder_output_dim
        if self._reliability_use_fused:
            reliability_input_dim += self.fused_feature_dim
        if self._reliability_use_raw:
            reliability_input_dim += self.reliability_raw_adapter_dim
            self.reliability_raw_adapters = nn.ModuleDict(
                {
                    name: build_mlp(
                        input_dim=OBS_DIMS[name],
                        hidden_dims=[],
                        output_dim=self.reliability_raw_adapter_dim,
                        output_activation=nn.Tanh(),
                    )
                    for name in MODALITY_NAMES
                }
            )
        else:
            self.reliability_raw_adapters = nn.ModuleDict()
        self.reliability_head = build_mlp(
            input_dim=reliability_input_dim,
            hidden_dims=reliability_hidden_dims,
            output_dim=1,
            output_activation=None,
        )
        explicit_local_modalities = [
            str(name) for name in (reliability_local_residual_enabled_modalities or []) if str(name) in MODALITY_NAMES
        ]
        if explicit_local_modalities:
            self.reliability_local_residual_enabled_modalities = list(dict.fromkeys(explicit_local_modalities))
            self.reliability_local_residual_adapter_dim = int(reliability_local_residual_adapter_dim)
            self.reliability_local_residual_input_mode = str(reliability_local_residual_input_mode)
            self.reliability_local_residual_apply_to_provisional = bool(
                reliability_local_residual_apply_to_provisional
            )
        elif bool(reliability_ego_residual_enabled):
            self.reliability_local_residual_enabled_modalities = ["ego"]
            self.reliability_local_residual_adapter_dim = self.reliability_ego_raw_adapter_dim
            self.reliability_local_residual_input_mode = self.reliability_ego_residual_input_mode
            self.reliability_local_residual_apply_to_provisional = self.reliability_ego_residual_apply_to_provisional
        else:
            self.reliability_local_residual_enabled_modalities = []
            self.reliability_local_residual_adapter_dim = int(reliability_local_residual_adapter_dim)
            self.reliability_local_residual_input_mode = str(reliability_local_residual_input_mode)
            self.reliability_local_residual_apply_to_provisional = bool(
                reliability_local_residual_apply_to_provisional
            )
        self.reliability_local_residual_enabled = bool(self.reliability_local_residual_enabled_modalities)
        self.reliability_ego_residual_enabled = "ego" in self.reliability_local_residual_enabled_modalities
        self.reliability_ego_residual_apply_to_provisional = bool(
            self.reliability_local_residual_apply_to_provisional if self.reliability_ego_residual_enabled else False
        )
        self.reliability_ego_residual_input_mode = (
            self.reliability_local_residual_input_mode
            if self.reliability_ego_residual_enabled
            else str(reliability_ego_residual_input_mode)
        )
        self.reliability_ego_raw_adapter_dim = (
            int(self.reliability_local_residual_adapter_dim)
            if self.reliability_ego_residual_enabled
            else int(reliability_ego_raw_adapter_dim)
        )
        self.reliability_local_residual_adapters = nn.ModuleDict()
        self.reliability_local_residual_heads = nn.ModuleDict()
        for modality_name in self.reliability_local_residual_enabled_modalities:
            input_dim = self._local_residual_input_dim(modality_name)
            self.reliability_local_residual_adapters[modality_name] = build_mlp(
                input_dim=input_dim,
                hidden_dims=[],
                output_dim=self.reliability_local_residual_adapter_dim,
                output_activation=nn.Tanh(),
            )
            head = build_mlp(
                input_dim=self.reliability_local_residual_adapter_dim,
                hidden_dims=[],
                output_dim=1,
                output_activation=None,
            )
            self._zero_last_linear(head)
            self.reliability_local_residual_heads[modality_name] = head
        if "ego" in self.reliability_local_residual_adapters:
            self.reliability_ego_raw_adapter = self.reliability_local_residual_adapters["ego"]
            self.reliability_ego_residual_head = self.reliability_local_residual_heads["ego"]
        else:
            self.reliability_ego_raw_adapter = nn.Identity()
            self.reliability_ego_residual_head = nn.Identity()

    def _local_residual_input_dim(self, modality_name: str) -> int:
        mode = self.reliability_local_residual_input_mode
        if mode == "raw_only":
            return OBS_DIMS[modality_name]
        if mode == "raw_route_fused_v1":
            if modality_name != "ego":
                raise ValueError("raw_route_fused_v1 is only supported for ego residual calibration")
            return OBS_DIMS["ego"] + OBS_DIMS["route"] + 2 + 2 + self.fused_feature_dim
        if mode == "raw_route_fused_v2":
            if modality_name == "vision":
                return OBS_DIMS["vision"] + self.fused_feature_dim + 2
            if modality_name == "ego":
                return OBS_DIMS["ego"] + OBS_DIMS["route"] + 2 + 2 + self.fused_feature_dim + 2
            return OBS_DIMS[modality_name] + self.fused_feature_dim + 2
        raise ValueError(f"Unsupported reliability_local_residual_input_mode: {mode}")

    @staticmethod
    def _zero_last_linear(module: nn.Module) -> None:
        for child in reversed(list(module.modules())):
            if isinstance(child, nn.Linear):
                nn.init.zeros_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
                return

    def _build_local_reliability_residual_inputs(
        self,
        modality_name: str,
        obs_dict: dict[str, torch.Tensor],
        fused: torch.Tensor,
        base_reliability_scores: torch.Tensor | None = None,
        base_reliability_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = self.reliability_local_residual_input_mode
        modality_idx = MODALITY_NAMES.index(modality_name)
        if mode == "raw_only":
            return obs_dict[modality_name]
        if mode == "raw_route_fused_v1":
            if modality_name != "ego":
                raise ValueError("raw_route_fused_v1 is only supported for ego residual calibration")
            ego = obs_dict["ego"]
            route = obs_dict["route"]
            heading_disagreement = ego[..., 3:5] - route[..., 3:5]
            speed_and_route = torch.cat([ego[..., 0:1], route[..., 2:3]], dim=-1)
            return torch.cat([ego, route, heading_disagreement, speed_and_route, fused.detach()], dim=-1)
        if mode == "raw_route_fused_v2":
            if base_reliability_scores is None or base_reliability_logits is None:
                raise ValueError("raw_route_fused_v2 requires base_reliability_scores and base_reliability_logits")
            base_score = base_reliability_scores[..., modality_idx : modality_idx + 1].detach()
            base_logit = base_reliability_logits[..., modality_idx : modality_idx + 1].detach()
            if modality_name == "vision":
                return torch.cat([obs_dict["vision"], fused.detach(), base_score, base_logit], dim=-1)
            if modality_name == "ego":
                ego = obs_dict["ego"]
                route = obs_dict["route"]
                heading_disagreement = ego[..., 3:5] - route[..., 3:5]
                speed_and_route = torch.cat([ego[..., 0:1], route[..., 2:3]], dim=-1)
                return torch.cat(
                    [ego, route, heading_disagreement, speed_and_route, fused.detach(), base_score, base_logit],
                    dim=-1,
                )
            return torch.cat([obs_dict[modality_name], fused.detach(), base_score, base_logit], dim=-1)
        raise ValueError(f"Unsupported reliability_local_residual_input_mode: {mode}")

    def _build_ego_reliability_residual_inputs(
        self,
        obs_dict: dict[str, torch.Tensor],
        fused: torch.Tensor,
    ) -> torch.Tensor:
        zeros = fused.new_zeros((fused.shape[0], len(MODALITY_NAMES)))
        mode = self.reliability_local_residual_input_mode
        try:
            self.reliability_local_residual_input_mode = self.reliability_ego_residual_input_mode
            return self._build_local_reliability_residual_inputs(
                "ego",
                obs_dict,
                fused,
                base_reliability_scores=zeros,
                base_reliability_logits=zeros,
            )
        finally:
            self.reliability_local_residual_input_mode = mode

    def _apply_local_reliability_residual(
        self,
        reliability_logits: torch.Tensor,
        obs_dict: dict[str, torch.Tensor],
        fused: torch.Tensor,
        base_reliability_scores: torch.Tensor,
        base_reliability_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residuals = torch.zeros_like(reliability_logits)
        if not self.reliability_local_residual_enabled_modalities:
            return reliability_logits, residuals
        for modality_name in self.reliability_local_residual_enabled_modalities:
            modality_idx = MODALITY_NAMES.index(modality_name)
            residual_inputs = self._build_local_reliability_residual_inputs(
                modality_name,
                obs_dict,
                fused,
                base_reliability_scores=base_reliability_scores,
                base_reliability_logits=base_reliability_logits,
            )
            residual_features = self.reliability_local_residual_adapters[modality_name](residual_inputs)
            residual_delta = self.reliability_local_residual_heads[modality_name](residual_features).squeeze(-1)
            residuals[..., modality_idx] = residual_delta
        return reliability_logits + residuals, residuals

    def _apply_ego_reliability_residual(
        self,
        reliability_logits: torch.Tensor,
        obs_dict: dict[str, torch.Tensor],
        fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_reliability_scores = torch.sigmoid(reliability_logits)
        return self._apply_local_reliability_residual(
            reliability_logits=reliability_logits,
            obs_dict=obs_dict,
            fused=fused,
            base_reliability_scores=base_reliability_scores,
            base_reliability_logits=reliability_logits,
        )

    def _build_reliability_inputs(
        self,
        stacked: torch.Tensor,
        fused: torch.Tensor,
        obs_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = [stacked]
        if self._reliability_use_fused:
            fused_expanded = fused.unsqueeze(1).expand(-1, stacked.shape[1], -1)
            parts.append(fused_expanded)
        if self._reliability_use_raw:
            raw_features = torch.stack(
                [self.reliability_raw_adapters[name](obs_dict[name]) for name in MODALITY_NAMES],
                dim=1,
            )
            parts.append(raw_features)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def freeze_policy_backbone(self) -> None:
        for module in [self.encoders, self.feature_norms, self.fusion_module, self.gru_cell, self.actor_head, self.critic_head]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.log_std.requires_grad = False

    def unfreeze_reliability_head(self) -> None:
        for parameter in self.reliability_head.parameters():
            parameter.requires_grad = True

    def freeze_ego_reliability_residual_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if not self.reliability_ego_residual_enabled:
            raise ValueError("freeze_ego_reliability_residual_only requires reliability_ego_residual_enabled")
        for module in [self.reliability_ego_raw_adapter, self.reliability_ego_residual_head]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def freeze_local_reliability_residual_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if not self.reliability_local_residual_enabled_modalities:
            raise ValueError("freeze_local_reliability_residual_only requires enabled local residual modalities")
        for modality_name in self.reliability_local_residual_enabled_modalities:
            for module in [
                self.reliability_local_residual_adapters[modality_name],
                self.reliability_local_residual_heads[modality_name],
            ]:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def freeze_attention_score_calibration_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.score_net.parameters():
            parameter.requires_grad = True

    def freeze_attention_score_adapter_calibration_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.score_net.parameters():
            parameter.requires_grad = True
        for parameter in self.fusion_module.score_adapter.parameters():
            parameter.requires_grad = True

    def freeze_attention_relative_score_calibration_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.score_adapter,
            self.fusion_module.base_score_head,
            self.fusion_module.relative_gap_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def freeze_attention_reliability_bias_calibration_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.score_adapter,
            self.fusion_module.base_score_head,
            self.fusion_module.relative_gap_head,
            self.fusion_module.reliability_bias_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def freeze_attention_gated_reliability_bias_calibration_only(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_head,
            self.fusion_module.reliability_bias_gate_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def freeze_attention_clean_preserving_bias_calibration_only(self, train_score_adapter: bool = False) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.reliability_bias_head.parameters():
            parameter.requires_grad = True
        if train_score_adapter:
            for parameter in self.fusion_module.score_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_gate_only_selective_finetune(self, train_gate_input_adapter: bool = False) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.reliability_bias_gate_head.parameters():
            parameter.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_residual_only(self, train_gate_input_adapter: bool = False) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.reliability_bias_gate_residual_head.parameters():
            parameter.requires_grad = True
        gate_residual_alpha = getattr(self.fusion_module, "gate_residual_alpha", None)
        if (
            isinstance(gate_residual_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_residual_alpha_mode", "fixed")) == "learnable"
        ):
            gate_residual_alpha.requires_grad = True
        gate_logit_offset = getattr(self.fusion_module, "gate_logit_offset", None)
        if (
            isinstance(gate_logit_offset, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_logit_offset_mode", "fixed")) == "learnable"
        ):
            gate_logit_offset.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_relative_micro_unfreeze(self, train_gate_input_adapter: bool = True) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.reliability_bias_gate_residual_head.parameters():
            parameter.requires_grad = True
        for parameter in self.fusion_module.relative_gap_head.parameters():
            parameter.requires_grad = True
        gate_residual_alpha = getattr(self.fusion_module, "gate_residual_alpha", None)
        if (
            isinstance(gate_residual_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_residual_alpha_mode", "fixed")) == "learnable"
        ):
            gate_residual_alpha.requires_grad = True
        gate_logit_offset = getattr(self.fusion_module, "gate_logit_offset", None)
        if (
            isinstance(gate_logit_offset, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_logit_offset_mode", "fixed")) == "learnable"
        ):
            gate_logit_offset.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_score_adapter_micro_unfreeze(self, train_gate_input_adapter: bool = True) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.fusion_module.reliability_bias_gate_residual_head.parameters():
            parameter.requires_grad = True
        for parameter in self.fusion_module.score_adapter.parameters():
            parameter.requires_grad = True
        gate_residual_alpha = getattr(self.fusion_module, "gate_residual_alpha", None)
        if (
            isinstance(gate_residual_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_residual_alpha_mode", "fixed")) == "learnable"
        ):
            gate_residual_alpha.requires_grad = True
        gate_logit_offset = getattr(self.fusion_module, "gate_logit_offset", None)
        if (
            isinstance(gate_logit_offset, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_logit_offset_mode", "fixed")) == "learnable"
        ):
            gate_logit_offset.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_adapter_coupled_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
        train_score_adapter: bool = False,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if train_score_adapter:
            for parameter in self.fusion_module.score_adapter.parameters():
                parameter.requires_grad = True
        gate_residual_alpha = getattr(self.fusion_module, "gate_residual_alpha", None)
        if (
            isinstance(gate_residual_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_residual_alpha_mode", "fixed")) == "learnable"
        ):
            gate_residual_alpha.requires_grad = True
        gate_logit_offset = getattr(self.fusion_module, "gate_logit_offset", None)
        if (
            isinstance(gate_logit_offset, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_logit_offset_mode", "fixed")) == "learnable"
        ):
            gate_logit_offset.requires_grad = True
        adapter_gate_alpha = getattr(self.fusion_module, "adapter_gate_alpha", None)
        if (
            isinstance(adapter_gate_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "adapter_gate_alpha_mode", "fixed")) == "learnable"
        ):
            adapter_gate_alpha.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_adapter_readout_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
        train_score_adapter: bool = False,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if train_score_adapter:
            for parameter in self.fusion_module.score_adapter.parameters():
                parameter.requires_grad = True
        gate_residual_alpha = getattr(self.fusion_module, "gate_residual_alpha", None)
        if (
            isinstance(gate_residual_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_residual_alpha_mode", "fixed")) == "learnable"
        ):
            gate_residual_alpha.requires_grad = True
        gate_logit_offset = getattr(self.fusion_module, "gate_logit_offset", None)
        if (
            isinstance(gate_logit_offset, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "gate_logit_offset_mode", "fixed")) == "learnable"
        ):
            gate_logit_offset.requires_grad = True
        adapter_gate_alpha = getattr(self.fusion_module, "adapter_gate_alpha", None)
        if (
            isinstance(adapter_gate_alpha, torch.nn.Parameter)
            and str(getattr(self.fusion_module, "adapter_gate_alpha_mode", "fixed")) == "learnable"
        ):
            adapter_gate_alpha.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_output_scale_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
        train_score_adapter: bool = False,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if train_score_adapter:
            for parameter in self.fusion_module.score_adapter.parameters():
                parameter.requires_grad = True
        for attr_name, mode_name in [
            ("gate_residual_alpha", "gate_residual_alpha_mode"),
            ("gate_logit_offset", "gate_logit_offset_mode"),
            ("adapter_gate_alpha", "adapter_gate_alpha_mode"),
            ("gate_output_scale", "gate_output_scale_mode"),
        ]:
            value = getattr(self.fusion_module, attr_name, None)
            if (
                isinstance(value, torch.nn.Parameter)
                and str(getattr(self.fusion_module, mode_name, "fixed")) == "learnable"
            ):
                value.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_bias_adapter_output_scale_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_head,
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
            self.fusion_module.score_adapter,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for attr_name, mode_name in [
            ("gate_residual_alpha", "gate_residual_alpha_mode"),
            ("gate_logit_offset", "gate_logit_offset_mode"),
            ("adapter_gate_alpha", "adapter_gate_alpha_mode"),
            ("gate_output_scale", "gate_output_scale_mode"),
            ("gate_source_logit_bias", "gate_source_logit_bias_mode"),
        ]:
            value = getattr(self.fusion_module, attr_name, None)
            if (
                isinstance(value, torch.nn.Parameter)
                and str(getattr(self.fusion_module, mode_name, "fixed")) == "learnable"
            ):
                value.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_threshold_shift_bias_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_head,
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
            self.fusion_module.reliability_bias_gate_threshold_shift_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for attr_name, mode_name in [
            ("gate_residual_alpha", "gate_residual_alpha_mode"),
            ("gate_logit_offset", "gate_logit_offset_mode"),
            ("adapter_gate_alpha", "adapter_gate_alpha_mode"),
            ("gate_output_scale", "gate_output_scale_mode"),
            ("gate_threshold_shift_alpha", "gate_threshold_shift_alpha_mode"),
            ("gate_source_logit_bias", "gate_source_logit_bias_mode"),
        ]:
            value = getattr(self.fusion_module, attr_name, None)
            if (
                isinstance(value, torch.nn.Parameter)
                and str(getattr(self.fusion_module, mode_name, "fixed")) == "learnable"
            ):
                value.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_threshold_shift_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
            self.fusion_module.reliability_bias_gate_threshold_shift_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for attr_name, mode_name in [
            ("gate_residual_alpha", "gate_residual_alpha_mode"),
            ("gate_logit_offset", "gate_logit_offset_mode"),
            ("adapter_gate_alpha", "adapter_gate_alpha_mode"),
            ("gate_output_scale", "gate_output_scale_mode"),
            ("gate_threshold_shift_alpha", "gate_threshold_shift_alpha_mode"),
            ("gate_source_logit_bias", "gate_source_logit_bias_mode"),
        ]:
            value = getattr(self.fusion_module, attr_name, None)
            if (
                isinstance(value, torch.nn.Parameter)
                and str(getattr(self.fusion_module, mode_name, "fixed")) == "learnable"
            ):
                value.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_cancel_contrast_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
            self.fusion_module.reliability_bias_gate_common_cancel_head,
            self.fusion_module.reliability_bias_gate_contrast_shift_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for attr_name, mode_name in [
            ("gate_residual_alpha", "gate_residual_alpha_mode"),
            ("gate_logit_offset", "gate_logit_offset_mode"),
            ("adapter_gate_alpha", "adapter_gate_alpha_mode"),
            ("gate_common_cancel_alpha", "gate_common_cancel_alpha_mode"),
            ("gate_contrast_shift_alpha", "gate_contrast_shift_alpha_mode"),
        ]:
            value = getattr(self.fusion_module, attr_name, None)
            if (
                isinstance(value, torch.nn.Parameter)
                and str(getattr(self.fusion_module, mode_name, "fixed")) == "learnable"
            ):
                value.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_cancel_rescue_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in [
            self.fusion_module.reliability_bias_gate_residual_head,
            self.fusion_module.reliability_bias_gate_adapter_input_adapter,
            self.fusion_module.reliability_bias_gate_adapter_coupler,
            self.fusion_module.reliability_bias_gate_common_cancel_head,
            self.fusion_module.reliability_bias_gate_mild_evidence_input_adapter,
            self.fusion_module.reliability_bias_gate_mild_rescue_gate_head,
            self.fusion_module.reliability_bias_gate_mild_value_input_adapter,
            self.fusion_module.reliability_bias_gate_mild_rescue_value_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for attr_name, mode_name in [
            ("gate_residual_alpha", "gate_residual_alpha_mode"),
            ("gate_logit_offset", "gate_logit_offset_mode"),
            ("adapter_gate_alpha", "adapter_gate_alpha_mode"),
            ("gate_common_cancel_alpha", "gate_common_cancel_alpha_mode"),
            ("gate_mild_rescue_alpha", "gate_mild_rescue_alpha_mode"),
        ]:
            value = getattr(self.fusion_module, attr_name, None)
            if (
                isinstance(value, torch.nn.Parameter)
                and str(getattr(self.fusion_module, mode_name, "fixed")) == "learnable"
            ):
                value.requires_grad = True
        gate_input_adapter = getattr(self.fusion_module, "reliability_bias_gate_input_adapter", None)
        if train_gate_input_adapter and gate_input_adapter is not None:
            for parameter in gate_input_adapter.parameters():
                parameter.requires_grad = True

    def freeze_attention_source_gate_excess_cancel_selective_rescue_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        self.freeze_attention_source_gate_cancel_rescue_micro_unfreeze(
            train_gate_input_adapter=train_gate_input_adapter
        )

    def freeze_attention_source_gate_anchor_cancel_selector_rescue_micro_unfreeze(
        self,
        train_gate_input_adapter: bool = True,
    ) -> None:
        self.freeze_attention_source_gate_cancel_rescue_micro_unfreeze(
            train_gate_input_adapter=train_gate_input_adapter
        )

    def forward_reliability(
        self,
        obs_dict: dict[str, torch.Tensor],
        gate_feature_context: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        _fused, aux_outputs = self.encode_and_fuse(obs_dict, gate_feature_context=gate_feature_context)
        return {
            "reliability_scores": aux_outputs["reliability_scores"],
            "reliability_logits": aux_outputs["reliability_logits"],
            "reliability_inputs": aux_outputs["reliability_inputs"],
            "encoder_features": aux_outputs["encoder_features"],
            "attention_weights": aux_outputs["attention_weights"],
            "attention_scores": aux_outputs["attention_scores"],
            "attention_base_scores": aux_outputs["attention_base_scores"],
            "attention_relative_gap_scores": aux_outputs["attention_relative_gap_scores"],
            "attention_relative_scores_before_bias": aux_outputs["attention_relative_scores_before_bias"],
            "attention_bias_scores": aux_outputs["attention_bias_scores"],
            "attention_source_bias_gates": aux_outputs["attention_source_bias_gates"],
            "attention_bias_gate_residuals": aux_outputs["attention_bias_gate_residuals"],
            "attention_bias_gate_adapter_logits": aux_outputs["attention_bias_gate_adapter_logits"],
            "attention_bias_gate_adapter_hidden": aux_outputs["attention_bias_gate_adapter_hidden"],
            "attention_bias_gate_adapter_hidden_norms": aux_outputs["attention_bias_gate_adapter_hidden_norms"],
            "attention_bias_gate_delta_logits": aux_outputs["attention_bias_gate_delta_logits"],
            "attention_bias_gate_scaled_delta_logits": aux_outputs["attention_bias_gate_scaled_delta_logits"],
            "attention_bias_gate_post_shift_excess": aux_outputs["attention_bias_gate_post_shift_excess"],
            "attention_bias_gate_common_cancel_gates": aux_outputs["attention_bias_gate_common_cancel_gates"],
            "attention_bias_gate_common_cancel_logits": aux_outputs["attention_bias_gate_common_cancel_logits"],
            "attention_bias_gate_post_cancel_logits": aux_outputs["attention_bias_gate_post_cancel_logits"],
            "attention_bias_gate_mild_rescue_gates": aux_outputs["attention_bias_gate_mild_rescue_gates"],
            "attention_bias_gate_mild_rescue_values": aux_outputs["attention_bias_gate_mild_rescue_values"],
            "attention_bias_gate_mild_rescue_logits": aux_outputs["attention_bias_gate_mild_rescue_logits"],
            "attention_bias_gates": aux_outputs["attention_bias_gates"],
            "attention_gate_reconstruction_error": aux_outputs["attention_gate_reconstruction_error"],
            "attention_residual_alpha": aux_outputs["attention_residual_alpha"],
            "attention_adapter_gate_alpha": aux_outputs["attention_adapter_gate_alpha"],
            "attention_gate_common_cancel_alpha": aux_outputs["attention_gate_common_cancel_alpha"],
            "attention_gate_mild_rescue_alpha": aux_outputs["attention_gate_mild_rescue_alpha"],
            "attention_gate_cancel_temperature": aux_outputs["attention_gate_cancel_temperature"],
            "attention_gate_selector_temperature": aux_outputs["attention_gate_selector_temperature"],
            "attention_gate_rescue_selector_floor": aux_outputs["attention_gate_rescue_selector_floor"],
            "attention_gate_mild_gate_min": aux_outputs["attention_gate_mild_gate_min"],
            "attention_gate_logit_offset": aux_outputs["attention_gate_logit_offset"],
            "attention_gate_output_scale": aux_outputs["attention_gate_output_scale"],
            "attention_final_scores": aux_outputs["attention_final_scores"],
            "attention_score_features": aux_outputs["attention_score_features"],
            "attention_score_adapter_delta": aux_outputs["attention_score_adapter_delta"],
            "attention_score_adapter_delta_norms": aux_outputs["attention_score_adapter_delta_norms"],
            "attention_relative_gap_norm": aux_outputs["attention_relative_gap_norm"],
            "attention_bias_norm": aux_outputs["attention_bias_norm"],
            "attention_bias_gate_norm": aux_outputs["attention_bias_gate_norm"],
            "fused_feature": aux_outputs["fused_feature"],
        }

    def encode_and_fuse(
        self,
        obs_dict: dict[str, torch.Tensor],
        gate_feature_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = {
            name: self.feature_norms[name](self.encoders[name](obs_dict[name]))
            for name in self.modality_names
        }
        stacked = torch.stack([encoded[name] for name in self.modality_names], dim=1)
        self.fusion_module.relative_gap_mode = getattr(self.fusion_module, "relative_gap_mode", "vision_anchor")
        if bool(getattr(self.fusion_module, "use_reliability_bias_head", False)):
            (
                provisional_scores,
                score_features,
                adapter_delta,
                relative_gap_scores,
                provisional_relative_scores_before_bias,
                _bias_scores,
                _gate_scores,
                _source_gate_scores,
                _gate_residuals,
                _gate_residual_logits,
                _gate_adapter_logits,
                _gate_delta_logits,
                _gate_scaled_delta_logits,
                _gate_threshold_shift_logits,
                _gate_post_shift_logits,
                _gate_post_shift_excess,
                _gate_common_cancel_gates,
                _gate_common_cancel_logits,
                _gate_post_cancel_logits,
                _gate_contrast_shift_logits,
                _gate_mild_rescue_gates,
                _gate_mild_rescue_selector_actives,
                _gate_mild_rescue_values,
                _gate_mild_rescue_logits,
            ) = self.fusion_module.compute_scores(stacked)
            provisional_scores = provisional_scores / max(self.attention_temperature, 1e-6)
            provisional_weights = torch.softmax(provisional_scores, dim=-1)
            provisional_weighted_sum = torch.sum(stacked * provisional_weights.unsqueeze(-1), dim=1)
            provisional_fused = (
                torch.cat([provisional_weighted_sum, stacked.mean(dim=1)], dim=-1)
                if self.append_mean_context
                else provisional_weighted_sum
            )
            provisional_reliability_inputs = self._build_reliability_inputs(
                stacked=stacked,
                fused=provisional_fused,
                obs_dict=obs_dict,
            )
            provisional_reliability_logits = self.reliability_head(provisional_reliability_inputs).squeeze(-1)
            provisional_base_reliability_logits = provisional_reliability_logits
            provisional_base_reliability_scores = torch.sigmoid(provisional_base_reliability_logits)
            if self.reliability_local_residual_apply_to_provisional:
                (
                    provisional_reliability_logits,
                    provisional_reliability_residuals,
                ) = self._apply_local_reliability_residual(
                    reliability_logits=provisional_reliability_logits,
                    obs_dict=obs_dict,
                    fused=provisional_fused,
                    base_reliability_scores=provisional_base_reliability_scores,
                    base_reliability_logits=provisional_base_reliability_logits,
                )
            else:
                provisional_reliability_residuals = torch.zeros_like(provisional_reliability_logits)
            provisional_reliability_scores = torch.sigmoid(provisional_reliability_logits)

            bias_inputs = torch.cat(
                [
                    score_features,
                    provisional_reliability_scores.unsqueeze(-1),
                    provisional_reliability_logits.unsqueeze(-1),
                ],
                dim=-1,
            )
            gate_input_mode = str(getattr(self.fusion_module, "gate_input_mode", "legacy"))
            gate_feature_stopgrad_on_bias = bool(getattr(self.fusion_module, "gate_feature_stopgrad_on_bias", False))
            gate_feature_stopgrad_on_source_gate = bool(
                getattr(self.fusion_module, "gate_feature_stopgrad_on_source_gate", False)
            )
            gate_feature_stopgrad_on_adapter_delta = bool(
                getattr(self.fusion_module, "gate_feature_stopgrad_on_adapter_delta", False)
            )
            bias_for_gate = _bias_scores.detach() if gate_feature_stopgrad_on_bias else _bias_scores
            source_gate_scores = None
            gate_feature_slices: dict[str, tuple[int, int]] = {}
            gate_feature_cursor = 0
            adapter_gate_inputs = None

            def _append_gate_feature(
                bucket: list[torch.Tensor],
                name: str,
                value: torch.Tensor,
            ) -> None:
                nonlocal gate_feature_cursor
                width = int(value.shape[-1])
                bucket.append(value)
                gate_feature_slices[name] = (gate_feature_cursor, gate_feature_cursor + width)
                gate_feature_cursor += width

            if gate_feature_context is not None:
                source_gate_scores = gate_feature_context.get("source_gate_scores")
            if gate_input_mode == "legacy":
                legacy_features = [
                    provisional_reliability_scores.unsqueeze(-1),
                    provisional_reliability_logits.unsqueeze(-1),
                ]
                gate_head = getattr(self.fusion_module, "reliability_bias_gate_head", None)
                first_layer = gate_head[0] if gate_head is not None and len(gate_head) > 0 else None
                legacy_input_width = 4
                if first_layer is not None:
                    weight = getattr(first_layer, "weight", None)
                    try:
                        legacy_input_width = int(weight.shape[-1])
                    except Exception:
                        legacy_input_width = int(getattr(first_layer, "in_features", 4) or 4)
                if legacy_input_width >= 4:
                    legacy_features.extend(
                        [
                            bias_for_gate.unsqueeze(-1),
                            provisional_relative_scores_before_bias.unsqueeze(-1),
                        ]
                    )
                gate_inputs = torch.cat(legacy_features, dim=-1)
                gate_feature_slices = {}
            elif gate_input_mode == "bias_aware_v1":
                gate_features: list[torch.Tensor] = []
                _append_gate_feature(gate_features, "score_features", score_features)
                _append_gate_feature(gate_features, "bias_scores", bias_for_gate.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_abs_bias_feature", False)):
                    _append_gate_feature(gate_features, "abs_bias_scores", bias_for_gate.abs().unsqueeze(-1))
                _append_gate_feature(gate_features, "reliability_scores", provisional_reliability_scores.unsqueeze(-1))
                _append_gate_feature(gate_features, "reliability_logits", provisional_reliability_logits.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_score_delta_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "score_before_bias",
                        provisional_relative_scores_before_bias.unsqueeze(-1),
                    )
                if bool(getattr(self.fusion_module, "gate_use_teacher_delta_feature", False)):
                    if gate_feature_context is None:
                        clean_teacher_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                        sign_teacher_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                    else:
                        clean_teacher_scores = gate_feature_context.get("clean_teacher_scores")
                        sign_teacher_scores = gate_feature_context.get("sign_teacher_scores")
                        if clean_teacher_scores is None or sign_teacher_scores is None:
                            raise ValueError(
                                "bias_aware_v1 with gate_use_teacher_delta_feature requires clean_teacher_scores and sign_teacher_scores"
                            )
                    _append_gate_feature(
                        gate_features,
                        "clean_teacher_delta",
                        (provisional_relative_scores_before_bias - clean_teacher_scores).unsqueeze(-1),
                    )
                    _append_gate_feature(
                        gate_features,
                        "sign_teacher_delta",
                        (provisional_relative_scores_before_bias - sign_teacher_scores).unsqueeze(-1),
                    )
                gate_inputs = torch.cat(gate_features, dim=-1)
            elif gate_input_mode == "source_residual_bias_aware_v1":
                if gate_feature_context is None:
                    clean_teacher_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                    sign_teacher_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                    source_gate_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                else:
                    clean_teacher_scores = gate_feature_context.get("clean_teacher_scores")
                    sign_teacher_scores = gate_feature_context.get("sign_teacher_scores")
                    source_gate_scores = gate_feature_context.get("source_gate_scores")
                    if clean_teacher_scores is None or sign_teacher_scores is None or source_gate_scores is None:
                        raise ValueError(
                            "source_residual_bias_aware_v1 requires clean_teacher_scores, sign_teacher_scores, and source_gate_scores"
                        )
                source_gate_for_features = (
                    source_gate_scores.detach() if gate_feature_stopgrad_on_source_gate else source_gate_scores
                )
                gate_features = []
                _append_gate_feature(gate_features, "score_features", score_features)
                _append_gate_feature(gate_features, "bias_scores", bias_for_gate.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_abs_bias_feature", False)):
                    _append_gate_feature(gate_features, "abs_bias_scores", bias_for_gate.abs().unsqueeze(-1))
                _append_gate_feature(gate_features, "reliability_scores", provisional_reliability_scores.unsqueeze(-1))
                _append_gate_feature(gate_features, "reliability_logits", provisional_reliability_logits.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_score_delta_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "score_before_bias",
                        provisional_relative_scores_before_bias.unsqueeze(-1),
                    )
                if bool(getattr(self.fusion_module, "gate_use_source_gate_feature", False)):
                    _append_gate_feature(gate_features, "source_gate_scores", source_gate_for_features.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_teacher_delta_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "clean_teacher_delta",
                        (provisional_relative_scores_before_bias - clean_teacher_scores).unsqueeze(-1),
                    )
                    _append_gate_feature(
                        gate_features,
                        "sign_teacher_delta",
                        (provisional_relative_scores_before_bias - sign_teacher_scores).unsqueeze(-1),
                    )
                gate_inputs = torch.cat(gate_features, dim=-1)
            elif gate_input_mode == "source_residual_adapter_delta_aware_v2":
                if gate_feature_context is None:
                    clean_teacher_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                    sign_teacher_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                    source_gate_scores = torch.zeros_like(provisional_relative_scores_before_bias)
                    clean_teacher_adapter_delta = torch.zeros_like(adapter_delta)
                    sign_teacher_adapter_delta = torch.zeros_like(adapter_delta)
                else:
                    clean_teacher_scores = gate_feature_context.get("clean_teacher_scores")
                    sign_teacher_scores = gate_feature_context.get("sign_teacher_scores")
                    source_gate_scores = gate_feature_context.get("source_gate_scores")
                    clean_teacher_adapter_delta = gate_feature_context.get("clean_teacher_adapter_delta")
                    sign_teacher_adapter_delta = gate_feature_context.get("sign_teacher_adapter_delta")
                    if (
                        clean_teacher_scores is None
                        or sign_teacher_scores is None
                        or source_gate_scores is None
                        or clean_teacher_adapter_delta is None
                        or sign_teacher_adapter_delta is None
                    ):
                        raise ValueError(
                            "source_residual_adapter_delta_aware_v2 requires clean/sign teacher scores, source gate scores, and clean/sign teacher adapter deltas"
                        )
                source_gate_for_features = (
                    source_gate_scores.detach() if gate_feature_stopgrad_on_source_gate else source_gate_scores
                )
                adapter_delta_for_features = adapter_delta.detach() if gate_feature_stopgrad_on_adapter_delta else adapter_delta
                gate_features = []
                _append_gate_feature(gate_features, "score_features", score_features)
                _append_gate_feature(gate_features, "bias_scores", bias_for_gate.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_abs_bias_feature", False)):
                    _append_gate_feature(gate_features, "abs_bias_scores", bias_for_gate.abs().unsqueeze(-1))
                _append_gate_feature(gate_features, "reliability_scores", provisional_reliability_scores.unsqueeze(-1))
                _append_gate_feature(gate_features, "reliability_logits", provisional_reliability_logits.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_score_delta_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "score_before_bias",
                        provisional_relative_scores_before_bias.unsqueeze(-1),
                    )
                if bool(getattr(self.fusion_module, "gate_use_source_gate_feature", False)):
                    _append_gate_feature(gate_features, "source_gate_scores", source_gate_for_features.unsqueeze(-1))
                if bool(getattr(self.fusion_module, "gate_use_teacher_delta_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "clean_teacher_delta",
                        (provisional_relative_scores_before_bias - clean_teacher_scores).unsqueeze(-1),
                    )
                    _append_gate_feature(
                        gate_features,
                        "sign_teacher_delta",
                        (provisional_relative_scores_before_bias - sign_teacher_scores).unsqueeze(-1),
                    )
                if bool(getattr(self.fusion_module, "gate_use_adapter_delta_feature", False)):
                    _append_gate_feature(gate_features, "adapter_delta", adapter_delta_for_features)
                if bool(getattr(self.fusion_module, "gate_use_adapter_norm_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "adapter_delta_norm",
                        torch.norm(adapter_delta_for_features, dim=-1, keepdim=True),
                    )
                if bool(getattr(self.fusion_module, "gate_use_adapter_teacher_delta_feature", False)):
                    _append_gate_feature(
                        gate_features,
                        "clean_teacher_adapter_delta",
                        adapter_delta_for_features - clean_teacher_adapter_delta,
                    )
                    _append_gate_feature(
                        gate_features,
                        "sign_teacher_adapter_delta",
                        adapter_delta_for_features - sign_teacher_adapter_delta,
                    )
                gate_inputs = torch.cat(gate_features, dim=-1)
                adapter_gate_features = [adapter_delta_for_features]
                if str(getattr(self.fusion_module, "adapter_gate_input_mode", "legacy")) == "dedicated_v1":
                    adapter_gate_features.append(adapter_delta_for_features.abs())
                if bool(getattr(self.fusion_module, "gate_use_adapter_norm_feature", False)):
                    adapter_gate_features.append(torch.norm(adapter_delta_for_features, dim=-1, keepdim=True))
                if bool(getattr(self.fusion_module, "gate_use_adapter_teacher_delta_feature", False)):
                    adapter_gate_features.append(adapter_delta_for_features - clean_teacher_adapter_delta)
                    adapter_gate_features.append(adapter_delta_for_features - sign_teacher_adapter_delta)
                if bool(getattr(self.fusion_module, "gate_use_source_gate_feature", False)):
                    adapter_gate_features.append(source_gate_for_features.unsqueeze(-1))
                adapter_gate_inputs_raw = torch.cat(adapter_gate_features, dim=-1)
                if str(getattr(self.fusion_module, "adapter_gate_input_mode", "legacy")) == "dedicated_v1":
                    adapter_gate_inputs = self.fusion_module.reliability_bias_gate_adapter_input_adapter(adapter_gate_inputs_raw)
                else:
                    adapter_gate_inputs = adapter_gate_inputs_raw
                adapter_gate_hidden = adapter_gate_inputs
                mild_evidence_inputs_raw = torch.cat(
                    [
                        adapter_delta_for_features - clean_teacher_adapter_delta,
                        adapter_delta_for_features - sign_teacher_adapter_delta,
                        (adapter_delta_for_features - clean_teacher_adapter_delta).abs(),
                        (adapter_delta_for_features - sign_teacher_adapter_delta).abs(),
                        (adapter_delta_for_features - clean_teacher_adapter_delta).abs()
                        - (adapter_delta_for_features - sign_teacher_adapter_delta).abs(),
                    ],
                    dim=-1,
                )
            else:
                raise ValueError(f"Unsupported gate_input_mode: {gate_input_mode}")
            (
                scores,
                score_features,
                adapter_delta,
                relative_gap_scores,
                relative_scores_before_bias,
                bias_scores,
                gate_scores,
                source_gate_outputs,
                gate_residuals,
                gate_residual_logits,
                gate_adapter_logits,
                gate_delta_logits,
                gate_scaled_delta_logits,
                gate_threshold_shift_logits,
                gate_post_shift_logits,
                gate_post_shift_excess,
                gate_common_cancel_gates,
                gate_common_cancel_logits,
                gate_post_cancel_logits,
                gate_contrast_shift_logits,
                gate_mild_rescue_gates,
                gate_mild_rescue_selector_actives,
                gate_mild_rescue_values,
                gate_mild_rescue_logits,
            ) = self.fusion_module.compute_scores(
                stacked,
                bias_inputs=bias_inputs,
                gate_inputs=gate_inputs,
                adapter_gate_inputs=adapter_gate_inputs,
                source_gate_scores=source_gate_scores,
                mild_rescue_inputs=mild_evidence_inputs_raw if "mild_evidence_inputs_raw" in locals() else None,
            )
        else:
            (
                scores,
                score_features,
                adapter_delta,
                relative_gap_scores,
                relative_scores_before_bias,
                bias_scores,
                gate_scores,
                source_gate_outputs,
                gate_residuals,
                gate_residual_logits,
                gate_adapter_logits,
                gate_delta_logits,
                gate_scaled_delta_logits,
                gate_threshold_shift_logits,
                gate_post_shift_logits,
                gate_post_shift_excess,
                gate_common_cancel_gates,
                gate_common_cancel_logits,
                gate_post_cancel_logits,
                gate_contrast_shift_logits,
                gate_mild_rescue_gates,
                gate_mild_rescue_selector_actives,
                gate_mild_rescue_values,
                gate_mild_rescue_logits,
            ) = self.fusion_module.compute_scores(stacked)
        gate_mode = str(getattr(self.fusion_module, "gate_mode", "absolute_v1"))
        if gate_mode == "source_residual_v1":
            if not torch.isfinite(source_gate_outputs).all():
                raise ValueError("source_residual_v1 requires finite attention_source_bias_gates")
            reconstructed_gate_scores = torch.clamp(source_gate_outputs + gate_residuals, 0.0, 1.0)
            gate_reconstruction_error = torch.mean(torch.abs(gate_scores - reconstructed_gate_scores), dim=-1)
            if torch.any(gate_reconstruction_error > 1e-6):
                raise ValueError("source_residual_v1 gate_reconstruction_error exceeded tolerance")
        elif gate_mode == "source_residual_logit_v2":
            if not torch.isfinite(source_gate_outputs).all():
                raise ValueError("source_residual_logit_v2 requires finite attention_source_bias_gates")
            source_gate_logits = torch.logit(source_gate_outputs.clamp(1.0e-4, 1.0 - 1.0e-4))
            residual_alpha = self.fusion_module.get_gate_residual_alpha_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            gate_logit_offset = self.fusion_module.get_gate_logit_offset_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            adapter_gate_alpha = self.fusion_module.get_adapter_gate_alpha_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            gate_output_scale = self.fusion_module.get_gate_output_scale_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            source_logit_bias = self.fusion_module.get_gate_source_logit_bias_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            reconstructed_gate_scores = torch.sigmoid(
                source_gate_logits
                + source_logit_bias
                + gate_output_scale * (gate_logit_offset + residual_alpha * gate_residuals + adapter_gate_alpha * gate_adapter_logits)
                + gate_threshold_shift_logits
                + gate_common_cancel_logits
                + gate_contrast_shift_logits
                + gate_mild_rescue_logits
            )
            gate_reconstruction_error = torch.mean(torch.abs(gate_scores - reconstructed_gate_scores), dim=-1)
            if torch.any(gate_reconstruction_error > 1e-6):
                raise ValueError("source_residual_logit_v2 gate_reconstruction_error exceeded tolerance")
        else:
            gate_reconstruction_error = torch.zeros_like(gate_scores[:, 0])
        scores = scores / max(self.attention_temperature, 1e-6)
        weights = torch.softmax(scores, dim=-1)
        weighted_sum = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        fused = (
            torch.cat([weighted_sum, stacked.mean(dim=1)], dim=-1)
            if self.append_mean_context
            else weighted_sum
        )
        reliability_inputs = self._build_reliability_inputs(
            stacked=stacked,
            fused=fused,
            obs_dict=obs_dict,
        )
        reliability_logits = self.reliability_head(reliability_inputs).squeeze(-1)
        base_reliability_logits = reliability_logits
        base_reliability_scores = torch.sigmoid(base_reliability_logits)
        reliability_logits, reliability_residuals = self._apply_local_reliability_residual(
            reliability_logits=reliability_logits,
            obs_dict=obs_dict,
            fused=fused,
            base_reliability_scores=base_reliability_scores,
            base_reliability_logits=base_reliability_logits,
        )
        reliability_scores = torch.sigmoid(reliability_logits)
        # This matches the additive contribution to `attention_final_scores`,
        # which are temperature-scaled before the softmax.
        gate_effect_scale = torch.as_tensor(
            float(getattr(self.fusion_module, "reliability_bias_residual_scale", 0.0))
            / max(self.attention_temperature, 1e-6),
            device=gate_scores.device,
            dtype=gate_scores.dtype,
        )
        attention_bias_gate_effects = gate_effect_scale * gate_scores * bias_scores
        aux_outputs = {
            "attention_weights": weights,
            "attention_scores": scores,
            "attention_base_scores": self.fusion_module.base_score_head(score_features).squeeze(-1),
            "attention_relative_gap_scores": relative_gap_scores,
            "attention_relative_scores_before_bias": relative_scores_before_bias,
            "attention_bias_scores": bias_scores,
            "attention_bias_gate_effects": attention_bias_gate_effects,
            "attention_source_bias_gates": source_gate_outputs,
            "attention_bias_gate_residuals": gate_residuals,
            "attention_bias_gate_residual_logits": gate_residual_logits,
            "attention_bias_gate_adapter_logits": gate_adapter_logits,
            "attention_bias_gate_adapter_hidden": adapter_gate_hidden if "adapter_gate_hidden" in locals() else torch.zeros_like(score_features),
            "attention_bias_gate_adapter_hidden_norms": torch.norm(
                adapter_gate_hidden if "adapter_gate_hidden" in locals() else torch.zeros_like(score_features),
                dim=-1,
            ),
            "attention_bias_gate_delta_logits": gate_delta_logits,
            "attention_bias_gate_scaled_delta_logits": gate_scaled_delta_logits,
            "attention_bias_gate_threshold_shift_logits": gate_threshold_shift_logits,
            "attention_bias_gate_post_shift_logits": gate_post_shift_logits,
            "attention_bias_gate_post_shift_excess": gate_post_shift_excess,
            "attention_bias_gate_common_cancel_gates": gate_common_cancel_gates,
            "attention_bias_gate_common_cancel_probs": gate_common_cancel_gates,
            "attention_bias_gate_common_cancel_logits": gate_common_cancel_logits,
            "attention_bias_gate_post_cancel_logits": gate_post_cancel_logits,
            "attention_bias_gate_contrast_shift_logits": gate_contrast_shift_logits,
            "attention_bias_gate_mild_rescue_gates": gate_mild_rescue_gates,
            "attention_bias_gate_selector_probs": gate_mild_rescue_gates,
            "attention_bias_gate_selector_actives": gate_mild_rescue_selector_actives,
            "attention_bias_gate_mild_rescue_values": gate_mild_rescue_values,
            "attention_bias_gate_mild_rescue_logits": gate_mild_rescue_logits,
            "attention_bias_gates": gate_scores,
            "attention_gate_reconstruction_error": gate_reconstruction_error,
            "attention_residual_alpha": self.fusion_module.get_gate_residual_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_residual_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_residual_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_adapter_gate_alpha": self.fusion_module.get_adapter_gate_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_adapter_gate_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_adapter_gate_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_logit_offset": self.fusion_module.get_gate_logit_offset_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_logit_offset_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_logit_offset_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_output_scale": self.fusion_module.get_gate_output_scale_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_output_scale_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_output_scale_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_threshold_shift_alpha": self.fusion_module.get_gate_threshold_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_threshold_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_threshold_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_source_logit_bias": self.fusion_module.get_gate_source_logit_bias_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_source_logit_bias_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_source_logit_bias_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_common_cancel_alpha": self.fusion_module.get_gate_common_cancel_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_common_cancel_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_common_cancel_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_contrast_shift_alpha": self.fusion_module.get_gate_contrast_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_contrast_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_contrast_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_mild_rescue_alpha": self.fusion_module.get_gate_mild_rescue_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.fusion_module.get_gate_mild_rescue_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).ndim > 0
            else self.fusion_module.get_gate_mild_rescue_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ).expand(gate_scores.shape[0]),
            "attention_gate_cancel_temperature": torch.full(
                (gate_scores.shape[0],),
                float(getattr(self.fusion_module, "cancel_temperature", 1.0)),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_selector_temperature": torch.full(
                (gate_scores.shape[0],),
                float(getattr(self.fusion_module, "selector_temperature", 1.0)),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_rescue_selector_floor": torch.full(
                (gate_scores.shape[0],),
                min(max(float(getattr(self.fusion_module, "rescue_selector_floor", 0.0)), 0.0), 0.999),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_mild_gate_min": torch.full(
                (gate_scores.shape[0],),
                min(max(float(getattr(self.fusion_module, "mild_gate_min", 0.08)), 1.0e-4), 1.0 - 1.0e-4),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_final_scores": scores,
            "attention_score_features": score_features,
            "attention_score_adapter_delta": adapter_delta,
            "attention_bias_gate_inputs": gate_inputs if "gate_inputs" in locals() else torch.zeros_like(score_features[..., :1]),
            "attention_bias_gate_input_slices": gate_feature_slices if "gate_feature_slices" in locals() else {},
            "attention_score_adapter_delta_norms": torch.norm(adapter_delta, dim=-1),
            "attention_relative_gap_norm": torch.norm(relative_gap_scores, dim=-1),
            "attention_bias_norm": torch.norm(bias_scores, dim=-1),
            "attention_bias_gate_norm": torch.norm(gate_scores, dim=-1),
            "attention_bias_gate_effect_norm": torch.norm(attention_bias_gate_effects, dim=-1),
            "attention_relative_gap_mode": str(getattr(self.fusion_module, "relative_gap_mode", "vision_anchor")),
            "reliability_scores": reliability_scores,
            "reliability_logits": reliability_logits,
            "reliability_local_residual_deltas": reliability_residuals,
            "provisional_reliability_local_residual_deltas": (
                provisional_reliability_residuals
                if "provisional_reliability_residuals" in locals()
                else torch.zeros_like(reliability_logits)
            ),
            "reliability_ego_residual_deltas": reliability_residuals,
            "provisional_reliability_ego_residual_deltas": (
                provisional_reliability_residuals
                if "provisional_reliability_residuals" in locals()
                else torch.zeros_like(reliability_logits)
            ),
            "ordered_modalities": list(MODALITY_NAMES),
            "encoder_features": stacked,
            "reliability_inputs": reliability_inputs,
            "fused_feature": fused,
            "encoder_feature_norms": torch.stack(
                [torch.norm(encoded[name], dim=-1) for name in self.modality_names],
                dim=-1,
            ),
            "fusion_append_mean_context": torch.full(
                (stacked.shape[0],),
                float(self.append_mean_context),
                device=stacked.device,
                dtype=stacked.dtype,
            ),
        }
        return fused, aux_outputs

    def forward_sequence(
        self,
        obs_seq: dict[str, torch.Tensor],
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
        gate_feature_context: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = episode_starts.shape
        self.validate_hidden_state(initial_hidden, batch_size)
        hidden = initial_hidden

        action_list = []
        log_prob_list = []
        entropy_list = []
        value_list = []
        hidden_norm_list = []
        attention_weight_list = []
        attention_score_list = []
        attention_final_score_list = []
        attention_relative_gap_list = []
        attention_relative_scores_before_bias_list = []
        attention_score_feature_list = []
        attention_score_adapter_delta_list = []
        attention_bias_list = []
        attention_bias_gate_effect_list = []
        attention_bias_gate_list = []
        attention_source_bias_gate_list = []
        attention_bias_gate_residual_list = []
        attention_bias_gate_residual_logit_list = []
        attention_bias_gate_adapter_logit_list = []
        attention_bias_gate_adapter_hidden_list = []
        attention_bias_gate_adapter_hidden_norm_list = []
        attention_gate_reconstruction_error_list = []
        attention_bias_norm_list = []
        attention_residual_alpha_list = []
        attention_adapter_gate_alpha_list = []
        attention_gate_logit_offset_list = []
        attention_gate_output_scale_list = []
        attention_gate_threshold_shift_alpha_list = []
        attention_gate_source_logit_bias_list = []
        attention_gate_common_cancel_alpha_list = []
        attention_gate_contrast_shift_alpha_list = []
        attention_gate_mild_rescue_alpha_list = []
        attention_gate_cancel_temperature_list = []
        attention_gate_selector_temperature_list = []
        attention_gate_rescue_selector_floor_list = []
        attention_gate_mild_gate_min_list = []
        attention_bias_gate_delta_logit_list = []
        attention_bias_gate_scaled_delta_logit_list = []
        attention_bias_gate_threshold_shift_logit_list = []
        attention_bias_gate_post_shift_logit_list = []
        attention_bias_gate_post_shift_excess_list = []
        attention_bias_gate_common_cancel_gate_list = []
        attention_bias_gate_common_cancel_logit_list = []
        attention_bias_gate_post_cancel_logit_list = []
        attention_bias_gate_contrast_shift_logit_list = []
        attention_bias_gate_mild_rescue_gate_list = []
        attention_bias_gate_selector_active_list = []
        attention_bias_gate_mild_rescue_value_list = []
        attention_bias_gate_mild_rescue_logit_list = []
        reliability_score_list = []
        reliability_logit_list = []
        reliability_local_residual_delta_list = []
        provisional_reliability_local_residual_delta_list = []
        reliability_ego_residual_delta_list = []
        provisional_reliability_ego_residual_delta_list = []
        action_mean_list = []
        action_std_list = []

        for timestep in range(seq_len):
            step_obs = {key: obs_seq[key][:, timestep] for key in self.modality_names}
            step_gate_context = None
            if gate_feature_context is not None:
                step_gate_context = {
                    key: value[:, timestep] if value.dim() >= 3 else value
                    for key, value in gate_feature_context.items()
                }
            step_output = self.step(
                obs_dict=step_obs,
                hidden_state=hidden,
                episode_start=episode_starts[:, timestep],
                deterministic=deterministic,
                gate_feature_context=step_gate_context,
            )
            hidden = step_output.hidden_state

            if actions is None:
                action = step_output.action
                log_prob = step_output.log_prob
            else:
                action = actions[:, timestep]
                pre_tanh_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
                dist = torch.distributions.Normal(
                    step_output.aux_outputs["action_mean"],
                    step_output.aux_outputs["action_std"],
                )
                log_prob = self._tanh_log_prob(dist, pre_tanh_action)

            action_list.append(action)
            log_prob_list.append(log_prob)
            entropy_list.append(step_output.entropy)
            value_list.append(step_output.value)
            hidden_norm_list.append(step_output.aux_outputs["hidden_norm"])
            attention_weight_list.append(step_output.aux_outputs["attention_weights"])
            attention_score_list.append(step_output.aux_outputs["attention_scores"])
            attention_final_score_list.append(step_output.aux_outputs["attention_final_scores"])
            attention_relative_gap_list.append(step_output.aux_outputs["attention_relative_gap_scores"])
            attention_relative_scores_before_bias_list.append(
                step_output.aux_outputs["attention_relative_scores_before_bias"]
            )
            attention_score_feature_list.append(step_output.aux_outputs["attention_score_features"])
            attention_score_adapter_delta_list.append(step_output.aux_outputs["attention_score_adapter_delta"])
            attention_bias_list.append(step_output.aux_outputs["attention_bias_scores"])
            attention_bias_gate_effect_list.append(step_output.aux_outputs["attention_bias_gate_effects"])
            attention_source_bias_gate_list.append(step_output.aux_outputs["attention_source_bias_gates"])
            attention_bias_gate_residual_list.append(step_output.aux_outputs["attention_bias_gate_residuals"])
            attention_bias_gate_residual_logit_list.append(step_output.aux_outputs["attention_bias_gate_residual_logits"])
            attention_bias_gate_adapter_logit_list.append(step_output.aux_outputs["attention_bias_gate_adapter_logits"])
            attention_bias_gate_adapter_hidden_list.append(step_output.aux_outputs["attention_bias_gate_adapter_hidden"])
            attention_bias_gate_adapter_hidden_norm_list.append(step_output.aux_outputs["attention_bias_gate_adapter_hidden_norms"])
            attention_bias_gate_delta_logit_list.append(step_output.aux_outputs["attention_bias_gate_delta_logits"])
            attention_bias_gate_scaled_delta_logit_list.append(step_output.aux_outputs["attention_bias_gate_scaled_delta_logits"])
            attention_bias_gate_threshold_shift_logit_list.append(
                step_output.aux_outputs["attention_bias_gate_threshold_shift_logits"]
            )
            attention_bias_gate_post_shift_logit_list.append(
                step_output.aux_outputs["attention_bias_gate_post_shift_logits"]
            )
            attention_bias_gate_post_shift_excess_list.append(
                step_output.aux_outputs["attention_bias_gate_post_shift_excess"]
            )
            attention_bias_gate_common_cancel_gate_list.append(
                step_output.aux_outputs["attention_bias_gate_common_cancel_gates"]
            )
            attention_bias_gate_common_cancel_logit_list.append(
                step_output.aux_outputs["attention_bias_gate_common_cancel_logits"]
            )
            attention_bias_gate_post_cancel_logit_list.append(
                step_output.aux_outputs["attention_bias_gate_post_cancel_logits"]
            )
            attention_bias_gate_contrast_shift_logit_list.append(
                step_output.aux_outputs["attention_bias_gate_contrast_shift_logits"]
            )
            attention_bias_gate_mild_rescue_gate_list.append(
                step_output.aux_outputs["attention_bias_gate_mild_rescue_gates"]
            )
            attention_bias_gate_selector_active_list.append(
                step_output.aux_outputs["attention_bias_gate_selector_actives"]
            )
            attention_bias_gate_mild_rescue_value_list.append(
                step_output.aux_outputs["attention_bias_gate_mild_rescue_values"]
            )
            attention_bias_gate_mild_rescue_logit_list.append(
                step_output.aux_outputs["attention_bias_gate_mild_rescue_logits"]
            )
            attention_gate_reconstruction_error_list.append(step_output.aux_outputs["attention_gate_reconstruction_error"])
            attention_bias_gate_list.append(step_output.aux_outputs["attention_bias_gates"])
            attention_bias_norm_list.append(step_output.aux_outputs["attention_bias_norm"])
            attention_residual_alpha_list.append(step_output.aux_outputs["attention_residual_alpha"])
            attention_adapter_gate_alpha_list.append(step_output.aux_outputs["attention_adapter_gate_alpha"])
            attention_gate_logit_offset_list.append(step_output.aux_outputs["attention_gate_logit_offset"])
            attention_gate_output_scale_list.append(step_output.aux_outputs["attention_gate_output_scale"])
            attention_gate_threshold_shift_alpha_list.append(step_output.aux_outputs["attention_gate_threshold_shift_alpha"])
            attention_gate_source_logit_bias_list.append(step_output.aux_outputs["attention_gate_source_logit_bias"])
            attention_gate_common_cancel_alpha_list.append(step_output.aux_outputs["attention_gate_common_cancel_alpha"])
            attention_gate_contrast_shift_alpha_list.append(step_output.aux_outputs["attention_gate_contrast_shift_alpha"])
            attention_gate_mild_rescue_alpha_list.append(step_output.aux_outputs["attention_gate_mild_rescue_alpha"])
            attention_gate_cancel_temperature_list.append(step_output.aux_outputs["attention_gate_cancel_temperature"])
            attention_gate_selector_temperature_list.append(step_output.aux_outputs["attention_gate_selector_temperature"])
            attention_gate_rescue_selector_floor_list.append(step_output.aux_outputs["attention_gate_rescue_selector_floor"])
            attention_gate_mild_gate_min_list.append(step_output.aux_outputs["attention_gate_mild_gate_min"])
            reliability_score_list.append(step_output.aux_outputs["reliability_scores"])
            reliability_logit_list.append(step_output.aux_outputs["reliability_logits"])
            reliability_local_residual_delta_list.append(step_output.aux_outputs["reliability_local_residual_deltas"])
            provisional_reliability_local_residual_delta_list.append(
                step_output.aux_outputs["provisional_reliability_local_residual_deltas"]
            )
            reliability_ego_residual_delta_list.append(step_output.aux_outputs["reliability_ego_residual_deltas"])
            provisional_reliability_ego_residual_delta_list.append(
                step_output.aux_outputs["provisional_reliability_ego_residual_deltas"]
            )
            action_mean_list.append(step_output.aux_outputs["action_mean"])
            action_std_list.append(step_output.aux_outputs["action_std"])

        return {
            "actions": torch.stack(action_list, dim=1),
            "log_probs": torch.stack(log_prob_list, dim=1),
            "entropies": torch.stack(entropy_list, dim=1),
            "values": torch.stack(value_list, dim=1),
            "hidden_norms": torch.stack(hidden_norm_list, dim=1),
            "attention_weights": torch.stack(attention_weight_list, dim=1),
            "attention_scores": torch.stack(attention_score_list, dim=1),
            "attention_final_scores": torch.stack(attention_final_score_list, dim=1),
            "attention_relative_gap_scores": torch.stack(attention_relative_gap_list, dim=1),
            "attention_relative_scores_before_bias": torch.stack(
                attention_relative_scores_before_bias_list,
                dim=1,
            ),
            "attention_score_features": torch.stack(attention_score_feature_list, dim=1),
            "attention_score_adapter_delta": torch.stack(attention_score_adapter_delta_list, dim=1),
            "attention_bias_scores": torch.stack(attention_bias_list, dim=1),
            "attention_bias_gate_effects": torch.stack(attention_bias_gate_effect_list, dim=1),
            "attention_source_bias_gates": torch.stack(attention_source_bias_gate_list, dim=1),
            "attention_bias_gate_residuals": torch.stack(attention_bias_gate_residual_list, dim=1),
            "attention_bias_gate_residual_logits": torch.stack(attention_bias_gate_residual_logit_list, dim=1),
            "attention_bias_gate_adapter_logits": torch.stack(attention_bias_gate_adapter_logit_list, dim=1),
            "attention_bias_gate_adapter_hidden": torch.stack(attention_bias_gate_adapter_hidden_list, dim=1),
            "attention_bias_gate_adapter_hidden_norms": torch.stack(attention_bias_gate_adapter_hidden_norm_list, dim=1),
            "attention_bias_gate_delta_logits": torch.stack(attention_bias_gate_delta_logit_list, dim=1),
            "attention_bias_gate_scaled_delta_logits": torch.stack(attention_bias_gate_scaled_delta_logit_list, dim=1),
            "attention_bias_gate_threshold_shift_logits": torch.stack(attention_bias_gate_threshold_shift_logit_list, dim=1),
            "attention_bias_gate_post_shift_logits": torch.stack(attention_bias_gate_post_shift_logit_list, dim=1),
            "attention_bias_gate_post_shift_excess": torch.stack(attention_bias_gate_post_shift_excess_list, dim=1),
            "attention_bias_gate_common_cancel_gates": torch.stack(attention_bias_gate_common_cancel_gate_list, dim=1),
            "attention_bias_gate_common_cancel_logits": torch.stack(attention_bias_gate_common_cancel_logit_list, dim=1),
            "attention_bias_gate_post_cancel_logits": torch.stack(attention_bias_gate_post_cancel_logit_list, dim=1),
            "attention_bias_gate_contrast_shift_logits": torch.stack(attention_bias_gate_contrast_shift_logit_list, dim=1),
            "attention_bias_gate_mild_rescue_gates": torch.stack(attention_bias_gate_mild_rescue_gate_list, dim=1),
            "attention_bias_gate_selector_actives": torch.stack(attention_bias_gate_selector_active_list, dim=1),
            "attention_bias_gate_mild_rescue_values": torch.stack(attention_bias_gate_mild_rescue_value_list, dim=1),
            "attention_bias_gate_mild_rescue_logits": torch.stack(attention_bias_gate_mild_rescue_logit_list, dim=1),
            "attention_gate_reconstruction_error": torch.stack(attention_gate_reconstruction_error_list, dim=1),
            "attention_bias_gates": torch.stack(attention_bias_gate_list, dim=1),
            "attention_bias_norms": torch.stack(attention_bias_norm_list, dim=1),
            "attention_residual_alphas": torch.stack(attention_residual_alpha_list, dim=1),
            "attention_adapter_gate_alphas": torch.stack(attention_adapter_gate_alpha_list, dim=1),
            "attention_gate_logit_offsets": torch.stack(attention_gate_logit_offset_list, dim=1),
            "attention_gate_output_scales": torch.stack(attention_gate_output_scale_list, dim=1),
            "attention_gate_threshold_shift_alphas": torch.stack(attention_gate_threshold_shift_alpha_list, dim=1),
            "attention_gate_source_logit_biases": torch.stack(attention_gate_source_logit_bias_list, dim=1),
            "attention_gate_common_cancel_alphas": torch.stack(attention_gate_common_cancel_alpha_list, dim=1),
            "attention_gate_contrast_shift_alphas": torch.stack(attention_gate_contrast_shift_alpha_list, dim=1),
            "attention_gate_mild_rescue_alphas": torch.stack(attention_gate_mild_rescue_alpha_list, dim=1),
            "attention_gate_cancel_temperatures": torch.stack(attention_gate_cancel_temperature_list, dim=1),
            "attention_gate_selector_temperatures": torch.stack(attention_gate_selector_temperature_list, dim=1),
            "attention_gate_rescue_selector_floors": torch.stack(attention_gate_rescue_selector_floor_list, dim=1),
            "attention_gate_mild_gate_mins": torch.stack(attention_gate_mild_gate_min_list, dim=1),
            "reliability_scores": torch.stack(reliability_score_list, dim=1),
            "reliability_logits": torch.stack(reliability_logit_list, dim=1),
            "reliability_local_residual_deltas": torch.stack(reliability_local_residual_delta_list, dim=1),
            "provisional_reliability_local_residual_deltas": torch.stack(
                provisional_reliability_local_residual_delta_list,
                dim=1,
            ),
            "reliability_ego_residual_deltas": torch.stack(reliability_ego_residual_delta_list, dim=1),
            "provisional_reliability_ego_residual_deltas": torch.stack(
                provisional_reliability_ego_residual_delta_list,
                dim=1,
            ),
            "action_means": torch.stack(action_mean_list, dim=1),
            "action_stds": torch.stack(action_std_list, dim=1),
            "final_hidden": hidden,
        }
