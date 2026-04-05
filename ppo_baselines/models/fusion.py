"""Fusion modules for multimodal baselines."""

from __future__ import annotations

import torch
from torch import nn


class EarlyFusion(nn.Module):
    """Concatenate modality features."""

    def forward(self, features_by_modality: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ordered_names = list(features_by_modality.keys())
        fused = torch.cat([features_by_modality[name] for name in ordered_names], dim=-1)
        aux = {
            "attention_weights": None,
            "ordered_modalities": ordered_names,
        }
        return fused, aux


class AttentionFusion(nn.Module):
    """Learn soft attention weights over modality embeddings."""

    def __init__(self, feature_dim: int, hidden_dim: int, append_mean_context: bool = False):
        super().__init__()
        self.append_mean_context = append_mean_context
        self.score_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, max(feature_dim // 4, 1)),
            nn.SiLU(),
            nn.Linear(max(feature_dim // 4, 1), feature_dim),
        )
        self.score_adapter_residual_scale = 0.0
        self.relative_gap_residual_scale = 0.0
        self.reliability_bias_residual_scale = 0.0
        self.reliability_bias_gate_temperature = 1.0
        self.reliability_bias_gate_threshold = 0.0
        self.gate_residual_scale = 0.0
        self.gate_residual_alpha = 1.0
        self.gate_residual_alpha_mode = "fixed"
        self.gate_residual_alpha_max = 8.0
        self.gate_residual_alpha_l2_coef = 0.0
        self.gate_logit_offset = 0.0
        self.gate_logit_offset_mode = "fixed"
        self.gate_logit_offset_max_abs = 2.0
        self.gate_logit_offset_l2_coef = 0.0
        self.adapter_gate_alpha = 0.0
        self.adapter_gate_alpha_mode = "fixed"
        self.adapter_gate_alpha_max = 8.0
        self.adapter_gate_alpha_l2_coef = 0.0
        self.gate_output_scale = 1.0
        self.gate_output_scale_mode = "fixed"
        self.gate_output_scale_max = 16.0
        self.gate_output_scale_l2_coef = 0.0
        self.gate_threshold_shift_alpha = 0.0
        self.gate_threshold_shift_alpha_mode = "fixed"
        self.gate_threshold_shift_alpha_max = 8.0
        self.gate_threshold_shift_alpha_l2_coef = 0.0
        self.gate_source_logit_bias = 0.0
        self.gate_source_logit_bias_mode = "fixed"
        self.gate_source_logit_bias_max_abs = 2.0
        self.gate_source_logit_bias_l2_coef = 0.0
        self.gate_common_cancel_alpha = 0.0
        self.gate_common_cancel_alpha_mode = "fixed"
        self.gate_common_cancel_alpha_max = 8.0
        self.gate_common_cancel_alpha_l2_coef = 0.0
        self.gate_contrast_shift_alpha = 0.0
        self.gate_contrast_shift_alpha_mode = "fixed"
        self.gate_contrast_shift_alpha_max = 8.0
        self.gate_contrast_shift_alpha_l2_coef = 0.0
        self.gate_mild_rescue_alpha = 0.0
        self.gate_mild_rescue_alpha_mode = "fixed"
        self.gate_mild_rescue_alpha_max = 8.0
        self.gate_mild_rescue_alpha_l2_coef = 0.0
        self.common_cancel_anchor_gate = 0.05
        self.cancel_temperature = 1.0
        self.selector_temperature = 1.0
        self.rescue_selector_floor = 0.0
        self.clean_gate_max = 0.02
        self.mild_gate_min = 0.08
        self.gate_gap_margin = 0.05
        self.adapter_gate_input_mode = "legacy"
        self.use_relative_score_head = False
        self.use_reliability_bias_head = False
        self.use_reliability_bias_gate = False
        self.gate_mode = "absolute_v1"
        self.gate_input_mode = "legacy"
        self.gate_use_abs_bias_feature = False
        self.gate_use_score_delta_feature = False
        self.gate_use_teacher_delta_feature = False
        self.gate_use_source_gate_feature = False
        self.gate_use_adapter_delta_feature = False
        self.gate_use_adapter_norm_feature = False
        self.gate_use_adapter_teacher_delta_feature = False
        self.gate_feature_stopgrad_on_bias = False
        self.gate_feature_stopgrad_on_source_gate = False
        self.gate_feature_stopgrad_on_adapter_delta = False
        self.relative_gap_mode = "vision_anchor"
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.base_score_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        pair_dim = feature_dim * 2
        self.relative_gap_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, max(feature_dim // 4, 1)),
            nn.SiLU(),
            nn.Linear(max(feature_dim // 4, 1), 1),
        )
        bias_input_dim = feature_dim + 2
        self.reliability_bias_head = nn.Sequential(
            nn.LayerNorm(bias_input_dim),
            nn.Linear(bias_input_dim, max(feature_dim // 4, 1)),
            nn.SiLU(),
            nn.Linear(max(feature_dim // 4, 1), 1),
        )
        self.reliability_bias_gate_head = nn.Sequential(
            nn.LazyLinear(8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )
        self.reliability_bias_gate_residual_head = nn.Sequential(
            nn.LazyLinear(8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )
        self.reliability_bias_gate_adapter_coupler = nn.Sequential(
            nn.LazyLinear(8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )
        self.reliability_bias_gate_adapter_input_adapter = nn.Sequential(
            nn.LazyLinear(32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )
        self.reliability_bias_gate_threshold_shift_head = nn.Sequential(
            nn.LazyLinear(16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.reliability_bias_gate_common_cancel_head = nn.Sequential(
            nn.LazyLinear(16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.reliability_bias_gate_contrast_shift_head = nn.Sequential(
            nn.LazyLinear(16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.reliability_bias_gate_mild_rescue_gate_head = nn.Sequential(
            nn.LazyLinear(16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.reliability_bias_gate_mild_evidence_input_adapter = nn.Sequential(
            nn.LazyLinear(32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )
        self.reliability_bias_gate_mild_value_input_adapter = nn.Sequential(
            nn.LazyLinear(32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )
        self.reliability_bias_gate_mild_rescue_value_head = nn.Sequential(
            nn.LazyLinear(16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.reliability_bias_gate_input_adapter: nn.Module | None = None

    def get_gate_residual_alpha_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha_value = getattr(self, "gate_residual_alpha", 1.0)
        if isinstance(alpha_value, torch.Tensor):
            alpha = alpha_value.to(device=device, dtype=dtype)
        else:
            alpha = torch.as_tensor(float(alpha_value), device=device, dtype=dtype)
        alpha = torch.clamp(alpha, min=0.0)
        alpha_max = getattr(self, "gate_residual_alpha_max", None)
        if alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(alpha_max))
        return alpha

    def get_gate_logit_offset_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        offset_value = getattr(self, "gate_logit_offset", 0.0)
        if isinstance(offset_value, torch.Tensor):
            offset = offset_value.to(device=device, dtype=dtype)
        else:
            offset = torch.as_tensor(offset_value, device=device, dtype=dtype)
        max_abs = getattr(self, "gate_logit_offset_max_abs", None)
        if max_abs is not None:
            offset = torch.clamp(offset, min=-float(max_abs), max=float(max_abs))
        return offset

    def get_adapter_gate_alpha_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha_value = getattr(self, "adapter_gate_alpha", 0.0)
        if isinstance(alpha_value, torch.Tensor):
            alpha = alpha_value.to(device=device, dtype=dtype)
        else:
            alpha = torch.as_tensor(float(alpha_value), device=device, dtype=dtype)
        alpha = torch.clamp(alpha, min=0.0)
        alpha_max = getattr(self, "adapter_gate_alpha_max", None)
        if alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(alpha_max))
        return alpha

    def get_gate_output_scale_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_value = getattr(self, "gate_output_scale", 1.0)
        if isinstance(scale_value, torch.Tensor):
            scale = scale_value.to(device=device, dtype=dtype)
        else:
            scale = torch.as_tensor(scale_value, device=device, dtype=dtype)
        scale = torch.clamp(scale, min=0.0)
        scale_max = getattr(self, "gate_output_scale_max", None)
        if scale_max is not None:
            scale = torch.clamp(scale, max=float(scale_max))
        return scale

    def get_gate_mild_rescue_alpha_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha_value = getattr(self, "gate_mild_rescue_alpha", 0.0)
        if isinstance(alpha_value, torch.Tensor):
            alpha = alpha_value.to(device=device, dtype=dtype)
        else:
            alpha = torch.as_tensor(float(alpha_value), device=device, dtype=dtype)
        alpha = torch.clamp(alpha, min=0.0)
        alpha_max = getattr(self, "gate_mild_rescue_alpha_max", None)
        if alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(alpha_max))
        return alpha

    def get_gate_threshold_shift_alpha_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha_value = getattr(self, "gate_threshold_shift_alpha", 0.0)
        if isinstance(alpha_value, torch.Tensor):
            alpha = alpha_value.to(device=device, dtype=dtype)
        else:
            alpha = torch.as_tensor(alpha_value, device=device, dtype=dtype)
        alpha_max = getattr(self, "gate_threshold_shift_alpha_max", None)
        if alpha_max is not None:
            alpha = torch.clamp(alpha, min=-float(alpha_max), max=float(alpha_max))
        return alpha

    def get_gate_source_logit_bias_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bias_value = getattr(self, "gate_source_logit_bias", 0.0)
        if isinstance(bias_value, torch.Tensor):
            bias = bias_value.to(device=device, dtype=dtype)
        else:
            bias = torch.as_tensor(bias_value, device=device, dtype=dtype)
        max_abs = getattr(self, "gate_source_logit_bias_max_abs", None)
        if max_abs is not None:
            bias = torch.clamp(bias, min=-float(max_abs), max=float(max_abs))
        return bias

    def get_gate_common_cancel_alpha_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha_value = getattr(self, "gate_common_cancel_alpha", 0.0)
        if isinstance(alpha_value, torch.Tensor):
            alpha = alpha_value.to(device=device, dtype=dtype)
        else:
            alpha = torch.as_tensor(alpha_value, device=device, dtype=dtype)
        alpha = torch.clamp(alpha, min=0.0)
        alpha_max = getattr(self, "gate_common_cancel_alpha_max", None)
        if alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(alpha_max))
        return alpha

    def get_gate_contrast_shift_alpha_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha_value = getattr(self, "gate_contrast_shift_alpha", 0.0)
        if isinstance(alpha_value, torch.Tensor):
            alpha = alpha_value.to(device=device, dtype=dtype)
        else:
            alpha = torch.as_tensor(alpha_value, device=device, dtype=dtype)
        alpha = torch.clamp(alpha, min=0.0)
        alpha_max = getattr(self, "gate_contrast_shift_alpha_max", None)
        if alpha_max is not None:
            alpha = torch.clamp(alpha, max=float(alpha_max))
        return alpha

    def adapt_score_features(self, stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.score_adapter(stacked)
        score_features = stacked + float(self.score_adapter_residual_scale) * delta
        return score_features, delta

    def initialize_base_score_head_from_score_net(self) -> None:
        for base_param, score_param in zip(self.base_score_head.parameters(), self.score_net.parameters(), strict=True):
            base_param.data.copy_(score_param.data)

    def compute_scores(
        self,
        stacked: torch.Tensor,
        bias_inputs: torch.Tensor | None = None,
        gate_inputs: torch.Tensor | None = None,
        adapter_gate_inputs: torch.Tensor | None = None,
        source_gate_scores: torch.Tensor | None = None,
        mild_rescue_inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        score_features, adapter_delta = self.adapt_score_features(stacked)
        legacy_scores = self.score_net(score_features).squeeze(-1)
        base_scores = self.base_score_head(score_features).squeeze(-1)
        relative_gap_mode = str(getattr(self, "relative_gap_mode", "vision_anchor"))
        if relative_gap_mode == "vision_anchor":
            pair_indices = [(0, 1), (0, 2), (0, 3)]
        elif relative_gap_mode == "all_pairs":
            pair_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        else:
            raise ValueError(f"Unsupported relative_gap_mode: {relative_gap_mode}")
        gap_pairs = torch.stack(
            [torch.cat([score_features[:, lhs], score_features[:, rhs]], dim=-1) for lhs, rhs in pair_indices],
            dim=1,
        )
        relative_gaps = self.relative_gap_head(gap_pairs).squeeze(-1)
        if bool(self.use_relative_score_head):
            gap_scale = float(self.relative_gap_residual_scale)
            scores = base_scores.clone()
            for pair_idx, (lhs, rhs) in enumerate(pair_indices):
                correction = gap_scale * relative_gaps[:, pair_idx]
                scores[:, lhs] = scores[:, lhs] + correction
                scores[:, rhs] = scores[:, rhs] - correction
            relative_gap_scores = relative_gaps
        else:
            scores = legacy_scores
            relative_gap_scores = torch.zeros_like(relative_gaps)
        relative_scores_before_bias = scores.clone()

        if bool(self.use_reliability_bias_head) and bias_inputs is not None:
            bias_scores = self.reliability_bias_head(bias_inputs).squeeze(-1)
            if bool(self.use_reliability_bias_gate) and gate_inputs is not None:
                gate_input_adapter = getattr(self, "reliability_bias_gate_input_adapter", None)
                if gate_input_adapter is not None:
                    gate_inputs = gate_input_adapter(gate_inputs)
                gate_mode = str(getattr(self, "gate_mode", "absolute_v1"))
                if gate_mode == "source_residual_v1":
                    if source_gate_scores is None:
                        raise ValueError("source_residual_v1 requires source_gate_scores")
                    residual_logits = self.reliability_bias_gate_residual_head(gate_inputs).squeeze(-1)
                    gate_residual_logits = residual_logits
                    gate_residuals = float(self.gate_residual_scale) * torch.tanh(residual_logits)
                    source_gate_outputs = source_gate_scores
                    adapter_gate_logits = torch.zeros_like(gate_residuals)
                    gate_delta_logits = gate_residuals
                    gate_scaled_delta_logits = gate_delta_logits
                    gate_scores = torch.clamp(source_gate_outputs + gate_residuals, 0.0, 1.0)
                    gate_threshold_shift_logits = torch.zeros_like(gate_scores)
                    gate_post_shift_logits = gate_scores
                    gate_post_shift_excess = torch.zeros_like(gate_scores)
                    gate_common_cancel_gates = torch.zeros_like(gate_scores)
                    gate_common_cancel_logits = torch.zeros_like(gate_scores)
                    gate_post_cancel_logits = gate_post_shift_logits
                    gate_contrast_shift_logits = torch.zeros_like(gate_scores)
                    gate_mild_rescue_gates = torch.zeros_like(gate_scores)
                    gate_mild_rescue_selector_actives = torch.zeros_like(gate_scores)
                    gate_mild_rescue_values = torch.zeros_like(gate_scores)
                    gate_mild_rescue_logits = torch.zeros_like(gate_scores)
                elif gate_mode == "source_residual_logit_v2":
                    if source_gate_scores is None:
                        raise ValueError("source_residual_logit_v2 requires source_gate_scores")
                    residual_logits = self.reliability_bias_gate_residual_head(gate_inputs).squeeze(-1)
                    gate_residual_logits = residual_logits
                    source_gate_outputs = source_gate_scores
                    source_gate_clamped = source_gate_outputs.clamp(1.0e-4, 1.0 - 1.0e-4)
                    source_gate_logits = torch.logit(source_gate_clamped)
                    gate_residuals = residual_logits
                    residual_alpha = self.get_gate_residual_alpha_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    gate_logit_offset = self.get_gate_logit_offset_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    if adapter_gate_inputs is not None:
                        adapter_gate_logits = self.reliability_bias_gate_adapter_coupler(adapter_gate_inputs).squeeze(-1)
                    else:
                        adapter_gate_logits = torch.zeros_like(residual_logits)
                    adapter_gate_alpha = self.get_adapter_gate_alpha_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    gate_output_scale = self.get_gate_output_scale_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    gate_delta_logits = (
                        gate_logit_offset
                        + residual_alpha * residual_logits
                        + adapter_gate_alpha * adapter_gate_logits
                    )
                    gate_scaled_delta_logits = gate_output_scale * gate_delta_logits
                    gate_threshold_shift_alpha = self.get_gate_threshold_shift_alpha_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    source_logit_bias = self.get_gate_source_logit_bias_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    if adapter_gate_inputs is not None:
                        threshold_inputs = torch.cat(
                            [
                                adapter_gate_inputs,
                                adapter_gate_logits.unsqueeze(-1),
                                gate_scaled_delta_logits.unsqueeze(-1),
                                source_gate_outputs.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        gate_threshold_shift_logits = gate_threshold_shift_alpha * torch.tanh(
                            self.reliability_bias_gate_threshold_shift_head(threshold_inputs).squeeze(-1)
                        )
                    else:
                        gate_threshold_shift_logits = torch.zeros_like(residual_logits)
                    gate_post_shift_logits = (
                        source_gate_logits + source_logit_bias + gate_scaled_delta_logits + gate_threshold_shift_logits
                    )
                    source_anchor_gate = float(getattr(self, "common_cancel_anchor_gate", 0.05))
                    source_anchor_gate = min(max(source_anchor_gate, 1.0e-4), 1.0 - 1.0e-4)
                    source_anchor_logit = torch.logit(torch.full_like(gate_post_shift_logits, source_anchor_gate))
                    gate_post_shift_excess = torch.relu(gate_post_shift_logits - source_anchor_logit)
                    gate_common_cancel_alpha = self.get_gate_common_cancel_alpha_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    if adapter_gate_inputs is not None:
                        cancel_inputs = torch.cat(
                            [
                                adapter_gate_inputs,
                                adapter_gate_logits.unsqueeze(-1),
                                gate_scaled_delta_logits.unsqueeze(-1),
                                source_gate_outputs.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        cancel_temperature = float(getattr(self, "cancel_temperature", 1.0))
                        gate_common_cancel_gates = torch.sigmoid(
                            cancel_temperature * self.reliability_bias_gate_common_cancel_head(cancel_inputs).squeeze(-1)
                        )
                        gate_common_cancel_logits = (
                            -gate_common_cancel_alpha * gate_common_cancel_gates * gate_post_shift_excess
                        )
                    else:
                        gate_common_cancel_gates = torch.zeros_like(residual_logits)
                        gate_common_cancel_logits = torch.zeros_like(residual_logits)
                    gate_post_cancel_logits = gate_post_shift_logits + gate_common_cancel_logits
                    gate_contrast_shift_alpha = self.get_gate_contrast_shift_alpha_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    if adapter_gate_inputs is not None:
                        contrast_inputs = torch.cat(
                            [
                                adapter_gate_inputs,
                                adapter_gate_logits.unsqueeze(-1),
                                gate_scaled_delta_logits.unsqueeze(-1),
                                gate_post_cancel_logits.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        gate_contrast_shift_logits = gate_contrast_shift_alpha * torch.tanh(
                            self.reliability_bias_gate_contrast_shift_head(contrast_inputs).squeeze(-1)
                        )
                    else:
                        gate_contrast_shift_logits = torch.zeros_like(residual_logits)
                    gate_mild_rescue_alpha = self.get_gate_mild_rescue_alpha_tensor(
                        device=residual_logits.device,
                        dtype=residual_logits.dtype,
                    )
                    if mild_rescue_inputs is not None:
                        rescue_inputs = torch.cat(
                            [
                                mild_rescue_inputs,
                                adapter_gate_logits.unsqueeze(-1),
                                (gate_post_cancel_logits - source_anchor_logit).unsqueeze(-1),
                                torch.relu(
                                    torch.logit(
                                        torch.full_like(
                                            gate_post_cancel_logits,
                                            min(max(float(getattr(self, "mild_gate_min", 0.08)), 1.0e-4), 1.0 - 1.0e-4),
                                        )
                                    )
                                    - gate_post_cancel_logits
                                ).unsqueeze(-1),
                                source_gate_outputs.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        rescue_hidden = self.reliability_bias_gate_mild_evidence_input_adapter(rescue_inputs)
                        selector_temperature = float(getattr(self, "selector_temperature", 1.0))
                        gate_mild_rescue_gates = torch.sigmoid(
                            selector_temperature
                            * self.reliability_bias_gate_mild_rescue_gate_head(rescue_hidden).squeeze(-1)
                        )
                        selector_floor = min(max(float(getattr(self, "rescue_selector_floor", 0.0)), 0.0), 0.999)
                        gate_mild_rescue_selector_actives = torch.relu(
                            gate_mild_rescue_gates
                            - torch.as_tensor(
                                selector_floor,
                                device=gate_mild_rescue_gates.device,
                                dtype=gate_mild_rescue_gates.dtype,
                            )
                        ) / max(1.0e-6, 1.0 - selector_floor)
                        mild_gate_need = torch.relu(
                            torch.logit(
                                torch.full_like(
                                    gate_post_cancel_logits,
                                    min(max(float(getattr(self, "mild_gate_min", 0.08)), 1.0e-4), 1.0 - 1.0e-4),
                                )
                            )
                            - gate_post_cancel_logits
                        )
                        rescue_value_inputs = torch.cat(
                            [
                                rescue_hidden,
                                gate_post_cancel_logits.unsqueeze(-1),
                                mild_gate_need.unsqueeze(-1),
                                adapter_gate_logits.unsqueeze(-1),
                            ],
                            dim=-1,
                        )
                        rescue_value_hidden = self.reliability_bias_gate_mild_value_input_adapter(
                            rescue_value_inputs
                        )
                        gate_mild_rescue_values = torch.nn.functional.softplus(
                            self.reliability_bias_gate_mild_rescue_value_head(rescue_value_hidden).squeeze(-1)
                        )
                        gate_mild_rescue_logits = (
                            gate_mild_rescue_alpha * gate_mild_rescue_selector_actives * gate_mild_rescue_values
                        )
                    else:
                        gate_mild_rescue_gates = torch.zeros_like(residual_logits)
                        gate_mild_rescue_selector_actives = torch.zeros_like(residual_logits)
                        gate_mild_rescue_values = torch.zeros_like(residual_logits)
                        gate_mild_rescue_logits = torch.zeros_like(residual_logits)
                    gate_scores = torch.sigmoid(
                        gate_post_cancel_logits + gate_contrast_shift_logits + gate_mild_rescue_logits
                    )
                else:
                    gate_logits = self.reliability_bias_gate_head(gate_inputs).squeeze(-1)
                    gate_scores = torch.sigmoid(
                        float(self.reliability_bias_gate_temperature)
                        * (gate_logits - float(self.reliability_bias_gate_threshold))
                    )
                    source_gate_outputs = torch.zeros_like(gate_scores)
                    gate_residuals = torch.zeros_like(gate_scores)
                    gate_residual_logits = torch.zeros_like(gate_scores)
                    adapter_gate_logits = torch.zeros_like(gate_scores)
                    gate_delta_logits = torch.zeros_like(gate_scores)
                    gate_scaled_delta_logits = torch.zeros_like(gate_scores)
                    gate_threshold_shift_logits = torch.zeros_like(gate_scores)
                    gate_post_shift_logits = torch.zeros_like(gate_scores)
                    gate_post_shift_excess = torch.zeros_like(gate_scores)
                    gate_common_cancel_gates = torch.zeros_like(gate_scores)
                    gate_common_cancel_logits = torch.zeros_like(gate_scores)
                    gate_post_cancel_logits = torch.zeros_like(gate_scores)
                    gate_contrast_shift_logits = torch.zeros_like(gate_scores)
                    gate_mild_rescue_gates = torch.zeros_like(gate_scores)
                    gate_mild_rescue_selector_actives = torch.zeros_like(gate_scores)
                    gate_mild_rescue_values = torch.zeros_like(gate_scores)
                    gate_mild_rescue_logits = torch.zeros_like(gate_scores)
            else:
                gate_scores = torch.ones_like(bias_scores)
                source_gate_outputs = torch.zeros_like(gate_scores)
                gate_residuals = torch.zeros_like(gate_scores)
                gate_residual_logits = torch.zeros_like(gate_scores)
                adapter_gate_logits = torch.zeros_like(gate_scores)
                gate_delta_logits = torch.zeros_like(gate_scores)
                gate_scaled_delta_logits = torch.zeros_like(gate_scores)
                gate_threshold_shift_logits = torch.zeros_like(gate_scores)
                gate_post_shift_logits = torch.zeros_like(gate_scores)
                gate_post_shift_excess = torch.zeros_like(gate_scores)
                gate_common_cancel_gates = torch.zeros_like(gate_scores)
                gate_common_cancel_logits = torch.zeros_like(gate_scores)
                gate_post_cancel_logits = torch.zeros_like(gate_scores)
                gate_contrast_shift_logits = torch.zeros_like(gate_scores)
                gate_mild_rescue_gates = torch.zeros_like(gate_scores)
                gate_mild_rescue_selector_actives = torch.zeros_like(gate_scores)
                gate_mild_rescue_values = torch.zeros_like(gate_scores)
                gate_mild_rescue_logits = torch.zeros_like(gate_scores)
            scores = scores + float(self.reliability_bias_residual_scale) * gate_scores * bias_scores
        else:
            bias_scores = torch.zeros_like(scores)
            gate_scores = torch.zeros_like(scores)
            source_gate_outputs = torch.zeros_like(scores)
            gate_residuals = torch.zeros_like(scores)
            gate_residual_logits = torch.zeros_like(scores)
            adapter_gate_logits = torch.zeros_like(scores)
            gate_delta_logits = torch.zeros_like(scores)
            gate_scaled_delta_logits = torch.zeros_like(scores)
            gate_threshold_shift_logits = torch.zeros_like(scores)
            gate_post_shift_logits = torch.zeros_like(scores)
            gate_post_shift_excess = torch.zeros_like(scores)
            gate_common_cancel_gates = torch.zeros_like(scores)
            gate_common_cancel_logits = torch.zeros_like(scores)
            gate_post_cancel_logits = torch.zeros_like(scores)
            gate_contrast_shift_logits = torch.zeros_like(scores)
            gate_mild_rescue_gates = torch.zeros_like(scores)
            gate_mild_rescue_selector_actives = torch.zeros_like(scores)
            gate_mild_rescue_values = torch.zeros_like(scores)
            gate_mild_rescue_logits = torch.zeros_like(scores)

        return (
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
            adapter_gate_logits,
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
        )

    def forward(self, features_by_modality: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ordered_names = list(features_by_modality.keys())
        stacked = torch.stack([features_by_modality[name] for name in ordered_names], dim=1)
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
            adapter_gate_logits,
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
        ) = self.compute_scores(stacked)
        gate_mode = str(getattr(self, "gate_mode", "absolute_v1"))
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
            residual_alpha = self.get_gate_residual_alpha_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            gate_logit_offset = self.get_gate_logit_offset_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            adapter_gate_alpha = self.get_adapter_gate_alpha_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            gate_output_scale = self.get_gate_output_scale_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            source_logit_bias = self.get_gate_source_logit_bias_tensor(
                device=gate_residuals.device,
                dtype=gate_residuals.dtype,
            )
            reconstructed_gate_scores = torch.sigmoid(
                source_gate_logits
                + source_logit_bias
                + gate_output_scale * (gate_logit_offset + residual_alpha * gate_residuals + adapter_gate_alpha * adapter_gate_logits)
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
        weights = torch.softmax(scores, dim=-1)
        weighted_sum = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        mean_context = stacked.mean(dim=1)
        fused = (
            torch.cat([weighted_sum, mean_context], dim=-1)
            if self.append_mean_context
            else weighted_sum
        )
        aux = {
            "attention_weights": weights,
            "attention_weighted_sum": weighted_sum,
            "attention_mean_context": mean_context,
            "attention_scores": scores,
            "attention_base_scores": self.base_score_head(score_features).squeeze(-1),
            "attention_relative_gap_scores": relative_gap_scores,
            "attention_relative_scores_before_bias": relative_scores_before_bias,
            "attention_bias_scores": bias_scores,
            "attention_source_bias_gates": source_gate_outputs,
            "attention_bias_gate_residuals": gate_residuals,
            "attention_bias_gate_residual_logits": gate_residual_logits,
            "attention_bias_gate_adapter_logits": adapter_gate_logits,
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
            "attention_residual_alpha": self.get_gate_residual_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_residual_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_residual_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_residual_alpha_vector": self.get_gate_residual_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_logit_offset": self.get_gate_logit_offset_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_logit_offset_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_logit_offset_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_logit_offset_vector": self.get_gate_logit_offset_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_adapter_gate_alpha": self.get_adapter_gate_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_adapter_gate_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_adapter_gate_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_adapter_gate_alpha_vector": self.get_adapter_gate_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_output_scale": self.get_gate_output_scale_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_output_scale_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_output_scale_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_output_scale_vector": self.get_gate_output_scale_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_threshold_shift_alpha": self.get_gate_threshold_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_threshold_shift_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_threshold_shift_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_threshold_shift_alpha_vector": self.get_gate_threshold_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_source_logit_bias": self.get_gate_source_logit_bias_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_source_logit_bias_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_source_logit_bias_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_source_logit_bias_vector": self.get_gate_source_logit_bias_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_common_cancel_alpha": self.get_gate_common_cancel_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_common_cancel_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_common_cancel_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_common_cancel_alpha_vector": self.get_gate_common_cancel_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_contrast_shift_alpha": self.get_gate_contrast_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_contrast_shift_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_contrast_shift_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_contrast_shift_alpha_vector": self.get_gate_contrast_shift_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_mild_rescue_alpha": self.get_gate_mild_rescue_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            )[0].expand(gate_scores.shape[0])
            if self.get_gate_mild_rescue_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).ndim > 0
            else self.get_gate_mild_rescue_alpha_tensor(device=gate_scores.device, dtype=gate_scores.dtype).expand(gate_scores.shape[0]),
            "attention_gate_mild_rescue_alpha_vector": self.get_gate_mild_rescue_alpha_tensor(
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_cancel_temperature": torch.full(
                (gate_scores.shape[0],),
                float(getattr(self, "cancel_temperature", 1.0)),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_selector_temperature": torch.full(
                (gate_scores.shape[0],),
                float(getattr(self, "selector_temperature", 1.0)),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_rescue_selector_floor": torch.full(
                (gate_scores.shape[0],),
                min(max(float(getattr(self, "rescue_selector_floor", 0.0)), 0.0), 0.999),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_gate_mild_gate_min": torch.full(
                (gate_scores.shape[0],),
                min(max(float(getattr(self, "mild_gate_min", 0.08)), 1.0e-4), 1.0 - 1.0e-4),
                device=gate_scores.device,
                dtype=gate_scores.dtype,
            ),
            "attention_final_scores": scores,
            "attention_score_features": score_features,
            "attention_score_adapter_delta": adapter_delta,
            "attention_score_adapter_delta_norms": torch.norm(adapter_delta, dim=-1),
            "attention_relative_gap_norm": torch.norm(relative_gap_scores, dim=-1),
            "attention_bias_norm": torch.norm(bias_scores, dim=-1),
            "attention_bias_gate_norm": torch.norm(gate_scores, dim=-1),
            "attention_relative_gap_mode": str(getattr(self, "relative_gap_mode", "vision_anchor")),
            "ordered_modalities": ordered_names,
        }
        return fused, aux
