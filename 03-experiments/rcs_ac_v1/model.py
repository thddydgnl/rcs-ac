"""Recurrent attention actor-critic for RCS-AC v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from ppo_baselines.models.baselines import OBS_DIMS
from ppo_baselines.models.encoders import ModalityEncoder, build_mlp
from ppo_baselines.models.fusion import AttentionFusion


MODALITY_NAMES = ["vision", "distance", "ego", "route"]


@dataclass
class RecurrentPolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    hidden_state: torch.Tensor
    aux_outputs: dict[str, torch.Tensor]


class RecurrentAttentionActorCritic(nn.Module):
    """Attention fusion + shared GRU actor-critic.

    This is intentionally close to baseline C, with the main difference being
    the shared recurrent core after fusion.
    """

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        encoder_output_dim: int,
        attention_hidden_dim: int,
        recurrent_hidden_dim: int,
        append_mean_context: bool,
        attention_temperature: float,
        use_feature_layer_norm: bool,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        action_dim: int,
        action_log_std_init: float,
    ):
        super().__init__()
        self.modality_names = list(MODALITY_NAMES)
        self.encoder_output_dim = encoder_output_dim
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.action_dim = action_dim
        self.attention_temperature = attention_temperature
        self.append_mean_context = bool(append_mean_context)
        self.fused_feature_dim = encoder_output_dim * (2 if self.append_mean_context else 1)

        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(
                    input_dim=OBS_DIMS[name],
                    hidden_dims=encoder_hidden_dims,
                    output_dim=encoder_output_dim,
                )
                for name in self.modality_names
            }
        )
        self.fusion_module = AttentionFusion(
            feature_dim=encoder_output_dim,
            hidden_dim=attention_hidden_dim,
            append_mean_context=self.append_mean_context,
        )
        self.feature_norms = nn.ModuleDict(
            {
                name: nn.LayerNorm(encoder_output_dim) if use_feature_layer_norm else nn.Identity()
                for name in self.modality_names
            }
        )
        self.gru_cell = nn.GRUCell(
            input_size=self.fused_feature_dim,
            hidden_size=recurrent_hidden_dim,
        )
        self.actor_head = build_mlp(
            input_dim=recurrent_hidden_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=action_dim,
            output_activation=None,
        )
        self.critic_head = build_mlp(
            input_dim=recurrent_hidden_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            output_activation=None,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), action_log_std_init))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.recurrent_hidden_dim, device=device, dtype=torch.float32)

    def validate_hidden_state(self, hidden_state: torch.Tensor, batch_size: int) -> None:
        expected_shape = (batch_size, self.recurrent_hidden_dim)
        if tuple(hidden_state.shape) != expected_shape:
            raise ValueError(
                f"Hidden state shape mismatch: expected {expected_shape}, got {tuple(hidden_state.shape)}"
            )

    def reset_hidden(self, hidden_state: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        """Selective hidden reset for done envs.

        Works for both single-env and future parallel-env settings.
        """
        done_mask = done_mask.to(hidden_state.device).float().view(-1, 1)
        return hidden_state * (1.0 - done_mask)

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
        scores = scores / max(self.attention_temperature, 1e-6)
        weights = torch.softmax(scores, dim=-1)
        weighted_sum = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        fused = (
            torch.cat([weighted_sum, stacked.mean(dim=1)], dim=-1)
            if self.append_mean_context
            else weighted_sum
        )
        aux_outputs = {
            "attention_weights": weights,
            "attention_scores": scores,
            "attention_base_scores": self.fusion_module.base_score_head(score_features).squeeze(-1),
            "attention_relative_gap_scores": relative_gap_scores,
            "attention_relative_scores_before_bias": relative_scores_before_bias,
            "attention_bias_scores": bias_scores,
            "attention_source_bias_gates": source_gate_outputs,
            "attention_bias_gate_residuals": gate_residuals,
            "attention_bias_gate_residual_logits": gate_residual_logits,
            "attention_bias_gates": gate_scores,
            "attention_gate_reconstruction_error": torch.zeros_like(gate_scores[:, 0]),
            "attention_final_scores": scores,
            "attention_score_features": score_features,
            "attention_score_adapter_delta_norms": torch.norm(adapter_delta, dim=-1),
            "attention_relative_gap_norm": torch.norm(relative_gap_scores, dim=-1),
            "attention_bias_norm": torch.norm(bias_scores, dim=-1),
            "attention_bias_gate_norm": torch.norm(gate_scores, dim=-1),
            "attention_relative_gap_mode": str(getattr(self.fusion_module, "relative_gap_mode", "vision_anchor")),
            "ordered_modalities": list(self.modality_names),
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

    @staticmethod
    def _squash_action(pre_tanh_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh_action)

    @staticmethod
    def _tanh_log_prob(dist: Normal, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        gaussian_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        correction = torch.log(1.0 - torch.tanh(pre_tanh_action).pow(2) + 1e-6).sum(dim=-1)
        return gaussian_log_prob - correction

    def step(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool = False,
        gate_feature_context: dict[str, torch.Tensor] | None = None,
    ) -> RecurrentPolicyOutput:
        batch_size = next(iter(obs_dict.values())).shape[0]
        self.validate_hidden_state(hidden_state, batch_size)
        hidden_state = self.reset_hidden(hidden_state, episode_start)

        fused, aux_outputs = self.encode_and_fuse(obs_dict, gate_feature_context=gate_feature_context)
        next_hidden = self.gru_cell(fused, hidden_state)
        hidden_norm = torch.norm(next_hidden, dim=-1)

        action_mean = self.actor_head(next_hidden)
        action_std = torch.exp(self.log_std).unsqueeze(0).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        pre_tanh_action = dist.mean if deterministic else dist.rsample()
        action = self._squash_action(pre_tanh_action)
        log_prob = self._tanh_log_prob(dist, pre_tanh_action)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(next_hidden).squeeze(-1)

        aux_outputs.update(
            {
                "action_mean": action_mean,
                "action_std": action_std,
                "hidden_norm": hidden_norm,
                "next_hidden": next_hidden,
            }
        )
        return RecurrentPolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            hidden_state=next_hidden,
            aux_outputs=aux_outputs,
        )

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
                dist = Normal(step_output.aux_outputs["action_mean"], step_output.aux_outputs["action_std"])
                log_prob = self._tanh_log_prob(dist, pre_tanh_action)

            action_list.append(action)
            log_prob_list.append(log_prob)
            entropy_list.append(step_output.entropy)
            value_list.append(step_output.value)
            hidden_norm_list.append(step_output.aux_outputs["hidden_norm"])
            attention_weight_list.append(step_output.aux_outputs["attention_weights"])
            attention_score_list.append(step_output.aux_outputs["attention_scores"])
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
            "action_means": torch.stack(action_mean_list, dim=1),
            "action_stds": torch.stack(action_std_list, dim=1),
            "final_hidden": hidden,
        }
