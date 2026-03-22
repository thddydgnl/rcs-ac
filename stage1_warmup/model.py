"""Stage 1 warm-up recurrent model with weak reliability head."""

from __future__ import annotations

import torch
from torch import nn

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
    ):
        super().__init__(
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            attention_hidden_dim=attention_hidden_dim,
            recurrent_hidden_dim=recurrent_hidden_dim,
            attention_temperature=attention_temperature,
            use_feature_layer_norm=use_feature_layer_norm,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
        self.reliability_input_mode = reliability_input_mode
        reliability_input_dim = encoder_output_dim if reliability_input_mode == "own" else 2 * encoder_output_dim
        self.reliability_head = build_mlp(
            input_dim=reliability_input_dim,
            hidden_dims=reliability_hidden_dims,
            output_dim=1,
            output_activation=None,
        )

    def freeze_policy_backbone(self) -> None:
        for module in [self.encoders, self.feature_norms, self.fusion_module, self.gru_cell, self.actor_head, self.critic_head]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.log_std.requires_grad = False

    def unfreeze_reliability_head(self) -> None:
        for parameter in self.reliability_head.parameters():
            parameter.requires_grad = True

    def forward_reliability(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _fused, aux_outputs = self.encode_and_fuse(obs_dict)
        return {
            "reliability_scores": aux_outputs["reliability_scores"],
            "reliability_logits": aux_outputs["reliability_logits"],
            "reliability_inputs": aux_outputs["reliability_inputs"],
            "encoder_features": aux_outputs["encoder_features"],
            "attention_weights": aux_outputs["attention_weights"],
            "fused_feature": aux_outputs["fused_feature"],
        }

    def encode_and_fuse(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = {
            name: self.feature_norms[name](self.encoders[name](obs_dict[name]))
            for name in self.modality_names
        }
        stacked = torch.stack([encoded[name] for name in self.modality_names], dim=1)
        scores = self.fusion_module.score_net(stacked).squeeze(-1)
        scores = scores / max(self.attention_temperature, 1e-6)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        if self.reliability_input_mode == "own_plus_fused":
            fused_expanded = fused.unsqueeze(1).expand(-1, stacked.shape[1], -1)
            reliability_inputs = torch.cat([stacked, fused_expanded], dim=-1)
        else:
            reliability_inputs = stacked
        reliability_logits = self.reliability_head(reliability_inputs).squeeze(-1)
        reliability_scores = torch.sigmoid(reliability_logits)
        aux_outputs = {
            "attention_weights": weights,
            "reliability_scores": reliability_scores,
            "reliability_logits": reliability_logits,
            "ordered_modalities": list(MODALITY_NAMES),
            "encoder_features": stacked,
            "reliability_inputs": reliability_inputs,
            "fused_feature": fused,
            "encoder_feature_norms": torch.stack(
                [torch.norm(encoded[name], dim=-1) for name in self.modality_names],
                dim=-1,
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
        reliability_score_list = []
        reliability_logit_list = []
        action_mean_list = []
        action_std_list = []

        for timestep in range(seq_len):
            step_obs = {key: obs_seq[key][:, timestep] for key in self.modality_names}
            step_output = self.step(
                obs_dict=step_obs,
                hidden_state=hidden,
                episode_start=episode_starts[:, timestep],
                deterministic=deterministic,
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
            reliability_score_list.append(step_output.aux_outputs["reliability_scores"])
            reliability_logit_list.append(step_output.aux_outputs["reliability_logits"])
            action_mean_list.append(step_output.aux_outputs["action_mean"])
            action_std_list.append(step_output.aux_outputs["action_std"])

        return {
            "actions": torch.stack(action_list, dim=1),
            "log_probs": torch.stack(log_prob_list, dim=1),
            "entropies": torch.stack(entropy_list, dim=1),
            "values": torch.stack(value_list, dim=1),
            "hidden_norms": torch.stack(hidden_norm_list, dim=1),
            "attention_weights": torch.stack(attention_weight_list, dim=1),
            "reliability_scores": torch.stack(reliability_score_list, dim=1),
            "reliability_logits": torch.stack(reliability_logit_list, dim=1),
            "action_means": torch.stack(action_mean_list, dim=1),
            "action_stds": torch.stack(action_std_list, dim=1),
            "final_hidden": hidden,
        }
