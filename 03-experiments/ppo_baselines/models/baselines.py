"""Baseline actor-critic models for RCSToyNavEnv."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from ppo_baselines.models.encoders import ModalityEncoder, build_mlp
from ppo_baselines.models.fusion import AttentionFusion, EarlyFusion


OBS_DIMS = {
    "vision": 8,
    "distance": 9,
    "ego": 5,
    "route": 5,
    "aux": 8,
}


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    aux_outputs: dict[str, torch.Tensor | None]


class GaussianActorCriticBase(nn.Module):
    """Shared actor-critic skeleton for all baselines."""

    def __init__(
        self,
        modality_names: list[str],
        encoder_hidden_dims: list[int],
        encoder_output_dim: int,
        trunk_hidden_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        action_dim: int,
        action_log_std_init: float,
    ):
        super().__init__()
        self.modality_names = modality_names
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(
                    input_dim=OBS_DIMS[name],
                    hidden_dims=encoder_hidden_dims,
                    output_dim=encoder_output_dim,
                )
                for name in modality_names
            }
        )
        self.encoder_output_dim = encoder_output_dim
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.full((action_dim,), action_log_std_init))

        self.fusion_module: nn.Module
        self.fused_dim: int

        self.shared_trunk: nn.Module
        self.actor_head: nn.Module
        self.critic_head: nn.Module

        self._build_heads(trunk_hidden_dims, actor_hidden_dims, critic_hidden_dims)

    def _build_heads(
        self,
        trunk_hidden_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
    ) -> None:
        self.shared_trunk = build_mlp(
            input_dim=self.fused_dim,
            hidden_dims=trunk_hidden_dims[:-1] if trunk_hidden_dims else [],
            output_dim=trunk_hidden_dims[-1] if trunk_hidden_dims else self.fused_dim,
            output_activation=nn.Tanh(),
        )
        trunk_out_dim = trunk_hidden_dims[-1] if trunk_hidden_dims else self.fused_dim
        self.actor_head = build_mlp(
            input_dim=trunk_out_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=self.action_dim,
            output_activation=None,
        )
        self.critic_head = build_mlp(
            input_dim=trunk_out_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            output_activation=None,
        )

    def encode_modalities(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.encoders[name](obs_dict[name]) for name in self.modality_names}

    def fuse_features(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        return self.fusion_module(encoded)

    def extract_features(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        encoded = self.encode_modalities(obs_dict)
        return self.fuse_features(encoded)

    def get_distribution(self, obs_dict: dict[str, torch.Tensor]) -> tuple[Normal, torch.Tensor, dict[str, torch.Tensor | None]]:
        fused, aux_outputs = self.extract_features(obs_dict)
        trunk = self.shared_trunk(fused)
        action_mean = self.actor_head(trunk)
        action_std = torch.exp(self.log_std).unsqueeze(0).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        value = self.critic_head(trunk).squeeze(-1)
        aux_outputs = dict(aux_outputs)
        aux_outputs["action_mean"] = action_mean
        aux_outputs["action_std"] = action_std
        return dist, value, aux_outputs

    @staticmethod
    def _squash_action(pre_tanh_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh_action)

    @staticmethod
    def _tanh_log_prob(dist: Normal, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        gaussian_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        correction = torch.log(1.0 - torch.tanh(pre_tanh_action).pow(2) + 1e-6).sum(dim=-1)
        return gaussian_log_prob - correction

    def act(self, obs_dict: dict[str, torch.Tensor], deterministic: bool = False) -> PolicyOutput:
        dist, value, aux_outputs = self.get_distribution(obs_dict)
        pre_tanh_action = dist.mean if deterministic else dist.rsample()
        action = self._squash_action(pre_tanh_action)
        log_prob = self._tanh_log_prob(dist, pre_tanh_action)
        entropy = dist.entropy().sum(dim=-1)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            action_mean=aux_outputs["action_mean"],
            action_std=aux_outputs["action_std"],
            aux_outputs=aux_outputs,
        )

    def evaluate_actions(
        self,
        obs_dict: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> PolicyOutput:
        unclipped_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        dist, value, aux_outputs = self.get_distribution(obs_dict)
        log_prob = self._tanh_log_prob(dist, unclipped_action)
        entropy = dist.entropy().sum(dim=-1)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            action_mean=aux_outputs["action_mean"],
            action_std=aux_outputs["action_std"],
            aux_outputs=aux_outputs,
        )


class BaselineAActorCritic(GaussianActorCriticBase):
    """Baseline A: ego + route only, concat fusion."""

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        encoder_output_dim: int,
        trunk_hidden_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        action_dim: int,
        action_log_std_init: float,
    ):
        self.fused_dim = encoder_output_dim * 2
        super().__init__(
            modality_names=["ego", "route"],
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            trunk_hidden_dims=trunk_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
        self.fusion_module = EarlyFusion()
        self.fused_dim = encoder_output_dim * 2
        self._build_heads(trunk_hidden_dims, actor_hidden_dims, critic_hidden_dims)


class BaselineBEarlyFusionActorCritic(GaussianActorCriticBase):
    """Baseline B: 4-modality early fusion."""

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        encoder_output_dim: int,
        trunk_hidden_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        action_dim: int,
        action_log_std_init: float,
    ):
        self.fused_dim = encoder_output_dim * 4
        super().__init__(
            modality_names=["vision", "distance", "ego", "route"],
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            trunk_hidden_dims=trunk_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
        self.fusion_module = EarlyFusion()
        self.fused_dim = encoder_output_dim * 4
        self._build_heads(trunk_hidden_dims, actor_hidden_dims, critic_hidden_dims)


class BaselineCAttentionFusionActorCritic(GaussianActorCriticBase):
    """Baseline C: 4-modality attention fusion."""

    def __init__(
        self,
        encoder_hidden_dims: list[int],
        encoder_output_dim: int,
        trunk_hidden_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        action_dim: int,
        action_log_std_init: float,
        attention_hidden_dim: int,
    ):
        self.fused_dim = encoder_output_dim * 2
        super().__init__(
            modality_names=["vision", "distance", "ego", "route"],
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            trunk_hidden_dims=trunk_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
        self.fusion_module = AttentionFusion(
            feature_dim=encoder_output_dim,
            hidden_dim=attention_hidden_dim,
            append_mean_context=True,
        )
        self.fused_dim = encoder_output_dim * 2
        self._build_heads(trunk_hidden_dims, actor_hidden_dims, critic_hidden_dims)


def build_policy_model(
    baseline_type: str,
    action_dim: int,
    encoder_hidden_dims: list[int],
    encoder_output_dim: int,
    trunk_hidden_dims: list[int],
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
    action_log_std_init: float,
    attention_hidden_dim: int,
) -> GaussianActorCriticBase:
    if baseline_type == "baseline_a":
        return BaselineAActorCritic(
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            trunk_hidden_dims=trunk_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
    if baseline_type == "baseline_b":
        return BaselineBEarlyFusionActorCritic(
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            trunk_hidden_dims=trunk_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
        )
    if baseline_type == "baseline_c":
        return BaselineCAttentionFusionActorCritic(
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_output_dim=encoder_output_dim,
            trunk_hidden_dims=trunk_hidden_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            action_dim=action_dim,
            action_log_std_init=action_log_std_init,
            attention_hidden_dim=attention_hidden_dim,
        )
    raise ValueError(f"Unsupported baseline_type: {baseline_type}")
