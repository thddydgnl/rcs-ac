"""Discrete actor-critic policy used by privileged Stage 2 smoke."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from rcs_ac_v2.env.discrete_actions import NUM_RAW2DRIVE_ACTIONS


@dataclass(frozen=True)
class ActorCriticConfig:
    latent_dim: int
    action_dim: int = NUM_RAW2DRIVE_ACTIONS
    hidden_dim: int = 128


@dataclass
class ActorCriticOutput:
    logits: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor

    @property
    def distribution(self) -> Categorical:
        return Categorical(logits=self.logits)


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor_trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.critic_trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.actor = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.critic = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> ActorCriticOutput:
        if latent.ndim != 3:
            raise ValueError(f"latent must be [B,T,D], got {tuple(latent.shape)}")
        if latent.shape[-1] != self.cfg.latent_dim:
            raise ValueError(f"latent dim must be {self.cfg.latent_dim}, got {latent.shape[-1]}")
        actor_features = self.actor_trunk(latent)
        critic_features = self.critic_trunk(latent)
        logits = self.actor(actor_features)
        value = self.critic(critic_features).squeeze(-1)
        entropy = Categorical(logits=logits).entropy()
        return ActorCriticOutput(logits=logits, value=value, entropy=entropy)

    def act(self, latent: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(latent)
        dist = output.distribution
        if deterministic:
            action = torch.argmax(output.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.to(torch.int64), log_prob
