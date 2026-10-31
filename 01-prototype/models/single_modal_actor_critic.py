from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    obs_dim: int = 13
    action_dim: int = 3
    hidden_dim: int = 128
    min_std: float = 0.05


class SingleModalActorCritic(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = nn.Sequential(
            nn.Linear(self.config.obs_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )

        self.actor_head = nn.Linear(self.config.hidden_dim, self.config.action_dim)
        self.critic_head = nn.Linear(self.config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(self.config.action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        policy_latent = self.actor_head(features)

        steer_mean = torch.tanh(policy_latent[..., 0:1])
        throttle_mean = torch.sigmoid(policy_latent[..., 1:2])
        brake_mean = torch.sigmoid(policy_latent[..., 2:3])
        action_mean = torch.cat([steer_mean, throttle_mean, brake_mean], dim=-1)

        value = self.critic_head(features).squeeze(-1)
        return action_mean, value

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp_min(self.config.min_std)
