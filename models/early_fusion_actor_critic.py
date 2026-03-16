from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EarlyFusionConfig:
    obs_dim: int
    action_dim: int = 3
    hidden_dim: int = 256
    min_std: float = 0.05
    rgb_shape: tuple[int, int, int] = (3, 90, 160)
    lidar_shape: tuple[int, int, int] = (1, 64, 64)
    ego_dim: int = 7
    route_dim: int = 6
    rgb_feature_dim: int = 64
    lidar_feature_dim: int = 32
    ego_feature_dim: int = 32
    route_feature_dim: int = 32


class EarlyFusionActorCritic(nn.Module):
    def __init__(self, config: EarlyFusionConfig) -> None:
        super().__init__()
        self.config = config

        self.rgb_numel = int(config.rgb_shape[0] * config.rgb_shape[1] * config.rgb_shape[2])
        self.lidar_numel = int(config.lidar_shape[0] * config.lidar_shape[1] * config.lidar_shape[2])
        self.ego_numel = int(config.ego_dim)
        self.route_numel = int(config.route_dim)

        expected_dim = self.rgb_numel + self.lidar_numel + self.ego_numel + self.route_numel
        if expected_dim != int(config.obs_dim):
            raise ValueError(f"EarlyFusionConfig.obs_dim mismatch: expected {expected_dim}, got {config.obs_dim}")

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(config.rgb_shape[0], 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.rgb_feature_dim),
            nn.ReLU(),
        )

        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(config.lidar_shape[0], 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, config.lidar_feature_dim),
            nn.ReLU(),
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(config.ego_dim, config.ego_feature_dim),
            nn.ReLU(),
            nn.Linear(config.ego_feature_dim, config.ego_feature_dim),
            nn.ReLU(),
        )

        self.route_encoder = nn.Sequential(
            nn.Linear(config.route_dim, config.route_feature_dim),
            nn.ReLU(),
            nn.Linear(config.route_feature_dim, config.route_feature_dim),
            nn.ReLU(),
        )

        fused_dim = (
            config.rgb_feature_dim
            + config.lidar_feature_dim
            + config.ego_feature_dim
            + config.route_feature_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.critic_head = nn.Linear(config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def _split_modalities(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_end = self.rgb_numel
        lidar_end = rgb_end + self.lidar_numel
        ego_end = lidar_end + self.ego_numel
        route_end = ego_end + self.route_numel

        rgb = obs[..., :rgb_end].reshape(-1, *self.config.rgb_shape)
        lidar = obs[..., rgb_end:lidar_end].reshape(-1, *self.config.lidar_shape)
        ego = obs[..., lidar_end:ego_end].reshape(-1, self.config.ego_dim)
        route = obs[..., ego_end:route_end].reshape(-1, self.config.route_dim)
        return rgb, lidar, ego, route

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, lidar, ego, route = self._split_modalities(obs)

        rgb_feature = self.rgb_encoder(rgb)
        lidar_feature = self.lidar_encoder(lidar)
        ego_feature = self.ego_encoder(ego)
        route_feature = self.route_encoder(route)

        fused = torch.cat([rgb_feature, lidar_feature, ego_feature, route_feature], dim=-1)
        fused_feature = self.fusion(fused)
        policy_latent = self.actor_head(fused_feature)

        steer_mean = torch.tanh(policy_latent[..., 0:1])
        throttle_mean = torch.sigmoid(policy_latent[..., 1:2])
        brake_mean = torch.sigmoid(policy_latent[..., 2:3])
        action_mean = torch.cat([steer_mean, throttle_mean, brake_mean], dim=-1)

        value = self.critic_head(fused_feature).squeeze(-1)
        return action_mean, value

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp_min(self.config.min_std)
