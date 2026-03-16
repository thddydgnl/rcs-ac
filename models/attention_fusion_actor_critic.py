from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AttentionFusionConfig:
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
    attention_heads: int = 4
    attention_layers: int = 1
    dropout: float = 0.0


class AttentionFusionActorCritic(nn.Module):
    def __init__(self, config: AttentionFusionConfig) -> None:
        super().__init__()
        self.config = config

        self.rgb_numel = int(config.rgb_shape[0] * config.rgb_shape[1] * config.rgb_shape[2])
        self.lidar_numel = int(config.lidar_shape[0] * config.lidar_shape[1] * config.lidar_shape[2])
        self.ego_numel = int(config.ego_dim)
        self.route_numel = int(config.route_dim)

        expected_dim = self.rgb_numel + self.lidar_numel + self.ego_numel + self.route_numel
        if expected_dim != int(config.obs_dim):
            raise ValueError(f"AttentionFusionConfig.obs_dim mismatch: expected {expected_dim}, got {config.obs_dim}")

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

        self.rgb_proj = nn.Linear(config.rgb_feature_dim, config.hidden_dim)
        self.lidar_proj = nn.Linear(config.lidar_feature_dim, config.hidden_dim)
        self.ego_proj = nn.Linear(config.ego_feature_dim, config.hidden_dim)
        self.route_proj = nn.Linear(config.route_feature_dim, config.hidden_dim)

        self.modality_embed = nn.Parameter(torch.zeros(4, config.hidden_dim))
        nn.init.normal_(self.modality_embed, mean=0.0, std=0.02)

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=config.hidden_dim,
                    num_heads=config.attention_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.attention_layers)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.attention_layers)])

        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.critic_head = nn.Linear(config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        self._last_attention_summary = {
            "attention_mean_rgb": 0.0,
            "attention_mean_lidar": 0.0,
            "attention_mean_ego": 0.0,
            "attention_mean_route": 0.0,
        }

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

    def _encode_modalities(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        rgb, lidar, ego, route = self._split_modalities(obs)

        rgb_token = self.rgb_proj(self.rgb_encoder(rgb))
        lidar_token = self.lidar_proj(self.lidar_encoder(lidar))
        ego_token = self.ego_proj(self.ego_encoder(ego))
        route_token = self.route_proj(self.route_encoder(route))

        tokens = torch.stack([rgb_token, lidar_token, ego_token, route_token], dim=1)
        tokens = tokens + self.modality_embed.unsqueeze(0)

        attention_weights = None
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            attended, attention_weights = attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
            tokens = norm(tokens + attended)

        return tokens, attention_weights

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, attention_weights = self._encode_modalities(obs)

        if attention_weights is not None:
            modality_scores = attention_weights.mean(dim=1).mean(dim=0)
            self._last_attention_summary = {
                "attention_mean_rgb": float(modality_scores[0].detach().cpu().item()),
                "attention_mean_lidar": float(modality_scores[1].detach().cpu().item()),
                "attention_mean_ego": float(modality_scores[2].detach().cpu().item()),
                "attention_mean_route": float(modality_scores[3].detach().cpu().item()),
            }

        pooled = tokens.mean(dim=1)
        fused_feature = self.fusion(pooled)
        policy_latent = self.actor_head(fused_feature)

        steer_mean = torch.tanh(policy_latent[..., 0:1])
        throttle_mean = torch.sigmoid(policy_latent[..., 1:2])
        brake_mean = torch.sigmoid(policy_latent[..., 2:3])
        action_mean = torch.cat([steer_mean, throttle_mean, brake_mean], dim=-1)

        value = self.critic_head(fused_feature).squeeze(-1)
        return action_mean, value

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp_min(self.config.min_std)

    def get_diagnostics(self) -> dict[str, float]:
        return dict(self._last_attention_summary)
