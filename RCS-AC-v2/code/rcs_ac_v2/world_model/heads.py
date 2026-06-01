"""Prediction heads for the Stage 1 privileged world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class WorldModelHeadConfig:
    latent_dim: int
    bev_channels: int = 43
    bev_height: int = 16
    bev_width: int = 16
    hidden_dim: int = 128


class BEVDecoder(nn.Module):
    def __init__(self, cfg: WorldModelHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.bev_channels * cfg.bev_height * cfg.bev_width),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = latent.shape[:2]
        logits = self.net(latent)
        return logits.reshape(
            batch_size,
            sequence_length,
            self.cfg.bev_channels,
            self.cfg.bev_height,
            self.cfg.bev_width,
        )


class ScalarHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class WorldModelHeads(nn.Module):
    def __init__(self, cfg: WorldModelHeadConfig) -> None:
        super().__init__()
        self.decoder = BEVDecoder(cfg)
        self.reward = ScalarHead(cfg.latent_dim, cfg.hidden_dim)
        self.continue_head = ScalarHead(cfg.latent_dim, cfg.hidden_dim)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "bev_logits": self.decoder(latent),
            "reward": self.reward(latent),
            "continue_logits": self.continue_head(latent),
        }
