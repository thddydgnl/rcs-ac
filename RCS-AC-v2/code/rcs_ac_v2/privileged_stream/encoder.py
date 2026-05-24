"""Privileged BEV + ego encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PrivilegedEncoderConfig:
    bev_channels: int = 43
    ego_dim: int = 8
    embed_dim: int = 128
    conv_channels: int = 32
    pooled_size: int = 4


class PrivilegedEncoder(nn.Module):
    def __init__(self, cfg: PrivilegedEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PrivilegedEncoderConfig()
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(self.cfg.bev_channels, self.cfg.conv_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.cfg.conv_channels, self.cfg.conv_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.cfg.pooled_size, self.cfg.pooled_size)),
            nn.Flatten(),
        )
        bev_dim = self.cfg.conv_channels * self.cfg.pooled_size * self.cfg.pooled_size
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.cfg.ego_dim, self.cfg.conv_channels),
            nn.SiLU(),
        )
        self.project = nn.Sequential(
            nn.Linear(bev_dim + self.cfg.conv_channels, self.cfg.embed_dim),
            nn.SiLU(),
        )

    def forward(self, privileged_bev: torch.Tensor, privileged_ego: torch.Tensor) -> torch.Tensor:
        if privileged_bev.ndim != 5:
            raise ValueError(f"privileged_bev must be [B,T,C,H,W], got {tuple(privileged_bev.shape)}")
        if privileged_ego.ndim != 3:
            raise ValueError(f"privileged_ego must be [B,T,D], got {tuple(privileged_ego.shape)}")
        batch_size, sequence_length, channels, height, width = privileged_bev.shape
        if channels != self.cfg.bev_channels:
            raise ValueError(f"expected {self.cfg.bev_channels} BEV channels, got {channels}")
        if tuple(privileged_ego.shape[:2]) != (batch_size, sequence_length):
            raise ValueError("privileged_ego must share [B,T] with privileged_bev")
        if privileged_ego.shape[-1] != self.cfg.ego_dim:
            raise ValueError(f"expected ego dim {self.cfg.ego_dim}, got {privileged_ego.shape[-1]}")

        bev_flat = privileged_bev.reshape(batch_size * sequence_length, channels, height, width)
        ego_flat = privileged_ego.reshape(batch_size * sequence_length, self.cfg.ego_dim)
        encoded = torch.cat([self.bev_encoder(bev_flat), self.ego_encoder(ego_flat)], dim=-1)
        embeds = self.project(encoded)
        return embeds.reshape(batch_size, sequence_length, self.cfg.embed_dim)
