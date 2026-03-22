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

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features_by_modality: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ordered_names = list(features_by_modality.keys())
        stacked = torch.stack([features_by_modality[name] for name in ordered_names], dim=1)
        scores = self.score_net(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        aux = {
            "attention_weights": weights,
            "ordered_modalities": ordered_names,
        }
        return fused, aux

