from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MaskedCriticConfig:
    fusion_dim: int = 128
    risk_context_dim: int = 64
    hidden_dim: int = 256


class MaskedCritic(nn.Module):
    def __init__(self, config: MaskedCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskedCriticConfig()
        input_dim = int(self.config.fusion_dim) + 4 + 4 + 1 + int(self.config.risk_context_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(self.config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.hidden_dim), int(self.config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.hidden_dim), 1),
        )

    def forward(
        self,
        fused_feature: torch.Tensor,
        modality_mask: torch.Tensor,
        reliability: torch.Tensor,
        risk_scalar: torch.Tensor,
        risk_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(fused_feature.shape[0])
        if tuple(fused_feature.shape) != (batch_size, int(self.config.fusion_dim)):
            raise ValueError(f"fused_feature shape mismatch: {tuple(fused_feature.shape)}")
        if tuple(modality_mask.shape) != (batch_size, 4):
            raise ValueError(f"modality_mask shape mismatch: {tuple(modality_mask.shape)}")
        if tuple(reliability.shape) != (batch_size, 4):
            raise ValueError(f"reliability shape mismatch: {tuple(reliability.shape)}")
        risk_scalar = risk_scalar.reshape(batch_size, 1)
        if tuple(risk_context.shape) != (batch_size, int(self.config.risk_context_dim)):
            raise ValueError(f"risk_context shape mismatch: {tuple(risk_context.shape)}")
        x = torch.cat(
            [
                fused_feature,
                modality_mask.to(dtype=fused_feature.dtype, device=fused_feature.device),
                reliability.to(dtype=fused_feature.dtype, device=fused_feature.device),
                risk_scalar.to(dtype=fused_feature.dtype, device=fused_feature.device),
                risk_context.to(dtype=fused_feature.dtype, device=fused_feature.device),
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)
