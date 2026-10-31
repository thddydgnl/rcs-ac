from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rcs_ac_first.runtime.auxiliary import DEFAULT_AUXILIARY_DIM, AuxiliaryObservationSpec
from rcs_ac_first.runtime.phase2_targets import reliability_prior_from_auxiliary_tensor


@dataclass
class ReliabilityConfig:
    auxiliary_dim: int = DEFAULT_AUXILIARY_DIM
    hidden_dim: int = 64
    minimum: float = 0.02
    maximum: float = 1.0


class RCSACReliabilityModule(nn.Module):
    def __init__(self, config: ReliabilityConfig | None = None, spec: AuxiliaryObservationSpec | None = None) -> None:
        super().__init__()
        self.config = config or ReliabilityConfig()
        self.spec = spec
        self.residual = nn.Sequential(
            nn.Linear(int(self.config.auxiliary_dim), int(self.config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.hidden_dim), 4),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, auxiliary: torch.Tensor) -> dict[str, torch.Tensor]:
        if auxiliary.ndim == 1:
            auxiliary = auxiliary.unsqueeze(0)
        if auxiliary.ndim != 2 or int(auxiliary.shape[-1]) != int(self.config.auxiliary_dim):
            raise ValueError(
                f"Expected auxiliary shape [B, {int(self.config.auxiliary_dim)}], got {tuple(auxiliary.shape)}"
            )
        prior = reliability_prior_from_auxiliary_tensor(
            auxiliary,
            spec=self.spec,
            minimum=float(self.config.minimum),
        )
        residual = 0.05 * torch.tanh(self.residual(auxiliary))
        reliability = (prior + residual).clamp(float(self.config.minimum), float(self.config.maximum))
        logits = torch.logit(reliability.clamp(1.0e-4, 1.0 - 1.0e-4))
        return {
            "reliability": reliability,
            "reliability_logits": logits,
            "reliability_prior": prior,
        }
