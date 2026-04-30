from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rcs_ac_first.runtime.auxiliary import AuxiliaryObservationSpec, build_auxiliary_observation_spec, split_flat_auxiliary_observation
from rcs_ac_first.runtime.phase2_targets import risk_prior_from_auxiliary_tensor


@dataclass
class RiskContextConfig:
    risk_stats_dim: int = 6
    route_commitment_dim: int = 5
    sensor_dropout_dim: int = 4
    reliability_dim: int = 4
    hidden_dim: int = 64
    context_dim: int = 64

    @property
    def input_dim(self) -> int:
        return int(self.risk_stats_dim + self.route_commitment_dim + self.sensor_dropout_dim + self.reliability_dim)


class RiskContextHead(nn.Module):
    def __init__(self, config: RiskContextConfig | None = None, spec: AuxiliaryObservationSpec | None = None) -> None:
        super().__init__()
        self.config = config or RiskContextConfig()
        self.spec = spec or build_auxiliary_observation_spec(
            risk_stats_dim=self.config.risk_stats_dim,
            route_commitment_stats_dim=self.config.route_commitment_dim,
            sensor_dropout_dim=self.config.sensor_dropout_dim,
        )
        self.context = nn.Sequential(
            nn.Linear(self.config.input_dim, int(self.config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.hidden_dim), int(self.config.context_dim)),
            nn.Tanh(),
        )
        self.risk_residual = nn.Sequential(
            nn.Linear(self.config.input_dim, int(self.config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.hidden_dim), 1),
        )
        nn.init.zeros_(self.risk_residual[-1].weight)
        nn.init.zeros_(self.risk_residual[-1].bias)

    def forward(self, auxiliary: torch.Tensor, reliability: torch.Tensor) -> dict[str, torch.Tensor]:
        if auxiliary.ndim == 1:
            auxiliary = auxiliary.unsqueeze(0)
        if reliability.ndim != 2 or int(reliability.shape[-1]) != int(self.config.reliability_dim):
            raise ValueError(f"Expected reliability shape [B, 4], got {tuple(reliability.shape)}")
        split = split_flat_auxiliary_observation(auxiliary, self.spec)
        inputs = torch.cat(
            [
                split["risk_stats"],
                split["route_commitment_stats"],
                split["sensor_dropout_mask"],
                reliability.detach(),
            ],
            dim=-1,
        )
        prior = risk_prior_from_auxiliary_tensor(auxiliary, spec=self.spec)
        risk_scalar = (prior + 0.05 * torch.tanh(self.risk_residual(inputs).squeeze(-1))).clamp(0.0, 1.0)
        risk_logits = torch.logit(risk_scalar.clamp(1.0e-4, 1.0 - 1.0e-4))
        return {
            "risk_scalar": risk_scalar,
            "risk_context": self.context(inputs),
            "risk_logits": risk_logits,
            "risk_prior": prior,
        }
