from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

MODALITY_ORDER: tuple[str, ...] = ("rgb", "lidar", "ego", "route")


@dataclass
class FusionGateConfig:
    feature_dim: int = 128
    fusion_dim: int = 128
    hidden_dim: int = 128
    temperature: float = 1.0
    route_weight_floor: float = 0.08
    ego_weight_floor: float = 0.05
    contribution_bias_scale: float = 0.0
    risk_context_dim: int = 64
    reliability_logit_scale: float = 1.0
    risk_logit_scale: float = 0.5
    use_risk_context: bool = False


@dataclass
class FusionGateOutput:
    fused: torch.Tensor
    weights: torch.Tensor
    logits: torch.Tensor
    projected_features: dict[str, torch.Tensor]
    diagnostics: dict[str, float]


class FusionGate(nn.Module):
    def __init__(self, config: FusionGateConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionGateConfig()
        feature_dim = int(self.config.feature_dim)
        fusion_dim = int(self.config.fusion_dim)
        hidden_dim = int(self.config.hidden_dim)
        gate_input_dim = fusion_dim + 3
        if bool(self.config.use_risk_context):
            gate_input_dim += int(self.config.risk_context_dim)
        self.projectors = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(feature_dim, fusion_dim),
                    nn.ReLU(),
                )
                for key in MODALITY_ORDER
            }
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        reliability: torch.Tensor | None = None,
        contribution: torch.Tensor | None = None,
        risk_scalar: torch.Tensor | None = None,
        risk_context: torch.Tensor | None = None,
        modality_mask: torch.Tensor | None = None,
    ) -> FusionGateOutput:
        missing = [key for key in MODALITY_ORDER if key not in features]
        if missing:
            raise KeyError(f"FusionGate missing modality features: {missing}")

        projected = {key: self.projectors[key](features[key]) for key in MODALITY_ORDER}
        batch_size = int(next(iter(projected.values())).shape[0])
        dtype = next(iter(projected.values())).dtype
        device = next(iter(projected.values())).device
        if reliability is None:
            reliability = torch.ones((batch_size, len(MODALITY_ORDER)), dtype=dtype, device=device)
        if contribution is None:
            contribution = torch.full(
                (batch_size, len(MODALITY_ORDER)),
                1.0 / float(len(MODALITY_ORDER)),
                dtype=dtype,
                device=device,
            )
        if risk_scalar is None:
            risk_scalar = torch.zeros((batch_size,), dtype=dtype, device=device)
        reliability = reliability.to(dtype=dtype, device=device).clamp(1.0e-4, 1.0)
        contribution = contribution.to(dtype=dtype, device=device)
        if tuple(contribution.shape) != (batch_size, len(MODALITY_ORDER)):
            raise ValueError(
                f"contribution shape mismatch: expected {(batch_size, len(MODALITY_ORDER))}, got {tuple(contribution.shape)}"
            )
        contribution = contribution.clamp_min(1.0e-8)
        contribution = contribution / contribution.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        risk_scalar = risk_scalar.to(dtype=dtype, device=device).reshape(batch_size, 1).clamp(0.0, 1.0)
        if modality_mask is None:
            modality_mask = torch.ones((batch_size, len(MODALITY_ORDER)), dtype=dtype, device=device)
        else:
            modality_mask = modality_mask.to(dtype=dtype, device=device)
            if modality_mask.ndim == 1:
                modality_mask = modality_mask.unsqueeze(0)
            if tuple(modality_mask.shape) != (batch_size, len(MODALITY_ORDER)):
                raise ValueError(
                    f"modality_mask shape mismatch: expected {(batch_size, len(MODALITY_ORDER))}, got {tuple(modality_mask.shape)}"
                )
            modality_mask = (modality_mask > 0.5).to(dtype=dtype)
        if bool((modality_mask.sum(dim=-1) <= 0.0).any()):
            raise ValueError("FusionGate.forward does not allow empty modality masks")
        risk_context_l1_mean = 0.0
        if bool(self.config.use_risk_context):
            if risk_context is None:
                risk_context = torch.zeros((batch_size, int(self.config.risk_context_dim)), dtype=dtype, device=device)
            risk_context = risk_context.to(dtype=dtype, device=device)
            if tuple(risk_context.shape) != (batch_size, int(self.config.risk_context_dim)):
                raise ValueError(
                    "risk_context shape mismatch: "
                    f"expected {(batch_size, int(self.config.risk_context_dim))}, got {tuple(risk_context.shape)}"
                )
            risk_context_l1_mean = float(risk_context.detach().abs().mean().cpu().item())
        else:
            risk_context = None

        gate_inputs: list[torch.Tensor] = []
        for index, key in enumerate(MODALITY_ORDER):
            pieces = [
                projected[key],
                reliability[:, index : index + 1],
                contribution[:, index : index + 1],
                risk_scalar,
            ]
            if risk_context is not None:
                pieces.append(risk_context)
            gate_inputs.append(torch.cat(pieces, dim=-1))
        logits = torch.cat([self.gate_mlp(item) for item in gate_inputs], dim=-1)
        reliability_bias = reliability.log()
        logits = logits + float(self.config.reliability_logit_scale) * reliability_bias
        if float(self.config.risk_logit_scale) != 0.0:
            logits = logits + float(self.config.risk_logit_scale) * risk_scalar * reliability_bias
        if float(self.config.contribution_bias_scale) != 0.0:
            logits = logits + float(self.config.contribution_bias_scale) * (
                contribution - (1.0 / float(len(MODALITY_ORDER)))
            )

        logits = logits.masked_fill(modality_mask <= 0.0, -1.0e9)
        raw_weights = torch.softmax(logits / float(self.config.temperature), dim=-1)
        weights = self._apply_safety_floors(raw_weights, modality_mask=modality_mask)
        stacked = torch.stack([projected[key] for key in MODALITY_ORDER], dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        diagnostics = self._build_diagnostics(
            fused=fused,
            weights=weights,
            raw_weights=raw_weights,
            reliability=reliability,
            contribution=contribution,
            risk_scalar=risk_scalar,
            risk_context_l1_mean=risk_context_l1_mean,
            modality_mask=modality_mask,
        )
        return FusionGateOutput(
            fused=fused,
            weights=weights,
            logits=logits,
            projected_features=projected,
            diagnostics=diagnostics,
        )

    def _apply_safety_floors(self, raw_weights: torch.Tensor, *, modality_mask: torch.Tensor) -> torch.Tensor:
        floor = torch.zeros_like(raw_weights)
        ego_index = MODALITY_ORDER.index("ego")
        route_index = MODALITY_ORDER.index("route")
        floor[:, ego_index] = float(self.config.ego_weight_floor) * modality_mask[:, ego_index]
        floor[:, route_index] = float(self.config.route_weight_floor) * modality_mask[:, route_index]
        remaining = (1.0 - floor.sum(dim=-1, keepdim=True)).clamp_min(1.0e-6)
        weights = floor + remaining * raw_weights * modality_mask
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _build_diagnostics(
        self,
        *,
        fused: torch.Tensor,
        weights: torch.Tensor,
        raw_weights: torch.Tensor,
        reliability: torch.Tensor,
        contribution: torch.Tensor,
        risk_scalar: torch.Tensor,
        risk_context_l1_mean: float,
        modality_mask: torch.Tensor,
    ) -> dict[str, float]:
        diagnostics = {
            "fused_feature_l1_mean": float(fused.detach().abs().mean().cpu().item()),
            "fusion_entropy": float(
                (-(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1)).mean().detach().cpu().item()
            ),
        }
        for index, key in enumerate(MODALITY_ORDER):
            diagnostics[f"fusion_weight_{key}"] = float(weights[:, index].mean().detach().cpu().item())
            diagnostics[f"reliability_{key}"] = float(reliability[:, index].mean().detach().cpu().item())
            diagnostics[f"contribution_{key}"] = float(contribution[:, index].mean().detach().cpu().item())
            diagnostics[f"modality_mask_{key}"] = float(modality_mask[:, index].mean().detach().cpu().item())
            diagnostics[f"shapley_raw_{key}"] = 0.0
        diagnostics["contribution_entropy"] = float(
            (-(contribution * contribution.clamp_min(1.0e-8).log()).sum(dim=-1)).mean().detach().cpu().item()
        )
        diagnostics["contribution_max"] = float(contribution.max(dim=-1).values.mean().detach().cpu().item())
        diagnostics["contribution_min"] = float(contribution.min(dim=-1).values.mean().detach().cpu().item())
        diagnostics["risk_scalar"] = float(risk_scalar.mean().detach().cpu().item())
        diagnostics["risk_scalar_mean"] = diagnostics["risk_scalar"]
        diagnostics["risk_context_l1_mean"] = float(risk_context_l1_mean)
        ego_index = MODALITY_ORDER.index("ego")
        route_index = MODALITY_ORDER.index("route")
        diagnostics["fusion_ego_floor_applied_rate"] = float(
            torch.logical_and(
                raw_weights[:, ego_index] < float(self.config.ego_weight_floor),
                modality_mask[:, ego_index] > 0.0,
            )
            .to(torch.float32)
            .mean()
            .cpu()
            .item()
        )
        diagnostics["fusion_route_floor_applied_rate"] = float(
            torch.logical_and(
                raw_weights[:, route_index] < float(self.config.route_weight_floor),
                modality_mask[:, route_index] > 0.0,
            )
            .to(torch.float32)
            .mean()
            .cpu()
            .item()
        )
        return diagnostics
