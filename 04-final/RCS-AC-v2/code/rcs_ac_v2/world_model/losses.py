"""Losses for Stage 1 privileged world-model smoke."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rcs_ac_v2.data.schema import PrivilegedSequenceBatch
from rcs_ac_v2.world_model.rssm import RSSMSequence


@dataclass(frozen=True)
class WorldModelLossConfig:
    decoder_weight: float = 1.0
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    dynamics_weight: float = 1.0
    representation_weight: float = 1.0
    free_nats: float = 1.0


@dataclass
class WorldModelLossBreakdown:
    total: torch.Tensor
    decoder: torch.Tensor
    reward: torch.Tensor
    continue_loss: torch.Tensor
    dynamics: torch.Tensor
    representation: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "decoder": float(self.decoder.detach().cpu()),
            "reward": float(self.reward.detach().cpu()),
            "continue": float(self.continue_loss.detach().cpu()),
            "dynamics": float(self.dynamics.detach().cpu()),
            "representation": float(self.representation.detach().cpu()),
        }


def compute_world_model_loss(
    *,
    predictions: dict[str, torch.Tensor],
    rssm_sequence: RSSMSequence,
    batch: PrivilegedSequenceBatch,
    cfg: WorldModelLossConfig | None = None,
) -> WorldModelLossBreakdown:
    cfg = cfg or WorldModelLossConfig()
    decoder = F.binary_cross_entropy_with_logits(
        predictions["bev_logits"],
        batch.privileged_bev.float(),
    )
    reward = F.mse_loss(predictions["reward"], batch.reward.float())
    continue_loss = F.binary_cross_entropy_with_logits(
        predictions["continue_logits"],
        batch.cont.float(),
    )
    dynamics = free_nats_loss(
        kl_diag_gaussian(
            mean_a=rssm_sequence.post_mean.detach(),
            std_a=rssm_sequence.post_std.detach(),
            mean_b=rssm_sequence.prior_mean,
            std_b=rssm_sequence.prior_std,
        ),
        cfg.free_nats,
    )
    representation = free_nats_loss(
        kl_diag_gaussian(
            mean_a=rssm_sequence.post_mean,
            std_a=rssm_sequence.post_std,
            mean_b=rssm_sequence.prior_mean.detach(),
            std_b=rssm_sequence.prior_std.detach(),
        ),
        cfg.free_nats,
    )
    total = (
        cfg.decoder_weight * decoder
        + cfg.reward_weight * reward
        + cfg.continue_weight * continue_loss
        + cfg.dynamics_weight * dynamics
        + cfg.representation_weight * representation
    )
    return WorldModelLossBreakdown(
        total=total,
        decoder=decoder,
        reward=reward,
        continue_loss=continue_loss,
        dynamics=dynamics,
        representation=representation,
    )


def kl_diag_gaussian(
    *,
    mean_a: torch.Tensor,
    std_a: torch.Tensor,
    mean_b: torch.Tensor,
    std_b: torch.Tensor,
) -> torch.Tensor:
    var_a = std_a.pow(2)
    var_b = std_b.pow(2)
    kl = torch.log(std_b / std_a) + (var_a + (mean_a - mean_b).pow(2)) / (2.0 * var_b) - 0.5
    return kl.sum(dim=-1)


def free_nats_loss(kl: torch.Tensor, free_nats: float) -> torch.Tensor:
    if free_nats < 0:
        raise ValueError("free_nats must be non-negative")
    return torch.clamp(kl, min=free_nats).mean()
