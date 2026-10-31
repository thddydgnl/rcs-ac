"""Rollout helpers for the privileged RSSM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rcs_ac_v2.world_model.heads import WorldModelHeads
from rcs_ac_v2.world_model.rssm import RSSM, RSSMSequence, RSSMState


@dataclass
class RolloutPrediction:
    sequence: RSSMSequence
    predictions: dict[str, torch.Tensor]


def rollout_prior(
    *,
    rssm: RSSM,
    heads: WorldModelHeads,
    initial_state: RSSMState,
    actions: torch.Tensor,
) -> RolloutPrediction:
    sequence = rssm.imagine(initial_state, actions)
    predictions = heads(sequence.latent)
    for name, tensor in predictions.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite rollout prediction: {name}")
    if not torch.isfinite(sequence.latent).all():
        raise ValueError("non-finite rollout latent")
    return RolloutPrediction(sequence=sequence, predictions=predictions)
