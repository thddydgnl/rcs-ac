"""Privileged world model for Stage 1 smoke."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rcs_ac_v2.data.schema import PrivilegedSequenceBatch, SchemaConfig, validate_privileged_batch
from rcs_ac_v2.env.discrete_actions import NUM_RAW2DRIVE_ACTIONS
from rcs_ac_v2.privileged_stream.encoder import PrivilegedEncoder, PrivilegedEncoderConfig
from rcs_ac_v2.world_model.heads import WorldModelHeadConfig, WorldModelHeads
from rcs_ac_v2.world_model.rssm import RSSM, RSSMConfig, RSSMSequence
from rcs_ac_v2.world_model.rollout import RolloutPrediction, rollout_prior


@dataclass(frozen=True)
class PrivilegedWorldModelConfig:
    bev_channels: int = 43
    bev_height: int = 16
    bev_width: int = 16
    ego_dim: int = 8
    action_dim: int = NUM_RAW2DRIVE_ACTIONS
    embed_dim: int = 128
    deterministic_dim: int = 128
    stochastic_dim: int = 32
    hidden_dim: int = 128


@dataclass
class PrivilegedWorldModelOutput:
    embeds: torch.Tensor
    rssm_sequence: RSSMSequence
    predictions: dict[str, torch.Tensor]


class PrivilegedWorldModel(nn.Module):
    def __init__(self, cfg: PrivilegedWorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PrivilegedWorldModelConfig()
        self.encoder = PrivilegedEncoder(
            PrivilegedEncoderConfig(
                bev_channels=self.cfg.bev_channels,
                ego_dim=self.cfg.ego_dim,
                embed_dim=self.cfg.embed_dim,
            )
        )
        self.rssm = RSSM(
            RSSMConfig(
                action_dim=self.cfg.action_dim,
                embed_dim=self.cfg.embed_dim,
                deterministic_dim=self.cfg.deterministic_dim,
                stochastic_dim=self.cfg.stochastic_dim,
                hidden_dim=self.cfg.hidden_dim,
            )
        )
        self.heads = WorldModelHeads(
            WorldModelHeadConfig(
                latent_dim=self.rssm.latent_dim,
                bev_channels=self.cfg.bev_channels,
                bev_height=self.cfg.bev_height,
                bev_width=self.cfg.bev_width,
                hidden_dim=self.cfg.hidden_dim,
            )
        )

    def forward(self, batch: PrivilegedSequenceBatch) -> PrivilegedWorldModelOutput:
        validate_privileged_batch(
            batch,
            SchemaConfig(
                privileged_bev_channels=self.cfg.bev_channels,
                ego_dim=self.cfg.ego_dim,
                num_actions=self.cfg.action_dim,
            ),
        )
        height, width = batch.privileged_bev.shape[-2:]
        if (height, width) != (self.cfg.bev_height, self.cfg.bev_width):
            raise ValueError(
                f"batch BEV size must be {(self.cfg.bev_height, self.cfg.bev_width)}, got {(height, width)}"
            )
        embeds = self.encoder(batch.privileged_bev, batch.privileged_ego)
        rssm_sequence = self.rssm.observe(embeds, batch.action)
        predictions = self.heads(rssm_sequence.latent)
        return PrivilegedWorldModelOutput(
            embeds=embeds,
            rssm_sequence=rssm_sequence,
            predictions=predictions,
        )

    def rollout_from_batch_final_state(
        self,
        batch: PrivilegedSequenceBatch,
        future_actions: torch.Tensor,
    ) -> RolloutPrediction:
        output = self.forward(batch)
        return rollout_prior(
            rssm=self.rssm,
            heads=self.heads,
            initial_state=output.rssm_sequence.final_state(),
            actions=future_actions,
        )
