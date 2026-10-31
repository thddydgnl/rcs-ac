from __future__ import annotations

import torch
import pytest

from rcs_ac_v2.world_model.rollout import rollout_prior
from rcs_ac_v2.world_model.heads import WorldModelHeadConfig, WorldModelHeads
from rcs_ac_v2.world_model.rssm import RSSM, RSSMConfig


def test_rssm_observe_shapes() -> None:
    cfg = RSSMConfig(embed_dim=16, deterministic_dim=24, stochastic_dim=8, hidden_dim=32)
    rssm = RSSM(cfg)
    embeds = torch.randn(3, 5, cfg.embed_dim)
    actions = torch.randint(0, cfg.action_dim, (3, 5), dtype=torch.int64)

    sequence = rssm.observe(embeds, actions)

    assert sequence.deter.shape == (3, 5, cfg.deterministic_dim)
    assert sequence.stoch.shape == (3, 5, cfg.stochastic_dim)
    assert sequence.prior_mean.shape == (3, 5, cfg.stochastic_dim)
    assert sequence.prior_std.shape == (3, 5, cfg.stochastic_dim)
    assert sequence.post_mean.shape == (3, 5, cfg.stochastic_dim)
    assert sequence.post_std.shape == (3, 5, cfg.stochastic_dim)
    assert sequence.latent.shape == (3, 5, cfg.deterministic_dim + cfg.stochastic_dim)
    assert torch.isfinite(sequence.latent).all()
    assert torch.all(sequence.prior_std > 0)
    assert torch.all(sequence.post_std > 0)


def test_rssm_rejects_bad_action_ids() -> None:
    cfg = RSSMConfig(embed_dim=16)
    rssm = RSSM(cfg)
    embeds = torch.randn(2, 3, cfg.embed_dim)
    actions = torch.full((2, 3), cfg.action_dim, dtype=torch.int64)
    with pytest.raises(ValueError, match="action ids"):
        rssm.observe(embeds, actions)


def test_rssm_uses_previous_action_time_convention() -> None:
    cfg = RSSMConfig(embed_dim=16, deterministic_dim=24, stochastic_dim=8, hidden_dim=32)
    torch.manual_seed(4)
    rssm = RSSM(cfg)
    embeds = torch.randn(2, 3, cfg.embed_dim)
    base_actions = torch.zeros(2, 3, dtype=torch.int64)
    changed_actions = base_actions.clone()
    changed_actions[:, 0] = 1

    base = rssm.observe(embeds, base_actions)
    changed = rssm.observe(embeds, changed_actions)

    assert torch.allclose(base.deter[:, 0], changed.deter[:, 0])
    assert not torch.allclose(base.deter[:, 1], changed.deter[:, 1])


def test_rollout_prior_has_finite_latent_reward_continue() -> None:
    cfg = RSSMConfig(embed_dim=16, deterministic_dim=24, stochastic_dim=8, hidden_dim=32)
    rssm = RSSM(cfg)
    heads = WorldModelHeads(
        WorldModelHeadConfig(
            latent_dim=rssm.latent_dim,
            bev_channels=43,
            bev_height=8,
            bev_width=8,
            hidden_dim=32,
        )
    )
    embeds = torch.randn(2, 4, cfg.embed_dim)
    actions = torch.randint(0, cfg.action_dim, (2, 4), dtype=torch.int64)
    observed = rssm.observe(embeds, actions)
    future_actions = torch.randint(0, cfg.action_dim, (2, 2), dtype=torch.int64)

    rollout = rollout_prior(
        rssm=rssm,
        heads=heads,
        initial_state=observed.final_state(),
        actions=future_actions,
    )

    assert rollout.sequence.latent.shape == (2, 2, rssm.latent_dim)
    assert rollout.predictions["bev_logits"].shape == (2, 2, 43, 8, 8)
    assert rollout.predictions["reward"].shape == (2, 2)
    assert rollout.predictions["continue_logits"].shape == (2, 2)
    assert torch.isfinite(rollout.sequence.latent).all()
    assert torch.isfinite(rollout.predictions["reward"]).all()
    assert torch.isfinite(rollout.predictions["continue_logits"]).all()
