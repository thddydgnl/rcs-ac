from __future__ import annotations

import torch

from rcs_ac_v2.data.schema import make_synthetic_privileged_batch
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.utils.checkpoint import load_checkpoint, save_checkpoint
from rcs_ac_v2.world_model.losses import compute_world_model_loss


def test_privileged_world_model_forward_shapes() -> None:
    cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    model = PrivilegedWorldModel(cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=4,
        height=8,
        width=8,
        ego_dim=8,
        seed=5,
    )
    output = model(batch)

    assert output.embeds.shape == (2, 4, cfg.embed_dim)
    assert output.rssm_sequence.latent.shape == (2, 4, cfg.deterministic_dim + cfg.stochastic_dim)
    assert output.predictions["bev_logits"].shape == (2, 4, 43, 8, 8)
    assert output.predictions["reward"].shape == (2, 4)
    assert output.predictions["continue_logits"].shape == (2, 4)
    assert torch.isfinite(output.predictions["bev_logits"]).all()


def test_privileged_ego_affects_encoded_latent() -> None:
    cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    torch.manual_seed(10)
    model = PrivilegedWorldModel(cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=3,
        height=8,
        width=8,
        ego_dim=8,
        seed=10,
    )
    changed = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=3,
        height=8,
        width=8,
        ego_dim=8,
        seed=10,
    )
    changed.privileged_ego = changed.privileged_ego + 3.0

    base_output = model(batch)
    changed_output = model(changed)

    assert not torch.allclose(base_output.embeds, changed_output.embeds)
    assert not torch.allclose(
        base_output.rssm_sequence.post_mean,
        changed_output.rssm_sequence.post_mean,
    )


def test_checkpoint_roundtrip_preserves_forward_loss(tmp_path) -> None:
    torch.manual_seed(123)
    cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    model = PrivilegedWorldModel(cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=3,
        height=8,
        width=8,
        ego_dim=8,
        seed=6,
    )
    output = model(batch)
    loss_before = compute_world_model_loss(
        predictions=output.predictions,
        rssm_sequence=output.rssm_sequence,
        batch=batch,
    ).total

    checkpoint_path = save_checkpoint(
        tmp_path / "roundtrip.pt",
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
    )
    payload = load_checkpoint(checkpoint_path)
    restored = PrivilegedWorldModel(PrivilegedWorldModelConfig(**payload["config"]))
    restored.load_state_dict(payload["model_state_dict"])
    restored_output = restored(batch)
    loss_after = compute_world_model_loss(
        predictions=restored_output.predictions,
        rssm_sequence=restored_output.rssm_sequence,
        batch=batch,
    ).total

    assert torch.allclose(loss_before, loss_after)


def test_privileged_world_model_rollout_from_final_state() -> None:
    cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    model = PrivilegedWorldModel(cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=3,
        height=8,
        width=8,
        ego_dim=8,
        seed=8,
    )
    future_actions = torch.randint(0, cfg.action_dim, (2, 2), dtype=torch.int64)
    rollout = model.rollout_from_batch_final_state(batch, future_actions)

    assert rollout.sequence.latent.shape == (2, 2, cfg.deterministic_dim + cfg.stochastic_dim)
    assert rollout.predictions["bev_logits"].shape == (2, 2, 43, 8, 8)
    assert torch.isfinite(rollout.predictions["reward"]).all()
    assert torch.isfinite(rollout.predictions["continue_logits"]).all()
