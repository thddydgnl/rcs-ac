from __future__ import annotations

import torch

from rcs_ac_v2.data.schema import make_synthetic_privileged_batch
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.training.train_stage1_privileged_world_model import run_synthetic_overfit_smoke
from rcs_ac_v2.world_model.losses import (
    WorldModelLossConfig,
    compute_world_model_loss,
    free_nats_loss,
    kl_diag_gaussian,
)


def test_world_model_loss_breakdown_is_finite() -> None:
    cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    model = PrivilegedWorldModel(cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=3,
        height=8,
        width=8,
        ego_dim=8,
    )
    output = model(batch)
    losses = compute_world_model_loss(
        predictions=output.predictions,
        rssm_sequence=output.rssm_sequence,
        batch=batch,
    )

    for value in losses.detached_scalars().values():
        assert torch.isfinite(torch.tensor(value))
    assert losses.total.requires_grad
    assert losses.decoder > 0
    assert losses.continue_loss > 0


def test_free_nats_clamps_kl_floor() -> None:
    mean_a = torch.zeros(2, 3, 4)
    std_a = torch.ones(2, 3, 4)
    mean_b = torch.zeros(2, 3, 4)
    std_b = torch.ones(2, 3, 4)
    kl = kl_diag_gaussian(mean_a=mean_a, std_a=std_a, mean_b=mean_b, std_b=std_b)
    assert torch.allclose(kl, torch.zeros(2, 3))
    assert free_nats_loss(kl, free_nats=1.0).item() == 1.0


def test_kl_stop_gradient_directions() -> None:
    torch.manual_seed(9)
    post_mean = torch.randn(2, 3, 4, requires_grad=True)
    post_std = (torch.rand(2, 3, 4) + 0.5).requires_grad_()
    prior_mean = torch.randn(2, 3, 4, requires_grad=True)
    prior_std = (torch.rand(2, 3, 4) + 0.5).requires_grad_()

    dyn = free_nats_loss(
        kl_diag_gaussian(
            mean_a=post_mean.detach(),
            std_a=post_std.detach(),
            mean_b=prior_mean,
            std_b=prior_std,
        ),
        free_nats=0.0,
    )
    dyn.backward()
    assert post_mean.grad is None
    assert post_std.grad is None
    assert prior_mean.grad is not None
    assert prior_std.grad is not None

    post_mean = torch.randn(2, 3, 4, requires_grad=True)
    post_std = (torch.rand(2, 3, 4) + 0.5).requires_grad_()
    prior_mean = torch.randn(2, 3, 4, requires_grad=True)
    prior_std = (torch.rand(2, 3, 4) + 0.5).requires_grad_()
    rep = free_nats_loss(
        kl_diag_gaussian(
            mean_a=post_mean,
            std_a=post_std,
            mean_b=prior_mean.detach(),
            std_b=prior_std.detach(),
        ),
        free_nats=0.0,
    )
    rep.backward()
    assert post_mean.grad is not None
    assert post_std.grad is not None
    assert prior_mean.grad is None
    assert prior_std.grad is None


def test_synthetic_one_episode_overfit_loss_decreases(tmp_path) -> None:
    result = run_synthetic_overfit_smoke(output_dir=tmp_path, steps=18, seed=11)
    assert result.final_loss < result.initial_loss
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["decoder"] > 0
    assert result.metrics["reward"] >= 0
    assert result.metrics["continue"] > 0
