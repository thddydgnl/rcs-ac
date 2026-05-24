"""Synthetic Stage 1 privileged world-model smoke training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rcs_ac_v2.data.schema import make_synthetic_privileged_batch
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.utils.checkpoint import save_checkpoint
from rcs_ac_v2.utils.seed import seed_everything
from rcs_ac_v2.world_model.losses import WorldModelLossConfig, compute_world_model_loss


@dataclass
class SyntheticOverfitResult:
    initial_loss: float
    final_loss: float
    loss_history: list[float]
    checkpoint_path: Path | None
    metrics: dict[str, float]


def run_synthetic_overfit_smoke(
    *,
    output_dir: str | Path | None = None,
    steps: int = 30,
    seed: int = 7,
    learning_rate: float = 3e-3,
) -> SyntheticOverfitResult:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    seed_everything(seed)
    cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=64)
    model = PrivilegedWorldModel(cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=4,
        height=cfg.bev_height,
        width=cfg.bev_width,
        ego_dim=cfg.ego_dim,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_cfg = WorldModelLossConfig()
    loss_history: list[float] = []
    latest_metrics: dict[str, float] = {}

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = compute_world_model_loss(
            predictions=output.predictions,
            rssm_sequence=output.rssm_sequence,
            batch=batch,
            cfg=loss_cfg,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        latest_metrics = losses.detached_scalars()
        loss_history.append(latest_metrics["total"])

    checkpoint_path: Path | None = None
    if output_dir is not None:
        checkpoint_path = Path(output_dir) / "stage1_privileged_world_model_smoke.pt"
        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "metrics": latest_metrics,
                "loss_history": loss_history,
            },
        )

    return SyntheticOverfitResult(
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=loss_history,
        checkpoint_path=checkpoint_path,
        metrics=latest_metrics,
    )


def main() -> None:
    result = run_synthetic_overfit_smoke(output_dir="RCS-AC-v2/experiments/01_privileged_world_model")
    print(
        {
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
            **result.metrics,
        }
    )


if __name__ == "__main__":
    main()
