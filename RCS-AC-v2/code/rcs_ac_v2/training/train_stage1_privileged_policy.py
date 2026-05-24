"""Local privileged policy smoke before Windows CARLA clean eval."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from rcs_ac_v2.data.schema import make_synthetic_privileged_batch
from rcs_ac_v2.privileged_stream.policy import (
    PrivilegedPolicy,
    PrivilegedPolicyConfig,
    imagine_policy_rollout,
)
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.policy.behavior_loss import BehaviorLossConfig, compute_behavior_loss
from rcs_ac_v2.utils.checkpoint import save_checkpoint, sha256_file
from rcs_ac_v2.utils.seed import seed_everything


@dataclass
class PrivilegedPolicySmokeResult:
    initial_critic_loss: float
    final_critic_loss: float
    final_total_loss: float
    final_entropy: float
    rollout_reward_mean: float
    rollout_continue_mean: float
    valid_action_ids: bool
    metrics: dict[str, float]
    world_model_checkpoint_path: Path | None = None
    policy_checkpoint_path: Path | None = None
    manifest_path: Path | None = None
    checkpoint_manifest: dict[str, object] | None = None


def run_privileged_policy_smoke(
    *,
    steps: int = 30,
    seed: int = 13,
    learning_rate: float = 3e-3,
    bev_height: int = 8,
    bev_width: int = 8,
    ego_dim: int = 8,
    hidden_dim: int = 64,
    output_dir: str | Path | None = None,
    export_checkpoints: bool = False,
) -> PrivilegedPolicySmokeResult:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    seed_everything(seed)
    wm_cfg = PrivilegedWorldModelConfig(
        bev_height=bev_height,
        bev_width=bev_width,
        ego_dim=ego_dim,
        hidden_dim=hidden_dim,
    )
    world_model = PrivilegedWorldModel(wm_cfg)
    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    batch = make_synthetic_privileged_batch(
        batch_size=3,
        sequence_length=5,
        height=wm_cfg.bev_height,
        width=wm_cfg.bev_width,
        ego_dim=wm_cfg.ego_dim,
        seed=seed,
    )
    with torch.no_grad():
        initial_state = world_model(batch).rssm_sequence.final_state()

    policy_cfg = PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=hidden_dim)
    policy = PrivilegedPolicy(policy_cfg)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    loss_cfg = BehaviorLossConfig(lambda_=1.0, actor_weight=0.1, entropy_weight=1e-3)
    critic_history: list[float] = []
    latest_metrics: dict[str, float] = {}
    latest_rollout_reward_mean = 0.0
    latest_rollout_continue_mean = 0.0

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rollout = imagine_policy_rollout(
            world_model=world_model,
            policy=policy,
            initial_state=initial_state,
            horizon=batch.action.shape[1],
            deterministic=False,
        )
        losses = compute_behavior_loss(
            logits=rollout.logits,
            value=rollout.value,
            action=rollout.action,
            reward=rollout.reward,
            cont=rollout.cont,
            bootstrap=torch.zeros(batch.reward.shape[0]),
            cfg=loss_cfg,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        latest_metrics = losses.detached_scalars()
        latest_rollout_reward_mean = float(rollout.reward.detach().mean().cpu())
        latest_rollout_continue_mean = float(rollout.cont.detach().mean().cpu())
        critic_history.append(latest_metrics["critic"])

    with torch.no_grad():
        final_rollout = imagine_policy_rollout(
            world_model=world_model,
            policy=policy,
            initial_state=initial_state,
            horizon=batch.action.shape[1],
            deterministic=True,
        )
        valid = bool(
            (final_rollout.action >= 0).all() and (final_rollout.action < wm_cfg.action_dim).all()
        )
    world_model_checkpoint_path: Path | None = None
    policy_checkpoint_path: Path | None = None
    manifest_path: Path | None = None
    manifest: dict[str, object] | None = None
    if export_checkpoints:
        if output_dir is None:
            raise ValueError("output_dir is required when export_checkpoints=True")
        export_dir = Path(output_dir)
        world_model_checkpoint_path = save_checkpoint(
            export_dir / "stage2_privileged_world_model.pt",
            {
                "model_state_dict": world_model.state_dict(),
                "config": wm_cfg.__dict__,
                "stage": "stage2_privileged_policy_clean",
                "checkpoint_role": "world_model",
                "metrics": latest_metrics,
                "seed": int(seed),
            },
        )
        policy_checkpoint_path = save_checkpoint(
            export_dir / "stage2_privileged_policy.pt",
            {
                "model_state_dict": policy.state_dict(),
                "config": policy_cfg.__dict__,
                "stage": "stage2_privileged_policy_clean",
                "checkpoint_role": "policy",
                "metrics": latest_metrics,
                "seed": int(seed),
                "valid_action_ids": valid,
            },
        )
        manifest = {
            "stage": "stage2_privileged_policy_clean",
            "source": "local_synthetic_smoke_not_carla",
            "clean_gate_evidence": False,
            "seed": int(seed),
            "steps": int(steps),
            "world_model": {
                "path": str(world_model_checkpoint_path),
                "sha256": sha256_file(world_model_checkpoint_path),
                "config": wm_cfg.__dict__,
            },
            "policy": {
                "path": str(policy_checkpoint_path),
                "sha256": sha256_file(policy_checkpoint_path),
                "config": policy_cfg.__dict__,
            },
            "metrics": {
                "initial_critic_loss": critic_history[0],
                "final_critic_loss": critic_history[-1],
                "final_total_loss": latest_metrics["total"],
                "final_entropy": latest_metrics["entropy"],
                "rollout_reward_mean": latest_rollout_reward_mean,
                "rollout_continue_mean": latest_rollout_continue_mean,
                "valid_action_ids": valid,
            },
            "clean_eval_config_patch": {
                "checkpoints": {
                    "world_model": {
                        "path": str(world_model_checkpoint_path),
                        "sha256": sha256_file(world_model_checkpoint_path),
                        "commit": None,
                    },
                    "policy": {
                        "path": str(policy_checkpoint_path),
                        "sha256": sha256_file(policy_checkpoint_path),
                        "commit": None,
                    },
                }
            },
            "warning": "Synthetic smoke checkpoint only; not clean-performance evidence.",
        }
        manifest_path = export_dir / "stage2_privileged_checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return PrivilegedPolicySmokeResult(
        initial_critic_loss=critic_history[0],
        final_critic_loss=critic_history[-1],
        final_total_loss=latest_metrics["total"],
        final_entropy=latest_metrics["entropy"],
        rollout_reward_mean=latest_rollout_reward_mean,
        rollout_continue_mean=latest_rollout_continue_mean,
        valid_action_ids=valid,
        metrics=latest_metrics,
        world_model_checkpoint_path=world_model_checkpoint_path,
        policy_checkpoint_path=policy_checkpoint_path,
        manifest_path=manifest_path,
        checkpoint_manifest=manifest,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2 privileged policy smoke/checkpoint export")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--bev-height", type=int, default=16)
    parser.add_argument("--bev-width", type=int, default=16)
    parser.add_argument("--ego-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--export-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    result = run_privileged_policy_smoke(
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        bev_height=args.bev_height,
        bev_width=args.bev_width,
        ego_dim=args.ego_dim,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir or None,
        export_checkpoints=args.export_checkpoints,
    )
    print(
        {
            "initial_critic_loss": result.initial_critic_loss,
            "final_critic_loss": result.final_critic_loss,
            "final_total_loss": result.final_total_loss,
            "final_entropy": result.final_entropy,
            "rollout_reward_mean": result.rollout_reward_mean,
            "rollout_continue_mean": result.rollout_continue_mean,
            "valid_action_ids": result.valid_action_ids,
            "world_model_checkpoint_path": (
                str(result.world_model_checkpoint_path) if result.world_model_checkpoint_path else None
            ),
            "policy_checkpoint_path": str(result.policy_checkpoint_path) if result.policy_checkpoint_path else None,
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
            **result.metrics,
        }
    )


if __name__ == "__main__":
    main()
