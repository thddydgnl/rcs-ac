"""Train Stage 2 privileged checkpoints from CARLA privileged replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from rcs_ac_v2.policy.behavior_loss import BehaviorLossConfig, compute_behavior_loss
from rcs_ac_v2.privileged_stream.policy import PrivilegedPolicy, PrivilegedPolicyConfig, imagine_policy_rollout
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.training.privileged_replay import (
    REPLAY_SOURCE,
    load_replay_episodes,
    PrivilegedReplaySampler,
    validate_replay_manifest,
)
from rcs_ac_v2.utils.checkpoint import save_checkpoint, sha256_file
from rcs_ac_v2.utils.seed import seed_everything
from rcs_ac_v2.world_model.losses import WorldModelLossConfig, compute_world_model_loss


@dataclass(frozen=True)
class ReplayTrainingResult:
    output_dir: Path
    world_model_checkpoint_path: Path
    policy_checkpoint_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def train_stage2_privileged_from_replay(
    *,
    replay_paths: list[str | Path],
    replay_manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 20260524,
    sequence_length: int = 16,
    batch_size: int = 8,
    world_model_steps: int = 200,
    policy_steps: int = 200,
    learning_rate: float = 3e-4,
    policy_learning_rate: float | None = None,
    hidden_dim: int = 128,
    device: str = "auto",
) -> ReplayTrainingResult:
    if world_model_steps <= 0 or policy_steps <= 0:
        raise ValueError("world_model_steps and policy_steps must be positive")
    seed_everything(seed)
    resolved_device = _resolve_device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay_manifest = validate_replay_manifest(replay_manifest_path, replay_paths=replay_paths)
    replay_files = replay_manifest["files"]
    episodes = load_replay_episodes(replay_paths)
    sampler = PrivilegedReplaySampler(
        episodes,
        sequence_length=sequence_length,
        seed=seed,
        device=resolved_device,
    )

    wm_cfg = PrivilegedWorldModelConfig(bev_height=16, bev_width=16, ego_dim=8, hidden_dim=hidden_dim)
    world_model = PrivilegedWorldModel(wm_cfg).to(resolved_device)
    wm_optimizer = torch.optim.AdamW(world_model.parameters(), lr=learning_rate)
    wm_loss_cfg = WorldModelLossConfig()
    wm_history: list[dict[str, float]] = []
    for _step in range(world_model_steps):
        batch = sampler.sample(batch_size)
        wm_optimizer.zero_grad(set_to_none=True)
        wm_output = world_model(batch)
        losses = compute_world_model_loss(
            predictions=wm_output.predictions,
            rssm_sequence=wm_output.rssm_sequence,
            batch=batch,
            cfg=wm_loss_cfg,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=10.0)
        wm_optimizer.step()
        wm_history.append(losses.detached_scalars())

    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)

    policy_lr = learning_rate if policy_learning_rate is None else policy_learning_rate
    policy_cfg = PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=hidden_dim)
    policy = PrivilegedPolicy(policy_cfg).to(resolved_device)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=policy_lr)
    behavior_cfg = BehaviorLossConfig(lambda_=1.0, actor_weight=0.1, entropy_weight=1e-3)
    policy_history: list[dict[str, float]] = []
    valid_action_ids = False
    for _step in range(policy_steps):
        batch = sampler.sample(batch_size)
        with torch.no_grad():
            initial_state = world_model(batch).rssm_sequence.final_state()
        policy_optimizer.zero_grad(set_to_none=True)
        rollout = imagine_policy_rollout(
            world_model=world_model,
            policy=policy,
            initial_state=initial_state,
            horizon=sequence_length,
            deterministic=False,
        )
        losses = compute_behavior_loss(
            logits=rollout.logits,
            value=rollout.value,
            action=rollout.action,
            reward=rollout.reward,
            cont=rollout.cont,
            bootstrap=torch.zeros(batch.reward.shape[0], device=resolved_device),
            cfg=behavior_cfg,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        policy_optimizer.step()
        policy_history.append(losses.detached_scalars())
        valid_action_ids = bool(
            (rollout.action.detach() >= 0).all()
            and (rollout.action.detach() < wm_cfg.action_dim).all()
            and torch.isfinite(rollout.logits.detach()).all()
        )

    checkpoint_common = {
        "stage": "stage2_privileged_policy_clean",
        "source": REPLAY_SOURCE,
        "clean_gate_evidence": False,
        "trainer": "train_stage2_privileged_from_replay",
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "batch_size": int(batch_size),
        "world_model_steps": int(world_model_steps),
        "policy_steps": int(policy_steps),
        "replay_manifest": {
            "path": str(replay_manifest_path),
            "sha256": sha256_file(Path(replay_manifest_path)),
        },
        "data_files": replay_files,
    }
    world_model_checkpoint_path = save_checkpoint(
        output / "stage2_privileged_world_model_replay.pt",
        {
            **checkpoint_common,
            "checkpoint_role": "world_model",
            "model_state_dict": world_model.state_dict(),
            "config": wm_cfg.__dict__,
            "metrics": {
                "initial_total": wm_history[0]["total"],
                "final_total": wm_history[-1]["total"],
                "final_decoder": wm_history[-1]["decoder"],
                "final_reward": wm_history[-1]["reward"],
                "final_continue": wm_history[-1]["continue"],
            },
        },
    )
    policy_checkpoint_path = save_checkpoint(
        output / "stage2_privileged_policy_replay.pt",
        {
            **checkpoint_common,
            "checkpoint_role": "policy",
            "model_state_dict": policy.state_dict(),
            "config": policy_cfg.__dict__,
            "metrics": {
                "initial_total": policy_history[0]["total"],
                "final_total": policy_history[-1]["total"],
                "final_actor": policy_history[-1]["actor"],
                "final_critic": policy_history[-1]["critic"],
                "final_entropy": policy_history[-1]["entropy"],
                "valid_action_ids": valid_action_ids,
            },
        },
    )
    manifest = {
        **checkpoint_common,
        "world_model": {
            "path": str(world_model_checkpoint_path),
            "sha256": sha256_file(world_model_checkpoint_path),
            "config": wm_cfg.__dict__,
            "metrics": {
                "initial_total": wm_history[0]["total"],
                "final_total": wm_history[-1]["total"],
            },
        },
        "policy": {
            "path": str(policy_checkpoint_path),
            "sha256": sha256_file(policy_checkpoint_path),
            "config": policy_cfg.__dict__,
            "metrics": {
                "initial_total": policy_history[0]["total"],
                "final_total": policy_history[-1]["total"],
                "valid_action_ids": valid_action_ids,
            },
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
        "warning": "Replay-trained checkpoints are inputs for clean eval, not clean PASS evidence.",
    }
    manifest_path = output / "stage2_privileged_replay_checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return ReplayTrainingResult(
        output_dir=output,
        world_model_checkpoint_path=world_model_checkpoint_path,
        policy_checkpoint_path=policy_checkpoint_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 privileged checkpoints from replay")
    parser.add_argument("--replay", action="append", required=True, help="Replay .npz file or directory; repeatable")
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--world-model-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = train_stage2_privileged_from_replay(
        replay_paths=[Path(item) for item in args.replay],
        replay_manifest_path=args.replay_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        world_model_steps=args.world_model_steps,
        policy_steps=args.policy_steps,
        learning_rate=args.learning_rate,
        policy_learning_rate=args.policy_learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"STAGE2_REPLAY_CHECKPOINT_MANIFEST={result.manifest_path}")
    print(f"STAGE2_REPLAY_WORLD_MODEL={result.world_model_checkpoint_path}")
    print(f"STAGE2_REPLAY_POLICY={result.policy_checkpoint_path}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


if __name__ == "__main__":
    main()
