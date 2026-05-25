"""Offline latent separability probe for Stage 2 privileged replay."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rcs_ac_v2.evaluation.checkpoint_action_probe import build_checkpoint_filled_config, _resolve_path
from rcs_ac_v2.evaluation.carla_adapter import load_stage2_policy_runner
from rcs_ac_v2.training.privileged_replay import validate_replay_manifest
from rcs_ac_v2.utils.checkpoint import sha256_file


def probe_replay_latent_separability(
    *,
    base_config_path: str | Path,
    checkpoint_manifest_path: str | Path,
    replay_manifest_path: str | Path,
    project_root: str | Path | None = None,
    train_steps: int = 300,
    learning_rate: float = 1e-2,
    seed: int = 20260524,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    torch.manual_seed(int(seed))
    root = Path(project_root) if project_root is not None else Path.cwd()
    manifest_path = Path(checkpoint_manifest_path)
    replay_manifest_resolved = _resolve_path(
        replay_manifest_path,
        root=root,
        fallback_dir=manifest_path.parent,
    )
    replay_manifest = validate_replay_manifest(replay_manifest_resolved)
    config = build_checkpoint_filled_config(
        base_config_path=base_config_path,
        checkpoint_manifest_path=manifest_path,
    )
    config["device"] = device
    runner = load_stage2_policy_runner(config)
    runner.world_model.eval()

    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for file_entry in replay_manifest["files"]:
        replay_path = _resolve_path(file_entry["path"], root=root, fallback_dir=replay_manifest_resolved.parent)
        with np.load(replay_path, allow_pickle=False) as data:
            bev_np = np.asarray(data["privileged_bev"], dtype=np.float32)
            ego_np = np.asarray(data["privileged_ego"], dtype=np.float32)
            action_np = np.asarray(data["action"], dtype=np.int64)
        bev = torch.as_tensor(bev_np[None, ...], dtype=torch.float32, device=runner.device)
        ego = torch.as_tensor(ego_np[None, ...], dtype=torch.float32, device=runner.device)
        actions = torch.as_tensor(action_np[None, ...], dtype=torch.int64, device=runner.device)
        with torch.no_grad():
            embeds = runner.world_model.encoder(bev, ego)
            sequence = runner.world_model.rssm.observe(embeds, actions)
            latents.append(sequence.latent[0].detach().cpu())
            labels.append(actions[0].detach().cpu())

    x = torch.cat(latents, dim=0)
    y = torch.cat(labels, dim=0)
    if x.shape[0] < 4:
        raise ValueError("latent probe requires at least 4 replay steps")
    permutation = torch.randperm(x.shape[0])
    split = max(1, int(0.8 * x.shape[0]))
    train_idx = permutation[:split]
    test_idx = permutation[split:]
    if test_idx.numel() == 0:
        test_idx = train_idx

    classifier = torch.nn.Linear(x.shape[-1], 39)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=float(learning_rate))
    for _ in range(int(train_steps)):
        logits = classifier(x[train_idx])
        loss = torch.nn.functional.cross_entropy(logits, y[train_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = classifier(x[train_idx]).argmax(dim=-1)
        test_pred = classifier(x[test_idx]).argmax(dim=-1)
    train_acc = float((train_pred == y[train_idx]).float().mean().item())
    test_acc = float((test_pred == y[test_idx]).float().mean().item())
    counts = Counter(int(item) for item in y.tolist())
    majority_count = max(counts.values()) if counts else 0
    majority_fraction = float(majority_count) / float(y.numel()) if y.numel() else 0.0
    result = {
        "stage": "stage2_privileged_policy_clean",
        "source": "stage2_replay_latent_probe",
        "clean_gate_evidence": False,
        "checkpoint_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "replay_manifest": {
            "path": str(replay_manifest_resolved),
            "sha256": sha256_file(replay_manifest_resolved),
        },
        "samples": int(y.numel()),
        "latent_dim": int(x.shape[-1]),
        "action_counts": {str(key): int(counts[key]) for key in sorted(counts)},
        "majority_action_fraction": majority_fraction,
        "linear_probe_train_accuracy": train_acc,
        "linear_probe_test_accuracy": test_acc,
        "train_steps": int(train_steps),
        "learning_rate": float(learning_rate),
        "interpretation": (
            "If linear_probe_test_accuracy is near majority_action_fraction, the world-model latent "
            "does not separate replay actions well enough for the current policy objective."
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe whether Stage 2 replay latents separate action labels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--project-root", default="")
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    result = probe_replay_latent_separability(
        base_config_path=args.config,
        checkpoint_manifest_path=args.checkpoint_manifest,
        replay_manifest_path=args.replay_manifest,
        project_root=args.project_root or None,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_path=args.output or None,
    )
    print(f"STAGE2_REPLAY_LATENT_PROBE={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
