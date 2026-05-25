"""Replay-observation action probe for Stage 2 clean checkpoint promotion."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
import yaml

from rcs_ac_v2.env.discrete_actions import ACTION_TABLE_VERSION, RAW2DRIVE_ACTIONS, get_discrete_action
from rcs_ac_v2.evaluation.carla_adapter import load_stage2_policy_runner
from rcs_ac_v2.training.privileged_replay import validate_replay_manifest
from rcs_ac_v2.utils.checkpoint import sha256_file


def build_checkpoint_filled_config(
    *,
    base_config_path: str | Path,
    checkpoint_manifest_path: str | Path,
    output_config_path: str | Path | None = None,
) -> dict[str, Any]:
    base_path = Path(base_config_path)
    manifest_path = Path(checkpoint_manifest_path)
    config = yaml.safe_load(base_path.read_text())
    checkpoint_manifest = _load_checkpoint_manifest(manifest_path)
    config["dry_run"]["allow_null_checkpoints"] = False
    config["checkpoints"]["world_model"].update(checkpoint_manifest["clean_eval_config_patch"]["checkpoints"]["world_model"])
    config["checkpoints"]["policy"].update(checkpoint_manifest["clean_eval_config_patch"]["checkpoints"]["policy"])
    git_commit = str(checkpoint_manifest.get("git_commit", "UNSET"))
    if git_commit != "UNSET":
        config["checkpoints"]["world_model"]["commit"] = git_commit
        config["checkpoints"]["policy"]["commit"] = git_commit
    config["checkpoint_training_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    if output_config_path is not None:
        target = Path(output_config_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def probe_checkpoint_actions(
    *,
    base_config_path: str | Path,
    checkpoint_manifest_path: str | Path,
    replay_manifest_path: str | Path | None = None,
    project_root: str | Path | None = None,
    max_steps_per_file: int = 32,
    min_nonzero_throttle_fraction: float = 0.5,
    min_nonzero_steer_fraction: float = 0.02,
    min_nonzero_steer_count: int = 12,
    max_nonzero_steer_fraction: float = 0.5,
    max_dominant_action_fraction: float = 0.85,
    max_previous_stuck_action8_fraction: float = 0.05,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_steps_per_file <= 0:
        raise ValueError("max_steps_per_file must be positive")
    manifest_path = Path(checkpoint_manifest_path)
    checkpoint_manifest = _load_checkpoint_manifest(manifest_path)
    replay_manifest_entry = checkpoint_manifest.get("replay_manifest", {})
    replay_manifest_raw = replay_manifest_path or replay_manifest_entry.get("path")
    if not replay_manifest_raw:
        raise ValueError("replay_manifest_path is required")
    root = Path(project_root) if project_root is not None else Path.cwd()
    replay_manifest_resolved = _resolve_path(replay_manifest_raw, root=root, fallback_dir=manifest_path.parent)
    replay_manifest = validate_replay_manifest(replay_manifest_resolved)

    config = build_checkpoint_filled_config(
        base_config_path=base_config_path,
        checkpoint_manifest_path=manifest_path,
    )
    config["device"] = device
    runner = load_stage2_policy_runner(config)

    action_counts: Counter[int] = Counter()
    label_action_counts: Counter[int] = Counter()
    reverse_count = 0
    nonzero_throttle_count = 0
    moving_steer_count = 0
    label_moving_steer_count = 0
    action_match_count = 0
    predicted_moving_on_label_nonmoving_count = 0
    predicted_nonmoving_on_label_moving_count = 0
    left_moving_steer_count = 0
    right_moving_steer_count = 0
    zero_throttle_steer_count = 0
    zero_throttle_count = 0
    probed_files: list[dict[str, Any]] = []
    for file_entry in replay_manifest["files"]:
        replay_path = _resolve_path(file_entry["path"], root=root, fallback_dir=replay_manifest_resolved.parent)
        with np.load(replay_path, allow_pickle=False) as data:
            bev = np.asarray(data["privileged_bev"], dtype=np.float32)
            ego = np.asarray(data["privileged_ego"], dtype=np.float32)
            replay_action = np.asarray(data["action"], dtype=np.int64)
        steps = min(max_steps_per_file, int(bev.shape[0]))
        runner.reset()
        for index in range(steps):
            obs = {
                "privileged_bev": torch.as_tensor(bev[index : index + 1][None, ...], dtype=torch.float32),
                "privileged_ego": torch.as_tensor(ego[index : index + 1][None, ...], dtype=torch.float32),
            }
            logits = runner.act(obs, {})["logits"]
            if tuple(logits.shape) != (1, 1, len(RAW2DRIVE_ACTIONS)) or not torch.isfinite(logits).all():
                raise ValueError("policy logits must be finite [1,1,39]")
            action_id = int(torch.argmax(logits, dim=-1).item())
            label_action_id = int(replay_action[index])
            runner.observe_action(action_id)
            action = get_discrete_action(action_id)
            label_action = get_discrete_action(label_action_id)
            action_is_moving_steer = _is_forward_steer_action(action)
            label_is_moving_steer = _is_forward_steer_action(label_action)
            action_counts[action_id] += 1
            label_action_counts[label_action_id] += 1
            action_match_count += int(action_id == label_action_id)
            reverse_count += int(action.reverse)
            nonzero_throttle_count += int(float(action.throttle) > 0.0 and not action.reverse)
            moving_steer_count += int(action_is_moving_steer)
            label_moving_steer_count += int(label_is_moving_steer)
            predicted_moving_on_label_nonmoving_count += int(action_is_moving_steer and not label_is_moving_steer)
            predicted_nonmoving_on_label_moving_count += int((not action_is_moving_steer) and label_is_moving_steer)
            left_moving_steer_count += int(action_is_moving_steer and float(action.steer) < 0.0)
            right_moving_steer_count += int(action_is_moving_steer and float(action.steer) > 0.0)
            zero_throttle_steer_count += int(
                abs(float(action.steer)) > 0.0 and float(action.throttle) == 0.0 and not action.reverse
            )
            zero_throttle_count += int(float(action.throttle) == 0.0)
        probed_files.append({"path": str(replay_path), "steps": steps})

    total = sum(action_counts.values())
    unique_action_ids = sorted(action_counts)
    nonzero_throttle_fraction = float(nonzero_throttle_count) / float(total) if total else 0.0
    moving_steer_fraction = float(moving_steer_count) / float(total) if total else 0.0
    label_moving_steer_fraction = float(label_moving_steer_count) / float(total) if total else 0.0
    action_match_fraction = float(action_match_count) / float(total) if total else 0.0
    predicted_moving_on_label_nonmoving_fraction = (
        float(predicted_moving_on_label_nonmoving_count) / float(total) if total else 0.0
    )
    predicted_nonmoving_on_label_moving_fraction = (
        float(predicted_nonmoving_on_label_moving_count) / float(total) if total else 0.0
    )
    zero_throttle_fraction = float(zero_throttle_count) / float(total) if total else 0.0
    zero_throttle_steer_fraction = float(zero_throttle_steer_count) / float(total) if total else 0.0
    previous_action8_count = int(action_counts.get(8, 0))
    previous_action8_fraction = float(previous_action8_count) / float(total) if total else 0.0
    dominant_action_count = max(action_counts.values()) if action_counts else 0
    dominant_action_fraction = float(dominant_action_count) / float(total) if total else 0.0
    recommended = bool(
        total > 0
        and reverse_count == 0
        and len(unique_action_ids) > 1
        and nonzero_throttle_fraction >= float(min_nonzero_throttle_fraction)
        and moving_steer_count >= int(min_nonzero_steer_count)
        and moving_steer_fraction >= float(min_nonzero_steer_fraction)
        and moving_steer_fraction <= float(max_nonzero_steer_fraction)
        and dominant_action_fraction <= float(max_dominant_action_fraction)
        and previous_action8_fraction <= float(max_previous_stuck_action8_fraction)
    )
    result = {
        "stage": "stage2_privileged_policy_clean",
        "source": "stage2_checkpoint_action_probe",
        "clean_gate_evidence": False,
        "checkpoint_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "replay_manifest": {
            "path": str(replay_manifest_resolved),
            "sha256": sha256_file(replay_manifest_resolved),
        },
        "action_table_version": ACTION_TABLE_VERSION,
        "probed_actions": total,
        "probed_files": probed_files,
        "action_counts": {str(key): int(action_counts[key]) for key in unique_action_ids},
        "label_action_counts": {str(key): int(label_action_counts[key]) for key in sorted(label_action_counts)},
        "unique_action_ids": unique_action_ids,
        "action_match_count": int(action_match_count),
        "action_match_fraction": action_match_fraction,
        "dominant_action_id": int(max(action_counts, key=action_counts.get)) if action_counts else None,
        "dominant_action_count": int(dominant_action_count),
        "dominant_action_fraction": dominant_action_fraction,
        "reverse_action_count": int(reverse_count),
        "nonzero_throttle_action_count": int(nonzero_throttle_count),
        "nonzero_throttle_fraction": nonzero_throttle_fraction,
        "moving_steer_action_count": int(moving_steer_count),
        "moving_steer_fraction": moving_steer_fraction,
        "label_moving_steer_action_count": int(label_moving_steer_count),
        "label_moving_steer_fraction": label_moving_steer_fraction,
        "predicted_moving_on_label_nonmoving_count": int(predicted_moving_on_label_nonmoving_count),
        "predicted_moving_on_label_nonmoving_fraction": predicted_moving_on_label_nonmoving_fraction,
        "predicted_nonmoving_on_label_moving_count": int(predicted_nonmoving_on_label_moving_count),
        "predicted_nonmoving_on_label_moving_fraction": predicted_nonmoving_on_label_moving_fraction,
        "nonzero_steer_action_count": int(moving_steer_count),
        "nonzero_steer_fraction": moving_steer_fraction,
        "left_moving_steer_action_count": int(left_moving_steer_count),
        "right_moving_steer_action_count": int(right_moving_steer_count),
        "zero_throttle_steer_action_count": int(zero_throttle_steer_count),
        "zero_throttle_steer_fraction": zero_throttle_steer_fraction,
        "zero_throttle_action_count": int(zero_throttle_count),
        "zero_throttle_fraction": zero_throttle_fraction,
        "single_action_collapse": len(unique_action_ids) == 1,
        "previous_stuck_action8_count": previous_action8_count,
        "previous_stuck_action8_fraction": previous_action8_fraction,
        "minimum_nonzero_throttle_fraction": float(min_nonzero_throttle_fraction),
        "minimum_nonzero_steer_fraction": float(min_nonzero_steer_fraction),
        "minimum_nonzero_steer_count": int(min_nonzero_steer_count),
        "maximum_moving_steer_fraction": float(max_nonzero_steer_fraction),
        "maximum_nonzero_steer_fraction": float(max_nonzero_steer_fraction),
        "maximum_dominant_action_fraction": float(max_dominant_action_fraction),
        "max_previous_stuck_action8_fraction": float(max_previous_stuck_action8_fraction),
        "recommended_for_clean_rerun": recommended,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe Stage 2 checkpoint actions on privileged replay observations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--replay-manifest", default="")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--output-config", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-steps-per-file", type=int, default=32)
    parser.add_argument("--min-nonzero-throttle-fraction", type=float, default=0.5)
    parser.add_argument("--min-nonzero-steer-fraction", type=float, default=0.02)
    parser.add_argument("--min-nonzero-steer-count", type=int, default=12)
    parser.add_argument("--max-nonzero-steer-fraction", type=float, default=0.5)
    parser.add_argument("--max-dominant-action-fraction", type=float, default=0.85)
    parser.add_argument("--max-previous-stuck-action8-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.output_config:
        build_checkpoint_filled_config(
            base_config_path=args.config,
            checkpoint_manifest_path=args.checkpoint_manifest,
            output_config_path=args.output_config,
        )
        print(f"PROMOTED_CONFIG={args.output_config}")
    result = probe_checkpoint_actions(
        base_config_path=args.config,
        checkpoint_manifest_path=args.checkpoint_manifest,
        replay_manifest_path=args.replay_manifest or None,
        project_root=args.project_root or None,
        max_steps_per_file=args.max_steps_per_file,
        min_nonzero_throttle_fraction=args.min_nonzero_throttle_fraction,
        min_nonzero_steer_fraction=args.min_nonzero_steer_fraction,
        min_nonzero_steer_count=args.min_nonzero_steer_count,
        max_nonzero_steer_fraction=args.max_nonzero_steer_fraction,
        max_dominant_action_fraction=args.max_dominant_action_fraction,
        max_previous_stuck_action8_fraction=args.max_previous_stuck_action8_fraction,
        device=args.device,
        output_path=args.output or None,
    )
    print(f"STAGE2_CHECKPOINT_ACTION_PROBE={json.dumps(result, sort_keys=True)}")


def _load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    return payload


def _is_forward_steer_action(action: Any) -> bool:
    return bool(abs(float(action.steer)) > 0.0 and float(action.throttle) > 0.0 and not action.reverse)


def _resolve_path(raw: str | Path, *, root: Path, fallback_dir: Path) -> Path:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(root / path)
        if os.sep != "\\" and "\\" in str(raw):
            candidates.append(root.joinpath(*PureWindowsPath(str(raw)).parts))
        candidates.append(fallback_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    main()
