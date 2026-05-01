from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

from rcs_ac_first.data.expert_dataset import load_expert_npz, save_expert_npz
from rcs_ac_first.scripts.train import _project_path

OPTIONAL_FLOAT_KEYS: tuple[str, ...] = (
    "policy_action_steer",
    "policy_action_longitudinal",
    "target_corrected_steer",
    "target_corrective_delta",
    "target_corrective_delta_available",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed expert NPZ from existing expert datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", action="append", nargs=3, metavar=("PATH", "COUNT", "LABEL"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-bad-transitions", action="store_true")
    parser.add_argument("--shuffle-output", action="store_true")
    return parser.parse_args()


def _select_indices(
    arrays: dict[str, np.ndarray],
    *,
    count: int,
    seed: int,
    include_bad_transitions: bool,
) -> np.ndarray:
    n = int(arrays["obs_flat"].shape[0])
    if count <= 0:
        raise ValueError(f"part count must be positive, got {count}")
    keep = np.arange(n, dtype=np.int64)
    if not include_bad_transitions:
        keep = keep[~arrays["bad_transition_mask"].astype(np.bool_)]
    if count > int(keep.shape[0]):
        raise ValueError(f"Requested {count} rows from dataset with only {int(keep.shape[0])} usable rows")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(keep, size=int(count), replace=False)).astype(np.int64, copy=False)


def _slice_part(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    label: str,
    episode_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    part = {
        key: value[indices].copy()
        for key, value in arrays.items()
        if value.shape[:1] == (int(arrays["obs_flat"].shape[0]),)
    }
    part["episode_id"] = part["episode_id"].astype(np.int32, copy=False) + int(episode_offset)
    max_episode_id = int(part["episode_id"].max()) if int(part["episode_id"].shape[0]) else -1
    route = part["route_progress"].astype(np.float32, copy=False)
    mask_available = part.get("aux_mask_targets_available")
    summary: dict[str, Any] = {
        "label": str(label),
        "selected_count": int(indices.shape[0]),
        "route_progress_min": float(np.min(route)),
        "route_progress_max": float(np.max(route)),
        "route_progress_mean": float(np.mean(route)),
        "bad_transition_count": int(np.count_nonzero(part["bad_transition_mask"])),
    }
    if mask_available is not None:
        summary["aux_mask_available_rate"] = float(np.mean(mask_available.astype(np.float32)))
    return part, summary, max_episode_id + 1


def _concat_parts(parts: list[dict[str, np.ndarray]], *, shuffle_output: bool, seed: int) -> dict[str, np.ndarray]:
    required_keys = {
        "obs_flat",
        "auxiliary",
        "expert_action",
        "expert_control",
        "reward",
        "done",
        "route_progress",
        "episode_id",
        "step_in_episode",
        "bad_transition_mask",
    }
    for part in parts:
        n = int(part["obs_flat"].shape[0])
        for key in OPTIONAL_FLOAT_KEYS:
            if key not in part:
                part[key] = np.zeros((n,), dtype=np.float32)
    common_keys = set(parts[0])
    for part in parts[1:]:
        common_keys &= set(part)
    keys = sorted(common_keys | required_keys)
    mixed = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    if shuffle_output:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(int(mixed["obs_flat"].shape[0]))
        mixed = {key: value[order] if value.shape[:1] == order.shape else value for key, value in mixed.items()}
    return mixed


def main() -> int:
    args = parse_args()
    parts: list[dict[str, np.ndarray]] = []
    summaries: list[dict[str, Any]] = []
    episode_offset = 0
    for index, (raw_path, raw_count, label) in enumerate(args.part):
        arrays = load_expert_npz(_project_path(raw_path))
        indices = _select_indices(
            arrays,
            count=int(raw_count),
            seed=int(args.seed) + index,
            include_bad_transitions=bool(args.include_bad_transitions),
        )
        part, summary, next_episode_offset = _slice_part(
            arrays,
            indices,
            label=str(label),
            episode_offset=episode_offset,
        )
        episode_offset += next_episode_offset
        parts.append(part)
        summaries.append(summary)

    mixed = _concat_parts(parts, shuffle_output=bool(args.shuffle_output), seed=int(args.seed))
    output_path = _project_path(args.output)
    summary = save_expert_npz(output_path, mixed)
    mask_available = mixed.get("aux_mask_targets_available")
    extra: dict[str, Any] = {
        "parts": summaries,
        "shuffle_output": bool(args.shuffle_output),
        "seed": int(args.seed),
    }
    if mask_available is not None:
        extra["aux_mask_available_rate"] = float(np.mean(mask_available.astype(np.float32)))
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary.to_dict() | extra, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] dataset={output_path} summary={summary_path} {summary.to_dict() | extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
