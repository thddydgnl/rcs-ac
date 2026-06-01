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
    "lane_offset",
    "heading_error",
    "speed_kmh",
    "curve_proxy",
    "policy_action_steer",
    "policy_action_longitudinal",
    "target_corrected_steer",
    "target_corrective_delta",
    "target_corrective_delta_available",
)

SOURCE_STRING_KEYS: tuple[str, ...] = (
    "source_label",
    "source_dataset",
)

SOURCE_INT_KEYS: tuple[str, ...] = (
    "source_dataset_index",
    "source_row_index",
    "source_episode_id",
    "source_step_in_episode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine compatible expert/recovery .npz datasets.")
    parser.add_argument("--input", action="append", required=True, help="Input .npz path. Repeat for each dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def _concat_key(key: str, values: list[np.ndarray]) -> np.ndarray:
    first = values[0]
    if first.ndim == 0:
        return np.asarray(values)
    return np.concatenate(values, axis=0)


def _reindex_episode_ids(arrays_list: list[dict[str, np.ndarray]]) -> None:
    offset = 0
    for arrays in arrays_list:
        episode_ids = arrays.get("episode_id")
        if episode_ids is None:
            continue
        unique = np.unique(episode_ids.astype(np.int32, copy=False))
        mapping = {int(old): int(index + offset) for index, old in enumerate(unique.tolist())}
        arrays["episode_id"] = np.asarray([mapping[int(item)] for item in episode_ids], dtype=np.int32)
        offset += int(unique.shape[0])


def _ensure_source_metadata(arrays_list: list[dict[str, np.ndarray]], input_paths: list[Path]) -> None:
    for dataset_index, (arrays, input_path) in enumerate(zip(arrays_list, input_paths, strict=True)):
        n = int(arrays["obs_flat"].shape[0])
        if "source_label" not in arrays:
            arrays["source_label"] = np.full((n,), input_path.stem, dtype="<U128")
        if "source_dataset" not in arrays:
            arrays["source_dataset"] = np.full((n,), str(input_path), dtype="<U512")
        if "source_dataset_index" not in arrays:
            arrays["source_dataset_index"] = np.full((n,), int(dataset_index), dtype=np.int32)
        if "source_row_index" not in arrays:
            arrays["source_row_index"] = np.arange(n, dtype=np.int32)
        if "source_episode_id" not in arrays:
            arrays["source_episode_id"] = arrays["episode_id"].astype(np.int32, copy=True)
        if "source_step_in_episode" not in arrays:
            arrays["source_step_in_episode"] = arrays["step_in_episode"].astype(np.int32, copy=True)
        for key in SOURCE_STRING_KEYS:
            arrays[key] = np.asarray(arrays[key]).astype(str, copy=False).reshape(-1)
        for key in SOURCE_INT_KEYS:
            arrays[key] = np.asarray(arrays[key], dtype=np.int32).reshape(-1)


def _combine(inputs: list[Path]) -> dict[str, np.ndarray]:
    arrays_list = [load_expert_npz(path) for path in inputs]
    _ensure_source_metadata(arrays_list, inputs)
    _reindex_episode_ids(arrays_list)
    for arrays in arrays_list:
        n = int(arrays["obs_flat"].shape[0])
        for key in OPTIONAL_FLOAT_KEYS:
            if key not in arrays:
                arrays[key] = np.zeros((n,), dtype=np.float32)
    common_keys = set(arrays_list[0].keys())
    for arrays in arrays_list[1:]:
        common_keys &= set(arrays.keys())
    if not common_keys:
        raise ValueError("Input datasets have no common keys.")

    lengths = [int(arrays["obs_flat"].shape[0]) for arrays in arrays_list]
    combined: dict[str, np.ndarray] = {}
    for key in sorted(common_keys):
        values = [arrays[key] for arrays in arrays_list]
        if any(value.shape[:1] != (length,) for value, length in zip(values, lengths, strict=True)):
            continue
        combined[key] = _concat_key(key, values)
    return combined


def main() -> int:
    args = parse_args()
    input_paths = [_project_path(item) for item in args.input]
    output_path = _project_path(args.output)
    if len(input_paths) < 2:
        raise ValueError("At least two --input paths are required.")
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")

    combined = _combine(input_paths)
    summary = save_expert_npz(output_path, combined)
    summary_payload: dict[str, Any] = summary.to_dict()
    summary_payload["input_paths"] = [str(path) for path in input_paths]
    summary_payload["input_counts"] = [int(load_expert_npz(path)["obs_flat"].shape[0]) for path in input_paths]
    summary_path = _project_path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[done] output={output_path} summary={summary_path} {summary_payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
