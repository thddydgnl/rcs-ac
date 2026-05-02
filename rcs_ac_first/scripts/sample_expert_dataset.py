from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rcs_ac_first.data.expert_dataset import load_expert_npz, save_expert_npz


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic row sample from an expert NPZ dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["first", "random"], default="random")
    return parser.parse_args()


def _sample_indices(n: int, *, max_rows: int, seed: int, mode: str) -> np.ndarray:
    if max_rows <= 0:
        raise ValueError("--max-rows must be positive.")
    count = min(int(max_rows), int(n))
    if str(mode) == "first":
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=count, replace=False))


def _sample_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    n = int(arrays["obs_flat"].shape[0])
    sampled: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.shape[:1] == (n,):
            sampled[key] = array[indices].copy()
        else:
            sampled[key] = array.copy()
    return sampled


def main() -> int:
    args = parse_args()
    input_path = _project_path(args.input)
    output_path = _project_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    arrays = load_expert_npz(input_path)
    n = int(arrays["obs_flat"].shape[0])
    indices = _sample_indices(n, max_rows=int(args.max_rows), seed=int(args.seed), mode=str(args.mode))
    sampled = _sample_arrays(arrays, indices)
    summary = save_expert_npz(output_path, sampled)
    payload: dict[str, Any] = summary.to_dict() | {
        "input": str(input_path),
        "output": str(output_path),
        "input_count": n,
        "sampled_count": int(indices.shape[0]),
        "seed": int(args.seed),
        "mode": str(args.mode),
        "selected_indices": [int(item) for item in indices.tolist()],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] dataset={output_path} summary={summary_path} {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
