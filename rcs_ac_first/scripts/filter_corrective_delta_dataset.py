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
    parser = argparse.ArgumentParser(description="Filter wrong-sign corrective-delta rows from an expert NPZ dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--route-min", type=float, default=0.20)
    parser.add_argument("--route-max", type=float, default=0.36)
    parser.add_argument("--abs-lane-offset-min", type=float, default=1.0)
    parser.add_argument("--abs-heading-error-min", type=float, default=6.0)
    parser.add_argument("--mode", choices=["strict-lane", "semantic"], default="semantic")
    parser.add_argument("--drop-ambiguous", action="store_true")
    return parser.parse_args()


def _float_array(arrays: dict[str, np.ndarray], key: str, *, default: np.ndarray | None = None) -> np.ndarray:
    if key in arrays:
        return np.asarray(arrays[key], dtype=np.float32).reshape(-1)
    if default is None:
        raise KeyError(f"Dataset is missing required key: {key}")
    return np.asarray(default, dtype=np.float32).reshape(-1)


def _filter_mask(
    arrays: dict[str, np.ndarray],
    *,
    route_min: float,
    route_max: float,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    mode: str,
    drop_ambiguous: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    route = _float_array(arrays, "route_progress")
    lane = _float_array(arrays, "lane_offset", default=np.zeros_like(route))
    heading = _float_array(arrays, "heading_error", default=np.zeros_like(route))
    delta = _float_array(arrays, "target_corrective_delta", default=np.zeros_like(route))
    available = _float_array(arrays, "target_corrective_delta_available", default=np.ones_like(route))

    n = int(route.shape[0])
    route_mask = (route >= float(route_min)) & (route <= float(route_max))
    available_mask = available > 0.5
    nonzero_delta = np.abs(delta) > 1.0e-5
    lane_dominant = np.abs(lane) >= float(abs_lane_offset_min)
    heading_dominant = (~lane_dominant) & (np.abs(heading) >= float(abs_heading_error_min))
    ambiguous = (~lane_dominant) & (np.abs(heading) < float(abs_heading_error_min))
    eligible = route_mask & available_mask & nonzero_delta & (lane_dominant | heading_dominant | ambiguous)

    lane_aligned = delta * (-lane) > 0.0
    heading_aligned = delta * np.sign(heading) > 0.0
    if str(mode) == "strict-lane":
        bad = eligible & (~lane_aligned)
    else:
        bad = eligible & ((lane_dominant & (~lane_aligned)) | (heading_dominant & (~heading_aligned)))
        if bool(drop_ambiguous):
            bad = bad | (eligible & ambiguous)

    keep = np.ones((n,), dtype=np.bool_)
    keep[bad] = False
    summary = {
        "input_count": n,
        "kept_count": int(np.sum(keep)),
        "dropped_count": int(np.sum(~keep)),
        "mode": str(mode),
        "route_min": float(route_min),
        "route_max": float(route_max),
        "abs_lane_offset_min": float(abs_lane_offset_min),
        "abs_heading_error_min": float(abs_heading_error_min),
        "drop_ambiguous": bool(drop_ambiguous),
        "eligible_count": int(np.sum(eligible)),
        "lane_dominant_count": int(np.sum(eligible & lane_dominant)),
        "heading_dominant_count": int(np.sum(eligible & heading_dominant)),
        "ambiguous_count": int(np.sum(eligible & ambiguous)),
        "dropped_left_count": int(np.sum(bad & (lane < 0.0))),
        "dropped_right_count": int(np.sum(bad & (lane > 0.0))),
        "dropped_lane_dominant_count": int(np.sum(bad & lane_dominant)),
        "dropped_heading_dominant_count": int(np.sum(bad & heading_dominant)),
        "dropped_ambiguous_count": int(np.sum(bad & ambiguous)),
    }
    return keep, summary


def main() -> int:
    args = parse_args()
    input_path = _project_path(args.input)
    output_path = _project_path(args.output)
    arrays = load_expert_npz(input_path)
    keep, summary = _filter_mask(
        arrays,
        route_min=float(args.route_min),
        route_max=float(args.route_max),
        abs_lane_offset_min=float(args.abs_lane_offset_min),
        abs_heading_error_min=float(args.abs_heading_error_min),
        mode=str(args.mode),
        drop_ambiguous=bool(args.drop_ambiguous),
    )
    n = int(arrays["obs_flat"].shape[0])
    filtered = {
        key: value[keep].copy() if np.asarray(value).shape[:1] == (n,) else value
        for key, value in arrays.items()
    }
    dataset_summary = save_expert_npz(output_path, filtered)
    payload = dataset_summary.to_dict() | summary | {"input": str(input_path), "output": str(output_path)}
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] dataset={output_path} summary={summary_path} {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
