from __future__ import annotations

import argparse
import csv
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

from rcs_ac_first.data.expert_dataset import load_expert_npz


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corrective-delta targets in an expert NPZ dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/corrective_delta_dataset_audit")
    parser.add_argument("--run-name", default="corrective_delta_dataset_audit")
    parser.add_argument("--route-min", type=float, default=0.025)
    parser.add_argument("--route-max", type=float, default=0.10)
    parser.add_argument("--abs-lane-offset-min", type=float, default=0.55)
    parser.add_argument("--abs-heading-error-min", type=float, default=3.0)
    parser.add_argument("--min-severe-count", type=int, default=200)
    parser.add_argument("--min-side-count", type=int, default=40)
    parser.add_argument("--min-sign-rate", type=float, default=0.98)
    parser.add_argument("--min-side-sign-rate", type=float, default=0.98)
    parser.add_argument("--min-abs-delta-mean", type=float, default=0.035)
    parser.add_argument("--min-abs-delta-p10", type=float, default=0.010)
    parser.add_argument("--max-target-steer-saturation-rate", type=float, default=0.05)
    parser.add_argument("--max-target-steer-consistency-mae", type=float, default=1.0e-5)
    parser.add_argument("--max-wrong-sign-rows", type=int, default=500)
    return parser.parse_args()


def _as_float_array(arrays: dict[str, np.ndarray], key: str, *, default: np.ndarray | None = None) -> np.ndarray:
    if key in arrays:
        return np.asarray(arrays[key], dtype=np.float32).reshape(-1)
    if default is None:
        raise KeyError(f"Dataset is missing required audit key: {key}")
    return np.asarray(default, dtype=np.float32).reshape(-1)


def _safe_rate(mask: np.ndarray) -> float:
    if int(mask.size) == 0:
        return 0.0
    return float(np.mean(mask.astype(np.float32)))


def _masked_stat(values: np.ndarray, mask: np.ndarray, fn: str) -> float:
    selected = np.asarray(values, dtype=np.float32).reshape(-1)[mask]
    if int(selected.size) == 0:
        return 0.0
    if fn == "mean":
        return float(np.mean(selected))
    if fn == "abs_mean":
        return float(np.mean(np.abs(selected)))
    if fn == "p50_abs":
        return float(np.percentile(np.abs(selected), 50))
    if fn == "p75_abs":
        return float(np.percentile(np.abs(selected), 75))
    if fn == "p90_abs":
        return float(np.percentile(np.abs(selected), 90))
    if fn == "p95_abs":
        return float(np.percentile(np.abs(selected), 95))
    if fn == "p10_abs":
        return float(np.percentile(np.abs(selected), 10))
    if fn == "p10_signed":
        return float(np.percentile(selected, 10))
    if fn == "min":
        return float(np.min(selected))
    if fn == "max":
        return float(np.max(selected))
    if fn == "max_abs":
        return float(np.max(np.abs(selected)))
    raise ValueError(f"Unknown stat: {fn}")


def _side_summary(
    *,
    name: str,
    side_mask: np.ndarray,
    target_delta: np.ndarray,
    lane_offset: np.ndarray,
    policy_steer: np.ndarray,
    expert_steer: np.ndarray,
) -> dict[str, float | int]:
    expected_signed = target_delta * (-lane_offset)
    sign_mask = expected_signed > 0.0
    return {
        f"{name}_count": int(np.sum(side_mask)),
        f"{name}_sign_rate_vs_neg_lane": _safe_rate(sign_mask[side_mask]),
        f"{name}_target_delta_mean": _masked_stat(target_delta, side_mask, "mean"),
        f"{name}_target_delta_abs_mean": _masked_stat(target_delta, side_mask, "abs_mean"),
        f"{name}_target_delta_abs_p10": _masked_stat(target_delta, side_mask, "p10_abs"),
        f"{name}_target_delta_abs_p50": _masked_stat(target_delta, side_mask, "p50_abs"),
        f"{name}_target_delta_abs_p75": _masked_stat(target_delta, side_mask, "p75_abs"),
        f"{name}_target_delta_abs_p90": _masked_stat(target_delta, side_mask, "p90_abs"),
        f"{name}_target_delta_abs_p95": _masked_stat(target_delta, side_mask, "p95_abs"),
        f"{name}_target_delta_abs_max": _masked_stat(target_delta, side_mask, "max_abs"),
        f"{name}_signed_delta_p10": _masked_stat(expected_signed, side_mask, "p10_signed"),
        f"{name}_policy_steer_mean": _masked_stat(policy_steer, side_mask, "mean"),
        f"{name}_expert_steer_mean": _masked_stat(expert_steer, side_mask, "mean"),
    }


def _optional_array(
    arrays: dict[str, np.ndarray],
    key: str,
    *,
    n: int,
    default: Any,
) -> np.ndarray:
    if key in arrays and np.asarray(arrays[key]).shape[:1] == (int(n),):
        return np.asarray(arrays[key])
    return np.asarray([default] * int(n))


def _route_bin(value: float) -> str:
    if value < 0.05:
        return "0.025-0.050"
    if value < 0.075:
        return "0.050-0.075"
    if value <= 0.10:
        return "0.075-0.100"
    return "outside"


def _rows_from_mask(
    arrays: dict[str, np.ndarray],
    *,
    mask: np.ndarray,
    expected_signed: np.ndarray,
    target_steer: np.ndarray,
    max_rows: int,
) -> list[dict[str, Any]]:
    n = int(np.asarray(arrays["route_progress"]).reshape(-1).shape[0])
    expert_action = np.asarray(arrays["expert_action"], dtype=np.float32)
    episode_id = _optional_array(arrays, "episode_id", n=n, default=-1).reshape(-1)
    step_in_episode = _optional_array(arrays, "step_in_episode", n=n, default=-1).reshape(-1)
    route_progress = _optional_array(arrays, "route_progress", n=n, default=0.0).astype(np.float32).reshape(-1)
    lane_offset = _optional_array(arrays, "lane_offset", n=n, default=0.0).astype(np.float32).reshape(-1)
    heading_error = _optional_array(arrays, "heading_error", n=n, default=0.0).astype(np.float32).reshape(-1)
    available = _optional_array(arrays, "target_corrective_delta_available", n=n, default=0.0).astype(np.float32).reshape(-1)
    target_delta = _optional_array(arrays, "target_corrective_delta", n=n, default=0.0).astype(np.float32).reshape(-1)
    policy_steer = _optional_array(arrays, "policy_action_steer", n=n, default=0.0).astype(np.float32).reshape(-1)
    target_corrected = _optional_array(arrays, "target_corrected_steer", n=n, default=0.0).astype(np.float32).reshape(-1)
    source_label = _optional_array(arrays, "source_label", n=n, default="").reshape(-1)
    source_dataset = _optional_array(arrays, "source_dataset", n=n, default="").reshape(-1)
    source_row_index = _optional_array(arrays, "source_row_index", n=n, default=-1).reshape(-1)
    lateral_active = _optional_array(arrays, "lateral_recovery_active", n=n, default=False).reshape(-1)
    lateral_branch = _optional_array(arrays, "lateral_recovery_branch_active", n=n, default=False).reshape(-1)
    lateral_correction = _optional_array(arrays, "lateral_recovery_correction", n=n, default=0.0).astype(np.float32).reshape(-1)
    lateral_raw = _optional_array(arrays, "lateral_recovery_raw_steer", n=n, default=0.0).astype(np.float32).reshape(-1)
    turn_boost = _optional_array(arrays, "lateral_recovery_turn_boost_active", n=n, default=False).reshape(-1)
    speed_gate = _optional_array(arrays, "lateral_recovery_speed_gate_active", n=n, default=False).reshape(-1)
    speed_kmh = _optional_array(arrays, "speed_kmh", n=n, default=0.0).astype(np.float32).reshape(-1)

    indices = np.flatnonzero(mask)
    order = sorted(
        indices.tolist(),
        key=lambda idx: (float(route_progress[idx]), int(episode_id[idx]), int(step_in_episode[idx]), int(idx)),
    )
    rows: list[dict[str, Any]] = []
    for idx in order[: max(0, int(max_rows))]:
        rows.append(
            {
                "row_index": int(idx),
                "source_label": str(source_label[idx]),
                "source_dataset": str(source_dataset[idx]),
                "source_row_index": int(source_row_index[idx]) if str(source_row_index[idx]).lstrip("-").isdigit() else str(source_row_index[idx]),
                "episode_id": int(episode_id[idx]),
                "step_in_episode": int(step_in_episode[idx]),
                "route_progress": float(route_progress[idx]),
                "route_bin": _route_bin(float(route_progress[idx])),
                "lane_offset": float(lane_offset[idx]),
                "heading_error": float(heading_error[idx]),
                "speed_kmh": float(speed_kmh[idx]),
                "target_corrective_delta_available": float(available[idx]),
                "target_corrective_delta": float(target_delta[idx]),
                "expected_signed": float(expected_signed[idx]),
                "policy_action_steer": float(policy_steer[idx]),
                "target_corrected_steer": float(target_corrected[idx]),
                "target_steer": float(target_steer[idx]),
                "expert_steer": float(expert_action[idx, 0]),
                "lateral_recovery_active": bool(lateral_active[idx]),
                "lateral_recovery_branch_active": bool(lateral_branch[idx]),
                "lateral_recovery_correction": float(lateral_correction[idx]),
                "lateral_recovery_raw_steer": float(lateral_raw[idx]),
                "lateral_recovery_turn_boost_active": bool(turn_boost[idx]),
                "lateral_recovery_speed_gate_active": bool(speed_gate[idx]),
                "classification": "unresolved",
            }
        )
    return rows


def _group_counts(values: np.ndarray, mask: np.ndarray, *, top_k: int = 20) -> list[dict[str, Any]]:
    selected = np.asarray(values).reshape(-1)[mask]
    if int(selected.size) == 0:
        return []
    unique, counts = np.unique(selected, return_counts=True)
    order = np.argsort(-counts)
    return [
        {"value": str(unique[index]), "count": int(counts[index])}
        for index in order[: int(top_k)]
    ]


def audit_arrays(
    arrays: dict[str, np.ndarray],
    *,
    route_min: float,
    route_max: float,
    abs_lane_offset_min: float,
    abs_heading_error_min: float,
    min_severe_count: int,
    min_side_count: int,
    min_sign_rate: float,
    min_side_sign_rate: float,
    min_abs_delta_mean: float,
    min_abs_delta_p10: float,
    max_target_steer_saturation_rate: float,
    max_target_steer_consistency_mae: float,
    max_wrong_sign_rows: int = 500,
) -> dict[str, Any]:
    route_progress = _as_float_array(arrays, "route_progress")
    lane_offset = _as_float_array(arrays, "lane_offset", default=np.zeros_like(route_progress))
    heading_error = _as_float_array(arrays, "heading_error", default=np.zeros_like(route_progress))
    target_delta = _as_float_array(arrays, "target_corrective_delta")
    available = _as_float_array(arrays, "target_corrective_delta_available", default=np.ones_like(route_progress))
    policy_steer = _as_float_array(arrays, "policy_action_steer", default=np.zeros_like(route_progress))
    target_corrected_steer = _as_float_array(
        arrays,
        "target_corrected_steer",
        default=policy_steer + target_delta,
    )
    expert_action = np.asarray(arrays["expert_action"], dtype=np.float32)
    expert_steer = expert_action[:, 0].reshape(-1)

    available_mask = available > 0.5
    route_mask = (route_progress >= float(route_min)) & (route_progress <= float(route_max))
    severe_mask = (np.abs(lane_offset) >= float(abs_lane_offset_min)) | (
        np.abs(heading_error) >= float(abs_heading_error_min)
    )
    nonzero_target_mask = np.abs(target_delta) > 1.0e-5
    eligible_mask = available_mask & route_mask & severe_mask & nonzero_target_mask
    left_mask = eligible_mask & (lane_offset < 0.0)
    right_mask = eligible_mask & (lane_offset > 0.0)
    expected_signed = target_delta * (-lane_offset)
    aligned_mask = eligible_mask & (expected_signed > 0.0)
    wrong_sign_mask = eligible_mask & (expected_signed <= 0.0)
    wrong_sign_left_mask = left_mask & (expected_signed <= 0.0)

    sign_rate = _safe_rate((expected_signed > 0.0)[eligible_mask])
    left_sign_rate = _safe_rate((expected_signed > 0.0)[left_mask])
    right_sign_rate = _safe_rate((expected_signed > 0.0)[right_mask])
    target_abs_mean = _masked_stat(target_delta, eligible_mask, "abs_mean")
    target_abs_p10 = _masked_stat(target_delta, eligible_mask, "p10_abs")
    target_steer = policy_steer + target_delta
    target_steer_consistency_mae = _masked_stat(target_corrected_steer - target_steer, available_mask, "abs_mean")
    target_steer_saturation_rate = _safe_rate((np.abs(target_steer) > 1.0)[eligible_mask])

    gate_failures: list[str] = []
    if int(np.sum(eligible_mask)) < int(min_severe_count):
        gate_failures.append("severe_count")
    if int(np.sum(left_mask)) < int(min_side_count):
        gate_failures.append("left_count")
    if int(np.sum(right_mask)) < int(min_side_count):
        gate_failures.append("right_count")
    if sign_rate < float(min_sign_rate):
        gate_failures.append("sign_rate")
    if left_sign_rate < float(min_side_sign_rate):
        gate_failures.append("left_sign_rate")
    if right_sign_rate < float(min_side_sign_rate):
        gate_failures.append("right_sign_rate")
    if target_abs_mean < float(min_abs_delta_mean):
        gate_failures.append("target_delta_abs_mean")
    if target_abs_p10 < float(min_abs_delta_p10):
        gate_failures.append("target_delta_abs_p10")
    if target_steer_saturation_rate > float(max_target_steer_saturation_rate):
        gate_failures.append("target_steer_saturation_rate")
    if target_steer_consistency_mae > float(max_target_steer_consistency_mae):
        gate_failures.append("target_steer_consistency_mae")

    route_bins: list[dict[str, float | int | str]] = []
    for start, end in ((0.025, 0.05), (0.05, 0.075), (0.075, 0.10)):
        upper = route_progress <= end if end >= 0.10 else route_progress < end
        bin_mask = available_mask & (route_progress >= start) & upper & severe_mask & nonzero_target_mask
        bin_left = bin_mask & (lane_offset < 0.0)
        bin_right = bin_mask & (lane_offset > 0.0)
        route_bins.append(
            {
                "bin": f"{start:.3f}-{end:.3f}",
                "count": int(np.sum(bin_mask)),
                "left_count": int(np.sum(bin_left)),
                "right_count": int(np.sum(bin_right)),
                "aligned_rate": _safe_rate((expected_signed > 0.0)[bin_mask]),
                "target_delta_abs_mean": _masked_stat(target_delta, bin_mask, "abs_mean"),
            }
        )

    source_keys = (
        "source_label",
        "source_dataset",
        "source_dataset_index",
        "source_row_index",
        "source_episode_id",
        "source_step_in_episode",
    )
    schema = {
        key: {
            "present": bool(key in arrays),
            "shape": list(np.asarray(arrays[key]).shape) if key in arrays else [],
            "finite_rate": _safe_rate(np.isfinite(np.asarray(arrays[key], dtype=np.float32)).reshape(-1))
            if key in arrays and np.issubdtype(np.asarray(arrays[key]).dtype, np.number)
            else 0.0,
        }
        for key in (
            "target_corrective_delta",
            "target_corrective_delta_available",
            "policy_action_steer",
            "target_corrected_steer",
            "lane_offset",
            "heading_error",
            "route_progress",
            *source_keys,
            "lateral_recovery_active",
            "lateral_recovery_branch_active",
            "lateral_recovery_correction",
            "lateral_recovery_raw_steer",
            "lateral_recovery_turn_boost_active",
            "lateral_recovery_speed_gate_active",
            "speed_kmh",
        )
    }
    missing_source_metadata = not all(bool(schema[key]["present"]) for key in source_keys)
    episode_ids = _optional_array(arrays, "episode_id", n=route_progress.shape[0], default=-1).reshape(-1)
    route_bin_values = np.asarray([_route_bin(float(value)) for value in route_progress])
    wrong_sign_rows = _rows_from_mask(
        arrays,
        mask=wrong_sign_left_mask,
        expected_signed=expected_signed,
        target_steer=target_steer,
        max_rows=int(max_wrong_sign_rows),
    )

    summary: dict[str, Any] = {
        "schema": schema,
        "num_transitions": int(route_progress.shape[0]),
        "route_min": float(route_min),
        "route_max": float(route_max),
        "abs_lane_offset_min": float(abs_lane_offset_min),
        "abs_heading_error_min": float(abs_heading_error_min),
        "available_count": int(np.sum(available_mask)),
        "available_rate": _safe_rate(available_mask),
        "route_count": int(np.sum(route_mask)),
        "route_rate": _safe_rate(route_mask),
        "eligible_severe_count": int(np.sum(eligible_mask)),
        "eligible_severe_rate": _safe_rate(eligible_mask),
        "aligned_count": int(np.sum(aligned_mask)),
        "aligned_rate_in_eligible": sign_rate,
        "wrong_sign_rate_in_eligible": 1.0 - sign_rate if int(np.sum(eligible_mask)) else 0.0,
        "wrong_sign_count": int(np.sum(wrong_sign_mask)),
        "wrong_sign_left_count": int(np.sum(wrong_sign_left_mask)),
        "wrong_sign_left_rate_in_left": _safe_rate(wrong_sign_left_mask[left_mask]),
        "missing_source_metadata": bool(missing_source_metadata),
        "source_metadata_present": {key: bool(schema[key]["present"]) for key in source_keys},
        "reconstructable_from_summary": "none" if bool(missing_source_metadata) else "row_metadata",
        "wrong_sign_left_by_episode_top": _group_counts(episode_ids, wrong_sign_left_mask),
        "wrong_sign_left_by_route_bin": _group_counts(route_bin_values, wrong_sign_left_mask),
        "wrong_sign_left_rows": wrong_sign_rows,
        "wrong_sign_left_rows_truncated": int(np.sum(wrong_sign_left_mask)) > int(max_wrong_sign_rows),
        "zero_delta_rate_in_route_severe": _safe_rate((~nonzero_target_mask)[available_mask & route_mask & severe_mask]),
        "target_delta_abs_mean": target_abs_mean,
        "target_delta_abs_p10": target_abs_p10,
        "target_delta_abs_p50": _masked_stat(target_delta, eligible_mask, "p50_abs"),
        "target_delta_abs_p75": _masked_stat(target_delta, eligible_mask, "p75_abs"),
        "target_delta_abs_p90": _masked_stat(target_delta, eligible_mask, "p90_abs"),
        "target_delta_abs_p95": _masked_stat(target_delta, eligible_mask, "p95_abs"),
        "target_delta_abs_max": _masked_stat(target_delta, eligible_mask, "max_abs"),
        "target_delta_min": _masked_stat(target_delta, eligible_mask, "min"),
        "target_delta_max": _masked_stat(target_delta, eligible_mask, "max"),
        "signed_delta_p10": _masked_stat(expected_signed, eligible_mask, "p10_signed"),
        "policy_steer_mean": _masked_stat(policy_steer, eligible_mask, "mean"),
        "expert_steer_mean": _masked_stat(expert_steer, eligible_mask, "mean"),
        "target_steer_mean": _masked_stat(target_steer, eligible_mask, "mean"),
        "target_steer_abs_mean": _masked_stat(target_steer, eligible_mask, "abs_mean"),
        "target_steer_min": _masked_stat(target_steer, eligible_mask, "min"),
        "target_steer_max": _masked_stat(target_steer, eligible_mask, "max"),
        "target_steer_saturation_rate": target_steer_saturation_rate,
        "target_steer_consistency_mae": target_steer_consistency_mae,
        "route_bins": route_bins,
        "gate_status": "PASS" if not gate_failures else "BLOCK",
        "gate_failures": gate_failures,
    }
    summary.update(
        _side_summary(
            name="left",
            side_mask=left_mask,
            target_delta=target_delta,
            lane_offset=lane_offset,
            policy_steer=policy_steer,
            expert_steer=expert_steer,
        )
    )
    summary.update(
        _side_summary(
            name="right",
            side_mask=right_mask,
            target_delta=target_delta,
            lane_offset=lane_offset,
            policy_steer=policy_steer,
            expert_steer=expert_steer,
        )
    )
    return summary


def main() -> int:
    args = parse_args()
    dataset_path = _project_path(args.dataset)
    arrays = load_expert_npz(dataset_path)
    summary = audit_arrays(
        arrays,
        route_min=float(args.route_min),
        route_max=float(args.route_max),
        abs_lane_offset_min=float(args.abs_lane_offset_min),
        abs_heading_error_min=float(args.abs_heading_error_min),
        min_severe_count=int(args.min_severe_count),
        min_side_count=int(args.min_side_count),
        min_sign_rate=float(args.min_sign_rate),
        min_side_sign_rate=float(args.min_side_sign_rate),
        min_abs_delta_mean=float(args.min_abs_delta_mean),
        min_abs_delta_p10=float(args.min_abs_delta_p10),
        max_target_steer_saturation_rate=float(args.max_target_steer_saturation_rate),
        max_target_steer_consistency_mae=float(args.max_target_steer_consistency_mae),
        max_wrong_sign_rows=int(args.max_wrong_sign_rows),
    )
    summary["dataset"] = str(dataset_path)

    output_dir = _project_path(args.output_dir) / str(args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    wrong_rows_path = output_dir / "wrong_sign_left_rows.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows = list(summary.get("wrong_sign_left_rows", []))
    if rows:
        with wrong_rows_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        wrong_rows_path.write_text("", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    summary["wrong_sign_left_rows_csv"] = str(wrong_rows_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
