from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.runtime.action import clamp_action_tensor, decode_stage4_action_v2
from rcs_ac_first.runtime.auxiliary import build_auxiliary_observation_spec, flatten_auxiliary_observation_dict
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec, flatten_observation_dict
from rcs_ac_first.scripts.train import _build_env, _project_path, _resolve_config_path, _set_seed, _timestamp

COLLISION_CLUSTER_ROUTES: tuple[tuple[str, float], ...] = (
    ("0179487", 0.1794871794871795),
    ("0743590", 0.7435897435897436),
)
COLLISION_CLUSTER_TOLERANCE = 0.02
COLLISION_CONTEXT_WINDOW_STEPS = 32

BC_EVAL_FIELDS: tuple[str, ...] = (
    "step",
    "episode",
    "steer",
    "longitudinal",
    "throttle",
    "brake",
    "reward",
    "done",
    "done_reason",
    "route_progress",
    "speed_kmh",
    "lane_offset",
    "heading_error",
    "collision",
    "collision_count_event",
    "collision_route_delta_0179487",
    "collision_route_delta_0743590",
    "collision_window_0179487",
    "collision_window_0743590",
    "near_collision_cluster",
    "offroad",
    "stuck",
    "steer_delta_abs",
    "longitudinal_delta_abs",
    "action_jitter_l1",
    "fusion_rgb",
    "fusion_lidar",
    "fusion_ego",
    "fusion_route",
    "reliability_rgb",
    "reliability_lidar",
    "reliability_ego",
    "reliability_route",
    "risk_scalar",
    "contribution_rgb",
    "contribution_lidar",
    "contribution_ego",
    "contribution_route",
    "compact_visual_features_enabled",
    "compact_visual_feature_dim",
    "compact_visual_feature_l1_mean",
    "road_compact_feature_l1_mean",
    "lane_compact_feature_l1_mean",
    "visual_road_x_mean",
    "visual_lane_x_mean",
    "visual_lane_center_error_mean",
    "compact_visual_actor_input_l1_mean",
    "visual_actor_delta_l1_mean",
    "visual_zero_action_l1_mean",
    "visual_zero_steer_delta_mean",
    "visual_zero_longitudinal_delta_mean",
    "lane_risk_active",
    "failure_window_active",
    "corrective_need_active",
    "visual_zero_steer_share",
    "visual_zero_signed_steer_delta",
    "visual_center_corrective_sign",
    "visual_center_correction_score",
    "route_offset_corrective_sign",
    "route_visual_center_sign_agreement",
    "visual_zero_corrective_sign",
    "visual_zero_corrective_steer_delta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Behavior Cloning checkpoint in CARLA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default="bc_clean_eval")
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def _load_model(*, config: Any, checkpoint_path: Path, device: torch.device) -> RCSACFirstModel:
    model = RCSACFirstModel(config.model, config.fusion).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "BC checkpoint state_dict mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    return model


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(sum(values) / len(values))


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    count = sum(1 for row in rows if bool(row.get(key, False)))
    return float(count / len(rows))


def _route_cluster_deltas(route_progress: float) -> dict[str, float]:
    return {
        label: abs(float(route_progress) - float(cluster_route))
        for label, cluster_route in COLLISION_CLUSTER_ROUTES
    }


def _nearest_route_cluster(route_progress: float, *, tolerance: float = COLLISION_CLUSTER_TOLERANCE) -> str | None:
    deltas = _route_cluster_deltas(route_progress)
    if not deltas:
        return None
    label, delta = min(deltas.items(), key=lambda item: item[1])
    return label if delta <= float(tolerance) else None


def _safe_diag(diagnostics: dict[str, float], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in diagnostics:
            value = float(diagnostics[name])
            return value if math.isfinite(value) else default
    return default


def _build_log_row(
    *,
    step: int,
    episode: int,
    action: np.ndarray,
    previous_action: np.ndarray | None,
    reward: float,
    done: bool,
    info: dict[str, Any],
    diagnostics: dict[str, float],
    visual_zero_deltas: dict[str, float],
    corrective_route_min: float,
    corrective_route_max: float,
    corrective_abs_lane_offset_min: float,
    corrective_abs_heading_error_min: float,
    visual_center_error_min: float,
) -> dict[str, Any]:
    decoded = decode_stage4_action_v2(action.tolist())
    if previous_action is None:
        steer_delta_abs = 0.0
        longitudinal_delta_abs = 0.0
    else:
        steer_delta_abs = abs(float(action[0]) - float(previous_action[0]))
        longitudinal_delta_abs = abs(float(action[1]) - float(previous_action[1]))
    route_progress = float(info.get("route_progress", 0.0))
    collision_cluster_deltas = _route_cluster_deltas(route_progress)
    nearest_collision_cluster = _nearest_route_cluster(route_progress)
    lane_offset = float(info.get("lane_offset", 0.0))
    heading_error = float(info.get("heading_error", 0.0))
    visual_zero_action_l1 = float(visual_zero_deltas.get("action_l1", 0.0))
    visual_zero_steer_delta = float(visual_zero_deltas.get("steer", 0.0))
    lane_risk_active = abs(lane_offset) >= float(corrective_abs_lane_offset_min) or abs(heading_error) >= float(
        corrective_abs_heading_error_min
    )
    failure_window_active = float(corrective_route_min) <= route_progress <= float(corrective_route_max)
    corrective_need_active = lane_risk_active and failure_window_active
    visual_zero_steer_share = visual_zero_steer_delta / max(visual_zero_action_l1, 1.0e-6)
    signed_steer_delta = float(visual_zero_deltas.get("signed_steer", 0.0))
    visual_center_error = _safe_diag(diagnostics, "visual_lane_center_error_mean")
    visual_center_active = abs(visual_center_error) >= float(visual_center_error_min)
    visual_center_corrective_sign = bool(visual_center_active and signed_steer_delta * visual_center_error < 0.0)
    visual_center_correction_score = float(signed_steer_delta * -visual_center_error)
    route_offset_corrective_sign = bool(lane_risk_active and signed_steer_delta * (-lane_offset) > 0.0)
    route_visual_center_sign_agreement = bool(
        lane_risk_active and visual_center_active and lane_offset * visual_center_error > 0.0
    )
    corrective_steer_delta = visual_zero_steer_delta if route_offset_corrective_sign else 0.0
    row: dict[str, Any] = {
        "step": int(step),
        "episode": int(episode),
        "steer": float(decoded["steer"]),
        "longitudinal": float(decoded["longitudinal"]),
        "throttle": float(decoded["throttle"]),
        "brake": float(decoded["brake"]),
        "reward": float(reward),
        "done": bool(done),
        "done_reason": str(info.get("done_reason", "running")),
        "route_progress": route_progress,
        "speed_kmh": float(info.get("speed_kmh", info.get("speed", 0.0))),
        "lane_offset": lane_offset,
        "heading_error": heading_error,
        "collision": bool(info.get("collision", False)),
        "collision_count_event": bool(info.get("collision", False) and done),
        "collision_route_delta_0179487": float(collision_cluster_deltas["0179487"]),
        "collision_route_delta_0743590": float(collision_cluster_deltas["0743590"]),
        "collision_window_0179487": bool(nearest_collision_cluster == "0179487"),
        "collision_window_0743590": bool(nearest_collision_cluster == "0743590"),
        "near_collision_cluster": bool(nearest_collision_cluster is not None),
        "offroad": bool(info.get("offroad", False)),
        "stuck": bool(info.get("stuck", False)),
        "steer_delta_abs": float(steer_delta_abs),
        "longitudinal_delta_abs": float(longitudinal_delta_abs),
        "action_jitter_l1": float(steer_delta_abs + longitudinal_delta_abs),
        "risk_scalar": _safe_diag(diagnostics, "risk_scalar_mean", "risk_scalar"),
        "visual_zero_action_l1_mean": visual_zero_action_l1,
        "visual_zero_steer_delta_mean": visual_zero_steer_delta,
        "visual_zero_longitudinal_delta_mean": float(visual_zero_deltas.get("longitudinal", 0.0)),
        "lane_risk_active": bool(lane_risk_active),
        "failure_window_active": bool(failure_window_active),
        "corrective_need_active": bool(corrective_need_active),
        "visual_zero_steer_share": float(visual_zero_steer_share),
        "visual_zero_signed_steer_delta": float(signed_steer_delta),
        "visual_center_corrective_sign": bool(visual_center_corrective_sign),
        "visual_center_correction_score": float(visual_center_correction_score),
        "route_offset_corrective_sign": bool(route_offset_corrective_sign),
        "route_visual_center_sign_agreement": bool(route_visual_center_sign_agreement),
        "visual_zero_corrective_sign": bool(route_offset_corrective_sign),
        "visual_zero_corrective_steer_delta": float(corrective_steer_delta),
    }
    for modality in ("rgb", "lidar", "ego", "route"):
        row[f"fusion_{modality}"] = _safe_diag(
            diagnostics,
            f"fusion_weight_{modality}",
            f"fusion_{modality}",
        )
        row[f"reliability_{modality}"] = _safe_diag(
            diagnostics,
            f"reliability_{modality}",
            default=1.0,
        )
        row[f"contribution_{modality}"] = _safe_diag(
            diagnostics,
            f"contribution_{modality}",
            default=0.25,
        )
    for key in (
        "compact_visual_features_enabled",
        "compact_visual_feature_dim",
        "compact_visual_feature_l1_mean",
        "road_compact_feature_l1_mean",
        "lane_compact_feature_l1_mean",
        "visual_road_x_mean",
        "visual_lane_x_mean",
        "visual_lane_center_error_mean",
        "compact_visual_actor_input_l1_mean",
        "visual_actor_delta_l1_mean",
    ):
        row[key] = _safe_diag(diagnostics, key)
    return row


def _finite_rows(rows: list[dict[str, Any]]) -> bool:
    numeric_keys = [field for field in BC_EVAL_FIELDS if field not in {"done", "done_reason", "collision", "offroad", "stuck"}]
    for row in rows:
        for key in numeric_keys:
            value = row.get(key)
            if value is None:
                return False
            try:
                if not math.isfinite(float(value)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _mean_abs(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    values = [abs(float(row.get(key, 0.0))) for row in rows]
    return float(sum(values) / len(values))


def _max_abs(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(abs(float(row.get(key, 0.0))) for row in rows))


def _collision_context_events(rows: list[dict[str, Any]], *, window_steps: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("collision_count_event", False)):
            continue
        episode = int(row["episode"])
        step = int(row["step"])
        window_start = step - int(window_steps) + 1
        window_rows = [
            candidate
            for candidate in rows
            if int(candidate["episode"]) == episode and window_start <= int(candidate["step"]) <= step
        ]
        route_progress = float(row["route_progress"])
        nearest_cluster = _nearest_route_cluster(route_progress) or "other"
        events.append(
            {
                "step": step,
                "episode": episode,
                "cluster": nearest_cluster,
                "route_progress": route_progress,
                "speed_kmh": float(row["speed_kmh"]),
                "lane_offset": float(row["lane_offset"]),
                "heading_error": float(row["heading_error"]),
                "steer": float(row["steer"]),
                "longitudinal": float(row["longitudinal"]),
                "risk_scalar": float(row.get("risk_scalar", 0.0)),
                "visual_zero_signed_steer_delta": float(row.get("visual_zero_signed_steer_delta", 0.0)),
                "visual_center_corrective_sign": bool(row.get("visual_center_corrective_sign", False)),
                "route_offset_corrective_sign": bool(row.get("route_offset_corrective_sign", False)),
                "route_visual_center_sign_agreement": bool(row.get("route_visual_center_sign_agreement", False)),
                "window_step_count": int(len(window_rows)),
                "window_mean_speed_kmh": _mean(window_rows, "speed_kmh"),
                "window_mean_abs_lane_offset": _mean_abs(window_rows, "lane_offset"),
                "window_max_abs_lane_offset": _max_abs(window_rows, "lane_offset"),
                "window_mean_abs_heading_error": _mean_abs(window_rows, "heading_error"),
                "window_max_abs_heading_error": _max_abs(window_rows, "heading_error"),
                "window_mean_action_jitter_l1": _mean(window_rows, "action_jitter_l1"),
                "window_mean_visual_zero_steer_delta": _mean(window_rows, "visual_zero_steer_delta_mean"),
                "window_visual_center_correction_rate": _rate(window_rows, "visual_center_corrective_sign"),
                "window_route_offset_corrective_sign_rate": _rate(window_rows, "route_offset_corrective_sign"),
                "window_route_visual_center_sign_agreement_rate": _rate(
                    window_rows,
                    "route_visual_center_sign_agreement",
                ),
            }
        )
    return events


def _collision_cluster_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(event.get("cluster", "other")) for event in events)
    return dict(counts)


def _build_summary(
    *,
    run_name: str,
    checkpoint_path: Path,
    num_steps: int,
    rows: list[dict[str, Any]],
    output_csv: Path,
    ppo_gate: dict[str, float],
) -> dict[str, Any]:
    completed_steps = len(rows)
    route_values = [float(row["route_progress"]) for row in rows]
    done_reasons = Counter(str(row["done_reason"]) for row in rows if bool(row["done"]))
    action_range_ok = all(-1.0001 <= float(row["steer"]) <= 1.0001 and -1.0001 <= float(row["longitudinal"]) <= 1.0001 for row in rows)
    finite_ok = _finite_rows(rows)
    mean_route_progress = _mean(rows, "route_progress")
    collision_rate = _rate(rows, "collision")
    collision_events = _collision_context_events(rows, window_steps=COLLISION_CONTEXT_WINDOW_STEPS)
    collision_count = len(collision_events)
    collision_window_rows = [
        row for row in rows if bool(row.get("near_collision_cluster", False))
    ]
    offroad_rate = _rate(rows, "offroad")
    stuck_rate = _rate(rows, "stuck")
    mean_steer_jitter = _mean(rows, "steer_delta_abs")
    mean_longitudinal_jitter = _mean(rows, "longitudinal_delta_abs")
    mean_action_jitter_l1 = _mean(rows, "action_jitter_l1")
    mean_visual_zero_action_l1 = _mean(rows, "visual_zero_action_l1_mean")
    mean_visual_zero_steer_delta = _mean(rows, "visual_zero_steer_delta_mean")
    mean_visual_zero_longitudinal_delta = _mean(rows, "visual_zero_longitudinal_delta_mean")
    corrective_rows = [row for row in rows if bool(row.get("corrective_need_active", False))]
    corrective_count = len(corrective_rows)
    mean_corrective_visual_zero_steer_delta = _mean(corrective_rows, "visual_zero_steer_delta_mean")
    mean_corrective_visual_zero_steer_share = _mean(corrective_rows, "visual_zero_steer_share")
    mean_corrective_steer_delta = _mean(corrective_rows, "visual_zero_corrective_steer_delta")
    route_offset_corrective_sign_rate = _rate(corrective_rows, "route_offset_corrective_sign")
    corrective_sign_rate = route_offset_corrective_sign_rate
    visual_center_correction_rate = _rate(corrective_rows, "visual_center_corrective_sign")
    mean_visual_center_correction_score = _mean(corrective_rows, "visual_center_correction_score")
    route_visual_center_sign_agreement_rate = _rate(corrective_rows, "route_visual_center_sign_agreement")
    early_offroad_count = sum(
        1 for row in rows if bool(row.get("offroad", False)) and float(row.get("route_progress", 0.0)) < 0.20
    )

    pass_reasons: list[str] = []
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    ppo_gate_fail_reasons: list[str] = []

    if completed_steps <= 0:
        fail_reasons.append("no rollout steps were completed")
    if completed_steps < int(num_steps):
        fail_reasons.append(f"completed_steps {completed_steps} < requested num_steps {num_steps}")
    if not finite_ok:
        fail_reasons.append("non-finite value detected in evaluation rows")
    if not action_range_ok:
        fail_reasons.append("action outside [-1, 1] detected")
    if max(route_values, default=0.0) <= 0.0:
        fail_reasons.append("route_progress never became positive")
    if mean_route_progress <= 0.0:
        fail_reasons.append("mean_route_progress is not positive")
    if collision_rate > 0.50:
        fail_reasons.append(f"collision_rate {collision_rate:.3f} > 0.50")
    if stuck_rate > 0.50:
        fail_reasons.append(f"stuck_rate {stuck_rate:.3f} > 0.50")
    if not math.isfinite(mean_action_jitter_l1):
        fail_reasons.append("action jitter is non-finite")

    if collision_rate > 0.30:
        warn_reasons.append(f"collision_rate {collision_rate:.3f} exceeds PPO fine-tuning recommendation 0.30")
    if stuck_rate > 0.30:
        warn_reasons.append(f"stuck_rate {stuck_rate:.3f} exceeds PPO fine-tuning recommendation 0.30")
    if mean_steer_jitter > 0.30:
        warn_reasons.append(f"mean_steer_jitter {mean_steer_jitter:.3f} exceeds recommendation 0.30")
    if mean_longitudinal_jitter > 0.35:
        warn_reasons.append(
            f"mean_longitudinal_jitter {mean_longitudinal_jitter:.3f} exceeds recommendation 0.35"
        )
    offroad_count = int(done_reasons.get("offroad", 0))
    if offroad_count > int(ppo_gate["max_offroad"]):
        ppo_gate_fail_reasons.append(f"offroad_count {offroad_count} > {int(ppo_gate['max_offroad'])}")
    if early_offroad_count > 0:
        ppo_gate_fail_reasons.append(f"early_offroad_count {early_offroad_count} > 0")
    if max(route_values, default=0.0) < float(ppo_gate["min_max_route"]):
        ppo_gate_fail_reasons.append(
            f"max_route_progress {max(route_values, default=0.0):.6f} < {float(ppo_gate['min_max_route']):.6f}"
        )
    if mean_route_progress < float(ppo_gate["min_mean_route"]):
        ppo_gate_fail_reasons.append(
            f"mean_route_progress {mean_route_progress:.6f} < {float(ppo_gate['min_mean_route']):.6f}"
        )
    if mean_action_jitter_l1 > float(ppo_gate["max_action_jitter"]):
        ppo_gate_fail_reasons.append(
            f"mean_action_jitter_l1 {mean_action_jitter_l1:.6f} > {float(ppo_gate['max_action_jitter']):.6f}"
        )
    if mean_visual_zero_action_l1 < float(ppo_gate["min_visual_zero_action_l1"]):
        ppo_gate_fail_reasons.append(
            f"mean_visual_zero_action_l1 {mean_visual_zero_action_l1:.6f} < "
            f"{float(ppo_gate['min_visual_zero_action_l1']):.6f}"
        )
    if mean_visual_zero_steer_delta < float(ppo_gate["min_visual_zero_steer_delta"]):
        ppo_gate_fail_reasons.append(
            f"mean_visual_zero_steer_delta {mean_visual_zero_steer_delta:.6f} < "
            f"{float(ppo_gate['min_visual_zero_steer_delta']):.6f}"
        )
    if corrective_count < int(ppo_gate["min_corrective_count"]):
        ppo_gate_fail_reasons.append(f"corrective_need_count {corrective_count} < {int(ppo_gate['min_corrective_count'])}")
    if mean_corrective_visual_zero_steer_delta < float(ppo_gate["min_corrective_visual_zero_steer_delta"]):
        ppo_gate_fail_reasons.append(
            f"mean_corrective_visual_zero_steer_delta {mean_corrective_visual_zero_steer_delta:.6f} < "
            f"{float(ppo_gate['min_corrective_visual_zero_steer_delta']):.6f}"
        )
    corrective_sign_source = str(ppo_gate.get("corrective_sign_source", "route_offset")).strip().lower()
    enforce_route_offset_sign = corrective_sign_source in {"route_offset", "lane_offset", "legacy", "both"}
    enforce_visual_center_sign = corrective_sign_source in {"visual_center", "rgb_center", "both"}
    if corrective_sign_source not in {"route_offset", "lane_offset", "legacy", "visual_center", "rgb_center", "both"}:
        ppo_gate_fail_reasons.append(f"unknown corrective_sign_source {corrective_sign_source!r}")
    if enforce_route_offset_sign and route_offset_corrective_sign_rate < float(ppo_gate["min_corrective_sign_rate"]):
        ppo_gate_fail_reasons.append(
            "route_offset_corrective_sign_rate "
            f"{route_offset_corrective_sign_rate:.6f} < {float(ppo_gate['min_corrective_sign_rate']):.6f}"
        )
    if enforce_visual_center_sign and visual_center_correction_rate < float(ppo_gate["min_visual_center_correction_rate"]):
        ppo_gate_fail_reasons.append(
            "visual_center_correction_rate "
            f"{visual_center_correction_rate:.6f} < {float(ppo_gate['min_visual_center_correction_rate']):.6f}"
        )

    if not fail_reasons:
        pass_reasons.append("BC clean rollout completed with finite in-range actions and positive route progress")
    if fail_reasons:
        pass_status = "FAIL"
        pass_reasons.extend(fail_reasons)
    elif warn_reasons:
        pass_status = "WARN"
        pass_reasons.extend(warn_reasons)
    else:
        pass_status = "PASS"
        pass_reasons.append("BC checkpoint satisfies initial and recommended I2 thresholds except random-policy comparison")

    return {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "num_steps": int(num_steps),
        "completed_steps": int(completed_steps),
        "episode_count": int(max((int(row["episode"]) for row in rows), default=0) + 1 if rows else 0),
        "final_route_progress": float(route_values[-1]) if route_values else 0.0,
        "mean_route_progress": float(mean_route_progress),
        "max_route_progress": float(max(route_values, default=0.0)),
        "collision_rate": float(collision_rate),
        "collision_count": int(collision_count),
        "collision_context_window_steps": int(COLLISION_CONTEXT_WINDOW_STEPS),
        "collision_cluster_counts": _collision_cluster_counts(collision_events),
        "collision_events": collision_events,
        "near_collision_cluster_row_count": int(len(collision_window_rows)),
        "near_collision_cluster_route_offset_corrective_sign_rate": _rate(
            collision_window_rows,
            "route_offset_corrective_sign",
        ),
        "near_collision_cluster_visual_center_correction_rate": _rate(
            collision_window_rows,
            "visual_center_corrective_sign",
        ),
        "near_collision_cluster_route_visual_agreement_rate": _rate(
            collision_window_rows,
            "route_visual_center_sign_agreement",
        ),
        "offroad_rate": float(offroad_rate),
        "stuck_rate": float(stuck_rate),
        "mean_steer_jitter": float(mean_steer_jitter),
        "mean_longitudinal_jitter": float(mean_longitudinal_jitter),
        "mean_action_jitter_l1": float(mean_action_jitter_l1),
        "mean_visual_zero_action_l1": float(mean_visual_zero_action_l1),
        "mean_visual_zero_steer_delta": float(mean_visual_zero_steer_delta),
        "mean_visual_zero_longitudinal_delta": float(mean_visual_zero_longitudinal_delta),
        "corrective_need_count": int(corrective_count),
        "mean_corrective_visual_zero_steer_delta": float(mean_corrective_visual_zero_steer_delta),
        "mean_corrective_visual_zero_steer_share": float(mean_corrective_visual_zero_steer_share),
        "mean_corrective_steer_delta": float(mean_corrective_steer_delta),
        "corrective_sign_rate": float(corrective_sign_rate),
        "corrective_sign_source": corrective_sign_source,
        "route_offset_corrective_sign_rate": float(route_offset_corrective_sign_rate),
        "visual_center_correction_rate": float(visual_center_correction_rate),
        "mean_visual_center_correction_score": float(mean_visual_center_correction_score),
        "route_visual_center_sign_agreement_rate": float(route_visual_center_sign_agreement_rate),
        "early_offroad_count": int(early_offroad_count),
        "ppo_gate_status": "PASS" if not ppo_gate_fail_reasons else "BLOCK",
        "ppo_gate_fail_reasons": ppo_gate_fail_reasons,
        "done_reason_counts": dict(done_reasons),
        "pass_status": pass_status,
        "pass_reasons": pass_reasons,
        "output_csv": str(output_csv),
    }


def main() -> int:
    args = parse_args()
    config_path = _resolve_config_path(str(args.config))
    config = load_phase1_config(config_path)
    eval_raw = config.raw.get("bc_eval") or config.raw.get("eval") or {}
    if not isinstance(eval_raw, dict):
        eval_raw = {}
    corrective_route_min = float(eval_raw.get("corrective_route_min", 0.12))
    corrective_route_max = float(eval_raw.get("corrective_route_max", 0.23))
    corrective_abs_lane_offset_min = float(eval_raw.get("corrective_abs_lane_offset_min", 1.8))
    corrective_abs_heading_error_min = float(eval_raw.get("corrective_abs_heading_error_min", 8.0))
    visual_center_error_min = float(eval_raw.get("visual_center_error_min", 0.05))
    ppo_gate = {
        "max_offroad": float(eval_raw.get("ppo_gate_max_offroad", 2)),
        "min_max_route": float(eval_raw.get("ppo_gate_min_max_route", 0.230769)),
        "min_mean_route": float(eval_raw.get("ppo_gate_min_mean_route", 0.0)),
        "max_action_jitter": float(eval_raw.get("ppo_gate_max_action_jitter", 0.07)),
        "min_visual_zero_action_l1": float(eval_raw.get("ppo_gate_min_visual_zero_action_l1", 0.012)),
        "min_visual_zero_steer_delta": float(eval_raw.get("ppo_gate_min_visual_zero_steer_delta", 0.003)),
        "min_corrective_count": float(eval_raw.get("ppo_gate_min_corrective_count", 1)),
        "min_corrective_visual_zero_steer_delta": float(
            eval_raw.get("ppo_gate_min_corrective_visual_zero_steer_delta", 0.004)
        ),
        "min_corrective_sign_rate": float(eval_raw.get("ppo_gate_min_corrective_sign_rate", 0.65)),
        "min_visual_center_correction_rate": float(eval_raw.get("ppo_gate_min_visual_center_correction_rate", 0.65)),
        "corrective_sign_source": str(eval_raw.get("ppo_gate_corrective_sign_source", "route_offset")),
    }
    device = torch.device(config.train.resolve_device(args.device))
    _set_seed(config.train.seed)

    checkpoint_path = _project_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")

    run_name = f"{args.run_name}_{_timestamp()}"
    output_dir = PROJECT_ROOT / "outputs" / "bc_eval" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "bc_eval_log.csv"
    summary_path = output_dir / "summary.json"

    obs_spec = build_policy_observation_spec(
        rgb_shape=config.model.rgb_shape,
        lidar_shape=config.model.lidar_shape,
        ego_dim=config.model.ego_dim,
        route_dim=config.model.route_dim,
    )
    aux_spec = build_auxiliary_observation_spec(
        quality_stats_dim=config.model.quality_stats_dim,
        risk_stats_dim=config.model.risk_stats_dim,
        route_commitment_stats_dim=config.model.route_commitment_stats_dim,
        sensor_dropout_dim=config.model.sensor_dropout_dim,
    )
    model = _load_model(config=config, checkpoint_path=checkpoint_path, device=device)
    env = _build_env(
        config,
        host=args.host or config.train.host,
        port=int(args.port if args.port is not None else config.train.port),
        output_dir=output_dir,
        run_name=run_name,
    )

    rows: list[dict[str, Any]] = []
    try:
        reset = env.reset()
        observation = flatten_observation_dict(reset["obs"], obs_spec)
        auxiliary = flatten_auxiliary_observation_dict(reset["obs"], aux_spec)
        hidden_state: torch.Tensor | None = None
        previous_action: np.ndarray | None = None
        episode = 0

        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BC_EVAL_FIELDS)
            writer.writeheader()

            for step in range(int(args.num_steps)):
                obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
                aux_t = torch.as_tensor(auxiliary, dtype=torch.float32, device=device).reshape(1, -1)
                with torch.no_grad():
                    outputs = model.forward_policy_outputs(
                        obs_t,
                        auxiliary=aux_t,
                        hidden_state=hidden_state,
                        track_diagnostics=True,
                    )
                    action_mean = outputs["action_mean"]
                    if not isinstance(action_mean, torch.Tensor):
                        raise RuntimeError("Model output action_mean is missing.")
                    if not torch.isfinite(action_mean).all():
                        raise RuntimeError("BC policy produced non-finite action_mean.")
                    action_t = clamp_action_tensor(action_mean[0])
                    zero_outputs = model.forward_policy_outputs(
                        obs_t,
                        auxiliary=aux_t,
                        hidden_state=hidden_state,
                        track_diagnostics=False,
                        zero_rgb_perception_actor_features=True,
                    )
                    zero_action_mean = zero_outputs["action_mean"]
                    if not isinstance(zero_action_mean, torch.Tensor):
                        raise RuntimeError("Model zero-visual output action_mean is missing.")
                    zero_action_t = clamp_action_tensor(zero_action_mean[0])
                    visual_zero_diff = (action_t - zero_action_t).detach().abs().cpu().numpy().astype(np.float32)
                    visual_zero_deltas = {
                        "steer": float(visual_zero_diff[0]),
                        "signed_steer": float((action_t[0] - zero_action_t[0]).detach().cpu().item()),
                        "longitudinal": float(visual_zero_diff[1]),
                        "action_l1": float(visual_zero_diff.sum()),
                    }
                    next_hidden = outputs["next_hidden_state"]
                    hidden_state = next_hidden.detach() if isinstance(next_hidden, torch.Tensor) else None
                action = action_t.detach().cpu().numpy().astype(np.float32)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError(f"BC policy produced non-finite action: {action}")

                diagnostics = model.get_diagnostics()
                result = env.step(action, policy_diagnostics=diagnostics)
                info = dict(result["info"])
                reward = float(result["reward"])
                done = bool(result["done"])
                row = _build_log_row(
                    step=step,
                    episode=episode,
                    action=action,
                    previous_action=previous_action,
                    reward=reward,
                    done=done,
                    info=info,
                    diagnostics=diagnostics,
                    visual_zero_deltas=visual_zero_deltas,
                    corrective_route_min=corrective_route_min,
                    corrective_route_max=corrective_route_max,
                    corrective_abs_lane_offset_min=corrective_abs_lane_offset_min,
                    corrective_abs_heading_error_min=corrective_abs_heading_error_min,
                    visual_center_error_min=visual_center_error_min,
                )
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                previous_action = action.copy()

                observation = flatten_observation_dict(result["obs"], obs_spec)
                auxiliary = flatten_auxiliary_observation_dict(result["obs"], aux_spec)
                if done and step + 1 < int(args.num_steps):
                    episode += 1
                    hidden_state = None
                    previous_action = None
                    reset = env.reset()
                    observation = flatten_observation_dict(reset["obs"], obs_spec)
                    auxiliary = flatten_auxiliary_observation_dict(reset["obs"], aux_spec)

        summary = _build_summary(
            run_name=run_name,
            checkpoint_path=checkpoint_path,
            num_steps=int(args.num_steps),
            rows=rows,
            output_csv=log_path,
            ppo_gate=ppo_gate,
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["pass_status"] in {"PASS", "WARN"} else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
