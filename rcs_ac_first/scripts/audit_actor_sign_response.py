from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rcs_ac_first.data.expert_dataset import load_expert_npz
from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.runtime.auxiliary import RISK_STATS_FIELDS, build_auxiliary_observation_spec, default_flat_auxiliary
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw actor steering sign response to scalar probes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-selector", default="index", choices=["index", "early_left", "early_any"])
    parser.add_argument("--early-route-max", type=float, default=0.08)
    parser.add_argument("--early-left-lane-max", type=float, default=-0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="outputs/actor_sign_audit")
    parser.add_argument("--run-name", default="actor_sign_audit")
    parser.add_argument("--probe-values", default="-1.0,0.0,1.0")
    parser.add_argument("--lane-offset-scale", type=float, default=1.0)
    parser.add_argument("--heading-error-scale", type=float, default=10.0)
    parser.add_argument("--route-y-index", type=int, default=1)
    parser.add_argument("--route-y-scale", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=0.005)
    return parser.parse_args()


def _resolve_device(config: Any, override: str) -> torch.device:
    if override == "auto":
        return torch.device(config.train.resolve_device(None))
    if override == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(override)


def _load_model(*, config: Any, checkpoint_path: Path, device: torch.device) -> RCSACFirstModel:
    model = RCSACFirstModel(config.model, config.fusion).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint state_dict mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    model.eval()
    return model


def _parse_probe_values(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) < 3:
        raise ValueError("--probe-values must contain at least negative, zero, and positive values.")
    if 0.0 not in values:
        values.append(0.0)
    return sorted(values)


def _load_base_tensors(
    *,
    config: Any,
    dataset_path: Path | None,
    sample_index: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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
    if dataset_path is not None:
        arrays = load_expert_npz(dataset_path)
        if int(arrays["obs_flat"].shape[1]) != int(obs_spec.total_dim):
            raise ValueError(f"Dataset obs dim {int(arrays['obs_flat'].shape[1])} != config obs dim {obs_spec.total_dim}")
        if int(arrays["auxiliary"].shape[1]) != int(aux_spec.total_dim):
            raise ValueError(
                f"Dataset auxiliary dim {int(arrays['auxiliary'].shape[1])} != config auxiliary dim {aux_spec.total_dim}"
            )
        if int(arrays["expert_action"].shape[1]) != int(config.model.action_dim):
            raise ValueError(
                f"Dataset action dim {int(arrays['expert_action'].shape[1])} != config action dim {config.model.action_dim}"
            )
        index = _select_sample_index(arrays, sample_index=sample_index, args=args)
        obs = torch.as_tensor(arrays["obs_flat"][index], dtype=torch.float32, device=device).unsqueeze(0)
        aux = torch.as_tensor(arrays["auxiliary"][index], dtype=torch.float32, device=device).unsqueeze(0)
        meta = {
            "source": "dataset",
            "dataset": str(dataset_path),
            "sample_selector": str(args.sample_selector),
            "sample_index": int(index),
            "episode_id": int(arrays["episode_id"][index]),
            "step_in_episode": int(arrays["step_in_episode"][index]),
            "route_progress": float(arrays["route_progress"][index]),
            "lane_offset": float(np.asarray(arrays.get("lane_offset", np.zeros_like(arrays["route_progress"]))).reshape(-1)[index]),
            "heading_error": float(np.asarray(arrays.get("heading_error", np.zeros_like(arrays["route_progress"]))).reshape(-1)[index]),
        }
    else:
        obs = torch.zeros((1, obs_spec.total_dim), dtype=torch.float32, device=device)
        aux = default_flat_auxiliary(1, spec=aux_spec, device=device)
        meta = {
            "source": "zero_template",
            "dataset": "",
            "sample_selector": "none",
            "sample_index": 0,
            "episode_id": -1,
            "step_in_episode": -1,
            "route_progress": 0.0,
            "lane_offset": 0.0,
            "heading_error": 0.0,
        }
    return obs, aux, meta


def _select_sample_index(arrays: dict[str, np.ndarray], *, sample_index: int, args: argparse.Namespace) -> int:
    n = int(arrays["obs_flat"].shape[0])
    if str(args.sample_selector) == "index":
        return int(np.clip(sample_index, 0, n - 1))
    route_progress = np.asarray(arrays["route_progress"], dtype=np.float32).reshape(-1)
    if str(args.sample_selector) == "early_any":
        candidates = np.flatnonzero(route_progress <= float(args.early_route_max))
    else:
        lane_offset = np.asarray(arrays.get("lane_offset", np.zeros_like(route_progress)), dtype=np.float32).reshape(-1)
        candidates = np.flatnonzero(
            (route_progress <= float(args.early_route_max)) & (lane_offset <= float(args.early_left_lane_max))
        )
    if candidates.size == 0:
        raise ValueError(
            f"No samples matched selector={args.sample_selector!r} "
            f"early_route_max={float(args.early_route_max)} early_left_lane_max={float(args.early_left_lane_max)}"
        )
    return int(candidates[0])


def _forward(model: RCSACFirstModel, obs: torch.Tensor, aux: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        outputs = model.forward_policy_outputs(obs, auxiliary=aux, track_diagnostics=True)
        zero_outputs = model.forward_policy_outputs(
            obs,
            auxiliary=aux,
            track_diagnostics=False,
            zero_rgb_perception_actor_features=True,
        )
    action = outputs["action_mean"]
    zero_action = zero_outputs["action_mean"]
    if not isinstance(action, torch.Tensor) or not isinstance(zero_action, torch.Tensor):
        raise RuntimeError("Model did not return action tensors.")
    if tuple(action.shape) != (1, 2):
        raise RuntimeError(f"Expected action_mean shape [1, 2], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise RuntimeError("action_mean contains non-finite values.")
    diagnostics = model.get_diagnostics()
    return {
        "steer": float(action[0, 0].detach().cpu().item()),
        "longitudinal": float(action[0, 1].detach().cpu().item()),
        "visual_zero_steer": float(zero_action[0, 0].detach().cpu().item()),
        "visual_zero_steer_delta": float((action[0, 0] - zero_action[0, 0]).detach().cpu().item()),
        "fusion_weight_rgb": float(diagnostics.get("fusion_weight_rgb", 0.0)),
        "fusion_weight_route": float(diagnostics.get("fusion_weight_route", 0.0)),
        "visual_lane_center_error": float(diagnostics.get("visual_lane_center_error_mean", 0.0)),
        "visual_actor_delta_l1": float(diagnostics.get("visual_actor_delta_l1_mean", 0.0)),
    }


def _risk_index(field: str) -> int:
    spec = build_auxiliary_observation_spec()
    risk = spec.item("risk_stats")
    return int(risk.start + RISK_STATS_FIELDS.index(field))


def _route_start(config: Any) -> int:
    obs_spec = build_policy_observation_spec(
        rgb_shape=config.model.rgb_shape,
        lidar_shape=config.model.lidar_shape,
        ego_dim=config.model.ego_dim,
        route_dim=config.model.route_dim,
    )
    return int(obs_spec.modal("route").start)


def _make_probe(
    *,
    config: Any,
    base_obs: torch.Tensor,
    base_aux: torch.Tensor,
    axis: str,
    value: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    obs = base_obs.clone()
    aux = base_aux.clone()
    if axis == "lane_offset":
        actual = float(value) * float(args.lane_offset_scale)
        aux[:, _risk_index("lane_offset")] = actual
    elif axis == "heading_error":
        actual = float(value) * float(args.heading_error_scale)
        aux[:, _risk_index("heading_error")] = actual
    elif axis == "route_target_y":
        route_index = int(args.route_y_index)
        if route_index < 0 or route_index >= int(config.model.route_dim):
            raise ValueError(f"route-y-index out of range: {route_index}")
        actual = float(value) * float(args.route_y_scale)
        obs[:, _route_start(config) + route_index] = actual
    else:
        raise ValueError(f"Unknown probe axis: {axis}")
    return obs, aux, actual


def _axis_summary(rows: list[dict[str, Any]], *, min_delta: float) -> dict[str, Any]:
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_axis.setdefault(str(row["axis"]), []).append(row)
    result: dict[str, Any] = {}
    for axis, axis_rows in by_axis.items():
        zero = min(axis_rows, key=lambda row: abs(float(row["probe_value"])))
        neg = min((row for row in axis_rows if float(row["probe_value"]) < 0.0), key=lambda row: abs(float(row["probe_value"])), default=None)
        pos = min((row for row in axis_rows if float(row["probe_value"]) > 0.0), key=lambda row: abs(float(row["probe_value"])), default=None)
        sign_pass = False
        monotonic_pass = False
        min_abs_delta = 0.0
        symmetry_error = 0.0
        if neg is not None and pos is not None:
            neg_delta = float(neg["steer"]) - float(zero["steer"])
            pos_delta = float(zero["steer"]) - float(pos["steer"])
            min_abs_delta = min(abs(neg_delta), abs(pos_delta))
            sign_pass = neg_delta > float(min_delta) and pos_delta > float(min_delta)
            ordered = sorted(axis_rows, key=lambda row: float(row["probe_value"]))
            monotonic_pass = all(float(left["steer"]) >= float(right["steer"]) for left, right in zip(ordered, ordered[1:]))
            symmetry_error = abs((float(neg["steer"]) + float(pos["steer"])) / 2.0 - float(zero["steer"]))
        result[axis] = {
            "zero_steer": float(zero["steer"]),
            "sign_pass": bool(sign_pass),
            "monotonic_pass": bool(monotonic_pass),
            "min_abs_delta": float(min_abs_delta),
            "symmetry_error": float(symmetry_error),
        }
    return result


def main() -> int:
    args = parse_args()
    config_path = _project_path(args.config)
    checkpoint_path = _project_path(args.checkpoint)
    dataset_path = _project_path(args.dataset) if args.dataset else None
    config = load_phase1_config(config_path)
    device = _resolve_device(config, args.device)
    model = _load_model(config=config, checkpoint_path=checkpoint_path, device=device)
    base_obs, base_aux, source_meta = _load_base_tensors(
        config=config,
        dataset_path=dataset_path,
        sample_index=int(args.sample_index),
        device=device,
        args=args,
    )
    values = _parse_probe_values(str(args.probe_values))
    axes = ("lane_offset", "heading_error", "route_target_y")
    rows: list[dict[str, Any]] = []
    for axis in axes:
        for value in values:
            obs, aux, actual = _make_probe(
                config=config,
                base_obs=base_obs,
                base_aux=base_aux,
                axis=axis,
                value=float(value),
                args=args,
            )
            metrics = _forward(model, obs, aux)
            rows.append(
                {
                    "axis": axis,
                    "probe_value": float(value),
                    "actual_value": float(actual),
                    **metrics,
                }
            )

    output_dir = _project_path(args.output_dir) / str(args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "actor_sign_audit.csv"
    summary_path = output_dir / "summary.json"
    fieldnames = [
        "axis",
        "probe_value",
        "actual_value",
        "steer",
        "longitudinal",
        "visual_zero_steer",
        "visual_zero_steer_delta",
        "fusion_weight_rgb",
        "fusion_weight_route",
        "visual_lane_center_error",
        "visual_actor_delta_l1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    axis_summary = _axis_summary(rows, min_delta=float(args.min_delta))
    sign_pass_rate = sum(1 for item in axis_summary.values() if item["sign_pass"]) / max(1, len(axis_summary))
    monotonic_rate = sum(1 for item in axis_summary.values() if item["monotonic_pass"]) / max(1, len(axis_summary))
    min_abs_delta = min((float(item["min_abs_delta"]) for item in axis_summary.values()), default=0.0)
    zero_left_bias = min((float(item["zero_steer"]) for item in axis_summary.values()), default=0.0)
    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "source": source_meta,
        "route_y_index": int(args.route_y_index),
        "min_delta": float(args.min_delta),
        "axis_summary": axis_summary,
        "actor_sign_response_all_axes_rate": float(sign_pass_rate),
        "actor_sign_response_monotonic_rate": float(monotonic_rate),
        "actor_sign_response_min_abs_delta": float(min_abs_delta),
        "actor_sign_response_left_bias_at_zero": float(zero_left_bias),
        "pass_status": "PASS"
        if sign_pass_rate >= 0.999 and monotonic_rate >= 0.999 and min_abs_delta >= float(args.min_delta) and zero_left_bias >= -0.03
        else "BLOCK",
        "csv": str(csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
