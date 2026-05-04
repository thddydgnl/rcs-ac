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

from rcs_ac_v1_1.adapters.carla_observation_adapter import build_v11_flat_obs_from_carla_step
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config
from rcs_ac_v1_1.runtime.observation import flatten_observation_dict

REQUIRED_LOG_COLUMNS: tuple[str, ...] = (
    "step",
    "episode_step",
    "reward",
    "done",
    "done_reason",
    "collision",
    "offroad",
    "stuck",
    "route_progress",
    "route_progress_delta",
    "speed_kmh",
    "steer",
    "throttle",
    "brake",
    "target_speed_kmh_raw",
    "target_speed_capped_kmh",
    "waypoint_valid",
    "path_valid",
    "used_route_fallback",
    "used_synthetic_fallback",
    "lookahead_m",
    "raw_steer",
    "smoothed_steer",
    "rate_limited_steer",
    "bev_fusion_weight_rgb",
    "bev_fusion_weight_lidar",
    "bev_fusion_weight_route",
    "bev_fusion_weight_ego",
    "obs_flat_dim",
    "obs_finite",
    "rgb_mean",
    "lidar_occupied_sum",
    "route_bev_path_sum",
    "ego_bev_footprint_sum",
    "route_point_count",
    "lidar_point_count",
    *tuple(f"sensor_health_{idx}" for idx in range(15)),
)


def build_flat_obs_from_components(components: dict[str, np.ndarray], config_path: str | Path) -> np.ndarray:
    runtime_config = load_v11_runtime_config(config_path)
    return flatten_observation_dict(components, runtime_config.observation_spec)


def load_policy_checkpoint(config_path: str | Path, checkpoint_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    runtime_config = load_v11_runtime_config(config_path)
    model = runtime_config.build_policy().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def synthetic_eval_step(config_path: str | Path, checkpoint_path: str | Path | None = None, device: str = "cpu") -> dict[str, object]:
    runtime_config = load_v11_runtime_config(config_path)
    model = runtime_config.build_policy().to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    obs = {
        "rgb": np.zeros(runtime_config.observation_spec.modal("rgb").shape, dtype=np.float32),
        "lidar_bev": np.zeros(runtime_config.observation_spec.modal("lidar_bev").shape, dtype=np.float32),
        "route_bev": np.zeros(runtime_config.observation_spec.modal("route_bev").shape, dtype=np.float32),
        "ego_bev": np.zeros(runtime_config.observation_spec.modal("ego_bev").shape, dtype=np.float32),
        "ego_scalar": np.zeros(runtime_config.observation_spec.modal("ego_scalar").shape, dtype=np.float32),
        "route_scalar": np.zeros(runtime_config.observation_spec.modal("route_scalar").shape, dtype=np.float32),
        "sensor_health": np.ones(runtime_config.observation_spec.modal("sensor_health").shape, dtype=np.float32),
    }
    flat = torch.as_tensor(flatten_observation_dict(obs, runtime_config.observation_spec), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(flat)
    controller = runtime_config.build_controller()
    control = controller.control(
        outputs["waypoints"][0].detach().cpu().numpy(),
        target_speed_kmh=float(outputs["target_speed_kmh"][0, 0].detach().cpu()),
        current_speed_kmh=0.0,
        dt=0.05,
    )
    return {"outputs": outputs, "control": control}


def run_carla_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not args.env_config:
        raise ValueError("--mode carla requires --env-config pointing to an existing rcs_ac_first CARLA config")
    runtime_config = load_v11_runtime_config(args.config)
    device = _resolve_device(args.device)
    model = runtime_config.build_policy().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    elif not args.allow_random_policy_smoke:
        raise ValueError("--mode carla requires --checkpoint unless --allow-random-policy-smoke is set")
    model.eval()

    env_config, env = _build_carla_env(args)
    del env_config
    output_dir = Path(args.output_dir or Path("logs") / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "v1_1_eval_steps.csv"
    summary_path = output_dir / "summary.json"
    controller = runtime_config.build_controller()
    rows: list[dict[str, Any]] = []
    previous_control: dict[str, float] | None = None
    previous_route_progress = 0.0
    episode_step = 0
    exit_status = "completed"
    exception_type = ""
    exception_message = ""
    try:
        reset = env.reset()
        current_obs = dict(reset.get("obs", {}))
        current_info = dict(reset.get("info", {}))
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_LOG_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for step in range(int(args.num_steps)):
                obs_flat, packet, ego_state, route_context, adapter_debug = build_v11_flat_obs_from_carla_step(
                    obs=current_obs,
                    info=current_info,
                    runtime_config=runtime_config,
                    previous_control=previous_control,
                    ego_vehicle=_get_ego_vehicle(env),
                )
                obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).reshape(1, -1)
                if not torch.isfinite(obs_t).all():
                    raise RuntimeError("v1.1 CARLA observation contains NaN or inf")
                with torch.no_grad():
                    outputs = model(obs_t)
                _validate_model_outputs(outputs)
                waypoints = outputs["waypoints"][0].detach().cpu().numpy()
                fusion = outputs["bev_fusion_weights"][0].detach().cpu().numpy()
                target_speed_raw = float(outputs["target_speed_kmh"][0, 0].detach().cpu())
                route_fallback = None if route_context is None else route_context.route_points_ego
                control = controller.control(
                    waypoints,
                    target_speed_kmh=target_speed_raw,
                    current_speed_kmh=float(ego_state.speed_kmh),
                    dt=float(args.dt),
                    route_fallback_waypoints=route_fallback,
                    uncertainty=0.0,
                )
                step_action = _control_to_env_action(control, args.action_format)
                result = _step_env(env, step_action)
                info = dict(result.get("info", {}))
                reward = float(result.get("reward", info.get("reward", 0.0)))
                done = bool(result.get("done", False))
                debug = dict(control["debug"])
                sensor_health = np.asarray(packet.__dict__.get("_unused", np.zeros(0)), dtype=np.float32)
                del sensor_health
                row = _build_log_row(
                    step=step,
                    episode_step=episode_step,
                    reward=reward,
                    done=done,
                    info=info,
                    ego_state=ego_state,
                    control=control,
                    target_speed_raw=target_speed_raw,
                    fusion=fusion,
                    adapter_debug=adapter_debug,
                    sensor_health=_sensor_health_from_obs(runtime_config, obs_flat),
                    route_progress_delta=float(info.get("route_progress", 0.0)) - previous_route_progress,
                )
                writer.writerow(row)
                rows.append(row)
                previous_route_progress = float(info.get("route_progress", previous_route_progress))
                previous_control = {
                    "steer": float(control["steer"]),
                    "throttle": float(control["throttle"]),
                    "brake": float(control["brake"]),
                    "longitudinal": float(control["throttle"]) - float(control["brake"]),
                }
                current_obs = dict(result.get("obs", {}))
                current_info = info
                episode_step += 1
                if done:
                    break
    except Exception as exc:
        exit_status = "exception"
        exception_type = type(exc).__name__
        exception_message = str(exc)
        raise
    finally:
        summary = _build_summary(
            args=args,
            rows=rows,
            exit_status=exit_status,
            exception_type=exception_type,
            exception_message=exception_message,
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if hasattr(env, "close"):
            env.close()
    return summary


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic":
        result = synthetic_eval_step(args.config, args.checkpoint if args.checkpoint else None, args.device)
        control = result["control"]
        print(
            {
                "steer": control["steer"],
                "throttle": control["throttle"],
                "brake": control["brake"],
                "debug_keys": sorted(control["debug"].keys()),
            }
        )
        return
    summary = run_carla_eval(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCS-AC v1.1 pure-pursuit evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--env-config", default="", help="rcs_ac_first CARLA config used only for --mode carla")
    parser.add_argument("--run-name", default="v1_1_synthetic_eval")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["synthetic", "carla"], default="synthetic")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--action-format", choices=["stage4_v2"], default="stage4_v2")
    parser.add_argument("--allow-random-policy-smoke", action="store_true")
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_carla_env(args: argparse.Namespace) -> tuple[Any, Any]:
    from rcs_ac_first.runtime.config import load_phase1_config
    from rcs_ac_first.scripts.train import _build_env, _resolve_config_path

    config = load_phase1_config(_resolve_config_path(str(args.env_config)))
    output_dir = Path(args.output_dir or Path("logs") / args.run_name)
    env = _build_env(config, host=str(args.host), port=int(args.port), output_dir=output_dir, run_name=str(args.run_name))
    return config, env


def _get_ego_vehicle(env: Any) -> Any:
    for name in ("ego_vehicle", "vehicle", "_ego_vehicle", "_vehicle"):
        if hasattr(env, name):
            return getattr(env, name)
    return None


def _step_env(env: Any, step_action: np.ndarray) -> dict[str, Any]:
    try:
        return env.step(step_action, policy_diagnostics={"v1_1": True})
    except TypeError:
        return env.step(step_action)


def _control_to_env_action(control: dict[str, Any], action_format: str) -> np.ndarray:
    steer = float(control["steer"])
    throttle = float(control["throttle"])
    brake = float(control["brake"])
    return np.asarray([steer, throttle - brake], dtype=np.float32)


def _validate_model_outputs(outputs: dict[str, torch.Tensor]) -> None:
    for key in ("waypoints", "target_speed_kmh", "bev_fusion_weights"):
        if key not in outputs:
            raise RuntimeError(f"v1.1 model output missing {key!r}")
        if not torch.isfinite(outputs[key]).all():
            raise RuntimeError(f"v1.1 model output {key!r} contains NaN or inf")
    if tuple(outputs["waypoints"].shape[1:]) != (4, 2):
        raise RuntimeError(f"unexpected waypoints shape: {tuple(outputs['waypoints'].shape)}")


def _sensor_health_from_obs(runtime_config: Any, obs_flat: np.ndarray) -> np.ndarray:
    spec = runtime_config.observation_spec.modal("sensor_health")
    return np.asarray(obs_flat[spec.start : spec.stop], dtype=np.float32)


def _build_log_row(
    *,
    step: int,
    episode_step: int,
    reward: float,
    done: bool,
    info: dict[str, Any],
    ego_state: Any,
    control: dict[str, Any],
    target_speed_raw: float,
    fusion: np.ndarray,
    adapter_debug: dict[str, Any],
    sensor_health: np.ndarray,
    route_progress_delta: float,
) -> dict[str, Any]:
    debug = dict(control["debug"])
    row: dict[str, Any] = {
        "step": int(step),
        "episode_step": int(episode_step),
        "reward": float(reward),
        "done": bool(done),
        "done_reason": str(info.get("done_reason", "running")),
        "collision": bool(info.get("collision", False)),
        "offroad": bool(info.get("offroad", False)),
        "stuck": bool(info.get("stuck", False)),
        "route_progress": _safe_float(info.get("route_progress", 0.0)),
        "route_progress_delta": float(route_progress_delta),
        "speed_kmh": float(ego_state.speed_kmh),
        "steer": float(control["steer"]),
        "throttle": float(control["throttle"]),
        "brake": float(control["brake"]),
        "target_speed_kmh_raw": float(target_speed_raw),
        "target_speed_capped_kmh": float(debug.get("target_speed_capped_kmh", 0.0)),
        "waypoint_valid": bool(debug.get("waypoint_valid", False)),
        "path_valid": bool(debug.get("path_valid", False)),
        "used_route_fallback": bool(debug.get("used_route_fallback", False)),
        "used_synthetic_fallback": bool(debug.get("used_synthetic_fallback", False)),
        "lookahead_m": float(debug.get("lookahead_m", 0.0)),
        "raw_steer": float(debug.get("raw_steer", 0.0)),
        "smoothed_steer": float(debug.get("smoothed_steer", 0.0)),
        "rate_limited_steer": float(debug.get("rate_limited_steer", 0.0)),
        "bev_fusion_weight_rgb": float(fusion[0]),
        "bev_fusion_weight_lidar": float(fusion[1]),
        "bev_fusion_weight_route": float(fusion[2]),
        "bev_fusion_weight_ego": float(fusion[3]),
        "obs_flat_dim": int(adapter_debug.get("adapter_obs_flat_dim", 0)),
        "obs_finite": bool(adapter_debug.get("adapter_obs_finite", False)),
        "rgb_mean": float(adapter_debug.get("rgb_mean", 0.0)),
        "lidar_occupied_sum": float(adapter_debug.get("lidar_occupied_sum", 0.0)),
        "route_bev_path_sum": float(adapter_debug.get("route_bev_path_sum", 0.0)),
        "ego_bev_footprint_sum": float(adapter_debug.get("ego_bev_footprint_sum", 0.0)),
        "route_point_count": int(adapter_debug.get("adapter_route_point_count", 0)),
        "lidar_point_count": int(adapter_debug.get("adapter_lidar_point_count", adapter_debug.get("lidar_point_count", 0))),
    }
    for idx in range(15):
        row[f"sensor_health_{idx}"] = float(sensor_health[idx]) if idx < sensor_health.shape[0] else 0.0
    return row


def _build_summary(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    exit_status: str,
    exception_type: str,
    exception_message: str,
) -> dict[str, Any]:
    return {
        "run_name": str(args.run_name),
        "num_steps_requested": int(args.num_steps),
        "num_steps_completed": int(len(rows)),
        "exit_status": str(exit_status),
        "crash_type": "",
        "exception_type": str(exception_type),
        "exception_message": str(exception_message),
        "collision_count": int(sum(bool(row.get("collision", False)) for row in rows)),
        "offroad_count": int(sum(bool(row.get("offroad", False)) for row in rows)),
        "stuck_count": int(sum(bool(row.get("stuck", False)) for row in rows)),
        "first_done_step": _first_done(rows)[0],
        "first_done_reason": _first_done(rows)[1],
        "max_route_progress": _max(rows, "route_progress"),
        "final_route_progress": float(rows[-1].get("route_progress", 0.0)) if rows else 0.0,
        "mean_speed_kmh": _mean(rows, "speed_kmh"),
        "mean_abs_steer": _mean_abs(rows, "steer"),
        "mean_target_speed_capped_kmh": _mean(rows, "target_speed_capped_kmh"),
        "waypoint_valid_rate": _rate(rows, "waypoint_valid"),
        "path_valid_rate": _rate(rows, "path_valid"),
        "used_route_fallback_rate": _rate(rows, "used_route_fallback"),
        "used_synthetic_fallback_rate": _rate(rows, "used_synthetic_fallback"),
        "obs_finite_rate": _rate(rows, "obs_finite"),
    }


def _first_done(rows: list[dict[str, Any]]) -> tuple[int, str]:
    for row in rows:
        if bool(row.get("done", False)):
            return int(row.get("step", -1)), str(row.get("done_reason", ""))
    return -1, ""


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(_safe_float(row.get(key, 0.0)) for row in rows) / len(rows))


def _mean_abs(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(abs(_safe_float(row.get(key, 0.0))) for row in rows) / len(rows))


def _max(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(_safe_float(row.get(key, 0.0)) for row in rows))


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(bool(row.get(key, False)) for row in rows) / len(rows))


if __name__ == "__main__":
    main()
