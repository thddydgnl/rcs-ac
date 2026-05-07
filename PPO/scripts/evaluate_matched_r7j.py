from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "RCS-AC-v1.1" / "code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from PPO.ppo_r7j.models import PPOMatchedR7JPolicy, PPOMatchedR7JPolicyConfig
from rcs_ac_v1_1.adapters.carla_observation_adapter import build_v11_flat_obs_from_carla_step
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config
from rcs_ac_v1_1.runtime.observation import flatten_observation_dict
from rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit import (
    REQUIRED_LOG_COLUMNS,
    _build_carla_env,
    _build_log_row,
    _build_summary,
    _control_to_env_action,
    _get_ego_vehicle,
    _resolve_device,
    _sensor_health_from_obs,
    _step_env,
)


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic":
        result = synthetic_eval_step(args.config, args.checkpoint if args.checkpoint else None, args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    summary = run_carla_eval(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO matched-r7j pure-pursuit evaluation.")
    parser.add_argument("--config", required=True, help="RCS-AC v1.1 r7j runtime config.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--env-config", default="")
    parser.add_argument("--run-name", default="ppo_matched_r7j_eval")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["synthetic", "carla"], default="synthetic")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--fixed-delta-seconds", type=float, default=None)
    parser.add_argument("--action-format", choices=["direct_3d", "stage4_v2"], default="direct_3d")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--allow-random-policy-smoke", action="store_true")
    parser.add_argument("--log-std-init", type=float, default=-1.5)
    return parser.parse_args()


def synthetic_eval_step(config_path: str | Path, checkpoint_path: str | Path | None = None, device: str = "cpu") -> dict[str, Any]:
    runtime_config = load_v11_runtime_config(config_path)
    torch_device = _resolve_device(device)
    model = PPOMatchedR7JPolicy(
        runtime_config,
        PPOMatchedR7JPolicyConfig(
            feature_dim=int(runtime_config.policy_config.feature_dim),
            log_std_init=-1.5,
        ),
    ).to(torch_device)
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, torch_device)
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
    flat = torch.as_tensor(flatten_observation_dict(obs, runtime_config.observation_spec), dtype=torch.float32, device=torch_device).unsqueeze(0)
    with torch.no_grad():
        outputs = model.act(flat, deterministic=True)
    controller = runtime_config.build_controller()
    control = controller.control(
        outputs["waypoints"][0].detach().cpu().numpy(),
        target_speed_kmh=float(outputs["target_speed_kmh"][0, 0].detach().cpu()),
        current_speed_kmh=0.0,
        dt=0.05,
    )
    return {
        "action_contract": "waypoints_plus_target_speed_to_shared_pure_pursuit_direct_3d",
        "waypoints_shape": list(outputs["waypoints"].shape),
        "target_speed_kmh": float(outputs["target_speed_kmh"][0, 0].detach().cpu()),
        "steer": float(control["steer"]),
        "throttle": float(control["throttle"]),
        "brake": float(control["brake"]),
        "controller_debug_keys": sorted(str(key) for key in control["debug"].keys()),
    }


def run_carla_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not args.env_config:
        raise ValueError("--mode carla requires --env-config")
    runtime_config = load_v11_runtime_config(args.config)
    device = _resolve_device(args.device)
    model = PPOMatchedR7JPolicy(
        runtime_config,
        PPOMatchedR7JPolicyConfig(
            feature_dim=int(runtime_config.policy_config.feature_dim),
            log_std_init=float(args.log_std_init),
        ),
    ).to(device)
    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint, device)
    elif not args.allow_random_policy_smoke:
        raise ValueError("--mode carla requires --checkpoint unless --allow-random-policy-smoke is set")
    model.eval()

    _, env = _build_carla_env(args)
    output_dir = Path(args.output_dir or Path("PPO") / "logs" / args.run_name)
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
                obs_flat, _, ego_state, route_context, adapter_debug = build_v11_flat_obs_from_carla_step(
                    obs=current_obs,
                    info=current_info,
                    runtime_config=runtime_config,
                    previous_control=previous_control,
                    ego_vehicle=_get_ego_vehicle(env),
                )
                obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).reshape(1, -1)
                if not torch.isfinite(obs_t).all():
                    raise RuntimeError("PPO matched-r7j CARLA observation contains NaN or inf")
                with torch.no_grad():
                    outputs = model.act(obs_t, deterministic=True)
                _validate_outputs(outputs, runtime_config.policy_config.waypoint_count)
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
                result = _step_env(env, _control_to_env_action(control, args.action_format))
                info = dict(result.get("info", {}))
                reward = float(result.get("reward", info.get("reward", 0.0)))
                done = bool(result.get("done", False))
                row = _build_log_row(
                    step=step,
                    episode_step=episode_step,
                    reward=reward,
                    done=done,
                    info=info,
                    ego_state=ego_state,
                    control=control,
                    target_speed_raw=target_speed_raw,
                    waypoints=waypoints,
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
        summary["baseline_family"] = "PPO matched-r7j"
        summary["controller_contract"] = "shared_r7j_pure_pursuit_direct_3d"
        summary["uses_rcs_reliability_fusion_shapley_risk"] = False
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if hasattr(env, "close"):
            env.close()
    return summary


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)


def _validate_outputs(outputs: dict[str, torch.Tensor], waypoint_count: int) -> None:
    for key in ("waypoints", "target_speed_kmh", "bev_fusion_weights"):
        if key not in outputs:
            raise RuntimeError(f"PPO matched-r7j output missing {key!r}")
        if not torch.isfinite(outputs[key]).all():
            raise RuntimeError(f"PPO matched-r7j output {key!r} contains NaN or inf")
    if tuple(outputs["waypoints"].shape[1:]) != (int(waypoint_count), 2):
        raise RuntimeError(f"unexpected waypoints shape: {tuple(outputs['waypoints'].shape)}")


if __name__ == "__main__":
    main()
