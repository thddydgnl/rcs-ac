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

from rcs_ac_v1_1.adapters.carla_observation_adapter import build_v11_state_from_carla_step
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config
from rcs_ac_v1_1.runtime.gt_bev_targets import build_gt_bev_target
from rcs_ac_v1_1.runtime.pipeline import build_policy_components
from rcs_ac_v1_1.runtime.waypoint_targets import build_distance_based_expert_waypoints, build_expert_target_speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a tiny CARLA BEV expert dataset for RCS-AC v1.1 smoke.")
    parser.add_argument("--config", required=True, help="v1.1 runtime config")
    parser.add_argument("--env-config", required=True, help="rcs_ac_first CARLA env config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--run-name", default="v1_1_bev_expert_tiny")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_steps) <= 0:
        raise ValueError("--num-steps must be > 0")
    runtime_config = load_v11_runtime_config(args.config)
    env_config, env = _build_carla_env(args)
    del env_config
    arrays: dict[str, list[Any]] = {
        "rgb": [],
        "lidar_bev": [],
        "route_bev": [],
        "ego_bev": [],
        "ego_scalar": [],
        "route_scalar": [],
        "sensor_health": [],
        "gt_bev_target": [],
        "expert_waypoints": [],
        "expert_target_speed": [],
        "bev_target_available": [],
        "waypoint_target_available": [],
        "episode_id": [],
        "step": [],
        "done": [],
        "route_progress": [],
        "done_reason": [],
    }
    previous_control: dict[str, float] | None = None
    accepted = 0
    episode_id = 0
    try:
        reset = env.reset()
        current_info = dict(reset.get("info", {}))
        _enable_expert_control(env)
        for step in range(int(args.num_steps)):
            result = _step_expert(env)
            obs = dict(result.get("obs", {}))
            info = dict(result.get("info", current_info))
            packet, ego_state, route_context, _ = build_v11_state_from_carla_step(
                obs=obs,
                info=info,
                runtime_config=runtime_config,
                previous_control=previous_control,
                ego_vehicle=_get_ego_vehicle(env),
            )
            components = build_policy_components(
                runtime_config=runtime_config,
                sensor_packet=packet,
                ego_state=ego_state,
                route_context=route_context,
            )
            gt_bev, bev_available = build_gt_bev_target(
                spec=runtime_config.bev_spec,
                map_adapter=None,
                actors_ego=None,
                route_context=route_context,
            )
            route_points = None if route_context is None else route_context.route_points_ego
            expert_waypoints, waypoint_available = build_distance_based_expert_waypoints(
                route_points,
                fallback_route_ego=route_points,
            )
            expert_speed = build_expert_target_speed(
                expert_speed_kmh=_safe_optional_float(info.get("target_speed_kmh")),
                route_target_speed_kmh=None if route_context is None else route_context.target_speed_kmh,
            )
            for key in ("rgb", "lidar_bev", "route_bev", "ego_bev", "ego_scalar", "route_scalar", "sensor_health"):
                arrays[key].append(components[key])
            arrays["gt_bev_target"].append(gt_bev)
            arrays["expert_waypoints"].append(expert_waypoints)
            arrays["expert_target_speed"].append(expert_speed)
            arrays["bev_target_available"].append(float(bev_available))
            arrays["waypoint_target_available"].append(float(waypoint_available))
            arrays["episode_id"].append(int(episode_id))
            arrays["step"].append(int(step))
            arrays["done"].append(bool(result.get("done", False)))
            arrays["route_progress"].append(float(info.get("route_progress", 0.0)))
            arrays["done_reason"].append(str(info.get("done_reason", "running")))
            previous_control = _previous_control_from_info(info)
            accepted += 1
            current_info = info
            if bool(result.get("done", False)):
                episode_id += 1
                reset = env.reset()
                current_info = dict(reset.get("info", {}))
                previous_control = None
        if accepted <= 0:
            raise RuntimeError("No v1.1 BEV expert samples collected")
    finally:
        _disable_expert_control(env)
        if hasattr(env, "close"):
            env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stacked = _stack_arrays(arrays)
    np.savez_compressed(output, **stacked)
    summary = {key: list(value.shape) for key, value in stacked.items() if hasattr(value, "shape")}
    summary["num_samples"] = int(accepted)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_carla_env(args: argparse.Namespace) -> tuple[Any, Any]:
    from rcs_ac_first.runtime.config import load_phase1_config
    from rcs_ac_first.scripts.train import _build_env, _resolve_config_path

    config = load_phase1_config(_resolve_config_path(str(args.env_config)))
    output_dir = Path("logs") / str(args.run_name)
    env = _build_env(config, host=str(args.host), port=int(args.port), output_dir=output_dir, run_name=str(args.run_name))
    return config, env


def _step_expert(env: Any) -> dict[str, Any]:
    if hasattr(env, "step_expert_autopilot"):
        result = env.step_expert_autopilot()
        return _normalize_expert_step_result(result)
    if hasattr(env, "_compute_heuristic_action"):
        action = np.asarray(env._compute_heuristic_action(), dtype=np.float32)
        result = env.step(action)
        result["expert_action"] = action
        return _normalize_expert_step_result(result)
    raise AttributeError("CarlaEnv does not expose step_expert_autopilot() or _compute_heuristic_action()")


def _normalize_expert_step_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("expert step must return a dict")
    if "obs" not in result:
        raise KeyError("expert step result missing 'obs'")
    info = dict(result.get("info", {}))
    action = np.asarray(result.get("expert_action", result.get("action", [0.0, 0.0])), dtype=np.float32).reshape(-1)
    if action.shape[0] == 3:
        action = np.asarray([action[0], action[1] - action[2]], dtype=np.float32)
    elif action.shape[0] != 2:
        action = np.zeros((2,), dtype=np.float32)
    result["expert_action"] = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
    if "expert_control" not in result:
        previous = _previous_control_from_info(info)
        result["expert_control"] = np.asarray(
            [previous["steer"], previous["throttle"], previous["brake"]],
            dtype=np.float32,
        )
    return result


def _enable_expert_control(env: Any) -> None:
    if hasattr(env, "enable_ego_autopilot"):
        env.enable_ego_autopilot()


def _disable_expert_control(env: Any) -> None:
    if hasattr(env, "disable_ego_autopilot"):
        env.disable_ego_autopilot()


def _get_ego_vehicle(env: Any) -> Any:
    for name in ("ego_vehicle", "vehicle", "_ego_vehicle", "_vehicle"):
        if hasattr(env, name):
            return getattr(env, name)
    return None


def _safe_optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _previous_control_from_info(info: dict[str, Any]) -> dict[str, float]:
    control = info.get("control")
    if not isinstance(control, dict):
        control = {}
    steer = _safe_float(control.get("steer"), default=_safe_float(info.get("applied_steer"), default=_safe_float(info.get("smoothed_steer"), default=_safe_float(info.get("steer"), default=0.0))))
    throttle = _safe_float(control.get("throttle"), default=_safe_float(info.get("applied_throttle"), default=_safe_float(info.get("throttle"), default=0.0)))
    brake = _safe_float(control.get("brake"), default=_safe_float(info.get("applied_brake"), default=_safe_float(info.get("brake"), default=0.0)))
    longitudinal = _safe_float(
        info.get("applied_longitudinal"),
        default=_safe_float(info.get("longitudinal"), default=throttle - brake),
    )
    return {"steer": steer, "throttle": throttle, "brake": brake, "longitudinal": longitudinal}


def _safe_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _stack_arrays(arrays: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        if key == "done_reason":
            stacked[key] = np.asarray(values, dtype="U64")
        elif key in {"episode_id", "step"}:
            stacked[key] = np.asarray(values, dtype=np.int32)
        elif key == "done":
            stacked[key] = np.asarray(values, dtype=np.bool_)
        elif key in {"bev_target_available", "waypoint_target_available", "route_progress"}:
            stacked[key] = np.asarray(values, dtype=np.float32)
        else:
            stacked[key] = np.stack(values).astype(np.float32, copy=False)
    return stacked


if __name__ == "__main__":
    raise SystemExit(main())
