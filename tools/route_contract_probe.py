from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path.cwd()
for path in (ROOT, ROOT.parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec, flatten_observation_dict
from rcs_ac_first.scripts.train import _build_env


INFO_KEYS = (
    "route_progress",
    "target_waypoint_index",
    "lane_offset",
    "heading_error",
    "route_target_x",
    "target_x",
    "route_target_local_x",
    "target_x_local",
    "route_target_y",
    "target_y",
    "route_target_local_y",
    "target_y_local",
    "route_target_heading",
    "route_heading",
    "route_heading_error",
    "route_heading_mismatch",
    "lookahead_turn_delta_rad",
    "route_waypoint_count",
    "route_total_length_m",
    "route_representation",
    "maneuver_command",
    "maneuver_turn_sign",
    "offroad",
    "done",
    "done_reason",
)


def _info_subset(info: dict[str, Any]) -> dict[str, Any]:
    return {key: info.get(key, "<MISSING>") for key in INFO_KEYS}


def _obs_summary(obs: dict[str, Any], obs_spec: Any) -> dict[str, Any]:
    flat = flatten_observation_dict(obs, obs_spec)
    route = np.asarray(obs["route"], dtype=np.float32).reshape(-1)
    ego = np.asarray(obs["ego"], dtype=np.float32).reshape(-1)
    return {
        "obs_keys": sorted(list(obs.keys())),
        "flat_dim": int(flat.shape[0]),
        "route_shape": list(np.asarray(obs["route"]).shape),
        "route": [float(value) for value in route.tolist()],
        "route_abs_sum": float(np.abs(route).sum()),
        "route_nonzero_count": int(np.count_nonzero(np.abs(route) > 1e-8)),
        "ego": [float(value) for value in ego.tolist()],
    }


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml"
    )
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/route_contract_probe_round90")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_phase1_config(config_path)
    obs_spec = build_policy_observation_spec(
        rgb_shape=config.model.rgb_shape,
        lidar_shape=config.model.lidar_shape,
        ego_dim=config.model.ego_dim,
        route_dim=config.model.route_dim,
    )
    env = _build_env(
        config,
        host="127.0.0.1",
        port=2000,
        output_dir=output_dir,
        run_name="route_contract_probe_round90",
    )

    result: dict[str, Any] = {
        "config_path": str(config_path),
        "config_route_representation": config.raw.get("env", {}).get("route_representation"),
        "model_route_dim": int(config.model.route_dim),
    }
    try:
        reset = env.reset()
        result["reset_info"] = _info_subset(dict(reset["info"]))
        result["reset_obs"] = _obs_summary(reset["obs"], obs_spec)
        if env.route_manager is not None and env.vehicle is not None:
            result["reset_route_debug_snapshot"] = env.route_manager.get_debug_snapshot(
                env.vehicle.get_transform()
            )
            result["reset_route_metadata"] = env.route_manager.get_route_metadata()

        step = env.step(np.asarray([0.0, 0.25, 0.0], dtype=np.float32))
        result["step_info"] = _info_subset(dict(step["info"]))
        result["step_obs"] = _obs_summary(step["obs"], obs_spec)
        if env.route_manager is not None and env.vehicle is not None:
            result["step_route_debug_snapshot"] = env.route_manager.get_debug_snapshot(
                env.vehicle.get_transform()
            )
            result["step_route_metadata"] = env.route_manager.get_route_metadata()
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - diagnostic best effort
            result["close_error"] = repr(exc)

    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
