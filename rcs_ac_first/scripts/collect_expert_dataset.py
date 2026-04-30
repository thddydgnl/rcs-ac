from __future__ import annotations

import argparse
import copy
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

from rcs_ac_first.data.expert_dataset import save_expert_npz
from rcs_ac_first.runtime.auxiliary import build_auxiliary_observation_spec, flatten_auxiliary_observation_dict
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec, flatten_observation_dict
from rcs_ac_first.scripts.train import _build_env, _project_path, _resolve_config_path, _set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CARLA autopilot expert dataset for RCS-AC-first BC.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-transitions", type=int, required=True)
    parser.add_argument("--run-name", default="expert_clean")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--weather", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--skip-bad-transitions", action="store_true")
    return parser.parse_args()


def _force_clean_collection_config(config: Any, args: argparse.Namespace) -> None:
    config.raw = copy.deepcopy(config.raw)
    env_raw = dict(config.raw.get("env") or {})
    env_raw["traffic_mode"] = "none"
    env_raw["corruption"] = {"enabled": False, "mode": "none", "level": 0.0}
    if args.weather is not None:
        env_raw["weather"] = str(args.weather)
    config.raw["env"] = env_raw
    if args.max_episode_steps is not None:
        config.train.max_episode_steps = int(args.max_episode_steps)
    if args.device is not None:
        config.train.device = str(args.device)


def _is_bad_transition(info: dict[str, Any], action: np.ndarray) -> bool:
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        return True
    if not np.all((action >= -1.0) & (action <= 1.0)):
        return True
    if bool(info.get("collision", False)):
        return True
    if bool(info.get("stuck", False)):
        return True
    lane_offset = abs(float(info.get("lane_offset", 0.0)))
    heading_error = abs(float(info.get("heading_error", 0.0)))
    return bool(lane_offset > 3.5 or heading_error > 80.0)


def _append_optional(target: dict[str, list[Any]], key: str, value: Any) -> None:
    target.setdefault(key, []).append(value)


def main() -> int:
    args = parse_args()
    if int(args.num_transitions) <= 0:
        raise ValueError("--num-transitions must be > 0")

    config = load_phase1_config(_resolve_config_path(str(args.config)))
    _force_clean_collection_config(config, args)
    _set_seed(config.train.seed)

    host = args.host or config.train.host
    port = int(args.port if args.port is not None else config.train.port)
    run_name = str(args.run_name)
    output_path = _project_path(args.output)
    output_dir = _project_path(config.train.output_root) / run_name

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

    arrays: dict[str, list[Any]] = {
        "obs_flat": [],
        "auxiliary": [],
        "expert_action": [],
        "expert_control": [],
        "reward": [],
        "done": [],
        "route_progress": [],
        "episode_id": [],
        "step_in_episode": [],
        "bad_transition_mask": [],
        "done_reason": [],
        "corruption_level": [],
        "drop_rgb": [],
        "drop_lidar": [],
        "drop_ego": [],
        "drop_route": [],
        "aux_road_mask_target": [],
        "aux_lane_edge_target": [],
        "aux_mask_targets_available": [],
    }
    accepted = 0
    attempted = 0
    episode_id = 0
    env = None
    try:
        env = _build_env(config, host=host, port=port, output_dir=output_dir, run_name=run_name)
        env.reset()
        env.enable_ego_autopilot()
        while accepted < int(args.num_transitions):
            step_result = env.step_expert_autopilot()
            attempted += 1
            info = dict(step_result.get("info", {}))
            obs_flat = flatten_observation_dict(step_result["obs"], obs_spec)
            auxiliary = flatten_auxiliary_observation_dict(step_result["obs"], aux_spec)
            expert_action = np.asarray(step_result["expert_action"], dtype=np.float32)
            expert_control = np.asarray(step_result["expert_control"], dtype=np.float32)
            bad_transition = _is_bad_transition(info, expert_action)
            if not args.skip_bad_transitions or not bad_transition:
                arrays["obs_flat"].append(obs_flat)
                arrays["auxiliary"].append(auxiliary)
                arrays["expert_action"].append(np.clip(expert_action, -1.0, 1.0))
                arrays["expert_control"].append(expert_control)
                arrays["reward"].append(float(step_result["reward"]))
                arrays["done"].append(bool(step_result["done"]))
                arrays["route_progress"].append(float(info.get("route_progress", 0.0)))
                arrays["episode_id"].append(int(episode_id))
                arrays["step_in_episode"].append(int(info.get("current_step", env.step_count)))
                arrays["bad_transition_mask"].append(bool(bad_transition))
                arrays["done_reason"].append(str(info.get("done_reason", "running")))
                arrays["corruption_level"].append(float(info.get("corruption_level", 0.0)))
                for key in ("rgb", "lidar", "ego", "route"):
                    arrays[f"drop_{key}"].append(float(info.get(f"drop_{key}", 0.0)))
                mask_available = bool(info.get("aux_mask_targets_available", False))
                road_mask = np.asarray(
                    info.get(
                        "aux_road_mask_target",
                        np.zeros((1, config.model.aux_mask_height, config.model.aux_mask_width)),
                    ),
                    dtype=np.float32,
                )
                lane_mask = np.asarray(
                    info.get(
                        "aux_lane_edge_target",
                        np.zeros((1, config.model.aux_mask_height, config.model.aux_mask_width)),
                    ),
                    dtype=np.float32,
                )
                expected_mask_shape = (1, int(config.model.aux_mask_height), int(config.model.aux_mask_width))
                if road_mask.shape != expected_mask_shape:
                    road_mask = np.zeros(expected_mask_shape, dtype=np.float32)
                    mask_available = False
                if lane_mask.shape != expected_mask_shape:
                    lane_mask = np.zeros(expected_mask_shape, dtype=np.float32)
                    mask_available = False
                arrays["aux_road_mask_target"].append(road_mask.astype(np.float32, copy=False))
                arrays["aux_lane_edge_target"].append(lane_mask.astype(np.float32, copy=False))
                arrays["aux_mask_targets_available"].append(float(mask_available))
                accepted += 1
            if bool(step_result["done"]):
                env.disable_ego_autopilot()
                env.reset()
                env.enable_ego_autopilot()
                episode_id += 1
            if attempted % 100 == 0:
                print(f"[collect] accepted={accepted} attempted={attempted} episodes={episode_id}")
    finally:
        if env is not None:
            try:
                env.disable_ego_autopilot()
            finally:
                env.close()

    stacked: dict[str, np.ndarray] = {
        "obs_flat": np.stack(arrays["obs_flat"]).astype(np.float32, copy=False),
        "auxiliary": np.stack(arrays["auxiliary"]).astype(np.float32, copy=False),
        "expert_action": np.stack(arrays["expert_action"]).astype(np.float32, copy=False),
        "expert_control": np.stack(arrays["expert_control"]).astype(np.float32, copy=False),
        "reward": np.asarray(arrays["reward"], dtype=np.float32),
        "done": np.asarray(arrays["done"], dtype=np.bool_),
        "route_progress": np.asarray(arrays["route_progress"], dtype=np.float32),
        "episode_id": np.asarray(arrays["episode_id"], dtype=np.int32),
        "step_in_episode": np.asarray(arrays["step_in_episode"], dtype=np.int32),
        "bad_transition_mask": np.asarray(arrays["bad_transition_mask"], dtype=np.bool_),
        "done_reason": np.asarray(arrays["done_reason"], dtype="<U64"),
        "corruption_level": np.asarray(arrays["corruption_level"], dtype=np.float32),
        "drop_rgb": np.asarray(arrays["drop_rgb"], dtype=np.float32),
        "drop_lidar": np.asarray(arrays["drop_lidar"], dtype=np.float32),
        "drop_ego": np.asarray(arrays["drop_ego"], dtype=np.float32),
        "drop_route": np.asarray(arrays["drop_route"], dtype=np.float32),
        "aux_road_mask_target": np.stack(arrays["aux_road_mask_target"]).astype(np.float32, copy=False),
        "aux_lane_edge_target": np.stack(arrays["aux_lane_edge_target"]).astype(np.float32, copy=False),
        "aux_mask_targets_available": np.asarray(arrays["aux_mask_targets_available"], dtype=np.float32),
    }
    summary = save_expert_npz(output_path, stacked)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary.to_dict() | {"attempted_transitions": int(attempted)}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] dataset={output_path} summary={summary_path} {summary.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
