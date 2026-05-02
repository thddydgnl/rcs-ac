from __future__ import annotations

import argparse
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
import torch

from rcs_ac_first.data.expert_dataset import save_expert_npz
from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.runtime.auxiliary import build_auxiliary_observation_spec, flatten_auxiliary_observation_dict
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec, flatten_observation_dict
from rcs_ac_first.scripts.train import _build_env, _project_path, _resolve_config_path, _set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect targeted branch-turn recovery imitation dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-transitions", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--run-name", default="recovery_dataset")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--route-progress-min", type=float, default=0.24)
    parser.add_argument("--route-progress-max", type=float, default=0.38)
    parser.add_argument("--lane-offset-threshold", type=float, default=0.55)
    parser.add_argument("--heading-error-threshold", type=float, default=8.0)
    parser.add_argument("--speed-kmh-min", type=float, default=None)
    parser.add_argument("--speed-kmh-max", type=float, default=None)
    parser.add_argument("--lane-offset-sign", choices=["any", "left", "right"], default="any")
    parser.add_argument("--require-corrective-sign-aligned", action="store_true")
    parser.add_argument("--accept-branch-only", action="store_true")
    parser.add_argument("--accept-lateral-branch-only", action="store_true")
    parser.add_argument("--require-turn-boost", action="store_true")
    parser.add_argument("--require-speed-gate-active", action="store_true")
    return parser.parse_args()


def _device(config: Any, override: str | None) -> torch.device:
    requested = override or config.train.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(str(requested))


def _load_model(config: Any, checkpoint_path: Path, device: torch.device) -> RCSACFirstModel:
    model = RCSACFirstModel(config.model, config.fusion).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _should_accept(info: dict[str, Any], args: argparse.Namespace) -> bool:
    if bool(info.get("collision", False)) or bool(info.get("stuck", False)) or bool(info.get("offroad", False)):
        return False
    progress = float(info.get("route_progress", 0.0))
    if progress < float(args.route_progress_min) or progress > float(args.route_progress_max):
        return False
    if bool(args.accept_branch_only) and not bool(info.get("junction_branch_choice_active", False)):
        return False
    if bool(args.accept_lateral_branch_only) and not bool(info.get("lateral_recovery_branch_active", False)):
        return False
    if bool(args.require_turn_boost) and not bool(info.get("lateral_recovery_turn_boost_active", False)):
        return False
    if bool(args.require_speed_gate_active) and not bool(info.get("lateral_recovery_speed_gate_active", False)):
        return False
    speed_kmh = float(info.get("speed_kmh", 0.0))
    if args.speed_kmh_min is not None and speed_kmh < float(args.speed_kmh_min):
        return False
    if args.speed_kmh_max is not None and speed_kmh > float(args.speed_kmh_max):
        return False
    signed_lane_offset = float(info.get("lane_offset", 0.0))
    if str(args.lane_offset_sign) == "left" and signed_lane_offset >= 0.0:
        return False
    if str(args.lane_offset_sign) == "right" and signed_lane_offset <= 0.0:
        return False
    lane_offset = abs(signed_lane_offset)
    heading_error = abs(float(info.get("heading_error", 0.0)))
    recovery_active = bool(info.get("lateral_recovery_active", False))
    turn_boost_active = bool(info.get("lateral_recovery_turn_boost_active", False))
    return bool(
        recovery_active
        or turn_boost_active
        or lane_offset >= float(args.lane_offset_threshold)
        or heading_error >= float(args.heading_error_threshold)
    )


def _corrective_sign_aligned(info: dict[str, Any], raw_action: np.ndarray, corrected_steer: float) -> bool:
    lane_offset = float(info.get("lane_offset", 0.0))
    if abs(lane_offset) <= 1e-6:
        return False
    policy_action = dict(info.get("policy_action", {}))
    policy_steer = float(policy_action.get("steer", raw_action[0]))
    target_delta = float(corrected_steer) - policy_steer
    if abs(target_delta) <= 1e-6:
        return False
    return bool(target_delta * (-lane_offset) > 0.0)


def _append(arrays: dict[str, list[Any]], key: str, value: Any) -> None:
    arrays.setdefault(key, []).append(value)


def main() -> int:
    args = parse_args()
    if int(args.num_transitions) <= 0:
        raise ValueError("--num-transitions must be > 0")

    config = load_phase1_config(_resolve_config_path(str(args.config)))
    _set_seed(config.train.seed)
    device = _device(config, args.device)
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
    model = _load_model(config, _project_path(args.checkpoint), device)

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
        "lane_offset": [],
        "heading_error": [],
        "speed_kmh": [],
        "curve_proxy": [],
        "junction_branch_choice_active": [],
        "lateral_recovery_branch_active": [],
        "lateral_recovery_active": [],
        "lateral_recovery_correction": [],
        "lateral_recovery_turn_boost_active": [],
        "lateral_recovery_speed_gate_active": [],
        "lateral_recovery_raw_steer": [],
        "policy_action_steer": [],
        "policy_action_longitudinal": [],
        "target_corrected_steer": [],
        "target_corrective_delta": [],
        "target_corrective_delta_available": [],
        "source_dataset": [],
        "source_dataset_index": [],
        "source_label": [],
        "source_episode_id": [],
        "source_step_in_episode": [],
        "source_row_index": [],
        "aux_road_mask_target": [],
        "aux_lane_edge_target": [],
        "aux_mask_targets_available": [],
    }

    accepted = 0
    attempted = 0
    episode_id = 0
    hidden_state: torch.Tensor | None = None
    env = None
    try:
        env = _build_env(config, host=host, port=port, output_dir=output_dir, run_name=run_name)
        reset_result = env.reset()
        observation = flatten_observation_dict(reset_result["obs"], obs_spec)
        auxiliary = flatten_auxiliary_observation_dict(reset_result["obs"], aux_spec)
        while accepted < int(args.num_transitions) and attempted < int(args.max_steps):
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            aux_tensor = torch.as_tensor(auxiliary, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                outputs = model.forward_policy_outputs(
                    obs_tensor,
                    auxiliary=aux_tensor,
                    hidden_state=hidden_state,
                    track_diagnostics=True,
                )
            action_mean = outputs["action_mean"]
            if not isinstance(action_mean, torch.Tensor):
                raise RuntimeError("Model output action_mean is missing.")
            action = action_mean.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            hidden = outputs["next_hidden_state"]
            hidden_state = hidden if isinstance(hidden, torch.Tensor) else None

            step_result = env.step(action, policy_diagnostics=model.get_diagnostics())
            attempted += 1
            info = dict(step_result.get("info", {}))
            if _should_accept(info, args):
                policy_action = dict(info.get("policy_action", {}))
                corrected_steer = float(info.get("lateral_recovery_corrected_steer", policy_action.get("steer", action[0])))
                if not bool(args.require_corrective_sign_aligned) or _corrective_sign_aligned(info, action, corrected_steer):
                    target_action = np.asarray(
                        [
                            np.clip(corrected_steer, -1.0, 1.0),
                            np.clip(float(policy_action.get("longitudinal", action[1])), -1.0, 1.0),
                        ],
                        dtype=np.float32,
                    )
                    control = dict(info.get("control", {}))
                    expert_control = np.asarray(
                        [
                            float(control.get("steer", info.get("steer", 0.0))),
                            float(control.get("throttle", info.get("throttle", 0.0))),
                            float(control.get("brake", info.get("brake", 0.0))),
                        ],
                        dtype=np.float32,
                    )
                    arrays["obs_flat"].append(observation.astype(np.float32, copy=False))
                    arrays["auxiliary"].append(auxiliary.astype(np.float32, copy=False))
                    arrays["expert_action"].append(target_action)
                    arrays["expert_control"].append(expert_control)
                    arrays["reward"].append(float(step_result.get("reward", 0.0)))
                    arrays["done"].append(bool(step_result.get("done", False)))
                    arrays["route_progress"].append(float(info.get("route_progress", 0.0)))
                    arrays["episode_id"].append(int(episode_id))
                    arrays["step_in_episode"].append(int(info.get("step", attempted)))
                    arrays["bad_transition_mask"].append(False)
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
                    if road_mask.shape != (1, int(config.model.aux_mask_height), int(config.model.aux_mask_width)):
                        road_mask = np.zeros(
                            (1, int(config.model.aux_mask_height), int(config.model.aux_mask_width)), dtype=np.float32
                        )
                        mask_available = False
                    if lane_mask.shape != (1, int(config.model.aux_mask_height), int(config.model.aux_mask_width)):
                        lane_mask = np.zeros(
                            (1, int(config.model.aux_mask_height), int(config.model.aux_mask_width)), dtype=np.float32
                        )
                        mask_available = False
                    arrays["aux_road_mask_target"].append(road_mask.astype(np.float32, copy=False))
                    arrays["aux_lane_edge_target"].append(lane_mask.astype(np.float32, copy=False))
                    arrays["aux_mask_targets_available"].append(float(mask_available))
                    for key in (
                        "done_reason",
                        "lane_offset",
                        "heading_error",
                        "speed_kmh",
                        "curve_proxy",
                        "junction_branch_choice_active",
                        "lateral_recovery_branch_active",
                        "lateral_recovery_active",
                        "lateral_recovery_correction",
                        "lateral_recovery_turn_boost_active",
                        "lateral_recovery_speed_gate_active",
                        "lateral_recovery_raw_steer",
                        "policy_action_steer",
                        "policy_action_longitudinal",
                        "target_corrected_steer",
                        "target_corrective_delta",
                        "target_corrective_delta_available",
                    ):
                        if key == "done_reason":
                            _append(arrays, key, str(info.get("done_reason", "running")))
                        elif key == "policy_action_steer":
                            _append(arrays, key, float(policy_action.get("steer", action[0])))
                        elif key == "policy_action_longitudinal":
                            _append(arrays, key, float(policy_action.get("longitudinal", action[1])))
                        elif key == "target_corrected_steer":
                            _append(arrays, key, float(target_action[0]))
                        elif key == "target_corrective_delta":
                            _append(arrays, key, float(target_action[0]) - float(policy_action.get("steer", action[0])))
                        elif key == "target_corrective_delta_available":
                            _append(arrays, key, 1.0)
                        elif key in {
                            "junction_branch_choice_active",
                            "lateral_recovery_branch_active",
                            "lateral_recovery_active",
                            "lateral_recovery_turn_boost_active",
                            "lateral_recovery_speed_gate_active",
                        }:
                            _append(arrays, key, bool(info.get(key, False)))
                        else:
                            _append(arrays, key, float(info.get(key, 0.0)))
                    arrays["source_dataset"].append(str(output_path))
                    arrays["source_dataset_index"].append(0)
                    arrays["source_label"].append(str(args.run_name))
                    arrays["source_episode_id"].append(int(episode_id))
                    arrays["source_step_in_episode"].append(int(info.get("step", attempted)))
                    arrays["source_row_index"].append(int(accepted))
                    accepted += 1

            if bool(step_result.get("done", False)):
                reset_result = env.reset()
                observation = flatten_observation_dict(reset_result["obs"], obs_spec)
                auxiliary = flatten_auxiliary_observation_dict(reset_result["obs"], aux_spec)
                hidden_state = None
                episode_id += 1
            else:
                observation = flatten_observation_dict(step_result["obs"], obs_spec)
                auxiliary = flatten_auxiliary_observation_dict(step_result["obs"], aux_spec)

            if attempted % 250 == 0:
                print(f"[collect-recovery] accepted={accepted} attempted={attempted} episodes={episode_id}")
    finally:
        if env is not None:
            env.close()

    if accepted <= 0:
        raise RuntimeError("No recovery transitions collected.")

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
        "lane_offset": np.asarray(arrays["lane_offset"], dtype=np.float32),
        "heading_error": np.asarray(arrays["heading_error"], dtype=np.float32),
        "speed_kmh": np.asarray(arrays["speed_kmh"], dtype=np.float32),
        "curve_proxy": np.asarray(arrays["curve_proxy"], dtype=np.float32),
        "junction_branch_choice_active": np.asarray(arrays["junction_branch_choice_active"], dtype=np.bool_),
        "lateral_recovery_branch_active": np.asarray(arrays["lateral_recovery_branch_active"], dtype=np.bool_),
        "lateral_recovery_active": np.asarray(arrays["lateral_recovery_active"], dtype=np.bool_),
        "lateral_recovery_correction": np.asarray(arrays["lateral_recovery_correction"], dtype=np.float32),
        "lateral_recovery_turn_boost_active": np.asarray(arrays["lateral_recovery_turn_boost_active"], dtype=np.bool_),
        "lateral_recovery_speed_gate_active": np.asarray(arrays["lateral_recovery_speed_gate_active"], dtype=np.bool_),
        "lateral_recovery_raw_steer": np.asarray(arrays["lateral_recovery_raw_steer"], dtype=np.float32),
        "policy_action_steer": np.asarray(arrays["policy_action_steer"], dtype=np.float32),
        "policy_action_longitudinal": np.asarray(arrays["policy_action_longitudinal"], dtype=np.float32),
        "target_corrected_steer": np.asarray(arrays["target_corrected_steer"], dtype=np.float32),
        "target_corrective_delta": np.asarray(arrays["target_corrective_delta"], dtype=np.float32),
        "target_corrective_delta_available": np.asarray(arrays["target_corrective_delta_available"], dtype=np.float32),
        "source_dataset": np.asarray(arrays["source_dataset"], dtype="<U512"),
        "source_dataset_index": np.asarray(arrays["source_dataset_index"], dtype=np.int32),
        "source_label": np.asarray(arrays["source_label"], dtype="<U128"),
        "source_episode_id": np.asarray(arrays["source_episode_id"], dtype=np.int32),
        "source_step_in_episode": np.asarray(arrays["source_step_in_episode"], dtype=np.int32),
        "source_row_index": np.asarray(arrays["source_row_index"], dtype=np.int32),
        "aux_road_mask_target": np.stack(arrays["aux_road_mask_target"]).astype(np.float32, copy=False),
        "aux_lane_edge_target": np.stack(arrays["aux_lane_edge_target"]).astype(np.float32, copy=False),
        "aux_mask_targets_available": np.asarray(arrays["aux_mask_targets_available"], dtype=np.float32),
    }
    summary = save_expert_npz(output_path, stacked)
    extra = {
        "attempted_transitions": int(attempted),
        "accepted_transitions": int(accepted),
        "acceptance_rate": float(accepted / max(1, attempted)),
        "route_progress_min_filter": float(args.route_progress_min),
        "route_progress_max_filter": float(args.route_progress_max),
        "lane_offset_threshold": float(args.lane_offset_threshold),
        "heading_error_threshold": float(args.heading_error_threshold),
        "speed_kmh_min_filter": None if args.speed_kmh_min is None else float(args.speed_kmh_min),
        "speed_kmh_max_filter": None if args.speed_kmh_max is None else float(args.speed_kmh_max),
        "lane_offset_sign_filter": str(args.lane_offset_sign),
        "require_corrective_sign_aligned": bool(args.require_corrective_sign_aligned),
        "accept_branch_only": bool(args.accept_branch_only),
        "accept_lateral_branch_only": bool(args.accept_lateral_branch_only),
        "require_turn_boost": bool(args.require_turn_boost),
        "require_speed_gate_active": bool(args.require_speed_gate_active),
        "target_corrective_delta_mean": float(np.mean(stacked["target_corrective_delta"])),
        "target_corrective_delta_abs_mean": float(np.mean(np.abs(stacked["target_corrective_delta"]))),
        "target_corrective_delta_sign_rate": float(
            np.mean(stacked["target_corrective_delta"] * (-stacked["lane_offset"]) > 0.0)
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary.to_dict() | extra, indent=2), encoding="utf-8")
    print(f"[done] dataset={output_path} summary={summary_path} {summary.to_dict() | extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
