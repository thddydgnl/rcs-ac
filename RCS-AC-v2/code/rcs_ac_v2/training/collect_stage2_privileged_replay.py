"""Collect Stage 2 privileged replay from Windows CARLA.

The collector uses the same Stage 2 CARLA adapter as clean eval, then saves
small `.npz` replay episodes for real world-model/policy training. It is not a
clean gate and does not write `summary.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rcs_ac_v2.env.discrete_actions import RAW2DRIVE_ACTIONS, get_discrete_action, to_carla_control
from rcs_ac_v2.evaluation.carla_adapter import SharedCoreCarlaEnvAdapter, _build_shared_core_env
from rcs_ac_v2.training.privileged_replay import (
    REPLAY_SOURCE,
    default_replay_metadata,
    replay_file_manifest,
    save_privileged_replay_episode,
)


def select_privileged_replay_action(
    info: dict[str, Any],
    *,
    target_speed_kmh: float = 18.0,
    steer_sign: float = -1.0,
) -> int:
    speed = _finite_float(info.get("speed_kmh", 0.0))
    heading_error = _finite_float(info.get("heading_error", 0.0))
    lane_offset = _finite_float(info.get("lane_offset", 0.0))
    desired_speed = _finite_float(info.get("target_speed_kmh", target_speed_kmh), default=target_speed_kmh)
    target_distance = _finite_float(info.get("target_distance", 25.0), default=25.0)

    desired_steer = float(np.clip(float(steer_sign) * (0.025 * heading_error + 0.18 * lane_offset), -0.75, 0.75))
    if target_distance < 2.0 or speed > desired_speed + 2.0:
        desired_throttle = 0.0
    elif speed < desired_speed - 6.0:
        desired_throttle = 0.7
    else:
        desired_throttle = 0.3
    if abs(desired_steer) > 0.45:
        desired_throttle = min(desired_throttle, 0.3)

    best_id = 0
    best_score = float("inf")
    for action in RAW2DRIVE_ACTIONS:
        if action.reverse:
            continue
        score = (
            3.0 * (float(action.steer) - desired_steer) ** 2
            + (float(action.throttle) - desired_throttle) ** 2
            + 0.5 * float(action.brake) ** 2
        )
        if score < best_score:
            best_score = score
            best_id = int(action.action_id)
    return best_id


def collect_episode(
    env: Any,
    *,
    max_steps: int,
    target_speed_kmh: float = 18.0,
    steer_sign: float = -1.0,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    reset = env.reset()
    obs = dict(reset["obs"])
    info = dict(reset.get("info", {}))
    bev_rows: list[np.ndarray] = []
    ego_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    reward_rows: list[float] = []
    cont_rows: list[float] = []
    done_reason = "max_steps_reached"
    for _step in range(max_steps):
        action_id = select_privileged_replay_action(
            info,
            target_speed_kmh=target_speed_kmh,
            steer_sign=steer_sign,
        )
        bev_rows.append(_stage2_bev_to_numpy(obs["privileged_bev"]))
        ego_rows.append(_stage2_ego_to_numpy(obs["privileged_ego"]))
        action_rows.append(action_id)

        control = to_carla_control(get_discrete_action(action_id))
        step = _step_env(env, control)
        next_info = dict(step.get("info", {}))
        done = bool(step.get("done", False))
        truncated = bool(step.get("truncated", False))
        reward_rows.append(_finite_float(step.get("reward", next_info.get("reward", 0.0))))
        cont_rows.append(0.0 if done and not truncated else 1.0)
        obs = dict(step["obs"])
        info = next_info
        if done:
            done_reason = str(info.get("done_reason", "done"))
            break
        if truncated:
            done_reason = str(info.get("done_reason", "truncated"))
            break

    if not action_rows:
        raise RuntimeError("collector produced an empty episode")
    return {
        "privileged_bev": np.stack(bev_rows).astype(np.float32),
        "privileged_ego": np.stack(ego_rows).astype(np.float32),
        "action": np.asarray(action_rows, dtype=np.int64),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "cont": np.asarray(cont_rows, dtype=np.float32),
        "metadata": {
            "episode_length": len(action_rows),
            "done_reason": done_reason,
            "target_speed_kmh": float(target_speed_kmh),
            "steer_sign": float(steer_sign),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Stage 2 privileged CARLA replay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--target-speed-kmh", type=float, default=18.0)
    parser.add_argument("--steer-sign", type=float, default=-1.0)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_env = _build_shared_core_env(config)
    env = SharedCoreCarlaEnvAdapter(raw_env, config)
    episode_paths: list[Path] = []
    try:
        for episode_index in range(int(args.episodes)):
            episode = collect_episode(
                env,
                max_steps=int(args.max_steps),
                target_speed_kmh=float(args.target_speed_kmh),
                steer_sign=float(args.steer_sign),
            )
            path = output_dir / f"episode_{episode_index:04d}.npz"
            metadata = default_replay_metadata(
                extra={
                    **episode["metadata"],
                    "config": str(config_path),
                    "episode_index": episode_index,
                }
            )
            episode_paths.append(
                save_privileged_replay_episode(
                    path,
                    privileged_bev=episode["privileged_bev"],
                    privileged_ego=episode["privileged_ego"],
                    action=episode["action"],
                    reward=episode["reward"],
                    cont=episode["cont"],
                    metadata=metadata,
                )
            )
    finally:
        env.close()

    manifest = {
        "source": REPLAY_SOURCE,
        "clean_gate_evidence": False,
        "config": str(config_path),
        "episodes": len(episode_paths),
        "files": replay_file_manifest(episode_paths),
    }
    manifest_path = output_dir / "stage2_privileged_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"STAGE2_PRIVILEGED_REPLAY_MANIFEST={manifest_path}")


def _stage2_bev_to_numpy(value: Any) -> np.ndarray:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tuple(tensor.shape[:2]) == (1, 1):
        tensor = tensor[0, 0]
    if tuple(tensor.shape) != (43, 16, 16):
        raise ValueError(f"expected privileged_bev [1,1,43,16,16] or [43,16,16], got {tuple(tensor.shape)}")
    return tensor.numpy()


def _stage2_ego_to_numpy(value: Any) -> np.ndarray:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tuple(tensor.shape[:2]) == (1, 1):
        tensor = tensor[0, 0]
    if tuple(tensor.shape) != (8,):
        raise ValueError(f"expected privileged_ego [1,1,8] or [8], got {tuple(tensor.shape)}")
    return tensor.numpy()


def _step_env(env: Any, control: dict[str, float | bool]) -> dict[str, Any]:
    try:
        return env.step(control, policy_diagnostics={"rcs_ac_v2_replay_collector": True})
    except TypeError:
        return env.step(control)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


if __name__ == "__main__":
    main()
