"""Collect Stage 2 privileged replay from Windows CARLA.

The collector uses the same Stage 2 CARLA adapter as clean eval, then saves
small `.npz` replay episodes for real world-model/policy training. It is not a
clean gate and does not write `summary.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
import yaml

from rcs_ac_v2.env.discrete_actions import RAW2DRIVE_ACTIONS, get_discrete_action, to_carla_control
from rcs_ac_v2.evaluation.carla_adapter import SharedCoreCarlaEnvAdapter, _build_shared_core_env, load_stage2_policy_runner
from rcs_ac_v2.evaluation.privileged_observation import (
    EGO8_SCHEMA_VERSION,
    PRIVILEGED_OBSERVATION_SOURCE,
    SEMANTIC43_CHANNEL_NAMES_SHA256,
    SEMANTIC43_SCHEMA_VERSION,
)
from rcs_ac_v2.training.privileged_replay import (
    default_replay_metadata,
    save_privileged_replay_episode,
    write_privileged_replay_manifest,
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
    desired_speed = _finite_float(target_speed_kmh, default=18.0)
    desired_steer = float(np.clip(float(steer_sign) * (0.025 * heading_error + 0.18 * lane_offset), -0.75, 0.75))
    if speed > desired_speed + 2.0:
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
    record_after_route_progress: float = 0.0,
    min_recorded_steps: int = 1,
    warmup_policy_runner: Any | None = None,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if record_after_route_progress < 0.0:
        raise ValueError("record_after_route_progress must be non-negative")
    if min_recorded_steps <= 0:
        raise ValueError("min_recorded_steps must be positive")
    reset = env.reset()
    obs = dict(reset["obs"])
    info = dict(reset.get("info", {}))
    bev_rows: list[np.ndarray] = []
    ego_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    reward_rows: list[float] = []
    cont_rows: list[float] = []
    done_reason = "max_steps_reached"
    warmup_steps = 0
    max_route_progress = _finite_float(info.get("route_progress", 0.0))
    for _step in range(max_steps):
        _validate_simulator_state_privileged_info(info)
        route_progress = _finite_float(info.get("route_progress", 0.0))
        max_route_progress = max(max_route_progress, route_progress)
        should_record = route_progress >= float(record_after_route_progress)
        if should_record or warmup_policy_runner is None:
            action_id = select_privileged_replay_action(
                info,
                target_speed_kmh=target_speed_kmh,
                steer_sign=steer_sign,
            )
        else:
            action_id = _select_warmup_policy_action(warmup_policy_runner, obs, info)
        if should_record:
            bev_rows.append(_stage2_bev_to_numpy(obs["privileged_bev"]))
            ego_rows.append(_stage2_ego_to_numpy(obs["privileged_ego"]))
            action_rows.append(action_id)
        else:
            warmup_steps += 1

        control = to_carla_control(get_discrete_action(action_id))
        step = _step_env(env, control)
        if not should_record and warmup_policy_runner is not None:
            warmup_policy_runner.observe_action(action_id)
        next_info = dict(step.get("info", {}))
        done = bool(step.get("done", False))
        truncated = bool(step.get("truncated", False))
        if should_record:
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

    if len(action_rows) < int(min_recorded_steps):
        raise RuntimeError(
            "collector produced too few recorded steps "
            f"({len(action_rows)} < {int(min_recorded_steps)}) after route_progress>={float(record_after_route_progress)}; "
            f"max_route_progress={max_route_progress}; done_reason={done_reason}"
        )
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
            "record_after_route_progress": float(record_after_route_progress),
            "max_route_progress": float(max_route_progress),
            "warmup_steps": int(warmup_steps),
            "recorded_steps": int(len(action_rows)),
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
    parser.add_argument("--record-after-route-progress", type=float, default=0.0)
    parser.add_argument("--min-recorded-steps", type=int, default=1)
    parser.add_argument("--warmup-policy-config", default="")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_env = _build_shared_core_env(config)
    env = SharedCoreCarlaEnvAdapter(raw_env, config)
    warmup_policy_runner = None
    if args.warmup_policy_config:
        warmup_policy_config = yaml.safe_load(Path(args.warmup_policy_config).read_text())
        warmup_policy_runner = load_stage2_policy_runner(warmup_policy_config)
    episode_paths: list[Path] = []
    try:
        for episode_index in range(int(args.episodes)):
            if warmup_policy_runner is not None:
                warmup_policy_runner.reset()
            episode = collect_episode(
                env,
                max_steps=int(args.max_steps),
                target_speed_kmh=float(args.target_speed_kmh),
                steer_sign=float(args.steer_sign),
                record_after_route_progress=float(args.record_after_route_progress),
                min_recorded_steps=int(args.min_recorded_steps),
                warmup_policy_runner=warmup_policy_runner,
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

    manifest_path = output_dir / "stage2_privileged_replay_manifest.json"
    write_privileged_replay_manifest(
        manifest_path,
        config_path=config_path,
        episode_paths=episode_paths,
        extra={
            "git_commit": _git_output(["rev-parse", "HEAD"]),
            "git_status_short": _git_output(["status", "--short", "RCS-AC-v2"]),
            "run_kind": "official_clean_clone_replay_collection",
            "target_speed_kmh": float(args.target_speed_kmh),
            "steer_sign": float(args.steer_sign),
            "record_after_route_progress": float(args.record_after_route_progress),
            "min_recorded_steps": int(args.min_recorded_steps),
            "warmup_policy_config": str(args.warmup_policy_config or ""),
        },
    )
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


def _select_warmup_policy_action(runner: Any, obs: dict[str, Any], info: dict[str, Any]) -> int:
    output = runner.act(obs, info)
    logits = output["logits"]
    if tuple(logits.shape) != (1, 1, len(RAW2DRIVE_ACTIONS)) or not torch.isfinite(logits).all():
        raise RuntimeError("warmup policy logits must be finite [1,1,39]")
    return int(torch.argmax(logits, dim=-1).item())


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _validate_simulator_state_privileged_info(info: dict[str, Any]) -> None:
    if info.get("privileged_observation_clean_gate_eligible") is not True:
        raise RuntimeError("replay collection requires clean-gate-eligible simulator-state privileged observations")
    if info.get("privileged_observation_source") != PRIVILEGED_OBSERVATION_SOURCE:
        raise RuntimeError("replay collection requires privileged_observation_source=carla_simulator_state")
    if info.get("privileged_observation_schema") != SEMANTIC43_SCHEMA_VERSION:
        raise RuntimeError("replay collection semantic schema mismatch")
    if info.get("privileged_bev_channel_names_sha256") != SEMANTIC43_CHANNEL_NAMES_SHA256:
        raise RuntimeError("replay collection semantic channel hash mismatch")
    if info.get("privileged_ego_schema") != EGO8_SCHEMA_VERSION:
        raise RuntimeError("replay collection ego schema mismatch")


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except Exception:
        return "UNSET"
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
