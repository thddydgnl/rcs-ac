"""Stage 0 validation for RCSToyNavEnv.

Purpose:
- verify environment stability before training,
- catch observation/reward/done/logging failures early,
- compare clean vs corrupted runs with random and heuristic policies,
- save episode logs for later reliability analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rcs_toy_nav_env import RCSToyNavEnv


DEFAULT_NUM_EPISODES = 20
DEFAULT_LOG_DIR = Path("/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/stage0_logs")
EXPECTED_OBS_KEYS = ("vision", "distance", "ego", "route", "aux")
EXPECTED_OBS_SHAPES = {
    "vision": (8,),
    "distance": (9,),
    "ego": (5,),
    "route": (5,),
    "aux": (8,),
}
EXPECTED_INFO_KEYS = (
    "corruption_flags",
    "quality_stats",
    "risk_stats",
    "goal_reached",
    "collision",
    "out_of_bounds",
    "timeout",
    "reward/total",
    "metric/goal_distance",
)
SUMMARY_FLOAT_KEYS = (
    "total_reward",
    "episode_length",
    "success",
    "collision",
    "out_of_bounds",
    "timeout",
    "min_goal_distance",
    "final_goal_distance",
    "vision_corrupted",
    "distance_corrupted",
    "ego_corrupted",
    "route_corrupted",
    "avg_vision_quality",
    "avg_distance_quality",
    "avg_risk",
)


def random_policy(env: RCSToyNavEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
    del obs, info
    return env.action_space.sample()


def heuristic_policy(env: RCSToyNavEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
    del info
    route = obs["route"]
    distance = obs["distance"]

    goal_rel_angle = float(np.arctan2(route[0], np.clip(route[1], -1.0, 1.0)))
    center_ray = len(distance) // 2
    left_window = float(np.mean(distance[:center_ray]))
    center_window = float(distance[center_ray])
    right_window = float(np.mean(distance[center_ray + 1 :]))

    avoid_turn = np.clip((right_window - left_window) * 2.0, -1.0, 1.0)
    goal_turn = np.clip(goal_rel_angle / np.pi, -1.0, 1.0)

    if center_window < 0.30:
        turn = 0.7 * avoid_turn + 0.3 * goal_turn
    else:
        turn = 0.8 * goal_turn + 0.2 * avoid_turn
    turn = float(np.clip(turn, -1.0, 1.0))

    goal_distance_norm = float(route[2])
    if goal_distance_norm > 0.30:
        accel = 0.9
    elif goal_distance_norm > 0.10:
        accel = 0.5
    else:
        accel = 0.1

    if center_window < 0.20:
        accel = min(accel, 0.2)

    return np.array([accel, turn], dtype=np.float32)


def validate_observation(
    obs: dict[str, np.ndarray],
    expected_shapes: dict[str, tuple[int, ...]] = EXPECTED_OBS_SHAPES,
) -> list[str]:
    errors: list[str] = []

    if not isinstance(obs, dict):
        return ["observation is not a dict"]

    for key in EXPECTED_OBS_KEYS:
        if key not in obs:
            errors.append(f"missing observation key: {key}")
            continue

        value = np.asarray(obs[key])
        if value.shape != expected_shapes[key]:
            errors.append(f"{key} shape mismatch: expected {expected_shapes[key]}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            errors.append(f"{key} contains NaN or Inf")

        if key in ("vision", "distance"):
            if np.any(value < -1e-6) or np.any(value > 1.0 + 1e-6):
                errors.append(f"{key} out of expected range [0, 1]")
        else:
            if np.any(value < -1.0 - 1e-6) or np.any(value > 1.0 + 1e-6):
                errors.append(f"{key} out of expected range [-1, 1]")

    return errors


def validate_step_result(
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    if not np.isfinite(reward):
        errors.append("reward is not finite")
    if not isinstance(terminated, bool):
        errors.append("terminated is not bool")
    if not isinstance(truncated, bool):
        errors.append("truncated is not bool")
    if not isinstance(info, dict):
        errors.append("info is not dict")
        return errors

    for key in EXPECTED_INFO_KEYS:
        if key not in info:
            errors.append(f"missing info key: {key}")

    if "reward/total" in info and not np.isfinite(info["reward/total"]):
        errors.append("info['reward/total'] is not finite")
    if "metric/goal_distance" in info and info["metric/goal_distance"] < -1e-6:
        errors.append("goal distance is negative")

    return errors


def infer_terminated_reason(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> str:
    if info.get("goal_reached", False):
        return "goal"
    if info.get("collision", False):
        return "collision"
    if info.get("out_of_bounds", False):
        return "out_of_bounds"
    if truncated or info.get("timeout", False):
        return "timeout"
    if terminated:
        return "terminated_unknown"
    return "running"


def run_episode(
    env: RCSToyNavEnv,
    policy_fn: Callable[[RCSToyNavEnv, dict[str, np.ndarray], dict[str, Any]], np.ndarray],
    episode_id: int,
    policy_type: str,
    corruption_mode: str,
    seed: int,
) -> dict[str, Any]:
    env.set_corruption_mode(corruption_mode)
    obs, info = env.reset(seed=seed)

    observation_errors = validate_observation(obs)
    if observation_errors:
        raise AssertionError(f"reset observation validation failed: {observation_errors}")

    total_reward = 0.0
    min_goal_distance = float("inf")
    quality_vision_values: list[float] = []
    quality_distance_values: list[float] = []
    risk_values: list[float] = []
    step_errors: list[str] = []
    final_info = info

    while True:
        action = policy_fn(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        observation_errors = validate_observation(obs)
        result_errors = validate_step_result(reward, terminated, truncated, info)
        if observation_errors or result_errors:
            step_errors.extend(observation_errors + result_errors)
            raise AssertionError(f"step validation failed: {step_errors}")

        total_reward += float(reward)
        min_goal_distance = min(min_goal_distance, float(info["metric/goal_distance"]))
        quality_vision_values.append(float(info["quality_stats"]["vision_quality"]))
        quality_distance_values.append(float(info["quality_stats"]["distance_quality"]))
        risk_values.append(float(info["risk_stats"]["fused_risk"]))
        final_info = info

        if terminated or truncated:
            break

    corruption_flags = final_info["corruption_flags"]
    summary = {
        "episode_id": episode_id,
        "policy_type": policy_type,
        "corruption_mode": corruption_mode,
        "total_reward": float(total_reward),
        "episode_length": int(env.step_count),
        "success": int(final_info["goal_reached"]),
        "collision": int(final_info["collision"]),
        "out_of_bounds": int(final_info["out_of_bounds"]),
        "timeout": int(final_info["timeout"]),
        "min_goal_distance": float(min_goal_distance),
        "final_goal_distance": float(final_info["metric/goal_distance"]),
        "vision_corrupted": int(corruption_flags["vision_corrupted"]),
        "distance_corrupted": int(corruption_flags["distance_corrupted"]),
        "ego_corrupted": int(corruption_flags["ego_corrupted"]),
        "route_corrupted": int(corruption_flags["route_corrupted"]),
        "avg_vision_quality": float(np.mean(quality_vision_values)),
        "avg_distance_quality": float(np.mean(quality_distance_values)),
        "avg_risk": float(np.mean(risk_values)),
        "terminated_reason": infer_terminated_reason(True, final_info["timeout"], final_info),
    }
    return summary


def run_evaluation(
    policy_name: str,
    policy_fn: Callable[[RCSToyNavEnv, dict[str, np.ndarray], dict[str, Any]], np.ndarray],
    corruption_mode: str,
    num_episodes: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    env = RCSToyNavEnv()
    episode_summaries: list[dict[str, Any]] = []

    for episode_idx in range(num_episodes):
        episode_summary = run_episode(
            env=env,
            policy_fn=policy_fn,
            episode_id=episode_idx,
            policy_type=policy_name,
            corruption_mode=corruption_mode,
            seed=base_seed + episode_idx,
        )
        episode_summaries.append(episode_summary)

    return episode_summaries


def save_episode_summaries_jsonl(episode_summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in episode_summaries:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_episode_summaries_csv(episode_summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(episode_summaries[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_summaries)


def aggregate_results(episode_summaries: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "num_episodes": float(len(episode_summaries)),
        "avg_total_reward": float(np.mean([row["total_reward"] for row in episode_summaries])),
        "avg_episode_length": float(np.mean([row["episode_length"] for row in episode_summaries])),
        "success_rate": float(np.mean([row["success"] for row in episode_summaries])),
        "collision_rate": float(np.mean([row["collision"] for row in episode_summaries])),
        "timeout_rate": float(np.mean([row["timeout"] for row in episode_summaries])),
        "avg_min_goal_distance": float(np.mean([row["min_goal_distance"] for row in episode_summaries])),
        "avg_final_goal_distance": float(np.mean([row["final_goal_distance"] for row in episode_summaries])),
        "avg_vision_quality": float(np.mean([row["avg_vision_quality"] for row in episode_summaries])),
        "avg_distance_quality": float(np.mean([row["avg_distance_quality"] for row in episode_summaries])),
        "avg_risk": float(np.mean([row["avg_risk"] for row in episode_summaries])),
    }


def print_summary_table(
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]]
) -> None:
    print("\nStage 0 Summary Statistics")
    for (policy_name, corruption_mode), rows in grouped_results.items():
        stats = aggregate_results(rows)
        print(
            f"- policy={policy_name:9s} mode={corruption_mode:10s} "
            f"episodes={int(stats['num_episodes'])} "
            f"avg_return={stats['avg_total_reward']:.3f} "
            f"success_rate={stats['success_rate']:.3f} "
            f"collision_rate={stats['collision_rate']:.3f} "
            f"timeout_rate={stats['timeout_rate']:.3f} "
            f"avg_risk={stats['avg_risk']:.3f}"
        )


def build_pass_fail_checklist(
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]],
    log_paths: list[Path],
    min_episodes: int,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    total_episode_count = sum(len(rows) for rows in grouped_results.values())
    checks.append(
        (
            "Episode count requirement",
            total_episode_count >= 4 * min_episodes,
            f"ran {total_episode_count} episodes across random/heuristic x clean/corrupted",
        )
    )

    clean_present = ("random", "clean") in grouped_results and ("heuristic", "clean") in grouped_results
    corrupted_present = ("random", "corrupted") in grouped_results and ("heuristic", "corrupted") in grouped_results
    checks.append(("Clean mode executed", clean_present, "random and heuristic clean runs present"))
    checks.append(("Corrupted mode executed", corrupted_present, "random and heuristic corrupted runs present"))

    log_success = all(path.exists() and path.stat().st_size > 0 for path in log_paths)
    checks.append(("Log files saved", log_success, ", ".join(str(path) for path in log_paths)))

    no_nan = True
    for rows in grouped_results.values():
        for row in rows:
            for key in SUMMARY_FLOAT_KEYS:
                if not np.isfinite(float(row[key])):
                    no_nan = False
                    break
    checks.append(("No NaN in episode summaries", no_nan, "all numeric summary fields are finite"))

    return checks


def print_pass_fail_checklist(checks: list[tuple[str, bool, str]]) -> None:
    print("\nStage 0 Pass/Fail Checklist")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"- [{status}] {name}: {detail}")


def run_stage0_validation(
    num_episodes: int = DEFAULT_NUM_EPISODES,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    policies = {
        "random": random_policy,
        "heuristic": heuristic_policy,
    }
    corruption_modes = ("clean", "corrupted")

    for policy_name, policy_fn in policies.items():
        for corruption_mode in corruption_modes:
            rows = run_evaluation(
                policy_name=policy_name,
                policy_fn=policy_fn,
                corruption_mode=corruption_mode,
                num_episodes=num_episodes,
                base_seed=1000 if corruption_mode == "clean" else 2000,
            )
            grouped_results[(policy_name, corruption_mode)] = rows

    all_rows = [row for rows in grouped_results.values() for row in rows]
    jsonl_path = log_dir / "stage0_episode_summaries.jsonl"
    csv_path = log_dir / "stage0_episode_summaries.csv"
    save_episode_summaries_jsonl(all_rows, jsonl_path)
    save_episode_summaries_csv(all_rows, csv_path)

    print_summary_table(grouped_results)
    checks = build_pass_fail_checklist(grouped_results, [jsonl_path, csv_path], num_episodes)
    print_pass_fail_checklist(checks)

    return grouped_results


def run_smoke_test() -> None:
    env = RCSToyNavEnv()

    for corruption_mode in ("clean", "corrupted"):
        env.set_corruption_mode(corruption_mode)
        obs, info = env.reset(seed=123)
        observation_errors = validate_observation(obs)
        if observation_errors:
            raise AssertionError(f"smoke test observation errors: {observation_errors}")

        action = heuristic_policy(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        observation_errors = validate_observation(obs)
        result_errors = validate_step_result(reward, terminated, truncated, info)
        if observation_errors or result_errors:
            raise AssertionError(f"smoke test step errors: {observation_errors + result_errors}")

    print("Stage 0 smoke test passed for clean and corrupted modes.")


if __name__ == "__main__":
    run_smoke_test()
    run_stage0_validation(num_episodes=20)
