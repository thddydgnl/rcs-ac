"""Sample script to smoke-test the CARLA F1 strategy environment with random actions."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EnvConfig
from carla_strategy_env.envs import CarlaF1StrategyEnv
from utils.codex_artifacts import CodexArtifactRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test for the CARLA F1 strategy environment.")
    parser.add_argument("--laps", type=int, default=3, help="Number of laps to configure for the episode.")
    parser.add_argument("--steps", type=int, default=20, help="Maximum number of strategy steps to run.")
    parser.add_argument(
        "--action-mode",
        choices=("random", "stay_out", "cycle"),
        default="random",
        help="How to select strategy actions during the run.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used for deterministic random smoke tests.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional output directory for Codex-friendly diagnostics and latest camera frame.",
    )
    return parser.parse_args()


def choose_action(step: int, action_space_n: int, mode: str, rng: random.Random) -> int:
    if mode == "stay_out":
        return 0
    if mode == "cycle":
        return step % action_space_n
    return rng.randrange(action_space_n)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    rng = random.Random(args.seed)

    config = EnvConfig()
    config.race.total_laps = args.laps
    env = CarlaF1StrategyEnv(config)
    recorder = CodexArtifactRecorder(args.artifacts_dir) if args.artifacts_dir is not None else None
    total_reward = 0.0
    steps_completed = 0
    last_info: dict[str, object] = {}
    last_reward = 0.0
    terminated = False
    truncated = False

    try:
        observation, info = env.reset()
        last_info = dict(info)
        logging.info("Initial observation keys: %s", observation.keys())
        logging.info("Initial strategy info: %s", info)
        if recorder is not None:
            frame_name = recorder.record_frame(env.render())
            recorder.append_step({"step": 0, "event": "reset", "info": info, "frame": frame_name})

        for step in range(args.steps):
            action = choose_action(step, env.action_space.n, args.action_mode, rng)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps_completed = step + 1
            last_info = dict(info)
            last_reward = reward
            logging.info(
                "step=%s action=%s reward=%.3f lap=%s remaining=%s tire=%s pit_count=%s",
                step,
                action,
                reward,
                info["current_lap"],
                info["laps_remaining"],
                info["tire_compound"],
                info["pit_count"],
            )
            if recorder is not None:
                frame_name = recorder.record_frame(env.render())
                recorder.append_step(
                    {
                        "step": steps_completed,
                        "action": action,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                        "frame": frame_name,
                    }
                )
            if terminated or truncated:
                logging.info("Episode ended: terminated=%s truncated=%s", terminated, truncated)
                break
    except Exception as exc:
        if recorder is not None:
            errors = [
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ]
            recorder.finalize(
                summary={
                    "status": "failed",
                    "mode": "carla",
                    "steps_requested": args.steps,
                    "steps_completed": steps_completed,
                    "action_mode": args.action_mode,
                },
                metrics={
                    "total_reward": total_reward,
                    "last_reward": last_reward,
                    "steps_completed": steps_completed,
                    "laps_requested": args.laps,
                    "seed": args.seed,
                },
                errors=errors,
            )
        raise
    finally:
        env.close()

    summary = {
        "status": "completed",
        "mode": "carla",
        "steps_requested": args.steps,
        "steps_completed": steps_completed,
        "action_mode": args.action_mode,
        "terminated": terminated,
        "truncated": truncated,
        "current_lap": last_info.get("current_lap"),
        "laps_remaining": last_info.get("laps_remaining"),
        "pit_count": last_info.get("pit_count"),
        "progress": last_info.get("progress"),
    }
    metrics = {
        "total_reward": total_reward,
        "last_reward": last_reward,
        "steps_completed": steps_completed,
        "laps_requested": args.laps,
        "seed": args.seed,
        "tire_compound": last_info.get("tire_compound"),
        "tire_wear": last_info.get("tire_wear"),
        "fuel_kg": last_info.get("fuel_kg"),
        "best_lap_time_seconds": last_info.get("best_lap_time_seconds"),
        "last_lap_time_seconds": last_info.get("last_lap_time_seconds"),
    }
    if recorder is not None:
        recorder.finalize(summary=summary, metrics=metrics, errors=[])
    print(json.dumps({"summary": summary, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
