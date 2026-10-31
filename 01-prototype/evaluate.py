from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _bootstrap_carla() -> None:
    try:
        import carla  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    search_roots: list[Path] = []

    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        search_roots.append(Path(env_root))

    search_roots.extend(
        [
            Path.home() / "Downloads" / "CARLA_0.9.16",
            Path("C:/CARLA_0.9.16"),
        ]
    )

    for root in search_roots:
        dist_dir = root / "PythonAPI" / "carla" / "dist"
        if not dist_dir.exists():
            continue

        matching = sorted(dist_dir.glob(f"carla-*-{py_tag}-{py_tag}-win_amd64.whl"))
        if matching:
            sys.path.append(str(matching[-1]))
            import carla  # noqa: F401
            return

    raise RuntimeError(
        "CARLA Python API was not found. Install the matching CARLA wheel "
        "or set CARLA_ROOT to your CARLA installation directory."
    )


_bootstrap_carla()

from env.carla_env import CarlaEnv
from rl.ppo_agent import PPOAgent, PPOConfig
from train import build_model_config, build_policy_observation, load_config, set_seed
from utils.common import ensure_dir, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO baselines on CARLA observations.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--config", required=True, help="Training config used to build the model.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--run-name", default="eval_run", help="Evaluation run name.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic mean action.")
    return parser.parse_args()


def open_eval_log(log_path: str) -> tuple[Any, csv.DictWriter]:
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "timestamp",
            "model_type",
            "episode_index",
            "episode_reward",
            "episode_length",
            "done_reason",
            "route_progress",
            "reached_goal",
            "collision",
            "offroad",
            "stuck",
        ],
    )
    writer.writeheader()
    return log_file, writer


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    model_type = str(checkpoint_config.get("model_type", config.get("model_type", "single_modal")))
    set_seed(int(config.get("seed", 7)))

    env_cfg_raw = config.get("env", {})
    output_root = ensure_dir(os.path.join("outputs", "eval", args.run_name))
    logs_dir = ensure_dir(os.path.join("logs", f"{args.run_name}_{timestamp_string().replace(':', '').replace('-', '')}"))
    eval_log_path = os.path.join(logs_dir, "evaluation_log.csv")

    env = CarlaEnv(
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 2000)),
        town=str(config.get("town", "Town04")),
        output_dir=output_root,
        image_every=int(env_cfg_raw.get("image_every", 50)),
        client_timeout=float(env_cfg_raw.get("client_timeout", 10.0)),
        route_length=int(env_cfg_raw.get("route_length", 40)),
        route_spacing=float(env_cfg_raw.get("route_spacing", 3.0)),
        max_steps=int(config.get("max_episode_steps", 128)),
    )

    log_file, log_writer = open_eval_log(eval_log_path)
    try:
        reset_result = env.reset()
        obs = reset_result["obs"]
        model_config = build_model_config(obs, config, model_type)

        agent = PPOAgent(
            PPOConfig(
                model_type=model_type,
                obs_dim=int(model_config["obs_dim"]),
                action_dim=int(model_config["action_dim"]),
                hidden_dim=int(model_config["hidden_dim"]),
                device="cpu",
                model_kwargs=dict(model_config["model_kwargs"]),
            )
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.model.eval()

        episode_rewards: list[float] = []
        for episode_index in range(int(args.episodes)):
            result = env.reset()
            observation = build_policy_observation(result["obs"], model_type)
            episode_reward = 0.0
            episode_length = 0
            done_reason = "running"
            route_progress = 0.0
            reached_goal = False
            collision = False
            offroad = False
            stuck = False

            while True:
                action, _, _ = agent.act(observation, deterministic=bool(args.deterministic))
                step_result = env.step(action)
                episode_reward += float(step_result["reward"])
                episode_length += 1
                done_reason = str(step_result["info"].get("done_reason", "running"))
                route_progress = float(step_result["info"].get("route_progress", 0.0))
                reached_goal = bool(step_result["info"].get("reached_goal", False))
                collision = bool(step_result["info"].get("collision", False))
                offroad = bool(step_result["info"].get("offroad", False))
                stuck = bool(step_result["info"].get("stuck", False))

                if step_result["done"]:
                    break

                observation = build_policy_observation(step_result["obs"], model_type)

            episode_rewards.append(episode_reward)
            log_writer.writerow(
                {
                    "timestamp": timestamp_string(),
                    "model_type": model_type,
                    "episode_index": episode_index,
                    "episode_reward": round(float(episode_reward), 6),
                    "episode_length": episode_length,
                    "done_reason": done_reason,
                    "route_progress": round(float(route_progress), 6),
                    "reached_goal": reached_goal,
                    "collision": collision,
                    "offroad": offroad,
                    "stuck": stuck,
                }
            )
            log_file.flush()
            print(
                f"[eval] episode={episode_index} reward={episode_reward:.3f} length={episode_length} "
                f"done_reason={done_reason} route_progress={route_progress:.3f} "
                f"goal={reached_goal} collision={collision} offroad={offroad} stuck={stuck}"
            )

        print(
            f"[eval-summary] model_type={model_type} episodes={len(episode_rewards)} "
            f"mean_reward={float(np.mean(episode_rewards)):.3f} log={eval_log_path}"
        )
        return 0
    finally:
        env.close()
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
