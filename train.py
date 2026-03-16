from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


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
from models.single_modal_actor_critic import ModelConfig
from rl.ppo_agent import PPOAgent, PPOConfig
from rl.rollout_buffer import RolloutBuffer
from utils.common import ensure_dir, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-modal PPO baseline on CARLA ego+route observations.")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "train_single_modal.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Optional override for total timesteps.")
    parser.add_argument("--run-name", default=None, help="Optional override for run name.")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping, got {type(data)}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_single_modal_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    ego = observation["ego"].astype(np.float32, copy=False)
    route = observation["route"].astype(np.float32, copy=False)
    state = np.concatenate([ego, route], axis=0).astype(np.float32, copy=False)
    if state.shape != (13,):
        raise ValueError(f"Expected single-modal state shape (13,), got {state.shape}")
    return state


def open_training_log(log_path: str) -> tuple[Any, csv.DictWriter]:
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "timestamp",
            "record_type",
            "update",
            "total_steps",
            "episode_index",
            "episode_reward",
            "episode_length",
            "done_reason",
            "route_progress",
            "actor_loss",
            "critic_loss",
            "entropy",
        ],
    )
    writer.writeheader()
    return log_file, writer


def write_training_log(
    writer: csv.DictWriter,
    log_file: Any,
    record_type: str,
    update: int,
    total_steps: int,
    episode_index: int,
    episode_reward: float,
    episode_length: int,
    done_reason: str,
    route_progress: float,
    metrics: dict[str, float],
) -> None:
    writer.writerow(
        {
            "timestamp": timestamp_string(),
            "record_type": record_type,
            "update": update,
            "total_steps": total_steps,
            "episode_index": episode_index,
            "episode_reward": round(float(episode_reward), 6),
            "episode_length": episode_length,
            "done_reason": done_reason,
            "route_progress": round(float(route_progress), 6),
            "actor_loss": round(float(metrics.get("actor_loss", 0.0)), 6),
            "critic_loss": round(float(metrics.get("critic_loss", 0.0)), 6),
            "entropy": round(float(metrics.get("entropy", 0.0)), 6),
        }
    )
    log_file.flush()


def save_checkpoint(agent: PPOAgent, checkpoint_path: str, update: int, total_steps: int, episode_index: int) -> None:
    agent.save(
        checkpoint_path,
        extra_state={
            "update": update,
            "total_steps": total_steps,
            "episode_index": episode_index,
        },
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.total_timesteps is not None:
        config["total_timesteps"] = int(args.total_timesteps)
    if args.run_name is not None:
        config["run_name"] = str(args.run_name)

    seed = int(config.get("seed", 7))
    set_seed(seed)

    run_name = f"{config.get('run_name', 'single_modal_baseline')}_{timestamp_string().replace(':', '').replace('-', '')}"
    logs_dir = ensure_dir(os.path.join(config.get("logs_dir", "logs"), run_name))
    checkpoints_dir = ensure_dir(os.path.join(config.get("checkpoints_dir", "checkpoints"), run_name))
    env_output_dir = ensure_dir(os.path.join(config.get("output_root", "outputs/ppo_single_modal_env"), run_name))
    training_log_path = os.path.join(logs_dir, "training_log.csv")

    model_cfg_raw = config.get("model", {})
    ppo_cfg_raw = config.get("ppo", {})
    env_cfg_raw = config.get("env", {})

    model_config = ModelConfig(
        obs_dim=int(model_cfg_raw.get("obs_dim", 13)),
        action_dim=int(model_cfg_raw.get("action_dim", 3)),
        hidden_dim=int(model_cfg_raw.get("hidden_dim", 128)),
    )
    ppo_config = PPOConfig(
        obs_dim=model_config.obs_dim,
        action_dim=model_config.action_dim,
        hidden_dim=model_config.hidden_dim,
        learning_rate=float(ppo_cfg_raw.get("learning_rate", 3e-4)),
        gamma=float(ppo_cfg_raw.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg_raw.get("gae_lambda", 0.95)),
        clip_coef=float(ppo_cfg_raw.get("clip_coef", 0.2)),
        value_coef=float(ppo_cfg_raw.get("value_coef", 0.5)),
        entropy_coef=float(ppo_cfg_raw.get("entropy_coef", 0.01)),
        max_grad_norm=float(ppo_cfg_raw.get("max_grad_norm", 0.5)),
        update_epochs=int(ppo_cfg_raw.get("update_epochs", 4)),
        minibatch_size=int(ppo_cfg_raw.get("minibatch_size", 64)),
        rollout_steps=int(config.get("rollout_steps", 128)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    agent = PPOAgent(config=ppo_config)
    buffer = RolloutBuffer(
        capacity=ppo_config.rollout_steps,
        obs_dim=model_config.obs_dim,
        action_dim=model_config.action_dim,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
    )

    env = CarlaEnv(
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 2000)),
        town=str(config.get("town", "Town04")),
        output_dir=env_output_dir,
        image_every=int(env_cfg_raw.get("image_every", 50)),
        client_timeout=float(env_cfg_raw.get("client_timeout", 10.0)),
        route_length=int(env_cfg_raw.get("route_length", 40)),
        route_spacing=float(env_cfg_raw.get("route_spacing", 3.0)),
        max_steps=int(config.get("max_episode_steps", 128)),
    )

    total_timesteps = int(config.get("total_timesteps", 512))
    save_every_updates = int(config.get("save_every_updates", 1))
    log_file, log_writer = open_training_log(training_log_path)

    total_steps = 0
    update_index = 0
    episode_index = 0
    episode_reward = 0.0
    episode_length = 0
    last_done_reason = "running"
    last_route_progress = 0.0

    try:
        reset_result = env.reset()
        observation = build_single_modal_state(reset_result["obs"])

        while total_steps < total_timesteps:
            buffer.reset()
            for _ in range(ppo_config.rollout_steps):
                action, log_prob, value = agent.act(observation, deterministic=False)
                step_result = env.step(action)

                next_observation = build_single_modal_state(step_result["obs"])
                reward = float(step_result["reward"])
                done = bool(step_result["done"])
                info = step_result["info"]

                if not np.isfinite(reward):
                    raise RuntimeError(f"Encountered non-finite reward: {reward}")
                if not np.all(np.isfinite(action)):
                    raise RuntimeError(f"Encountered non-finite action: {action}")

                buffer.add(
                    observation=observation,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                )

                episode_reward += reward
                episode_length += 1
                total_steps += 1
                observation = next_observation
                last_done_reason = str(info.get("done_reason", "running"))
                last_route_progress = float(info.get("route_progress", 0.0))

                if done:
                    write_training_log(
                        writer=log_writer,
                        log_file=log_file,
                        record_type="episode",
                        update=update_index,
                        total_steps=total_steps,
                        episode_index=episode_index,
                        episode_reward=episode_reward,
                        episode_length=episode_length,
                        done_reason=last_done_reason,
                        route_progress=last_route_progress,
                        metrics={},
                    )
                    print(
                        f"[episode] idx={episode_index} total_steps={total_steps} "
                        f"reward={episode_reward:.3f} length={episode_length} "
                        f"done_reason={last_done_reason} route_progress={last_route_progress:.3f}"
                    )
                    episode_index += 1
                    episode_reward = 0.0
                    episode_length = 0
                    observation = build_single_modal_state(env.reset()["obs"])

                if total_steps >= total_timesteps:
                    break

            with torch.no_grad():
                if buffer.ptr == 0:
                    break
                if buffer.dones[buffer.ptr - 1] >= 1.0:
                    last_value = 0.0
                    last_done = True
                else:
                    _, _, last_value = agent.act(observation, deterministic=True)
                    last_done = False

            buffer.compute_returns_and_advantages(last_value=last_value, last_done=last_done)
            metrics = agent.update(buffer.get_training_batch(agent.device))
            update_index += 1

            print(
                f"[update] idx={update_index} total_steps={total_steps} "
                f"actor_loss={metrics['actor_loss']:.4f} critic_loss={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )

            write_training_log(
                writer=log_writer,
                log_file=log_file,
                record_type="update",
                update=update_index,
                total_steps=total_steps,
                episode_index=episode_index,
                episode_reward=episode_reward,
                episode_length=episode_length,
                done_reason=last_done_reason,
                route_progress=last_route_progress,
                metrics=metrics,
            )

            checkpoint_path = os.path.join(checkpoints_dir, f"update_{update_index:04d}.pt")
            if update_index % save_every_updates == 0:
                save_checkpoint(
                    agent=agent,
                    checkpoint_path=checkpoint_path,
                    update=update_index,
                    total_steps=total_steps,
                    episode_index=episode_index,
                )

        final_checkpoint = os.path.join(checkpoints_dir, "last.pt")
        save_checkpoint(
            agent=agent,
            checkpoint_path=final_checkpoint,
            update=update_index,
            total_steps=total_steps,
            episode_index=episode_index,
        )
        print(f"[done] Training finished. logs={training_log_path} checkpoint={final_checkpoint}")
        return 0
    finally:
        env.close()
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
