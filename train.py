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
from env.reward import RewardConfig
from rl.ppo_agent import PPOAgent, PPOConfig
from rl.rollout_buffer import RolloutBuffer
from utils.common import ensure_dir, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO baselines on CARLA observations.")
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "train_single_modal.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Optional override for total timesteps.")
    parser.add_argument("--run-name", default=None, help="Optional override for run name.")
    parser.add_argument("--model-type", default=None, choices=["single_modal", "early_fusion", "attention_fusion"], help="Optional override for model type.")
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


def build_early_fusion_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    rgb = observation["rgb"].astype(np.float32, copy=False).reshape(-1)
    lidar_bev = observation["lidar_bev"].astype(np.float32, copy=False).reshape(-1)
    ego = observation["ego"].astype(np.float32, copy=False).reshape(-1)
    route = observation["route"].astype(np.float32, copy=False).reshape(-1)
    return np.concatenate([rgb, lidar_bev, ego, route], axis=0).astype(np.float32, copy=False)


def build_policy_observation(observation: dict[str, np.ndarray], model_type: str) -> np.ndarray:
    if model_type == "single_modal":
        return build_single_modal_state(observation)
    if model_type in {"early_fusion", "attention_fusion"}:
        return build_early_fusion_state(observation)
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_model_config(observation: dict[str, np.ndarray], config: dict[str, Any], model_type: str) -> dict[str, Any]:
    model_cfg_raw = config.get("model", {})
    if model_type == "single_modal":
        return {
            "obs_dim": int(model_cfg_raw.get("obs_dim", 13)),
            "action_dim": int(model_cfg_raw.get("action_dim", 3)),
            "hidden_dim": int(model_cfg_raw.get("hidden_dim", 128)),
            "model_kwargs": {"min_std": float(model_cfg_raw.get("min_std", 0.05))},
        }

    rgb_shape = tuple(int(v) for v in observation["rgb"].shape)
    lidar_shape = tuple(int(v) for v in observation["lidar_bev"].shape)
    ego_dim = int(observation["ego"].shape[0])
    route_dim = int(observation["route"].shape[0])
    obs_dim = (
        int(np.prod(rgb_shape))
        + int(np.prod(lidar_shape))
        + ego_dim
        + route_dim
    )
    return {
        "obs_dim": obs_dim,
        "action_dim": int(model_cfg_raw.get("action_dim", 3)),
        "hidden_dim": int(model_cfg_raw.get("hidden_dim", 256)),
        "model_kwargs": {
            "min_std": float(model_cfg_raw.get("min_std", 0.05)),
            "rgb_shape": rgb_shape,
            "lidar_shape": lidar_shape,
            "ego_dim": ego_dim,
            "route_dim": route_dim,
            "rgb_feature_dim": int(model_cfg_raw.get("rgb_feature_dim", 64)),
            "lidar_feature_dim": int(model_cfg_raw.get("lidar_feature_dim", 32)),
            "ego_feature_dim": int(model_cfg_raw.get("ego_feature_dim", 32)),
            "route_feature_dim": int(model_cfg_raw.get("route_feature_dim", 32)),
            "attention_heads": int(model_cfg_raw.get("attention_heads", 4)),
            "attention_layers": int(model_cfg_raw.get("attention_layers", 1)),
            "dropout": float(model_cfg_raw.get("dropout", 0.0)),
        },
    }


def open_training_log(log_path: str) -> tuple[Any, csv.DictWriter]:
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "timestamp",
            "model_type",
            "seed",
            "record_type",
            "update",
            "total_steps",
            "episode_idx",
            "episode_index",
            "episode_reward",
            "episode_length",
            "done_reason",
            "route_progress",
            "reached_goal",
            "collision",
            "offroad",
            "stuck",
            "average_reward",
            "average_route_progress",
            "policy_loss",
            "value_loss",
            "actor_loss",
            "critic_loss",
            "entropy",
            "attention_mean_rgb",
            "attention_mean_lidar",
            "attention_mean_ego",
            "attention_mean_route",
            "spawn_point_index",
            "initial_speed_kmh",
            "initial_heading_error",
            "initial_lane_offset",
            "initial_target_distance",
            "average_speed_kmh",
            "low_speed_step_count",
            "stuck_trigger_step",
            "recent_control_window",
            "recent_steer_mean",
            "recent_throttle_mean",
            "recent_brake_mean",
            "recent_speed_mean_kmh",
            "pedal_overlap_suppressed",
            "low_speed_brake_suppressed",
            "final_speed_kmh",
            "lane_offset",
            "heading_error",
            "target_distance",
            "offroad_candidate",
            "offroad_steps",
            "offroad_threshold",
            "offroad_grace_steps",
            "first_offroad_step",
            "final_action_steer",
            "final_action_throttle",
            "final_action_brake",
        ],
    )
    writer.writeheader()
    return log_file, writer


def write_training_log(
    writer: csv.DictWriter,
    log_file: Any,
    model_type: str,
    seed: int,
    record_type: str,
    update: int,
    total_steps: int,
    episode_index: int,
    episode_reward: float,
    episode_length: int,
    done_reason: str,
    route_progress: float,
    reached_goal: bool,
    collision: bool,
    offroad: bool,
    stuck: bool,
    average_reward: float,
    average_route_progress: float,
    average_speed_kmh: float,
    initial_info: dict[str, Any] | None,
    latest_info: dict[str, Any] | None,
    metrics: dict[str, float],
) -> None:
    initial_info = initial_info or {}
    latest_info = latest_info or {}
    control = latest_info.get("control", {})
    writer.writerow(
        {
            "timestamp": timestamp_string(),
            "model_type": model_type,
            "seed": seed,
            "record_type": record_type,
            "update": update,
            "total_steps": total_steps,
            "episode_idx": episode_index,
            "episode_index": episode_index,
            "episode_reward": round(float(episode_reward), 6),
            "episode_length": episode_length,
            "done_reason": done_reason,
            "route_progress": round(float(route_progress), 6),
            "reached_goal": bool(reached_goal),
            "collision": bool(collision),
            "offroad": bool(offroad),
            "stuck": bool(stuck),
            "average_reward": round(float(average_reward), 6),
            "average_route_progress": round(float(average_route_progress), 6),
            "policy_loss": round(float(metrics.get("actor_loss", 0.0)), 6),
            "value_loss": round(float(metrics.get("critic_loss", 0.0)), 6),
            "actor_loss": round(float(metrics.get("actor_loss", 0.0)), 6),
            "critic_loss": round(float(metrics.get("critic_loss", 0.0)), 6),
            "entropy": round(float(metrics.get("entropy", 0.0)), 6),
            "attention_mean_rgb": round(float(metrics.get("attention_mean_rgb", 0.0)), 6),
            "attention_mean_lidar": round(float(metrics.get("attention_mean_lidar", 0.0)), 6),
            "attention_mean_ego": round(float(metrics.get("attention_mean_ego", 0.0)), 6),
            "attention_mean_route": round(float(metrics.get("attention_mean_route", 0.0)), 6),
            "spawn_point_index": int(initial_info.get("spawn_point_index", latest_info.get("spawn_point_index", -1))),
            "initial_speed_kmh": round(float(initial_info.get("speed_kmh", 0.0)), 6),
            "initial_heading_error": round(float(initial_info.get("heading_error", 0.0)), 6),
            "initial_lane_offset": round(float(initial_info.get("lane_offset", 0.0)), 6),
            "initial_target_distance": round(float(initial_info.get("target_distance", 0.0)), 6),
            "average_speed_kmh": round(float(average_speed_kmh), 6),
            "low_speed_step_count": int(latest_info.get("low_speed_steps", 0)),
            "stuck_trigger_step": int(latest_info.get("stuck_trigger_step", -1)),
            "recent_control_window": int(latest_info.get("recent_control_window", 0)),
            "recent_steer_mean": round(float(latest_info.get("recent_steer_mean", 0.0)), 6),
            "recent_throttle_mean": round(float(latest_info.get("recent_throttle_mean", 0.0)), 6),
            "recent_brake_mean": round(float(latest_info.get("recent_brake_mean", 0.0)), 6),
            "recent_speed_mean_kmh": round(float(latest_info.get("recent_speed_mean_kmh", 0.0)), 6),
            "pedal_overlap_suppressed": bool(latest_info.get("pedal_overlap_suppressed", False)),
            "low_speed_brake_suppressed": bool(latest_info.get("low_speed_brake_suppressed", False)),
            "final_speed_kmh": round(float(latest_info.get("speed_kmh", 0.0)), 6),
            "lane_offset": round(float(latest_info.get("lane_offset", 0.0)), 6),
            "heading_error": round(float(latest_info.get("heading_error", 0.0)), 6),
            "target_distance": round(float(latest_info.get("target_distance", 0.0)), 6),
            "offroad_candidate": bool(latest_info.get("offroad_candidate", False)),
            "offroad_steps": int(latest_info.get("offroad_steps", 0)),
            "offroad_threshold": round(float(latest_info.get("offroad_threshold", 0.0)), 6),
            "offroad_grace_steps": int(latest_info.get("offroad_grace_steps", 0)),
            "first_offroad_step": int(latest_info.get("first_offroad_step", -1)),
            "final_action_steer": round(float(control.get("steer", 0.0)), 6),
            "final_action_throttle": round(float(control.get("throttle", 0.0)), 6),
            "final_action_brake": round(float(control.get("brake", 0.0)), 6),
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
    if args.model_type is not None:
        config["model_type"] = str(args.model_type)

    seed = int(config.get("seed", 7))
    set_seed(seed)
    model_type = str(config.get("model_type", "single_modal"))

    run_name = f"{config.get('run_name', model_type)}_{timestamp_string().replace(':', '').replace('-', '')}"
    logs_dir = ensure_dir(os.path.join(config.get("logs_dir", "logs"), run_name))
    checkpoints_dir = ensure_dir(os.path.join(config.get("checkpoints_dir", "checkpoints"), run_name))
    env_output_dir = ensure_dir(os.path.join(config.get("output_root", "outputs/ppo_single_modal_env"), run_name))
    training_log_path = os.path.join(logs_dir, "training_log.csv")

    ppo_cfg_raw = config.get("ppo", {})
    env_cfg_raw = config.get("env", {})
    reward_cfg_raw = config.get("reward", {})

    reward_config = RewardConfig(
        progress_weight=float(reward_cfg_raw.get("progress_weight", 5.0)),
        lane_weight=float(reward_cfg_raw.get("lane_weight", 0.1)),
        collision_penalty=float(reward_cfg_raw.get("collision_penalty", 10.0)),
        offroad_penalty=float(reward_cfg_raw.get("offroad_penalty", 5.0)),
        lane_invasion_penalty=float(reward_cfg_raw.get("lane_invasion_penalty", 1.0)),
        steer_change_weight=float(reward_cfg_raw.get("steer_change_weight", 0.05)),
        throttle_change_weight=float(reward_cfg_raw.get("throttle_change_weight", 0.02)),
        brake_change_weight=float(reward_cfg_raw.get("brake_change_weight", 0.02)),
        goal_bonus=float(reward_cfg_raw.get("goal_bonus", 20.0)),
        stuck_penalty=float(reward_cfg_raw.get("stuck_penalty", 3.0)),
        progress_delta_clip=float(reward_cfg_raw.get("progress_delta_clip", 1.0)),
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
        reward_config=reward_config,
        stuck_speed_threshold_mps=float(env_cfg_raw.get("stuck_speed_threshold_mps", 0.3)),
        stuck_min_steps=int(env_cfg_raw.get("stuck_min_steps", 40)),
        offroad_min_steps=int(env_cfg_raw.get("offroad_min_steps", 5)),
        offroad_grace_steps=int(env_cfg_raw.get("offroad_grace_steps", 8)),
        offroad_lane_offset_threshold=(
            None
            if env_cfg_raw.get("offroad_lane_offset_threshold") is None
            else float(env_cfg_raw.get("offroad_lane_offset_threshold"))
        ),
        suppress_conflicting_pedals=bool(env_cfg_raw.get("suppress_conflicting_pedals", False)),
        low_speed_brake_suppression=bool(env_cfg_raw.get("low_speed_brake_suppression", False)),
        low_speed_brake_speed_threshold_kmh=float(env_cfg_raw.get("low_speed_brake_speed_threshold_kmh", 0.0)),
        low_speed_brake_max=float(env_cfg_raw.get("low_speed_brake_max", 1.0)),
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
    last_reached_goal = False
    last_collision = False
    last_offroad = False
    last_stuck = False
    last_info: dict[str, Any] | None = None
    episode_speed_sum = 0.0
    episode_speed_count = 0
    completed_episode_rewards: list[float] = []
    completed_episode_progresses: list[float] = []

    try:
        reset_result = env.reset()
        initial_obs = reset_result["obs"]
        episode_initial_info = dict(reset_result["info"])
        last_info = dict(reset_result["info"])
        model_config = build_model_config(initial_obs, config, model_type)
        policy_observation = build_policy_observation(initial_obs, model_type)

        ppo_config = PPOConfig(
            model_type=model_type,
            obs_dim=int(model_config["obs_dim"]),
            action_dim=int(model_config["action_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
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
            model_kwargs=dict(model_config["model_kwargs"]),
        )
        agent = PPOAgent(config=ppo_config)
        buffer = RolloutBuffer(
            capacity=ppo_config.rollout_steps,
            obs_dim=ppo_config.obs_dim,
            action_dim=ppo_config.action_dim,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )
        observation = policy_observation

        while total_steps < total_timesteps:
            buffer.reset()
            for _ in range(ppo_config.rollout_steps):
                action, log_prob, value = agent.act(observation, deterministic=False)
                step_result = env.step(action)

                next_observation = build_policy_observation(step_result["obs"], model_type)
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
                last_reached_goal = bool(info.get("reached_goal", False))
                last_collision = bool(info.get("collision", False))
                last_offroad = bool(info.get("offroad", False))
                last_stuck = bool(info.get("stuck", False))
                last_info = dict(info)
                episode_speed_sum += float(info.get("speed_kmh", 0.0))
                episode_speed_count += 1

                if done:
                    completed_episode_rewards.append(episode_reward)
                    completed_episode_progresses.append(last_route_progress)
                    average_speed_kmh = episode_speed_sum / max(1, episode_speed_count)
                    write_training_log(
                        writer=log_writer,
                        log_file=log_file,
                        model_type=model_type,
                        seed=seed,
                        record_type="episode",
                        update=update_index,
                        total_steps=total_steps,
                        episode_index=episode_index,
                        episode_reward=episode_reward,
                        episode_length=episode_length,
                        done_reason=last_done_reason,
                        route_progress=last_route_progress,
                        reached_goal=last_reached_goal,
                        collision=last_collision,
                        offroad=last_offroad,
                        stuck=last_stuck,
                        average_reward=float(np.mean(completed_episode_rewards)),
                        average_route_progress=float(np.mean(completed_episode_progresses)),
                        average_speed_kmh=average_speed_kmh,
                        initial_info=episode_initial_info,
                        latest_info=last_info,
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
                    episode_speed_sum = 0.0
                    episode_speed_count = 0
                    reset_result = env.reset()
                    episode_initial_info = dict(reset_result["info"])
                    last_info = dict(reset_result["info"])
                    observation = build_policy_observation(reset_result["obs"], model_type)

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
            metrics.update(agent.get_diagnostics())
            update_index += 1

            print(
                f"[update] idx={update_index} total_steps={total_steps} "
                f"actor_loss={metrics['actor_loss']:.4f} critic_loss={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )

            write_training_log(
                writer=log_writer,
                log_file=log_file,
                model_type=model_type,
                seed=seed,
                record_type="update",
                update=update_index,
                total_steps=total_steps,
                episode_index=episode_index,
                episode_reward=episode_reward,
                episode_length=episode_length,
                done_reason=last_done_reason,
                route_progress=last_route_progress,
                reached_goal=last_reached_goal,
                collision=last_collision,
                offroad=last_offroad,
                stuck=last_stuck,
                average_reward=float(np.mean(completed_episode_rewards)) if completed_episode_rewards else 0.0,
                average_route_progress=float(np.mean(completed_episode_progresses)) if completed_episode_progresses else 0.0,
                average_speed_kmh=(episode_speed_sum / max(1, episode_speed_count)),
                initial_info=episode_initial_info,
                latest_info=last_info,
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
