from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "RCS-AC-v1.1" / "code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from PPO.ppo_r7j.models import PPOMatchedR7JPolicy, PPOMatchedR7JPolicyConfig
from rcs_ac_v1_1.adapters.carla_observation_adapter import build_v11_flat_obs_from_carla_step
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config
from rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit import (
    _build_carla_env,
    _control_to_env_action,
    _get_ego_vehicle,
    _resolve_device,
    _step_env,
)


def main() -> None:
    args = parse_args()
    train_from_config(args.config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO matched to the RCS-AC v1.1 r7j runtime contract.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("PPO matched-r7j config must be a mapping")
    runtime_config = load_v11_runtime_config(_cfg_path(cfg_path, cfg["runtime_config"]))
    device = _resolve_device(str(cfg.get("device", "auto")))
    ppo_cfg = dict(cfg.get("ppo", {}))
    model_cfg = dict(cfg.get("model", {}))
    model = PPOMatchedR7JPolicy(
        runtime_config,
        PPOMatchedR7JPolicyConfig(
            feature_dim=int(model_cfg.get("feature_dim", runtime_config.policy_config.feature_dim)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            log_std_init=float(model_cfg.get("log_std_init", -1.5)),
            min_log_std=float(model_cfg.get("min_log_std", -4.0)),
            max_log_std=float(model_cfg.get("max_log_std", 0.0)),
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ppo_cfg.get("learning_rate", 3.0e-5)))

    output_dir = Path(str(cfg.get("output_dir", "PPO/results/ppo_matched_r7j_train")))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_args = SimpleNamespace(
        env_config=str(_cfg_path(cfg_path, cfg["env_config"])),
        output_dir=str(output_dir / "env"),
        run_name=str(cfg.get("run_name", "ppo_matched_r7j_train")),
        host=str(cfg.get("host", "127.0.0.1")),
        port=int(cfg.get("port", 2000)),
        fixed_delta_seconds=cfg.get("fixed_delta_seconds"),
    )
    _, env = _build_carla_env(eval_args)
    controller = runtime_config.build_controller()

    rollout_steps = int(ppo_cfg.get("rollout_steps", 128))
    total_timesteps = int(ppo_cfg.get("total_timesteps", rollout_steps))
    update_epochs = int(ppo_cfg.get("update_epochs", 4))
    minibatch_size = int(ppo_cfg.get("minibatch_size", 64))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_coef = float(ppo_cfg.get("clip_coef", 0.2))
    value_coef = float(ppo_cfg.get("value_loss_weight", 0.5))
    entropy_coef = float(ppo_cfg.get("entropy_loss_weight", 0.0))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    dt = float(cfg.get("dt", 0.05))
    action_format = str(cfg.get("action_format", "direct_3d"))

    log_path = output_dir / "training_log.csv"
    rows: list[dict[str, Any]] = []
    reset = env.reset()
    current_obs = dict(reset.get("obs", {}))
    current_info = dict(reset.get("info", {}))
    previous_control: dict[str, float] | None = None
    global_step = 0
    update_idx = 0
    try:
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "update",
                    "global_step",
                    "mean_reward",
                    "mean_route_progress",
                    "last_route_progress",
                    "collision_count",
                    "offroad_count",
                    "stuck_count",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                ],
            )
            writer.writeheader()
            while global_step < total_timesteps:
                rollout = _collect_rollout(
                    model=model,
                    env=env,
                    runtime_config=runtime_config,
                    controller=controller,
                    current_obs=current_obs,
                    current_info=current_info,
                    previous_control=previous_control,
                    rollout_steps=min(rollout_steps, total_timesteps - global_step),
                    device=device,
                    dt=dt,
                    action_format=action_format,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                current_obs = rollout.pop("current_obs")
                current_info = rollout.pop("current_info")
                previous_control = rollout.pop("previous_control")
                metrics = _ppo_update(
                    model=model,
                    optimizer=optimizer,
                    rollout=rollout,
                    update_epochs=update_epochs,
                    minibatch_size=minibatch_size,
                    clip_coef=clip_coef,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    max_grad_norm=max_grad_norm,
                )
                global_step += int(rollout["rewards"].shape[0])
                row = {
                    "update": update_idx,
                    "global_step": global_step,
                    "mean_reward": float(rollout["rewards"].mean().cpu()),
                    "mean_route_progress": float(np.mean(rollout["route_progress"])),
                    "last_route_progress": float(rollout["route_progress"][-1]) if rollout["route_progress"] else 0.0,
                    "collision_count": int(sum(rollout["collision"])),
                    "offroad_count": int(sum(rollout["offroad"])),
                    "stuck_count": int(sum(rollout["stuck"])),
                    **metrics,
                }
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                _save_checkpoint(output_dir / "last.pt", model, optimizer, cfg, update_idx, global_step)
                update_idx += 1
    finally:
        if hasattr(env, "close"):
            env.close()
    summary = {"updates": update_idx, "global_step": global_step, "last": rows[-1] if rows else {}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _collect_rollout(
    *,
    model: PPOMatchedR7JPolicy,
    env: Any,
    runtime_config: Any,
    controller: Any,
    current_obs: dict[str, Any],
    current_info: dict[str, Any],
    previous_control: dict[str, float] | None,
    rollout_steps: int,
    device: torch.device,
    dt: float,
    action_format: str,
    gamma: float,
    gae_lambda: float,
) -> dict[str, Any]:
    obs_list: list[torch.Tensor] = []
    action_list: list[torch.Tensor] = []
    logprob_list: list[torch.Tensor] = []
    value_list: list[torch.Tensor] = []
    reward_list: list[float] = []
    done_list: list[float] = []
    route_progress: list[float] = []
    collision: list[bool] = []
    offroad: list[bool] = []
    stuck: list[bool] = []
    model.eval()
    for _ in range(int(rollout_steps)):
        obs_flat, _, ego_state, route_context, _ = build_v11_flat_obs_from_carla_step(
            obs=current_obs,
            info=current_info,
            runtime_config=runtime_config,
            previous_control=previous_control,
            ego_vehicle=_get_ego_vehicle(env),
        )
        obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device).reshape(1, -1)
        with torch.no_grad():
            outputs = model.act(obs_t, deterministic=False)
        waypoints = outputs["waypoints"][0].detach().cpu().numpy()
        target_speed = float(outputs["target_speed_kmh"][0, 0].detach().cpu())
        route_fallback = None if route_context is None else route_context.route_points_ego
        control = controller.control(
            waypoints,
            target_speed_kmh=target_speed,
            current_speed_kmh=float(ego_state.speed_kmh),
            dt=float(dt),
            route_fallback_waypoints=route_fallback,
            uncertainty=0.0,
        )
        result = _step_env(env, _control_to_env_action(control, action_format))
        info = dict(result.get("info", {}))
        reward = float(result.get("reward", info.get("reward", 0.0)))
        done = bool(result.get("done", False))
        obs_list.append(obs_t.squeeze(0))
        action_list.append(outputs["raw_action"].squeeze(0).detach())
        logprob_list.append(outputs["log_prob"].squeeze(0).detach())
        value_list.append(outputs["value"].squeeze(0).detach())
        reward_list.append(reward)
        done_list.append(float(done))
        route_progress.append(float(info.get("route_progress", 0.0)))
        collision.append(bool(info.get("collision", False)))
        offroad.append(bool(info.get("offroad", False)))
        stuck.append(bool(info.get("stuck", False)))
        previous_control = {
            "steer": float(control["steer"]),
            "throttle": float(control["throttle"]),
            "brake": float(control["brake"]),
            "longitudinal": float(control["throttle"]) - float(control["brake"]),
        }
        if done:
            reset = env.reset()
            current_obs = dict(reset.get("obs", {}))
            current_info = dict(reset.get("info", {}))
            previous_control = None
        else:
            current_obs = dict(result.get("obs", {}))
            current_info = info
    with torch.no_grad():
        next_obs_flat, _, _, _, _ = build_v11_flat_obs_from_carla_step(
            obs=current_obs,
            info=current_info,
            runtime_config=runtime_config,
            previous_control=previous_control,
            ego_vehicle=_get_ego_vehicle(env),
        )
        next_value = model(
            torch.as_tensor(next_obs_flat, dtype=torch.float32, device=device).reshape(1, -1)
        )["value"].reshape(-1)[0]
    rewards = torch.as_tensor(reward_list, dtype=torch.float32, device=device)
    dones = torch.as_tensor(done_list, dtype=torch.float32, device=device)
    values = torch.stack(value_list).reshape(-1)
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, device=device)
    for idx in reversed(range(int(rewards.shape[0]))):
        next_nonterminal = 1.0 - dones[idx]
        next_val = next_value if idx == int(rewards.shape[0]) - 1 else values[idx + 1]
        delta = rewards[idx] + gamma * next_val * next_nonterminal - values[idx]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[idx] = last_advantage
    returns = advantages + values
    return {
        "obs": torch.stack(obs_list),
        "raw_actions": torch.stack(action_list),
        "old_log_probs": torch.stack(logprob_list).reshape(-1),
        "values": values,
        "rewards": rewards,
        "advantages": advantages,
        "returns": returns,
        "route_progress": route_progress,
        "collision": collision,
        "offroad": offroad,
        "stuck": stuck,
        "current_obs": current_obs,
        "current_info": current_info,
        "previous_control": previous_control,
    }


def _ppo_update(
    *,
    model: PPOMatchedR7JPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    obs = rollout["obs"]
    raw_actions = rollout["raw_actions"]
    old_log_probs = rollout["old_log_probs"]
    returns = rollout["returns"]
    advantages = rollout["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
    count = int(obs.shape[0])
    last_policy_loss = torch.tensor(0.0, device=obs.device)
    last_value_loss = torch.tensor(0.0, device=obs.device)
    last_entropy = torch.tensor(0.0, device=obs.device)
    last_kl = torch.tensor(0.0, device=obs.device)
    for _ in range(int(update_epochs)):
        permutation = torch.randperm(count, device=obs.device)
        for start in range(0, count, int(minibatch_size)):
            idx = permutation[start : start + int(minibatch_size)]
            outputs = model.evaluate_raw_action(obs[idx], raw_actions[idx])
            new_log_probs = outputs["log_prob"].reshape(-1)
            entropy = outputs["entropy"].reshape(-1).mean()
            values = outputs["value"].reshape(-1)
            log_ratio = new_log_probs - old_log_probs[idx]
            ratio = torch.exp(log_ratio)
            unclipped = -advantages[idx] * ratio
            clipped = -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(values - returns[idx]).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            last_policy_loss = policy_loss.detach()
            last_value_loss = value_loss.detach()
            last_entropy = entropy.detach()
            last_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    return {
        "policy_loss": float(last_policy_loss.cpu()),
        "value_loss": float(last_value_loss.cpu()),
        "entropy": float(last_entropy.cpu()),
        "approx_kl": float(last_kl.cpu()),
    }


def _save_checkpoint(
    path: Path,
    model: PPOMatchedR7JPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    update_idx: int,
    global_step: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "update": int(update_idx),
            "global_step": int(global_step),
            "baseline_family": "PPO matched-r7j",
        },
        path,
    )


def _cfg_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (config_path.parent / path).resolve()
    return candidate if candidate.exists() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
