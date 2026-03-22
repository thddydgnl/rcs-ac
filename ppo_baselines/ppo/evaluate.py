"""Evaluation helpers for PPO baselines."""

from __future__ import annotations

import numpy as np
import torch

from ppo_baselines.models.baselines import BaselineCAttentionFusionActorCritic, GaussianActorCriticBase


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in obs.items()
    }


@torch.no_grad()
def evaluate_policy(
    env,
    model: GaussianActorCriticBase,
    device: torch.device,
    num_episodes: int,
) -> dict[str, float]:
    model.eval()
    returns = []
    successes = []
    collisions = []
    timeouts = []
    progress_values = []
    attention_success: list[np.ndarray] = []
    attention_failure: list[np.ndarray] = []
    attention_all: list[np.ndarray] = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=10000 + episode_idx)
        episode_return = 0.0
        episode_progress = []
        episode_attention = []

        while True:
            obs_tensor = obs_to_tensor(obs, device)
            output = model.act(obs_tensor, deterministic=True)
            action = output.action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_progress.append(float(info.get("progress_delta", 0.0)))

            attention_weights = output.aux_outputs.get("attention_weights")
            if attention_weights is not None:
                episode_attention.append(attention_weights.squeeze(0).cpu().numpy())

            if terminated or truncated:
                returns.append(episode_return)
                successes.append(float(info["goal_reached"]))
                collisions.append(float(info["collision"]))
                timeouts.append(float(info["timeout"]))
                progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                if episode_attention:
                    stacked = np.stack(episode_attention, axis=0).mean(axis=0)
                    attention_all.append(stacked)
                    if info["goal_reached"]:
                        attention_success.append(stacked)
                    else:
                        attention_failure.append(stacked)
                break

    metrics = {
        "eval/return_mean": float(np.mean(returns)),
        "eval/success_rate": float(np.mean(successes)),
        "eval/collision_rate": float(np.mean(collisions)),
        "eval/timeout_rate": float(np.mean(timeouts)),
        "eval/avg_progress_to_goal": float(np.mean(progress_values)),
    }

    if isinstance(model, BaselineCAttentionFusionActorCritic):
        ordered_modalities = model.modality_names
        if attention_all:
            all_mean = np.mean(np.stack(attention_all, axis=0), axis=0)
            all_std = np.std(np.stack(attention_all, axis=0), axis=0)
            for name, value in zip(ordered_modalities, all_mean):
                metrics[f"eval_attention_mean/{name}"] = float(value)
            for name, value in zip(ordered_modalities, all_std):
                metrics[f"eval_attention_std/{name}"] = float(value)
        if attention_success:
            success_mean = np.mean(np.stack(attention_success, axis=0), axis=0)
            for name, value in zip(ordered_modalities, success_mean):
                metrics[f"eval_attention_success/{name}"] = float(value)
        if attention_failure:
            failure_mean = np.mean(np.stack(attention_failure, axis=0), axis=0)
            for name, value in zip(ordered_modalities, failure_mean):
                metrics[f"eval_attention_failure/{name}"] = float(value)

    return metrics
