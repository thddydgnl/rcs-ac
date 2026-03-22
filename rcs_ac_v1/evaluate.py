"""Evaluation for recurrent RCS-AC v1."""

from __future__ import annotations

import numpy as np
import torch

from rcs_ac_v1.model import MODALITY_NAMES, RecurrentAttentionActorCritic


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


@torch.no_grad()
def evaluate_recurrent_policy(
    env,
    model: RecurrentAttentionActorCritic,
    device: torch.device,
    num_episodes: int,
) -> dict[str, float]:
    model.eval()
    returns = []
    successes = []
    collisions = []
    timeouts = []
    progress_values = []
    hidden_norm_values = []
    attention_values = []
    success_attention = []
    failure_attention = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=20000 + episode_idx)
        del info
        hidden = model.initial_hidden(batch_size=1, device=device)
        episode_start = torch.ones(1, device=device)
        episode_return = 0.0
        episode_progress = []
        episode_attention = []

        while True:
            obs_tensor = obs_to_tensor(obs, device)
            output = model.step(
                obs_dict=obs_tensor,
                hidden_state=hidden,
                episode_start=episode_start,
                deterministic=True,
            )
            hidden = output.hidden_state
            action = output.action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_progress.append(float(info.get("progress_delta", 0.0)))
            hidden_norm_values.append(float(output.aux_outputs["hidden_norm"].mean().cpu().item()))
            attention = output.aux_outputs["attention_weights"].squeeze(0).cpu().numpy()
            attention_values.append(attention)
            episode_attention.append(attention)
            episode_start = torch.tensor([float(terminated or truncated)], device=device)

            if terminated or truncated:
                returns.append(episode_return)
                successes.append(float(info["goal_reached"]))
                collisions.append(float(info["collision"]))
                timeouts.append(float(info["timeout"]))
                progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                if episode_attention:
                    mean_episode_attention = np.mean(np.stack(episode_attention, axis=0), axis=0)
                    if info["goal_reached"]:
                        success_attention.append(mean_episode_attention)
                    else:
                        failure_attention.append(mean_episode_attention)
                break

    metrics = {
        "eval/return_mean": float(np.mean(returns)),
        "eval/success_rate": float(np.mean(successes)),
        "eval/collision_rate": float(np.mean(collisions)),
        "eval/timeout_rate": float(np.mean(timeouts)),
        "eval/avg_progress_to_goal": float(np.mean(progress_values)),
        "eval/hidden_norm_mean": float(np.mean(hidden_norm_values)) if hidden_norm_values else 0.0,
        "eval/hidden_norm_std": float(np.std(hidden_norm_values)) if hidden_norm_values else 0.0,
    }

    if attention_values:
        attention_stack = np.stack(attention_values, axis=0)
        att_mean = attention_stack.mean(axis=0)
        att_std = attention_stack.std(axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_attention_mean/{name}"] = float(att_mean[idx])
            metrics[f"eval_attention_std/{name}"] = float(att_std[idx])
        route_idx = MODALITY_NAMES.index("route")
        other_mean = np.mean(np.delete(att_mean, route_idx))
        metrics["eval_attention/route_dominance_ratio"] = float(att_mean[route_idx] / max(other_mean, 1e-6))
    if success_attention:
        success_stack = np.mean(np.stack(success_attention, axis=0), axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_attention_success/{name}"] = float(success_stack[idx])
    if failure_attention:
        failure_stack = np.mean(np.stack(failure_attention, axis=0), axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_attention_failure/{name}"] = float(failure_stack[idx])
    return metrics
