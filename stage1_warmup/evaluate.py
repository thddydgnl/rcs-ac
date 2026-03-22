"""Evaluation helpers for Stage 1 warm-up."""

from __future__ import annotations

import numpy as np
import torch

from rcs_ac_v1.model import MODALITY_NAMES


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


def aux_to_reliability_targets(aux_obs: np.ndarray) -> np.ndarray:
    """Map aux state to weak reliability labels in [0, 1]."""
    aux = np.asarray(aux_obs, dtype=np.float32)
    vision = np.clip(aux[..., 0], 0.0, 1.0)
    distance = np.clip(aux[..., 1], 0.0, 1.0)
    ego = np.clip(1.0 - aux[..., 2], 0.0, 1.0)
    route = np.clip(1.0 - aux[..., 3], 0.0, 1.0)
    return np.stack([vision, distance, ego, route], axis=-1).astype(np.float32)


def _safe_corr(attention_stack: np.ndarray, reliability_stack: np.ndarray) -> float:
    if attention_stack.size == 0 or reliability_stack.size == 0:
        return 0.0
    x = attention_stack.reshape(-1)
    y = reliability_stack.reshape(-1)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def evaluate_stage1_policy(
    env,
    model,
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
    reliability_values = []
    reliability_target_values = []
    success_attention = []
    failure_attention = []
    success_reliability = []
    failure_reliability = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=30000 + episode_idx)
        del info
        hidden = model.initial_hidden(batch_size=1, device=device)
        episode_start = torch.ones(1, device=device)
        episode_return = 0.0
        episode_progress = []
        episode_attention = []
        episode_reliability = []

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
            reliability = output.aux_outputs["reliability_scores"].squeeze(0).cpu().numpy()
            targets = aux_to_reliability_targets(obs["aux"])
            attention_values.append(attention)
            reliability_values.append(reliability)
            reliability_target_values.append(targets)
            episode_attention.append(attention)
            episode_reliability.append(reliability)
            episode_start = torch.tensor([float(terminated or truncated)], device=device)

            if terminated or truncated:
                returns.append(episode_return)
                successes.append(float(info["goal_reached"]))
                collisions.append(float(info["collision"]))
                timeouts.append(float(info["timeout"]))
                progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                if episode_attention:
                    mean_episode_attention = np.mean(np.stack(episode_attention, axis=0), axis=0)
                    mean_episode_reliability = np.mean(np.stack(episode_reliability, axis=0), axis=0)
                    if info["goal_reached"]:
                        success_attention.append(mean_episode_attention)
                        success_reliability.append(mean_episode_reliability)
                    else:
                        failure_attention.append(mean_episode_attention)
                        failure_reliability.append(mean_episode_reliability)
                break

    metrics = {
        "eval/return_mean": float(np.mean(returns)),
        "eval/success_rate": float(np.mean(successes)),
        "eval/collision_rate": float(np.mean(collisions)),
        "eval/timeout_rate": float(np.mean(timeouts)),
        "eval/avg_progress_to_goal": float(np.mean(progress_values)),
        "eval/hidden_norm_mean": float(np.mean(hidden_norm_values)) if hidden_norm_values else 0.0,
        "eval/hidden_norm_std": float(np.std(hidden_norm_values)) if hidden_norm_values else 0.0,
        "eval/reliability_nan_detected": float(
            0.0 if not reliability_values else float(not np.all(np.isfinite(np.stack(reliability_values, axis=0))))
        ),
        "eval/quality_label_if_available": 1.0,
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

    if reliability_values:
        reliability_stack = np.stack(reliability_values, axis=0)
        rel_mean = reliability_stack.mean(axis=0)
        rel_std = reliability_stack.std(axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_reliability_mean/{name}"] = float(rel_mean[idx])
            metrics[f"eval_reliability_std/{name}"] = float(rel_std[idx])

    if attention_values and reliability_values:
        metrics["eval_attention_reliability/correlation"] = _safe_corr(
            np.stack(attention_values, axis=0),
            np.stack(reliability_values, axis=0),
        )

    if success_attention:
        success_stack = np.mean(np.stack(success_attention, axis=0), axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_attention_success/{name}"] = float(success_stack[idx])
    if failure_attention:
        failure_stack = np.mean(np.stack(failure_attention, axis=0), axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_attention_failure/{name}"] = float(failure_stack[idx])
    if success_reliability:
        success_rel_stack = np.mean(np.stack(success_reliability, axis=0), axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_reliability_success/{name}"] = float(success_rel_stack[idx])
    if failure_reliability:
        failure_rel_stack = np.mean(np.stack(failure_reliability, axis=0), axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"eval_reliability_failure/{name}"] = float(failure_rel_stack[idx])

    if reliability_values and reliability_target_values:
        reliability_stack = np.stack(reliability_values, axis=0)
        target_stack = np.stack(reliability_target_values, axis=0)
        metrics["eval/reliability_mse"] = float(np.mean((reliability_stack - target_stack) ** 2))

    return metrics
