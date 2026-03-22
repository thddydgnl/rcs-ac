"""Shared PPO trainer for baseline A/B/C."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ppo_baselines.config import ExperimentConfig
from ppo_baselines.models.baselines import build_policy_model
from ppo_baselines.ppo.buffer import DictRolloutBuffer
from ppo_baselines.ppo.evaluate import evaluate_policy, obs_to_tensor
from ppo_baselines.utils.logger import ExperimentLogger
from rcs_toy_nav_env import RCSToyNavEnv


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(config: ExperimentConfig) -> RCSToyNavEnv:
    env = RCSToyNavEnv()
    env.set_corruption_mode(config.env.corruption_mode)
    return env


class PPOTrainer:
    """Single shared PPO loop for all baseline variants."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)

        self.train_env = make_env(config)
        self.eval_env = make_env(config)
        self.obs_keys = ["vision", "distance", "ego", "route", "aux"]
        self.obs_shapes = {
            key: self.train_env.observation_space[key].shape
            for key in self.obs_keys
        }

        self.model = build_policy_model(
            baseline_type=config.model.baseline_type,
            action_dim=int(self.train_env.action_space.shape[0]),
            encoder_hidden_dims=config.model.encoder_hidden_dims,
            encoder_output_dim=config.model.encoder_output_dim,
            trunk_hidden_dims=config.model.trunk_hidden_dims,
            actor_hidden_dims=config.model.actor_hidden_dims,
            critic_hidden_dims=config.model.critic_hidden_dims,
            action_log_std_init=config.model.action_log_std_init,
            attention_hidden_dim=config.model.attention_hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.ppo.learning_rate,
        )
        self.buffer = DictRolloutBuffer(
            rollout_steps=config.ppo.rollout_steps,
            obs_keys=self.obs_keys,
            obs_shapes=self.obs_shapes,
            action_dim=int(self.train_env.action_space.shape[0]),
        )
        self.logger = ExperimentLogger(config.logging.output_dir)
        self.logger.save_config(asdict(config))
        self.nan_detected = False

    @staticmethod
    def _has_non_finite(metrics: dict[str, float]) -> bool:
        for value in metrics.values():
            if isinstance(value, (float, int)) and not np.isfinite(value):
                return True
        return False

    def save_checkpoint(self, update_idx: int) -> None:
        checkpoint_path = Path(self.config.logging.output_dir) / f"checkpoint_{update_idx:05d}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "update_idx": update_idx,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def collect_rollout(self, global_step: int) -> dict[str, float]:
        obs, info = self.train_env.reset(seed=self.config.seed + global_step)
        del info
        self.buffer.reset()

        episode_returns = []
        episode_lengths = []
        success_flags = []
        collision_flags = []
        timeout_flags = []
        progress_values = []
        attention_weights = []

        current_return = 0.0
        current_length = 0
        current_progress = []

        for _ in range(self.config.ppo.rollout_steps):
            obs_tensor = obs_to_tensor(obs, self.device)
            with torch.no_grad():
                output = self.model.act(obs_tensor, deterministic=False)

            action = output.action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            self.buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                done=done,
                value=float(output.value.squeeze(0).cpu().item()),
                log_prob=float(output.log_prob.squeeze(0).cpu().item()),
            )

            current_return += float(reward)
            current_length += 1
            current_progress.append(float(info.get("progress_delta", 0.0)))

            if output.aux_outputs.get("attention_weights") is not None:
                attention_weights.append(output.aux_outputs["attention_weights"].squeeze(0).cpu().numpy())

            obs = next_obs

            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                success_flags.append(float(info["goal_reached"]))
                collision_flags.append(float(info["collision"]))
                timeout_flags.append(float(info["timeout"]))
                progress_values.append(float(np.mean(current_progress)) if current_progress else 0.0)
                current_return = 0.0
                current_length = 0
                current_progress = []
                obs, info = self.train_env.reset()
                del info

        with torch.no_grad():
            obs_tensor = obs_to_tensor(obs, self.device)
            last_value = float(self.model.act(obs_tensor, deterministic=False).value.squeeze(0).cpu().item())
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=False,
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
        )

        metrics = {
            "train/episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "train/episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "train/success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            "train/collision_rate": float(np.mean(collision_flags)) if collision_flags else 0.0,
            "train/timeout_rate": float(np.mean(timeout_flags)) if timeout_flags else 0.0,
            "train/avg_progress_to_goal": float(np.mean(progress_values)) if progress_values else 0.0,
        }

        if attention_weights:
            stacked = np.stack(attention_weights, axis=0)
            modality_names = self.model.modality_names
            mean_weights = stacked.mean(axis=0)
            std_weights = stacked.std(axis=0)
            for idx, name in enumerate(modality_names):
                metrics[f"train_attention/mean_{name}"] = float(mean_weights[idx])
                metrics[f"train_attention/std_{name}"] = float(std_weights[idx])
            metrics["train_attention/weight_std"] = float(np.mean(std_weights))

        return metrics

    def update_model(self) -> dict[str, float]:
        update_metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
        }
        num_updates = 0

        for _ in range(self.config.ppo.update_epochs):
            for batch in self.buffer.get_minibatches(self.config.ppo.minibatch_size, self.device):
                output = self.model.evaluate_actions(batch.observations, batch.actions)
                ratio = torch.exp(output.log_prob - batch.log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_eps,
                    1.0 + self.config.ppo.clip_eps,
                )
                policy_loss = -torch.min(
                    ratio * batch.advantages,
                    clipped_ratio * batch.advantages,
                ).mean()
                value_loss = 0.5 * torch.mean((output.value - batch.returns) ** 2)
                entropy_loss = output.entropy.mean()

                loss = (
                    policy_loss
                    + self.config.ppo.value_loss_coef * value_loss
                    - self.config.ppo.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.ppo.max_grad_norm)
                self.optimizer.step()

                update_metrics["loss/actor"] += float(policy_loss.item())
                update_metrics["loss/critic"] += float(value_loss.item())
                update_metrics["loss/entropy"] += float(entropy_loss.item())
                num_updates += 1

        if num_updates > 0:
            for key in update_metrics:
                update_metrics[key] /= num_updates
        return update_metrics

    def train(self) -> None:
        num_updates = self.config.ppo.total_steps // self.config.ppo.rollout_steps
        start_time = time.time()
        for update_idx in range(1, num_updates + 1):
            rollout_metrics = self.collect_rollout(global_step=update_idx * self.config.ppo.rollout_steps)
            loss_metrics = self.update_model()
            self.nan_detected = self.nan_detected or self._has_non_finite(rollout_metrics) or self._has_non_finite(loss_metrics)

            log_payload = {
                "update": update_idx,
                "train_step": update_idx * self.config.ppo.rollout_steps,
                "wall_clock_time": float(time.time() - start_time),
                "finished_flag": 0,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
            }
            log_payload.update(rollout_metrics)
            log_payload.update(loss_metrics)
            self.logger.log_train(log_payload)

            if update_idx % self.config.logging.eval_interval_updates == 0:
                eval_metrics = evaluate_policy(
                    env=self.eval_env,
                    model=self.model,
                    device=self.device,
                    num_episodes=self.config.ppo.num_eval_episodes,
                )
                self.nan_detected = self.nan_detected or self._has_non_finite(eval_metrics)
                eval_payload = {
                    "update": update_idx,
                    "train_step": update_idx * self.config.ppo.rollout_steps,
                    "wall_clock_time": float(time.time() - start_time),
                    "finished_flag": 0,
                    "nan_detected": int(self.nan_detected),
                    "crashed_flag": 0,
                }
                eval_payload.update(eval_metrics)
                self.logger.log_eval(eval_payload)
                log_payload.update(eval_metrics)

            self.logger.log(log_payload)

            if update_idx % self.config.logging.save_interval_updates == 0:
                self.save_checkpoint(update_idx)

        final_status = {
            "finished_flag": 1,
            "nan_detected": int(self.nan_detected),
            "crashed_flag": 0,
            "wall_clock_time": float(time.time() - start_time),
            "total_updates": num_updates,
        }
        self.logger.save_status(final_status)


def train_from_config(config: ExperimentConfig) -> PPOTrainer:
    trainer = PPOTrainer(config)
    trainer.train()
    return trainer
