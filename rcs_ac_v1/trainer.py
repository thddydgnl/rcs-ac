"""Recurrent PPO trainer for RCS-AC v1."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ppo_baselines.models.baselines import BaselineCAttentionFusionActorCritic
from ppo_baselines.utils.logger import ExperimentLogger
from rcs_ac_v1.config import ExperimentConfig
from rcs_ac_v1.evaluate import evaluate_recurrent_policy, obs_to_tensor
from rcs_ac_v1.model import MODALITY_NAMES, RecurrentAttentionActorCritic
from rcs_ac_v1.recurrent_buffer import RecurrentRolloutBuffer
from rcs_toy_nav_env import RCSToyNavEnv


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(config: ExperimentConfig) -> RCSToyNavEnv:
    env = RCSToyNavEnv()
    env.set_corruption_mode(config.env.corruption_mode)
    return env


class RecurrentPPOTrainer:
    """Sequence-aware PPO with shared recurrent attention trunk."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        set_seed(config.seed)

        self.train_env = make_env(config)
        self.eval_env = make_env(config)
        self.obs_shapes = {
            key: self.train_env.observation_space[key].shape
            for key in MODALITY_NAMES
        }

        self.model = RecurrentAttentionActorCritic(
            encoder_hidden_dims=config.model.encoder_hidden_dims,
            encoder_output_dim=config.model.encoder_output_dim,
            attention_hidden_dim=config.model.attention_hidden_dim,
            recurrent_hidden_dim=config.model.recurrent_hidden_dim,
            attention_temperature=config.model.attention_temperature,
            use_feature_layer_norm=config.model.use_feature_layer_norm,
            actor_hidden_dims=config.model.actor_hidden_dims,
            critic_hidden_dims=config.model.critic_hidden_dims,
            action_dim=int(self.train_env.action_space.shape[0]),
            action_log_std_init=config.model.action_log_std_init,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.ppo.learning_rate)
        self.buffer = RecurrentRolloutBuffer(
            rollout_steps=config.ppo.rollout_steps,
            sequence_length=config.ppo.sequence_length,
            obs_shapes=self.obs_shapes,
            action_dim=int(self.train_env.action_space.shape[0]),
            hidden_dim=config.model.recurrent_hidden_dim,
            obs_keys=list(MODALITY_NAMES),
        )
        self.logger = ExperimentLogger(config.logging.output_dir)
        self.logger.save_config(asdict(config))
        self.nan_detected = False
        self.hidden_validation = {
            "hidden_zero_init_ok": True,
            "hidden_reset_ok": True,
            "hidden_leakage_detected": False,
            "hidden_shape_mismatch_detected": False,
            "hidden_nan_detected": False,
            "hidden_reset_count": 0,
        }
        if config.model.warm_start_baseline_c_checkpoint:
            self._warm_start_from_baseline_c(config.model.warm_start_baseline_c_checkpoint)

    def _warm_start_from_baseline_c(self, checkpoint_path: str) -> None:
        """Optional warm-start from baseline C encoders and attention score net.

        This keeps the recurrent core new while initializing shared multimodal
        front-end weights from the feedforward attention baseline.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        feedforward = BaselineCAttentionFusionActorCritic(
            encoder_hidden_dims=self.config.model.encoder_hidden_dims,
            encoder_output_dim=self.config.model.encoder_output_dim,
            trunk_hidden_dims=[128, 128],
            actor_hidden_dims=[64],
            critic_hidden_dims=[64],
            action_dim=int(self.train_env.action_space.shape[0]),
            action_log_std_init=self.config.model.action_log_std_init,
            attention_hidden_dim=self.config.model.attention_hidden_dim,
        ).to(self.device)
        feedforward.load_state_dict(checkpoint["model_state_dict"])
        for name in self.model.modality_names:
            self.model.encoders[name].load_state_dict(feedforward.encoders[name].state_dict())
        self.model.fusion_module.score_net.load_state_dict(feedforward.fusion_module.score_net.state_dict())

    @staticmethod
    def _metrics_have_nan(metrics: dict[str, float]) -> bool:
        for value in metrics.values():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
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
        hidden = self.model.initial_hidden(batch_size=1, device=self.device)
        if not torch.allclose(hidden, torch.zeros_like(hidden)):
            self.hidden_validation["hidden_zero_init_ok"] = False
        episode_start_flag = torch.ones(1, device=self.device)

        episode_returns = []
        episode_lengths = []
        success_flags = []
        collision_flags = []
        timeout_flags = []
        progress_values = []
        attention_values = []
        feature_norm_values = []
        hidden_norm_values = []

        current_return = 0.0
        current_length = 0
        current_progress = []

        for _ in range(self.config.ppo.rollout_steps):
            if bool(episode_start_flag.item()) and not torch.allclose(hidden, torch.zeros_like(hidden), atol=1e-6):
                self.hidden_validation["hidden_leakage_detected"] = True
            if not torch.all(torch.isfinite(hidden)):
                self.hidden_validation["hidden_nan_detected"] = True

            obs_tensor = obs_to_tensor(obs, self.device)
            hidden_before = hidden.clone()
            try:
                with torch.no_grad():
                    output = self.model.step(
                        obs_dict=obs_tensor,
                        hidden_state=hidden,
                        episode_start=episode_start_flag,
                        deterministic=False,
                    )
            except ValueError:
                self.hidden_validation["hidden_shape_mismatch_detected"] = True
                raise

            action = output.action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            self.buffer.add(
                obs={key: obs[key] for key in MODALITY_NAMES},
                action=action,
                reward=float(reward),
                done=done,
                episode_start=bool(episode_start_flag.item()),
                value=float(output.value.squeeze(0).cpu().item()),
                log_prob=float(output.log_prob.squeeze(0).cpu().item()),
                hidden_state=hidden_before.squeeze(0).cpu().numpy(),
            )

            hidden = output.hidden_state
            hidden_norm_values.append(float(output.aux_outputs["hidden_norm"].mean().cpu().item()))
            attention_values.append(output.aux_outputs["attention_weights"].squeeze(0).cpu().numpy())
            feature_norm_values.append(output.aux_outputs["encoder_feature_norms"].squeeze(0).cpu().numpy())

            current_return += float(reward)
            current_length += 1
            current_progress.append(float(info.get("progress_delta", 0.0)))
            obs = next_obs
            episode_start_flag = torch.tensor([float(done)], device=self.device)
            hidden = self.model.reset_hidden(hidden, episode_start_flag)
            if done:
                self.hidden_validation["hidden_reset_count"] += 1
                if not torch.allclose(hidden, torch.zeros_like(hidden), atol=1e-6):
                    self.hidden_validation["hidden_reset_ok"] = False

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
            last_output = self.model.step(
                obs_dict=obs_tensor,
                hidden_state=hidden,
                episode_start=episode_start_flag,
                deterministic=False,
            )
        self.buffer.compute_returns_and_advantages(
            last_value=float(last_output.value.squeeze(0).cpu().item()),
            last_done=bool(episode_start_flag.item()),
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
            "train/hidden_norm_mean": float(np.mean(hidden_norm_values)) if hidden_norm_values else 0.0,
            "train/hidden_norm_std": float(np.std(hidden_norm_values)) if hidden_norm_values else 0.0,
            "train/hidden_reset_count": float(self.hidden_validation["hidden_reset_count"]),
            "train/sequence_length": float(self.config.ppo.sequence_length),
            "train/masked_timestep_ratio": float(self.buffer.masked_ratio()),
            "train/hidden_zero_init_ok": float(self.hidden_validation["hidden_zero_init_ok"]),
            "train/hidden_reset_ok": float(self.hidden_validation["hidden_reset_ok"]),
            "train/hidden_leakage_detected": float(self.hidden_validation["hidden_leakage_detected"]),
            "train/hidden_shape_mismatch_detected": float(self.hidden_validation["hidden_shape_mismatch_detected"]),
            "train/hidden_nan_detected": float(self.hidden_validation["hidden_nan_detected"]),
        }
        if attention_values:
            stacked = np.stack(attention_values, axis=0)
            mean_weights = stacked.mean(axis=0)
            std_weights = stacked.std(axis=0)
            for idx, name in enumerate(MODALITY_NAMES):
                metrics[f"train_attention/mean_{name}"] = float(mean_weights[idx])
                metrics[f"train_attention/std_{name}"] = float(std_weights[idx])
            route_idx = MODALITY_NAMES.index("route")
            other_mean = np.mean(np.delete(mean_weights, route_idx))
            metrics["train_attention/route_dominance_ratio"] = float(
                mean_weights[route_idx] / max(other_mean, 1e-6)
            )
        if feature_norm_values:
            feature_norm_stack = np.stack(feature_norm_values, axis=0)
            for idx, name in enumerate(MODALITY_NAMES):
                metrics[f"train_feature_norm/mean_{name}"] = float(feature_norm_stack[:, idx].mean())
                metrics[f"train_feature_norm/std_{name}"] = float(feature_norm_stack[:, idx].std())
        return metrics

    def update_model(self) -> dict[str, float]:
        metrics = {
            "loss/actor": 0.0,
            "loss/critic": 0.0,
            "loss/entropy": 0.0,
            "loss/grad_norm": 0.0,
            "loss/hidden_norm_mean": 0.0,
            "loss/hidden_norm_std": 0.0,
        }
        num_updates = 0

        for _ in range(self.config.ppo.update_epochs):
            for batch in self.buffer.get_sequence_minibatches(
                minibatch_num_sequences=self.config.ppo.minibatch_num_sequences,
                device=self.device,
            ):
                outputs = self.model.forward_sequence(
                    obs_seq=batch.observations,
                    episode_starts=batch.episode_starts,
                    initial_hidden=batch.initial_hidden,
                    actions=batch.actions,
                    deterministic=False,
                )
                valid_mask = batch.valid_mask
                log_probs = outputs["log_probs"]
                values = outputs["values"]
                entropies = outputs["entropies"]
                hidden_norms = outputs["hidden_norms"]

                ratio = torch.exp(log_probs - batch.old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_eps,
                    1.0 + self.config.ppo.clip_eps,
                )

                valid_count = torch.clamp(valid_mask.sum(), min=1.0)
                actor_loss = -(
                    torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages) * valid_mask
                ).sum() / valid_count
                critic_loss = 0.5 * (((values - batch.returns) ** 2) * valid_mask).sum() / valid_count
                entropy = (entropies * valid_mask).sum() / valid_count

                loss = (
                    actor_loss
                    + self.config.ppo.value_loss_coef * critic_loss
                    - self.config.ppo.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.ppo.max_grad_norm).item()
                )
                self.optimizer.step()

                metrics["loss/actor"] += float(actor_loss.item())
                metrics["loss/critic"] += float(critic_loss.item())
                metrics["loss/entropy"] += float(entropy.item())
                metrics["loss/grad_norm"] += grad_norm
                metrics["loss/hidden_norm_mean"] += float((hidden_norms * valid_mask).sum().item() / valid_count.item())
                centered = (hidden_norms - hidden_norms.mean()) ** 2
                metrics["loss/hidden_norm_std"] += float(torch.sqrt((centered * valid_mask).sum() / valid_count).item())
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates
        return metrics

    def train(self) -> None:
        num_updates = self.config.ppo.total_steps // self.config.ppo.rollout_steps
        start_time = time.time()
        for update_idx in range(1, num_updates + 1):
            rollout_metrics = self.collect_rollout(global_step=update_idx * self.config.ppo.rollout_steps)
            loss_metrics = self.update_model()
            self.nan_detected = self.nan_detected or self._metrics_have_nan(rollout_metrics) or self._metrics_have_nan(loss_metrics)

            train_payload = {
                "update": update_idx,
                "train_step": update_idx * self.config.ppo.rollout_steps,
                "wall_clock_time": float(time.time() - start_time),
                "finished_flag": 0,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
            }
            train_payload.update(rollout_metrics)
            train_payload.update(loss_metrics)
            self.logger.log_train(train_payload)
            self.logger.log(train_payload)

            if update_idx % self.config.logging.eval_interval_updates == 0:
                eval_metrics = evaluate_recurrent_policy(
                    env=self.eval_env,
                    model=self.model,
                    device=self.device,
                    num_episodes=self.config.ppo.num_eval_episodes,
                )
                self.nan_detected = self.nan_detected or self._metrics_have_nan(eval_metrics)
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
                self.logger.log(eval_payload)

            if update_idx % self.config.logging.save_interval_updates == 0:
                self.save_checkpoint(update_idx)

        self.logger.save_status(
            {
                "finished_flag": 1,
                "nan_detected": int(self.nan_detected),
                "crashed_flag": 0,
                "wall_clock_time": float(time.time() - start_time),
                "total_updates": num_updates,
                "hidden_validation": self.hidden_validation,
            }
        )
