from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from models.single_modal_actor_critic import ModelConfig, SingleModalActorCritic
from utils.common import clamp


@dataclass
class PPOConfig:
    obs_dim: int = 13
    action_dim: int = 3
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 256
    device: str = "cpu"


class PPOAgent:
    def __init__(self, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        self.model = SingleModalActorCritic(
            config=ModelConfig(
                obs_dim=self.config.obs_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.hidden_dim,
            )
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _build_distribution(self, observations: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        action_mean, value = self.model(observations)
        std = self.model.get_std().expand_as(action_mean)
        return Normal(action_mean, std), value

    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution, value = self._build_distribution(observation_tensor)
            if deterministic:
                raw_action = distribution.mean
            else:
                raw_action = distribution.sample()
            bounded_action = raw_action.clone()
            bounded_action[..., 0:1] = bounded_action[..., 0:1].clamp(-1.0, 1.0)
            bounded_action[..., 1:2] = bounded_action[..., 1:2].clamp(0.0, 1.0)
            bounded_action[..., 2:3] = bounded_action[..., 2:3].clamp(0.0, 1.0)
            log_prob = distribution.log_prob(bounded_action).sum(dim=-1)

        action = bounded_action.squeeze(0).cpu().numpy().astype(np.float32)
        action[0] = np.float32(clamp(float(action[0]), -1.0, 1.0))
        action[1] = np.float32(clamp(float(action[1]), 0.0, 1.0))
        action[2] = np.float32(clamp(float(action[2]), 0.0, 1.0))
        return action, float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._build_distribution(observations)
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy, values

    def update(self, rollout_batch: Any) -> dict[str, float]:
        batch_size = rollout_batch.observations.shape[0]
        indices = np.arange(batch_size)

        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0
        update_steps = 0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_indices = indices[start:end]

                obs_batch = rollout_batch.observations[batch_indices]
                action_batch = rollout_batch.actions[batch_indices]
                old_log_prob_batch = rollout_batch.log_probs[batch_indices]
                advantage_batch = rollout_batch.advantages[batch_indices]
                return_batch = rollout_batch.returns[batch_indices]

                new_log_probs, entropy, values = self.evaluate_actions(obs_batch, action_batch)
                log_ratio = new_log_probs - old_log_prob_batch
                ratio = log_ratio.exp()

                unclipped = ratio * advantage_batch
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * advantage_batch
                actor_loss = -torch.min(unclipped, clipped).mean()

                critic_loss = 0.5 * (return_batch - values).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = actor_loss + self.config.value_coef * critic_loss - self.config.entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                actor_loss_total += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())
                entropy_total += float(entropy_bonus.item())
                update_steps += 1

        if update_steps == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        return {
            "actor_loss": actor_loss_total / update_steps,
            "critic_loss": critic_loss_total / update_steps,
            "entropy": entropy_total / update_steps,
        }

    def save(self, checkpoint_path: str, extra_state: dict[str, Any]) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "extra_state": extra_state,
            },
            checkpoint_path,
        )
