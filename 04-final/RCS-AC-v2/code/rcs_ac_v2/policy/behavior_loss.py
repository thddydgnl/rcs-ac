"""Dreamer-style actor-critic losses for Stage 2 local smoke."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.distributions import Categorical


@dataclass(frozen=True)
class BehaviorLossConfig:
    gamma: float = 0.99
    lambda_: float = 0.95
    critic_weight: float = 1.0
    actor_weight: float = 1.0
    entropy_weight: float = 1e-3
    normalize_advantage: bool = True
    advantage_epsilon: float = 1e-6


@dataclass
class BehaviorLossBreakdown:
    total: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    entropy: torch.Tensor
    lambda_return: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "actor": float(self.actor.detach().cpu()),
            "critic": float(self.critic.detach().cpu()),
            "entropy": float(self.entropy.detach().cpu()),
            "lambda_return_mean": float(self.lambda_return.detach().mean().cpu()),
        }


def lambda_returns(
    *,
    reward: torch.Tensor,
    cont: torch.Tensor,
    value: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    if reward.shape != cont.shape or reward.shape != value.shape:
        raise ValueError("reward, cont, and value must share [B,T]")
    if bootstrap.shape != reward[:, -1].shape:
        raise ValueError("bootstrap must be [B]")
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be in [0,1]")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0,1]")

    returns: list[torch.Tensor] = []
    next_return = bootstrap
    sequence_length = reward.shape[1]
    for step in reversed(range(sequence_length)):
        next_value = bootstrap if step == sequence_length - 1 else value[:, step + 1]
        mixed_next = (1.0 - lambda_) * next_value + lambda_ * next_return
        next_return = reward[:, step] + gamma * cont[:, step] * mixed_next
        returns.append(next_return)
    returns.reverse()
    return torch.stack(returns, dim=1)


def compute_behavior_loss(
    *,
    logits: torch.Tensor,
    value: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    cont: torch.Tensor,
    bootstrap: torch.Tensor | None = None,
    cfg: BehaviorLossConfig | None = None,
) -> BehaviorLossBreakdown:
    cfg = cfg or BehaviorLossConfig()
    if logits.ndim != 3:
        raise ValueError("logits must be [B,T,A]")
    if action.shape != value.shape:
        raise ValueError("action and value must share [B,T]")
    if action.dtype != torch.int64:
        raise ValueError("action must be torch.int64")
    bootstrap = value[:, -1].detach() if bootstrap is None else bootstrap
    returns = lambda_returns(
        reward=reward,
        cont=cont,
        value=value.detach(),
        bootstrap=bootstrap.detach(),
        gamma=cfg.gamma,
        lambda_=cfg.lambda_,
    )
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(action)
    entropy = dist.entropy().mean()
    advantage = returns - value
    actor_advantage = advantage.detach()
    if cfg.normalize_advantage:
        actor_advantage = (actor_advantage - actor_advantage.mean()) / (
            actor_advantage.std(unbiased=False) + cfg.advantage_epsilon
        )
    actor = -(log_prob * actor_advantage).mean()
    critic = F.mse_loss(value, returns.detach())
    total = cfg.actor_weight * actor + cfg.critic_weight * critic - cfg.entropy_weight * entropy
    return BehaviorLossBreakdown(
        total=total,
        actor=actor,
        critic=critic,
        entropy=entropy,
        lambda_return=returns,
    )
