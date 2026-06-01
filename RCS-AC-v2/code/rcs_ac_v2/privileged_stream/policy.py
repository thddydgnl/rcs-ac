"""Privileged latent policy wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rcs_ac_v2.policy.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticOutput
from rcs_ac_v2.world_model.rssm import RSSMState


@dataclass(frozen=True)
class PrivilegedPolicyConfig:
    latent_dim: int
    hidden_dim: int = 128


class PrivilegedPolicy(nn.Module):
    def __init__(self, cfg: PrivilegedPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor_critic = ActorCritic(
            ActorCriticConfig(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        )

    def forward(self, latent: torch.Tensor) -> ActorCriticOutput:
        return self.actor_critic(latent)

    def act(self, latent: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_critic.act(latent, deterministic=deterministic)


@dataclass
class PolicyRollout:
    latent: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor


def imagine_policy_rollout(
    *,
    world_model,
    policy: PrivilegedPolicy,
    initial_state: RSSMState,
    horizon: int,
    deterministic: bool = False,
) -> PolicyRollout:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    current_state = initial_state
    latents: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    conts: list[torch.Tensor] = []

    for _ in range(horizon):
        latent = current_state.latent.unsqueeze(1)
        output = policy(latent)
        dist = output.distribution
        action = torch.argmax(output.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        next_sequence = world_model.rssm.imagine(current_state, action)
        predictions = world_model.heads(next_sequence.latent)
        reward = predictions["reward"][:, 0].detach()
        cont = torch.sigmoid(predictions["continue_logits"][:, 0]).detach()

        latents.append(latent[:, 0])
        logits.append(output.logits[:, 0])
        values.append(output.value[:, 0])
        actions.append(action[:, 0])
        log_probs.append(log_prob[:, 0])
        entropies.append(output.entropy[:, 0])
        rewards.append(reward)
        conts.append(cont)
        current_state = next_sequence.final_state()

    rollout = PolicyRollout(
        latent=torch.stack(latents, dim=1),
        logits=torch.stack(logits, dim=1),
        value=torch.stack(values, dim=1),
        action=torch.stack(actions, dim=1).to(torch.int64),
        log_prob=torch.stack(log_probs, dim=1),
        entropy=torch.stack(entropies, dim=1),
        reward=torch.stack(rewards, dim=1),
        cont=torch.stack(conts, dim=1),
    )
    for name in ("latent", "logits", "value", "reward", "cont"):
        if not torch.isfinite(getattr(rollout, name)).all():
            raise ValueError(f"non-finite policy rollout field: {name}")
    return rollout
