"""Compact Gaussian RSSM used for Stage 1 privileged world-model smoke."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rcs_ac_v2.env.discrete_actions import NUM_RAW2DRIVE_ACTIONS


@dataclass(frozen=True)
class RSSMConfig:
    action_dim: int = NUM_RAW2DRIVE_ACTIONS
    embed_dim: int = 128
    deterministic_dim: int = 128
    stochastic_dim: int = 32
    hidden_dim: int = 128
    min_std: float = 0.1


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    post_mean: torch.Tensor
    post_std: torch.Tensor

    @property
    def latent(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)


@dataclass
class RSSMSequence:
    deter: torch.Tensor
    stoch: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    post_mean: torch.Tensor
    post_std: torch.Tensor

    @property
    def latent(self) -> torch.Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)

    def final_state(self) -> RSSMState:
        return RSSMState(
            deter=self.deter[:, -1],
            stoch=self.stoch[:, -1],
            prior_mean=self.prior_mean[:, -1],
            prior_std=self.prior_std[:, -1],
            post_mean=self.post_mean[:, -1],
            post_std=self.post_std[:, -1],
        )


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RSSMConfig()
        self.recurrent = nn.GRUCell(
            self.cfg.stochastic_dim + self.cfg.action_dim,
            self.cfg.deterministic_dim,
        )
        self.prior_net = _mlp(
            self.cfg.deterministic_dim,
            self.cfg.hidden_dim,
            self.cfg.stochastic_dim * 2,
        )
        self.posterior_net = _mlp(
            self.cfg.deterministic_dim + self.cfg.embed_dim,
            self.cfg.hidden_dim,
            self.cfg.stochastic_dim * 2,
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.deterministic_dim + self.cfg.stochastic_dim

    def initial(self, batch_size: int, *, device: torch.device | None = None) -> RSSMState:
        device = device or next(self.parameters()).device
        deter = torch.zeros(batch_size, self.cfg.deterministic_dim, device=device)
        stoch = torch.zeros(batch_size, self.cfg.stochastic_dim, device=device)
        mean = torch.zeros(batch_size, self.cfg.stochastic_dim, device=device)
        std = torch.ones(batch_size, self.cfg.stochastic_dim, device=device)
        return RSSMState(
            deter=deter,
            stoch=stoch,
            prior_mean=mean,
            prior_std=std,
            post_mean=mean,
            post_std=std,
        )

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor) -> RSSMSequence:
        if embeds.ndim != 3:
            raise ValueError(f"embeds must be [B,T,E], got {tuple(embeds.shape)}")
        batch_size, sequence_length, embed_dim = embeds.shape
        if embed_dim != self.cfg.embed_dim:
            raise ValueError(f"embed dim must be {self.cfg.embed_dim}, got {embed_dim}")
        action_onehot = _to_action_onehot(actions, self.cfg.action_dim)
        if tuple(action_onehot.shape[:2]) != (batch_size, sequence_length):
            raise ValueError("actions must share [B,T] with embeds")

        state = self.initial(batch_size, device=embeds.device)
        states: list[RSSMState] = []
        zero_action = torch.zeros(batch_size, self.cfg.action_dim, device=embeds.device)

        for step in range(sequence_length):
            prev_action = zero_action if step == 0 else action_onehot[:, step - 1]
            state = self._observe_step(state, prev_action, embeds[:, step])
            states.append(state)

        return _stack_states(states)

    def imagine(self, state: RSSMState, actions: torch.Tensor) -> RSSMSequence:
        action_onehot = _to_action_onehot(actions, self.cfg.action_dim)
        if action_onehot.ndim != 3:
            raise ValueError("actions must be [B,T] ids or [B,T,A] one-hot/probabilities")
        states: list[RSSMState] = []
        current = state
        for step in range(action_onehot.shape[1]):
            current = self._imagine_step(current, action_onehot[:, step])
            states.append(current)
        return _stack_states(states)

    def _observe_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
    ) -> RSSMState:
        deter = self.recurrent(torch.cat([prev_state.stoch, prev_action], dim=-1), prev_state.deter)
        prior_mean, prior_std = _stats_from_output(self.prior_net(deter), self.cfg.min_std)
        post_mean, post_std = _stats_from_output(
            self.posterior_net(torch.cat([deter, embed], dim=-1)),
            self.cfg.min_std,
        )
        stoch = post_mean
        return RSSMState(
            deter=deter,
            stoch=stoch,
            prior_mean=prior_mean,
            prior_std=prior_std,
            post_mean=post_mean,
            post_std=post_std,
        )

    def _imagine_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        deter = self.recurrent(torch.cat([prev_state.stoch, action], dim=-1), prev_state.deter)
        prior_mean, prior_std = _stats_from_output(self.prior_net(deter), self.cfg.min_std)
        stoch = prior_mean
        return RSSMState(
            deter=deter,
            stoch=stoch,
            prior_mean=prior_mean,
            prior_std=prior_std,
            post_mean=prior_mean,
            post_std=prior_std,
        )


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _stats_from_output(output: torch.Tensor, min_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw_std = torch.chunk(output, chunks=2, dim=-1)
    std = F.softplus(raw_std) + min_std
    return mean, std


def _to_action_onehot(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    if actions.ndim == 2:
        if actions.dtype != torch.int64:
            raise ValueError("action ids must be torch.int64")
        if actions.numel() > 0:
            if int(actions.min()) < 0 or int(actions.max()) >= action_dim:
                raise ValueError(f"action ids must be in [0, {action_dim - 1}]")
        return F.one_hot(actions, num_classes=action_dim).float()
    if actions.ndim == 3:
        if actions.shape[-1] != action_dim:
            raise ValueError(f"action vector dim must be {action_dim}")
        return actions.float()
    raise ValueError(f"actions must be [B,T] or [B,T,A], got {tuple(actions.shape)}")


def _stack_states(states: list[RSSMState]) -> RSSMSequence:
    if not states:
        raise ValueError("cannot stack empty RSSM state sequence")
    return RSSMSequence(
        deter=torch.stack([state.deter for state in states], dim=1),
        stoch=torch.stack([state.stoch for state in states], dim=1),
        prior_mean=torch.stack([state.prior_mean for state in states], dim=1),
        prior_std=torch.stack([state.prior_std for state in states], dim=1),
        post_mean=torch.stack([state.post_mean for state in states], dim=1),
        post_std=torch.stack([state.post_std for state in states], dim=1),
    )
