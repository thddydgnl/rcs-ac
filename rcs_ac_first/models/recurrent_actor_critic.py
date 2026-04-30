from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RecurrentActorCriticConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    layers: int = 1
    action_dim: int = 2
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_init_by_dim: tuple[float, ...] = (-0.2, -0.2)
    min_std_by_dim: tuple[float, ...] = (0.05, 0.05)
    max_std_by_dim: tuple[float, ...] = (0.8, 0.8)


@dataclass
class RecurrentActorCriticOutput:
    action_mean: torch.Tensor
    value: torch.Tensor
    next_hidden: torch.Tensor
    recurrent_feature: torch.Tensor


class RecurrentActorCritic(nn.Module):
    def __init__(self, config: RecurrentActorCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or RecurrentActorCriticConfig()
        self.gru = nn.GRU(
            input_size=int(self.config.input_dim),
            hidden_size=int(self.config.hidden_dim),
            num_layers=int(self.config.layers),
            batch_first=True,
        )
        self.actor_head = nn.Sequential(
            nn.Linear(int(self.config.hidden_dim), int(self.config.actor_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.actor_hidden_dim), int(self.config.action_dim)),
            nn.Tanh(),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(int(self.config.hidden_dim), int(self.config.critic_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(self.config.critic_hidden_dim), 1),
        )
        self.log_std = nn.Parameter(torch.as_tensor(self.config.log_std_init_by_dim, dtype=torch.float32))
        self.register_buffer(
            "_min_std_by_dim",
            torch.as_tensor(self.config.min_std_by_dim, dtype=torch.float32).clamp_min(1.0e-6),
            persistent=False,
        )
        self.register_buffer(
            "_max_std_by_dim",
            torch.as_tensor(self.config.max_std_by_dim, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        fused: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> RecurrentActorCriticOutput:
        if fused.ndim != 2:
            raise ValueError(f"Expected fused feature with shape [B, D], got {tuple(fused.shape)}")
        batch_size = int(fused.shape[0])
        if hidden_state is not None:
            expected = (int(self.config.layers), batch_size, int(self.config.hidden_dim))
            if tuple(hidden_state.shape) != expected:
                raise ValueError(f"Expected hidden_state shape {expected}, got {tuple(hidden_state.shape)}")
        out_seq, next_hidden = self.gru(fused.unsqueeze(1), hidden_state)
        recurrent_feature = out_seq[:, -1, :]
        action_mean = self.actor_head(recurrent_feature)
        value = self.critic_head(recurrent_feature).squeeze(-1)
        return RecurrentActorCriticOutput(
            action_mean=action_mean,
            value=value,
            next_hidden=next_hidden,
            recurrent_feature=recurrent_feature,
        )

    def get_std(self) -> torch.Tensor:
        std = torch.exp(self.log_std)
        std = torch.maximum(std, self._min_std_by_dim)
        return torch.minimum(std, torch.maximum(self._max_std_by_dim, self._min_std_by_dim))

    def evaluate_value(
        self,
        fused: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(fused, hidden_state=hidden_state).value
