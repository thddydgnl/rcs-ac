from __future__ import annotations

import torch
from torch.distributions import Normal


class BoundedSquashedNormal:
    def __init__(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        action_bounds: list[tuple[float, float]],
    ) -> None:
        if action_mean.ndim != 2:
            raise ValueError(f"Expected action_mean shape [B, A], got {tuple(action_mean.shape)}")
        if len(action_bounds) != int(action_mean.shape[-1]):
            raise ValueError(f"Expected {int(action_mean.shape[-1])} action bounds, got {len(action_bounds)}")
        self.eps = 1.0e-6
        lower = torch.as_tensor(
            [float(bound[0]) for bound in action_bounds],
            dtype=action_mean.dtype,
            device=action_mean.device,
        )
        upper = torch.as_tensor(
            [float(bound[1]) for bound in action_bounds],
            dtype=action_mean.dtype,
            device=action_mean.device,
        )
        self.lower = lower
        self.upper = upper
        self.scale = ((upper - lower) * 0.5).clamp_min(self.eps)
        self.bias = (upper + lower) * 0.5
        normalized_mean = ((action_mean - self.bias) / self.scale).clamp(-1.0 + self.eps, 1.0 - self.eps)
        raw_loc = torch.atanh(normalized_mean)
        raw_scale = (action_std.reshape(1, -1) / self.scale).clamp_min(self.eps)
        self.base = Normal(raw_loc, raw_scale)

    @property
    def mean(self) -> torch.Tensor:
        return self._squash(self.base.loc)

    def sample(self) -> torch.Tensor:
        return self._squash(self.base.sample())

    def rsample(self) -> torch.Tensor:
        return self._squash(self.base.rsample())

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_action(action)
        raw_action = torch.atanh(normalized)
        log_det = torch.log(self.scale) + torch.log((1.0 - normalized.pow(2)).clamp_min(self.eps))
        return (self.base.log_prob(raw_action) - log_det).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.base.entropy().sum(dim=-1)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return ((action - self.bias) / self.scale).clamp(-1.0 + self.eps, 1.0 - self.eps)

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        normalized = torch.tanh(raw_action)
        action = self.bias + self.scale * normalized
        return torch.maximum(torch.minimum(action, self.upper), self.lower)
