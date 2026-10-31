"""Common observation encoders."""

from __future__ import annotations

from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    output_activation: nn.Module | None = nn.Tanh(),
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ModalityEncoder(nn.Module):
    """Simple per-modality MLP encoder."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        self.network = build_mlp(input_dim, hidden_dims, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
