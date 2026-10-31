from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def _ensure_finite_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.numel() == 0:
        raise ValueError(f"{name} tensor is empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} tensor contains NaN or inf values.")
    return tensor


def _to_batch_tensor(value: Any, name: str, device: torch.device | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    tensor = tensor.to(device=device, dtype=torch.float32) if device is not None else tensor.to(dtype=torch.float32)
    if tensor.ndim == 0:
        raise ValueError(f"{name} tensor must have at least one dimension.")
    return _ensure_finite_tensor(tensor, name)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, hidden_channels: tuple[int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = hidden_channels
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected image batch with 4 dims, got shape {tuple(x.shape)}")
        return _ensure_finite_tensor(self.net(_ensure_finite_tensor(x, "image_input")), "image_feature")


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected vector batch with 2 dims, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected vector feature dim {self.input_dim}, got {x.shape[-1]}")
        return _ensure_finite_tensor(self.net(_ensure_finite_tensor(x, "vector_input")), "vector_feature")


def prepare_modal_batch(
    observation: dict[str, Any],
    rgb_shape: tuple[int, int, int],
    lidar_shape: tuple[int, int, int],
    ego_dim: int,
    route_dim: int,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    rgb = _to_batch_tensor(observation["rgb"], "rgb", device=device)
    lidar = _to_batch_tensor(observation["lidar"], "lidar", device=device)
    ego = _to_batch_tensor(observation["ego"], "ego", device=device)
    route = _to_batch_tensor(observation["route"], "route", device=device)

    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if lidar.ndim == 3:
        lidar = lidar.unsqueeze(0)
    if ego.ndim == 1:
        ego = ego.unsqueeze(0)
    if route.ndim == 1:
        route = route.unsqueeze(0)

    if tuple(rgb.shape[1:]) != tuple(rgb_shape):
        raise ValueError(f"RGB batch shape mismatch: expected {rgb_shape}, got {tuple(rgb.shape[1:])}")
    if tuple(lidar.shape[1:]) != tuple(lidar_shape):
        raise ValueError(f"LiDAR batch shape mismatch: expected {lidar_shape}, got {tuple(lidar.shape[1:])}")
    if int(ego.shape[-1]) != int(ego_dim):
        raise ValueError(f"Ego dim mismatch: expected {ego_dim}, got {ego.shape[-1]}")
    if int(route.shape[-1]) != int(route_dim):
        raise ValueError(f"Route dim mismatch: expected {route_dim}, got {route.shape[-1]}")

    batch_size = rgb.shape[0]
    if not (lidar.shape[0] == ego.shape[0] == route.shape[0] == batch_size):
        raise ValueError(
            "Batch size mismatch across modalities: "
            f"rgb={rgb.shape[0]} lidar={lidar.shape[0]} ego={ego.shape[0]} route={route.shape[0]}"
        )

    return {"rgb": rgb, "lidar": lidar, "ego": ego, "route": route}
