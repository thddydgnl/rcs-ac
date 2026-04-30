from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RGBEncoderOutput:
    feature: torch.Tensor
    feature_map: torch.Tensor


@dataclass
class EncoderOutput:
    rgb: torch.Tensor
    rgb_feature_map: torch.Tensor
    lidar: torch.Tensor
    ego: torch.Tensor
    route: torch.Tensor
    diagnostics: dict[str, float]


class RGBEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feature_map(rgb).feature

    def forward_with_feature_map(self, rgb: torch.Tensor) -> RGBEncoderOutput:
        feature_map = self.stem(rgb.to(dtype=torch.float32))
        feature = self.pool_head(feature_map)
        return RGBEncoderOutput(feature=feature, feature_map=feature_map)


class LiDAREncoder(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.net(lidar.to(dtype=torch.float32))


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.net(vector.to(dtype=torch.float32))


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        *,
        rgb_shape: tuple[int, int, int],
        lidar_shape: tuple[int, int, int],
        ego_dim: int,
        route_dim: int,
        feature_dim: int = 128,
        vector_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.rgb_encoder = RGBEncoder(in_channels=int(rgb_shape[0]), feature_dim=int(feature_dim))
        self.lidar_encoder = LiDAREncoder(in_channels=int(lidar_shape[0]), feature_dim=int(feature_dim))
        self.ego_encoder = VectorEncoder(
            input_dim=int(ego_dim),
            hidden_dim=int(vector_hidden_dim),
            feature_dim=int(feature_dim),
        )
        self.route_encoder = VectorEncoder(
            input_dim=int(route_dim),
            hidden_dim=int(vector_hidden_dim),
            feature_dim=int(feature_dim),
        )

    def forward(self, split_observation: dict[str, torch.Tensor]) -> EncoderOutput:
        rgb_input = split_observation["rgb"]
        lidar_input = split_observation["lidar"]
        ego_input = split_observation["ego"]
        route_input = split_observation["route"]
        rgb_output = self.rgb_encoder.forward_with_feature_map(rgb_input)
        rgb = rgb_output.feature
        lidar = self.lidar_encoder(lidar_input)
        ego = self.ego_encoder(ego_input)
        route = self.route_encoder(route_input)
        diagnostics = {
            "rgb_input_mean": _finite_mean(rgb_input),
            "lidar_input_mean": _finite_mean(lidar_input),
            "rgb_feature_l1_mean": _feature_l1_mean(rgb),
            "rgb_feature_map_l1_mean": _feature_l1_mean(rgb_output.feature_map),
            "lidar_feature_l1_mean": _feature_l1_mean(lidar),
            "ego_feature_l1_mean": _feature_l1_mean(ego),
            "route_feature_l1_mean": _feature_l1_mean(route),
        }
        return EncoderOutput(
            rgb=rgb,
            rgb_feature_map=rgb_output.feature_map,
            lidar=lidar,
            ego=ego,
            route=route,
            diagnostics=diagnostics,
        )


def _finite_mean(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(dtype=torch.float32).mean().cpu().item())


def _feature_l1_mean(feature: torch.Tensor) -> float:
    return float(feature.detach().abs().mean().cpu().item())
