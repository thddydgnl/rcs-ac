from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rcs_ac_v1_1.runtime.bev_spec import BEVChannelSpec, BEVSpec, DEFAULT_BEV_CHANNELS, DEFAULT_BEV_SPEC
from rcs_ac_v1_1.runtime.observation import (
    DEFAULT_EGO_SCALAR_DIM,
    DEFAULT_RGB_SHAPE,
    DEFAULT_ROUTE_SCALAR_DIM,
    DEFAULT_SENSOR_HEALTH_DIM,
    DEFAULT_WAYPOINT_COUNT,
    V11PolicyObservationSpec,
    build_v11_policy_observation_spec,
    split_flat_observation,
)


@dataclass(frozen=True)
class V11BEVPolicyConfig:
    rgb_shape: tuple[int, int, int] = DEFAULT_RGB_SHAPE
    bev_spec: BEVSpec = DEFAULT_BEV_SPEC
    bev_channels: BEVChannelSpec = DEFAULT_BEV_CHANNELS
    ego_scalar_dim: int = DEFAULT_EGO_SCALAR_DIM
    route_scalar_dim: int = DEFAULT_ROUTE_SCALAR_DIM
    sensor_health_dim: int = DEFAULT_SENSOR_HEALTH_DIM
    feature_dim: int = 128
    fusion_dim: int = 128
    waypoint_count: int = DEFAULT_WAYPOINT_COUNT
    waypoint_mode: str = "distance"
    waypoint_distances_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    waypoint_gap_residual_scale_m: float = 0.75
    waypoint_min_gap_m: float = 0.75
    waypoint_max_lateral_m: float = 6.0
    waypoint_parameterization: str = "monotonic_bounded"
    min_target_speed_kmh: float = 0.0
    max_target_speed_kmh: float = 28.0
    use_bev_prediction_head: bool = True

    def __post_init__(self) -> None:
        for name in (
            "ego_scalar_dim",
            "route_scalar_dim",
            "sensor_health_dim",
            "feature_dim",
            "fusion_dim",
            "waypoint_count",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.waypoint_mode != "distance":
            raise ValueError("Only waypoint_mode='distance' is supported in v1.1 scaffold")
        if self.waypoint_parameterization != "monotonic_bounded":
            raise ValueError("Only waypoint_parameterization='monotonic_bounded' is supported")
        if len(self.waypoint_distances_m) != int(self.waypoint_count):
            raise ValueError("waypoint_distances_m length must match waypoint_count")
        previous = 0.0
        for distance in self.waypoint_distances_m:
            if float(distance) <= previous:
                raise ValueError("waypoint_distances_m must be strictly increasing and positive")
            previous = float(distance)
        for name in ("waypoint_gap_residual_scale_m", "waypoint_min_gap_m", "waypoint_max_lateral_m"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if float(self.max_target_speed_kmh) <= float(self.min_target_speed_kmh):
            raise ValueError("max_target_speed_kmh must be greater than min_target_speed_kmh")


def parameterize_waypoints(
    raw_waypoint: torch.Tensor,
    *,
    distances_m: tuple[float, ...],
    gap_residual_scale_m: float,
    min_gap_m: float,
    max_lateral_m: float,
) -> torch.Tensor:
    """Convert unconstrained raw waypoint output into a forward local path."""

    if raw_waypoint.ndim != 2:
        raise ValueError(f"raw_waypoint must be [B, 2*K], got {tuple(raw_waypoint.shape)}")
    waypoint_count = len(distances_m)
    if int(raw_waypoint.shape[-1]) != 2 * waypoint_count:
        raise ValueError(
            f"raw_waypoint last dim must be {2 * waypoint_count}, got {int(raw_waypoint.shape[-1])}"
        )
    distances = torch.as_tensor(distances_m, dtype=raw_waypoint.dtype, device=raw_waypoint.device)
    base_gaps = torch.empty_like(distances)
    base_gaps[0] = distances[0]
    base_gaps[1:] = distances[1:] - distances[:-1]

    raw_gap = raw_waypoint[:, :waypoint_count]
    raw_y = raw_waypoint[:, waypoint_count:]
    gap_residual = float(gap_residual_scale_m) * torch.tanh(raw_gap)
    gaps = torch.clamp(base_gaps.unsqueeze(0) + gap_residual, min=float(min_gap_m))
    x = torch.cumsum(gaps, dim=-1)
    y = float(max_lateral_m) * torch.tanh(raw_y)
    return torch.stack((x, y), dim=-1)


class BEVSourceEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.net(bev.to(dtype=torch.float32))


class RGBContextEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb.to(dtype=torch.float32))


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 64),
            nn.ReLU(),
            nn.Linear(64, int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.net(vector.to(dtype=torch.float32))


class SourceLevelBEVFusion(nn.Module):
    """Small source-level fusion module for v1.1 smoke experiments.

    This is intentionally source-level, not channel-level, to preserve the
    RCS-AC reliability/fusion interpretation from the v1 design.
    """

    source_order: tuple[str, ...] = ("rgb", "lidar_bev", "route_bev", "ego_bev")

    def __init__(self, feature_dim: int, fusion_dim: int, health_dim: int) -> None:
        super().__init__()
        self.project = nn.ModuleDict(
            {key: nn.Linear(int(feature_dim), int(fusion_dim)) for key in self.source_order}
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(int(health_dim), 64),
            nn.ReLU(),
            nn.Linear(64, len(self.source_order)),
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        sensor_health: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.stack([self.project[key](features[key]) for key in self.source_order], dim=1)
        reliability_logits = self.reliability_head(sensor_health.to(dtype=projected.dtype, device=projected.device))
        weights = torch.softmax(reliability_logits, dim=-1)
        fused = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        return fused, weights


class RCSACV11BEVPolicy(nn.Module):
    """RCS-AC v1.1 Full-BEV input policy scaffold.

    Policy inputs are sensor-derived observations only. GT BEV targets and
    expert waypoints are deliberately absent from ``forward`` to prevent
    evaluation-time leakage.
    """

    def __init__(self, config: V11BEVPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or V11BEVPolicyConfig()
        self.obs_spec: V11PolicyObservationSpec = build_v11_policy_observation_spec(
            rgb_shape=self.config.rgb_shape,
            bev_spec=self.config.bev_spec,
            bev_channels=self.config.bev_channels,
            ego_scalar_dim=self.config.ego_scalar_dim,
            route_scalar_dim=self.config.route_scalar_dim,
            sensor_health_dim=self.config.sensor_health_dim,
        )
        feature_dim = int(self.config.feature_dim)
        self.rgb_context_encoder = RGBContextEncoder(self.config.rgb_shape[0], feature_dim)
        self.lidar_bev_encoder = BEVSourceEncoder(self.config.bev_channels.lidar_channels, feature_dim)
        self.route_bev_encoder = BEVSourceEncoder(self.config.bev_channels.route_channels, feature_dim)
        self.ego_bev_encoder = BEVSourceEncoder(self.config.bev_channels.ego_channels, feature_dim)
        self.ego_scalar_encoder = VectorEncoder(self.config.ego_scalar_dim, feature_dim)
        self.route_scalar_encoder = VectorEncoder(self.config.route_scalar_dim, feature_dim)
        self.fusion = SourceLevelBEVFusion(
            feature_dim=feature_dim,
            fusion_dim=int(self.config.fusion_dim),
            health_dim=int(self.config.sensor_health_dim),
        )
        policy_input_dim = int(self.config.fusion_dim) + 2 * feature_dim
        self.policy_body = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.waypoint_head = nn.Linear(256, int(self.config.waypoint_count) * 2)
        self.target_speed_head = nn.Linear(256, 1)
        self.value_head = nn.Linear(256, 1)
        self.bev_prediction_head = nn.Sequential(
            nn.Linear(256, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, int(self.config.bev_channels.target_channels), kernel_size=4, stride=2, padding=1),
        ) if bool(self.config.use_bev_prediction_head) else None

    def forward(self, flat_observation: torch.Tensor) -> dict[str, torch.Tensor]:
        split = split_flat_observation(flat_observation, self.obs_spec)
        rgb_feature = self.rgb_context_encoder(split["rgb"])
        features = {
            "rgb": rgb_feature,
            "lidar_bev": self.lidar_bev_encoder(split["lidar_bev"]),
            "route_bev": self.route_bev_encoder(split["route_bev"]),
            "ego_bev": self.ego_bev_encoder(split["ego_bev"]),
        }
        fused_bev, fusion_weights = self.fusion(features, split["sensor_health"])
        ego_feature = self.ego_scalar_encoder(split["ego_scalar"])
        route_feature = self.route_scalar_encoder(split["route_scalar"])
        body_input = torch.cat([fused_bev, ego_feature, route_feature], dim=-1)
        hidden = self.policy_body(body_input)
        raw_waypoint = self.waypoint_head(hidden)
        waypoints = parameterize_waypoints(
            raw_waypoint,
            distances_m=self.config.waypoint_distances_m,
            gap_residual_scale_m=float(self.config.waypoint_gap_residual_scale_m),
            min_gap_m=float(self.config.waypoint_min_gap_m),
            max_lateral_m=float(self.config.waypoint_max_lateral_m),
        )
        speed_unit = torch.sigmoid(self.target_speed_head(hidden))
        target_speed = speed_unit * (
            float(self.config.max_target_speed_kmh) - float(self.config.min_target_speed_kmh)
        ) + float(self.config.min_target_speed_kmh)
        result = {
            "waypoints": waypoints,
            "target_speed_kmh": target_speed,
            "value": self.value_head(hidden),
            "bev_fusion_weights": fusion_weights,
        }
        if self.bev_prediction_head is not None:
            result["bev_logits"] = self.bev_prediction_head(hidden)
        return result
