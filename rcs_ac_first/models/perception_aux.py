from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_rgb_perception_raw_features(
    road_mask_logits: torch.Tensor,
    lane_edge_mask_logits: torch.Tensor,
) -> torch.Tensor:
    road, lane = _validate_and_sigmoid_perception_logits(road_mask_logits, lane_edge_mask_logits)
    _, _, height, width = road.shape
    lower_start = max(0, height // 2)
    bottom_start = max(0, int(height * 3 // 4))
    center_left = max(0, width // 3)
    center_right = min(width, int(width * 2 // 3))
    left_right = max(1, width // 2)

    road_lower = road[:, :, lower_start:, :]
    lane_lower = lane[:, :, lower_start:, :]
    lane_left = lane[:, :, :, :left_right]
    lane_right = lane[:, :, :, left_right:]
    road_center = road[:, :, :, center_left:center_right]
    lane_center = lane[:, :, :, center_left:center_right]
    road_bottom = road[:, :, bottom_start:, :]
    lane_bottom = lane[:, :, bottom_start:, :]

    x_coords = torch.linspace(-1.0, 1.0, width, dtype=road.dtype, device=road.device).view(1, 1, 1, width)
    y_coords = torch.linspace(0.0, 1.0, height, dtype=road.dtype, device=road.device).view(1, 1, height, 1)

    road_x = _weighted_x_mean(road_lower, x_coords)
    lane_x = _weighted_x_mean(lane_lower, x_coords)
    road_y = (road * y_coords).sum(dim=(1, 2, 3)) / road.sum(dim=(1, 2, 3)).clamp_min(1.0e-6)
    lane_y = (lane * y_coords).sum(dim=(1, 2, 3)) / lane.sum(dim=(1, 2, 3)).clamp_min(1.0e-6)

    features = [
        road.mean(dim=(1, 2, 3)),
        road_lower.mean(dim=(1, 2, 3)),
        road_center.mean(dim=(1, 2, 3)),
        road_bottom.mean(dim=(1, 2, 3)),
        lane.mean(dim=(1, 2, 3)),
        lane_lower.mean(dim=(1, 2, 3)),
        lane_center.mean(dim=(1, 2, 3)),
        lane_left.mean(dim=(1, 2, 3)),
        lane_right.mean(dim=(1, 2, 3)),
        lane_bottom.mean(dim=(1, 2, 3)),
        road_x,
        lane_x,
        road_y,
        lane_y,
    ]
    raw = torch.stack(features, dim=-1)
    if not torch.isfinite(raw).all():
        raise ValueError("RGB perception actor features contain non-finite values.")
    return raw


def build_estimated_scalar_visual_features(
    road_mask_logits: torch.Tensor,
    lane_edge_mask_logits: torch.Tensor,
) -> torch.Tensor:
    raw = build_rgb_perception_raw_features(road_mask_logits, lane_edge_mask_logits)
    road, lane = _validate_and_sigmoid_perception_logits(road_mask_logits, lane_edge_mask_logits)
    _, _, height, width = road.shape

    far_start = max(0, height // 4)
    far_stop = max(far_start + 1, height // 2)
    mid_start = max(0, height // 2)
    mid_stop = max(mid_start + 1, int(height * 3 // 4))
    near_start = max(0, int(height * 3 // 4))
    near_stop = height

    x_coords = torch.linspace(-1.0, 1.0, width, dtype=road.dtype, device=road.device).view(1, 1, 1, width)

    road_far = road[:, :, far_start:far_stop, :]
    road_mid = road[:, :, mid_start:mid_stop, :]
    road_near = road[:, :, near_start:near_stop, :]
    lane_far = lane[:, :, far_start:far_stop, :]
    lane_mid = lane[:, :, mid_start:mid_stop, :]
    lane_near = lane[:, :, near_start:near_stop, :]

    road_far_x = _weighted_x_mean(road_far, x_coords)
    road_mid_x = _weighted_x_mean(road_mid, x_coords)
    road_near_x = _weighted_x_mean(road_near, x_coords)
    lane_far_x = _weighted_x_mean(lane_far, x_coords)
    lane_mid_x = _weighted_x_mean(lane_mid, x_coords)
    lane_near_x = _weighted_x_mean(lane_near, x_coords)

    extended = torch.stack(
        [
            road_far_x,
            road_mid_x,
            road_near_x,
            road_far_x - road_near_x,
            road_far.mean(dim=(1, 2, 3)),
            road_mid.mean(dim=(1, 2, 3)),
            road_near.mean(dim=(1, 2, 3)),
            lane_far_x,
            lane_mid_x,
            lane_near_x,
            lane_far_x - lane_near_x,
            lane_far.mean(dim=(1, 2, 3)),
            lane_mid.mean(dim=(1, 2, 3)),
            lane_near.mean(dim=(1, 2, 3)),
        ],
        dim=-1,
    )
    features = torch.cat([raw, extended], dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("Estimated scalar visual features contain non-finite values.")
    return features


def _validate_and_sigmoid_perception_logits(
    road_mask_logits: torch.Tensor,
    lane_edge_mask_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if road_mask_logits.ndim != 4 or lane_edge_mask_logits.ndim != 4:
        raise ValueError("Perception logits must have shape [B, 1, H, W].")
    if tuple(road_mask_logits.shape) != tuple(lane_edge_mask_logits.shape):
        raise ValueError(
            "road/lane logits shape mismatch: "
            f"{tuple(road_mask_logits.shape)} vs {tuple(lane_edge_mask_logits.shape)}"
        )
    if int(road_mask_logits.shape[1]) != 1:
        raise ValueError(f"Expected single-channel logits, got {tuple(road_mask_logits.shape)}")

    road = torch.sigmoid(road_mask_logits.to(dtype=torch.float32))
    lane = torch.sigmoid(lane_edge_mask_logits.to(dtype=torch.float32))
    return road, lane


def _weighted_x_mean(mask: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
    return (mask * x_coords).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0e-6)


class RGBPerceptionAuxHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        hidden_channels: int = 64,
        output_size: tuple[int, int] = (45, 80),
    ) -> None:
        super().__init__()
        self.output_size = (int(output_size[0]), int(output_size[1]))
        if self.output_size[0] <= 0 or self.output_size[1] <= 0:
            raise ValueError(f"output_size must be positive, got {output_size!r}")
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(int(hidden_channels), 2, kernel_size=1)

    def forward(self, rgb_feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.net(rgb_feature_map.to(dtype=torch.float32))
        resized = F.interpolate(
            feature,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.classifier(resized)
        return {
            "pred_road_mask_logits": logits[:, 0:1, :, :],
            "pred_lane_edge_mask_logits": logits[:, 1:2, :, :],
        }


class RGBPerceptionActorFeatureExtractor(nn.Module):
    raw_feature_dim: int = 14

    def __init__(
        self,
        *,
        output_dim: int = 16,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim!r}")
        self.projector = nn.Sequential(
            nn.Linear(self.raw_feature_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        road_mask_logits: torch.Tensor,
        lane_edge_mask_logits: torch.Tensor,
        *,
        zero_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw = self._build_raw_features(road_mask_logits, lane_edge_mask_logits)
        compact = self.projector(raw)
        if bool(zero_features):
            compact = torch.zeros_like(compact)
        return {
            "raw_features": raw,
            "compact_features": compact,
        }

    def _build_raw_features(
        self,
        road_mask_logits: torch.Tensor,
        lane_edge_mask_logits: torch.Tensor,
    ) -> torch.Tensor:
        return build_rgb_perception_raw_features(road_mask_logits, lane_edge_mask_logits)


class EstimatedScalarHead(nn.Module):
    input_dim: int = 28

    def __init__(
        self,
        *,
        route_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.route_dim = int(route_dim)
        if self.route_dim <= 0:
            raise ValueError(f"route_dim must be positive, got {route_dim!r}")
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + self.route_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )
        self._initialize_residual_output()

    def forward(
        self,
        visual_geometry_features: torch.Tensor,
        route_input: torch.Tensor,
    ) -> torch.Tensor:
        if visual_geometry_features.ndim != 2 or int(visual_geometry_features.shape[-1]) != self.input_dim:
            raise ValueError(
                "visual_geometry_features must have shape "
                f"[B, {self.input_dim}], got {tuple(visual_geometry_features.shape)}"
            )
        route = route_input.to(device=visual_geometry_features.device, dtype=visual_geometry_features.dtype)
        if route.ndim != 2 or int(route.shape[-1]) != self.route_dim:
            raise ValueError(f"route_input must have shape [B, {self.route_dim}], got {tuple(route.shape)}")
        estimated = self.net(torch.cat([visual_geometry_features, route], dim=-1))
        if not torch.isfinite(estimated).all():
            raise ValueError("Estimated scalar head produced non-finite values.")
        return estimated

    def _initialize_residual_output(self) -> None:
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            return
        with torch.no_grad():
            output.weight.zero_()
            output.bias.zero_()
