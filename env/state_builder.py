from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.common import compute_speed_mps, world_to_local_2d

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass
class ObservationConfig:
    rgb_width: int = 160
    rgb_height: int = 90
    rgb_channels_first: bool = True
    bev_x_bins: int = 64
    bev_y_bins: int = 64
    bev_x_range: tuple[float, float] = (0.0, 30.0)
    bev_y_range: tuple[float, float] = (-15.0, 15.0)
    bev_z_min: float = -2.5
    bev_z_max: float = 1.0
    route_lookahead_count: int = 3


class StateBuilder:
    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()

    def build_rgb(self, image: Any) -> np.ndarray:
        if image is None:
            rgb = np.zeros((self.config.rgb_height, self.config.rgb_width, 3), dtype=np.float32)
        else:
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            rgb_uint8 = raw[:, :, :3][:, :, ::-1]
            resized = self._resize_rgb(rgb_uint8, self.config.rgb_width, self.config.rgb_height)
            rgb = resized.astype(np.float32) / 255.0

        if self.config.rgb_channels_first:
            rgb = np.transpose(rgb, (2, 0, 1))
        return rgb.astype(np.float32, copy=False)

    def build_lidar_bev(self, points: np.ndarray) -> np.ndarray:
        bev = np.zeros((self.config.bev_y_bins, self.config.bev_x_bins), dtype=np.float32)
        if points is None or points.size == 0:
            return bev[np.newaxis, :, :]

        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError(f"Expected LiDAR points with shape (N,3) or (N,4), got {points.shape}")

        xyz = points[:, :3]
        x_min, x_max = self.config.bev_x_range
        y_min, y_max = self.config.bev_y_range
        z_min = self.config.bev_z_min
        z_max = self.config.bev_z_max

        valid = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] <= y_max)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
        )
        xyz = xyz[valid]
        if xyz.shape[0] == 0:
            return bev[np.newaxis, :, :]

        hist, _, _ = np.histogram2d(
            xyz[:, 1],
            xyz[:, 0],
            bins=[self.config.bev_y_bins, self.config.bev_x_bins],
            range=[self.config.bev_y_range, self.config.bev_x_range],
        )
        hist = hist.astype(np.float32)
        max_value = float(hist.max())
        if max_value > 0.0:
            hist /= max_value
        return hist[np.newaxis, :, :].astype(np.float32, copy=False)

    def build_ego(self, vehicle: Any, control: Any, route_metrics: Any) -> np.ndarray:
        speed_mps = compute_speed_mps(vehicle)
        angular_velocity = vehicle.get_angular_velocity()
        yaw_rate = np.deg2rad(float(angular_velocity.z))

        acceleration = vehicle.get_acceleration()
        world_acc_location = type(vehicle.get_location())(
            x=vehicle.get_transform().location.x + acceleration.x,
            y=vehicle.get_transform().location.y + acceleration.y,
            z=vehicle.get_transform().location.z + acceleration.z,
        )
        accel_x, accel_y = world_to_local_2d(vehicle.get_transform(), world_acc_location)

        features = np.asarray(
            [
                speed_mps,
                yaw_rate,
                accel_x,
                accel_y,
                float(control.steer),
                float(route_metrics.heading_error),
                float(route_metrics.lane_offset),
            ],
            dtype=np.float32,
        )
        return features

    def build_route(self, route_manager: Any, vehicle_transform: Any) -> np.ndarray:
        route_vec = route_manager.build_route_vector(
            vehicle_transform=vehicle_transform,
            count=self.config.route_lookahead_count,
        )
        expected_size = self.config.route_lookahead_count * 2
        if route_vec.shape[0] < expected_size:
            padded = np.zeros((expected_size,), dtype=np.float32)
            padded[: route_vec.shape[0]] = route_vec
            return padded
        return route_vec.astype(np.float32, copy=False)

    def build_observation(
        self,
        image: Any,
        lidar_points: np.ndarray,
        vehicle: Any,
        control: Any,
        route_manager: Any,
        route_metrics: Any,
    ) -> dict[str, np.ndarray]:
        observation = {
            "rgb": self.build_rgb(image),
            "lidar_bev": self.build_lidar_bev(lidar_points),
            "ego": self.build_ego(vehicle, control, route_metrics),
            "route": self.build_route(route_manager, vehicle.get_transform()),
        }
        return {key: value.astype(np.float32, copy=False) for key, value in observation.items()}

    def _resize_rgb(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return self._nearest_resize(image, width, height)

    def _nearest_resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        if src_h == height and src_w == width:
            return image

        y_idx = np.linspace(0, src_h - 1, height).astype(np.int32)
        x_idx = np.linspace(0, src_w - 1, width).astype(np.int32)
        return image[y_idx][:, x_idx]
