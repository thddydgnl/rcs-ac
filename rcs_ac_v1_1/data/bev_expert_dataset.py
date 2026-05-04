from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rcs_ac_v1_1.runtime.config import V11RuntimeConfig
from rcs_ac_v1_1.runtime.observation import flatten_observation_dict

REQUIRED_KEYS: tuple[str, ...] = (
    "rgb",
    "lidar_bev",
    "route_bev",
    "ego_bev",
    "ego_scalar",
    "route_scalar",
    "sensor_health",
    "gt_bev_target",
    "expert_waypoints",
    "expert_target_speed",
    "bev_target_available",
    "waypoint_target_available",
)


class BEVExpertDataset(Dataset):
    def __init__(self, path: str | Path, runtime_config: V11RuntimeConfig) -> None:
        self.path = Path(path)
        self.runtime_config = runtime_config
        self.arrays = {key: value for key, value in np.load(self.path, allow_pickle=False).items()}
        self._validate()
        self.length = int(self.arrays["rgb"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = int(index)
        observation = {
            "rgb": self.arrays["rgb"][idx],
            "lidar_bev": self.arrays["lidar_bev"][idx],
            "route_bev": self.arrays["route_bev"][idx],
            "ego_bev": self.arrays["ego_bev"][idx],
            "ego_scalar": self.arrays["ego_scalar"][idx],
            "route_scalar": self.arrays["route_scalar"][idx],
            "sensor_health": self.arrays["sensor_health"][idx],
        }
        obs_flat = flatten_observation_dict(observation, self.runtime_config.observation_spec)
        targets = {
            "gt_bev_target": torch.as_tensor(self.arrays["gt_bev_target"][idx], dtype=torch.float32),
            "expert_waypoints": torch.as_tensor(self.arrays["expert_waypoints"][idx], dtype=torch.float32),
            "expert_target_speed": torch.as_tensor(self.arrays["expert_target_speed"][idx], dtype=torch.float32),
            "bev_target_available": torch.as_tensor(self.arrays["bev_target_available"][idx], dtype=torch.float32),
            "waypoint_target_available": torch.as_tensor(
                self.arrays["waypoint_target_available"][idx], dtype=torch.float32
            ),
        }
        meta = {}
        for key in ("episode_id", "step", "done"):
            if key in self.arrays:
                meta[key] = self.arrays[key][idx].item() if np.asarray(self.arrays[key][idx]).ndim == 0 else self.arrays[key][idx]
        return {
            "obs_flat": torch.as_tensor(obs_flat, dtype=torch.float32),
            "targets": targets,
            "meta": meta,
        }

    def _validate(self) -> None:
        missing = [key for key in REQUIRED_KEYS if key not in self.arrays]
        if missing:
            raise KeyError(f"BEVExpertDataset missing keys: {missing}")
        n = int(self.arrays["rgb"].shape[0])
        if n <= 0:
            raise ValueError("BEVExpertDataset must contain at least one sample")
        for key in REQUIRED_KEYS:
            if int(self.arrays[key].shape[0]) != n:
                raise ValueError(f"Dataset key {key!r} first dimension does not match N={n}")
            if not np.isfinite(self.arrays[key]).all():
                raise ValueError(f"Dataset key {key!r} contains NaN or inf")
        expected_shapes = {
            "rgb": self.runtime_config.observation_spec.modal("rgb").shape,
            "lidar_bev": self.runtime_config.observation_spec.modal("lidar_bev").shape,
            "route_bev": self.runtime_config.observation_spec.modal("route_bev").shape,
            "ego_bev": self.runtime_config.observation_spec.modal("ego_bev").shape,
            "ego_scalar": self.runtime_config.observation_spec.modal("ego_scalar").shape,
            "route_scalar": self.runtime_config.observation_spec.modal("route_scalar").shape,
            "sensor_health": self.runtime_config.observation_spec.modal("sensor_health").shape,
            "gt_bev_target": self.runtime_config.target_spec.gt_bev_shape,
            "expert_waypoints": self.runtime_config.target_spec.expert_waypoint_shape,
            "expert_target_speed": self.runtime_config.target_spec.expert_target_speed_shape,
        }
        for key, expected in expected_shapes.items():
            if tuple(self.arrays[key].shape[1:]) != tuple(expected):
                raise ValueError(f"Dataset key {key!r} expected sample shape {expected}, got {self.arrays[key].shape[1:]}")
        for key in ("bev_target_available", "waypoint_target_available"):
            values = np.asarray(self.arrays[key], dtype=np.float32)
            if values.ndim != 1:
                raise ValueError(f"Dataset key {key!r} must be [N]")
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"Dataset key {key!r} must be in [0, 1]")
