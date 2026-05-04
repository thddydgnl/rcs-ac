from __future__ import annotations

import numpy as np
import pytest
import torch

from rcs_ac_v1_1.data.bev_expert_dataset import BEVExpertDataset
from rcs_ac_v1_1.rl.bev_losses import compute_bev_loss, compute_waypoint_speed_loss
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config
from rcs_ac_v1_1.runtime.gt_bev_targets import V11ActorBox, build_gt_bev_target
from rcs_ac_v1_1.runtime.state import V11RouteContext
from rcs_ac_v1_1.runtime.waypoint_targets import (
    build_distance_based_expert_waypoints,
    build_expert_target_speed,
    validate_waypoint_target,
)


class _Map:
    def is_drivable_ego_xy(self, x_m: float, y_m: float) -> bool:
        return -2.0 <= y_m <= 2.0 and -8.0 <= x_m <= 40.0

    def is_lane_marking_ego_xy(self, x_m: float, y_m: float) -> bool:
        return abs(abs(y_m) - 1.8) < 0.2


def _runtime_config():
    return load_v11_runtime_config("configs/v1_1/full_bev_smoke.yaml")


def test_gt_bev_target_shape_and_obstacle_exclusion() -> None:
    cfg = _runtime_config()
    route = V11RouteContext(np.asarray([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32), (6.0, 0.0), "straight")
    target, available = build_gt_bev_target(
        spec=cfg.bev_spec,
        map_adapter=_Map(),
        actors_ego=[V11ActorBox(center_x_m=5.0, center_y_m=0.0, length_m=4.0, width_m=2.0)],
        route_context=route,
    )
    assert target.shape == (5, 128, 128)
    assert available == 1.0
    assert target[2].max() == 1.0
    assert np.all(target[3][target[2] > 0] == 0.0)
    assert target[4].max() > 0.0


def test_waypoint_target_and_speed_builders() -> None:
    path = np.asarray([[2.0, 0.0], [4.0, 0.5], [8.0, 1.0]], dtype=np.float32)
    waypoints, available = build_distance_based_expert_waypoints(path)
    assert available == 1.0
    assert waypoints.shape == (4, 2)
    assert validate_waypoint_target(waypoints) == 1.0
    speed = build_expert_target_speed(expert_speed_kmh=None, route_target_speed_kmh=50.0)
    assert speed.shape == (1,)
    assert speed[0] == 28.0
    observed_speed = build_expert_target_speed(expert_speed_kmh=0.0, route_target_speed_kmh=0.0, observed_speed_kmh=16.0)
    assert observed_speed[0] == 16.0


def test_curved_waypoint_target_allows_lateral_progress() -> None:
    path = np.asarray([[2.0, 0.0], [4.0, 1.0], [4.5, 3.0], [4.5, 6.0]], dtype=np.float32)
    waypoints, available = build_distance_based_expert_waypoints(path)
    assert available == 1.0
    assert validate_waypoint_target(waypoints) == 1.0


def test_bev_and_waypoint_losses() -> None:
    logits = torch.zeros((2, 5, 8, 8))
    target = torch.zeros((2, 5, 8, 8))
    target[:, :, 2:5, 2:5] = 1.0
    loss, metrics = compute_bev_loss(logits, target, torch.tensor([1.0, 0.0]))
    assert torch.isfinite(loss)
    assert metrics["bev_target_available_rate"] == 0.5
    outputs = {
        "waypoints": torch.tensor([[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]]),
        "target_speed_kmh": torch.tensor([[10.0]]),
    }
    targets = {
        "expert_waypoints": outputs["waypoints"].clone(),
        "expert_target_speed": torch.tensor([[12.0]]),
        "waypoint_target_available": torch.tensor([1.0]),
    }
    wp_loss, wp_metrics = compute_waypoint_speed_loss(outputs, targets)
    assert torch.isfinite(wp_loss)
    assert wp_metrics["waypoint_l1"] == 0.0


def test_bev_expert_dataset_loads_npz(tmp_path) -> None:
    cfg = _runtime_config()
    n = 2
    arrays = {
        "rgb": np.zeros((n, 3, 90, 160), dtype=np.float32),
        "lidar_bev": np.zeros((n, 5, 128, 128), dtype=np.float32),
        "route_bev": np.zeros((n, 4, 128, 128), dtype=np.float32),
        "ego_bev": np.zeros((n, 3, 128, 128), dtype=np.float32),
        "ego_scalar": np.zeros((n, 6), dtype=np.float32),
        "route_scalar": np.zeros((n, 10), dtype=np.float32),
        "sensor_health": np.ones((n, 15), dtype=np.float32),
        "gt_bev_target": np.zeros((n, 5, 128, 128), dtype=np.float32),
        "expert_waypoints": np.tile(np.asarray([[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]], dtype=np.float32), (n, 1, 1)),
        "expert_target_speed": np.ones((n, 1), dtype=np.float32) * 10.0,
        "bev_target_available": np.ones(n, dtype=np.float32),
        "waypoint_target_available": np.ones(n, dtype=np.float32),
        "episode_id": np.arange(n, dtype=np.int32),
        "step": np.arange(n, dtype=np.int32),
    }
    path = tmp_path / "tiny.npz"
    np.savez(path, **arrays)
    dataset = BEVExpertDataset(path, cfg)
    item = dataset[0]
    assert len(dataset) == n
    assert tuple(item["obs_flat"].shape) == (239839,)
    assert tuple(item["targets"]["gt_bev_target"].shape) == (5, 128, 128)


def test_bev_expert_dataset_rejects_missing_key(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, rgb=np.zeros((1, 3, 90, 160), dtype=np.float32))
    with pytest.raises(KeyError):
        BEVExpertDataset(path, _runtime_config())
