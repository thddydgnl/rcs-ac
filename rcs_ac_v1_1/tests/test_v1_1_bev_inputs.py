from __future__ import annotations

import inspect

import numpy as np
import torch

from rcs_ac_v1_1.models.bev_policy import RCSACV11BEVPolicy
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, DEFAULT_BEV_CHANNELS, GT_BEV_TARGET_CHANNELS
from rcs_ac_v1_1.runtime.observation import (
    build_v11_policy_observation_spec,
    build_v11_target_spec,
    flatten_observation_dict,
    split_flat_observation,
)
from rcs_ac_v1_1.runtime.pid_controller import V11WaypointPIDController


def test_bev_spec_grid_shape() -> None:
    spec = BEVSpec()
    assert spec.shape_hw == (128, 128)
    assert spec.channel_shape(5) == (5, 128, 128)


def test_v11_policy_observation_shapes_and_flat_dim() -> None:
    spec = build_v11_policy_observation_spec()
    summary = {item.key: item.shape for item in spec.modalities}
    assert summary["rgb"] == (3, 90, 160)
    assert summary["lidar_bev"] == (5, 128, 128)
    assert summary["route_bev"] == (4, 128, 128)
    assert summary["ego_bev"] == (3, 128, 128)
    assert summary["ego_scalar"] == (6,)
    assert summary["route_scalar"] == (10,)
    assert summary["sensor_health"] == (15,)
    assert spec.total_dim == 239839


def test_flatten_and_split_preserve_v11_shapes() -> None:
    spec = build_v11_policy_observation_spec()
    observation = {
        item.key: np.zeros(item.shape, dtype=np.float32)
        for item in spec.modalities
    }
    flat = flatten_observation_dict(observation, spec)
    split = split_flat_observation(torch.from_numpy(flat).unsqueeze(0), spec)
    for item in spec.modalities:
        assert tuple(split[item.key].shape) == (1,) + item.shape


def test_v11_target_spec_keeps_gt_bev_target_out_of_policy_inputs() -> None:
    policy_spec = build_v11_policy_observation_spec()
    target_spec = build_v11_target_spec()
    assert "gt_bev_target" not in policy_spec.keys
    assert "actor_positions" not in policy_spec.keys
    assert "gt_bboxes" not in policy_spec.keys
    assert target_spec.gt_bev_shape == (len(GT_BEV_TARGET_CHANNELS), 128, 128)
    assert target_spec.expert_waypoint_shape == (4, 2)


def test_v11_policy_forward_shapes_and_fusion_weights() -> None:
    model = RCSACV11BEVPolicy()
    obs = torch.zeros((2, model.obs_spec.total_dim), dtype=torch.float32)
    outputs = model(obs)
    assert tuple(outputs["waypoints"].shape) == (2, 4, 2)
    assert tuple(outputs["target_speed_kmh"].shape) == (2, 1)
    assert tuple(outputs["value"].shape) == (2, 1)
    assert tuple(outputs["bev_fusion_weights"].shape) == (2, 4)
    assert tuple(outputs["bev_logits"].shape) == (2, 5, 128, 128)
    torch.testing.assert_close(outputs["bev_fusion_weights"].sum(dim=-1), torch.ones(2))


def test_pid_controller_output_range() -> None:
    controller = V11WaypointPIDController()
    control = controller.control(
        [(3.0, 0.2), (6.0, 0.4), (9.0, 0.5), (12.0, 0.6)],
        target_speed_kmh=10.0,
        current_speed_kmh=8.0,
        dt=0.05,
    )
    assert -1.0 <= control["steer"] <= 1.0
    assert 0.0 <= control["throttle"] <= 0.5
    assert 0.0 <= control["brake"] <= 0.4


def test_no_gt_bev_in_policy_forward_signature() -> None:
    parameters = set(inspect.signature(RCSACV11BEVPolicy.forward).parameters)
    forbidden = {"gt_bev_target", "actor_positions", "gt_bboxes", "carla_actor_state"}
    assert parameters.isdisjoint(forbidden)


def test_default_target_channel_count_matches_config() -> None:
    assert DEFAULT_BEV_CHANNELS.target_channels == len(GT_BEV_TARGET_CHANNELS)
