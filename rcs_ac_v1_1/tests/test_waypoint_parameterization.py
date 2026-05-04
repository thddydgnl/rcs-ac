from __future__ import annotations

import torch

from rcs_ac_v1_1.models.bev_policy import RCSACV11BEVPolicy, parameterize_waypoints


def test_parameterized_waypoint_x_is_positive_and_increasing() -> None:
    raw = torch.randn(16, 8)
    waypoints = parameterize_waypoints(
        raw,
        distances_m=(2.0, 4.0, 6.0, 8.0),
        gap_residual_scale_m=0.75,
        min_gap_m=0.75,
        max_lateral_m=6.0,
    )
    assert torch.all(waypoints[:, :, 0] > 0.0)
    assert torch.all(torch.diff(waypoints[:, :, 0], dim=1) >= 0.75)


def test_parameterized_waypoint_y_is_bounded() -> None:
    raw = torch.full((4, 8), 1000.0)
    waypoints = parameterize_waypoints(
        raw,
        distances_m=(2.0, 4.0, 6.0, 8.0),
        gap_residual_scale_m=0.75,
        min_gap_m=0.75,
        max_lateral_m=6.0,
    )
    assert torch.max(torch.abs(waypoints[:, :, 1])).item() <= 6.0


def test_policy_forward_uses_constrained_waypoints() -> None:
    model = RCSACV11BEVPolicy()
    obs = torch.zeros((2, model.obs_spec.total_dim), dtype=torch.float32)
    outputs = model(obs)
    waypoints = outputs["waypoints"]
    assert torch.all(waypoints[:, :, 0] > 0.0)
    assert torch.all(torch.diff(waypoints[:, :, 0], dim=1) >= 0.75)
    assert torch.max(torch.abs(waypoints[:, :, 1])).item() <= 6.0
