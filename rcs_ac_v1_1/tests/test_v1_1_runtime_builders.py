from __future__ import annotations

import numpy as np
import torch

from rcs_ac_v1_1.models.rgb_bev_head import RGBBEVHead
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, ego_xy_to_bev_row_col
from rcs_ac_v1_1.runtime.ego_bev import build_ego_bev
from rcs_ac_v1_1.runtime.lidar_bev import build_lidar_bev
from rcs_ac_v1_1.runtime.pipeline import build_flat_policy_observation, build_policy_components
from rcs_ac_v1_1.runtime.rgb_bev import build_rgb_bev_unavailable
from rcs_ac_v1_1.runtime.route_bev import build_route_bev
from rcs_ac_v1_1.runtime.scalars import build_ego_scalar, build_route_scalar
from rcs_ac_v1_1.runtime.sensor_health import build_sensor_health
from rcs_ac_v1_1.runtime.state import V11EgoState, V11RouteContext, V11SensorPacket
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config


def _route(maneuver: str = "left") -> V11RouteContext:
    return V11RouteContext(
        route_points_ego=np.asarray([[0.0, 0.0], [4.0, 0.4], [8.0, 0.8]], dtype=np.float32),
        target_point_ego=(6.0, 0.6),
        maneuver=maneuver,
        target_speed_kmh=16.0,
        distance_to_next_turn_m=12.0,
        heading_error_rad=0.1,
    )


def test_sensor_health_shape_range_and_dropouts() -> None:
    health = build_sensor_health(
        rgb_front=np.ones((3, 90, 160), dtype=np.float32) * 0.5,
        lidar_points_ego=np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32),
        route_context=_route(),
        ego_state=V11EgoState(speed_kmh=10.0, yaw_rate_rad_s=0.1),
        camera_dropout=True,
        weather_severity_norm=2.0,
    )
    assert health.shape == (15,)
    assert health.dtype == np.float32
    assert np.all((0.0 <= health) & (health <= 1.0))
    assert health[0] == 0.0
    assert health[12] == 1.0
    assert health[14] == 1.0


def test_scalars_shape_and_safe_invalid_route_default() -> None:
    ego = build_ego_scalar(V11EgoState(speed_kmh=20.0, yaw_rate_rad_s=0.3, previous_steer=1.2))
    route = build_route_scalar(None)
    assert ego.shape == (6,)
    assert route.shape == (10,)
    assert route[3] == 1.0
    assert route[-1] == 0.0


def test_route_bev_channels_and_turn_sign() -> None:
    spec = BEVSpec()
    bev = build_route_bev(_route("right"), spec)
    assert bev.shape == (4, 128, 128)
    assert bev[0].max() > 0.0
    assert bev[1].max() > 0.9
    assert bev[2].min() < 0.0
    assert bev[3].max() > 0.0


def test_ego_bev_contains_origin_and_signed_control() -> None:
    spec = BEVSpec()
    bev = build_ego_bev(V11EgoState(speed_kmh=0.0, yaw_rate_rad_s=0.0, previous_steer=-0.5), spec)
    row, col, valid = ego_xy_to_bev_row_col(0.0, 0.0, spec)
    assert valid
    assert bev.shape == (3, 128, 128)
    assert bev[0, row, col] > 0.0
    assert bev[2, row, col] < 0.0


def test_lidar_bev_occupancy_free_unknown() -> None:
    spec = BEVSpec()
    points = np.asarray([[6.0, 0.0, 0.0], [6.0, 0.0, 0.5]], dtype=np.float32)
    bev = build_lidar_bev(points, spec)
    row, col, valid = ego_xy_to_bev_row_col(6.0, 0.0, spec)
    assert valid
    assert bev.shape == (5, 128, 128)
    assert bev[0, row, col] == 1.0
    assert bev[1][bev[0] > 0].sum() == 0.0
    assert np.all((0.0 <= bev) & (bev <= 1.0))


def test_rgb_bev_placeholder_and_head_shape() -> None:
    spec = BEVSpec()
    placeholder = build_rgb_bev_unavailable(spec)
    assert placeholder.shape == (4, 128, 128)
    assert placeholder[3].sum() == 0.0
    head = RGBBEVHead()
    out = head(torch.zeros((2, 3, 90, 160)))
    assert tuple(out["rgb_bev_logits"].shape) == (2, 4, 128, 128)
    assert tuple(out["rgb_bev"].shape) == (2, 4, 128, 128)


def test_policy_component_pipeline_builds_flat_observation() -> None:
    runtime_config = load_v11_runtime_config("configs/v1_1/full_bev_smoke.yaml")
    packet = V11SensorPacket(
        rgb_front=np.zeros((90, 160, 3), dtype=np.float32),
        lidar_points_ego=np.asarray([[5.0, 0.0, 0.0], [6.0, 1.0, 0.2]], dtype=np.float32),
    )
    ego = V11EgoState(speed_kmh=8.0, yaw_rate_rad_s=0.02)
    components = build_policy_components(
        runtime_config=runtime_config,
        sensor_packet=packet,
        ego_state=ego,
        route_context=_route(),
    )
    assert components["rgb"].shape == (3, 90, 160)
    assert components["rgb_bev"].shape == (4, 128, 128)
    flat = build_flat_policy_observation(
        runtime_config=runtime_config,
        sensor_packet=packet,
        ego_state=ego,
        route_context=_route(),
    )
    assert flat.shape == (runtime_config.observation_spec.total_dim,)
