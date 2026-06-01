from __future__ import annotations

import numpy as np

from rcs_ac_v1_1.adapters.carla_lidar_adapter import coerce_lidar_points, extract_lidar_points_ego, lidar_bev_debug
from rcs_ac_v1_1.adapters.carla_observation_adapter import build_v11_flat_obs_from_carla_step
from rcs_ac_v1_1.adapters.carla_route_adapter import build_route_context_from_info, build_route_context_from_obs_info, world_xy_to_ego_xy
from rcs_ac_v1_1.runtime.bev_spec import BEVSpec
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config


def test_world_xy_to_ego_xy_uses_left_positive_y() -> None:
    points = world_xy_to_ego_xy(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw_rad=0.0,
    )
    assert np.allclose(points[0], [1.0, -0.0])
    assert np.allclose(points[1], [0.0, -1.0])


def test_build_route_context_from_target_info() -> None:
    context, debug = build_route_context_from_info({"target_x_local": 8.0, "target_y_local": 1.5, "heading_error": 10.0})
    assert context is not None
    assert context.route_points_ego.shape == (2, 2)
    assert context.target_point_ego == (8.0, 1.5)
    assert np.isclose(context.heading_error_rad, np.deg2rad(10.0))
    assert int(debug["adapter_route_available"]) == 1


def test_build_route_context_can_fallback_to_obs_route_vector() -> None:
    context, debug = build_route_context_from_obs_info({"route": np.asarray([0.2, 0.1] + [0.0] * 8)}, {})
    assert context is not None
    assert np.allclose(context.target_point_ego, (8.0, 1.2))
    assert debug["adapter_route_source"] == "obs_route_vector"


def test_route_context_uses_lookahead_when_current_target_is_behind() -> None:
    route = np.asarray([-0.5, -1.0, 3.0, -0.2, 6.0, -0.3, -0.3, -0.1, 0.0, 0.0], dtype=np.float32)
    context, debug = build_route_context_from_obs_info({"route": route}, {"target_x_local": -0.5, "target_y_local": -1.0})
    assert context is not None
    assert context.route_points_ego.shape[0] >= 2
    assert context.target_point_ego is not None
    assert context.target_point_ego[0] > 0.0
    assert debug["adapter_route_source"] == "obs_route_vector_lookahead"


def test_lidar_adapter_accepts_flat_xyzi_buffer() -> None:
    raw = np.asarray([1.0, 0.0, 0.1, 0.5, 2.0, 1.0, 0.2, 0.5], dtype=np.float32)
    points = coerce_lidar_points(raw)
    assert points is not None
    assert points.shape == (2, 3)
    debug = lidar_bev_debug(points, BEVSpec())
    assert debug["lidar_point_count"] == 2
    assert debug["lidar_inside_bev_ratio"] > 0.0


def test_lidar_adapter_detects_legacy_bev_without_treating_as_points() -> None:
    points, debug = extract_lidar_points_ego({"lidar": np.zeros((1, 64, 64), dtype=np.float32)})
    assert points is None
    assert debug["adapter_lidar_legacy_bev_detected"] == 1


def test_carla_observation_adapter_builds_v11_flat_obs() -> None:
    cfg = load_v11_runtime_config("configs/v1_1/full_bev_smoke.yaml")
    obs = {
        "rgb": np.zeros((90, 160, 3), dtype=np.float32),
        "lidar": np.asarray([[4.0, 0.0, 0.1], [5.0, 1.0, 0.2]], dtype=np.float32),
    }
    info = {
        "speed_kmh": 5.0,
        "target_x_local": 8.0,
        "target_y_local": 0.5,
        "route_progress": 0.1,
    }
    flat, packet, ego, route, debug = build_v11_flat_obs_from_carla_step(
        obs=obs,
        info=info,
        runtime_config=cfg,
        previous_control={"steer": 0.1, "throttle": 0.2, "brake": 0.0, "longitudinal": 0.2},
    )
    assert flat.shape == (cfg.observation_spec.total_dim,)
    assert np.isfinite(flat).all()
    assert packet.rgb_front is not None
    assert ego.previous_steer == 0.1
    assert route is not None
    assert debug["adapter_obs_flat_dim"] == cfg.observation_spec.total_dim
