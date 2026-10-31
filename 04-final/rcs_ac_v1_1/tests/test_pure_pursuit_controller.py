from __future__ import annotations

from rcs_ac_v1_1.runtime.pid_controller import (
    PurePursuitControllerConfig,
    V11SpeedAdaptivePurePursuitController,
)


def test_pure_pursuit_straight_waypoints_near_zero_steer() -> None:
    controller = V11SpeedAdaptivePurePursuitController()
    control = controller.control(
        [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0)],
        target_speed_kmh=12.0,
        current_speed_kmh=8.0,
        dt=0.05,
    )
    assert abs(float(control["steer"])) < 1.0e-6


def test_pure_pursuit_positive_y_waypoints_negative_carla_steer() -> None:
    controller = V11SpeedAdaptivePurePursuitController()
    control = controller.control(
        [(2.0, 0.5), (4.0, 1.0), (6.0, 1.5), (8.0, 2.0)],
        target_speed_kmh=12.0,
        current_speed_kmh=8.0,
        dt=0.05,
    )
    assert float(control["steer"]) < 0.0


def test_pure_pursuit_rate_limit_respected() -> None:
    config = PurePursuitControllerConfig(max_steer_rate_per_s=0.5, steer_lowpass_tau_s=0.0)
    controller = V11SpeedAdaptivePurePursuitController(config)
    control = controller.control(
        [(2.0, 6.0), (4.0, 6.0), (6.0, 6.0), (8.0, 6.0)],
        target_speed_kmh=12.0,
        current_speed_kmh=8.0,
        dt=0.1,
    )
    assert abs(float(control["steer"])) <= 0.05 + 1.0e-6


def test_invalid_waypoints_use_route_fallback() -> None:
    controller = V11SpeedAdaptivePurePursuitController()
    control = controller.control(
        [(-2.0, 0.0), (0.1, 20.0)],
        target_speed_kmh=12.0,
        current_speed_kmh=8.0,
        dt=0.05,
        route_fallback_waypoints=[(3.0, 0.3), (6.0, 0.6), (9.0, 0.9)],
    )
    debug = control["debug"]
    assert debug["waypoint_valid"] is True
    assert debug["used_route_fallback"] is True
    assert debug["path_valid"] is True
    assert debug["target_x_m"] > 0.0


def test_invalid_waypoints_without_route_fallback_slow_to_stop() -> None:
    controller = V11SpeedAdaptivePurePursuitController()
    control = controller.control(
        [(-2.0, 0.0), (0.1, 20.0)],
        target_speed_kmh=12.0,
        current_speed_kmh=8.0,
        dt=0.05,
    )
    debug = control["debug"]
    assert debug["waypoint_valid"] is False
    assert debug["used_synthetic_fallback"] is True
    assert debug["target_speed_capped_kmh"] == 0.0
    assert float(control["throttle"]) == 0.0
    assert float(control["brake"]) > 0.0
