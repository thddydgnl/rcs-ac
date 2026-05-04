from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PIDState:
    integral: float = 0.0
    previous_error: float = 0.0


@dataclass(frozen=True)
class PIDControllerConfig:
    lateral_waypoint_index: int = 2
    steer_kp: float = 0.8
    steer_ki: float = 0.0
    steer_kd: float = 0.08
    speed_kp: float = 0.08
    speed_ki: float = 0.0
    speed_kd: float = 0.02
    max_throttle: float = 0.5
    max_brake: float = 0.4


class V11WaypointPIDController:
    def __init__(self, config: PIDControllerConfig | None = None) -> None:
        self.config = config or PIDControllerConfig()
        self.lateral = PIDState()
        self.longitudinal = PIDState()

    def reset(self) -> None:
        self.lateral = PIDState()
        self.longitudinal = PIDState()

    def control(
        self,
        waypoints: list[tuple[float, float]],
        *,
        target_speed_kmh: float,
        current_speed_kmh: float,
        dt: float,
    ) -> dict[str, float]:
        if not waypoints:
            raise ValueError("waypoints must not be empty")
        index = min(max(0, int(self.config.lateral_waypoint_index)), len(waypoints) - 1)
        x, y = waypoints[index]
        heading_error = math.atan2(float(y), max(1.0e-6, float(x)))
        steer = self._pid(heading_error, self.lateral, self.config.steer_kp, self.config.steer_ki, self.config.steer_kd, dt)
        speed_error = float(target_speed_kmh) - float(current_speed_kmh)
        longitudinal = self._pid(
            speed_error,
            self.longitudinal,
            self.config.speed_kp,
            self.config.speed_ki,
            self.config.speed_kd,
            dt,
        )
        throttle = max(0.0, min(float(self.config.max_throttle), longitudinal))
        brake = max(0.0, min(float(self.config.max_brake), -longitudinal))
        return {
            "steer": max(-1.0, min(1.0, steer)),
            "throttle": throttle,
            "brake": brake,
            "selected_waypoint_x": float(x),
            "selected_waypoint_y": float(y),
            "speed_error_kmh": speed_error,
        }

    @staticmethod
    def _pid(error: float, state: PIDState, kp: float, ki: float, kd: float, dt: float) -> float:
        safe_dt = max(float(dt), 1.0e-6)
        state.integral += float(error) * safe_dt
        derivative = (float(error) - state.previous_error) / safe_dt
        state.previous_error = float(error)
        return float(kp) * float(error) + float(ki) * state.integral + float(kd) * derivative


@dataclass(frozen=True)
class PurePursuitControllerConfig:
    wheel_base_m: float = 2.85
    min_lookahead_m: float = 3.5
    max_lookahead_m: float = 12.0
    lookahead_base_m: float = 3.0
    lookahead_gain_per_mps: float = 0.45
    max_steer: float = 1.0
    steer_gain: float = 1.0
    steer_lowpass_tau_s: float = 0.25
    max_steer_rate_per_s: float = 1.2
    max_abs_target_y_m: float = 8.0
    min_valid_target_x_m: float = 0.5
    min_valid_path_length_m: float = 2.0
    speed_kp: float = 0.08
    speed_ki: float = 0.0
    speed_kd: float = 0.02
    max_throttle: float = 0.5
    max_brake: float = 0.4
    uncertainty_slowdown_threshold: float = 0.45
    uncertainty_stop_threshold: float = 0.85
    max_speed_under_high_uncertainty_kmh: float = 8.0
    min_uncertain_speed_kmh: float = 4.0
    invalid_path_target_speed_kmh: float = 0.0
    curvature_speed_cap_enabled: bool = True
    curvature_speed_gain: float = 2.2
    min_curve_speed_kmh: float = 6.0
    route_fallback_lookahead_m: float = 6.0
    route_fallback_y_m: float = 0.0


def sanitize_waypoints(
    waypoints: list[tuple[float, float]] | np.ndarray,
    config: PurePursuitControllerConfig,
) -> tuple[np.ndarray, bool]:
    arr = np.asarray(waypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return np.asarray([[config.route_fallback_lookahead_m, config.route_fallback_y_m]], dtype=np.float32), False
    finite_mask = np.isfinite(arr).all(axis=1)
    x_valid = arr[:, 0] >= float(config.min_valid_target_x_m)
    y_valid = np.abs(arr[:, 1]) <= float(config.max_abs_target_y_m)
    filtered = arr[finite_mask & x_valid & y_valid]
    if filtered.shape[0] == 0:
        return np.asarray([[config.route_fallback_lookahead_m, config.route_fallback_y_m]], dtype=np.float32), False
    order = np.argsort(filtered[:, 0], kind="stable")
    return filtered[order].astype(np.float32, copy=False), True


def compute_lookahead_m(speed_kmh: float, config: PurePursuitControllerConfig) -> float:
    speed_mps = max(0.0, float(speed_kmh) / 3.6)
    lookahead = float(config.lookahead_base_m) + float(config.lookahead_gain_per_mps) * speed_mps
    return float(np.clip(lookahead, float(config.min_lookahead_m), float(config.max_lookahead_m)))


def _path_length_m(waypoints: np.ndarray) -> float:
    if waypoints.shape[0] <= 1:
        return float(np.linalg.norm(waypoints[0]))
    anchored = np.vstack([np.zeros((1, 2), dtype=np.float32), waypoints])
    return float(np.linalg.norm(np.diff(anchored, axis=0), axis=1).sum())


def interpolate_path_by_arc_length(waypoints: np.ndarray, lookahead_m: float) -> tuple[float, float]:
    anchored = np.vstack([np.zeros((1, 2), dtype=np.float32), np.asarray(waypoints, dtype=np.float32)])
    segment = np.diff(anchored, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    total = float(lengths.sum())
    if total <= 1.0e-6:
        return float(waypoints[-1, 0]), float(waypoints[-1, 1])

    target = min(max(0.0, float(lookahead_m)), total)
    cumulative = 0.0
    for idx, length in enumerate(lengths):
        length_f = float(length)
        if cumulative + length_f >= target and length_f > 1.0e-6:
            alpha = (target - cumulative) / length_f
            point = anchored[idx] + alpha * segment[idx]
            return float(point[0]), float(point[1])
        cumulative += length_f
    return float(anchored[-1, 0]), float(anchored[-1, 1])


def compute_pure_pursuit_steer(
    target_x_m: float,
    target_y_m: float,
    config: PurePursuitControllerConfig,
) -> tuple[float, float]:
    ld2 = max(float(target_x_m) ** 2 + float(target_y_m) ** 2, 1.0e-6)
    curvature = 2.0 * float(target_y_m) / ld2
    steer_rad = math.atan(float(config.wheel_base_m) * curvature)
    steer = float(config.steer_gain) * steer_rad
    return float(np.clip(steer, -float(config.max_steer), float(config.max_steer))), float(curvature)


def lowpass_steer(raw_steer: float, previous_steer: float, dt: float, tau_s: float) -> float:
    if float(tau_s) <= 0.0:
        return float(raw_steer)
    alpha = float(dt) / (float(tau_s) + float(dt))
    return float(previous_steer) + alpha * (float(raw_steer) - float(previous_steer))


def rate_limit_steer(steer: float, previous_steer: float, dt: float, max_rate_per_s: float) -> float:
    max_delta = max(0.0, float(max_rate_per_s)) * max(float(dt), 1.0e-6)
    delta = float(np.clip(float(steer) - float(previous_steer), -max_delta, max_delta))
    return float(previous_steer) + delta


def curvature_speed_cap_kmh(curvature: float, config: PurePursuitControllerConfig) -> float:
    if not bool(config.curvature_speed_cap_enabled):
        return float("inf")
    cap = float(config.max_speed_under_high_uncertainty_kmh) + float(config.curvature_speed_gain) / (
        abs(float(curvature)) + 1.0e-3
    )
    return max(float(config.min_curve_speed_kmh), float(cap))


def apply_uncertainty_speed_cap(
    target_speed_kmh: float,
    uncertainty: float,
    config: PurePursuitControllerConfig,
) -> float:
    target = max(0.0, float(target_speed_kmh))
    uncertainty_f = float(np.clip(uncertainty, 0.0, 1.0))
    if uncertainty_f < float(config.uncertainty_slowdown_threshold):
        return target
    if uncertainty_f >= float(config.uncertainty_stop_threshold):
        return min(target, float(config.min_uncertain_speed_kmh))
    span = max(
        float(config.uncertainty_stop_threshold) - float(config.uncertainty_slowdown_threshold),
        1.0e-6,
    )
    ratio = (uncertainty_f - float(config.uncertainty_slowdown_threshold)) / span
    high_cap = float(config.max_speed_under_high_uncertainty_kmh)
    stop_cap = float(config.min_uncertain_speed_kmh)
    cap = (1.0 - ratio) * high_cap + ratio * stop_cap
    return min(target, cap)


def compute_longitudinal_control(
    *,
    target_speed_kmh: float,
    current_speed_kmh: float,
    dt: float,
    state: PIDState,
    config: PurePursuitControllerConfig,
) -> tuple[float, float, float]:
    speed_error = float(target_speed_kmh) - float(current_speed_kmh)
    safe_dt = max(float(dt), 1.0e-6)
    state.integral += speed_error * safe_dt
    derivative = (speed_error - state.previous_error) / safe_dt
    state.previous_error = speed_error
    longitudinal = (
        float(config.speed_kp) * speed_error
        + float(config.speed_ki) * state.integral
        + float(config.speed_kd) * derivative
    )
    throttle = float(np.clip(longitudinal, 0.0, float(config.max_throttle)))
    brake = float(np.clip(-longitudinal, 0.0, float(config.max_brake)))
    return throttle, brake, speed_error


class V11SpeedAdaptivePurePursuitController:
    def __init__(self, config: PurePursuitControllerConfig | None = None) -> None:
        self.config = config or PurePursuitControllerConfig()
        self.longitudinal = PIDState()
        self.previous_steer = 0.0

    def reset(self) -> None:
        self.longitudinal = PIDState()
        self.previous_steer = 0.0

    def control(
        self,
        waypoints: list[tuple[float, float]] | np.ndarray,
        *,
        target_speed_kmh: float,
        current_speed_kmh: float,
        dt: float,
        route_fallback_waypoints: list[tuple[float, float]] | np.ndarray | None = None,
        uncertainty: float = 0.0,
    ) -> dict[str, object]:
        config = self.config
        clean_waypoints, waypoint_valid = sanitize_waypoints(waypoints, config)
        used_route_fallback = False
        used_synthetic_fallback = not waypoint_valid
        if not waypoint_valid and route_fallback_waypoints is not None:
            route_waypoints, route_valid = sanitize_waypoints(route_fallback_waypoints, config)
            if route_valid:
                clean_waypoints = route_waypoints
                waypoint_valid = True
                used_route_fallback = True
                used_synthetic_fallback = False
        path_valid = _path_length_m(clean_waypoints) >= float(config.min_valid_path_length_m)
        if not path_valid:
            clean_waypoints = np.asarray([[config.route_fallback_lookahead_m, config.route_fallback_y_m]], dtype=np.float32)
            used_synthetic_fallback = True

        lookahead = compute_lookahead_m(current_speed_kmh, config)
        target_x, target_y = interpolate_path_by_arc_length(clean_waypoints, lookahead)
        raw_steer, curvature = compute_pure_pursuit_steer(target_x, target_y, config)
        filtered = lowpass_steer(raw_steer, self.previous_steer, dt, config.steer_lowpass_tau_s)
        steer = rate_limit_steer(filtered, self.previous_steer, dt, config.max_steer_rate_per_s)
        steer = float(np.clip(steer, -float(config.max_steer), float(config.max_steer)))
        self.previous_steer = steer

        if used_synthetic_fallback:
            capped_speed = min(float(target_speed_kmh), float(config.invalid_path_target_speed_kmh))
        else:
            capped_speed = apply_uncertainty_speed_cap(target_speed_kmh, uncertainty, config)
        curve_cap = curvature_speed_cap_kmh(curvature, config)
        capped_speed = min(capped_speed, curve_cap)
        throttle, brake, speed_error = compute_longitudinal_control(
            target_speed_kmh=capped_speed,
            current_speed_kmh=current_speed_kmh,
            dt=dt,
            state=self.longitudinal,
            config=config,
        )
        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "debug": {
                "waypoint_valid": bool(waypoint_valid),
                "path_valid": bool(path_valid),
                "used_route_fallback": bool(used_route_fallback),
                "used_synthetic_fallback": bool(used_synthetic_fallback),
                "lookahead_m": float(lookahead),
                "target_x_m": float(target_x),
                "target_y_m": float(target_y),
                "raw_steer": float(raw_steer),
                "smoothed_steer": float(filtered),
                "rate_limited_steer": float(steer),
                "curvature": float(curvature),
                "uncertainty": float(uncertainty),
                "target_speed_capped_kmh": float(capped_speed),
                "speed_cap_kmh": float(capped_speed),
                "speed_error_kmh": float(speed_error),
            },
        }
