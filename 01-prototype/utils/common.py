from __future__ import annotations

from datetime import datetime
from math import cos, radians, sin, sqrt


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_dir(path):
    import os

    os.makedirs(path, exist_ok=True)
    return path


def compute_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed_mps * 3.6


def compute_speed_mps(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def angle_diff_deg(target_deg, current_deg):
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return diff


def world_to_local_2d(transform, location) -> tuple[float, float]:
    delta_x = location.x - transform.location.x
    delta_y = location.y - transform.location.y
    yaw_rad = radians(transform.rotation.yaw)

    local_x = cos(yaw_rad) * delta_x + sin(yaw_rad) * delta_y
    local_y = -sin(yaw_rad) * delta_x + cos(yaw_rad) * delta_y
    return float(local_x), float(local_y)


def timestamp_string():
    return datetime.now().isoformat(timespec="seconds")
