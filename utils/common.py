from datetime import datetime
from math import sqrt


def ensure_dir(path):
    import os

    os.makedirs(path, exist_ok=True)
    return path


def compute_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    speed_mps = sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed_mps * 3.6


def angle_diff_deg(target_deg, current_deg):
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return diff


def timestamp_string():
    return datetime.now().isoformat(timespec="seconds")
