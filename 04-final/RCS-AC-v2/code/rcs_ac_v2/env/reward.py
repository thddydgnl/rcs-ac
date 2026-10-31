"""Stage 0 reward and continuation contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class RewardConfig:
    route_progress_weight: float = 10.0
    lane_centering_weight: float = 0.2
    speed_error_weight: float = 0.5
    collision_penalty: float = 100.0
    offroad_penalty: float = 50.0
    red_light_penalty: float = 20.0
    stuck_penalty: float = 10.0
    comfort_penalty_weight: float = 0.0
    terminal_failure_penalty: float = 0.0


@dataclass(frozen=True)
class RewardInputs:
    route_progress_prev: float
    route_progress_curr: float
    speed_kmh: float
    target_speed_kmh: float
    lane_offset_m: float
    collision: bool
    offroad: bool
    red_light: bool
    stuck: bool
    done: bool
    comfort_cost: float = 0.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    route_progress: float
    lane_centering: float
    speed_efficiency: float
    collision: float
    offroad: float
    red_light: float
    stuck: float
    comfort: float
    terminal: float

    @property
    def components(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "route_progress": self.route_progress,
                "lane_centering": self.lane_centering,
                "speed_efficiency": self.speed_efficiency,
                "collision": self.collision,
                "offroad": self.offroad,
                "red_light": self.red_light,
                "stuck": self.stuck,
                "comfort": self.comfort,
                "terminal": self.terminal,
            }
        )


def _ensure_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _validate_inputs(inputs: RewardInputs, cfg: RewardConfig) -> None:
    for name in (
        "route_progress_prev",
        "route_progress_curr",
        "speed_kmh",
        "target_speed_kmh",
        "lane_offset_m",
        "comfort_cost",
    ):
        _ensure_finite(name, getattr(inputs, name))
    for name, value in cfg.__dict__.items():
        _ensure_finite(name, value)
    for name in ("collision", "offroad", "red_light", "stuck", "done"):
        if not isinstance(getattr(inputs, name), bool):
            raise ValueError(f"{name} must be bool")
    if inputs.target_speed_kmh < 0.0:
        raise ValueError("target_speed_kmh must be non-negative")
    if inputs.speed_kmh < 0.0:
        raise ValueError("speed_kmh must be non-negative")
    if inputs.comfort_cost < 0.0:
        raise ValueError("comfort_cost must be non-negative")


def compute_reward(inputs: RewardInputs, cfg: RewardConfig | None = None) -> RewardBreakdown:
    cfg = cfg or RewardConfig()
    _validate_inputs(inputs, cfg)

    progress_delta = inputs.route_progress_curr - inputs.route_progress_prev
    route_progress = cfg.route_progress_weight * progress_delta
    lane_centering = -cfg.lane_centering_weight * abs(inputs.lane_offset_m)

    speed_scale = max(inputs.target_speed_kmh, 1.0)
    speed_error = abs(inputs.speed_kmh - inputs.target_speed_kmh) / speed_scale
    speed_efficiency = -cfg.speed_error_weight * speed_error

    collision = -cfg.collision_penalty if inputs.collision else 0.0
    offroad = -cfg.offroad_penalty if inputs.offroad else 0.0
    red_light = -cfg.red_light_penalty if inputs.red_light else 0.0
    stuck = -cfg.stuck_penalty if inputs.stuck else 0.0
    comfort = -cfg.comfort_penalty_weight * inputs.comfort_cost

    terminal_failed = inputs.done and (
        inputs.collision or inputs.offroad or inputs.red_light or inputs.stuck
    )
    terminal = -cfg.terminal_failure_penalty if terminal_failed else 0.0

    total = sum(
        (
            route_progress,
            lane_centering,
            speed_efficiency,
            collision,
            offroad,
            red_light,
            stuck,
            comfort,
            terminal,
        )
    )
    return RewardBreakdown(
        total=float(total),
        route_progress=float(route_progress),
        lane_centering=float(lane_centering),
        speed_efficiency=float(speed_efficiency),
        collision=float(collision),
        offroad=float(offroad),
        red_light=float(red_light),
        stuck=float(stuck),
        comfort=float(comfort),
        terminal=float(terminal),
    )


def compute_continue(done: bool, truncated: bool) -> float:
    if not isinstance(done, bool):
        raise ValueError("done must be bool")
    if not isinstance(truncated, bool):
        raise ValueError("truncated must be bool")
    if truncated:
        return 1.0
    return 0.0 if done else 1.0
