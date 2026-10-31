"""Waypoint-following fallback driver when BehaviorAgent is unavailable."""

from __future__ import annotations

from collections import deque
import logging
from typing import Any

from config import RaceConfig
from carla_strategy_env.drivers.base import BaseDriverController, DriverStepResult


LOGGER = logging.getLogger(__name__)


class SimpleWaypointDriver(BaseDriverController):
    def __init__(self, config: RaceConfig) -> None:
        self._config = config
        self._vehicle: Any | None = None
        self._route: list[Any] = []
        self._speed_history: deque[float] = deque(maxlen=config.stuck_tick_limit)
        self._route_index: int = 0
        self._collided = False

    def setup(self, vehicle: Any, world: Any, road_map: Any, route: list[Any]) -> None:
        self._vehicle = vehicle
        self._route = route
        self._route_index = 0
        self._collided = False
        self._speed_history.clear()
        LOGGER.info("Fallback driver configured with %s route waypoints.", len(route))

    def set_collision_state(self, collided: bool) -> None:
        self._collided = collided

    def tick(self) -> DriverStepResult:
        if self._vehicle is None:
            raise RuntimeError("SimpleWaypointDriver.setup() must be called before tick().")

        import carla

        throttle = 0.55
        steer = 0.0
        brake = 0.0

        if self._route:
            waypoint = self._route[min(self._route_index, len(self._route) - 1)]
            vehicle_transform = self._vehicle.get_transform()
            target_location = waypoint.transform.location
            delta_y = target_location.y - vehicle_transform.location.y
            steer = max(-0.5, min(0.5, delta_y * 0.02))
            self._route_index = min(self._route_index + 1, len(self._route) - 1)

        self._vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        velocity = self._vehicle.get_velocity()
        speed_mps = float((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5)
        self._speed_history.append(speed_mps)
        stuck = (
            len(self._speed_history) == self._speed_history.maxlen
            and sum(self._speed_history) / len(self._speed_history) < self._config.stuck_speed_threshold_mps
        )
        progress_delta = 1.0 / max(1, len(self._route))
        return DriverStepResult(
            progress_delta=progress_delta,
            speed_mps=speed_mps,
            collision=self._collided,
            stuck=stuck,
        )

    def destroy(self) -> None:
        self._vehicle = None
        self._route = []
