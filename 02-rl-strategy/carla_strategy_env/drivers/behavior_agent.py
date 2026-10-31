"""BehaviorAgent-backed driver adapter with a simple progress and stuck monitor."""

from __future__ import annotations

from collections import deque
import importlib.util
import logging
from typing import Any

from config import RaceConfig
from carla_strategy_env.drivers.base import BaseDriverController, DriverStepResult


LOGGER = logging.getLogger(__name__)


class BehaviorAgentDriver(BaseDriverController):
    def __init__(self, config: RaceConfig, behavior_mode: str = "normal") -> None:
        self._config = config
        self._behavior_mode = behavior_mode
        self._vehicle: Any | None = None
        self._agent: Any | None = None
        self._route: list[Any] = []
        self._progress_per_tick: float = 0.0
        self._speed_history: deque[float] = deque(maxlen=config.stuck_tick_limit)
        self._route_index: int = 0
        self._collided = False

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("agents.navigation.behavior_agent") is not None

    def setup(self, vehicle: Any, world: Any, road_map: Any, route: list[Any]) -> None:
        self._vehicle = vehicle
        self._route = route
        self._progress_per_tick = 1.0 / max(1, len(route))
        self._route_index = 0
        self._collided = False
        self._speed_history.clear()

        try:
            from agents.navigation.behavior_agent import BehaviorAgent
        except ImportError as exc:
            raise RuntimeError(
                "BehaviorAgent could not be imported. Ensure CARLA PythonAPI agents are on PYTHONPATH."
            ) from exc

        self._agent = BehaviorAgent(vehicle, behavior=self._behavior_mode)
        destination = route[-1].transform.location if route else vehicle.get_transform().location
        self._agent.set_destination(destination)
        LOGGER.info("BehaviorAgent driver configured with %s route waypoints.", len(route))

    def set_collision_state(self, collided: bool) -> None:
        self._collided = collided

    def tick(self) -> DriverStepResult:
        if self._vehicle is None or self._agent is None:
            raise RuntimeError("BehaviorAgentDriver.setup() must be called before tick().")

        control = self._agent.run_step()
        self._vehicle.apply_control(control)

        velocity = self._vehicle.get_velocity()
        speed_mps = float((velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5)
        self._speed_history.append(speed_mps)

        if self._route:
            self._route_index = min(self._route_index + 1, len(self._route))
            progress_delta = self._progress_per_tick
        else:
            progress_delta = 0.01

        stuck = (
            len(self._speed_history) == self._speed_history.maxlen
            and sum(self._speed_history) / len(self._speed_history) < self._config.stuck_speed_threshold_mps
        )
        return DriverStepResult(
            progress_delta=progress_delta,
            speed_mps=speed_mps,
            collision=self._collided,
            stuck=stuck,
        )

    def destroy(self) -> None:
        self._vehicle = None
        self._agent = None
        self._route = []
