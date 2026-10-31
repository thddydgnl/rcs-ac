"""Abstract driver interface for delegating low-level control away from the strategist."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DriverStepResult:
    progress_delta: float
    speed_mps: float
    collision: bool
    stuck: bool


class BaseDriverController(ABC):
    @abstractmethod
    def setup(self, vehicle: Any, world: Any, road_map: Any, route: list[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def tick(self) -> DriverStepResult:
        raise NotImplementedError

    @abstractmethod
    def destroy(self) -> None:
        raise NotImplementedError
