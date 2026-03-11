"""CARLA client session management with safe world setup and teardown."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from config import CarlaConfig


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CarlaSession:
    config: CarlaConfig
    client: Any | None = None
    world: Any | None = None
    road_map: Any | None = None
    traffic_manager: Any | None = None
    spawned_actors: list[Any] = field(default_factory=list)
    original_settings: Any | None = None

    def connect(self) -> None:
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError("CARLA Python API is not installed or not on PYTHONPATH.") from exc

        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout_seconds)
        self.world = self.client.get_world()
        self.road_map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(self.config.traffic_manager_port)
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = self.config.synchronous_mode
        settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        self.world.apply_settings(settings)
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(self.config.synchronous_mode)

        LOGGER.info(
            "Connected to CARLA %s:%s on map '%s'.",
            self.config.host,
            self.config.port,
            self.road_map.name if self.road_map is not None else "unknown",
        )

    def validate_map(self) -> None:
        if self.world is None or self.road_map is None:
            raise RuntimeError("CARLA session is not connected.")

        active_name = getattr(self.road_map, "name", "")
        if self.config.map_name.lower() not in active_name.lower():
            raise RuntimeError(
                "Expected Spa map to be loaded in CARLA. "
                f"Configured map_name='{self.config.map_name}', active map='{active_name}'. "
                f"Repository assets live at '{self.config.spa_xodr_path}'. Import the Spa map into CARLA before running."
            )

    def tick(self) -> Any:
        if self.world is None:
            raise RuntimeError("CARLA session is not connected.")
        return self.world.tick() if self.config.synchronous_mode else self.world.wait_for_tick()

    def register_actor(self, actor: Any) -> None:
        self.spawned_actors.append(actor)

    def cleanup(self) -> None:
        for actor in reversed(self.spawned_actors):
            try:
                actor.destroy()
            except Exception:
                LOGGER.exception("Failed to destroy actor cleanly.")
        self.spawned_actors.clear()

        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                LOGGER.exception("Failed to restore original world settings.")
