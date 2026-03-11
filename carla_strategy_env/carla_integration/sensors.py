"""Sensor wrappers for front-camera capture and collision bookkeeping."""

from __future__ import annotations

from collections import deque
import logging
from typing import Any

import numpy as np

from config import SensorConfig
from carla_strategy_env.carla_integration.client import CarlaSession


LOGGER = logging.getLogger(__name__)


class RgbCameraSensor:
    def __init__(self, session: CarlaSession, vehicle: Any, config: SensorConfig) -> None:
        self._session = session
        self._vehicle = vehicle
        self._config = config
        self._sensor: Any | None = None
        self._latest_frame = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)

    def spawn(self) -> None:
        import carla

        if self._session.world is None:
            raise RuntimeError("CARLA session is not connected.")

        blueprint = self._session.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self._config.image_width))
        blueprint.set_attribute("image_size_y", str(self._config.image_height))
        blueprint.set_attribute("fov", str(self._config.image_fov))
        transform = carla.Transform(carla.Location(x=self._config.camera_x, z=self._config.camera_z))
        self._sensor = self._session.world.spawn_actor(blueprint, transform, attach_to=self._vehicle)
        self._session.register_actor(self._sensor)
        self._sensor.listen(self._on_image)
        LOGGER.info("RGB camera sensor spawned.")

    def _on_image(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self._latest_frame = array[:, :, ::-1]

    def get_observation(self) -> np.ndarray:
        return self._latest_frame.copy()


class CollisionSensor:
    def __init__(self, session: CarlaSession, vehicle: Any, config: SensorConfig) -> None:
        self._session = session
        self._vehicle = vehicle
        self._config = config
        self._sensor: Any | None = None
        self._collision_events: deque[float] = deque(maxlen=20)

    def spawn(self) -> None:
        import carla

        if self._session.world is None:
            raise RuntimeError("CARLA session is not connected.")

        blueprint = self._session.world.get_blueprint_library().find("sensor.other.collision")
        blueprint.set_attribute("sensor_tick", str(self._config.collision_sensor_tick))
        self._sensor = self._session.world.spawn_actor(blueprint, carla.Transform(), attach_to=self._vehicle)
        self._session.register_actor(self._sensor)
        self._sensor.listen(self._on_collision)
        LOGGER.info("Collision sensor spawned.")

    def _on_collision(self, event: Any) -> None:
        impulse = event.normal_impulse
        magnitude = float((impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5)
        self._collision_events.append(magnitude)
        LOGGER.warning("Collision detected: impulse=%.3f", magnitude)

    def has_collision(self) -> bool:
        return bool(self._collision_events)

    def clear(self) -> None:
        self._collision_events.clear()
