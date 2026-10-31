"""Vehicle spawning helpers for the CARLA strategy environment."""

from __future__ import annotations

import logging
import random
from typing import Any

from config import EnvConfig
from carla_strategy_env.carla_integration.client import CarlaSession


LOGGER = logging.getLogger(__name__)


def spawn_ego_vehicle(session: CarlaSession, config: EnvConfig) -> Any:
    if session.world is None:
        raise RuntimeError("CARLA session is not connected.")

    blueprint_library = session.world.get_blueprint_library()
    blueprints = list(blueprint_library.filter(config.vehicle_filter))
    if not blueprints:
        raise RuntimeError(f"No vehicle blueprints matched filter '{config.vehicle_filter}'.")

    preferred = [bp for bp in blueprints if bp.id in config.preferred_vehicle_ids]
    vehicle_bp = random.choice(preferred or blueprints)
    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "svmm_strategist_ego")

    spawn_points = session.world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("The loaded CARLA map has no spawn points.")

    vehicle = None
    for spawn_point in spawn_points:
        vehicle = session.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            session.register_actor(vehicle)
            LOGGER.info("Spawned ego vehicle '%s' at %s.", vehicle_bp.id, spawn_point.location)
            return vehicle

    raise RuntimeError("Failed to spawn ego vehicle at any available spawn point.")
