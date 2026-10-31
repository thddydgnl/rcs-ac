"""Track helpers for Spa map validation, route generation, and lap-progress estimation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackRoute:
    waypoints: list[Any]
    route_length_meters: float


class TrackManager:
    def __init__(self, expected_map_name: str, spa_xodr_path: Path) -> None:
        self.expected_map_name = expected_map_name
        self.spa_xodr_path = spa_xodr_path

    def validate_runtime_map(self, road_map: Any) -> None:
        active_name = getattr(road_map, "name", "")
        if self.expected_map_name.lower() not in active_name.lower():
            raise RuntimeError(
                f"Spa map '{self.expected_map_name}' is not active in CARLA. "
                f"Current map is '{active_name}'. Import and load the map described by '{self.spa_xodr_path}'."
            )

    def build_route(self, road_map: Any) -> TrackRoute:
        import carla

        spawn_points = road_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Cannot build route because the active CARLA map has no spawn points.")

        start = spawn_points[0].location
        current = road_map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Driving)
        if current is None:
            raise RuntimeError("Failed to get a starting waypoint for route generation.")

        waypoints: list[Any] = []
        route_length = 0.0
        visited = set()
        for _ in range(400):
            key = (round(current.transform.location.x, 1), round(current.transform.location.y, 1))
            if key in visited and len(waypoints) > 20:
                break
            visited.add(key)
            waypoints.append(current)
            next_points = current.next(10.0)
            if not next_points:
                break
            nxt = next_points[0]
            route_length += current.transform.location.distance(nxt.transform.location)
            current = nxt

        if len(waypoints) < 10:
            raise RuntimeError("Generated route is too short for a closed-loop strategy environment.")

        LOGGER.info("Track route generated with %s waypoints and %.1f meters.", len(waypoints), route_length)
        return TrackRoute(waypoints=waypoints, route_length_meters=max(route_length, 1.0))
