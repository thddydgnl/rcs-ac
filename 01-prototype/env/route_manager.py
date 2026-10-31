from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import carla

from utils.common import world_to_local_2d


@dataclass
class RouteMetrics:
    target_index: int
    route_progress: float
    lane_offset: float
    heading_error: float
    reached_goal: bool


class RouteManager:
    def __init__(
        self,
        world_map: carla.Map,
        start_waypoint: carla.Waypoint,
        route_length: int = 40,
        waypoint_spacing: float = 3.0,
        goal_radius: float = 3.0,
    ) -> None:
        self.world_map = world_map
        self.route_length = route_length
        self.waypoint_spacing = waypoint_spacing
        self.goal_radius = max(0.0, float(goal_radius))
        self.route: List[carla.Waypoint] = self._generate_route(start_waypoint)
        self.target_index = 1 if len(self.route) > 1 else 0
        self.completed = False

    def _generate_route(self, start_waypoint: carla.Waypoint) -> List[carla.Waypoint]:
        route = [start_waypoint]
        current = start_waypoint

        for _ in range(self.route_length - 1):
            next_candidates = current.next(self.waypoint_spacing)
            if not next_candidates:
                break
            current = self._select_best_candidate(current, next_candidates)
            route.append(current)

        if len(route) < 2:
            raise RuntimeError("Failed to generate a usable waypoint route.")
        return route

    def _select_best_candidate(self, current_waypoint: carla.Waypoint, candidates: List[carla.Waypoint]) -> carla.Waypoint:
        current_yaw = current_waypoint.transform.rotation.yaw
        current_road_id = current_waypoint.road_id
        current_lane_id = current_waypoint.lane_id

        def score(waypoint: carla.Waypoint) -> tuple[float, float]:
            yaw_diff = abs(self._angle_diff_deg(waypoint.transform.rotation.yaw, current_yaw))
            road_penalty = 0.0 if waypoint.road_id == current_road_id else 100.0
            lane_penalty = 0.0 if waypoint.lane_id == current_lane_id else 10.0
            distance = waypoint.transform.location.distance(current_waypoint.transform.location)
            return road_penalty + lane_penalty + yaw_diff, distance

        return min(candidates, key=score)

    def update_progress(self, vehicle_location: carla.Location, reach_threshold: float = 4.0, search_window: int = 6) -> None:
        start_idx = self.target_index
        end_idx = min(len(self.route), self.target_index + search_window)

        best_index = self.target_index
        best_distance = vehicle_location.distance(self.route[self.target_index].transform.location)

        for idx in range(start_idx, end_idx):
            distance = vehicle_location.distance(self.route[idx].transform.location)
            if distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_distance <= reach_threshold:
            self.target_index = best_index
            if self.target_index < len(self.route) - 1:
                next_distance = vehicle_location.distance(self.route[self.target_index + 1].transform.location)
                if next_distance < best_distance:
                    self.target_index += 1

        self.completed = self.reached_goal(vehicle_location)

    def get_target_waypoint(self, vehicle_location: carla.Location, lookahead: float = 5.0) -> carla.Waypoint:
        self.update_progress(vehicle_location)

        for idx in range(self.target_index, len(self.route)):
            waypoint = self.route[idx]
            if vehicle_location.distance(waypoint.transform.location) >= lookahead:
                return waypoint
        return self.route[-1]

    def get_current_target_waypoint(self) -> carla.Waypoint:
        return self.route[self.target_index]

    def get_lookahead_waypoints(self, count: int = 3) -> List[carla.Waypoint]:
        waypoints: List[carla.Waypoint] = []
        for offset in range(count):
            idx = min(self.target_index + offset, len(self.route) - 1)
            waypoints.append(self.route[idx])
        return waypoints

    def vehicle_to_waypoint_relative(self, vehicle_transform: carla.Transform, waypoint: carla.Waypoint) -> tuple[float, float]:
        local_x, local_y = world_to_local_2d(vehicle_transform, waypoint.transform.location)
        distance = float((local_x ** 2 + local_y ** 2) ** 0.5)
        angle = float(np.arctan2(local_y, max(local_x, 1e-6)))
        return distance, angle

    def build_route_vector(self, vehicle_transform: carla.Transform, count: int = 3) -> np.ndarray:
        features: List[float] = []
        for waypoint in self.get_lookahead_waypoints(count=count):
            distance, angle = self.vehicle_to_waypoint_relative(vehicle_transform, waypoint)
            features.extend([distance, angle])
        return np.asarray(features, dtype=np.float32)

    def compute_route_progress(self) -> float:
        if len(self.route) <= 1:
            return 1.0
        return min(1.0, self.target_index / float(len(self.route) - 1))

    def compute_lane_offset(self, vehicle_transform: carla.Transform) -> float:
        reference_waypoint = self.get_current_target_waypoint()
        ref_transform = reference_waypoint.transform
        dx = vehicle_transform.location.x - ref_transform.location.x
        dy = vehicle_transform.location.y - ref_transform.location.y
        yaw_rad = self._deg_to_rad(ref_transform.rotation.yaw)

        right_x = -self._sin(yaw_rad)
        right_y = self._cos(yaw_rad)
        return dx * right_x + dy * right_y

    def compute_heading_error(self, vehicle_transform: carla.Transform) -> float:
        target_waypoint = self.get_current_target_waypoint()
        return self._angle_diff_deg(target_waypoint.transform.rotation.yaw, vehicle_transform.rotation.yaw)

    def reached_goal(self, vehicle_location: carla.Location, threshold: float = 3.0) -> bool:
        effective_threshold = self.goal_radius if threshold == 3.0 else threshold
        final_distance = self.final_goal_distance(vehicle_location)
        return self.target_index >= len(self.route) - 1 and final_distance <= effective_threshold

    def final_goal_distance(self, vehicle_location: carla.Location) -> float:
        return vehicle_location.distance(self.route[-1].transform.location)

    def build_metrics(self, vehicle_transform: carla.Transform) -> RouteMetrics:
        self.update_progress(vehicle_transform.location)
        return RouteMetrics(
            target_index=self.target_index,
            route_progress=self.compute_route_progress(),
            lane_offset=self.compute_lane_offset(vehicle_transform),
            heading_error=self.compute_heading_error(vehicle_transform),
            reached_goal=self.reached_goal(vehicle_transform.location),
        )

    def get_debug_snapshot(self, vehicle_transform: carla.Transform) -> Dict[str, object]:
        metrics = self.build_metrics(vehicle_transform)
        lookahead = self.get_lookahead_waypoints(count=3)
        return {
            "target_waypoint_index": metrics.target_index,
            "route_progress": metrics.route_progress,
            "lane_offset": metrics.lane_offset,
            "heading_error": metrics.heading_error,
            "reached_goal": metrics.reached_goal,
            "lookahead_waypoints": [
                {
                    "x": wp.transform.location.x,
                    "y": wp.transform.location.y,
                    "z": wp.transform.location.z,
                    "yaw": wp.transform.rotation.yaw,
                }
                for wp in lookahead
            ],
        }

    def remaining_waypoints(self) -> int:
        return max(0, len(self.route) - 1 - self.target_index)

    def is_completed(self) -> bool:
        return self.completed

    @staticmethod
    def _angle_diff_deg(target_deg: float, current_deg: float) -> float:
        return (target_deg - current_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _deg_to_rad(degrees: float) -> float:
        return degrees * 3.141592653589793 / 180.0

    @staticmethod
    def _sin(value: float) -> float:
        import math

        return math.sin(value)

    @staticmethod
    def _cos(value: float) -> float:
        import math

        return math.cos(value)
