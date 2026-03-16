from __future__ import annotations

import csv
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

import carla
import numpy as np

from env.route_manager import RouteManager
from env.sensors import CollisionSensor, LidarSensor, LaneInvasionSensor, RgbCamera, SensorSyncError
from utils.common import angle_diff_deg, compute_speed_kmh, ensure_dir, timestamp_string


@dataclass
class StepResult:
    obs: Dict[str, Any]
    reward: Optional[float]
    done: bool
    info: Dict[str, Any]


class CarlaEnv:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        town: str = "Town04",
        output_dir: str = "outputs/carla_run",
        image_every: int = 10,
        client_timeout: float = 10.0,
        route_length: int = 40,
        route_spacing: float = 3.0,
        stuck_speed_threshold_kmh: float = 0.5,
        stuck_time_seconds: float = 3.0,
    ) -> None:
        self.host = host
        self.port = port
        self.town = town
        self.output_dir = ensure_dir(output_dir)
        self.image_every = max(1, int(image_every))
        self.client_timeout = client_timeout
        self.route_length = route_length
        self.route_spacing = route_spacing
        self.stuck_speed_threshold_kmh = stuck_speed_threshold_kmh
        self.stuck_time_seconds = stuck_time_seconds

        self.images_dir = ensure_dir(os.path.join(self.output_dir, "rgb"))
        self.logs_path = os.path.join(self.output_dir, "logs.csv")

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.spectator: Optional[carla.Actor] = None
        self.sensors: Dict[str, Any] = {}
        self.route_manager: Optional[RouteManager] = None
        self.csv_file: Optional[Any] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        self.step_count = 0
        self.previous_settings: Optional[carla.WorldSettings] = None
        self.last_controls: Deque[carla.VehicleControl] = deque(maxlen=10)
        self.low_speed_steps = 0
        self.fixed_delta_seconds = 0.05

    def reset(self) -> Dict[str, Any]:
        self.close()
        self._connect_and_load_world()
        self._spawn_vehicle()
        self._setup_sensors()
        self._open_log()
        self.spectator = self.world.get_spectator() if self.world is not None else None

        initial_frame = self.world.tick()
        self._setup_route()
        self._update_spectator()
        sensor_snapshot = self._collect_sensor_data(initial_frame)
        self.step_count = 0
        self.low_speed_steps = 0

        observation = self._build_observation(sensor_snapshot)
        info = self._build_info(
            frame=initial_frame,
            sensor_snapshot=sensor_snapshot,
            control=carla.VehicleControl(),
            observation=observation,
        )
        self._write_log_row(frame=initial_frame, control=carla.VehicleControl(), info=info, image_path="")

        return {
            "frame": initial_frame,
            "route_waypoints": len(self.route_manager.route),
            "route_progress": info["route_progress"],
        }

    def step(self, action: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if self.vehicle is None or self.world is None or self.route_manager is None:
            raise RuntimeError("Environment must be reset before stepping.")

        control = self._select_control(action)
        self.vehicle.apply_control(control)
        self.last_controls.append(control)

        frame = self.world.tick()
        self._update_spectator()
        sensor_snapshot = self._collect_sensor_data(frame)
        observation = self._build_observation(sensor_snapshot)
        info = self._build_info(frame=frame, sensor_snapshot=sensor_snapshot, control=control, observation=observation)
        image_path = ""
        if self.step_count % self.image_every == 0:
            image_path = self._save_rgb_image(sensor_snapshot["rgb"], frame)

        self._write_log_row(frame=frame, control=control, info=info, image_path=image_path)
        self.step_count += 1

        result = StepResult(
            obs=observation,
            reward=None,
            done=bool(info["reached_goal"] or info["collision"] or info["stuck"] or info["offroad"]),
            info=info,
        )

        return {
            "obs": result.obs,
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
            "frame": info["frame"],
            "speed_kmh": info["speed_kmh"],
            "target_waypoint_index": info["target_waypoint_index"],
            "route_completed": info["reached_goal"],
            "collision": info["collision"],
            "lane_invasion": info["lane_invasion"],
            "image_path": image_path,
            "lidar_points": info["lidar_points"],
            "route_progress": info["route_progress"],
            "lane_offset": info["lane_offset"],
            "heading_error": info["heading_error"],
            "stuck": info["stuck"],
            "offroad": info["offroad"],
        }

    def close(self) -> None:
        if self.vehicle is not None:
            try:
                self.vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            except RuntimeError:
                pass

        for sensor in self.sensors.values():
            try:
                sensor.destroy()
            except Exception:
                pass
        self.sensors.clear()

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                pass
            self.vehicle = None

        if self.world is not None and self.previous_settings is not None:
            try:
                self.world.apply_settings(self.previous_settings)
            except RuntimeError:
                pass
            self.previous_settings = None

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.route_manager = None
        self.spectator = None

    def _connect_and_load_world(self) -> None:
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.client_timeout)
        self.world = self.client.load_world(self.town)
        self.map = self.world.get_map()

        self.previous_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

    def _spawn_vehicle(self) -> None:
        if self.world is None or self.map is None:
            raise RuntimeError("World must be loaded before spawning vehicle.")

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        if vehicle_bp is None:
            vehicle_bp = blueprint_library.filter("vehicle.*")[0]

        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in map.")

        self.vehicle = None
        for spawn_transform in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.vehicle.set_autopilot(False)

    def _setup_sensors(self) -> None:
        if self.world is None or self.vehicle is None:
            raise RuntimeError("World and vehicle must exist before sensor setup.")

        self.sensors = {
            "rgb": RgbCamera(self.world, self.vehicle),
            "lidar": LidarSensor(self.world, self.vehicle),
            "collision": CollisionSensor(self.world, self.vehicle),
            "lane_invasion": LaneInvasionSensor(self.world, self.vehicle),
        }

    def _setup_route(self) -> None:
        if self.map is None or self.vehicle is None:
            raise RuntimeError("Map and vehicle must exist before route setup.")

        start_waypoint = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_waypoint is None:
            raise RuntimeError("Could not project ego vehicle to a driving waypoint.")

        self.route_manager = RouteManager(
            world_map=self.map,
            start_waypoint=start_waypoint,
            route_length=self.route_length,
            waypoint_spacing=self.route_spacing,
        )

    def _collect_sensor_data(self, frame: int) -> Dict[str, Any]:
        try:
            rgb = self.sensors["rgb"].get_data(frame)
            lidar = self.sensors["lidar"].get_data(frame)
            collision_events = self.sensors["collision"].get_data(frame)
            lane_events = self.sensors["lane_invasion"].get_data(frame)
        except SensorSyncError as exc:
            raise RuntimeError(f"Sensor synchronization failed at frame {frame}: {exc}") from exc

        return {
            "rgb": rgb,
            "lidar": lidar,
            "lidar_points": int(lidar.shape[0]) if isinstance(lidar, np.ndarray) else 0,
            "collision_events": collision_events,
            "lane_events": lane_events,
        }

    def _select_control(self, action: Optional[Dict[str, float]]) -> carla.VehicleControl:
        if action is not None:
            return carla.VehicleControl(
                throttle=float(max(0.0, min(1.0, action.get("throttle", 0.0)))),
                steer=float(max(-1.0, min(1.0, action.get("steer", 0.0)))),
                brake=float(max(0.0, min(1.0, action.get("brake", 0.0)))),
                hand_brake=bool(action.get("hand_brake", False)),
                reverse=bool(action.get("reverse", False)),
                manual_gear_shift=False,
            )
        return self._compute_heuristic_control()

    def _compute_heuristic_control(self) -> carla.VehicleControl:
        if self.vehicle is None or self.route_manager is None:
            raise RuntimeError("Vehicle and route manager are required for control.")

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        speed_kmh = compute_speed_kmh(self.vehicle)
        lookahead_distance = max(5.0, min(12.0, speed_kmh * 0.35))
        target_waypoint = self.route_manager.get_target_waypoint(vehicle_location, lookahead=lookahead_distance)

        local_target = self._world_to_vehicle_local(vehicle_transform, target_waypoint.transform.location)
        lookahead = max(1.0, math.sqrt(local_target.x ** 2 + local_target.y ** 2))
        alpha = math.atan2(local_target.y, max(local_target.x, 1e-3))
        wheel_base = 2.8
        steer_rad = math.atan2(2.0 * wheel_base * math.sin(alpha), lookahead)
        steer = max(-0.55, min(0.55, steer_rad))

        heading_error = angle_diff_deg(target_waypoint.transform.rotation.yaw, vehicle_transform.rotation.yaw)
        target_speed_kmh = 20.0
        if abs(heading_error) > 20.0:
            target_speed_kmh = 12.0
        if abs(heading_error) > 35.0:
            target_speed_kmh = 8.0

        speed_error = target_speed_kmh - speed_kmh
        throttle = max(0.18, min(0.55, 0.06 * speed_error))
        brake = 0.0

        if abs(heading_error) > 35.0:
            throttle = min(throttle, 0.18)
        if speed_error < -4.0:
            throttle = 0.0
            brake = min(0.5, 0.06 * abs(speed_error))

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    def _build_observation(self, sensor_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if self.vehicle is None or self.route_manager is None:
            raise RuntimeError("Vehicle and route manager are required to build observations.")

        vehicle_transform = self.vehicle.get_transform()
        route_snapshot = self.route_manager.get_debug_snapshot(vehicle_transform)
        return {
            "rgb": sensor_snapshot["rgb"],
            "lidar": sensor_snapshot["lidar"],
            "route_features": route_snapshot,
            "vehicle": {
                "x": vehicle_transform.location.x,
                "y": vehicle_transform.location.y,
                "z": vehicle_transform.location.z,
                "yaw": vehicle_transform.rotation.yaw,
                "speed_kmh": compute_speed_kmh(self.vehicle),
            },
        }

    def _build_info(
        self,
        frame: int,
        sensor_snapshot: Dict[str, Any],
        control: carla.VehicleControl,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.vehicle is None or self.map is None or self.route_manager is None:
            raise RuntimeError("Vehicle, map, and route manager are required to build info.")

        vehicle_transform = self.vehicle.get_transform()
        route_metrics = self.route_manager.build_metrics(vehicle_transform)
        speed_kmh = compute_speed_kmh(self.vehicle)
        collision_events = sensor_snapshot["collision_events"]
        lane_events = sensor_snapshot["lane_events"]
        latest_collision = collision_events[-1] if collision_events else None
        lane_markings = sorted({mark for event in lane_events for mark in event["markings"]})
        offroad = self._is_offroad(vehicle_transform, route_metrics.lane_offset)
        stuck = self._is_stuck(speed_kmh, route_metrics.reached_goal)

        return {
            "timestamp": timestamp_string(),
            "frame": frame,
            "step": self.step_count,
            "speed_kmh": speed_kmh,
            "collision": latest_collision is not None,
            "collision_events": collision_events,
            "collision_actor": "" if latest_collision is None else latest_collision["other_actor"],
            "collision_intensity": 0.0 if latest_collision is None else latest_collision["intensity"],
            "lane_invasion": bool(lane_markings),
            "lane_invasion_events": lane_events,
            "lane_markings": lane_markings,
            "route_progress": route_metrics.route_progress,
            "target_waypoint_index": route_metrics.target_index,
            "lane_offset": route_metrics.lane_offset,
            "heading_error": route_metrics.heading_error,
            "reached_goal": route_metrics.reached_goal,
            "stuck": stuck,
            "offroad": offroad,
            "lidar_points": sensor_snapshot["lidar_points"],
            "control": {
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
            },
            "observation_keys": list(observation.keys()),
        }

    def _is_stuck(self, speed_kmh: float, reached_goal: bool) -> bool:
        if reached_goal:
            self.low_speed_steps = 0
            return False

        if speed_kmh < self.stuck_speed_threshold_kmh:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0

        threshold_steps = max(1, int(self.stuck_time_seconds / self.fixed_delta_seconds))
        return self.low_speed_steps >= threshold_steps

    def _is_offroad(self, vehicle_transform: carla.Transform, lane_offset: float) -> bool:
        if self.map is None:
            return False

        waypoint = self.map.get_waypoint(
            vehicle_transform.location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return True

        lane_half_width = max(0.5, waypoint.lane_width * 0.5)
        return abs(lane_offset) > lane_half_width * 1.2

    def _save_rgb_image(self, image: Any, frame: int) -> str:
        if image is None:
            return ""
        image_name = f"{frame:06d}.png"
        image_path = os.path.join(self.images_dir, image_name)
        image.save_to_disk(image_path)
        return image_path

    def _open_log(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()

        self.csv_file = open(self.logs_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "frame",
                "step",
                "x",
                "y",
                "z",
                "yaw",
                "speed",
                "steer",
                "throttle",
                "brake",
                "target_waypoint_index",
                "route_progress",
                "heading_error",
                "lane_offset",
                "collision",
                "lane_invasion",
                "offroad",
                "stuck",
                "reached_goal",
                "collision_actor",
                "collision_intensity",
                "lane_markings",
                "lidar_points",
                "rgb_image_path",
            ],
        )
        self.csv_writer.writeheader()

    def _write_log_row(self, frame: int, control: carla.VehicleControl, info: Dict[str, Any], image_path: str) -> None:
        if self.csv_writer is None or self.csv_file is None or self.vehicle is None:
            return

        transform = self.vehicle.get_transform()
        row = {
            "timestamp": info["timestamp"],
            "frame": frame,
            "step": info["step"],
            "x": round(transform.location.x, 3),
            "y": round(transform.location.y, 3),
            "z": round(transform.location.z, 3),
            "yaw": round(transform.rotation.yaw, 3),
            "speed": round(info["speed_kmh"], 3),
            "steer": round(control.steer, 4),
            "throttle": round(control.throttle, 4),
            "brake": round(control.brake, 4),
            "target_waypoint_index": info["target_waypoint_index"],
            "route_progress": round(info["route_progress"], 4),
            "heading_error": round(info["heading_error"], 4),
            "lane_offset": round(info["lane_offset"], 4),
            "collision": info["collision"],
            "lane_invasion": info["lane_invasion"],
            "offroad": info["offroad"],
            "stuck": info["stuck"],
            "reached_goal": info["reached_goal"],
            "collision_actor": info["collision_actor"],
            "collision_intensity": round(info["collision_intensity"], 3),
            "lane_markings": "|".join(info["lane_markings"]),
            "lidar_points": info["lidar_points"],
            "rgb_image_path": image_path,
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def _update_spectator(self) -> None:
        if self.spectator is None or self.vehicle is None:
            return

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        forward = vehicle_transform.get_forward_vector()

        spectator_location = carla.Location(
            x=vehicle_location.x - forward.x * 8.0,
            y=vehicle_location.y - forward.y * 8.0,
            z=vehicle_location.z + 3.5,
        )
        spectator_rotation = carla.Rotation(
            pitch=-15.0,
            yaw=vehicle_transform.rotation.yaw,
            roll=0.0,
        )
        self.spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

    def _world_to_vehicle_local(self, vehicle_transform: carla.Transform, world_location: carla.Location) -> carla.Location:
        delta_x = world_location.x - vehicle_transform.location.x
        delta_y = world_location.y - vehicle_transform.location.y
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)

        local_x = math.cos(yaw_rad) * delta_x + math.sin(yaw_rad) * delta_y
        local_y = -math.sin(yaw_rad) * delta_x + math.cos(yaw_rad) * delta_y
        return carla.Location(x=local_x, y=local_y, z=0.0)
