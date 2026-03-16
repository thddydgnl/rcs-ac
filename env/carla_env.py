from __future__ import annotations

import csv
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

import carla
import numpy as np

from env.reward import RewardCalculator, RewardConfig
from env.route_manager import RouteManager
from env.sensors import CollisionSensor, LidarSensor, LaneInvasionSensor, RgbCamera, SensorSyncError
from env.state_builder import ObservationConfig, StateBuilder
from utils.common import angle_diff_deg, clamp, compute_speed_kmh, compute_speed_mps, ensure_dir, timestamp_string


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: float
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
        max_steps: int = 500,
        observation_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        stuck_speed_threshold_mps: float = 0.3,
        stuck_min_steps: int = 40,
        offroad_min_steps: int = 5,
        offroad_lane_offset_threshold: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.town = town
        self.output_dir = ensure_dir(output_dir)
        self.image_every = max(1, int(image_every))
        self.client_timeout = client_timeout
        self.route_length = route_length
        self.route_spacing = route_spacing
        self.max_steps = max_steps
        self.stuck_speed_threshold_mps = stuck_speed_threshold_mps
        self.stuck_min_steps = stuck_min_steps
        self.offroad_min_steps = offroad_min_steps
        self.offroad_lane_offset_threshold = offroad_lane_offset_threshold

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
        self.fixed_delta_seconds = 0.05

        self.state_builder = StateBuilder(observation_config or ObservationConfig())
        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())

        self.prev_obs: Optional[Dict[str, np.ndarray]] = None
        self.prev_info: Optional[Dict[str, Any]] = None
        self.prev_action: np.ndarray = np.zeros((3,), dtype=np.float32)
        self.low_speed_steps = 0
        self.offroad_steps = 0

    def reset(self) -> Dict[str, Any]:
        self._ensure_world_connected()
        self._reset_episode_actors()
        self._spawn_vehicle()
        self._setup_sensors()
        self._setup_route()
        self._open_log()

        self.spectator = self.world.get_spectator() if self.world is not None else None
        initial_frame = self.world.tick()
        self._update_spectator()
        sensor_snapshot = self._collect_sensor_data(initial_frame)
        self.step_count = 0
        self.low_speed_steps = 0
        self.offroad_steps = 0
        self.prev_action = np.zeros((3,), dtype=np.float32)

        route_metrics = self.route_manager.build_metrics(self.vehicle.get_transform())
        initial_control = carla.VehicleControl()
        observation = self._build_observation(sensor_snapshot, initial_control, route_metrics)
        info = self._build_info(
            frame=initial_frame,
            current_step=0,
            sensor_snapshot=sensor_snapshot,
            action_vector=self.prev_action,
            control=initial_control,
            route_metrics=route_metrics,
            observation=observation,
            reward=0.0,
            reward_breakdown=self._zero_reward_breakdown(),
            done=False,
            done_reason="running",
        )

        self._write_log_row(frame=initial_frame, control=initial_control, info=info, image_path="")
        self.prev_obs = observation
        self.prev_info = info

        return {
            "obs": observation,
            "reward": 0.0,
            "done": False,
            "info": info,
            "frame": initial_frame,
            "route_waypoints": len(self.route_manager.route),
            "route_progress": info["route_progress"],
        }

    def step(self, action: Optional[Any] = None) -> Dict[str, Any]:
        if self.vehicle is None or self.world is None or self.route_manager is None:
            raise RuntimeError("Environment must be reset before stepping.")

        action_vector = self._normalize_action(action) if action is not None else self._compute_heuristic_action()
        control = self._action_to_control(action_vector)
        self.vehicle.apply_control(control)
        self.last_controls.append(control)

        frame = self.world.tick()
        self._update_spectator()
        sensor_snapshot = self._collect_sensor_data(frame)
        current_step = self.step_count + 1
        route_metrics = self.route_manager.build_metrics(self.vehicle.get_transform())
        observation = self._build_observation(sensor_snapshot, control, route_metrics)

        base_info = self._build_info(
            frame=frame,
            current_step=current_step,
            sensor_snapshot=sensor_snapshot,
            action_vector=action_vector,
            control=control,
            route_metrics=route_metrics,
            observation=observation,
            reward=0.0,
            reward_breakdown=self._zero_reward_breakdown(),
            done=False,
            done_reason="running",
        )

        prev_state = None if self.prev_info is None else {"obs": self.prev_obs, "info": self.prev_info}
        current_state = {"obs": observation, "info": base_info}
        reward, reward_breakdown = self.reward_calculator.compute_reward(
            prev_state=prev_state,
            current_state=current_state,
            info=base_info,
            prev_control=self.prev_action,
            current_control=action_vector,
        )

        done_reason = self._resolve_done_reason(base_info, current_step)
        done = done_reason != "running"
        base_info["reward"] = reward
        base_info["reward_breakdown"] = reward_breakdown
        base_info["done"] = done
        base_info["done_reason"] = done_reason
        base_info["max_steps_reached"] = current_step >= self.max_steps

        image_path = ""
        if current_step % self.image_every == 0:
            image_path = self._save_rgb_image(sensor_snapshot["rgb"], frame)

        self._write_log_row(frame=frame, control=control, info=base_info, image_path=image_path)

        self.step_count = current_step
        self.prev_obs = observation
        self.prev_info = base_info
        self.prev_action = action_vector

        result = StepResult(obs=observation, reward=reward, done=done, info=base_info)
        return {
            "obs": result.obs,
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
            "frame": base_info["frame"],
            "speed_kmh": base_info["speed_kmh"],
            "target_waypoint_index": base_info["target_waypoint_index"],
            "route_completed": base_info["reached_goal"],
            "collision": base_info["collision"],
            "lane_invasion": base_info["lane_invasion"],
            "image_path": image_path,
            "lidar_points": base_info["lidar_points"],
            "route_progress": base_info["route_progress"],
            "lane_offset": base_info["lane_offset"],
            "heading_error": base_info["heading_error"],
            "stuck": base_info["stuck"],
            "offroad": base_info["offroad"],
            "done_reason": done_reason,
        }

    def close(self) -> None:
        self._reset_episode_actors()
        self._close_log_file()

        if self.world is not None and self.previous_settings is not None:
            try:
                self.world.apply_settings(self.previous_settings)
            except RuntimeError:
                pass

        self.previous_settings = None
        self.spectator = None
        self.route_manager = None
        self.prev_obs = None
        self.prev_info = None
        self.prev_action = np.zeros((3,), dtype=np.float32)
        self.low_speed_steps = 0
        self.offroad_steps = 0
        self.step_count = 0
        self.last_controls.clear()

        self.map = None
        self.world = None
        self.client = None

    def _ensure_world_connected(self) -> None:
        if self.world is not None and self.map is not None and self.client is not None:
            return
        self._connect_and_load_world()

    def _reset_episode_actors(self) -> None:
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

        self.route_manager = None
        self.prev_obs = None
        self.prev_info = None
        self.prev_action = np.zeros((3,), dtype=np.float32)
        self.low_speed_steps = 0
        self.offroad_steps = 0
        self.step_count = 0
        self.last_controls.clear()

    def _close_log_file(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

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

    def _normalize_action(self, action: Any) -> np.ndarray:
        if isinstance(action, dict):
            vector = np.asarray(
                [
                    action.get("steer", 0.0),
                    action.get("throttle", 0.0),
                    action.get("brake", 0.0),
                ],
                dtype=np.float32,
            )
        else:
            vector = np.asarray(action, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f"Expected action with shape (3,), got {vector.shape}")

        vector = vector.astype(np.float32, copy=False)
        vector[0] = float(clamp(float(vector[0]), -1.0, 1.0))
        vector[1] = float(clamp(float(vector[1]), 0.0, 1.0))
        vector[2] = float(clamp(float(vector[2]), 0.0, 1.0))
        return vector

    def _action_to_control(self, action_vector: np.ndarray) -> carla.VehicleControl:
        return carla.VehicleControl(
            steer=float(action_vector[0]),
            throttle=float(action_vector[1]),
            brake=float(action_vector[2]),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    def _compute_heuristic_action(self) -> np.ndarray:
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
        steer = clamp(float(steer_rad), -0.55, 0.55)

        heading_error = angle_diff_deg(target_waypoint.transform.rotation.yaw, vehicle_transform.rotation.yaw)
        target_speed_kmh = 20.0
        if abs(heading_error) > 20.0:
            target_speed_kmh = 12.0
        if abs(heading_error) > 35.0:
            target_speed_kmh = 8.0

        speed_error = target_speed_kmh - speed_kmh
        throttle = clamp(0.06 * speed_error, 0.18, 0.55)
        brake = 0.0

        if abs(heading_error) > 35.0:
            throttle = min(throttle, 0.18)
        if speed_error < -4.0:
            throttle = 0.0
            brake = min(0.5, 0.06 * abs(speed_error))

        return np.asarray([steer, throttle, brake], dtype=np.float32)

    def _build_observation(self, sensor_snapshot: Dict[str, Any], control: carla.VehicleControl, route_metrics: Any) -> Dict[str, np.ndarray]:
        if self.vehicle is None or self.route_manager is None:
            raise RuntimeError("Vehicle and route manager are required to build observations.")

        observation = self.state_builder.build_observation(
            image=sensor_snapshot["rgb"],
            lidar_points=sensor_snapshot["lidar"],
            vehicle=self.vehicle,
            control=control,
            route_manager=self.route_manager,
            route_metrics=route_metrics,
        )
        return observation

    def _build_info(
        self,
        frame: int,
        current_step: int,
        sensor_snapshot: Dict[str, Any],
        action_vector: np.ndarray,
        control: carla.VehicleControl,
        route_metrics: Any,
        observation: Dict[str, np.ndarray],
        reward: float,
        reward_breakdown: Dict[str, float],
        done: bool,
        done_reason: str,
    ) -> Dict[str, Any]:
        if self.vehicle is None or self.map is None or self.route_manager is None:
            raise RuntimeError("Vehicle, map, and route manager are required to build info.")

        vehicle_transform = self.vehicle.get_transform()
        speed_kmh = compute_speed_kmh(self.vehicle)
        speed_mps = compute_speed_mps(self.vehicle)
        collision_events = sensor_snapshot["collision_events"]
        lane_events = sensor_snapshot["lane_events"]
        latest_collision = collision_events[-1] if collision_events else None
        lane_markings = sorted({mark for event in lane_events for mark in event["markings"]})
        is_low_speed, stuck = self._update_stuck_status(speed_mps, route_metrics.reached_goal)
        offroad_candidate, offroad = self._update_offroad_status(vehicle_transform, route_metrics.lane_offset, route_metrics.reached_goal)

        return {
            "timestamp": timestamp_string(),
            "frame": frame,
            "step": current_step,
            "speed_kmh": speed_kmh,
            "speed_mps": speed_mps,
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
            "is_low_speed": is_low_speed,
            "low_speed_steps": self.low_speed_steps,
            "offroad_candidate": offroad_candidate,
            "offroad_steps": self.offroad_steps,
            "lidar_points": sensor_snapshot["lidar_points"],
            "action": action_vector.copy(),
            "control": {
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
            },
            "observation_keys": list(observation.keys()),
            "observation_shapes": {key: tuple(value.shape) for key, value in observation.items()},
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "done": done,
            "done_reason": done_reason,
            "max_steps_reached": current_step >= self.max_steps,
        }

    def _update_stuck_status(self, speed_mps: float, reached_goal: bool) -> tuple[bool, bool]:
        if reached_goal:
            self.low_speed_steps = 0
            return False, False

        is_low_speed = speed_mps < self.stuck_speed_threshold_mps
        if is_low_speed:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0

        return is_low_speed, self.low_speed_steps >= self.stuck_min_steps

    def _update_offroad_status(self, vehicle_transform: carla.Transform, lane_offset: float, reached_goal: bool) -> tuple[bool, bool]:
        if reached_goal or self.map is None:
            self.offroad_steps = 0
            return False, False

        waypoint = self.map.get_waypoint(
            vehicle_transform.location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            offroad_candidate = True
        else:
            threshold = self.offroad_lane_offset_threshold
            if threshold is None:
                threshold = max(float(waypoint.lane_width) * 0.6, 1.75)
            offroad_candidate = abs(float(lane_offset)) > threshold

        if offroad_candidate:
            self.offroad_steps += 1
        else:
            self.offroad_steps = 0

        return offroad_candidate, self.offroad_steps >= self.offroad_min_steps

    def _resolve_done_reason(self, info: Dict[str, Any], current_step: int) -> str:
        if bool(info["reached_goal"]):
            return "goal"
        if bool(info["collision"]):
            return "collision"
        if bool(info["offroad"]):
            return "offroad"
        if bool(info["stuck"]):
            return "stuck"
        if current_step >= self.max_steps:
            return "max_steps"
        return "running"

    def _save_rgb_image(self, image: Any, frame: int) -> str:
        if image is None:
            return ""
        image_name = f"{frame:06d}.png"
        image_path = os.path.join(self.images_dir, image_name)
        image.save_to_disk(image_path)
        return image_path

    def _open_log(self) -> None:
        self._close_log_file()
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
                "reward",
                "progress_reward",
                "lane_reward",
                "collision_penalty",
                "offroad_penalty",
                "lane_invasion_penalty",
                "control_cost",
                "goal_bonus",
                "stuck_penalty",
                "done",
                "done_reason",
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
        reward_breakdown = info.get("reward_breakdown", self._zero_reward_breakdown())
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
            "reward": round(float(info.get("reward", 0.0)), 6),
            "progress_reward": round(float(reward_breakdown["progress_reward"]), 6),
            "lane_reward": round(float(reward_breakdown["lane_reward"]), 6),
            "collision_penalty": round(float(reward_breakdown["collision_penalty"]), 6),
            "offroad_penalty": round(float(reward_breakdown["offroad_penalty"]), 6),
            "lane_invasion_penalty": round(float(reward_breakdown["lane_invasion_penalty"]), 6),
            "control_cost": round(float(reward_breakdown["control_cost"]), 6),
            "goal_bonus": round(float(reward_breakdown["goal_bonus"]), 6),
            "stuck_penalty": round(float(reward_breakdown["stuck_penalty"]), 6),
            "done": bool(info.get("done", False)),
            "done_reason": info.get("done_reason", "running"),
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

    def _zero_reward_breakdown(self) -> Dict[str, float]:
        return {
            "progress_reward": 0.0,
            "lane_reward": 0.0,
            "collision_penalty": 0.0,
            "offroad_penalty": 0.0,
            "lane_invasion_penalty": 0.0,
            "control_cost": 0.0,
            "goal_bonus": 0.0,
            "stuck_penalty": 0.0,
        }
