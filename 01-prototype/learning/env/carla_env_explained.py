"""
이 파일 전체 역할:
CARLA 시뮬레이터를 하나의 강화학습 환경처럼 사용할 수 있게 해 주는 핵심 파일이다.
reset(), step(), close() 구조를 중심으로 world 연결, 차량 생성, 센서 수집, observation 생성,
reward 계산, done 판정, 로그 저장까지 전체 흐름을 관리한다.

이 파일을 읽을 때 큰 흐름:
1. __init__에서 환경 설정과 내부 상태를 준비한다.
2. reset()에서 새 episode를 시작한다.
3. step()에서 action을 적용하고 한 tick 진행한다.
4. close()에서 리소스를 정리한다.
"""

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
    """
    이 클래스의 역할:
    한 번의 step 결과를 구조적으로 묶어서 표현하는 간단한 컨테이너다.

    이 클래스가 언제 사용되는지:
    CarlaEnv.step() 내부에서 최종 결과를 정리할 때 사용된다.

    입력 값 설명:
    obs: 현재 step observation.
    reward: 현재 step scalar reward.
    done: 종료 여부.
    info: 디버깅/로그용 부가 정보.

    출력 값 설명:
    StepResult 객체 하나로 step 결과를 다룰 수 있다.
    """

    obs: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, Any]


class CarlaEnv:
    """
    이 클래스의 역할:
    CARLA 기반 자율주행 RL 환경의 최상위 컨트롤러이다.

    이 클래스의 전체 구조:
    - __init__: 환경 설정 및 내부 상태 준비
    - reset: episode 시작
    - step: 한 step 진행
    - close: 리소스 정리
    - helper 메서드들: world 연결, vehicle spawn, sensor setup, observation/info/log 생성
    """

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
        """
        이 함수의 역할:
        환경에서 사용할 연결 정보, route 설정, 종료 조건, observation/reward builder를 준비한다.

        이 함수가 언제 호출되는지:
        env = CarlaEnv(...)로 환경 객체를 생성할 때 호출된다.

        입력 값 설명:
        host, port: CARLA 서버 연결 정보.
        town: 로드할 CARLA 맵 이름.
        output_dir: 로그/이미지 저장 폴더.
        image_every: 몇 step마다 RGB 이미지를 저장할지 결정.
        client_timeout: client 연결 timeout.
        route_length, route_spacing: waypoint route 생성 설정.
        max_steps: episode 최대 step 수.
        observation_config: observation 전처리 설정.
        reward_config: reward 계산 설정.
        stuck_speed_threshold_mps, stuck_min_steps: stuck 판정 기준.
        offroad_min_steps, offroad_lane_offset_threshold: offroad 판정 기준.

        출력 값 설명:
        반환값은 없고, 모든 설정과 내부 상태를 self에 저장한다.
        """
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
        """
        이 함수의 역할:
        새로운 episode를 시작하기 위해 world, ego vehicle, 센서, route를 준비하고 첫 observation을 반환한다.

        이 함수가 언제 호출되는지:
        episode 시작 시 가장 먼저 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        dict: 초기 obs/reward/done/info를 포함한 결과를 반환한다.
        """
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
        """
        이 함수의 역할:
        action을 적용하고 시뮬레이션을 한 tick 진행한 뒤 observation, reward, done, info를 계산한다.

        이 함수가 언제 호출되는지:
        reset() 이후 episode 동안 반복 호출된다.

        입력 값 설명:
        action:
            None이면 heuristic controller 사용,
            아니면 [steer, throttle, brake] 형태 action을 사용한다.

        출력 값 설명:
        dict: obs, reward, done, info 및 편의용 필드를 포함한 step 결과.
        """
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
        """
        이 함수의 역할:
        센서, 차량, world 설정, 로그 파일 등 사용한 리소스를 정리한다.

        이 함수가 언제 호출되는지:
        reset() 직전이나 프로그램 종료 시 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, 내부 리소스를 안전하게 정리한다.
        """
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
        self.prev_obs = None
        self.prev_info = None
        self.prev_action = np.zeros((3,), dtype=np.float32)
        self.low_speed_steps = 0
        self.offroad_steps = 0

    def _connect_and_load_world(self) -> None:
        """
        이 함수의 역할:
        CARLA 서버에 연결하고 town을 로드한 뒤 synchronous mode를 설정한다.

        이 함수가 언제 호출되는지:
        reset() 초반에 world를 준비할 때 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, self.client, self.world, self.map을 준비한다.
        """
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.client_timeout)
        self.world = self.client.load_world(self.town)
        self.map = self.world.get_map()

        self.previous_settings = self.world.get_settings()
        settings = self.world.get_settings()
        # synchronous mode를 사용하는 이유:
        # RL에서는 action 적용 후 정확히 한 tick만 진행시키고,
        # 그 결과로 얻은 센서 데이터를 현재 step observation으로 써야 재현성과 디버깅이 쉬워진다.
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

    def _spawn_vehicle(self) -> None:
        """
        이 함수의 역할:
        ego vehicle을 world의 spawn point 중 하나에 생성한다.

        이 함수가 언제 호출되는지:
        reset()에서 world 연결 후 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, self.vehicle을 생성한다.
        """
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
        """
        이 함수의 역할:
        ego vehicle에 RGB 카메라, LiDAR, collision sensor, lane invasion sensor를 부착한다.

        이 함수가 언제 호출되는지:
        reset()에서 차량 생성 후 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, self.sensors dict를 채운다.
        """
        if self.world is None or self.vehicle is None:
            raise RuntimeError("World and vehicle must exist before sensor setup.")

        self.sensors = {
            "rgb": RgbCamera(self.world, self.vehicle),
            "lidar": LidarSensor(self.world, self.vehicle),
            "collision": CollisionSensor(self.world, self.vehicle),
            "lane_invasion": LaneInvasionSensor(self.world, self.vehicle),
        }

    def _setup_route(self) -> None:
        """
        이 함수의 역할:
        현재 ego vehicle 위치를 기준으로 waypoint route를 생성한다.

        이 함수가 언제 호출되는지:
        reset()에서 첫 tick 후 route 준비 단계에서 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, self.route_manager를 생성한다.
        """
        if self.map is None or self.vehicle is None:
            raise RuntimeError("Map and vehicle must exist before route setup.")

        start_waypoint = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_waypoint is None:
            raise RuntimeError("Could not project ego vehicle to a driving waypoint.")

        # RouteManager의 역할:
        # route 생성, route_progress 계산, lane_offset/heading_error/reached_goal 계산을 담당한다.
        self.route_manager = RouteManager(
            world_map=self.map,
            start_waypoint=start_waypoint,
            route_length=self.route_length,
            waypoint_spacing=self.route_spacing,
        )

    def _collect_sensor_data(self, frame: int) -> Dict[str, Any]:
        """
        이 함수의 역할:
        현재 world tick frame에 맞는 센서 데이터를 한 번에 수집한다.

        이 함수가 언제 호출되는지:
        reset()과 step()에서 sensor snapshot이 필요할 때 호출된다.

        입력 값 설명:
        frame: 현재 world.tick()으로 얻은 frame 번호.

        출력 값 설명:
        Dict[str, Any]: rgb, lidar, event 정보를 담은 snapshot dict.
        """
        try:
            # 센서 동기화 방식:
            # world가 한 tick 진행된 뒤, 각 센서 queue에서 정확히 같은 frame 번호의 데이터를 찾는다.
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
        """
        이 함수의 역할:
        다양한 형태의 action 입력을 내부 표준인 (3,) float32 벡터로 통일한다.

        이 함수가 언제 호출되는지:
        step(action)이 사용자 입력 action을 직접 받을 때 호출된다.

        입력 값 설명:
        action: dict 또는 list/tuple/ndarray 형태의 [steer, throttle, brake].

        출력 값 설명:
        np.ndarray: 정규화된 [steer, throttle, brake] 벡터.
        """
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
        """
        이 함수의 역할:
        내부 action 벡터를 CARLA VehicleControl로 바꾼다.

        이 함수가 언제 호출되는지:
        step()에서 차량에 제어를 적용하기 직전에 호출된다.

        입력 값 설명:
        action_vector: [steer, throttle, brake] float32 벡터.

        출력 값 설명:
        carla.VehicleControl: CARLA가 이해하는 제어 명령.
        """
        return carla.VehicleControl(
            steer=float(action_vector[0]),
            throttle=float(action_vector[1]),
            brake=float(action_vector[2]),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    def _compute_heuristic_action(self) -> np.ndarray:
        """
        이 함수의 역할:
        정책 네트워크가 없을 때 최소 route-following을 위한 heuristic action을 계산한다.

        이 함수가 언제 호출되는지:
        step(action=None)일 때 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        np.ndarray: [steer, throttle, brake] float32 action 벡터.
        """
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
        """
        이 함수의 역할:
        현재 센서 데이터와 차량 상태를 RL용 observation dict로 만든다.

        이 함수가 언제 호출되는지:
        reset()과 step()에서 observation이 필요할 때 호출된다.

        입력 값 설명:
        sensor_snapshot: 현재 frame의 센서 데이터 묶음.
        control: 현재 적용된 VehicleControl.
        route_metrics: route 관련 metric.

        출력 값 설명:
        Dict[str, np.ndarray]: rgb, lidar_bev, ego, route로 구성된 observation dict.
        """
        if self.vehicle is None or self.route_manager is None:
            raise RuntimeError("Vehicle and route manager are required to build observations.")

        # observation 구성 의미:
        # rgb는 전방 이미지, lidar_bev는 공간 구조, ego는 차량 상태, route는 앞으로 갈 길의 요약이다.
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
        """
        이 함수의 역할:
        현재 step의 디버깅/로그/분석용 정보를 하나의 info dict로 정리한다.

        이 함수가 언제 호출되는지:
        reset()과 step()에서 반환용 info를 만들 때 호출된다.

        입력 값 설명:
        frame, current_step, sensor_snapshot, action_vector, control, route_metrics, observation,
        reward, reward_breakdown, done, done_reason이 들어온다.

        출력 값 설명:
        Dict[str, Any]: 현재 step에 대한 풍부한 상태 정보 dict.
        """
        if self.vehicle is None or self.map is None or self.route_manager is None:
            raise RuntimeError("Vehicle, map, and route manager are required to build info.")

        vehicle_transform = self.vehicle.get_transform()
        speed_kmh = compute_speed_kmh(self.vehicle)
        speed_mps = compute_speed_mps(self.vehicle)
        collision_events = sensor_snapshot["collision_events"]
        lane_events = sensor_snapshot["lane_events"]
        latest_collision = collision_events[-1] if collision_events else None
        lane_markings = sorted({mark for event in lane_events for mark in event["markings"]})
        offroad_candidate, offroad = self._update_offroad_status(vehicle_transform, route_metrics.lane_offset, route_metrics.reached_goal)
        is_low_speed, stuck = self._update_stuck_status(speed_mps, route_metrics.reached_goal)

        # info는 학습 입력은 아니지만, 로그/디버깅/실험 분석에 매우 중요한 역할을 한다.
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
        """
        이 함수의 역할:
        차량이 너무 오랫동안 저속 상태에 머물렀는지(stuck) 판정한다.

        이 함수가 언제 호출되는지:
        _build_info() 내부에서 매 step 호출된다.

        입력 값 설명:
        speed_mps: 현재 차량 속도(m/s).
        reached_goal: goal 도달 여부.

        출력 값 설명:
        tuple[bool, bool]: (현재 저속 상태 여부, stuck 여부)
        """
        # stuck 판정 로직:
        # 단일 step의 저속은 stuck이 아니고, 저속 상태가 연속으로 이어질 때만 stuck으로 본다.
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
        """
        이 함수의 역할:
        차량이 도로를 벗어났는지(offroad)를 연속 판정 기준으로 계산한다.

        이 함수가 언제 호출되는지:
        _build_info() 내부에서 매 step 호출된다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.
        lane_offset: 차선 중심 대비 오프셋.
        reached_goal: goal 도달 여부.

        출력 값 설명:
        tuple[bool, bool]: (offroad 후보 여부, 최종 offroad 여부)
        """
        # offroad 판정 로직:
        # project_to_road=False로 driving waypoint가 없거나 lane_offset이 threshold를 크게 넘으면 후보로 본다.
        # 하지만 한 step만 후보라고 바로 종료하지 않고, 여러 step 연속일 때만 offroad=True가 된다.
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
        """
        이 함수의 역할:
        episode를 종료해야 하는지와 종료 사유를 문자열로 결정한다.

        이 함수가 언제 호출되는지:
        step()에서 reward 계산 후 done 판정 시 호출된다.

        입력 값 설명:
        info: 현재 step 정보 dict.
        current_step: 현재 step 번호.

        출력 값 설명:
        str: "running", "goal", "collision", "offroad", "stuck", "max_steps" 중 하나.
        """
        # episode 종료 조건:
        # goal / collision / offroad / stuck / max_steps
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
        """
        이 함수의 역할:
        현재 RGB 이미지를 디스크에 저장한다.

        이 함수가 언제 호출되는지:
        step()에서 image_every 주기마다 호출된다.

        입력 값 설명:
        image: CARLA RGB image 객체.
        frame: 현재 frame 번호.

        출력 값 설명:
        str: 저장된 이미지 경로. 이미지가 없으면 빈 문자열.
        """
        if image is None:
            return ""
        image_name = f"{frame:06d}.png"
        image_path = os.path.join(self.images_dir, image_name)
        image.save_to_disk(image_path)
        return image_path

    def _open_log(self) -> None:
        """
        이 함수의 역할:
        logs.csv 파일을 열고 로그 헤더를 준비한다.

        이 함수가 언제 호출되는지:
        reset()에서 새로운 episode 로그를 시작할 때 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, self.csv_file과 self.csv_writer를 초기화한다.
        """
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
        """
        이 함수의 역할:
        현재 step 정보를 logs.csv 한 줄로 저장한다.

        이 함수가 언제 호출되는지:
        reset()과 step()에서 info가 준비된 뒤 호출된다.

        입력 값 설명:
        frame: 현재 frame 번호.
        control: 현재 VehicleControl.
        info: 현재 step 정보 dict.
        image_path: 저장된 RGB 이미지 경로.

        출력 값 설명:
        반환값은 없고, CSV 파일에 한 줄을 추가한다.
        """
        if self.csv_writer is None or self.csv_file is None or self.vehicle is None:
            return

        transform = self.vehicle.get_transform()
        reward_breakdown = info.get("reward_breakdown", self._zero_reward_breakdown())
        # 로그 CSV에 기록되는 데이터 의미:
        # 위치, 자세, 제어 입력, route 진행도, 종료 정보, 보상 breakdown을 모두 저장해
        # 나중에 실험 분석과 디버깅에 활용할 수 있게 한다.
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
        """
        이 함수의 역할:
        spectator 카메라를 ego 차량 뒤쪽 추적 시점으로 갱신한다.

        이 함수가 언제 호출되는지:
        reset() 후와 step()마다 world tick 직후 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, spectator transform을 바꾼다.
        """
        if self.spectator is None or self.vehicle is None:
            return

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        forward = vehicle_transform.get_forward_vector()

        # spectator 카메라 업데이트 목적:
        # 개발자가 ego 차량을 게임처럼 뒤에서 따라가며 쉽게 디버깅하기 위함이다.
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
        """
        이 함수의 역할:
        world 좌표를 차량 local 좌표로 변환한다.

        이 함수가 언제 호출되는지:
        heuristic controller가 target waypoint를 차량 기준 상대 위치로 볼 때 호출된다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.
        world_location: world 좌표계의 위치.

        출력 값 설명:
        carla.Location: 차량 기준 local 좌표.
        """
        # 좌표 변환 의미:
        # world 좌표는 지도 기준 위치이고, local 좌표는 차량 기준 앞/뒤/좌/우 위치를 알려 준다.
        delta_x = world_location.x - vehicle_transform.location.x
        delta_y = world_location.y - vehicle_transform.location.y
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)

        local_x = math.cos(yaw_rad) * delta_x + math.sin(yaw_rad) * delta_y
        local_y = -math.sin(yaw_rad) * delta_x + math.cos(yaw_rad) * delta_y
        return carla.Location(x=local_x, y=local_y, z=0.0)

    def _zero_reward_breakdown(self) -> Dict[str, float]:
        """
        이 함수의 역할:
        reward breakdown dict를 0으로 초기화한 기본값으로 만든다.

        이 함수가 언제 호출되는지:
        reset() 직후나 reward 계산 전 기본 schema가 필요할 때 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        Dict[str, float]: 모든 reward 항목이 0.0인 dict.
        """
        # 보상 구조 초기화 목적:
        # 아직 reward를 계산하지 않았더라도 info/log의 형태를 일정하게 유지하기 위함이다.
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
