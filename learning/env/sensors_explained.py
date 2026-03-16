"""
이 파일 전체 역할:
CARLA 센서를 Python 코드에서 다루기 쉽게 감싸고,
동기식 시뮬레이션에서 frame 단위로 센서 데이터를 안정적으로 맞춰 가져오기 위한 학습용 설명 파일이다.
원본 sensors.py의 로직은 그대로 유지하고 설명만 추가했다.
"""

from __future__ import annotations

import math
import queue
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import carla
import numpy as np


class SensorSyncError(RuntimeError):
    """
    이 클래스의 역할:
    센서 동기화가 실패했을 때 발생시키는 예외이다.

    이 클래스가 언제 사용되는지:
    timeout, future frame mismatch 같은 센서 수집 오류가 발생할 때 사용된다.

    입력 값 설명:
    일반 RuntimeError처럼 에러 메시지를 담는다.

    출력 값 설명:
    예외 클래스이므로 정상 반환값은 없다.
    """


@dataclass
class SensorDebugState:
    """
    이 클래스의 역할:
    특정 센서의 상태를 디버깅하기 위한 정보를 저장한다.

    이 클래스가 언제 사용되는지:
    BaseSensor가 frame 처리 상황을 추적할 때 사용된다.

    입력 값 설명:
    sensor_name, last_frame_seen, dropped_frames, timeouts를 저장한다.

    출력 값 설명:
    센서 상태를 담은 dataclass 객체를 제공한다.
    """

    sensor_name: str
    last_frame_seen: Optional[int] = None
    dropped_frames: int = 0
    timeouts: int = 0


class BaseSensor:
    """
    이 클래스의 역할:
    모든 센서 wrapper의 공통 기반 클래스이다.

    이 클래스가 언제 사용되는지:
    RGB 카메라, LiDAR, collision, lane invasion 센서 모두 이 클래스를 기반으로 동작한다.

    입력 값 설명:
    world, parent_actor, blueprint_id, transform, sensor_name, attributes 등을 받는다.

    출력 값 설명:
    센서 actor를 생성하고 queue/pending/debug 상태를 초기화한다.
    """

    def __init__(
        self,
        world: carla.World,
        parent_actor: carla.Actor,
        blueprint_id: str,
        transform: carla.Transform,
        sensor_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        queue_size: int = 32,
    ) -> None:
        """
        이 함수의 역할:
        실제 CARLA 센서를 생성하고 callback queue를 준비한다.

        이 함수가 언제 호출되는지:
        하위 센서 클래스가 super().__init__()를 호출할 때 사용된다.

        입력 값 설명:
        world: CARLA world 객체.
        parent_actor: 센서를 부착할 actor, 보통 ego vehicle.
        blueprint_id: 생성할 센서 blueprint 이름.
        transform: 센서 위치/자세.
        sensor_name: 디버깅용 이름.
        attributes: blueprint 속성 설정.
        queue_size: 내부 queue 최대 크기.

        출력 값 설명:
        반환값은 없고, 센서 actor와 내부 상태를 생성한다.
        """
        blueprint = world.get_blueprint_library().find(blueprint_id)
        if blueprint is None:
            raise RuntimeError(f"Blueprint not found: {blueprint_id}")

        for key, value in (attributes or {}).items():
            blueprint.set_attribute(key, str(value))

        self.sensor_name = sensor_name
        self.sensor: Optional[carla.Sensor] = world.spawn_actor(blueprint, transform, attach_to=parent_actor)
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        self._pending: Deque[Any] = deque()
        self.debug = SensorDebugState(sensor_name=sensor_name)

        self.sensor.listen(self._on_data)

    def _on_data(self, data: Any) -> None:
        """
        이 함수의 역할:
        센서 callback으로 들어온 데이터를 queue에 저장한다.

        이 함수가 언제 호출되는지:
        CARLA sensor.listen()이 새 데이터가 생길 때 자동 호출한다.

        입력 값 설명:
        data: 센서가 생성한 원본 데이터.

        출력 값 설명:
        반환값은 없고, queue에 데이터를 넣는다.
        """
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # queue가 가득 차면 가장 오래된 데이터를 버리고 최신 데이터를 유지한다.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.debug.dropped_frames += 1
            self._queue.put_nowait(data)

    def _convert(self, data: Any) -> Any:
        """
        이 함수의 역할:
        센서 원본 데이터를 외부에서 쓰기 좋은 형식으로 바꾸는 hook이다.

        이 함수가 언제 호출되는지:
        get_data()가 올바른 frame을 찾은 뒤 반환 직전에 호출된다.

        입력 값 설명:
        data: 원본 센서 데이터.

        출력 값 설명:
        Any: 기본 구현은 그대로 반환하며 하위 클래스에서 override 가능하다.
        """
        return data

    def _get_next_raw(self, timeout: float) -> Any:
        """
        이 함수의 역할:
        pending 버퍼 또는 queue에서 다음 raw 데이터를 하나 가져온다.

        이 함수가 언제 호출되는지:
        get_data()가 특정 frame 데이터를 찾을 때 반복 호출된다.

        입력 값 설명:
        timeout: queue 대기 시간.

        출력 값 설명:
        Any: raw 데이터 하나를 반환하며 timeout이면 None.
        """
        if self._pending:
            return self._pending.popleft()

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            self.debug.timeouts += 1
            return None

    def _store_pending_left(self, data: Any) -> None:
        """
        이 함수의 역할:
        아직 처리하지 않을 future frame 데이터를 pending 버퍼에 저장한다.

        이 함수가 언제 호출되는지:
        현재 기다리는 frame보다 미래 frame이 먼저 도착했을 때 호출된다.

        입력 값 설명:
        data: 나중에 다시 쓸 센서 데이터.

        출력 값 설명:
        반환값은 없고 pending deque 앞쪽에 저장한다.
        """
        self._pending.appendleft(data)

    def get_data(self, frame: int, timeout: float = 2.0) -> Any:
        """
        이 함수의 역할:
        특정 frame 번호에 해당하는 센서 데이터를 찾아 반환한다.

        이 함수가 언제 호출되는지:
        CarlaEnv._collect_sensor_data()에서 현재 world tick에 맞는 센서 데이터를 수집할 때 호출된다.

        입력 값 설명:
        frame: 기대하는 현재 frame 번호.
        timeout: 데이터 대기 시간.

        출력 값 설명:
        Any: 해당 frame의 변환된 센서 데이터를 반환한다.
        """
        while True:
            data = self._get_next_raw(timeout)
            if data is None:
                raise SensorSyncError(self._build_timeout_message(frame))

            self.debug.last_frame_seen = int(data.frame)

            if data.frame < frame:
                self.debug.dropped_frames += 1
                continue

            if data.frame > frame:
                self._store_pending_left(data)
                raise SensorSyncError(self._build_future_frame_message(expected_frame=frame, received_frame=int(data.frame)))

            return self._convert(data)

    def _build_timeout_message(self, frame: int) -> str:
        """
        이 함수의 역할:
        timeout 발생 시 디버깅에 필요한 메시지를 만든다.

        이 함수가 언제 호출되는지:
        get_data()가 timeout으로 실패할 때 호출된다.

        입력 값 설명:
        frame: 기다리던 frame 번호.

        출력 값 설명:
        str: 디버깅용 상세 메시지.
        """
        return (
            f"[{self.sensor_name}] timeout while waiting for frame {frame}. "
            f"last_frame_seen={self.debug.last_frame_seen} "
            f"dropped_frames={self.debug.dropped_frames} "
            f"timeouts={self.debug.timeouts}"
        )

    def _build_future_frame_message(self, expected_frame: int, received_frame: int) -> str:
        """
        이 함수의 역할:
        future frame mismatch가 발생했을 때 메시지를 만든다.

        이 함수가 언제 호출되는지:
        get_data()가 기대한 frame보다 더 큰 frame을 먼저 받았을 때 호출된다.

        입력 값 설명:
        expected_frame: 현재 기다리는 frame.
        received_frame: 실제 도착한 frame.

        출력 값 설명:
        str: mismatch 설명 메시지.
        """
        return (
            f"[{self.sensor_name}] future frame received while waiting for synchronous tick. "
            f"expected={expected_frame} received={received_frame} "
            f"last_frame_seen={self.debug.last_frame_seen}"
        )

    def destroy(self) -> None:
        """
        이 함수의 역할:
        센서를 정지시키고 CARLA world에서 제거한다.

        이 함수가 언제 호출되는지:
        CarlaEnv.close()에서 센서 리소스를 정리할 때 호출된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        반환값은 없고, 센서를 안전하게 종료/삭제한다.
        """
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except RuntimeError:
                pass
            try:
                self.sensor.destroy()
            except RuntimeError:
                pass
            self.sensor = None


class StreamSensor(BaseSensor):
    """
    이 클래스의 역할:
    카메라나 LiDAR처럼 매 frame 데이터를 내보내는 스트림 센서용 래퍼이다.

    이 클래스가 언제 사용되는지:
    RgbCamera와 LidarSensor가 이 클래스를 상속한다.

    입력 값 설명:
    BaseSensor와 동일하다.

    출력 값 설명:
    현재 frame에 맞는 스트림 데이터를 반환한다.
    """

    def get_data(self, frame: int, timeout: float = 2.0) -> Any:
        """
        이 함수의 역할:
        스트림 센서의 현재 frame 데이터를 좀 더 유연하게 동기화해서 반환한다.

        이 함수가 언제 호출되는지:
        RGB/LiDAR 데이터를 수집할 때 호출된다.

        입력 값 설명:
        frame: 기대하는 frame 번호.
        timeout: 대기 시간.

        출력 값 설명:
        Any: 현재 frame에 해당하는 데이터.
        """
        while True:
            data = self._get_next_raw(timeout)
            if data is None:
                raise SensorSyncError(self._build_timeout_message(frame))

            self.debug.last_frame_seen = int(data.frame)

            if data.frame < frame:
                self.debug.dropped_frames += 1
                continue

            if data.frame > frame:
                self._store_pending_left(data)
                data = self._get_next_raw(timeout)
                if data is None:
                    raise SensorSyncError(self._build_timeout_message(frame))
                self.debug.last_frame_seen = int(data.frame)
                if data.frame > frame:
                    self._store_pending_left(data)
                    raise SensorSyncError(self._build_future_frame_message(expected_frame=frame, received_frame=int(data.frame)))
                if data.frame < frame:
                    self.debug.dropped_frames += 1
                    continue

            return self._convert(data)


class EventSensor(BaseSensor):
    """
    이 클래스의 역할:
    특정 frame에 0개 또는 여러 이벤트가 발생할 수 있는 이벤트 센서용 래퍼이다.

    이 클래스가 언제 사용되는지:
    collision sensor와 lane invasion sensor가 이 클래스를 상속한다.

    입력 값 설명:
    BaseSensor와 동일하다.

    출력 값 설명:
    한 frame에 발생한 이벤트들을 리스트로 반환한다.
    """

    def get_data(self, frame: int, timeout: float = 0.02) -> List[Any]:
        """
        이 함수의 역할:
        해당 frame에서 발생한 모든 이벤트를 리스트로 모아 반환한다.

        이 함수가 언제 호출되는지:
        collision/lane invasion 데이터를 수집할 때 호출된다.

        입력 값 설명:
        frame: 기대하는 frame 번호.
        timeout: 추가 이벤트를 기다릴 짧은 시간.

        출력 값 설명:
        List[Any]: 현재 frame 이벤트 리스트.
        """
        matched: List[Any] = []

        while True:
            data = self._get_next_raw(timeout)
            if data is None:
                return matched

            self.debug.last_frame_seen = int(data.frame)

            if data.frame < frame:
                self.debug.dropped_frames += 1
                continue

            if data.frame > frame:
                self._store_pending_left(data)
                return matched

            matched.append(self._convert(data))


class RgbCamera(StreamSensor):
    """
    이 클래스의 역할:
    ego 차량 앞쪽 시야를 보는 RGB 카메라 센서를 생성한다.
    """

    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
        """
        이 함수의 역할:
        RGB 카메라 센서를 차량에 부착한다.

        이 함수가 언제 호출되는지:
        CarlaEnv._setup_sensors()에서 호출된다.

        입력 값 설명:
        world: CARLA world.
        parent_actor: 센서를 부착할 ego vehicle.

        출력 값 설명:
        반환값은 없고, 카메라 센서 actor를 생성한다.
        """
        super().__init__(
            world=world,
            parent_actor=parent_actor,
            blueprint_id="sensor.camera.rgb",
            transform=carla.Transform(carla.Location(x=1.5, z=2.2)),
            sensor_name="rgb_camera",
            attributes={
                "image_size_x": 800,
                "image_size_y": 450,
                "fov": 100,
                "sensor_tick": 0.0,
            },
        )


class LidarSensor(StreamSensor):
    """
    이 클래스의 역할:
    ego 차량 주변 3D point cloud를 수집하는 LiDAR 센서를 생성한다.
    """

    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
        """
        이 함수의 역할:
        LiDAR 센서를 차량 위쪽에 부착한다.

        이 함수가 언제 호출되는지:
        CarlaEnv._setup_sensors()에서 호출된다.

        입력 값 설명:
        world: CARLA world.
        parent_actor: 센서를 부착할 ego vehicle.

        출력 값 설명:
        반환값은 없고, LiDAR 센서 actor를 생성한다.
        """
        super().__init__(
            world=world,
            parent_actor=parent_actor,
            blueprint_id="sensor.lidar.ray_cast",
            transform=carla.Transform(carla.Location(z=2.2)),
            sensor_name="lidar",
            attributes={
                "channels": 16,
                "range": 40.0,
                "rotation_frequency": 20.0,
                "points_per_second": 20000,
                "upper_fov": 10.0,
                "lower_fov": -20.0,
                "sensor_tick": 0.0,
            },
        )

    def _convert(self, data: carla.LidarMeasurement) -> np.ndarray:
        """
        이 함수의 역할:
        LiDAR raw buffer를 numpy (N,4) 배열로 바꾼다.

        이 함수가 언제 호출되는지:
        현재 frame LiDAR 데이터를 반환하기 직전에 호출된다.

        입력 값 설명:
        data: CARLA LiDAR measurement.

        출력 값 설명:
        np.ndarray: [x, y, z, intensity] 형태 float32 배열.
        """
        raw = np.frombuffer(data.raw_data, dtype=np.float32)
        if raw.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        # LiDAR를 numpy 배열로 바꾸면 이후 BEV 변환과 상태 구성이 쉬워진다.
        return raw.reshape((-1, 4)).copy()


class CollisionSensor(EventSensor):
    """
    이 클래스의 역할:
    다른 actor와의 충돌 이벤트를 수집한다.
    """

    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
        """
        이 함수의 역할:
        collision sensor를 ego 차량에 부착한다.

        이 함수가 언제 호출되는지:
        CarlaEnv._setup_sensors()에서 호출된다.

        입력 값 설명:
        world: CARLA world.
        parent_actor: ego vehicle.

        출력 값 설명:
        반환값은 없고, collision sensor actor를 생성한다.
        """
        super().__init__(
            world=world,
            parent_actor=parent_actor,
            blueprint_id="sensor.other.collision",
            transform=carla.Transform(),
            sensor_name="collision",
        )

    def _convert(self, data: carla.CollisionEvent) -> Dict[str, Any]:
        """
        이 함수의 역할:
        충돌 이벤트를 로그/디버깅에 적합한 dict로 바꾼다.

        이 함수가 언제 호출되는지:
        EventSensor.get_data()가 collision 이벤트를 반환할 때 호출된다.

        입력 값 설명:
        data: collision event 객체.

        출력 값 설명:
        Dict[str, Any]: frame, other_actor, intensity를 담은 dict.
        """
        impulse = data.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        other_actor = "unknown"
        if data.other_actor is not None:
            other_actor = f"{data.other_actor.type_id}:{data.other_actor.id}"
        return {
            "frame": int(data.frame),
            "other_actor": other_actor,
            "intensity": float(intensity),
        }


class LaneInvasionSensor(EventSensor):
    """
    이 클래스의 역할:
    차선 마킹을 침범했을 때 lane invasion 이벤트를 수집한다.
    """

    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
        """
        이 함수의 역할:
        lane invasion sensor를 ego 차량에 부착한다.

        이 함수가 언제 호출되는지:
        CarlaEnv._setup_sensors()에서 호출된다.

        입력 값 설명:
        world: CARLA world.
        parent_actor: ego vehicle.

        출력 값 설명:
        반환값은 없고, lane invasion sensor actor를 생성한다.
        """
        super().__init__(
            world=world,
            parent_actor=parent_actor,
            blueprint_id="sensor.other.lane_invasion",
            transform=carla.Transform(),
            sensor_name="lane_invasion",
        )

    def _convert(self, data: carla.LaneInvasionEvent) -> Dict[str, Any]:
        """
        이 함수의 역할:
        lane invasion 이벤트를 dict 형태로 바꾼다.

        이 함수가 언제 호출되는지:
        EventSensor.get_data()가 lane invasion 이벤트를 반환할 때 호출된다.

        입력 값 설명:
        data: lane invasion event 객체.

        출력 값 설명:
        Dict[str, Any]: frame과 침범한 lane marking 목록을 담은 dict.
        """
        markings = sorted({str(marking.type) for marking in data.crossed_lane_markings})
        return {
            "frame": int(data.frame),
            "markings": markings,
        }
