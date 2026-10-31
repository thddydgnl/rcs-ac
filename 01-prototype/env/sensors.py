from __future__ import annotations

import math
import queue
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import carla
import numpy as np


class SensorSyncError(RuntimeError):
    """Raised when synchronous sensor collection fails."""


@dataclass
class SensorDebugState:
    sensor_name: str
    last_frame_seen: Optional[int] = None
    dropped_frames: int = 0
    timeouts: int = 0


class BaseSensor:
    """Common synchronous sensor wrapper with buffered frame matching."""

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
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.debug.dropped_frames += 1
            self._queue.put_nowait(data)

    def _convert(self, data: Any) -> Any:
        return data

    def _get_next_raw(self, timeout: float) -> Any:
        if self._pending:
            return self._pending.popleft()

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            self.debug.timeouts += 1
            return None

    def _store_pending_left(self, data: Any) -> None:
        self._pending.appendleft(data)

    def get_data(self, frame: int, timeout: float = 2.0) -> Any:
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
        return (
            f"[{self.sensor_name}] timeout while waiting for frame {frame}. "
            f"last_frame_seen={self.debug.last_frame_seen} "
            f"dropped_frames={self.debug.dropped_frames} "
            f"timeouts={self.debug.timeouts}"
        )

    def _build_future_frame_message(self, expected_frame: int, received_frame: int) -> str:
        return (
            f"[{self.sensor_name}] future frame received while waiting for synchronous tick. "
            f"expected={expected_frame} received={received_frame} "
            f"last_frame_seen={self.debug.last_frame_seen}"
        )

    def destroy(self) -> None:
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
    """Frame-synchronous stream sensor such as camera or LiDAR."""

    def get_data(self, frame: int, timeout: float = 2.0) -> Any:
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
    """Event sensor that may emit zero or many events for a given frame."""

    def get_data(self, frame: int, timeout: float = 0.02) -> List[Any]:
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
    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
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
    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
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
        raw = np.frombuffer(data.raw_data, dtype=np.float32)
        if raw.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return raw.reshape((-1, 4)).copy()


class CollisionSensor(EventSensor):
    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
        super().__init__(
            world=world,
            parent_actor=parent_actor,
            blueprint_id="sensor.other.collision",
            transform=carla.Transform(),
            sensor_name="collision",
        )

    def _convert(self, data: carla.CollisionEvent) -> Dict[str, Any]:
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
    def __init__(self, world: carla.World, parent_actor: carla.Actor) -> None:
        super().__init__(
            world=world,
            parent_actor=parent_actor,
            blueprint_id="sensor.other.lane_invasion",
            transform=carla.Transform(),
            sensor_name="lane_invasion",
        )

    def _convert(self, data: carla.LaneInvasionEvent) -> Dict[str, Any]:
        markings = sorted({str(marking.type) for marking in data.crossed_lane_markings})
        return {
            "frame": int(data.frame),
            "markings": markings,
        }
