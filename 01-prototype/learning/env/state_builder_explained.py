"""
이 파일 전체 역할:
CARLA에서 들어오는 원시 센서 데이터와 차량 상태를,
강화학습에서 바로 쓰기 쉬운 observation 형태로 바꾸는 전처리 모듈이다.

이 파일을 읽을 때 핵심 포인트:
1. RGB 이미지는 신경망에 넣기 쉬운 float32 tensor로 바뀐다.
2. LiDAR point cloud는 2D BEV 격자로 바뀐다.
3. 차량 자체 상태는 ego vector로 정리된다.
4. route 정보는 lookahead waypoint 기준 route vector로 정리된다.
"""

from __future__ import annotations

# dataclass:
# observation 설정값을 구조적으로 보관하기 위해 사용한다.
from dataclasses import dataclass
# Any:
# CARLA 객체처럼 구체 타입을 엄격히 고정하기 어려운 입력에 사용한다.
from typing import Any

# numpy:
# 이미지/포인트클라우드/벡터를 배열 형태로 다루기 위해 사용한다.
import numpy as np

# 공용 유틸:
# 차량 속도 계산, 월드 좌표를 차량 local 좌표로 바꾸는 helper를 사용한다.
from utils.common import compute_speed_mps, world_to_local_2d

try:
    # OpenCV가 있으면 resize 품질과 성능이 좋아서 우선 사용한다.
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    # OpenCV가 없으면 numpy 기반 fallback resize를 사용한다.
    cv2 = None


@dataclass
class ObservationConfig:
    """
    이 클래스의 역할:
    observation 전처리 과정에서 사용할 해상도, 격자 크기, 범위 등을 한 곳에 모아 두는 설정 클래스이다.

    이 클래스가 언제 호출되는지:
    StateBuilder가 생성될 때 기본 설정 또는 사용자 지정 설정으로 사용된다.

    입력 값 설명:
    각 필드는 RGB 크기, BEV 격자 크기, LiDAR 필터링 범위, route lookahead 개수를 정의한다.

    출력 값 설명:
    이 클래스 자체는 값을 계산하지 않고, StateBuilder가 참조할 설정 묶음을 제공한다.
    """

    rgb_width: int = 160
    rgb_height: int = 90
    rgb_channels_first: bool = True
    bev_x_bins: int = 64
    bev_y_bins: int = 64
    bev_x_range: tuple[float, float] = (0.0, 30.0)
    bev_y_range: tuple[float, float] = (-15.0, 15.0)
    bev_z_min: float = -2.5
    bev_z_max: float = 1.0
    route_lookahead_count: int = 3


class StateBuilder:
    """
    이 클래스의 역할:
    RGB, LiDAR, ego state, route 정보를 최종 observation dict로 조합하는 클래스이다.

    이 클래스가 언제 사용되는지:
    CarlaEnv.reset(), CarlaEnv.step()에서 센서와 차량 상태를 observation으로 바꿀 때 사용된다.

    입력 값 설명:
    CARLA camera image, LiDAR point cloud, vehicle 객체, control, route_manager, route_metrics 등이 들어온다.

    출력 값 설명:
    RL-friendly한 4모달 observation dict를 반환한다.
    """

    def __init__(self, config: ObservationConfig | None = None) -> None:
        """
        이 함수의 역할:
        StateBuilder가 사용할 전처리 설정을 저장한다.

        이 함수가 언제 호출되는지:
        CarlaEnv가 초기화될 때 StateBuilder를 생성하면서 호출된다.

        입력 값 설명:
        config:
            ObservationConfig 객체. None이면 기본 ObservationConfig()를 사용한다.

        출력 값 설명:
        반환값은 없고, self.config에 설정이 저장된다.
        """
        self.config = config or ObservationConfig()

    def build_rgb(self, image: Any) -> np.ndarray:
        """
        이 함수의 역할:
        CARLA raw image를 RL 모델 입력에 적합한 RGB float32 배열로 변환한다.

        이 함수가 언제 호출되는지:
        build_observation() 내부에서 RGB 모달을 만들 때 호출된다.

        입력 값 설명:
        image:
            CARLA camera sensor가 반환한 이미지 객체. None이면 0으로 채운 이미지를 만든다.

        출력 값 설명:
        np.ndarray:
            shape는 (3, H, W), dtype은 float32이며 값 범위는 [0, 1]이다.
        """
        if image is None:
            # 센서 데이터가 없더라도 shape를 항상 고정하기 위해 0으로 채운 이미지를 만든다.
            rgb = np.zeros((self.config.rgb_height, self.config.rgb_width, 3), dtype=np.float32)
        else:
            # CARLA camera는 raw_data를 BGRA(uint8) 버퍼 형태로 제공한다.
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            # alpha 채널을 제거하고 BGR -> RGB 순서로 바꾼다.
            rgb_uint8 = raw[:, :, :3][:, :, ::-1]
            resized = self._resize_rgb(rgb_uint8, self.config.rgb_width, self.config.rgb_height)
            # 신경망 입력에 맞게 [0, 1] 범위 float32로 정규화한다.
            rgb = resized.astype(np.float32) / 255.0

        if self.config.rgb_channels_first:
            # PyTorch 스타일 입력을 염두에 두고 채널 축을 앞으로 보낸다.
            rgb = np.transpose(rgb, (2, 0, 1))
        return rgb.astype(np.float32, copy=False)

    def build_lidar_bev(self, points: np.ndarray) -> np.ndarray:
        """
        이 함수의 역할:
        LiDAR point cloud를 2D BEV(위에서 내려다본 격자 지도)로 변환한다.

        이 함수가 언제 호출되는지:
        build_observation() 내부에서 LiDAR 모달을 만들 때 호출된다.

        입력 값 설명:
        points:
            shape가 (N,4) 또는 (N,3)인 LiDAR 포인트 배열이다.

        출력 값 설명:
        np.ndarray:
            shape는 (1, H, W), dtype은 float32인 density 기반 BEV 격자이다.
        """
        bev = np.zeros((self.config.bev_y_bins, self.config.bev_x_bins), dtype=np.float32)
        if points is None or points.size == 0:
            return bev[np.newaxis, :, :]

        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError(f"Expected LiDAR points with shape (N,3) or (N,4), got {points.shape}")

        # intensity가 있더라도 현재 단계에서는 xyz만 사용해 occupancy/density를 만든다.
        xyz = points[:, :3]
        x_min, x_max = self.config.bev_x_range
        y_min, y_max = self.config.bev_y_range
        z_min = self.config.bev_z_min
        z_max = self.config.bev_z_max

        # 관찰하고 싶은 영역만 남기고 나머지 포인트는 제거한다.
        valid = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] <= y_max)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
        )
        xyz = xyz[valid]
        if xyz.shape[0] == 0:
            return bev[np.newaxis, :, :]

        # 2D histogram으로 각 칸에 몇 개의 포인트가 떨어졌는지 계산한다.
        hist, _, _ = np.histogram2d(
            xyz[:, 1],
            xyz[:, 0],
            bins=[self.config.bev_y_bins, self.config.bev_x_bins],
            range=[self.config.bev_y_range, self.config.bev_x_range],
        )
        hist = hist.astype(np.float32)
        max_value = float(hist.max())
        if max_value > 0.0:
            # 격자 값을 [0, 1] 범위의 normalized density로 정규화한다.
            hist /= max_value
        return hist[np.newaxis, :, :].astype(np.float32, copy=False)

    def build_ego(self, vehicle: Any, control: Any, route_metrics: Any) -> np.ndarray:
        """
        이 함수의 역할:
        차량 자체 상태를 고정 길이 ego vector로 추출한다.

        이 함수가 언제 호출되는지:
        build_observation() 내부에서 ego 모달을 만들 때 호출된다.

        입력 값 설명:
        vehicle:
            CARLA vehicle 객체.
        control:
            현재 step에 적용된 VehicleControl.
        route_metrics:
            heading_error, lane_offset 등이 들어 있는 route metric 객체.

        출력 값 설명:
        np.ndarray:
            shape는 (7,), dtype은 float32인 ego 상태 벡터이다.
        """
        speed_mps = compute_speed_mps(vehicle)
        angular_velocity = vehicle.get_angular_velocity()
        # CARLA angular velocity의 z는 deg/s이므로 rad/s로 바꾼다.
        yaw_rate = np.deg2rad(float(angular_velocity.z))

        acceleration = vehicle.get_acceleration()
        # acceleration은 world frame 값이므로, local frame 가속도를 얻기 위해 위치처럼 다룰 점을 만든다.
        world_acc_location = type(vehicle.get_location())(
            x=vehicle.get_transform().location.x + acceleration.x,
            y=vehicle.get_transform().location.y + acceleration.y,
            z=vehicle.get_transform().location.z + acceleration.z,
        )
        accel_x, accel_y = world_to_local_2d(vehicle.get_transform(), world_acc_location)

        features = np.asarray(
            [
                speed_mps,
                yaw_rate,
                accel_x,
                accel_y,
                float(control.steer),
                float(route_metrics.heading_error),
                float(route_metrics.lane_offset),
            ],
            dtype=np.float32,
        )
        return features

    def build_route(self, route_manager: Any, vehicle_transform: Any) -> np.ndarray:
        """
        이 함수의 역할:
        lookahead waypoint 정보를 고정 길이 route vector로 변환한다.

        이 함수가 언제 호출되는지:
        build_observation() 내부에서 route 모달을 만들 때 호출된다.

        입력 값 설명:
        route_manager:
            현재 route를 관리하는 객체.
        vehicle_transform:
            ego 차량의 현재 transform.

        출력 값 설명:
        np.ndarray:
            shape는 (6,), dtype은 float32이며
            [dist1, angle1, dist2, angle2, dist3, angle3] 구조를 가진다.
        """
        route_vec = route_manager.build_route_vector(
            vehicle_transform=vehicle_transform,
            count=self.config.route_lookahead_count,
        )
        expected_size = self.config.route_lookahead_count * 2
        if route_vec.shape[0] < expected_size:
            # route 길이가 짧아 부족한 경우에도 shape를 고정하기 위해 0 padding을 한다.
            padded = np.zeros((expected_size,), dtype=np.float32)
            padded[: route_vec.shape[0]] = route_vec
            return padded
        return route_vec.astype(np.float32, copy=False)

    def build_observation(
        self,
        image: Any,
        lidar_points: np.ndarray,
        vehicle: Any,
        control: Any,
        route_manager: Any,
        route_metrics: Any,
    ) -> dict[str, np.ndarray]:
        """
        이 함수의 역할:
        RGB, LiDAR BEV, ego vector, route vector를 묶어서 최종 observation dict를 만든다.

        이 함수가 언제 호출되는지:
        CarlaEnv.reset(), CarlaEnv.step()에서 observation을 만들 때 호출된다.

        입력 값 설명:
        image:
            CARLA 카메라 이미지 객체.
        lidar_points:
            numpy 형태 LiDAR point cloud.
        vehicle:
            ego vehicle 객체.
        control:
            현재 제어 명령.
        route_manager:
            route 관련 계산을 담당하는 객체.
        route_metrics:
            현재 route 진행 상태를 담은 metric 객체.

        출력 값 설명:
        dict[str, np.ndarray]:
            {
              "rgb": (3,H,W),
              "lidar_bev": (1,H,W),
              "ego": (7,),
              "route": (6,)
            }
            형태의 observation dict를 반환한다.
        """
        observation = {
            "rgb": self.build_rgb(image),
            "lidar_bev": self.build_lidar_bev(lidar_points),
            "ego": self.build_ego(vehicle, control, route_metrics),
            "route": self.build_route(route_manager, vehicle.get_transform()),
        }
        # 모든 observation 값의 dtype을 float32로 통일한다.
        return {key: value.astype(np.float32, copy=False) for key, value in observation.items()}

    def _resize_rgb(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        이 함수의 역할:
        RGB 이미지를 지정된 크기로 resize한다.

        이 함수가 언제 호출되는지:
        build_rgb()에서 원본 camera 해상도를 학습용 입력 크기로 줄일 때 호출된다.

        입력 값 설명:
        image:
            RGB uint8 이미지 배열.
        width, height:
            목표 해상도.

        출력 값 설명:
        np.ndarray:
            resize된 RGB 이미지 배열을 반환한다.
        """
        if cv2 is not None:
            # OpenCV가 있으면 더 안정적이고 빠른 resize를 사용한다.
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return self._nearest_resize(image, width, height)

    def _nearest_resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        이 함수의 역할:
        OpenCV가 없을 때 사용할 간단한 nearest-neighbor resize를 수행한다.

        이 함수가 언제 호출되는지:
        _resize_rgb()에서 cv2가 설치되지 않은 환경일 때 fallback으로 호출된다.

        입력 값 설명:
        image:
            resize 대상 이미지.
        width, height:
            목표 해상도.

        출력 값 설명:
        np.ndarray:
            nearest-neighbor 방식으로 resize된 이미지 배열을 반환한다.
        """
        src_h, src_w = image.shape[:2]
        if src_h == height and src_w == width:
            return image

        # 원본 이미지에서 샘플링할 index를 선형 간격으로 만든다.
        y_idx = np.linspace(0, src_h - 1, height).astype(np.int32)
        x_idx = np.linspace(0, src_w - 1, width).astype(np.int32)
        return image[y_idx][:, x_idx]
