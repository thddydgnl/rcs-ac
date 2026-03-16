"""
이 파일 전체 역할:
ego 차량이 따라갈 waypoint route를 생성하고, 현재 route 진행 상태를 계산하는 학습용 설명 파일이다.
원본 route_manager.py의 로직은 그대로 유지하고, 한국어 설명만 추가했다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import carla

from utils.common import world_to_local_2d


@dataclass
class RouteMetrics:
    """
    이 클래스의 역할:
    현재 route 기준 주행 상태를 요약하는 데이터 묶음이다.

    이 클래스가 언제 사용되는지:
    CarlaEnv가 route_progress, lane_offset, heading_error, goal 도달 여부를 한 번에 전달할 때 사용된다.

    입력 값 설명:
    target_index, route_progress, lane_offset, heading_error, reached_goal 값을 저장한다.

    출력 값 설명:
    계산된 route 상태를 담아 전달한다.
    """

    target_index: int
    route_progress: float
    lane_offset: float
    heading_error: float
    reached_goal: bool


class RouteManager:
    """
    이 클래스의 역할:
    waypoint route를 생성하고, 차량이 그 route를 얼마나 따라갔는지 관리하는 핵심 클래스이다.

    이 클래스가 언제 사용되는지:
    CarlaEnv._setup_route()에서 생성되고, 이후 매 step마다 target waypoint와 route metric 계산에 사용된다.

    입력 값 설명:
    world_map, start_waypoint, route_length, waypoint_spacing을 받는다.

    출력 값 설명:
    route 리스트와 관련 계산 메서드들을 제공한다.
    """

    def __init__(self, world_map: carla.Map, start_waypoint: carla.Waypoint, route_length: int = 40, waypoint_spacing: float = 3.0) -> None:
        """
        이 함수의 역할:
        RouteManager를 초기화하고 시작 waypoint부터 route를 생성한다.

        이 함수가 언제 호출되는지:
        CarlaEnv._setup_route()에서 차량 spawn 후 route를 만들 때 호출된다.

        입력 값 설명:
        world_map: CARLA 지도 객체.
        start_waypoint: route 시작점.
        route_length: 최대 waypoint 개수.
        waypoint_spacing: waypoint 간격(미터).

        출력 값 설명:
        반환값은 없고, self.route와 target_index를 초기화한다.
        """
        self.world_map = world_map
        self.route_length = route_length
        self.waypoint_spacing = waypoint_spacing
        self.route: List[carla.Waypoint] = self._generate_route(start_waypoint)
        self.target_index = 1 if len(self.route) > 1 else 0
        self.completed = False

    def _generate_route(self, start_waypoint: carla.Waypoint) -> List[carla.Waypoint]:
        """
        이 함수의 역할:
        시작점부터 앞으로 이어지는 waypoint들을 선택해 route 리스트를 만든다.

        이 함수가 언제 호출되는지:
        __init__ 내부에서 route를 처음 생성할 때 호출된다.

        입력 값 설명:
        start_waypoint: route 생성 시작 waypoint.

        출력 값 설명:
        List[carla.Waypoint]: 생성된 route waypoint 리스트.
        """
        route = [start_waypoint]
        current = start_waypoint

        for _ in range(self.route_length - 1):
            next_candidates = current.next(self.waypoint_spacing)
            if not next_candidates:
                break
            # 여러 후보 중 현재 차선/방향과 가장 자연스럽게 이어지는 후보를 선택한다.
            current = self._select_best_candidate(current, next_candidates)
            route.append(current)

        if len(route) < 2:
            raise RuntimeError("Failed to generate a usable waypoint route.")
        return route

    def _select_best_candidate(self, current_waypoint: carla.Waypoint, candidates: List[carla.Waypoint]) -> carla.Waypoint:
        """
        이 함수의 역할:
        다음 waypoint 후보 중 가장 적절한 후보를 점수화해서 선택한다.

        이 함수가 언제 호출되는지:
        _generate_route()가 route를 늘려 갈 때 반복적으로 호출된다.

        입력 값 설명:
        current_waypoint: 현재 waypoint.
        candidates: 다음 후보 waypoint 리스트.

        출력 값 설명:
        carla.Waypoint: 선택된 다음 waypoint.
        """
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
        """
        이 함수의 역할:
        현재 차량 위치를 기준으로 route 상 어느 waypoint까지 진행했는지 갱신한다.

        이 함수가 언제 호출되는지:
        target waypoint를 찾거나 route metric을 계산하기 전에 호출된다.

        입력 값 설명:
        vehicle_location: ego 차량 현재 위치.
        reach_threshold: waypoint 도달로 볼 거리 기준.
        search_window: 앞으로 탐색할 waypoint 개수.

        출력 값 설명:
        반환값은 없고, self.target_index와 self.completed를 갱신한다.
        """
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

        self.completed = self.reached_goal(vehicle_location, threshold=reach_threshold)

    def get_target_waypoint(self, vehicle_location: carla.Location, lookahead: float = 5.0) -> carla.Waypoint:
        """
        이 함수의 역할:
        현재 차량 기준 일정 거리 앞의 target waypoint를 반환한다.

        이 함수가 언제 호출되는지:
        heuristic controller가 조향 목표점을 정할 때 주로 호출된다.

        입력 값 설명:
        vehicle_location: 현재 차량 위치.
        lookahead: 현재 위치에서 최소 몇 미터 앞 waypoint를 고를지 정하는 값.

        출력 값 설명:
        carla.Waypoint: 현재 조향 대상 waypoint.
        """
        self.update_progress(vehicle_location)

        for idx in range(self.target_index, len(self.route)):
            waypoint = self.route[idx]
            if vehicle_location.distance(waypoint.transform.location) >= lookahead:
                return waypoint
        return self.route[-1]

    def get_current_target_waypoint(self) -> carla.Waypoint:
        """
        이 함수의 역할:
        현재 target_index가 가리키는 waypoint를 반환한다.

        이 함수가 언제 호출되는지:
        lane_offset, heading_error 계산 시 사용된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        carla.Waypoint: 현재 target waypoint.
        """
        return self.route[self.target_index]

    def get_lookahead_waypoints(self, count: int = 3) -> List[carla.Waypoint]:
        """
        이 함수의 역할:
        현재 target 이후의 lookahead waypoint들을 여러 개 반환한다.

        이 함수가 언제 호출되는지:
        route vector를 만들거나 디버깅 정보를 만들 때 호출된다.

        입력 값 설명:
        count: 몇 개의 waypoint를 미리 볼지 정하는 값.

        출력 값 설명:
        List[carla.Waypoint]: lookahead waypoint 리스트.
        """
        waypoints: List[carla.Waypoint] = []
        for offset in range(count):
            idx = min(self.target_index + offset, len(self.route) - 1)
            waypoints.append(self.route[idx])
        return waypoints

    def vehicle_to_waypoint_relative(self, vehicle_transform: carla.Transform, waypoint: carla.Waypoint) -> tuple[float, float]:
        """
        이 함수의 역할:
        waypoint를 차량 local frame 기준의 거리와 상대 각도로 바꾼다.

        이 함수가 언제 호출되는지:
        route vector를 만들 때 waypoint를 숫자 feature로 변환하기 위해 사용된다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.
        waypoint: 상대 위치를 계산할 waypoint.

        출력 값 설명:
        tuple[float, float]: (거리 meter, 상대 각도 radian)
        """
        local_x, local_y = world_to_local_2d(vehicle_transform, waypoint.transform.location)
        distance = float((local_x ** 2 + local_y ** 2) ** 0.5)
        angle = float(np.arctan2(local_y, max(local_x, 1e-6)))
        return distance, angle

    def build_route_vector(self, vehicle_transform: carla.Transform, count: int = 3) -> np.ndarray:
        """
        이 함수의 역할:
        lookahead waypoint들을 [dist1, angle1, dist2, angle2, ...] 형태의 벡터로 만든다.

        이 함수가 언제 호출되는지:
        StateBuilder가 route observation을 구성할 때 호출된다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.
        count: 사용할 lookahead waypoint 수.

        출력 값 설명:
        np.ndarray: route feature float32 벡터.
        """
        features: List[float] = []
        for waypoint in self.get_lookahead_waypoints(count=count):
            distance, angle = self.vehicle_to_waypoint_relative(vehicle_transform, waypoint)
            features.extend([distance, angle])
        return np.asarray(features, dtype=np.float32)

    def compute_route_progress(self) -> float:
        """
        이 함수의 역할:
        전체 route 중 현재 얼마나 진행했는지를 0~1 범위로 계산한다.

        이 함수가 언제 호출되는지:
        reward 계산, info 구성, 로그 기록 등에 사용된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        float: route progress 값.
        """
        if len(self.route) <= 1:
            return 1.0
        return min(1.0, self.target_index / float(len(self.route) - 1))

    def compute_lane_offset(self, vehicle_transform: carla.Transform) -> float:
        """
        이 함수의 역할:
        차선 중심선 대비 ego 차량의 횡방향 오프셋을 계산한다.

        이 함수가 언제 호출되는지:
        reward, offroad 판단, info 구성 등에 사용된다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.

        출력 값 설명:
        float: lane center 기준 lateral offset.
        """
        reference_waypoint = self.get_current_target_waypoint()
        ref_transform = reference_waypoint.transform
        dx = vehicle_transform.location.x - ref_transform.location.x
        dy = vehicle_transform.location.y - ref_transform.location.y
        yaw_rad = self._deg_to_rad(ref_transform.rotation.yaw)

        right_x = -self._sin(yaw_rad)
        right_y = self._cos(yaw_rad)
        return dx * right_x + dy * right_y

    def compute_heading_error(self, vehicle_transform: carla.Transform) -> float:
        """
        이 함수의 역할:
        차량 진행 방향과 target waypoint 방향의 차이를 계산한다.

        이 함수가 언제 호출되는지:
        controller, reward, info 구성에 사용된다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.

        출력 값 설명:
        float: heading error(degree).
        """
        target_waypoint = self.get_current_target_waypoint()
        return self._angle_diff_deg(target_waypoint.transform.rotation.yaw, vehicle_transform.rotation.yaw)

    def reached_goal(self, vehicle_location: carla.Location, threshold: float = 3.0) -> bool:
        """
        이 함수의 역할:
        마지막 waypoint 근처에 도달했는지 판정한다.

        이 함수가 언제 호출되는지:
        episode 종료 조건(goal) 판단에 사용된다.

        입력 값 설명:
        vehicle_location: 차량 현재 위치.
        threshold: 마지막 waypoint에 충분히 가까운 거리 기준.

        출력 값 설명:
        bool: goal 도달 여부.
        """
        final_distance = vehicle_location.distance(self.route[-1].transform.location)
        return self.target_index >= len(self.route) - 1 and final_distance <= threshold

    def build_metrics(self, vehicle_transform: carla.Transform) -> RouteMetrics:
        """
        이 함수의 역할:
        현재 차량 상태를 기준으로 핵심 route metric들을 한 번에 계산한다.

        이 함수가 언제 호출되는지:
        CarlaEnv가 observation과 info를 만들기 전에 호출한다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.

        출력 값 설명:
        RouteMetrics: route 관련 핵심 상태 묶음.
        """
        self.update_progress(vehicle_transform.location)
        return RouteMetrics(
            target_index=self.target_index,
            route_progress=self.compute_route_progress(),
            lane_offset=self.compute_lane_offset(vehicle_transform),
            heading_error=self.compute_heading_error(vehicle_transform),
            reached_goal=self.reached_goal(vehicle_transform.location),
        )

    def get_debug_snapshot(self, vehicle_transform: carla.Transform) -> Dict[str, object]:
        """
        이 함수의 역할:
        디버깅에 유용한 route 상태 snapshot dict를 만든다.

        이 함수가 언제 호출되는지:
        observation 구성이나 상태 확인 시 사용될 수 있다.

        입력 값 설명:
        vehicle_transform: 차량 현재 transform.

        출력 값 설명:
        Dict[str, object]: route 관련 수치와 lookahead waypoint 좌표 정보.
        """
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
        """
        이 함수의 역할:
        아직 남아 있는 waypoint 개수를 반환한다.

        이 함수가 언제 호출되는지:
        로그/디버깅 시 잔여 route 길이를 보고 싶을 때 사용된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        int: 남은 waypoint 수.
        """
        return max(0, len(self.route) - 1 - self.target_index)

    def is_completed(self) -> bool:
        """
        이 함수의 역할:
        route 완료 여부를 간단히 반환한다.

        이 함수가 언제 호출되는지:
        외부에서 완료 여부만 빠르게 확인할 때 사용된다.

        입력 값 설명:
        없음.

        출력 값 설명:
        bool: route 완료 여부.
        """
        return self.completed

    @staticmethod
    def _angle_diff_deg(target_deg: float, current_deg: float) -> float:
        """
        이 함수의 역할:
        두 각도 차이를 -180~180 범위로 정규화한다.

        이 함수가 언제 호출되는지:
        heading error나 waypoint 후보 점수 계산 시 사용된다.

        입력 값 설명:
        target_deg: 목표 각도.
        current_deg: 현재 각도.

        출력 값 설명:
        float: 정규화된 각도 차이(degree).
        """
        return (target_deg - current_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _deg_to_rad(degrees: float) -> float:
        """
        이 함수의 역할:
        degree를 radian으로 바꾼다.

        이 함수가 언제 호출되는지:
        lane offset 계산에서 sin/cos를 사용할 때 호출된다.

        입력 값 설명:
        degrees: degree 각도 값.

        출력 값 설명:
        float: radian 값.
        """
        return degrees * 3.141592653589793 / 180.0

    @staticmethod
    def _sin(value: float) -> float:
        """
        이 함수의 역할:
        sin 계산 helper.

        이 함수가 언제 호출되는지:
        lane offset 계산 시 사용된다.

        입력 값 설명:
        value: radian 값.

        출력 값 설명:
        float: sin(value)
        """
        import math

        return math.sin(value)

    @staticmethod
    def _cos(value: float) -> float:
        """
        이 함수의 역할:
        cos 계산 helper.

        이 함수가 언제 호출되는지:
        lane offset 계산 시 사용된다.

        입력 값 설명:
        value: radian 값.

        출력 값 설명:
        float: cos(value)
        """
        import math

        return math.cos(value)
