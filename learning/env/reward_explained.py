"""
이 파일 전체 역할:
CARLA 환경에서 한 step이 끝났을 때 보상을 어떻게 계산할지 담당하는 학습용 설명 파일이다.
원본 reward.py의 로직은 그대로 유지하면서, 초보자도 보상 구조를 이해할 수 있도록 한국어 설명만 추가했다.

이 파일을 읽을 때 핵심 포인트:
1. 보상은 한 가지 값이 아니라 여러 항목의 합과 차로 구성된다.
2. 진행(progress)을 장려하고, 차선 이탈/충돌/급격한 제어 변화는 불이익을 준다.
3. breakdown dict를 통해 "왜 이런 reward가 나왔는지"를 디버깅할 수 있다.
"""

from __future__ import annotations

# dataclass:
# 설정값만 모아 둔 간단한 클래스를 만들 때 사용한다.
from dataclasses import dataclass
# Any:
# 타입이 매우 유연한 입력을 받을 때 사용한다.
from typing import Any

# numpy:
# 벡터 차이 계산, clip, 절댓값 계산 등에 사용한다.
import numpy as np


@dataclass
class RewardConfig:
    """
    이 클래스의 역할:
    보상 계산에 필요한 하이퍼파라미터(가중치, 패널티 크기 등)를 한 곳에 모아 두는 설정 클래스이다.

    이 클래스가 언제 사용되는지:
    RewardCalculator가 생성될 때 전달되며, 이후 매 step마다 reward를 계산할 때 기준값으로 사용된다.

    입력 값 설명:
    각 필드는 보상 항목의 상대적 크기를 정하는 실수 값이다.

    출력 값 설명:
    이 클래스 자체가 값을 계산하지는 않고, RewardCalculator가 참조할 설정 묶음을 제공한다.
    """

    # route_progress 증가량에 곱해지는 가중치
    progress_weight: float = 5.0
    # lane_offset이 작을수록 주는 보상의 크기
    lane_weight: float = 0.1
    # 충돌 시 부여하는 큰 음수 패널티
    collision_penalty: float = 10.0
    # offroad 상태일 때 부여하는 음수 패널티
    offroad_penalty: float = 5.0
    # lane invasion이 감지되면 부여하는 음수 패널티
    lane_invasion_penalty: float = 1.0
    # steer 변화량에 대한 비용 가중치
    steer_change_weight: float = 0.05
    # throttle 변화량에 대한 비용 가중치
    throttle_change_weight: float = 0.02
    # brake 변화량에 대한 비용 가중치
    brake_change_weight: float = 0.02
    # 목표 지점 도달 시 주는 보너스
    goal_bonus: float = 20.0
    # stuck 상태에 빠졌을 때 주는 패널티
    stuck_penalty: float = 3.0
    # progress delta가 비정상적으로 크게 튀지 않도록 잘라내는 범위
    progress_delta_clip: float = 1.0


class RewardCalculator:
    """
    이 클래스의 역할:
    이전 상태와 현재 상태를 비교해서 최종 scalar reward를 계산하는 클래스이다.

    이 클래스가 언제 사용되는지:
    CarlaEnv.step() 내부에서 observation과 info가 만들어진 뒤 reward를 계산할 때 사용된다.

    입력 값 설명:
    prev_state, current_state, info, 이전 action, 현재 action 등이 들어온다.

    출력 값 설명:
    (최종 reward, reward breakdown dict)를 반환한다.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """
        이 함수의 역할:
        보상 계산기 객체를 만들고 사용할 설정을 저장한다.

        이 함수가 언제 호출되는지:
        CarlaEnv가 초기화될 때 RewardCalculator를 생성하면서 호출된다.

        입력 값 설명:
        config:
            RewardConfig 객체이며, None이면 기본값 RewardConfig()를 생성해서 사용한다.

        출력 값 설명:
        반환값은 없고, self.config에 설정을 저장한다.
        """
        self.config = config or RewardConfig()

    def compute_reward(
        self,
        prev_state: dict[str, Any] | None,
        current_state: dict[str, Any],
        info: dict[str, Any],
        prev_control: np.ndarray,
        current_control: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """
        이 함수의 역할:
        이전 step과 현재 step을 비교해 최종 reward와 세부 항목별 breakdown을 계산한다.

        이 함수가 언제 호출되는지:
        CarlaEnv.step() 내부에서 현재 info가 준비된 뒤, done 판정 전후 reward를 계산할 때 호출된다.

        입력 값 설명:
        prev_state:
            이전 step의 obs/info를 담은 dict 또는 첫 step이면 None.
        current_state:
            현재 step의 obs/info를 담은 dict.
        info:
            현재 step에서 계산된 route_progress, lane_offset, collision 등의 부가 정보.
        prev_control:
            이전 step에서 사용한 action 벡터 [steer, throttle, brake].
        current_control:
            현재 step에서 사용한 action 벡터 [steer, throttle, brake].

        출력 값 설명:
        tuple[float, dict[str, float]]:
            첫 번째 값은 최종 reward,
            두 번째 값은 항목별 보상/패널티를 기록한 breakdown dict이다.
        """
        prev_progress = 0.0
        if prev_state is not None:
            # 첫 step이 아니면 이전 info에서 route_progress를 읽는다.
            prev_info = prev_state.get("info", {})
            prev_progress = float(prev_info.get("route_progress", 0.0))
        current_progress = float(info.get("route_progress", 0.0))

        # progress delta는 갑자기 너무 크게 튀지 않도록 clip 처리한다.
        progress_delta = float(
            np.clip(
                current_progress - prev_progress,
                -self.config.progress_delta_clip,
                self.config.progress_delta_clip,
            )
        )
        progress_reward = self.config.progress_weight * progress_delta

        # lane_offset 절댓값이 작을수록 보상이 커진다.
        lane_offset = abs(float(info.get("lane_offset", 0.0)))
        lane_reward = self.config.lane_weight * max(0.0, 1.0 - lane_offset)

        # 아래 항목들은 특정 이벤트가 발생하면 패널티/보너스를 부여한다.
        collision_penalty = self.config.collision_penalty if bool(info.get("collision", False)) else 0.0
        offroad_penalty = self.config.offroad_penalty if bool(info.get("offroad", False)) else 0.0
        lane_invasion_penalty = self.config.lane_invasion_penalty if bool(info.get("lane_invasion", False)) else 0.0
        goal_bonus = self.config.goal_bonus if bool(info.get("reached_goal", False)) else 0.0
        stuck_penalty = self.config.stuck_penalty if bool(info.get("stuck", False)) else 0.0

        # 제어 변화량이 클수록 부드럽지 않은 주행이라고 보고 비용을 준다.
        control_delta = np.abs(current_control.astype(np.float32) - prev_control.astype(np.float32))
        control_cost = (
            float(control_delta[0]) * self.config.steer_change_weight
            + float(control_delta[1]) * self.config.throttle_change_weight
            + float(control_delta[2]) * self.config.brake_change_weight
        )

        # 최종 reward는 여러 보상과 패널티를 합친 값이다.
        reward = (
            progress_reward
            + lane_reward
            - collision_penalty
            - offroad_penalty
            - lane_invasion_penalty
            - control_cost
            + goal_bonus
            - stuck_penalty
        )

        # breakdown은 로그/디버깅/학습 분석 시 매우 중요하다.
        breakdown = {
            "progress_reward": float(progress_reward),
            "lane_reward": float(lane_reward),
            "collision_penalty": float(collision_penalty),
            "offroad_penalty": float(offroad_penalty),
            "lane_invasion_penalty": float(lane_invasion_penalty),
            "control_cost": float(control_cost),
            "goal_bonus": float(goal_bonus),
            "stuck_penalty": float(stuck_penalty),
        }
        return float(reward), breakdown
