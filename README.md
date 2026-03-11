# CARLA F1 Strategy MVP

이 저장소는 CARLA를 "드라이버 학습"이 아니라 "레이스 전략 의사결정" 검증용 테스트베드로 사용하는 최소 실행 가능 프로토타입입니다. 상위 RL 에이전트는 피트 전략만 결정하고, 저수준 차량 제어는 `BehaviorAgent` 또는 폴백 드라이버가 담당합니다.

## 프로젝트 구조

```text
carla_strategy_env/
  carla_integration/   # CARLA 연결, 액터/센서 생성, 트랙 헬퍼
  drivers/             # BehaviorAgent 및 폴백 주행 드라이버
  envs/                # Gymnasium 스타일 전략 환경
  strategy/            # 타이어/연료/피트스톱/랩타임 가상 물리 모델
config.py              # 모든 경로/환경/보상 설정
scripts/run_env.py     # 랜덤 액션 기반 smoke test
train.py               # 추후 RL 학습 진입점
data/maps/spa/         # Spa 트랙 자산
```

## 핵심 환경 설계

- 행동 공간: `Discrete(4)`
  - `0`: stay out
  - `1`: pit + soft
  - `2`: pit + medium
  - `3`: pit + hard
- 관측 공간: `Dict`
  - `vision`: 전방 RGB 카메라 이미지
  - `strategy`: 랩 수, 남은 랩 수, 타이어 종류, 마모도, 연료량, 직전 랩타임, 피트 횟수, 그립 등 수치 벡터
- 보상 요소:
  - 완주 보상
  - 충돌/주행 불능 패널티
  - 랩타임 개선 보상
  - 피트스톱 손실 반영
  - 진행률 보상
  - 약한 생존 페널티

## 사전 준비

### 1. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. CARLA 준비

- CARLA 서버가 실행 중이어야 합니다.
- CARLA Python API가 현재 Python 환경에서 import 가능해야 합니다.
- `BehaviorAgent`를 사용하려면 CARLA `PythonAPI/carla` 및 `PythonAPI/carla/agents` 경로가 `PYTHONPATH`에 포함되어야 합니다.
- 현재 코드는 Spa 맵이 이미 CARLA 런타임에 import/load 되어 있다고 가정합니다.

### 3. Spa 맵 관련

이 저장소에는 다음 Spa 자산이 포함되어 있습니다.

- `data/maps/spa/Spa_Circuit_Base.xodr`
- `data/maps/spa/Spa_Circuit_Base.udatasmith`
- `data/maps/spa/Spa_Circuit_Base.fbx`

중요:
- 이 MVP는 저장소 내부 자산을 직접 CARLA에 import하지 않습니다.
- 먼저 외부 CARLA/Unreal 워크플로우로 Spa 맵을 빌드해 CARLA에서 로드 가능한 상태로 준비해야 합니다.
- 실행 시 활성 맵 이름에 `config.py`의 `CarlaConfig.map_name` 값이 포함되지 않으면 명확한 오류를 발생시킵니다.

## 실행 방법

### 환경 smoke test

```bash
python scripts/run_env.py --laps 3 --steps 20
```

### 추후 학습 진입점

```bash
python train.py
```

## 설정 위치

모든 주요 설정은 `config.py`에 모여 있습니다.

- CARLA 접속 정보
- Spa 자산 경로와 기대 맵 이름
- 센서 해상도/FOV
- 랩 수, 피트 손실, 연료 소모량, 타이어 마모량
- 보상 가중치

## 확장 포인트

- `carla_strategy_env/strategy/models.py`
  - TODO: modality별 기여도 로깅 및 SVMM 연결 포인트 추가
  - TODO: 날씨/노면 변화 모델 고도화
- `carla_strategy_env/drivers/`
  - TODO: 더 안정적인 커스텀 레이싱 라인 추종 드라이버 추가
- `carla_strategy_env/envs/f1_strategy_env.py`
  - TODO: 멀티카 경쟁, 언더컷/오버컷, 세이프티카 이벤트 확장
- `train.py`
  - TODO: SB3 또는 사용자 정의 SVMM 학습 루프 연결

## 주의사항

- 이 프로젝트는 실제 F1 물리 재현이 목적이 아닙니다.
- 현재 MVP는 단일 차량 기준 전략 최적화 환경입니다.
- CARLA의 맵 import 상태, spawn point 구성, `BehaviorAgent` 가용성에 따라 일부 튜닝이 필요할 수 있습니다.
