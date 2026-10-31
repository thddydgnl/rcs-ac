# rcs-ac-01-prototype

CARLA 환경에서 자율주행 에이전트 학습을 실험하기 위한 초기 프로토타입입니다. 카메라/상태 관측을 처리하고, PPO 기반 학습 루프와 단일 모달·초기 멀티모달 비교 실험을 구성했습니다.

## 주요 내용

- CARLA 서버 연결과 경로 주행 환경 구성
- RGB/상태 관측 수집 및 저장
- PPO 학습 루프와 rollout buffer 구성
- single modal, early fusion, attention fusion 비교 실험
- 학습 로그, 체크포인트, 결과 요약 스크립트 포함

## 프로젝트 흐름

이 저장소는 RCS-AC 계열 프로젝트의 1단계 작업입니다.

1. `rcs-ac-01-prototype`: CARLA + PPO 초기 프로토타입
2. `rcs-ac-02-rl-strategy`: CARLA 기반 전략 환경 MVP
3. `rcs-ac-03-experiments`: RCS-AC 실험 검증
4. `rcs-ac-04-final`: 최종 구조화 및 안정화 버전

## 실행 예시

```bash
python main.py --host 127.0.0.1 --port 2000 --town Town04
python train.py --config configs/train_single_modal.yaml
```

CARLA Python API가 필요하며, `CARLA_ROOT` 환경변수 또는 로컬 CARLA 설치 경로를 사용합니다.

## 참고

초기 실험 저장소라 체크포인트, 로그, 결과 산출물이 함께 포함되어 있습니다. 포트폴리오에서는 이후 단계 저장소와 함께 프로젝트 발전 과정을 보여주는 용도로 보는 것이 좋습니다.