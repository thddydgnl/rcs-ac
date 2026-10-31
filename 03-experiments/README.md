# rcs-ac-03-experiments

RCS-AC 구조를 검증하기 위한 실험 저장소입니다. reliability calibration, corruption robustness, counterfactual Shapley alignment 등 여러 실험 단계를 스크립트와 결과 로그로 관리했습니다.

## 주요 내용

- Stage 1 reliability calibration 실험
- Stage 2 mild corruption / full validation 실험
- counterfactual Shapley 기반 contribution 방향성 검증
- vision/ego signal 분리와 logit calibration 실험
- baseline 비교, multi-seed confirm, pilot/full run 관리

## 프로젝트 흐름

이 저장소는 RCS-AC 계열 프로젝트의 3단계 작업입니다.

1. `rcs-ac-01-prototype`: CARLA + PPO 초기 프로토타입
2. `rcs-ac-02-rl-strategy`: CARLA 기반 전략 환경 MVP
3. `rcs-ac-03-experiments`: RCS-AC 실험 검증
4. `rcs-ac-04-final`: 최종 구조화 및 안정화 버전

## 구조

```text
configs/                 # 실험 설정
rcs_ac_v1/               # RCS-AC v1 구현
ppo_baselines/           # PPO baseline 비교 코드
docs/                    # 실험 문서와 분석 기록
results*/                # 실험 결과 산출물
run_*.py                 # 단계별 실험 실행 스크립트
tests/                   # 핵심 동작 검증 테스트
```

## 참고

이 저장소는 연구/실험 과정의 흔적을 보존하는 성격이 강합니다. `.DS_Store`, `__pycache__`, 로그, CSV 결과물이 일부 포함되어 있어 최종 제품형 저장소라기보다 실험 재현과 진행 기록에 가깝습니다.