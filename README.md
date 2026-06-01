# rcs-ac-04-final

RCS-AC 계열 프로젝트의 최종 정리 버전입니다. reliability, counterfactual contribution, risk-aware weighting을 결합한 멀티센서 actor-critic 구조를 구현하고, PPO/BC/안정화 실험을 정리했습니다.

## 주요 내용

- RGB, LiDAR, ego, route encoder 기반 멀티모달 입력 처리
- fusion gate, GRU actor-critic, PPO rollout/update 구현
- reliability head와 risk/context head 구성
- current critic 기반 Shapley contribution 계산
- masked value path, dropout consistency, Shapley evaluation
- expert dataset 수집, behavior cloning 학습/평가, 안정화 실험 기록

## 프로젝트 흐름

이 저장소는 RCS-AC 계열 프로젝트의 4단계 작업입니다.

1. `rcs-ac-01-prototype`: CARLA + PPO 초기 프로토타입
2. `rcs-ac-02-rl-strategy`: CARLA 기반 전략 환경 MVP
3. `rcs-ac-03-experiments`: RCS-AC 실험 검증
4. `rcs-ac-04-final`: 최종 구조화 및 안정화 버전

## 빠른 검증

CARLA 없이 로컬에서 모델 wiring, loss, checkpoint, recurrent rollout update를 확인합니다.

```bash
python -m rcs_ac_first.scripts.smoke_check --device cpu
python -m pytest tests
```

CARLA 환경이 준비되어 있으면 PPO smoke/train은 아래 entrypoint를 사용합니다.

```bash
python -m rcs_ac_first.scripts.train --config configs/phase3_full_shapley_smoke.yaml
```

BC 관련 entrypoint:

```bash
python -m rcs_ac_first.scripts.train_bc --help
python -m rcs_ac_first.scripts.evaluate_bc --help
```

Shapley 평가:

```bash
python -m rcs_ac_first.scripts.evaluate_shapley --help
```

## 구조

```text
configs/                  # 실행용 YAML config
data/expert/              # expert/BC dataset summary
docs/                     # 설계, 결과, 운영, 안정화 기록
rcs_ac_first/
  data/                   # dataset utilities
  models/                 # encoders, fusion, reliability/risk, actor-critic, masked critic
  rl/                     # PPO, rollout buffer, losses, distributions, Shapley
  runtime/                # observation/action/config/auxiliary contracts
  scripts/                # train, smoke, Shapley eval, BC tools, dataset collection
tests/                    # core contract regression tests
```

## 참고 문서

- `docs/00_three_phase_master_plan.md`: 3단계 구현 철학과 설계 기준
- `docs/01_phase_checklist.md`: Phase별 완료 기록
- `docs/12_phase3_counterfactual_shapley_result.md`: Phase 3 결과
- `docs/14_imitation_rl_execution_plan.md`: imitation/BC 이후 계획
- `docs/15_clean_stability_foundation_plan.md`: clean 주행 안정화 기반 계획