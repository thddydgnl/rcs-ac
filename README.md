# RCS-AC-first

작성일: 2026-04-26  
현재 상태 갱신: 2026-05-01

`RCS-AC-first`는 `RCS-AC_1-12_상세계획서.pdf`를 기준으로 reliability, counterfactual contribution, risk-aware weighting을 결합한 멀티센서 actor-critic을 구현하는 작업 공간이다.

현재 코드는 초기 scaffold 단계를 지나 다음 범위까지 구현되어 있다.

- Phase 1: RGB/LiDAR/ego/route encoder, fusion gate, GRU actor-critic, PPO rollout/update
- Phase 1.1: RGB dense perception auxiliary head와 mask loss
- Phase 2: reliability head, risk/context head, corruption-aware target, risk-aware fusion/loss
- Phase 3: current critic 기반 Shapley contribution, masked value path, dropout consistency, Shapley evaluation
- BC/안정화: expert dataset 수집, behavior cloning 학습/평가, stabilization round 기록

## 빠른 검증

CARLA 없이 로컬에서 모델 wiring, loss, checkpoint, recurrent rollout update를 확인한다.

```bash
python -m rcs_ac_first.scripts.smoke_check --device cpu
python -m pytest tests
```

CARLA 환경이 준비되어 있으면 PPO smoke/train은 아래 entrypoint를 사용한다.

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

## 주요 문서

- `docs/00_three_phase_master_plan.md`: 원래 3단계 구현 철학과 설계 기준
- `docs/01_phase_checklist.md`: Phase별 완료 기록과 과거 세부 체크리스트
- `docs/10_phase1_core_result.md`: Phase 1 결과
- `docs/11_phase2_reliability_risk_result.md`: Phase 2 결과
- `docs/12_phase3_counterfactual_shapley_result.md`: Phase 3 결과
- `docs/14_imitation_rl_execution_plan.md`: imitation/BC 이후 계획
- `docs/15_clean_stability_foundation_plan.md`: clean 주행 안정화 기반 계획
- `docs/stabilization_rounds/`: 안정화 실험 기록
- `docs/operating_guides/`: 원격 Mac/Windows 실험 운영 규칙

## 기준 참고 자료

- 원문 PDF: `C:/Users/thddy/Documents/code/RCS-AC/docs/RCS-AC_1-12_상세계획서.pdf`
- 텍스트 추출본: `C:/Users/thddy/Documents/code/RCS-AC/docs/RCS-AC_1-12_상세계획서.txt`
- 기존 smoke 구현 참고: `C:/Users/thddy/Documents/code/RCS-AC/rcs_ac`
- shared runtime 참고: `C:/Users/thddy/Documents/code/shared_core`
- PPO 학습 루프 참고: `C:/Users/thddy/Documents/code/attention`

## 현재 코드 구조

```text
RCS-AC-first/
  configs/                  # 실행용 YAML config
  data/expert/              # expert/BC dataset summary
  docs/                     # 설계, 결과, 운영, 안정화 기록
  rcs_ac_first/
    data/                   # dataset utilities
    models/                 # encoders, fusion, reliability/risk, actor-critic, masked critic
    rl/                     # PPO, rollout buffer, losses, distributions, Shapley
    runtime/                # observation/action/config/auxiliary contracts
    scripts/                # train, smoke, Shapley eval, BC tools, dataset collection
  tests/                    # pytest regression tests for core contracts
```
