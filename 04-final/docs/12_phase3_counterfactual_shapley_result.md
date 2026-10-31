# Phase 3 Counterfactual Shapley Full RCS-AC 결과

작성일: 2026-04-26

## Summary

Phase 3는 `docs/05_phase3_counterfactual_shapley_execution_plan.md` 기준으로 구현 및 검증했다.

최종 판단: **Phase 3 구현 완료, 안정화 라운드 진입 가능**

완료한 핵심 목표:

- uniform placeholder였던 `contribution`을 current critic 기반 exact-ish Shapley contribution으로 교체
- `contribution`을 fusion gate 입력과 logit bias에 연결
- `shapley_alignment_loss`와 `dropout_consistency_loss`를 PPO update total loss에 연결
- representation-level modality mask 기반 dropout consistency 구현
- Phase 3 config, local smoke, CARLA clean smoke, CARLA mild corruption smoke, exact-ish eval script 구현

## 구현 모듈

주요 추가/변경 파일:

- `rcs_ac_first/runtime/modality_mask.py`
- `rcs_ac_first/runtime/config.py`
- `rcs_ac_first/models/fusion_gate.py`
- `rcs_ac_first/models/recurrent_actor_critic.py`
- `rcs_ac_first/models/masked_critic.py`
- `rcs_ac_first/models/rcs_ac_model.py`
- `rcs_ac_first/rl/shapley.py`
- `rcs_ac_first/rl/losses.py`
- `rcs_ac_first/rl/ppo_agent.py`
- `rcs_ac_first/scripts/smoke_check.py`
- `rcs_ac_first/scripts/train.py`
- `rcs_ac_first/scripts/evaluate_shapley.py`
- `configs/phase3_full_shapley_smoke.yaml`
- `rcs_ac_first/configs/phase3_full_shapley_smoke.yaml`

## 실행 명령어

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
py -3.12 -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_full_shapley_smoke.yaml
py -3.12 -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase2_reliability_risk_smoke.yaml
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase3_full_shapley_smoke.yaml --total-timesteps 512 --run-name phase3_clean_512 --device auto
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase3_full_shapley_smoke.yaml --total-timesteps 2048 --run-name phase3_mild_corruption_2048 --device auto
py -3.12 -B -m rcs_ac_first.scripts.evaluate_shapley --config .\configs\phase3_full_shapley_smoke.yaml --checkpoint .\checkpoints\phase3_mild_corruption_2048_20260426T202556\last.pt --run-name phase3_eval_256 --num-steps 256 --device auto
```

## Local Smoke 결과

통과.

```text
contribution shape=[4, 4]
contribution row sum=1.0
Shapley raw shape=[4, 4]
subset_count=16
shapley_alignment_loss=0.001207
dropout_consistency_loss=0.0
checkpoint save/load=pass
NaN/Inf=none
```

## CARLA Clean Smoke 결과

통과.

```text
run=phase3_clean_512_20260426T202419
total_steps=512
updates=4
env_rows=516
checkpoint=True
max_corruption=0
shapley_alignment_loss=0.004698911216109991
dropout_consistency_loss=2.586915343272267e-06
contribution_sum=1.0
```

생성 파일:

```text
logs/phase3_clean_512_20260426T202419/training_log.csv
outputs/phase3_full_shapley_env/phase3_clean_512_20260426T202419/env_step_log.csv
checkpoints/phase3_clean_512_20260426T202419/last.pt
```

## CARLA Mild Corruption Smoke 결과

통과.

```text
run=phase3_mild_corruption_2048_20260426T202556
total_steps=2048
updates=16
env_rows=2062
checkpoint=True
max_corruption=0.6
drop_rgb_rows=256
drop_lidar_rows=192
drop_ego_rows=53
drop_route_rows=48
shapley_alignment_loss=0.01141210738569498
dropout_consistency_loss=1.3646986189996824e-05
contribution_sum=1.00000002980232
```

생성 파일:

```text
logs/phase3_mild_corruption_2048_20260426T202556/training_log.csv
outputs/phase3_full_shapley_env/phase3_mild_corruption_2048_20260426T202556/env_step_log.csv
checkpoints/phase3_mild_corruption_2048_20260426T202556/last.pt
```

Checkpoint reload:

```text
{'update': 16, 'total_steps': 2048, 'episode_index': 13}
```

## Exact-ish Shapley Eval 결과

통과.

```text
run=phase3_eval_256_20260426T203040
steps=256
final_route_progress=0.02564102564102564
max_corruption=0.0
```

생성 파일:

```text
outputs/phase3_shapley_eval/phase3_eval_256_20260426T203040/contribution_timeseries.csv
outputs/phase3_shapley_eval/phase3_eval_256_20260426T203040/fusion_vs_contribution.csv
outputs/phase3_shapley_eval/phase3_eval_256_20260426T203040/scenario_summary.json
```

## 실패/보정 항목

formal validation 기준의 실패 항목은 없다.

주의:

- Phase 3 v1은 current critic 기반 exact-ish Shapley를 공식 acceptance path로 사용한다.
- `MaskedCritic`은 인터페이스와 smoke forward까지 구현했지만 기본 학습 경로에는 넣지 않았다.
- dropout consistency는 observation-level dropout이 아니라 representation-level modality mask로 구현했다.
- 성능 자체는 안정화 라운드에서 별도로 튜닝해야 한다.

## 안정화 라운드 진입 판단

안정화 라운드 진입 가능.

다음 라운드에서 볼 항목:

- Shapley/dropout consistency loss weight 조정
- contribution collapse 여부
- action jitter 감소 여부
- clean route progress 악화 여부
- mild/heavy corruption에서 modality contribution이 기대 방향으로 움직이는지
- masked critic을 기본 Shapley value source로 전환할지 여부
