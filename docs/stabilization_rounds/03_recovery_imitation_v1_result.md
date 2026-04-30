# Stabilization Round 03: Recovery Imitation v1

## Summary

이번 라운드는 `lateral_recovery_v2`의 남은 실패 원인을 reward 추가 조정보다 더 직접적으로 다루기 위해, 반복 offroad 구간의 recovery target을 수집하고 sequence BC로 보정한 뒤 PPO smoke를 다시 실행했다.

핵심 판단:

- v2의 실패는 대부분 route progress `0.28~0.33` branch-turn 구간에서 반복됐다.
- offroad 직전에는 lateral recovery correction이 이미 포화되는 경우가 많아, 단순 correction cap 증가나 reward penalty 추가만으로는 한계가 있다.
- 따라서 이번 라운드는 `recovery_imitation_v1`: 실패 구간의 corrected steering을 supervised target으로 학습하는 방식으로 진행했다.

## Subagent Findings

### Log Analyst

- `lateral_recovery_v2_4096`은 collision/stuck은 억제했지만 offroad가 8회 남았다.
- terminal offroad route progress는 대부분 `0.308~0.333` 근처였다.
- left/right branch 모두에서 lane offset과 heading error가 커진 뒤 correction이 포화됐다.
- 추천: branch-turn recovery 구간만 타겟팅한 imitation 보정.

### Code/Data Auditor

- 기존 expert dataset, sequence BC, init-checkpoint 로드는 대부분 재사용 가능했다.
- 누락된 부분은 “off-policy recovery 구간만 수집하는 dataset collector”였다.
- 최소 구현은 현재 policy rollout 중 lateral recovery corrected action을 target으로 저장하는 방식이 적절하다고 판단했다.

### Imitation/RL Specialist

- 추천 방식: `recovery_imitation_v1_branch_turn_bc`
- sequence length `32`, stride `4`, low learning rate `2e-5`
- BC는 전체 주행을 다시 학습하는 것이 아니라 실패 구간 correction만 약하게 보정하는 용도로 제한.

## Implemented Changes

### New Script

`rcs_ac_first/scripts/collect_recovery_dataset.py`

기능:

- checkpoint policy를 deterministic rollout
- route progress / lane offset / heading error / recovery active 조건으로 recovery transition 필터링
- target action:
  - steer: `lateral_recovery_corrected_steer`
  - longitudinal: current policy longitudinal
- `.npz` expert dataset 저장
- recovery 관련 optional metadata 저장

### Updated Script

`rcs_ac_first/scripts/train_bc.py`

추가:

- `--init-checkpoint`
- `--init-strict`

목적:

- 기존 v2 checkpoint weight에서 sequence BC를 시작할 수 있게 함
- optimizer는 새로 시작

### New Config

`configs/bc_sequence_recovery_imitation_v1_smoke.yaml`

핵심 설정:

- `bc.mode: sequence`
- `epochs: 3`
- `batch_size: 8`
- `learning_rate: 0.00002`
- `sequence_length: 32`
- `sequence_stride: 4`
- `include_bad_transitions: false`

## Commands

### Static Validation

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Result: PASS

### Recovery Dataset Collection

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_recovery_dataset `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v2_4096_20260429T103740\last.pt `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v1_256.npz `
  --num-transitions 256 `
  --max-steps 8192 `
  --run-name recovery_imitation_v1_collect_256 `
  --device auto
```

Result:

- attempted: `2500`
- accepted: `256`
- acceptance rate: `0.1024`
- route progress range: `0.2564~0.3590`
- route progress mean: `0.2847`
- bad transitions: `0`
- done transitions: `0`

### Sequence BC

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_sequence_recovery_imitation_v1_smoke.yaml `
  --dataset C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v1_256.npz `
  --run-name recovery_imitation_v1_bc_256 `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v2_4096_20260429T103740\last.pt `
  --init-strict true `
  --device auto
```

Result:

- epoch 1: train loss `0.031355`, val loss `0.015318`
- epoch 2: train loss `0.020863`, val loss `0.018681`
- epoch 3: train loss `0.031898`, val loss `0.023163`

Decision:

- `last.pt`는 overfit 경향이 있어 PPO init에는 사용하지 않음.
- `epoch_001.pt`를 사용.

### BC Eval

`last.pt`:

- final route progress: `0.5897`
- offroad/collision/stuck: `0`
- pass: `PASS`

`epoch_001.pt`:

- final route progress: `0.8205`
- offroad/collision/stuck: `0`
- pass: `PASS`

Decision:

- `epoch_001.pt`가 더 적합.

### PPO 2048 Smoke

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v1_bc_256\epoch_001.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v1_ppo_2048 `
  --device auto
```

Result:

- updates: `16`
- max_steps episodes: `2`
- offroad episodes: `4`
- collision/stuck: `0`
- route progress max: `0.692`

### PPO 4096 Formal Smoke

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v1_bc_256\epoch_001.pt `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name recovery_imitation_v1_ppo_4096 `
  --device auto
```

Result:

- updates: `32`
- max_steps episodes: `5`
- offroad episodes: `6`
- collision/stuck: `0`
- route progress max: `0.7692`
- route progress mean by episode: `0.4872`

Comparison with `lateral_recovery_v2_4096`:

| Metric | v2 4096 | recovery imitation v1 4096 |
|---|---:|---:|
| max_steps episodes | 4 | 5 |
| offroad episodes | 8 | 6 |
| collision episodes | 0 | 0 |
| stuck episodes | 0 | 0 |
| episode route progress mean | 0.4260 | 0.4872 |
| episode route progress max | 0.7692 | 0.7692 |

### Final Deterministic Eval

Checkpoint:

`checkpoints/recovery_imitation_v1_ppo_4096_20260429T164950/last.pt`

Result:

- completed steps: `512`
- done reason: `max_steps`
- final route progress: `0.9231`
- max route progress: `0.9231`
- offroad/collision/stuck: `0`
- mean steer jitter: `0.0060`
- mean longitudinal jitter: `0.0724`
- pass: `PASS`

Comparison with v2 final eval:

| Metric | v2 final eval | recovery imitation v1 final eval |
|---|---:|---:|
| final route progress | 0.8974 | 0.9231 |
| max route progress | 0.8974 | 0.9231 |
| offroad/collision/stuck | 0 | 0 |
| mean steer jitter | 0.0042 | 0.0060 |
| mean action jitter L1 | 0.0724 | 0.0784 |

## Diagnosis

이번 라운드는 효과가 있었다.

개선:

- 학습 중 offroad가 `8 -> 6`으로 감소
- max_steps episode가 `4 -> 5`로 증가
- episode route progress mean이 `0.4260 -> 0.4872`로 증가
- deterministic eval route progress가 `0.8974 -> 0.9231`로 증가

남은 문제:

- offroad terminal route progress는 여전히 `0.282~0.359`에 몰려 있다.
- 일부 terminal state에서는 correction이 여전히 포화된다.
- 특히 lane offset이 `2.5m~3.4m` 수준까지 커진 뒤에는 steer correction만으로 복구하기 어렵다.
- speed는 branch-turn/recovery 구간에서 v2보다 약간 올라갔다. 다음 라운드에서는 turn-zone speed cap을 같이 조정해야 한다.

## Decision

Status: `WARN-PASS`

이유:

- clean eval 기준은 개선되어 다음 단계 진행 가능.
- training smoke에서도 offroad가 줄어든 것은 의미 있는 개선.
- 하지만 training offroad가 완전히 사라지지 않았기 때문에 clean stabilization 완료로 보기는 어렵다.

## Recommended Next Round

`recovery_imitation_v2_speed_gated_branch_turn`

우선순위:

1. route progress `0.26~0.36` branch-turn 구간에서 target speed cap을 더 낮춘다.
2. lateral recovery가 active이고 abs lane offset이 큰 경우 throttle cap을 더 강하게 건다.
3. recovery imitation dataset을 `512~1024` transitions로 늘리되, epoch 1~2 early checkpoint를 선택한다.
4. PPO 4096 smoke 후 offroad terminal lane offset과 heading error를 다시 비교한다.

성공 기준:

- 4096 training offroad episodes `<= 4`
- collision/stuck `0`
- final deterministic eval route progress `>= 0.92`
- steer/action jitter가 v1보다 크게 증가하지 않을 것
