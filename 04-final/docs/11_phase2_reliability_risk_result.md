# Phase 2 Reliability + Risk-aware Fusion 결과

작성일: 2026-04-26

## Summary

Phase 2는 `docs/04_phase2_reliability_risk_execution_plan.md` 기준으로 구현 및 검증했다.

최종 판단: **Phase 2 완료, Phase 3 진입 가능**

Phase 2에서 완료한 핵심 목표는 다음과 같다.

- auxiliary observation `[quality_stats, risk_stats, route_commitment_stats, sensor_dropout_mask]`를 모델/PPO에 연결
- reliability head와 risk/context head 추가
- fusion gate가 reliability, risk scalar, risk context를 실제 gate input으로 사용
- rollout buffer와 PPO update에 auxiliary, reliability target, risk target, dropout mask 저장
- weak corruption controller를 `shared_core` CARLA env에 추가
- clean CARLA smoke와 mild corruption CARLA 2048-step smoke 통과

## 구현 모듈

주요 추가/변경 파일:

- `rcs_ac_first/runtime/auxiliary.py`
- `rcs_ac_first/runtime/phase2_targets.py`
- `rcs_ac_first/models/reliability.py`
- `rcs_ac_first/models/risk_context.py`
- `rcs_ac_first/models/fusion_gate.py`
- `rcs_ac_first/models/rcs_ac_model.py`
- `rcs_ac_first/rl/rollout_buffer.py`
- `rcs_ac_first/rl/ppo_agent.py`
- `rcs_ac_first/rl/losses.py`
- `rcs_ac_first/scripts/train.py`
- `rcs_ac_first/scripts/smoke_check.py`
- `configs/phase2_reliability_risk_smoke.yaml`
- `shared_core/env/corruption.py`
- `shared_core/env/carla_env.py`
- `shared_core/env/log_schema.py`

## 실행한 명령어

Static:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Local smoke:

```powershell
py -3.12 -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase2_reliability_risk_smoke.yaml
```

CARLA preflight:

```powershell
py -3.12 -B -c "from rcs_ac_first.scripts.train import _bootstrap_carla; _bootstrap_carla(); import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())"
```

CARLA clean:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase2_reliability_risk_smoke.yaml --total-timesteps 512 --run-name phase2_clean_512 --device auto
```

CARLA mild corruption:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase2_reliability_risk_smoke.yaml --total-timesteps 2048 --run-name phase2_mild_corruption_2048 --device auto
```

Checkpoint reload:

```powershell
py -3.12 -B -c "from pathlib import Path; from rcs_ac_first.runtime.config import load_phase1_config; from rcs_ac_first.rl.ppo_agent import PPOAgent, PPOConfig; cfg=load_phase1_config(Path('configs/phase2_reliability_risk_smoke.yaml')); agent=PPOAgent(PPOConfig.from_phase1_config(cfg, device_override='auto')); ck=agent.load('checkpoints/phase2_mild_corruption_2048_20260426T145626/last.pt', map_location=agent.device); print(ck.get('extra_state', {}))"
```

## Local Smoke 결과

통과.

확인된 항목:

- obs dim: `47311`
- auxiliary shape: `[4, 30]`
- reliability shape: `[4, 4]`
- risk scalar shape: `[4]`
- risk context shape: `[4, 64]`
- fusion weights row sum: `1.0`
- reliability/risk losses finite
- PPO backward 성공
- checkpoint save/load 성공
- recurrent rollout update 성공
- forced corruption 반응 성공

Forced corruption 요약:

```text
clean_reliability_rgb=1.0
corrupt_reliability_rgb=0.05
clean_fusion_rgb=0.218567
corrupt_fusion_rgb=0.006997

clean_reliability_lidar=1.0
corrupt_reliability_lidar=0.05
clean_fusion_lidar=0.217323
corrupt_fusion_lidar=0.007024
```

## CARLA Clean Smoke 결과

통과.

Run:

```text
phase2_clean_512_20260426T145510
```

생성 파일:

```text
logs/phase2_clean_512_20260426T145510/training_log.csv
outputs/phase2_reliability_risk_env/phase2_clean_512_20260426T145510/env_step_log.csv
checkpoints/phase2_clean_512_20260426T145510/last.pt
```

요약:

- total_steps: `512`
- PPO updates: `4`
- env_step rows: `515`
- max corruption: `0`
- `sensor_dropout_mask_all_zero=False` rows: `0`
- checkpoint 생성: 성공

## CARLA Mild Corruption Smoke 결과

통과.

Run:

```text
phase2_mild_corruption_2048_20260426T145626
```

생성 파일:

```text
logs/phase2_mild_corruption_2048_20260426T145626/training_log.csv
outputs/phase2_reliability_risk_env/phase2_mild_corruption_2048_20260426T145626/env_step_log.csv
checkpoints/phase2_mild_corruption_2048_20260426T145626/last.pt
```

최종 update row:

```text
total_steps=2048
update=16
actor_loss=8.844956755638123e-05
critic_loss=7.105638027191162
entropy=2.391589879989624
reliability_loss=0.0018716549966484308
risk_supervision_loss=1.230102384397469e-06
risk_regularization=0.010313273407518864
risk_scalar=0.09567467868328094
risk_target=0.09761261940002441
corruption_level=0.15000000596046448
drop_rgb=0.125
drop_lidar=0.125
```

env_step corruption 요약:

```text
rows=2062
max_corruption=0.6
drop_rgb_rows=240
drop_lidar_rows=192
drop_ego_rows=99
drop_route_rows=32
sensor_dropout_mask_all_zero=False rows=563
phase2_fields_present=True
```

Checkpoint reload:

```text
{'update': 16, 'total_steps': 2048, 'episode_index': 13}
```

## 실패/보정 항목

처음 mild corruption run에서 `sensor_dropout_mask_all_zero`가 env step CSV에 없어서 acceptance 확인이 불완전했다.

조치:

- `shared_core/env/log_schema.py`에 `sensor_dropout_mask_all_zero` 추가
- `shared_core/env/carla_env.py` CSV row에 해당 필드 기록
- rollout buffer에 dropout mask 저장
- PPO update metrics에 `drop_rgb/drop_lidar/drop_ego/drop_route` 평균 기록
- clean 512와 mild corruption 2048을 재실행해 통과 확인

## Phase 3 진입 판단

Phase 3 진입 가능.

Phase 2에서 의도한 reliability/risk-aware fusion 기반은 닫혔다. Phase 3에서는 Shapley contribution, dropout consistency, full robustness evaluation을 구현하면 된다.
