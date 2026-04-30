# Phase 1 Core 결과 기록

작성일: 2026-04-26

## 구현 파일

- `rcs_ac_first/runtime/observation.py`
- `rcs_ac_first/runtime/action.py`
- `rcs_ac_first/runtime/config.py`
- `rcs_ac_first/models/encoders.py`
- `rcs_ac_first/models/fusion_gate.py`
- `rcs_ac_first/models/recurrent_actor_critic.py`
- `rcs_ac_first/models/rcs_ac_model.py`
- `rcs_ac_first/rl/distributions.py`
- `rcs_ac_first/rl/losses.py`
- `rcs_ac_first/rl/rollout_buffer.py`
- `rcs_ac_first/rl/ppo_agent.py`
- `rcs_ac_first/scripts/smoke_check.py`
- `rcs_ac_first/scripts/train.py`
- `configs/phase1_core_smoke.yaml`

## Local 검증

- compile: 통과
  - command: `py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first`
- shared runtime compile: 통과
  - command: `py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core`
- smoke check: 통과
  - command: `py -3.12 -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase1_core_smoke.yaml`
- dummy forward: 통과
  - action mean: `[4, 2]`
  - value: `[4]`
  - fusion weights: `[4, 4]`
  - fusion weight sums: `1.0`
  - road mask logits: `[4, 1, 45, 80]`
  - lane edge logits: `[4, 1, 45, 80]`
- dummy backward: 통과
  - total loss finite
  - finite grad parameter count: `59`
- checkpoint: 통과
  - path: `C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\smoke_check\last.pt`
- recurrent rollout update smoke: 통과
  - hidden state batch shape: `[8, 1, 256]`
  - initial hidden L1: `0.0`
  - max post-initial hidden L1: `0.03675998`
  - PPO update consumed stored hidden states without NaN

## CARLA Smoke

- CARLA Python API/server preflight: 통과
  - command: `py -3.12 -B -c "from rcs_ac_first.scripts.train import _bootstrap_carla; _bootstrap_carla(); import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())"`
  - result: `0.9.16`
- 128-step probe: 통과
  - command: `py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase1_core_smoke.yaml --total-timesteps 128 --run-name phase1_probe_128 --device auto`
  - run: `phase1_probe_128_20260426T140820`
  - updates: `1`
- 512-step smoke: 통과
  - command: `py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase1_core_smoke.yaml --total-timesteps 512 --run-name phase1_smoke_512 --device auto`
  - run: `phase1_smoke_512_20260426T140853`
  - updates: `4`
  - episodes: `2`
- 2048-step smoke: 통과
  - command: `py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase1_core_smoke.yaml --total-timesteps 2048 --run-name phase1_gate_2048 --device auto`
  - run: `phase1_gate_2048_20260426T141000`
  - updates: `16`
  - episodes: `11`
  - final checkpoint reload: 통과

## 생성 산출물

- 2048 training log:
  - `C:\Users\thddy\Documents\code\RCS-AC-first\logs\phase1_gate_2048_20260426T141000\training_log.csv`
  - rows: `27`
  - update rows: `16`
  - episode rows: `11`
- 2048 env step log:
  - `C:\Users\thddy\Documents\code\RCS-AC-first\outputs\phase1_core_env\phase1_gate_2048_20260426T141000\env_step_log.csv`
  - rows: `2060`
- 2048 checkpoint:
  - `C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase1_gate_2048_20260426T141000\last.pt`
  - extra state: `update=16`, `total_steps=2048`, `episode_index=11`

## 주요 로그

- final actor loss: `-2.72504985332489e-05`
- final critic loss: `4.370137691497803`
- final entropy: `2.391589879989624`
- final policy std mean: `0.800000011920929`
- final fusion weights:
  - RGB: `0.2153720259666443`
  - LiDAR: `0.21537655591964722`
  - ego: `0.2618982791900635`
  - route: `0.3073531687259674`
- final route progress: `0.02564102564102564`
- final done reason: `running`
- env action contract: `stage4_action_v2`
- env last action/control:
  - policy steer: `-0.4524`
  - policy longitudinal: `-0.9286`
  - applied throttle: `0.0`
  - applied brake: `0.05`

## 문제와 조치

- 2048-step smoke의 episode 종료는 대부분 `stuck`이다.
- Phase 1 목표는 성능이 아니라 end-to-end wiring 검증이므로 `goal_count=0`과 stuck episode는 Phase 1 실패로 보지 않는다.
- route progress, done reason, reward, action, value, std, fusion diagnostics가 기록되므로 Phase 2에서 reliability/risk와 주행 안정화 개선을 진행할 수 있다.

## Phase 2 진입 판단

- [x] 진입 가능
- [ ] 보류

판단 근거:

- Phase 1 local smoke가 재현 가능하게 통과했다.
- recurrent hidden state가 rollout/update 경로에 연결되었다.
- CARLA 128, 512, 2048-step smoke가 모두 통과했다.
- `training_log.csv`, `env_step_log.csv`, checkpoint가 생성되고 주요 필드가 finite로 확인되었다.
