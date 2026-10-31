# Stabilization Round 04: Speed Gate v2b

## Hypothesis

`recovery_imitation_v2_speed_gate_2048` failed because the speed gate did not stay active through the real offroad tail and because the gate was applied before action smoothing.

This round keeps the hypothesis narrow:

```text
Make the existing branch-turn speed gate actually activate through the branch-exit tail and enforce its throttle/brake effect after smoothing.
```

## Baseline

Baseline:

```text
outputs/phase3_bc_clean_lateral_recovery_v2_env/recovery_imitation_v1_ppo_2048_20260429T164500/env_step_log.csv
```

Previous candidate:

```text
outputs/phase3_bc_clean_recovery_imitation_v2_speed_gate_env/recovery_imitation_v2_speed_gate_2048_20260429T175429/env_step_log.csv
```

## Regression-First Summary

The previous `speed_gate_2048` candidate should not be promoted to 4096.

Regressions versus `recovery_imitation_v1_2048`:

| Metric | v1 2048 | speed_gate 2048 | Judgment |
|---|---:|---:|---|
| offroad episodes | 4 | 4 | unchanged |
| max_steps episodes | 2 | 2 | unchanged |
| best route progress | 0.6923 | 0.6667 | regressed |
| terminal episode mean progress | about 0.414 | about 0.388 | regressed |
| low-speed exposure | lower | higher | regression risk |

False-win risk: `HIGH`

Reason:

```text
The gate reduced speed in some windows, but did not reduce offroad and slightly reduced route progress.
```

## Mechanism Analysis

The previous speed gate did activate:

```text
active rows: 51 / 2055
active rows in route progress 0.26~0.36: 48
active mean abs lane offset: about 2.04 m
active mean speed: about 9.9 km/h
active mean target speed: about 6.8 km/h
```

But it activated too late:

```text
min abs lane offset while active: about 1.70 m
mean abs lane offset while active: about 2.04 m
```

This means the gate mostly acted after the vehicle was already near offroad.

The curve threshold also caused misses:

```text
Some right-side branch-exit failures had lane offset > 2.0 m,
but curve_proxy was about 7.6, below speed_gate_curve_threshold 14.0.
So the gate turned off or never activated in the terminal tail.
```

Code audit also found a real implementation issue:

```text
The speed gate was applied before action smoothing.
With speed_gate_throttle_cap=0.0 and throttle_fall_limit=0.25,
final control could still contain throttle after the gate.
```

## Changes

### Code

File:

```text
shared_core/env/carla_env.py
```

Changes:

```text
1. Added complete lateral recovery default fields to step_expert_autopilot action_resolution.
   This prevents the expert autopilot path from missing required _build_info keys.

2. Re-applied speed-gate throttle/brake enforcement after action smoothing.
   When lateral_recovery_speed_gate_active is true:
     - final throttle is capped by speed_gate_throttle_cap
     - brake is raised to the gated brake request if speed is above the gated target
     - smoothed_throttle/smoothed_brake are updated to match the final control
```

### Config

New config:

```text
configs/phase3_bc_clean_recovery_imitation_v2b_speed_gate_smoke.yaml
```

Base:

```text
configs/phase3_bc_clean_recovery_imitation_v2_speed_gate_smoke.yaml
```

Changed:

```yaml
env:
  lateral_recovery:
    speed_gate_curve_threshold: 6.0
```

Kept:

```yaml
env:
  lateral_recovery:
    speed_gate_lane_offset_threshold_m: 1.60
    speed_gate_target_speed_kmh: 8.0
    speed_gate_throttle_cap: 0.0
    speed_gate_brake_kp: 0.080
    speed_gate_max_brake: 0.35
```

Reverted reward turn-speed changes to the v2 baseline values:

```yaml
reward:
  turn_speed_penalty_weight: 0.008
  turn_speed_delta_threshold_rad: 0.20
  turn_speed_free_kmh: 20.0
```

Reason:

```text
This isolates the experiment to speed-gate activation/enforcement instead of mixing in reward changes.
```

## Validation

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Result: `PASS`

Import/log schema validation:

```powershell
py -3.12 -B -c "from shared_core.env.carla_env import CarlaEnv; from shared_core.env.log_schema import ENV_STEP_LOG_FIELDS; print('import_ok', len(ENV_STEP_LOG_FIELDS))"
```

Result:

```text
import_ok 308
```

## CARLA 2048 Attempt 1

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2b_speed_gate_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v1_bc_256\epoch_001.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v2b_speed_gate_2048 `
  --device auto
```

Result: `BLOCKED`

Reason:

```text
CARLA simulator did not respond on 127.0.0.1:2000.
RuntimeError: time-out of 30000ms while waiting for the simulator.
```

Preflight:

```powershell
py -3.12 -B -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())"
```

Result:

```text
RuntimeError: time-out of 10000ms while waiting for the simulator.
```

## Resume Execution

CARLA was later restarted and responded successfully:

```text
CARLA server version: 0.9.16
```

## CARLA 2048 Result

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2b_speed_gate_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v1_bc_256\epoch_001.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v2b_speed_gate_2048 `
  --device auto
```

Run:

```text
logs/recovery_imitation_v2b_speed_gate_2048_20260429T202202/training_log.csv
outputs/phase3_bc_clean_recovery_imitation_v2b_speed_gate_env/recovery_imitation_v2b_speed_gate_2048_20260429T202202/env_step_log.csv
checkpoints/recovery_imitation_v2b_speed_gate_2048_20260429T202202/last.pt
```

Surface result:

```text
updates: 16
max_steps episodes: 2
offroad episodes: 4
collision/stuck: 0
route_progress max: 0.7949
```

2048 gate diagnostics:

| Metric | v1 2048 | speed_gate 2048 | v2b 2048 |
|---|---:|---:|---:|
| offroad episodes | 4 | 4 | 4 |
| max_steps episodes | 2 | 2 | 2 |
| terminal route max | 0.6923 | 0.6667 | 0.7949 |
| route 0.26~0.36 speed mean | 13.88 | 12.26 | 10.50 |
| high-lane zone gate rate | 0.00 | 0.585 | 0.957 |
| offroad last-10 gate rate | 0.00 | 0.50 | 1.00 |
| offroad last-10 speed mean | 16.62 | 10.30 | 8.25 |
| offroad last-10 lane abs mean | 2.33 | 2.16 | 2.20 |
| offroad last-10 throttle max | 0.468 | 0.48 | 0.00 |

2048 gate decision:

```text
CONDITIONAL PASS
```

Reason:

```text
The primary failure count did not improve, but the mechanism activation and final throttle enforcement were fixed:
  - last-10 gate active rate reached 1.00
  - last-10 speed dropped to 8.25 km/h
  - final throttle was actually capped to 0.00 in the offroad tail
  - route max recovered above the v1 2048 baseline
```

This was enough to run 4096 as a formal diagnostic.

## CARLA 4096 Result

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2b_speed_gate_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v1_bc_256\epoch_001.pt `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name recovery_imitation_v2b_speed_gate_4096 `
  --device auto
```

Run:

```text
logs/recovery_imitation_v2b_speed_gate_4096_20260429T202733/training_log.csv
outputs/phase3_bc_clean_recovery_imitation_v2b_speed_gate_env/recovery_imitation_v2b_speed_gate_4096_20260429T202733/env_step_log.csv
checkpoints/recovery_imitation_v2b_speed_gate_4096_20260429T202733/last.pt
```

Surface result:

| Metric | recovery imitation v1 4096 | speed gate v2b 4096 |
|---|---:|---:|
| max_steps episodes | 5 | 5 |
| offroad episodes | 6 | 6 |
| collision episodes | 0 | 0 |
| stuck episodes | 0 | 0 |
| terminal route progress mean | 0.4872 | 0.4637 |
| terminal route progress max | 0.7692 | 0.7692 |

Mechanism diagnostics:

| Metric | recovery imitation v1 4096 | speed gate v2b 4096 |
|---|---:|---:|
| route 0.26~0.36 speed mean | 13.43 | 12.18 |
| route 0.26~0.36 lane abs mean | 0.90 | 1.23 |
| high-lane zone speed mean | 16.02 | 7.65 |
| high-lane zone gate rate | 0.00 | 0.886 |
| speed-gate active rows | 0 | 181 |
| gate throttle max | n/a | 0.00 |
| offroad last-10 gate rate | 0.00 | 1.00 |
| offroad last-10 speed mean | 15.29 | 8.64 |
| offroad last-10 lane abs mean | 2.39 | 2.29 |
| offroad last-10 brake mean | 0.102 | 0.204 |

Interpretation:

```text
The speed gate now activates and enforces throttle correctly.
However, offroad count did not decrease.
The car is slower near offroad, but lateral recovery still does not pull it back into lane.
```

## Deterministic Eval

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2b_speed_gate_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2b_speed_gate_4096_20260429T202733\last.pt `
  --run-name recovery_imitation_v2b_speed_gate_eval_512 `
  --num-steps 512 `
  --device auto
```

Result:

| Metric | recovery imitation v1 eval | speed gate v2b eval |
|---|---:|---:|
| pass status | PASS | PASS |
| final route progress | 0.9231 | 0.8205 |
| max route progress | 0.9231 | 0.8205 |
| collision/offroad/stuck | 0/0/0 | 0/0/0 |
| mean steer jitter | 0.0060 | 0.0067 |
| mean action jitter L1 | 0.0784 | 0.0805 |

## Regression-First Final Summary

Regressed:

```text
1. Deterministic eval route progress fell from 0.9231 to 0.8205.
2. 4096 terminal route progress mean fell from 0.4872 to 0.4637.
3. Route 0.26~0.36 lane abs mean increased from 0.90 to 1.23.
4. Action jitter increased mildly.
```

Improved:

```text
1. Speed gate activation now reaches the offroad tail.
2. Final throttle is actually capped to 0 while gate is active.
3. High-lane-zone speed dropped from 16.02 km/h to 7.65 km/h.
4. Offroad last-10 speed dropped from 15.29 km/h to 8.64 km/h.
```

Unchanged:

```text
1. 4096 offroad episodes remained 6.
2. 4096 max_steps episodes remained 5.
3. Collision and stuck remained 0.
```

## Final Judgment

Status: `FAIL-AS-STABILIZATION / PASS-AS-DIAGNOSTIC`

Reason:

```text
The round proved that speed gating can be made active and enforce final throttle/brake correctly.
But slowing alone did not reduce offroad and worsened deterministic eval route progress.
This should not replace recovery_imitation_v1 as the current best checkpoint.
```

Best current checkpoint remains:

```text
checkpoints/recovery_imitation_v1_ppo_4096_20260429T164950/last.pt
```

Do not promote:

```text
checkpoints/recovery_imitation_v2b_speed_gate_4096_20260429T202733/last.pt
```

## Next Ablation

The next hypothesis should not be "more speed gate".

Evidence says:

```text
The vehicle can be slowed near offroad, but lane offset still stays around 2.3m and offroad still occurs.
The missing component is lateral recovery behavior under low speed/high offset, not just longitudinal slowing.
```

Recommended next experiment:

```text
recovery_imitation_v2_lateral_target
```

Primary change:

```text
Collect a larger recovery dataset from the v1 checkpoint or v2b control wrapper,
but train the policy on corrected steering targets only.
Do not train gated longitudinal targets yet, because v2b showed aggressive slowdowns hurt eval progress.
```

Suggested dataset:

```text
num_transitions: 512
route_progress_min: 0.24
route_progress_max: 0.38
lane_offset_threshold: 0.45
heading_error_threshold: 6.0
```

2048 gate for next round:

```text
offroad <= 3
or offroad remains 4 but offroad terminal lane_abs_mean decreases by at least 10%
route max >= 0.69
deterministic eval should not fall below 0.90 if 4096 is run
```
