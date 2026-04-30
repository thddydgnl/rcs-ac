# Lateral Recovery v1 Stabilization Result

Created: 2026-04-29

## 1. Trigger

User trigger:

```text
서브에이전트로 안정화 진행해줘
```

Workflow source:

```text
docs/stabilization_rounds/00_agent_workflow_rules.md
```

## 2. Baseline

Primary baseline:

```text
control_v2b training:
  outputs/phase3_bc_clean_control_v2_env/phase3_bc_clean_control_v2b_4096_20260427T174353/env_step_log.csv
  logs/phase3_bc_clean_control_v2b_4096_20260427T174353/training_log.csv

control_v2b checkpoint:
  checkpoints/phase3_bc_clean_control_v2b_4096_20260427T174353/last.pt

control_v2b eval:
  outputs/bc_eval/phase3_bc_clean_control_v2b_eval_512_20260427T175252
```

Baseline metrics:

```text
control_v2b_4096:
  done_reason:
    offroad: 31
  collision: 0
  stuck: 0
  route_mean: 0.0965
  route_max: 0.3077
  pre_done_lane_abs: 2.472 m
  pre_done_heading_abs: 19.486 deg

control_v2b_final_eval_512:
  done_reason:
    offroad: 2
  collision: 0
  stuck: 0
  route_mean: 0.1803
  route_max: 0.5128
```

## 3. Sub-Agent Findings

### 3.1 Log Analyst

Key findings:

```text
1. Dominant failure is lateral drift into lane-offset offroad.
   v2b training ended 31/31 completed episodes as offroad.

2. Pre-offroad correction is insufficient.
   v2b pre-offroad windows:
     20-step |lane_offset|: 2.038 m
     10-step |lane_offset|: 2.429 m
     5-step  |lane_offset|: 2.639 m

3. Steering did not scale enough before offroad.
   pre-10 |steer|: 0.103
   requested |steer|: 0.126

4. Speed reward remained active while lateral error was already large.
   pre-10 speed: about 17.7 km/h
```

Recommendation:

```text
Increase lateral recovery pressure before offroad.
Do not loosen offroad detection.
Keep control_v2b anti-stuck behavior.
```

### 3.2 Code Auditor

Key findings:

```text
Confirmed:
  - action contract is [steer, longitudinal]
  - target-speed mapping is correct
  - train/eval env config path is consistent
  - offroad detection is not obviously broken
  - reward signs are correct

Risk:
  - reward_terminal_component was stale in terminal rows because done_reason was updated after _build_info.
  - lane_offset sign/heading_error sign needed explicit verification before steering correction.
```

Required fix:

```text
Update reward_terminal_component after final done_reason is resolved.
```

### 3.3 Control/RL Specialist

Recommendation:

```text
Implement lateral_recovery_v1:
  - residual steering correction
  - activated by lane_offset / heading_error thresholds
  - preserve control_v2b longitudinal controller
  - log raw/corrected steer and correction state
```

## 4. Code And Config Changes

Modified:

```text
shared_core/env/carla_env.py
shared_core/env/log_schema.py
RCS-AC-first/rcs_ac_first/scripts/train.py
```

Added configs:

```text
configs/phase3_bc_clean_lateral_recovery_v1_smoke.yaml
configs/phase3_bc_clean_lateral_recovery_v1b_smoke.yaml
configs/phase3_bc_clean_lateral_recovery_v1c_smoke.yaml
configs/phase3_bc_clean_lateral_recovery_v1d_smoke.yaml
```

New env config block:

```yaml
env:
  lateral_recovery:
    enabled: true
    lane_offset_threshold_m: 0.55
    heading_error_threshold_deg: 8.0
    lane_gain: -0.080
    heading_gain: 0.006
    max_abs_correction: 0.22
```

Final accepted config:

```text
configs/phase3_bc_clean_lateral_recovery_v1d_smoke.yaml
```

New log fields:

```text
lateral_recovery_enabled
lateral_recovery_active
lateral_recovery_lane_error
lateral_recovery_heading_error
lateral_recovery_correction
lateral_recovery_raw_steer
lateral_recovery_corrected_steer
```

## 5. Validation Commands

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Probe and smoke commands:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v1d_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_control_v2b_4096_20260427T174353\last.pt `
  --run-name lateral_recovery_v1d_probe_eval_512 `
  --num-steps 512 `
  --device auto
```

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v1d_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_control_v2b_4096_20260427T174353\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name lateral_recovery_v1d_2048 `
  --device auto
```

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v1d_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_control_v2b_4096_20260427T174353\last.pt `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name lateral_recovery_v1d_4096 `
  --device auto
```

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v1d_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v1d_4096_20260429T025058\last.pt `
  --run-name lateral_recovery_v1d_final_eval_512 `
  --num-steps 512 `
  --device auto
```

## 6. Iteration Notes

### 6.1 v1

Initial v1:

```text
lane_offset_threshold_m: 0.55
heading_error_threshold_deg: 7.0
lane_gain: -0.045
heading_gain: 0.006
max_abs_correction: 0.16
```

Probe result:

```text
offroad: 0
stuck: 0
collision: 1
route_max: 0.7949
active_rate: 0.3346
```

2048 result:

```text
done_reason:
  collision: 3
  offroad: 4
route_max: 0.7949
active_rate: 0.6046
```

Judgment:

```text
Direction was correct, but correction behavior caused collision risk.
```

### 6.2 v1b / v1c

v1b raised activation thresholds. Collision still appeared.

v1c lowered max target speed and raised thresholds further. Collision still appeared with static vegetation.

Important sign check:

```text
lane_offset was decreasing under negative correction.
The correction sign was not reversed.
The issue was insufficient centering before vegetation collision, not opposite-sign steering.
```

### 6.3 v1d

Final v1d used:

```text
max_target_speed_kmh: 28.0
lane_offset_threshold_m: 0.55
heading_error_threshold_deg: 8.0
lane_gain: -0.080
heading_gain: 0.006
max_abs_correction: 0.22
```

Probe result:

```text
done_reason:
  max_steps: 1
collision: 0
offroad: 0
stuck: 0
route_max: 0.6923
active_rate: 0.1774
correction_abs_p90: 0.0658
```

## 7. Formal Results

### 7.1 2048 Smoke

Run:

```text
lateral_recovery_v1d_2048_20260429T024605
```

Result:

```text
updates: 16
done_reason:
  max_steps: 2
  offroad: 4
collision: 0
stuck: 0
route_max: 0.7692
route_mean: 0.2495
lane_abs_p90: 0.941
heading_abs_p90: 9.782
```

### 7.2 4096 Formal Smoke

Run:

```text
lateral_recovery_v1d_4096_20260429T025058
```

Checkpoint:

```text
checkpoints/lateral_recovery_v1d_4096_20260429T025058/last.pt
```

Result:

```text
updates: 32
done_reason:
  max_steps: 4
  offroad: 8
collision: 0
stuck: 0
route_mean: 0.2456
route_max: 0.7436
speed_mean: 11.554 km/h
speed_p90: 17.585 km/h
lane_abs_mean: 0.524 m
lane_abs_p90: 0.903 m
heading_abs_mean: 4.634 deg
heading_abs_p90: 10.309 deg
pre_done_lane_abs: 1.761 m
pre_done_heading_abs: 18.567 deg
lateral_recovery_active_rate: 0.4967
correction_abs_mean: 0.0430
correction_abs_p90: 0.1039
correction_abs_max: 0.2200
```

### 7.3 Final 512 Eval

Run:

```text
lateral_recovery_v1d_final_eval_512_20260429T030000
```

Result:

```text
status: PASS
done_reason:
  max_steps: 1
collision: 0
offroad: 0
stuck: 0
route_mean: 0.3229
route_max: 0.6667
speed_mean: 11.316 km/h
speed_p90: 17.715 km/h
lane_abs_mean: 0.410 m
lane_abs_p90: 0.598 m
heading_abs_mean: 1.916 deg
heading_abs_p90: 6.651 deg
lateral_recovery_active_rate: 0.1793
correction_abs_p90: 0.0621
```

## 8. Baseline Comparison

Training comparison:

```text
control_v2b_4096:
  offroad: 31
  max_steps: 0
  collision: 0
  stuck: 0
  route_mean: 0.0965
  route_max: 0.3077
  lane_abs_p90: 1.937

lateral_recovery_v1d_4096:
  offroad: 8
  max_steps: 4
  collision: 0
  stuck: 0
  route_mean: 0.2456
  route_max: 0.7436
  lane_abs_p90: 0.903
```

Eval comparison:

```text
control_v2b_final_eval_512:
  offroad: 2
  max_steps: 0
  collision: 0
  stuck: 0
  route_mean: 0.1803
  route_max: 0.5128

lateral_recovery_v1d_final_eval_512:
  offroad: 0
  max_steps: 1
  collision: 0
  stuck: 0
  route_mean: 0.3229
  route_max: 0.6667
```

## 9. Judgment

Status:

```text
PASS
```

Reasons:

```text
1. Static validation passed for RCS-AC-first and shared_core.
2. 512 probe passed.
3. 2048 smoke completed with 16 updates.
4. 4096 formal smoke completed with 32 updates.
5. Final checkpoint eval passed.
6. Collision remained 0 in accepted v1d 2048/4096/final eval.
7. Stuck remained 0.
8. Offroad decreased from 31 to 8 in 4096 training.
9. Route progress improved substantially.
10. Lane error distribution improved substantially.
```

Remaining issue:

```text
Offroad still occurs in training, especially around route_progress 0.28~0.33.
The residual controller improves clean stability but is still a wrapper, not learned policy competence.
```

## 10. Recommended Next Step

Next stabilization target:

```text
lateral_recovery_v2 or recovery_imitation_v1
```

Recommended order:

```text
1. Analyze the 8 remaining offroad episodes from lateral_recovery_v1d_4096.
2. Check whether they are caused by correction saturation, route geometry, or policy under-steer.
3. If correction saturation dominates, tune lateral_recovery_v2.
4. If policy under-steer dominates, collect recovery imitation data or DAgger-style expert labels near high lane offset.
5. Do not proceed to weather/heavy corruption until clean offroad count is lower and route progress is stable.
```

