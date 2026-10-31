# Lateral Recovery v2 Stabilization Result

Created: 2026-04-29

## 1. Goal

Continue the sub-agent stabilization workflow after `lateral_recovery_v1d`.

The target was to reduce the remaining 8 offroad episodes from:

```text
outputs/phase3_bc_clean_lateral_recovery_v1d_env/lateral_recovery_v1d_4096_20260429T025058/env_step_log.csv
```

while preserving:

```text
collision: 0
stuck: 0
final eval PASS
```

## 2. Sub-Agent Diagnosis

### 2.1 Log Analyst

Main findings:

```text
All 8 remaining offroad failures are concentrated in turn/junction branch segments.
They occur around route_progress 0.2821~0.3333.

Pre-offroad aggregate:
  20-step |lane_offset|: 1.998 m
  10-step |lane_offset|: 2.349 m
  5-step  |lane_offset|: 2.499 m

Correction saturation:
  last 10 steps before offroad: 100%
  last 5 steps before offroad: 100%

Target speed is a contributor but not the root cause:
  some failures happen at about 6~8 km/h.
```

Interpretation:

```text
The remaining failures are not general clean driving failures.
They are branch-turn recovery failures where the lateral recovery correction is already saturated.
```

### 2.2 Code Auditor

Main findings:

```text
No confirmed implementation bug causing the remaining offroad.

Confirmed:
  - lateral recovery does not call build_metrics() and does not mutate route progress.
  - lane/heading units are stable.
  - train config pass-through includes lateral_recovery.
  - target-speed control remains active.
  - reward_terminal_component fix is present.

Risk:
  - env logs had behavior fields but not lateral recovery config values.
```

Action taken:

```text
Added lateral recovery config values to env_step_log.csv.
```

### 2.3 Control/RL Specialist

Recommendation:

```text
Test lateral_recovery_v2 by increasing recovery authority slightly.
Keep target speed, PPO, reward, smoothing, and offroad detection unchanged.
```

## 3. Implemented Change

Modified:

```text
shared_core/env/carla_env.py
shared_core/env/log_schema.py
```

Added:

```text
configs/phase3_bc_clean_lateral_recovery_v2_smoke.yaml
```

New v2 behavior:

```text
Base lateral recovery remains v1d.
Turn boost raises correction cap only when:
  lateral_recovery_turn_boost_enabled == true
  junction_branch_choice_active == true
  abs(curve_proxy) >= 18.0
  abs(lane_offset) >= 1.8
```

Config:

```yaml
env:
  lateral_recovery:
    enabled: true
    lane_offset_threshold_m: 0.55
    heading_error_threshold_deg: 8.0
    lane_gain: -0.080
    heading_gain: 0.006
    max_abs_correction: 0.22
    turn_boost_enabled: true
    turn_boost_curve_threshold: 18.0
    turn_boost_lane_offset_threshold_m: 1.80
    turn_boost_max_abs_correction: 0.32
```

New log fields:

```text
lateral_recovery_curve_proxy
lateral_recovery_branch_active
lateral_recovery_turn_boost_active
lateral_recovery_effective_max_abs_correction
lateral_recovery_lane_offset_threshold_m
lateral_recovery_heading_error_threshold_deg
lateral_recovery_lane_gain
lateral_recovery_heading_gain
lateral_recovery_max_abs_correction
lateral_recovery_turn_boost_enabled
lateral_recovery_turn_boost_curve_threshold
lateral_recovery_turn_boost_lane_offset_threshold_m
lateral_recovery_turn_boost_max_abs_correction
```

## 4. Commands

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Probe:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v1d_4096_20260429T025058\last.pt `
  --run-name lateral_recovery_v2_probe_eval_512 `
  --num-steps 512 `
  --device auto
```

2048 smoke:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v1d_4096_20260429T025058\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name lateral_recovery_v2_2048 `
  --device auto
```

4096 smoke:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v1d_4096_20260429T025058\last.pt `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name lateral_recovery_v2_4096 `
  --device auto
```

Final eval:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lateral_recovery_v2_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\lateral_recovery_v2_4096_20260429T103740\last.pt `
  --run-name lateral_recovery_v2_final_eval_512 `
  --num-steps 512 `
  --device auto
```

## 5. Results

### 5.1 Probe Eval

Run:

```text
lateral_recovery_v2_probe_eval_512_20260429T103052
```

Result:

```text
status: PASS
done_reason:
  max_steps: 1
collision: 0
offroad: 0
stuck: 0
route_max: 0.7179
turn_boost_rate: 0.0
```

Interpretation:

```text
Normal clean eval path was not affected by turn boost.
```

### 5.2 2048 Smoke

Run:

```text
lateral_recovery_v2_2048_20260429T103236
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
pre_offroad_turn_boost_rate: 0.9773
pre_offroad_correction_abs_mean: 0.3026
```

Interpretation:

```text
Boost activated in the intended failure windows.
Offroad count did not improve in 2048 compared with v1d.
```

### 5.3 4096 Smoke

Run:

```text
lateral_recovery_v2_4096_20260429T103740
```

Checkpoint:

```text
checkpoints/lateral_recovery_v2_4096_20260429T103740/last.pt
```

Result:

```text
updates: 32
done_reason:
  max_steps: 4
  offroad: 8
collision: 0
stuck: 0
route_mean: 0.2369
route_max: 0.7692
lane_abs_p90: 0.915
heading_abs_p90: 8.532
turn_boost_rate: 0.0243
pre_offroad_turn_boost_rate: 0.9091
```

### 5.4 Final Eval

Run:

```text
lateral_recovery_v2_final_eval_512_20260429T104640
```

Result:

```text
status: PASS
done_reason:
  max_steps: 1
collision: 0
offroad: 0
stuck: 0
route_mean: 0.4236
route_max: 0.8974
active_rate: 0.3314
turn_boost_rate: 0.0
```

## 6. Comparison

Training comparison:

```text
lateral_recovery_v1d_4096:
  max_steps: 4
  offroad: 8
  collision: 0
  stuck: 0
  route_mean: 0.2456
  route_max: 0.7436
  lane_abs_p90: 0.903
  heading_abs_p90: 10.309

lateral_recovery_v2_4096:
  max_steps: 4
  offroad: 8
  collision: 0
  stuck: 0
  route_mean: 0.2369
  route_max: 0.7692
  lane_abs_p90: 0.915
  heading_abs_p90: 8.532
```

Eval comparison:

```text
lateral_recovery_v1d_final_eval_512:
  max_steps: 1
  collision/offroad/stuck: 0
  route_mean: 0.3229
  route_max: 0.6667

lateral_recovery_v2_final_eval_512:
  max_steps: 1
  collision/offroad/stuck: 0
  route_mean: 0.4236
  route_max: 0.8974
```

## 7. Judgment

Status:

```text
WARN
```

Reasons:

```text
PASS aspects:
  - compileall passed
  - 512 probe passed
  - 2048 and 4096 PPO smoke completed
  - collision stayed 0
  - stuck stayed 0
  - final eval improved route progress substantially

WARN aspects:
  - 4096 training offroad did not improve: 8 -> 8
  - route_mean slightly decreased in training
  - pre-offroad boost activated, so simple correction-cap increase is not enough
```

Conclusion:

```text
lateral_recovery_v2 is safe and improves final deterministic eval, but it does not solve the remaining training offroad failure mode.
The remaining issue is not just insufficient correction cap.
The policy needs better turn recovery behavior around the branch-turn segment.
```

## 8. Next Recommendation

Next target:

```text
recovery_imitation_v1
```

Recommended scope:

```text
1. Collect or relabel expert recovery actions around high lane-offset turn failures.
2. Target the repeated route_progress 0.28~0.33 branch-turn segment.
3. Train a small recovery BC/fine-tune stage or add DAgger-style expert correction labels.
4. Keep lateral_recovery_v1d/v2 as a safety wrapper.
5. Do not keep increasing correction cap without new data; it already activates in failure windows.
```

Alternative narrow config experiment:

```text
lateral_recovery_v2_speed_gate:
  apply target-speed cap 10~12 km/h when:
    junction_branch_choice_active
    abs(curve_proxy) >= 18
    abs(lane_offset) >= 1.5

This is lower priority because two failures already occurred near 6~8 km/h.
```

