# Stabilization Round 08: Narrow Pre-Offroad Speed Guard v2

## Summary

This round adjusted the Round 07 guard to reduce deterministic evaluation jitter while keeping the guard narrow.

Result:

```text
Status: PASS-AS-DIAGNOSTIC / HOLD-AS-STABILIZATION
4096 run: skipped
Promotion: no
Best checkpoint remains unchanged
```

Current best remains:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

## Subagent Decision

Subagents agreed that Round 07 proved the narrow guard direction was useful, but exact promotion gates had to stay conservative.

Key conclusions:

```text
Round 07 reduced training risk but failed eval jitter.
Round 08 should soften brake/throttle behavior.
Heading 10 -> 8 had weaker evidence than the jitter changes.
Promotion requires training distribution and deterministic eval, not eval alone.
```

## Config Changes

New configs:

```text
configs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_smoke.yaml
rcs_ac_first/configs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_smoke.yaml
```

Changed from Round 07:

```yaml
lateral_recovery:
  speed_gate_heading_error_threshold_deg: 8.0
  speed_gate_target_speed_kmh: 10.0
  speed_gate_throttle_cap: 0.06
  speed_gate_brake_kp: 0.040
  speed_gate_max_brake: 0.20
```

Kept unchanged:

```yaml
lateral_recovery:
  speed_gate_lane_offset_threshold_m: 1.80
  speed_gate_min_speed_kmh: 16.0
  speed_gate_curve_threshold: 0.0
  speed_gate_branch_only: false
```

Intent:

```text
Lower jitter by making the guard less abrupt.
Slightly widen heading coverage.
Avoid broad speed governance by preserving lane and speed thresholds.
```

## Code/Logging Fix

File:

```text
shared_core/env/carla_env.py
```

Fixed info forwarding for:

```text
lateral_recovery_speed_gate_heading_error_threshold_deg
lateral_recovery_speed_gate_min_speed_kmh
```

Round 07 behavior used the configured values, but CSV logging for those two thresholds defaulted to zero because `_build_info()` did not forward them. This has been fixed for future runs.

## Validation

Static checks:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Result:

```text
PASS
```

Config load:

```text
model_variant: phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v2
speed_gate_heading_error_threshold_deg: 8.0
speed_gate_min_speed_kmh: 16.0
speed_gate_target_speed_kmh: 10.0
speed_gate_throttle_cap: 0.06
speed_gate_brake_kp: 0.04
speed_gate_max_brake: 0.20
```

CARLA preflight:

```text
0.9.16
```

## 2048 Smoke

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048 `
  --device auto
```

Outputs:

```text
logs/recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048_20260430T030449/training_log.csv
outputs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_env/recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048_20260430T030449/env_step_log.csv
checkpoints/recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048_20260430T030449/last.pt
```

Console episodes:

```text
max_steps: 3
offroad: 2
collision: 0
stuck: 0
```

CSV analysis using speed at action resolution:

```text
rows: 2054
terminal_route_mean: 0.66668
terminal_route_max: 0.9487
risk10_rows: 14
risk10_rate: 0.00682
risk8_rows: 26
risk8_rate: 0.01266
guard_rows: 21
guard_rate: 0.01022
guard_on_risk8: 0.769
false_guard_outside_risk8: 1
pre20_guard_rate: 0.15
pre20_risk8_rate: 0.20
pre20_speed_at_resolution_mean: 15.42 km/h
pre20_lane_abs_mean: 2.166 m
pre20_heading_abs_mean: 18.552 deg
early_offroad_under_0.20: 0
```

Compared with Round 07:

```text
eval jitter improved
early offroad improved from 1 to 0
route mean improved from 0.6359 to 0.6667
strict risk10 rate held at 0.0068
intended v2 risk8 rate rose to 0.0127
pre20 guard coverage stayed low: 0.175 -> 0.15
```

## 2048 Deterministic Eval

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048_20260430T030449\last.pt `
  --run-name recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048_eval_512 `
  --num-steps 512 `
  --device auto
```

Output:

```text
outputs/bc_eval/recovery_imitation_v2_narrow_preoffroad_speed_guard_v2_2048_eval_512_20260430T031038/summary.json
```

Result:

```text
pass_status: PASS
done_reason_counts: goal 1
max_route_progress: 1.0
collision/offroad/stuck: 0/0/0
mean_action_jitter_l1: 0.09488
mean_longitudinal_jitter: 0.08703
mean_steer_jitter: 0.00784
```

This fixed the Round 07 eval jitter failure:

```text
Round 07 jitter: 0.10737
Round 08 jitter: 0.09488
Hard ceiling: 0.100
Current best jitter: 0.09357
```

## Gate Decision

Round 08 passed the eval-jitter problem and improved several 2048 training signals.

However, it is not promotion-ready:

```text
intended v2 risk8_rate = 0.01266, above 0.007 gate
guard_on_risk8 = 0.769, below 0.95 target
pre20_guard_rate = 0.15, below 0.45 target
training terminal max route did not reach 1.0
```

Decision:

```text
Do not run 4096 for this exact v2 config.
Do not promote the 2048 checkpoint.
Keep current best unchanged.
```

## Interpretation

Round 08 shows the guard intensity can be softened enough to preserve clean eval smoothness.

But the mechanism is still not catching the actual pre-offroad tail often enough. In particular, pre20 guard coverage did not improve from Round 07. This means the remaining offroad tail is not simply a high-speed/high-heading condition; the vehicle can be near offroad with speed just below the speed threshold or with timing that misses the guard.

## Next Recommendation

Recommended next round should not further broaden heading. Instead:

```text
Keep the softer v2 control strength.
Keep heading threshold at 8.
Test a slightly lower min speed threshold, 14 km/h, while adding a stricter false-activation gate.
```

Candidate:

```yaml
lateral_recovery:
  speed_gate_lane_offset_threshold_m: 1.80
  speed_gate_heading_error_threshold_deg: 8.0
  speed_gate_min_speed_kmh: 14.0
  speed_gate_target_speed_kmh: 10.0
  speed_gate_throttle_cap: 0.06
  speed_gate_brake_kp: 0.035
  speed_gate_max_brake: 0.18
```

Why:

```text
Round 08 pre20 speed mean was 15.42 km/h, so a 16 km/h threshold misses much of the real tail.
Control should remain softer than Round 07 to preserve eval jitter.
```

Promotion gate should remain strict:

```text
2048 offroad <= 2
terminal_route_mean >= 0.65
risk10_rate <= 0.007
false guard outside intended predicate <= 0.002
pre20_guard_rate >= 0.45
eval safety 0/0/0
eval max route 1.0
eval action jitter <= 0.095
```

