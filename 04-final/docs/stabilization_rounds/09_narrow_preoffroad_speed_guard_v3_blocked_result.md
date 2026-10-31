# Stabilization Round 09: Narrow Pre-Offroad Speed Guard v3

## Summary

This round followed the updated sub-agent workflow:

```text
sub-agents first
experiment brief second
config-only modification third
2048 diagnostic run fourth
document fifth
```

Result:

```text
Status: FAIL-AS-STABILIZATION / PASS-AS-DIAGNOSTIC
4096 run: skipped
deterministic eval: skipped
Promotion: no
Best checkpoint remains unchanged
```

Current best remains:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

## Experiment Brief

Round name:

```text
recovery_imitation_v2_narrow_preoffroad_speed_guard_v3
```

Starting checkpoint:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

Primary hypothesis:

```text
Round 08 missed part of the pre-offroad tail because speed_gate_min_speed_kmh=16 was too high.
Lowering it to 14 should catch the 14-16 km/h tail without becoming broad speed governance.
```

Allowed changes:

```text
config only
speed_gate_min_speed_kmh: 16.0 -> 14.0
speed_gate_brake_kp: 0.040 -> 0.035
speed_gate_max_brake: 0.20 -> 0.18
```

Rejected changes:

```text
no reward changes
no PPO hyperparameter changes
no BC/data changes
no code changes
no automatic 4096
```

## Sub-Agent Inputs

Agent A - Log Analyst:

```text
GO with WARN constraints.
The 14-16 km/h band should increase projected pre20 coverage from 0.15 to about 0.30.
This is useful as a 2048 diagnostic, not as a promotion-track change.
```

Agent B - Code Auditor:

```text
No blockers.
The path is config-only.
Round 08 logging fix is present in current source.
```

Agent C - Mechanism Verifier:

```text
Key proof is not more guard rows.
The new rows must concentrate in the 14-16 km/h pre-offroad tail without clean-state activation.
```

Agent D - Safety Regression:

```text
2048 continue requires offroad <=2, route mean >=0.65, risk10 <=0.007,
guard broadness controlled, and eval jitter <=0.095 if eval is reached.
```

## Config Changes

New configs:

```text
configs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_smoke.yaml
rcs_ac_first/configs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_smoke.yaml
```

Changed from Round 08:

```yaml
lateral_recovery:
  speed_gate_min_speed_kmh: 14.0
  speed_gate_brake_kp: 0.035
  speed_gate_max_brake: 0.18
```

Kept from Round 08:

```yaml
lateral_recovery:
  speed_gate_lane_offset_threshold_m: 1.80
  speed_gate_heading_error_threshold_deg: 8.0
  speed_gate_target_speed_kmh: 10.0
  speed_gate_throttle_cap: 0.06
  speed_gate_curve_threshold: 0.0
  speed_gate_branch_only: false
```

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
model_variant: phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v3
speed_gate_min_speed_kmh: 14.0
speed_gate_brake_kp: 0.035
speed_gate_max_brake: 0.18
```

CARLA preflight:

```text
0.9.16
```

## 2048 Smoke

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_2048 `
  --device auto
```

Outputs:

```text
logs/recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_2048_20260430T033539/training_log.csv
outputs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_env/recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_2048_20260430T033539/env_step_log.csv
checkpoints/recovery_imitation_v2_narrow_preoffroad_speed_guard_v3_2048_20260430T033539/last.pt
```

Console episodes:

```text
max_steps: 2
offroad: 4
collision: 0
stuck: 0
```

CSV analysis:

```text
rows: 2055
terminal_route_mean: 0.51283
terminal_route_max: 0.9231
risk10_rows: 18
risk10_rate: 0.00876
v3_intended_rows: 27
v3_intended_rate: 0.01314
guard_rows: 25
guard_rate: 0.01217
guard_recall: 0.852
guard_precision: 0.920
false_outside_v3_intended_rate: 0.00097
false_outside_narrow_geometry_rate: 0.00097
clean_speed_governance_rate: 0.0
subthreshold_activation_rate: 0.0
pre20_rows: 80
pre20_guard_rate: 0.175
pre20_v3_intended_rate: 0.175
pre20_v3_guard_recall: 0.929
pre20_speed_mean: 13.281 km/h
early_offroad_under_0.20: 1
```

## Gate Decision

Round 09 failed the 2048 stop/continue gate.

Hard failures:

```text
offroad = 4, gate required <=2
terminal_route_mean = 0.5128, gate required >=0.65
risk10_rate = 0.00876, gate required <=0.007
early_offroad_under_0.20 = 1, preferred 0
pre20_guard_rate = 0.175, far below target 0.45
```

Good diagnostic signals:

```text
guard_precision = 0.920
false activation stayed low
clean_speed_governance_rate = 0.0
subthreshold_activation_rate = 0.0
```

Interpretation:

```text
The lower speed threshold did not turn into a broad speed governor,
but it also did not solve the offroad tail.
The real tail now includes substantial rows below 14 km/h, so lowering
speed threshold alone is not enough and can hurt route distribution.
```

Decision:

```text
Do not run 4096.
Do not run deterministic eval.
Do not promote the v3 checkpoint.
Keep recovery_imitation_v2_lateral_target_4096 as current best.
```

## Comparison

```text
Round 08:
  offroad: 2
  route_mean: 0.66668
  route_max: 0.9487
  risk10_rate: 0.00682
  pre20_guard_rate: 0.15
  eval jitter: 0.09488

Round 09:
  offroad: 4
  route_mean: 0.51283
  route_max: 0.9231
  risk10_rate: 0.00876
  pre20_guard_rate: 0.175
  eval skipped
```

Round 09 caught slightly more intended rows but worsened core stabilization metrics.

## Next Recommendation

Do not continue lowering the speed threshold.

Recommended next direction:

```text
Return to Round 08 settings as the safer guard baseline.
Investigate why eligible rows are missed and why pre20 tail remains below the speed-gate condition.
The next experiment should focus on geometry/timing, not more speed-threshold lowering.
```

Candidate next hypotheses:

```text
1. Extend guard hold for 3-5 steps after activation, instead of lowering speed further.
2. Use offroad_candidate or lane_offset persistence as a timing signal.
3. Improve lateral steering correction timing rather than more brake intervention.
```

Preferred next small experiment:

```text
Round 10: preoffroad guard hold v1
Base: Round 08 v2 guard settings
Change: if the narrow guard activates, keep throttle cap / mild brake eligibility for 3 steps
Goal: cover intermittent tail misses without broadening the predicate.
```

