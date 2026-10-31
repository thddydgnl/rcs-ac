# Stabilization Round 07: Narrow Pre-Offroad Speed Guard

## Summary

This round tested a narrow runtime speed/control guard on top of the current best v2 lateral-target checkpoint.

The goal was not to add more imitation data or rewrite rewards. The hypothesis was:

```text
When the policy enters a high-risk pre-offroad state, steering recovery alone is too late.
Apply a narrow target-speed/throttle/brake guard only under the high-risk condition.
```

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

## Subagent Inputs

Four subagents were used:

```text
Agent A - Log Analyst
Agent B - Code Auditor
Agent C - Mechanism Verifier
Agent D - Safety Regression Analyst
```

Shared conclusion:

```text
GO for a narrow pre-offroad speed/control guard.
Do not retry broad speed gates.
Do not add more recovery imitation data in this round.
Do not promote on clean eval alone.
```

The key high-risk predicate was:

```text
abs(lane_offset) >= 1.8 m
abs(heading_error) >= 10 deg
speed >= 16 km/h
```

## Code Changes

### Runtime Guard Conditions

File:

```text
shared_core/env/carla_env.py
```

Added two explicit lateral recovery speed-gate thresholds:

```yaml
speed_gate_heading_error_threshold_deg
speed_gate_min_speed_kmh
```

The speed gate now requires:

```text
lateral_recovery_speed_gate_enabled
target_speed action contract
branch condition, unless branch_only=false
curve threshold
lane offset threshold
heading error threshold
minimum speed threshold
```

This keeps the gate narrow and prevents it from becoming a global speed governor.

### Logging

Files:

```text
shared_core/env/log_schema.py
shared_core/env/carla_env.py
```

Added env-step log fields:

```text
lateral_recovery_speed_gate_heading_error_threshold_deg
lateral_recovery_speed_gate_min_speed_kmh
```

## Config Changes

New configs:

```text
configs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_smoke.yaml
rcs_ac_first/configs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_smoke.yaml
```

Important guard settings:

```yaml
lateral_recovery:
  speed_gate_enabled: true
  speed_gate_branch_only: false
  speed_gate_curve_threshold: 0.0
  speed_gate_lane_offset_threshold_m: 1.80
  speed_gate_heading_error_threshold_deg: 10.0
  speed_gate_min_speed_kmh: 16.0
  speed_gate_target_speed_kmh: 9.0
  speed_gate_throttle_cap: 0.04
  speed_gate_brake_kp: 0.060
  speed_gate_max_brake: 0.25
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

CARLA preflight:

```powershell
py -3.12 -B -c "from rcs_ac_first.scripts.train import _bootstrap_carla; _bootstrap_carla(); import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())"
```

Result:

```text
0.9.16
```

## 2048 Smoke

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v2_narrow_preoffroad_speed_guard_2048 `
  --device auto
```

Outputs:

```text
logs/recovery_imitation_v2_narrow_preoffroad_speed_guard_2048_20260430T025417/training_log.csv
outputs/phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_env/recovery_imitation_v2_narrow_preoffroad_speed_guard_2048_20260430T025417/env_step_log.csv
checkpoints/recovery_imitation_v2_narrow_preoffroad_speed_guard_2048_20260430T025417/last.pt
```

Console episodes:

```text
max_steps: 2
offroad: 2
goal: 1
collision: 0
stuck: 0
```

CSV analysis:

```text
rows: 2054
terminal_route_mean: 0.6359
terminal_route_max: 1.0
risk_rows: 14
risk_rate: 0.0068
guard_rows: 14
guard_rate: 0.0068
pre20_rows: 40
pre20_speed_mean: 17.5389 km/h
pre20_lane_abs_mean: 2.1537 m
pre20_heading_abs_mean: 8.9231 deg
pre20_guard_rate: 0.175
early_offroad_under_0.20: 1
```

Compared with blocked v3:

```text
v3 offroad: 7 -> narrow guard: 2
v3 terminal route mean: 0.3333 -> narrow guard: 0.6359
v3 risk rate: 0.0262 -> narrow guard: 0.0068
```

This is a real diagnostic improvement.

## 2048 Deterministic Eval

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_narrow_preoffroad_speed_guard_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_narrow_preoffroad_speed_guard_2048_20260430T025417\last.pt `
  --run-name recovery_imitation_v2_narrow_preoffroad_speed_guard_2048_eval_512 `
  --num-steps 512 `
  --device auto
```

Output:

```text
outputs/bc_eval/recovery_imitation_v2_narrow_preoffroad_speed_guard_2048_eval_512_20260430T025931/summary.json
```

Result:

```text
pass_status: PASS
done_reason_counts: goal 1
max_route_progress: 1.0
collision/offroad/stuck: 0/0/0
mean_action_jitter_l1: 0.10737
mean_longitudinal_jitter: 0.09979
mean_steer_jitter: 0.00758
```

Safety passed, but action jitter exceeded the hard stabilization ceiling:

```text
hard ceiling: 0.100
observed: 0.10737
```

## Gate Decision

2048 smoke improved training behavior substantially:

```text
offroad <= 2
goal reached during training
terminal route mean improved
risk rate reduced
no collision/stuck
```

But it failed promotion criteria because:

```text
deterministic eval action jitter > 0.100
one early offroad remained at route progress < 0.20
guard coverage in pre20 offroad windows was only 17.5%
```

Decision:

```text
Do not run 4096 in this exact configuration.
Do not promote the 2048 checkpoint.
Keep v2 lateral-target 4096 as current best.
```

## Interpretation

The narrow guard is directionally useful. It reduced the high-risk exposure and greatly improved 2048 training distribution compared with v3.

However, the current settings are still not promotion-ready. The guard appears to catch the strict high-lane/high-heading/high-speed condition, but it does not cover enough of the actual pre-offroad tail because many tail rows have heading error just below 10 degrees.

The evaluation jitter issue also suggests that even if safety is clean, the PPO-updated checkpoint is less smooth than the current best.

## Next Recommendation

Recommended next round:

```text
recovery_imitation_v2_narrow_preoffroad_speed_guard_v2
```

Start again from the current best, not from this 2048 checkpoint:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

Use a slightly more covering but still narrow guard:

```yaml
speed_gate_lane_offset_threshold_m: 1.80
speed_gate_heading_error_threshold_deg: 8.0
speed_gate_min_speed_kmh: 16.0
speed_gate_target_speed_kmh: 10.0
speed_gate_throttle_cap: 0.06
speed_gate_brake_kp: 0.040
speed_gate_max_brake: 0.20
```

Rationale:

```text
Lower heading threshold from 10 to 8 to cover the real offroad tail.
Raise target speed/cap slightly and lower brake strength to reduce action jitter.
Keep lane and speed thresholds unchanged to avoid broad speed gating.
```

Promotion criteria should remain conservative:

```text
2048 offroad <= 2
terminal_route_mean >= 0.60
risk_rate <= 0.007
eval safety 0/0/0
eval goal or max_route_progress 1.0
eval action jitter <= 0.10
```

