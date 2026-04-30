# Stabilization Round 05: Lateral Target v2

## Summary

Trigger:

```text
다음 단계를 서브에이전트로 안정화 진행해줘
```

Primary hypothesis:

```text
Speed gating proved that slowing the car is not enough.
The remaining clean instability is mostly lateral recovery steering around route_progress 0.24~0.38.
```

Final judgment:

```text
PASS-AS-CLEAN-EVAL / WARN-AS-TRAINING-STABILITY
```

The round produced the first clean deterministic eval that reached `goal` within 512 steps:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

However, training offroad episodes did not improve versus the previous best.

## Subagent Findings

### Log Analyst

Key finding:

```text
speed_gate_v2b slowed the vehicle, but offroad remained.
The failure is a high-offset lateral recovery dead zone, not pure speed.
```

Important diagnostics:

```text
v2b eval route: 0.8205 vs v1 eval route: 0.9231
v2b train offroad: 6, all around route_progress 0.2821~0.3333
one dead-zone example: speed about 4.23 km/h, abs lane offset about 2.09m, route_progress stuck
```

### Code Auditor

Important issues found:

```text
1. train_bc.py had steer_loss_weight / longitudinal_loss_weight in config but did not use them.
2. collect_recovery_dataset.py needed narrower recovery filters and extra metadata.
3. sequence BC over accepted-only sparse recovery rows can fake temporal continuity.
```

Implemented response:

```text
1. Added weighted action BC loss.
2. Added recovery dataset filters and metadata.
3. Used frame BC for this v2 smoke instead of sparse sequence BC.
```

### Safety Regression Analyst

Regression guard:

```text
Hard fail if deterministic eval route < 0.85, any eval safety event, jitter > 0.12,
training collision/stuck > 0, or training offroad >= 8.
```

Best-candidate criteria:

```text
eval route >= 0.90
eval safety all zero
jitter <= 0.10
training route mean >= 0.4872 or training offroad < 6
```

### Ablation Planner

Recommended experiment:

```text
base checkpoint: recovery_imitation_v1_ppo_4096
dataset: 512 transitions
route window: 0.24~0.38
lane_offset threshold: 0.45m
heading_error threshold: 6.0deg
target: corrected steering only, preserve policy longitudinal
```

## Code Changes

### train_bc.py

File:

```text
rcs_ac_first/scripts/train_bc.py
```

Change:

```text
steer_loss_weight and longitudinal_loss_weight are now used in frame and sequence BC losses.
```

Reason:

```text
The v2 lateral target experiment needs stronger steering imitation without aggressively rewriting longitudinal behavior.
```

### collect_recovery_dataset.py

File:

```text
rcs_ac_first/scripts/collect_recovery_dataset.py
```

Added filters:

```text
--speed-kmh-min
--speed-kmh-max
--accept-lateral-branch-only
--require-turn-boost
--require-speed-gate-active
```

Added dataset fields:

```text
speed_kmh
lateral_recovery_branch_active
lateral_recovery_speed_gate_active
lateral_recovery_raw_steer
```

### Configs

New configs:

```text
configs/bc_frame_recovery_imitation_v2_lateral_target_smoke.yaml
configs/bc_frame_recovery_imitation_v2_lateral_target_gentle_smoke.yaml
configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml
```

Package copies:

```text
rcs_ac_first/configs/bc_frame_recovery_imitation_v2_lateral_target_smoke.yaml
rcs_ac_first/configs/bc_frame_recovery_imitation_v2_lateral_target_gentle_smoke.yaml
rcs_ac_first/configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml
```

## Validation

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Result: `PASS`

CARLA preflight:

```text
CARLA 0.9.16
```

## Dataset

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_recovery_dataset `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v1_ppo_4096_20260429T164950\last.pt `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v2_lateral_target_512.npz `
  --num-transitions 512 `
  --max-steps 8192 `
  --run-name recovery_imitation_v2_lateral_target_collect_512 `
  --route-progress-min 0.24 `
  --route-progress-max 0.38 `
  --lane-offset-threshold 0.45 `
  --heading-error-threshold 6.0 `
  --device auto
```

Result:

```text
accepted: 512 / 3313
acceptance_rate: 0.1545
route_progress min/mean/max: 0.2564 / 0.3004 / 0.3590
bad_transition_count: 0
done_count: 0
```

Dataset:

```text
data/expert/recovery_imitation_v2_lateral_target_512.npz
```

## BC Attempts

### Strong Frame BC

Config:

```text
configs/bc_frame_recovery_imitation_v2_lateral_target_smoke.yaml
```

Settings:

```text
epochs: 3
learning_rate: 2e-5
steer_loss_weight: 4.0
longitudinal_loss_weight: 0.10
```

Result:

```text
epoch_001 eval route: 0.7692, no safety event
epoch_002 eval route: 0.2821, offroad 2
```

Judgment:

```text
NO-GO. Strong BC overwrote the good clean policy too much.
```

### Gentle Frame BC

Config:

```text
configs/bc_frame_recovery_imitation_v2_lateral_target_gentle_smoke.yaml
```

Settings:

```text
epochs: 1
learning_rate: 2e-6
steer_loss_weight: 2.0
longitudinal_loss_weight: 1.0
```

Result:

```text
eval route: 0.8718
collision/offroad/stuck: 0/0/0
mean action jitter L1: 0.0891
```

Judgment:

```text
CANDIDATE. Below the strict 0.90 eval threshold, but safe enough for a 2048 PPO gate.
```

Checkpoint:

```text
checkpoints/recovery_imitation_v2_lateral_target_gentle_bc_512/last.pt
```

## PPO 2048 Gate

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_gentle_bc_512\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v2_lateral_target_2048 `
  --device auto
```

Result:

```text
updates: 16
max_steps episodes: 3
offroad episodes: 2
collision/stuck: 0/0
terminal route mean: 0.5513
terminal route max: 0.9231
```

Deterministic eval:

```text
goal reached: 1
max route progress: 1.0
collision/offroad/stuck: 0/0/0
mean action jitter L1: 0.0920
```

Checkpoint:

```text
checkpoints/recovery_imitation_v2_lateral_target_2048_20260429T230049/last.pt
```

Judgment:

```text
PASS. Promoted to 4096 formal smoke.
```

## PPO 4096 Formal

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_gentle_bc_512\last.pt `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name recovery_imitation_v2_lateral_target_4096 `
  --device auto
```

Result:

```text
updates: 32
max_steps episodes: 5
offroad episodes: 7
collision/stuck: 0/0
terminal route mean: 0.5227
terminal route max: 0.9231
```

Comparison with previous best training:

| Metric | v1 best 4096 | v2 lateral target 4096 |
|---|---:|---:|
| max_steps episodes | 5 | 5 |
| offroad episodes | 6 | 7 |
| collision/stuck | 0/0 | 0/0 |
| terminal route mean | 0.4872 | 0.5227 |
| terminal route max | 0.7692 | 0.9231 |
| offroad tail abs lane mean | 2.3882 | 2.3312 |

Training judgment:

```text
Mixed. Route progress improved, but offroad count increased from 6 to 7.
```

Checkpoint:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

## Final Deterministic Eval

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --run-name recovery_imitation_v2_lateral_target_4096_eval_512 `
  --num-steps 512 `
  --device auto
```

Result:

```text
goal reached: 1
max route progress: 1.0
collision/offroad/stuck: 0/0/0
mean action jitter L1: 0.0936
mean steer jitter: 0.0082
```

Comparison with previous best eval:

| Metric | v1 best eval | v2 lateral target eval |
|---|---:|---:|
| done reason | max_steps | goal |
| max route progress | 0.9231 | 1.0000 |
| collision/offroad/stuck | 0/0/0 | 0/0/0 |
| mean action jitter L1 | 0.0784 | 0.0936 |
| mean steer jitter | 0.0060 | 0.0082 |

Eval judgment:

```text
PASS. This is the best clean deterministic eval so far.
```

## Current Best

Best clean deterministic eval checkpoint:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

Best conservative fallback checkpoint:

```text
checkpoints/recovery_imitation_v1_ppo_4096_20260429T164950/last.pt
```

Reason:

```text
v2 reaches goal in deterministic eval, but training offroad is still 7.
v1 has worse eval max route, but slightly lower training offroad.
```

## Next Recommendation

Do not add more speed gate yet.

Next round should target:

```text
reduce training offroad without losing v2's goal-reaching eval behavior
```

Recommended next experiment:

```text
recovery_imitation_v2_lateral_target_1024_gentle
```

But change only one thing:

```text
collect more diverse recovery samples, keep gentle BC strength
```

Suggested gate:

```text
4096 offroad <= 6
terminal route mean >= 0.52
eval reaches goal or max route >= 0.95
eval safety all zero
mean action jitter L1 <= 0.10
```

