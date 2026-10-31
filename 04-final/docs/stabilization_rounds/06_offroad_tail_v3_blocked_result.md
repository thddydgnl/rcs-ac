# Stabilization Round 06: Offroad Tail v3

## Summary

Trigger:

```text
다음 라운드 서브 에이전트로 안정화 진행해줘
```

Goal:

```text
Preserve recovery_imitation_v2_lateral_target goal-reaching eval behavior while reducing training offroad.
```

Final judgment:

```text
FAIL-AS-STABILIZATION / PASS-AS-DIAGNOSTIC
```

Reason:

```text
The new offroad-tail imitation data preserved deterministic eval goal behavior after PPO,
but 2048 training offroad increased and route mean collapsed.
The 2048 gate failed, so 4096 formal smoke was intentionally skipped.
```

Best checkpoint remains:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

Do not promote:

```text
checkpoints/recovery_imitation_v3_offroad_tail_2048_20260430T011615/last.pt
```

## Subagent Findings

### Log Analyst

Key finding:

```text
v2 deterministic eval succeeds because lane_offset never enters the high-risk band.
v2 training offroad happens when lane_offset >= 1.8m, heading_error >= 10deg,
and speed remains around or above 16 km/h while lateral recovery is active.
```

Important v2 4096 diagnostics:

```text
offroad episodes: 7
offroad pre20 abs(lane_offset) mean: 2.027
offroad pre20 abs(heading_error) mean: 14.790
offroad pre20 speed mean: about 17.386 km/h
lateral_recovery_active: 140/140
lateral_recovery_branch_active: 140/140
lateral_recovery_speed_gate_active: 0/140
```

Primary precondition metric:

```text
rate(abs(lane_offset) >= 1.8 and abs(heading_error) >= 10deg and speed >= 16km/h)
```

### Code Auditor

Important findings:

```text
1. The core train/eval env config is connected correctly.
2. PPO init path works when --init-checkpoint is passed.
3. BC eval must use the phase3 env config, not the minimal BC config.
4. Training/eval CSVs still omit some lateral recovery fields, but env_step_log has enough detail.
5. Speed-gate target longitudinal is not trained by collect_recovery_dataset.py, but this round did not use speed-gate targets.
```

### Mechanism Verifier

Key finding:

```text
The v2 lateral target mechanism was real:
target_corrected_steer - raw_steer matched lateral_recovery_correction,
and the correction direction opposed lane offset in about 98.1% of relevant samples.
```

Runtime finding:

```text
Lateral recovery is active before offroad and usually points in the correct direction.
The remaining issue is not "mechanism did not activate"; it is that drift is already too large/fast.
```

### Safety Regression Analyst

Gate requirement:

```text
2048 gate:
  eval safety = 0
  eval goal or max_route >= 1.0
  training offroad <= 6
  training route_mean >= 0.5227
  training route_max >= 0.9231
```

### Ablation / Data Planner

Recommended experiment:

```text
recovery_imitation_v3_offroad_tail_1024_gentle
```

Hypothesis:

```text
The v2 policy lacks recovery coverage for both the old 0.24~0.38 route window and the new 0.44~0.50 offroad tail.
```

## Code / Config Changes

### New utility

File:

```text
rcs_ac_first/scripts/combine_expert_datasets.py
```

Purpose:

```text
Combine compatible .npz expert/recovery datasets into one training file.
Episode ids are reindexed to avoid collisions.
```

### New configs

Project configs:

```text
configs/bc_frame_recovery_imitation_v3_offroad_tail_1024_gentle_smoke.yaml
configs/phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml
```

Package copies:

```text
rcs_ac_first/configs/bc_frame_recovery_imitation_v3_offroad_tail_1024_gentle_smoke.yaml
rcs_ac_first/configs/phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml
```

BC settings:

```text
mode: frame
epochs: 1
learning_rate: 1e-6
steer_loss_weight: 2.0
longitudinal_loss_weight: 1.0
```

## Validation

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Result: `PASS`

Config load:

```text
bc_frame_recovery_imitation_v3_offroad_tail_1024_gentle
phase3_bc_clean_recovery_imitation_v3_offroad_tail
```

Result: `PASS`

CARLA preflight:

```text
CARLA 0.9.16
```

## Dataset Collection

### Primary Window

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_recovery_dataset `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v3_offroad_tail_primary_768.npz `
  --num-transitions 768 `
  --max-steps 12000 `
  --run-name recovery_imitation_v3_offroad_tail_collect_primary_768 `
  --route-progress-min 0.24 `
  --route-progress-max 0.38 `
  --lane-offset-threshold 0.45 `
  --heading-error-threshold 6.0 `
  --accept-lateral-branch-only `
  --device auto
```

Result:

```text
accepted: 768 / 5622
acceptance_rate: 0.1366
route_progress min/mean/max: 0.2564 / 0.2997 / 0.3590
bad_transition_count: 0
done_count: 0
```

### Late Tail Window

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_recovery_dataset `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v3_offroad_tail_late_256.npz `
  --num-transitions 256 `
  --max-steps 12000 `
  --run-name recovery_imitation_v3_offroad_tail_collect_late_256 `
  --route-progress-min 0.44 `
  --route-progress-max 0.50 `
  --lane-offset-threshold 0.45 `
  --heading-error-threshold 6.0 `
  --accept-lateral-branch-only `
  --device auto
```

Result:

```text
accepted: 256 / 5080
acceptance_rate: 0.0504
route_progress min/mean/max: 0.4615 / 0.4750 / 0.4872
bad_transition_count: 0
done_count: 0
```

### Combined Dataset

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.combine_expert_datasets `
  --input C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v3_offroad_tail_primary_768.npz `
  --input C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v3_offroad_tail_late_256.npz `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v3_offroad_tail_1024.npz
```

Result:

```text
num_transitions: 1024
route_progress min/mean/max: 0.2564 / 0.3435 / 0.4872
input_counts: [768, 256]
```

## BC Training

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_frame_recovery_imitation_v3_offroad_tail_1024_gentle_smoke.yaml `
  --dataset C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\recovery_imitation_v3_offroad_tail_1024.npz `
  --run-name recovery_imitation_v3_offroad_tail_gentle_bc_1024 `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --init-strict true `
  --device auto
```

Result:

```text
train_loss: 0.022614
val_loss: 0.020749
```

Checkpoint:

```text
checkpoints/recovery_imitation_v3_offroad_tail_gentle_bc_1024/last.pt
```

## BC Deterministic Eval

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v3_offroad_tail_gentle_bc_1024\last.pt `
  --run-name recovery_imitation_v3_offroad_tail_gentle_bc_eval_512 `
  --num-steps 512 `
  --device auto
```

Result:

```text
done_reason: max_steps
final/max route_progress: 0.9744
collision/offroad/stuck: 0/0/0
mean_action_jitter_l1: 0.0908
```

Judgment:

```text
Safe enough to try 2048 PPO gate.
```

## PPO 2048 Gate

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v3_offroad_tail_gentle_bc_1024\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name recovery_imitation_v3_offroad_tail_2048 `
  --device auto
```

Result:

```text
updates: 16
max_steps episodes: 2
offroad episodes: 7
collision/stuck: 0/0
terminal route mean: 0.3333
terminal route max: 0.9231
```

Comparison:

| Metric | v2 2048 | v2 4096 | v3 2048 |
|---|---:|---:|---:|
| max_steps episodes | 3 | 5 | 2 |
| offroad episodes | 2 | 7 | 7 |
| terminal route mean | 0.5513 | 0.5227 | 0.3333 |
| terminal route max | 0.9231 | 0.9231 | 0.9231 |
| risk row rate | 0.0136 | 0.0175 | 0.0262 |
| offroad tail abs lane mean | 2.0660 | 2.0741 | 2.0815 |
| offroad tail speed mean | 55.55 km/h | 62.39 km/h | 63.43 km/h |

2048 gate judgment:

```text
FAIL.
training offroad did not decrease
training route mean collapsed
risk row rate increased
```

The planned 4096 formal smoke was skipped.

## 2048 Checkpoint Eval

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v3_offroad_tail_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v3_offroad_tail_2048_20260430T011615\last.pt `
  --run-name recovery_imitation_v3_offroad_tail_2048_eval_512 `
  --num-steps 512 `
  --device auto
```

Result:

```text
goal reached: 1
max_route_progress: 1.0
collision/offroad/stuck: 0/0/0
mean_action_jitter_l1: 0.1039
```

Interpretation:

```text
Eval goal behavior survived, but action jitter exceeded the <=0.10 preferred gate
and training distribution regressed badly.
```

## Regression-First Summary

Regressed:

```text
1. 2048 training offroad rose to 7.
2. 2048 terminal route mean fell to 0.3333.
3. max_steps episodes fell to 2.
4. risk row rate increased versus v2.
5. deterministic eval jitter increased to 0.1039.
```

Improved or preserved:

```text
1. deterministic eval still reached goal after 2048 PPO.
2. collision and stuck stayed zero.
3. combined dataset successfully covered both primary and late-tail windows.
```

Unchanged:

```text
1. terminal route max remained 0.9231 in 2048 training.
2. offroad tail lane magnitude remained about 2.08m.
```

## Mechanism Activation

The imitation mechanism activated:

```text
new dataset collected from current v2 policy
primary + late tail windows were combined
gentle BC completed with finite loss
BC deterministic eval had safety all zero
```

But the mechanism did not produce the desired training effect:

```text
more offroad-tail imitation did not reduce the high-lane/high-heading/high-speed precondition.
It likely overfit or shifted the policy enough to introduce early offroad at route_progress 0.103.
```

## Final Judgment

Status:

```text
FAIL-AS-STABILIZATION / PASS-AS-DIAGNOSTIC
```

Do not promote:

```text
checkpoints/recovery_imitation_v3_offroad_tail_2048_20260430T011615/last.pt
```

Keep best:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

## Next Recommendation

Do not continue increasing recovery imitation data in the same form.

The most useful next single hypothesis is:

```text
The high-risk offroad precondition is not solved by more corrected-steer BC;
it needs a narrow runtime guard for branch/lateral recovery states where:
  abs(lane_offset) >= 1.8
  abs(heading_error) >= 10deg
  speed >= 16km/h
```

Recommended next experiment:

```text
recovery_imitation_v2_narrow_preoffroad_speed_guard
```

Use current best as init:

```text
checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
```

Keep it narrow:

```text
No new BC dataset.
No reward rewrite.
No broad speed gate.
Only a late/pre-offroad guard active under the high-risk condition.
```

Gate:

```text
2048 must have:
  offroad <= 6
  terminal route mean >= 0.5227
  eval goal or max route 1.0
  eval safety all zero
  action jitter <= 0.10
```

