# Round 21 - Corrective Sign Gate Audit

Date: 2026-05-01

## Goal

Audit the Round 20 blocker before running more training:

- Keep the Round 20 checkpoint fixed.
- Split the old route/lane-offset corrective sign metric from the new RGB visual-center correction metric.
- Let `bc_eval` choose which corrective sign source gates PPO.
- Re-evaluate the Round 20 checkpoint with the visual-center sign source.

## Code Changes

Commit:

- `44ff2f1` - Split BC corrective sign gate sources

Main changes:

- `evaluate_bc.py` now logs both sign contracts:
  - `route_offset_corrective_sign`
  - `visual_center_corrective_sign`
- It also logs whether route offset and visual center agree:
  - `route_visual_center_sign_agreement`
- Summary now includes:
  - `corrective_sign_source`
  - `route_offset_corrective_sign_rate`
  - `visual_center_correction_rate`
  - `route_visual_center_sign_agreement_rate`
- PPO gate can select the corrective sign source:
  - `ppo_gate_corrective_sign_source: route_offset`
  - `ppo_gate_corrective_sign_source: visual_center`
  - `ppo_gate_corrective_sign_source: both`

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_gate_audit_round21.yaml
```

Local verification:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_evaluate_bc_gate.py tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py -q
13 passed

python -m py_compile rcs_ac_first/scripts/evaluate_bc.py tests/test_evaluate_bc_gate.py
PASS
```

Windows verification:

```text
py -3.12 -B -m py_compile rcs_ac_first/scripts/evaluate_bc.py tests/test_evaluate_bc_gate.py
PASS
```

## Sign Audit

Subagent/code audit conclusion:

- `target_corrective_delta = corrected_steer - policy_steer` is internally consistent.
- The old route-offset convention is also internally consistent if positive `lane_offset` means the vehicle is right of route center.
- `heading_error` only activates the risk mask by absolute value; it does not provide a correction sign.
- `visual_center_error = lane_x - road_x` is internally consistent for the RGB-center objective.
- The conflict is gate semantics: Round 20 required both `signed_steer_delta * (-lane_offset) > 0` and `signed_steer_delta * visual_center_error < 0`, even though the late failure window can make those disagree.

Round 20 local log audit:

```text
corrective rows: 124
visual_center_corrective_sign: 124 / 124
route_offset_corrective_sign: 0 / 124
lane_offset mean: 1.1882
visual_center_error mean: -1.7468
signed visual steer delta mean: 0.00612
```

## CARLA Eval

Checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_visual_center_sign_bc_round20/last.pt
```

Run:

```text
rgb_perception_actor_features_v2_gate_audit_eval_512_round21_20260501T204250
```

Note:

- The first eval attempt hit a CARLA streaming/load-world timeout.
- `ensure_carla_ready.ps1 -ForceRestart` recovered CARLA and the second eval completed.

Summary:

```text
completed_steps: 512
pass_status: PASS
ppo_gate_status: BLOCK
corrective_sign_source: visual_center
max_route_progress: 0.256410
mean_route_progress: 0.118389
offroad_count: 2
early_offroad_count: 0
corrective_need_count: 131
route_offset_corrective_sign_rate: 0.000000
visual_center_correction_rate: 1.000000
route_visual_center_sign_agreement_rate: 0.000000
mean_visual_zero_action_l1: 0.011208
mean_visual_zero_steer_delta: 0.005328
mean_corrective_visual_zero_steer_delta: 0.005922
mean_action_jitter_l1: 0.052413
```

PPO gate fail reasons:

```text
max_route_progress 0.256410 < 0.282051
mean_route_progress 0.118389 < 0.132762
mean_visual_zero_action_l1 0.011208 < 0.012000
```

Compared with Round 20:

- The old `corrective_sign_rate` blocker is gone.
- Offroad count remains `2`.
- Early offroad remains `0`.
- Max route remains `0.256410`.
- Mean route improves slightly from `0.115435` to `0.118389`.
- `mean_visual_zero_action_l1` drops slightly from `0.012060` to `0.011208`, just under the gate.

## Decision

Do not enter PPO yet.

Round 21 confirms that the old corrective sign gate was stale for the RGB-center correction objective. The true remaining blockers are:

- Route progress is still too low.
- RGB visual ablation is active in steering, but total action influence is barely below the configured threshold.

Next round should not re-fight the sign gate. It should target route progress directly:

- Keep visual-center sign source as the PPO gate source.
- Add route-aware lateral correction so RGB center correction does not fight route intent in the late window.
- Consider a small longitudinal/route progress adjustment, because the remaining route gap is now larger than the sign gap.
