# Round 20 - Visual Center Sign Correction

Date: 2026-05-01

## Goal

Continue the stabilization round from the Round 19 failure mode:

- Keep `target_corrective_delta = corrected_steer - policy_steer` as the supervised corrective target.
- Add an explicit RGB visual-center sign objective so the RGB-dependent steering delta points back toward the detected road/lane center.
- Do not enter PPO unless the 512-step BC gate passes.

## Code Changes

Commit:

- `7bf38cc` - Add visual center corrective sign loss

Main changes:

- `train_bc.py` adds `visual_center_corrective_sign_loss`.
- The loss uses RGB perception raw features:
  - `road_x = raw_features[..., 10]`
  - `lane_x = raw_features[..., 11]`
  - `visual_center_error = lane_x - road_x`
- The loss is active only when:
  - `target_corrective_delta` is available,
  - the sample is valid,
  - the corrective target has non-trivial magnitude,
  - the visual center error has non-trivial magnitude,
  - `-visual_center_error` and `target_corrective_delta` agree in sign.
- `evaluate_bc.py` now logs:
  - `visual_center_corrective_sign`
  - `visual_center_correction_score`
  - `visual_center_correction_rate`
- Config:
  - `configs/bc_frame_rgb_perception_actor_features_v2_visual_center_sign_round20.yaml`

Local verification:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py -q
11 passed

python -m py_compile rcs_ac_first/scripts/train_bc.py rcs_ac_first/scripts/evaluate_bc.py tests/test_train_bc_perception.py
PASS
```

Windows syntax verification:

```text
py -3.12 -B -m py_compile rcs_ac_first/scripts/train_bc.py rcs_ac_first/scripts/evaluate_bc.py tests/test_train_bc_perception.py
PASS
```

## BC Result

Run:

```text
rgb_perception_actor_features_v2_visual_center_sign_bc_round20
```

Checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_visual_center_sign_bc_round20/last.pt
```

Final validation:

```text
bc_loss: 0.006972
steer_mse: 0.004483
visual_zero_action_l1: 0.025577
visual_zero_steer_delta: 0.005557
target_corrective_delta_l1: 0.048448
target_corrective_delta_sign_rate: 0.908214
visual_zero_target_delta_l1: 0.092934
visual_zero_target_delta_sign_rate: 0.923855
visual_center_corrective_sign_rate: 1.000000
visual_center_correction_rate: 1.000000
visual_center_corrective_mask_rate: 0.537634
visual_center_signed_delta_mean: 0.006245
fusion_rgb: 0.004995
```

Compared with Round 19:

- `bc_loss` improved from `0.008050` to `0.006972`.
- `visual_zero_target_delta_sign_rate` improved from `0.091786` to `0.923855`.
- The RGB visual-zero delta now matches the corrective target direction offline.

## CARLA Eval

Run:

```text
rgb_perception_actor_features_v2_visual_center_sign_eval_512_round20_20260501T202939
```

Summary:

```text
completed_steps: 512
pass_status: PASS
ppo_gate_status: BLOCK
max_route_progress: 0.256410
mean_route_progress: 0.115435
offroad_count: 2
early_offroad_count: 0
corrective_need_count: 124
corrective_sign_rate: 0.000000
visual_center_correction_rate: 1.000000
mean_visual_center_correction_score: 0.010508
mean_visual_zero_steer_delta: 0.005339
mean_corrective_visual_zero_steer_delta: 0.006124
mean_action_jitter_l1: 0.064797
```

PPO gate fail reasons:

```text
max_route_progress 0.256410 < 0.282051
mean_route_progress 0.115435 < 0.132762
corrective_sign_rate 0.000000 < 0.700000
```

Compared with Round 19:

- Offroad count improved from `4` to `2`.
- Early offroad count improved from `3` to `0`.
- Mean route progress improved from `0.070262` to `0.115435`.
- Max route progress stayed at `0.256410`.
- PPO remains blocked.

## Interpretation

Round 20 successfully fixed the RGB visual-center correction direction:

```text
visual_center_correction_rate: 1.000000
visual_zero_target_delta_sign_rate: 0.923855
```

However, the old `lane_offset`-based corrective sign metric still reports `0.0`. The eval rows show the reason: the model is now producing a visual-center-consistent positive visual delta, while the route/lane-offset sign check can disagree with the visual center error in the late failure window.

Failure-window audit:

```text
corrective_need rows: 124
lane_offset_mean: 1.1882
heading_error_mean: -15.3324
steer_mean: 0.0523
visual_zero_signed_steer_delta_mean: 0.00612
visual_center_correction_score_mean: 0.01051
visual_center_corrective_sign_count: 124 / 124
```

Offroad rows:

```text
step 233, route 0.230769, lane_offset 2.1362, heading_error -13.1049, steer -0.0852, visual_center_corrective_sign True
step 472, route 0.256410, lane_offset 2.0822, heading_error -10.5550, steer -0.0871, visual_center_corrective_sign True
```

## Decision

Do not enter PPO.

The RGB center correction objective is working, but route progress is still below the PPO gate and the legacy corrective sign gate is no longer aligned with the new RGB-center objective. The next round should focus on the route/RGB sign contract rather than simply increasing RGB weight:

- Audit whether `lane_offset`, `visual_center_error`, and steer sign conventions are all expressed in the same coordinate frame.
- Replace or split the PPO gate corrective sign metric so visual-center correction is not judged by an incompatible route-offset sign.
- Add a small route-aware correction target: route decides intended path direction, RGB center error decides lateral recovery inside the lane.
