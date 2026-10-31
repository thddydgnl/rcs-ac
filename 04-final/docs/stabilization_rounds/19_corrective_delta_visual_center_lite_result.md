# Round 19 - Corrective Delta Visual Center Lite

Date: 2026-05-01

## Goal

Run the lite version of the next stabilization round:

- Train against `target_corrective_delta = corrected_steer - policy_steer`.
- Expose RGB perception center diagnostics (`visual_road_x`, `visual_lane_x`, `visual_lane_center_error`).
- Keep the experiment narrow before trying RGB fusion weight boosts or PPO.

## Code Changes

Commits:

- `0fa80f8` - Add corrective delta BC round 19
- `013f667` - Derive corrective delta from recovery datasets
- `57073d6` - Preserve risk fields in mixed expert datasets

Main changes:

- `collect_recovery_dataset.py` now stores `target_corrective_delta` and availability.
- `ExpertDataset` derives `target_corrective_delta` from older recovery datasets when `target_corrected_steer` and `policy_action_steer` exist.
- `train_bc.py` adds corrective-delta loss/metrics:
  - `target_corrective_delta_loss`
  - `target_corrective_delta_l1`
  - `target_corrective_delta_sign_rate`
  - `visual_zero_target_delta_loss`
  - `visual_zero_target_delta_sign_rate`
- `evaluate_bc.py` logs `visual_zero_signed_steer_delta` and uses the configured risk threshold for corrective sign checks.
- `rcs_ac_model.py` logs visual center diagnostics.

Local verification:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py -q
8 passed

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_corrective_delta_round19.yaml --device cpu
PASS
```

Windows syntax verification:

```text
py -3.12 -B -m py_compile rcs_ac_first/data/expert_dataset.py rcs_ac_first/scripts/train_bc.py rcs_ac_first/scripts/evaluate_bc.py
PASS
```

## Dataset

Output:

```text
data/expert/rgb_perception_actor_features_v2_corrective_delta_mix_928_round19.npz
```

Mix:

- clean: 384
- drift: 192
- bridge: 256
- recovery_tail: 96

Audit:

```text
num_transitions: 928
target_corrective_delta_available_rate: 0.5862
target_corrective_delta_available_count: 544
target_corrective_delta_abs_mean: 0.09684
target_corrective_delta_sign_rate_vs_-lane_offset: 0.9743
available_lane_offset_mean: -1.1201
```

## BC Result

Run:

```text
rgb_perception_actor_features_v2_corrective_delta_bc_round19
```

Checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_corrective_delta_bc_round19/last.pt
```

Final validation:

```text
bc_loss: 0.008050
steer_mse: 0.005080
action_l1: 0.073562
visual_zero_action_l1: 0.028159
visual_zero_steer_delta: 0.004252
target_corrective_delta_l1: 0.056074
target_corrective_delta_sign_rate: 0.908214
visual_zero_target_delta_l1: 0.102266
visual_zero_target_delta_sign_rate: 0.091786
target_corrective_delta_mask_rate: 0.580645
fusion_rgb: 0.004995
fusion_ego: 0.799551
fusion_route: 0.172027
visual_road_x_mean: 0.011808
visual_lane_x_mean: -0.468607
visual_lane_center_error_mean: -0.480415
visual_actor_delta_l1_mean: 0.126640
```

Interpretation:

- The full action output learned the corrective-delta target direction (`0.908` sign rate).
- The visual-zero delta still points the wrong way (`0.092` sign rate), so the RGB-specific pathway is not yet providing a reliable corrective steering effect.

## CARLA Eval

Run:

```text
rgb_perception_actor_features_v2_corrective_delta_eval_512_round19_20260501T195158
```

Summary:

```text
completed_steps: 512
pass_status: PASS
ppo_gate_status: BLOCK
max_route_progress: 0.256410
mean_route_progress: 0.070262
offroad_count: 4
early_offroad_count: 3
corrective_need_count: 28
corrective_sign_rate: 0.0
mean_visual_zero_action_l1: 0.018878
mean_visual_zero_steer_delta: 0.004414
mean_corrective_visual_zero_steer_delta: 0.004942
```

Compared with Round 18:

- Max route improved from `0.205128` to `0.256410`.
- PPO remains blocked.
- Early offroad and corrective sign failure remain.

Failure bucket audit:

```text
route 0.2052-0.2565:
  n: 15
  lane_offset_mean: -2.286
  heading_error_mean: 12.436
  steer_mean: -0.0675
  visual_zero_signed_steer_delta_mean: -0.00550
  visual_corrective_sign_count: 0
  visual_road_x_mean: 0.003
  visual_lane_x_mean: -0.856
  visual_lane_center_error_mean: -0.859
```

Offroad rows all show the same pattern:

- `lane_offset` is strongly negative, meaning the vehicle is left of route center.
- `steer` remains negative.
- `visual_zero_signed_steer_delta` remains negative.
- `visual_lane_center_error` is strongly negative.

## Decision

Do not enter PPO.

Round 19 confirmed that corrective-delta supervision helps the total action somewhat, but the RGB perception pathway is still mapped in the wrong steering direction. The next round should explicitly map visual center error to steering sign, for example:

- Add a visual-center sign loss: `visual_zero_signed_steer_delta * (-visual_lane_center_error) > 0`.
- Or add a small direct visual correction head initialized so negative visual center error produces positive steer correction.
- Keep route as the progress/intent source, but make RGB center error directly responsible for lateral correction.
