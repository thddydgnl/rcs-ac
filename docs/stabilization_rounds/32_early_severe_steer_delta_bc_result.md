# Round 32 Early Severe Steer Delta BC Result

Date: 2026-05-02

## Goal

Fix the Round31 blocker by adding a BC loss that directly supervises raw actor steering against `policy_action_steer + target_corrective_delta` on early severe recovery rows.

This round deliberately did not change runtime control, reward, PPO, model architecture, CARLA environment semantics, or speed/brake loss.

## Baseline

Round31 blocker:

```text
checkpoint: checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt
early-left actor_sign_response_all_axes_rate: 0.0
early-left actor_sign_response_min_abs_delta: 0.0001499
early-left zero steer bias: -0.070969
```

Round30 simulator blocker:

```text
maneuver-off 256 eval
early_offroad_count: 3
max_route_progress: 0.076923
initial_audit.left_steer_rate: 0.871094
initial_audit.mean_steer: -0.095287
```

## Code Changes

Added early severe steer correction metrics/loss to:

```text
rcs_ac_first/scripts/train_bc.py
```

New config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_early_severe_delta_round32.yaml
```

New BC config fields:

```yaml
early_severe_steer_delta_loss_weight: 1.0
early_severe_steer_sign_loss_weight: 0.5
early_severe_route_min: 0.025
early_severe_route_max: 0.10
early_severe_abs_lane_offset_min: 0.55
early_severe_abs_heading_error_min: 3.0
early_severe_steer_margin: 0.04
early_severe_side_balanced: true
```

The old one-sided `early_route_offset_agreement_loss_weight` was disabled in this config to isolate the new raw steer target-delta mechanism:

```yaml
early_route_offset_agreement_loss_weight: 0.0
```

## Local Validation

```text
python -m py_compile rcs_ac_first/scripts/train_bc.py
python -m pytest tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py tests/test_evaluate_bc_gate.py -q
```

Result:

```text
31 passed
```

## Remote Execution Notes

Windows `git pull --ff-only` still fails in non-interactive SSH due Git Credential Manager:

```text
fatal: Unable to persist credentials with the 'wincredman' credential store
```

Files were copied to Windows with SCP and the experiment was run without CARLA.

Training command used:

```text
py -3.12 -B -m rcs_ac_first.scripts.train_bc
  --config configs/bc_frame_rgb_perception_actor_features_v2_early_severe_delta_round32.yaml
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
  --init-checkpoint checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt
  --init-strict true
  --run-name rgb_perception_actor_features_v2_early_severe_delta_bc_round32
  --device cuda
```

Output:

```text
log: C:\Users\thddy\Documents\code\RCS-AC-first\logs\rgb_perception_actor_features_v2_early_severe_delta_bc_round32\bc_training_log.csv
checkpoint: C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\rgb_perception_actor_features_v2_early_severe_delta_bc_round32\last.pt
```

## BC Result

Dataset:

```text
num_transitions: 4256
route_progress_min: 0.025641
route_progress_max: 0.487179
route_progress_mean: 0.253892
```

Training:

```text
epoch 1 val_loss: 0.011186
epoch 2 val_loss: 0.010995
epoch 3 val_loss: 0.010969
epoch 4 val_loss: 0.011287
epoch 5 val_loss: 0.010517
```

Final validation mechanism metrics:

```text
early_severe_steer_mask_rate: 0.065728
early_severe_steer_delta_l1: 0.004342
early_severe_steer_sign_loss: 0.000619
early_severe_steer_sign_rate: 0.095015
early_severe_steer_left_sign_rate: 0.090245
early_severe_steer_right_sign_rate: 0.098592
early_severe_steer_left_mask_rate: 0.023474
early_severe_steer_right_mask_rate: 0.042254
target_corrective_delta_mask_rate: 0.896714
```

The new loss activated, and severe-mask coverage exists on both sides, but validation sign rates are far below the required gate.

## Offline Actor Sign Audit

### Early Left

Run:

```text
round32_early_severe_delta_early_left
```

Result:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.00023355
actor_sign_response_left_bias_at_zero: -0.052492
```

Comparison to Round31:

```text
zero steer bias improved: -0.070969 -> -0.052492
min_abs_delta improved: 0.0001499 -> 0.0002336
all_axes_rate unchanged: 0.0 -> 0.0
```

### Early Any

Run:

```text
round32_early_severe_delta_early_any
```

Result:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.00000025
actor_sign_response_left_bias_at_zero: -0.042832
```

Comparison to Round31:

```text
zero steer bias improved: -0.067882 -> -0.042832
all_axes_rate unchanged: 0.0 -> 0.0
```

## Decision

Status: `BLOCK`

Do not run CARLA 512.

Reason:

- The new loss reduced the left prior somewhat, but did not create usable signed actor response.
- Offline actor audit still fails all axes.
- Validation early severe sign rate is only `0.095`, with both side splits around `0.09-0.10`.
- The pre-CARLA gate from Round31/Agent C was not met.

## Interpretation

This confirms that simply adding the loss on the current trainable scope is not enough.

The likely issue is one or more of:

1. `trainable_module_names` only updates:
   - `perception_aux_head`
   - `rgb_perception_actor_features`
   - `actor_input_adapter`
   while the recurrent actor head remains frozen.
2. The current dataset still has weak or inconsistent target-delta signal in the early severe validation split.
3. The sign response audit perturbs scalar inputs directly, but the trainable path may not route those scalar changes strongly into raw actor steering.

## Next Recommendation

Next round should not go to PPO or CARLA yet.

Use one of these narrow next probes:

1. Train with `recurrent_actor_critic` included in `trainable_module_names` for a very short BC run, then repeat offline actor sign audit.
2. Run a dataset-side target audit on Windows to confirm early severe target signs and magnitudes by side before changing trainable scope.
3. If target signs are correct, use a stronger short BC run with:
   - `early_severe_steer_delta_loss_weight: 2.0`
   - `early_severe_steer_sign_loss_weight: 1.0`
   - actor head unfrozen

Required next gate remains:

```text
early-left actor_sign_response_all_axes_rate >= 0.90
early-left actor_sign_response_min_abs_delta >= 0.005
abs(early-left zero steer bias) <= 0.03
early severe validation side sign rates >= 0.85
```
