# Rounds 39-41 Scalar Actor Probe Result

Date: 2026-05-02

Status: PARTIAL PROGRESS, BLOCKED BEFORE CARLA

## Purpose

Rounds 37-38 showed that semantic dataset labels pass and that opening `recurrent_actor_critic` improves BC loss, but scalar lane/heading/route probes still barely affected raw actor steering. These rounds add a policy-internal scalar actor feature path and test whether it creates direct signed recovery response.

This is not a runtime steering override. It is a trainable feature path inside the policy.

## Code Changes

`RCSACFirstModel` now supports optional scalar actor features:

```text
use_scalar_actor_features
scalar_actor_adapter
scalar_actor_lane_offset_scale
scalar_actor_heading_error_scale
scalar_actor_route_y_scale
```

The scalar feature vector is:

```text
lane_offset / scalar_actor_lane_offset_scale
heading_error / scalar_actor_heading_error_scale
route_target_y / scalar_actor_route_y_scale
```

It is passed through a zero-initialized residual adapter into actor input:

```text
actor_input = actor_input + scalar_actor_adapter(scalar_features)
```

Default behavior is unchanged because `use_scalar_actor_features` defaults to `false`.

`audit_actor_sign_response.py` now supports:

```text
--sign-convention legacy
--sign-convention semantic
```

Semantic convention:

```text
lane_offset: increasing offset should reduce steer
heading_error: increasing heading error should increase steer
route_target_y: increasing target y should increase steer
```

This fixes the previous false block where heading/route were judged with the lane-offset direction.

## Local Validation

```bash
python -m py_compile rcs_ac_first/scripts/audit_actor_sign_response.py rcs_ac_first/models/rcs_ac_model.py rcs_ac_first/runtime/config.py
python -m pytest tests/test_audit_actor_sign_response.py tests/test_rgb_perception_adapter.py tests/test_train_bc_perception.py tests/test_audit_corrective_delta_dataset.py -q
```

Result:

```text
40 passed in 1.26s
```

## Round 39

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_actor_probe_round39.yaml
```

Trainable scope:

```text
perception_aux_head
rgb_perception_actor_features
actor_input_adapter
scalar_actor_adapter
recurrent_actor_critic
```

Init checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt
```

Init used `strict=false` because the scalar adapter is new.

BC result:

```text
val_loss: 0.005941
```

Semantic actor sign audit:

```text
early_left:
  pass_status: BLOCK
  actor_sign_response_all_axes_rate: 1.0
  actor_sign_response_monotonic_rate: 1.0
  actor_sign_response_min_abs_delta: 0.008956
  actor_sign_response_left_bias_at_zero: -0.085952

early_any:
  pass_status: BLOCK
  actor_sign_response_all_axes_rate: 1.0
  actor_sign_response_monotonic_rate: 1.0
  actor_sign_response_min_abs_delta: 0.007733
  actor_sign_response_left_bias_at_zero: -0.060240
```

Interpretation:

The scalar actor path solves signed sensitivity, but the zero-steer bias is too far left.

## Round 40

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_adapter_only_probe_round40.yaml
```

Trainable scope:

```text
scalar_actor_adapter
```

Init checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt
```

BC result:

```text
val_loss: 0.009136
early_severe_steer_delta_l1: 0.004712
```

Semantic actor sign audit:

```text
early_left:
  pass_status: BLOCK
  actor_sign_response_all_axes_rate: 1.0
  actor_sign_response_min_abs_delta: 0.005265
  actor_sign_response_left_bias_at_zero: -0.065524

early_any:
  pass_status: BLOCK
  actor_sign_response_all_axes_rate: 1.0
  actor_sign_response_min_abs_delta: 0.005019
  actor_sign_response_left_bias_at_zero: -0.067881
```

Interpretation:

Training only the scalar adapter creates just enough signed sensitivity, but it preserves too much of the original left bias.

## Round 41

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_on_actor_unfreeze_probe_round41.yaml
```

Trainable scope:

```text
scalar_actor_adapter
```

Init checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_semantic_actor_unfreeze_probe_bc_round38/last.pt
```

This uses Round38 as the base because Round38 improved the neutral early-any prior while still lacking scalar sensitivity.

BC result:

```text
val_loss: 0.006850
early_severe_steer_delta_l1: 0.004851
```

Semantic actor sign audit:

```text
early_left:
  pass_status: BLOCK
  actor_sign_response_all_axes_rate: 1.0
  actor_sign_response_monotonic_rate: 1.0
  actor_sign_response_min_abs_delta: 0.006398
  actor_sign_response_left_bias_at_zero: -0.058431

early_any:
  pass_status: PASS
  actor_sign_response_all_axes_rate: 1.0
  actor_sign_response_monotonic_rate: 1.0
  actor_sign_response_min_abs_delta: 0.005995
  actor_sign_response_left_bias_at_zero: 0.013238
```

## Decision

Status: `BLOCK`

Do not run CARLA 256/512 yet.

The sign-response blocker is mostly solved:

```text
all_axes_rate target: >= 0.90
Round41 early_left: 1.0
Round41 early_any: 1.0

min_abs_delta target: >= 0.005
Round41 early_left: 0.006398
Round41 early_any: 0.005995
```

The remaining blocker is early-left zero bias:

```text
target abs bias: <= 0.03
Round41 early_left bias: -0.058431
```

Next round should keep the scalar actor path and add a narrow early-left bias correction objective or sampling strategy. It should not run CARLA until early-left zero bias is within target.
