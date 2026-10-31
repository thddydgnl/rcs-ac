# Rounds 42-48 Probe Absolute Bias and CARLA Result

Date: 2026-05-02

Status: PARTIAL PROGRESS, BLOCKED BEFORE PPO

## Purpose

Rounds 39-41 made scalar lane/heading/route probes influence the raw actor, but `early_left` zero-steer bias still blocked the offline actor sign gate.

These rounds add offline-only corrective losses to reduce that bias without adding runtime steering rules, lane-offset overrides, or privileged inference inputs.

The privileged simulator fields used here (`lane_offset`, `heading_error`, `route_progress`, `target_corrective_delta`) are used only for BC loss masks and counterfactual training probes.

## Code Changes

`train_bc.py` now logs and optimizes optional early-severe absolute steer losses:

```text
early_severe_absolute_steer_loss
early_severe_absolute_steer_sign_rate
early_severe_probe_absolute_steer_loss
early_severe_probe_absolute_steer_sign_rate
early_severe_probe_absolute_steer_mask_rate
```

`early_severe_absolute_steer_loss` penalizes raw actor steer that points opposite the semantic recovery target on early severe rows.

`early_severe_probe_absolute_steer_loss` performs offline counterfactual probes by zeroing configured scalar axes (`lane_offset`, `heading_error`, `route_target_y`) and requiring the raw actor to keep the recovery-side steer sign. This directly targets the actor sign audit probe. It is a training-time objective only.

## Local Validation

```bash
python -m py_compile rcs_ac_first/scripts/train_bc.py
python -m pytest tests/test_train_bc_perception.py tests/test_audit_actor_sign_response.py tests/test_rgb_perception_adapter.py -q
```

Result:

```text
35 passed in 1.18s
```

## Round 42

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_absolute_bias_probe_round42.yaml
```

Change:

```text
early_severe_absolute_steer_loss_weight: 0.8
trainable_module_names: [scalar_actor_adapter]
init: Round41 checkpoint
```

BC:

```text
val_loss: 0.006654
```

Actor sign audit:

```text
early_left: BLOCK
  all_axes_rate: 1.0
  min_abs_delta: 0.012166
  left_bias_at_zero: -0.054651

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.008869
  left_bias_at_zero: 0.013239
```

## Round 43

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_absolute_bias_probe_round43.yaml
```

Change:

```text
early_severe_absolute_steer_loss_weight: 5.0
epochs: 5
trainable_module_names: [scalar_actor_adapter]
init: Round41 checkpoint
```

BC:

```text
val_loss: 0.007384
```

Actor sign audit:

```text
early_left: BLOCK
  all_axes_rate: 1.0
  min_abs_delta: 0.011136
  left_bias_at_zero: -0.053432

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.007884
  left_bias_at_zero: 0.013239
```

The lane and route zero probes improved, but `heading_error=0` remained the blocking axis.

## Round 44

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_heading_probe_round44.yaml
```

Change:

```text
early_severe_probe_absolute_steer_loss_weight: 3.0
early_severe_probe_axes: ["heading_error"]
trainable_module_names: [scalar_actor_adapter]
init: Round41 checkpoint
```

BC:

```text
val_loss: 0.007821
```

Actor sign audit:

```text
early_left: BLOCK
  all_axes_rate: 1.0
  min_abs_delta: 0.019140
  left_bias_at_zero: -0.047642

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.013408
  left_bias_at_zero: 0.013239
```

## Rounds 45-47

Configs:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_all_probe_round45.yaml
configs/bc_frame_rgb_perception_actor_features_v2_scalar_all_probe_continue_round46.yaml
configs/bc_frame_rgb_perception_actor_features_v2_scalar_all_probe_continue_round47.yaml
```

Change:

```text
early_severe_probe_absolute_steer_loss_weight: 10.0
early_severe_probe_axes: ["lane_offset", "heading_error", "route_target_y"]
trainable_module_names: [scalar_actor_adapter]
Round45 init: Round44 checkpoint
Round46 init: Round45 checkpoint
Round47 init: Round46 checkpoint
```

Actor sign audit progression:

```text
Round45 early_left left_bias_at_zero: -0.037780 BLOCK
Round46 early_left left_bias_at_zero: -0.035025 BLOCK
Round47 early_left left_bias_at_zero: -0.034288 BLOCK
```

Scalar adapter only was directionally correct but not strong enough to clear the `-0.03` gate.

## Round 48

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_scalar_all_probe_actor_unfreeze_round48.yaml
```

Change:

```text
trainable_module_names:
  - scalar_actor_adapter
  - recurrent_actor_critic
init: Round47 checkpoint
epochs: 1
```

BC:

```text
val_loss: 0.009531
selected_params: 429189
total_params: 955328
```

Actor sign audit:

```text
early_left: PASS
  all_axes_rate: 1.0
  monotonic_rate: 1.0
  min_abs_delta: 0.017438
  left_bias_at_zero: -0.025071

early_any: PASS
  all_axes_rate: 1.0
  monotonic_rate: 1.0
  min_abs_delta: 0.011120
  left_bias_at_zero: 0.085784
```

This is the first checkpoint in this sequence to satisfy the offline actor sign gate for `early_left`.

## CARLA Eval

Checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_scalar_all_probe_actor_unfreeze_bc_round48/last.pt
```

256-step eval:

```text
run: rgb_perception_actor_features_v2_scalar_all_probe_actor_unfreeze_eval_256_round48_retry
completed_steps: 256
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
final_route_progress: 0.461538
pass_status: PASS
ppo_gate_status: BLOCK
ppo_gate_fail_reasons:
  - mean_action_jitter_l1 0.101465 > 0.070000
```

512-step eval:

```text
run: rgb_perception_actor_features_v2_scalar_all_probe_actor_unfreeze_eval_512_round48_retry
completed_steps: 512
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
pass_status: PASS
ppo_gate_status: BLOCK
ppo_gate_fail_reasons:
  - mean_action_jitter_l1 0.114223 > 0.070000
  - visual_center_correction_rate 0.698113 < 0.700000
```

2048-step eval:

```text
run: rgb_perception_actor_features_v2_scalar_all_probe_actor_unfreeze_eval_2048_round48
completed_steps: 2048
collision_count: 0
offroad_count: 10
early_offroad_count: 1
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
pass_status: PASS
ppo_gate_status: BLOCK
ppo_gate_fail_reasons:
  - offroad_count 10 > 2
  - early_offroad_count 1 > 0
  - mean_action_jitter_l1 0.094641 > 0.070000
  - visual_center_correction_rate 0.331169 < 0.700000
```

Per the stabilization goal, the 2048 result is a safety failure even though `evaluate_bc.py` reports `pass_status: PASS`. The objective requires collision/offroad/severe early offroad/stuck all zero at 2048, and this checkpoint has 10 offroad episodes plus 1 early offroad.

## Decision

Do not start PPO.

Round48 proves that the offline actor sign gate and 512-step clean safety gate can be passed, but long-horizon 2048 safety is not stable yet. The next round should keep the Round48 offline sign improvement and reduce:

```text
2048 offroad_count
early_offroad_count
mean_action_jitter_l1
low visual_center_correction_rate
```

without adding runtime steering overrides or presenting guard-assisted behavior as raw RCS-AC policy performance.
