# Rounds 53-59 Strict Bias And Late Recovery Result

Date: 2026-05-02

Status: PARTIAL PROGRESS, BLOCKED BEFORE PPO

## Starting Point

Rounds 51-52 reduced the long-horizon failure from Round48, but 2048-step clean eval still had two non-early offroad terminations around route progress `0.307`.

The next issue was that the actor sign audit only blocked negative zero-steer bias. A model could pass with a large positive zero-steer bias, which violates the stabilization target:

```text
early-left zero steer bias abs <= 0.03
```

## Code Changes

`audit_actor_sign_response.py` now has a strict absolute zero-bias gate:

```text
--max-zero-bias: default 0.03
actor_sign_response_max_abs_zero_steer
pass_status requires max_abs_zero_steer <= max_zero_bias
```

`train_bc.py` now supports two offline-only losses:

```text
late_recovery_absolute_steer_loss
early_center_zero_steer_loss
```

`late_recovery_absolute_steer_loss` applies a route-windowed absolute recovery steering target. `early_center_zero_steer_loss` constrains very early, near-centered rows so the raw actor does not learn a start-of-episode steering bias.

Both losses use privileged simulator fields only as offline BC masks. They do not add runtime steering overrides, runtime speed guards, or inference-time privileged inputs.

## Local Validation

```bash
python -m py_compile rcs_ac_first/scripts/train_bc.py rcs_ac_first/scripts/audit_actor_sign_response.py
python -m pytest tests/test_audit_actor_sign_response.py tests/test_rgb_perception_adapter.py tests/test_train_bc_perception.py tests/test_audit_corrective_delta_dataset.py tests/test_expert_dataset_provenance.py tests/test_expert_dataset_sequence.py tests/test_evaluate_bc_gate.py -q
```

Result:

```text
53 passed
```

## Round 53

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_absolute_recovery_round53.yaml
```

Result:

```text
val_loss: 0.060510
early_left strict: BLOCK
zero_steer bias: about +0.21
```

Decision: do not run CARLA. The strong absolute recovery target over-biased the actor.

## Round 54

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_window_absolute_recovery_round54.yaml
```

Result:

```text
val_loss: 0.029454
early_left strict: BLOCK
early_left max_abs_zero_steer: 0.119524
```

Decision: do not run CARLA. Late-only recovery still leaked into zero-bias.

## Round 55

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_window_absolute_recovery_round55.yaml
```

Change:

```text
late_recovery_absolute_steer_loss_weight: 0.15
late_recovery_absolute_steer_margin: 0.30
init: Round52 checkpoint
```

Result:

```text
val_loss: 0.009407
early_left strict: PASS
  max_abs_zero_steer: 0.025733
  min_abs_delta: 0.014188
early_any strict: BLOCK
  max_abs_zero_steer: 0.044844
```

Decision: do not run CARLA. The early-left sample passed, but the centered early-any sample still had too much zero-steer bias.

## Round 56

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_window_absolute_recovery_round56.yaml
```

Change:

```text
early_center_zero_steer_loss_weight: 1.0
early_center_zero_steer_route_min: 0.025
early_center_zero_steer_route_max: 0.085
early_center_zero_steer_abs_lane_offset_max: 0.20
early_center_zero_steer_abs_heading_error_max: 2.0
```

Result:

```text
val_loss: 0.008229
early_left strict: PASS
  max_abs_zero_steer: 0.010389
early_any strict: BLOCK
  max_abs_zero_steer: 0.036173
```

Decision: do not run CARLA. The zero-bias correction was useful but slightly too strong in the negative direction.

## Round 57

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_window_absolute_recovery_round57.yaml
```

Change:

```text
early_center_zero_steer_loss_weight: 0.5
```

Offline:

```text
val_loss: 0.009065
early_left strict: PASS
  max_abs_zero_steer: 0.027693
  min_abs_delta: 0.012175
early_any strict: PASS
  max_abs_zero_steer: 0.000652
  min_abs_delta: 0.017422
```

512 eval:

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
ppo_gate_status: BLOCK
```

2048 eval:

```text
collision_count: 0
offroad_count: 2
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
ppo_gate_status: BLOCK
```

Decision: no PPO. This is the cleanest offline gate result, but long-horizon offroad remains.

## Round 58

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_window_absolute_recovery_round58.yaml
```

Change:

```text
early_severe_longitudinal_ceiling_loss_weight: 0.8
early_severe_longitudinal_ceiling_route_min: 0.28
early_severe_longitudinal_ceiling_route_max: 0.36
early_severe_longitudinal_ceiling: 0.25
```

Offline:

```text
val_loss: 0.017780
early_left strict: PASS
  max_abs_zero_steer: 0.017348
early_any strict: PASS
  max_abs_zero_steer: 0.028741
```

512 eval:

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.564103
done_reason_counts: {"max_steps": 1}
```

2048 eval:

```text
collision_count: 0
offroad_count: 1
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.615385
ppo_gate_status: BLOCK
```

Decision: best long-horizon safety so far by offroad count, but too slow and still not offroad-zero.

## Round 59

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_window_absolute_recovery_round59.yaml
```

Change:

```text
early_severe_longitudinal_ceiling_loss_weight: 0.4
early_severe_longitudinal_ceiling: 0.35
```

Offline:

```text
val_loss: 0.013926
early_left strict: PASS
  max_abs_zero_steer: 0.020649
early_any strict: PASS
  max_abs_zero_steer: 0.025460
```

512 eval:

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.666667
```

2048 eval:

```text
collision_count: 0
offroad_count: 2
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.666667
ppo_gate_status: BLOCK
```

Decision: no PPO. Speed ceiling was less damaging to progress than Round58, but safety regressed back to two offroads.

## Current Best

For the strict objective:

```text
Best offline gate: Round57
  early_left strict PASS
  early_any strict PASS
  512 safety PASS
  2048 offroad_count: 2

Best 2048 offroad count: Round58
  early_left strict PASS
  early_any strict PASS
  512 safety PASS
  2048 offroad_count: 1
  max_route_progress: 0.615385
```

Do not start PPO.

The current bottleneck is no longer early-left sign convention or initial zero-bias. The remaining failure is late-route recovery around route progress `0.28-0.36`: the actor produces generally correct recovery signs, but lateral recovery is too late or too slow, and adding speed suppression helps safety while hurting route progress.

Next round should avoid broad speed suppression and instead target the late-window recovery timing directly, preferably with a smoother route-window objective or a dataset relabel/filter focused on the exact route `0.28-0.36`, `lane_offset < -1.5`, `heading_error > 10` rows that precede the remaining offroads.
