# Rounds 49-52 Long-Horizon Offroad Result

Date: 2026-05-02

Status: PARTIAL PROGRESS, BLOCKED BEFORE PPO

## Starting Point

Round48 was the first checkpoint to pass the offline actor sign gate and 512-step clean eval, but it failed the 2048 target:

```text
Round48 2048:
collision_count: 0
offroad_count: 10
early_offroad_count: 1
severe_early_offroad_count: 0
stuck_rate: 0.0
ppo_gate_status: BLOCK
```

The 2048 CSV showed that most offroad events happened around route progress `0.18-0.31`. The actor usually produced the correct recovery sign, but it often reacted late after `lane_offset` had already grown past roughly `2.1-2.9`. Some right-side events also kept longitudinal action very high during lateral recovery.

## Code Change

`train_bc.py` now supports an optional offline-only severe lateral recovery speed objective:

```text
early_severe_longitudinal_ceiling_loss
early_severe_longitudinal_ceiling_rate
early_severe_longitudinal_ceiling_loss_weight
early_severe_longitudinal_ceiling_route_min
early_severe_longitudinal_ceiling_route_max
early_severe_longitudinal_ceiling
```

This loss uses privileged simulator fields only as an offline BC mask. It does not add a runtime speed controller, steering override, or inference-time privileged signal.

## Local Validation

```bash
python -m py_compile rcs_ac_first/scripts/train_bc.py
python -m pytest tests/test_train_bc_perception.py -q
```

Result:

```text
28 passed in 1.17s
```

## Round 49

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_lateral_speed_ceiling_round49.yaml
```

Change:

```text
early_severe_longitudinal_ceiling_loss_weight: 1.5
early_severe_longitudinal_ceiling_route_min: 0.10
early_severe_longitudinal_ceiling_route_max: 0.32
early_severe_longitudinal_ceiling: 0.25
init: Round48 checkpoint
```

BC:

```text
val_loss: 0.018102
```

Actor sign audit:

```text
early_left: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.014967
  left_bias_at_zero: -0.021854

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.009393
  left_bias_at_zero: 0.063526
```

512 eval:

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.512821
done_reason_counts: {"max_steps": 1}
ppo_gate_status: BLOCK
```

Decision: too much speed suppression. Safety stayed clean at 512, but route progress stalled and visual-center correction collapsed.

## Round 50

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_lateral_speed_ceiling_round50.yaml
```

Change:

```text
early_severe_longitudinal_ceiling_loss_weight: 0.35
early_severe_longitudinal_ceiling_route_min: 0.16
early_severe_longitudinal_ceiling_route_max: 0.32
early_severe_longitudinal_ceiling: 0.45
init: Round48 checkpoint
```

BC:

```text
val_loss: 0.008921
```

Actor sign audit:

```text
early_left: BLOCK
  all_axes_rate: 1.0
  min_abs_delta: 0.015308
  left_bias_at_zero: -0.041199

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.009717
  left_bias_at_zero: 0.017211
```

Decision: do not run CARLA. The weaker speed ceiling broke the offline early-left bias gate.

## Round 51

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_mid_route_recovery_round51.yaml
```

Change:

```text
early_severe_route_max: 0.32
init: Round48 checkpoint
```

This tests a wider steering recovery window instead of speed suppression.

BC:

```text
val_loss: 0.010347
```

Actor sign audit:

```text
early_left: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.013492
  left_bias_at_zero: 0.026148

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.009568
  left_bias_at_zero: 0.101666
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
mean_action_jitter_l1: 0.216895
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
mean_action_jitter_l1: 0.168530
```

Decision: clear improvement over Round48 long horizon, but still not 2048 safe because offroad must be zero.

## Round 52

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_left_recovery_round52.yaml
```

Change:

```text
early_severe_route_max: 0.36
early_severe_steer_margin: 0.08
init: Round51 checkpoint
```

BC:

```text
val_loss: 0.008797
```

Actor sign audit:

```text
early_left: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.010741
  left_bias_at_zero: 0.007626

early_any: PASS
  all_axes_rate: 1.0
  min_abs_delta: 0.013224
  left_bias_at_zero: 0.036249
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
mean_action_jitter_l1: 0.140223
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
mean_action_jitter_l1: 0.142699
```

Decision: no 2048 safety promotion. It keeps early safety clean and reduces offroad versus Round48, but still fails the strict 2048 offroad-zero objective.

## Current Best

For strict objective progress:

```text
Round48: first offline sign PASS and 512 PASS, but 2048 offroad_count 10 and early_offroad_count 1.
Round51/Round52: 2048 early_offroad_count 0 and offroad_count 2, but jitter is high and offroad is not zero.
```

Do not start PPO.

The next round should focus on removing the remaining late-route offroad without increasing action jitter. The likely issue is not sign convention anymore; it is recovery timing and control smoothness around route progress `0.30-0.36` when lane offset can exceed `3.0`.
