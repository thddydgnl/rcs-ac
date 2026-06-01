# Rounds 75-76 Late Recovery Brake Result

Date: 2026-05-02

Status: BLOCKED BEFORE PPO

## Purpose

Round63 remained the best CARLA candidate but failed 2048 once near:

```text
route_progress: 0.307692 -> 0.333333
lane_offset: about -2.1 -> -2.23
heading_error: about 21 -> 14
```

Rounds 75-76 targeted this window with offline privileged supervision only. No runtime steering override, lane controller, or hidden simulator signal was added to inference.

## Round75

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_sustained_late_recovery_round75.yaml
```

Changes from Round63:

```text
late_recovery_route_min: 0.30
late_recovery_route_max: 0.34
late_recovery_abs_lane_offset_min: 1.8
late_recovery_abs_heading_error_min: 12.0
late_recovery_absolute_steer_margin: 0.55
late_recovery_absolute_steer_loss_weight: 0.50

early_severe_longitudinal_ceiling_route_min: 0.30
early_severe_longitudinal_ceiling_route_max: 0.34
early_severe_longitudinal_ceiling: 0.05
early_severe_longitudinal_ceiling_loss_weight: 0.5
```

BC:

```text
init_checkpoint: checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt
trainable_module_names: scalar_actor_adapter, recurrent_actor_critic
val_loss: 0.033082
```

Separated offline gates:

```text
early_any neutral zero/sign, max_samples=16: PASS
  sample_pass_rate: 1.0
  max_abs_zero_steer: 0.014233
  min_abs_delta: 0.012164

early_left corrective sign, max_samples=16, zero-bias not enforced: PASS
  sample_pass_rate: 1.0
  min_abs_delta: 0.010990

legacy single early_left gate: PASS
  max_abs_zero_steer: 0.013801
  min_abs_delta: 0.011837
```

CARLA 512:

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.538462
pass_status: PASS
ppo_gate_status: BLOCK
```

The first 2048 attempt failed at simulator reset/load_world with CARLA connection refused/timeout, so it was discarded and CARLA was force-restarted.

CARLA 2048 retry:

```text
collision_count: 0
offroad_count: 1
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.538462
corrective_sign_rate: 0.902778
mean_corrective_visual_zero_steer_delta: 0.035732
ppo_gate_status: BLOCK
```

Failure window:

```text
step 810: route 0.307692, lane_offset -1.276, heading 23.615, steer 0.188, longitudinal 0.222
step 815: route 0.307692, lane_offset -1.530, heading 20.430, steer 0.171, longitudinal 0.212
step 819: route 0.307692, lane_offset -2.053, heading 48.494, steer 0.177, longitudinal 0.192
step 821: route 0.307692, lane_offset -2.330, heading 46.587, steer 0.309, longitudinal 0.239
step 824: route 0.307692, lane_offset -2.706, heading 42.325, steer 0.368, longitudinal 0.224
step 826: route 0.333333, lane_offset -2.933, heading 39.230, steer 0.360, longitudinal 0.219
step 829: route 0.333333, lane_offset -3.244, heading 34.673, steer 0.337, longitudinal 0.187, offroad
```

Interpretation: Round75 improved corrective sign diagnostics, but actual recovery was too weak and still accelerated through the dangerous late window. It also regressed route progress compared with Round63.

## Round76

Round76 increased the same late-window intervention:

```text
configs/bc_frame_rgb_perception_actor_features_v2_strong_late_brake_round76.yaml

late_recovery_absolute_steer_loss_weight: 1.0
late_recovery_absolute_steer_margin: 0.75
early_severe_longitudinal_ceiling_loss_weight: 2.0
early_severe_longitudinal_ceiling: -0.20
```

BC:

```text
val_loss: 0.088327
```

Offline gates:

```text
early_any neutral zero/sign, max_samples=16: BLOCK
  sample_pass_rate: 0.5625
  max_abs_zero_steer: 0.043226
  min_abs_delta: 0.014472

early_left corrective sign, max_samples=16, zero-bias not enforced: PASS
  sample_pass_rate: 1.0
  min_abs_delta: 0.014444
```

Decision: do not run CARLA.

## Interpretation

Round75 shows the current late-window loss can point the actor in the right direction but does not produce enough steering/braking in closed loop. Round76 shows simply increasing the same loss damages neutral zero-bias before CARLA.

The next useful direction is not another broad strength increase. It should use a narrower trainable scope or a more direct late-window corrective dataset/objective so the recovery behavior changes without shifting centered driving.
