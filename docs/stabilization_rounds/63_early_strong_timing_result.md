# Round 63 Early Strong Timing Result

Date: 2026-05-02

Status: BLOCKED BEFORE PPO

## Purpose

Round62 preserved offline actor sign gates and 512 safety, but still failed 2048 with two offroad terminations around route progress `0.307692`. Round63 tested whether the same late recovery window should activate earlier and with a stronger absolute steering margin.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_early_strong_timing_round63.yaml
```

Change from Round60:

```text
late_recovery_absolute_steer_loss_weight: 0.35
late_recovery_route_min: 0.28
late_recovery_route_max: 0.32
late_recovery_abs_lane_offset_min: 1.2
late_recovery_abs_heading_error_min: 8.0
late_recovery_absolute_steer_margin: 0.42
early_center_zero_steer_loss_weight: 0.5
```

## BC

```text
val_loss: 0.011842
```

## Offline Actor Sign Audit

```text
early_left: PASS
  max_abs_zero_steer: 0.026566
  min_abs_delta: 0.014469

early_any: PASS
  max_abs_zero_steer: 0.018031
  min_abs_delta: 0.017449
```

## CARLA 512 Eval

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
ppo_gate_status: BLOCK
```

PPO gate remained blocked by action jitter and correction diagnostics, so PPO was not started.

## CARLA 2048 Eval

```text
collision_count: 0
offroad_count: 1
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
corrective_sign_rate: 0.822222
visual_center_correction_rate: 0.655556
ppo_gate_status: BLOCK
```

Decision: do not start PPO.

## Failure Window

The remaining offroad happened around the same late route region. The actor initially produced strong right recovery and braking, but the steering then collapsed for several steps while lane offset kept growing.

```text
step 680: route 0.307692, lane_offset -2.099, heading 21.638, steer 0.667, longitudinal -0.985
step 681: route 0.307692, lane_offset -2.102, heading 20.081, steer 0.666, longitudinal -0.992
step 682: route 0.307692, lane_offset -2.104, heading 18.821, steer 0.558, longitudinal -1.000
step 683: route 0.307692, lane_offset -2.110, heading 17.842, steer 0.288, longitudinal 0.357
step 684: route 0.307692, lane_offset -2.123, heading 17.247, steer 0.122, longitudinal 0.242
step 685: route 0.307692, lane_offset -2.143, heading 16.983, steer 0.038, longitudinal 0.208
step 686: route 0.307692, lane_offset -2.166, heading 16.851, steer 0.047, longitudinal 0.128
step 687: route 0.333333, lane_offset -2.189, heading 16.643, steer 0.216, longitudinal 0.158
step 688: route 0.333333, lane_offset -2.207, heading 16.205, steer 0.460, longitudinal -0.215
step 689: route 0.333333, lane_offset -2.219, heading 15.514, steer 0.471, longitudinal 0.021
step 690: route 0.333333, lane_offset -2.226, heading 14.635, steer 0.415, longitudinal 0.234, offroad
```

## Interpretation

Round63 improved the long-horizon result versus Round62 from two offroad terminations to one, and the 2048 corrective sign rate rose to `0.822222`. The failure mode changed from simply responding too late to failing to sustain recovery through the `0.307692 -> 0.333333` route transition.

The next useful round should target recovery persistence in the same route area, without adding runtime steering overrides. A narrow offline loss that keeps right recovery active across `route_progress 0.30-0.34` when `lane_offset <= -1.8` and `heading_error >= 12` is more promising than further broad margin increases.
