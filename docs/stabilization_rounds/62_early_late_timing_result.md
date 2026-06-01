# Round 62 Early Late Timing Result

Date: 2026-05-02

Status: BLOCKED BEFORE PPO

## Purpose

Round60 produced the first observed 2048 clean safety pass, but the repeat 2048 eval failed again with two offroad terminations. Round62 tested whether activating the late timing loss earlier and more gently would improve repeatability without adding runtime guards.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_early_late_timing_round62.yaml
```

Change from Round60:

```text
late_recovery_absolute_steer_loss_weight: 0.25
late_recovery_route_min: 0.28
late_recovery_route_max: 0.32
late_recovery_abs_lane_offset_min: 1.2
late_recovery_abs_heading_error_min: 8.0
late_recovery_absolute_steer_margin: 0.36
early_center_zero_steer_loss_weight: 0.5
```

## BC

```text
val_loss: 0.009428
```

## Offline Actor Sign Audit

```text
early_left: PASS
  max_abs_zero_steer: 0.019748
  min_abs_delta: 0.011045

early_any: PASS
  max_abs_zero_steer: 0.017903
  min_abs_delta: 0.014068
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

## CARLA 2048 Eval

```text
collision_count: 0
offroad_count: 2
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
ppo_gate_status: BLOCK
```

Decision: do not start PPO.

## Failure Window

Offroad still occurred around route progress `0.307692`.

First failure:

```text
step 661: lane_offset -2.237, heading 26.770, steer 0.364, longitudinal 0.433
step 666: lane_offset -2.518, heading 17.419, steer 0.420, longitudinal 0.114
step 669: lane_offset -2.581, heading 11.750, steer 0.501, longitudinal -0.439, offroad
```

Second failure:

```text
step 1846: lane_offset -2.194, heading 19.548, steer 0.403, longitudinal 0.232
step 1848: lane_offset -2.252, heading 15.329, steer 0.561, longitudinal -0.666
step 1854: lane_offset -2.626, heading 31.882, steer 0.633, longitudinal -0.963, offroad
```

## Interpretation

Round62 preserved strict offline gates and 512 safety, but did not improve 2048 repeat reliability. The actor still lets lane offset grow past roughly `-2.2` before recovery becomes decisive. Stronger braking eventually appears, but often after the vehicle is already too close to the road edge.

Next work should not proceed to PPO yet. The likely next useful direction is a more direct offline target for earlier correction timing in the `route 0.28-0.32` window, possibly by relabeling or upweighting rows before `lane_offset` exceeds `2.0`, rather than pushing broad speed or steering margins after the state is already severe.
