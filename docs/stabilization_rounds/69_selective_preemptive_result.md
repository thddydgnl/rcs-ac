# Round 69 Selective Preemptive Recovery Result

Date: 2026-05-02

Status: BLOCKED BEFORE PPO

## Purpose

Round67 and Round68 showed that a broad preemptive mask with `abs_lane_offset_min: 0.9` breaks the early-left zero-bias gate. Round69 kept the earlier route trigger but narrowed the lane-offset threshold back to `1.2`, targeting the observed `route 0.256410` / lane around `-1.3` region without covering the broader `-1.0` region.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_selective_preemptive_round69.yaml
```

Change from Round63:

```text
late_recovery_route_min: 0.24
late_recovery_route_max: 0.32
late_recovery_abs_lane_offset_min: 1.2
late_recovery_abs_heading_error_min: 8.0
late_recovery_absolute_steer_margin: 0.42
```

## BC

```text
val_loss: 0.011745
```

## Offline Actor Sign Audit

```text
early_left: PASS
  max_abs_zero_steer: 0.023657
  min_abs_delta: 0.015177

early_any: PASS
  max_abs_zero_steer: 0.024977
  min_abs_delta: 0.016712
```

## CARLA 512 Eval

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.846154
done_reason_counts: {"max_steps": 1}
ppo_gate_status: BLOCK
```

The 512 run was safety-clean but did not complete the route.

## CARLA 2048 Eval

```text
collision_count: 0
offroad_count: 2
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 0.871795
corrective_sign_rate: 0.752747
visual_center_correction_rate: 0.538462
ppo_gate_status: BLOCK
```

Decision: do not start PPO.

## Failure Windows

First offroad:

```text
step 681: route 0.282051, lane_offset -2.022, heading 29.355, steer 0.356, longitudinal 0.637
step 682: route 0.282051, lane_offset -2.105, heading 27.744, steer 0.394, longitudinal 0.588
step 683: route 0.282051, lane_offset -2.179, heading 25.938, steer 0.402, longitudinal 0.579
step 684: route 0.307692, lane_offset -2.245, heading 24.038, steer 0.401, longitudinal 0.569
step 685: route 0.307692, lane_offset -2.303, heading 22.095, steer 0.399, longitudinal 0.558
step 686: route 0.307692, lane_offset -2.355, heading 20.165, steer 0.379, longitudinal 0.561
step 687: route 0.307692, lane_offset -2.401, heading 18.288, steer 0.365, longitudinal 0.554
step 688: route 0.307692, lane_offset -2.441, heading 16.457, steer 0.341, longitudinal 0.565
step 689: route 0.307692, lane_offset -2.474, heading 14.674, steer 0.377, longitudinal 0.275
step 690: route 0.307692, lane_offset -2.497, heading 12.768, steer 0.405, longitudinal 0.360
step 691: route 0.307692, lane_offset -2.507, heading 10.708, steer 0.477, longitudinal 0.082, offroad
```

Second offroad:

```text
step 1877: route 0.282051, lane_offset -1.904, heading 5.786, steer 0.444, longitudinal 0.142
step 1878: route 0.282051, lane_offset -2.115, heading 29.234, steer 0.488, longitudinal -0.213
step 1879: route 0.282051, lane_offset -2.182, heading 26.890, steer 0.612, longitudinal -0.661
step 1880: route 0.282051, lane_offset -2.229, heading 24.302, steer 0.665, longitudinal -0.868
step 1881: route 0.282051, lane_offset -2.257, heading 21.567, steer 0.683, longitudinal -0.944
step 1882: route 0.282051, lane_offset -2.271, heading 18.811, steer 0.677, longitudinal -0.966
step 1883: route 0.307692, lane_offset -2.273, heading 16.134, steer 0.663, longitudinal -0.973
step 1884: route 0.307692, lane_offset -2.266, heading 13.554, steer 0.658, longitudinal -0.973
step 1885: route 0.307692, lane_offset -2.252, heading 11.125, steer 0.623, longitudinal -0.973
step 1886: route 0.307692, lane_offset -2.231, heading 8.837, steer 0.608, longitudinal -0.977
step 1887: route 0.307692, lane_offset -2.207, heading 6.724, steer 0.564, longitudinal -0.975, offroad
```

## Interpretation

Round69 restored the strict offline gates, but it did not improve long-horizon safety or route progress. In one failure the actor still accelerated while far off center, and in the other it braked strongly but still started too late to avoid offroad.

The config-only late recovery mask family appears insufficient. The next useful direction is not another broad margin/window tweak, but targeted data or relabeling for pre-offroad states where lane offset is about `-1.3` to `-2.0`, with explicit longitudinal reduction and recovery steering before the state reaches `-2.1`.
