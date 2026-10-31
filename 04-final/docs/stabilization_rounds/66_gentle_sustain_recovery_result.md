# Round 66 Gentle Sustain Recovery Result

Date: 2026-05-02

Status: BLOCKED BEFORE PPO

## Purpose

Round64 and Round65 showed that extending the late recovery route window through `0.34` can break the early-left zero-bias gate when the absolute steering margin is too strong. Round66 kept the sustained route window and reduced the margin to `0.34`.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_gentle_sustain_recovery_round66.yaml
```

Change from Round65:

```text
late_recovery_absolute_steer_margin: 0.34
```

The sustained late route window was kept:

```text
late_recovery_route_min: 0.28
late_recovery_route_max: 0.34
late_recovery_abs_lane_offset_min: 1.2
late_recovery_abs_heading_error_min: 8.0
```

## BC

```text
val_loss: 0.009692
```

## Offline Actor Sign Audit

```text
early_left: PASS
  max_abs_zero_steer: 0.018275
  min_abs_delta: 0.015031

early_any: PASS
  max_abs_zero_steer: 0.029006
  min_abs_delta: 0.017436
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
corrective_sign_rate: 0.710692
visual_center_correction_rate: 0.572327
ppo_gate_status: BLOCK
```

Decision: do not start PPO.

## Failure Windows

First offroad:

```text
step 661: route 0.282051, lane_offset -2.174, heading 26.293, steer 0.568, longitudinal -0.406
step 662: route 0.282051, lane_offset -2.223, heading 23.873, steer 0.610, longitudinal -0.728
step 663: route 0.307692, lane_offset -2.255, heading 21.317, steer 0.629, longitudinal -0.883
step 664: route 0.307692, lane_offset -2.273, heading 18.731, steer 0.637, longitudinal -0.940
step 665: route 0.307692, lane_offset -2.279, heading 16.197, steer 0.631, longitudinal -0.963
step 666: route 0.307692, lane_offset -2.276, heading 13.751, steer 0.618, longitudinal -0.972
step 667: route 0.307692, lane_offset -2.266, heading 11.430, steer 0.600, longitudinal -0.978
step 668: route 0.307692, lane_offset -2.250, heading 9.252, steer 0.570, longitudinal -0.982
step 669: route 0.307692, lane_offset -2.229, heading 7.227, steer 0.547, longitudinal -0.984, offroad
```

Second offroad:

```text
step 1855: route 0.307692, lane_offset -2.218, heading 28.731, steer 0.536, longitudinal -0.956
step 1856: route 0.307692, lane_offset -2.256, heading 27.049, steer 0.618, longitudinal -0.970
step 1857: route 0.307692, lane_offset -2.283, heading 25.218, steer 0.639, longitudinal -0.977
step 1858: route 0.307692, lane_offset -2.303, heading 23.392, steer 0.620, longitudinal -0.969
step 1859: route 0.307692, lane_offset -2.316, heading 21.611, steer 0.616, longitudinal -0.980
step 1860: route 0.307692, lane_offset -2.326, heading 19.893, steer 0.588, longitudinal -0.972
step 1861: route 0.307692, lane_offset -2.331, heading 18.258, steer 0.589, longitudinal -0.984
step 1862: route 0.333333, lane_offset -2.335, heading 16.753, steer 0.545, longitudinal -0.974
step 1863: route 0.333333, lane_offset -2.336, heading 15.308, steer 0.560, longitudinal -0.965, offroad
```

## Interpretation

Round66 restored the offline gates and 512 safety, but long-horizon safety regressed to two offroad terminations. Unlike Round63, the actor did not collapse steering in the failure window; it produced sustained right recovery and strong braking, but only after lane offset was already around `-2.17` to `-2.33`.

This means the route-window extension did not solve the root timing issue. The remaining problem is not just sustaining steering through route `0.333333`; the policy must start recovery before lane offset reaches roughly `-2.1`. The next useful direction is to move the supervised recovery trigger earlier in state severity, or to create targeted failure-window data/relabeling for lane offsets around `-1.5` to `-2.0` before offroad is unavoidable.
