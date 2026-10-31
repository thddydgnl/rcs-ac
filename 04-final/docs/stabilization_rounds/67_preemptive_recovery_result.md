# Round 67 Preemptive Recovery Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round66 showed that sustained recovery through route progress `0.34` did not solve long-horizon offroad. Inspecting the earlier failure window showed that at route progress `0.256410`, lane offset was already around `-1.03` to `-1.32`, heading error was `18` to `20`, and the actor still produced weak or wrong-direction steering while accelerating. Round67 tested a preemptive recovery mask that starts before the lane offset exceeds `-2.0`.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_preemptive_recovery_round67.yaml
```

Change from Round63:

```text
late_recovery_route_min: 0.24
late_recovery_route_max: 0.32
late_recovery_abs_lane_offset_min: 0.9
late_recovery_abs_heading_error_min: 8.0
late_recovery_absolute_steer_margin: 0.42
```

## BC

```text
val_loss: 0.013315
```

## Offline Actor Sign Audit

```text
early_left: BLOCK
  max_abs_zero_steer: 0.040396
  min_abs_delta: 0.019195

early_any: PASS
  max_abs_zero_steer: 0.001719
  min_abs_delta: 0.020077
```

## Decision

Do not run CARLA and do not start PPO.

The preemptive trigger increased the actor's response strength, but it also pushed the early-left route-target-y zero-steer probe beyond the strict `0.03` limit. This is too biased for promotion.

## Interpretation

Round67 confirms that the earlier route `0.256410` / lane `-1.0` to `-1.3` area is a high-leverage target, but the direct `0.42` margin is too strong. The next round should keep the preemptive mask but soften the margin so the early-left zero-bias gate survives.
