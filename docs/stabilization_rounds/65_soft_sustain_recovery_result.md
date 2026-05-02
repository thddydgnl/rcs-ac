# Round 65 Soft Sustain Recovery Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round64 extended the late recovery route window through `0.34`, but failed the early-left zero-bias gate. Round65 kept the same sustained route window and softened the absolute steering margin from `0.42` to `0.38`.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_soft_sustain_recovery_round65.yaml
```

Change from Round64:

```text
late_recovery_absolute_steer_margin: 0.38
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
val_loss: 0.011023
```

## Offline Actor Sign Audit

```text
early_left: BLOCK
  max_abs_zero_steer: 0.030494
  min_abs_delta: 0.016391

early_any: PASS
  max_abs_zero_steer: 0.016349
  min_abs_delta: 0.018835
```

## Decision

Do not run CARLA and do not start PPO.

The result was very close to the strict gate, but still exceeded the early-left zero-bias limit of `0.03`. This confirms that widening the existing late recovery mask to include route progress `0.333333` is sensitive even when the margin is softened.

## Interpretation

Round65 suggests the route extension may still be viable, but the margin needs to be softer or the mask needs to be more selective. Since both Round64 and Round65 failed on the route-target-y zero-steer probe rather than sign direction or min delta, the next low-risk config-only probe should reduce the sustained-window margin further while preserving the `0.34` route max.
