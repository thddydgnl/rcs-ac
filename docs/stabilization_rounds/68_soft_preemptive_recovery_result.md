# Round 68 Soft Preemptive Recovery Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round67 showed that starting the late recovery mask at route progress `0.24` and lane offset `0.9` targets the observed pre-offroad window, but the `0.42` absolute steer margin broke the early-left zero-bias gate. Round68 kept the preemptive mask and softened the margin to `0.34`.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_soft_preemptive_recovery_round68.yaml
```

Change from Round67:

```text
late_recovery_absolute_steer_margin: 0.34
```

The preemptive mask was kept:

```text
late_recovery_route_min: 0.24
late_recovery_route_max: 0.32
late_recovery_abs_lane_offset_min: 0.9
late_recovery_abs_heading_error_min: 8.0
```

## BC

```text
val_loss: 0.013002
```

## Offline Actor Sign Audit

```text
early_left: BLOCK
  max_abs_zero_steer: 0.050245
  min_abs_delta: 0.019124

early_any: PASS
  max_abs_zero_steer: 0.009678
  min_abs_delta: 0.017977
```

## Decision

Do not run CARLA and do not start PPO.

Softening the margin did not recover the strict early-left zero-bias gate. The broader preemptive mask appears to leak into the early-left probe behavior.

## Interpretation

The preemptive route window is still the right area to investigate, but `abs_lane_offset_min: 0.9` is too broad for the current loss form. The next round should keep the earlier route trigger but make the mask more selective, targeting the observed `route 0.256410` / lane offset around `-1.3` area rather than all rows above `0.9`.
