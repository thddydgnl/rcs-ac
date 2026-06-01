# Round 64 Sustain Recovery Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round63 reduced the 2048 offroad count from two to one, but the remaining failure happened when recovery steering collapsed around the route transition from `0.307692` to `0.333333`. Round64 tested the narrowest config-only change: keep the same late recovery target active through route progress `0.34`.

This is offline privileged supervision only. No runtime steering override, lane controller, or hidden privileged inference input was added. `lateral_recovery.enabled` remained `false`.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_sustain_recovery_round64.yaml
```

Change from Round63:

```text
late_recovery_route_max: 0.34
```

The rest of the late recovery window was kept unchanged:

```text
late_recovery_absolute_steer_loss_weight: 0.35
late_recovery_route_min: 0.28
late_recovery_abs_lane_offset_min: 1.2
late_recovery_abs_heading_error_min: 8.0
late_recovery_absolute_steer_margin: 0.42
```

## BC

```text
val_loss: 0.011396
```

## Offline Actor Sign Audit

```text
early_left: BLOCK
  max_abs_zero_steer: 0.031652
  min_abs_delta: 0.016239

early_any: PASS
  max_abs_zero_steer: 0.010019
  min_abs_delta: 0.019123
```

## Decision

Do not run CARLA and do not start PPO.

Round64 strengthened the actor response, but the early-left strict zero-bias gate regressed just beyond the allowed `0.03` threshold. Extending the late recovery route window globally from `0.32` to `0.34` is therefore too broad for the current loss shape.

## Interpretation

The Round63 failure still suggests that recovery needs to persist through route progress `0.333333`, but Round64 shows that simply widening the existing late recovery mask leaks into the early-left bias behavior. The next round should avoid broadening the shared late mask further. A better next step is a more selective persistence target for the observed failure region, or a softer route-window extension with a lower margin/weight that preserves the early-left zero-bias gate.
