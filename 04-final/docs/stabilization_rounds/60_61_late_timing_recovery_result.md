# Rounds 60-61 Late Timing Recovery Result

Date: 2026-05-02

Status: PARTIAL PROGRESS, BLOCKED BEFORE PPO

## Starting Point

Rounds 57-59 showed that early-left sign and initial zero-bias were mostly fixed, but long-horizon offroad remained around route progress `0.28-0.36`.

The Windows dataset audit on `rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz` showed:

```text
late_left_preoffroad count: 320
route range: 0.282051-0.307692
lane range: -2.876949 to -1.500554
heading range: 10.183887 to 37.517056
target corrective sign rate: 1.0
target corrected steer mean: 0.320446
target corrected steer >= 0.35: 0.40625
```

Interpretation: label sign is not the main problem in this window. The remaining issue is recovery timing and magnitude when lateral drift is already large.

## Round 60

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_timing_recovery_round60.yaml
```

Change from Round57:

```text
late_recovery_absolute_steer_loss_weight: 0.35
late_recovery_route_min: 0.28
late_recovery_route_max: 0.32
late_recovery_abs_lane_offset_min: 1.5
late_recovery_abs_heading_error_min: 10.0
late_recovery_absolute_steer_margin: 0.40
early_center_zero_steer_loss_weight: 0.5
```

This targets only the repeated late-left pre-offroad window. It is offline privileged supervision only; no runtime steering override or route/lane controller was added.

BC:

```text
val_loss: 0.010826
```

Offline actor sign audit:

```text
early_left: PASS
  max_abs_zero_steer: 0.015635
  min_abs_delta: 0.012432

early_any: PASS
  max_abs_zero_steer: 0.024162
  min_abs_delta: 0.016009
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
```

2048 eval, first run:

```text
collision_count: 0
offroad_rate: 0.0
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
done_reason_counts: {"goal": 2, "max_steps": 2}
ppo_gate_status: BLOCK
```

This was the first observed 2048 clean safety pass for this stabilization branch.

2048 eval, repeat:

```text
collision_count: 0
offroad_count: 2
early_offroad_count: 0
severe_early_offroad_count: 0
stuck_rate: 0.0
max_route_progress: 1.0
ppo_gate_status: BLOCK
```

Decision: do not start PPO. Round60 is a meaningful improvement signal, but the clean 2048 result did not reproduce.

## Repeat Failure Window

The repeat 2048 offroad events still occurred near route progress `0.307692-0.333333`. Round60 produced strong corrective steering and heavy braking after drift became severe:

```text
step 651: lane_offset -1.70, heading 19.56, steer 0.398, longitudinal 0.558
step 659: lane_offset -2.25, heading 30.89, steer 0.491, longitudinal 0.109
step 662: lane_offset -2.41, heading 23.32, steer 0.704, longitudinal -0.892
step 667: lane_offset -2.39, heading 9.79, steer 0.667, longitudinal -0.978, offroad
```

Interpretation: once the actor commits, the recovery is strong. The failure is that speed remains high while lane offset grows from roughly `-1.7` to `-2.25`; the corrective response is still late.

## Round 61

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_late_timing_speed_round61.yaml
```

Change from Round60:

```text
early_severe_longitudinal_ceiling_loss_weight: 0.4
early_severe_longitudinal_ceiling_route_min: 0.28
early_severe_longitudinal_ceiling_route_max: 0.32
early_severe_longitudinal_ceiling: 0.25
```

BC:

```text
val_loss: 0.018302
```

Offline actor sign audit:

```text
early_left: PASS
  max_abs_zero_steer: 0.019426

early_any: BLOCK
  max_abs_zero_steer: 0.031047
```

Decision: do not run CARLA. Even the narrowed speed ceiling pushed early-any zero-bias just outside the strict gate.

## Current Best

```text
Round60:
  strict offline gates: PASS
  512 safety: PASS
  2048 first run: safety PASS
  2048 repeat: offroad_count 2

Round58:
  best repeat-style long-horizon offroad count so far: 1
  too slow, max_route_progress 0.615385
```

Do not start PPO yet.

The next round should focus on earlier response before `lane_offset` reaches about `-2.0`, without broad speed suppression. A candidate direction is a narrower timing loss that activates at `route 0.28-0.32`, `abs(lane_offset) >= 1.2`, `abs(heading_error) >= 8`, and encourages visual/route corrective action earlier while keeping the early-center zero-bias loss active.
