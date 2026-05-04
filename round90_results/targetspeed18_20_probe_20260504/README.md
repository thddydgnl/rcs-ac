# Round90 Target-Speed Wrapper Sweep

Date: 2026-05-04

## Purpose

The full runtime metadata restore showed that the current shared-core runtime
drives Round90 faster than the stored offroad-zero PASS run. This probe checks
whether the late offroad is primarily sensitive to the shared longitudinal
target-speed wrapper rather than a steering-rule change.

This is **not** a promoted RCS-AC baseline and not a hidden steering assist:

- checkpoint unchanged:
  `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- base config unchanged except `env.longitudinal_control.max_target_speed_kmh`
- no lane-offset steering override
- no lateral recovery assist
- no route/lane runtime steering rule
- same change would have to be applied to PPO/SAC if promoted as a common action
  wrapper contract

## Configs

- `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_shared_reward_targetspeed16_probe.yaml`
- `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_shared_reward_targetspeed18_probe.yaml`
- `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_shared_reward_targetspeed20_probe.yaml`

The active Round90 shared-reward baseline still uses max target speed `28.0`.

## Results

| Condition | Steps | Done reasons | Safety | Max route | Mean route | Mean speed | Mean target speed |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| max 28 current | 2048 | `max_steps:2, offroad:2` | FAIL | `0.4615` | `0.1868` | `7.638` | `10.369` |
| max 16 | 512 | `max_steps:1` | PASS | `0.2821` | `0.1463` | `4.719` | `6.696` |
| max 16 | 2048 | `max_steps:4` | PASS | `0.2821` | `0.1393` | `4.645` | `6.603` |
| max 18 | 512 | `max_steps:1` | PASS | `0.2821` | `0.1477` | `4.748` | `6.805` |
| max 18 | 2048 | `max_steps:4` | PASS | `0.3333` | `0.1491` | `4.996` | `7.042` |
| max 20 | 512 | `max_steps:1` | PASS | `0.3590` | `0.1652` | `5.945` | `8.026` |
| max 20 | 2048 | `max_steps:3, offroad:1` | FAIL | `0.3590` | `0.1536` | `5.470` | `7.676` |

Safety means collision/offroad/severe early offroad/stuck all zero.

## Interpretation

Lowering only the global target-speed wrapper is enough to remove the current
2048 offroad at max `16` and max `18`, while max `20` still produces a late
offroad. This strongly supports the latest diagnosis:

1. Round90 is not failing because the checkpoint/config suddenly changed.
2. Current shared-core target-speed behavior can push the same policy into a
   faster, narrower-margin trajectory.
3. The clean-safety boundary for this checkpoint is between target-speed max
   `18` and `20` under the current runtime.

However, max `16` and max `18` are too conservative to promote directly:

- max route drops to `0.282-0.333`
- mean route is lower than the current max `28` run
- PPO gate remains blocked
- route progress is not good enough for the final comparison story

## Decision

Do not promote max `16`, `18`, or `20` as the final clean baseline yet. Treat
this as a cause-isolation result. The next useful stabilization step is a
proper action-contract fix or training-time longitudinal objective that makes
the policy's second action dimension match the runtime target-speed semantics,
instead of only lowering the global speed cap.

Any future comparison baseline must report this explicitly:

- raw actor output
- target-speed wrapper settings
- actual CARLA throttle/brake after wrapper resolution
- whether the same wrapper is used by RCS-AC, PPO, and SAC
