# Round90 Shared Reward Reverification

Date: 2026-05-04

Status: **512 clean safety PASS, 2048 strict clean safety FAIL**

## Purpose

Re-run the active `Round90 shared reward` baseline after promoting the shared
reward config as the current RCS-AC clean baseline.

This run uses the same Round90 checkpoint and the active shared reward config:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_shared_reward_baseline.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Windows eval worktree: `C:\Users\thddy\Documents\code\RCS-AC-first-round90-shared-reward-reverify`
- `env.lateral_recovery.enabled: false`
- no runtime lane-offset steering override
- no handcoded steering rule

## Results

| Run | Steps | Done reasons | Strict safety | Max route | Mean route | Action jitter | Visual-center correction |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| previous probe 512 | 512 | `{"max_steps": 1}` | PASS | 0.384615 | 0.192458 | 0.133210 | 0.000000 |
| previous probe 2048 | 2048 | `{"max_steps": 4}` | PASS | 0.435897 | 0.192558 | 0.151773 | 0.470968 |
| reverify 512 | 512 | `{"max_steps": 1}` | PASS | 0.461538 | 0.226262 | 0.134540 | 0.647059 |
| reverify 2048 | 2048 | `{"max_steps": 2, "offroad": 2}` | FAIL | 0.461538 | 0.186624 | 0.136260 | 0.485106 |
| reverify 2048 retry1 | 2048 | `{"max_steps": 2, "offroad": 2}` | FAIL | 0.461538 | 0.196565 | 0.137118 | 0.474860 |

Strict safety requires:

- collision count `0`
- offroad rate `0.0`
- severe early offroad count `0`
- stuck rate `0.0`
- completed steps equal requested steps

The 512-step reverify passed, but both 2048-step attempts completed with two
offroad terminations. These are model/environment rollout results, not CARLA
crashes or timeouts.

## Interpretation

The previous shared-reward probe showed a clean 2048 pass, but the fresh
reverification did not reproduce strict long-horizon clean safety. The failures
were not severe early offroad; they happened later in route progress. This means
Round90 shared reward should not be treated as a fully locked long-horizon clean
baseline without either:

1. another controlled rerun matrix that explains the variability, or
2. a stronger baseline candidate that passes repeated 2048 clean evals.

The persistent weak signals remain:

- `mean_action_jitter_l1` is above the PPO gate threshold.
- `visual_center_correction_rate` stays below the target `0.70`.
- 2048 rollout can still drift into offroad despite zero collision/stuck.

## Sensor-Input Relevance

This failure is consistent with the earlier diagnosis: the policy can complete
short clean runs, but its RGB/route visual-centering correction is not robust
enough across longer episodes. The actor still relies on fused RGB/LiDAR/ego/route
features plus RGB-perception compact features and oracle scalar actor features.
The visual-center correction signal is diagnostic, not a hard runtime steering
override.

## Decision

Do not advance this checkpoint/config to PPO fine-tuning or robustness comparison
as a strict clean-safe 2048 baseline until the long-horizon offroad regression is
resolved or a repeated 2048 pass is obtained under the same protocol.
