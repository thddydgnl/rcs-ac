# Round90 Original No-Shaping Reverification

Date: 2026-05-04

Status: **512 clean safety PASS, 2048 strict clean safety FAIL**

## Purpose

Re-run the original Round90 config that previously produced a stored 2048
offroad-zero result, then compare it against the shared-reward revalidation.

This run uses:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Reward block: none in config, so this is the original no-shaping contract
- Windows eval worktree: `C:\Users\thddy\Documents\code\RCS-AC-first-round90-shared-reward-reverify`
- `env.lateral_recovery.enabled: false`
- no runtime lane-offset steering override
- no handcoded steering rule

## Results

| Run | Steps | Done reasons | Strict safety | Max route | Mean route | Action jitter | Visual-center correction |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| stored original 2048 | 2048 | `{"max_steps": 4}` | PASS | 0.410256 | 0.183681 | 0.146305 | 0.490798 |
| previous shared 2048 | 2048 | `{"max_steps": 4}` | PASS | 0.435897 | 0.192558 | 0.151773 | 0.470968 |
| shared reverify 2048 | 2048 | `{"max_steps": 2, "offroad": 2}` | FAIL | 0.461538 | 0.186624 | 0.136260 | 0.485106 |
| shared reverify 2048 retry1 | 2048 | `{"max_steps": 2, "offroad": 2}` | FAIL | 0.461538 | 0.196565 | 0.137118 | 0.474860 |
| original reverify 512 | 512 | `{"max_steps": 1}` | PASS | 0.461538 | 0.226212 | 0.131149 | 0.571429 |
| original reverify 2048 | 2048 | `{"max_steps": 2, "offroad": 2}` | FAIL | 0.461538 | 0.196101 | 0.153061 | 0.489899 |
| original reverify 2048 retry1 | 2048 | `{"max_steps": 3, "offroad": 1}` | FAIL | 0.461538 | 0.197967 | 0.146829 | 0.504425 |

## Interpretation

The original no-shaping Round90 result also failed to reproduce strict
offroad-zero 2048 clean safety in the fresh re-run. This means the fresh
shared-reward failures are not strong evidence that reward shaping alone caused
the long-horizon offroad issue.

The more likely current readout is:

- Round90 remains short-horizon clean-safe at 512 steps.
- Both original and shared-reward variants are fragile at 2048 under the current
  CARLA/runtime revalidation condition.
- The repeated failure is later-route offroad, not severe early offroad.
- Action jitter and visual-center correction remain weak in both variants.

## What Differs From The Stored PASS Runs

Stored original/shared 2048 PASS runs had exactly four `max_steps` episodes.
Fresh revalidation produced five episodes because offroad terminations reset
early and evaluation continued until total requested steps were accumulated.

This is a model rollout failure under the strict clean gate, not a CARLA crash
or timeout. The CARLA readiness guard passed and the Python eval commands
completed normally.

## Decision

Do not use either original Round90 or Round90 shared reward as a locked
2048-clean baseline until a repeated controlled 2048 matrix passes strict
offroad-zero safety or a stronger replacement checkpoint is produced.
