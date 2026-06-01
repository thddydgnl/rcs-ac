# Round101 Round90 Clean Baseline Confirmation

## Goal

After Rounds 93-100 failed to improve on Round90, rerun the current best clean-safety checkpoint as a baseline confirmation before moving to PPO or robustness comparison work.

## Setup

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Eval: 2048 clean steps
- CARLA was force restarted before the run.

No model or runtime steering changes were made.

## Result

2048 clean eval: PASS

- completed steps: `2048`
- episodes: `4`
- done reasons: `max_steps: 4`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- max route progress: `0.384615`
- final route progress: `0.358974`
- mean action jitter L1: `0.138907`
- steer jitter: `0.080903`
- longitudinal jitter: `0.058004`
- corrective need count: `162`
- route-offset corrective sign rate: `0.833333`
- visual-center correction rate: `0.462963`
- PPO gate: BLOCK
  - `mean_action_jitter_l1 0.138907 > 0.070000`
  - `visual_center_correction_rate 0.462963 < 0.700000`

## Decision

Round90 remains the best clean-safety baseline. It has now passed a fresh 2048 clean rerun after the failed Round93-100 tuning branch.

This does not complete the full `/goal` because PPO route-progress improvement and robustness comparison are still pending. It does, however, give a stronger basis for treating Round90 as the RCS-AC clean baseline candidate for fair weather/dropout comparison experiments.

## Next Direction

Do not replace Round90 with any of Rounds 93-100.

Proceed with two clearly separated tracks:

1. Baseline comparison track: use Round90 as the clean-safe RCS-AC checkpoint and begin matching PPO/SAC clean-baseline setup.
2. Optional improvement track: run any PPO route-progress probes as separate candidates, and only promote them if they preserve 512/2048 safety.
