# Round 14 - Masked Recovery Visual Pretrain

Date: 2026-05-01

## Goal

Use the working RGB/semantic mask path from Round 13 on the actual failure window instead of the clean early-route window. Round 13 proved the mask supervision path is active, but the no-lateral policy still drifted offroad. This round collects recovery-focused masked transitions around route progress 0.24-0.38 and trains the RGB perception actor adapter on those recovery targets.

## Hypothesis

The policy is failing because the visual adapter learned mostly clean-lane evidence and did not see enough near-failure corrections. A recovery-focused dataset should make the RGB/lane-edge features more useful for steering recovery before PPO explores from the no-lateral configuration.

## Procedure

1. Run local static/smoke validation before using Windows CARLA.
2. Sync the latest Mac branch to Windows with `git pull --ff-only`.
3. Start CARLA through the interactive scheduled-task helper if SSH cannot launch a visible CARLA window directly.
4. Collect a masked recovery dataset with lateral recovery enabled:
   - config: `configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml`
   - checkpoint: `checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt`
   - route window: 0.24-0.38
   - mask keys required: `aux_road_mask_target`, `aux_lane_edge_target`, `aux_mask_targets_available`
5. BC pretrain only the perception/adapter path:
   - config: `configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml`
   - trainable modules: `perception_aux_head`, `rgb_perception_actor_features`, `actor_input_adapter`
6. Evaluate the pretrain checkpoint without lateral recovery.
7. Run low-std PPO only if the pretrain gate is healthy and the evaluation is not clearly worse than Round 13.
8. Pull back the complete text/table logs for every run under one local round folder.

## Gates

- Dataset gate: mask availability is nonzero and road/lane target means are plausible.
- BC gate: perception auxiliary loss is logged, visual actor delta is nonzero, and action loss remains finite.
- Eval gate: compare route progress, offroad/stuck termination, lane offset, and heading error against Round 13.

## Result

Pending.
