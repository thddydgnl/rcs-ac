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

Completed on Windows via `song@100.87.219.58`.

Local validation before the remote run:

- `python -m pytest tests`: 9 passed.
- `python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml --device cpu`: passed.
- `python -m rcs_ac_first.scripts.smoke_check --config configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml --device cpu`: passed.

Code/setup changes made before the run:

- Raised `client_timeout` from 30s to 120s for the lateral-recovery collection config.
- Fixed `smoke_check` checkpoint writes to use a config/pid-specific path. Running two smoke checks in parallel had corrupted the shared `checkpoints/smoke_check/last.pt`.

Dataset:

```text
file: data/expert/rgb_perception_actor_features_v2_masked_recovery_256.npz
accepted: 256 / 1871 attempted
acceptance_rate: 0.1368
route_progress: min 0.2564, mean 0.2995, max 0.3590
mask_available_mean: 1.0
road_mask_mean: 0.2771
lane_edge_mean: 0.0152
steer_target_range: -0.4235 to 0.4432
```

BC pretrain:

```text
run: rgb_perception_actor_features_v2_masked_recovery_pretrain_bc
checkpoint: checkpoints/rgb_perception_actor_features_v2_masked_recovery_pretrain_bc/last.pt
epoch 3 val_loss: 0.026519
epoch 3 val perception_aux_loss: 0.326835
epoch 3 val aux_mask_available_rate: 1.0
epoch 3 val visual_actor_delta_l1_mean: 0.038602
```

Pretrain evaluation without lateral recovery:

```text
run: rgb_perception_actor_features_v2_masked_recovery_pretrain_eval_512_retry_20260501T033921
completed_steps: 512
done_reason_counts: offroad 8
max_route_progress: 0.0769
mean_route_progress: 0.0345
final_route_progress: 0.0256
```

Low-std PPO probe:

```text
run: rgb_perception_actor_features_v2_masked_recovery_lowstd_2048_retry_20260501T034330
checkpoint: checkpoints/rgb_perception_actor_features_v2_masked_recovery_lowstd_2048_retry_20260501T034330/last.pt
episodes: 32
done_reason: all offroad
episode route_progress range: 0.0513-0.0769
final update total_steps: 2048
final update critic_loss: 4.610916
final update visual_actor_delta_l1_mean: 0.052218
```

Final PPO checkpoint evaluation:

```text
run: rgb_perception_actor_features_v2_masked_recovery_lowstd_2048_eval_512_retry2_20260501T035017
completed_steps: 512
done_reason_counts: offroad 8
max_route_progress: 0.0769
mean_route_progress: 0.0353
final_route_progress: 0.0769
```

Pulled logs:

```text
/Users/song-yonghwi/code/windows_results/rcs-ac-first/experiments/rgb_perception_actor_features_v2_masked_recovery_round14
```

## Conclusion

The recovery-focused masked dataset was valid, and the RGB perception actor adapter was trained with real semantic mask supervision. However, this direction regressed the no-lateral rollout relative to Round 13: max route progress fell from roughly 0.179 to 0.077, and PPO did not recover from that basin.

The likely failure mode is distribution mismatch. Samples collected only from route progress 0.24-0.38 teach recovery behavior, but they do not preserve the early route behavior needed to reach that window when lateral recovery is disabled. The visual adapter becomes active, but the policy collapses into early offroad before the recovery knowledge can help.

Recommended next round:

1. Do not continue this pure recovery-only dataset path.
2. Build a mixed masked dataset: mostly clean early-route samples plus a smaller recovery slice, with balanced sampling or explicit loss weights.
3. Add a BC validation gate split by route window: early route 0.00-0.12, transition 0.12-0.24, recovery 0.24-0.38.
4. Only launch PPO if the no-lateral pretrain eval reaches at least the Round 13 max route progress baseline.

Operational note:

CARLA exited between some scripts. Restarting it with `scripts/windows/start_carla_interactive.ps1` before the next command fixed the issue. The operating guide now calls out this re-check/restart step explicitly.
