# Round81 Longitudinal Row Tune

## Intent

Round78 passed 512/2048 clean safety but relied on heavy late braking. Round79 and Round80 showed that softening the longitudinal objective through the shared actor update shifted steering neutral and failed the offline zero-bias gate.

Round81 tested a narrower training-scope change:

- initialize from the Round78 checkpoint
- train only `recurrent_actor_critic.actor_head.2`
- preserve actor output row `0` (steering) after every optimizer step
- allow only actor output row `1` (longitudinal) to change

This is a BC training-scope isolation experiment, not a runtime steering/braking rule.

## Code Change

`train_bc.py` now supports:

- `bc.actor_head_output_trainable_indices`

When this option is set, frozen action rows in the final actor output layer are restored after each optimizer step. This prevents optimizer updates and weight decay from silently moving the frozen steering row.

Round81 config uses:

- `trainable_module_names: ["recurrent_actor_critic.actor_head.2"]`
- `actor_head_output_trainable_indices: [1]`
- `weight_decay: 0.0`
- `steer_loss_weight: 0.0`
- `longitudinal_loss_weight: 1.0`
- steering-specific auxiliary losses disabled
- `early_severe_longitudinal_ceiling: 0.0`

## Verification

Local checks:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`
- `python -m pytest tests/test_audit_actor_sign_response.py tests/test_audit_corrective_delta_dataset.py -q`
- Result: 9 passed

Windows BC training:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_longitudinal_row_tune_round81.yaml`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_preemptive_late_recovery_bc_round78/last.pt`
- Run: `rgb_perception_actor_features_v2_longitudinal_row_tune_bc_round81`
- Result: `val_loss=0.107777`

Offline actor sign audits:

- `early_any`: PASS
  - all axes rate: `1.0`
  - min abs delta: `0.019830`
  - max abs zero steer: `0.015059`
- `early_left`: PASS with relaxed zero-bias gate
  - all axes rate: `1.0`
  - min abs delta: `0.020008`

Compared with Round78 `early_any` zero-bias `0.026600`, the steering gate improved after row isolation.

## CARLA Results

512-step clean eval:

- Run: `rgb_perception_actor_features_v2_longitudinal_row_tune_eval_512_round81`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- final/max route progress: `0.538462`
- corrective sign rate: `1.0`
- visual center correction rate: `1.0`
- mean action jitter L1: `0.135167`
- PPO gate: BLOCK due action jitter and corrective count

2048-step clean eval:

- First attempt failed before rollout because CARLA timed out during reset. It was not counted as a model result.
- Retry run: `rgb_perception_actor_features_v2_longitudinal_row_tune_eval_2048_retry_round81`
- collision: `0`
- offroad: `1`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- max route progress: `0.564103`
- final route progress: `0.205128`
- corrective sign rate: `0.834711`
- visual center correction rate: `0.528926`
- mean action jitter L1: `0.152791`
- PPO gate: BLOCK

## Offroad Window

The offroad happened in the same late route region that Round78 had stabilized:

| step | episode_step | route | lane_offset | heading_error | steer | longitudinal | done_reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 820 | 308 | 0.307692 | -1.596 | 14.025 | 0.373 | 0.293 | running |
| 822 | 310 | 0.307692 | -2.011 | 41.542 | 0.318 | 0.252 | running |
| 824 | 312 | 0.307692 | -2.256 | 37.994 | 0.484 | 0.308 | running |
| 826 | 314 | 0.307692 | -2.443 | 33.395 | 0.615 | -0.075 | running |
| 828 | 316 | 0.333333 | -2.561 | 28.003 | 0.707 | -0.617 | running |
| 830 | 318 | 0.333333 | -2.613 | 22.266 | 0.741 | -0.878 | running |
| 832 | 320 | 0.333333 | -2.616 | 16.701 | 0.731 | -0.930 | offroad |

Round81 delayed braking too much: the policy was still producing positive longitudinal action while lane offset was already around `-2.0` to `-2.3`. By the time strong braking appeared, the lane error was near the offroad threshold.

## Decision

Round81 is not promoted:

- 512 clean safety improved route progress
- offline steering gate improved
- but 2048 clean safety failed with one offroad

The experiment is still useful. It confirms the Round79/80 diagnosis: the shared actor update was part of the neutral-steering instability, and output-row isolation can preserve steering. However, longitudinal-only tuning from Round78 made late braking too weak too early, causing the same late offroad mode to return.

Next direction:

- keep actor output row isolation available
- do not use Round81 as the PPO seed
- return to Round78 as the safety baseline
- try a less aggressive longitudinal row tune, likely with `early_severe_longitudinal_ceiling` between `-0.05` and `0.0`, or a route/lane-window-specific longitudinal loss that preserves early braking once `abs(lane_offset) >= 1.5`
- keep 512 -> 2048 gating before any PPO
