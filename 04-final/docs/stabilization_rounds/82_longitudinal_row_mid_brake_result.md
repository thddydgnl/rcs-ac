# Round82 Longitudinal Row Mid Brake

## Intent

Round81 proved that actor output-row isolation can preserve steering, but `early_severe_longitudinal_ceiling: 0.0` delayed braking too much and caused one 2048-step offroad.

Round82 kept the same row-isolated training scope and tested a midpoint longitudinal ceiling:

- initialize from Round78
- train only `recurrent_actor_critic.actor_head.2`
- restore frozen steering row `0` after every optimizer step
- train only longitudinal row `1`
- set `early_severe_longitudinal_ceiling: -0.025`

This is offline BC supervision and trainable-scope isolation only. It does not add runtime steering or braking rules.

## Config

Config: `configs/bc_frame_rgb_perception_actor_features_v2_longitudinal_row_mid_brake_round82.yaml`

Key settings:

- `trainable_module_names: ["recurrent_actor_critic.actor_head.2"]`
- `actor_head_output_trainable_indices: [1]`
- `steer_loss_weight: 0.0`
- `longitudinal_loss_weight: 1.0`
- steering-specific auxiliary losses disabled
- `early_severe_longitudinal_ceiling: -0.025`

## Verification

Local checks:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`
- `python -m pytest tests/test_audit_actor_sign_response.py tests/test_audit_corrective_delta_dataset.py -q`
- Result: 9 passed

Windows BC training:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_preemptive_late_recovery_bc_round78/last.pt`
- Run: `rgb_perception_actor_features_v2_longitudinal_row_mid_brake_bc_round82`
- Result: `val_loss=0.111021`

Offline actor sign audits:

- `early_any`: PASS
  - all axes rate: `1.0`
  - min abs delta: `0.019830`
  - max abs zero steer: `0.015059`
- `early_left`: PASS with relaxed zero-bias gate
  - all axes rate: `1.0`
  - min abs delta: `0.020008`

## CARLA Results

512-step clean eval:

- Run: `rgb_perception_actor_features_v2_longitudinal_row_mid_brake_eval_512_round82`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- final/max route progress: `0.538462`
- mean action jitter L1: `0.138450`
- PPO gate: BLOCK

2048-step clean eval:

- First attempt failed before rollout with CARLA `reset()` timeout and was not counted as a model result.
- Retry run: `rgb_perception_actor_features_v2_longitudinal_row_mid_brake_eval_2048_retry_round82`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- final/max route progress: `0.564103`
- corrective sign rate: `0.827068`
- visual center correction rate: `0.421053`
- mean action jitter L1: `0.145582`
- PPO gate: BLOCK due action jitter and visual-center correction rate

## Decision

Round82 is a better clean-driving candidate than Round81:

- restores 2048 clean safety to zero offroad
- improves max route progress over Round78 (`0.564103` vs `0.512821`)
- preserves offline steering sign response and zero-bias gate

Round82 still does not complete the `/goal`:

- PPO gate remains BLOCK
- action jitter is about `0.146`, above the current `0.07` gate
- visual-center correction rate remains too low for PPO readiness

Current best candidates:

- safest proven BC baseline: Round78
- best route-progress clean BC candidate: Round82

Next direction:

- keep Round82 as the new candidate for clean 512/2048 safety
- do not start PPO while PPO gate is BLOCK
- reduce jitter without adding runtime smoothing as a hidden performance crutch
- investigate whether the PPO gate should use separate steer/longitudinal jitter thresholds, because Round82's steering jitter is near `0.067` while longitudinal jitter is `0.078`
- if training changes are used, prefer a mild action-smooth BC objective or final-row regularization that preserves the 2048 zero-offroad result
