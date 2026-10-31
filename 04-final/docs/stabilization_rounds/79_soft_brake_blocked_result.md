# Round79 Soft Brake Probe

## Intent

Round78 was the first recent round to pass both 512 and 2048 clean safety, but it relied on heavy braking in the late recovery window and still had PPO gate issues. Round79 tested whether braking could be softened while preserving the Round78 clean behavior.

This was a config-only probe. It did not add runtime steering or braking rules.

## Config

Config: `configs/bc_frame_rgb_perception_actor_features_v2_preemptive_late_recovery_soft_brake_round79.yaml`

Base: Round78 / Round63 checkpoint.

Key changes from Round78:

- `early_severe_longitudinal_ceiling: 0.0`
- `late_recovery_abs_lane_offset_min: 1.35`
- `late_recovery_abs_heading_error_min: 9.0`

## Verification

Local checks:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`
- `python -m pytest tests/test_audit_actor_sign_response.py tests/test_audit_corrective_delta_dataset.py -q`
- Result: 9 passed

Windows BC training:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt`
- Run: `rgb_perception_actor_features_v2_preemptive_late_recovery_soft_brake_bc_round79`
- Result: `val_loss=0.039879`

Training metrics:

- val `bc_sample_weight_mean=0.926761`
- val `bc_sample_weight_late_mask_rate=0.091549`
- val `late_recovery_absolute_steer_loss=0.034127`
- val `late_recovery_absolute_steer_sign_rate=0.600939`
- val `early_severe_steer_sign_rate=0.996995`

## Offline Gate

`early_any`, max 16:

- all axes rate: `1.0`
- min abs delta: `0.049374`
- max abs zero steer: `0.031148`
- pass status: BLOCK

`early_left`, max 16, max zero bias 1.0:

- all axes rate: `1.0`
- min abs delta: `0.049374`
- max abs zero steer: `0.031148`
- pass status: PASS

## Decision

Round79 was not promoted to CARLA evaluation. It missed the active neutral zero-bias gate:

- target: `early-left/neutral zero steer bias abs <= 0.03`
- Round79: `0.031148`

This suggests the softer-brake/later-mask change reduces the safety margin that kept Round78 neutral enough. Round78 remains the current best clean-safety candidate despite heavy braking.

Next work should start from Round78, not Round79. The safer direction is to keep the preemptive mask but add a neutral-bias preservation constraint or reduce braking through a narrower longitudinal-only adjustment that does not shift zero steer.
