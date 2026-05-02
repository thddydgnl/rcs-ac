# Round80 Neutral Hold Soft Brake Probe

## Intent

Round78 passed both 512 and 2048 clean safety, but used heavy braking. Round79 tried to soften braking and move the late mask slightly later, but failed the neutral zero-bias gate. Round80 isolated the braking change more carefully:

- keep the Round78 preemptive late recovery mask
- soften only the longitudinal ceiling
- strengthen the early-center zero-steer loss

This was a config-only offline-supervision probe. It did not add runtime control rules.

## Config

Config: `configs/bc_frame_rgb_perception_actor_features_v2_preemptive_late_neutral_hold_round80.yaml`

Base: Round78 / Round63 checkpoint.

Key changes from Round78:

- `early_severe_longitudinal_ceiling: 0.0`
- `early_center_zero_steer_loss_weight: 1.0`
- kept `late_recovery_abs_lane_offset_min: 1.25`
- kept `late_recovery_abs_heading_error_min: 8.0`

## Verification

Local checks:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`
- `python -m pytest tests/test_audit_actor_sign_response.py tests/test_audit_corrective_delta_dataset.py -q`
- Result: 9 passed

Windows BC training:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt`
- Run: `rgb_perception_actor_features_v2_preemptive_late_neutral_hold_bc_round80`
- Result: `val_loss=0.041421`

Training metrics:

- val `bc_sample_weight_mean=0.913615`
- val `bc_sample_weight_late_mask_rate=0.107981`
- val `late_recovery_absolute_steer_loss=0.059556`
- val `early_center_zero_steer_abs_mean=0.003310`
- val `early_severe_longitudinal_ceiling_rate=0.712424`
- val `early_severe_steer_sign_rate=0.996995`

## Offline Gate

`early_any`, max 16:

- all axes rate: `1.0`
- min abs delta: `0.049179`
- max abs zero steer: `0.040754`
- pass status: BLOCK

`early_left`, max 16, max zero bias 1.0:

- all axes rate: `1.0`
- min abs delta: `0.049179`
- max abs zero steer: `0.040754`
- pass status: PASS

## Decision

Round80 was not promoted to CARLA evaluation. It failed the neutral zero-bias gate by a larger margin than Round79:

- target: `early-left/neutral zero steer bias abs <= 0.03`
- Round79: `0.031148`
- Round80: `0.040754`

Increasing early-center zero-steer loss did not preserve the audit zero point. The likely issue is not simply too little center loss; the longitudinal softening and late recovery objective are coupled through the shared actor adapter/recurrent actor head and shift neutral steer.

Round78 remains the current best candidate:

- offline zero bias inside gate
- 512 clean safety PASS
- 2048 clean safety PASS
- PPO still blocked due jitter/visual-center metrics and heavy braking

Next direction should avoid changing the shared steering objective while reducing braking. A safer next probe would isolate longitudinal parameters or use a separate longitudinal-only loss/head path if available, rather than trying to soften braking through the same scalar actor update.
