# Round78 Preemptive Late Recovery

## Intent

Round77 passed 512 clean safety but still had one 2048 offroad. The offroad window showed correct recovery direction, but the recovery became strong only after lane offset had already grown to about `-2.8`.

Round78 tested a config-only change to start the late recovery objective earlier in the same route window. This is privileged offline supervision for BC loss masking only, not a runtime steering override.

## Config

Config: `configs/bc_frame_rgb_perception_actor_features_v2_preemptive_late_recovery_round78.yaml`

Base: Round77 / Round63 checkpoint.

Key changes from Round77:

- `late_recovery_abs_lane_offset_min: 1.25`
- `late_recovery_abs_heading_error_min: 8.0`
- `early_severe_longitudinal_ceiling: -0.05`
- kept `late_recovery_absolute_steer_margin: 0.60`
- kept `late_recovery_bc_loss_scale: 0.20`

## Verification

Local checks:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`
- `python -m pytest tests/test_audit_actor_sign_response.py tests/test_audit_corrective_delta_dataset.py -q`
- Result: 9 passed

Windows BC training:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt`
- Run: `rgb_perception_actor_features_v2_preemptive_late_recovery_bc_round78`
- Result: `val_loss=0.044770`

Training metrics:

- val `bc_sample_weight_mean=0.913615`
- val `bc_sample_weight_min=0.519249`
- val `bc_sample_weight_late_mask_rate=0.107981`
- val `late_recovery_absolute_steer_sign_rate=0.600939`
- val `early_severe_steer_sign_rate=0.996995`

Offline actor sign audits:

- `early_any`, max 16: PASS
  - all axes rate: `1.0`
  - min abs delta: `0.048092`
  - max abs zero steer: `0.026600`
- `early_left`, max 16, max zero bias 1.0: PASS
  - all axes rate: `1.0`
  - min abs delta: `0.048092`
  - max abs zero steer: `0.026600`

The neutral zero-bias gate still passes but is closer to the `0.03` limit than Round77.

## CARLA Results

512-step clean eval:

- Run: `rgb_perception_actor_features_v2_preemptive_late_recovery_eval_512_round78`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- final/max route progress: `0.487179`
- mean action jitter L1: `0.119345`
- clean safety: PASS
- PPO gate: BLOCK

2048-step clean eval:

- Run: `rgb_perception_actor_features_v2_preemptive_late_recovery_eval_2048_round78`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- max route progress: `0.512821`
- final route progress: `0.487179`
- corrective sign rate: `0.791367`
- visual center correction rate: `0.460432`
- mean action jitter L1: `0.118379`
- clean safety: PASS
- PPO gate: BLOCK due jitter and visual-center correction metric

## Recovery Window Sample

Round78 avoided the Round77 offroad by braking and steering earlier:

| step | route | lane_offset | heading_error | steer | longitudinal | done |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 800 | 0.282051 | -0.617 | 12.152 | 0.218 | 0.083 | running |
| 810 | 0.307692 | -1.120 | 21.362 | 0.673 | -0.924 | running |
| 820 | 0.307692 | -1.266 | 35.917 | 0.062 | -0.700 | running |
| 828 | 0.307692 | -1.681 | 29.590 | 0.727 | -0.910 | running |

The earlier mask moved recovery before lane offset reached the Round77 offroad range, but it also produced heavy braking and lower route progress in several episodes.

## Decision

Round78 is the first recent round to satisfy the strict BC clean safety target for both 512 and 2048:

- 512: collision/offroad/severe early offroad/stuck all zero
- 2048: collision/offroad/severe early offroad/stuck all zero

This does not complete the full `/goal` yet. PPO remains blocked because the evaluator reports high action jitter and low visual-center correction rate, and the policy appears to rely on heavy braking during late recovery. The next step should either:

- reduce excessive late braking while preserving the 2048 clean result, or
- run a very conservative PPO-readiness audit before any PPO fine-tuning.
