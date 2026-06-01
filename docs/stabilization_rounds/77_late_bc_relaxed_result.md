# Round77 Late Recovery BC Relaxation

## Intent

Round75 improved closed-loop safety to 512 steps but still failed 2048 with one late offroad near route progress 0.33. The failure window showed correct steering direction, but the actor started strong recovery too late while still carrying speed. Round76 pushed late recovery harder, but damaged neutral early steering.

Round77 tested an offline-only training objective change: reduce the main BC imitation weight only inside the late recovery mask so the late absolute steer and braking losses are not muted by weak expert labels. This uses privileged offline supervision from route progress, lane offset, and heading error for loss masking only. It is not a runtime steering rule or lane-offset controller.

## Code Change

- Added `late_recovery_bc_loss_scale` to `train_bc.py`.
- The option applies per-sample BC weights on the late recovery mask:
  - normal samples keep weight `1.0`
  - late recovery samples use the configured scale
- Added log fields:
  - `bc_sample_weight_mean`
  - `bc_sample_weight_min`
  - `bc_sample_weight_late_mask_rate`
- Kept runtime inference unchanged.

## Config

Config: `configs/bc_frame_rgb_perception_actor_features_v2_late_bc_relaxed_round77.yaml`

Base: Round75 from the Round63 checkpoint.

Key changes:

- `late_recovery_absolute_steer_loss_weight: 0.75`
- `late_recovery_absolute_steer_margin: 0.60`
- `late_recovery_bc_loss_scale: 0.20`

## Verification

Local checks:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`
- `python -m pytest tests/test_audit_actor_sign_response.py tests/test_audit_corrective_delta_dataset.py -q`
- Result: 9 passed

Windows BC training:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt`
- Run: `rgb_perception_actor_features_v2_late_bc_relaxed_bc_round77`
- Result: `val_loss=0.036460`

Training log confirms the new mask weighting was active:

- val `bc_sample_weight_mean=0.941784`
- val `bc_sample_weight_min=0.519249`
- val `bc_sample_weight_late_mask_rate=0.072770`
- val `late_recovery_absolute_steer_sign_rate=0.600939`
- val `early_severe_steer_sign_rate=0.994405`

Offline actor sign audits:

- `early_any`, max 16: PASS
  - all axes rate: `1.0`
  - min abs delta: `0.054672`
  - max abs zero steer: `0.015894`
- `early_left`, max 16, max zero bias 1.0: PASS
  - all axes rate: `1.0`
  - min abs delta: `0.054672`
  - max abs zero steer: `0.015894`

## CARLA Results

512-step clean eval:

- Run: `rgb_perception_actor_features_v2_late_bc_relaxed_eval_512_round77`
- collision: `0`
- offroad: `0`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- final/max route progress: `0.512821`
- clean safety: PASS
- PPO gate: BLOCK due jitter and corrective metrics

2048-step clean eval:

- Run: `rgb_perception_actor_features_v2_late_bc_relaxed_eval_2048_round77`
- collision: `0`
- offroad: `1`
- early offroad: `0`
- severe early offroad: `0`
- stuck: `0`
- max route progress: `0.512821`
- corrective sign rate: `0.895105`
- visual center correction rate: `0.489510`
- clean safety: BLOCK
- PPO gate: BLOCK

## Failure Window

The 2048 offroad occurred at global step 828, episode step 316, route progress 0.333333.

Recent window:

| step | route | lane_offset | heading_error | steer | longitudinal | done |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 816 | 0.307692 | -1.781 | 46.439 | 0.233 | 0.186 | running |
| 818 | 0.307692 | -2.039 | 43.932 | 0.402 | 0.223 | running |
| 820 | 0.307692 | -2.264 | 40.472 | 0.445 | 0.180 | running |
| 822 | 0.333333 | -2.458 | 36.644 | 0.454 | 0.174 | running |
| 824 | 0.333333 | -2.623 | 32.607 | 0.522 | 0.095 | running |
| 825 | 0.333333 | -2.689 | 30.379 | 0.599 | -0.085 | running |
| 826 | 0.333333 | -2.740 | 27.937 | 0.686 | -0.412 | running |
| 827 | 0.333333 | -2.775 | 25.300 | 0.733 | -0.706 | running |
| 828 | 0.333333 | -2.794 | 22.556 | 0.748 | -0.830 | offroad |

The sign is correct and the actor eventually reaches strong rightward recovery plus braking, but the response is still too late for the accumulated lateral error. This is better than a wrong-sign bug, but it is not yet enough for the 2048 clean target.

## Decision

Round77 does not meet the active stabilization goal because 2048 clean eval still has `offroad=1`. PPO remains blocked.

Next round should not simply raise late recovery strength globally. The more likely fix is to start recovery earlier in the route 0.30-0.34 window or train a preemptive late-window response before lane offset exceeds about 1.8, while preserving the neutral zero-bias gate.
