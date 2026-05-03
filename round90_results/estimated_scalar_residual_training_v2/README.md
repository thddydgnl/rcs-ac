# Round90 Estimated Scalar Residual Training v2

This run trains only the estimated scalar residual head so that
`estimated_lane_offset` and `estimated_heading_error` match the existing
privileged `lane_offset` and `heading_error` targets more closely.

The objective intentionally stays unbiased:

- `SmoothL1(estimated_lane_offset, lane_offset)`
- `SmoothL1(estimated_heading_error, heading_error)`
- no side weighting
- no early-severe weighting
- no sign-specific loss
- no runtime steering rule or lane-offset steering override

Compared with the first residual experiment, this run uses a larger residual
head (`estimated_scalar_hidden_dim=128`) and selects the best checkpoint on a
held-out validation split.

## Setup

- Source config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Source checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Samples: 4,256
- Train/val split: 80/20
- Epochs: 80
- Best epoch: 79
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Output checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v2/last.pt`

The output checkpoint is local only because `checkpoints/` is ignored by git.

## Difference From Existing Scalar Targets

| Metric | Before residual | v1 residual | v2 residual |
| --- | ---: | ---: | ---: |
| lane MAE | 0.9448 | 0.1305 | 0.0755 |
| heading MAE | 7.5204 | 2.6155 | 1.7278 |
| lane sign agreement | 0.7066 | 0.9788 | 0.9845 |
| heading sign agreement | 0.4468 | 0.8567 | 0.9011 |
| early-severe lane sign | 0.7025 | 0.9771 | 1.0000 |
| early-severe heading sign | 0.4365 | 0.8563 | 1.0000 |
| left lane sign | 0.8644 | 1.0000 | 0.9849 |
| right lane sign | 0.1367 | 1.0000 | 0.9832 |

Validation split at the best epoch:

- lane MAE: 0.0783
- heading MAE: 1.8068
- lane sign agreement: 0.9768
- heading sign agreement: 0.9058

The validation numbers are close to the all-sample numbers, so the improved
match is not only memorizing the full dataset.

## Actor Action Delta After v2

Compared with the original oracle-scalar Round90 actor path:

- steer absolute mean delta: 0.0128
- steer absolute p90 delta: 0.0319
- steer absolute max delta: 0.2197
- longitudinal absolute mean delta: 0.0050
- longitudinal absolute p90 delta: 0.0116
- longitudinal absolute max delta: 0.1443

## Interpretation

The v2 residual head is much closer to the existing `lane_offset` and
`heading_error` targets than both the untrained bootstrap and the v1 residual
run, without changing the scalar matching loss into a sign-biased or
early-severe-biased objective.

This is still offline privileged supervision. It shows that the RGB/route
estimated scalar path can imitate the old scalar targets on the Round36 mix
dataset, but it does not yet prove clean CARLA stability or robustness under
weather/sensor corruption.
