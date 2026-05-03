# Round90 Estimated Scalar Residual Training v5

This run keeps the requested constraints:

- no Round90 BC/action loss change
- no PPO reward/loss change
- no action target change
- no runtime steering override
- no oracle actor action distillation
- no side/sign/early-severe scalar loss weighting

The scalar imitation objective remains:

- `SmoothL1(estimated_lane_offset, lane_offset)`
- `SmoothL1(estimated_heading_error, heading_error)`

## What Changed

v5 combines the two allowed improvement directions:

1. Richer visual geometry features for the estimated scalar path only.
2. A stronger residual head architecture.

The compact RGB actor feature path still uses the original 14 raw features, so
Round90 checkpoint compatibility is preserved for the actor-side visual
adapter. Only `EstimatedScalarHead` now receives a 64-dimensional visual
geometry vector.

The 64-dimensional vector contains:

- original road/lane summary features
- far/mid/near road and lane center x
- far/mid/near left/right edge proxy x
- far/mid/near width proxy
- far/mid/near confidence for full/left/right bands
- road and lane far-mid, mid-near, far-near slope proxies
- simple curvature proxies from slope differences

The residual head now projects visual geometry and route features separately,
then fuses them with a LayerNorm/SiLU MLP. The final residual output is still
zero-initialized so old checkpoints start from the trained aux-head baseline.

## Setup

- Source config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Source checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Samples: 4,256
- Train/val split: 80/20
- Epochs: 240
- Best epoch: 240
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Hidden dim: 384
- Output checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v5/last.pt`

The output checkpoint is local only because `checkpoints/` is ignored by git.

## Difference From Existing Scalar Targets

| Metric | Before residual | v2 residual | v4 residual | v5 residual |
| --- | ---: | ---: | ---: | ---: |
| lane MAE | 0.9448 | 0.0755 | 0.0554 | 0.0604 |
| heading MAE | 7.5204 | 1.7278 | 1.2009 | 0.4123 |
| lane sign agreement | 0.7066 | 0.9845 | 0.9954 | 0.9889 |
| heading sign agreement | 0.4468 | 0.9011 | 0.9215 | 0.9613 |
| early-severe lane sign | 0.7025 | 1.0000 | 1.0000 | 1.0000 |
| early-severe heading sign | 0.4365 | 1.0000 | 1.0000 | 1.0000 |
| left lane sign | 0.8644 | 0.9849 | 0.9940 | 0.9866 |
| right lane sign | 0.1367 | 0.9832 | 1.0000 | 0.9966 |

Validation split at the best epoch:

- lane MAE: 0.0652
- heading MAE: 0.4657
- lane sign agreement: 0.9871
- heading sign agreement: 0.9587

## Actor Action Delta After v5

Compared with the original oracle-scalar Round90 actor path:

| Metric | v4 | v5 |
| --- | ---: | ---: |
| steer absolute mean delta | 0.0108 | 0.0092 |
| steer absolute p90 delta | 0.0239 | 0.0145 |
| steer absolute max delta | 0.2416 | 0.3618 |
| longitudinal absolute mean delta | 0.0038 | 0.0024 |
| longitudinal absolute p90 delta | 0.0084 | 0.0046 |
| longitudinal absolute max delta | 0.1435 | 0.2413 |

## Interpretation

v5 is substantially closer on heading than v4 while staying within the allowed
scope. Average heading error dropped from `1.2009` to `0.4123`, and validation
heading error dropped from `1.3202` to `0.4657`.

Lane MAE is slightly worse than v4 (`0.0604` vs `0.0554`) but still much better
than v2. Actor mean and p90 deltas improved, but max deltas worsened on a small
number of samples. This means v5 is a better average replacement candidate, but
CARLA clean eval is still required before calling it behavior-equivalent to
Round90.
