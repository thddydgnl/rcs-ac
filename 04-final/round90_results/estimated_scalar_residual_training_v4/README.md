# Round90 Estimated Scalar Residual Training v4

This run keeps the scalar imitation objective unchanged and improves only the
estimated scalar path.

Allowed changes used here:

- added estimated-scalar-only visual geometry features
- increased the estimated scalar residual head to `hidden_dim=256`
- selected the best checkpoint on a held-out validation split

Unchanged:

- Round90 BC/action loss
- PPO reward/loss
- action targets
- runtime steering logic
- scalar imitation objective

The objective remains:

- `SmoothL1(estimated_lane_offset, lane_offset)`
- `SmoothL1(estimated_heading_error, heading_error)`
- no lane/heading side weighting
- no sign loss
- no early-severe weighting
- no oracle actor action distillation

## Feature Change

The original compact visual actor feature path still uses the existing 14 raw
features, preserving Round90 checkpoint compatibility.

Only the estimated scalar head now receives an extended 28-dimensional visual
geometry vector:

- original road/lane mask summary features
- far/mid/near road center x
- far/mid/near lane center x
- road and lane near-far center slope proxies
- far/mid/near road and lane confidence

These features give the residual head more direct evidence for lane position
and heading while leaving the actor, BC losses, and PPO path unchanged.

## Setup

- Source config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Source checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Samples: 4,256
- Train/val split: 80/20
- Epochs: 160
- Best epoch: 160
- Learning rate: `7.5e-4`
- Weight decay: `1e-4`
- Output checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v4/last.pt`

The output checkpoint is local only because `checkpoints/` is ignored by git.

## Difference From Existing Scalar Targets

| Metric | Before residual | v1 residual | v2 residual | v4 residual |
| --- | ---: | ---: | ---: | ---: |
| lane MAE | 0.9448 | 0.1305 | 0.0755 | 0.0554 |
| heading MAE | 7.5204 | 2.6155 | 1.7278 | 1.2009 |
| lane sign agreement | 0.7066 | 0.9788 | 0.9845 | 0.9954 |
| heading sign agreement | 0.4468 | 0.8567 | 0.9011 | 0.9215 |
| early-severe lane sign | 0.7025 | 0.9771 | 1.0000 | 1.0000 |
| early-severe heading sign | 0.4365 | 0.8563 | 1.0000 | 1.0000 |
| left lane sign | 0.8644 | 1.0000 | 0.9849 | 1.0000 |
| right lane sign | 0.1367 | 1.0000 | 0.9832 | 0.9888 |

Validation split at the best epoch:

- lane MAE: 0.0584
- heading MAE: 1.3202
- lane sign agreement: 0.9935
- heading sign agreement: 0.9213

## Actor Action Delta After v4

Compared with the original oracle-scalar Round90 actor path:

- steer absolute mean delta: 0.0108
- steer absolute p90 delta: 0.0239
- steer absolute max delta: 0.2416
- longitudinal absolute mean delta: 0.0038
- longitudinal absolute p90 delta: 0.0084
- longitudinal absolute max delta: 0.1435

## Interpretation

v4 is the closest estimated scalar path so far while staying inside the
requested constraints. It does not use actor action distillation and does not
change Round90's BC/action/PPO losses.

The remaining gap is mostly heading error. The all-sample heading MAE is now
about `1.20`, with validation heading MAE about `1.32`. This is much closer
than v2 but still not mathematically identical to the old privileged scalar
values, so CARLA validation is still required before treating it as a
behavior-preserving replacement.
