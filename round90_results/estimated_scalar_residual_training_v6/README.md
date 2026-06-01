# Round90 Estimated Scalar Residual Training v6

This run changes only the offline estimated scalar head training objective.
RCS-AC BC/action/PPO losses, action targets, runtime control, and reward logic
are unchanged.

The goal is still to reproduce the existing privileged scalar targets:

- `lane_offset`
- `heading_error`

No oracle actor action distillation, sign loss, side-specific loss, or
early-severe weighting is used.

## Loss

v6 uses a normalized scalar imitation loss:

```text
SmoothL1((estimated_lane_offset - lane_offset) / train_lane_std)
+ SmoothL1((estimated_heading_error - heading_error) / train_heading_std)
```

Best checkpoint selection also uses the normalized validation MAE:

```text
val_lane_mae / train_lane_std + val_heading_mae / train_heading_std
```

The train-set scales were:

- `train_lane_std`: 0.9584
- `train_heading_std`: 11.6854

## Setup

- Source config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Source checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Samples: 4,256
- Train/val split: 80/20
- Epochs: 240
- Best epoch: 215
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Hidden dim: 384
- Visual feature dim: 64
- Output checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v6/last.pt`

The output checkpoint is local only because `checkpoints/` is ignored by git.

## Difference From Existing Scalar Targets

| Metric | v5 residual | v6 residual |
| --- | ---: | ---: |
| lane MAE | 0.0604 | 0.0337 |
| heading MAE | 0.4123 | 0.5097 |
| validation lane MAE | 0.0652 | 0.0384 |
| validation heading MAE | 0.4657 | 0.5371 |
| lane sign agreement | 0.9889 | 0.9930 |
| heading sign agreement | 0.9613 | 0.9427 |
| early-severe lane sign | 1.0000 | 1.0000 |
| early-severe heading sign | 1.0000 | 1.0000 |
| left lane sign | 0.9866 | 0.9923 |
| right lane sign | 0.9966 | 0.9955 |

## Actor Action Delta

Compared with the original oracle-scalar Round90 actor path:

| Metric | v5 | v6 |
| --- | ---: | ---: |
| steer absolute mean delta | 0.0092 | 0.0086 |
| steer absolute p90 delta | 0.0145 | 0.0155 |
| steer absolute max delta | 0.3618 | 0.3191 |
| longitudinal absolute mean delta | 0.0024 | 0.0023 |
| longitudinal absolute p90 delta | 0.0046 | 0.0045 |
| longitudinal absolute max delta | 0.2413 | 0.2334 |

## Interpretation

v6 is more balanced than v5. It substantially improves lane matching and
slightly improves mean/max action deltas, but it gives back some heading
accuracy. If the priority is lowest heading error, v5 remains better. If the
priority is normalized scalar similarity and lower average actor change, v6 is
the better candidate.

Both v5 and v6 still need CARLA clean eval before either can be treated as a
behavior-preserving replacement for Round90 oracle scalar inputs.
