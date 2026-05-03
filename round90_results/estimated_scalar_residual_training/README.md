# Round90 Estimated Scalar Residual Training

Date: 2026-05-03

Purpose: train the new `estimated_scalar_head` offline so `estimated_lane_offset` and `estimated_heading_error` move closer to the previous oracle `lane_offset` and `heading_error` targets.

## Inputs

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Sample count: `4256`

## Training Setup

- Trained module: `estimated_scalar_head` only
- Frozen modules: encoder, perception aux head, actor, critic, reliability, risk, scalar adapter
- Trainable parameters: `1922`
- Epochs: `30`
- Learning rate: `0.003`
- Loss: Smooth L1 for `estimated_lane_offset` and `estimated_heading_error`
- Supervision type: privileged offline supervision only

## Before vs After

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| lane offset MAE | `0.944770` | `0.130541` | `-0.814229` |
| heading error MAE | `7.520399` | `2.615464` | `-4.904934` |
| lane sign agreement, all | `0.706612` | `0.978822` | `+0.272211` |
| heading sign agreement, all | `0.446798` | `0.856663` | `+0.409866` |
| lane sign agreement, early-severe | `0.702461` | `0.977069` | `+0.274609` |
| heading sign agreement, early-severe | `0.436521` | `0.856264` | `+0.419743` |
| lane sign agreement, left | `0.864387` | `1.000000` | `+0.135613` |
| lane sign agreement, right | `0.136701` | `1.000000` | `+0.863299` |

## Interpretation

The residual head now predicts scalar lane/heading values much closer to the previous oracle targets on the Round90 training dataset.

Remaining gap:

- Heading error is still less accurate than lane offset.
- This is an offline fit on the Round90 dataset, not yet a CARLA clean eval pass.
- The next gate should be an offline actor sign audit using `scalar_actor_feature_source: estimated`, followed by 512/2048 clean eval only if the audit passes.

Outputs:

- `summary.json`
- `training_log.csv`
- checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90/last.pt`
