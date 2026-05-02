# Round 72 Semantic Recovery Low-LR Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round71 showed that strict-lane filtered balanced recovery data fixed dataset left/right sign rates, but overcorrected the raw actor zero-steer bias. Round72 tested whether using the semantic-filtered dataset and a lower BC learning rate would preserve the early zero-bias gate.

This is still offline privileged supervision. Runtime lateral recovery remains disabled in the train/eval config.

## Config

```text
configs/bc_frame_rgb_perception_actor_features_v2_semantic_recovery_lr_round72.yaml
```

Changes from Round71:

```text
learning_rate: 0.00005
dataset: data/expert/rgb_perception_actor_features_v2_balanced_recovery_semantic_filtered_round71.npz
lateral_recovery.enabled: false
```

## Dataset Gate

The semantic-filtered dataset passed semantic corrective-delta audit:

```text
semantic_gate_status: PASS
heading_aware_lane_dominant_alignment_rate: 1.0
heading_aware_heading_dominant_alignment_rate: 1.0
heading_aware_conflict_alignment_vs_heading_rate: 1.0
```

It did not pass strict lane split:

```text
strict_lane_left_sign_rate_vs_neg_lane: 0.898925
strict_lane_right_sign_rate_vs_neg_lane: 1.0
```

## BC

```text
init_checkpoint: checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt
val_loss: 0.011441
```

## Offline Actor Sign Audit

```text
early_left: BLOCK
  max_abs_zero_steer: 0.148717
  min_abs_delta: 0.021508

early_any: BLOCK
  max_abs_zero_steer: 0.148717
  min_abs_delta: 0.021508
```

Decision: do not run CARLA and do not start PPO.

## Interpretation

Lowering the learning rate and using semantic-filtered data did not protect the early zero-steer bias. The actor became even more biased than Round71. This suggests that the issue is not only strict-vs-semantic filtering; the full trainable scope over the mixed recovery dataset is too powerful and shifts the baseline steering response.

Next useful directions:

```text
1. Reduce trainable scope, for example scalar adapter only or a separate small correction head.
2. Reduce recovery row count sharply instead of training on the full filtered dataset.
3. Treat recovery rows as an auxiliary corrective-delta objective rather than changing the main expert_action distribution.
```

Round63 remains the best current CARLA candidate.
