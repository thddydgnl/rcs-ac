# Round 71 Balanced Recovery Dataset Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round70 produced a clean left-side recovery slice but no natural right-side slice. Round71 added collection-only steering perturbation support to generate a right-side recovery slice, then built a balanced recovery dataset and tested whether BC could learn earlier recovery without runtime steering overrides.

This remains privileged offline supervision. The collection config can use lateral recovery and collection-time perturbation, but the raw train/eval config keeps `lateral_recovery.enabled: false`.

## Tool Changes

`rcs_ac_first/scripts/collect_recovery_dataset.py` now supports collection-only perturbation:

```text
--collection-steer-bias
--collection-bias-route-min
--collection-bias-route-max
--collection-bias-abs-lane-offset-max
```

Accepted rows record:

```text
collection_steer_bias_applied
```

`rcs_ac_first/scripts/filter_corrective_delta_dataset.py` was added to create corrected datasets by dropping wrong-sign corrective-delta rows.

## Right-Side Collection

Dataset:

```text
data/expert/rgb_perception_actor_features_v2_recovery_right_biased_aligned_32_round71.npz
```

Collection settings:

```text
lane_offset_sign: right
require_corrective_sign_aligned: true
collection_steer_bias: 0.18
collection_bias_route_min: 0.05
collection_bias_route_max: 0.36
collection_bias_abs_lane_offset_max: 2.0
```

Summary:

```text
num_transitions: 32
accepted_transitions: 32
attempted_transitions: 1178
target_corrective_delta_sign_rate: 1.0
collection_steer_bias_applied_rate: 1.0
```

Audit:

```text
gate_status: PASS
right_count: 32
right_sign_rate_vs_neg_lane: 1.0
wrong_sign_count: 0
source_metadata_present: true
```

## Mixed Dataset

Dataset:

```text
data/expert/rgb_perception_actor_features_v2_balanced_recovery_mix_4320_round71.npz
```

Parts:

```text
base_round36: 4256
left_recovery_round70: 32
right_biased_recovery_round71: 32
```

The unfiltered mix failed strict lane audit because the base Round36 rows still contained wrong-sign left rows:

```text
strict_lane_left_sign_rate_vs_neg_lane: 0.903738
wrong_sign_count: 188
wrong_sign source_label: base_round36
```

## Filtered Datasets

Semantic filter:

```text
data/expert/rgb_perception_actor_features_v2_balanced_recovery_semantic_filtered_round71.npz
dropped_count: 107
semantic_gate_status: PASS
strict_lane_left_sign_rate_vs_neg_lane: 0.898925
```

Strict-lane filter:

```text
data/expert/rgb_perception_actor_features_v2_balanced_recovery_strict_filtered_round71.npz
dropped_count: 188
strict_lane_gate_status: PASS
strict_lane_left_sign_rate_vs_neg_lane: 1.0
strict_lane_right_sign_rate_vs_neg_lane: 1.0
wrong_sign_count: 0
semantic_gate_status: BLOCK
```

Round71 used the strict-lane filtered dataset because the current stabilization goal requires left/right corrective sign rates to pass separately.

## BC

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_balanced_recovery_mix_round71.yaml
```

Training:

```text
init_checkpoint: checkpoints/rgb_perception_actor_features_v2_early_strong_timing_bc_round63/last.pt
dataset: data/expert/rgb_perception_actor_features_v2_balanced_recovery_strict_filtered_round71.npz
val_loss: 0.011097
```

## Offline Actor Sign Audit

```text
early_left: BLOCK
  max_abs_zero_steer: 0.118910
  min_abs_delta: 0.021109

early_any: BLOCK
  max_abs_zero_steer: 0.118910
  min_abs_delta: 0.021109
```

Decision: do not run CARLA and do not start PPO.

## Interpretation

Round71 proved that a right-side recovery slice can be generated with collection-only perturbation and audited cleanly. It also confirmed that the base dataset still contains strict-lane wrong-sign left rows in the route `0.20-0.36` severe window.

However, training on the strict-filtered balanced recovery dataset overcorrected the raw actor. The actor response direction and delta strength were good, but zero-steer bias became far too large. This is unsafe for CARLA promotion.

The next round should reduce recovery supervision influence rather than adding more recovery rows. Useful options:

```text
1. Mix fewer recovery rows, for example 8 left + 8 right.
2. Lower learning rate or train only a smaller adapter subset.
3. Keep semantic-filtered labels and add a separate strict-lane audit rather than forcing all heading-dominant rows to lane convention.
```

Round63 remains the best current CARLA candidate.
