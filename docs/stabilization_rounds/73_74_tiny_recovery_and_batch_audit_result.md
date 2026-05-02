# Rounds 73-74 Tiny Recovery And Batch Audit Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Rounds 71-72 showed that full recovery/filter datasets overcorrected the actor. Rounds 73-74 tested a smaller intervention:

```text
base Round36 + 8 left aligned recovery rows + 8 right biased aligned recovery rows
trainable scope: scalar_actor_adapter only
init checkpoint: Round63
lateral_recovery.enabled: false
```

This remains offline privileged supervision. No runtime steering override or hidden simulator signal was added to inference.

## Round73 Dataset

Raw tiny mix:

```text
data/expert/rgb_perception_actor_features_v2_tiny_balanced_recovery_mix_4272_round73.npz
base_round36: 4256
left_recovery_round70: 8
right_biased_recovery_round71: 8
```

The raw tiny mix still inherited the known Round36 wrong-sign rows:

```text
left_sign_rate_vs_neg_lane: 0.902540
right_sign_rate_vs_neg_lane: 1.0
wrong_sign_count: 188
```

Therefore it was not used for BC. A strict-lane filtered version was created:

```text
data/expert/rgb_perception_actor_features_v2_tiny_balanced_recovery_strict_filtered_round73.npz
num_transitions: 4084
strict_lane_gate_status: PASS
strict_lane_left_sign_rate_vs_neg_lane: 1.0
strict_lane_right_sign_rate_vs_neg_lane: 1.0
wrong_sign_count: 0
```

The semantic heading-dominant gate still blocks, so this dataset should be interpreted only as a lane-offset corrective probe.

## Round73 BC

```text
config: configs/bc_frame_rgb_perception_actor_features_v2_tiny_balanced_recovery_round73.yaml
learning_rate: 0.00005
trainable_module_names: scalar_actor_adapter
selected_params: 384
val_loss: 0.012112
```

Single-sample actor audit:

```text
early_left: BLOCK
  max_abs_zero_steer: 0.107602
  min_abs_delta: 0.019772

early_any: PASS
  max_abs_zero_steer: 0.018029
  min_abs_delta: 0.018260
```

Decision: do not run CARLA.

## Round74 BC

Round74 repeated the same strict-filtered tiny dataset and scalar-only scope with a 10x smaller learning rate:

```text
config: configs/bc_frame_rgb_perception_actor_features_v2_tiny_balanced_recovery_lr5e6_round74.yaml
learning_rate: 0.000005
selected_params: 384
val_loss: 0.011901
```

Single-sample actor audit:

```text
early_left: BLOCK
  max_abs_zero_steer: 0.107276
  min_abs_delta: 0.018657

early_any: PASS
  max_abs_zero_steer: 0.018029
  min_abs_delta: 0.017236
```

Decision: do not run CARLA.

## Audit Tool Finding

The actor sign audit previously selected only the first matching sample for `early_left`. After dataset shuffle/filter changes, that first sample can change, which makes PASS/BLOCK unstable. The audit tool now supports:

```text
--max-samples N
```

This runs the scalar probe over multiple matching samples and reports aggregate sample pass rate, max zero steer, and min delta.

Batch audit exposed that Round63 itself does not satisfy robust early-left zero-bias on the first 16 early-left samples:

```text
Round63 original dataset, early_left, max_samples=16:
  sample_pass_rate: 0.0625
  max_abs_zero_steer: 0.279893
  min_abs_delta: 0.014203
  pass_status: BLOCK

Round74 strict-filtered dataset, early_left, max_samples=16:
  sample_pass_rate: 0.0
  max_abs_zero_steer: 0.412775
  min_abs_delta: 0.013204
  pass_status: BLOCK
```

This means the old Round63 PASS was too dependent on a single favorable sample.

## Interpretation

The useful result is not that Round74 improved driving. It did not advance to CARLA. The useful result is that the current early-left zero-bias gate is partly measuring non-neutral visual/route context, not just scalar recovery behavior. In an early-left sample, RGB and route can still legitimately imply nonzero steering even when one probed scalar is set to zero.

The actor sign direction and monotonicity are strong, but zero-bias is not stable across real early-left samples. The next stabilization step should separate:

```text
1. neutral-center zero-bias gate: evaluated only on centered/neutral samples
2. early-left/right corrective sign gate: evaluated on off-center samples, without requiring zero steer
3. sustained late recovery gate: focused on route 0.30-0.34, lane_offset <= -1.8, heading_error >= 12
```

Round63 remains the best CARLA candidate so far, but the audit criteria need this separation before another BC/CARLA promotion.
