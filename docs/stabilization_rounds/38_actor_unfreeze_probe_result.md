# Round 38 Actor Unfreeze Probe Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round 37 showed that the semantic dataset gate passes and semantic BC improves validation loss slightly, but raw actor sign response remains blocked. Round 38 tests whether a very short trainable-scope expansion can route the corrected recovery supervision into raw actor steering.

This is a Full-workflow type change because it changes trainable scope. It still does not change runtime control, reward, PPO, CARLA settings, or model architecture.

## Config

New config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_semantic_actor_unfreeze_probe_round38.yaml
```

Derived from Round37 with:

```text
epochs: 2
early_severe_alignment_mode: semantic
trainable_module_names:
  - perception_aux_head
  - rgb_perception_actor_features
  - actor_input_adapter
  - recurrent_actor_critic
```

The short 2-epoch run is intentionally a probe, not a promotion run.

## Local Validation

```bash
python -m pytest tests/test_train_bc_perception.py tests/test_audit_corrective_delta_dataset.py -q
```

Result:

```text
32 passed in 1.21s
```

## BC Training

Windows command:

```text
py -3.12 -B -m rcs_ac_first.scripts.train_bc \
  --config configs/bc_frame_rgb_perception_actor_features_v2_semantic_actor_unfreeze_probe_round38.yaml \
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz \
  --init-checkpoint checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt \
  --init-strict true \
  --run-name rgb_perception_actor_features_v2_semantic_actor_unfreeze_probe_bc_round38 \
  --device cuda
```

Trainable scope:

```text
selected_params: 559223
total_params: 954944
```

Validation:

```text
epoch 1 val_loss: 0.009307
epoch 2 val_loss: 0.007900
```

Final validation metrics:

```text
early_severe_steer_mask_rate: 0.077465
early_severe_steer_sign_rate: 0.095587
early_severe_steer_left_sign_rate: 0.092813
early_severe_steer_right_sign_rate: 0.098592
early_severe_steer_delta_l1: 0.003515
```

Compared with Round37, validation loss and delta L1 improve:

```text
Round37 val_loss: 0.010276
Round38 val_loss: 0.007900
Round37 early_severe_steer_delta_l1: 0.004084
Round38 early_severe_steer_delta_l1: 0.003515
```

But the training metrics still do not indicate useful signed recovery response.

## Offline Actor Sign Audit

Early-left:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.000328
actor_sign_response_left_bias_at_zero: -0.065140
```

Early-any:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.000000805
actor_sign_response_left_bias_at_zero: 0.013237
```

## Decision

Status: `BLOCK`

Do not run CARLA 256/512.

Opening `recurrent_actor_critic` improves BC loss and can move the zero-steer prior, especially early-any bias:

```text
Round37 early_any zero bias: -0.045246
Round38 early_any zero bias: 0.013237
```

However, the scalar probe sensitivity remains far below the goal:

```text
required min_abs_delta: >= 0.005
Round38 early_left min_abs_delta: 0.000328
Round38 early_any min_abs_delta: 0.000000805
```

The next blocker is not dataset label semantics. It is that scalar route/lane/heading perturbations still barely affect raw actor steering. Next work should inspect the feature path from `lane_offset`, `heading_error`, and route target fields into `recurrent_actor_critic`, and consider an explicit scalar corrective branch or stronger actor-input adapter supervision before further simulator runs.
