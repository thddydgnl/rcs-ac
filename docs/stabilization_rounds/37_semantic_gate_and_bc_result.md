# Round 37 Semantic Gate And BC Result

Date: 2026-05-02

Status: BLOCKED BEFORE CARLA

## Purpose

Round 36 resolved row-level provenance for the rebuilt early-route mix. This round promotes the heading-aware label semantics into an explicit offline dataset gate, then checks whether BC with the same semantics improves raw actor recovery response.

This round does not change runtime control, CARLA behavior, PPO, reward, or model architecture.

## Code Changes

`rcs_ac_first/scripts/audit_corrective_delta_dataset.py`

- Adds explicit semantic dataset gate fields:

```text
semantic_gate_status
semantic_gate_failures
```

- Makes public `gate_status` and `gate_failures` follow the semantic gate.
- Preserves the old lane-only gate as diagnostics:

```text
strict_lane_gate_status
strict_lane_gate_failures
strict_lane_aligned_rate_in_eligible
strict_lane_left_sign_rate_vs_neg_lane
strict_lane_right_sign_rate_vs_neg_lane
```

`rcs_ac_first/scripts/train_bc.py`

- Adds `early_severe_alignment_mode`.
- Default remains `lane`, preserving previous behavior.
- New `semantic` mode uses the same offline privileged convention as the dataset gate:

```text
lane-dominant expected sign: -sign(lane_offset)
heading-dominant expected sign: sign(heading_error)
```

This is privileged offline supervision only. It is not a runtime steering rule and does not add privileged signals to inference.

`configs/bc_frame_rgb_perception_actor_features_v2_semantic_early_severe_delta_round37.yaml`

- New Round37 BC config derived from Round32.
- Enables:

```text
early_severe_alignment_mode: semantic
```

## Local Validation

```bash
python -m py_compile rcs_ac_first/scripts/train_bc.py rcs_ac_first/scripts/audit_corrective_delta_dataset.py
python -m pytest tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_provenance.py tests/test_audit_corrective_delta_dataset.py tests/test_expert_dataset_sequence.py -q
```

Result:

```text
41 passed in 1.36s
```

## Dataset Gate

Windows command:

```text
py -3.12 -B -m rcs_ac_first.scripts.audit_corrective_delta_dataset \
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz \
  --run-name round37_semantic_gate_audit \
  --output-dir remote_runs/round37_semantic_gate_audit \
  --max-wrong-sign-rows 500
```

Result:

```text
gate_status: PASS
gate_failures: []
semantic_gate_status: PASS
semantic_gate_failures: []
missing_source_metadata: false
heading_aware_gate_status: PASS
heading_aware_gate_failures: []
strict_lane_gate_status: BLOCK
strict_lane_gate_failures: ["strict_lane_sign_rate", "strict_lane_left_sign_rate"]
```

Key metrics:

```text
heading_aware_eligible_count: 340
heading_aware_lane_dominant_alignment_rate: 1.0
heading_aware_heading_dominant_alignment_rate: 1.0
heading_aware_lane_heading_conflict_count: 58
heading_aware_conflict_alignment_vs_heading_rate: 1.0
heading_aware_ambiguous_count: 8
heading_aware_ambiguous_rate: 0.023529
target_delta_abs_mean: 0.096613
target_delta_abs_p10: 0.049289
target_steer_saturation_rate: 0.0
target_steer_consistency_mae: 1.06e-9
```

Decision: the rebuilt Round36 dataset passes the semantic offline dataset gate.

## BC Training

Windows command:

```text
py -3.12 -B -m rcs_ac_first.scripts.train_bc \
  --config configs/bc_frame_rgb_perception_actor_features_v2_semantic_early_severe_delta_round37.yaml \
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz \
  --init-checkpoint checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt \
  --init-strict true \
  --run-name rgb_perception_actor_features_v2_semantic_early_severe_delta_bc_round37 \
  --device cuda
```

Output:

```text
log: logs/rgb_perception_actor_features_v2_semantic_early_severe_delta_bc_round37/bc_training_log.csv
checkpoint: checkpoints/rgb_perception_actor_features_v2_semantic_early_severe_delta_bc_round37/last.pt
```

Training result:

```text
epoch 1 val_loss: 0.011127
epoch 2 val_loss: 0.011146
epoch 3 val_loss: 0.010736
epoch 4 val_loss: 0.010717
epoch 5 val_loss: 0.010276
```

Final validation metrics:

```text
visual_zero_action_l1: 0.142400
visual_zero_steer_delta: 0.022376
target_corrective_delta_sign_rate: 0.924622
target_corrective_delta_mask_rate: 0.896714
early_severe_steer_mask_rate: 0.077465
early_severe_steer_sign_rate: 0.095587
early_severe_steer_left_sign_rate: 0.092813
early_severe_steer_right_sign_rate: 0.098592
early_severe_steer_delta_l1: 0.004084
early_severe_steer_sign_loss: 0.000432
early_severe_steer_left_mask_rate: 0.035211
early_severe_steer_right_mask_rate: 0.042254
```

Comparison to Round32:

```text
val_loss: 0.010517 -> 0.010276
early_severe_steer_mask_rate: 0.065728 -> 0.077465
early_severe_steer_delta_l1: 0.004342 -> 0.004084
early_severe_steer_sign_rate: 0.095015 -> 0.095587
early_severe_steer_left_sign_rate: 0.090245 -> 0.092813
```

The semantic mask increases coverage and slightly improves L1, but it does not create a usable signed raw actor response.

## Offline Actor Sign Audit

Early-left result:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.000224
actor_sign_response_left_bias_at_zero: -0.055056
```

Early-any result:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.000000197
actor_sign_response_left_bias_at_zero: -0.045246
```

## Decision

Status: `BLOCK`

Do not run CARLA 256/512.

The dataset semantics are no longer the main blocker. The remaining blocker is that the current trainable scope:

```text
perception_aux_head
rgb_perception_actor_features
actor_input_adapter
```

still does not route scalar recovery cues strongly enough into raw actor steering.

Next round should test a very short, tightly gated BC run that includes `recurrent_actor_critic` in the trainable scope, then repeat the same offline actor sign audit before any simulator evaluation.
