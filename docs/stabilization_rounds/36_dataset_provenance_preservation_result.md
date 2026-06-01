# Round 36 Dataset Provenance Preservation

Date: 2026-05-02

Status: IMPLEMENTED, REBUILT DATASET AUDITED, BC/CARLA/PPO STILL BLOCKED

## Purpose

Round 35 showed that the strict lane-offset audit was false-blocking heading-dominant early-left rows, but the mixed Round 29 dataset still lacked row-level provenance. Without provenance, wrong-sign or ambiguous rows cannot be traced back to original component datasets, episodes, and steps.

This round implements provenance preservation for future mixed/combined expert datasets. It does not change training targets, runtime control, CARLA behavior, PPO, or any inference input.

## Changes

`rcs_ac_first/scripts/mix_expert_datasets.py`

- Adds row-level provenance to every sliced part before episode reindexing:

```text
source_label
source_dataset
source_dataset_index
source_row_index
source_episode_id
source_step_in_episode
```

- Keeps `episode_id` reindexing behavior unchanged.
- Keeps shuffle behavior unchanged, with provenance arrays shuffled along with their rows.
- Adds `dataset_index` and `dataset_path` to the part summary JSON.

`rcs_ac_first/scripts/combine_expert_datasets.py`

- Fills missing row-level provenance for each input dataset before episode reindexing.
- Preserves existing provenance fields when they already exist.
- Stores original `episode_id` and `step_in_episode` in `source_episode_id` and `source_step_in_episode` before combine-time episode reindexing.

## Guardrails

This is artifact metadata only.

No changes were made to:

```text
target_corrective_delta
target_corrected_steer
expert_action
train_bc.py
evaluate_bc.py
CARLA runtime control
PPO
reward
model architecture
```

The new fields are for offline audit/provenance. They must not be used as runtime policy inputs or as a steering override.

## Validation

Commands run on Mac:

```bash
python -m py_compile rcs_ac_first/scripts/mix_expert_datasets.py rcs_ac_first/scripts/combine_expert_datasets.py
python -m pytest tests/test_expert_dataset_provenance.py tests/test_audit_corrective_delta_dataset.py tests/test_expert_dataset_sequence.py -q
```

Result:

```text
9 passed in 1.20s
```

Additional focused validation after audit output was extended with source episode/step fields:

```bash
python -m py_compile rcs_ac_first/scripts/audit_corrective_delta_dataset.py
python -m pytest tests/test_audit_corrective_delta_dataset.py tests/test_expert_dataset_provenance.py -q
```

Result:

```text
8 passed in 1.19s
```

New tests verify:

- `mix_expert_datasets.py` writes per-row source metadata.
- source metadata remains row-aligned after output shuffle.
- `combine_expert_datasets.py` captures original episode/step before episode reindexing.

## Windows Rebuild

Windows `git pull --ff-only` was blocked by Git Credential Manager requiring an interactive TTY, so the verified Mac scripts were copied directly to Windows for this run.

The Round 29 component datasets were present on Windows:

```text
rgb_perception_actor_features_v2_near_cluster_mix_3744_round27b.npz
rgb_perception_actor_features_v2_early_route_recovery_256_round29.npz
rgb_perception_actor_features_v2_early_route_failure_recovery_256_round29.npz
```

Rebuilt dataset:

```text
data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz
num_transitions: 4256
input_counts: [3744, 256, 256]
route_progress_min: 0.025641
route_progress_max: 0.487179
route_progress_mean: 0.253892
```

Pulled local artifacts:

```text
round36_results/early_route_mix_4256_round36_provenance.summary.json
round36_results/provenance_source_audit_summary.json
round36_results/wrong_sign_left_rows_with_source_episode.csv
```

## Audit Result

Command on Windows:

```text
py -3.12 -B -m rcs_ac_first.scripts.audit_corrective_delta_dataset \
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz \
  --run-name round36_provenance_source_audit \
  --output-dir remote_runs/round36_provenance_source_audit \
  --max-wrong-sign-rows 500
```

Strict lane-offset audit:

```text
gate_status: BLOCK
gate_failures: ["sign_rate", "left_sign_rate"]
eligible_severe_count: 332
aligned_rate_in_eligible: 0.825301
left_sign_rate_vs_neg_lane: 0.705584
right_sign_rate_vs_neg_lane: 1.0
wrong_sign_left_count: 58
```

Heading-aware audit:

```text
heading_aware_gate_status: PASS
heading_aware_eligible_count: 340
heading_aware_lane_dominant_alignment_rate: 1.0
heading_aware_heading_dominant_alignment_rate: 1.0
heading_aware_lane_heading_conflict_count: 58
heading_aware_conflict_alignment_vs_heading_rate: 1.0
heading_aware_ambiguous_count: 8
heading_aware_ambiguous_rate: 0.023529
```

Provenance:

```text
missing_source_metadata: false
reconstructable_from_summary: row_metadata
```

Wrong-left strict rows by source:

```text
rgb_perception_actor_features_v2_early_route_recovery_256_round29: 38
rgb_perception_actor_features_v2_early_route_failure_recovery_256_round29: 20
```

Wrong-left strict rows by source episode:

```text
source_episode_id=1: 17
source_episode_id=2: 16
source_episode_id=3: 16
source_episode_id=0: 9
```

The exported CSV now includes:

```text
source_dataset
source_label
source_row_index
source_episode_id
source_step_in_episode
```

## Current Gate State

Goal progression remains blocked before BC/CARLA/PPO.

The provenance blocker is resolved for the rebuilt Round36 dataset, but no corrected training dataset has been approved yet. The strict lane-only gate still blocks, while the heading-aware gate passes. This means the next decision must be semantic:

```text
keep heading-dominant rows as valid exceptions
filter only truly ambiguous rows
or create a corrected audit gate that separates lane-dominant and heading-dominant recovery
```

No BC, CARLA eval, PPO, or robustness run should start until that dataset gate is explicitly defined and passes.
