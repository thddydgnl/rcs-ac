# Round 36 Dataset Provenance Preservation

Date: 2026-05-02

Status: IMPLEMENTED, DATASET REBUILD STILL REQUIRED

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

New tests verify:

- `mix_expert_datasets.py` writes per-row source metadata.
- source metadata remains row-aligned after output shuffle.
- `combine_expert_datasets.py` captures original episode/step before episode reindexing.

## Current Gate State

Goal progression remains blocked before BC/CARLA/PPO.

Reason:

```text
The existing Round 29 mixed dataset was created before this provenance change.
```

Local Mac does not currently have the Round 29 component NPZ files needed to rebuild:

```text
data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
```

Therefore this round implements the missing provenance mechanism, but does not yet produce a provenance-backed corrected/rebuilt Round 29 dataset.

## Next Step

On Windows, locate the original Round 29 component datasets and rebuild the mixed dataset with the updated `mix_expert_datasets.py`.

Then rerun:

```text
strict lane-offset corrective-delta audit
heading-aware corrective-delta audit
```

Only after the rebuilt dataset contains provenance should the next round decide whether rows are kept as heading-dominant exceptions, filtered as ambiguous, or relabeled.
