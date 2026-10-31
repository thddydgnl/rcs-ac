# Round 34 Early-Left Wrong-Sign Row Audit Result

Date: 2026-05-02

## Status

`BLOCK`

## Mode

Full workflow escalation.

Reason: this round audits dataset target semantics and privileged offline supervision. It intentionally did not run BC, CARLA, PPO, relabeling, filtering, or any runtime/controller change.

## Baseline

Last safety baseline remains Round29 guarded 512:

```text
collision_count: 0
early_offroad_count: 0
max_route_progress: 0.564103
mean_route_progress: 0.271484
mean_action_jitter_l1: 0.073864
```

## Latest Blocker

Round33 dataset audit on:

```text
data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
```

blocked because:

```text
eligible_severe_count: 332
wrong_sign_count: 58
left_count: 197
left_sign_rate_vs_neg_lane: 0.705584
right_sign_rate_vs_neg_lane: 1.0
```

## Experiment Brief

Primary hypothesis:

```text
The apparent early-left wrong-sign rows may be caused by lane/heading disagreement,
not simply weak target magnitude or a global target-delta sign bug.
```

Allowed changes:

```text
Extend the dataset audit script to export row-level wrong-sign left rows.
```

Rejected changes:

```text
No BC training.
No CARLA.
No PPO.
No relabeling.
No filtering.
No runtime steering rule.
No lane_offset-based controller.
```

Mechanism metrics:

```text
wrong_sign_left_count
wrong_sign_left_by_episode_top
wrong_sign_left_by_route_bin
wrong_sign_left_rows.csv
source metadata presence
lane_offset / heading_error distributions for wrong rows
```

Stop condition:

```text
Do not proceed to corrected dataset or BC if source provenance is missing or rows remain semantically ambiguous.
```

## Sub-Agent Summary

Agent A:

- Wrong-sign eligible rows are effectively all left-side.
- `58/197` early-left rows are wrong by the `target_corrective_delta * (-lane_offset)` test.
- Most wrong rows are concentrated in route bin `0.075-0.100`.

Agent F:

- Episode/step audit is possible.
- Source-level audit is not reliable from the current NPZ.
- `mix_expert_datasets.py` accepts a part label but only writes it to summary JSON, not per-row NPZ fields.

Agent C:

- Stop before training/CARLA/PPO.
- A corrected dataset candidate must preserve traceability and pass side-split gates.
- Do not pass by deleting hard examples.

Agent J:

- Treat `lane_offset` and `heading_error` as audit stratifiers, not runtime control rules.
- Do not relabel solely from `-lane_offset`.
- Report this as privileged offline supervision, separate from raw policy behavior.

## Code Changes

Updated:

```text
rcs_ac_first/scripts/audit_corrective_delta_dataset.py
tests/test_audit_corrective_delta_dataset.py
```

The audit now writes:

```text
summary.json
wrong_sign_left_rows.csv
```

The row table includes:

```text
row_index
episode_id
step_in_episode
route_progress
route_bin
lane_offset
heading_error
speed_kmh
policy_action_steer
target_corrected_steer
target_corrective_delta
expected_signed
source metadata fields when present
controller provenance fields when present
classification
```

The script does not change the dataset and does not relabel any target.

## Local Validation

```text
python -m py_compile rcs_ac_first/scripts/audit_corrective_delta_dataset.py
python -m pytest tests/test_audit_corrective_delta_dataset.py tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py tests/test_evaluate_bc_gate.py -q
```

Result:

```text
34 passed
```

## Remote Run

Windows command:

```text
py -3.12 -B -m rcs_ac_first.scripts.audit_corrective_delta_dataset
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
  --run-name round34_wrong_sign_row_audit
  --output-dir remote_runs/round34_wrong_sign_row_audit
  --max-wrong-sign-rows 500
```

Pulled local artifacts:

```text
round34_results/wrong_sign_row_audit_summary.json
round34_results/wrong_sign_left_rows.csv
```

## Result

The Round33 aggregate result reproduced:

```text
gate_status: BLOCK
gate_failures: sign_rate, left_sign_rate
wrong_sign_count: 58
wrong_sign_left_count: 58
wrong_sign_left_rate_in_left: 0.294416
left_sign_rate_vs_neg_lane: 0.705584
right_sign_rate_vs_neg_lane: 1.0
```

Source metadata is missing:

```text
missing_source_metadata: true
reconstructable_from_summary: none
source_label: missing
source_dataset: missing
source_row_index: missing
source_episode_id: missing
source_step_in_episode: missing
```

Wrong-sign left rows by route bin:

```text
0.050-0.075: 9
0.075-0.100: 49
```

Wrong-sign left rows by episode:

```text
episode 101: 10
episode 102: 10
episode 100: 9
episode 103: 9
episode 106: 7
episode 108: 7
episode 107: 6
```

Wrong-sign row distribution:

```text
count: 58
lane_offset_min: -0.282535
lane_offset_max: -0.001588
lane_offset_mean: -0.121468
heading_error_min: -13.337110
heading_error_max: -8.363881
heading_error_mean: -10.301501
target_corrective_delta_min: -0.071823
target_corrective_delta_max: -0.022494
target_corrective_delta_mean: -0.047149
speed_kmh_mean: 7.706409
```

Important observation:

```text
lane_abs >= 0.55 count: 0 / 58
abs(heading_error) >= 3 count: 58 / 58
heading_error < 0 count: 58 / 58
```

## Interpretation

The Round33 "wrong-sign early-left" result was not actually dominated by rows that are far left of lane center.

Every exported wrong-sign row has:

```text
lane_offset < 0
abs(lane_offset) < 0.55
heading_error < -8.36
target_corrective_delta < 0
```

So the rows fail the `target_corrective_delta * (-lane_offset) > 0` audit because lane offset is slightly negative, but heading error is strongly negative and the target asks for negative correction.

This strongly suggests a lane/heading conflict case:

```text
Using -lane_offset alone as the expected correction sign is too strict when lane offset is small
and heading error dominates the recovery direction.
```

It does not prove that the labels are correct, because source metadata and controller provenance fields are missing. It does show that the Round33 blocker should not be fixed by blindly flipping all early-left target signs.

## Gate Decision

Status remains `BLOCK`.

Do not create a corrected dataset yet.

Do not run BC.

Do not run CARLA or PPO.

Reason:

- All wrong-sign rows are still `classification=unresolved`.
- Source metadata is missing from the mixed NPZ.
- The apparent wrong-sign rows may be valid heading-dominant exceptions.
- Relabeling from `-lane_offset` alone would risk creating a privileged lane-offset controller through labels.

## Next Recommendation

Next round should split the audit expectation into lane-dominant and heading-dominant cases before any dataset filtering or relabeling.

Recommended narrow changes:

1. Add a heading-aware dataset target audit metric:

```text
lane_dominant_mask: abs(lane_offset) >= 0.55
heading_dominant_mask: abs(lane_offset) < 0.55 and abs(heading_error) >= 3.0
lane_expected_sign: -sign(lane_offset)
heading_expected_sign: sign(heading_error)
```

2. Report side splits separately:

```text
lane_dominant_alignment
heading_dominant_alignment
lane_heading_conflict_count
```

3. Preserve the original strict Round33 audit as a diagnostic, but do not use it alone as the corrected-dataset gate.

4. Add source metadata to future mixed datasets:

```text
source_label
source_dataset_index
source_row_index
source_episode_id
source_step_in_episode
```

No runtime behavior should change.
