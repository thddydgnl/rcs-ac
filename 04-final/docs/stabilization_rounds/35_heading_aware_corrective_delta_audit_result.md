# Round 35 Heading-Aware Corrective Delta Audit Result

Date: 2026-05-02

## Status

`BLOCK`

## Mode

Full workflow escalation.

Reason: this round audits privileged offline target semantics. It did not create or modify any dataset, relabel targets, filter rows, train BC, run CARLA, run PPO, or change runtime behavior.

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

Round34 showed that the strict lane-offset audit still reproduced the Round33 block:

```text
gate_status: BLOCK
wrong_sign_left_count: 58
left_sign_rate_vs_neg_lane: 0.705584
right_sign_rate_vs_neg_lane: 1.0
```

But all `58/58` wrong-sign rows had:

```text
abs(lane_offset) < 0.55
abs(heading_error) >= 3.0
heading_error < 0
target_corrective_delta < 0
```

That suggested the strict `-sign(lane_offset)` expected sign may false-block heading-dominant recovery rows.

## Experiment Brief

Primary hypothesis:

```text
Strict lane-offset-only alignment false-blocks rows where heading error dominates the corrective target.
```

Allowed changes:

```text
Add output-only heading-aware diagnostic metrics to the dataset audit script.
```

Rejected changes:

```text
No dataset write.
No target mutation.
No relabeling.
No filtering.
No BC training.
No CARLA.
No PPO.
No train_bc.py changes.
No runtime steering rule.
```

Expected signs used only for offline diagnostics:

```text
lane_dominant_expected_sign = -sign(lane_offset)
heading_dominant_expected_sign = sign(heading_error)
```

## Sub-Agent Summary

Agent C:

- Keep Round35 pre-training and pre-CARLA.
- Use heading-aware metrics as an audit only.
- Continue only if lane-dominant and heading-dominant target labels are both clean.

Agent J:

- Do not turn heading/lane expected signs into relabeling or runtime control logic.
- Keep the script output-only.
- Preserve the strict Round33 metric as diagnostic, not as the sole corrected-dataset gate.

Agent B:

- Add metrics only inside `audit_arrays()`.
- Keep existing gates unchanged.
- Add tests for heading-dominant alignment and ambiguous rows.

## Code Changes

Updated:

```text
rcs_ac_first/scripts/audit_corrective_delta_dataset.py
tests/test_audit_corrective_delta_dataset.py
```

New summary fields include:

```text
heading_aware_gate_status
heading_aware_eligible_count
heading_aware_lane_dominant_count
heading_aware_lane_dominant_alignment_rate
heading_aware_heading_dominant_count
heading_aware_heading_dominant_alignment_rate
heading_aware_lane_heading_conflict_count
heading_aware_conflict_alignment_vs_heading_rate
heading_aware_conflict_alignment_vs_lane_rate
heading_aware_ambiguous_count
heading_aware_ambiguous_rate
```

Existing strict lane-offset fields are preserved.

## Local Validation

```text
python -m py_compile rcs_ac_first/scripts/audit_corrective_delta_dataset.py
python -m pytest tests/test_audit_corrective_delta_dataset.py tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py tests/test_evaluate_bc_gate.py -q
```

Result:

```text
36 passed
```

## Remote Run

Windows command:

```text
py -3.12 -B -m rcs_ac_first.scripts.audit_corrective_delta_dataset
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
  --run-name round35_heading_aware_audit
  --output-dir remote_runs/round35_heading_aware_audit
  --max-wrong-sign-rows 500
```

Pulled local artifacts:

```text
round35_results/heading_aware_audit_summary.json
round35_results/wrong_sign_left_rows.csv
```

## Result

Strict lane-offset gate remains blocked:

```text
gate_status: BLOCK
gate_failures: sign_rate, left_sign_rate
left_sign_rate_vs_neg_lane: 0.705584
right_sign_rate_vs_neg_lane: 1.0
wrong_sign_left_count: 58
```

Heading-aware diagnostic gate passes:

```text
heading_aware_gate_status: PASS
heading_aware_eligible_count: 340
heading_aware_lane_dominant_count: 163
heading_aware_lane_dominant_alignment_rate: 1.0
heading_aware_heading_dominant_count: 169
heading_aware_heading_dominant_alignment_rate: 1.0
heading_aware_lane_heading_conflict_count: 58
heading_aware_conflict_alignment_vs_heading_rate: 1.0
heading_aware_conflict_alignment_vs_lane_rate: 0.0
heading_aware_ambiguous_count: 8
heading_aware_ambiguous_rate: 0.023529
```

Route-bin heading-aware result:

```text
0.025-0.050: strict 1.000000, heading_dominant_count 16, heading_rate 1.000000
0.050-0.075: strict 0.913462, heading_dominant_count 14, heading_rate 1.000000
0.075-0.100: strict 0.752525, heading_dominant_count 139, heading_rate 1.000000
```

Source metadata remains missing:

```text
missing_source_metadata: true
source_label: missing
source_dataset: missing
source_row_index: missing
source_episode_id: missing
source_step_in_episode: missing
```

## Interpretation

Round35 explains the Round33/Round34 block:

```text
The strict lane-offset-only audit labels 58 heading-dominant conflict rows as wrong.
Those same 58 rows align perfectly with the heading-dominant expected sign.
```

So the current Round29 mixed dataset is not proven bad by the strict `-lane_offset` metric alone. The likely issue is that early-severe target semantics need to distinguish:

```text
lane-dominant recovery
heading-dominant recovery
lane/heading conflict recovery
ambiguous low-error rows
```

However, this is still not a corrected-dataset pass.

Reason:

- Source metadata is absent.
- Controller provenance fields such as `lateral_recovery_active` and `lateral_recovery_correction` are absent in the mixed NPZ.
- The new heading-aware metric is an offline privileged diagnostic, not a training target.

## Gate Decision

Do not run BC.

Do not run CARLA.

Do not run PPO.

Do not create a corrected dataset from this result alone.

The audit removes one suspected blocker:

```text
Round33 strict left-sign failure is mostly a heading-dominant audit-definition issue.
```

But another blocker remains:

```text
The mixed dataset lacks row-level source/provenance metadata needed for safe filtering or relabeling.
```

## Next Recommendation

Next round should add row-level source/provenance metadata to future mixed datasets and, if possible, reconstruct or regenerate the Round29 mix with provenance.

Recommended narrow path:

1. Update `mix_expert_datasets.py` and `combine_expert_datasets.py` to preserve:

```text
source_label
source_dataset_index
source_row_index
source_episode_id
source_step_in_episode
```

2. Rebuild the Round29 mixed dataset from original component NPZs if they are still present on Windows.

3. Re-run both audits:

```text
strict lane-offset audit
heading-aware audit
```

4. Only after provenance exists, decide whether any rows should be filtered, kept as heading-dominant exceptions, or marked ambiguous.

No runtime behavior should change.
