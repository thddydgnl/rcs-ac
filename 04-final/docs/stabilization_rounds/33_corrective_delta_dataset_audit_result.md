# Round 33 Corrective Delta Dataset Audit Result

Date: 2026-05-02

## Goal

Before unfreezing the actor head after Round32, verify whether the Round29 mixed recovery dataset has clean early-severe `target_corrective_delta` labels.

This round deliberately did not run CARLA, PPO, or another BC training pass.

## Reason

Round32 added direct early-severe steer delta supervision, but the offline actor sign audit still failed:

```text
early-left actor_sign_response_all_axes_rate: 0.0
early-left actor_sign_response_min_abs_delta: 0.00023355
early-left zero steer bias: -0.052492
validation early_severe_steer_sign_rate: 0.095015
```

The next possible causes were:

1. The trainable actor path was too narrow.
2. The dataset target signs were weak or inconsistent.

Round33 checked the second cause first.

## Code Changes

Added a dataset audit script:

```text
rcs_ac_first/scripts/audit_corrective_delta_dataset.py
```

Added tests:

```text
tests/test_audit_corrective_delta_dataset.py
```

The script audits:

- key presence and finite rates
- early-severe counts by side
- `target_corrective_delta * (-lane_offset)` sign agreement
- target delta magnitude distribution
- target steer consistency with `policy_action_steer + target_corrective_delta`
- target steer saturation rate
- route-bin side and sign summaries

## Local Validation

```text
python -m py_compile rcs_ac_first/scripts/audit_corrective_delta_dataset.py
python -m pytest tests/test_audit_corrective_delta_dataset.py tests/test_train_bc_perception.py tests/test_expert_dataset_sequence.py tests/test_evaluate_bc_gate.py -q
```

Result:

```text
33 passed
```

## Remote Execution

Windows dataset:

```text
data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
```

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.audit_corrective_delta_dataset
  --dataset data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
  --run-name round33_target_audit
  --output-dir remote_runs/round33_target_audit
```

Pulled local summary:

```text
round33_results/target_audit_summary.json
```

## Result

Status: `BLOCK`

Gate failures:

```text
sign_rate
left_sign_rate
```

Key metrics:

```text
eligible_severe_count: 332
left_count: 197
right_count: 135
aligned_rate_in_eligible: 0.825301
wrong_sign_rate_in_eligible: 0.174699
left_sign_rate_vs_neg_lane: 0.705584
right_sign_rate_vs_neg_lane: 1.0
target_delta_abs_mean: 0.096613
target_delta_abs_p10: 0.049289
target_steer_saturation_rate: 0.0
target_steer_consistency_mae: 0.0000000011
zero_delta_rate_in_route_severe: 0.341270
```

Route bins:

```text
0.025-0.050: count 30, left 30, right 0, aligned_rate 1.000000
0.050-0.075: count 104, left 104, right 0, aligned_rate 0.913462
0.075-0.100: count 198, left 63, right 135, aligned_rate 0.752525
```

## Interpretation

The target magnitudes are strong enough and target steer is not saturating, but the early-severe target signs are not clean enough.

The main failure is on left-side rows:

```text
left_sign_rate_vs_neg_lane: 0.705584
required: >= 0.98
```

That means about 29% of eligible left-side early-severe rows ask for a correction that does not agree with `-lane_offset`.

This explains why Round32's direct target-delta loss could lower left bias slightly without producing a reliable signed actor response. The model may be receiving mixed supervision in exactly the early-left slice that causes simulator collapse.

## Decision

Do not run actor-head-unfrozen BC yet.

Do not run CARLA.

Reason:

- Dataset target audit failed before the trainable-scope experiment.
- Unfreezing `recurrent_actor_critic` against inconsistent labels can memorize or amplify the wrong early-left correction.
- Round32's actor sign audit already blocks simulator promotion.

## Next Recommendation

Next round should fix the dataset target semantics before another BC run.

Recommended narrow path:

1. Audit wrong-sign early-left rows by episode/step and target source.
2. Determine whether the sign issue comes from:
   - `lane_offset` convention mismatch,
   - controller target generation,
   - mixed clean/recovery rows,
   - route/heading cases where `-lane_offset` alone is the wrong reference.
3. If it is a label issue, filter or relabel early-severe target rows and regenerate a small corrected dataset.
4. Re-run this dataset audit before any BC training.

Promotion gate remains:

```text
eligible early-severe count >= 200
left/right count >= 40
aligned_rate_in_eligible >= 0.98
left_sign_rate_vs_neg_lane >= 0.98
right_sign_rate_vs_neg_lane >= 0.98
target_delta_abs_mean >= 0.035
target_delta_abs_p10 >= 0.010
target_steer_saturation_rate <= 0.05
```
