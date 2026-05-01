# Round 26 - Signed Correction Audit

Date: 2026-05-01

## Goal

Check whether the signed correction failure from Round 25 is:

```text
1. a pure sign bug,
2. a loss/masking design issue,
3. a visual_center_error sign interpretation issue.
```

This round did not run PPO or CARLA. It was a code/log audit plus a diagnostic logging/test patch.

## Evidence

Round 25 exact collision cluster:

```text
route_progress: 0.16-0.20
lane_offset: > +1.5
heading_error: < -8
risk rows: 23
visual_center_corrective_sign_rate: 1.000000
route_offset_corrective_sign_rate: 0.000000
route_visual_center_sign_agreement_rate: 0.000000
visual_zero_signed_steer_delta mean: +0.024834
visual_lane_center_error_mean: -1.933721
collision terminals: 4
```

For positive `lane_offset`, route-offset corrective visual steer should be negative:

```text
route_offset_corrective_sign = visual_zero_signed_steer_delta * (-lane_offset) > 0
```

The cluster instead has positive visual-zero signed steer, so the policy is being pushed in the wrong route-offset direction.

## Agent Findings

Agent A:

```text
Evidence does not support a global sign flip.
The failure is localized around route_progress 0.179487.
Loss/masking issue is more likely because train/val masked metrics look perfect while eval cluster agreement is 0.
```

Agent C:

```text
For route_progress 0.16-0.20, lane_offset > +1.5, heading_error < -8, signed visual steer must be negative.
Current exact-mask rows have mean signed steer +0.0248 and route-offset corrective rate 0/23.
visual_center_error cannot be trusted as sole sign source in that cluster.
```

Agent B:

```text
raw_features[..., 10] is road_x and raw_features[..., 11] is lane_x.
x_coords run from -1 left to +1 right.
No isolated code sign inversion was found.
visual-center losses only train rows where target_delta and visual_center_error already agree, so conflicting rows can be silently masked out.
```

## Conclusion

Pure one-line sign bug:

```text
Unlikely.
```

The road/lane raw feature indices and `lane_x - road_x` computation are internally consistent. The evaluator and training losses use compatible visual-center sign rules.

Loss/masking design issue:

```text
Very likely.
```

The visual-center losses currently compute:

```text
center_consistent = (-visual_center_error * target_delta) > 0
corrective_mask = availability * sign_active * center_active * center_consistent
```

That means rows where visual-center sign disagrees with target/route direction are excluded before the loss is applied. This can produce perfect masked metrics while the exact collision cluster remains wrong.

visual_center_error sign interpretation issue:

```text
Confirmed as a local trust issue, not a raw index bug.
```

At the collision cluster:

```text
lane_offset > 0
visual_center_error < 0
visual_zero_signed_steer_delta > 0
```

So visual-center correction is active, but it is not route-offset corrective.

## Code Change

Files:

```text
rcs_ac_first/scripts/train_bc.py
tests/test_train_bc_perception.py
```

Added train/val metrics:

```text
visual_center_target_consistency_rate
visual_center_target_consistency_mask_rate
route_aware_visual_center_target_consistency_rate
route_aware_visual_center_target_consistency_mask_rate
```

These metrics measure target/visual-center sign consistency before applying the existing `center_consistent` mask. They make the silent exclusion visible without changing the loss behavior.

Added tests:

```text
RGB raw feature index 10 tracks road_x.
RGB raw feature index 11 tracks lane_x.
Inconsistent route-aware visual-center target rows report target_consistency_rate=0 and loss mask=0.
```

## Verification

```text
python -m py_compile rcs_ac_first/scripts/train_bc.py tests/test_train_bc_perception.py
PASS

PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py -q
19 passed

PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
20 passed

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --device cpu
PASS
```

## Decision

Do not flip the sign globally.

Do not enter PPO.

The next stabilization experiment should add a narrow collision-cluster route-offset agreement loss for:

```text
route_progress: 0.16-0.20
lane_offset: > +1.5
heading_error: < -8
```

Gate the next fix on:

```text
exact cluster visual_zero_signed_steer_delta < 0
exact cluster route_offset_corrective_sign_rate >= 0.80
collision_count = 0 in 512 before any 2048 continuation
no early_offroad
no runtime steering assist
```
