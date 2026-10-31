# Estimated Scalar Actor Features Plan

Status: **design plan**

Purpose: replace the current scalar actor feature contract
`lane_offset, heading_error, route_target_y` with inference-available estimates:

- `estimated_lane_offset`
- `estimated_heading_error`
- `route_target_y`

This plan exists because Round90 used scalar actor features, while the first PPO clean baseline disabled them. If the scalar features are derived from CARLA auxiliary ground truth at inference time, they are privileged simulator signals and should not be treated as ordinary sensor inputs. The better comparison setup is to make the scalar features available from the same deployed sensors, then use that same representation for RCS-AC, PPO-matched-input, and later SAC-matched-input baselines.

## Current State

Current model path:

- `rcs_ac_first/models/rcs_ac_model.py`
- `RCSACFirstModel._scalar_actor_features()`

Current scalar actor feature vector:

```text
[lane_offset, heading_error, route_y]
```

Current source:

- `lane_offset`: `auxiliary.risk_stats["lane_offset"]`
- `heading_error`: `auxiliary.risk_stats["heading_error"]`
- `route_y`: `route_input[:, 1]`

Current use:

- If `model.use_scalar_actor_features: true`, the vector is normalized, passed through `scalar_actor_adapter`, and added to the actor input.
- Round90 has `use_scalar_actor_features: true`.
- PPO strict baseline currently has scalar actor features disabled.

Important interpretation:

- `route_target_y` is part of the route observation and is acceptable as an inference input if route is part of the policy observation contract.
- `lane_offset` and `heading_error` from CARLA auxiliary/risk stats are not acceptable as normal inference inputs for a fair non-oracle baseline, because they encode simulator-derived localization/road geometry.
- If the existing values are used during evaluation, the run must be labeled `oracle-scalar` or `privileged-scalar`, not clean sensor-only inference.

## Target Contract

Replace the actor-facing scalar vector with:

```text
[estimated_lane_offset, estimated_heading_error, route_target_y]
```

Definitions:

- `estimated_lane_offset`: signed lateral offset estimate inferred from deployed sensor observations, primarily RGB road/lane perception and route geometry.
- `estimated_heading_error`: signed heading/yaw alignment estimate inferred from deployed sensor observations, primarily RGB road/lane orientation and route geometry.
- `route_target_y`: signed lateral coordinate of the next route target from the existing route observation. This remains route-derived, not privileged CARLA risk auxiliary.

The actor should not know whether the first two scalars came from ground truth labels or sensor estimation. The policy input contract should only expose estimated values at inference time.

## Sign Convention

Before implementation, the sign convention must be frozen and audited:

- Positive `estimated_lane_offset` must mean the same physical side as the current `lane_offset` training target.
- Positive `estimated_heading_error` must mean the same rotational error direction as the current `heading_error` training target.
- Corrective steering labels must be checked against these signs:
  - left-side drift should require rightward corrective steering under the repo's steering convention.
  - right-side drift should require leftward corrective steering under the repo's steering convention.

Do not rely on aggregate sign rate alone. Always report left/right splits.

Required audit metrics:

- `estimated_lane_offset_sign_agreement`
- `estimated_heading_error_sign_agreement`
- `early_severe_lane_offset_sign_agreement`
- `early_severe_heading_error_sign_agreement`
- `left_corrective_sign_rate`
- `right_corrective_sign_rate`

Recommended pass thresholds before CARLA evaluation:

- lane offset sign agreement >= 0.90
- heading error sign agreement >= 0.85
- severe early drift sign agreement >= 0.95
- left/right corrective sign rate >= 0.98 when using corrected recovery labels

## Estimation Sources

### RGB Perception Masks

Use the existing RGB perception auxiliary head:

- `pred_road_mask_logits`
- `pred_lane_edge_mask_logits`

Current compact visual features already compute:

- road mask mass and lower-half mass
- lane mask mass and lower-half mass
- road/lane center x
- road/lane vertical distribution
- `visual_lane_center_error = lane_x - road_x`

This should be extended from generic compact features into explicit scalar estimates.

Suggested raw signals:

- `road_center_x_lower`
- `lane_center_x_lower`
- `road_center_x_bottom`
- `lane_center_x_bottom`
- `road_center_slope`
- `lane_center_slope`
- `road_mass_lower`
- `lane_mass_lower`
- `lane_left_right_balance`

The first implementation can be a learned estimator head that consumes RGB perception mask-derived features. Avoid hardcoded runtime steering rules.

### Route Geometry

Use route observation as context, not as a direct steering override:

- `route_target_x`
- `route_target_y`
- route heading deltas if present in the route representation
- maneuver hint if present

`route_target_y` remains in the scalar vector because it is already part of the deployed route input. It should be normalized and documented as route-derived, not risk-stat-derived.

### LiDAR And Ego State

LiDAR/ego can optionally support the estimator later, but the first clean implementation should keep scope narrow:

- RGB masks estimate lane/road alignment.
- Route target disambiguates intended direction.
- Ego state can normalize speed-dependent noise if needed.

Do not add new privileged map/lane center inputs to the inference policy.

## Recommended Architecture

Add a small estimator module, for example:

```text
RGB mask logits
  -> explicit visual geometry features
  -> estimated_scalar_head
  -> [estimated_lane_offset, estimated_heading_error]
```

Then build:

```text
estimated_scalar_actor_features =
  [
    estimated_lane_offset / scalar_actor_lane_offset_scale,
    estimated_heading_error / scalar_actor_heading_error_scale,
    route_target_y / scalar_actor_route_y_scale,
  ]
```

Configuration should distinguish source:

```yaml
model:
  use_scalar_actor_features: true
  scalar_actor_feature_source: estimated
  scalar_actor_feature_dim: 3
```

Allowed source values:

- `disabled`: no scalar actor features.
- `oracle_auxiliary`: current CARLA auxiliary/risk-stat values. Only for ablation and upper-bound diagnosis.
- `estimated`: inference-available sensor estimate. This should become the default for fair comparison.

Do not silently reuse `oracle_auxiliary` when a config claims sensor-only inference.

## Training Supervision

Use privileged simulator values only as offline supervision targets:

- target `lane_offset` supervises `estimated_lane_offset`
- target `heading_error` supervises `estimated_heading_error`

This must be documented as **privileged offline supervision**, not privileged inference.

Suggested losses:

```text
estimated_lane_offset_loss = smooth_l1(estimated_lane_offset, target_lane_offset)
estimated_heading_error_loss = smooth_l1(estimated_heading_error, target_heading_error)
estimated_scalar_sign_loss = optional sign/alignment penalty on severe drift rows
```

The actor can consume estimated scalars during training. To avoid train/eval mismatch:

- Do not feed ground-truth lane offset to the actor in normal training if the final policy will use estimates.
- If teacher forcing is used for a warmup, label the run separately and anneal to estimated scalars before accepting it as a baseline.

Recommended training stages:

1. Train or warm up RGB perception masks.
2. Train estimated scalar head against offline privileged targets.
3. Train BC actor using estimated scalar features, not oracle scalar features.
4. Run offline actor sign audit with estimated scalar features.
5. Run CARLA 512 and 2048 clean eval.
6. Only after clean safety passes, use the same representation for robustness comparisons.

## Baseline Fairness Matrix

Keep these baselines separate:

| Baseline | Scalar features | RCS-AC modules | Interpretation |
| --- | --- | --- | --- |
| PPO-strict | disabled | disabled | Plain same-sensor PPO lower-complexity baseline |
| PPO-oracle-scalar | oracle auxiliary | disabled | Upper-bound/diagnostic only, not a fair deployed baseline |
| PPO-matched-estimated-scalar | estimated scalars | disabled | Fair representation-matched PPO baseline |
| RCS-AC-oracle-scalar | oracle auxiliary | enabled | Diagnostic only if used |
| RCS-AC-estimated-scalar | estimated scalars | enabled | Preferred final RCS-AC comparison model |

The main robustness claim should compare:

```text
RCS-AC-estimated-scalar
vs
PPO-matched-estimated-scalar
vs
SAC-matched-estimated-scalar
```

The `PPO-strict` baseline can still be reported, but it should not be the only baseline if RCS-AC uses extra actor-facing scalar representation.

## Implementation Steps

### Step 1: Add explicit config

Add config fields:

```yaml
model:
  scalar_actor_feature_source: estimated
  estimated_scalar_hidden_dim: 32
  estimated_scalar_loss_weight: 1.0
  estimated_scalar_sign_loss_weight: 0.0
```

Keep backward compatibility by mapping existing configs with `use_scalar_actor_features: true` and no source to `oracle_auxiliary`, then document those runs as old/oracle-scalar runs.

### Step 2: Build visual geometry features

Extend the perception feature extractor or add a sibling utility that returns named geometry values:

- `visual_road_center_x`
- `visual_lane_center_x`
- `visual_lane_center_error`
- `visual_road_heading_proxy`
- `visual_lane_heading_proxy`
- `visual_mask_confidence`

These are diagnostics and estimator inputs, not steering rules.

### Step 3: Add estimated scalar head

Add a small MLP head that predicts:

```text
estimated_lane_offset
estimated_heading_error
```

Inputs may include:

- visual geometry features
- compact RGB perception actor features
- route target features

The head output should be finite-clamped only for numeric stability. Do not clamp in a way that hides sign errors.

### Step 4: Route scalar source through actor

Change scalar feature construction from:

```text
risk_stats.lane_offset
risk_stats.heading_error
route[1]
```

to:

```text
estimated_lane_offset
estimated_heading_error
route[1]
```

when `scalar_actor_feature_source: estimated`.

When `scalar_actor_feature_source: oracle_auxiliary`, preserve current behavior for ablation.

### Step 5: Add diagnostics

Log both target and estimated scalar quality:

- `estimated_lane_offset_mean`
- `estimated_heading_error_mean`
- `estimated_lane_offset_mae`
- `estimated_heading_error_mae`
- `estimated_lane_offset_sign_agreement`
- `estimated_heading_error_sign_agreement`
- `estimated_scalar_actor_delta_l1_mean`
- `oracle_scalar_actor_delta_l1_mean` if oracle ablation is run

For CARLA eval, also log:

- scalar source used by the actor
- whether privileged auxiliary was consumed by actor inference
- side-split corrective sign rates

### Step 6: Add ablations

Minimum required ablations:

1. Round90 current behavior, normal eval.
2. Round90 with scalar actor features disabled.
3. Round90 with oracle scalar but RGB perception actor features disabled.
4. Estimated scalar candidate with estimated scalar enabled.
5. Estimated scalar candidate with scalar disabled.

These tell us whether scalar features are carrying the clean driving behavior and whether the estimated replacement preserves that behavior.

## Acceptance Gate

Before accepting `estimated` scalar features as the new comparison contract:

- offline estimated scalar sign audit must pass
- actor sign response audit must pass with estimated scalars
- CARLA 512 clean eval must have collision/offroad/severe early offroad/stuck `0/0/0/0`
- CARLA 2048 clean eval must have collision/offroad/severe early offroad/stuck `0/0/0/0`
- PPO-matched-estimated-scalar must be trained/evaluated under the same scalar input contract
- all manifests must state `scalar_actor_feature_source: estimated`

Do not promote a run if it only passes with `oracle_auxiliary`.

## Reporting Language

Use this wording in papers or experiment summaries:

> Lane offset and heading error are used as privileged offline supervision targets for an RGB/route-based scalar estimator. At deployment and evaluation time, the actor receives only estimated scalar features derived from policy-available observations.

Avoid this wording unless the run is explicitly oracle:

> The policy receives lane offset and heading error.

That sentence implies privileged inference unless the source is clearly described as estimated from onboard sensors.

## Risks

- RGB mask quality may be good in clean weather but degrade under adverse weather.
- Estimated scalar errors can produce wrong-direction recovery if sign convention is not audited.
- If the scalar estimator is too strong, it may become the main controller and obscure whether RCS-AC reliability/Shapley modules add robustness.
- If PPO does not receive the same estimated scalar contract, the comparison will mix representation advantage with algorithmic advantage.

Mitigation:

- Report scalar estimator quality under clean and corrupted conditions.
- Run matched-input PPO/SAC baselines.
- Keep RCS-AC module ablations separate from representation ablations.
- Never use estimated scalars as a hardcoded steering override.

## Next Recommended Work

1. Add scalar source config and diagnostics without changing behavior.
2. Run Round90 scalar ablation to measure current dependency.
3. Implement estimated scalar head from RGB perception and route features.
4. Train/audit estimated scalar quality offline.
5. Rebuild RCS-AC and PPO matched-input baselines using the estimated scalar contract.
