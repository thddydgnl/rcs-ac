# Round90 Scalar Actor Feature Comparison

Date: 2026-05-03

Purpose: compare Round90's previous scalar actor feature path with the newly implemented estimated scalar actor feature path.

## Inputs

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Sample count: `4256`

The Round90 checkpoint was trained before `EstimatedScalarHead` existed. Loading it with `scalar_actor_feature_source: estimated` leaves the estimated scalar head newly initialized. Therefore this result measures the **untrained estimated-source gap**, not the performance of a trained estimated-scalar candidate.

## Compared Paths

- `oracle_auxiliary`: previous Round90 behavior. Actor receives `risk_stats.lane_offset`, `risk_stats.heading_error`, and `route_target_y`.
- `estimated`: new estimated scalar behavior. Actor receives `estimated_lane_offset`, `estimated_heading_error`, and `route_target_y`.
- `disabled`: scalar actor features removed.

## Scalar Quality

Estimated scalar quality against oracle targets:

- lane offset MAE: `1.032764`
- heading error MAE: `7.532130`
- lane offset sign agreement, all rows: `0.230114`
- heading error sign agreement, all rows: `0.389463`
- lane offset sign agreement, early-severe rows: `0.249161`
- heading error sign agreement, early-severe rows: `0.390101`
- lane offset sign agreement, left rows: `0.0`
- lane offset sign agreement, right rows: `1.0`

This means the untrained estimated scalar head currently has a strong sign mismatch, especially for left-drift rows.

## Actor Output Delta

Mean absolute action delta:

| Comparison | Steer L1 mean | Steer L1 p95 | Longitudinal L1 mean | Action L1 mean |
| --- | ---: | ---: | ---: | ---: |
| oracle vs disabled | `0.155236` | `0.446837` | `0.027680` | `0.091458` |
| estimated vs disabled | `0.049750` | `0.146462` | `0.008075` | `0.028912` |
| oracle vs estimated | `0.109725` | `0.327831` | `0.027101` | `0.068413` |

Early-severe steer delta:

| Comparison | Steer L1 mean | Steer L1 p95 |
| --- | ---: | ---: |
| oracle vs estimated | `0.124170` | `0.343992` |
| oracle vs disabled | `0.175425` | `0.470324` |
| estimated vs disabled | `0.054521` | `0.151172` |

## Interpretation

Round90 relies materially on the old scalar actor feature path:

- Removing scalar features changes steering by mean `0.155236`.
- Replacing oracle scalar with the untrained estimated scalar still changes steering by mean `0.109725`.
- The untrained estimated scalar path is much closer to scalar-disabled than to oracle-scalar behavior.

The next required step is not CARLA promotion. The estimated scalar head first needs offline supervised training/audit until sign agreement is acceptable, then actor sign audit and 512/2048 clean eval can be attempted.

Raw outputs:

- `summary.json`
- `action_delta_summary.csv`
