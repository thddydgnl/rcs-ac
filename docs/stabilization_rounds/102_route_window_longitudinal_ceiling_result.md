# Round102 Route-Window Longitudinal Ceiling

## Motivation

Round90 remains the clean baseline reference, but the 2026-05-04 re-verification exposed a current-runtime failure mode around route progress `0.25-0.35`: the global target-speed wrapper can remove offroad when capped to `16-18 km/h`, but that collapses route progress and should not be promoted as the final RCS-AC baseline.

The route-bin diagnostic in `round90_results/action_log_contract_fix_20260504/target_speed_route_bins.json` points to a narrower fix: preserve the Round90 steering policy and train only the longitudinal actor row so this route window is less aggressive under the normal `max_target_speed_kmh: 28.0` contract.

## Change

Added a training-time `route_progress_longitudinal_ceiling_*` BC loss in `rcs_ac_first/scripts/train_bc.py`.

This is offline privileged supervision because it uses dataset `route_progress` as a mask during BC training. It is not a runtime steering rule, target-speed override, lateral recovery assist, or lane-offset based control rule.

Round102 config:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_round102_route_window_longitudinal_ceiling.yaml`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Trainable module: `recurrent_actor_critic.actor_head.2`
- Trainable action row: `[1]` only, preserving steering row `[0]`
- Added loss:
  - `route_progress_longitudinal_ceiling_loss_weight: 0.25`
  - `route_progress_longitudinal_ceiling_route_min: 0.25`
  - `route_progress_longitudinal_ceiling_route_max: 0.35`
  - `route_progress_longitudinal_ceiling: -0.68`

## Local Verification

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`: PASS
- Config load for Round102: PASS
- Tensor-level loss probe: PASS, only rows in the route window were penalized.

## CARLA Results

Pending Windows BC train and clean eval.

