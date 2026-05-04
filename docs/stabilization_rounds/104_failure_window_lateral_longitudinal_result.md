# Round104 Failure-Window Lateral + Longitudinal Probe

## Motivation

Rounds 102 and 103 showed that route-window longitudinal tuning alone does not solve the current 2048 clean offroad. Round103's terminal offroad occurred at:

- route progress: `0.282051`
- lane offset: `-2.804958`
- heading error: `42.213886`
- policy longitudinal: `-0.230216`
- target speed: `14.467625 km/h`
- speed: `10.154786 km/h`
- applied steer: `0.133588`

The actor was steering in the recovery direction, but too weakly and while still requesting a high target speed. Dataset audit also showed that the matching route/lane/heading rows contain mixed longitudinal labels, so route-only speed objectives are underconstrained.

## Change

Round104 is a narrow BC probe, not a runtime rule:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_round104_failure_window_lateral_longitudinal.yaml`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Trainable module: `recurrent_actor_critic.actor_head.2`
- Trainable action rows: `[0, 1]`
- Learning rate: `0.00001`
- Steer imitation weight: `0.0`
- Failure-window late recovery:
  - route `0.25-0.30`
  - `abs(lane_offset) >= 1.50`
  - `abs(heading_error) >= 10.0`
  - absolute steer margin `0.55`
- Failure-window longitudinal ceiling:
  - route `0.25-0.30`
  - severe mask from `early_severe_abs_lane_offset_min: 0.55`, `early_severe_abs_heading_error_min: 3.0`
  - longitudinal ceiling `-0.75`

This is privileged offline supervision using route/lane/heading masks. It is not a runtime steering override, lane controller, target-speed override, or lateral recovery assist.

## Verification

Pending Windows BC train and clean eval.

