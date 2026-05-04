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

Windows BC train:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_failure_window_lateral_longitudinal_bc_round104/last.pt`
- Trainable parameters: `514`
- Train loss: `0.238370`
- Val loss: `0.196728`
- Val late-recovery absolute steer loss: `0.100906`
- Val late-recovery sign rate: `0.751174`
- Val late-recovery mask rate: `0.171362`
- Val early-severe longitudinal ceiling loss: `0.378089`
- Val early-severe longitudinal ceiling rate: `0.082176`
- Val early-center zero steer abs mean: `0.005704`

512 clean eval:

- Run: `rgb_perception_actor_features_v2_failure_window_lateral_longitudinal_eval_512_round104_20260504T192812`
- Completed steps: `512`
- Done reasons: `{"max_steps": 1}`
- Collision: `0`
- Offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- Mean route progress: `0.222656`
- Mean action jitter L1: `0.129396`
- Mean steer jitter: `0.076463`
- Mean longitudinal jitter: `0.052933`
- Corrective sign rate: `0.869565`
- Visual-center correction rate: `0.652174`
- PPO gate: `BLOCK`

2048 clean eval:

- Run: `rgb_perception_actor_features_v2_failure_window_lateral_longitudinal_eval_2048_round104_20260504T192929`
- Completed steps: `2048`
- Done reasons: `{"max_steps": 3, "offroad": 1}`
- Collision: `0`
- Offroad: `1`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- Mean route progress: `0.195701`
- Mean action jitter L1: `0.145403`
- Mean steer jitter: `0.082343`
- Mean longitudinal jitter: `0.063060`
- Corrective sign rate: `0.651934`
- Visual-center correction rate: `0.458564`
- PPO gate: `BLOCK`

2048 initial audit:

- Mean applied steer: `-0.078968`
- Left steer rate: `0.657500`
- Right steer rate: `0.340000`
- Final lane offset after initial audit window: `-0.697489`
- Final heading error after initial audit window: `8.798908`

2048 route-bin detail:

| route bin | rows | speed mean | target mean | policy longitudinal mean | done |
| --- | ---: | ---: | ---: | ---: | --- |
| `0.25-0.30` | 267 | `7.012` | `9.099` | `-0.718` | none |
| `0.30-0.35` | 164 | `9.143` | `11.112` | `-0.535` | `{"offroad": 1}` |
| `0.35-0.40` | 179 | `7.454` | `10.722` | `-0.571` | none |

## Verdict

BLOCK.

Round104 passed 512 safety and improved the short-run corrective sign rate, but it did not solve 2048 safety. It also worsened initial left steer bias and action jitter after opening the steering row. Do not promote Round104 over Round90.

## Interpretation

Opening the steering row with a narrow failure-window target is risky: it can improve one short corrective metric while degrading long-horizon correction balance. The remaining offroad also shifted back into route `0.30-0.35`, while the tuned `0.25-0.30` bin became safer but not enough for the full route sequence.

The next productive step is not another actor-head-only BC tweak. The likely bottleneck is missing state coverage for transition rows where route, lane offset, heading error, and speed jointly enter the failure regime. A better next round should collect or synthesize targeted recovery data for the actual pre-offroad state sequence, then audit label sign and speed targets before BC.
