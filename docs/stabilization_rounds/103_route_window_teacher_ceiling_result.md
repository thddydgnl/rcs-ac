# Round103 Route-Window Teacher Ceiling

## Motivation

Round102 showed that a plain route-window longitudinal ceiling is not enough. The 512 clean eval passed, but 2048 still produced `offroad: 1` and action jitter increased. The route-bin analysis showed that the trained actor still produced a relatively high longitudinal mean around route progress `0.30-0.35`.

Round103 changes the training target shape: preserve the Round90 actor as a frozen teacher and only clip the teacher longitudinal target in the problematic route window. This should keep the actor closer to Round90 outside the speed-sensitive window while lowering the local target speed.

## Change

Added `route_progress_longitudinal_teacher_*` BC loss in `rcs_ac_first/scripts/train_bc.py`.

This is privileged offline supervision because it uses dataset `route_progress` as the loss mask. It is not a runtime target-speed override, steering rule, lane-offset based control rule, or lateral recovery assist.

Round103 config:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_round103_route_window_teacher_ceiling.yaml`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Frozen teacher checkpoint: same as init checkpoint unless overridden
- Trainable module: `recurrent_actor_critic.actor_head.2`
- Trainable action row: `[1]`
- Teacher target:
  - `route_progress_longitudinal_teacher_loss_weight: 0.45`
  - `route_progress_longitudinal_teacher_route_min: 0.25`
  - `route_progress_longitudinal_teacher_route_max: 0.40`
  - `route_progress_longitudinal_teacher_ceiling: -0.72`

## Verification

Local:

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`: PASS
- Round103 config load: PASS
- Tensor-level teacher clipping probe: PASS

Windows BC train:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_route_window_teacher_ceiling_bc_round103/last.pt`
- Trainable parameters: `514`
- Train loss: `0.321018`
- Val loss: `0.317988`
- Val teacher loss: `0.082682`
- Val teacher L1: `0.360465`
- Val teacher mask rate: `0.636150`
- Val teacher target mean: `-0.658842`

512 clean eval:

- Run: `rgb_perception_actor_features_v2_route_window_teacher_ceiling_eval_512_round103_20260504T191656`
- Completed steps: `512`
- Done reasons: `{"max_steps": 1}`
- Collision: `0`
- Offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- Mean route progress: `0.220202`
- Mean action jitter L1: `0.131299`
- Mean longitudinal jitter: `0.051523`
- Visual-center correction rate: `0.714286`
- PPO gate: `BLOCK`

2048 clean eval:

- Run: `rgb_perception_actor_features_v2_route_window_teacher_ceiling_eval_2048_round103_20260504T191813`
- Completed steps: `2048`
- Done reasons: `{"max_steps": 3, "offroad": 1}`
- Collision: `0`
- Offroad: `1`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- Mean route progress: `0.191294`
- Mean action jitter L1: `0.147069`
- Mean longitudinal jitter: `0.066491`
- Visual-center correction rate: `0.524862`
- PPO gate: `BLOCK`

2048 route-bin detail:

| route bin | rows | speed mean | target mean | policy longitudinal mean | done |
| --- | ---: | ---: | ---: | ---: | --- |
| `0.25-0.30` | 267 | `7.083` | `9.109` | `-0.717` | `{"offroad": 1}` |
| `0.30-0.35` | 182 | `7.935` | `9.952` | `-0.641` | none |
| `0.35-0.40` | 163 | `8.230` | `10.798` | `-0.564` | `{"max_steps": 1}` |

## Verdict

BLOCK.

Round103 improved the 512 visual-center correction rate and reduced the `0.30-0.35` target mean versus Round102, but it moved the remaining 2048 offroad into `0.25-0.30` and increased action/longitudinal jitter. Do not promote Round103 over Round90.

## Interpretation

The remaining failure is not solved by simply lowering longitudinal output in the broad route window. The offroad can shift earlier while jitter increases. This suggests the late-window issue couples longitudinal timing with lateral recovery state, not just target-speed magnitude.

The next useful round should avoid another scalar-only longitudinal row update unless it explicitly conditions on lateral state or episode phase. Candidate directions:

- Audit the exact 2048 offroad episode rows around route `0.25-0.30` for lane offset, heading error, applied steer, target speed, and maneuver command.
- Build a failure-window relabel/filter dataset around the actual terminal context instead of applying a broad route-only mask.
- If adding a loss, condition it on `abs(lane_offset)`/`abs(heading_error)` and keep it offline-only; do not add a runtime override.
