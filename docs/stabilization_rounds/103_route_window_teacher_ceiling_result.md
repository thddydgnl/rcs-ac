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

Pending Windows BC train and clean eval.

