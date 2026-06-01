# Round 30 Initial Steering Audit Result

Date: 2026-05-02

## Goal

Check whether the repeated "turns left immediately after spawn" failure is caused by a route maneuver command issue, or by a deeper policy/action/perception bias.

## Code Changes

- Added initial 80-step audit fields to `rcs_ac_first/scripts/evaluate_bc.py`.
- Logged raw/applied/smoothed steering, route target aliases, heading/lane offsets, and maneuver command.
- Added audit configs:
  - `configs/bc_frame_rgb_perception_actor_features_v2_round30_audit_cycle.yaml`
  - `configs/bc_frame_rgb_perception_actor_features_v2_round30_audit_fixed_straight.yaml`
  - `configs/bc_frame_rgb_perception_actor_features_v2_round30_audit_maneuver_off.yaml`
- Fixed a logging validation bug where string `maneuver_command` was treated as a numeric field.

## Remote Execution Notes

- Mac pushed commit `6b359a8` to `origin/codex-stabilization-rgb-v2`.
- Windows `git pull --ff-only` was blocked by Git Credential Manager in the non-interactive SSH session:
  - `fatal: Unable to persist credentials with the 'wincredman' credential store`
- To keep the round moving, the updated `evaluate_bc.py` was copied to Windows over SSH/SCP.
- CARLA was force-restarted before each evaluation.

## Checkpoint

```text
checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt
```

## Evaluation Results

### Cycle Maneuver

Run:

```text
rgb_perception_actor_features_v2_round30_audit_cycle_256_rerun
```

Output:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\rgb_perception_actor_features_v2_round30_audit_cycle_256_rerun_20260502T042157\bc_eval_log.csv
```

Key metrics:

```text
completed_steps: 256
done_reason_counts: {"offroad": 3}
early_offroad_count: 3
max_route_progress: 0.076923
ppo_gate_status: BLOCK
initial_audit.left_steer_rate: 0.871094
initial_audit.right_steer_rate: 0.128906
initial_audit.mean_steer: -0.103551
initial_audit.final_lane_offset: -2.906541
initial_audit.final_heading_error: 27.878848
initial_audit.maneuver_command_counts: {"straight": 126, "left": 65, "right": 65}
```

### Fixed Straight Maneuver

Run:

```text
rgb_perception_actor_features_v2_round30_audit_fixed_straight_256
```

Output:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\rgb_perception_actor_features_v2_round30_audit_fixed_straight_256_20260502T042332\bc_eval_log.csv
```

Key metrics:

```text
completed_steps: 256
done_reason_counts: {"offroad": 3}
early_offroad_count: 3
max_route_progress: 0.076923
ppo_gate_status: BLOCK
initial_audit.left_steer_rate: 0.855469
initial_audit.right_steer_rate: 0.144531
initial_audit.mean_steer: -0.093409
initial_audit.final_lane_offset: -3.005331
initial_audit.final_heading_error: 27.488788
initial_audit.maneuver_command_counts: {"straight": 256}
```

### Maneuver Disabled

Run:

```text
rgb_perception_actor_features_v2_round30_audit_maneuver_off_256
```

Output:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\rgb_perception_actor_features_v2_round30_audit_maneuver_off_256_20260502T042515\bc_eval_log.csv
```

Key metrics:

```text
completed_steps: 256
done_reason_counts: {"offroad": 3}
early_offroad_count: 3
max_route_progress: 0.076923
ppo_gate_status: BLOCK
initial_audit.left_steer_rate: 0.871094
initial_audit.right_steer_rate: 0.128906
initial_audit.mean_steer: -0.095287
initial_audit.final_lane_offset: -3.086543
initial_audit.final_heading_error: 30.263133
initial_audit.maneuver_command_counts: {"": 256}
```

## Interpretation

The immediate left turn is not explained by the left/right maneuver command itself.

Reason:

- Cycle, straight-only, and maneuver-disabled evaluations all show almost the same early left steering bias.
- Initial left steering rate remains around 85.5-87.1%.
- Final lane offset stays strongly negative around -2.9 to -3.1.
- All three runs terminate with three early offroad episodes and the same low route progress ceiling.

This points away from a simple "left command is injected at spawn" bug. The likely problem is earlier in the policy behavior under the initial observation distribution, or in the route/perception/action sign contract.

## Current Diagnosis

Most likely:

1. The policy has learned an initial negative-steer prior that dominates the early route/perception inputs.
2. The route target heading aliases are still slightly negative on average, but this alone does not explain the large lane drift.
3. The visual correction path still has zero effective corrective signal:
   - `corrective_need_count: 0`
   - `mean_corrective_visual_zero_steer_delta: 0.0`
   - `visual_center_correction_rate: 0.0`

The current RGB/perception path is therefore not acting as a reliable "if drifting left, steer right" correction mechanism.

## Next Round Recommendation

Do not continue PPO from this checkpoint yet.

Next stabilization round should run a small, code-level sign audit before more BC/PPO training:

1. Add an offline batch audit that feeds symmetric synthetic scalar features into the actor:
   - `lane_offset = -x, 0, +x`
   - `visual_center_error = -x, 0, +x`
   - `route_target_y = -x, 0, +x`
   - fixed RGB-zero or recorded RGB
2. Measure whether predicted steer changes in the expected opposite direction.
3. If sign response is wrong or near zero, fix the corrective loss/feature contract before training more.
4. If sign response is correct offline, inspect CARLA observation values at spawn and route target sign conventions.

The next useful experiment is not another large dataset round; it is a minimal actor response audit that proves whether the control policy can produce right correction when the vehicle is left of center.
