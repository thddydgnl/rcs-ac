# Round 25 - Collision Context Logging

Date: 2026-05-01

## Mode

This round used the Lite Subagent Workflow.

The round stayed Lite because it was diagnostic only:

```text
No training
No PPO
No reward/control/action/model changes
No runtime assist
No gate relaxation
```

## Starting Point

Round 24 produced a useful but non-promotable safety patch:

```text
patched 2048:
  ppo_gate_status: PASS
  collision_count: 5
  offroad_count: 2
  early_offroad_count: 0
  route_visual_center_sign_agreement_rate: 0.395161
```

Decision from Round 24:

```text
Do not enter PPO.
Add collision-context logging or collision-window aggregation before changing training again.
Recurring collision clusters were near route_progress 0.179487 and 0.743590.
```

## Sub-Agent Consensus

Agent A:

```text
Latest failed run is the Round 24 patched 2048 retry.
Configured gate passed, but decision blocks because collision_count=5.
The narrow next experiment is collision-context logging / window aggregation.
```

Agent C:

```text
Add collision_count, collision clusters, and pre-collision window aggregates.
No promotion from ppo_gate_status PASS when collision_count is nonzero.
Stop if repeated collision appears at known clusters.
```

Agent B:

```text
evaluate_bc.py logging path is behavior-neutral after env.step.
CSV additions must be numeric/bool.
Summary JSON can include event lists and cluster stats.
No blocker for a minimal logging-only patch.
```

## Code Change

Commit:

- `5f631c1` - Add BC eval collision context logging

Files:

```text
rcs_ac_first/scripts/evaluate_bc.py
tests/test_evaluate_bc_gate.py
```

Added CSV fields:

```text
collision_count_event
collision_route_delta_0179487
collision_route_delta_0743590
collision_window_0179487
collision_window_0743590
near_collision_cluster
```

Added summary fields:

```text
collision_count
collision_context_window_steps
collision_cluster_counts
collision_events
near_collision_cluster_row_count
near_collision_cluster_route_offset_corrective_sign_rate
near_collision_cluster_visual_center_correction_rate
near_collision_cluster_route_visual_agreement_rate
```

The collision context window is 32 eval steps.

Local verification:

```text
python -m py_compile rcs_ac_first/scripts/evaluate_bc.py tests/test_evaluate_bc_gate.py
PASS

PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_evaluate_bc_gate.py -q
3 passed

PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
18 passed
```

## Diagnostic Eval

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --checkpoint checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt --run-name rgb_perception_actor_features_v2_route_offset_agreement_collision_context_eval_2048_round25 --num-steps 2048 --device cuda --host 127.0.0.1 --port 2000
```

Result:

```text
completed_steps: 2048
pass_status: PASS
ppo_gate_status: BLOCK
ppo_gate_fail_reasons: ["offroad_count 3 > 2"]
done_reason_counts: {"collision": 5, "offroad": 3}
collision_count: 5
collision_cluster_counts: {"0179487": 4, "other": 1}
max_route_progress: 0.666667
mean_route_progress: 0.153521
early_offroad_count: 0
route_offset_corrective_sign_rate: 0.474074
route_visual_center_sign_agreement_rate: 0.474074
visual_center_correction_rate: 1.000000
```

Near-cluster aggregate:

```text
near_collision_cluster_row_count: 134
near_collision_cluster_route_offset_corrective_sign_rate: 0.000000
near_collision_cluster_route_visual_agreement_rate: 0.000000
near_collision_cluster_visual_center_correction_rate: 1.000000
```

## Collision Events

```text
step 465, episode 0:
  cluster: other
  route_progress: 0.666667
  lane_offset: +1.290337
  heading_error: -3.762321
  window_route_offset_corrective_sign_rate: 0.000000
  window_route_visual_center_sign_agreement_rate: 0.000000
  window_visual_center_correction_rate: 1.000000

step 869, episode 2:
  cluster: 0179487
  route_progress: 0.179487
  lane_offset: +1.717111
  heading_error: -16.471272
  window_route_offset_corrective_sign_rate: 0.000000
  window_route_visual_center_sign_agreement_rate: 0.000000
  window_visual_center_correction_rate: 1.000000

step 1042, episode 3:
  cluster: 0179487
  route_progress: 0.179487
  lane_offset: +1.698240
  heading_error: -17.149258
  window_route_offset_corrective_sign_rate: 0.000000
  window_route_visual_center_sign_agreement_rate: 0.000000
  window_visual_center_correction_rate: 0.968750

step 1674, episode 6:
  cluster: 0179487
  route_progress: 0.179487
  lane_offset: +1.713478
  heading_error: -16.606350
  window_route_offset_corrective_sign_rate: 0.000000
  window_route_visual_center_sign_agreement_rate: 0.000000
  window_visual_center_correction_rate: 1.000000

step 1845, episode 7:
  cluster: 0179487
  route_progress: 0.179487
  lane_offset: +1.715613
  heading_error: -16.574307
  window_route_offset_corrective_sign_rate: 0.000000
  window_route_visual_center_sign_agreement_rate: 0.000000
  window_visual_center_correction_rate: 1.000000
```

## Interpretation

The diagnostic logging confirms that the dominant collision cluster is not caused by a missing RGB visual-center response. The visual-center correction is active almost perfectly before collisions.

The failure is that the visual-center proxy is aligned against the route/lane offset at the recurring collision cluster:

```text
visual_center_correction: active
route_offset_corrective_sign: inactive/wrong
route_visual_center_agreement: inactive/wrong
```

At route_progress `0.179487`, the policy repeatedly approaches the same right-offset, negative-heading state:

```text
lane_offset ~= +1.70
heading_error ~= -16 to -17
speed at collision ~= 2.8 to 4.8 km/h
visual_zero_signed_steer_delta ~= +0.026
```

That positive visual steer delta is treated as visually corrective, but it is not route-offset corrective for this state.

## Decision

Do not enter PPO.

Round 25 is a successful diagnostic round, not a stabilization improvement. It confirms the next fix should target the route/visual sign disagreement around the `0.179487` collision cluster.

## Next Round

Recommended next experiment:

```text
Add a narrow collision-cluster BC safety loss or config-only route-offset agreement window around route_progress 0.16-0.20, lane_offset > +1.5, heading_error < -8.
The goal is to force visual-zero signed steer delta to become route-offset corrective in this cluster.
```

Preserve:

```text
No runtime steering assist.
No PPO.
No route-progress throttle/brake floor.
No gate relaxation.
Keep visual-center correction active, but do not let it override route-offset safety in this cluster.
```
