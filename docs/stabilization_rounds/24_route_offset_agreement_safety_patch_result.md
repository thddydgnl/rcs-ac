# Round 24 - Route-Offset Agreement Safety Patch

Date: 2026-05-01

## Trigger And Mode

This round started in Lite Subagent Workflow mode using the documented Mac-Windows remote CARLA workflow.

The initial no-code 2048 eval exposed a safety regression, so the round escalated to Full workflow before making any fix.

## Starting Point

Round 23 was the current best short-horizon result:

```text
512 eval x2
max_route_progress: 0.743590
mean_route_progress: 0.3244
offroad_count: 0
early_offroad_count: 0
collision_count: 1 each eval
visual_center_correction_rate: 1.000000
ppo_gate_status: BLOCK only because corrective_need_count < 32
```

The next documented question was whether the remaining collision was deterministic and whether the `corrective_need_count` blocker was stale.

## Lite Eval - Round 23 Checkpoint At 2048 Steps

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_aware_visual_center_round23.yaml --checkpoint checkpoints/rgb_perception_actor_features_v2_route_aware_visual_center_bc_round23/last.pt --run-name rgb_perception_actor_features_v2_route_aware_visual_center_eval_2048_round24 --num-steps 2048 --device cuda --host 127.0.0.1 --port 2000
```

Result:

```text
completed_steps: 2048
pass_status: PASS
ppo_gate_status: BLOCK
done_reason_counts: {"collision": 1, "offroad": 6}
early_offroad_count: 1
max_route_progress: 0.743590
mean_route_progress: 0.196627
corrective_need_count: 186
route_offset_corrective_sign_rate: 0.182796
route_visual_center_sign_agreement_rate: 0.182796
visual_center_correction_rate: 1.000000
mean_action_jitter_l1: 0.056718
ppo_gate_fail_reasons:
  - offroad_count 6 > 2
  - early_offroad_count 1 > 0
```

Safety pattern:

```text
Most offroads were right-side route/lane failures:
  lane_offset > +2.10
  heading_error negative
  visual_center_corrective_sign True
  route_visual_center_sign_agreement False

The visual-center metric stayed perfect, but route-offset agreement was very low.
```

Decision:

```text
Escalate from Lite to Full.
Do not enter PPO.
Treat the 2048 result as a real safety regression, not a gate-only issue.
```

## Full-Agent Findings

Agent A:

```text
The latest blocker is no longer only corrective_need_count.
The 2048 eval has actual offroad and early-offroad failures.
The next experiment should stay narrow and eval/train only.
```

Agent C:

```text
Visual-center correction must remain active past old route cap.
False-win risk: route progress improves while safety regresses.
Stop if offroad, early offroad, deterministic collision, or visual-center route-window failure appears.
```

Agent B:

```text
Runtime path is confirmed.
The gate result matches actual logged offroad events.
visual_center_correction_rate can look healthy while route-offset agreement is poor.
This is a real longer-horizon safety regression.
```

Agent D:

```text
Dominant failure cluster is right-side offroad with route/visual disagreement.
visual_center_correction_rate is healthy but route agreement is not.
Stop PPO; continue only with a narrow BC safety patch.
```

Agent J:

```text
Round 23 did not add hardcoded steering assist.
Runtime lateral_recovery is disabled.
Reject runtime steering/controller assist, route-progress throttle floors, and gate relaxation.
Config/training-only changes using existing logged labels are acceptable.
```

## Config-Only Safety Patch

Commit:

- `f665a76` - Add route offset agreement round 24 config

New config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml
```

Change:

```text
visual_zero_route_min: 0.10
visual_zero_route_max: 0.60
visual_zero_abs_lane_offset_min: 1.5
visual_zero_corrective_steer_margin: 0.012
visual_zero_corrective_sign_loss_weight: 0.30
visual_center_corrective_sign_loss_weight: 0.20
route_aware_visual_center_loss_weight: 0.15
```

No code, model architecture, reward, action contract, dataset, or runtime controller was changed.

Local verification:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
17 passed

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --device cpu
PASS
```

## BC Training

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.train_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --dataset data/expert/rgb_perception_actor_features_v2_corrective_delta_mix_928_round19.npz --init-checkpoint checkpoints/rgb_perception_actor_features_v2_route_aware_visual_center_bc_round23/last.pt --init-strict true --run-name rgb_perception_actor_features_v2_route_offset_agreement_bc_round24 --device cuda
```

Result:

```text
epochs: 5
final val bc_loss: 0.005694
final val steer_mse: 0.003601
final val longitudinal_mse: 0.018254
visual_zero_action_l1: 0.056119
visual_zero_steer_delta: 0.020045
visual_zero_corrective_steer_delta: 0.016969
target_corrective_delta_sign_rate: 0.946128
visual_zero_target_delta_sign_rate: 0.923855
visual_center_correction_rate: 1.000000
route_aware_visual_center_correction_rate: 1.000000
```

## Patched 512 Eval

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --checkpoint checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt --run-name rgb_perception_actor_features_v2_route_offset_agreement_eval_512_round24 --num-steps 512 --device cuda --host 127.0.0.1 --port 2000
```

Result:

```text
completed_steps: 512
done_reason_counts: {"max_steps": 1}
ppo_gate_status: BLOCK
ppo_gate_fail_reasons: ["corrective_need_count 25 < 32"]
max_route_progress: 0.769231
mean_route_progress: 0.340294
collision_rate: 0.000000
offroad_rate: 0.000000
early_offroad_count: 0
corrective_need_count: 25
route_offset_corrective_sign_rate: 0.280000
route_visual_center_sign_agreement_rate: 0.280000
visual_center_correction_rate: 1.000000
mean_action_jitter_l1: 0.068906
```

Interpretation:

```text
Short-horizon safety improved.
The route-offset agreement metric improved over the pre-patch 2048 value but is still low.
```

## Patched 2048 Eval

The first patched 2048 attempt failed before rollout during CARLA reset/load-world with a simulator timeout. CARLA was force restarted and the same eval was retried.

Retry command:

```text
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --checkpoint checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt --run-name rgb_perception_actor_features_v2_route_offset_agreement_eval_2048_round24_retry --num-steps 2048 --device cuda --host 127.0.0.1 --port 2000
```

Result:

```text
completed_steps: 2048
pass_status: PASS
ppo_gate_status: PASS
done_reason_counts: {"collision": 5, "offroad": 2}
early_offroad_count: 0
max_route_progress: 0.743590
mean_route_progress: 0.216572
corrective_need_count: 124
route_offset_corrective_sign_rate: 0.395161
route_visual_center_sign_agreement_rate: 0.395161
visual_center_correction_rate: 1.000000
mean_action_jitter_l1: 0.066185
mean_visual_zero_steer_delta: 0.018646
ppo_gate_fail_reasons: []
```

Terminal events:

```text
collision route 0.179487, lane_offset +1.713, heading -16.63
collision route 0.179487, lane_offset +1.702, heading -17.02
offroad   route 0.282051, lane_offset +2.910, heading -21.24
collision route 0.743590, lane_offset +1.503, heading -5.04
offroad   route 0.307692, lane_offset -2.816, heading +42.34
collision route 0.179487, lane_offset +1.716, heading -16.49
collision route 0.743590, lane_offset +1.730, heading -7.76
```

## Comparison

Pre-patch Round24 2048:

```text
collision_count: 1
offroad_count: 6
early_offroad_count: 1
mean_route_progress: 0.196627
route_visual_center_sign_agreement_rate: 0.182796
ppo_gate_status: BLOCK
```

Patched Round24 2048:

```text
collision_count: 5
offroad_count: 2
early_offroad_count: 0
mean_route_progress: 0.216572
route_visual_center_sign_agreement_rate: 0.395161
ppo_gate_status: PASS
```

## Decision

Do not enter PPO.

Although the configured PPO gate passes after the patch, the policy is not safe enough because collision count increased to `5`. The patch partially worked for the original offroad problem:

```text
offroad_count: 6 -> 2
early_offroad_count: 1 -> 0
route agreement: 0.1828 -> 0.3952
```

But it traded part of the failure mode into collisions, especially near:

```text
route_progress ~= 0.179487
route_progress ~= 0.743590
```

Round 24 should be treated as a useful diagnostic patch, not a promotable checkpoint.

## Next Round

The next round should stay Full or safety-focused Lite only if the change is purely diagnostic.

Recommended next step:

```text
Add collision-context logging or collision-window aggregation before changing training again.
Then target the recurring collision clusters at route_progress 0.179487 and 0.743590.
```

Reject:

```text
PPO entry from the patched checkpoint.
Gate relaxation based only on ppo_gate_status PASS.
Runtime steering assist or lateral_recovery controller.
Longitudinal throttle/brake floor keyed by route_progress.
```
