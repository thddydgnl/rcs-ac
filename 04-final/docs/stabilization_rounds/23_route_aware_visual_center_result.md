# Round 23 - Route-Aware Visual-Center Correction

Date: 2026-05-01

## Goal

Round 22 showed that route progress should not be improved with longitudinal floor losses or route fusion floors. Round 23 returned to the stronger Round 20 visual-center checkpoint and added a narrow route-aware visual correction loss around the previous route cap.

The intended behavior was:

- keep RGB visual-center correction active;
- apply extra supervision only in the route window where progress previously stalled;
- avoid weakening base BC or adding throttle/brake floors;
- only enter PPO if the configured BC gate passes.

## Code And Config Changes

Commit:

- `4d08ba7` - Add route-aware visual center BC loss

Main files:

```text
rcs_ac_first/scripts/train_bc.py
tests/test_train_bc_perception.py
configs/bc_frame_rgb_perception_actor_features_v2_route_aware_visual_center_round23.yaml
```

New BC terms:

```text
route_aware_visual_center_loss
route_aware_visual_center_sign_rate
route_aware_visual_center_correction_rate
route_aware_visual_center_mask_rate
route_aware_visual_center_signed_delta_mean
```

Config:

```text
route_aware_visual_center_loss_weight: 0.20
route_aware_visual_center_route_min: 0.205
route_aware_visual_center_route_max: 0.36
route_aware_visual_center_margin: 0.007
ppo_gate_corrective_sign_source: visual_center
```

Local verification:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
17 passed

python -m py_compile rcs_ac_first/scripts/train_bc.py tests/test_train_bc_perception.py
PASS

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_route_aware_visual_center_round23.yaml --device cpu
PASS
```

Windows syntax verification:

```text
py -3.12 -B -m py_compile rcs_ac_first/scripts/train_bc.py tests/test_train_bc_perception.py
PASS
```

## BC Training

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.train_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_aware_visual_center_round23.yaml --dataset data/expert/rgb_perception_actor_features_v2_corrective_delta_mix_928_round19.npz --init-checkpoint checkpoints/rgb_perception_actor_features_v2_visual_center_sign_bc_round20/last.pt --init-strict true --run-name rgb_perception_actor_features_v2_route_aware_visual_center_bc_round23 --device cuda
```

Result:

```text
epochs: 5
final val bc_loss: 0.006220
final val steer_mse: 0.003998
final val longitudinal_mse: 0.019549
visual_zero_action_l1: 0.036834
visual_zero_steer_delta: 0.010616
visual_zero_corrective_steer_delta: 0.010884
target_corrective_delta_sign_rate: 0.908214
visual_zero_target_delta_sign_rate: 0.923855
visual_center_correction_rate: 1.000000
route_aware_visual_center_loss: 0.000000
route_aware_visual_center_sign_rate: 1.000000
route_aware_visual_center_correction_rate: 1.000000
route_aware_visual_center_mask_rate: 0.376344
route_aware_visual_center_signed_delta_mean: 0.012128
```

## CARLA Eval 1

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_aware_visual_center_round23.yaml --checkpoint checkpoints/rgb_perception_actor_features_v2_route_aware_visual_center_bc_round23/last.pt --run-name rgb_perception_actor_features_v2_route_aware_visual_center_eval_512_round23 --num-steps 512 --device cuda --host 127.0.0.1 --port 2000
```

Summary:

```text
completed_steps: 512
pass_status: PASS
ppo_gate_status: BLOCK
max_route_progress: 0.743590
mean_route_progress: 0.324419
done_reason_counts: {"collision": 1}
collision_rate: 0.001953
offroad_rate: 0.000000
early_offroad_count: 0
corrective_need_count: 19
visual_center_correction_rate: 1.000000
mean_visual_zero_action_l1: 0.016194
mean_visual_zero_steer_delta: 0.009930
mean_corrective_visual_zero_steer_delta: 0.009645
ppo_gate_fail_reasons: ["corrective_need_count 19 < 32"]
```

## CARLA Eval 2

Command:

```text
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs/bc_frame_rgb_perception_actor_features_v2_route_aware_visual_center_round23.yaml --checkpoint checkpoints/rgb_perception_actor_features_v2_route_aware_visual_center_bc_round23/last.pt --run-name rgb_perception_actor_features_v2_route_aware_visual_center_eval_512_round23_repeat --num-steps 512 --device cuda --host 127.0.0.1 --port 2000
```

Summary:

```text
completed_steps: 512
pass_status: PASS
ppo_gate_status: BLOCK
max_route_progress: 0.743590
mean_route_progress: 0.324619
done_reason_counts: {"collision": 1}
collision_rate: 0.001953
offroad_rate: 0.000000
early_offroad_count: 0
corrective_need_count: 16
visual_center_correction_rate: 1.000000
mean_visual_zero_action_l1: 0.016570
mean_visual_zero_steer_delta: 0.010007
mean_corrective_visual_zero_steer_delta: 0.012056
ppo_gate_fail_reasons: ["corrective_need_count 16 < 32"]
```

## Comparison

Round 21 baseline:

```text
max_route_progress: 0.256410
mean_route_progress: 0.118389
offroad_count: 2
early_offroad_count: 0
visual_center_correction_rate: 1.000000
```

Round 23:

```text
eval 1:
  max_route_progress: 0.743590
  mean_route_progress: 0.324419
  offroad_count: 0
  early_offroad_count: 0
  collision_count: 1
  visual_center_correction_rate: 1.000000

eval 2:
  max_route_progress: 0.743590
  mean_route_progress: 0.324619
  offroad_count: 0
  early_offroad_count: 0
  collision_count: 1
  visual_center_correction_rate: 1.000000
```

## Decision

Do not enter PPO yet because the configured BC gate still blocks on `corrective_need_count < 32`.

This blocker is likely conservative for the new behavior: corrective windows became rarer because the route and offroad behavior improved sharply. The model now reaches `0.743590` route progress in repeated 512-step evals with zero offroad and stable visual-center correction.

Round 23 is the current best stabilization round. The next round should either:

- update the PPO gate so low `corrective_need_count` is not treated as failure when route progress, offroad, and visual-center correction are strong; or
- run a longer repeated eval to confirm whether the remaining collision is deterministic before allowing PPO.
