# Round 22 - Route Progress Floor Probes

Date: 2026-05-01

## Goal

Round 21 removed the stale route-offset sign gate as a blocker. The remaining PPO blockers were:

```text
max_route_progress 0.256410 < 0.282051
mean_route_progress 0.118389 < 0.132762
mean_visual_zero_action_l1 0.011208 < 0.012000
```

Round 22 tested whether route progress could be improved with narrow floor-style interventions while preserving the Round 20 visual-center correction behavior.

## Code And Config Changes

Commits:

- `2ca9fca` - Add route progress longitudinal floor loss
- `4843bf1` - Add weaker route progress floor round 22 config
- `4a6dfe8` - Add route floor eval round 22 config

New configs:

```text
configs/bc_frame_rgb_perception_actor_features_v2_route_progress_floor_round22.yaml
configs/bc_frame_rgb_perception_actor_features_v2_route_progress_floor_v2_round22.yaml
configs/bc_frame_rgb_perception_actor_features_v2_route_floor_eval_round22.yaml
```

Local verification:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
15 passed

python -m py_compile rcs_ac_first/scripts/train_bc.py tests/test_train_bc_perception.py
PASS

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_route_progress_floor_round22.yaml --device cpu
PASS

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_route_progress_floor_v2_round22.yaml --device cpu
PASS

python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_route_floor_eval_round22.yaml --device cpu
PASS
```

Windows syntax verification:

```text
py -3.12 -B -m py_compile rcs_ac_first/scripts/train_bc.py tests/test_train_bc_perception.py
PASS
```

## Probe 1 - Longitudinal Floor V1

Change:

```text
route_progress_longitudinal_floor_loss_weight: 0.20
route_progress_longitudinal_floor_route_min: 0.10
route_progress_longitudinal_floor_route_max: 0.32
route_progress_longitudinal_floor_speed_max_mps: 3.5
route_progress_longitudinal_floor: 0.05
epochs: 5
```

BC validation:

```text
bc_loss: 0.039528
steer_mse: 0.004878
longitudinal_mse: 0.247428
visual_center_correction_rate: 1.000000
route_progress_longitudinal_floor_rate: 1.000000
route_progress_longitudinal_mean: 0.077130
```

CARLA eval:

```text
completed_steps: 512
ppo_gate_status: BLOCK
max_route_progress: 0.128205
mean_route_progress: 0.053836
offroad_count: 5
early_offroad_count: 5
corrective_need_count: 0
visual_center_correction_rate: 0.000000
```

Interpretation:

- The floor worked offline, but it pushed the policy away from expert longitudinal behavior too much.
- Early offroad became much worse.
- This setting should not be reused.

## Probe 2 - Longitudinal Floor V2

Change:

```text
route_progress_longitudinal_floor_loss_weight: 0.05
route_progress_longitudinal_floor_route_min: 0.10
route_progress_longitudinal_floor_route_max: 0.24
route_progress_longitudinal_floor_speed_max_mps: 3.0
route_progress_longitudinal_floor: -0.10
epochs: 2
```

BC validation:

```text
bc_loss: 0.015897
steer_mse: 0.004460
longitudinal_mse: 0.084521
visual_center_correction_rate: 1.000000
route_progress_longitudinal_floor_rate: 0.057348
route_progress_longitudinal_mean: -0.398687
```

CARLA eval:

```text
completed_steps: 512
ppo_gate_status: BLOCK
max_route_progress: 0.128205
mean_route_progress: 0.051232
offroad_count: 5
early_offroad_count: 5
corrective_need_count: 0
visual_center_correction_rate: 0.000000
```

Interpretation:

- Weakening the floor did not recover stability.
- The route ceiling got worse in nearly the same way as V1.
- Longitudinal floor supervision is the wrong lever for the current route-progress blocker.

## Probe 3 - Route Fusion Floor Eval

Change:

```text
fusion.route_weight_floor: 0.12
```

No BC training was run. The Round 20 checkpoint was evaluated under the higher route floor.

CARLA eval:

```text
completed_steps: 512
ppo_gate_status: BLOCK
max_route_progress: 0.256410
mean_route_progress: 0.101963
offroad_count: 2
early_offroad_count: 1
corrective_need_count: 63
visual_center_correction_rate: 1.000000
mean_visual_zero_action_l1: 0.011003
mean_visual_zero_steer_delta: 0.005139
mean_corrective_visual_zero_steer_delta: 0.006086
```

Interpretation:

- Visual-center correction survived.
- Route progress did not improve.
- Early offroad regressed from `0` to `1`.
- Raising route floor alone does not solve the route cap.

## Comparison

Baseline Round 21:

```text
max_route_progress: 0.256410
mean_route_progress: 0.118389
offroad_count: 2
early_offroad_count: 0
visual_center_correction_rate: 1.000000
```

Round 22 probes:

```text
longitudinal floor v1:
  max_route_progress: 0.128205
  mean_route_progress: 0.053836
  offroad_count: 5
  early_offroad_count: 5

longitudinal floor v2:
  max_route_progress: 0.128205
  mean_route_progress: 0.051232
  offroad_count: 5
  early_offroad_count: 5

route floor eval:
  max_route_progress: 0.256410
  mean_route_progress: 0.101963
  offroad_count: 2
  early_offroad_count: 1
```

## Decision

Do not enter PPO.

Round 22 shows that route progress should not be attacked by simply reducing brake or raising route fusion floor. Both interventions either damage early stability or fail to improve the route ceiling.

The next round should target route-aware visual correction rather than longitudinal or raw fusion floors:

- Keep `ppo_gate_corrective_sign_source: visual_center`.
- Keep the Round 20 checkpoint as the current best base.
- Add a route-window visual-center correction loss that is active around and beyond the current cap, e.g. `route_progress >= 0.205` and `<= 0.36`.
- Extend the route-aware correction window beyond `0.256410` so the model is trained on the segment it currently cannot pass.
- Do not reuse the Round 22 longitudinal floor configs except as negative controls.
