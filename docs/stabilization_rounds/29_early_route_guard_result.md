# Round 29 Early Route Guard Result

Date: 2026-05-02

## Goal

Round28 failed before the prior near-cluster training window:

- `route_progress=0.076923`
- `lane_offset≈-3m`
- `heading_error≈+23..29deg`
- repeated early offroad without collision

Round29 tested whether early-route recovery data plus an early-route signed correction loss can stabilize this failure.

## Code/config changes

- Added optional `early_route_offset_agreement_*` BC metrics and loss in `rcs_ac_first/scripts/train_bc.py`.
- Kept existing `collision_cluster_route_offset_agreement_*` metrics separate.
- Added Round29 configs:
  - `configs/bc_frame_rgb_perception_actor_features_v2_early_route_recovery_collect_round29.yaml`
  - `configs/bc_frame_rgb_perception_actor_features_v2_early_route_offset_round29.yaml`
  - `configs/bc_frame_rgb_perception_actor_features_v2_early_route_guard_round29.yaml`
- Added unit coverage for:
  - early-route negative-lane / positive-heading correction sign
  - early-route loss disabled after the early route window
  - collision-cluster and early-route metrics not overwriting each other

Local verification:

```text
python -m pytest tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py -q
27 passed
python -m py_compile rcs_ac_first/scripts/train_bc.py
PASS
```

## Data collection

Collected two early-route recovery datasets on Windows/CARLA.

Round24 policy collection:

```text
output: data/expert/rgb_perception_actor_features_v2_early_route_recovery_256_round29.npz
accepted: 256 / 1862
route_progress: 0.025641-0.076923
target_corrective_delta_abs_mean: 0.038891
```

This set did not contain severe early drift; max observed lane offset was only about `-0.64m`.

Round28 failure-policy collection:

```text
output: data/expert/rgb_perception_actor_features_v2_early_route_failure_recovery_256_round29.npz
accepted: 256 / 1362
route_progress: 0.025641-0.076923
target_corrective_delta_abs_mean: 0.087752
```

Broad early failure coverage:

```text
lane < -0.55 and heading > 3: 52 rows, right-delta sign rate 1.0
lane > 0.55 and heading < -3: 45 rows, left-delta sign rate 1.0
```

Combined dataset:

```text
output: data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
inputs: [3744, 256, 256]
```

## BC result

Training command used Round24 as the initialization checkpoint:

```text
checkpoint: checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt
run: rgb_perception_actor_features_v2_early_route_offset_bc_round29
val_loss: 0.011058
```

Final validation highlights:

```text
visual_zero_action_l1: 0.098373
visual_zero_steer_delta: 0.021650
target_corrective_delta_sign_rate: 0.911080
early_route_offset_agreement_mask_rate: 0.018779
early_route_offset_agreement_negative_sign_rate: 0.098592
collision_cluster_route_offset_agreement_sign_rate: 0.617393
```

Interpretation: BC learned strong RGB sensitivity and target corrective delta imitation, but the early-route RGB-vs-zero signed correction metric remained weak.

## 512 eval without guard

Run:

```text
rgb_perception_actor_features_v2_early_route_offset_eval_512_round29
```

Result:

```text
collision_count: 0
done_reason_counts: {"offroad": 7}
early_offroad_count: 7
severe_early_offroad_count: 7
severe_early_terminal_offroad_count: 7
max_route_progress: 0.076923
mean_route_progress: 0.034856
mean_action_jitter_l1: 0.081806
ppo_gate_status: BLOCK
```

The raw model still reproduced the Round28 early offroad pattern.

## 512 eval with lateral recovery guard

Run:

```text
rgb_perception_actor_features_v2_early_route_guard_eval_512_round29
```

Result:

```text
collision_count: 0
done_reason_counts: {"max_steps": 1}
early_offroad_count: 0
severe_early_offroad_count: 0
max_route_progress: 0.564103
mean_route_progress: 0.271484
mean_action_jitter_l1: 0.073864
ppo_gate_status: BLOCK
```

Only remaining blocker was jitter.

## 512 eval with guard + stronger smoothing

Run:

```text
rgb_perception_actor_features_v2_early_route_guard_smooth_eval_512_round29
```

Result:

```text
collision_count: 1
done_reason_counts: {"collision": 1}
early_offroad_count: 0
severe_early_offroad_count: 0
max_route_progress: 0.435897
mean_route_progress: 0.185697
mean_action_jitter_l1: 0.066269
ppo_gate_fail_reasons: []
ppo_gate_status: PASS
```

The guard solved early offroad and smoothing passed the configured PPO gate, but a mid-route collision remained.

## PPO 2048

Run:

```text
rgb_perception_actor_features_v2_early_route_guard_ppo2048_round29_20260502T030140
```

Result:

```text
total_steps: 2048
offroad episodes: 7
collision episodes: 1
best route_progress during PPO: 0.564
typical offroad route_progress: 0.103-0.179
```

PPO 2048 was not stable. It degraded from the guarded 512 eval and repeatedly ended offroad.

## Conclusion

Round29 did make one real improvement: lateral recovery guard plus stronger smoothing prevents the severe early offroad failure in 512-step evaluation.

It did not make the raw BC policy stable, and PPO 2048 is still not ready. The remaining issue is not a simple sign bug: the model often steers in the correct direction, but the correction is too late and too speed-coupled. PPO then amplifies mid-route offroad/collision risk.

## Next recommendation

Next round should not go directly deeper into PPO. Use the guarded config as a safety baseline, then focus on:

- collecting more early-route rows with larger lane offset before terminal offroad
- adding an explicit early speed/brake correction loss for `route_progress <= 0.10`
- making the PPO gate reject collision even when the current gate list is empty
- evaluating one 2048 guarded BC rollout before PPO, so PPO is only started if the guarded policy is stable for longer than 512 steps
