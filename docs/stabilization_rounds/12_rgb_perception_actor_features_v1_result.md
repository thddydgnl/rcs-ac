# Stabilization Round 12: RGB Perception Actor Features v1

## Hypothesis

Hardcoded `lateral_recovery` should be replaced by learned RGB road/lane perception features inside the policy.

This v1 round adds compact RGB perception actor features without adding a post-action steering residual and keeps `env.lateral_recovery.enabled=false`.

## Baseline

Current best engineering checkpoint:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt
```

Important prior diagnostic:

```text
Round 11 lateral_recovery-off eval:
  offroad: 7
  max_route_progress: 0.0769
  lateral_recovery_enabled_rate: 0.0
  result: controller dependency confirmed
```

## Changes

Implemented files:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\runtime\config.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\perception_aux.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\rcs_ac_model.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\__init__.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\evaluate_bc.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py
```

Created configs:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml
```

New model config:

```yaml
model:
  use_rgb_perception_actor_features: true
  rgb_perception_actor_feature_dim: 16
```

The env keeps:

```yaml
env:
  lateral_recovery:
    enabled: false
    turn_boost_enabled: false
```

## Mechanism

The existing RGB perception head predicts:

```text
pred_road_mask_logits
pred_lane_edge_mask_logits
```

v1 adds:

```text
road/lane mask probabilities
  -> compact raw visual summaries
  -> 16-dim compact visual actor feature
  -> actor input adapter with fused policy feature
  -> GRU actor-critic
```

This is not an action residual:

```text
No final steer correction is added after policy output.
No route-manager lane_offset/heading_error is used to change action.
```

The adapter preserves checkpoint compatibility by keeping the GRU input dimension at `fusion_dim`.

## Commands

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Local smoke:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml
```

128-step CARLA probe:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt --init-strict false --total-timesteps 128 --run-name rgb_perception_actor_features_v1_probe_128 --device auto
```

2048-step diagnostic:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt --init-strict false --total-timesteps 2048 --run-name rgb_perception_actor_features_v1_2048 --device auto
```

## Outputs

128 probe:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\logs\rgb_perception_actor_features_v1_probe_128_20260430T094250\training_log.csv
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\rgb_perception_actor_features_v1_probe_128_20260430T094250\last.pt
```

2048 diagnostic:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\logs\rgb_perception_actor_features_v1_2048_20260430T094332\training_log.csv
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\phase3_bc_clean_rgb_perception_actor_features_v1_env\rgb_perception_actor_features_v1_2048_20260430T094332\env_step_log.csv
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\rgb_perception_actor_features_v1_2048_20260430T094332\last.pt
```

## Results

Local smoke:

```text
PASS
device: cuda
rgb_perception_actor_features_shape: [4, 16]
visual_feature_zero_action_l1_mean: 0.00025944
loss/backward: PASS
checkpoint save/load: PASS
```

128 probe:

```text
init loaded keys: 77
skipped keys: 0
first episode: offroad
route_progress: 0.077
PPO update: 1
```

2048 diagnostic:

```text
updates: 16
episodes: 30
done_reason:
  offroad: 30
terminal_route_mean: 0.0778
terminal_route_max: 0.1026
early_offroad_under_0.20: 30
collision/stuck: 0/0
```

## Mechanism Activation

Hardcoded controller remained disabled:

```text
lateral_recovery_enabled_rate: 0.0
lateral_recovery_active_rate: 0.0
lateral_recovery_correction_max: 0.0
```

RGB visual feature path was active:

```text
compact_visual_enabled_mean: 1.0
compact_visual_feature_l1_mean: 0.0896
road_compact_feature_l1_mean: 0.3155
lane_compact_feature_l1_mean: 0.1138
compact_visual_actor_input_l1_mean: 0.3216
visual_actor_delta_l1_mean: 0.00223
```

Dense RGB aux targets were available:

```text
aux_mask_available_rate: 1.0
aux_road_mask_loss: 0.5845
aux_lane_edge_mask_loss: 0.2682
```

But prediction quality metrics were not useful yet:

```text
aux_road_mask_iou: 0.0
aux_lane_edge_mask_iou: 0.0
```

RGB fusion was effectively collapsed:

```text
fusion_rgb_mean: 0.0033
rgb_feature_map_l1_mean: 1.3482
```

## Regressions

Compared with current best engineering eval:

| Metric | Current best with lateral recovery | RGB actor features v1 |
|---|---:|---:|
| goal | 1 | 0 |
| max route progress | 1.0000 | 0.1026 |
| offroad | 0 in eval | 30 in 2048 diagnostic |
| lateral recovery active | about 82.5% | 0.0% |

This confirms the new feature path is not enough to replace the controller yet.

## Hardcoding / Assist Audit

This round is clean from the hardcoding perspective:

```text
No post-action lateral recovery.
No route-manager lane_offset/heading_error action correction.
No speed gate / turn boost.
```

Paper-relevance label for this round:

```text
ALGORITHM-ONLY-DIAGNOSTIC / FAILING-POLICY
```

It is not a performance candidate, but it is a valid diagnostic because the learned-policy path is being tested without the hardcoded controller.

## Gate Decision

2048 gate: **FAIL**

Reasons:

```text
offroad: 30
terminal_route_mean: 0.0778 < 0.55
terminal_route_max: 0.1026 < 0.9231
early_offroad_under_0.20: 30
visual_actor_delta_l1_mean only 0.00223
fusion_rgb_mean collapsed to 0.0033
```

4096: **SKIPPED**

Deterministic eval after 2048: **SKIPPED**

Reason:

```text
The 2048 diagnostic failed catastrophically and did not satisfy any promotion gate.
```

## Judgment

Implementation succeeded, but v1 is not behaviorally strong enough.

The compact visual features are present and connected to the actor, but their influence is too weak and the RGB fusion path is nearly ignored. The policy still collapses immediately without hardcoded lateral recovery.

## Next Step

Recommended v2 direction:

```text
1. Do not return to hardcoded lateral_recovery for final performance.
2. Pretrain or fine-tune perception/actor features before PPO-only recovery.
3. Increase visual feature influence in a controlled learned way:
   - stronger actor adapter visual weights, or
   - freeze old actor briefly and train visual adapter/perception head with BC/recovery data, or
   - add dense aux mask targets to BC datasets and train perception head before PPO.
4. Add an explicit no-lateral-recovery recovery dataset that does not use corrected steer from lateral_recovery as final policy target.
5. Keep hardcoded controller only as teacher/diagnostic, not as final eval assist.
```

Most likely next experiment:

```text
rgb_perception_actor_features_v2_pretrain

Goal:
  train compact visual actor adapter/perception head with supervised BC or recovery data before PPO,
  while keeping env.lateral_recovery disabled during final eval.
```

