# Round 13 - RGB Perception Actor Features v2 Pretrain

Date: 2026-05-01

## Goal

Use RGB road/lane perception as a learned stabilization signal, not as a final hardcoded controller.

Round 12 connected compact RGB perception actor features to the policy, but the learned policy still collapsed without `lateral_recovery`. The main failure was not absence of a visual path; it was that the policy did not use the visual lane/drivable signal strongly enough.

## Direction

Train the RGB perception actor path before PPO:

1. Keep final eval without hardcoded `lateral_recovery`.
2. Use semantic/lane mask targets as supervised perception pressure.
3. Train only the visual/perception adapter path first:
   - `perception_aux_head`
   - `rgb_perception_actor_features`
   - `actor_input_adapter`
4. Keep the base actor mostly preserved during the first pretrain pass.
5. After the visual path affects action, continue with low-std PPO.

## Code/config changes

Updated:

```text
configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml
rcs_ac_first/configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml
rcs_ac_first/scripts/collect_expert_dataset.py
```

Key settings:

```yaml
model:
  use_rgb_perception_actor_features: true
  rgb_perception_actor_feature_dim: 16
  rgb_perception_actor_adapter_init_std: 0.05

bc:
  perception_aux_loss_weight: 0.5
  road_mask_weight: 0.25
  lane_edge_weight: 1.0
  lane_edge_pos_weight: 3.0
  trainable_module_names:
    - perception_aux_head
    - rgb_perception_actor_features
    - actor_input_adapter
```

Added model config:

```text
rgb_perception_actor_adapter_init_std
```

This keeps the old adapter default at `0.005`, while v2 pretrain can start with stronger visual adapter weights.

`collect_expert_dataset.py` now stores semantic auxiliary masks in the NPZ when the environment provides them:

```text
aux_road_mask_target
aux_lane_edge_target
aux_mask_targets_available
```

## Local verification

Passed:

```bash
python -m pytest tests/test_observation.py tests/test_fusion_gate.py tests/test_rollout_buffer.py tests/test_rgb_perception_adapter.py
python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml --device cpu
```

Important smoke signal:

```text
visual_feature_zero_action_l1_mean: 0.00264087
```

This confirms that zeroing compact visual actor features changes the action path in the smoke model.

## Windows experiment plan

After pushing the current code to GitHub and pulling on Windows, collect or select a dataset with mask targets, then run BC pretrain.

Candidate dataset should contain:

```text
aux_road_mask_target
aux_lane_edge_mask_target
aux_mask_targets_available
```

If `aux_mask_available_rate` is `0.0`, the dataset is not useful for this round. Re-collect with a config that has `env.semantic_aux_camera_enabled: true`.

Recommended command shape:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml `
  --dataset C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\<DATASET>.npz `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt `
  --init-strict false `
  --run-name rgb_perception_actor_features_v2_pretrain_bc `
  --device auto
```

Then evaluate without lateral recovery:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\rgb_perception_actor_features_v2_pretrain_bc\last.pt `
  --run-name rgb_perception_actor_features_v2_pretrain_eval_512 `
  --num-steps 512 `
  --device auto
```

## Gate

This round is promising only if:

- `aux_mask_available_rate` is high.
- `aux_lane_edge_mask_loss` and `aux_road_mask_loss` move down during BC.
- `visual_actor_delta_l1_mean` is non-trivial and stays finite.
- no-lateral-recovery eval survives longer than Round 11/12 immediate collapse.
- route progress improves before offroad.

If the visual ablation delta remains near zero after BC, do not move to PPO. Increase visual adapter training pressure or inspect mask target quality first.

## 2026-05-01 first remote attempt

Windows branch:

```text
codex-stabilization-rgb-v2
```

Remote smoke passed:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.smoke_check `
  --config configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml `
  --device cpu
```

Smoke visual path signal:

```text
visual_feature_zero_action_l1_mean: 0.00264087
```

BC pretrain on the existing recovery dataset completed:

```text
dataset: data/expert/recovery_imitation_v2_lateral_target_512.npz
run: rgb_perception_actor_features_v2_pretrain_bc
checkpoint: checkpoints/rgb_perception_actor_features_v2_pretrain_bc/last.pt
```

But the round gate failed because the existing dataset has no usable mask targets:

```text
aux_mask_available_rate: 0.0
perception_aux_loss: 0.0
aux_road_mask_loss: 0.0
aux_lane_edge_mask_loss: 0.0
```

The action path did learn from BC through the adapter:

```text
visual_actor_delta_l1_mean: 0.06789840012788773
val bc_loss: 0.039427537471055984
```

This is not enough for the RGB/line round, because the perception head was not trained against road/lane masks.

Follow-up implemented in code:

```text
collect_expert_dataset.py now saves aux road/lane mask targets when semantic aux camera data is available.
```

A masked clean dataset collection was attempted:

```text
run: rgb_perception_actor_features_v2_masked_clean_collect_256
output: data/expert/rgb_perception_actor_features_v2_masked_clean_256.npz
```

It failed before the experiment script because CARLA did not open RPC port `2000` within `120` seconds:

```text
remote_runs/logs/run_20260501_023627.txt
remote_runs/summary.json
```

Next operational step:

1. Fix/retry CARLA startup on Windows.
2. Re-run masked dataset collection.
3. Re-run v2 BC pretrain only after `aux_mask_available_rate > 0`.
4. Then evaluate no-lateral-recovery and proceed to low-std PPO only if visual mask training passes the gate.

## 2026-05-01 continuation

CARLA startup fix:

```text
Direct Start-Process over SSH was unstable.
Starting CARLA through an interactive scheduled task for song\song opened port 2000 and allowed collection/eval to run.
```

Additional code fix:

```text
train_bc.py now accepts the dataset key aux_lane_edge_target as the lane mask target.
```

Before this fix, the new masked dataset contained valid masks, but BC still logged zero perception loss because `train_bc.py` looked only for `aux_lane_edge_mask_target`.

Masked dataset collected:

```text
data/expert/rgb_perception_actor_features_v2_masked_clean_256.npz
num_transitions: 256
route_progress_mean: 0.10436698794364929
aux_mask_targets_available mean: 1.0
road mask mean: 0.2783322334289551
lane edge mask mean: 0.01923177018761635
```

BC pretrain with real mask supervision:

```text
run: rgb_perception_actor_features_v2_masked_clean_pretrain_bc_v2
checkpoint: checkpoints/rgb_perception_actor_features_v2_masked_clean_pretrain_bc_v2/last.pt
val bc_loss: 0.020064804702997208
val perception_aux_loss: 0.37155163288116455
val aux_mask_available_rate: 1.0
val aux_road_mask_loss: 0.4151459038257599
val aux_lane_edge_mask_loss: 0.2677651643753052
val visual_actor_delta_l1_mean: 0.06980796158313751
```

No-lateral eval after BC pretrain:

```text
run: rgb_perception_actor_features_v2_masked_clean_pretrain_eval_512_v2_20260501T030210
completed_steps: 512
episode_count: 4
done_reason_counts: {"offroad": 3}
max_route_progress: 0.1794871794871795
mean_route_progress: 0.07366786858974358
collision_rate: 0.0
stuck_rate: 0.0
```

Low-std PPO from the masked pretrain checkpoint:

```text
run: rgb_perception_actor_features_v2_masked_clean_lowstd_2048_20260501T030328
checkpoint: checkpoints/rgb_perception_actor_features_v2_masked_clean_lowstd_2048_20260501T030328/last.pt
total_steps: 2048
max training route_progress observed in episode logs: 0.231
all completed episodes still ended offroad
final logged aux_mask_available_rate: 1.0
final logged visual_actor_delta_l1_mean: 0.06174435466527939
```

No-lateral eval after low-std PPO:

```text
run: rgb_perception_actor_features_v2_masked_clean_lowstd_2048_eval_512_retry_20260501T031015
completed_steps: 512
episode_count: 4
done_reason_counts: {"offroad": 3}
max_route_progress: 0.1794871794871795
mean_route_progress: 0.08443509615384615
collision_rate: 0.0
stuck_rate: 0.0
```

Judgment:

```text
MASK SUPERVISION PATH: PASS
NO-LATERAL CLEAN POLICY: STILL FAILING
LOW-STD PPO 2048: NOT ENOUGH
```

The v2 direction is technically alive: real semantic mask supervision reaches BC, visual features affect action, and no-lateral eval completes 512 steps without collision/stuck. But it still repeatedly ends offroad and does not materially improve max eval progress beyond the masked pretrain baseline.

Recommended next round:

```text
Collect a recovery-focused masked dataset, not only clean early-route frames.
Use collect_recovery_dataset.py or a no-lateral recovery collector over route_progress 0.24-0.38,
require aux_mask_targets_available > 0,
then pretrain the same visual adapter path before PPO.
```

## Next step after pass

Run low-std PPO from the v2 pretrain checkpoint:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lowstd_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\rgb_perception_actor_features_v2_pretrain_bc\last.pt `
  --init-strict false `
  --override-log-std -2.0 -2.0 `
  --total-timesteps 2048 `
  --run-name rgb_perception_actor_features_v2_lowstd_2048 `
  --device auto
```
