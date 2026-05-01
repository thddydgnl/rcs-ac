# Round 15 - Mixed Masked Visual BC Gate

Date: 2026-05-01

## Goal

Test the distribution-collapse diagnosis from Round 14 before launching another PPO run.

Round 14 used a valid recovery-only masked dataset, but no-lateral rollouts failed around route progress `0.05-0.08`, far before the recovery data window `0.24-0.38`. This round uses a mixed masked dataset so the visual adapter preserves early-route behavior while retaining a smaller recovery slice.

## Experiment Brief

```text
Round name: 15_mixed_masked_visual_bc_gate
Starting checkpoint: checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
Baseline: Round 13 masked-clean no-lateral eval max_route_progress 0.1795
Blocked run: Round 14 recovery-only final eval max_route_progress 0.0769, PPO 32/32 offroad
Primary hypothesis: recovery-only masked BC is off-support and weakens early-route behavior
Allowed changes: Shapley correctness/logging fix, eval visual-zero action logging, mixed dataset BC/eval
Rejected changes: reward rewrite, lateral recovery tuning, speed gate changes, 4096 promotion
```

## Sub-Agent Consensus

- Log Analyst: Round 14 is a regression; all PPO episodes ended offroad and final eval max route stayed `0.0769`.
- Code Auditor: current Shapley subset limiting needed correction before any Shapley-enabled PPO run.
- Mechanism Verifier: require visual-zero action delta in eval logs before claiming camera-derived action effect.
- Safety Regression Analyst: do not run 4096; 2048 promotion requires offroad `<=2`, collision/stuck `0`, early offroad `0`.
- Data Auditor: build a mixed masked dataset instead of continuing pure recovery-only.
- Hardcoding Auditor: report controller assistance clearly; any improvement needs controller-off/visual-zero ablations.

## Planned Procedure

1. Fix local blockers and validate on Mac.
2. Push branch and pull on Windows.
3. Ensure CARLA is ready before every CARLA command.
4. Build a mixed masked dataset:

```text
target: data/expert/rgb_perception_actor_features_v2_masked_mix_512.npz
preferred composition:
  clean early masked: 384 rows, route 0.00-0.16
  recovery masked: 128 rows, route 0.24-0.38
fallback composition:
  clean early masked: 256 existing rows
  recovery masked: 128 existing rows
```

5. Train BC with:

```text
configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml
init checkpoint: checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
run: rgb_perception_actor_features_v2_masked_mix_pretrain_bc
```

6. Evaluate without lateral recovery:

```text
run: rgb_perception_actor_features_v2_masked_mix_pretrain_eval_512
```

7. Only run 2048 PPO if the BC/eval gate passes.

## Gates

BC/pre-PPO continue gate:

```text
aux_mask_available_rate >= 0.95
aux_road_mask_iou >= 0.50 or road loss falls meaningfully
aux_lane_edge_mask_loss finite and not flat zero
visual_actor_delta_l1_mean >= 0.03
mean_visual_zero_action_l1 >= 0.01 preferred
no-lateral eval max_route_progress >= 0.1795
offroad eval count not worse than Round 13
```

2048 PPO gate, if reached:

```text
collision = 0
stuck = 0
offroad episodes <= 2
early offroad at route_progress < 0.20 = 0
max route_progress >= 0.90
deterministic eval safety clean
```

## Result

Completed on Windows via `song@100.87.219.58`.

Local validation before Windows execution:

```text
python -m compileall -q rcs_ac_first tests
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests -q
python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml --device cpu
python -m rcs_ac_first.scripts.smoke_check --config configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml --device cpu
```

Validation passed:

```text
tests: 13 passed
RGB pretrain smoke: passed, Shapley subset_count 6
lateral target smoke: passed, Shapley subset_count 6
```

Code changes before the run:

```text
Shapley subset selection now keeps pair-complete singleton evidence instead of biased early binary masks.
Skipped Shapley updates no longer train alignment against the default contribution proxy.
BC eval now logs lane_offset, heading_error, and visual-zero action deltas.
Added rcs_ac_first.scripts.mix_expert_datasets for reproducible mixed NPZ construction.
```

Windows sync:

```text
commit: 3264c86 Stabilize masked visual BC workflow
Windows git pull --ff-only: success
```

Dataset collection:

```text
run: rgb_perception_actor_features_v2_masked_clean_384_round15
file: data/expert/rgb_perception_actor_features_v2_masked_clean_384_round15.npz
num_transitions: 384
bad_transition_count: 0
route_progress: min 0.0256, mean 0.1064, max 0.2051
attempted_transitions: 386
```

Mixed dataset:

```text
file: data/expert/rgb_perception_actor_features_v2_masked_mix_512_round15.npz
num_transitions: 512
bad_transition_count: 0
aux_mask_available_rate: 1.0
route_progress: min 0.0256, mean 0.1549, max 0.3590

parts:
  clean_early: 384 rows, route 0.0256-0.2051, mask availability 1.0
  recovery: 128 rows, route 0.2564-0.3590, mask availability 1.0
```

BC pretrain:

```text
run: rgb_perception_actor_features_v2_masked_mix_pretrain_bc_round15
checkpoint: checkpoints/rgb_perception_actor_features_v2_masked_mix_pretrain_bc_round15/last.pt
epoch 3 val bc_loss: 0.021231
epoch 3 val perception_aux_loss: 0.325867
epoch 3 val aux_mask_available_rate: 1.0
epoch 3 val aux_road_mask_iou: 0.606665
epoch 3 val aux_lane_edge_mask_iou: 0.0
epoch 3 val visual_actor_delta_l1_mean: 0.055377
```

No-lateral eval:

```text
run: rgb_perception_actor_features_v2_masked_mix_pretrain_eval_512_round15_retry_20260501T112000
completed_steps: 512
done_reason_counts: offroad 3
collision_rate: 0.0
stuck_rate: 0.0
max_route_progress: 0.179487
mean_route_progress: 0.083033
final_route_progress: 0.025641
mean_action_jitter_l1: 0.054720
mean_visual_zero_action_l1: 0.004148
mean_visual_zero_steer_delta: 0.000719
mean_visual_zero_longitudinal_delta: 0.003429
```

Eval offroad rows:

```text
step 158: route 0.1795, lane_offset -2.6012, heading_error 11.5909
step 318: route 0.1795, lane_offset -2.5785, heading_error 11.4589
step 477: route 0.1795, lane_offset -2.5870, heading_error 11.6606
```

Pulled logs:

```text
/Users/song-yonghwi/code/windows_results/rcs-ac-first/experiments/rgb_perception_actor_features_v2_masked_mix_round15
```

## Judgment

Status: **WARN / do not run PPO 2048 yet**.

The mixed dataset fixed the Round 14 recovery-only collapse. The eval returned to the Round 13 masked-clean level:

```text
Round 13 max route: 0.1795
Round 14 recovery-only max route: 0.0769
Round 15 mixed max route: 0.1795
```

However, it did not beat the Round 13 gate, and the new visual-zero action metric shows weak camera-derived action causality:

```text
mean_visual_zero_action_l1: 0.004148
preferred gate: >= 0.01
stop-risk threshold from mechanism verifier: < 0.005
```

Therefore PPO was intentionally not launched. Running PPO from this checkpoint would likely repeat the Round 13 plateau rather than test a stronger mechanism.

## Next Action

Do not tune reward/control yet.

The next smallest useful change is to strengthen actor-facing visual causality before PPO:

```text
1. Keep the mixed dataset.
2. Add/try a stronger BC action-effect gate, such as visual-zero consistency or adapter-only stronger initialization/training.
3. Keep road mask supervision; lane-edge IoU remains 0.0, so do not claim lane-edge learning yet.
4. Re-run BC/eval and require mean_visual_zero_action_l1 >= 0.01 and max_route_progress > 0.1795 before PPO.
```
