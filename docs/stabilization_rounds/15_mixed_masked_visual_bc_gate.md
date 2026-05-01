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

Pending.
