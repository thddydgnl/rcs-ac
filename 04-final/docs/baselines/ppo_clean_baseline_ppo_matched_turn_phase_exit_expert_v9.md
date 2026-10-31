# PPO Matched Clean Baseline Manifest: exit_expert_v9

Date: 2026-05-04

Status: accepted PPO matched clean baseline candidate.

## Baseline Identity

- Run family: `ppo_matched_turn_phase_exit_expert_v9`
- Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_exit_expert_v9.yaml`
- Checkpoint: `checkpoints/ppo_matched_turn_phase_exit_expert_v9_bc_20260504T0635/last.pt`
- Windows checkpoint source: `C:\Users\thddy\Documents\code\RCS-AC-first-codex-sync\checkpoints\ppo_matched_turn_phase_exit_expert_v9_bc_20260504T0635\last.pt`
- Parent checkpoint: `checkpoints/ppo_matched_turn_phase_early_steer_slow_v4_bc_20260504T0415/last.pt`
- Base candidate: `ppo_matched_turn_phase_early_steer_slow_v4`

## Fairness Contract

- Observation keys: RGB, LiDAR, ego, route.
- Matched PPO features: estimated scalar actor features and RGB perception actor features are enabled to match the RCS-AC clean input contract.
- RCS-AC-only modules are disabled: reliability, risk context, Shapley, dropout consistency.
- Runtime assist is disabled: no lateral recovery assist, no lane-offset runtime steering override, no hardcoded steering rule.
- Environment: Town04 clean, no traffic, no corruption, default weather.
- Action wrapper: same target-speed two-dimensional action and action smoothing wrapper as the matched baseline family.
- Reward: unchanged from the V4 matched clean baseline config.

## Offline Supervision Change

V9 adds a targeted offline imitation data patch for the clean route-exit gap that blocked V4.

- New data: `data/expert/ppo_matched_exit_expert_route036_049_256_20260504.npz`
- Mixed data: `data/expert/ppo_matched_exit_expert_mix_4512_20260504.npz`
- New expert route range: `0.384615` to `0.487179`
- New expert row count: `256`
- Mixed row count: `4512`
- Collection source: clean route-window expert/heuristic controller samples from the same CARLA environment.

This is offline imitation supervision only. It is not a runtime controller or privileged inference input.

## Training Scope

- Trainable module: `recurrent_actor_critic.actor_head.2`
- Trainable actor output row: longitudinal only, index `1`
- Frozen actor output row: steering, index `0`
- Epochs: `3`
- Learning rate: `8e-5`
- Longitudinal loss weight: `1.5`
- Route-window longitudinal floor: route `0.384-0.488`, floor `0.15`, weight `0.80`

## Clean Eval Results

512-step clean eval:

- Run: `ppo_matched_turn_phase_exit_expert_v9_eval_512_20260504T0650_20260504T060911`
- Collision: `0`
- Offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.5128205128`
- Max target waypoint index: `20`
- Mean action jitter L1: `0.1288284499`

2048-step clean eval:

- Run: `ppo_matched_turn_phase_exit_expert_v9_eval_2048_20260504T0710_20260504T061116`
- Collision: `0`
- Offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.5128205128`
- Max target waypoint index: `20`
- Mean action jitter L1: `0.1188556407`

## Comparison To RCS-AC Round90

- RCS-AC Round90 estimated scalar v6 clean route target: `0.435897`
- PPO matched V4 route: `0.410256`
- PPO matched V9 route: `0.512821`

V9 preserves the V4 clean safety target and exceeds the Round90 route-progress reference in both 512 and 2048 clean eval.

## Notes

- The generic `ppo_gate_status` remains `BLOCK` due to jitter and visual-center corrective metrics. In this clean baseline goal, those metrics are reference diagnostics only; the acceptance criteria prioritize collision/offroad/severe early offroad/stuck `0` and route progress.
- This candidate should be used as the PPO matched clean baseline for the next robustness comparison stage, while reporting the targeted offline exit supervision in the baseline protocol.

## Artifacts

- Local pullback: `remote_runs/ppo_matched_turn_phase_exit_expert_v9_20260504/`
- 512 summary: `remote_runs/ppo_matched_turn_phase_exit_expert_v9_20260504/ppo_matched_turn_phase_exit_expert_v9_eval_512_20260504T0650_20260504T060911/summary.json`
- 2048 summary: `remote_runs/ppo_matched_turn_phase_exit_expert_v9_20260504/ppo_matched_turn_phase_exit_expert_v9_eval_2048_20260504T0710_20260504T061116/summary.json`
