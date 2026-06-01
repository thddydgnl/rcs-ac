# Round 17 - Drift-Window Visual-Steer BC Gate

Date: 2026-05-01

Status: BC/eval complete - PPO blocked

## Baseline

Round 16 is the current blocked run:

- BC val loss: `0.0204317`
- Eval max route: `0.205128`
- Eval mean route: `0.091096`
- Offroad: `3/512`
- Early offroad: route `0.153846`
- Mean visual-zero action L1: `0.006131`
- Mean visual-zero steer delta: `0.001017`
- Mean visual-zero longitudinal delta: `0.005114`
- Mean action jitter L1: `0.069248`

The failure window is route `0.15-0.23`, left drift lane offset about `-2.2` to `-2.75 m`, heading error about `5.3` to `16.3 deg`.

## Subagent Consensus

- The problem is still lateral drift plus weak steering-relevant RGB causality.
- Round 16's generic visual-zero margin mostly increased longitudinal dependence, not steering dependence.
- PPO remains blocked.
- The smallest useful next step is targeted drift-window data, not another loss-only tweak.
- Direct line-follow steering control is not allowed; RGB lane/road information must remain inside the learned policy path.

## Intervention

Round 17 combines:

1. Collect `128-256` masked drift-window transitions around route `0.15-0.23`.
2. Mix drift-window data with Round 15 clean/recovery data.
3. Train conservative frame BC using:
   - `perception_aux_head`
   - `rgb_perception_actor_features`
   - `actor_input_adapter`
4. Add a small BC-only steering-specific visual-zero margin.
5. Add eval logging for corrective-window visual steering metrics.

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_drift_window_round17.yaml
```

No direct action override, no lateral recovery in final eval, and no PPO before the BC/eval gate passes.

## Planned Remote Steps

1. Windows pull latest code.
2. Run `ensure_carla_ready.ps1`.
3. Collect drift-window data with `collect_recovery_dataset.py`:
   - config: `configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml`
   - checkpoint: `checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt`
   - route window: `0.15-0.23`
   - lane offset threshold: `0.45`
   - heading threshold: `5.0`
4. Mix:
   - Round 15 clean early: `384`
   - Round 17 drift-window: `192` if available
   - Round 15 recovery: `128`
5. Train Round 17 BC.
6. Run 512-step no-lateral eval.
7. Pull logs and summaries.
8. Judge gate before any PPO.

## Gate

PPO 2048 may run only if all pass:

- early offroad before route `0.20`: `0`
- total offroad: `<= 2/512`
- max route: `>= 0.230769`, preferred `>= 0.256410`
- collision/stuck: `0`
- mean action jitter L1: `<= 0.07`
- rollout mean visual-zero action L1: `>= 0.012`
- rollout mean visual-zero steer delta: `>= 0.003`
- corrective-window mean visual-zero steer delta: `>= 0.004`
- corrective sign rate: `>= 0.65`

If BC improves but rollout steering causality remains weak, stop before PPO.

## Result

Mac commit pushed and Windows pulled:

- `2c5a21c Add round 17 drift-window visual steering gate`

CARLA readiness:

- Pre-collect guard: `CARLA_READY=True`
- Post-collect guard: `CARLA_READY=True`
- First eval attempt: invalid infra run, failed at `env.reset()` with CARLA streaming/load-world timeout before rollout.
- Retry used `-ForceRestart`, then guard passed with `CARLA_READY=True`.
- Post-eval guard: `CARLA_READY=True`

Dataset collection:

- Output: `data/expert/rgb_perception_actor_features_v2_drift_window_192_round17.npz`
- Accepted: `192 / 1994`
- Acceptance rate: `0.0963`
- Route range: `0.153846-0.205128`
- Route mean: `0.178285`
- Bad transitions: `0`
- Mask availability: `1.0`

Mixed dataset:

- Output: `data/expert/rgb_perception_actor_features_v2_drift_mix_704_round17.npz`
- Total transitions: `704`
- Parts:
  - clean early: `384`, route `0.025641-0.205128`
  - drift window: `192`, route `0.153846-0.205128`
  - recovery: `128`, route `0.256410-0.358974`
- Bad transitions: `0`
- Mask availability: `1.0`

BC:

- Run: `rgb_perception_actor_features_v2_drift_window_bc_round17`
- Init: `checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt`
- Trainable params: `130418 / 954944`
- Epoch 5 val `bc_loss`: `0.017550`
- Epoch 5 val `action_l1`: `0.111493`
- Epoch 5 val `perception_aux_loss`: `0.304767`
- Epoch 5 val `aux_road_mask_iou`: `0.625252`
- Epoch 5 val `aux_lane_edge_mask_iou`: `0.0`
- Epoch 5 val `visual_actor_delta_l1_mean`: `0.066281`
- Epoch 5 val `visual_zero_action_l1`: `0.022247`
- Epoch 5 val `visual_zero_steer_delta`: `0.001104`
- Epoch 5 val `visual_zero_longitudinal_delta`: `0.021143`

Eval retry:

- Run: `rgb_perception_actor_features_v2_drift_window_eval_512_round17_retry_20260501T125253`
- Completed steps: `512`
- Script status: `PASS`
- Max route progress: `0.282051`
- Mean route progress: `0.132762`
- Final route progress: `0.025641`
- Done reasons: `{"offroad": 2}`
- Collision rate: `0.0`
- Stuck rate: `0.0`
- Mean action jitter L1: `0.057810`
- Mean steer jitter: `0.005620`
- Mean longitudinal jitter: `0.052190`
- Mean visual-zero action L1: `0.007283`
- Mean visual-zero steer delta: `0.001037`
- Mean visual-zero longitudinal delta: `0.006246`
- Corrective need count: `16`
- Corrective sign rate: `0.625`
- Mean corrective visual-zero steer delta: `0.000121`
- Mean corrective steer delta: `0.000068`

Offroad rows:

- Step `242`, episode `0`: route `0.256410`, lane offset `-2.1757`, heading error `7.8436`
- Step `496`, episode `1`: route `0.282051`, lane offset `-3.0236`, heading error `24.7985`

Lane/heading envelope:

- `abs(lane_offset)` mean `1.1174`, p95 `2.0123`, max `3.0236`
- `abs(heading_error)` mean `3.6991`, p95 `9.6022`, max `24.7985`

## Judgment

Round 17 is a real safety/progress improvement over Round 16, but PPO remains blocked.

Improved versus Round 16:

- Max route: `0.205128 -> 0.282051`
- Mean route: `0.091096 -> 0.132762`
- Offroad: `3/512 -> 2/512`
- Early offroad before route `0.20`: `1 -> 0`
- Mean action jitter: `0.069248 -> 0.057810`
- BC val loss: `0.020432 -> 0.017550`

Still blocked:

- Rollout visual-zero action L1 is `0.007283`, below gate `0.012`.
- Rollout visual-zero steer delta is `0.001037`, below gate `0.003`.
- Corrective-window evidence is weak: only `16` rows and mean corrective steer delta `0.000121`.
- Final offroad shifted later but still occurs with severe left drift and heading error at route `0.282051`.

Decision: do not run PPO 2048. The drift-window data helped route/safety, but the model still does not prove steering-relevant RGB causality in rollout.
