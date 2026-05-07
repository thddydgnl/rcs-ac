# PPO Matched-r7j Contract

Created: 2026-05-07

## Purpose

This baseline aligns PPO with the RCS-AC v1.1 r7j runtime contract so the paper comparison keeps the runtime, observation, controller, reward, and evaluation protocol matched. The intended remaining difference is the learning algorithm and policy architecture, not hidden runtime assistance.

RCS-AC r7j is not a newly trained checkpoint. It is the frozen r6 imitation/BC checkpoint:

```text
RCS-AC-v1.1/results/frozen_baselines/r6_controller_compatible_waypoints_20260506/checkpoint/last.pt
```

with the r7j runtime/controller config:

```text
RCS-AC-v1.1/configs/v1_1/full_bev_smoke_shared_agile_midbalanced_route_geometry_cc4_r7j_target_y_lowpass_light.yaml
```

## Matched Contract Table

| Area | RCS-AC v1.1 r7j | PPO matched-r7j | Status |
|---|---|---|---|
| CARLA/Town | CARLA 0.9.16, Town04 | Same env config target | Matched by config |
| Weather | Clean/default | Clean/default | Matched |
| Eval start sets | r7j multistart config, seed/start list | Same env config/start list should be used | Matched when using same generated env configs |
| RGB input | `rgb [3,90,160]` | Same flat observation slice | Matched |
| LiDAR input | `lidar_bev [5,128,128]` | Same flat observation slice | Matched |
| Route BEV | `route_bev [4,128,128]` | Same flat observation slice | Matched |
| Ego BEV | `ego_bev [3,128,128]` | Same flat observation slice | Matched |
| Ego scalar | 6D r7j list | Same flat observation slice | Matched |
| Route scalar | 20D r7j route geometry list | Same flat observation slice | Matched |
| Sensor health | 15D | Same flat observation slice | Matched |
| GT BEV target | Training target only, not policy input | Not used at inference | Matched |
| Policy output | 4 local waypoints + target speed km/h | 4 local waypoints + target speed km/h | Matched |
| Controller | r7j shared speed-adaptive pure pursuit | Same `V11SpeedAdaptivePurePursuitController` | Matched |
| `target_y_lowpass_tau_s` | `0.06` | Loaded from same r7j config | Matched |
| Steer smoothing/rate limit | r7j config | Loaded from same r7j config | Matched |
| Target speed cap | r7j config | Loaded from same r7j config | Matched |
| Curvature speed cap | r7j config | Loaded from same r7j config | Matched |
| CARLA action | `direct_3d [steer, throttle, brake]` | Same default | Matched |
| Raw/controller/applied logs | Separate fields in `v1_1_eval_steps.csv` | Same field names | Matched |
| Eval horizon | 512 and 2048 | 512 and 2048 configs added | Matched |
| Result folder | RCS-AC result folders | PPO result folders only | Separated |

## Deliberately Different

| Area | Difference | Reason |
|---|---|---|
| Learning algorithm | PPO loss, advantage, clipping, value/entropy optimization | This is the baseline algorithm being compared. |
| Policy architecture | PPO uses a simple concatenation actor-critic | PPO must not inherit RCS-AC reliability/fusion/Shapley/risk modules. |
| Source-level fusion | RCS-AC r7j policy has source-level fusion weights; PPO matched-r7j does not use learned RCS-AC fusion | RCS-AC fusion is part of the algorithm under test. |
| Shapley/reliability/risk | RCS-AC-specific modules are excluded from PPO | Required for a fair algorithm comparison. |

## Must-fix Items Found in the Previous PPO Baseline

| Item | Old PPO state | Matched-r7j fix |
|---|---|---|
| Action contract | Existing PPO configs use `action_dim: 2` direct steer/longitudinal style | New PPO emits waypoint + target speed, then uses r7j controller |
| Controller contract | Previous action wrapper/controller differed from v1.1 r7j | New eval/train scripts instantiate r7j controller from the same runtime config |
| Observation contract | Previous PPO was built around v1 RGB/LiDAR/ego/route tensors | New PPO consumes r7j full-BEV flat observation via the v1.1 adapter |
| Logging | Previous PPO logs mixed `bc_eval_log.csv`/old schema | New eval writes `v1_1_eval_steps.csv` plus `summary.json` |
| Result separation | Older PPO logs live under `PPO/remote_runs`, RCS logs elsewhere | New matched outputs default to `PPO/results/...` |

## Optional / Later

| Item | Why not required now |
|---|---|
| SAC matched-r7j | Same contract should be reused after PPO is stable. |
| Multistart PowerShell wrapper | r7j already has a wrapper; PPO can get an equivalent wrapper after the first synthetic/import smoke passes. |
| BC/imitation initialization for PPO | Fairness depends on comparing final protocols clearly. If PPO receives imitation pretraining, SAC and paper text must label it as imitation-initialized PPO. |

## Paper Wording

Use wording like:

```text
PPO-matched-r7j uses the same CARLA environment, sensor preprocessing, BEV observation contract, waypoint/target-speed action contract, pure-pursuit controller, target-speed caps, curvature speed cap, steering low-pass/rate limit, and 512/2048 evaluation protocol as RCS-AC v1.1 r7j. PPO does not use RCS-AC reliability estimation, source-level reliability fusion, Shapley attribution, or risk context modules.
```

Do not call the PPO result "raw direct-action PPO" when it uses this controller. Report it as "PPO with the shared r7j waypoint-controller action contract."

## Added Files

```text
PPO/ppo_r7j/models.py
PPO/scripts/train_matched_r7j.py
PPO/scripts/evaluate_matched_r7j.py
PPO/configs/matched_r7j/ppo_matched_r7j_train_seed7.yaml
PPO/configs/matched_r7j/ppo_matched_r7j_eval_seed7_512.yaml
PPO/configs/matched_r7j/ppo_matched_r7j_eval_seed7_2048.yaml
PPO/docs/ppo_matched_r7j_logging_schema.md
```

## Remaining Known Difference

The compatibility log fields named `bev_fusion_weight_*` are filled by PPO as sensor-availability weights, not learned RCS-AC reliability fusion. This preserves the r7j CSV schema for the jitter analyzer but must be documented as a logging compatibility field, not an algorithmic fusion module.
