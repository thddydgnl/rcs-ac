# PPO Clean Matched-r7j Baseline Manifest - 2026-05-07

## Decision

Accept `imitation-initialized PPO matched-r7j` as the PPO clean matched
baseline for robustness comparison against RCS-AC v1.1 r7j.

The checkpoint passed the required clean safety gates under the matched r7j
sensor, observation, action, controller, reward, termination, and evaluation
contract. It is slower than RCS-AC r7j, but remains inside the clean comparison
speed band and satisfies the route-progress and jitter criteria.

## Candidate

| Item | Value |
| --- | --- |
| Baseline label | `imitation-initialized PPO matched-r7j` |
| Train config | `PPO/configs/matched_r7j/ppo_matched_r7j_train_seed7_imitation_teacher.yaml` |
| Eval configs | `PPO/configs/matched_r7j/ppo_matched_r7j_eval_seed7_512.yaml`, `PPO/configs/matched_r7j/ppo_matched_r7j_eval_seed7_2048.yaml` |
| Checkpoint | `PPO/results/windows_pullback/ppo_matched_r7j_seed7_imitation_teacher_train/last.pt` |
| Teacher checkpoint | `RCS-AC-v1.1/portable_snapshots/r7j_generalized_20260507/results/frozen_baselines/r6_controller_compatible_waypoints_20260506/checkpoint/last.pt` |
| RCS-AC r7j runtime config | `RCS-AC-v1.1/configs/v1_1/full_bev_smoke_shared_agile_midbalanced_route_geometry_cc4_r7j_target_y_lowpass_light.yaml` |
| Latest contract audit | `PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/contract_audit_latest.json` |
| Evidence root | `PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507` |
| Source commit before manifest | `b4c23b0` |

The teacher was used only for offline waypoint/target-speed distillation during
PPO training. PPO inference does not use RCS-AC reliability, source-level
reliability fusion, Shapley, or risk-context modules.

## Matched Contract

| Contract item | Status |
| --- | --- |
| CARLA/Town04/clean weather/fixed-delta protocol | matched to r7j |
| Sensor setup and observation preprocessing | matched to r7j |
| RGB shape, BEV spec, ego/route/sensor-health scalar contract | matched to r7j |
| Policy output | waypoint `4x2` plus `target_speed_kmh` |
| Controller/action path | shared r7j pure-pursuit controller to direct_3d `[steer, throttle, brake]` |
| Target-speed cap and curvature speed cap | matched to r7j |
| Steer lowpass/rate limit and `target_y_lowpass_tau_s=0.06` | matched to r7j |
| Reward, termination, horizon, route/start/seed protocol | matched to r7j |
| PPO-only steering override or privileged inference input | absent |
| RCS-AC reliability/fusion/Shapley/risk modules in PPO | absent |
| Result folder separation | PPO results stored under `PPO/results` only |

## Clean Gate Summary

| Gate | Runs | Safety | Route progress | mean_speed_kmh | steer_jitter | raw_steer_jitter | target_y_jitter | waypoint_y_jitter |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Primary 512 | 9 | 9/9 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | 13.5316192179 | 0.0096486062 | 0.0186091452 | n/a | n/a |
| Primary 2048 | 9 | 9/9 pass, collision/offroad/stuck 0 | avg 1.0, 9/9 goal | 13.6367680177 | 0.0083660715 | 0.0173332023 | n/a | n/a |
| Generalization 512 | 36 | 36/36 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | 13.5308524831 | 0.0097225591 | 0.0186434548 | 0.0533415079 | 0.0421959021 |
| Hidden 512 | 18 | 18/18 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | 13.5332448681 | 0.0096917216 | 0.0185869402 | 0.0531762641 | 0.0422157163 |
| Generalization 2048 subset | 6 | 6/6 pass, collision/offroad/stuck 0 | avg 1.0, 6/6 goal | 13.6321007498 | 0.0084267295 | 0.0167460605 | 0.0458853666 | 0.0391047836 |

All recorded clean gates had:

```text
obs_finite_rate=1.0
waypoint_valid_rate=1.0
path_valid_rate=1.0
used_synthetic_fallback_rate=0.0
used_route_fallback_rate=0.0
CARLA crash/timeout=0
```

## RCS-AC r7j Comparison

RCS-AC r7j reference values are from
`RCS-AC-v1.1/results/frozen_candidates/r7j_target_y_lowpass_light_generalized_20260507/README.md`.

| Metric | RCS-AC v1.1 r7j | PPO clean matched-r7j |
| --- | ---: | ---: |
| Primary 2048 safety | 9/9 | 9/9 |
| Primary 2048 route goal | 9/9 | 9/9 |
| Primary 2048 mean_speed_kmh | 14.069821 | 13.636768 |
| Primary 2048 steer_jitter | 0.019461 | 0.008366 |
| Generalization 512 safety | 36/36 | 36/36 |
| Generalization 512 mean_speed_kmh | 14.561688 | 13.530852 |
| Generalization 512 steer_jitter | 0.018047 | 0.009723 |
| Hidden 512 safety | 18/18 | 18/18 |
| Hidden 512 mean_speed_kmh | 14.469590 | 13.533245 |
| Hidden 512 steer_jitter | 0.018123 | 0.009692 |
| Generalization 2048 subset safety | 6/6 | 6/6 |
| Generalization 2048 subset route goal | 6/6 | 6/6 |
| Generalization 2048 subset mean_speed_kmh | 14.865491 | 13.632101 |
| Generalization 2048 subset steer_jitter | 0.016469 | 0.008427 |

Interpretation: PPO now has comparable clean safety and route completion under
the matched controller/runtime. RCS-AC r7j remains faster. PPO's measured
steering jitter is lower, largely because the accepted PPO candidate drives
more conservatively.

## Eval Commands

Primary, generalization, and hidden gates were run on Windows CARLA through the
PPO multistart runner:

```powershell
.\PPO\scripts\windows\run_ppo_matched_r7j_multistart_eval.ps1 `
  -SpawnIndices <set> `
  -Steps 512 `
  -Seed <seed> `
  -RunPrefix <run-prefix> `
  -Checkpoint 'PPO\results\ppo_matched_r7j_seed7_imitation_teacher_train\last.pt'

.\PPO\scripts\windows\run_ppo_matched_r7j_multistart_eval.ps1 `
  -SpawnIndices <set> `
  -Steps 2048 `
  -Seed <seed> `
  -RunPrefix <run-prefix> `
  -Checkpoint 'PPO\results\ppo_matched_r7j_seed7_imitation_teacher_train\last.pt'
```

Jitter was computed with:

```bash
PYTHONPATH=RCS-AC-v1.1/code:. python -m rcs_ac_v1_1.scripts.analyze_eval_jitter \
  --log-roots <pulled PPO multistart roots> \
  --output-dir <PPO jitter output dir>
```

## Logging Schema

The eval logs separate:

```text
raw policy output
controller output
CARLA applied control
collision/offroad/stuck
route_progress
mean_speed_kmh
raw_steer_jitter
steer_jitter
steer_sign_change_rate
controller_target_y_jitter
controller_curvature_jitter
policy_waypoint_y_jitter
obs_finite_rate
waypoint_valid_rate
path_valid_rate
used_synthetic_fallback_rate
used_route_fallback_rate
```

See `PPO/docs/ppo_matched_r7j_logging_schema.md` for the detailed schema.

## Paper Wording

Use this label:

```text
imitation-initialized PPO matched-r7j
```

Recommended wording:

```text
The PPO baseline uses the same sensor preprocessing, waypoint-plus-speed action
contract, shared pure-pursuit controller, reward, termination, start sets, and
evaluation protocol as RCS-AC v1.1 r7j. PPO is initialized with the same
imitation teacher family used to produce the frozen r6/r7j waypoint policy, but
does not use RCS-AC reliability, source-level fusion, Shapley attribution, or
risk-context modules at inference.
```

Do not call this a plain from-scratch PPO baseline. If a scratch PPO result is
reported, keep it as a separate ablation.

## Next Robustness Plan

Only after this manifest, run paired RCS-AC r7j vs PPO matched-r7j robustness
experiments under the same route/start/seed/controller/eval contract:

1. Clean recheck subset if the environment or logging code changes.
2. Weather degradation: rain, fog, wet road, low sun/night variants.
3. Sensor dropout: RGB dropout, LiDAR dropout, delayed frames, intermittent
   missing sensor packets.
4. Sensor noise/corruption: RGB blur/noise/exposure, LiDAR sparsity/dropout,
   calibration jitter if supported by the shared environment.
5. Report clean-normalized degradation for safety, route completion, speed,
   jitter, validity, and fallback rates.

Keep all PPO robustness outputs under `PPO/results` and all RCS-AC outputs under
their RCS-AC result roots.
