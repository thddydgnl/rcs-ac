# PPO Matched-r7j Clean Candidate - Imitation Teacher 2026-05-07

## Status

Primary clean candidate. This checkpoint passed primary 512 and primary 2048
9-start clean multistart gates. It is not yet the final paper-ready PPO clean
matched-r7j baseline because the generalization and hidden gates have not run.

## Candidate

| Item | Value |
| --- | --- |
| Label | imitation-initialized PPO matched-r7j |
| Train config | `PPO/configs/matched_r7j/ppo_matched_r7j_train_seed7_imitation_teacher.yaml` |
| Checkpoint | `PPO/results/windows_pullback/ppo_matched_r7j_seed7_imitation_teacher_train/last.pt` |
| Teacher checkpoint | `RCS-AC-v1.1/portable_snapshots/r7j_generalized_20260507/results/frozen_baselines/r6_controller_compatible_waypoints_20260506/checkpoint/last.pt` |
| Runtime config | `RCS-AC-v1.1/configs/v1_1/full_bev_smoke_shared_agile_midbalanced_route_geometry_cc4_r7j_target_y_lowpass_light.yaml` |
| Contract audit | `PPO/results/windows_pullback/windows_imitation_contract_audit.json` |

The teacher is used only as offline waypoint/target-speed distillation during
PPO training. PPO inference still uses the matched-r7j observation contract and
does not use RCS-AC reliability, source-level reliability fusion, Shapley, or
risk-context modules.

## Training Result

Windows CARLA training completed 4096 PPO timesteps. The final training rollout
ended with:

| Metric | Value |
| --- | ---: |
| last_route_progress | 0.8461538462 |
| mean_route_progress | 0.7684294872 |
| collision_count | 0 |
| offroad_count | 0 |
| stuck_count | 0 |
| imitation_loss | 0.0816092864 |

Earlier exploratory rollout updates still had collision/offroad events, so the
training log should be interpreted as optimization telemetry rather than the
clean safety gate.

## Single-Start Clean Eval

| Gate | Safety | max_route_progress | mean_speed_kmh | done |
| --- | --- | ---: | ---: | --- |
| 512 | PASS, collision/offroad/stuck 0 | 0.8205128205 | 13.5293498663 | completed 512/512 |
| 2048 | PASS, collision/offroad/stuck 0 | 1.0 | 13.6385935138 | goal at step 619 |

Validity/fallback was clean in both runs:

| Metric | 512 | 2048 |
| --- | ---: | ---: |
| obs_finite_rate | 1.0 | 1.0 |
| waypoint_valid_rate | 1.0 | 1.0 |
| path_valid_rate | 1.0 | 1.0 |
| used_synthetic_fallback_rate | 0.0 | 0.0 |
| used_route_fallback_rate | 0.0 | 0.0 |

## Jitter

The combined 512/2048 jitter analysis is in
`PPO/results/windows_pullback/ppo_matched_r7j_imitation_teacher_jitter_20260507`.

| Metric | Average | Target |
| --- | ---: | ---: |
| steer_jitter | 0.0090182046 | <= 0.022 |
| raw_steer_jitter | 0.0176990338 | <= 0.032 |
| controller_target_y_jitter | 0.0512811238 | <= 0.065 |
| policy_waypoint_y_jitter | 0.0419311782 | <= 0.080 |
| mean_speed_kmh | 13.5839716900 | 13.5-15.0 |

## Primary Multistart Clean Eval

Primary start set:

```text
seed 7, spawn indices 22, 160, 161, 162, 176, 192, 193, 232, 234
```

| Gate | Safety | Route progress | Speed | Jitter |
| --- | --- | --- | --- | --- |
| Primary 512 | 9/9 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | avg 13.5316192179 | steer 0.0096486062, raw 0.0186091452 |
| Primary 2048 | 9/9 pass, collision/offroad/stuck 0 | avg 1.0, 9/9 goal | avg 13.6367680177 | steer 0.0083660715, raw 0.0173332023 |

Validity/fallback was clean across all primary runs:

```text
obs_finite_rate=1.0
waypoint_valid_rate=1.0
path_valid_rate=1.0
used_synthetic_fallback_rate=0.0
used_route_fallback_rate=0.0
CARLA crash/timeout=0
```

Primary evidence files:

```text
PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/primary_512
PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/primary_2048
```

## RCS-AC r7j Comparison

| Metric | RCS-AC r7j reference | PPO candidate |
| --- | ---: | ---: |
| Primary 512 safety | 9/9 | 9/9 |
| Primary 512 avg max_route_progress | 0.840456 | 0.820513 |
| Primary 512 min max_route_progress | 0.794872 | 0.820513 |
| Primary 2048 safety | 9/9 | 9/9 |
| Primary 2048 avg max_route_progress | 1.0 | 1.0 |
| Primary 2048 route goal | 9/9 | 9/9 |
| Mean speed | 13.74-14.07 | 13.53-13.64 |
| steer_jitter | 0.019-0.020 | 0.008-0.010 |

The PPO candidate now matches the primary clean safety/progress gate. It is not
yet a paper-ready clean baseline because the generalization and hidden gates are
still pending.

## Next Gates

1. Run generalization 512 36-start clean gate.
2. If 36/36 safety passes, run hidden 512 18-start clean gate.
3. Run generalization 2048 subset gate.
4. Only after these pass, write the final PPO clean matched-r7j baseline
   manifest.

## Notes For Paper Wording

Use the label `imitation-initialized PPO matched-r7j`. This is fair to compare
against r7j because r7j itself is a frozen imitation/BC checkpoint plus the
shared r7j controller/runtime config. The paper should still report that PPO
received teacher distillation before PPO updates, while RCS-AC additionally has
its own reliability/fusion/Shapley/risk design in the RCS-AC algorithm family.
