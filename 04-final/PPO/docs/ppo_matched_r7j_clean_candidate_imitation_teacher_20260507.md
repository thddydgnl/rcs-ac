# PPO Matched-r7j Clean Candidate - Imitation Teacher 2026-05-07

## Status

Clean matched-r7j baseline candidate. This checkpoint passed primary 512,
primary 2048, generalization 512, hidden 512, and generalization 2048 subset
clean gates under the PPO matched-r7j contract.

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

## Hidden Quick Gate

Hidden quick seed2 set:

```text
seed 2, spawn indices 24, 163, 177, 190, 230, 364
```

| Gate | Safety | Route progress | Speed | Jitter |
| --- | --- | --- | --- | --- |
| Hidden quick 512 | 6/6 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | avg 13.5333655915 | steer 0.0094685034, raw 0.0182670101 |

Validity/fallback was clean across all hidden quick runs:

```text
obs_finite_rate=1.0
waypoint_valid_rate=1.0
path_valid_rate=1.0
used_synthetic_fallback_rate=0.0
used_route_fallback_rate=0.0
CARLA crash/timeout=0
```

Evidence:

```text
PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/hidden_quick_512_seed2
```

## Hidden Full Gate

Hidden full set:

```text
seed 2, spawn indices 24, 163, 177, 190, 230, 364
seed 5, spawn indices 151, 174, 191, 233, 366, 369
seed 9, spawn indices 20, 23, 164, 178, 231, 370
```

| Gate | Safety | Route progress | Speed | Jitter |
| --- | --- | --- | --- | --- |
| Hidden 512 full | 18/18 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | avg 13.5332448681 | steer 0.0096917216, raw 0.0185869402 |

Validity/fallback was clean across all hidden full runs:

```text
obs_finite_rate=1.0
waypoint_valid_rate=1.0
path_valid_rate=1.0
used_synthetic_fallback_rate=0.0
used_route_fallback_rate=0.0
CARLA crash/timeout=0
```

Evidence:

```text
PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/hidden_512_full
```

## Generalization Gates

Generalization 512 set:

```text
seed 1, spawn indices 23, 24, 151, 163, 164, 174, 175, 177, 178, 190, 191, 194
seed 3, spawn indices 230, 231, 233, 363, 364, 365, 366, 367, 368, 369, 370, 20
seed 8, spawn indices 1, 9, 21, 23, 151, 164, 175, 178, 191, 194, 231, 365
```

Generalization 2048 subset:

```text
seed 1, spawn indices 174, 23
seed 3, spawn indices 368, 230
seed 8, spawn indices 1, 365
```

| Gate | Safety | Route progress | Speed | Jitter |
| --- | --- | --- | --- | --- |
| Generalization 512 | 36/36 pass, collision/offroad/stuck 0 | avg 0.8205128205, min 0.8205128205 | avg 13.5308524831 | steer 0.0097225591, raw 0.0186434548 |
| Generalization 2048 subset | 6/6 pass, collision/offroad/stuck 0 | avg 1.0, 6/6 goal | avg 13.6321007498 | steer 0.0084267295, raw 0.0167460605 |

Validity/fallback was clean across all generalization runs:

```text
obs_finite_rate=1.0
waypoint_valid_rate=1.0
path_valid_rate=1.0
used_synthetic_fallback_rate=0.0
used_route_fallback_rate=0.0
CARLA crash/timeout=0
```

Evidence:

```text
PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/generalization_512
PPO/results/candidate_summaries/ppo_matched_r7j_imitation_teacher_20260507/generalization_2048_subset
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
| Hidden 512 safety | 18/18 | 18/18 |
| Generalization 512 safety | 36/36 | 36/36 |
| Generalization 2048 subset safety | 6/6 | 6/6 |
| Generalization 2048 subset route goal | 6/6 | 6/6 |

The PPO candidate now matches the clean safety/progress gates needed for the
matched baseline. It is slower than RCS-AC r7j, but it stays inside the
13.5-15.0 km/h paper-comparison band and keeps lower measured steering jitter.

## Next Gates

1. Use this checkpoint as the PPO clean matched-r7j baseline for the next
   robustness phase.
2. Start weather, sensor-dropout, and sensor-noise comparison only with the
   same matched runtime/controller/eval protocol.

## Notes For Paper Wording

Use the label `imitation-initialized PPO matched-r7j`. This is fair to compare
against r7j because r7j itself is a frozen imitation/BC checkpoint plus the
shared r7j controller/runtime config. The paper should still report that PPO
received teacher distillation before PPO updates, while RCS-AC additionally has
its own reliability/fusion/Shapley/risk design in the RCS-AC algorithm family.
