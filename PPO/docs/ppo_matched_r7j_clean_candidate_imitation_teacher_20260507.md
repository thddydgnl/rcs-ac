# PPO Matched-r7j Clean Candidate - Imitation Teacher 2026-05-07

## Status

Candidate only. This checkpoint passed the first single-start 512/2048 clean
gate, but it is not yet the final PPO clean matched-r7j baseline because the
primary 9-start multistart, generalization, and hidden gates have not run.

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

## RCS-AC r7j Comparison

| Metric | RCS-AC r7j reference | PPO candidate |
| --- | ---: | ---: |
| Primary 512 safety | 9/9 | single-start PASS |
| Primary 512 avg max_route_progress | 0.840456 | 0.820513 |
| Primary 512 min max_route_progress | 0.794872 | 0.820513 |
| Primary 2048 safety | 9/9 | single-start PASS |
| Primary 2048 avg max_route_progress | 1.0 | 1.0 |
| Primary 2048 route goal | 9/9 | single-start goal |
| Mean speed | 13.74-14.07 | 13.53-13.64 |
| steer_jitter | 0.019-0.020 | 0.008-0.010 |

The PPO candidate is close enough on the first start to justify primary
multistart testing. It is not yet a paper-ready clean baseline.

## Next Gates

1. Build or reuse a PPO-only multistart runner that writes results under
   `PPO/results`.
2. Run primary 512 9-start clean gate.
3. If 9/9 safety passes, run primary 2048 9-start clean gate.
4. Only after primary gates pass, run generalization 512, hidden 512, and
   generalization 2048 subset gates.
5. Write the final PPO clean matched-r7j baseline manifest only after these
   gates pass.

## Notes For Paper Wording

Use the label `imitation-initialized PPO matched-r7j`. This is fair to compare
against r7j because r7j itself is a frozen imitation/BC checkpoint plus the
shared r7j controller/runtime config. The paper should still report that PPO
received teacher distillation before PPO updates, while RCS-AC additionally has
its own reliability/fusion/Shapley/risk design in the RCS-AC algorithm family.
