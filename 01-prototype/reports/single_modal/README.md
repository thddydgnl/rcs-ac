# Single-Modal Baseline Package

Current official `single_modal` baseline is the fresh Pair 1 run backed by [`train_single_modal.yaml`](/C:/Users/thddy/Documents/code/configs/train_single_modal.yaml).

## Official Baseline

- Model: `single_modal`
- Baseline config: [`train_single_modal.yaml`](/C:/Users/thddy/Documents/code/configs/train_single_modal.yaml)
- Official training log: [`training_log.csv`](/C:/Users/thddy/Documents/code/logs/inspect_pair1_baseline_fresh_20260318T200348/training_log.csv)
- Official checkpoint: [`last.pt`](/C:/Users/thddy/Documents/code/checkpoints/inspect_pair1_baseline_fresh_20260318T200348/last.pt)
- Official env log: [`logs.csv`](/C:/Users/thddy/Documents/code/outputs/ppo_single_modal_env/inspect_pair1_baseline_fresh_20260318T200348/logs.csv)

## Result Summary

| Metric | Value |
| --- | ---: |
| goal_count | 1 |
| goal_ratio | 0.034 |
| done_reason_distribution | `{'collision': 4, 'offroad': 13, 'max_steps': 11, 'goal': 1}` |
| avg_route_progress | 0.404951 |
| max_route_progress | 1.000000 |
| min_target_distance_min | 1.282000 |
| avg_episode_speed_kmh | 12.621632 |
| offroad_ratio | 0.448 |
| max_steps_ratio | 0.379 |
| stuck_ratio | 0.000 |

## Interpretation

This baseline preserves the forward-driving behavior learned by Pair 1 and now includes a first successful `goal` episode.

Route reachability is meaningfully established: `max_route_progress=1.0` and `min_target_distance_min=1.282` show that the policy can finish the route.

The main remaining weakness is late-run stability, because `offroad_ratio=0.448` indicates that final-lap drift and offroad failure still occur regularly.

## Fixed Comparison Metrics

Use the following metrics for all future `early_fusion` and other model result tables so comparisons remain directly aligned:

| Metric | Purpose |
| --- | --- |
| `goal_count` | Whether the model actually finishes episodes |
| `goal_ratio` | Goal success frequency |
| `done_reason_distribution` | Failure mode mix |
| `avg_route_progress` | Average forward progress |
| `max_route_progress` | Best achieved route completion |
| `min_target_distance_min` | Closest approach to the goal |
| `avg_episode_speed_kmh` | Average movement level |
| `offroad_ratio` | Lane departure / route failure rate |
| `max_steps_ratio` | How often episodes survive to the time budget |
| `stuck_ratio` | Whether the policy collapses into immobility |

## Deprecated Experiments

- `pair2`: combined extra reward axes and degraded both stability and offroad behavior.
- `steps448`: did not show reliable improvement over the 384-step Pair 1 baseline.
- `laneoffset/control smoothing` follow-ups: cancelled or discarded because they did not establish a better baseline than the fresh Pair 1 run.
