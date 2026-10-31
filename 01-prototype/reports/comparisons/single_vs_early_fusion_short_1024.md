# Single vs Early Fusion Short Comparison

## Runs

- Single: [training_log.csv](C:/Users/thddy/Documents/code/logs/compare_single_short_1024_20260318T215212/training_log.csv)
- Early fusion: [training_log.csv](C:/Users/thddy/Documents/code/logs/compare_early_fusion_short_1024_retry_20260318T215407/training_log.csv)

## Fixed Conditions

- `town=Town04`
- `weather=default`
- `traffic_mode=none`
- `seed=7`
- `total_timesteps=1024`
- `max_episode_steps=384`
- reward version: `pair1_frozen_v1`
- same done conditions
- same action semantics

## Comparison Table

| metric | single | early_fusion |
| --- | ---: | ---: |
| episodes | 4 | 4 |
| avg_route_progress | 0.064103 | 0.064103 |
| max_route_progress | 0.128205 | 0.153846 |
| avg_episode_speed_kmh | 2.927888 | 3.912167 |
| offroad_ratio | 0.750 | 0.750 |
| collision_ratio | 0.000 | 0.000 |
| max_steps_ratio | 0.250 | 0.000 |
| stuck_ratio | 0.000 | 0.250 |
| goal_count | 0 | 0 |
| min_target_distance_min | 91.017433 | 88.839455 |

## Interpretation

- Early fusion is trainable under the same budget and conditions as single.
- Early fusion did not improve average route progress yet, but it slightly improved best route progress and minimum target distance.
- Early fusion also produced one `stuck` episode where single produced one `max_steps` episode.
- At this stage, early fusion looks operational but not clearly better than single under a 1024-step short budget.
