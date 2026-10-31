# Early Fusion Short Validation

## Baseline

- Model: `early_fusion`
- Variant: `early_fusion_concat`
- Config: [train_early_fusion.yaml](C:/Users/thddy/Documents/code/configs/train_early_fusion.yaml)
- Comparison target: [train_single_modal.yaml](C:/Users/thddy/Documents/code/configs/train_single_modal.yaml)

## Short Training Run

- Training log: [training_log.csv](C:/Users/thddy/Documents/code/logs/compare_early_fusion_short_1024_retry_20260318T215407/training_log.csv)
- Step log: [logs.csv](C:/Users/thddy/Documents/code/outputs/ppo_early_fusion_env/compare_early_fusion_short_1024_retry_20260318T215407/logs.csv)
- Checkpoint: [last.pt](C:/Users/thddy/Documents/code/checkpoints/compare_early_fusion_short_1024_retry_20260318T215407/last.pt)

## Fixed Comparison Conditions

- `town=Town04`
- `weather=default`
- `traffic_mode=none`
- `seed=7`
- `total_timesteps=1024`
- `max_episode_steps=384`
- reward version: `pair1_frozen_v1`
- done thresholds and action semantics unchanged from single baseline

## Short-Run Summary

| metric | value |
| --- | ---: |
| goal_count | 0 |
| goal_ratio | 0.000 |
| done_reason_distribution | `{'stuck': 1, 'offroad': 3}` |
| avg_episode_length | 189.500 |
| avg_route_progress | 0.064103 |
| max_route_progress | 0.153846 |
| avg_episode_speed_kmh | 3.912167 |
| offroad_ratio | 0.750 |
| max_steps_ratio | 0.000 |
| stuck_ratio | 0.250 |
| min_target_distance_min | 88.839455 |

## Initial Interpretation

- Early fusion completed updates and produced episode-level logs, so the baseline is operational.
- In this short budget it did not collapse to NaN or action/log-prob failure.
- It slightly exceeded single on best route progress, but introduced one `stuck` episode and did not reduce offroad.
- This is enough to move to the next baseline stage, but not enough to claim performance gains yet.
