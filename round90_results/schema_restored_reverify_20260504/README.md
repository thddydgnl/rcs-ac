# Round90 Schema-Restored Revalidation

Date: 2026-05-04

## Purpose

Re-run original Round90 after restoring the Windows `shared_core` environment log
contract to produce `env_step_log.csv` with route geometry fields.

This was done to make the fresh revalidation comparable with the stored
offroad-zero logs.

## Shared-Core Schema Restore

The Windows runtime at:

`C:\Users\thddy\Documents\code\shared_core\env\carla_env.py`

was patched to:

- write `env_step_log.csv`
- use `ENV_STEP_LOG_FIELDS`
- include route geometry in `info` and env logs:
  - `route_representation`
  - `target_x_local`, `target_y_local`
  - `lookahead_turn_delta_rad`
  - `route_waypoint_count`
  - `route_total_length_m`
  - `route_heading_mismatch`

The patch helper is:

`tools/patch_shared_core_route_log.py`

## Results

| Run | Steps | Done reasons | Offroad | Env log | Route fields |
| --- | ---: | --- | ---: | --- | --- |
| original schema-restored 512 | 512 | `{"max_steps": 1}` | 0 | `env_step_log.csv` | populated |
| original schema-restored 2048 | 2048 | `{"max_steps": 2, "offroad": 2}` | 2 | `env_step_log.csv` | populated |

2048 details:

- collision count: `0`
- severe early offroad count: `0`
- stuck rate: `0.0`
- max route progress: `0.461538`
- mean route progress: `0.198255`
- mean action jitter L1: `0.137479`
- visual-center correction rate: `0.562500`
- mean route target heading: `-0.549803`
- `route_target_x` nonzero rows: `2048/2048`
- `route_waypoint_count` nonzero rows: `2048/2048`

## Interpretation

The original Round90 logs are now schema-comparable with the stored PASS logs:
route geometry is present and nonzero throughout the eval.

The 2048 clean safety result still fails with late offroad. That means the
remaining mismatch with the old offroad-zero runs is not caused by route fields
being missing from the policy observation or from the merged evaluator log.

Round90 should not be considered a repeatably locked 2048 clean baseline under
the current Windows CARLA/shared-core runtime until repeated 2048 runs return
offroad `0`.

