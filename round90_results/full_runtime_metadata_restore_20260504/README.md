# Round90 Full Runtime Metadata Restore

Date: 2026-05-04

## Purpose

After PPO baseline work, Round90 clean revalidation no longer reproduced the
stored offroad-zero 2048 result. Earlier checks showed the checkpoint, config,
and evaluator were not the main difference. This pass restored the missing
shared-core runtime/action metadata so that the current runtime can be compared
against the old PASS logs without blind spots.

## Shared-Core Logging Fixes

Windows shared-core `env/carla_env.py` was patched to restore logging only. No
steering rule, runtime assist, reward, checkpoint, or policy control behavior was
changed.

Restored fields include:

- `policy_diagnostics` accepted by `env.step(...)` instead of being discarded
- fusion/reliability/risk diagnostics in `env_step_log.csv`
- target-speed action resolution: `target_speed_kmh`, `speed_error_kmh`,
  `raw_requested_throttle`, `raw_requested_brake`, smoothed action values
- action/runtime metadata: action contract, action space, smoothing flags,
  lateral recovery flags
- sensor quality/spec metadata: RGB brightness/contrast, LiDAR valid ratio,
  RGB/LiDAR specs, corruption/dropout metadata
- append-mode `env_step_log.csv`, so reset across multiple episodes no longer
  overwrites earlier rows in the same evaluation run

Repository helper scripts:

- `tools/patch_shared_core_runtime_metadata.py`
- `tools/patch_shared_core_append_env_log.py`
- `scripts/windows/check_shared_core_carla_env.ps1`

## Verification

### 128-step metadata probe

Run:

`rgb_perception_actor_features_v2_round90_full_runtime_metadata_probe_128_20260504T_20260504T160453`

Result:

- completed steps: `128`
- collision/offroad/stuck: `0/0/0`
- all checked runtime metadata columns were non-empty for `129/129` rows
- mean `fusion_weight_rgb`: `0.00578`
- mean `fusion_weight_route`: `0.20070`
- mean `rgb_brightness`: `0.44765`
- mean `lidar_valid_ratio`: `0.40181`

### 2048 revalidation before append fix

Run:

`rgb_perception_actor_features_v2_round90_full_runtime_metadata_2048_20260504T1610_20260504T160621`

Result:

- done reasons: `{"max_steps": 3, "offroad": 1}`
- collision/stuck/severe early offroad: `0/0/0`
- max route: `0.461538`
- full runtime fields were present in `bc_eval_log.csv`
- `env_step_log.csv` still only kept the last episode, revealing the overwrite
  issue fixed afterward

### 2048 revalidation after append fix

Run:

`rgb_perception_actor_features_v2_round90_full_runtime_metadata_append_2048_20260504T1620_20260504T161237`

Result:

- done reasons: `{"max_steps": 2, "offroad": 2}`
- collision/stuck/severe early offroad: `0/0/0`
- max route: `0.461538`
- env rows: `2053`
- env episode ids: `1, 2, 3, 4, 5`

This confirms the full run-level `env_step_log.csv` is now preserved across
episode resets.

## Current Versus Stored PASS

Stored Round90 PASS 2048 log:

- done reasons: `{"max_steps": 4}`
- mean speed: `6.3005 km/h`
- mean `fusion_rgb/lidar/ego/route`: `0.0144 / 0.0693 / 0.6509 / 0.2653`

Current append-fixed 2048:

- done reasons: `{"max_steps": 2, "offroad": 2}`
- mean speed: `7.6381 km/h`
- mean `fusion_rgb/lidar/ego/route`: `0.0070 / 0.0332 / 0.7862 / 0.1736`

Critical route window `0.30-0.34`:

| Metric | Stored PASS | Current append-fixed |
| --- | ---: | ---: |
| rows | `279` | `112` |
| speed mean km/h | `6.3047` | `8.8360` |
| eval brake/longitudinal mean | `0.7705` | `0.6282` |
| steer mean | `0.0459` | `-0.0723` |
| fusion RGB mean | `0.0162` | `0.0066` |
| fusion LiDAR mean | `0.0760` | `0.0289` |
| fusion ego mean | `0.6515` | `0.8203` |
| fusion route mean | `0.2563` | `0.1442` |
| target speed mean km/h | unavailable in old log | `10.0899` |
| speed error mean km/h | unavailable in old log | `1.2201` |

Important note: in `bc_eval_log.csv`, `brake`/`longitudinal` represent the
policy/evaluator action-side longitudinal signal. The restored `env_step_log.csv`
now also records actual CARLA control throttle/brake after target-speed
resolution and smoothing. In the append-fixed 2048 run, actual env control over
all rows was mean throttle `0.1683`, mean brake `0.0192`, while the evaluator
longitudinal column was mostly negative. This distinction matters for diagnosing
why the current runtime drives faster.

## Interpretation

The remaining difference is not the Round90 checkpoint or YAML. It is still a
runtime/action-contract margin issue under the current shared-core behavior:

1. Current runs are faster than the stored PASS run, especially near route
   `0.30-0.34`.
2. Current fusion is much more ego-dominant and gives less weight to RGB/LiDAR
   and route.
3. The same left/turn-heavy route window repeatedly produces late offroad.
4. Because Round90 was already a narrow clean baseline, this runtime drift is
   enough to turn a previously offroad-zero run into offroad 1-2/2048.

## Next Step

Do not treat Round90 as a robust final clean checkpoint under the current
shared-core runtime. The next useful step is to decide whether to:

- preserve Round90 as a historical clean baseline with its original runtime
  contract, or
- re-stabilize/evaluate a new clean baseline under the current shared reward and
  shared-core runtime contract, using the restored metadata to constrain speed,
  fusion, and route-window failure behavior.
