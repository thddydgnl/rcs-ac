# PPO Matched-r7j Logging Schema

Created: 2026-05-07

PPO matched-r7j eval writes the same per-step CSV schema used by:

```text
RCS-AC-v1.1/code/rcs_ac_v1_1/scripts/evaluate_bev_pure_pursuit.py
```

Primary per-step log:

```text
v1_1_eval_steps.csv
```

Primary summary:

```text
summary.json
```

Important metric groups:

| Group | Fields |
|---|---|
| Safety | `collision`, `offroad`, `stuck`, `done_reason` |
| Progress | `route_progress`, `route_progress_delta` |
| Speed/control | `speed_kmh`, `target_speed_kmh_raw`, `target_speed_capped_kmh`, `steer`, `throttle`, `brake` |
| Raw policy | `policy_waypoint_0_x/y` through `policy_waypoint_3_x/y`, `target_speed_kmh_raw` |
| Controller | `controller_target_y_m`, `controller_raw_target_y_m`, `controller_curvature`, `raw_steer`, `smoothed_steer`, `rate_limited_steer` |
| Validity | `waypoint_valid`, `path_valid`, `used_synthetic_fallback`, `used_route_fallback`, `obs_finite` |
| Sensor health | `sensor_health_0` through `sensor_health_14`, camera/lidar/route/ego quality aliases |
| Jitter analysis | compatible with `analyze_eval_jitter.py` |

Compatibility note:

```text
bev_fusion_weight_rgb/lidar/route/ego are not PPO learned reliability fusion.
They are availability-normalized logging compatibility weights so the shared
r7j analyzer can read the same CSV schema.
```

Required reporting distinction:

```text
raw policy output = PPO waypoints + PPO target_speed_kmh
controller output = r7j pure-pursuit steer/throttle/brake before env step
CARLA applied control = shared env direct_3d [steer, throttle, brake]
```
