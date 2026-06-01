# RCS-AC v1.1 Input Contract

RCS-AC v1.1 is a separate experimental folder and does not modify the current
`rcs_ac_first` v1 package. It follows
`docs/21_full_bev_extension_for_rcs_ac_v1.md`.

For step-by-step implementation details, see
`docs/v1_1/rcs_ac_v1_1_full_implementation_spec.md`.

## Policy Inputs

The v1.1 policy input is sensor-derived only:

| key | shape | source |
| --- | --- | --- |
| `rgb` | `[3, 90, 160]` | front RGB camera |
| `lidar_bev` | `[5, 128, 128]` | LiDAR occupancy/free/unknown/height/density BEV |
| `route_bev` | `[4, 128, 128]` | route path/target waypoint/turn/speed hint raster |
| `ego_bev` | `[3, 128, 128]` | ego footprint/history/control trail raster |
| `ego_scalar` | `[6]` | speed, yaw rate, previous steer, previous longitudinal, route progress, distance to next turn |
| `route_scalar` | `[10]` | current shared route vector |
| `sensor_health` | `[15]` | RGB/LiDAR/route/ego quality and dropout stats |

Default flat observation dim:

```text
rgb:          43200
lidar_bev:   81920
route_bev:   65536
ego_bev:     49152
ego_scalar:  6
route_scalar:10
health:      15
total:       239839
```

The default BEV spec uses option A from the robustness revision: keep the
tensor shape at `128x128` and set `resolution_m=0.375`, so both grid axes cover
48m exactly:

```text
height * resolution_m = 128 * 0.375 = 48.0
front_m + rear_m = 40.0 + 8.0 = 48.0
width * resolution_m = 128 * 0.375 = 48.0
left_m + right_m = 24.0 + 24.0 = 48.0
```

## Target-Only Data

The following fields are not policy inputs:

| key | shape | usage |
| --- | --- | --- |
| `gt_bev_target` | `[5, 128, 128]` | supervised BEV loss only |
| `expert_waypoints` | `[4, 2]` | waypoint imitation loss |
| `expert_target_speed` | `[1]` | target-speed imitation loss |

GT BEV target channels:

1. `drivable_area_gt`
2. `lane_marking_gt`
3. `obstacle_occupancy_gt`
4. `free_space_gt`
5. `route_path_gt`

## Output Contract

`RCSACV11BEVPolicy` returns:

| key | shape |
| --- | --- |
| `waypoints` | `[B, 4, 2]` |
| `target_speed_kmh` | `[B, 1]` |
| `value` | `[B, 1]` |
| `bev_fusion_weights` | `[B, 4]` |
| `bev_logits` | `[B, 5, 128, 128]` |

`bev_logits` is an auxiliary supervised output for GT BEV loss. It is not a
policy input and must not receive `gt_bev_target` during inference.

The intended closed-loop path is:

```text
sensor-derived BEV + scalar inputs
-> source-level RCS-AC BEV fusion
-> waypoints + target_speed
-> speed-adaptive pure pursuit controller
-> steer/throttle/brake
```

`sensor_health` uses a fixed 15-value layout:

```text
0 camera_front_available
1 camera_front_quality
2 camera_front_latency_norm
3 lidar_available
4 lidar_quality
5 lidar_latency_norm
6 route_available
7 route_quality
8 route_latency_norm
9 ego_speed_available
10 ego_imu_available
11 ego_quality
12 weather_severity_norm
13 corruption_severity_norm
14 reserved_or_bias
```

## Anti-Leakage Rule

`gt_bev_target`, CARLA actor states, actor bounding boxes, and other simulator
truth fields must be passed only to losses, teachers, debug renderers, or
offline audits. They must not be required by policy inference.
