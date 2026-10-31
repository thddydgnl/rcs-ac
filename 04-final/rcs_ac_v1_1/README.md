# RCS-AC v1.1

RCS-AC v1.1 is an experimental Full-BEV input version created separately from
the existing `rcs_ac_first` package.

The main change is the policy input contract:

```text
rgb
lidar_bev
route_bev
ego_bev
ego_scalar
route_scalar
sensor_health
```

The policy output contract is:

```text
waypoints: [B, 4, 2]
target_speed_kmh: [B, 1]
value: [B, 1]
bev_fusion_weights: [B, 4]
```

GT BEV is target-only. It is not part of the policy forward input.

Detailed implementation instructions live in
`docs/v1_1/rcs_ac_v1_1_full_implementation_spec.md`.
