# RCS-AC v1.1 Full Implementation Specification

작성일: 2026-05-05

이 문서는 `docs/21_full_bev_extension_for_rcs_ac_v1.md`의 설계를 실제 구현자가 문서만 보고 그대로 구현할 수 있도록 풀어 쓴 RCS-AC v1.1 상세 구현 명세다.

중요한 원칙:

- 기존 `rcs_ac_first` v1 패키지는 수정하지 않는다.
- 새 구현은 `rcs_ac_v1_1` 패키지에 둔다.
- policy inference 입력에는 센서/route/ego에서 만들 수 있는 값만 넣는다.
- CARLA 내부 정답으로 만든 GT BEV는 학습 target, teacher, debug, offline audit 전용이다.
- waypoint/target speed를 policy가 출력하고, speed-adaptive pure pursuit controller가 `steer/throttle/brake`로 변환한다.
- RCS-AC의 핵심 차별점인 source-level reliability fusion은 유지한다.

## 1. 폴더 구조

현재 만든 v1.1 기본 폴더:

```text
rcs_ac_v1_1/
  __init__.py
  README.md
  runtime/
    __init__.py
    bev_spec.py
    observation.py
    pid_controller.py
  models/
    __init__.py
    bev_policy.py
  tests/
    __init__.py
    test_bev_spec.py
    test_pure_pursuit_controller.py
    test_v1_1_bev_inputs.py
    test_waypoint_parameterization.py

configs/v1_1/
  full_bev_smoke.yaml

docs/v1_1/
  rcs_ac_v1_1_input_contract.md
  rcs_ac_v1_1_full_implementation_spec.md
```

앞으로 추가할 파일:

```text
rcs_ac_v1_1/runtime/
  bev_rasterizer.py
  gt_bev_targets.py
  debug_render.py

rcs_ac_v1_1/data/
  __init__.py
  bev_expert_dataset.py

rcs_ac_v1_1/rl/
  __init__.py
  bev_losses.py

rcs_ac_v1_1/scripts/
  __init__.py
  collect_bev_expert_dataset.py
  train_bev_bc.py
  evaluate_bev_pure_pursuit.py
  render_bev_dataset_sample.py

rcs_ac_v1_1/tests/
  test_bev_rasterizer.py
  test_bev_losses.py
  test_bev_dataset.py
  test_v1_1_no_gt_leakage.py
```

## 2. 좌표계와 BEV grid

파일: `rcs_ac_v1_1/runtime/bev_spec.py`

기본 class:

```python
@dataclass(frozen=True)
class BEVSpec:
    height: int = 128
    width: int = 128
    resolution_m: float = 0.375
    front_m: float = 40.0
    rear_m: float = 8.0
    left_m: float = 24.0
    right_m: float = 24.0
```

The default follows option A from
`/Users/song-yonghwi/Downloads/rcs_ac_v1_1_robust_control_revision.md`: keep
the tensor shape at `128x128` and set `resolution_m=0.375`.

```text
height * resolution_m = 128 * 0.375 = 48.0
front_m + rear_m = 40.0 + 8.0 = 48.0
width * resolution_m = 128 * 0.375 = 48.0
left_m + right_m = 24.0 + 24.0 = 48.0
```

`BEVSpec.validate()` must reject mismatched grid extents when
`strict_extent_check=True`.

좌표계:

```text
x: ego 기준 전방 meters
y: ego 기준 좌/우 meters
y > 0: left
y < 0: right
```

BEV tensor layout:

```text
[C, H, W]
H: x 방향 grid index
W: y 방향 grid index
```

권장 변환 규칙:

```python
def ego_xy_to_bev_row_col(x_m, y_m, spec):
    row_float = (spec.front_m - x_m) / spec.resolution_m
    col_float = (spec.left_m - y_m) / spec.resolution_m
    row = int(round(row_float))
    col = int(round(col_float))
    valid = 0 <= row < spec.height and 0 <= col < spec.width
    return row, col, valid
```

이 규칙을 쓰면:

- 전방으로 갈수록 `row`가 작아진다.
- ego 위치 `x=0, y=0`은 대략 `row = front_m / resolution`, `col = left_m / resolution`에 놓인다.
- 좌측 `y>0`은 `col`이 작아지고, 우측 `y<0`은 `col`이 커진다.

반드시 추가할 test:

```text
test_bev_spec_grid_shape
test_ego_origin_maps_inside_grid
test_forward_point_has_smaller_row_than_ego
test_left_point_has_smaller_col_than_ego
```

## 3. Channel contract

파일: `rcs_ac_v1_1/runtime/bev_spec.py`

현재 정의:

```python
@dataclass(frozen=True)
class BEVChannelSpec:
    rgb_channels: int = 4
    lidar_channels: int = 5
    route_channels: int = 4
    ego_channels: int = 3
    target_channels: int = 5
```

### 3.1 Policy input channel

Policy input에는 GT가 들어가지 않는다.

LiDAR BEV `lidar_bev: [5, 128, 128]`:

```text
0 lidar_occupancy
1 lidar_free_space
2 lidar_unknown
3 lidar_height
4 lidar_density
```

Route BEV `route_bev: [4, 128, 128]`:

```text
0 route_path
1 target_waypoint_heatmap
2 turn_region
3 speed_hint
```

Ego BEV `ego_bev: [3, 128, 128]`:

```text
0 ego_mask
1 past_ego_trajectory
2 previous_control_trail
```

RGB는 v1.1 smoke에서는 raw front RGB를 입력으로 유지한다.

```text
rgb: [3, 90, 160]
```

RGB-derived BEV channel은 다음 단계에서 `RGB -> predicted BEV` head로 만든다. 이 predicted RGB BEV는 policy 내부 feature로 들어갈 수 있지만, 지금 flat input에는 저장하지 않는다.

### 3.2 Target-only GT BEV channel

GT BEV target `gt_bev_target: [5, 128, 128]`:

```text
0 drivable_area_gt
1 lane_marking_gt
2 obstacle_occupancy_gt
3 free_space_gt
4 route_path_gt
```

이 tensor는 학습 loss에서만 사용한다.

금지:

```text
model.forward(..., gt_bev_target=...)
model.forward(..., actor_positions=...)
model.forward(..., gt_bboxes=...)
eval loop에서 gt_bev_target을 policy observation에 concatenate
```

## 4. v1.1 observation contract

파일: `rcs_ac_v1_1/runtime/observation.py`

기본 modality 순서:

```python
POLICY_MODALITY_ORDER = (
    "rgb",
    "lidar_bev",
    "route_bev",
    "ego_bev",
    "ego_scalar",
    "route_scalar",
    "sensor_health",
)
```

기본 shape:

```text
rgb:           [3, 90, 160]      = 43,200
lidar_bev:     [5, 128, 128]     = 88,320
route_bev:     [4, 128, 128]     = 70,656
ego_bev:       [3, 128, 128]     = 52,992
ego_scalar:    [6]               = 6
route_scalar:  [10]              = 10
sensor_health: [15]              = 15
total_dim:                       = 239,839
```

`build_v11_policy_observation_spec()`는 위 순서대로 flat observation slice를 만든다.

구현 규칙:

1. 모든 observation 값은 `np.float32` 또는 `torch.float32`로 변환한다.
2. flat observation은 `[B, total_dim]`이어야 한다.
3. 단일 sample `[total_dim]`은 내부에서 `[1, total_dim]`으로 확장한다.
4. NaN/inf가 있으면 즉시 `ValueError`를 낸다.
5. dict observation flatten 시 key 누락은 `KeyError`를 낸다.

현재 구현된 함수:

```python
build_v11_policy_observation_spec(...)
build_v11_target_spec(...)
split_flat_observation(flat_observation, spec)
validate_observation_dict(observation, spec)
flatten_observation_dict(observation, spec)
summarize_v11_policy_observation_spec(spec)
```

## 5. Sensor-derived BEV 생성

추가 파일: `rcs_ac_v1_1/runtime/bev_rasterizer.py`

### 5.1 LiDAR BEV rasterizer

입력 후보:

```text
lidar_points: [N, 3] 또는 [N, 4]
각 row: x, y, z, optional intensity
좌표계: ego 기준
```

출력:

```text
lidar_bev: [5, H, W]
```

구현 알고리즘:

```python
def rasterize_lidar_points(points, spec):
    lidar_bev = zeros([5, H, W], float32)
    observed = zeros([H, W], bool)

    for point in points:
        x, y, z = point[:3]
        row, col, valid = ego_xy_to_bev_row_col(x, y, spec)
        if not valid:
            continue

        occupancy[row, col] = 1
        density[row, col] += 1
        height[row, col] = max(height[row, col], z)
        observed[row, col] = True

    density = log1p(density) / log1p(density_max)
    height = clip_normalize(height, z_min=-2.0, z_max=3.0)
    free_space = raycast_free_cells_from_ego_to_each_point(...)
    unknown = 1 - clip(occupancy + free_space, 0, 1)
```

초기 smoke에서는 free-space raycast를 단순화할 수 있다:

```text
free_space = 0
unknown = 1 - occupancy
```

하지만 closed-loop 전에는 raycast를 구현한다.

반드시 저장할 값 범위:

```text
occupancy: 0 or 1
free_space: 0 or 1
unknown: 0 or 1
height: [0, 1]
density: [0, 1]
```

테스트:

```text
test_lidar_bev_rasterizer_preserves_batch_shape
test_lidar_point_in_front_sets_occupancy
test_lidar_out_of_range_point_ignored
test_lidar_density_finite
```

### 5.2 Route BEV rasterizer

입력 후보:

```text
route_waypoints_ego: [K, 2]
target_waypoint_ego: [2]
maneuver_command: straight|left|right
target_speed_hint_kmh or curvature
```

출력:

```text
route_bev: [4, H, W]
```

Channel 구현:

```text
0 route_path:
  route waypoint polyline을 BEV에 그린다.

1 target_waypoint_heatmap:
  target waypoint cell 중심에 Gaussian blob을 그린다.

2 turn_region:
  maneuver command가 left/right이면 target 주변 또는 route curve 주변을 1로 표시한다.
  straight이면 0 또는 약한 값.

3 speed_hint:
  감속 필요 구간은 낮은 값, 직선/안전 구간은 높은 값.
  처음에는 모든 route_path cell에 normalized target_speed를 넣는다.
```

Polyline rasterization 규칙:

```python
for each segment p_i -> p_{i+1}:
    sample every spec.resolution_m * 0.5
    convert sample x,y to row,col
    route_path[row, col] = 1
```

Gaussian heatmap:

```python
heatmap[row, col] = max(heatmap[row, col], exp(-d2 / (2*sigma_cells**2)))
```

테스트:

```text
test_route_rasterizer_places_target_in_expected_cell
test_route_path_forward_has_nonzero_cells
test_left_turn_region_nonzero_for_left_command
test_route_rasterizer_does_not_write_outside_grid
```

### 5.3 Ego BEV rasterizer

입력:

```text
ego_bbox_m: length, width
past_ego_positions_ego: [T, 2]
previous_steer
previous_longitudinal
```

출력:

```text
ego_bev: [3, H, W]
```

Channel 구현:

```text
0 ego_mask:
  ego footprint rectangle.

1 past_ego_trajectory:
  past positions polyline. 오래된 위치는 낮은 값, 최신 위치는 높은 값.

2 previous_control_trail:
  previous control magnitude를 ego 주변 또는 trajectory 주변에 기록.
```

초기 smoke:

```text
ego_mask만 정확히 구현
past_ego_trajectory와 previous_control_trail은 0으로 시작 가능
```

closed-loop 전:

```text
past ego 위치와 previous control trail까지 구현
```

## 6. GT BEV target 생성

추가 파일: `rcs_ac_v1_1/runtime/gt_bev_targets.py`

GT BEV는 CARLA 내부 정답을 사용하지만 학습 target 전용이다.

입력 후보:

```text
world
ego_transform
route_waypoints
nearby vehicles/pedestrians/static obstacles
semantic map or waypoint topology
```

출력:

```text
gt_bev_target: [5, H, W]
bev_target_available: float 0 or 1
```

Channel:

```text
0 drivable_area_gt
1 lane_marking_gt
2 obstacle_occupancy_gt
3 free_space_gt
4 route_path_gt
```

### 6.1 drivable_area_gt

구현 방법:

1. BEV 각 cell의 ego 좌표 `(x, y)`를 world 좌표로 변환한다.
2. CARLA map의 `get_waypoint(location, project_to_road=False, lane_type=Driving)`을 호출한다.
3. waypoint가 있고 cell이 driving lane 안이면 `1`, 아니면 `0`.

초기 최적화:

- 모든 cell을 매 step 계산하면 느리다.
- debug/smoke에서는 낮은 resolution 또는 route 주변 crop만 허용한다.
- 이후 cache를 둔다.

### 6.2 lane_marking_gt

구현 방법:

1. route 주변 waypoint를 샘플링한다.
2. lane center와 lane width를 이용해 좌/우 lane boundary를 계산한다.
3. boundary polyline을 BEV에 rasterize한다.

### 6.3 obstacle_occupancy_gt

구현 방법:

1. CARLA actors 중 ego 제외 vehicle/pedestrian/static obstacle을 모은다.
2. actor bounding box corner를 world 좌표로 구한다.
3. ego 좌표로 변환한다.
4. polygon fill로 BEV cell을 채운다.

주의:

- 이 값은 policy input 금지.
- dataset target 저장과 debug render만 허용.

### 6.4 free_space_gt

초기 구현:

```text
free_space_gt = drivable_area_gt * (1 - obstacle_occupancy_gt)
```

정교한 구현:

```text
LiDAR ray visibility 또는 semantic visibility를 반영
```

### 6.5 route_path_gt

route BEV의 `route_path`와 동일하게 route waypoint polyline을 GT target channel로 rasterize한다.

## 7. Dataset format

추가 파일: `rcs_ac_v1_1/data/bev_expert_dataset.py`

NPZ key:

```text
obs_flat: [N, 239839]

gt_bev_target: [N, 5, 128, 128]
bev_target_available: [N]

expert_waypoints: [N, 4, 2]
waypoint_target_available: [N]

expert_target_speed: [N, 1]
target_speed_available: [N]

expert_control: [N, 3]
reward: [N]
done: [N]
route_progress: [N]
episode_id: [N]
step_in_episode: [N]
done_reason: [N]
source_label: [N]
```

dtype:

```text
obs_flat: float32
gt_bev_target: float32
bev_target_available: float32
expert_waypoints: float32
waypoint_target_available: float32
expert_target_speed: float32
target_speed_available: float32
expert_control: float32
reward: float32
done: bool
route_progress: float32
episode_id: int32
step_in_episode: int32
done_reason: unicode string
source_label: unicode string
```

Validation:

```python
assert obs_flat.shape[1] == build_v11_policy_observation_spec().total_dim
assert gt_bev_target.shape[1:] == build_v11_target_spec().gt_bev_shape
assert expert_waypoints.shape[1:] == (waypoint_count, 2)
assert expert_target_speed.shape[1:] == (1,)
assert all finite float arrays
```

## 8. Model architecture

파일: `rcs_ac_v1_1/models/bev_policy.py`

현재 구현 class:

```text
V11BEVPolicyConfig
BEVSourceEncoder
RGBContextEncoder
VectorEncoder
SourceLevelBEVFusion
RCSACV11BEVPolicy
```

Forward signature:

```python
def forward(self, flat_observation: torch.Tensor) -> dict[str, torch.Tensor]:
```

금지된 forward argument:

```text
gt_bev_target
actor_positions
gt_bboxes
carla_actor_state
```

Source encoders:

```text
RGBContextEncoder:
  rgb [B, 3, 90, 160] -> feature [B, 128]

BEVSourceEncoder:
  lidar_bev [B, 5, 128, 128] -> feature [B, 128]
  route_bev [B, 4, 128, 128] -> feature [B, 128]
  ego_bev [B, 3, 128, 128] -> feature [B, 128]

VectorEncoder:
  ego_scalar [B, 6] -> feature [B, 128]
  route_scalar [B, 10] -> feature [B, 128]
```

Fusion:

```text
features:
  rgb, lidar_bev, route_bev, ego_bev

sensor_health [B, 15]
-> reliability logits [B, 4]
-> softmax fusion weights [B, 4]
-> weighted sum projected source features [B, 128]
```

Policy body:

```text
concat:
  fused_bev [B, 128]
  ego_scalar_feature [B, 128]
  route_scalar_feature [B, 128]
-> [B, 384]
-> MLP 384 -> 256 -> 256
```

Output:

```text
waypoints: [B, 4, 2]
target_speed_kmh: [B, 1], sigmoid scaled to [min_speed, max_speed]
value: [B, 1]
bev_fusion_weights: [B, 4]
bev_logits: [B, 5, 128, 128]
```

## 9. BEV prediction head

현재 구현 상태:

```text
RCSACV11BEVPolicy auxiliary BEV prediction head: implemented in rcs_ac_v1_1/models/bev_policy.py
RGBBEVHead scaffold: implemented in rcs_ac_v1_1/models/rgb_bev_head.py
```

구현 위치:

```text
rcs_ac_v1_1/models/bev_policy.py
rcs_ac_v1_1/models/rgb_bev_head.py
```

첫 구현:

```python
class BEVPredictionHead(nn.Module):
    def __init__(self, fusion_dim, target_channels, height, width):
        ...

    def forward(self, fused_feature):
        return logits  # [B, target_channels, H, W]
```

주의:

- 이 head는 `gt_bev_target`을 입력으로 받지 않는다.
- loss 함수에서만 `bev_logits`와 `gt_bev_target`을 비교한다.

Output dict에 추가할 key:

```text
bev_logits: [B, 5, 128, 128]
```

## 10. Loss implementation

추가 파일: `rcs_ac_v1_1/rl/bev_losses.py`

### 10.1 BEV target loss

Input:

```text
bev_logits: [B, 5, H, W]
gt_bev_target: [B, 5, H, W]
bev_target_available: [B]
```

Weights:

```text
drivable: 1.0
lane: 2.0
obstacle: 3.0
free_space: 1.0
route: 1.5
```

Algorithm:

```python
loss_by_channel = BCEWithLogits(pred[:, c], target[:, c], reduction='none')
loss_by_sample = mean over H,W
masked_loss = sum(loss_by_sample * available) / available.sum().clamp_min(1)
total = weighted sum channels
```

Metrics:

```text
bev_drivable_loss
bev_lane_loss
bev_obstacle_loss
bev_free_loss
bev_route_loss
bev_drivable_iou
bev_lane_iou
bev_obstacle_iou
bev_target_available_rate
```

IoU:

```python
pred_mask = sigmoid(logits) > 0.5
target_mask = target > 0.5
iou = intersection / union.clamp_min(1)
```

### 10.2 Waypoint loss

Input:

```text
pred_waypoints: [B, 4, 2]
expert_waypoints: [B, 4, 2]
waypoint_target_available: [B]
```

Loss:

```python
SmoothL1(pred_waypoints, expert_waypoints)
```

Metrics:

```text
waypoint_l1
waypoint_x_l1
waypoint_y_l1
waypoint_target_available_rate
```

### 10.3 Target speed loss

Input:

```text
pred_target_speed_kmh: [B, 1]
expert_target_speed: [B, 1]
target_speed_available: [B]
```

Loss:

```python
SmoothL1(pred_target_speed_kmh, expert_target_speed)
```

Metrics:

```text
target_speed_mae
target_speed_available_rate
```

### 10.4 Total BC loss

Config:

```yaml
bev_loss:
  enabled: true
  drivable_weight: 1.0
  lane_weight: 2.0
  obstacle_weight: 3.0
  free_space_weight: 1.0
  route_weight: 1.5

waypoint_loss:
  enabled: true
  waypoint_weight: 1.0
  target_speed_weight: 0.5
```

Total:

```text
L_total =
  L_bev
  + waypoint_weight * L_waypoint
  + target_speed_weight * L_target_speed
```

## 11. Controller

파일: `rcs_ac_v1_1/runtime/pid_controller.py`

Debug/smoke용 legacy class:

```text
PIDControllerConfig
PIDState
V11WaypointPIDController
```

v1.1 closed-loop 기본 controller:

```text
PurePursuitControllerConfig
V11SpeedAdaptivePurePursuitController
```

Input:

```python
control(
    waypoints: list[tuple[float, float]],
    *,
    target_speed_kmh: float,
    current_speed_kmh: float,
    dt: float,
    uncertainty: float = 0.0,
) -> dict[str, object]
```

Lateral:

```text
sanitize predicted waypoints
speed-dependent lookahead distance 선택
arc-length 기준 target waypoint 보간
pure pursuit curvature 계산
steer low-pass smoothing
steer rate-limit 적용
steer clipped to [-max_steer, max_steer]
```

Longitudinal:

```text
target_speed에 uncertainty speed cap 적용
target_speed에 curvature speed cap 적용
speed_error = capped_target_speed_kmh - current_speed_kmh
longitudinal = PID(speed_error)
if longitudinal >= 0:
  throttle = clip(longitudinal, 0, max_throttle)
  brake = 0
else:
  throttle = 0
  brake = clip(-longitudinal, 0, max_brake)
```

Legacy `V11WaypointPIDController`는 유지하지만 기본 evaluation/config는
`V11SpeedAdaptivePurePursuitController`를 사용한다.

Required logging:

```text
steer
throttle
brake
lookahead_m
target_x_m
target_y_m
raw_steer
curvature
target_speed_capped_kmh
speed_error_kmh
waypoint_valid
path_valid
```

## 12. Data collection script

추가 파일: `rcs_ac_v1_1/scripts/collect_bev_expert_dataset.py`

CLI:

```text
python -m rcs_ac_v1_1.scripts.collect_bev_expert_dataset \
  --config configs/v1_1/full_bev_smoke.yaml \
  --output data/expert/v1_1_bev_smoke.npz \
  --num-transitions 1024 \
  --host 127.0.0.1 \
  --port 2000 \
  --device cuda
```

Loop:

```python
env = _build_env(...)
obs = env.reset()
for step in range(num_transitions):
    raw_sensor = obs
    lidar_bev = rasterize_lidar(...)
    route_bev = rasterize_route(...)
    ego_bev = rasterize_ego(...)
    obs_flat = flatten_v11_observation(...)

    gt_bev_target = build_gt_bev_target(world, ego, route)
    expert_waypoints = build_expert_waypoints(route or future expert ego)
    expert_target_speed = build_expert_target_speed(info/control)

    save arrays
    action = expert/autopilot/control policy
    obs, reward, done, info = env.step(action)
```

Expert waypoint source priority:

1. Future ego positions from expert rollout if available.
2. Shared route planner waypoints in ego coordinates.
3. CARLA autopilot local planner trajectory.

Expert target speed source priority:

1. shared_core target speed if present in `info["target_speed_kmh"]`.
2. expert speed setpoint.
3. current speed clipped to `[0, 28]`.

Do not use GT BEV as policy input when collecting.

## 13. BC training script

추가 파일: `rcs_ac_v1_1/scripts/train_bev_bc.py`

CLI:

```text
python -m rcs_ac_v1_1.scripts.train_bev_bc \
  --config configs/v1_1/full_bev_smoke.yaml \
  --dataset data/expert/v1_1_bev_smoke.npz \
  --run-name v1_1_bev_bc_smoke \
  --device cuda
```

Training loop:

```python
model = RCSACV11BEVPolicy(config)
dataset = BEVExpertDataset(dataset_path)
for batch in train_loader:
    outputs = model(batch["obs_flat"])
    losses = compute_bev_losses(outputs, batch)
    total_loss.backward()
    optimizer.step()
```

Checkpoint fields:

```text
model_state_dict
optimizer_state_dict
config
dataset_path
epoch
global_step
val_metrics
```

Log CSV fields:

```text
epoch
global_step
train_loss
val_loss
bev_drivable_loss
bev_lane_loss
bev_obstacle_loss
bev_free_loss
bev_route_loss
bev_drivable_iou
bev_lane_iou
bev_obstacle_iou
waypoint_l1
target_speed_mae
bev_fusion_weight_rgb
bev_fusion_weight_lidar
bev_fusion_weight_route
bev_fusion_weight_ego
```

## 14. Closed-loop eval script

추가 파일: `rcs_ac_v1_1/scripts/evaluate_bev_pure_pursuit.py`

CLI:

```text
python -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit \
  --config configs/v1_1/full_bev_smoke.yaml \
  --checkpoint checkpoints/v1_1_bev_bc_smoke/last.pt \
  --run-name v1_1_bev_pure_pursuit_eval_512 \
  --num-steps 512 \
  --host 127.0.0.1 \
  --port 2000 \
  --device cuda
```

Loop:

```python
obs = env.reset()
controller = V11SpeedAdaptivePurePursuitController(...)
hidden = None
for step in range(num_steps):
    obs_flat = build_v11_policy_obs_from_sensor(obs, info)
    outputs = model(obs_flat)
    waypoints = outputs["waypoints"][0]
    target_speed = outputs["target_speed_kmh"][0]
    control = controller.control(
        waypoints,
        target_speed_kmh=target_speed,
        current_speed_kmh=info["speed_kmh"],
        dt=env.fixed_delta_seconds,
        uncertainty=info.get("sensor_uncertainty", 0.0),
    )
    obs, reward, done, info = env.step(control)
    log everything
```

Eval summary:

```text
collision_count
offroad_count
severe_early_offroad_count
stuck_count
done_reason_counts
max_route_progress
final_route_progress
mean_action_jitter_l1
mean_steer_jitter
mean_longitudinal_jitter
mean_waypoint_y_abs
mean_target_speed_kmh
bev_fusion_weight means
```

Promotion gate:

```text
512 clean: collision/offroad/severe early/stuck all 0
2048 clean: collision/offroad/severe early/stuck all 0
No PPO/robustness until both pass
```

## 15. Debug rendering

추가 파일: `rcs_ac_v1_1/runtime/debug_render.py`

Required render:

```text
lidar_occupancy
lidar_free_space
route_path
target_waypoint_heatmap
ego_mask
gt_drivable
gt_lane
gt_obstacle
gt_route
pred_drivable
pred_lane
pred_obstacle
```

Output:

```text
outputs/v1_1_debug/<run_name>/bev_step_000123.png
```

Render rules:

- ego center marker must be visible.
- forward direction arrow must be visible.
- route path must appear in front of ego for straight route.
- obstacle boxes must match CARLA debug positions in offline checks.

## 16. Anti-leakage checklist

Every implementation step must satisfy:

```text
model.forward only accepts flat_observation
gt_bev_target is only present in dataset batch/loss
actor_positions are only used inside GT target builder or debug renderer
eval policy observation is built from sensor-derived BEV, route, ego, health
no CARLA actor bbox tensor is concatenated to obs_flat
```

Required tests:

```text
test_no_gt_bev_in_policy_forward_signature
test_no_gt_bev_required_for_inference
test_eval_observation_builder_rejects_gt_keys
test_dataset_separates_obs_and_targets
```

## 17. Implementation order

Do the work in this exact order:

1. Keep current `rcs_ac_v1_1` package and tests green.
2. Implement coordinate helpers in `runtime/bev_rasterizer.py`.
3. Implement route BEV rasterizer first.
4. Implement ego BEV rasterizer.
5. Implement LiDAR BEV rasterizer with occupancy/height/density, then add free/unknown.
6. Implement GT route target first.
7. Implement obstacle occupancy target.
8. Implement drivable/lane target.
9. Add debug renderer and render synthetic tests.
10. Add `BEVExpertDataset` loader/validator.
11. Add data collection script.
12. Collect a tiny dataset, e.g. 16 transitions.
13. Add BEV loss.
14. Add waypoint/target-speed loss.
15. Add BEV prediction head.
16. Train BC smoke on tiny dataset.
17. Run offline forward/loss tests.
18. Run CARLA 256 or 512 closed-loop pure pursuit eval.
19. Only after 512 clean pass, run 2048 clean.
20. Only after clean 512/2048 pass, start corruption/reliability work.

## 18. Current implementation status

Implemented now:

```text
rcs_ac_v1_1/runtime/bev_spec.py
rcs_ac_v1_1/runtime/config.py
rcs_ac_v1_1/runtime/observation.py
rcs_ac_v1_1/runtime/pid_controller.py
rcs_ac_v1_1/models/bev_policy.py
configs/v1_1/full_bev_smoke.yaml
docs/v1_1/rcs_ac_v1_1_input_contract.md
docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
docs/v1_1/rcs_ac_v1_1_remaining_implementation_design.md
rcs_ac_v1_1/tests/test_bev_spec.py
rcs_ac_v1_1/tests/test_pure_pursuit_controller.py
rcs_ac_v1_1/tests/test_v1_1_bev_inputs.py
rcs_ac_v1_1/tests/test_v1_1_config.py
rcs_ac_v1_1/tests/test_v1_1_runtime_builders.py
rcs_ac_v1_1/tests/test_v1_1_scripts.py
rcs_ac_v1_1/tests/test_v1_1_targets_dataset_losses.py
rcs_ac_v1_1/tests/test_waypoint_parameterization.py
```

Verified:

```text
python -m py_compile rcs_ac_v1_1/runtime/bev_spec.py \
  rcs_ac_v1_1/runtime/config.py \
  rcs_ac_v1_1/runtime/observation.py \
  rcs_ac_v1_1/runtime/pid_controller.py \
  rcs_ac_v1_1/runtime/state.py \
  rcs_ac_v1_1/runtime/sensor_health.py \
  rcs_ac_v1_1/runtime/scalars.py \
  rcs_ac_v1_1/runtime/bev_rasterizer.py \
  rcs_ac_v1_1/runtime/route_bev.py \
  rcs_ac_v1_1/runtime/ego_bev.py \
  rcs_ac_v1_1/runtime/lidar_bev.py \
  rcs_ac_v1_1/runtime/rgb_bev.py \
  rcs_ac_v1_1/runtime/gt_bev_targets.py \
  rcs_ac_v1_1/runtime/waypoint_targets.py \
  rcs_ac_v1_1/runtime/pipeline.py \
  rcs_ac_v1_1/models/bev_policy.py \
  rcs_ac_v1_1/models/rgb_bev_head.py \
  rcs_ac_v1_1/data/bev_expert_dataset.py \
  rcs_ac_v1_1/rl/bev_losses.py \
  rcs_ac_v1_1/adapters/carla_lidar_adapter.py \
  rcs_ac_v1_1/adapters/carla_map_actor_adapter.py \
  rcs_ac_v1_1/adapters/carla_observation_adapter.py \
  rcs_ac_v1_1/adapters/carla_route_adapter.py \
  rcs_ac_v1_1/scripts/collect_bev_expert_dataset.py \
  rcs_ac_v1_1/scripts/train_bev_bc.py \
  rcs_ac_v1_1/scripts/evaluate_bev_pure_pursuit.py

python -m pytest rcs_ac_v1_1/tests -q
```

Current test result:

```text
47 passed
```

Implemented offline/synthetic pipeline:

```text
sensor_health builder
ego_scalar / route_scalar builder
route BEV rasterizer
ego BEV rasterizer
LiDAR BEV rasterizer
RGB perception BEV channel builder
GT BEV target builder
BEVExpertDataset loader
waypoint target builder
waypoint availability mask
BEV loss
waypoint/target-speed BC loss
train_bev_bc.py
evaluate_bev_pure_pursuit.py
CARLA adapter scaffold
collect_bev_expert_dataset.py
sensor packet -> policy component pipeline
RCSACV11BEVPolicy bev_logits auxiliary head
RGBBEVHead scaffold
```

Remaining before real CARLA experiments:

```text
Windows CARLA adapter smoke validation
tiny CARLA dataset collection run
real CARLA 64/128 eval run
RGB learned BEV checkpoint/training integration
debug renderer
```

Detailed design for these remaining items lives in:

```text
docs/v1_1/rcs_ac_v1_1_remaining_implementation_design.md
```

The concrete plan for reaching Windows CARLA 64/128-step smoke runs lives in:

```text
docs/v1_1/rcs_ac_v1_1_carla_64_128_smoke_plan.md
```

## 19. Completion criteria for v1.1 input modification

The v1.1 input modification is complete only when:

1. `obs_flat` can be built from `rgb/lidar_bev/route_bev/ego_bev/ego_scalar/route_scalar/sensor_health`.
2. `RCSACV11BEVPolicy.forward(obs_flat)` returns waypoints and target speed without GT inputs.
3. GT BEV target is stored separately and only used by losses.
4. Unit tests prove shape correctness, BEV extent validation, constrained waypoint output, controller output range, fusion weight normalization, and no-GT forward signature.
5. A tiny offline dataset can load and compute loss without NaN.
6. A CARLA closed-loop smoke can run with pure pursuit using predicted waypoints/target speed.

At the current state, items 1-4 are implemented for the scaffold, including config parsing and consistency checks. Items 5-6 remain future work.
