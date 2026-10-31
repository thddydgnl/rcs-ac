# RCS-AC v1.1 Remaining Implementation Design

작성일: 2026-05-05

이 문서는 `rcs_ac_v1_1`에서 누락되어 있던 offline/synthetic 구현부를 실제 코드로 옮기기 위해 작성한 상세 구현 명세다.
현재 이 문서의 1-14번 구현 항목은 코드에 반영되었고, 남은 작업은 CARLA adapter, 실제 CARLA dataset collection/eval loop, RGB learned BEV checkpoint, robustness 실험 연결이다.

읽는 순서:

```text
1. docs/rcs_ac_project_handoff_for_new_codex_chat.md
2. docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
3. 이 문서
```

현재 완료된 기반:

```text
BEVSpec / coordinate helpers
observation flatten/split contract
RCSACV11BEVPolicy scaffold
monotonic/bounded waypoint parameterization
speed-adaptive pure pursuit controller
config parser and consistency checks
runtime builders
BEV rasterizers
target builders
BEVExpertDataset
BEV/waypoint/speed losses
train_bev_bc.py CPU smoke
evaluate_bev_pure_pursuit.py synthetic smoke
sensor packet -> policy component/flat observation pipeline
CARLA adapter scaffold
collect_bev_expert_dataset.py
evaluate_bev_pure_pursuit.py --mode carla
unit tests: 47 passed
```

이 문서의 구현 대상과 현재 상태:

```text
1. sensor_health builder: implemented
2. ego_scalar / route_scalar builder: implemented
3. route BEV rasterizer: implemented
4. ego BEV rasterizer: implemented
5. LiDAR BEV rasterizer: implemented
6. RGB perception BEV channel builder: placeholder/head scaffold implemented
7. GT BEV target builder: synthetic adapter contract implemented
8. BEVExpertDataset loader: implemented
9. waypoint target builder: implemented
10. waypoint availability mask: implemented
11. BEV loss: implemented
12. waypoint/target-speed BC loss: implemented
13. train_bev_bc.py: offline CPU smoke implemented
14. evaluate_bev_pure_pursuit.py: synthetic smoke implemented
15. sensor packet -> policy component/flat observation pipeline: implemented
16. CARLA env-specific adapter scaffold and real closed-loop eval mode: implemented
17. Windows CARLA 64/128 smoke execution: remaining
```

---

## 1. 전체 구현 원칙

정책 inference input에는 다음만 들어간다.

```text
rgb
lidar_bev
route_bev
ego_bev
ego_scalar
route_scalar
sensor_health
```

다음은 policy inference input에 들어가면 안 된다.

```text
gt_bev_target
CARLA actor positions
CARLA bounding boxes
CARLA semantic ground truth
future expert path
expert control
```

허용되는 사용:

```text
GT BEV: 학습 target, teacher, debug, offline audit
future expert path: waypoint target 생성, dataset target
CARLA actor state: GT BEV target 생성, debug
```

권장 구현 방식:

```text
runtime 모듈은 numpy/torch 기반 순수 함수로 만든다.
CARLA 객체를 직접 받는 코드는 scripts 또는 adapters에 둔다.
unit test는 synthetic dict/array로 검증한다.
```

이 원칙을 지키면 CARLA가 없는 Mac에서도 대부분의 로직을 테스트할 수 있다.

---

## 2. 공통 데이터 계약

새 runtime builder들은 CARLA 객체 대신 다음 형태의 dict 또는 dataclass를 받도록 설계한다.

### 2.1 Ego State

권장 파일:

```text
rcs_ac_v1_1/runtime/state.py
```

권장 dataclass:

```python
@dataclass(frozen=True)
class V11EgoState:
    speed_kmh: float
    yaw_rate_rad_s: float
    acceleration_mps2: float = 0.0
    previous_steer: float = 0.0
    previous_throttle: float = 0.0
    previous_brake: float = 0.0
    previous_longitudinal: float = 0.0
    route_progress: float = 0.0
    distance_to_next_turn_m: float = 0.0
```

`previous_longitudinal` 정의:

```text
previous_throttle - previous_brake
```

단, 기존 env/action wrapper가 longitudinal을 별도로 기록하면 그 값을 우선 사용한다.

### 2.2 Route Context

권장 dataclass:

```python
@dataclass(frozen=True)
class V11RouteContext:
    route_points_ego: np.ndarray        # [N, 2], ego frame, x forward, y left
    target_point_ego: tuple[float, float] | None
    maneuver: str                       # "left", "right", "straight", "lane_follow", "unknown"
    route_progress: float = 0.0
    distance_to_next_turn_m: float = 0.0
    target_speed_kmh: float = 0.0
    curvature: float = 0.0
    heading_error_rad: float = 0.0
    available: bool = True
```

`route_points_ego`는 ego 기준 전방 route centerline이다.

검증:

```text
shape == [N,2]
finite
N >= 2이면 route_available=True
x가 모두 음수이면 route_available=False
```

### 2.3 Sensor Packet

권장 dataclass:

```python
@dataclass(frozen=True)
class V11SensorPacket:
    rgb_front: np.ndarray | None        # [3,H,W] or [H,W,3]
    lidar_points_ego: np.ndarray | None # [N,3], x/y/z ego frame
    camera_latency_ms: float = 0.0
    lidar_latency_ms: float = 0.0
    route_latency_ms: float = 0.0
    weather_severity_norm: float = 0.0
    corruption_severity_norm: float = 0.0
    camera_dropout: bool = False
    lidar_dropout: bool = False
    route_dropout: bool = False
```

scripts에서 CARLA observation/info를 이 dataclass로 변환한다.

---

## 3. sensor_health Builder

권장 파일:

```text
rcs_ac_v1_1/runtime/sensor_health.py
```

출력:

```text
sensor_health: np.ndarray shape [15], dtype float32
```

layout은 현재 input contract와 동일하다.

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

### 3.1 함수 계약

```python
def build_sensor_health(
    *,
    rgb_front: np.ndarray | None,
    lidar_points_ego: np.ndarray | None,
    route_context: V11RouteContext | None,
    ego_state: V11EgoState | None,
    camera_latency_ms: float = 0.0,
    lidar_latency_ms: float = 0.0,
    route_latency_ms: float = 0.0,
    weather_severity_norm: float = 0.0,
    corruption_severity_norm: float = 0.0,
    camera_dropout: bool = False,
    lidar_dropout: bool = False,
    route_dropout: bool = False,
) -> np.ndarray:
```

### 3.2 Camera availability

```python
available = (
    rgb_front is not None
    and not camera_dropout
    and rgb_front.size > 0
    and np.isfinite(rgb_front).mean() > 0.99
)
```

### 3.3 Camera quality

RGB는 `[0,1]` float 또는 `[0,255]` uint 모두 허용한다. 내부에서 `[0,1]`로 정규화한다.

권장 proxy:

```text
finite_score = finite ratio
brightness_score = 1 - abs(mean_luma - 0.5) / 0.5
contrast_score = clip(std_luma / 0.25, 0, 1)
edge_score = clip(mean_abs_gradient / 0.10, 0, 1)
dropout_score = 0 if dropout else 1
quality = finite_score * dropout_score * mean(brightness_score, contrast_score, edge_score)
```

`edge_score`는 OpenCV 없이 구현한다.

```python
gx = np.abs(np.diff(luma, axis=-1)).mean()
gy = np.abs(np.diff(luma, axis=-2)).mean()
mean_abs_gradient = gx + gy
```

값은 `[0,1]`로 clip한다.

### 3.4 LiDAR availability

```python
available = (
    lidar_points_ego is not None
    and not lidar_dropout
    and lidar_points_ego.ndim == 2
    and lidar_points_ego.shape[1] >= 3
    and finite point count >= min_lidar_points
)
```

기본:

```text
min_lidar_points = 32
expected_lidar_points = 4096
```

### 3.5 LiDAR quality

권장 proxy:

```text
finite_ratio
count_score = clip(log1p(valid_count) / log1p(expected_lidar_points), 0, 1)
range_score = clip(points_inside_bev_count / max(valid_count,1), 0, 1)
height_score = 1 if z finite distribution is not degenerate else 0.5
quality = finite_ratio * mean(count_score, range_score, height_score)
```

### 3.6 Route availability and quality

availability:

```text
route_context is not None
route_context.available
not route_dropout
route_points_ego shape [N,2]
N >= 2
finite
at least one point has x > 0
```

quality:

```text
continuity_score = fraction of segment lengths in [0.1m, 8.0m]
forward_score = fraction of points with x >= -1.0
length_score = clip(total_forward_path_length / 20.0, 0, 1)
heading_score = 1 - clip(abs(heading_error_rad) / pi, 0, 1)
quality = mean(continuity_score, forward_score, length_score, heading_score)
```

### 3.7 Ego availability and quality

```text
ego_speed_available = finite speed_kmh
ego_imu_available = finite yaw_rate_rad_s and acceleration_mps2
ego_quality = 1.0 if both available else 0.0 or 0.5
```

Hard fail:

```text
NaN/inf가 sensor_health에 남으면 안 된다.
```

### 3.8 Tests

권장 테스트 파일:

```text
rcs_ac_v1_1/tests/test_sensor_health.py
```

테스트:

```text
test_sensor_health_shape_and_range
test_camera_dropout_sets_availability_zero
test_lidar_empty_sets_availability_zero
test_route_invalid_sets_availability_zero
test_ego_nan_reduces_quality
test_weather_and_corruption_severity_are_clipped
```

완료 기준:

```text
shape == (15,)
dtype == np.float32
all finite
all values in [0,1]
dropout source availability == 0
reserved_or_bias == 1
```

---

## 4. ego_scalar / route_scalar Builder

권장 파일:

```text
rcs_ac_v1_1/runtime/scalars.py
```

### 4.1 Ego scalar

현재 observation contract:

```text
ego_scalar_dim = 6
```

순서:

```text
0 speed_norm
1 yaw_rate_norm
2 previous_steer
3 previous_longitudinal
4 route_progress_norm
5 distance_to_next_turn_norm
```

함수:

```python
def build_ego_scalar(
    ego_state: V11EgoState,
    *,
    max_speed_kmh: float = 40.0,
    max_abs_yaw_rate_rad_s: float = 1.5,
    max_distance_to_turn_m: float = 100.0,
) -> np.ndarray:
```

계산:

```text
speed_norm = clip(speed_kmh / max_speed_kmh, 0, 1)
yaw_rate_norm = clip(yaw_rate_rad_s / max_abs_yaw_rate_rad_s, -1, 1)
previous_steer = clip(previous_steer, -1, 1)
previous_longitudinal = clip(previous_longitudinal, -1, 1)
route_progress_norm = clip(route_progress, 0, 1)
distance_to_next_turn_norm = clip(distance_to_next_turn_m / max_distance_to_turn_m, 0, 1)
```

주의:

```text
문서 표기는 speed/yaw_rate지만 실제 policy 입력은 정규화된 값으로 고정한다.
학습 로그에는 raw 값과 norm 값을 모두 저장하는 것이 좋다.
```

### 4.2 Route scalar

현재 observation contract:

```text
route_scalar_dim = 10
```

순서:

```text
0 target_x_norm
1 target_y_norm
2 heading_error_sin
3 heading_error_cos
4 curvature_norm
5 distance_to_next_turn_norm
6 maneuver_left
7 maneuver_right
8 maneuver_straight_or_lane_follow
9 route_available
```

함수:

```python
def build_route_scalar(
    route_context: V11RouteContext | None,
    *,
    max_target_x_m: float = 40.0,
    max_abs_target_y_m: float = 12.0,
    max_abs_curvature: float = 0.5,
    max_distance_to_turn_m: float = 100.0,
) -> np.ndarray:
```

계산:

```text
target_x_norm = clip(target_x / max_target_x_m, 0, 1)
target_y_norm = clip(target_y / max_abs_target_y_m, -1, 1)
heading_error_sin = sin(heading_error_rad)
heading_error_cos = cos(heading_error_rad)
curvature_norm = clip(curvature / max_abs_curvature, -1, 1)
distance_to_next_turn_norm = clip(distance_to_next_turn_m / max_distance_to_turn_m, 0, 1)
maneuver_left = 1 if maneuver == "left" else 0
maneuver_right = 1 if maneuver == "right" else 0
maneuver_straight_or_lane_follow = 1 if maneuver in {"straight","lane_follow"} else 0
route_available = 1 if valid else 0
```

route invalid이면:

```text
all zeros except heading_error_cos = 1.0
route_available = 0.0
```

### 4.3 Tests

권장 테스트:

```text
test_ego_scalar_shape_range
test_route_scalar_shape_range
test_route_scalar_left_right_one_hot
test_route_scalar_invalid_route_safe_default
test_scalar_nan_rejected_or_sanitized
```

완료 기준:

```text
ego_scalar.shape == (6,)
route_scalar.shape == (10,)
finite
documented order와 config/order가 일치
```

---

## 5. 공통 BEV Rasterizer Utility

권장 파일:

```text
rcs_ac_v1_1/runtime/bev_rasterizer.py
```

공통 primitive:

```python
def make_empty_bev(channels: int, spec: BEVSpec) -> np.ndarray
def draw_disk(channel: np.ndarray, row: int, col: int, radius_cells: int, value: float) -> None
def draw_polyline(channel: np.ndarray, points_ego: np.ndarray, spec: BEVSpec, radius_cells: int, value: float) -> None
def draw_gaussian(channel: np.ndarray, x_m: float, y_m: float, spec: BEVSpec, sigma_m: float, peak: float = 1.0) -> None
def rasterize_points(points_ego: np.ndarray, spec: BEVSpec) -> tuple[np.ndarray, np.ndarray]
```

구현 원칙:

```text
OpenCV 없이 numpy만으로 unit test 가능해야 한다.
모든 좌표 변환은 ego_xy_to_bev_row_col()을 사용한다.
channel 값은 특별히 signed로 정의한 경우를 제외하고 [0,1]로 clip한다.
```

`draw_polyline` 첫 구현:

```text
연속 point 사이를 일정 간격으로 linear interpolation한다.
간격은 resolution_m * 0.5 이하.
각 sample point에 disk를 찍는다.
```

이 방식은 Bresenham보다 느리지만 구현과 테스트가 쉽다.

---

## 6. Route BEV Rasterizer

권장 파일:

```text
rcs_ac_v1_1/runtime/route_bev.py
```

출력:

```text
route_bev: [4, H, W], dtype float32
```

channel:

```text
0 route_path
1 target_waypoint_heatmap
2 turn_region
3 speed_hint
```

함수:

```python
def build_route_bev(
    route_context: V11RouteContext | None,
    spec: BEVSpec,
    *,
    path_radius_m: float = 0.75,
    target_sigma_m: float = 1.0,
    turn_radius_m: float = 2.0,
    max_speed_kmh: float = 40.0,
) -> np.ndarray:
```

### 6.1 route_path

```text
route_points_ego를 polyline으로 rasterize한다.
value = 1.0
radius_cells = ceil(path_radius_m / resolution_m)
```

### 6.2 target_waypoint_heatmap

```text
target_point_ego가 있으면 Gaussian heatmap을 찍는다.
target_point_ego가 없으면 route_points_ego에서 x가 6~8m 근처인 point를 target으로 쓴다.
```

### 6.3 turn_region

signed channel로 정의한다.

```text
left turn: +1.0
right turn: -1.0
straight/lane_follow/unknown: 0.0
```

route path 중 target 주변 또는 distance_to_next_turn_m 주변에 disk를 찍는다.

주의:

```text
turn_region은 signed channel이므로 [0,1] clip을 하지 않는다.
```

### 6.4 speed_hint

route path 위에 normalized target speed를 rasterize한다.

```text
speed_hint_value = clip(target_speed_kmh / max_speed_kmh, 0, 1)
```

route invalid이면:

```text
모든 channel zero
```

### 6.5 Tests

```text
test_route_bev_shape
test_route_path_forward_points_are_drawn
test_route_target_heatmap_peak_near_target
test_route_turn_region_left_positive_right_negative
test_route_invalid_returns_zero
```

완료 기준:

```text
shape == (4,128,128)
finite
route forward point가 expected cell 근처에 rasterize
target heatmap max > 0.9
no GT/CARLA privileged dependency
```

---

## 7. Ego BEV Rasterizer

권장 파일:

```text
rcs_ac_v1_1/runtime/ego_bev.py
```

출력:

```text
ego_bev: [3, H, W], dtype float32
```

channel:

```text
0 ego_mask
1 past_ego_trajectory
2 previous_control_trail
```

함수:

```python
def build_ego_bev(
    ego_state: V11EgoState,
    spec: BEVSpec,
    *,
    past_positions_ego: np.ndarray | None = None,  # [T,2]
    vehicle_length_m: float = 4.7,
    vehicle_width_m: float = 1.9,
) -> np.ndarray:
```

### 7.1 ego_mask

ego vehicle footprint를 origin 주변 rectangle로 rasterize한다.

```text
x range: [-vehicle_length_m/2, vehicle_length_m/2]
y range: [-vehicle_width_m/2, vehicle_width_m/2]
```

각 cell center를 ego xy로 변환해 rectangle 내부이면 1.0.

### 7.2 past_ego_trajectory

`past_positions_ego`를 polyline으로 그린다.

decay:

```text
가장 최근 position value = 1.0
오래된 position value = 0.2까지 선형 감소
```

past_positions가 없으면 ego origin만 작은 disk로 표시한다.

### 7.3 previous_control_trail

signed channel로 정의한다.

```text
value = clip(previous_steer, -1, 1)
```

ego footprint 또는 최근 trajectory 위에 찍는다.

이 channel은 steer 방향 history를 policy가 볼 수 있게 한다.

### 7.4 Tests

```text
test_ego_bev_shape
test_ego_mask_contains_origin
test_past_trajectory_draws_previous_points
test_previous_control_trail_signed_value
```

완료 기준:

```text
shape == (3,128,128)
ego origin cell value > 0
finite
previous_control_trail signed value preserved
```

---

## 8. LiDAR BEV Rasterizer

권장 파일:

```text
rcs_ac_v1_1/runtime/lidar_bev.py
```

출력:

```text
lidar_bev: [5, H, W], dtype float32
```

channel:

```text
0 lidar_occupancy
1 lidar_free_space
2 lidar_unknown
3 lidar_height
4 lidar_density
```

함수:

```python
def build_lidar_bev(
    lidar_points_ego: np.ndarray | None,
    spec: BEVSpec,
    *,
    min_z_m: float = -2.5,
    max_z_m: float = 3.0,
    max_density_per_cell: int = 8,
    ray_step_m: float | None = None,
) -> np.ndarray:
```

### 8.1 Point filtering

```text
finite x/y/z
within BEV extent
min_z_m <= z <= max_z_m
```

### 8.2 occupancy

각 valid point가 들어간 cell:

```text
occupancy[row, col] = 1.0
```

### 8.3 density

cell별 count:

```text
density = clip(log1p(count) / log1p(max_density_per_cell), 0, 1)
```

### 8.4 height

cell별 height:

```text
height_norm = clip((max_z_in_cell - min_z_m) / (max_z_m - min_z_m), 0, 1)
```

point가 없는 cell은 0.

### 8.5 free_space

각 point에 대해 ego origin cell부터 point cell까지 ray를 sample한다.

```text
step = ray_step_m or resolution_m
sample points from origin to point
mark free_space=1 until just before occupancy cell
```

occupancy cell은 free_space보다 우선한다.

```python
free_space[occupancy > 0] = 0
```

### 8.6 unknown

```python
observed = maximum(occupancy, free_space)
unknown = 1.0 - observed
```

LiDAR unavailable이면:

```text
occupancy = 0
free_space = 0
unknown = 1
height = 0
density = 0
```

### 8.7 Tests

```text
test_lidar_bev_shape
test_lidar_point_sets_occupancy_cell
test_lidar_ray_marks_free_space_before_point
test_lidar_empty_returns_unknown_all_one
test_lidar_height_density_finite_and_bounded
```

완료 기준:

```text
shape == (5,128,128)
finite
all channels in [0,1]
occupancy/free_space conflict 없음
```

---

## 9. RGB Perception BEV Channel Builder

이 항목은 가장 주의해야 한다. RGB에서 BEV channel을 만들려면 학습된 projection/prediction이 필요하다. GT BEV를 inference에 직접 넣으면 안 된다.

권장 파일:

```text
rcs_ac_v1_1/runtime/rgb_bev.py
rcs_ac_v1_1/models/rgb_bev_head.py
```

출력:

```text
rgb_bev: [4, H, W], dtype float32
```

channel:

```text
0 drivable_area_pred
1 lane_marking_pred
2 road_boundary_pred
3 rgb_bev_confidence
```

### 9.1 Phase 1: Safe placeholder builder

학습된 RGB BEV head가 없을 때는 fake road/lane을 만들지 않는다.

```python
def build_rgb_bev_unavailable(spec: BEVSpec) -> np.ndarray:
    rgb_bev = zeros([4,H,W])
    rgb_bev[3] = 0.0  # confidence
    return rgb_bev
```

의미:

```text
RGB BEV unavailable이면 confidence 0으로 두고, fusion/reliability가 RGB source를 낮추게 한다.
```

### 9.2 Phase 2: Learned RGB BEV head

권장 model:

```python
class RGBBEVHead(nn.Module):
    forward(rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "rgb_bev_logits": [B,4,H,W],
            "rgb_bev": sigmoid/logit processed output,
        }
```

학습 target:

```text
drivable_area_gt
lane_marking_gt
road_boundary_gt or obstacle/free proxy
confidence target = 1 for valid GT, lower under synthetic corruption
```

주의:

```text
초기 current policy의 RGBContextEncoder는 vector encoder이므로 BEV map decoder가 아니다.
RGB BEV를 학습하려면 별도 spatial decoder/head가 필요하다.
```

### 9.3 Phase 3: Optional IPM/Homography

카메라 calibration이 명확하면 image road/lane mask를 BEV로 inverse perspective mapping할 수 있다.

단, 이 경로는 calibration/ground plane assumption에 민감하다. 첫 구현에서는 learned head 또는 unavailable placeholder를 추천한다.

### 9.4 Tests

```text
test_rgb_bev_unavailable_shape_and_zero_confidence
test_rgb_bev_head_output_shape
test_rgb_bev_no_gt_required_for_forward
test_rgb_bev_corruption_lowers_confidence_target
```

완료 기준:

```text
inference에서 GT BEV를 사용하지 않음
RGB unavailable 시 confidence 0
학습된 head 사용 시 output shape == (4,128,128)
```

---

## 10. GT BEV Target Builder

권장 파일:

```text
rcs_ac_v1_1/runtime/gt_bev_targets.py
```

출력:

```text
gt_bev_target: [5, H, W], dtype float32
bev_target_available: float32 scalar or bool
```

channel:

```text
0 drivable_area_gt
1 lane_marking_gt
2 obstacle_occupancy_gt
3 free_space_gt
4 route_path_gt
```

### 10.1 함수 계약

CARLA 없는 unit test를 위해 map/actor 정보를 adapter 형태로 받는다.

```python
def build_gt_bev_target(
    *,
    spec: BEVSpec,
    map_adapter: V11MapAdapter | None,
    actors_ego: list[V11ActorBox] | None,
    route_context: V11RouteContext | None,
) -> tuple[np.ndarray, float]:
```

권장 adapter:

```python
class V11MapAdapter(Protocol):
    def is_drivable_ego_xy(self, x_m: float, y_m: float) -> bool: ...
    def is_lane_marking_ego_xy(self, x_m: float, y_m: float) -> bool: ...
```

권장 actor box:

```python
@dataclass(frozen=True)
class V11ActorBox:
    center_x_m: float
    center_y_m: float
    length_m: float
    width_m: float
    yaw_rad: float = 0.0
    actor_type: str = "vehicle"
```

### 10.2 drivable_area_gt

각 cell center를 ego xy로 변환해 map_adapter에 질의한다.

```text
drivable=1 if is_drivable_ego_xy(x,y)
```

map_adapter가 없으면:

```text
drivable_area_gt = 0
bev_target_available = 0
```

### 10.3 lane_marking_gt

첫 구현에서는 map_adapter query 기반으로 둔다.

```text
lane=1 if is_lane_marking_ego_xy(x,y)
```

CARLA map에서 정확한 lane marking 추출이 어렵다면 route centerline 주변 lane proxy를 debug-only로 사용할 수 있지만, 문서에 proxy임을 명시해야 한다.

### 10.4 obstacle_occupancy_gt

actor boxes를 ego frame rectangle로 rasterize한다.

```text
cell center를 actor local frame으로 회전
abs(local_x) <= length/2
abs(local_y) <= width/2 이면 occupancy=1
```

차량/보행자/정적 장애물을 모두 포함한다.

### 10.5 free_space_gt

초기 정의:

```python
free_space_gt = drivable_area_gt * (1 - obstacle_occupancy_gt)
```

주의:

```text
이 free_space는 센서 관측 free space가 아니라 GT drivable free region이다.
문서와 로그에 "gt_drivable_free_space"로 해석해야 한다.
```

### 10.6 route_path_gt

route rasterizer의 `route_path` channel과 같은 알고리즘을 사용한다.

```text
route_context가 invalid이면 route_path_gt = 0
```

### 10.7 Tests

```text
test_gt_bev_target_shape
test_gt_drivable_map_adapter_fills_expected_cells
test_gt_actor_box_sets_obstacle_occupancy
test_gt_free_space_excludes_obstacles
test_gt_route_path_matches_route_rasterizer
test_gt_unavailable_without_map_adapter
```

완료 기준:

```text
shape == (5,128,128)
finite
values in [0,1]
GT target never required by policy forward
availability mask 정확
```

---

## 11. Waypoint Target Builder

권장 파일:

```text
rcs_ac_v1_1/runtime/waypoint_targets.py
```

출력:

```text
expert_waypoints: [4,2], dtype float32
waypoint_target_available: float32 scalar
expert_target_speed: [1], dtype float32
```

함수:

```python
def build_distance_based_expert_waypoints(
    expert_path_ego: np.ndarray | None,
    *,
    distances_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0),
    fallback_route_ego: np.ndarray | None = None,
    min_path_length_m: float = 4.0,
    allow_extrapolation: bool = True,
    max_extrapolation_m: float = 4.0,
) -> tuple[np.ndarray, float]:
```

### 11.1 Path source priority

```text
1. expert_path_ego if valid
2. fallback_route_ego if valid
3. unavailable
```

### 11.2 Valid path

```text
shape [N,2]
finite
N >= 2
forward length >= min_path_length_m
not all x <= 0
```

뒤쪽 point는 제거한다.

```text
keep x >= -0.5
```

### 11.3 Arc-length interpolation

```text
origin [0,0]을 path 앞에 붙인다.
segment length 누적.
distances_m 각 위치에서 linear interpolation.
```

path가 8m보다 짧으면:

```text
allow_extrapolation=True이고 부족분 <= max_extrapolation_m이면 마지막 segment 방향으로 extrapolate
아니면 available=0
```

### 11.4 Safety validation

생성 후:

```text
all finite
x strictly increasing
all x > 0
abs(y) <= 8.0 권장
```

위반 시:

```text
available=0
waypoints=zeros([4,2])
```

### 11.5 Target speed

함수:

```python
def build_expert_target_speed(
    *,
    expert_speed_kmh: float | None,
    route_target_speed_kmh: float | None,
    default_speed_kmh: float = 12.0,
    min_speed_kmh: float = 0.0,
    max_speed_kmh: float = 28.0,
) -> np.ndarray:
```

priority:

```text
1. expert_speed_kmh if finite
2. route_target_speed_kmh if finite
3. default_speed_kmh
```

### 11.6 Tests

```text
test_waypoint_target_interpolates_fixed_distances
test_waypoint_target_falls_back_to_route
test_waypoint_target_unavailable_when_path_too_short
test_waypoint_target_extrapolation_limited
test_waypoint_target_x_increasing
test_target_speed_priority_and_clipping
```

---

## 12. Waypoint Availability Mask

권장 위치:

```text
rcs_ac_v1_1/runtime/waypoint_targets.py
```

함수:

```python
def validate_waypoint_target(
    waypoints: np.ndarray,
    *,
    min_gap_m: float = 0.25,
    max_abs_lateral_m: float = 8.0,
) -> float:
```

available 조건:

```text
shape == [4,2]
finite
all x > 0
all diff(x) >= min_gap_m
all abs(y) <= max_abs_lateral_m
```

출력:

```text
1.0 if valid else 0.0
```

loss에서 사용:

```python
loss_per_sample = smooth_l1(...).mean(dim=(1,2))
loss = (loss_per_sample * waypoint_target_available).sum() / waypoint_target_available.sum().clamp_min(1.0)
```

완료 기준:

```text
invalid waypoint가 loss에 조용히 섞이지 않는다.
train log에 waypoint_target_available_rate를 기록한다.
```

---

## 13. BEVExpertDataset Loader

권장 파일:

```text
rcs_ac_v1_1/data/bev_expert_dataset.py
```

지원 포맷:

```text
.npz first
zarr/hdf5 later if dataset gets large
```

### 13.1 NPZ keys

필수:

```text
rgb: [N,3,90,160]
lidar_bev: [N,5,128,128]
route_bev: [N,4,128,128]
ego_bev: [N,3,128,128]
ego_scalar: [N,6]
route_scalar: [N,10]
sensor_health: [N,15]
gt_bev_target: [N,5,128,128]
expert_waypoints: [N,4,2]
expert_target_speed: [N,1]
bev_target_available: [N]
waypoint_target_available: [N]
```

권장 metadata:

```text
episode_id: [N]
step: [N]
done: [N]
source: [N] or string metadata
config_version: string
bev_spec_json: string
channel_names_json: string
```

### 13.2 Dataset return

`__getitem__` 반환:

```python
{
    "obs_flat": torch.float32 [239839],
    "targets": {
        "gt_bev_target": torch.float32 [5,128,128],
        "expert_waypoints": torch.float32 [4,2],
        "expert_target_speed": torch.float32 [1],
        "bev_target_available": torch.float32 [],
        "waypoint_target_available": torch.float32 [],
    },
    "meta": {
        "episode_id": ...,
        "step": ...,
    },
}
```

`obs_flat`은 `flatten_observation_dict()`로 생성한다.

### 13.3 Validation

load 시 검증:

```text
all required keys exist
first dimension N matches
shapes match V11RuntimeConfig observation/target spec
finite check
mask values in [0,1]
no gt_bev_target inside obs_flat
```

### 13.4 Split

초기 구현:

```text
random_split by episode_id if episode_id exists
otherwise deterministic index split
```

sequence training은 나중에 추가한다. 첫 smoke는 frame dataset으로 고정한다.

### 13.5 Tests

```text
test_bev_expert_dataset_loads_minimal_npz
test_dataset_rejects_missing_key
test_dataset_rejects_shape_mismatch
test_dataset_returns_flat_obs_and_targets
test_dataset_does_not_include_gt_in_policy_input
```

---

## 14. BEV Loss

권장 파일:

```text
rcs_ac_v1_1/rl/bev_losses.py
```

중요한 현재 코드 상태:

```text
현재 RCSACV11BEVPolicy는 `bev_logits`를 출력한다.
따라서 `train_bev_bc.py`는 waypoint/speed loss와 함께 GT BEV auxiliary loss를 계산할 수 있다.
단, 이 BEV prediction head는 offline scaffold이며, 실제 CARLA GT adapter와 RGB learned BEV checkpoint 품질 검증은 별도 단계다.
```

### 14.1 권장 output contract

policy auxiliary head output:

```python
outputs["bev_logits"]  # [B,5,128,128]
```

channel order:

```text
0 drivable_area
1 lane_marking
2 obstacle_occupancy
3 free_space
4 route_path
```

### 14.2 Loss function

```python
def compute_bev_loss(
    bev_logits: torch.Tensor,
    gt_bev_target: torch.Tensor,
    available: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

base:

```text
BCEWithLogitsLoss(reduction="none")
mean over C,H,W per sample
mask by bev_target_available
weighted channel sum
```

default weights:

```text
drivable: 1.0
lane: 2.0
obstacle: 3.0
free_space: 1.0
route: 1.5
```

### 14.3 Metrics

```text
bev_drivable_loss
bev_lane_loss
bev_obstacle_loss
bev_free_loss
bev_route_loss
bev_total_loss
bev_target_available_rate
bev_drivable_iou
bev_lane_iou
bev_obstacle_iou
```

IoU:

```text
sigmoid(logits) > 0.5
target > 0.5
```

### 14.4 Tests

```text
test_bev_loss_no_nan_all_zero_target
test_bev_loss_masks_unavailable_samples
test_bev_loss_rejects_shape_mismatch
test_bev_iou_perfect_prediction_is_one
```

---

## 15. Waypoint / Target-Speed BC Loss

권장 파일:

```text
rcs_ac_v1_1/rl/bev_losses.py
```

함수:

```python
def compute_waypoint_speed_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    waypoint_weight: float = 1.0,
    target_speed_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

입력:

```text
outputs["waypoints"]: [B,4,2]
outputs["target_speed_kmh"]: [B,1]
targets["expert_waypoints"]: [B,4,2]
targets["expert_target_speed"]: [B,1]
targets["waypoint_target_available"]: [B]
```

loss:

```text
SmoothL1(pred_waypoints, expert_waypoints)
SmoothL1(pred_target_speed, expert_target_speed)
waypoint loss는 waypoint_target_available로 mask
speed loss는 finite target이면 사용하되, 별도 speed_available mask를 나중에 추가 가능
```

metrics:

```text
waypoint_l1
waypoint_x_l1
waypoint_y_l1
waypoint_heading_error
target_speed_mae
waypoint_target_available_rate
```

`waypoint_heading_error`:

```text
atan2(pred_y_last, pred_x_last) - atan2(target_y_last, target_x_last)
```

### Tests

```text
test_waypoint_speed_loss_no_nan
test_waypoint_loss_masks_unavailable_samples
test_target_speed_loss_basic
test_waypoint_heading_error_zero_for_same_direction
```

---

## 16. train_bev_bc.py

권장 파일:

```text
rcs_ac_v1_1/scripts/train_bev_bc.py
```

### 16.1 CLI

```bash
python -m rcs_ac_v1_1.scripts.train_bev_bc \
  --config configs/v1_1/full_bev_smoke.yaml \
  --dataset data/v1_1/bev_expert_smoke.npz \
  --output-dir checkpoints/v1_1_bev_bc_smoke \
  --epochs 5 \
  --batch-size 16 \
  --device cuda
```

### 16.2 Flow

```text
1. load_v11_runtime_config
2. BEVExpertDataset load
3. train/val split
4. model = config.build_policy()
5. optimizer AdamW
6. forward(obs_flat)
7. waypoint/speed loss
8. optional BEV loss if outputs has "bev_logits"
9. backward/clip grad
10. log train/val metrics
11. save last.pt and best.pt
```

### 16.3 Checkpoint

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": raw_config,
    "epoch": epoch,
    "metrics": metrics,
}
```

### 16.4 CSV log

```text
epoch
split
loss
waypoint_l1
waypoint_x_l1
waypoint_y_l1
target_speed_mae
waypoint_target_available_rate
bev_loss if available
```

### 16.5 Tests

```text
test_train_bev_bc_smoke_one_epoch_cpu
test_train_bev_bc_saves_checkpoint
test_train_bev_bc_does_not_require_gt_in_forward
```

---

## 17. evaluate_bev_pure_pursuit.py

권장 파일:

```text
rcs_ac_v1_1/scripts/evaluate_bev_pure_pursuit.py
```

### 17.1 CLI

```bash
python -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit \
  --config configs/v1_1/full_bev_smoke.yaml \
  --checkpoint checkpoints/v1_1_bev_bc_smoke/best.pt \
  --run-name v1_1_bev_pure_pursuit_eval_512 \
  --num-steps 512 \
  --host 127.0.0.1 \
  --port 2000 \
  --device cuda
```

### 17.2 Flow

```text
1. load config/checkpoint
2. connect CARLA env
3. reset env
4. each step:
   a. convert CARLA obs/info to V11SensorPacket/V11EgoState/V11RouteContext
   b. build sensor_health
   c. build ego_scalar/route_scalar
   d. build lidar_bev/route_bev/ego_bev
   e. build rgb_bev or unavailable placeholder
   f. flatten policy observation
   g. model forward
   h. controller.control(...)
   i. env.step(control)
   j. log raw outputs and controller debug
5. write logs
```

### 17.3 Required logs

```text
step
reward
done
done_reason
collision
offroad
stuck
route_progress
speed_kmh
steer
throttle
brake
target_speed_kmh_raw
target_speed_capped_kmh
waypoint_valid
path_valid
used_route_fallback
used_synthetic_fallback
lookahead_m
raw_steer
smoothed_steer
rate_limited_steer
bev_fusion_weight_rgb
bev_fusion_weight_lidar
bev_fusion_weight_route
bev_fusion_weight_ego
sensor_health_0..14
```

### 17.4 Safety criteria

```text
512 clean: collision 0, offroad 0, stuck 0
2048 clean: collision 0, offroad 0, stuck 0
used_synthetic_fallback_rate should be low in clean
target_speed_capped_kmh should not stay 0 for most clean steps
```

### 17.5 Tests

CARLA 없는 tests:

```text
test_eval_builds_flat_obs_from_synthetic_packet
test_eval_controller_debug_columns_exist
test_eval_checkpoint_loads_policy_state_dict
```

CARLA integration은 Windows에서 별도 smoke로 수행한다.

---

## 18. 구현 순서 재정렬

권장 순서:

```text
1. state.py dataclasses: done
2. sensor_health.py: done
3. scalars.py: done
4. bev_rasterizer.py common primitives: done
5. route_bev.py: done
6. ego_bev.py: done
7. lidar_bev.py: done
8. waypoint_targets.py: done
9. gt_bev_targets.py synthetic adapter tests: done
10. data/bev_expert_dataset.py: done
11. rl/bev_losses.py waypoint/speed + BEV loss: done
12. train_bev_bc.py waypoint/speed/BEV smoke: done
13. rgb_bev.py unavailable placeholder + RGBBEVHead scaffold: done
14. evaluate_bev_pure_pursuit.py synthetic builder smoke: done
15. BEV prediction head + BEV loss activation: done
16. sensor packet -> policy component pipeline: done
17. CARLA collection/eval adapters: scaffold implemented
18. RGB learned BEV checkpoint/training integration: remaining
19. robustness/corruption training: remaining
```

이 순서를 추천하는 이유:

```text
route/ego/lidar BEV와 waypoint/speed/BEV auxiliary loss는 RGB learned BEV checkpoint 없이도 offline smoke 가능하다.
BEV loss는 현재 모델의 `bev_logits`로 활성화되어 있다.
GT BEV는 dataset/loss를 위해 필요하지만, policy forward에는 절대 들어가지 않는다.
```

---

## 19. 문서/코드 불일치 점검 결과

점검 결과:

```text
docs/21_full_bev_extension_for_rcs_ac_v1.md는 v1 원 설계 문서라 PID/0.35m 예시가 남아 있다.
v1.1 현재 구현은 pure pursuit와 0.375m/128x128을 사용한다.
따라서 v1.1 구현 시 docs/v1_1 문서를 우선한다.
```

현재 코드 기준으로 바로 수정해야 하는 문서:

```text
docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
docs/v1_1/rcs_ac_v1_1_full_implementation_spec.md
docs/rcs_ac_project_handoff_for_new_codex_chat.md
```

수정 기준:

```text
offline/synthetic v1.1 pipeline 구현 상태를 반영한다.
테스트 기대값은 47 passed로 갱신한다.
남은 구현 항목은 CARLA adapter와 real closed-loop 연결로 구분한다.
```
