# RCS-AC v1.1 Current Code Implementation Specification

작성일: 2026-05-05

이 문서는 현재 `rcs_ac_v1_1` 폴더에 실제 구현된 코드만 기준으로 작성한 상세 구현 명세다. 목적은 문서만 보고도 현재 scaffold를 같은 동작으로 재구현할 수 있게 만드는 것이다.

범위:

```text
rcs_ac_v1_1/
  adapters/
    __init__.py
    carla_lidar_adapter.py
    carla_map_actor_adapter.py
    carla_observation_adapter.py
    carla_route_adapter.py
  runtime/
    bev_spec.py
    bev_rasterizer.py
    config.py
    ego_bev.py
    gt_bev_targets.py
    lidar_bev.py
    observation.py
    pid_controller.py
    rgb_bev.py
    route_bev.py
    scalars.py
    sensor_health.py
    state.py
    waypoint_targets.py
  models/
    bev_policy.py
    rgb_bev_head.py
  data/
    bev_expert_dataset.py
  rl/
    bev_losses.py
  scripts/
    collect_bev_expert_dataset.py
    train_bev_bc.py
    evaluate_bev_pure_pursuit.py
  tests/
    test_v1_1_carla_adapter_contract.py
    test_bev_spec.py
    test_pure_pursuit_controller.py
    test_v1_1_bev_inputs.py
    test_v1_1_config.py
    test_v1_1_runtime_builders.py
    test_v1_1_scripts.py
    test_v1_1_targets_dataset_losses.py
    test_waypoint_parameterization.py

configs/v1_1/
  full_bev_smoke.yaml
```

현재 구현된 추가 항목:

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
CARLA adapter contract
collect_bev_expert_dataset.py
```

각 항목의 상세 설계와 CARLA 통합 시 남은 경계는 다음 문서에 둔다.

```text
docs/v1_1/rcs_ac_v1_1_remaining_implementation_design.md
```

---

## 1. 전체 원칙

RCS-AC v1.1은 기존 `rcs_ac_first` 패키지를 수정하지 않고, 새 패키지 `rcs_ac_v1_1` 안에 별도로 둔다.

정책 입력 원칙:

```text
policy inference input에는 센서/route/ego에서 만들 수 있는 값만 넣는다.
GT BEV, CARLA actor state, simulator privileged signal은 policy forward에 넣지 않는다.
GT BEV는 학습 target, teacher, debug, offline audit 전용이다.
```

정책 출력 원칙:

```text
model은 직접 steer/throttle/brake를 출력하지 않는다.
model은 local waypoints와 target_speed_kmh를 출력한다.
controller가 waypoints와 target_speed_kmh를 steer/throttle/brake로 변환한다.
```

RCS-AC 구조 원칙:

```text
RGB, LiDAR BEV, route BEV, ego BEV를 source 단위 feature로 인코딩한다.
sensor_health를 이용해 source-level reliability/fusion weight를 만든다.
fusion은 channel-level이 아니라 source-level이다.
```

---

## 2. BEV 좌표계

파일:

```text
rcs_ac_v1_1/runtime/bev_spec.py
```

좌표계:

```text
x > 0: ego 기준 전방
x < 0: ego 기준 후방
y > 0: ego 기준 왼쪽
y < 0: ego 기준 오른쪽
BEV tensor layout: [C, H, W]
row 증가 방향: 전방에서 후방으로
col 증가 방향: 왼쪽에서 오른쪽으로
```

기본 BEVSpec:

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
    strict_extent_check: bool = True
    extent_tolerance_m: float = 1.0e-4
```

기본 grid extent:

```text
height * resolution_m = 128 * 0.375 = 48.0m
front_m + rear_m = 40.0 + 8.0 = 48.0m

width * resolution_m = 128 * 0.375 = 48.0m
left_m + right_m = 24.0 + 24.0 = 48.0m
```

`strict_extent_check=True`이면 다음 불변식이 깨질 때 `ValueError`를 발생시킨다.

```text
abs(height * resolution_m - (front_m + rear_m)) <= extent_tolerance_m
abs(width * resolution_m - (left_m + right_m)) <= extent_tolerance_m
```

에러 메시지에는 `"extent mismatch"`가 포함되어야 한다.

---

## 3. BEVSpec Properties

`BEVSpec`은 다음 property를 제공한다.

```python
x_extent_m = front_m + rear_m
y_extent_m = left_m + right_m
grid_x_extent_m = height * resolution_m
grid_y_extent_m = width * resolution_m
ego_row_float = front_m / resolution_m
ego_col_float = left_m / resolution_m
shape_hw = (height, width)
```

기본값에서:

```text
x_extent_m = 48.0
y_extent_m = 48.0
grid_x_extent_m = 48.0
grid_y_extent_m = 48.0
ego_row_float = 106.666...
ego_col_float = 64.0
shape_hw = (128, 128)
```

채널 shape 생성:

```python
channel_shape(channels: int) -> tuple[int, int, int]
```

예:

```python
BEVSpec().channel_shape(5) == (5, 128, 128)
```

`channels <= 0`이면 `ValueError`를 발생시킨다.

---

## 4. Ego XY -> BEV Row/Col 변환

함수:

```python
ego_xy_to_bev_row_col(
    x_m: float,
    y_m: float,
    spec: BEVSpec | None = None,
) -> tuple[int, int, bool]
```

계산:

```python
row_float = (front_m - x_m) / resolution_m
col_float = (left_m - y_m) / resolution_m
valid = (
    -extent_tolerance_m <= row_float <= height + extent_tolerance_m
    and -extent_tolerance_m <= col_float <= width + extent_tolerance_m
)
row = floor(row_float)
col = floor(col_float)
if valid:
    row = clamp(row, 0, height - 1)
    col = clamp(col, 0, width - 1)
```

이 방식은 front/left/rear/right 경계를 모두 grid 안으로 포함한다.

기본값 예:

```text
(x=40.0, y=0.0)  -> (row=0,   col=64,  valid=True)
(x=-8.0, y=0.0)  -> (row=127, col=64,  valid=True)
(x=0.0,  y=24.0) -> (row=106, col=0,   valid=True)
(x=0.0,  y=-24.0)-> (row=106, col=127, valid=True)
```

주의:

```text
row/col은 cell index다.
경계값은 clamp되어 마지막 cell에 들어간다.
```

---

## 5. BEV Row/Col -> Ego XY 변환

함수:

```python
bev_row_col_to_ego_xy(
    row: int,
    col: int,
    spec: BEVSpec | None = None,
) -> tuple[float, float]
```

계산:

```python
x_m = front_m - (row + 0.5) * resolution_m
y_m = left_m - (col + 0.5) * resolution_m
```

이 함수는 cell boundary가 아니라 cell center 좌표를 반환한다.

예:

```text
(row=0, col=64)    -> x ~= 39.8125, y ~= -0.1875
(row=127, col=64)  -> x ~= -7.8125, y ~= -0.1875
(row=106, col=0)   -> x ~= 0.0625, y ~= 23.8125
(row=106, col=127) -> x ~= 0.0625, y ~= -23.8125
```

---

## 6. BEV Channel Contract

파일:

```text
rcs_ac_v1_1/runtime/bev_spec.py
```

기본 채널 수:

```python
@dataclass(frozen=True)
class BEVChannelSpec:
    rgb_channels: int = 4
    lidar_channels: int = 5
    route_channels: int = 4
    ego_channels: int = 3
    target_channels: int = 5
```

sensor-derived BEV input channels:

```text
rgb_drivable_area_pred
rgb_lane_marking_pred
rgb_road_boundary_pred
rgb_bev_confidence
lidar_occupancy
lidar_free_space
lidar_unknown
lidar_height
lidar_density
route_path
target_waypoint_heatmap
turn_region
speed_hint
ego_mask
past_ego_trajectory
previous_control_trail
```

GT BEV target channels:

```text
drivable_area_gt
lane_marking_gt
obstacle_occupancy_gt
free_space_gt
route_path_gt
```

중요:

```text
GT BEV target channels는 policy input에 들어가지 않는다.
```

---

## 7. Policy Observation Contract

파일:

```text
rcs_ac_v1_1/runtime/observation.py
```

기본 상수:

```python
DEFAULT_RGB_SHAPE = (3, 90, 160)
DEFAULT_ROUTE_SCALAR_DIM = 10
DEFAULT_EGO_SCALAR_DIM = 6
DEFAULT_SENSOR_HEALTH_DIM = 15
DEFAULT_WAYPOINT_COUNT = 4
```

policy modality order:

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

flat observation은 반드시 위 순서로 concat된다.

---

## 8. Policy Input Shapes

기본 policy input:

| key | shape | 설명 |
| --- | --- | --- |
| `rgb` | `[3, 90, 160]` | front camera RGB |
| `lidar_bev` | `[5, 128, 128]` | LiDAR BEV |
| `route_bev` | `[4, 128, 128]` | route BEV |
| `ego_bev` | `[3, 128, 128]` | ego/history/control BEV |
| `ego_scalar` | `[6]` | ego scalar |
| `route_scalar` | `[10]` | route scalar |
| `sensor_health` | `[15]` | sensor reliability/quality |

flat dim:

```text
rgb:           3 * 90 * 160      = 43,200
lidar_bev:     5 * 128 * 128     = 81,920
route_bev:     4 * 128 * 128     = 65,536
ego_bev:       3 * 128 * 128     = 49,152
ego_scalar:                         6
route_scalar:                      10
sensor_health:                     15
total_dim:                    239,839
```

`build_v11_policy_observation_spec()`는 `ObservationSliceSpec` 목록과 `total_dim`을 만든다.

---

## 9. Observation Slice Spec

구조:

```python
@dataclass(frozen=True)
class ObservationSliceSpec:
    key: str
    shape: tuple[int, ...]
    start: int
    stop: int
```

property:

```python
size = stop - start
slice = slice(start, stop)
```

`V11PolicyObservationSpec`:

```python
@dataclass(frozen=True)
class V11PolicyObservationSpec:
    modalities: tuple[ObservationSliceSpec, ...]
    total_dim: int
    bev_spec: BEVSpec
    bev_channels: BEVChannelSpec
```

method:

```python
keys -> tuple[str, ...]
modal(key: str) -> ObservationSliceSpec
```

없는 key를 요청하면 `KeyError`를 발생시킨다.

---

## 10. Observation Validation and Flattening

함수:

```python
validate_observation_dict(
    observation: dict[str, Any],
    spec: V11PolicyObservationSpec,
) -> dict[str, np.ndarray]
```

검증:

```text
모든 policy key가 있어야 한다.
각 value shape가 spec shape와 정확히 같아야 한다.
각 value는 finite float32로 변환 가능해야 한다.
empty array는 허용하지 않는다.
```

함수:

```python
flatten_observation_dict(
    observation: dict[str, Any],
    spec: V11PolicyObservationSpec,
) -> np.ndarray
```

동작:

```text
1. validate_observation_dict() 호출
2. POLICY_MODALITY_ORDER에 해당하는 spec.modalities 순서대로 flatten
3. np.concatenate
4. float32 반환
```

함수:

```python
split_flat_observation(
    flat_observation: Any,
    spec: V11PolicyObservationSpec,
) -> dict[str, torch.Tensor]
```

동작:

```text
1. torch.float32 tensor로 변환
2. 1D면 batch dimension 추가
3. [B, total_dim]인지 확인
4. NaN/inf 확인
5. 각 slice를 원래 modality shape로 reshape
```

오류 조건:

```text
flat_observation.ndim이 1 또는 2가 아니면 ValueError
last dim이 total_dim과 다르면 ValueError
NaN/inf가 있으면 ValueError
```

---

## 11. Target Spec

함수:

```python
build_v11_target_spec(
    bev_spec: BEVSpec = DEFAULT_BEV_SPEC,
    bev_channels: BEVChannelSpec = DEFAULT_BEV_CHANNELS,
    waypoint_count: int = DEFAULT_WAYPOINT_COUNT,
) -> V11TargetSpec
```

반환:

```python
V11TargetSpec(
    gt_bev_shape=(5, 128, 128),
    expert_waypoint_shape=(4, 2),
    expert_target_speed_shape=(1,),
)
```

주의:

```text
V11TargetSpec은 학습 target 계약이다.
policy forward input 계약이 아니다.
```

---

## 12. Sensor Health Contract

`sensor_health`는 `[15]` 고정 길이다.

layout:

```text
0  camera_front_available
1  camera_front_quality
2  camera_front_latency_norm
3  lidar_available
4  lidar_quality
5  lidar_latency_norm
6  route_available
7  route_quality
8  route_latency_norm
9  ego_speed_available
10 ego_imu_available
11 ego_quality
12 weather_severity_norm
13 corruption_severity_norm
14 reserved_or_bias
```

권장 값 범위:

```text
available: 0.0 또는 1.0
quality: 0.0 ~ 1.0
latency_norm: 0.0 ~ 1.0
severity_norm: 0.0 ~ 1.0
reserved_or_bias: 기본 1.0 권장
```

현재 코드는 값 범위 clamp를 강제하지 않고 finite 여부만 검사한다. 실제 data builder에서 range normalization을 책임져야 한다.

---

## 13. Policy Config

파일:

```text
rcs_ac_v1_1/models/bev_policy.py
```

config:

```python
@dataclass(frozen=True)
class V11BEVPolicyConfig:
    rgb_shape: tuple[int, int, int] = (3, 90, 160)
    bev_spec: BEVSpec = DEFAULT_BEV_SPEC
    bev_channels: BEVChannelSpec = DEFAULT_BEV_CHANNELS
    ego_scalar_dim: int = 6
    route_scalar_dim: int = 10
    sensor_health_dim: int = 15
    feature_dim: int = 128
    fusion_dim: int = 128
    waypoint_count: int = 4
    waypoint_mode: str = "distance"
    waypoint_distances_m: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    waypoint_gap_residual_scale_m: float = 0.75
    waypoint_min_gap_m: float = 0.75
    waypoint_max_lateral_m: float = 6.0
    waypoint_parameterization: str = "monotonic_bounded"
    min_target_speed_kmh: float = 0.0
    max_target_speed_kmh: float = 28.0
```

validation:

```text
ego_scalar_dim > 0
route_scalar_dim > 0
sensor_health_dim > 0
feature_dim > 0
fusion_dim > 0
waypoint_count > 0
waypoint_mode == "distance"
waypoint_parameterization == "monotonic_bounded"
len(waypoint_distances_m) == waypoint_count
waypoint_distances_m는 strictly increasing and positive
waypoint_gap_residual_scale_m > 0
waypoint_min_gap_m > 0
waypoint_max_lateral_m > 0
max_target_speed_kmh > min_target_speed_kmh
```

---

## 14. Waypoint Parameterization

함수:

```python
parameterize_waypoints(
    raw_waypoint: torch.Tensor,
    *,
    distances_m: tuple[float, ...],
    gap_residual_scale_m: float,
    min_gap_m: float,
    max_lateral_m: float,
) -> torch.Tensor
```

입력:

```text
raw_waypoint: [B, 2*K]
first K values: x gap residual logits
last K values: lateral y logits
```

기본 K:

```text
K = 4
raw_waypoint shape = [B, 8]
```

계산:

```python
distances = tensor(distances_m)
base_gaps[0] = distances[0]
base_gaps[1:] = distances[1:] - distances[:-1]

raw_gap = raw_waypoint[:, :K]
raw_y = raw_waypoint[:, K:]

gap_residual = gap_residual_scale_m * tanh(raw_gap)
gaps = clamp(base_gaps + gap_residual, min=min_gap_m)
x = cumsum(gaps)
y = max_lateral_m * tanh(raw_y)
waypoints = stack((x, y), dim=-1)
```

보장:

```text
waypoints shape = [B, K, 2]
모든 x > 0
x는 waypoint index가 증가할수록 증가
각 gap >= min_gap_m
모든 y는 [-max_lateral_m, max_lateral_m]
```

기본값에서 raw가 모두 0이면:

```text
x ~= [2.0, 4.0, 6.0, 8.0]
y ~= [0.0, 0.0, 0.0, 0.0]
```

---

## 15. Encoders

파일:

```text
rcs_ac_v1_1/models/bev_policy.py
```

### 15.1 BEVSourceEncoder

입력:

```text
bev: [B, C, 128, 128]
```

network:

```python
Conv2d(C, 32, kernel_size=5, stride=2, padding=2)
ReLU
Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
ReLU
Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
ReLU
AdaptiveAvgPool2d((1, 1))
Flatten
Linear(96, feature_dim)
ReLU
```

출력:

```text
[B, feature_dim]
```

사용 source:

```text
lidar_bev
route_bev
ego_bev
```

### 15.2 RGBContextEncoder

입력:

```text
rgb: [B, 3, 90, 160]
```

network:

```python
Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
ReLU
Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
ReLU
AdaptiveAvgPool2d((1, 1))
Flatten
Linear(64, feature_dim)
ReLU
```

출력:

```text
[B, feature_dim]
```

### 15.3 VectorEncoder

입력:

```text
ego_scalar: [B, 6]
route_scalar: [B, 10]
```

network:

```python
Linear(input_dim, 64)
ReLU
Linear(64, feature_dim)
ReLU
```

출력:

```text
[B, feature_dim]
```

---

## 16. Source-Level BEV Fusion

class:

```python
class SourceLevelBEVFusion(nn.Module)
```

source order:

```python
("rgb", "lidar_bev", "route_bev", "ego_bev")
```

입력:

```python
features: dict[str, torch.Tensor]
sensor_health: torch.Tensor
```

각 feature shape:

```text
features["rgb"]       = [B, feature_dim]
features["lidar_bev"] = [B, feature_dim]
features["route_bev"] = [B, feature_dim]
features["ego_bev"]   = [B, feature_dim]
sensor_health         = [B, 15]
```

source projection:

```python
project[source] = Linear(feature_dim, fusion_dim)
```

reliability head:

```python
Linear(sensor_health_dim, 64)
ReLU
Linear(64, 4)
softmax
```

fusion:

```python
projected = stack(projected_source_features, dim=1)  # [B, 4, fusion_dim]
weights = softmax(reliability_logits, dim=-1)        # [B, 4]
fused = sum(projected * weights[..., None], dim=1)   # [B, fusion_dim]
```

출력:

```text
fused: [B, fusion_dim]
weights: [B, 4]
```

테스트에서 `weights.sum(dim=-1) == 1`을 확인한다.

---

## 17. RCSACV11BEVPolicy Forward

class:

```python
class RCSACV11BEVPolicy(nn.Module)
```

초기화:

```text
1. V11BEVPolicyConfig 생성
2. build_v11_policy_observation_spec()로 obs_spec 생성
3. RGB/LiDAR/route/ego/scalar encoders 생성
4. SourceLevelBEVFusion 생성
5. policy_body 생성
6. waypoint_head, target_speed_head, value_head 생성
```

policy body input:

```text
fused_bev: [B, fusion_dim]
ego_feature: [B, feature_dim]
route_feature: [B, feature_dim]
body_input = concat([fused_bev, ego_feature, route_feature])
body_input_dim = fusion_dim + 2 * feature_dim
```

기본값:

```text
fusion_dim = 128
feature_dim = 128
body_input_dim = 384
```

policy body:

```python
Linear(384, 256)
ReLU
Linear(256, 256)
ReLU
```

heads:

```python
waypoint_head = Linear(256, waypoint_count * 2)
target_speed_head = Linear(256, 1)
value_head = Linear(256, 1)
```

forward 입력:

```python
flat_observation: torch.Tensor  # [B, 239839] or [239839]
```

forward 출력:

```python
{
    "waypoints": torch.Tensor,          # [B, 4, 2]
    "target_speed_kmh": torch.Tensor,   # [B, 1]
    "value": torch.Tensor,              # [B, 1]
    "bev_fusion_weights": torch.Tensor, # [B, 4]
}
```

target speed 계산:

```python
speed_unit = sigmoid(target_speed_head(hidden))
target_speed = speed_unit * (max_target_speed_kmh - min_target_speed_kmh) + min_target_speed_kmh
```

기본 범위:

```text
0.0 <= target_speed_kmh <= 28.0
```

---

## 18. Anti-Leakage Contract

`RCSACV11BEVPolicy.forward()`는 다음 인자를 받지 않아야 한다.

```text
gt_bev_target
actor_positions
gt_bboxes
carla_actor_state
```

테스트:

```python
inspect.signature(RCSACV11BEVPolicy.forward).parameters
```

위 forbidden parameter가 없음을 확인한다.

---

## 19. Legacy Waypoint PID Controller

파일:

```text
rcs_ac_v1_1/runtime/pid_controller.py
```

legacy class:

```python
PIDControllerConfig
PIDState
V11WaypointPIDController
```

이 controller는 smoke/debug 용도다. v1.1 closed-loop 기본 controller는 pure pursuit다.

legacy lateral:

```python
selected waypoint = waypoints[lateral_waypoint_index]
heading_error = atan2(y, max(x, 1e-6))
steer = PID(heading_error)
steer clipped to [-1, 1]
```

legacy longitudinal:

```python
speed_error = target_speed_kmh - current_speed_kmh
longitudinal = PID(speed_error)
throttle = clip(longitudinal, 0, max_throttle)
brake = clip(-longitudinal, 0, max_brake)
```

---

## 20. Pure Pursuit Controller Config

class:

```python
@dataclass(frozen=True)
class PurePursuitControllerConfig:
    wheel_base_m: float = 2.85
    min_lookahead_m: float = 3.5
    max_lookahead_m: float = 12.0
    lookahead_base_m: float = 3.0
    lookahead_gain_per_mps: float = 0.45
    max_steer: float = 1.0
    steer_gain: float = 1.0
    steer_lowpass_tau_s: float = 0.25
    max_steer_rate_per_s: float = 1.2
    max_abs_target_y_m: float = 8.0
    min_valid_target_x_m: float = 0.5
    min_valid_path_length_m: float = 2.0
    speed_kp: float = 0.08
    speed_ki: float = 0.0
    speed_kd: float = 0.02
    max_throttle: float = 0.5
    max_brake: float = 0.4
    uncertainty_slowdown_threshold: float = 0.45
    uncertainty_stop_threshold: float = 0.85
    max_speed_under_high_uncertainty_kmh: float = 8.0
    min_uncertain_speed_kmh: float = 4.0
    invalid_path_target_speed_kmh: float = 0.0
    curvature_speed_cap_enabled: bool = True
    curvature_speed_gain: float = 2.2
    min_curve_speed_kmh: float = 6.0
    route_fallback_lookahead_m: float = 6.0
    route_fallback_y_m: float = 0.0
```

핵심 안전 설정:

```text
invalid_path_target_speed_kmh = 0.0
```

의미:

```text
predicted waypoint가 invalid이고 route fallback도 없으면 계속 주행하지 않고 감속/정지한다.
```

---

## 21. Waypoint Sanity Check

함수:

```python
sanitize_waypoints(
    waypoints: list[tuple[float, float]] | np.ndarray,
    config: PurePursuitControllerConfig,
) -> tuple[np.ndarray, bool]
```

입력 검증:

```text
waypoints는 [N, 2] 형태여야 한다.
finite여야 한다.
x >= min_valid_target_x_m이어야 한다.
abs(y) <= max_abs_target_y_m이어야 한다.
```

유효하지 않으면 synthetic fallback point를 반환한다.

```python
[[route_fallback_lookahead_m, route_fallback_y_m]]
```

기본값:

```text
[[6.0, 0.0]]
```

반환 flag:

```text
True: original waypoint에서 유효한 forward path를 만들 수 있음
False: synthetic fallback을 반환했거나 path가 invalid
```

주의:

```text
synthetic fallback point는 steer 계산을 안전하게 하기 위한 임시 geometry다.
이 경우 target_speed는 invalid_path_target_speed_kmh로 낮춰야 한다.
```

---

## 22. Pure Pursuit Lateral Control

class:

```python
V11SpeedAdaptivePurePursuitController
```

state:

```python
self.longitudinal = PIDState()
self.previous_steer = 0.0
```

reset:

```python
self.longitudinal = PIDState()
self.previous_steer = 0.0
```

control signature:

```python
control(
    waypoints: list[tuple[float, float]] | np.ndarray,
    *,
    target_speed_kmh: float,
    current_speed_kmh: float,
    dt: float,
    route_fallback_waypoints: list[tuple[float, float]] | np.ndarray | None = None,
    uncertainty: float = 0.0,
) -> dict[str, object]
```

flow:

```text
1. predicted waypoints sanitize
2. invalid이면 route_fallback_waypoints sanitize 시도
3. route fallback도 없거나 invalid이면 synthetic fallback 사용
4. path length 검사
5. speed-dependent lookahead 계산
6. arc-length 기준 target point 보간
7. pure pursuit steer 계산
8. low-pass smoothing
9. rate limit
10. target speed cap
11. longitudinal PID
12. control dict 반환
```

lookahead:

```python
speed_mps = max(0.0, current_speed_kmh / 3.6)
lookahead = lookahead_base_m + lookahead_gain_per_mps * speed_mps
lookahead = clip(lookahead, min_lookahead_m, max_lookahead_m)
```

기본 예:

```text
current_speed_kmh = 8.0
speed_mps ~= 2.22
lookahead ~= 3.0 + 0.45 * 2.22 = 4.0m
```

arc-length interpolation:

```text
ego origin [0,0]을 path 시작점으로 붙인다.
waypoints까지 segment length를 누적한다.
lookahead_m 위치의 point를 선형 보간한다.
```

pure pursuit:

```python
ld2 = target_x_m ** 2 + target_y_m ** 2
curvature = 2.0 * target_y_m / ld2
steer_rad = atan(wheel_base_m * curvature)
steer = steer_gain * steer_rad
steer = clip(steer, -max_steer, max_steer)
```

steer smoothing:

```python
alpha = dt / (steer_lowpass_tau_s + dt)
smoothed = previous_steer + alpha * (raw_steer - previous_steer)
```

rate limit:

```python
max_delta = max_steer_rate_per_s * dt
steer = previous_steer + clip(smoothed - previous_steer, -max_delta, max_delta)
```

---

## 23. Pure Pursuit Speed Control

uncertainty speed cap:

```python
if uncertainty < uncertainty_slowdown_threshold:
    target remains unchanged
elif uncertainty >= uncertainty_stop_threshold:
    target = min(target, min_uncertain_speed_kmh)
else:
    linearly interpolate cap from max_speed_under_high_uncertainty_kmh
    to min_uncertain_speed_kmh
```

invalid path speed cap:

```python
if used_synthetic_fallback:
    capped_speed = min(target_speed_kmh, invalid_path_target_speed_kmh)
else:
    capped_speed = apply_uncertainty_speed_cap(...)
```

기본값:

```text
invalid_path_target_speed_kmh = 0.0
```

curvature speed cap:

```python
cap = max_speed_under_high_uncertainty_kmh + curvature_speed_gain / (abs(curvature) + 1e-3)
cap = max(min_curve_speed_kmh, cap)
capped_speed = min(capped_speed, cap)
```

longitudinal PID:

```python
speed_error = capped_target_speed_kmh - current_speed_kmh
integral += speed_error * dt
derivative = (speed_error - previous_error) / dt
longitudinal = kp * error + ki * integral + kd * derivative
throttle = clip(longitudinal, 0, max_throttle)
brake = clip(-longitudinal, 0, max_brake)
```

invalid waypoint 예:

```python
control(
    [(-2.0, 0.0), (0.1, 20.0)],
    target_speed_kmh=12.0,
    current_speed_kmh=8.0,
    dt=0.05,
)
```

기대:

```text
waypoint_valid = False
used_synthetic_fallback = True
target_speed_capped_kmh = 0.0
throttle = 0.0
brake > 0.0
```

---

## 24. Pure Pursuit Output Contract

반환:

```python
{
    "steer": float,
    "throttle": float,
    "brake": float,
    "debug": {
        "waypoint_valid": bool,
        "path_valid": bool,
        "used_route_fallback": bool,
        "used_synthetic_fallback": bool,
        "lookahead_m": float,
        "target_x_m": float,
        "target_y_m": float,
        "raw_steer": float,
        "smoothed_steer": float,
        "rate_limited_steer": float,
        "curvature": float,
        "uncertainty": float,
        "target_speed_capped_kmh": float,
        "speed_cap_kmh": float,
        "speed_error_kmh": float,
    },
}
```

해석:

```text
waypoint_valid:
  predicted waypoint 또는 route fallback waypoint가 valid한지 여부

path_valid:
  최종 clean_waypoints의 path length가 min_valid_path_length_m 이상인지 여부

used_route_fallback:
  predicted waypoint가 invalid라서 route_fallback_waypoints를 사용했는지 여부

used_synthetic_fallback:
  valid path가 없어 synthetic straight geometry를 사용했는지 여부

target_speed_capped_kmh:
  invalid path / uncertainty / curvature cap을 모두 반영한 최종 target speed
```

---

## 25. Config Contract

파일:

```text
configs/v1_1/full_bev_smoke.yaml
```

중요 설정:

```yaml
bev_spec:
  height: 128
  width: 128
  resolution_m: 0.375
  front_m: 40.0
  rear_m: 8.0
  left_m: 24.0
  right_m: 24.0
  strict_extent_check: true
```

model shape 설정도 동일해야 한다.

```yaml
model:
  bev_height: 128
  bev_width: 128
  bev_resolution_m: 0.375
  bev_front_m: 40.0
  bev_rear_m: 8.0
  bev_left_m: 24.0
  bev_right_m: 24.0
```

controller:

```yaml
controller:
  type: speed_adaptive_pure_pursuit
  wheel_base_m: 2.85
  min_lookahead_m: 3.5
  max_lookahead_m: 12.0
  lookahead_base_m: 3.0
  lookahead_gain_per_mps: 0.45
  steer_lowpass_tau_s: 0.25
  max_steer_rate_per_s: 1.2
  speed_kp: 0.08
  speed_ki: 0.0
  speed_kd: 0.02
  max_throttle: 0.5
  max_brake: 0.4
  uncertainty_slowdown_threshold: 0.45
  uncertainty_stop_threshold: 0.85
  max_speed_under_high_uncertainty_kmh: 8.0
  min_uncertain_speed_kmh: 4.0
  invalid_path_target_speed_kmh: 0.0
```

현재 구현:

```text
rcs_ac_v1_1/runtime/config.py
```

`load_v11_runtime_config(path)`는 YAML을 읽어 다음 객체를 만든다.

```text
BEVSpec
BEVChannelSpec
V11BEVPolicyConfig
PurePursuitControllerConfig
V11PolicyObservationSpec
V11TargetSpec
```

`model.bev_*`와 `bev_spec.*`가 다르면 즉시 `ValueError`를 발생시킨다.
`policy_inputs.*`와 `target_only.*`의 선언 shape도 실제 observation/target spec과 맞는지 검증한다.

반환 객체:

```python
V11RuntimeConfig(
    version: str,
    description: str,
    bev_spec: BEVSpec,
    bev_channels: BEVChannelSpec,
    policy_config: V11BEVPolicyConfig,
    controller_config: PurePursuitControllerConfig,
    observation_spec: V11PolicyObservationSpec,
    target_spec: V11TargetSpec,
    raw: dict[str, Any],
)
```

편의 method:

```python
build_policy() -> RCSACV11BEVPolicy
build_controller() -> V11SpeedAdaptivePurePursuitController
```

---

## 26. Unit Tests

현재 테스트:

```text
rcs_ac_v1_1/tests/test_bev_spec.py
rcs_ac_v1_1/tests/test_pure_pursuit_controller.py
rcs_ac_v1_1/tests/test_v1_1_bev_inputs.py
rcs_ac_v1_1/tests/test_waypoint_parameterization.py
rcs_ac_v1_1/tests/test_v1_1_config.py
rcs_ac_v1_1/tests/test_v1_1_runtime_builders.py
rcs_ac_v1_1/tests/test_v1_1_targets_dataset_losses.py
rcs_ac_v1_1/tests/test_v1_1_scripts.py
```

실행:

```bash
python -m pytest rcs_ac_v1_1/tests -q
```

현재 기대 결과:

```text
47 passed
```

---

## 27. BEVSpec Tests

검증 항목:

```text
1. default BEVSpec validate 통과
2. grid extent와 configured extent 일치
3. ego origin이 grid 안에 있음
4. forward point는 ego보다 row가 작음
5. left point는 ego보다 col이 작음
6. row/col -> ego xy round trip이 cell center 오차 안에 있음
7. front/rear/left/right boundary가 모두 grid 안으로 mapping됨
8. invalid extent는 ValueError 발생
```

boundary 기대:

```text
(40.0, 0.0)   -> (0,   64,  True)
(-8.0, 0.0)   -> (127, 64,  True)
(0.0, 24.0)   -> (106, 0,   True)
(0.0, -24.0)  -> (106, 127, True)
```

---

## 28. Observation/Policy Tests

검증 항목:

```text
1. BEVSpec shape가 (128,128)
2. lidar_bev shape가 (5,128,128)
3. route_bev shape가 (4,128,128)
4. ego_bev shape가 (3,128,128)
5. total_dim == 239839
6. flatten 후 split하면 shape가 보존됨
7. GT BEV target은 policy input keys에 없음
8. policy forward output shape 확인
9. fusion weights sum == 1
10. forward signature에 GT/CARLA privileged input 없음
11. target channel count가 GT_BEV_TARGET_CHANNELS 길이와 일치
```

---

## 29. Waypoint Parameterization Tests

검증 항목:

```text
1. parameterized waypoint x는 모두 양수
2. x diff는 min_gap_m 이상
3. y는 max_lateral_m 범위 안
4. policy forward도 constrained waypoints를 반환
```

기본 기대:

```text
waypoints.shape == [B, 4, 2]
all(x > 0)
all(diff(x) >= 0.75)
max(abs(y)) <= 6.0
```

---

## 30. Pure Pursuit Tests

검증 항목:

```text
1. straight waypoints는 steer ~= 0
2. left waypoints는 positive steer
3. rate limit이 max_steer_rate_per_s * dt를 넘지 않음
4. invalid predicted waypoint가 route fallback을 사용함
5. invalid predicted waypoint가 route fallback 없이 들어오면 target_speed가 0으로 capped되고 brake가 걸림
```

invalid without route fallback 기대:

```text
waypoint_valid = False
used_synthetic_fallback = True
target_speed_capped_kmh = 0.0
throttle = 0.0
brake > 0.0
```

---

## 31. Known Current Limitations

현재 scaffold의 한계:

```text
1. CARLA observation/info -> V11SensorPacket/V11EgoState/V11RouteContext adapter contract는 구현됐지만 Windows CARLA 실기 검증은 아직 없다.
2. RGB BEV는 unavailable placeholder와 learned head scaffold만 있고, 학습된 perception checkpoint는 없다.
3. GT BEV target은 synthetic map_adapter/actor box 계약과 CARLA map/actor adapter scaffold가 있으나 Windows CARLA 실기 검증은 아직 없다.
4. evaluate_bev_pure_pursuit.py는 --mode carla loop가 구현됐지만 Windows CARLA 64/128 smoke는 아직 실행 전이다.
5. corruption/weather robustness 실험 config와 collection pipeline은 아직 없다.
```

즉, 현재 상태는 `offline/synthetic v1.1 pipeline scaffold + CARLA smoke adapter scaffold + unit tests` 단계다.

---

## 32. 구현 순서 권장

다음 구현 순서:

```text
1. Windows git pull 후 adapter contract가 shared_core env obs/info와 맞는지 확인
2. tiny CARLA dataset collection script로 64~128 samples 수집
3. train_bev_bc.py로 real tiny dataset 1 epoch smoke
4. evaluate_bev_pure_pursuit.py --mode carla 64-step smoke
5. evaluate_bev_pure_pursuit.py --mode carla 128-step smoke
6. CARLA 256-step smoke
7. CARLA 512/2048 clean eval
8. RGB BEV learned head 학습/연결
9. corruption/dropout/weather robustness 실험
```

64/128 smoke까지의 자세한 실행 계획은 다음 문서를 따른다.

```text
docs/v1_1/rcs_ac_v1_1_carla_64_128_smoke_plan.md
```

---

## 33. 하드코딩 주의사항

허용되는 고정값:

```text
BEV default spec
channel order
sensor_health layout
waypoint anchor distances
controller safety caps
```

허용되지 않는 것:

```text
lane_offset을 이용한 runtime steering override
route/lane 기반 직접 steer rule을 policy 성능으로 포장
CARLA privileged state를 policy input으로 사용
GT BEV를 inference input에 포함
특정 route/seed에 맞춘 steering constant
invalid waypoint 상황에서 조용히 full target speed 유지
```

runtime guard나 controller safety cap을 사용할 때:

```text
raw policy output과 controller/guard output을 분리 기록해야 한다.
used_route_fallback, used_synthetic_fallback, target_speed_capped_kmh를 로그에 남겨야 한다.
```

---

## 34. Verification Commands

문법 확인:

```bash
python -m py_compile \
  rcs_ac_v1_1/runtime/bev_spec.py \
  rcs_ac_v1_1/runtime/observation.py \
  rcs_ac_v1_1/runtime/pid_controller.py \
  rcs_ac_v1_1/runtime/config.py \
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
  rcs_ac_v1_1/models/bev_policy.py
```

테스트:

```bash
python -m pytest rcs_ac_v1_1/tests -q
```

config smoke:

```bash
python - <<'PY'
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config

config = load_v11_runtime_config("configs/v1_1/full_bev_smoke.yaml")
model = config.build_policy()
controller = config.build_controller()
print(config.observation_spec.total_dim)
print(config.bev_spec.shape_hw)
print(config.target_spec.gt_bev_shape)
PY
```

기대 출력:

```text
239839
(128, 128)
(5, 128, 128)
```

---

## 35. Current Pass Criteria

현재 scaffold가 정상이라고 판단하는 기준:

```text
1. BEVSpec default validate 통과
2. BEV boundary mapping symmetric pass
3. observation total_dim == 239839
4. policy forward output shape pass
5. fusion weight normalization pass
6. waypoint x monotonic/bounded y pass
7. pure pursuit straight/left/rate-limit tests pass
8. invalid waypoint without route fallback triggers stop cap
9. config parser rejects BEV/model mismatch
10. config parser rejects declared input/target shape mismatch
11. sensor/scalar/rasterizer/target builders pass synthetic tests
12. BEVExpertDataset loads tiny NPZ
13. BEV and waypoint/speed losses are finite
14. train_bev_bc.py CPU smoke saves checkpoint
15. evaluate_bev_pure_pursuit.py synthetic step returns controller debug
16. sensor packet -> policy component/flat observation pipeline pass
17. CARLA adapter contract tests pass
18. evaluate_bev_pure_pursuit.py synthetic CLI smoke pass
19. pytest 47 passed
```

현재 상태:

```text
python -m pytest rcs_ac_v1_1/tests -q
47 passed
```
