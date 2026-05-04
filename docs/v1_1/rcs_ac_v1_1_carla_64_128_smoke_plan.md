# RCS-AC v1.1 CARLA 64/128-Step Smoke Plan

작성일: 2026-05-05

이 문서는 RCS-AC v1.1을 현재 offline/synthetic scaffold 상태에서 Windows CARLA 64-step, 128-step smoke run까지 가져가기 위한 상세 진행 문서다.

목표는 512/2048 clean eval이 아니다. 먼저 CARLA adapter, real observation build, controller execution, logging, crash separation이 정상 동작하는지 짧은 run으로 확인한다.

---

## 1. 현재 출발점

현재 구현된 것:

```text
rcs_ac_v1_1 runtime BEV spec/observation contract
sensor/scalar builders
route/ego/LiDAR/RGB placeholder BEV builders
GT BEV synthetic target builder
waypoint target builder
BEVExpertDataset
BEV + waypoint/target-speed BC loss
train_bev_bc.py offline smoke
evaluate_bev_pure_pursuit.py synthetic smoke
sensor packet -> policy component/flat observation pipeline
CARLA adapter scaffold
collect_bev_expert_dataset.py smoke script
evaluate_bev_pure_pursuit.py --mode carla loop
RCSACV11BEVPolicy + bev_logits auxiliary head
pure pursuit controller
unit tests: 47 passed
```

현재 구현됐지만 Windows CARLA에서 아직 실기 검증이 필요한 것:

```text
1. CARLA observation/info adapter가 shared_core 실제 obs/info key와 맞는지 확인
2. route planner/local planner key가 route_points_ego로 정상 변환되는지 확인
3. LiDAR raw data 좌표계가 ego-frame lidar_points_ego와 맞는지 확인
4. CARLA map/actor adapter가 실제 world/map/actor 객체에서 작동하는지 확인
5. tiny CARLA BEV dataset collection script Windows 실행
6. evaluate_bev_pure_pursuit.py --mode carla Windows 실행
7. 64/128 smoke logging and summary 확인
```

현재 adapter compatibility guard:

```text
obs["route"] 10-d vector만 있는 경우 target proxy로 route_context를 만든다.
heading_error / route_heading_error는 degree로 해석하고, *_rad key만 radian으로 해석한다.
obs["lidar"]가 legacy [C,H,W] BEV이면 raw point cloud로 오인하지 않고 legacy_bev_detected debug를 남긴다.
previous control은 info["control"]이 없으면 top-level steer/throttle/brake/smoothed/applied 값을 fallback으로 쓴다.
CARLA eval action format은 shared_core STAGE4_ACTION_V2 [steer, longitudinal]만 허용한다.
```

이번 단계에서 아직 하지 않는 것:

```text
temporal smoothness loss
corruption/dropout training
RGB learned BEV checkpoint 학습
PPO/SAC baseline
512/2048 long eval
weather/noise robustness 비교
```

이 항목들은 64/128 smoke가 안정적으로 통과한 뒤에 붙인다.

---

## 2. Smoke Run의 판정 기준

64-step smoke 통과 기준:

```text
CARLA process crash 없음
Python exception 없음
obs_flat shape == 239839
model forward NaN/inf 없음
waypoints shape == [1,4,2]
target_speed_kmh finite
pure pursuit control finite
steer/throttle/brake가 CARLA range 안에 있음
env.step이 64번 또는 정상 done까지 진행
logs.csv 또는 v1_1_eval_steps.csv 생성
summary.json 생성
```

128-step smoke 통과 기준:

```text
64-step 기준 모두 만족
collision/offroad/stuck이 발생하면 done_reason과 step이 정확히 기록됨
waypoint_valid/path_valid/used_route_fallback/used_synthetic_fallback 기록됨
sensor_health_0..14 기록됨
fusion weight 4개 기록됨
route_progress가 finite
speed_kmh가 finite
controller safety cap이 계속 0 speed로 묶어두지 않음
```

64/128 smoke는 성능 합격이 아니라 연결 합격이다. offroad가 나오더라도 adapter/log가 정상이라면 다음 디버깅으로 넘어갈 수 있다.

실패를 반드시 구분한다:

```text
CARLA crash: Unreal/CARLA 프로세스 오류, TaskGraphThread, RPC timeout
adapter failure: shape mismatch, coordinate mismatch, NaN, missing key
model failure: checkpoint mismatch, model forward NaN, invalid waypoint
controller failure: invalid control, target_speed cap stuck
driving failure: collision/offroad/stuck/done_reason
```

---

## 3. 추가할 권장 파일 구조

```text
rcs_ac_v1_1/adapters/
  __init__.py
  carla_types.py
  carla_observation_adapter.py
  carla_route_adapter.py
  carla_lidar_adapter.py
  carla_map_actor_adapter.py

rcs_ac_v1_1/scripts/
  collect_bev_expert_dataset.py
  evaluate_bev_pure_pursuit.py  # 기존 파일을 real CARLA mode로 확장

rcs_ac_v1_1/tests/
  test_v1_1_carla_adapter_contract.py
```

Mac에서 CARLA package import가 불가능할 수 있으므로 adapter는 두 층으로 나눈다:

```text
pure functions:
  numpy/dict/dataclass만 받음
  Mac unit test 가능

CARLA binding:
  carla.Actor, carla.World, carla.Map, sensor raw event를 읽음
  Windows smoke에서 검증
```

CARLA binding 파일은 import 시점에 `carla`가 없어도 전체 test가 깨지지 않게 지연 import한다.

---

## 4. CARLA Observation Adapter

목표:

```text
CARLA obs/info/env state
-> V11SensorPacket
-> V11EgoState
-> V11RouteContext
-> build_policy_components()
-> obs_flat
```

권장 함수:

```python
def build_v11_state_from_carla_step(
    *,
    obs: dict,
    info: dict,
    world,
    ego_vehicle,
    route_source,
    previous_control: dict | None,
    runtime_config,
) -> tuple[V11SensorPacket, V11EgoState, V11RouteContext | None, dict]:
    ...
```

`V11SensorPacket` 필수 매핑:

```text
rgb_front:
  shape [3,90,160] 또는 [90,160,3]
  float32 [0,1] 권장
  현재 pipeline이 HWC -> CHW 변환 가능

lidar_points_ego:
  shape [N,3] float32
  coordinate: x forward, y left, z up

camera_latency_ms:
  sensor timestamp와 world timestamp 차이를 알면 기록
  모르면 0.0

lidar_latency_ms:
  sensor timestamp와 world timestamp 차이를 알면 기록
  모르면 0.0

route_latency_ms:
  route update 지연을 알면 기록
  모르면 0.0

weather_severity_norm:
  clean smoke에서는 0.0

corruption_severity_norm:
  clean smoke에서는 0.0

camera_dropout/lidar_dropout/route_dropout:
  clean smoke에서는 실제 missing일 때만 true
```

`V11EgoState` 필수 매핑:

```text
speed_kmh:
  ego velocity magnitude * 3.6

yaw_rate_rad_s:
  ego angular_velocity.z를 rad/s로 변환
  CARLA가 deg/s로 주는지 rad/s로 주는지 반드시 확인 후 기록

acceleration_mps2:
  ego acceleration magnitude 또는 longitudinal acceleration

previous_steer:
  직전 applied steer

previous_throttle:
  직전 applied throttle

previous_brake:
  직전 applied brake

previous_longitudinal:
  throttle - brake

route_progress:
  info["route_progress"]가 있으면 사용
  없으면 0.0

distance_to_next_turn_m:
  route adapter에서 계산
```

debug에 반드시 넣을 값:

```text
adapter_rgb_shape
adapter_lidar_point_count
adapter_route_point_count
adapter_obs_flat_dim
adapter_route_available
adapter_camera_dropout
adapter_lidar_dropout
adapter_route_dropout
adapter_lidar_coord_checked
```

---

## 5. Route Adapter

목표:

```text
CARLA route/local planner waypoints
-> ego frame route_points_ego [N,2]
-> target_point_ego
-> maneuver
-> curvature
-> heading_error_rad
-> distance_to_next_turn_m
```

좌표계:

```text
ego x: 차량 전방 +
ego y: 차량 좌측 +
route_points_ego[:,0]는 대부분 0 이상이어야 함
왼쪽 목표는 y > 0
오른쪽 목표는 y < 0
```

권장 pure helper:

```python
def world_xy_to_ego_xy(
    world_xy: np.ndarray,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
) -> np.ndarray:
    ...
```

검증:

```text
직선 route:
  target_y_abs_mean < 1.0m

왼쪽 route:
  future route y mean > 0

오른쪽 route:
  future route y mean < 0

전방 waypoint:
  forward point ratio >= 0.7
```

route가 없을 때:

```text
route_context = None 또는 available=False
sensor_health route availability = 0
controller가 synthetic fallback이면 target_speed_capped_kmh = 0으로 떨어져야 함
```

---

## 6. LiDAR Adapter

목표:

```text
CARLA LiDAR raw buffer
-> np.ndarray [N,3]
-> v1.1 ego coordinate
-> build_lidar_bev()
```

검증해야 하는 흔한 오류:

```text
x/y 축 뒤바뀜
y 좌우 부호 반전
z scale 오류
sensor local frame과 vehicle frame 혼동
world frame point를 ego frame으로 변환하지 않음
```

64 smoke 전에 할 최소 검증:

```text
lidar_point_count > 32
finite ratio >= 0.99
BEV grid inside ratio > 0.05
occupied channel sum > 0
unknown channel이 전부 1이 아님
```

로그 필드:

```text
lidar_point_count
lidar_finite_ratio
lidar_inside_bev_ratio
lidar_occupied_sum
lidar_free_sum
lidar_unknown_mean
```

---

## 7. CARLA Map/Actor GT BEV Adapter

64/128 smoke에서 GT BEV는 policy input이 아니다. 다만 dataset collection과 BEV auxiliary loss를 위해 real dataset에는 필요하다.

권장 구현:

```text
CarlaMapAdapter(V11MapAdapter):
  is_drivable_ego_xy(x_m, y_m)
  is_lane_marking_ego_xy(x_m, y_m)

collect_v11_actor_boxes(world, ego_vehicle):
  list[V11ActorBox]
```

주의:

```text
GT BEV target은 train target/debug 전용
policy forward/input에는 절대 넣지 않는다
actor positions/bounding boxes를 obs_flat에 concat하지 않는다
```

smoke 전 최소 검증:

```text
gt_bev_target.shape == [5,128,128]
bev_target_available in {0,1}
drivable channel finite
route_path target channel finite
obstacle channel은 actor가 없으면 0이어도 됨
```

CARLA map adapter가 늦어지면:

```text
64/128 eval smoke는 GT target 없이 진행 가능
dataset smoke는 bev_target_available=0으로 시작 가능
단, 이 상태로 BEV auxiliary quality를 평가하면 안 된다
```

---

## 8. Tiny CARLA Dataset Collection

목표:

```text
CARLA에서 64~256 samples 정도의 v1.1 NPZ를 생성한다.
train_bev_bc.py가 real CARLA data를 읽고 1 epoch smoke를 통과하는지 확인한다.
```

권장 CLI:

```bash
python -m rcs_ac_v1_1.scripts.collect_bev_expert_dataset \
  --config configs/v1_1/full_bev_smoke.yaml \
  --output data/v1_1/bev_expert_tiny_carla_smoke.npz \
  --num-steps 128 \
  --host 127.0.0.1 \
  --port 2000 \
  --town Town04 \
  --seed 90
```

필수 저장 key:

```text
rgb
lidar_bev
route_bev
ego_bev
ego_scalar
route_scalar
sensor_health
gt_bev_target
expert_waypoints
expert_target_speed
bev_target_available
waypoint_target_available
episode_id
step
done
route_progress
done_reason
```

shape:

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

수집 방식:

```text
1. expert/autopilot/local planner가 차량을 움직인다.
2. 매 step마다 v1.1 policy input components를 build한다.
3. expert future path로 expert_waypoints를 만든다.
4. expert/local planner target speed를 expert_target_speed로 저장한다.
5. GT BEV target이 준비되지 않았으면 bev_target_available=0으로 저장한다.
```

수집 smoke pass:

```text
NPZ 생성
BEVExpertDataset load 성공
finite check 통과
shape check 통과
policy input에 gt_bev_target 누수 없음
```

---

## 9. Tiny Real Dataset Training Smoke

권장 CLI:

```bash
python -m rcs_ac_v1_1.scripts.train_bev_bc \
  --config configs/v1_1/full_bev_smoke.yaml \
  --dataset data/v1_1/bev_expert_tiny_carla_smoke.npz \
  --output-dir checkpoints/v1_1_bev_bc_tiny_carla_smoke \
  --epochs 1 \
  --batch-size 8 \
  --device cuda
```

pass 기준:

```text
loss finite
waypoint_l1 finite
target_speed_mae finite
bev loss는 bev_target_available=0이면 0 또는 mask 처리
last.pt 저장
best.pt 저장 또는 last.pt만 저장 정책 명확
metrics.csv 생성
```

실패 시 확인:

```text
shape mismatch -> dataset key shape 확인
NaN loss -> obs/target finite check
waypoint loss 폭발 -> expert_waypoints 좌표계 확인
target speed loss 폭발 -> km/h vs m/s 단위 확인
```

---

## 10. Real CARLA Eval Loop

`evaluate_bev_pure_pursuit.py`를 다음 흐름으로 확장한다.

```text
1. load runtime config
2. load checkpoint or randomly initialized smoke model
3. connect CARLA/shared_core env
4. reset env
5. for step in num_steps:
   a. CARLA obs/info -> V11SensorPacket/V11EgoState/V11RouteContext
   b. build_policy_components()
   c. build_flat_policy_observation()
   d. model forward
   e. controller.control()
   f. convert controller output to CARLA control/action
   g. env.step(control)
   h. write row
6. write summary.json
```

권장 CLI:

```bash
python -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit \
  --config configs/v1_1/full_bev_smoke.yaml \
  --checkpoint checkpoints/v1_1_bev_bc_tiny_carla_smoke/last.pt \
  --run-name v1_1_carla_eval_64_smoke \
  --num-steps 64 \
  --host 127.0.0.1 \
  --port 2000 \
  --device cuda \
  --mode carla
```

128 smoke:

```bash
python -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit \
  --config configs/v1_1/full_bev_smoke.yaml \
  --checkpoint checkpoints/v1_1_bev_bc_tiny_carla_smoke/last.pt \
  --run-name v1_1_carla_eval_128_smoke \
  --num-steps 128 \
  --host 127.0.0.1 \
  --port 2000 \
  --device cuda \
  --mode carla
```

checkpoint가 아직 없으면:

```text
--allow-random-policy-smoke true
```

처럼 명시 옵션을 둔다. random policy smoke 결과를 성능으로 해석하면 안 된다.

---

## 11. Required Eval Logs

step log 필수 컬럼:

```text
step
episode_step
reward
done
done_reason
collision
offroad
stuck
route_progress
route_progress_delta
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
sensor_health_0
sensor_health_1
sensor_health_2
sensor_health_3
sensor_health_4
sensor_health_5
sensor_health_6
sensor_health_7
sensor_health_8
sensor_health_9
sensor_health_10
sensor_health_11
sensor_health_12
sensor_health_13
sensor_health_14
obs_flat_dim
obs_finite
rgb_mean
lidar_occupied_sum
route_bev_path_sum
ego_bev_footprint_sum
route_point_count
lidar_point_count
```

summary 필수 field:

```text
run_name
num_steps_requested
num_steps_completed
exit_status
crash_type
exception_type
exception_message
collision_count
offroad_count
stuck_count
first_done_step
first_done_reason
max_route_progress
final_route_progress
mean_speed_kmh
mean_abs_steer
mean_target_speed_capped_kmh
waypoint_valid_rate
path_valid_rate
used_route_fallback_rate
used_synthetic_fallback_rate
obs_finite_rate
```

---

## 12. Mac-Windows 실행 흐름

Mac에서 코드 수정 후:

```bash
python -m pytest rcs_ac_v1_1/tests -q
git diff -- rcs_ac_v1_1 docs/v1_1 configs/v1_1
git push
```

Windows에서:

```powershell
Set-Location 'C:\Users\thddy\Documents\code\RCS-AC-first'
git pull
.\scripts\windows\ensure_carla_ready.ps1
```

통합 runner:

```powershell
.\scripts\windows\run_v1_1_smoke.ps1 `
  -EnvConfig configs\<existing-clean-carla-config>.yaml `
  -Stage all
```

단계별 runner:

```powershell
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage health
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage collect
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage train
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage eval64
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage eval128
```

64 smoke:

```powershell
py -3.12 -B -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit `
  --config configs/v1_1/full_bev_smoke.yaml `
  --checkpoint checkpoints/v1_1_bev_bc_tiny_carla_smoke/last.pt `
  --run-name v1_1_carla_eval_64_smoke `
  --num-steps 64 `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000 `
  --mode carla
```

128 smoke:

```powershell
py -3.12 -B -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit `
  --config configs/v1_1/full_bev_smoke.yaml `
  --checkpoint checkpoints/v1_1_bev_bc_tiny_carla_smoke/last.pt `
  --run-name v1_1_carla_eval_128_smoke `
  --num-steps 128 `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000 `
  --mode carla
```

원격 SSH 실행 시 권장:

```text
ServerAliveInterval=30
ServerAliveCountMax=6
ControlMaster=auto
ControlPersist=8h
CARLA 중복 실행 금지
ensure_carla_ready.ps1 먼저 실행
```

CARLA가 TaskGraphThread 오류로 죽으면:

```text
1. CARLA 프로세스가 2개 떠 있는지 확인
2. 기존 CARLA 종료
3. ensure_carla_ready.ps1 재실행
4. 같은 run_name을 재사용하지 말고 crash suffix를 붙임
5. crash를 모델 실패로 기록하지 않음
```

---

## 13. Failure Triage

### 13.1 Shape mismatch

확인:

```text
obs_flat_dim == 239839
rgb shape [3,90,160]
lidar_bev [5,128,128]
route_bev [4,128,128]
ego_bev [3,128,128]
ego_scalar [6]
route_scalar [10]
sensor_health [15]
```

### 13.2 시작하자마자 steer가 한쪽으로 고정

확인:

```text
route_points_ego y sign
ego yaw rad/deg 변환
world->ego transform
target_point_ego y
pure pursuit raw_steer
previous_steer 초기화
```

### 13.3 target speed가 계속 0

확인:

```text
waypoint_valid
path_valid
used_synthetic_fallback
route_context available
waypoint x monotonic
uncertainty speed cap
curvature speed cap
```

### 13.4 LiDAR BEV가 거의 비어 있음

확인:

```text
lidar_point_count
lidar_inside_bev_ratio
x/y/z coordinate convention
sensor transform applied twice 여부
z min/max filter
```

### 13.5 route BEV가 비어 있음

확인:

```text
route_point_count
route points forward ratio
ego frame transform
route_context.available
target_point_ego
```

---

## 14. 하드코딩 금지

금지:

```text
lane_offset 기반 runtime steer override
route y sign을 보고 직접 steer rule 적용
특정 route/seed 전용 steer constant
offroad 방지를 위해 CARLA 위치를 직접 policy input에 concat
GT BEV target을 inference input으로 사용
random policy smoke를 clean 성능으로 기록
controller safety cap 결과를 raw policy 성능으로 포장
```

허용:

```text
shape validation
NaN guard
invalid waypoint 시 stop cap
pure pursuit controller safety cap
debug-only GT BEV rendering
dataset target 생성을 위한 privileged CARLA signal
```

보고 시 반드시 분리:

```text
raw policy output
controller output
safety capped output
CARLA applied control
```

---

## 15. 64/128 Smoke 완료 후 다음 단계

64/128 smoke가 통과하면:

```text
1. route/LiDAR/map adapter audit 문서화
2. tiny dataset을 512~2048 samples로 확장
3. BC train을 5~20 epoch smoke로 확장
4. 256-step clean eval
5. 512-step clean eval
6. 2048-step clean eval
```

64/128 smoke에서 driving failure가 있어도:

```text
adapter/log/model/controller loop가 정상이라면 구현 단계는 통과
주행 안정성 문제는 256 전 디버깅 단계로 분리
```

64/128 smoke에서 adapter failure가 있으면:

```text
512/2048로 절대 넘어가지 않는다
temporal smoothness loss나 corruption training도 시작하지 않는다
```

---

## 16. 최종 체크리스트

구현 체크:

```text
[x] carla_observation_adapter.py
[x] carla_route_adapter.py
[x] carla_lidar_adapter.py
[x] carla_map_actor_adapter.py
[x] collect_bev_expert_dataset.py
[x] evaluate_bev_pure_pursuit.py --mode carla
[x] scripts/windows/run_v1_1_smoke.ps1
[x] adapter contract unit tests
[x] Mac pytest pass
[ ] Windows git pull
[ ] CARLA health check pass
[ ] tiny dataset collection pass
[ ] train_bev_bc real tiny dataset 1 epoch pass
[ ] 64-step smoke pass
[ ] 128-step smoke pass
[ ] summary/logs documented
```

64/128 smoke 완료 산출물:

```text
logs/v1_1_carla_eval_64_smoke/*.csv
logs/v1_1_carla_eval_64_smoke/summary.json
logs/v1_1_carla_eval_128_smoke/*.csv
logs/v1_1_carla_eval_128_smoke/summary.json
docs/v1_1 라운드 결과 기록
git diff 확인
git push 가능 상태
```
