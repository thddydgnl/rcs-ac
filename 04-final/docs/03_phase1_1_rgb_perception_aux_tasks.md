# RCS-AC-first Phase 1.1: RGB Perception Auxiliary Task 상세 설계

작성일: 2026-04-26

## 1. 목표

Phase 1 Core는 RGB, LiDAR, ego, route 입력을 fusion gate와 GRU actor-critic으로 연결해 PPO 루프가 깨지지 않는 것을 증명했다. Phase 1.1의 목표는 RGB encoder가 도로 구조를 더 안정적으로 배우도록 다음 dense perception 보조 task를 추가하는 것이다.

- Drivable area / road mask 예측
- Lane boundary / lane mask 예측

이 기능은 policy action contract, reward, route logic을 바꾸지 않는다. RGB에서 나온 feature map 위에 작은 segmentation head를 붙이고, CARLA semantic segmentation camera가 만든 road / roadline target으로 보조 loss만 더한다.

## 2. 참고 자료에서 가져올 설계 포인트

### 2.1 1Kaustubh122/Self_Driving_Car

참고 저장소는 CARLA에서 PPO baseline으로 환경과 제어 루프를 먼저 검증한 뒤, expert RGB + segmentation 데이터를 이용해 Behavior Cloning을 학습하고, SAC로 fine-tuning하는 흐름을 사용한다. README 기준으로 BC 단계는 RGB + segmentation 이미지를 수집하고 5-channel ResNet 계열 입력으로 steering, throttle, brake를 회귀한다.

RCS-AC-first에는 이 방식을 그대로 복사하지 않는다. segmentation을 policy observation의 추가 채널로 넣으면 Phase 1 observation contract가 바뀌고, CARLA semantic camera가 없을 때 정책 입력 자체가 흔들린다. 따라서 참고점은 다음 하나로 제한한다.

- RGB만으로 제어를 배우는 것보다 road/lane 구조 정보를 함께 학습시키면 CARLA RL/IL 흐름에서 representation이 좋아질 수 있다.

RCS-AC-first의 선택은 "segmentation as input"이 아니라 "segmentation as auxiliary supervision"이다.

### 2.2 CARLA semantic segmentation camera

CARLA semantic segmentation camera는 각 pixel을 scene object class label로 제공한다. RCS-AC-first의 목표 target은 semantic label image에서 직접 만들 수 있다.

- road target: `carla.CityObjectLabel.Roads`
- lane / roadline target: `carla.CityObjectLabel.RoadLines`
- target resize: binary mask이므로 nearest-neighbor resize
- policy input: 기존 RGB `(3, 90, 160)` 유지
- auxiliary target resolution: 기존 config와 맞춰 `(1, 45, 80)` 권장

현재 로컬 `shared_core`에는 이미 이 방향의 구현 단서가 있다.

- `C:\Users\thddy\Documents\code\shared_core\env\sensors.py`
  - `SemanticSegmentationCamera`
  - `sensor.camera.semantic_segmentation`
- `C:\Users\thddy\Documents\code\shared_core\env\carla_env.py`
  - `semantic_aux_camera_enabled`
  - `aux_mask_width`, `aux_mask_height`
  - `aux_road_mask_target`
  - `aux_lane_edge_target`
  - `aux_mask_targets_available`

로컬 구현은 roadline mask뿐 아니라 road mask의 경계도 lane edge target에 합친다. Phase 1.1에서는 이 방식을 채택한다.

```text
road_mask      = semantic_labels == Roads
roadline_mask  = semantic_labels == RoadLines
road_boundary  = edge(road_mask)
lane_edge_mask = roadline_mask OR road_boundary
```

이렇게 하면 lane marking이 약하게 보이는 구간에서도 road boundary supervision이 RGB encoder에 들어간다.

### 2.3 TwinLiteNet

TwinLiteNet은 drivable area segmentation과 lane detection을 함께 수행하는 lightweight dual segmentation 구조를 제공한다. 핵심 아이디어는 shared image encoder 뒤에 두 개의 작은 decoder/head를 둬서 road와 lane을 동시에 학습시키는 것이다.

RCS-AC-first에는 전체 TwinLiteNet을 넣지 않는다. Phase 1.1은 actor-critic이 주 모델이므로, TwinLiteNet에서 가져올 부분은 다음이다.

- 하나의 RGB feature extractor를 공유한다.
- drivable area와 lane line을 분리된 output channel로 예측한다.
- lightweight upsampling head로 계산량을 작게 유지한다.

### 2.4 YOLOP

YOLOP는 object detection, drivable area segmentation, lane detection을 하나의 multi-task perception network 안에서 같이 학습한다. YOLOP의 중요한 참고점은 multi-task를 end-to-end로 같이 학습해도 perception branch가 shared encoder의 표현을 보강할 수 있다는 점이다.

RCS-AC-first에는 YOLOP의 detection branch나 anchor/grid 설계를 넣지 않는다. 가져올 점은 다음이다.

- shared encoder + task-specific heads
- segmentation head는 policy output과 분리
- auxiliary loss weight로 RL objective를 압도하지 않게 조절

## 3. 최종 설계 결정

### 3.1 Phase 이름

이 작업은 Phase 2가 아니라 Phase 1.1로 둔다.

- Phase 1: Core end-to-end PPO loop
- Phase 1.1: RGB perception auxiliary supervision
- Phase 2: reliability + risk-aware fusion
- Phase 3: counterfactual Shapley full RCS-AC

이유는 perception auxiliary task가 reliability, risk, Shapley와 독립적이고, policy loop 안정성을 해치지 않는 Core 확장에 가깝기 때문이다.

### 3.2 입력과 출력 contract

Policy observation contract는 유지한다.

```text
rgb   : (3, 90, 160)
lidar : (1, 64, 64)
ego   : (5,)
route : (10,)
```

보조 target은 rollout buffer에 별도로 저장한다.

```text
aux_road_mask_target       : (1, 45, 80), float32, {0, 1}
aux_lane_edge_target       : (1, 45, 80), float32, {0, 1}
aux_mask_targets_available : scalar bool/float
aux_mask_target_source     : "semantic_camera" | "unavailable"
```

모델 prediction key는 다음으로 고정한다.

```text
pred_road_mask_logits      : (B, 1, 45, 80)
pred_lane_edge_mask_logits : (B, 1, 45, 80)
```

logit을 반환하고 loss에서 `BCEWithLogitsLoss`를 적용한다. 모델 forward에서 sigmoid를 미리 씌우지 않는다.

## 4. Target 생성 설계

### 4.1 CARLA semantic camera 설정

`configs/phase1_core_smoke.yaml`의 env block은 이미 다음 값을 갖고 있다.

```yaml
env:
  semantic_aux_camera_enabled: true
  semantic_sensor:
    image_size_x: 800
    image_size_y: 450
    fov: 100
    sensor_tick: 0.0
  aux_mask_width: 80
  aux_mask_height: 45
```

Phase 1.1에서는 이 설정을 유지하고, train env wrapper가 info dict 또는 observation auxiliary dict를 통해 mask target을 PPO buffer에 넘기도록 연결한다.

### 4.2 Road mask

road mask는 CARLA semantic label이 `Roads`인 pixel이다.

```text
road_mask = semantic_labels == SEMANTIC_ROAD_LABEL
```

결과는 nearest resize로 `(45, 80)`에 맞춘 뒤 channel dimension을 붙인다.

```text
aux_road_mask_target.shape = (1, 45, 80)
```

### 4.3 Lane edge mask

lane edge는 두 신호를 합친다.

- semantic roadline: `semantic_labels == SEMANTIC_ROAD_LINE_LABEL`
- road boundary: road mask의 4-neighbor 또는 8-neighbor 경계

```text
lane_edge_mask = roadline_mask OR road_boundary
```

Phase 1.1의 이름은 `lane_edge`로 둔다. 사용자가 말한 "lane boundary / lane mask"를 모두 포괄하기 위해, 나중에 필요하면 `aux_lane_line_mask_target`을 따로 분리할 수 있다. 지금은 sparse lane line만 쓰면 loss가 너무 약해질 수 있으므로 road boundary를 포함한 `lane_edge`가 더 안정적이다.

### 4.4 Availability rule

semantic camera가 아직 frame을 주지 않았거나 CARLA 없이 local smoke를 돌리는 경우에는 loss를 꺼야 한다.

```text
aux_mask_targets_available = 1.0  # semantic target valid
aux_mask_targets_available = 0.0  # target unavailable
```

loss 계산은 available sample만 사용한다. batch 전체가 unavailable이면 mask loss는 zero tensor로 반환한다.

## 5. 모델 구조 설계

### 5.1 RGBEncoder 변경

현재 `RGBEncoder`는 conv stack 뒤에 `AdaptiveAvgPool2d`, `Flatten`, `Linear`까지 한 번에 실행한다. Dense segmentation head를 붙이려면 pooling 전 spatial feature map이 필요하다.

권장 변경:

```text
RGBEncoder
  stem:
    Conv2d 3 -> 32, stride 2
    ReLU
    Conv2d 32 -> 64, stride 2
    ReLU
    Conv2d 64 -> 128, stride 2
    ReLU
  pool_head:
    AdaptiveAvgPool2d(1, 1)
    Flatten
    Linear 128 -> feature_dim
    ReLU
```

public method:

```python
def forward(self, rgb) -> torch.Tensor:
    return self.forward_with_feature_map(rgb).feature

def forward_with_feature_map(self, rgb) -> RGBEncoderOutput:
    return RGBEncoderOutput(feature=pooled_feature, feature_map=spatial_feature_map)
```

예상 feature map 크기:

```text
input  : (B, 3, 90, 160)
stride : 2 * 2 * 2 = 8
map    : approximately (B, 128, 12, 20)
```

정확한 크기에 의존하지 않고 aux head에서 `F.interpolate(..., size=(45, 80))`를 사용한다.

### 5.2 RGBPerceptionAuxHead 추가

새 파일 후보:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\perception_aux.py
```

권장 구조:

```python
class RGBPerceptionAuxHead(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_size=(45, 80)):
        ...

    def forward(self, rgb_feature_map):
        x = Conv2d(in_channels, hidden_channels, 3, padding=1)
        x = ReLU
        x = Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        x = ReLU
        x = interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        logits = Conv2d(hidden_channels, 2, 1)
        return {
            "pred_road_mask_logits": logits[:, 0:1],
            "pred_lane_edge_mask_logits": logits[:, 1:2],
        }
```

왜 이 정도만 쓰는가:

- Phase 1.1은 robust representation을 위한 보조 학습이 목표다.
- 전체 TwinLiteNet/YOLOP decoder를 넣으면 PPO smoke의 계산량과 디버깅 범위가 커진다.
- 출력 해상도는 `(45, 80)`이라 작은 head로 충분하다.

### 5.3 RCSACFirstModel 변경

`RCSACFirstModel.forward_policy_outputs`는 기존 output을 유지하고, feature map이 있을 때 perception logits를 추가한다.

추가 output:

```text
pred_road_mask_logits
pred_lane_edge_mask_logits
```

`predict_auxiliary_targets`도 scalar aux와 dense aux를 함께 반환한다.

```python
return {
    "pred_lane_offset": ...,
    "pred_heading_error": ...,
    "pred_road_mask_logits": ...,
    "pred_lane_edge_mask_logits": ...,
}
```

중요한 제한:

- perception logits를 fusion gate 입력으로 넣지 않는다.
- perception logits를 reward에 넣지 않는다.
- perception logits를 action sampling에 직접 사용하지 않는다.

Phase 1.1의 효과는 RGB encoder gradient를 보강하는 데만 둔다.

## 6. Loss 설계

### 6.1 Config 추가

`RCSACFirstModelConfig`:

```python
enable_rgb_perception_aux: bool = True
aux_mask_height: int = 45
aux_mask_width: int = 80
aux_perception_hidden_channels: int = 64
```

`RCSACFirstLossConfig`:

```python
aux_road_mask_loss_weight: float = 0.25
aux_lane_edge_mask_loss_weight: float = 1.0
aux_lane_edge_pos_weight: float = 3.0
```

smoke 안정성을 우선한다면 최초 local smoke에서는 loss weight를 0으로 두고 shape/backward만 확인한 뒤, CARLA semantic target이 안정적으로 들어오는 것을 확인하면 위 값을 켠다.

### 6.2 BCE loss

road는 면적이 넓고 lane edge는 sparse하므로 lane 쪽에 positive weight를 준다.

```text
road_loss = BCEWithLogits(pred_road_mask_logits, road_target)
lane_loss = BCEWithLogits(pred_lane_edge_mask_logits, lane_target, pos_weight=lane_edge_pos_weight)
```

sample availability 적용:

```text
valid = aux_mask_targets_available > 0.5
if valid.sum() == 0:
    dense_aux_loss = 0
else:
    dense_aux_loss =
      road_weight * road_loss(valid)
      + lane_weight * lane_loss(valid)
```

### 6.3 PPO total loss에 더하는 위치

현재 total loss:

```text
actor_loss
+ value_loss_weight * critic_loss
+ entropy_loss_weight * entropy_loss
+ scalar_aux_loss
```

Phase 1.1 total loss:

```text
actor_loss
+ value_loss_weight * critic_loss
+ entropy_loss_weight * entropy_loss
+ scalar_aux_loss
+ dense_perception_aux_loss
```

주의할 점:

- `dense_perception_aux_loss`는 actor/value loss보다 커지기 쉽다.
- 첫 CARLA smoke에서는 road 0.10, lane 0.50처럼 낮게 시작해도 된다.
- policy가 움직이지 않거나 fusion weight가 RGB에 과도하게 몰리면 loss weight를 낮춘다.

## 7. RolloutBuffer 변경

`RolloutBatch`에 필드를 추가한다.

```python
aux_road_mask_targets: torch.Tensor          # (T, 1, 45, 80)
aux_lane_edge_targets: torch.Tensor          # (T, 1, 45, 80)
aux_mask_targets_available: torch.Tensor     # (T,)
```

`RolloutBuffer.reset`:

```python
self.aux_road_mask_targets = np.zeros((capacity, 1, 45, 80), dtype=np.float32)
self.aux_lane_edge_targets = np.zeros((capacity, 1, 45, 80), dtype=np.float32)
self.aux_mask_targets_available = np.zeros((capacity,), dtype=np.float32)
```

`RolloutBuffer.add`:

```python
aux_road_mask_target=None
aux_lane_edge_target=None
aux_mask_targets_available=False
```

unavailable이면 zero mask를 저장하되 availability를 0으로 둔다.

## 8. Train/env 연결

`scripts/train.py`가 env step 후 info 또는 observation auxiliary dict에서 다음 키를 읽도록 한다.

```text
aux_road_mask_target
aux_lane_edge_target
aux_mask_targets_available
aux_mask_target_source
```

현재 `shared_core`가 이 키 이름을 이미 쓰고 있으므로, RCS-AC-first 쪽도 이름을 맞추는 것이 좋다.

CARLA 없는 local smoke에서는 dummy target을 만든다.

```text
road dummy: center-lower rectangle 또는 zeros
lane dummy: two diagonal thin lines 또는 zeros
available: 1.0 for model/loss smoke, 0.0 for unavailable smoke
```

local smoke는 두 케이스를 모두 검증한다.

## 9. Logging 설계

`training_log.csv`에 필드 추가:

```text
aux_road_mask_loss
aux_lane_edge_mask_loss
aux_road_mask_mae
aux_lane_edge_mask_mae
aux_road_mask_iou
aux_lane_edge_mask_iou
aux_road_mask_positive_ratio
aux_lane_edge_positive_ratio
aux_mask_available_rate
```

`env_step_log.csv`에 필드 추가:

```text
aux_mask_targets_available
aux_mask_target_source
aux_road_mask_positive_ratio
aux_lane_edge_positive_ratio
```

IoU 계산은 sigmoid threshold 0.5로 충분하다.

```text
pred_mask = sigmoid(logits) >= 0.5
iou = intersection / union
if union == 0: iou = 1.0 if both empty else 0.0
```

## 10. 구현 순서

1. Config 확장
   - `enable_rgb_perception_aux`
   - `aux_mask_height`
   - `aux_mask_width`
   - dense aux loss weights

2. RGB encoder feature map 노출
   - `RGBEncoderOutput` dataclass 추가
   - `forward_with_feature_map` 추가
   - 기존 `forward` 호환 유지

3. `RGBPerceptionAuxHead` 추가
   - `models/perception_aux.py`
   - 2-channel logits 출력

4. `RCSACFirstModel` 연결
   - `forward_policy_outputs` output key 추가
   - `predict_auxiliary_targets`에 dense logits 추가
   - `get_stage_b_warmup_modules`에 aux head 포함

5. Dense aux loss 추가
   - `rl/losses.py`
   - `compute_dense_perception_aux_loss`
   - availability masking
   - MAE / IoU / positive ratio metrics

6. Rollout buffer 확장
   - mask target arrays
   - batch tensor conversion

7. PPO update 연결
   - scalar aux loss 뒤에 dense aux loss 추가
   - metrics accumulator에 dense aux metrics 포함

8. Train env bridge 연결
   - `shared_core` info key 수집
   - 없으면 unavailable 처리

9. Smoke check 확장
   - dummy observation forward
   - dense logits shape check
   - available / unavailable loss check
   - checkpoint save/load

10. 로그 필드 추가
   - `training_log.csv`
   - `env_step_log.csv`

## 11. 검증 플랜

### 11.1 Static validation

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
```

통과 기준:

- syntax error 없음

### 11.2 Local perception smoke

```powershell
py -3.12 -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase1_core_smoke.yaml
```

추가 통과 기준:

- `pred_road_mask_logits.shape == (B, 1, 45, 80)`
- `pred_lane_edge_mask_logits.shape == (B, 1, 45, 80)`
- dense aux loss backward 성공
- availability가 0인 batch에서 dense aux loss가 0으로 반환
- availability가 1인 batch에서 finite loss 반환
- NaN/Inf 없음

### 11.3 CARLA semantic target probe

```powershell
py -3.12 -m rcs_ac_first.scripts.train --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase1_core_smoke.yaml --total-timesteps 128 --run-name phase1_1_semantic_probe_128
```

통과 기준:

- CARLA semantic camera spawn 성공
- `aux_mask_targets_available`가 최소 1회 true
- road mask positive ratio가 0보다 큼
- lane edge mask positive ratio 기록
- env close 정상 수행

### 11.4 CARLA PPO perception smoke

```powershell
py -3.12 -m rcs_ac_first.scripts.train --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase1_core_smoke.yaml --total-timesteps 512 --run-name phase1_1_perception_smoke_512
```

통과 기준:

- PPO update 최소 1회 성공
- dense aux loss가 finite
- `training_log.csv`에 road/lane loss 기록
- `env_step_log.csv`에 target availability 기록
- checkpoint reload 가능
- action/log_prob/value/fusion diagnostics 기존 기준 유지

## 12. 실패 시 우선 점검 순서

1. `aux_mask_targets_available`가 계속 0
   - semantic camera spawn 실패 여부
   - sensor queue sync 여부
   - env info key 이름 불일치 여부

2. road mask가 전부 0
   - CARLA label id mismatch
   - raw semantic image channel decode 방식 확인
   - `CityObjectLabel.Roads` fallback 값 확인

3. lane edge loss가 너무 큼
   - lane target positive ratio 확인
   - `aux_lane_edge_pos_weight` 낮추기
   - lane loss weight를 0.25 또는 0.5로 낮추기

4. PPO가 불안정
   - dense aux loss weight를 0으로 두고 Core smoke 재확인
   - road/lane loss를 켠 뒤 actor/value loss scale 비교
   - RGB feature map head가 encoder gradient를 과하게 키우는지 grad norm 확인

5. 메모리/속도 저하
   - aux mask resolution을 `(32, 64)`로 낮추기
   - aux head hidden channels를 32로 낮추기
   - mask target을 float16으로 저장하지 말고, buffer에서는 float32 유지

## 13. Phase 2/3와의 관계

Phase 1.1은 perception auxiliary supervision만 추가한다.

- reliability head는 아직 비활성화
- risk context는 아직 비활성화
- Shapley contribution은 아직 uniform placeholder
- dropout consistency는 아직 zero loss

다만 Phase 2에서 reliability를 켤 때 perception metrics를 sensor reliability pseudo-label 후보로 사용할 수 있다.

예:

```text
rgb_reliability_hint =
  high if road/lane aux loss is low and mask availability is stable
  low  if aux loss spikes under fog/rain/camera corruption
```

이 연결은 Phase 2 문서에서 별도로 설계한다.

## 14. 수용 기준

Phase 1.1 완료 기준:

- 기존 Phase 1 Core smoke가 그대로 통과한다.
- RGB perception aux head가 dense logits를 반환한다.
- semantic camera target이 buffer에 저장된다.
- target unavailable 상황에서 학습이 깨지지 않는다.
- PPO update에서 dense aux loss backward가 된다.
- 로그에 road/lane loss와 mask availability가 남는다.
- checkpoint save/load가 기존처럼 동작한다.

성능 기준은 아직 두지 않는다. 이 단계의 목표는 주행 성능 향상이 아니라 RGB encoder가 road/lane supervision을 받는 학습 루프를 안정적으로 연결하는 것이다.

## 15. 참고 링크

- 1Kaustubh122/Self_Driving_Car: https://github.com/1Kaustubh122/Self_Driving_Car
- 1Kaustubh122 custom 5-channel ResNet 코드: https://raw.githubusercontent.com/1Kaustubh122/Self_Driving_Car/main/Imitation_Learning_RL/model/ImitationResNet.py
- CARLA semantic segmentation camera: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
- CARLA CityObjectLabel source: https://carla.org/Doxygen/html/d0/dea/ObjectLabel_8h_source.html
- TwinLiteNet: https://github.com/chequanghuy/TwinLiteNet
- YOLOP: https://github.com/hustvl/YOLOP
- YOLOP paper: https://arxiv.org/abs/2108.11250
