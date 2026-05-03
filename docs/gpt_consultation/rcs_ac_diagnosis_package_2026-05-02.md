# RCS-AC 알고리즘 안정화 진단 패키지

작성일: 2026-05-02  
저장소: `/Users/song-yonghwi/code/RCS-AC-first`  
브랜치: `codex-stabilization-rgb-v2`  
최신 관련 커밋: `21ee7f1 Record round 29 early route guard result`

이 문서는 외부 GPT 또는 연구/코드 리뷰 모델에게 RCS-AC 알고리즘의 현재 구현 상태와 안정화 실험 결과를 전달하기 위한 진단 패키지다. 목적은 다음 질문에 답을 얻는 것이다.

1. 왜 매 episode 시작 직후 왼쪽으로 크게 휘는가?
2. 현재 RCS-AC 구조에서 안정화가 어려운 근본 원인은 무엇인가?
3. BC, RGB perception, route/lane correction, PPO fine-tuning을 어떤 순서와 방식으로 고쳐야 하는가?
4. lateral recovery guard 같은 런타임 보조가 알고리즘 성능을 가리는지, 혹은 안정화에 필요한 안전층인지 어떻게 분리 검증해야 하는가?

---

## 1. 프로젝트와 목표

RCS-AC-first는 CARLA에서 동작하는 multi-modal actor-critic 기반 자율주행 정책이다. 현재는 PPO 단독 학습보다 BC pretraining 및 안정화 gate를 먼저 통과시키는 흐름으로 진행 중이다.

주요 목표:

- RGB, LiDAR, ego state, route 정보를 융합해 steer/longitudinal action을 출력한다.
- RGB road/lane perception auxiliary head를 사용해 actor가 lane/road 정보를 행동에 반영하도록 만든다.
- reliability, risk context, Shapley contribution, dropout consistency 등의 RCS-AC 요소를 넣어 modality contribution을 안정적으로 다루려 한다.
- CARLA Town04 route에서 collision/offroad/stuck 없이 route progress를 늘리는 것이 안정화 목표다.

현재 액션 계약:

- 파일: `rcs_ac_first/runtime/action.py`
- action shape: `(2,)`
- order: `("steer", "longitudinal")`
- bounds: steer `[-1, 1]`, longitudinal `[-1, 1]`
- longitudinal > 0은 throttle, longitudinal < 0은 brake로 decode된다.

---

## 2. 현재 모델 구조 요약

핵심 클래스는 `RCSACFirstModel`이다.

파일: `rcs_ac_first/models/rcs_ac_model.py`

구성:

- `MultiModalEncoder`
  - 입력: RGB, LiDAR, ego, route
  - 각 modality를 feature_dim 128 수준으로 encoding
- `FusionGate`
  - modality order: `rgb`, `lidar`, `ego`, `route`
  - reliability, contribution, risk scalar/context를 받아 modality weight를 만든다.
  - route/ego에는 safety floor가 있다.
- `RecurrentActorCritic`
  - GRU 기반 actor-critic
  - action mean: steer, longitudinal
  - value output
- `RGBPerceptionAuxHead`
  - RGB feature map에서 road mask와 lane edge mask logits 예측
- `RGBPerceptionActorFeatureExtractor`
  - road/lane mask logits를 14개 raw statistic으로 압축하고 16-d compact feature로 project
  - actor input adapter가 fused feature와 compact RGB perception feature를 결합
- reliability module, risk context head, masked critic 포함

forward 흐름:

```text
flat obs
  -> split RGB/LiDAR/ego/route
  -> MultiModalEncoder
  -> RGB perception aux head: road/lane mask logits
  -> RGB perception actor feature extractor: 14 raw stats -> 16 compact features
  -> reliability/risk/context/contribution 계산
  -> FusionGate로 rgb/lidar/ego/route fused feature 생성
  -> actor_input_adapter(fused + rgb_perception_actor_features)
  -> recurrent actor critic
  -> action_mean = [steer, longitudinal]
```

중요한 구현 위치:

- `rcs_ac_first/models/rcs_ac_model.py:22-148`: 모델 구성
- `rcs_ac_first/models/rcs_ac_model.py:150-190`: forward 시작, RGB perception actor feature 생성
- `rcs_ac_first/models/perception_aux.py:43-120`: RGB perception actor feature extractor
- `rcs_ac_first/models/fusion_gate.py`: modality fusion

---

## 3. RGB perception actor feature의 실제 의미

현재 RGB perception actor feature는 segmentation mask 자체를 actor에 넣는 것이 아니라, road/lane mask logits에서 다음 통계값을 만든다.

파일: `rcs_ac_first/models/perception_aux.py`

raw feature 14개:

1. road mean
2. road lower mean
3. road center mean
4. road bottom mean
5. lane mean
6. lane lower mean
7. lane center mean
8. lane left mean
9. lane right mean
10. lane bottom mean
11. road_x
12. lane_x
13. road_y
14. lane_y

여기서 `road_x`, `lane_x`는 lower half mask의 x-coordinate weighted center다. 이후 코드에서 `visual_center_error = lane_x - road_x`로 해석한다.

중요한 한계:

- visual center는 mask 통계 기반이지 실제 lane centerline geometry가 아니다.
- lane edge mask가 좌/우 차선을 안정적으로 분리하지 않는다.
- road_x, lane_x의 sign convention이 CARLA lane_offset sign과 항상 일관되는지 강하게 검증되지 않았다.
- actor는 compact 16-d feature를 보지만, fusion weight 관점에서는 RGB가 낮게 쓰이는 경우가 많았다.

Round29 eval 예시:

- severe early offroad 시점에 `fusion_rgb`가 약 `0.001~0.023` 수준으로 낮았다.
- `reliability_rgb`는 1.0인데도 gate는 RGB를 크게 쓰지 않았다.
- 하지만 `visual_zero_steer_delta_mean`은 커져서 RGB feature가 actor input에는 영향을 주기 시작했다.

즉 현재 RGB는 “죽어 있지는 않지만, 조향을 충분히 빠르고 크게 바로잡는 주된 신호”가 되지는 못한다.

---

## 4. FusionGate와 RCS-AC contribution 상태

파일: `rcs_ac_first/models/fusion_gate.py`

FusionGate는 각 modality feature에 대해 MLP logit을 만들고, reliability log, risk scalar, contribution bias를 더한 뒤 softmax를 사용한다.

특징:

- modality order: `rgb`, `lidar`, `ego`, `route`
- reliability는 log bias로 들어간다.
- risk scalar는 reliability bias를 다시 scale한다.
- contribution bias scale은 config에서 0.5인 경우가 있다.
- route와 ego에는 minimum floor가 있다.

Round29 guard config:

```yaml
fusion:
  route_weight_floor: 0.08
  ego_weight_floor: 0.05
  contribution_bias_scale: 0.5
  reliability_logit_scale: 1.0
  risk_logit_scale: 0.5
  use_risk_context: true
```

가능한 안정화 문제:

- route/ego floor가 RGB의 행동 지배력을 낮출 수 있다.
- reliability_rgb가 1.0이어도 fusion_rgb가 낮은 현상이 관찰됐다.
- Shapley contribution은 개념상 RCS-AC 핵심이지만, 현재 안정화 config에서는 대개 disabled다.
- actor input adapter로 RGB perception compact feature를 fused feature 뒤에 다시 붙이므로, fusion_rgb와 actor-facing RGB influence가 서로 다르게 측정된다.

---

## 5. PPO와 Shapley 구현 상태

파일:

- `rcs_ac_first/rl/ppo_agent.py`
- `rcs_ac_first/rl/shapley.py`

현재 PPO:

- PPOAgent가 RCSACFirstModel을 포함한다.
- action distribution은 bounded squashed normal이다.
- recurrent hidden state를 rollout buffer에 저장한다.
- update 시 actor loss, critic loss, aux lane/heading loss, dense perception aux loss, entropy, reliability/risk/shapley/dropout consistency loss 등을 합산한다.

Round29 guard config PPO:

```yaml
ppo:
  learning_rate: 0.000005
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.06
  update_epochs: 1
  minibatch_size: 64
  max_grad_norm: 0.5
  override_loaded_log_std: true
  loaded_log_std_by_dim: [-2.0, -2.0]
```

Shapley 관련:

- `shapley.enabled: false`인 config가 많다.
- 코드상 `every_updates`가 PPO update에서 반영되도록 최근 수정된 상태다.
- Shapley subset estimator는 `subsets_per_sample`, `batch_fraction`을 사용한다.

중요한 안정화 관찰:

- 512-step deterministic BC eval이 gate를 통과해도 PPO 2048 training rollout에서 offroad/collision이 다시 발생한다.
- PPO fine-tuning 중 exploration, critic error, reward shaping, guard/smoothing 상호작용이 BC 안정성을 쉽게 무너뜨린다.
- 따라서 현재는 PPO보다 deterministic BC/eval 안정화가 선행되어야 한다.

---

## 6. BC 학습 loss 구성

파일: `rcs_ac_first/scripts/train_bc.py`

기본 BC:

- steer MSE weight: 보통 3.0
- longitudinal MSE weight: 보통 0.5
- perception aux loss: road/lane mask supervised loss

추가 안정화 loss:

1. visual-zero margin loss
   - RGB perception actor feature를 zero로 만든 출력과 원래 출력 차이를 키워 RGB 영향력을 강제한다.

2. target corrective delta loss
   - `target_corrective_delta`를 직접 imitation한다.
   - target은 recovery collection에서 `corrected_steer - policy_steer` 형태로 저장된다.

3. visual center corrective sign loss
   - `visual_center_error = lane_x - road_x`
   - RGB feature가 만든 steer delta가 visual center error를 줄이는 방향인지 본다.

4. route-aware visual center loss
   - route_progress window 안에서 visual center correction을 강제한다.

5. collision-cluster route-offset agreement loss
   - 특정 route/lane/heading cluster에서 desired sign을 강제한다.
   - Round28에서 symmetric `abs` mask와 `desired_sign = -sign(lane_offset)`로 개선됐다.

6. early-route offset agreement loss
   - Round29에서 추가.
   - early route window에서 lane/heading 조건이 맞을 때 RGB-vs-zero steer delta의 방향을 강제한다.

핵심 구현:

- `rcs_ac_first/scripts/train_bc.py:517-587`: visual center corrective loss
- `rcs_ac_first/scripts/train_bc.py:603-680`: route-aware visual center loss
- `rcs_ac_first/scripts/train_bc.py:715-804`: collision cluster route-offset loss
- `rcs_ac_first/scripts/train_bc.py:807-838`: early route offset loss
- `rcs_ac_first/scripts/train_bc.py:2031-2080`: BC total loss 합산

---

## 7. 현재 대표 config: Round29 guarded config

파일: `configs/bc_frame_rgb_perception_actor_features_v2_early_route_guard_round29.yaml`

주요 모델 설정:

```yaml
rgb_shape: [3, 90, 160]
lidar_shape: [1, 64, 64]
ego_dim: 5
route_dim: 10
feature_dim: 128
fusion_dim: 128
recurrent_hidden_dim: 256
actor_hidden_dim: 256
critic_hidden_dim: 256
enable_rgb_perception_aux: true
use_rgb_perception_actor_features: true
rgb_perception_actor_feature_dim: 16
enable_reliability: true
enable_risk_context: true
enable_shapley: true
```

BC loss 설정:

```yaml
steer_loss_weight: 3.0
longitudinal_loss_weight: 0.5
perception_aux_loss_weight: 0.5
visual_zero_corrective_sign_loss_weight: 0.30
target_corrective_delta_loss_weight: 0.40
visual_zero_target_delta_loss_weight: 0.20
visual_center_corrective_sign_loss_weight: 0.20
route_aware_visual_center_loss_weight: 0.15
collision_cluster_route_offset_agreement_loss_weight: 0.60
```

guard/smoothing 설정:

```yaml
action_smoothing:
  steer_alpha: 0.85
  throttle_rise_limit: 0.06
  throttle_fall_limit: 0.18
  brake_rise_limit: 0.08
  brake_fall_limit: 0.18

lateral_recovery:
  enabled: true
  lane_offset_threshold_m: 0.55
  heading_error_threshold_deg: 8.0
  lane_gain: -0.080
  heading_gain: 0.006
  max_abs_correction: 0.22
  speed_gate_enabled: true
  speed_gate_min_speed_kmh: 16.0
  speed_gate_target_speed_kmh: 10.0
  speed_gate_throttle_cap: 0.06
  speed_gate_max_brake: 0.20
```

주의:

- 이 guard는 policy output 이후 action을 보정하는 runtime assist다.
- 순수 RCS-AC policy 성능과 guard 포함 stack 성능을 분리해서 평가해야 한다.

---

## 8. 안정화 라운드 요약

### Round 17

- drift-window visual steer BC gate.
- BC val은 좋아졌지만 steering-relevant RGB causality가 약했다.
- 512 eval은 Round16보다 개선됐으나 PPO는 blocked.
- early offroad before route 0.20은 줄었지만 final offroad는 남음.

### Round 19

- corrective delta와 visual center 관련 loss 도입.
- `visual_zero_steer_delta`가 약간 개선됐다.
- 하지만 `visual_zero_signed_steer_delta` 방향성이 음수로 남아 RGB steering direction 문제가 지속됐다.

### Round 23

- route-aware visual center가 강해져 512 eval에서 route progress가 크게 개선됐다.
- 그러나 2048 eval에서 offroad/collision regression이 드러났다.

### Round 24

시작점:

- Round23 512 eval은 좋았지만 2048 eval에서 offroad 6, early offroad 1, collision 1이 발생.
- visual_center_correction_rate는 1.0인데 route_offset agreement는 낮았다.

개입:

- route-offset agreement safety patch.
- BC train 결과:
  - val loss 0.005694
  - visual_zero_steer_delta 0.020045
  - target_corrective_delta_sign_rate 0.946128
  - visual_center_correction_rate 1.0

결과:

- 512 eval: offroad 0, collision 0, max route 0.769231
- 2048 eval retry: PPO gate PASS였지만 collision 5, offroad 2가 있었다.

해석:

- short horizon은 개선됐지만 long horizon safety는 아직 불완전.

### Round 27

목표:

- collision cluster around route 0.16-0.20, lane_offset > +1.5, heading_error < -8를 겨냥.

문제:

- 기존 데이터셋에 exact cluster row가 없었다.
- near-cluster fallback을 썼고, sign rate는 0.61-0.64 수준.

결과:

- collision은 줄었지만 512 eval이 early offroad 7회로 무너졌다.
- terminal offroad:
  - route_progress 0.076923
  - lane_offset 약 -3m
  - heading_error +21~27deg
  - collision false

해석:

- near-cluster loss가 너무 넓거나 early route behavior를 망가뜨렸다.
- 혹은 early route failure가 training window 밖이라 loss가 닿지 않았다.

### Round 28

목표:

- Round27 loss가 positive-lane-only였던 문제를 확인/수정.

개입:

- collision_cluster loss에 `abs` lane/heading mode 추가.
- desired sign을 `-sign(lane_offset)`로 변경.
- side split metrics 추가.

BC 결과:

- overall sign rate 0.860795
- negative-lane sign rate 1.0
- positive-lane sign rate 0.199643

512 eval:

- collision 0
- offroad 7
- early/severe early offroad 7
- max route 0.076923

해석:

- symmetric loss는 코드/design issue를 일부 고쳤지만, early offroad는 그대로였다.
- 실패 지점이 route 0.10 이전이라 near-cluster loss가 실제 실패에 닿지 않는다.

### Round 29

목표:

- early route recovery data와 early-route signed correction loss 추가.

데이터 수집:

- Round24 policy collection:
  - 256 rows
  - route 0.025641-0.076923
  - severe drift 부족, lane min 약 -0.64m
- Round28 failure-policy collection:
  - 256 rows
  - broad early failure coverage:
    - lane < -0.55 and heading > 3: 52 rows, right-delta sign rate 1.0
    - lane > 0.55 and heading < -3: 45 rows, left-delta sign rate 1.0
- combined dataset: 4256 rows

BC 결과:

- val_loss 0.011058
- visual_zero_steer_delta 0.021650
- target_corrective_delta_sign_rate 0.911080
- early_route_offset_agreement_mask_rate 0.018779
- early_route_offset_agreement_negative_sign_rate 0.098592

raw 512 eval:

- collision 0
- offroad 7
- early/severe early offroad 7
- max_route_progress 0.076923

guard 512 eval:

- collision 0
- offroad 0
- max_route_progress 0.564103
- mean_action_jitter 0.073864
- PPO gate BLOCK only because jitter slightly high

guard + stronger smoothing 512 eval:

- early offroad 0
- severe early offroad 0
- mean_action_jitter 0.066269
- PPO gate PASS
- 하지만 collision 1회 발생 at route 0.435897

PPO 2048:

- offroad episodes 7
- collision episodes 1
- best route_progress 0.564
- typical offroad route_progress 0.103-0.179

해석:

- guard+smoothing은 early offroad를 막는다.
- raw policy는 여전히 early offroad.
- PPO는 BC 안정성을 악화시킨다.

---

## 9. 현재 가장 중요한 증상: 매 episode 시작 직후 왼쪽 커브

관찰:

- Round27/28/29 raw eval에서 terminal offroad가 반복적으로 route_progress 0.076923 부근에 집중.
- terminal state는 lane_offset 약 -3m, heading_error +20~30deg.
- 즉 차량이 초반에 한쪽으로 크게 밀려난다.
- 사용자의 체감: “매판 시작하자마자 왼쪽으로 커브한다.”

코드/데이터 관점의 가능한 원인:

### 원인 A: 초기 route 또는 maneuver command가 실제 도로와 비대칭

config에는 다음이 있다.

```yaml
route_representation: target_xy_plus_two_distance_heading_plus_third_heading_plus_turn_delta_plus_maneuver_hint_v1
maneuver_command_enabled: true
maneuver_command_mode: cycle
maneuver_command_values: ["straight", "left", "right"]
```

episode마다 maneuver command가 cycle이면, 초기 route에서 left command가 과대표현되거나 route heading mismatch가 있으면 매판 유사한 early turn bias가 생길 수 있다.

확인 필요:

- episode 시작 시 route target heading이 실제 lane heading과 일치하는가?
- 첫 30 step의 route vector, maneuver command, heading_error sign이 항상 같은가?
- left/right/straight command별 초기 steer mean 분포는 어떤가?

### 원인 B: steer sign convention mismatch

현재 loss들은 `desired_sign = -sign(lane_offset)`을 사용한다.

이 가정은 다음을 뜻한다.

- lane_offset > 0이면 오른쪽으로 치우쳤다고 보고 왼쪽 steer가 필요
- lane_offset < 0이면 왼쪽으로 치우쳤다고 보고 오른쪽 steer가 필요

만약 CARLA env의 lane_offset sign 또는 steering sign이 이와 다르면, correction loss가 반대로 걸릴 수 있다.

하지만 Round29 terminal CSV에서는 lane_offset이 -2.8m일 때 steer가 +0.43까지 가며 `route_offset_corrective_sign=True`였다. 이것은 완전한 sign bug라기보다 “보정 방향은 맞지만 늦고 약함” 가능성을 높인다.

그래도 확인할 것:

- positive steer가 실제 월드 좌표에서 오른쪽 회전인가?
- lane_offset < 0이 실제 차량이 왼쪽에 있음을 뜻하는가?
- heading_error > 0일 때 필요한 steering sign은 무엇인가?

### 원인 C: BC 데이터 초기 구간에 severe early recovery row가 부족

Round29에서 확인:

- Round24 policy로 수집한 early route data는 lane min이 약 -0.64m뿐이었다.
- raw failure는 lane -3m까지 간다.
- 즉 학습 데이터는 “이미 크게 벗어난 초기 상태”를 충분히 포함하지 않는다.

이 때문에 raw policy는 보정 sign을 배웠더라도, 실제 failure distribution에서 충분히 빠르고 강한 action을 내지 못한다.

### 원인 D: RGB correction이 action에 늦게 반영됨

Round29 terminal CSV:

- lane_offset이 -1.85 -> -2.86까지 커지는 동안 steer가 -0.15에서 +0.43까지 올라간다.
- 즉 처음에는 반대 방향 또는 약한 보정이었다가 나중에야 큰 오른쪽 steer가 나온다.
- 차량 속도는 약 17km/h 수준이라 lateral drift가 이미 커진 뒤다.
- 급브레이크가 늦게 들어간다.

이것은 “direction은 결국 맞지만 phase lag가 크다”는 증상이다.

### 원인 E: longitudinal/speed와 lateral correction이 분리되지 않음

early failure에서 steer correction만으로는 부족하고 속도를 줄여야 한다. Round29 guard가 성공한 이유 중 하나는 speed gate다.

guard 설정:

```yaml
speed_gate_min_speed_kmh: 16.0
speed_gate_target_speed_kmh: 10.0
speed_gate_throttle_cap: 0.06
speed_gate_max_brake: 0.20
```

raw policy에는 early route speed/brake correction loss가 충분히 명시되어 있지 않다.

### 원인 F: visual_center_error가 실제 lane center error가 아님

`visual_center_error = lane_x - road_x`는 road/lane mask 통계 차이다.

문제:

- 차선 edge mask가 좌우 차선 둘 다를 포함하면 lane_x가 도로 중심이 아니라 edge density 중심이 된다.
- curve/교차로/초기 route segment에서 lane_x-road_x가 실제 lateral error와 반대로 나올 수 있다.
- road_x가 이미 한쪽으로 치우친 road mask를 따라가면 offset이 왜곡된다.

확인 필요:

- visual_center_error sign과 simulator lane_offset sign의 상관관계
- road_x/lane_x time series와 실제 RGB 이미지 overlay
- lane_center 대신 road drivable area center를 사용할지 여부

### 원인 G: fusion과 actor-facing RGB 경로가 불일치

fusion_rgb가 낮아도 actor_input_adapter로 RGB perception compact feature가 들어가므로, RGB 영향 측정이 두 갈래로 나뉜다.

이는 debugging을 어렵게 한다.

확인 필요:

- fusion_rgb가 낮은데 visual_zero_steer_delta가 큰 이유
- actor_input_adapter가 초기 hidden state와 결합될 때 특정 steer bias를 만드는지
- zero_rgb_perception_actor_features=True baseline의 steer가 매번 어느 방향인지

---

## 10. 현재 안정화가 어려운 이유

### 10.1 실패 분포가 학습 분포와 다르다

실제 failure는 route_progress 0.076923, lane -3m 수준인데, 많은 loss window는 route 0.10 이후 또는 mild offset을 본다.

### 10.2 지표가 부분적으로 false positive를 만든다

예:

- visual_center_correction_rate = 1.0이어도 offroad 발생.
- PPO gate fail reasons가 empty인데 collision 1회가 있을 수 있었다.
- visual_zero_steer_delta가 커져도 실제 rollout에서는 너무 늦게 보정될 수 있다.

### 10.3 BC는 deterministic eval에서는 좋아도 PPO training distribution에서 무너진다

Round29:

- guard+smoothing 512 eval은 PPO gate PASS
- PPO 2048은 offroad 7, collision 1

즉 PPO fine-tuning의 rollout distribution이 BC deterministic evaluation보다 훨씬 거칠고, 짧은 eval gate가 충분하지 않다.

### 10.4 lateral recovery guard가 성능을 가릴 수 있다

guard는 안정화에 효과적이다. 그러나 이것이 RCS-AC policy 자체의 학습 결과인지, hand-coded controller의 개입인지 분리해야 한다.

따라서 앞으로는 반드시 다음을 나눠 기록해야 한다.

- raw policy eval
- guard-only eval
- guard+smoothing eval
- PPO with same guard
- PPO/SAC/RCS-AC baseline에도 동일 guard 적용 여부

### 10.5 left/right side balance가 안 맞는다

Round28:

- negative-lane sign rate는 1.0
- positive-lane sign rate는 0.199643

Round29:

- early negative sign rate가 val에서 0.098592

즉 side-specific supervision이 매우 불균형하다.

---

## 11. 코드상 특히 의심할 지점

외부 GPT에게 우선 리뷰를 요청할 파일:

1. `rcs_ac_first/runtime/observation.py`
   - route vector, route heading, target xy, maneuver hint encoding sign/scale 확인

2. `rcs_ac_first/runtime/auxiliary.py`
   - lane_offset, heading_error, route_progress가 aux에 어떤 순서/scale로 들어가는지 확인

3. CARLA env wrapper 파일
   - 현재 검색 목록에는 env 파일이 명확히 보이지 않으므로, 실제 environment implementation 위치를 추가 확인해야 한다.
   - lane_offset/heading_error 계산식, steering sign, route initialization, maneuver_command cycle 구현 확인 필요

4. `rcs_ac_first/models/perception_aux.py`
   - `road_x`, `lane_x`, `visual_center_error`가 실제 lateral error와 같은 sign인지 확인

5. `rcs_ac_first/scripts/train_bc.py`
   - correction loss masks가 실제 데이터에 충분히 켜지는지
   - early-route loss가 broad/mild rows에만 걸려 severe rows에는 여전히 약한지

6. `rcs_ac_first/scripts/evaluate_bc.py`
   - PPO gate가 collision을 제대로 fail로 처리하는지
   - early/severe terminal offroad count가 gate에 반영되는지

7. `rcs_ac_first/scripts/train.py`
   - PPO rollout 중 guard/smoothing 적용 순서
   - deterministic eval과 stochastic PPO action 차이
   - hidden state reset timing

---

## 12. 해결 방향 제안

### 12.1 먼저 진짜 sign convention audit

episode 시작 후 0-80 step에 대해 다음을 저장/검증한다.

- steer action
- wheel yaw or vehicle yaw change
- lane_offset
- heading_error
- route target heading
- maneuver command
- visual_center_error
- visual_zero_signed_steer_delta
- road_x, lane_x

검증 질문:

- steer +가 실제 오른쪽 회전인가?
- lane_offset -가 실제 왼쪽 이탈인가?
- heading_error +에서 필요한 steering sign은 무엇인가?
- visual_center_error sign이 lane_offset sign과 일관적인가?

### 12.2 early route 전용 데이터 수집 강화

현재 severe early failure row가 부족하다. 다음 조건으로 데이터를 더 수집해야 한다.

```text
route_progress: 0.025-0.10
abs(lane_offset): >= 1.2, preferably >= 2.0
abs(heading_error): >= 10 deg
include both signs
store corrected_steer - policy_steer
store corrected_longitudinal - policy_longitudinal
```

특히 “terminal 직전”이 아니라 “terminal 20-40 step 전” 복귀 데이터가 필요하다.

### 12.3 steer correction과 speed correction을 분리해서 학습

현재는 lateral correction은 여러 loss가 있지만, early speed/brake correction은 guard에 더 많이 의존한다.

제안:

- `target_corrective_steer_delta`
- `target_corrective_longitudinal_delta`
- `early_route_speed_gate_loss`
- `early_route_brake_margin_loss`

route <= 0.10, abs lane/heading 조건에서 speed를 줄이는 목표를 명시한다.

### 12.4 visual lane center를 더 해석 가능하게 만들기

현재 `lane_x - road_x` 대신 다음 feature를 명시적으로 만든다.

- visual_road_center_x
- visual_lane_center_x
- visual_lane_offset
- visual_heading_error proxy
- drivable road center
- left/right lane edge confidence

그리고 simulator lane_offset/heading_error와 상관관계를 offline에서 측정한다.

### 12.5 PPO 전 gate 강화

PPO 2048 전에 다음 gate가 필요하다.

- raw BC 512
- guard BC 512
- guard BC 2048
- collision_count must be 0
- early_offroad_count 0
- severe_early_terminal_offroad_count 0
- mean_action_jitter <= 0.07
- route_offset/visual_center correction sign rate threshold
- no hidden gate loophole where fail_reasons empty but collision exists

### 12.6 PPO는 더 보수적으로

PPO가 BC를 망가뜨리는 현상이 있으므로:

- lower learning rate
- shorter rollout but stricter abort on offroad
- KL target / early stop
- freeze RGB perception and actor_input_adapter initially
- freeze actor, train critic first
- use deterministic warmup eval before stochastic PPO
- offroad/collision transition replay 또는 imitation anchor loss 유지

---

## 13. 외부 GPT에게 물어볼 핵심 질문

아래 프롬프트를 그대로 붙여넣고, 이 PDF를 첨부하면 된다.

```text
나는 CARLA 자율주행용 RCS-AC-first 알고리즘을 안정화하고 있다. 첨부 PDF에는 현재 모델 구조, 센서 사용 방식, RGB perception actor feature 구현, BC/PPO 학습 코드, 안정화 Round17-29 실험 결과가 정리되어 있다.

내가 가장 알고 싶은 것은 다음이다.

1. 매 episode 시작 직후 차량이 왼쪽으로 크게 커브하는 원인이 코드상 sign convention 문제인지, route/maneuver 초기화 문제인지, 데이터 분포 문제인지, controller/longitudinal coupling 문제인지 우선순위를 매겨 진단해줘.
2. 현재 RCS-AC 구조에서 안정화가 어려운 근본 원인을 모델 구조, loss 설계, 데이터 수집, PPO fine-tuning, runtime guard 관점에서 분석해줘.
3. 특히 다음 구현을 자세히 리뷰해줘.
   - RGBPerceptionActorFeatureExtractor의 road_x/lane_x/visual_center_error 계산
   - train_bc.py의 visual_center_corrective_loss, route_aware_visual_center_loss, collision_cluster_route_offset_loss, early_route_offset_loss
   - FusionGate에서 RGB reliability는 높지만 fusion_rgb가 낮게 나오는 구조
   - PPO가 512-step BC gate 통과 후 2048 training에서 offroad를 다시 만드는 이유
4. “보정 방향은 맞지만 너무 늦다”는 Round29 관찰을 바탕으로, steer correction과 speed/brake correction을 어떻게 분리해서 학습해야 할지 구체적인 loss와 데이터 수집 조건을 제안해줘.
5. lateral recovery guard가 성능을 가리는 hand-coded controller인지, 혹은 안전층으로 유지해도 되는지 평가 프로토콜을 제안해줘.
6. 다음 안정화 라운드에서 가장 작은 코드 변경 3개와, 각 변경의 성공/실패 gate를 제안해줘.

답변은 다음 형식으로 해줘.

A. 가장 가능성 높은 원인 Top 5
B. 코드상 즉시 점검할 파일/함수/라인 단위 포인트
C. 실험 결과 해석: Round24, 27, 28, 29가 각각 무엇을 증명했는지
D. 다음 라운드 구현안: 데이터, loss, config, eval gate
E. PPO를 재개해도 되는 조건
F. lateral recovery guard를 논문/보고서에서 공정하게 다루는 방법

중요: 막연한 조언 말고, 첨부 PDF의 수치와 구현 세부사항을 근거로 답해줘.
```

---

## 14. 요약 결론

현재 RCS-AC-first는 RGB perception과 route-aware correction을 학습하기 시작했지만, raw policy는 초기 route 구간에서 반복적으로 왼쪽 drift/offroad를 만든다. Round29에서 guard+smoothing은 early offroad를 막았지만, raw policy 자체는 여전히 실패했고 PPO 2048은 다시 offroad/collision을 만들었다.

따라서 현재 병목은 단순히 “PPO hyperparameter가 나쁨”이 아니다. 더 근본적으로:

- early-route severe recovery data 부족
- visual center feature의 sign/geometry 불확실성
- lateral correction과 speed correction의 결합 실패
- RGB influence가 늦게 행동에 반영되는 문제
- PPO rollout distribution이 deterministic BC gate보다 훨씬 위험한 문제

가 동시에 존재한다.

다음 라운드의 가장 중요한 질문은 “왜 시작 직후 왼쪽으로 휘는지”를 sign/route/data/control 차원에서 분리 검증하는 것이다. 그 다음에 early speed-brake correction loss와 severe early recovery dataset을 넣어야 한다.
