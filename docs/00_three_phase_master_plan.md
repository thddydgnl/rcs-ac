# RCS-AC-first 3단계 구현 마스터 플랜

작성일: 2026-04-26

## 0. 이 문서의 목적

이 문서는 `RCS-AC_1-12_상세계획서.pdf`를 기준으로 RCS-AC 완성본을 새 폴더 `RCS-AC-first` 안에서 구현하기 위한 실행 계획서다.

상세계획서의 핵심은 다음 문장으로 요약된다.

```text
RCS-AC는 reliability, counterfactual contribution, risk-aware weighting을 함께 사용해서
현재 믿을 수 있고 실제 가치 추정에 도움이 되며 위험 상황에서도 안전한 모달을 우선하는
멀티센서 actor-critic이다.
```

이 목표를 한 번에 구현하면 실패 원인이 섞인다. 그래서 구현은 크게 3단계로 나눈다.

- Phase 1: RCS-AC Core
- Phase 2: Reliability + Risk-aware Fusion
- Phase 3: Counterfactual Shapley Full RCS-AC

각 단계는 독립적으로 compile, dummy forward, smoke run, short train, log inspection을 통과해야 다음 단계로 넘어간다.

## 1. 상세계획서 핵심 요구사항 분해

상세계획서 1~12장의 구현 요소를 코드 단위로 풀면 다음과 같다.

| 계획서 요소 | 코드 모듈 | 구현 단계 |
| --- | --- | --- |
| RGB encoder | `models/encoders.py` | Phase 1 |
| LiDAR encoder | `models/encoders.py` | Phase 1 |
| Ego encoder | `models/encoders.py` | Phase 1 |
| Route encoder | `models/encoders.py` | Phase 1 |
| Fusion output | `models/fusion_gate.py` | Phase 1부터 skeleton, Phase 2부터 risk-aware |
| Recurrent backbone | `models/recurrent_actor_critic.py` | Phase 1 |
| Actor head | `models/recurrent_actor_critic.py` | Phase 1 |
| Critic head | `models/recurrent_actor_critic.py` | Phase 1 |
| Reliability head | `models/reliability.py` | Phase 2 |
| Risk/context head | `models/risk_context.py` | Phase 2 |
| Sensor corruption | `env/corruption.py` 또는 shared env adapter | Phase 2 |
| Reliability supervision loss | `rl/losses.py` | Phase 2 |
| Risk-aware regularization | `rl/losses.py` | Phase 2 |
| Counterfactual contribution estimator | `rl/shapley.py` | Phase 3 |
| Masked critic | `models/masked_critic.py` | Phase 3 |
| Shapley alignment loss | `rl/losses.py` | Phase 3 |
| Dropout consistency loss | `rl/losses.py` | Phase 3 |
| Exact-ish Shapley eval | `scripts/evaluate_shapley.py` | Phase 3 |

## 2. 새 폴더의 역할

`RCS-AC-first`는 다음을 목표로 한다.

- 상세계획서 기준의 완성형 RCS-AC 구현
- 기존 smoke scaffold와 독립적인 새 구조
- 다만 CARLA env, reward, observation contract는 처음부터 새로 발명하지 않고 기존 `shared_core`를 최대한 참조
- 최종적으로는 `RCS-AC-first`만 보고도 모델 구조와 실험 의도를 이해할 수 있게 구성

초기 구현에서는 기존 `attention` 학습 루프를 완전히 복사하지 않는다. 대신 필요한 부분을 adapter 형태로 가져오거나, Phase 1에서는 wrapper를 써서 학습 루프 drift를 줄인다.

## 3. 추천 폴더 구조

```text
RCS-AC-first/
  README.md
  docs/
    00_three_phase_master_plan.md
    01_phase_checklist.md
    02_interface_contract.md
    03_experiment_protocol.md
    04_risk_and_reliability_design.md
    05_shapley_design.md
  rcs_ac_first/
    __init__.py
    configs/
      phase1_core_smoke.yaml
      phase1_core_train.yaml
      phase2_reliability_risk.yaml
      phase3_full_shapley.yaml
    models/
      __init__.py
      encoders.py
      reliability.py
      risk_context.py
      fusion_gate.py
      recurrent_actor_critic.py
      masked_critic.py
      rcs_ac_model.py
    rl/
      __init__.py
      distributions.py
      losses.py
      rollout_buffer.py
      ppo_agent.py
      shapley.py
    runtime/
      __init__.py
      observation.py
      action.py
      config.py
      logging.py
      checkpoint.py
      seed.py
    scripts/
      train.py
      smoke_check.py
      evaluate.py
      evaluate_shapley.py
      summarize_logs.py
    tests/
      test_model_shapes.py
      test_fusion_gate.py
      test_losses.py
      test_shapley_estimator.py
  outputs/
  logs/
  checkpoints/
```

## 4. 공통 계약

### 4.1 입력 모달

상세계획서의 입력은 네 모달이다.

| key | 의미 | Phase 1 기본 shape | 비고 |
| --- | --- | --- | --- |
| `rgb` | 전방 RGB image | `(3, 90, 160)` | 기존 shared contract 우선 사용 |
| `lidar` | LiDAR BEV/grid | `(1, 64, 64)` | 기존 shared contract 우선 사용 |
| `ego` | 차량 상태 | `(5,)` 또는 확장 | speed, yaw rate, steer, heading error, lane offset |
| `route` | 경로 정보 | `(10,)` 기본 | target, lookahead, turn delta, maneuver hint |

Phase 2부터 auxiliary context를 추가한다.

| key | 의미 | 사용 위치 |
| --- | --- | --- |
| `quality_stats` | RGB/LiDAR 품질 통계 | reliability head |
| `risk_stats` | speed, lane offset, heading error, curve proxy 등 | risk/context head |
| `route_commitment_stats` | route progress, target distance 등 | risk/context 또는 route reliability |
| `sensor_dropout_mask` | 모달별 dropout/corruption flag | reliability label, dropout consistency |

원칙:

- policy input flatten 순서는 `rgb -> lidar -> ego -> route`로 고정한다.
- auxiliary는 Phase 1에서는 policy input에 직접 넣지 않는다.
- Phase 2부터 auxiliary는 별도 branch 입력으로 받는다.
- obs_dim은 숫자 하드코딩이 아니라 shape spec에서 계산한다.

### 4.2 출력 action

상세계획서 원문은 `steering`, `throttle`, `brake` 3개 action을 말한다.

하지만 현재 shared CARLA contract와 실험 안정성을 고려하면 내부 policy action은 다음 2차원을 우선 추천한다.

```text
action = [steer, longitudinal]
steer in [-1, 1]
longitudinal in [-1, 1]
```

env decode:

```text
throttle = max(longitudinal, 0)
brake = max(-longitudinal, 0)
```

이 선택의 이유:

- throttle/brake 동시 양수 충돌을 구조적으로 줄인다.
- 기존 stage4 action contract와 맞다.
- smoke와 first train에서 제어 실패 원인을 줄인다.

Phase 3 이후 논문식 3-action policy가 필요하면 `action_mode: steer_throttle_brake`를 별도 config로 열 수 있다.

## 5. Phase 1: RCS-AC Core

### 5.1 목표

Phase 1의 목표는 RCS-AC 완성본의 골격을 세우는 것이다.

구체적으로는 다음 질문에 답해야 한다.

- 네 모달 encoder가 정상적으로 feature를 만드는가?
- recurrent actor-critic이 PPO rollout/update에 연결되는가?
- recurrent hidden state reset, detach, batch 처리 정책이 명확한가?
- future reliability/risk/Shapley를 넣을 interface가 열려 있는가?
- CARLA smoke run에서 최소 1 update 이상 깨지지 않는가?

성능 목표는 낮게 잡는다. Phase 1에서 goal을 못 가도 실패가 아니다. 다만 차량이 완전히 제자리에서만 멈춘다면 action bias, reward, exploration std를 점검한다.

### 5.2 구현 범위

Phase 1에서 구현할 것:

- `PolicyObservationSpec`
- `RCSACFirstConfig`
- `RGBEncoder`
- `LiDAREncoder`
- `VectorEncoder` for ego/route
- modality projection `P_m(h_m)`
- temporary contribution prior
- temporary reliability prior
- simple fusion gate
- safety floor for ego/route weights
- GRU recurrent backbone
- actor head
- critic head
- scalar auxiliary heads
  - lane offset
  - heading error
- PPO-style clipped actor loss
- value loss
- entropy loss
- checkpoint save/load
- training log
- dummy tests
- smoke script

Phase 1에서 만들지만 끄는 것:

- reliability supervision loss
- risk-aware regularization
- Shapley alignment loss
- dropout consistency loss
- masked critic training

### 5.3 Phase 1 모델 흐름

```text
flat_obs
  -> split rgb/lidar/ego/route
  -> RGBEncoder -> h_rgb
  -> LiDAREncoder -> h_lidar
  -> EgoEncoder -> h_ego
  -> RouteEncoder -> h_route
  -> projection per modality -> p_rgb, p_lidar, p_ego, p_route
  -> simple fusion gate -> w_rgb, w_lidar, w_ego, w_route
  -> floor route/ego weights
  -> z_t = sum_m w_m * p_m
  -> GRU(z_t, hidden)
  -> actor head -> action distribution
  -> critic head -> V_t
```

### 5.4 Phase 1 fusion gate

Phase 1에서도 full RCS-AC와 같은 gate API를 사용한다.

```python
fusion_outputs = fusion_gate(
    modality_features=encoded,
    reliability=None,
    contribution=None,
    risk_context=None,
)
```

내부 기본값:

```text
q_m = 1.0
c_hat_m = uniform
r_t = 0.0
context = zero vector
```

이렇게 하면 Phase 2/3에서 API를 바꾸지 않고 `reliability`, `risk_context`, `contribution`만 채워 넣을 수 있다.

### 5.5 Phase 1 recurrent 설계

GRU를 우선 사용한다.

권장 설정:

```yaml
model:
  recurrent_type: gru
  recurrent_hidden_dim: 256
  recurrent_layers: 1
  recurrent_detach_interval: rollout
```

처음에는 rollout buffer가 hidden state 전체 sequence를 완벽히 다루지 못할 수 있다. 그러면 Phase 1a에서는 single-step GRU로 시작한다.

Phase 1a:

- hidden state를 episode별로 유지
- update batch에서는 hidden 없이 `z_t`만 쓰거나 zero hidden으로 학습
- smoke 안정성 우선

Phase 1b:

- rollout buffer에 initial hidden 저장
- sequence minibatch 추가
- truncated BPTT 선택적 지원

첫 구현은 Phase 1a를 추천한다. recurrent 구조를 모델에 포함시키되 학습 복잡도는 천천히 올린다.

### 5.6 Phase 1 loss

활성화:

```text
L_actor
L_value
L_entropy
L_aux_lane_offset
L_aux_heading_error
```

0으로 고정:

```text
L_rel = 0
L_risk = 0
L_shap = 0
L_cons = 0
```

### 5.7 Phase 1 성공 기준

필수:

- 모든 Python 파일 compile 통과
- dummy observation forward 통과
- action mean shape `[B, 2]`
- value shape `[B]`
- fusion weights shape `[B, 4]`
- fusion weights sum ~= 1
- ego/route floor 적용 후 음수 weight 없음
- GRU hidden shape 정상
- PPOAgent 생성 통과
- 512~2048 timestep smoke run에서 최소 1 update
- checkpoint 생성
- training log 생성
- NaN 없음

권장:

- feature L1 diagnostics가 0이 아님
- policy std가 지나치게 작지 않음
- `longitudinal_mean`이 항상 음수/양수로 포화되지 않음

## 6. Phase 2: Reliability + Risk-aware Fusion

### 6.1 목표

Phase 2는 RCS-AC가 기존 attention fusion과 달라지는 첫 핵심 단계다.

상세계획서에 따르면 RCS-AC는 “중요해 보이는 모달”이 아니라 “믿을 수 있고 위험 상황에서 안전한 모달”을 선택해야 한다.

이 단계의 목표:

- quality stats를 reliability score로 변환
- risk stats를 context embedding과 risk scalar로 변환
- sensor corruption을 명시적으로 학습에 주입
- fusion gate가 reliability/risk에 반응하도록 만들기
- clean 환경 성능을 유지하면서 weak corruption에서 무너짐을 줄이기

### 6.2 Reliability head 설계

입력:

```text
h_rgb + rgb_quality_stats + rgb_dropout_flag
h_lidar + lidar_quality_stats + lidar_dropout_flag
h_ego + ego_quality_stats + ego_dropout_flag
h_route + route_quality_stats + route_dropout_flag
```

출력:

```text
q_rgb, q_lidar, q_ego, q_route in [0, 1]
```

권장 구현:

```python
class ModalityReliabilityHead(nn.Module):
    def forward(self, feature, quality_vector, dropout_flag):
        return sigmoid(mlp(concat(feature, quality_vector, dropout_flag)))
```

처음에는 quality vector를 간단히 나눈다.

RGB:

- brightness
- contrast
- entropy
- blur proxy
- frame dropout flag
- weather visibility proxy

LiDAR:

- point count norm
- valid ratio
- density
- sparsity
- point drop flag

Ego:

- speed finite flag
- yaw rate finite flag
- jerk abnormality proxy
- state noise flag

Route:

- route availability
- target distance valid
- heading mismatch
- route corruption flag

### 6.3 Reliability pseudo-label

Phase 2에서 supervised label은 실제 sensor health ground truth가 아니라 corruption metadata 기반 pseudo-label을 쓴다.

예시:

```text
q_tilde_rgb = 1.0 - rgb_corruption_severity
q_tilde_lidar = 1.0 - lidar_drop_rate
q_tilde_ego = 1.0 - ego_noise_severity
q_tilde_route = 1.0 - route_corruption_probability
```

clamp:

```text
q_tilde_m in [0.05, 1.0]
```

완전 dropout이면 0.0이 아니라 0.05 floor를 둔다. 완전 0은 gate와 gradient가 너무 빨리 죽을 수 있다.

### 6.4 Risk/context head 설계

입력:

- speed
- lane offset
- heading error
- curve proxy
- target distance
- recent collision/near-miss proxy
- weather severity
- sensor dropout mask
- LiDAR density
- RGB visibility proxy

출력:

```text
context c_t in R^64
risk r_t in [0, 1]
```

권장 구현:

```python
context = mlp(risk_stats)
risk = sigmoid(linear(context))
```

risk scalar는 다음 상황에서 높아져야 한다.

- lane offset이 커짐
- heading error가 커짐
- curve/junction/turn delta가 큼
- speed가 높은데 heading/lane이 불안정함
- sensor dropout이 발생함
- target distance가 줄어들고 command turn이 active

### 6.5 Risk-aware fusion gate

상세계획서의 gate 식:

```text
g_m = f_g(h_m, q_m, c_hat_m, r_t, c_t)
w_m = softmax(g_m / tau_w)
```

Phase 2에서는 아직 `c_hat_m`은 contribution estimator가 없으므로 uniform 또는 detached prior를 쓴다.

권장 gate 입력:

```text
gate_input_m = concat(
    projected_modality_feature_m,
    reliability_q_m,
    contribution_prior_m,
    risk_scalar,
    risk_context
)
```

ego/route floor:

```yaml
fusion:
  route_weight_floor: 0.08
  ego_weight_floor: 0.05
  temperature: 1.0
```

floor 적용 절차:

1. raw softmax로 `w` 계산
2. `w_route = max(w_route, eps_route)`
3. `w_ego = max(w_ego, eps_ego)`
4. 전체 weight 재정규화

### 6.6 Phase 2 loss

활성화:

```text
L_actor
L_value
L_entropy
L_aux_lane_offset
L_aux_heading_error
L_rel
L_risk
```

낮게 또는 0:

```text
L_cons = 0 or small
L_shap = 0
```

Reliability loss:

```text
L_rel = mean_m mse(q_m, q_tilde_m)
```

Risk-aware regularization:

```text
L_risk = mean(r_t * sum_m w_m * (1 - q_m))
```

추천 초기 weight:

```yaml
loss:
  reliability_loss_weight: 0.10
  risk_regularization_weight: 0.05
```

### 6.7 Sensor corruption curriculum

Phase 2a:

- clean weather
- per-episode weak corruption only
- RGB dropout 0.05~0.10
- LiDAR point drop 0.05~0.15
- route corruption 0.0~0.05

Phase 2b:

- light rain/fog
- RGB blur/downscale weak
- LiDAR sparsity weak
- route corruption 0.05~0.10

Phase 2c:

- mild mixed corruption
- time-step corruption 일부 허용
- dropout consistency 준비

### 6.8 Phase 2 성공 기준

필수:

- clean smoke가 Phase 1보다 깨지지 않음
- reliability output이 `[B, 4]`
- risk scalar가 `[B]`
- fusion weights가 `[B, 4]`
- corruption 발생 시 해당 modality reliability가 내려감
- risk가 높은 상황에서 low reliability modality weight가 내려감
- `L_rel`, `L_risk`가 NaN 없이 기록됨

정성 기준:

- RGB blur/drop 상황에서 RGB weight가 줄어드는지
- LiDAR sparse/drop 상황에서 LiDAR weight가 줄어드는지
- route corruption 상황에서도 route floor 때문에 route가 완전히 0이 되지 않는지
- clean 환경에서 reliability가 모두 1에 가깝게 유지되는지

## 7. Phase 3: Counterfactual Shapley Full RCS-AC

### 7.1 목표

Phase 3은 상세계획서의 핵심 차별점인 counterfactual contribution과 Shapley-inspired fusion을 완성하는 단계다.

이 단계부터 RCS-AC는 다음 세 요소를 모두 쓴다.

```text
reliability q_m
contribution c_hat_m
risk/context r_t, c_t
```

### 7.2 Masked critic 설계

모달 집합:

```text
M = {rgb, lidar, ego, route}
```

mask:

```text
mu in {0, 1}^4
```

masked critic 입력 선택지:

Option A:

```text
masked projected features + mask embedding -> shared recurrent/trunk -> V_mask
```

Option B:

```text
current recurrent hidden u_t + mask embedding -> mask-conditioned critic head
```

초기 추천은 Option A다. 이유는 contribution이 실제 modality subset에 반응하는지 확인하기 쉽기 때문이다.

### 7.3 Counterfactual contribution estimator

상세계획서 식:

```text
phi_m ~= 1/K * sum_{S subset M\{m}} [V(S union {m}) - V(S)]
```

훈련 중 approximate 방식:

- every `shapley_every_updates`마다 계산
- batch의 `shapley_batch_fraction`만 사용
- sample당 subset `K=3~8`개 샘플링
- empty/full subset은 일부 강제로 포함
- value target은 critic bootstrap target 또는 detached value estimate를 사용

초기 설정:

```yaml
shapley:
  enabled: true
  every_updates: 4
  batch_fraction: 0.25
  subsets_per_sample: 4
  contribution_temperature: 1.0
```

### 7.4 Contribution normalization

raw contribution:

```text
phi_m
```

normalized contribution:

```text
c_hat = softmax(phi / tau_c)
```

주의:

- phi가 모두 비슷하면 uniform에 가까워야 한다.
- phi가 음수가 될 수 있으므로 softmax 전 clipping 또는 normalization을 검토한다.
- contribution은 gate target이지 반드시 gate 자체가 아니다.

### 7.5 Shapley alignment loss

기본:

```text
L_shap = mean(sum_m (w_m - stopgrad(c_hat_m))^2)
```

대안:

```text
L_shap_kl = KL(stopgrad(c_hat) || w)
```

초기 추천은 MSE다. KL은 0 근처 weight에서 민감할 수 있다.

초기 weight:

```yaml
loss:
  shapley_alignment_loss_weight: 0.05
```

너무 크게 시작하면 gate가 contribution estimator 오류를 그대로 따라가므로 천천히 올린다.

### 7.6 Dropout consistency loss

상세계획서의 목적:

```text
full input과 dropout input 사이에서 정책이 과도하게 흔들리지 않게 한다.
```

구현:

1. 같은 observation batch에서 corrupted copy 생성
2. full policy distribution 계산
3. corrupted policy distribution 계산
4. action mean MSE 또는 approximate KL 계산

초기 추천:

```text
L_cons = mean(||a_full - a_drop||^2)
```

KL은 squashed distribution에서 구현이 까다로울 수 있다. 먼저 action mean consistency로 시작한다.

초기 weight:

```yaml
loss:
  dropout_consistency_loss_weight: 0.05
```

### 7.7 Full fusion gate

Phase 3의 최종 gate:

```text
g_m = f_g(P_m(h_m), q_m, c_hat_m, r_t, c_t)
w_m = softmax(g_m / tau_w)
floor route/ego
renormalize
z_t = sum_m w_m * P_m(h_m)
```

diagnostics:

- `fusion_weight_rgb`
- `fusion_weight_lidar`
- `fusion_weight_ego`
- `fusion_weight_route`
- `reliability_rgb`
- `reliability_lidar`
- `reliability_ego`
- `reliability_route`
- `contribution_rgb`
- `contribution_lidar`
- `contribution_ego`
- `contribution_route`
- `risk_scalar`
- `shapley_alignment_loss`
- `dropout_consistency_loss`

### 7.8 Phase 3 eval

평가 시에는 모달 수가 4개라 exact-ish subset 계산이 가능하다.

가능한 subset:

```text
2^4 = 16
```

평가 절차:

1. 고정 scenario 선택
2. 각 timestep마다 16개 subset value 계산
3. exact Shapley contribution 계산
4. fusion weight와 비교
5. weather/dropout별 평균 contribution plot 저장

평가 산출물:

- `contribution_timeseries.csv`
- `fusion_vs_contribution.csv`
- `scenario_summary.json`
- `modality_contribution_plot.png`
- `risk_reliability_fusion_plot.png`

### 7.9 Phase 3 성공 기준

필수:

- masked critic forward 통과
- approximate Shapley batch 계산 통과
- contribution shape `[B, 4]`
- contribution softmax sum ~= 1
- Shapley loss NaN 없음
- dropout consistency loss NaN 없음
- full training 2048+ steps 통과

성능 기준:

- clean 환경에서 Phase 2 대비 크게 하락하지 않음
- mild corruption에서 Phase 2보다 안정적
- heavy corruption에서 attention baseline 대비 degradation 감소
- contribution이 항상 uniform 또는 한 모달 독점으로 붕괴하지 않음

해석 기준:

- RGB 열화 시 RGB reliability와 contribution이 분리되어 보임
- LiDAR sparse 상황에서 LiDAR reliability가 낮아짐
- route가 실제 주행 방향 결정에 중요할 때 route contribution이 높아짐
- risk가 높은 상황에서 low reliability modality weight가 낮아짐

## 8. 단계별 config 계획

### 8.1 Phase 1 config

```yaml
experiment:
  name: rcs_ac_first_phase1_core_smoke
  seed: 7
  total_timesteps: 2048
  rollout_steps: 128

model:
  phase: phase1_core
  action_dim: 2
  fusion_dim: 256
  recurrent_type: gru
  recurrent_hidden_dim: 256
  recurrent_layers: 1
  enable_reliability: false
  enable_risk_context: false
  enable_shapley: false
  enable_dropout_consistency: false

loss:
  actor_loss_weight: 1.0
  value_loss_weight: 0.5
  entropy_loss_weight: 0.01
  aux_lane_offset_loss_weight: 1.0
  aux_heading_error_loss_weight: 1.0
  reliability_loss_weight: 0.0
  risk_regularization_weight: 0.0
  shapley_alignment_loss_weight: 0.0
  dropout_consistency_loss_weight: 0.0
```

### 8.2 Phase 2 config

```yaml
model:
  phase: phase2_reliability_risk
  enable_reliability: true
  enable_risk_context: true
  enable_shapley: false
  enable_dropout_consistency: false

fusion:
  temperature: 1.0
  route_weight_floor: 0.08
  ego_weight_floor: 0.05

loss:
  reliability_loss_weight: 0.10
  risk_regularization_weight: 0.05
  shapley_alignment_loss_weight: 0.0
  dropout_consistency_loss_weight: 0.0

corruption:
  enabled: true
  level: mild
  rgb_dropout_prob: 0.05
  lidar_drop_prob: 0.10
  ego_noise_std: 0.01
  route_corruption_prob: 0.05
```

### 8.3 Phase 3 config

```yaml
model:
  phase: phase3_full_shapley
  enable_reliability: true
  enable_risk_context: true
  enable_shapley: true
  enable_dropout_consistency: true

shapley:
  every_updates: 4
  batch_fraction: 0.25
  subsets_per_sample: 4
  contribution_temperature: 1.0
  include_empty_and_full_subsets: true

loss:
  reliability_loss_weight: 0.10
  risk_regularization_weight: 0.05
  shapley_alignment_loss_weight: 0.05
  dropout_consistency_loss_weight: 0.05
```

## 9. 로그 설계

training log에는 적어도 다음 필드가 있어야 한다.

기본 학습:

- `actor_loss`
- `critic_loss`
- `entropy`
- `approx_kl`
- `clipfrac`
- `explained_variance`
- `policy_std_mean`
- `policy_std_dim0`
- `policy_std_dim1`

주행:

- `episode_reward`
- `route_progress`
- `done_reason`
- `goal_count`
- `collision_rate`
- `offroad_rate`
- `stuck_rate`
- `average_speed_kmh`

모달 feature:

- `rgb_feature_l1_mean`
- `lidar_feature_l1_mean`
- `ego_feature_l1_mean`
- `route_feature_l1_mean`
- `fused_feature_l1_mean`

Fusion:

- `fusion_weight_rgb`
- `fusion_weight_lidar`
- `fusion_weight_ego`
- `fusion_weight_route`
- `fusion_entropy`
- `fusion_route_floor_applied_rate`
- `fusion_ego_floor_applied_rate`

Reliability:

- `reliability_rgb`
- `reliability_lidar`
- `reliability_ego`
- `reliability_route`
- `reliability_loss`

Risk:

- `risk_scalar_mean`
- `risk_scalar_max`
- `risk_regularization_loss`

Shapley:

- `contribution_rgb`
- `contribution_lidar`
- `contribution_ego`
- `contribution_route`
- `shapley_alignment_loss`
- `masked_critic_loss`
- `shapley_sample_count`

Consistency:

- `dropout_consistency_loss`
- `full_drop_action_l1`
- `full_drop_value_l1`

## 10. 테스트 계획

### 10.1 Unit tests

`test_model_shapes.py`

- dummy flat observation split
- encoder output shape
- fusion weight shape
- actor output shape
- critic output shape
- recurrent hidden shape

`test_fusion_gate.py`

- weight sum equals 1
- route/ego floor 적용
- temperature 변화 확인
- low reliability modality weight 감소 확인

`test_losses.py`

- PPO loss finite
- reliability loss finite
- risk regularization finite
- dropout consistency finite
- Shapley alignment finite

`test_shapley_estimator.py`

- subset mask 생성
- contribution shape
- contribution normalization
- exact subset eval sanity

### 10.2 Smoke checks

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py
```

확인:

- import success
- config load
- model build
- dummy forward
- dummy loss backward
- checkpoint save/load

### 10.3 CARLA smoke run

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\train.py --config .\RCS-AC-first\rcs_ac_first\configs\phase1_core_smoke.yaml
```

성공 기준:

- reset success
- first observation shape 정상
- rollout buffer add 정상
- minimum one update
- checkpoint 저장
- log 저장

## 11. 구현 순서

### 11.1 Phase 1 구현 순서

1. `RCS-AC-first/rcs_ac_first` package scaffold 생성
2. config dataclass 생성
3. observation spec 생성
4. action spec 생성
5. encoder 4종 구현
6. projection module 구현
7. fusion gate skeleton 구현
8. GRU actor-critic 구현
9. top-level `RCSACFirstModel` 구현
10. losses 기본 구현
11. PPOAgent 최소 구현 또는 existing PPO adapter 작성
12. dummy smoke check 작성
13. compile/test
14. CARLA smoke

### 11.2 Phase 2 구현 순서

1. auxiliary observation adapter 작성
2. quality/risk stats schema 정리
3. reliability head 구현
4. risk/context head 구현
5. corruption metadata schema 구현
6. reliability pseudo-label 생성
7. `L_rel` 구현
8. `L_risk` 구현
9. fusion gate에 q/risk/context 연결
10. log field 확장
11. mild corruption smoke
12. clean vs mild 비교

### 11.3 Phase 3 구현 순서

1. modality mask representation 정의
2. masked feature builder 구현
3. masked critic 구현
4. subset sampler 구현
5. approximate Shapley estimator 구현
6. contribution normalization 구현
7. gate에 contribution 연결
8. `L_shap` 구현
9. dropout consistency batch 생성
10. `L_cons` 구현
11. full train smoke
12. exact-ish eval script 구현
13. contribution report 생성

## 12. 중단 조건과 롤백 기준

각 Phase는 다음 문제가 발생하면 다음 단계로 넘어가지 않는다.

- compile 실패
- dummy forward 실패
- NaN 발생
- action shape mismatch
- fusion weight sum 오류
- checkpoint save/load 실패
- CARLA smoke에서 rollout buffer add 전 실패
- Phase 2에서 clean 성능이 Phase 1보다 크게 붕괴
- Phase 3에서 contribution이 항상 uniform 또는 단일 모달로 붕괴

문제가 생기면 먼저 아래 순서로 좁힌다.

1. observation shape
2. action contract
3. model forward
4. distribution/log_prob
5. loss finite check
6. optimizer step
7. env step
8. logging/checkpoint

## 13. 최종 목표

최종 RCS-AC-first 완성본은 다음을 만족해야 한다.

- 상세계획서의 네 모달 구조를 유지한다.
- reliability, contribution, risk를 모두 fusion에 반영한다.
- recurrent actor-critic을 사용한다.
- PPO-style clipped actor loss로 안정적으로 학습한다.
- reliability supervision, Shapley alignment, dropout consistency, risk regularization을 config로 켜고 끌 수 있다.
- 악천후와 센서 드랍에서 attention baseline 대비 성능 감소폭을 줄이는 것을 목표 지표로 삼는다.
- 평가 시 contribution map을 뽑아 해석 가능한 결과를 만든다.

