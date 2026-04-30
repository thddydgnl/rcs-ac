# RCS-AC-first Clean Stability Foundation 실행 계획

작성일: 2026-04-27

## 0. 문서 목적

이 문서는 `RCS-AC-first`에서 clean 환경 주행이 아직 충분히 안정되지 않는 원인을 reward tuning만으로 보지 않고, 아래 3가지 구조적 문제를 먼저 해결하기 위한 상세 실행 계획이다.

1. PPO std / exploration 과다
2. BC recurrent hidden mismatch
3. longitudinal throttle 포화

현재까지의 clean stabilization v1/v2/v3 실험은 reward를 바꾸면 route progress와 offroad/collision 비율이 일부 움직인다는 것을 보여줬다. 하지만 로그와 코드 확인 결과, clean instability의 가장 큰 병목은 reward 하나가 아니라 policy sampling, recurrent BC mismatch, speed/action control contract 쪽에 있다.

따라서 다음 단계의 목표는 reward를 계속 크게 바꾸는 것이 아니라, 먼저 policy가 clean에서 안정적으로 rollout될 수 있는 기반을 만드는 것이다.

## 1. 현재 관찰된 핵심 증거

### 1.1 PPO exploration이 너무 큼

현재 actor-critic의 std는 아래 구조를 가진다.

```text
log_std_init_by_dim = [-0.2, -0.2]
exp(-0.2) ~= 0.82
max_std_by_dim = [0.8, 0.8]
실제 policy_std_mean = 0.8
```

즉 PPO rollout에서 action sampling noise가 계속 크게 유지된다.

실제 로그 비교:

```text
v3 training 4096:
  steer_abs_mean ~= 0.4908
  steer_abs_p90  ~= 0.8682

v3 deterministic eval 512:
  steer_abs_mean ~= 0.0402
  steer_abs_p90  ~= 0.1139
```

같은 checkpoint라도 stochastic training rollout은 조향이 매우 크게 흔들리고, deterministic eval은 훨씬 안정적이다.

해석:

```text
현재 clean training collision/offroad 중 상당 부분은 policy mean 자체보다
큰 exploration noise 때문에 발생할 가능성이 높다.
```

### 1.2 BC recurrent hidden mismatch

BC 학습은 shuffled frame batch로 수행된다.

```text
train_bc.py:
  DataLoader(..., shuffle=True)
  model.forward_policy_outputs(..., hidden_state=None)
```

하지만 실제 평가와 PPO rollout은 GRU hidden state를 step마다 이어서 사용한다.

```text
evaluate_bc.py:
  hidden_state -> next_hidden_state -> 다음 step hidden_state
```

직접 probe 결과:

```text
BC checkpoint, hidden carry:
  longitudinal mean ~= 0.970
  speed p90 ~= 48.1 km/h

BC checkpoint, hidden reset every step:
  longitudinal mean ~= 0.649
  speed p90 ~= 32.8 km/h
```

해석:

```text
BC는 recurrent hidden을 이어 쓰는 조건으로 학습하지 않았는데,
실제 주행에서는 hidden을 이어 쓰고 있다.
그 결과 hidden accumulation이 longitudinal throttle 포화 방향으로 drift할 수 있다.
```

### 1.3 Longitudinal throttle 포화

현재 action contract:

```text
action = [steer, longitudinal]
throttle = max(longitudinal, 0)
brake = max(-longitudinal, 0)
```

Expert dataset 통계:

```text
expert longitudinal mean ~= 0.033
expert throttle mean ~= 0.331
expert brake mean ~= 0.298
expert longitudinal > 0.95 rate = 0.0
```

하지만 BC / fine-tuned eval policy:

```text
v3 deterministic eval:
  longitudinal mean ~= 0.984
  longitudinal > 0.95 rate ~= 95.9%
  speed p90 ~= 74.0 km/h
```

해석:

```text
policy가 expert의 속도 제어를 제대로 유지하지 못하고,
longitudinal action이 거의 full throttle로 포화되고 있다.
```

이 문제는 reward penalty만으로 늦게 교정하기 어렵다. action decoding 또는 target speed controller 계층이 필요하다.

## 2. 전체 해결 전략

3개 문제를 아래 순서로 해결한다.

```text
Stage S1: Low-std PPO stabilization
Stage S2: Recurrent sequence BC
Stage S3: Target-speed / action smoothing control layer
Stage S4: 통합 clean stabilization 재검증
```

우선순위는 S1 -> S2 -> S3이다.

이유:

- S1은 가장 빠르게 적용 가능하며, 현재 training rollout 사고를 즉시 줄일 수 있다.
- S2는 BC checkpoint 자체의 recurrent 실행 조건을 바로잡는다.
- S3는 throttle 포화 문제를 제어 계약 차원에서 막는다.
- reward tuning은 S1~S3 이후에 다시 수행해야 효과가 명확하다.

## 3. Stage S1: PPO std / exploration 과다 해결

### 3.1 목표

Clean fine-tuning 중 stochastic action이 deterministic policy mean에서 과도하게 벗어나지 않도록 한다.

현재 목표는 exploration을 없애는 것이 아니라, BC-initialized policy를 망가뜨리지 않는 범위에서 작게 유지하는 것이다.

### 3.2 변경 대상

주요 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\recurrent_actor_critic.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\runtime\config.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train.py
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lowstd_smoke.yaml
```

### 3.3 구현 방식

#### 3.3.1 Config에서 std를 더 낮게 설정

새 config:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lowstd_smoke.yaml
```

기준:

```yaml
model_variant: phase3_bc_clean_lowstd
training_stage: phase3_bc_clean_lowstd_smoke
output_root: outputs/phase3_bc_clean_lowstd_env

model:
  log_std_init_by_dim: [-2.0, -2.0]
  min_std_by_dim: [0.03, 0.03]
  max_std_by_dim: [0.20, 0.25]

loss:
  entropy_loss_weight: 0.0

ppo:
  learning_rate: 0.000005
  clip_coef: 0.06
  update_epochs: 1
```

예상 std:

```text
exp(-2.0) ~= 0.135
steer std ~= 0.135
longitudinal std ~= 0.135
```

초기 검증에서는 더 보수적으로 아래도 가능하다.

```yaml
max_std_by_dim: [0.15, 0.20]
```

#### 3.3.2 Init checkpoint 로드 후 std override 지원

BC checkpoint는 기존 `log_std` 파라미터를 가지고 있다. config만 바꿔도 strict load 시 checkpoint의 `log_std`가 config 초기값을 덮어쓴다.

따라서 `train.py`에 선택 옵션을 추가한다.

```text
--override-log-std -2.0 -2.0
```

또는 config 기반으로 아래 플래그를 둔다.

```yaml
ppo:
  override_loaded_log_std: true
  loaded_log_std_by_dim: [-2.0, -2.0]
```

권장 구현:

```text
init checkpoint strict load
-> model.actor_critic.log_std만 config 값으로 재설정
-> 나머지 model weight는 BC/PPO checkpoint 유지
```

이렇게 해야 BC checkpoint weight는 유지하면서 exploration만 낮출 수 있다.

#### 3.3.3 Training log에 std override 기록

`training_log.csv`에 아래 필드를 추가한다.

```text
log_std_dim0
log_std_dim1
std_override_enabled
std_override_source
```

Acceptance:

```text
policy_std_dim0 <= 0.20
policy_std_dim1 <= 0.25
entropy가 기존 2.39보다 크게 감소
```

### 3.4 S1 검증 계획

시작 checkpoint:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_stabilize_v3_4096_20260427T152736\last.pt
```

2048 smoke:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lowstd_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_stabilize_v3_4096_20260427T152736\last.pt `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name phase3_bc_clean_lowstd_2048 `
  --device auto
```

4096 smoke:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lowstd_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_stabilize_v3_4096_20260427T152736\last.pt `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name phase3_bc_clean_lowstd_4096 `
  --device auto
```

Pass 기준:

```text
PPO update 정상 수행
NaN/Inf 없음
policy_std_mean <= 0.25
training steer_abs_mean이 v3 training 0.49보다 크게 감소
collision/offroad가 v3보다 증가하지 않음
deterministic eval PASS
```

## 4. Stage S2: BC recurrent hidden mismatch 해결

### 4.1 목표

BC 학습을 실제 주행 조건과 맞춘다.

현재 BC는 shuffled single-frame supervised learning이다. GRU 모델을 실제 주행에서 hidden carry로 사용할 거라면, BC도 episode sequence 단위로 hidden을 이어서 학습해야 한다.

### 4.2 변경 대상

주요 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\data\expert_dataset.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train_bc.py
C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_sequence_clean_smoke.yaml
```

### 4.3 Dataset 변경

새 Dataset class:

```python
SequenceExpertDataset
```

반환 schema:

```text
obs:           [L, obs_dim]
auxiliary:     [L, 30]
expert_action: [L, 2]
done_mask:     [L]
valid_mask:    [L]
episode_id:    scalar
start_index:   scalar
```

sequence length 후보:

```text
smoke: 16
main: 32 또는 64
```

episode boundary 처리:

- 같은 `episode_id` 내부에서만 sequence를 구성한다.
- episode 길이가 짧으면 padding하고 `valid_mask=0`으로 둔다.
- `done=True` 이후 transition은 다음 sequence와 섞지 않는다.

Train/val split:

```text
frame 단위 split 금지
episode_id 단위 split 사용
```

이유:

```text
같은 episode frame이 train/val에 섞이면 recurrent generalization을 과대평가한다.
```

### 4.4 Model forward 변경 옵션

현재 `RecurrentActorCritic.forward()`는 `[B, D]` fused feature만 받고 내부에서 sequence length 1로 만든다.

S2에서는 2가지 방법 중 하나를 선택한다.

권장 A안:

```text
RCSACFirstModel에 forward_sequence(...) 추가
```

입력:

```text
obs_sequence:       [B, L, obs_dim]
auxiliary_sequence: [B, L, 30]
hidden_state:       [layers, B, H] 또는 None
```

출력:

```text
action_mean: [B, L, 2]
value:       [B, L]
next_hidden_state
```

내부 처리:

```text
각 timestep obs를 encoder/fusion에 통과
fused_sequence: [B, L, fusion_dim]
GRU에 fused_sequence를 그대로 입력
actor/critic head를 모든 timestep recurrent feature에 적용
```

권장 B안:

```text
train_bc.py에서 sequence를 loop로 돌며 hidden carry
```

장점:

- 구현이 빠르다.
- 기존 model API를 거의 바꾸지 않는다.

단점:

- 학습 속도가 느리다.
- BPTT가 깔끔하지 않다.

초기 smoke는 B안으로 시작하고, 안정화 후 A안으로 정리해도 된다.

### 4.5 Sequence BC loss

기본 loss:

```text
bc_loss = valid_mask 적용 MSE(action_mean, expert_action)
```

권장 weighted loss:

```text
steer_loss = MSE(action_mean[..., 0], expert_action[..., 0])
long_loss  = MSE(action_mean[..., 1], expert_action[..., 1])

total = 1.0 * steer_loss + 1.0 * long_loss
```

추가 안정화 loss:

```text
action_smooth_loss = L1(action_mean[t] - action_mean[t-1])
```

초기 권장:

```yaml
bc:
  sequence_length: 32
  action_smooth_loss_weight: 0.02
```

주의:

```text
expert action이 실제로 급격히 바뀌는 상황까지 너무 매끈하게 만들면 반응성이 떨어질 수 있으므로,
action_smooth_loss는 작게 시작한다.
```

### 4.6 Hidden mismatch 검증

S2 통과 조건은 단순 val loss가 아니라, 아래 probe가 개선되는지 확인한다.

Probe:

```text
same checkpoint, same CARLA route

hidden carry:
  longitudinal mean
  speed p90
  max route progress
  offroad count

hidden reset each step:
  longitudinal mean
  speed p90
  max route progress
  offroad count
```

Pass 기준:

```text
hidden carry와 hidden reset의 longitudinal mean 차이가 크게 줄어야 함
hidden carry가 full throttle로 drift하지 않아야 함
```

초기 목표:

```text
abs(longitudinal_mean_carry - longitudinal_mean_reset) <= 0.20
hidden carry longitudinal mean <= 0.75
speed p90 <= 45 km/h
```

## 5. Stage S3: Longitudinal throttle 포화 해결

### 5.1 목표

Policy가 `longitudinal ~= 1.0`을 내더라도 env가 무조건 full throttle로 가속하지 않도록 한다.

이 단계는 최종 자율주행 성능을 속임수로 만드는 것이 아니라, 실제 차량 제어처럼 target speed와 smoothing을 둬서 제어 가능한 action space를 만든다.

### 5.2 변경 대상

주요 파일:

```text
C:\Users\thddy\Documents\code\shared_core\env\carla_env.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\runtime\action.py
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_control_smoke.yaml
```

### 5.3 권장 action 해석 변경

현재:

```text
longitudinal > 0 -> throttle
longitudinal < 0 -> brake
```

권장:

```text
longitudinal -> target_speed_kmh
low=-1 -> target_speed 0 km/h 또는 brake
high=1 -> target_speed max_target_speed_kmh
```

예시:

```text
target_speed_kmh = ((longitudinal + 1) / 2) * max_target_speed_kmh
```

초기 max target speed:

```text
clean smoke: 35 km/h
main clean: 45 km/h
```

그 다음 low-level speed controller:

```text
speed_error = target_speed_kmh - current_speed_kmh

if speed_error > deadband:
    throttle = kp * speed_error
    brake = 0
elif speed_error < -deadband:
    throttle = 0
    brake = kb * abs(speed_error)
else:
    throttle = small_hold_throttle or 0
    brake = 0
```

Clamp:

```text
throttle <= max_throttle
brake <= max_brake
```

초기값:

```yaml
env:
  longitudinal_control:
    mode: target_speed
    max_target_speed_kmh: 35.0
    speed_deadband_kmh: 1.5
    throttle_kp: 0.08
    brake_kp: 0.06
    max_throttle: 0.55
    max_brake: 0.45
```

### 5.4 Action smoothing

제어 적용 전 smoothing:

```text
applied_steer[t] = alpha * requested_steer[t] + (1-alpha) * applied_steer[t-1]
applied_throttle[t] = min(requested_throttle, prev_throttle + throttle_rise_limit)
applied_brake[t] = min(requested_brake, prev_brake + brake_rise_limit)
```

초기값:

```yaml
env:
  action_smoothing:
    enabled: true
    steer_alpha: 0.45
    throttle_rise_limit: 0.08
    throttle_fall_limit: 0.20
    brake_rise_limit: 0.12
    brake_fall_limit: 0.25
```

주의:

```text
steer smoothing이 너무 크면 급커브 대응이 늦어진다.
처음에는 steering보다 longitudinal smoothing을 우선한다.
```

### 5.5 Logging 추가

env step log에 아래 필드를 추가한다.

```text
longitudinal_control_mode
target_speed_kmh
speed_error_kmh
raw_requested_throttle
raw_requested_brake
smoothed_throttle
smoothed_brake
smoothed_steer
throttle_cap_applied
brake_cap_applied
```

Pass 기준:

```text
policy_longitudinal이 높아도 speed_p90이 target speed 근처로 제한됨
throttle_mean이 0.9대에서 크게 하락
brake가 필요할 때 실제로 발생
collision/offroad 감소 또는 유지
route_progress가 0으로 붕괴하지 않음
```

## 6. Stage S4: 통합 clean stabilization 재검증

### 6.1 통합 config

새 config:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_foundation_smoke.yaml
```

포함:

```yaml
model:
  log_std_init_by_dim: [-2.0, -2.0]
  max_std_by_dim: [0.20, 0.25]

ppo:
  learning_rate: 0.000005
  clip_coef: 0.06
  update_epochs: 1

loss:
  entropy_loss_weight: 0.0

env:
  longitudinal_control:
    mode: target_speed
    max_target_speed_kmh: 35.0
  action_smoothing:
    enabled: true
```

### 6.2 실행 순서

1. S1만 적용한 low-std run
2. S2 sequence BC checkpoint 생성
3. sequence BC checkpoint 단독 eval
4. S2 checkpoint에서 low-std PPO fine-tuning
5. S3 target-speed/smoothing 적용 후 clean PPO fine-tuning
6. 최종 deterministic clean eval

### 6.3 비교 기준

현재 기준선:

```text
v3 deterministic clean eval 512:
  mean_route_progress ~= 0.114
  max_route_progress ~= 0.333
  collision_rate = 0.0
  offroad_rate ~= 0.0117
  speed_p90 ~= 74.0 km/h
```

S4 목표:

```text
speed_p90 <= 45 km/h
collision_rate <= v3
offroad_rate <= v3
mean_route_progress >= v3 또는 유지
action_jitter <= v3와 유사
```

학습 smoke 목표:

```text
training steer_abs_mean이 0.49에서 명확히 감소
policy_std_mean <= 0.25
collision/offroad가 v3보다 증가하지 않음
PPO update crash 없음
```

## 7. 구현 순서 상세 체크리스트

### S1 체크리스트

- [ ] `runtime/config.py`에서 log std config가 checkpoint load 이후 override될 수 있는지 확인
- [ ] `train.py`에 `--override-log-std` 또는 config 기반 override 추가
- [ ] training log에 `log_std_dim0`, `log_std_dim1`, `std_override_enabled` 추가
- [ ] `phase3_bc_clean_lowstd_smoke.yaml` 생성
- [ ] 2048 smoke 실행
- [ ] 4096 smoke 실행
- [ ] deterministic eval 512 실행
- [ ] v3와 action/std/collision/offroad 비교

### S2 체크리스트

- [ ] `SequenceExpertDataset` 추가
- [ ] episode 단위 train/val split 구현
- [ ] sequence padding + valid mask 구현
- [ ] `train_bc.py`에 `--sequence` 또는 config `bc.mode: sequence` 추가
- [ ] hidden carry BC 학습 구현
- [ ] optional action smooth loss 추가
- [ ] `bc_sequence_clean_smoke.yaml` 생성
- [ ] sequence BC smoke 학습
- [ ] hidden carry vs reset probe 실행
- [ ] sequence BC checkpoint deterministic eval 실행

### S3 체크리스트

- [ ] `carla_env.py`에 longitudinal control mode 추가
- [ ] target speed controller 구현
- [ ] action smoothing 구현
- [ ] env log 필드 추가
- [ ] `phase3_bc_clean_control_smoke.yaml` 생성
- [ ] deterministic eval에서 speed cap 작동 확인
- [ ] PPO smoke에서 route progress 붕괴 여부 확인

### S4 체크리스트

- [ ] S1/S2/S3 통합 config 생성
- [ ] 2048 smoke
- [ ] 4096 smoke
- [ ] deterministic eval 512
- [ ] 2048 또는 4096 extended eval
- [ ] clean 안정화 결과 문서 작성

## 8. 실패 시 분기

### 8.1 Low std 후 route progress가 떨어지는 경우

가능 원인:

```text
exploration이 줄면서 policy가 local behavior에 갇힘
BC checkpoint 자체가 충분히 route-following을 못 함
```

대응:

```text
std를 [0.20, 0.25]에서 [0.25, 0.30]으로 소폭 증가
learning rate는 유지
entropy는 여전히 0 또는 매우 작게 유지
```

### 8.2 Sequence BC 후 throttle drift가 계속되는 경우

가능 원인:

```text
dataset 2000 transitions가 너무 작음
expert autopilot data가 route command와 충분히 맞지 않음
longitudinal loss가 steer loss에 비해 부족함
```

대응:

```yaml
bc:
  longitudinal_loss_weight: 2.0
  include_bad_transitions: false
```

또는 dataset을 확장한다.

```text
clean_20000.npz
```

### 8.3 Target-speed controller 후 너무 느려지는 경우

가능 원인:

```text
max_target_speed_kmh가 너무 낮음
throttle_kp가 너무 낮음
brake가 과하게 개입
```

대응:

```yaml
max_target_speed_kmh: 40.0
throttle_kp: 0.10
brake_kp: 0.04
max_throttle: 0.65
```

### 8.4 Collision은 줄고 offroad가 늘어나는 경우

가능 원인:

```text
속도는 줄었지만 steering response가 늦음
steer smoothing이 과함
route/heading correction이 부족함
```

대응:

```yaml
steer_alpha: 0.65
heading_error_penalty_weight: +10~20%
heading_recovery_reward_weight: +10~20%
```

## 9. 최종 권장 실행 순서

바로 다음 구현은 S1만 먼저 한다.

```text
1. log_std override 구현
2. lowstd config 생성
3. v3 checkpoint에서 2048/4096 smoke
4. deterministic clean eval
```

S1이 통과하면 S2로 넘어간다.

```text
5. SequenceExpertDataset 구현
6. sequence BC smoke
7. hidden carry drift probe
```

S2 후에도 throttle 포화가 남으면 S3를 구현한다.

```text
8. target speed controller
9. action smoothing
10. 통합 clean foundation run
```

## 10. 판단 기준

이 문서의 핵심 판단은 아래와 같다.

```text
reward tuning은 필요하지만, 현재 clean instability의 1순위 원인은 아니다.
먼저 exploration std, recurrent BC mismatch, throttle 포화 문제를 줄여야 한다.
그 후 reward를 다시 조정해야 clean 안정화 실험의 신호가 훨씬 깨끗해진다.
```

이 순서로 진행하면 이후 heavy corruption / weather robustness 비교에서도 더 공정한 구조가 된다. Clean에서 정책이 불안정하면 corruption에서 RCS-AC의 강건성 효과와 단순 주행 실패가 섞여 논문용 해석이 어려워지기 때문이다.

## 11. 2026-04-27 실행 결과

### 11.1 구현 완료 항목

S1, S2, S3의 1차 구현을 완료했다.

구현/수정 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\rcs_ac_model.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\data\expert_dataset.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train_bc.py
C:\Users\thddy\Documents\code\shared_core\env\carla_env.py
C:\Users\thddy\Documents\code\shared_core\env\log_schema.py
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_lowstd_smoke.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_sequence_clean_smoke.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_control_smoke.yaml
```

### 11.2 S1 Low-std PPO 결과

시작 checkpoint:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\phase3_bc_clean_stabilize_v3_4096_20260427T152736\last.pt
```

결과:

```text
2048 smoke:
  run: phase3_bc_clean_lowstd_2048_20260427T163120
  updates: 16
  final policy_std_mean: 약 0.135
  entropy: 약 -1.162

4096 smoke:
  run: phase3_bc_clean_lowstd_4096_20260427T163602
  updates: 32
  final policy_std_mean: 약 0.135
  entropy: 약 -1.162
```

v3 대비 training action:

```text
v3 training steer_abs_mean:      약 0.491
lowstd training steer_abs_mean:  약 0.113
```

판단:

```text
S1은 성공. PPO exploration 과다 문제는 줄었다.
하지만 longitudinal throttle 포화는 여전히 남았다.
```

### 11.3 S2 Sequence BC 결과

생성 checkpoint:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_sequence_clean_smoke\last.pt
```

Sequence BC 학습:

```text
dataset: clean_smoke_2000.npz
sequence_length: 32
sequence_stride: 8
epochs: 8
final val_loss: 약 0.0205
```

Hidden carry probe:

```text
기존 frame BC, hidden carry:
  longitudinal mean ~= 0.970

sequence BC, hidden carry:
  longitudinal mean ~= 0.368
  speed p90 ~= 18.3 km/h
  128 step collision/offroad 없음
```

512-step eval:

```text
run: bc_sequence_clean_eval_512_20260427T165831
collision_rate: 0.0
offroad_rate: 0.0
stuck_rate: 약 0.0039
mean_route_progress: 약 0.182
max_route_progress: 약 0.333
```

판단:

```text
S2는 성공. recurrent hidden mismatch와 full-throttle drift가 크게 줄었다.
대신 속도가 낮아져 stuck 문제가 일부 생겼다.
```

### 11.4 S3 Target-speed / smoothing 결과

구현:

```text
env.longitudinal_control.mode = target_speed
env.action_smoothing.enabled = true
```

Sequence BC checkpoint + control eval:

```text
run: bc_sequence_clean_control_eval_512_20260427T170249
collision_rate: 0.0
offroad_rate: 약 0.0039
stuck_rate: 0.0
mean_route_progress: 약 0.176
max_route_progress: 약 0.513
speed_p90: 약 18.3 km/h
```

env log 기준:

```text
target_speed_mean: 약 23.8 km/h
speed_p90: 약 18.3 km/h
throttle_mean: 약 0.326
brake_mean: 약 0.004
throttle_cap_rate: 약 0.416
```

판단:

```text
S3는 1차 성공. speed cap과 throttle control이 작동했고 stuck은 사라졌다.
다만 max speed가 낮아 clean 주행이 아직 보수적이다.
```

### 11.5 S4 통합 PPO smoke 결과

시작 checkpoint:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_sequence_clean_smoke\last.pt
```

2048 PPO smoke:

```text
run: phase3_bc_clean_control_2048_20260427T170423
updates: 16
collision episode: 0
주요 done_reason: offroad, stuck
```

4096 PPO smoke:

```text
run: phase3_bc_clean_control_4096_20260427T170917
updates: 32
training done_reason:
  stuck: 9
  offroad: 19
  collision: 0
```

4096 checkpoint deterministic eval:

```text
run: phase3_bc_clean_control_eval_512_20260427T171817
collision_rate: 약 0.002
offroad_rate: 약 0.002
stuck_rate: 0.0
mean_route_progress: 약 0.208
max_route_progress: 약 0.564
speed_p90: 약 18.4 km/h
```

판단:

```text
통합 방향은 유효하다.
v3 대비 clean eval route progress는 개선됐고, speed 폭주는 제거됐다.
다만 policy가 아직 느리고, PPO 후 eval에서 collision 1회가 발생했다.
다음 튜닝은 reward보다 target_speed 상향과 stuck/offroad 균형 조정이 우선이다.
```

### 11.6 다음 권장 단계

다음 config는 `control_v2`로 둔다.

권장 변경:

```yaml
env:
  longitudinal_control:
    max_target_speed_kmh: 40.0
    throttle_kp: 0.10
    brake_kp: 0.04
    max_throttle: 0.65
    max_brake: 0.35
  action_smoothing:
    steer_alpha: 0.60
    throttle_rise_limit: 0.10
    throttle_fall_limit: 0.25

reward:
  stuck_penalty: 5.0
  heading_recovery_reward_weight: 0.014
```

목표:

```text
speed_p90: 22~30 km/h
stuck episode 감소
collision 0 유지
offroad 감소 또는 유지
mean_route_progress >= 0.20 유지
```
## 12. Control v2 Detailed Result - 2026-04-27

### 12.1 Source Log Diagnosis

Analyzed run:

```text
logs/phase3_bc_clean_control_4096_20260427T170917/training_log.csv
outputs/phase3_bc_clean_control_env/phase3_bc_clean_control_4096_20260427T170917/env_step_log.csv
```

Key findings:

```text
episodes done: 28
done_reason:
  stuck: 9
  offroad: 19
  collision: 0

speed_mean: 11.84 km/h
speed_p90: 18.31 km/h
target_speed_mean: 20.36 km/h
policy_longitudinal_p10: -0.9965
low_speed_steps_max: 60

pre-done lane_abs_mean: 1.91 m
pre-done heading_abs_mean: 13.83 deg
```

Interpretation:

```text
1. The old control config still allowed policy longitudinal output to fall near -1.0.
   In target-speed mode that could map to almost zero desired speed, causing stuck episodes.

2. Offroad was not primarily caused by uncontrolled high-speed driving.
   It appeared when lane offset and heading error accumulated near episode termination.

3. Therefore control_v2 needed two changes:
   - add a minimum target-speed floor to prevent zero-speed collapse
   - keep max target speed conservative enough that heading/lane recovery remains possible
```

### 12.2 Code And Config Changes

Implemented:

```text
shared_core/env/carla_env.py
  - longitudinal_control.min_target_speed_kmh support
  - target_speed_kmh = min_speed + normalized_longitudinal * (max_speed - min_speed)

configs/phase3_bc_clean_control_v2_smoke.yaml
  - target-speed floor and lower max speed
  - conservative throttle gain/cap
  - slightly more responsive steering smoothing
  - stronger curve/lane speed penalties
```

Final control_v2b values used for the accepted runs:

```yaml
longitudinal_control:
  mode: target_speed
  min_target_speed_kmh: 6.0
  max_target_speed_kmh: 32.0
  speed_deadband_kmh: 1.5
  throttle_kp: 0.075
  brake_kp: 0.05
  max_throttle: 0.48
  max_brake: 0.40

action_smoothing:
  steer_alpha: 0.75
  throttle_rise_limit: 0.08
  throttle_fall_limit: 0.25
  brake_rise_limit: 0.10
  brake_fall_limit: 0.25
```

### 12.3 Control v2 Runs

Pre-PPO deterministic sequence BC evaluation:

```text
run: bc_sequence_clean_control_v2b_eval_512_20260427T173804
checkpoint: checkpoints/bc_sequence_clean_smoke/last.pt
status: PASS
collision_rate: 0.00195
offroad_rate: 0.00195
stuck_rate: 0.0
mean_route_progress: 0.1975
max_route_progress: 0.5641
```

2048 PPO smoke:

```text
run: phase3_bc_clean_control_v2b_2048_20260427T173915
updates: 16
done_reason:
  offroad: 17
  stuck: 0
  collision: 0
route_max: 0.3077
speed_mean: 13.74 km/h
speed_p90: 18.24 km/h
target_speed_mean: 24.29 km/h
pre-done lane_abs_mean: 2.45 m
pre-done heading_abs_mean: 17.65 deg
```

4096 PPO smoke:

```text
run: phase3_bc_clean_control_v2b_4096_20260427T174353
updates: 32
checkpoint: checkpoints/phase3_bc_clean_control_v2b_4096_20260427T174353/last.pt
done_reason:
  offroad: 31
  stuck: 0
  collision: 0
route_mean: 0.0965
route_max: 0.3077
speed_mean: 13.94 km/h
speed_p90: 18.26 km/h
target_speed_mean: 23.94 km/h
target_speed_p90: 27.83 km/h
pre-done lane_abs_mean: 2.47 m
pre-done heading_abs_mean: 19.49 deg
```

4096 checkpoint deterministic eval:

```text
run: phase3_bc_clean_control_v2b_eval_512_20260427T175252
status: PASS
collision_rate: 0.0
offroad_rate: 0.00391
stuck_rate: 0.0
mean_route_progress: 0.1812
max_route_progress: 0.5128
checkpoint: checkpoints/phase3_bc_clean_control_v2b_4096_20260427T174353/last.pt
```

### 12.4 Comparison To Control v1

Training comparison:

```text
control_v1_4096:
  done: stuck 9, offroad 19, collision 0
  route_mean: 0.1014
  route_max: 0.3077
  speed_p90: 18.31
  target_speed_mean: 20.36
  pre-done lane_abs_mean: 1.91
  pre-done heading_abs_mean: 13.83

control_v2b_4096:
  done: stuck 0, offroad 31, collision 0
  route_mean: 0.0965
  route_max: 0.3077
  speed_p90: 18.26
  target_speed_mean: 23.94
  pre-done lane_abs_mean: 2.47
  pre-done heading_abs_mean: 19.49
```

Evaluation comparison:

```text
control_v1_eval_512:
  collision: 1
  offroad: 1
  stuck: 0
  mean_route_progress: 0.2079
  max_route_progress: 0.5641

control_v2b_eval_512:
  collision: 0
  offroad: 2
  stuck: 0
  mean_route_progress: 0.1812
  max_route_progress: 0.5128
```

### 12.5 Judgment

Control v2b is a partial success:

```text
PASS:
  - compileall passed for RCS-AC-first and shared_core
  - sequence BC eval passed
  - PPO 2048/4096 smoke completed
  - checkpoint reload/eval passed
  - stuck collapse was removed
  - collision did not appear during PPO training

WARN:
  - offroad is now the dominant failure mode
  - route progress did not improve over control_v1
  - pre-done heading/lane errors got worse than v1
```

Conclusion:

```text
The clean instability is no longer best explained by PPO std or longitudinal zero-speed collapse.
Those were improved. The remaining clean bottleneck is lateral recovery:
the policy does not correct lane offset and heading error early enough before offroad termination.
```

Recommended next step:

```text
Build lateral_recovery_v1 rather than another pure reward-only run.

Candidate changes:
1. Add a lane/heading recovery controller wrapper or auxiliary steering correction with small gain.
2. Add lane-offset/heading-error fields to BC/RL analysis summaries as first-class metrics.
3. Train/fine-tune with a small lateral recovery imitation target or DAgger-style expert relabel near high lane offset.
4. Keep control_v2b target-speed floor because it removes stuck, but consider max_target_speed_kmh 28~30 if lateral recovery remains weak.
```
