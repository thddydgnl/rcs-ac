# Phase 1 Core 진행 및 검증 상세 문서

작성일: 2026-04-26

## 0. 문서 목적

이 문서는 `RCS-AC-first`의 Phase 1, 즉 `RCS-AC Core`를 실제로 구현하고 검증하기 위한 작업 절차서다.

`00_three_phase_master_plan.md`가 전체 지도라면, 이 문서는 Phase 1 작업자의 손에 들려줄 실행 매뉴얼이다.

Phase 1의 핵심 목표는 다음이다.

```text
RCS-AC 완성본의 최종 모듈 자리를 미리 열어둔 상태로,
RGB/LiDAR/ego/route encoder, fusion interface, recurrent actor-critic,
PPO 연결, scalar auxiliary target, logging/checkpoint까지 깨지지 않게 연결한다.
```

Phase 1은 성능 최적화 단계가 아니다. `goal`을 잘 가는지보다 `구조가 올바르게 굴러가는지`를 먼저 본다.

## 1. Phase 1에서 만들 것과 만들지 않을 것

### 1.1 Phase 1에서 반드시 구현할 것

- 새 Python package scaffold
- config dataclass
- observation split/spec
- action contract
- RGB encoder
- LiDAR encoder
- ego encoder
- route encoder
- simple fusion gate
- ego/route safety floor
- recurrent backbone, 우선 GRU
- actor head
- critic head
- action distribution
- scalar auxiliary heads
  - lane offset
  - heading error
- PPO-style actor/value/entropy loss
- rollout buffer 또는 existing rollout adapter
- checkpoint save/load
- training log schema
- dummy smoke check
- CARLA smoke run 진입점

### 1.2 Phase 1에서 코드 자리는 만들되 꺼둘 것

아래 기능은 config, interface, placeholder key는 만들되 기본 weight는 0으로 둔다.

- reliability head
- risk/context head
- masked critic
- Shapley estimator
- Shapley alignment loss
- reliability supervision loss
- risk-aware regularization
- dropout consistency loss

이렇게 하는 이유는 Phase 2/3에서 기존 파일 구조를 크게 갈아엎지 않기 위해서다.

### 1.3 Phase 1에서 절대 하지 않을 것

- 상세계획서의 모든 loss를 한 번에 켜지 않는다.
- Shapley subset sampling을 Phase 1 학습 루프에 넣지 않는다.
- sensor corruption curriculum을 Phase 1 기본값으로 켜지 않는다.
- action contract를 실험 중간에 2차원과 3차원으로 흔들지 않는다.
- CARLA env/reward/route 규칙을 Phase 1에서 새로 설계하지 않는다.

## 2. Phase 1 완료 정의

Phase 1은 아래 조건을 만족하면 완료로 본다.

### 2.1 Local 완료 조건

- `py -3.12 -m compileall` 통과
- config load 통과
- model build 통과
- dummy flat observation split 통과
- dummy forward 통과
- dummy action distribution 통과
- dummy loss backward 통과
- checkpoint save/load 통과
- fusion weights finite
- fusion weights sum ~= 1
- action shape `[B, 2]`
- value shape `[B]`
- auxiliary prediction shape `[B]`
- NaN/inf 없음

### 2.2 CARLA smoke 완료 조건

- CARLA Python API import 성공
- env reset 성공
- 첫 observation shape 기록
- policy action 생성
- env step 성공
- rollout buffer add 성공
- 최소 1회 PPO update 성공
- `training_log.csv` 생성
- `env_step_log.csv` 생성
- checkpoint 생성
- run 종료 시 env close 성공

### 2.3 Phase 2 진입 조건

아래가 모두 만족되어야 Phase 2로 넘어간다.

- Phase 1 local smoke가 안정적으로 재현됨
- CARLA 2048-step smoke run이 최소 1회 통과
- training log에 model/fusion/action/std diagnostics가 들어옴
- `goal_count=0`이어도 상관없지만 모든 episode가 즉시 failure로 끝나지는 않음
- stuck만 발생하더라도 action/log/reward 분석이 가능할 정도의 정보가 남음

## 3. 권장 구현 순서 요약

Phase 1은 아래 순서로 진행한다.

1. Package scaffold
2. Runtime contract
3. Encoders
4. Fusion gate
5. Recurrent actor-critic
6. Top-level model
7. Distribution/loss
8. PPO/rollout adapter
9. Scripts/config
10. Local validation
11. CARLA smoke validation
12. Phase 1 결과 문서화

각 단계마다 멈춰서 검증한다. 구현을 한꺼번에 많이 밀면 실패 지점이 흐려진다.

## 4. Step 1: Package scaffold

### 4.1 생성할 폴더

```text
RCS-AC-first/
  rcs_ac_first/
    __init__.py
    configs/
    models/
      __init__.py
    rl/
      __init__.py
    runtime/
      __init__.py
    scripts/
      __init__.py
    tests/
```

### 4.2 생성 명령 예시

PowerShell:

```powershell
New-Item -ItemType Directory -Force .\RCS-AC-first\rcs_ac_first\configs
New-Item -ItemType Directory -Force .\RCS-AC-first\rcs_ac_first\models
New-Item -ItemType Directory -Force .\RCS-AC-first\rcs_ac_first\rl
New-Item -ItemType Directory -Force .\RCS-AC-first\rcs_ac_first\runtime
New-Item -ItemType Directory -Force .\RCS-AC-first\rcs_ac_first\scripts
New-Item -ItemType Directory -Force .\RCS-AC-first\rcs_ac_first\tests
```

실제 파일 생성은 구현 단계에서 `apply_patch`로 한다.

### 4.3 검증

```powershell
Get-ChildItem -Recurse .\RCS-AC-first\rcs_ac_first
```

통과 기준:

- 모든 하위 폴더가 존재한다.
- package import가 가능한 `__init__.py`가 있다.

## 5. Step 2: Runtime contract

Runtime contract는 모델 구현 전 반드시 고정한다.

### 5.1 구현 파일

```text
rcs_ac_first/runtime/observation.py
rcs_ac_first/runtime/action.py
rcs_ac_first/runtime/config.py
```

### 5.2 observation.py 요구사항

필수 클래스:

```python
ObservationSliceSpec
PolicyObservationSpec
```

필수 함수:

```python
build_policy_observation_spec(
    rgb_shape: tuple[int, int, int],
    lidar_shape: tuple[int, int, int],
    ego_dim: int,
    route_dim: int,
) -> PolicyObservationSpec

split_flat_observation(
    flat_observation,
    spec: PolicyObservationSpec,
) -> dict[str, torch.Tensor]
```

기본 shape:

```text
rgb:   (3, 90, 160) = 43200
lidar: (1, 64, 64)  = 4096
ego:   (5,)         = 5
route: (10,)        = 10
total: 47311
```

주의:

- `47311`을 코드 곳곳에 하드코딩하지 않는다.
- shape mismatch가 나면 어느 modality에서 틀렸는지 에러 메시지에 포함한다.
- NaN/inf가 있으면 바로 `ValueError`를 낸다.

### 5.3 action.py 요구사항

Phase 1 기본 action contract:

```text
stage4_action_v2
order = ("steer", "longitudinal")
shape = (2,)
bounds = {
  "steer": (-1.0, 1.0),
  "longitudinal": (-1.0, 1.0),
}
```

필수 함수:

```python
action_contract_spec(version: str) -> dict
action_dim_for_contract(version: str) -> int
validate_action_tensor(action: torch.Tensor, version: str) -> None
```

### 5.4 config.py 요구사항

권장 dataclass:

```python
RCSACFirstModelConfig
RCSACFirstFusionConfig
RCSACFirstLossConfig
RCSACFirstTrainConfig
```

Phase 1에서 Phase 2/3용 필드도 포함한다.

```python
enable_reliability: bool = False
enable_risk_context: bool = False
enable_shapley: bool = False
enable_dropout_consistency: bool = False
```

loss weight 기본값:

```python
reliability_loss_weight = 0.0
risk_regularization_weight = 0.0
shapley_alignment_loss_weight = 0.0
dropout_consistency_loss_weight = 0.0
```

### 5.5 Runtime 검증 명령

```powershell
py -3.12 -m compileall -q .\RCS-AC-first\rcs_ac_first\runtime
```

dummy observation split:

```powershell
py -3.12 -c "import sys, torch; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); from rcs_ac_first.runtime.observation import build_policy_observation_spec, split_flat_observation; spec=build_policy_observation_spec((3,90,160),(1,64,64),5,10); x=torch.zeros(2, spec.total_dim); d=split_flat_observation(x, spec); print(spec.total_dim, d['rgb'].shape, d['lidar'].shape, d['ego'].shape, d['route'].shape)"
```

기대 출력:

```text
47311 torch.Size([2, 3, 90, 160]) torch.Size([2, 1, 64, 64]) torch.Size([2, 5]) torch.Size([2, 10])
```

## 6. Step 3: Encoders

### 6.1 구현 파일

```text
rcs_ac_first/models/encoders.py
```

### 6.2 RGBEncoder

입력:

```text
[B, 3, 90, 160]
```

출력:

```text
[B, feature_dim]
```

권장 구조:

```text
Conv2d(3, 32, kernel=5, stride=2)
ReLU
Conv2d(32, 64, kernel=3, stride=2)
ReLU
Conv2d(64, 128, kernel=3, stride=2)
ReLU
AdaptiveAvgPool2d(1)
Flatten
Linear(128, feature_dim)
ReLU
```

초기 `feature_dim=128` 또는 `256`을 추천한다. Phase 1 smoke에서는 128이 빠르고 안정적이다.

### 6.3 LiDAREncoder

입력:

```text
[B, 1, 64, 64]
```

출력:

```text
[B, feature_dim]
```

권장 구조:

```text
Conv2d(1, 16, kernel=5, stride=2)
ReLU
Conv2d(16, 32, kernel=3, stride=2)
ReLU
Conv2d(32, 64, kernel=3, stride=2)
ReLU
AdaptiveAvgPool2d(1)
Flatten
Linear(64, feature_dim)
ReLU
```

### 6.4 VectorEncoder

ego/route 공통 MLP:

```text
Linear(input_dim, hidden_dim)
ReLU
Linear(hidden_dim, feature_dim)
ReLU
```

권장:

```text
ego hidden_dim = 64
route hidden_dim = 64
feature_dim = 128
```

### 6.5 EncoderOutput

권장 dataclass:

```python
EncoderOutput:
    rgb: torch.Tensor
    lidar: torch.Tensor
    ego: torch.Tensor
    route: torch.Tensor
    diagnostics: dict[str, float]
```

diagnostics:

- `rgb_feature_l1_mean`
- `lidar_feature_l1_mean`
- `ego_feature_l1_mean`
- `route_feature_l1_mean`
- `rgb_input_mean`
- `lidar_input_mean`

### 6.6 Encoder 검증

```powershell
py -3.12 -c "import sys, torch; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); from rcs_ac_first.models.encoders import RGBEncoder, LiDAREncoder, VectorEncoder; rgb=RGBEncoder(3,128); lidar=LiDAREncoder(1,128); ego=VectorEncoder(5,64,128); route=VectorEncoder(10,64,128); print(rgb(torch.zeros(2,3,90,160)).shape, lidar(torch.zeros(2,1,64,64)).shape, ego(torch.zeros(2,5)).shape, route(torch.zeros(2,10)).shape)"
```

기대 출력:

```text
torch.Size([2, 128]) torch.Size([2, 128]) torch.Size([2, 128]) torch.Size([2, 128])
```

## 7. Step 4: Fusion gate

### 7.1 구현 파일

```text
rcs_ac_first/models/fusion_gate.py
```

### 7.2 Phase 1 gate의 역할

Phase 1 gate는 아직 진짜 reliability/risk/Shapley를 쓰지 않는다. 하지만 API는 완성형 기준으로 만든다.

```python
fusion_output = fusion_gate(
    projected_features=features,
    reliability=None,
    contribution=None,
    risk_scalar=None,
    risk_context=None,
)
```

Phase 1 기본값:

```text
reliability = ones([B, 4])
contribution = uniform([B, 4])
risk_scalar = zeros([B])
risk_context = zeros([B, context_dim])
```

### 7.3 projection

각 modality feature를 같은 `fusion_dim`으로 projection한다.

```text
p_rgb = P_rgb(h_rgb)
p_lidar = P_lidar(h_lidar)
p_ego = P_ego(h_ego)
p_route = P_route(h_route)
```

Phase 1에서는 feature_dim과 fusion_dim이 같아도 projection layer를 둔다. Phase 2/3에서 gate input을 확장하기 쉬워진다.

### 7.4 weight 계산

```text
logits = gate_mlp(...)
w = softmax(logits / temperature)
```

Phase 1 간단 구현:

```text
gate input per modality = projected_feature_m
```

또는 gate MLP 없이 learnable modality logits로 시작할 수 있다. 하지만 Phase 2 확장을 생각하면 MLP가 낫다.

### 7.5 safety floor

기본값:

```yaml
route_weight_floor: 0.08
ego_weight_floor: 0.05
```

적용:

```text
w_route = max(w_route, route_floor)
w_ego = max(w_ego, ego_floor)
w = w / sum(w)
```

### 7.6 FusionGateOutput

권장 dataclass:

```python
FusionGateOutput:
    fused: torch.Tensor
    weights: torch.Tensor
    logits: torch.Tensor
    projected_features: dict[str, torch.Tensor]
    diagnostics: dict[str, float]
```

diagnostics:

- `fusion_weight_rgb`
- `fusion_weight_lidar`
- `fusion_weight_ego`
- `fusion_weight_route`
- `fusion_entropy`
- `fusion_ego_floor_applied_rate`
- `fusion_route_floor_applied_rate`
- `fused_feature_l1_mean`

### 7.7 Fusion 검증

```powershell
py -3.12 -c "import sys, torch; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); from rcs_ac_first.models.fusion_gate import FusionGate, FusionGateConfig; gate=FusionGate(FusionGateConfig(feature_dim=128, fusion_dim=128)); feats={k: torch.randn(3,128) for k in ['rgb','lidar','ego','route']}; out=gate(feats); print(out.fused.shape, out.weights.shape, out.weights.sum(-1))"
```

기대:

```text
torch.Size([3, 128]) torch.Size([3, 4]) tensor([1., 1., 1.], ...)
```

허용 오차:

```text
abs(weights.sum(-1) - 1) < 1e-5
```

## 8. Step 5: Recurrent actor-critic

### 8.1 구현 파일

```text
rcs_ac_first/models/recurrent_actor_critic.py
```

### 8.2 기본 구조

입력:

```text
z_t: [B, fusion_dim]
hidden: [num_layers, B, recurrent_hidden_dim] 또는 None
```

출력:

```text
action_mean: [B, 2]
value: [B]
next_hidden: [num_layers, B, recurrent_hidden_dim]
```

### 8.3 GRU 처리

GRU는 sequence 입력을 기대하므로 single-step에서는 다음처럼 처리한다.

```text
z_seq = z_t.unsqueeze(1)       # [B, 1, D]
out_seq, next_hidden = gru(z_seq, hidden)
u_t = out_seq[:, -1, :]        # [B, H]
```

`batch_first=True`를 권장한다.

### 8.4 Actor head

```text
Linear(H, hidden_dim)
ReLU
Linear(hidden_dim, action_dim)
Tanh
```

출력 의미:

```text
action_mean[:, 0] = steer mean
action_mean[:, 1] = longitudinal mean
```

### 8.5 Critic head

```text
Linear(H, hidden_dim)
ReLU
Linear(hidden_dim, 1)
Squeeze(-1)
```

### 8.6 Std

Phase 1은 state-independent std를 쓴다.

```python
log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
std = exp(log_std).clamp(min_std, max_std)
```

추천:

```yaml
log_std_init_by_dim: [-0.2, -0.2]
min_std_by_dim: [0.05, 0.05]
max_std_by_dim: [0.8, 0.8]
```

### 8.7 Recurrent 검증

```powershell
py -3.12 -c "import sys, torch; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); from rcs_ac_first.models.recurrent_actor_critic import RecurrentActorCritic, RecurrentActorCriticConfig; m=RecurrentActorCritic(RecurrentActorCriticConfig(input_dim=128, hidden_dim=256, action_dim=2)); x=torch.randn(4,128); out=m(x); print(out.action_mean.shape, out.value.shape, out.next_hidden.shape, m.get_std().shape)"
```

기대:

```text
torch.Size([4, 2]) torch.Size([4]) torch.Size([1, 4, 256]) torch.Size([2])
```

## 9. Step 6: Top-level model

### 9.1 구현 파일

```text
rcs_ac_first/models/rcs_ac_model.py
```

### 9.2 필수 public API

기존 PPO path와 맞추기 위해 아래 메서드는 반드시 제공한다.

```python
forward_policy_outputs(
    obs: torch.Tensor,
    *,
    hidden_state: torch.Tensor | None = None,
    track_diagnostics: bool = True,
    apply_route_dropout: bool | None = None,
) -> dict[str, torch.Tensor]

forward(obs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]

predict_auxiliary_targets(
    obs: torch.Tensor,
    *,
    hidden_state: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]

get_std() -> torch.Tensor
get_diagnostics() -> dict[str, float]
get_stage_b_warmup_modules() -> list[nn.Module]
get_auxiliary_losses() -> dict[str, torch.Tensor]
```

### 9.3 forward_policy_outputs 반환 key

```python
{
    "action_mean": Tensor[B, 2],
    "steer_mean": Tensor[B, 1],
    "longitudinal_mean": Tensor[B, 1],
    "value": Tensor[B],
    "fusion_weights": Tensor[B, 4],
    "next_hidden_state": Tensor[num_layers, B, hidden_dim],
}
```

Phase 2/3 확장 key placeholder:

```python
{
    "reliability": Tensor[B, 4] or None,
    "risk_scalar": Tensor[B] or None,
    "contribution": Tensor[B, 4] or None,
}
```

Phase 1에서는 None 또는 기본값으로 둔다.

### 9.4 auxiliary targets

Phase 1 필수:

```python
{
    "pred_lane_offset": Tensor[B],
    "pred_heading_error": Tensor[B],
}
```

Phase 1에서는 dense aux를 넣지 않는다.

### 9.5 diagnostics

최소 diagnostics:

- `policy_std_mean`
- `policy_std_dim0`
- `policy_std_dim1`
- `rgb_feature_l1_mean`
- `lidar_feature_l1_mean`
- `ego_feature_l1_mean`
- `route_feature_l1_mean`
- `fused_feature_l1_mean`
- `fusion_weight_rgb`
- `fusion_weight_lidar`
- `fusion_weight_ego`
- `fusion_weight_route`
- `fusion_entropy`
- `recurrent_hidden_l1_mean`

### 9.6 Top-level 검증

```powershell
py -3.12 -c "import sys, torch; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel, RCSACFirstModelConfig; m=RCSACFirstModel(RCSACFirstModelConfig(obs_dim=47311)); x=torch.randn(4,47311); out=m.forward_policy_outputs(x); aux=m.predict_auxiliary_targets(x); print(out['action_mean'].shape, out['value'].shape, out['fusion_weights'].shape, aux['pred_lane_offset'].shape, m.get_std().shape)"
```

기대:

```text
torch.Size([4, 2]) torch.Size([4]) torch.Size([4, 4]) torch.Size([4]) torch.Size([2])
```

## 10. Step 7: Distribution and losses

### 10.1 Distribution

구현 파일:

```text
rcs_ac_first/rl/distributions.py
```

Phase 1 추천:

- bounded squashed normal
- action bounds 기반 scaling
- `sample()`
- `rsample()`
- `log_prob(action)`
- `entropy()`

주의:

- tanh squash log-det correction을 넣는다.
- entropy는 base normal entropy 근사로 시작해도 된다.
- action은 최종적으로 bounds로 clamp한다.

### 10.2 Losses

구현 파일:

```text
rcs_ac_first/rl/losses.py
```

필수 함수:

```python
compute_ppo_actor_loss(new_log_probs, old_log_probs, advantages, clip_coef)
compute_value_loss(values, returns)
compute_entropy_loss(entropy)
compute_scalar_aux_loss(pred_lane_offset, target_lane_offset, pred_heading_error, target_heading_error)
```

placeholder:

```python
compute_reliability_loss(...) -> zero
compute_risk_regularization(...) -> zero
compute_shapley_alignment_loss(...) -> zero
compute_dropout_consistency_loss(...) -> zero
```

### 10.3 Loss 검증

```powershell
py -3.12 -c "import sys, torch; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); from rcs_ac_first.rl.losses import compute_ppo_actor_loss; old=torch.zeros(8); new=torch.randn(8)*0.1; adv=torch.randn(8); loss=compute_ppo_actor_loss(new, old, adv, 0.2); loss.backward() if loss.requires_grad else None; print(float(loss.detach()))"
```

통과 기준:

- finite scalar 출력
- NaN 없음

## 11. Step 8: Rollout/PPO adapter

### 11.1 선택지

Phase 1에서는 두 방법 중 하나를 선택한다.

Option A: 기존 `attention` PPO 루프 재사용 wrapper

- 장점: 빠름, 기존 CARLA 경로 안정
- 단점: recurrent hidden state 지원이 제한될 수 있음

Option B: `RCS-AC-first` 안에 PPOAgent/RolloutBuffer 구현

- 장점: 완성형 RCS-AC에 맞게 확장 쉬움
- 단점: 구현량 증가, env loop 버그 가능성 증가

추천:

```text
Phase 1a는 Option A로 smoke를 빠르게 통과한다.
Phase 1b에서 Option B 또는 recurrent-aware buffer로 확장한다.
```

### 11.2 Phase 1a minimal adapter

기존 `attention/rl/ppo_agent.py` 방식과 호환되도록 model API를 맞춘다.

필요:

- `forward_policy_outputs`
- `predict_auxiliary_targets`
- `get_std`
- `get_diagnostics`

Phase 1a에서 recurrent hidden은 모델 내부에서 zero hidden으로 시작해도 된다. 이건 완성형 recurrent 학습은 아니지만 구조 smoke에는 충분하다.

### 11.3 Phase 1b recurrent-aware buffer

추가 저장:

- `initial_hidden_state`
- `episode_start`
- `sequence_id`
- `step_in_sequence`

추가 검증:

- hidden reset on done
- hidden detach per rollout
- truncated sequence minibatch

Phase 1b는 Phase 1a smoke 이후 진행한다.

## 12. Step 9: Config and scripts

### 12.1 Phase 1 smoke config

파일:

```text
rcs_ac_first/configs/phase1_core_smoke.yaml
```

권장 내용:

```yaml
seed: 7
model_type: rcs_ac_first
model_variant: phase1_core
training_stage: phase1_core_smoke
run_name: rcs_ac_first_phase1_core_smoke
total_timesteps: 2048
rollout_steps: 128
max_episode_steps: 384
device: auto

model:
  action_dim: 2
  rgb_shape: [3, 90, 160]
  lidar_shape: [1, 64, 64]
  ego_dim: 5
  route_dim: 10
  feature_dim: 128
  fusion_dim: 128
  recurrent_hidden_dim: 256
  recurrent_layers: 1
  min_std_by_dim: [0.05, 0.05]
  max_std_by_dim: [0.8, 0.8]
  log_std_init_by_dim: [-0.2, -0.2]
  enable_reliability: false
  enable_risk_context: false
  enable_shapley: false
  enable_dropout_consistency: false

fusion:
  temperature: 1.0
  route_weight_floor: 0.08
  ego_weight_floor: 0.05

loss:
  value_loss_weight: 0.5
  entropy_loss_weight: 0.01
  aux_lane_offset_loss_weight: 1.0
  aux_heading_error_loss_weight: 1.0
  reliability_loss_weight: 0.0
  risk_regularization_weight: 0.0
  shapley_alignment_loss_weight: 0.0
  dropout_consistency_loss_weight: 0.0
```

### 12.2 smoke_check.py

필수 검사:

1. config load
2. model build
3. dummy observation forward
4. auxiliary prediction
5. distribution sample/log_prob
6. loss backward
7. checkpoint save/load

명령:

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py --config .\RCS-AC-first\rcs_ac_first\configs\phase1_core_smoke.yaml
```

### 12.3 train.py

Phase 1a에서는 기존 `attention/scripts/train.py`를 subprocess로 호출하는 wrapper여도 된다.

하지만 새 모델을 쓰려면 아래 중 하나가 필요하다.

1. `attention/rl/ppo_agent.py`에 `rcs_ac_first` branch 추가
2. `RCS-AC-first` 안의 train loop가 직접 `RCSACFirstModel`을 사용

Phase 1 설계상 더 깔끔한 것은 2번이다. 다만 빠른 smoke 목적이면 1번 adapter도 허용한다.

## 13. Local validation 절차

### 13.1 전체 compile

```powershell
py -3.12 -m compileall -q .\RCS-AC-first
```

실패 시:

- SyntaxError 파일과 line 확인
- import path 확인
- circular import 확인

### 13.2 import check

```powershell
py -3.12 -c "import sys; sys.path.insert(0, r'C:\Users\thddy\Documents\code\RCS-AC-first'); import rcs_ac_first; print('ok')"
```

기대:

```text
ok
```

### 13.3 model shape check

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py --only-model
```

통과 기준:

- action mean `[B, 2]`
- value `[B]`
- std `[2]`
- fusion weights `[B, 4]`
- auxiliary predictions `[B]`

### 13.4 loss backward check

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py --only-loss
```

통과 기준:

- total loss finite
- `loss.backward()` 성공
- at least one parameter grad finite
- gradient norm finite

### 13.5 checkpoint check

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py --only-checkpoint
```

통과 기준:

- checkpoint file 생성
- state_dict load 성공
- load 후 dummy forward 성공

## 14. CARLA smoke validation 절차

### 14.1 사전 조건

- CARLA 서버 실행
- Python 3.12에서 CARLA API import 가능
- `CARLA_ROOT` 또는 기본 경로의 wheel이 맞음
- 포트 기본값 `2000`

확인:

```powershell
py -3.12 -c "import carla; print('carla ok')"
```

만약 실패하면 기존 `attention/scripts/train.py`의 bootstrap 로직을 참고한다.

### 14.2 128-step probe

가장 먼저 아주 짧게 돌린다.

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\train.py --config .\RCS-AC-first\rcs_ac_first\configs\phase1_core_smoke.yaml --total-timesteps 128 --run-name rcs_ac_first_phase1_probe_128 --device auto
```

통과 기준:

- env reset
- first action
- at least one env step
- no shape mismatch

### 14.3 512-step smoke

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\train.py --config .\RCS-AC-first\rcs_ac_first\configs\phase1_core_smoke.yaml --total-timesteps 512 --run-name rcs_ac_first_phase1_smoke_512 --device auto
```

통과 기준:

- rollout buffer add
- 최소 1 update 또는 buffer flush
- log 생성
- checkpoint 생성

### 14.4 2048-step smoke

```powershell
py -3.12 .\RCS-AC-first\rcs_ac_first\scripts\train.py --config .\RCS-AC-first\rcs_ac_first\configs\phase1_core_smoke.yaml --total-timesteps 2048 --run-name rcs_ac_first_phase1_smoke_2048 --device auto
```

통과 기준:

- 최소 1 update 이상
- checkpoint `last.pt`
- `training_log.csv`
- `env_step_log.csv`
- `nan_detected=false`
- action/value/log_prob finite

## 15. Log inspection 절차

### 15.1 training_log.csv 확인

PowerShell:

```powershell
Get-ChildItem -Recurse .\RCS-AC-first\logs -Filter training_log.csv
```

Python summary:

```powershell
py -3.12 -c "import csv, pathlib; p=sorted(pathlib.Path(r'C:\Users\thddy\Documents\code\RCS-AC-first\logs').rglob('training_log.csv'))[-1]; rows=list(csv.DictReader(p.open(encoding='utf-8'))); print(p); print('rows', len(rows)); print(rows[-1])"
```

필수 확인 필드:

- `model_type`
- `model_variant`
- `record_type`
- `total_steps`
- `update`
- `actor_loss`
- `critic_loss`
- `entropy`
- `policy_std_mean`
- `route_progress`
- `done_reason`
- `rgb_feature_l1_mean`
- `lidar_feature_l1_mean`
- `ego_feature_l1_mean`
- `route_feature_l1_mean`
- `fused_feature_l1_mean`
- `fusion_weight_rgb`
- `fusion_weight_lidar`
- `fusion_weight_ego`
- `fusion_weight_route`

### 15.2 env_step_log.csv 확인

필수 확인:

- action contract version
- steer
- longitudinal
- throttle
- brake
- speed
- lane offset
- heading error
- reward
- done reason

### 15.3 정상 범위 가이드

Phase 1 smoke에서 기대할 수 있는 범위:

```text
policy_std_mean: 0.05 ~ 0.8
fusion weights: each 0.0 ~ 1.0, sum ~= 1.0
feature_l1_mean: finite, usually > 0
actor_loss: finite
critic_loss: finite
entropy: finite
```

주의:

- `goal_count=0`은 Phase 1 실패가 아니다.
- `done_reason=stuck`이 많을 수 있다.
- `route_progress`가 낮아도 wiring 검증은 통과할 수 있다.
- 다만 action이 항상 0이거나 NaN이면 실패다.

## 16. 실패별 진단표

### 16.1 import 실패

증상:

```text
ModuleNotFoundError: No module named 'rcs_ac_first'
```

확인:

- `sys.path`에 `RCS-AC-first` root가 들어갔는가
- `__init__.py`가 있는가
- 파일명과 패키지명이 일치하는가

### 16.2 observation dim mismatch

증상:

```text
Expected flat observation dim 47311, got ...
```

확인:

- 실제 env observation shape
- config의 `rgb_shape`, `lidar_shape`, `ego_dim`, `route_dim`
- route representation이 10차원인지
- auxiliary가 ego 뒤에 붙어 obs_dim이 바뀌었는지

### 16.3 action shape mismatch

증상:

```text
Expected action_dim=2, got 3
```

확인:

- action contract version
- model `action_dim`
- distribution action bounds
- env decode

### 16.4 fusion weight NaN

확인:

- input feature NaN 여부
- gate logits overflow
- temperature가 0 또는 너무 작은지
- floor 적용 후 normalization denominator가 0인지

### 16.5 value loss explosion

확인:

- reward scale
- returns finite 여부
- critic output NaN 여부
- learning rate
- grad clipping

### 16.6 action이 항상 brake 쪽으로 감

확인:

- longitudinal mean 평균
- log_std 너무 작은지
- action distribution sample이 deterministic처럼 되는지
- low-speed brake suppression 로그
- reward가 forward progress를 충분히 주는지

### 16.7 CARLA reset 실패

확인:

- CARLA 서버 실행 여부
- port
- town 로드 여부
- synchronous mode cleanup
- 이전 actor 잔여물

## 17. Phase 1 결과 문서 템플릿

Phase 1 구현 후 아래 문서를 추가한다.

```text
RCS-AC-first/docs/10_phase1_core_result.md
```

템플릿:

```markdown
# Phase 1 Core 결과 기록

작성일:

## 구현 파일

- ...

## Local 검증

- compile:
- import:
- dummy forward:
- dummy backward:
- checkpoint:

## CARLA smoke

- command:
- total_timesteps:
- update_count:
- episode_count:
- final checkpoint:
- training_log:
- env_step_log:

## 주요 로그

- average_reward:
- average_route_progress:
- done_reason_distribution:
- policy_std_mean:
- fusion weights:
- feature L1 means:

## 문제와 조치

- ...

## Phase 2 진입 판단

- [ ] 진입 가능
- [ ] 보류
```

## 18. Phase 1 최종 체크리스트

- [ ] package scaffold 생성
- [ ] runtime observation contract 구현
- [ ] action contract 구현
- [ ] config dataclass 구현
- [ ] RGB encoder 구현
- [ ] LiDAR encoder 구현
- [ ] ego encoder 구현
- [ ] route encoder 구현
- [ ] fusion gate 구현
- [ ] route/ego safety floor 구현
- [ ] GRU recurrent actor-critic 구현
- [ ] top-level model 구현
- [ ] scalar auxiliary heads 구현
- [ ] distribution 구현
- [ ] PPO losses 구현
- [ ] rollout/PPO adapter 구현
- [ ] smoke_check 구현
- [ ] train entrypoint 구현
- [ ] phase1 config 구현
- [ ] compile 통과
- [ ] import 통과
- [ ] dummy forward 통과
- [ ] dummy backward 통과
- [ ] checkpoint save/load 통과
- [ ] CARLA 128-step probe 통과
- [ ] CARLA 512-step smoke 통과
- [ ] CARLA 2048-step smoke 통과
- [ ] training log 생성
- [ ] env step log 생성
- [ ] NaN 없음
- [ ] Phase 1 결과 문서 작성

## 19. Phase 1 한 줄 결론

Phase 1은 RCS-AC의 영혼인 Shapley를 아직 켜지 않는 단계다. 하지만 이 단계가 튼튼해야 Phase 2의 reliability/risk와 Phase 3의 Shapley가 의미 있게 올라간다.

따라서 Phase 1의 성공 기준은 성능이 아니라 다음이다.

```text
네 모달을 받아 recurrent actor-critic으로 action/value를 만들고,
future RCS-AC 모듈 자리를 보존한 채 PPO smoke run을 안정적으로 통과한다.
```

