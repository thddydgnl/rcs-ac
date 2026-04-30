# Phase 3 Counterfactual Shapley Full RCS-AC 상세 구현 계획서

작성일: 2026-04-26

## 0. 문서 목적

이 문서는 `RCS-AC-first`의 Phase 3를 실제 코드로 구현하기 위한 상세 실행 계획서다.

Phase 1은 RGB/LiDAR/ego/route 입력, fusion gate, GRU actor-critic, PPO loop, auxiliary perception head까지 연결했다. Phase 2는 auxiliary observation, reliability, risk context, active corruption, risk-aware fusion을 연결했고 clean CARLA 512-step과 mild corruption CARLA 2048-step smoke를 통과했다.

Phase 3의 목표는 이제 `contribution` placeholder를 실제 counterfactual contribution으로 바꾸고, dropout consistency를 PPO 학습에 연결해서 완성형 RCS-AC 구조를 닫는 것이다.

Phase 3에서 구현할 핵심 질문은 다음과 같다.

```text
현재 action/value 판단에서 RGB, LiDAR, ego, route가 각각 얼마나 기여했는가?
그 기여도와 fusion weight가 일관되는가?
센서 일부가 빠지거나 망가져도 policy가 과도하게 흔들리지 않는가?
```

## 1. 현재 코드 기준 상태

### 1.1 이미 구현된 기반

현재 Phase 3가 기대할 수 있는 기반은 다음과 같다.

- `rcs_ac_first/runtime/observation.py`
  - flat observation split
  - observation order: `rgb -> lidar -> ego -> route`
  - obs dim: `47311`
- `rcs_ac_first/runtime/auxiliary.py`
  - auxiliary order: `quality_stats -> risk_stats -> route_commitment_stats -> sensor_dropout_mask`
  - auxiliary dim: `30`
- `rcs_ac_first/models/rcs_ac_model.py`
  - `RCSACFirstModel.forward_policy_outputs(obs, auxiliary=None, hidden_state=None, ...)`
  - reliability/risk/fusion/recurrent actor-critic 연결 완료
- `rcs_ac_first/models/fusion_gate.py`
  - modality order: `rgb, lidar, ego, route`
  - reliability/risk/context/contribution 인자를 이미 받음
  - contribution은 gate input과 optional logit bias로 들어갈 수 있음
- `rcs_ac_first/rl/rollout_buffer.py`
  - auxiliary observation 저장
  - hidden state 저장
  - reliability target, risk target, dropout mask 저장
- `rcs_ac_first/rl/ppo_agent.py`
  - `act`, `estimate_value`, `evaluate_actions`, `update`가 auxiliary와 hidden state를 사용
  - Phase 2 losses는 실제 계산됨
  - `shapley_alignment_loss`, `dropout_consistency_loss`는 아직 0으로 기록
- `shared_core/env/corruption.py`
  - active corruption controller 구현됨
  - sensor dropout mask와 corruption metadata 제공

### 1.2 현재 Phase 3 placeholder

`RCSACFirstModel.forward_policy_outputs()`는 현재 아래처럼 contribution을 고정한다.

```python
contribution = torch.full_like(reliability, 0.25)
```

즉 현재 fusion gate는 contribution 입력을 받을 수 있지만, 값은 항상 uniform이다.

`rl/losses.py`에는 아래 placeholder가 있다.

```python
compute_shapley_alignment_loss(...) -> zero
compute_dropout_consistency_loss(...) -> zero
```

`ppo_agent.py`의 update metrics에도 아래 값이 고정으로 남아 있다.

```python
"shapley_alignment_loss": 0.0
"dropout_consistency_loss": 0.0
```

Phase 3는 이 세 지점을 실제 구현으로 바꾸는 작업이다.

## 2. Phase 3 최종 목표

Phase 3 완료 시점의 RCS-AC는 아래 흐름을 가져야 한다.

```text
flat_obs + auxiliary
  -> RGB/LiDAR/ego/route encoder
  -> reliability q_m
  -> risk scalar/context
  -> counterfactual contribution c_m
  -> reliability + risk + contribution 기반 fusion gate
  -> GRU actor-critic
  -> PPO actor/value update
  -> reliability/risk/Shapley/dropout consistency losses
```

최종 output key는 Phase 2 key를 유지하면서 아래 key를 실제 값으로 채운다.

```text
contribution                  [B, 4]
counterfactual_value_full      [B]
shapley_raw                   [B, 4]
shapley_normalized            [B, 4]
dropout_consistency_loss       scalar metric
shapley_alignment_loss         scalar metric
masked_value_loss              scalar metric
```

`contribution` order는 반드시 기존 modality order와 동일하다.

```text
rgb, lidar, ego, route
```

## 3. Phase 3에서 하지 않을 것

범위를 명확히 자른다.

- action contract는 계속 2D로 유지한다.
  - `action = [steer, longitudinal]`
  - 3D throttle/brake action은 Phase 3 안정화 이후 별도 판단
- CARLA reward, route manager, low-level control은 재설계하지 않는다.
- Phase 2 reliability/risk target 공식을 크게 바꾸지 않는다.
- Phase 3의 1차 완료 기준은 주행 성능 향상이 아니라 full RCS-AC wiring 검증이다.
- heavy corruption 장기 성능 평가는 Phase 3 구현 후 안정화 라운드에서 수행한다.

## 4. 구현 단계 요약

Phase 3는 한 번에 크게 붙이지 않고 6단계로 닫는다.

```text
Phase 3A: Config / mask contract / model interface 확장
Phase 3B: Counterfactual masked value evaluator
Phase 3C: Approximate Shapley estimator
Phase 3D: Shapley-aware fusion + PPO loss 연결
Phase 3E: Dropout consistency training
Phase 3F: Evaluation / logging / result documentation
```

각 단계는 local smoke를 통과해야 다음 단계로 넘어간다.

## 5. Phase 3A: Config / Mask Contract / Interface

### 5.1 새 config 파일

아래 파일을 만든다.

```text
configs/phase3_full_shapley_smoke.yaml
rcs_ac_first/configs/phase3_full_shapley_smoke.yaml
```

Phase 2 config를 기반으로 하되 아래 값을 켠다.

```yaml
model:
  enable_reliability: true
  enable_risk_context: true
  enable_shapley: true
  enable_dropout_consistency: true

fusion:
  contribution_bias_scale: 0.5
  reliability_logit_scale: 1.0
  risk_logit_scale: 0.5
  use_risk_context: true

loss:
  reliability_loss_weight: 0.10
  risk_supervision_loss_weight: 0.05
  risk_regularization_weight: 0.03
  shapley_alignment_loss_weight: 0.03
  dropout_consistency_loss_weight: 0.03
```

초기에는 Shapley/dropout consistency loss weight를 작게 둔다. 두 loss는 학습 안정성에 영향을 크게 줄 수 있으므로 `0.03`에서 시작하고, 안정화 라운드에서 `0.05 ~ 0.10` 사이로 조정한다.

### 5.2 Config dataclass 확장

`runtime/config.py`에 Phase 3 전용 config를 추가한다.

권장 dataclass:

```python
@dataclass
class RCSACFirstShapleyConfig:
    enabled: bool = False
    estimator: str = "approximate"
    contribution_temperature: float = 1.0
    subsets_per_sample: int = 4
    batch_fraction: float = 0.25
    every_updates: int = 1
    include_empty_subset: bool = True
    include_full_subset: bool = True
    normalize: str = "softmax"
    value_source: str = "current_critic"
    detach_contribution_target: bool = True
    min_contribution: float = 0.02
```

```python
@dataclass
class RCSACFirstDropoutConsistencyConfig:
    enabled: bool = False
    mode: str = "auxiliary_mask"
    batch_fraction: float = 0.25
    max_masks_per_sample: int = 1
    action_mean_weight: float = 1.0
    value_weight: float = 0.25
    reliability_floor: float = 0.05
    detach_clean_policy: bool = True
```

`Phase1CoreConfig`에 아래 필드를 추가한다.

```python
shapley: RCSACFirstShapleyConfig
dropout_consistency: RCSACFirstDropoutConsistencyConfig
```

기존 Phase 1/2 config는 이 block이 없어도 기본값으로 동작해야 한다.

### 5.3 Modality mask contract

새 파일:

```text
rcs_ac_first/runtime/modality_mask.py
```

정의:

```python
MODALITY_ORDER = ("rgb", "lidar", "ego", "route")
FULL_MODALITY_MASK = [1, 1, 1, 1]
EMPTY_MODALITY_MASK = [0, 0, 0, 0]
```

mask 의미:

```text
1 = modality available
0 = modality counterfactually removed
```

mask tensor shape:

```text
[B, 4]
```

Shapley subset에서는 empty subset도 계산할 수 있어야 한다. 다만 empty subset은 모든 projected feature를 zero로 두고, mask embedding과 risk context만 남긴다.

### 5.4 Model interface 확장

`RCSACFirstModel.forward_policy_outputs()`에 optional contribution override를 추가한다.

```python
forward_policy_outputs(
    obs,
    *,
    auxiliary=None,
    hidden_state=None,
    contribution_override=None,
    track_diagnostics=True,
    apply_route_dropout=None,
)
```

규칙:

- `contribution_override is None`이면 모델 내부 estimator 또는 uniform fallback 사용
- shape는 `[B, 4]`
- sum은 row-wise `1.0`
- NaN/Inf 발생 시 uniform fallback 또는 명시적 error

또한 Shapley estimator가 projected features를 사용할 수 있도록 내부 representation method를 분리한다.

권장 추가 method:

```python
encode_modalities(obs) -> dict
build_policy_context(obs, auxiliary) -> dict
evaluate_masked_values(context, masks, hidden_state=None) -> Tensor
```

최소 구현에서는 `forward_policy_outputs()` 내부에서 중복 계산이 생겨도 괜찮다. 단, Phase 3 안정화 때 성능 병목이 생기면 context 분리로 리팩터링한다.

## 6. Phase 3B: Counterfactual Masked Value Evaluator

### 6.1 목표

Shapley는 subset value가 필요하다.

```text
V(S)
```

여기서 `S`는 사용 가능한 modality subset이다. 예를 들어 `S={rgb, route}`이면 LiDAR와 ego를 제거한 상태에서 value를 평가한다.

### 6.2 구현 옵션

현재 코드 구조상 두 가지가 가능하다.

#### Option A: current critic 기반 masked fusion value

현재 `FusionGateOutput`은 `projected_features`를 가지고 있다. 이를 이용해 subset별 masked fused feature를 만들고, 기존 `RecurrentActorCritic` critic으로 value를 평가한다.

장점:

- 별도 critic 학습이 없어 빠르게 구현 가능
- PPO critic과 contribution이 같은 value scale을 사용
- local smoke가 쉽다

단점:

- recurrent actor head까지 같이 지나가므로 subset value 계산 비용이 있다
- subset별 counterfactual distribution이 완전히 독립적인 critic은 아니다

#### Option B: 별도 masked critic

새 파일:

```text
rcs_ac_first/models/masked_critic.py
```

입력:

```text
projected_features        [B, 4, fusion_dim]
modality_mask             [B, 4]
reliability               [B, 4]
risk_scalar               [B]
risk_context              [B, risk_context_dim]
```

출력:

```text
masked_value              [B]
masked_feature            [B, hidden_dim]
```

장점:

- subset value 계산이 빠름
- Shapley estimator 전용으로 학습 가능
- exact-ish eval script 작성이 깔끔함

단점:

- masked subset에 대한 직접 return target이 약함
- 학습 초기에 Shapley signal이 불안정할 수 있음

### 6.3 Phase 3 권장안

구현은 두 단계로 한다.

1. **3B-1 current critic evaluator**
   - 먼저 기존 critic으로 masked value를 계산해 Shapley wiring을 검증한다.
   - 이 경로는 local smoke와 CARLA smoke의 1차 기준이다.

2. **3B-2 masked critic module**
   - 이후 `models/masked_critic.py`를 추가해 amortized masked value estimator를 붙인다.
   - Phase 3 최종 결과 문서에는 current critic과 masked critic 중 어떤 경로를 사용했는지 명시한다.

이 순서가 좋은 이유는 Shapley 수식과 PPO 연결을 먼저 검증할 수 있기 때문이다.

### 6.4 Masked fused feature 생성

새 helper:

```text
rcs_ac_first/rl/shapley.py
```

또는 model 내부 static method로 시작한다.

입력:

```python
projected_features: dict[str, Tensor]  # each [B, D]
fusion_weights: Tensor                 # [B, 4]
mask: Tensor                           # [B, 4]
```

계산:

```text
masked_weight = fusion_weights * mask
masked_weight = masked_weight / clamp(masked_weight.sum(-1, keepdim=True), min=eps)
```

empty subset 처리:

```text
if mask.sum == 0:
  masked_fused = learned_empty_feature or zeros([B, D])
```

초기 구현은 zero empty feature로 충분하다.

### 6.5 Current critic evaluator method

`RecurrentActorCritic`에 아래 helper를 추가한다.

```python
def evaluate_value(
    self,
    fused: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
) -> torch.Tensor:
    return self.forward(fused, hidden_state=hidden_state).value
```

이 helper는 actor output을 쓰지 않고 value만 반환한다. 내부적으로는 같은 GRU와 critic head를 통과한다.

## 7. Phase 3C: Approximate Shapley Estimator

### 7.1 Shapley 정의

모달리티 집합:

```text
M = {rgb, lidar, ego, route}
```

각 modality `m`의 기여도:

```text
phi_m = E_S [ V(S union {m}) - V(S) ]
where S subset M \ {m}
```

모달리티가 4개뿐이므로 exact subset은 16개다. 하지만 online PPO update에서 매번 16 subset을 모두 계산하면 비용이 커질 수 있다. 따라서 training 중에는 approximate sampling을 쓰고, eval script에서는 exact-ish 16 subset sweep을 쓴다.

### 7.2 새 파일

```text
rcs_ac_first/rl/shapley.py
```

구성:

```python
@dataclass
class ShapleyEstimatorConfig:
    num_modalities: int = 4
    subsets_per_sample: int = 4
    include_empty_subset: bool = True
    include_full_subset: bool = True
    contribution_temperature: float = 1.0
    min_contribution: float = 0.02

class ModalitySubsetSampler:
    ...

class ApproximateShapleyEstimator:
    ...
```

필수 함수:

```python
build_all_subset_masks(num_modalities: int = 4) -> torch.Tensor
sample_shapley_subsets(batch_size, num_modalities, subsets_per_sample, device) -> torch.Tensor
normalize_contributions(phi, temperature=1.0, min_contribution=0.02) -> torch.Tensor
compute_shapley_alignment_loss(fusion_weights, contribution) -> torch.Tensor
```

### 7.3 Subset mask examples

```text
empty:         [0, 0, 0, 0]
rgb only:      [1, 0, 0, 0]
lidar only:    [0, 1, 0, 0]
ego only:      [0, 0, 1, 0]
route only:    [0, 0, 0, 1]
full:          [1, 1, 1, 1]
```

### 7.4 Approximate 계산 절차

PPO update minibatch에서 아래 절차를 수행한다.

```text
1. batch obs/aux/hidden으로 base policy output 계산
2. projected features, fusion weights, reliability/risk context 확보
3. 각 sample마다 modality m 선택
4. S subset 샘플링
5. V(S union {m})와 V(S) 계산
6. delta = V(S union {m}) - V(S)
7. modality별 delta 평균 -> phi
8. normalize(phi) -> contribution
```

초기에는 각 sample에서 모든 modality에 대해 `subsets_per_sample`개씩 계산한다.

비용:

```text
B minibatch, modalities=4, K=4
value forward count ~= B * 4 * K * 2
```

이 비용이 너무 크면 `batch_fraction=0.25`를 적용한다.

### 7.5 Contribution normalization

raw phi는 음수가 될 수 있다. fusion gate에 넣을 contribution은 확률 simplex여야 한다.

초기 normalization:

```text
contribution = softmax(phi / temperature)
```

collapse guard:

```text
contribution = (1 - 4 * min_c) * contribution + min_c
```

`min_c=0.02`이면 어떤 modality도 완전히 0이 되지 않는다.

row sum은 다시 1로 normalize한다.

### 7.6 detach 규칙

기본은 contribution target을 detach한다.

```text
shapley_contribution = shapley_contribution.detach()
```

이유:

- Shapley estimator가 gate를 지나 역전파로 자기 자신을 맞추는 순환 경로를 만들 수 있다.
- 초기에는 contribution을 supervision target처럼 사용하고, policy/fusion은 그 target에 맞춘다.

## 8. Phase 3D: Shapley-aware Fusion + PPO Loss

### 8.1 Fusion gate 연결

현재 `FusionGate.forward()`는 이미 contribution을 받는다.

```python
fusion_gate(
    features,
    reliability=reliability,
    contribution=contribution,
    risk_scalar=risk_scalar,
    risk_context=risk_context,
)
```

Phase 3 변경:

- `RCSACFirstModel`에서 uniform contribution 대신 estimator output 사용
- `contribution_bias_scale`을 0보다 크게 설정
- diagnostics에 contribution mean 추가

추가 diagnostics:

```text
contribution_rgb
contribution_lidar
contribution_ego
contribution_route
shapley_raw_rgb
shapley_raw_lidar
shapley_raw_ego
shapley_raw_route
contribution_entropy
contribution_max
contribution_min
```

### 8.2 Shapley alignment loss

`rl/losses.py`의 placeholder를 실제 구현으로 교체한다.

```python
def compute_shapley_alignment_loss(
    fusion_weights: torch.Tensor,
    contribution: torch.Tensor,
    *,
    mode: str = "mse",
    detach_target: bool = True,
) -> torch.Tensor:
    target = contribution.detach() if detach_target else contribution
    return F.mse_loss(fusion_weights, target)
```

초기에는 MSE만 사용한다. KL은 contribution/fusion weight가 floor 때문에 0 근처에서 불안정해질 수 있으므로 안정화 이후 선택한다.

### 8.3 PPOAgent.update 연결

`ppo_agent.py`의 update 내부에서 다음을 추가한다.

```python
if config.model.enable_shapley:
    shapley_outputs = self.shapley_estimator(...)
    contribution = shapley_outputs["contribution"]
    outputs = self.model.forward_policy_outputs(
        batch.observations[mb_idx],
        auxiliary=batch.auxiliary_observations[mb_idx],
        hidden_state=hidden,
        contribution_override=contribution,
    )
    shapley_alignment_loss = compute_shapley_alignment_loss(
        outputs["fusion_weights"],
        contribution,
    )
else:
    shapley_alignment_loss = zero
```

주의:

- 현재 `evaluate_actions()`가 model forward를 한 번 수행한다.
- Shapley contribution이 필요하면 `evaluate_actions()` 내부 또는 그 직전에 contribution을 계산해야 한다.
- 중복 forward가 커질 수 있으므로 1차 구현은 명확성을 우선하고, 안정화 때 최적화한다.

### 8.4 Total loss

Phase 3 total loss:

```text
total_loss =
    actor_loss
  + value_loss_weight * critic_loss
  + entropy_loss_weight * entropy_loss
  + scalar_aux_loss
  + dense_perception_aux_loss
  + reliability_loss_weight * reliability_loss
  + risk_supervision_loss_weight * risk_supervision_loss
  + risk_regularization_weight * risk_regularization
  + shapley_alignment_loss_weight * shapley_alignment_loss
  + dropout_consistency_loss_weight * dropout_consistency_loss
```

NaN guard:

```python
if not torch.isfinite(total_loss):
    raise RuntimeError(...)
```

## 9. Phase 3E: Dropout Consistency

### 9.1 목표

센서가 일부 빠진 counterfactual observation에서 policy가 과도하게 튀지 않게 한다.

clean output:

```text
action_mean_full
value_full
```

masked output:

```text
action_mean_drop
value_drop
```

loss:

```text
L_cons =
    action_mean_weight * mse(action_mean_drop, stopgrad(action_mean_full))
  + value_weight * mse(value_drop, stopgrad(value_full))
```

### 9.2 Observation-level dropout vs representation-level dropout

두 방법이 있다.

#### Option A: observation-level dropout

flat observation에서 RGB/LiDAR/ego/route slice를 직접 zero/noise 처리한다.

장점:

- 실제 센서 dropout과 가장 비슷함
- encoder까지 robustness를 학습함

단점:

- 구현이 조금 더 복잡함
- RGB/LiDAR dense auxiliary target과 충돌할 수 있음

#### Option B: representation-level dropout

encoder output 또는 projected feature에서 modality를 mask한다.

장점:

- 구현이 빠르고 안정적
- Shapley masked value와 같은 mask helper를 공유 가능

단점:

- encoder 자체의 dropout robustness는 덜 학습됨

### 9.3 Phase 3 권장안

1차 구현은 **representation-level dropout consistency**로 간다.

이유:

- Phase 2 active corruption이 이미 observation-level corruption을 제공한다.
- Phase 3의 핵심은 contribution/fusion/recurrent policy 안정성이다.
- Shapley mask와 같은 subset mask를 재사용할 수 있다.

안정화 라운드에서 observation-level consistency를 추가한다.

### 9.4 Dropout mask 샘플링

학습 중 사용할 mask:

```text
single modality drop:
  [0, 1, 1, 1]
  [1, 0, 1, 1]
  [1, 1, 0, 1]
  [1, 1, 1, 0]

paired weak drop:
  [0, 0, 1, 1]
  [0, 1, 0, 1]
  [1, 0, 1, 0]
```

초기에는 single modality drop만 사용한다.

ego/route는 safety floor가 있으므로 완전 제거 대신 reliability floor를 적용한 soft mask도 허용한다.

### 9.5 Loss 구현

`rl/losses.py`에 구현:

```python
def compute_dropout_consistency_loss(
    clean_action_mean: torch.Tensor,
    drop_action_mean: torch.Tensor,
    clean_value: torch.Tensor | None = None,
    drop_value: torch.Tensor | None = None,
    *,
    action_weight: float = 1.0,
    value_weight: float = 0.25,
    detach_clean: bool = True,
) -> torch.Tensor:
    ...
```

metrics:

```text
dropout_consistency_loss
dropout_action_l1
dropout_action_mse
dropout_value_l1
dropout_mask_rgb_rate
dropout_mask_lidar_rate
dropout_mask_ego_rate
dropout_mask_route_rate
```

### 9.6 PPOAgent 연결

`update()` minibatch에서:

```python
clean_outputs = outputs
drop_outputs = model.forward_policy_outputs(
    observations,
    auxiliary=drop_auxiliary_or_same_auxiliary,
    hidden_state=hidden,
    contribution_override=drop_contribution_or_same_contribution,
    modality_mask=sampled_mask,
)
dropout_consistency_loss = compute_dropout_consistency_loss(...)
```

이를 위해 model에 `modality_mask` 인자를 추가하거나, Shapley evaluator helper를 통해 masked fused value/action을 구한다.

초기 구현은 action/value consistency를 `modality_mask` path로 직접 구하는 것이 가장 깔끔하다.

## 10. Model API 최종 제안

Phase 3 후 `forward_policy_outputs()` signature:

```python
def forward_policy_outputs(
    self,
    obs: torch.Tensor,
    *,
    auxiliary: torch.Tensor | None = None,
    hidden_state: torch.Tensor | None = None,
    contribution_override: torch.Tensor | None = None,
    modality_mask: torch.Tensor | None = None,
    track_diagnostics: bool = True,
    apply_route_dropout: bool | None = None,
) -> dict[str, torch.Tensor | None]:
```

`modality_mask` 규칙:

- `None`: full mask `[1,1,1,1]`
- shape `[B,4]`
- projected feature에 적용
- reliability는 직접 0으로 죽이지 않고 fusion gate input에는 유지
- masked feature는 fusion 전 projected feature에 적용

output 추가:

```text
fusion_logits
projected_feature_stack       optional [B, 4, fusion_dim]
modality_mask                 [B, 4]
contribution                  [B, 4]
shapley_raw                   optional [B, 4]
```

dict 안에 `projected_features` dict를 넣으면 typing이 복잡해지므로, training path에서는 stack tensor를 선호한다.

## 11. Logging 설계

### 11.1 training_log.csv 추가 field

`scripts/train.py`의 `TRAINING_LOG_FIELDS`에 추가한다.

```text
shapley_alignment_loss
dropout_consistency_loss
masked_value_loss
contribution_rgb
contribution_lidar
contribution_ego
contribution_route
shapley_raw_rgb
shapley_raw_lidar
shapley_raw_ego
shapley_raw_route
contribution_entropy
contribution_max
contribution_min
dropout_action_l1
dropout_value_l1
dropout_mask_rgb_rate
dropout_mask_lidar_rate
dropout_mask_ego_rate
dropout_mask_route_rate
shapley_subset_count
shapley_batch_fraction
```

### 11.2 env_step_log.csv

env step log는 shared_core가 담당하므로 Phase 3에서 env schema를 크게 바꾸지 않는다.

필요하면 policy diagnostics로 아래 값만 넘긴다.

```text
contribution_rgb
contribution_lidar
contribution_ego
contribution_route
```

하지만 Phase 3 1차 구현에서는 training log 중심으로 충분하다.

## 12. Smoke Check 확장

`scripts/smoke_check.py`에 Phase 3 check를 추가한다.

### 12.1 Model shape check

Pass criteria:

```text
contribution shape [B,4]
contribution finite
contribution row sum ~= 1.0
contribution min >= min_contribution
fusion weights row sum ~= 1.0
action_mean shape [B,2]
value shape [B]
```

### 12.2 Shapley estimator check

Dummy projected features와 dummy value function으로 검증한다.

간단한 deterministic test:

```text
V(S) = sum(feature_strength_m for m in S)
```

이 경우 Shapley contribution은 feature strength에 비례해야 한다.

Pass criteria:

```text
strong modality contribution > weak modality contribution
contribution row sum ~= 1
no NaN/Inf
```

### 12.3 Dropout consistency check

Pass criteria:

```text
clean forward 성공
masked forward 성공
dropout_consistency_loss finite
backward 성공
```

### 12.4 Recurrent PPO update smoke

기존 recurrent rollout smoke에 Phase 3 loss가 포함되어야 한다.

Pass criteria:

```text
PPOAgent.update() 성공
metrics에 shapley_alignment_loss 존재
metrics에 dropout_consistency_loss 존재
둘 다 finite
hidden_state shape 유지
checkpoint save/load 성공
```

## 13. Evaluation Script

새 파일:

```text
rcs_ac_first/scripts/evaluate_shapley.py
```

목표:

- checkpoint load
- fixed CARLA scenario 또는 saved log 기반 evaluation
- 16 subset exact-ish contribution 계산
- fusion weight와 Shapley contribution 비교

### 13.1 CLI

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_shapley `
  --config .\configs\phase3_full_shapley_smoke.yaml `
  --checkpoint .\checkpoints\<run>\last.pt `
  --run-name phase3_shapley_eval `
  --num-steps 256 `
  --device auto
```

### 13.2 Output

```text
outputs/phase3_shapley_eval/<run>/
  contribution_timeseries.csv
  fusion_vs_contribution.csv
  scenario_summary.json
```

CSV fields:

```text
step
route_progress
risk_scalar
corruption_level
fusion_rgb
fusion_lidar
fusion_ego
fusion_route
contribution_rgb
contribution_lidar
contribution_ego
contribution_route
reliability_rgb
reliability_lidar
reliability_ego
reliability_route
value_full
value_empty
done_reason
```

Plot PNG는 없어도 Phase 3 smoke 통과에는 충분하다. 안정화 라운드에서 plot을 추가한다.

## 14. CARLA 검증 계획

### 14.1 Static validation

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Pass:

```text
syntax error 없음
```

### 14.2 Local smoke

```powershell
py -3.12 -m rcs_ac_first.scripts.smoke_check --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_full_shapley_smoke.yaml
```

Pass:

```text
model build 성공
Shapley estimator 성공
contribution shape [B,4]
contribution row sum ~= 1
dropout consistency finite
PPO backward 성공
checkpoint save/load 성공
NaN/Inf 없음
```

### 14.3 CARLA clean regression smoke

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase3_full_shapley_smoke.yaml --total-timesteps 512 --run-name phase3_clean_512 --device auto
```

Pass:

```text
env reset 성공
최소 1 PPO update
training_log.csv 생성
checkpoint 생성
shapley/dropout metrics finite
clean corruption fields all zero 또는 auto clean window
```

### 14.4 CARLA mild corruption smoke

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase3_full_shapley_smoke.yaml --total-timesteps 2048 --run-name phase3_mild_corruption_2048 --device auto
```

Pass:

```text
최소 1 PPO update
권장 16 PPO updates
non-zero corruption row 존재
contribution fields finite
shapley_alignment_loss finite
dropout_consistency_loss finite
fusion/reliability/risk fields finite
checkpoint reload 가능
```

### 14.5 Exact-ish Shapley eval

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_shapley --config .\configs\phase3_full_shapley_smoke.yaml --checkpoint .\checkpoints\phase3_mild_corruption_2048_<timestamp>\last.pt --run-name phase3_eval_256 --num-steps 256 --device auto
```

Pass:

```text
16 subset sweep 성공
contribution_timeseries.csv 생성
fusion_vs_contribution.csv 생성
scenario_summary.json 생성
contribution row sum ~= 1
NaN/Inf 없음
```

## 15. 구현 파일별 작업 목록

### 15.1 Runtime

수정:

```text
rcs_ac_first/runtime/config.py
```

추가:

```text
rcs_ac_first/runtime/modality_mask.py
```

작업:

- Shapley config dataclass 추가
- Dropout consistency config dataclass 추가
- config loader가 `shapley`, `dropout_consistency` block을 읽도록 확장
- modality mask validation helper 추가

### 15.2 Models

수정:

```text
rcs_ac_first/models/rcs_ac_model.py
rcs_ac_first/models/fusion_gate.py
rcs_ac_first/models/recurrent_actor_critic.py
rcs_ac_first/models/__init__.py
```

추가:

```text
rcs_ac_first/models/masked_critic.py
```

작업:

- `forward_policy_outputs()`에 `contribution_override`, `modality_mask` 추가
- contribution fallback은 uniform 유지
- output에 contribution diagnostics 추가
- projected feature stack optional output 추가
- recurrent actor-critic에 `evaluate_value()` 추가
- masked critic module 추가

### 15.3 RL

수정:

```text
rcs_ac_first/rl/losses.py
rcs_ac_first/rl/ppo_agent.py
rcs_ac_first/rl/__init__.py
```

추가:

```text
rcs_ac_first/rl/shapley.py
```

작업:

- subset sampler 구현
- approximate Shapley estimator 구현
- contribution normalization 구현
- Shapley alignment loss 구현
- dropout consistency loss 구현
- PPO update total loss에 Phase 3 losses 연결
- metrics accumulator에 Phase 3 metrics 추가

### 15.4 Scripts / Config

수정:

```text
rcs_ac_first/scripts/smoke_check.py
rcs_ac_first/scripts/train.py
```

추가:

```text
rcs_ac_first/scripts/evaluate_shapley.py
configs/phase3_full_shapley_smoke.yaml
rcs_ac_first/configs/phase3_full_shapley_smoke.yaml
```

작업:

- smoke_check에 Shapley/dropout consistency 검증 추가
- training log fields 추가
- evaluate_shapley script 추가
- Phase 3 config 추가

### 15.5 Docs

추가:

```text
docs/12_phase3_counterfactual_shapley_result.md
```

수정:

```text
docs/01_phase_checklist.md
```

작업:

- Phase 3 구현 결과 문서 작성
- checklist의 Phase 3 항목 갱신
- 성공/실패 항목 기록

## 16. 안정성 규칙

### 16.1 Contribution collapse guard

아래 상태면 실패로 본다.

```text
contribution_rgb/lidar/ego/route 중 하나가 전체 update 내내 0.95 이상
또는 모든 contribution이 항상 0.25 근처로 고정
```

local smoke에서는 uniform이어도 통과 가능하지만, CARLA mild corruption run에서는 최소한 corruption 구간에서 contribution 분산이 생겨야 한다.

### 16.2 Loss weight warmup

Phase 3 초기에는 아래 warmup을 적용한다.

```text
0 ~ 1 update: shapley/dropout loss weight 0
2 ~ 4 update: target weight의 50%
5 update 이후: target weight 100%
```

구현은 PPOAgent.update에서 update index를 받기 어렵다면 config 상 weight를 낮게 시작하는 것으로 대체한다.

### 16.3 No-grad / detach 기준

기본:

- Shapley contribution target은 detach
- clean policy output은 dropout consistency에서 detach
- risk/reliability는 Phase 2 regularization과 동일하게 필요한 곳에서 detach

이유:

- 초기에 여러 auxiliary loss가 서로를 밀어내며 폭주하는 것을 막기 위함
- PPO actor/value loss가 주 signal이고, Phase 3 losses는 regularizer로 시작해야 함

### 16.4 Runtime fallback

`enable_shapley=False`이면:

```text
contribution = uniform
shapley_alignment_loss = 0
```

`enable_dropout_consistency=False`이면:

```text
dropout_consistency_loss = 0
```

Phase 1/2 config가 절대 깨지면 안 된다.

## 17. 완료 기준

Phase 3 완료는 아래 조건을 모두 만족해야 한다.

### 17.1 Local 완료

- compileall 통과
- phase3 smoke_check 통과
- contribution shape `[B,4]`
- contribution row sum `1.0`
- Shapley loss finite
- dropout consistency loss finite
- recurrent PPO update smoke 통과
- checkpoint save/load 통과

### 17.2 CARLA 완료

- phase3 clean 512-step smoke 통과
- phase3 mild corruption 2048-step smoke 통과
- training log에 Phase 3 fields 기록
- checkpoint reload 가능
- action/value/fusion/reliability/risk/contribution 모두 finite

### 17.3 Eval 완료

- exact-ish 16 subset eval script 통과
- contribution CSV 생성
- fusion vs contribution CSV 생성
- scenario summary 생성

### 17.4 결과 문서 완료

아래 문서를 작성한다.

```text
docs/12_phase3_counterfactual_shapley_result.md
```

포함 내용:

- 구현한 모듈 목록
- 실행한 명령어
- local smoke 결과
- CARLA clean 결과
- CARLA mild corruption 결과
- exact-ish Shapley eval 결과
- 로그/체크포인트 위치
- 실패/보정 항목
- 안정화 라운드 진입 가능 여부

## 18. Phase 3 이후 안정화 라운드에서 볼 것

Phase 3 구현이 끝난 뒤 성능 안정화 라운드를 별도로 잡는다.

안정화에서 볼 항목:

- Shapley loss weight `0.03 -> 0.05 -> 0.10`
- dropout consistency loss weight `0.03 -> 0.05 -> 0.10`
- contribution collapse 여부
- fusion weight와 contribution alignment 과도 여부
- clean route progress 악화 여부
- mild corruption에서 action jitter 감소 여부
- RGB/LiDAR dropout 시 해당 modality contribution 하락 여부
- route가 실제 maneuver 결정 구간에서 높은 contribution을 갖는지 여부

Phase 3 구현 완료는 안정화 완료를 의미하지 않는다. Phase 3 구현 완료는 full RCS-AC 구조가 end-to-end로 돌아가고, counterfactual contribution과 consistency signal이 학습 루프에 들어갔음을 증명하는 단계다.

## 19. 권장 구현 순서

1. `runtime/config.py`에 Shapley/dropout config 추가
2. `runtime/modality_mask.py` 추가
3. `rl/shapley.py`에 subset sampler와 contribution normalization 구현
4. `models/recurrent_actor_critic.py`에 `evaluate_value()` 추가
5. `models/rcs_ac_model.py`에 `contribution_override`, `modality_mask` 추가
6. `rl/losses.py`의 Shapley/dropout placeholder를 실제 loss로 교체
7. `rl/ppo_agent.py` update에 Phase 3 loss 연결
8. `scripts/smoke_check.py`에 Phase 3 local smoke 추가
9. `configs/phase3_full_shapley_smoke.yaml` 생성
10. compileall + smoke_check 실행
11. CARLA clean 512 실행
12. CARLA mild corruption 2048 실행
13. `scripts/evaluate_shapley.py` 추가
14. exact-ish eval 실행
15. `docs/12_phase3_counterfactual_shapley_result.md` 작성
16. `docs/01_phase_checklist.md` 갱신

## 20. 최종 판단

현재 Phase 2 코드는 Phase 3 진입이 가능하다.

가장 중요한 구현 포인트는 다음 세 가지다.

```text
1. uniform contribution placeholder를 counterfactual Shapley contribution으로 교체
2. fusion weight와 contribution을 alignment loss로 연결
3. masked/dropout forward를 만들어 dropout consistency loss를 PPO update에 연결
```

이 세 가지가 local smoke, CARLA clean smoke, CARLA mild corruption smoke, exact-ish eval까지 통과하면 RCS-AC-first는 상세계획서 기준의 full RCS-AC 구조에 도달한다.
