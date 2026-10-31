# Phase 2 Reliability + Risk-aware Fusion 상세계획서

작성일: 2026-04-26

## 0. 문서 목적

이 문서는 `RCS-AC-first`의 Phase 2, 즉 `Reliability + Risk-aware Fusion`을 실제로 구현하고 검증하기 위한 상세 실행 계획서다.

Phase 1은 이미 다음을 통과했다.

- local compile/smoke
- recurrent hidden rollout/update 정합성
- CARLA 128-step probe
- CARLA 512-step smoke
- CARLA 2048-step smoke
- `training_log.csv`, `env_step_log.csv`, checkpoint 생성

따라서 Phase 2는 더 이상 "학습 루프가 도는가"가 아니라, 다음 질문에 답해야 한다.

```text
각 모달이 현재 믿을 수 있는지 q_m으로 추정하고,
현재 주행 위험도가 높은지 r_t와 context로 추정하며,
fusion gate가 q_m과 risk context에 반응해서 위험한 상황에서 불안정한 센서를 덜 쓰는가?
```

Phase 2는 아직 Shapley contribution을 구현하지 않는다. contribution은 uniform placeholder로 유지한다.

## 1. Phase 2 최종 목표

Phase 2 완료 시점의 RCS-AC-first는 다음을 만족해야 한다.

- policy observation contract는 Phase 1 그대로 유지한다.
  - `rgb -> lidar -> ego -> route`
  - flat obs dim `47311`
- auxiliary observation을 별도 branch로 받는다.
  - `quality_stats`
  - `risk_stats`
  - `route_commitment_stats`
  - `sensor_dropout_mask`
  - `corruption_metadata`
- reliability head가 `q_rgb, q_lidar, q_ego, q_route`를 `[0, 1]` 범위로 출력한다.
- risk/context head가 `risk_scalar`와 `risk_context`를 출력한다.
- fusion gate가 reliability와 risk context를 실제 gate input으로 사용한다.
- corruption 또는 dropout이 발생하면 해당 모달의 reliability target이 낮아진다.
- high risk 상황에서는 low reliability 모달의 fusion weight가 내려가도록 regularization한다.
- clean CARLA smoke와 mild corruption CARLA smoke를 모두 통과한다.
- Phase 3 Shapley 구현을 위해 contribution input 자리는 유지한다.

## 2. Phase 2에서 하지 않을 것

Phase 2 범위를 명확히 제한한다.

- Shapley subset sampling을 구현하지 않는다.
- masked critic을 구현하지 않는다.
- exact-ish Shapley eval을 구현하지 않는다.
- dropout consistency loss를 기본 활성화하지 않는다.
- action contract를 3D throttle/brake로 바꾸지 않는다.
- reward, route manager, low-speed brake suppression을 Phase 2 핵심 작업으로 재설계하지 않는다.
- Phase 2의 성공 기준을 goal 도달 성능으로 잡지 않는다.

Phase 2의 성공 기준은 reliability/risk/fusion wiring과 mild corruption 대응 검증이다.

## 3. 현재 코드 상태 요약

### 3.1 Phase 1 구현 상태

이미 구현된 Core:

- `runtime/observation.py`
- `runtime/action.py`
- `runtime/config.py`
- `models/encoders.py`
- `models/fusion_gate.py`
- `models/recurrent_actor_critic.py`
- `models/rcs_ac_model.py`
- `models/perception_aux.py`
- `rl/distributions.py`
- `rl/losses.py`
- `rl/rollout_buffer.py`
- `rl/ppo_agent.py`
- `scripts/smoke_check.py`
- `scripts/train.py`

Phase 1.1로 이미 추가된 perception auxiliary:

- `pred_road_mask_logits`
- `pred_lane_edge_mask_logits`
- dense BCE loss
- mask target buffer/logging

### 3.2 현재 placeholder

`RCSACFirstModel.forward_policy_outputs()`는 현재 다음을 placeholder로 만든다.

```text
reliability = ones([B, 4])
contribution = uniform([B, 4])
risk_scalar = zeros([B])
risk_context = None
```

`FusionGate.forward()`는 현재 `risk_scalar`, `risk_context`를 인자로 받지만 내부에서 삭제한다.

```python
del risk_scalar, risk_context
```

즉 Phase 2의 핵심 변경은 이 placeholder를 실제 모듈로 교체하는 것이다.

### 3.3 shared_core auxiliary schema

`shared_core.env.state_builder` 기준으로 이미 구현된 auxiliary key는 다음이다.

```text
quality_stats
risk_stats
route_commitment_stats
sensor_dropout_mask
```

정확한 field와 dim:

```text
quality_stats: (15,)
  0  rgb_brightness
  1  rgb_contrast
  2  rgb_entropy
  3  rgb_blur_proxy
  4  rgb_dropout_flag
  5  lidar_point_count
  6  lidar_valid_ratio
  7  lidar_sparsity
  8  lidar_drop_ratio
  9  ego_jerk_magnitude
  10 ego_jerk_anomaly
  11 ego_state_mismatch_flag
  12 route_continuity
  13 route_heading_mismatch
  14 route_availability_flag

risk_stats: (6,)
  0 speed_mps
  1 lane_offset
  2 heading_error
  3 curve_proxy
  4 target_distance
  5 weather_severity

route_commitment_stats: (5,)
  0 route_progress
  1 remaining_route_ratio
  2 target_distance_ratio
  3 route_heading_mismatch_norm
  4 route_continuity_score

sensor_dropout_mask: (4,)
  0 rgb
  1 lidar
  2 ego
  3 route

corruption_metadata:
  drop_rgb
  drop_lidar
  drop_ego
  drop_route
  corruption_level
  corruption_window_id
  corruption_mode
```

중요한 현재 상태:

- `shared_core`는 auxiliary observation key를 이미 observation dict에 넣는다.
- `RCS-AC-first/scripts/train.py`는 현재 policy key만 flatten하고 auxiliary는 버퍼에 저장하지 않는다.
- `shared_core.env.carla_env._build_corruption_metadata()`는 현재 기본 zero metadata만 반환한다.
- 따라서 Phase 2 구현은 두 축이다.
  - RCS-AC-first 모델/학습 루프가 auxiliary를 소비하게 만들기
  - active corruption을 켜서 reliability target이 실제로 내려가는 학습 데이터를 만들기

## 4. Phase 2 단계 분할

Phase 2는 한 번에 전부 켜지 않는다. 네 하위 단계로 진행한다.

### 4.1 Phase 2A: Passive Reliability/Risk Wiring

목표:

- auxiliary vector를 `RCS-AC-first` 모델까지 전달한다.
- reliability/risk 모듈을 만들고 shape/loss/logging을 검증한다.
- active corruption은 아직 켜지 않는다.

성공 기준:

- dummy auxiliary forward 통과
- reliability shape `[B, 4]`
- risk scalar shape `[B]`
- risk context shape `[B, 64]`
- losses finite
- Phase 1 clean smoke가 깨지지 않음

### 4.2 Phase 2B: Active Corruption Controller

목표:

- CARLA observation에 약한 corruption을 주입한다.
- corruption metadata와 sensor dropout mask가 env info/log에 기록된다.
- reliability pseudo-label이 corruption severity에 따라 내려간다.

성공 기준:

- RGB dropout 시 `drop_rgb > 0`
- LiDAR drop 시 `drop_lidar > 0`
- route corruption 시 `drop_route > 0`
- sensor_dropout_mask가 all-zero가 아닌 episode가 생김
- target reliability가 해당 모달에서 낮아짐

### 4.3 Phase 2C: Risk-aware Fusion Training

목표:

- reliability와 risk context가 fusion gate input으로 들어간다.
- low reliability 모달 weight가 high risk 상황에서 줄어든다.
- clean smoke와 mild corruption smoke를 모두 통과한다.

성공 기준:

- `fusion_weight_*`, `reliability_*`, `risk_scalar_*` 로그 기록
- forced corruption local test에서 해당 모달 weight 감소
- CARLA mild corruption 2048-step smoke 통과

### 4.4 Phase 2D: 결과 문서화

목표:

- Phase 2 결과 문서를 작성한다.
- Phase 3 진입 가능 여부를 판단한다.

생성 문서:

```text
docs/11_phase2_reliability_risk_result.md
```

## 5. Runtime Auxiliary Contract

### 5.1 새 파일

```text
rcs_ac_first/runtime/auxiliary.py
```

### 5.2 상수

`RCS-AC-first` 모델이 `shared_core`에 강하게 의존하지 않도록 field order는 `runtime/auxiliary.py`에 명시적으로 둔다.

```python
QUALITY_STATS_FIELDS = (
    "rgb_brightness",
    "rgb_contrast",
    "rgb_entropy",
    "rgb_blur_proxy",
    "rgb_dropout_flag",
    "lidar_point_count",
    "lidar_valid_ratio",
    "lidar_sparsity",
    "lidar_drop_ratio",
    "ego_jerk_magnitude",
    "ego_jerk_anomaly",
    "ego_state_mismatch_flag",
    "route_continuity",
    "route_heading_mismatch",
    "route_availability_flag",
)

RISK_STATS_FIELDS = (
    "speed_mps",
    "lane_offset",
    "heading_error",
    "curve_proxy",
    "target_distance",
    "weather_severity",
)

ROUTE_COMMITMENT_STATS_FIELDS = (
    "route_progress",
    "remaining_route_ratio",
    "target_distance_ratio",
    "route_heading_mismatch_norm",
    "route_continuity_score",
)

SENSOR_DROPOUT_KEYS = ("rgb", "lidar", "ego", "route")
CORRUPTION_METADATA_FIELDS = (
    "drop_rgb",
    "drop_lidar",
    "drop_ego",
    "drop_route",
    "corruption_level",
    "corruption_window_id",
    "corruption_mode",
)
```

### 5.3 Dataclass

```python
@dataclass(frozen=True)
class AuxiliaryObservationSpec:
    quality_dim: int = 15
    risk_dim: int = 6
    route_commitment_dim: int = 5
    sensor_dropout_dim: int = 4
    total_dim: int = 30
```

`total_dim` 계산:

```text
15 + 6 + 5 + 4 = 30
```

corruption metadata는 문자열 필드가 섞여 있으므로 model input tensor에 직접 넣지 않는다. 대신 reliability target 생성에만 사용한다.

### 5.4 필수 함수

```python
build_auxiliary_observation_spec() -> AuxiliaryObservationSpec

validate_auxiliary_observation_dict(
    observation: dict[str, Any],
    spec: AuxiliaryObservationSpec,
) -> dict[str, np.ndarray]

flatten_auxiliary_observation_dict(
    observation: dict[str, Any],
    spec: AuxiliaryObservationSpec,
) -> np.ndarray

split_flat_auxiliary_observation(
    auxiliary: Any,
    spec: AuxiliaryObservationSpec,
) -> dict[str, torch.Tensor]

build_default_auxiliary_batch(
    batch_size: int,
    spec: AuxiliaryObservationSpec,
    device: torch.device,
) -> dict[str, torch.Tensor]
```

### 5.5 Missing key 처리

CARLA reset/step에서 auxiliary가 없거나 local smoke에서 dummy만 쓰는 경우가 있다. 기본값은 다음이다.

```text
quality_stats:
  rgb/lidar/ego/route quality는 default vector
risk_stats:
  zeros
route_commitment_stats:
  [0.0, 1.0, 0.0, 0.0, 0.0]
sensor_dropout_mask:
  [0.0, 0.0, 0.0, 0.0]
```

기본 auxiliary는 clean 상태로 해석한다.

## 6. Config 설계

### 6.1 Model config 추가

`RCSACFirstModelConfig`에 추가한다.

```python
quality_stats_dim: int = 15
risk_stats_dim: int = 6
route_commitment_stats_dim: int = 5
sensor_dropout_dim: int = 4
reliability_hidden_dim: int = 64
risk_hidden_dim: int = 64
risk_context_dim: int = 64
reliability_min: float = 0.02
reliability_max: float = 1.0
```

Phase 2 config에서는:

```yaml
model:
  model_variant: phase2_reliability_risk
  enable_reliability: true
  enable_risk_context: true
  enable_shapley: false
  enable_dropout_consistency: false
  quality_stats_dim: 15
  risk_stats_dim: 6
  route_commitment_stats_dim: 5
  sensor_dropout_dim: 4
  reliability_hidden_dim: 64
  risk_hidden_dim: 64
  risk_context_dim: 64
```

### 6.2 Fusion config 추가

`RCSACFirstFusionConfig`에 추가한다.

```python
risk_context_dim: int = 64
reliability_logit_scale: float = 1.0
risk_logit_scale: float = 0.5
use_risk_context: bool = False
```

Phase 2 config:

```yaml
fusion:
  temperature: 1.0
  route_weight_floor: 0.08
  ego_weight_floor: 0.05
  contribution_bias_scale: 0.0
  reliability_logit_scale: 1.0
  risk_logit_scale: 0.5
  use_risk_context: true
  risk_context_dim: 64
```

### 6.3 Loss config 추가

`RCSACFirstLossConfig`에 추가한다.

```python
risk_supervision_loss_weight: float = 0.05
```

기존 필드 중 다음을 Phase 2에서 활성화한다.

```yaml
loss:
  reliability_loss_weight: 0.10
  risk_supervision_loss_weight: 0.05
  risk_regularization_weight: 0.03
  shapley_alignment_loss_weight: 0.0
  dropout_consistency_loss_weight: 0.0
```

`risk_supervision_loss_weight`가 필요한 이유:

- `risk_regularization = risk * unreliable_weight_mass`만 두면 risk head가 risk를 0으로 낮추는 방향으로 도망갈 수 있다.
- 따라서 heuristic risk target을 두고 MSE로 붙잡는다.
- risk regularization은 risk scalar와 reliability를 detach해서 fusion weight를 주로 조절하게 한다.

### 6.4 Corruption config 추가

config root에 `corruption` block을 둔다.

```yaml
corruption:
  enabled: true
  mode: episode_window
  level: mild
  seed: 7
  window_min_steps: 32
  window_max_steps: 96
  episode_apply_prob: 0.60
  rgb_dropout_prob: 0.05
  rgb_zero_prob: 0.03
  rgb_noise_std: 0.02
  rgb_blur_prob: 0.05
  lidar_drop_prob: 0.10
  lidar_zero_prob: 0.03
  ego_noise_std: 0.01
  route_noise_std: 0.03
  route_dropout_prob: 0.05
```

Phase 2 최초 smoke에서는 mild만 쓴다. heavy corruption은 Phase 2 결과를 확인한 뒤 별도 config로 둔다.

### 6.5 새 config 파일

```text
configs/phase2_reliability_risk_smoke.yaml
rcs_ac_first/configs/phase2_reliability_risk_smoke.yaml
```

초기 값:

```yaml
model_type: rcs_ac_first
model_variant: phase2_reliability_risk
training_stage: phase2_reliability_risk_smoke
run_name: rcs_ac_first_phase2_reliability_risk_smoke
total_timesteps: 2048
rollout_steps: 128
max_episode_steps: 384
device: auto
output_root: outputs/phase2_reliability_risk_env
```

## 7. Reliability Module 설계

### 7.1 새 파일

```text
rcs_ac_first/models/reliability.py
```

### 7.2 입력 slice

modality별 quality slice:

```text
rgb_quality:   quality_stats[:, 0:5]
lidar_quality: quality_stats[:, 5:9]
ego_quality:   quality_stats[:, 9:12]
route_quality: quality_stats[:, 12:15]
```

dropout scalar:

```text
rgb_dropout:   sensor_dropout_mask[:, 0:1]
lidar_dropout: sensor_dropout_mask[:, 1:2]
ego_dropout:   sensor_dropout_mask[:, 2:3]
route_dropout: sensor_dropout_mask[:, 3:4]
```

### 7.3 ModalityReliabilityHead

각 모달 head는 해당 feature와 quality slice, dropout scalar를 받는다.

```python
class ModalityReliabilityHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        quality_dim: int,
        hidden_dim: int,
        reliability_min: float = 0.02,
        reliability_max: float = 1.0,
    ):
        ...

    def forward(
        self,
        feature: torch.Tensor,
        quality: torch.Tensor,
        dropout: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([feature, quality, dropout], dim=-1)
        raw = self.mlp(x).squeeze(-1)
        q = reliability_min + (reliability_max - reliability_min) * torch.sigmoid(raw)
        return q
```

출력:

```text
q_m: [B]
```

### 7.4 RCSACReliabilityModule

```python
class RCSACReliabilityModule(nn.Module):
    def forward(
        self,
        features: dict[str, torch.Tensor],
        quality_stats: torch.Tensor,
        sensor_dropout_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
        return {
            "reliability": q,                 # [B, 4]
            "reliability_logits": logits,     # [B, 4]
        }
```

modality order:

```text
rgb, lidar, ego, route
```

### 7.5 Reliability diagnostics

`RCSACFirstModel.get_diagnostics()`에 추가한다.

```text
reliability_rgb
reliability_lidar
reliability_ego
reliability_route
reliability_min
reliability_mean
reliability_entropy
```

## 8. Reliability Target 설계

### 8.1 새 파일

```text
rcs_ac_first/runtime/reliability_targets.py
```

또는 `runtime/auxiliary.py` 안에 포함해도 된다. 추천은 별도 파일이다.

### 8.2 입력

```python
build_reliability_targets(
    quality_stats: np.ndarray | torch.Tensor,
    sensor_dropout_mask: np.ndarray | torch.Tensor,
    corruption_metadata: dict[str, Any] | None = None,
) -> np.ndarray | torch.Tensor
```

### 8.3 기본 target 공식

corruption metadata의 `drop_*` 값은 severity `0.0 ~ 1.0`으로 해석한다.

```text
drop_rgb   = max(corruption.drop_rgb,   sensor_dropout_mask_rgb,   rgb_dropout_flag)
drop_lidar = max(corruption.drop_lidar, sensor_dropout_mask_lidar, lidar_drop_ratio)
drop_ego   = max(corruption.drop_ego,   sensor_dropout_mask_ego,   ego_state_mismatch_flag)
drop_route = max(corruption.drop_route, sensor_dropout_mask_route, 1 - route_availability_flag)
```

target:

```text
q_target_m = clamp(1.0 - 0.95 * drop_m, 0.05, 1.0)
```

추가 penalty:

```text
ego_jerk_anomaly가 true면 q_target_ego -= 0.15
route_heading_mismatch가 큰 경우 q_target_route -= 0.10 * clamp01(abs(route_heading_mismatch) / 30)
```

최종:

```text
q_target = clamp(q_target, 0.05, 1.0)
```

### 8.4 왜 target floor를 0.05로 두는가

`FusionGate`는 reliability를 logit에 `log(q)`로 넣는다. target이 정확히 0이면 학습 초기에 gate가 해당 모달을 완전히 죽이는 방향으로 가기 쉽다. Phase 2에서는 hard masking이 아니라 soft distrust를 학습한다.

## 9. Risk/Context Module 설계

### 9.1 새 파일

```text
rcs_ac_first/models/risk_context.py
```

### 9.2 입력

RiskContextHead 입력은 다음을 concat한다.

```text
risk_stats:              [B, 6]
route_commitment_stats:  [B, 5]
sensor_dropout_mask:     [B, 4]
reliability:             [B, 4]
```

총 input dim:

```text
6 + 5 + 4 + 4 = 19
```

### 9.3 출력

```text
risk_context: [B, 64]
risk_scalar:  [B]
risk_logits:  [B]
```

구조:

```python
class RiskContextHead(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, context_dim=64):
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
        )
        self.risk_head = nn.Linear(context_dim, 1)

    def forward(...):
        context = self.trunk(x)
        risk_scalar = torch.sigmoid(self.risk_head(context)).squeeze(-1)
        return RiskContextOutput(context=context, risk_scalar=risk_scalar, logits=...)
```

### 9.4 Risk target 공식

`runtime/risk_targets.py` 또는 `runtime/auxiliary.py`에 구현한다.

normalized components:

```text
speed_norm     = clamp01(speed_mps / 12.0)
lane_norm      = clamp01(abs(lane_offset) / 2.5)
heading_norm   = clamp01(abs(heading_error) / 30.0)
curve_norm     = clamp01(abs(curve_proxy))
target_close   = 1.0 - clamp01(target_distance / 20.0)
weather_norm   = clamp01(weather_severity)
dropout_norm   = max(sensor_dropout_mask)
route_mismatch = route_commitment_stats[route_heading_mismatch_norm]
```

target:

```text
risk_target =
    0.25 * lane_norm
  + 0.20 * heading_norm
  + 0.15 * speed_norm * max(lane_norm, heading_norm)
  + 0.10 * curve_norm
  + 0.10 * target_close
  + 0.10 * weather_norm
  + 0.05 * dropout_norm
  + 0.05 * route_mismatch
```

최종:

```text
risk_target = clamp01(risk_target)
```

### 9.5 Risk diagnostics

추가 field:

```text
risk_scalar_mean
risk_scalar_max
risk_target_mean
risk_target_max
risk_context_l1_mean
```

## 10. Fusion Gate 변경

### 10.1 현재 구조

현재 gate input은 모달별 projected feature만 쓴다.

```text
gate_mlp(projected_feature_m) -> scalar logit_m
```

그리고 reliability는 `log(q)`로만 더한다.

### 10.2 Phase 2 구조

Phase 2 gate input:

```text
gate_input_m = concat(
    projected_feature_m,     # [B, fusion_dim]
    reliability_m,           # [B, 1]
    contribution_prior_m,    # [B, 1], Phase 2에서는 0.25
    risk_scalar,             # [B, 1]
    risk_context             # [B, risk_context_dim]
)
```

input dim:

```text
fusion_dim + 1 + 1 + 1 + risk_context_dim
= 128 + 3 + 64
= 195
```

`FusionGateConfig.use_risk_context`가 false이면 기존 Phase 1 구조를 유지한다.

### 10.3 Reliability logit bias

추가:

```text
logits = logits + reliability_logit_scale * log(clamp(q, 1e-4, 1.0))
```

`reliability_logit_scale` 기본값:

```text
1.0
```

### 10.4 Risk context 사용 방식

risk context는 `gate_mlp` input에 넣는다. risk가 high일 때 어떤 모달을 줄일지는 gate가 학습한다.

risk scalar는 두 방식으로 쓴다.

1. `gate_mlp` input에 포함
2. risk regularization loss에서 weighting factor로 사용

### 10.5 Safety floor 유지

Phase 2에서도 floor는 유지한다.

```text
ego floor   = 0.05
route floor = 0.08
```

floor 적용 후 전체 weight를 다시 normalize한다.

## 11. Top-level Model API 변경

### 11.1 forward_policy_outputs signature

현재:

```python
forward_policy_outputs(
    obs,
    hidden_state=None,
    track_diagnostics=True,
    apply_route_dropout=None,
)
```

Phase 2:

```python
forward_policy_outputs(
    obs: torch.Tensor,
    *,
    auxiliary: torch.Tensor | dict[str, torch.Tensor] | None = None,
    hidden_state: torch.Tensor | None = None,
    track_diagnostics: bool = True,
    apply_route_dropout: bool | None = None,
) -> dict[str, torch.Tensor | None]
```

### 11.2 auxiliary input 규칙

`auxiliary`가 `None`이면 clean default auxiliary를 생성한다.

`auxiliary`가 tensor이면 shape `[B, 30]`으로 보고 split한다.

`auxiliary`가 dict이면 다음 key를 기대한다.

```text
quality_stats
risk_stats
route_commitment_stats
sensor_dropout_mask
```

### 11.3 output key 추가

```text
reliability                 [B, 4]
reliability_logits          [B, 4]
risk_scalar                 [B]
risk_context                [B, 64]
risk_logits                 [B]
contribution                [B, 4]  # Phase 2 uniform
reliability_target          optional [B, 4]
risk_target                 optional [B]
```

`reliability_target`, `risk_target`는 model output에 넣기보다 PPO batch/loss 쪽에서 관리해도 된다. 권장 구현은 batch/loss 쪽에서 관리한다.

## 12. Rollout Buffer 변경

### 12.1 RolloutBatch 추가 field

```python
auxiliary_observations: torch.Tensor      # [T, 30]
reliability_targets: torch.Tensor         # [T, 4]
risk_targets: torch.Tensor                # [T]
corruption_levels: torch.Tensor           # [T]
```

필요하면 logging용으로 다음도 저장한다.

```python
sensor_dropout_masks: torch.Tensor        # [T, 4]
```

하지만 `auxiliary_observations` 안에 이미 dropout mask가 있으므로 필수는 아니다.

### 12.2 add() 추가 인자

```python
auxiliary_observation: np.ndarray | None = None
reliability_target: np.ndarray | None = None
risk_target: float = 0.0
corruption_level: float = 0.0
```

### 12.3 get_training_batch()

device tensor로 변환:

```python
auxiliary_observations=torch.as_tensor(..., device=device)
reliability_targets=torch.as_tensor(..., device=device)
risk_targets=torch.as_tensor(..., device=device)
corruption_levels=torch.as_tensor(..., device=device)
```

## 13. PPOAgent 변경

### 13.1 act()

현재:

```python
act(observation, deterministic=False)
```

Phase 2:

```python
act(
    observation,
    auxiliary_observation=None,
    deterministic=False,
)
```

내부:

```python
outputs = model.forward_policy_outputs(
    obs,
    auxiliary=aux,
    hidden_state=hidden_before,
)
```

### 13.2 estimate_value()

```python
estimate_value(
    observation,
    auxiliary_observation=None,
    hidden_state=None,
)
```

### 13.3 evaluate_actions()

```python
evaluate_actions(
    observations,
    actions,
    hidden_states,
    auxiliary_observations=None,
)
```

### 13.4 update()

update minibatch에서:

```python
outputs, new_log_probs, entropy = evaluate_actions(
    batch.observations[mb_idx],
    batch.actions[mb_idx],
    batch.hidden_states[mb_idx],
    batch.auxiliary_observations[mb_idx],
)
```

loss:

```text
total_loss =
    actor_loss
  + value_weight * critic_loss
  + entropy_weight * entropy_loss
  + scalar_aux_loss
  + dense_perception_aux_loss
  + reliability_loss_weight * reliability_loss
  + risk_supervision_loss_weight * risk_supervision_loss
  + risk_regularization_weight * risk_regularization
```

## 14. Loss 설계

### 14.1 새 함수

`rl/losses.py`에 추가한다.

```python
compute_reliability_supervision_loss(
    reliability: torch.Tensor,
    reliability_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]

compute_risk_supervision_loss(
    risk_scalar: torch.Tensor,
    risk_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]

compute_risk_aware_regularization(
    fusion_weights: torch.Tensor,
    reliability: torch.Tensor,
    risk_scalar: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]
```

### 14.2 Reliability loss

```text
L_rel = smooth_l1_loss(q, q_target)
```

metrics:

```text
reliability_loss
reliability_mae
reliability_rgb
reliability_lidar
reliability_ego
reliability_route
reliability_target_rgb
reliability_target_lidar
reliability_target_ego
reliability_target_route
```

### 14.3 Risk supervision loss

```text
L_risk_sup = smooth_l1_loss(risk_scalar, risk_target)
```

metrics:

```text
risk_supervision_loss
risk_target_mae
risk_scalar_mean
risk_target_mean
```

### 14.4 Risk-aware regularization

목표:

```text
high risk 상황에서 low reliability 모달에 높은 weight를 주지 않도록 유도
```

공식:

```text
unreliable_weight_mass = sum_m w_m * (1 - stopgrad(q_m))
L_risk_reg = mean(stopgrad(risk_scalar) * unreliable_weight_mass)
```

detach 이유:

- risk head가 regularization을 피하려고 risk를 0으로 낮추는 것을 막는다.
- reliability head가 regularization을 피하려고 q를 무작정 1로 올리는 것을 막는다.
- 이 loss는 주로 fusion weight를 조정한다.

metrics:

```text
risk_regularization
unreliable_weight_mass
risk_weighted_unreliable_mass
```

## 15. Active Corruption 설계

### 15.1 구현 위치

추천 파일:

```text
shared_core/env/corruption.py
```

CARLA env에서 observation을 만든 뒤 policy observation에 corruption을 적용한다. 이렇게 해야 `env_step_log.csv`에 `drop_*`, `corruption_level`, `corruption_mode`가 남는다.

### 15.2 CorruptionConfig

```python
@dataclass
class CorruptionConfig:
    enabled: bool = False
    mode: str = "episode_window"
    level: str = "mild"
    seed: int = 7
    episode_apply_prob: float = 0.60
    window_min_steps: int = 32
    window_max_steps: int = 96
    rgb_dropout_prob: float = 0.05
    rgb_zero_prob: float = 0.03
    rgb_noise_std: float = 0.02
    rgb_blur_prob: float = 0.05
    lidar_drop_prob: float = 0.10
    lidar_zero_prob: float = 0.03
    ego_noise_std: float = 0.01
    route_noise_std: float = 0.03
    route_dropout_prob: float = 0.05
```

### 15.3 적용 위치

`CarlaEnv._build_observation()` 흐름:

현재:

```text
sensor_snapshot
-> quality_snapshot
-> risk_snapshot
-> corruption_metadata = zero
-> StateBuilder.build_observation(...)
```

Phase 2:

```text
sensor_snapshot
-> quality_snapshot
-> risk_snapshot
-> corruption_metadata = corruption_controller.step(...)
-> StateBuilder.build_observation(...)
-> apply corruption to observation["rgb"/"lidar"/"ego"/"route"]
-> rebuild sensor_dropout_mask from metadata
```

### 15.4 Modality별 corruption

RGB:

- dropout: `observation["rgb"] = 0`
- noise: `rgb += normal(0, rgb_noise_std)`
- blur/downscale: local smoke에서는 nearest/downscale placeholder 허용
- metadata:
  - `drop_rgb = severity`
  - `corruption_mode = "rgb_dropout" | "rgb_noise" | "rgb_blur" | "mixed"`

LiDAR:

- random cell drop: BEV grid 일부를 0으로 설정
- zero: 전체 LiDAR grid 0
- metadata:
  - `drop_lidar = severity`

Ego:

- speed/yaw/accel/steer vector에 작은 Gaussian noise
- state mismatch flag 유도
- metadata:
  - `drop_ego = severity`

Route:

- route vector noise
- route dropout: target/maneuver 일부 0
- route floor 때문에 fusion weight는 완전히 0이 되지 않음
- metadata:
  - `drop_route = severity`

### 15.5 Phase 2 mild corruption 제한

처음부터 heavy corruption을 쓰지 않는다.

```text
max corruption_level = 0.30
single episode corruption window = 32~96 steps
episode_apply_prob = 0.60
```

## 16. Train Loop 변경

### 16.1 auxiliary flatten

`scripts/train.py`에서 reset/step observation dict에서 auxiliary를 flatten한다.

```python
aux_spec = build_auxiliary_observation_spec()
auxiliary = flatten_auxiliary_observation_dict(reset_result["obs"], aux_spec)
```

act:

```python
action, log_prob, value = agent.act(
    observation,
    auxiliary_observation=auxiliary,
)
```

step 이후:

```python
next_auxiliary = flatten_auxiliary_observation_dict(step_result["obs"], aux_spec)
```

buffer.add:

```python
buffer.add(
    observation=observation,
    auxiliary_observation=auxiliary,
    reliability_target=build_reliability_targets(...),
    risk_target=build_risk_target(...),
    ...
)
```

그 다음:

```python
observation = next_observation
auxiliary = next_auxiliary
```

### 16.2 bootstrap value

```python
last_value = agent.estimate_value(
    observation,
    auxiliary_observation=auxiliary,
    hidden_state=agent.hidden_state,
)
```

## 17. Logging 설계

### 17.1 training_log.csv 추가 field

```text
reliability_loss
reliability_mae
reliability_rgb
reliability_lidar
reliability_ego
reliability_route
reliability_target_rgb
reliability_target_lidar
reliability_target_ego
reliability_target_route
risk_supervision_loss
risk_target_mae
risk_regularization
risk_scalar_mean
risk_scalar_max
risk_target_mean
risk_target_max
risk_context_l1_mean
unreliable_weight_mass
risk_weighted_unreliable_mass
drop_rgb
drop_lidar
drop_ego
drop_route
corruption_level
corruption_mode
sensor_dropout_rgb
sensor_dropout_lidar
sensor_dropout_ego
sensor_dropout_route
```

### 17.2 env_step_log.csv

`shared_core`는 이미 다음을 로그에 갖고 있다.

```text
drop_rgb
drop_lidar
drop_ego
drop_route
corruption_level
corruption_window_id
corruption_mode
rgb_dropout_flag
lidar_drop_ratio
route_availability_flag
curve_proxy
weather_severity
```

active corruption 구현 시 이 필드가 all zero가 아니어야 한다.

### 17.3 Diagnostics mapping

`RCSACFirstModel.get_diagnostics()`는 기존 attention-compatible key를 유지한다.

```text
attention_mean_rgb = fusion_weight_rgb
attention_mean_lidar = fusion_weight_lidar
attention_mean_ego = fusion_weight_ego
attention_mean_route = fusion_weight_route
```

추가로 reliability/risk key를 넣는다.

## 18. Smoke Check 변경

### 18.1 Local clean smoke

`scripts/smoke_check.py`는 기본 auxiliary를 넣고 forward한다.

검증:

```text
reliability shape [B, 4]
reliability finite
reliability range [0, 1]
risk_scalar shape [B]
risk_context shape [B, 64]
fusion weights sum ~= 1
loss backward finite
checkpoint load finite
```

### 18.2 Local corruption smoke

dummy auxiliary에서:

```text
sensor_dropout_mask = [1, 0, 0, 0]
drop_rgb = 1
```

검증:

```text
reliability_target_rgb < reliability_target_lidar
reliability_loss finite
risk_target finite
risk_regularization finite
```

### 18.3 Forced fusion response smoke

학습 전 reliability head output은 랜덤일 수 있다. 따라서 forced reliability를 `FusionGate`에 직접 넣어 gate 반응을 확인한다.

조건:

```text
clean reliability = [1, 1, 1, 1]
rgb corrupted reliability = [0.05, 1, 1, 1]
```

검증:

```text
fusion_weight_rgb_corrupted < fusion_weight_rgb_clean
sum weights ~= 1
ego/route floor 유지
```

## 19. CARLA 검증 절차

### 19.1 Phase 2 clean regression smoke

Phase 2 모듈을 켠 상태로 corruption은 끈다.

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase2_reliability_risk_smoke.yaml --total-timesteps 512 --run-name phase2_clean_512 --device auto
```

통과 기준:

- 최소 1 update
- `reliability_*` 기록
- `risk_scalar_*` 기록
- NaN 없음
- checkpoint 생성

### 19.2 Phase 2 mild corruption 512

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase2_reliability_risk_smoke.yaml --total-timesteps 512 --run-name phase2_mild_corruption_512 --device auto
```

통과 기준:

- `corruption_level > 0` row 존재
- `sensor_dropout_mask` all zero가 아닌 row 존재
- target reliability가 1보다 작은 row 존재
- 최소 1 PPO update

### 19.3 Phase 2 mild corruption 2048

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config .\configs\phase2_reliability_risk_smoke.yaml --total-timesteps 2048 --run-name phase2_mild_corruption_2048 --device auto
```

통과 기준:

- 최소 1 update 이상, 권장 16 update
- `training_log.csv` 생성
- `env_step_log.csv` 생성
- checkpoint reload 가능
- reliability/risk/fusion fields finite
- forced corruption window가 로그에 보임

### 19.4 로그 inspection

필수 확인:

```text
training_log rows > 0
update rows > 0
actor_loss finite
critic_loss finite
entropy finite
reliability_loss finite
risk_supervision_loss finite
risk_regularization finite
fusion weights finite
risk scalar finite
corruption_level finite
```

corruption response 확인:

```text
drop_rgb > 0인 row에서 reliability_target_rgb < 1
drop_lidar > 0인 row에서 reliability_target_lidar < 1
drop_route > 0인 row에서 reliability_target_route < 1
```

초기 2048-step에서는 학습량이 짧아서 predicted reliability가 완전히 따라오지 않아도 된다. target/loss/logging이 정상이고 update가 깨지지 않으면 Phase 2 wiring은 통과다.

## 20. 파일별 구현 순서

### 20.1 Runtime

1. `runtime/auxiliary.py` 생성
2. `runtime/reliability_targets.py` 생성
3. `runtime/risk_targets.py` 생성
4. `runtime/config.py`에 Phase 2 config field 추가
5. `runtime/__init__.py` export 갱신

### 20.2 Models

1. `models/reliability.py` 생성
2. `models/risk_context.py` 생성
3. `models/fusion_gate.py`를 risk-aware input 구조로 확장
4. `models/rcs_ac_model.py`에 auxiliary input, reliability, risk 연결
5. `models/__init__.py` export 갱신

### 20.3 RL

1. `rl/rollout_buffer.py`에 auxiliary/targets 저장 추가
2. `rl/losses.py`에 reliability/risk losses 추가
3. `rl/ppo_agent.py` act/evaluate/update에 auxiliary 연결
4. `rl/__init__.py` 필요 시 export 갱신

### 20.4 Scripts/config

1. `scripts/smoke_check.py`에 clean/corruption auxiliary smoke 추가
2. `scripts/train.py`에 auxiliary flatten/target 생성/logging 추가
3. `configs/phase2_reliability_risk_smoke.yaml` 생성
4. `rcs_ac_first/configs/phase2_reliability_risk_smoke.yaml` 복사본 생성

### 20.5 shared_core active corruption

1. `shared_core/env/corruption.py` 생성
2. `shared_core/env/carla_env.py`에 corruption config parsing 추가
3. observation build 이후 policy observation corruption 적용
4. corruption metadata를 info/log에 반영
5. clean config에서는 default off 유지

## 21. 구현 중 주의사항

### 21.1 Policy observation contract 유지

`flat_observation`에 auxiliary를 붙이지 않는다.

이유:

- Phase 1 checkpoint/model path와 호환성이 유지된다.
- baseline policy input rule과 충돌하지 않는다.
- Phase 2/3 모듈을 꺼도 Core가 그대로 동작한다.

### 21.2 Reliability target과 prediction 분리

`reliability` prediction은 model output이다.

`reliability_target`은 runtime/rollout/loss에서 만든다.

model 안에서 target을 만들지 않는다.

### 21.3 Risk regularization detach

`risk_regularization`은 다음처럼 detach한다.

```text
mean(stopgrad(risk_scalar) * sum(w_m * (1 - stopgrad(q_m))))
```

이렇게 해야 regularizer가 risk/reliability head를 속이는 방향으로 밀지 않고 fusion weight를 조절한다.

### 21.4 Contribution은 uniform

Phase 2에서 contribution은 계속:

```text
contribution = [0.25, 0.25, 0.25, 0.25]
```

`contribution_bias_scale`도 기본 `0.0`을 유지한다.

### 21.5 Route/ego floor

route/ego floor는 active corruption에서도 유지한다.

route가 corrupt되더라도 route weight가 완전히 0이 되면 route-following policy가 불안정해질 수 있다.

### 21.6 Clean regression

Phase 2 모듈을 켰을 때 clean 512/2048 smoke가 Phase 1보다 구조적으로 불안정해지면 안 된다.

허용:

- reward 낮음
- stuck 많음
- goal_count 0

불허:

- NaN
- rollout add 실패
- checkpoint load 실패
- fusion weight NaN
- reliability/risk field missing
- every episode가 reset 직후 즉시 failure

## 22. Phase 2 완료 정의

Phase 2는 아래 조건을 만족하면 완료로 본다.

### 22.1 Local 완료 조건

- `compileall` 통과
- `smoke_check` 통과
- clean auxiliary forward 통과
- corrupted auxiliary forward 통과
- reliability loss finite
- risk supervision loss finite
- risk regularization finite
- forced reliability fusion response 통과
- checkpoint save/load 통과

### 22.2 CARLA 완료 조건

- clean 512-step smoke 통과
- mild corruption 512-step smoke 통과
- mild corruption 2048-step smoke 통과
- `training_log.csv`에 reliability/risk/loss fields 기록
- `env_step_log.csv`에 corruption metadata 기록
- checkpoint reload 가능

### 22.3 Phase 3 진입 조건

- Phase 2 local smoke가 재현 가능함
- Phase 2 CARLA mild corruption 2048-step이 통과함
- reliability target이 corruption에 반응함
- fusion gate가 forced reliability test에서 expected direction으로 반응함
- risk scalar와 risk target이 finite로 기록됨
- Shapley 관련 key는 아직 uniform/zero placeholder로 유지됨

## 23. Phase 2 결과 문서 템플릿

생성 파일:

```text
docs/11_phase2_reliability_risk_result.md
```

템플릿:

```markdown
# Phase 2 Reliability + Risk-aware Fusion 결과 기록

작성일:

## 구현 파일

- ...

## Local 검증

- compile:
- smoke_check:
- forced fusion response:
- checkpoint:

## CARLA Clean Smoke

- command:
- run:
- updates:
- episodes:
- training_log:
- env_step_log:
- checkpoint:

## CARLA Mild Corruption Smoke

- command:
- run:
- updates:
- corruption rows:
- reliability target response:
- risk fields:

## 주요 로그

- reliability means:
- reliability targets:
- risk scalar:
- risk target:
- fusion weights:
- corruption distribution:

## 문제와 조치

- ...

## Phase 3 진입 판단

- [ ] 진입 가능
- [ ] 보류
```

## 24. Phase 2 한 줄 결론

Phase 2는 RCS-AC가 단순 attention 모델에서 벗어나는 단계다.

이 단계가 끝나면 모델은 다음을 할 수 있어야 한다.

```text
현재 센서가 믿을 수 있는지 q_m으로 보고,
현재 주행 위험도가 높은지 r_t로 보고,
위험한 상황에서 불안정한 모달의 fusion weight를 낮춘다.
```

아직 "이 모달이 value에 얼마나 기여했는가"는 계산하지 않는다. 그것은 Phase 3의 Shapley가 맡는다.
