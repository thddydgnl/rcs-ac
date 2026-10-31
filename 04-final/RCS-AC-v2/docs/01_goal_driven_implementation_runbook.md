# Goal-driven Raw2Drive Implementation Runbook

작성일: 2026-05-24

목표:

```text
RCS-AC-v2를 Raw2Drive PDF 원형에 맞춰 처음부터 구현한다.
Codex goal continuation이 끊기더라도 이 문서만 보고 다음 구현/검증/문서화 단계를 이어갈 수 있게 한다.
```

참조 우선순위:

1. `RCS-AC-v2/docs/reference/2505.16394v2.pdf`
2. `RCS-AC-v2/docs/00_raw2drive_exact_implementation_plan.md`
3. 이 문서
4. `RCS-AC-v2/docs/02_persistent_subagent_goal_protocol.md`
5. `RCS-AC-v2/docs/03_windows_carla_remote_execution_workflow.md`
6. `RCS-AC-v2/docs/04_git_sync_artifact_cleanup_policy.md`
7. `RCS-AC-v2/docs/05_clean_performance_goal_prompt.md`
8. `RCS-AC-v2/GOAL_STATUS.md`
9. 현재 구현 파일과 테스트
10. v1.1 코드는 참고만 하고 final v2 runtime dependency로 가져오지 않는다.

## 0. Goal 운영 규칙

Codex goal로 이 작업을 진행할 때는 매 turn마다 다음 순서를 따른다.

```text
1. 현재 stage 확인
2. 현재 stage의 gate와 TODO 확인
3. Lite/Full subagent mode 결정
4. Sub-Agent Protocol State와 Ledger 확인
5. stage entry / major edit / failure / gate trigger이면 Agent A + Agent C를 기본 spawn
6. code/config/test/logging/checkpoint wiring을 건드리면 Agent B 추가
7. subagent findings를 통합
8. Implementation Brief 작성
9. 구현 전 관련 파일/테스트 읽기
10. 작은 단위로 구현
11. 해당 stage test 실행
12. 해당 stage result note 업데이트
13. gate 통과 여부 기록
14. GOAL_STATUS.md 업데이트
15. 다음 stage로 넘어갈지 명시
```

금지:

```text
clean gate 통과 전 fusion ablation 구현에 몰입하지 않는다.
reward/action/observation contract를 stage 중간에 조용히 바꾸지 않는다.
raw stream policy inference에 privileged input을 넣지 않는다.
v1.1 waypoint + pure-pursuit 구조를 v2 final policy로 사용하지 않는다.
raw reward/continue head를 직접 학습하지 않는다.
서브에이전트를 초반 1회만 쓰고 이후 gate/failure/major edit에서 재검토하지 않는 것을 금지한다.
```

서브에이전트 운영은 별도 문서를 따른다.

```text
RCS-AC-v2/docs/02_persistent_subagent_goal_protocol.md
```

편집 전 barrier:

```text
Sub-agents first
Implementation Brief second
Edit third
Validate fourth
Document fifth
```

Windows CARLA 실행 barrier:

```text
Mac에서 구현/검증/문서화
Windows에서 CARLA 실행
CARLA readiness guard before every CARLA command
Pull back text/table logs only by default
```

상세 절차:

```text
RCS-AC-v2/docs/03_windows_carla_remote_execution_workflow.md
```

GitHub sync / artifact cleanup barrier:

```text
Official Windows result는 commit/push된 코드에서만 생성한다.
Scratch Windows run은 infrastructure probe로만 사용하고 clean 성능 근거로 쓰지 않는다.
Root .git은 삭제하지 않는다.
Heavy artifact는 .gitignore와 local artifact storage로 차단한다.
```

상세 절차:

```text
RCS-AC-v2/docs/04_git_sync_artifact_cleanup_policy.md
```

Goal complete 조건:

```text
최소 Stage 0 contract 구현과 테스트가 끝나고,
Stage 1 구현 시작 가능한 코드 skeleton/test/config가 생겼을 때만
첫 구현 goal을 complete로 판단한다.
```

전체 Raw2Drive 구현 goal은 다음을 모두 만족해야 complete다.

```text
Stage 0: contract tests pass
Stage 1: privileged world model smoke pass
Stage 2: privileged policy clean gate pass
Stage 3: raw world model alignment gate pass
Stage 4: raw policy clean gate pass
Stage 5: fusion ablation on clean-capable variants
Stage 6: closed-loop clean/stress report written
```

## 1. Stage 0 상세 작업표

Stage 0 목적:

```text
Raw2Drive 원형 구현의 contract를 잠근다.
모델 코딩 전에 action, observation, reward, buffer, config, leakage audit를 먼저 고정한다.
```

### 1.1 파일 생성

구현 파일:

```text
RCS-AC-v2/code/rcs_ac_v2/__init__.py
RCS-AC-v2/code/rcs_ac_v2/env/discrete_actions.py
RCS-AC-v2/code/rcs_ac_v2/env/reward.py
RCS-AC-v2/code/rcs_ac_v2/data/schema.py
RCS-AC-v2/code/rcs_ac_v2/data/replay_buffer.py
RCS-AC-v2/code/rcs_ac_v2/data/sequence_sampler.py
RCS-AC-v2/code/rcs_ac_v2/utils/seed.py
RCS-AC-v2/code/rcs_ac_v2/utils/checkpoint.py
```

테스트 파일:

```text
RCS-AC-v2/tests/test_discrete_actions.py
RCS-AC-v2/tests/test_data_schema.py
RCS-AC-v2/tests/test_replay_buffer.py
RCS-AC-v2/tests/test_reward_contract.py
RCS-AC-v2/tests/test_privileged_leakage_contract.py
```

설정 파일:

```text
RCS-AC-v2/configs/eval/preflight_contract.yaml
```

결과 문서:

```text
RCS-AC-v2/experiments/00_preflight_contract/result.md
```

### 1.2 discrete action contract

PDF Appendix C.2의 39 discrete actions를 구현한다.

데이터 구조:

```python
@dataclass(frozen=True)
class DiscreteAction:
    action_id: int
    throttle: float
    brake: float
    steer: float
    reverse: bool
```

필수 API:

```python
NUM_RAW2DRIVE_ACTIONS = 39
RAW2DRIVE_ACTIONS: tuple[DiscreteAction, ...]
get_discrete_action(action_id: int) -> DiscreteAction
to_carla_control(action: DiscreteAction) -> dict[str, float | bool]
validate_action_table() -> None
```

테스트:

```text
39개 action이 있다.
action_id가 0..38 연속이다.
throttle/brake는 [0,1]이다.
steer는 [-1,1]이다.
reverse는 bool이다.
action lookup은 out-of-range에서 ValueError를 낸다.
to_carla_control은 deterministic이다.
```

주의:

```text
PDF 표의 action order를 바꾸지 않는다.
나중에 policy checkpoint/action logits와 직접 연결되므로 action table version을 config에 기록한다.
```

### 1.3 data schema contract

Raw2Drive Stage 1/2 모두 sequence batch를 사용한다.

기본 dataclass:

```python
@dataclass
class PrivilegedSequenceBatch:
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    next_privileged_bev: torch.Tensor | None = None

@dataclass
class RawSensorSequenceBatch:
    raw_images: torch.Tensor
    raw_imu: torch.Tensor
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
```

Shape convention:

```text
privileged_bev: [B, T, 43, H, W]
privileged_ego: [B, T, D_ego]
raw_images: [B, T, N_cam, 3, H_img, W_img]
raw_imu: [B, T, D_imu]
action: [B, T] int64
reward: [B, T] float32
cont: [B, T] float32 in {0,1}
```

필수 API:

```python
validate_privileged_batch(batch: PrivilegedSequenceBatch, cfg: SchemaConfig) -> None
validate_raw_sensor_batch(batch: RawSensorSequenceBatch, cfg: SchemaConfig) -> None
make_synthetic_privileged_batch(...) -> PrivilegedSequenceBatch
make_synthetic_raw_sensor_batch(...) -> RawSensorSequenceBatch
```

테스트:

```text
valid synthetic batch 통과
잘못된 channel 수 실패
잘못된 action dtype 실패
action id가 39 이상이면 실패
cont 값이 0/1 범위를 벗어나면 실패
raw batch에 privileged target은 있어도 raw policy input view에는 노출되지 않음
```

### 1.4 replay buffer contract

Raw2Drive는 trajectory collection과 sequence sampling을 반복한다.

필수 API:

```python
class SequenceReplayBuffer:
    add_episode(episode: EpisodeData) -> None
    sample(batch_size: int, sequence_length: int, *, seed: int | None = None) -> EpisodeBatch
    __len__() -> int
```

EpisodeData fields:

```text
privileged_bev
privileged_ego
raw_images
raw_imu
action
reward
cont
done
metadata
```

테스트:

```text
짧은 episode는 sequence_length보다 짧으면 sample 후보에서 제외
sample shape가 정확함
seed 고정 시 sample deterministic
episode boundary를 넘어 sample하지 않음
done 이후 cont가 0으로 유지됨
```

### 1.5 reward/continue contract

PDF는 Think2Drive reward를 따른다고만 쓰므로, v2는 우선 명시 contract를 둔다.

초기 reward components:

```text
route_progress_delta
lane_centering
speed_efficiency
collision_penalty
offroad_penalty
red_light_penalty
stuck_penalty
comfort_penalty
terminal_bonus_or_penalty
```

필수 API:

```python
@dataclass(frozen=True)
class RewardInputs:
    route_progress_prev: float
    route_progress_curr: float
    speed_kmh: float
    target_speed_kmh: float
    lane_offset_m: float
    collision: bool
    offroad: bool
    red_light: bool
    stuck: bool
    done: bool

compute_reward(inputs: RewardInputs, cfg: RewardConfig) -> RewardBreakdown
compute_continue(done: bool, truncated: bool) -> float
```

테스트:

```text
route progress 증가 시 reward 증가
collision/offroad/red_light/stuck penalty가 음수
done이면 continue=0
running이면 continue=1
reward breakdown 합이 total과 일치
NaN 입력은 ValueError
```

### 1.6 leakage audit contract

Raw policy inference에서 금지되는 것:

```text
privileged_bev
privileged_ego 중 simulator-only field
CARLA actor box
HD map GT
reward
continue
next state
route completion future value
```

허용되는 것:

```text
raw_images
raw_imu
route command if real navigation provides it
past action/state history if onboard-measurable
```

필수 테스트:

```text
RawPolicyInput dataclass에 privileged fields가 없음
RawSensorSequenceBatch에서 policy_input_view()를 만들 때 privileged fields가 제외됨
Stage 2 training batch에는 privileged target이 있으나 inference view에는 없음
```

## 2. Stage 1 상세 작업표

Stage 1 목적:

```text
privileged BEV semantic mask 기반 world model을 구현하고 synthetic/tiny replay에서 overfit smoke를 통과한다.
```

구현 파일:

```text
code/rcs_ac_v2/world_model/rssm.py
code/rcs_ac_v2/world_model/heads.py
code/rcs_ac_v2/world_model/losses.py
code/rcs_ac_v2/world_model/rollout.py
code/rcs_ac_v2/privileged_stream/encoder.py
code/rcs_ac_v2/privileged_stream/world_model.py
code/rcs_ac_v2/training/train_stage1_privileged_world_model.py
```

테스트 파일:

```text
tests/test_rssm_shapes.py
tests/test_world_model_losses.py
tests/test_privileged_world_model_forward.py
```

RSSM 최소 구현:

```text
deterministic state h_t
categorical or Gaussian stochastic state s_t
prior p(s_t | h_t)
posterior q(s_t | h_t, e_t)
GRU transition f(h_{t-1}, a_{t-1}, s_{t-1})
```

World model heads:

```text
decoder: latent -> privileged_bev reconstruction logits
reward: latent -> scalar reward distribution or scalar regression
continue: latent -> Bernoulli continue logit
```

Loss:

```text
prediction loss:
  decoder reconstruction
  reward prediction
  continue prediction

dynamics loss:
  KL(stopgrad(posterior) || prior), free_nats >= 1

representation loss:
  KL(posterior || stopgrad(prior)), free_nats >= 1
```

Gate:

```text
synthetic one-episode overfit loss decreases
forward shapes pass
one rollout step has finite latent/reward/continue
checkpoint save/load roundtrip works
```

Stage 1 result note:

```text
experiments/01_privileged_world_model/result.md
```

필수 기록:

```text
config
synthetic loss start/end
decoder loss
reward loss
continue loss
KL dyn/rep
NaN check
checkpoint path
```

## 3. Stage 2 상세 작업표

Stage 2 목적:

```text
privileged world model rollout 안에서 privileged actor-critic policy를 학습한다.
```

구현 파일:

```text
code/rcs_ac_v2/policy/actor_critic.py
code/rcs_ac_v2/policy/behavior_loss.py
code/rcs_ac_v2/privileged_stream/policy.py
code/rcs_ac_v2/training/train_stage1_privileged_policy.py
code/rcs_ac_v2/evaluation/evaluate_clean.py
```

Policy output:

```text
action_logits: [B, T, 39]
value distribution or scalar value
entropy
```

Loss:

```text
lambda return
critic negative log likelihood or value regression
actor policy gradient with advantage normalization
entropy regularization
```

Clean gate sequence:

```text
local world-model rollout smoke
CARLA 128-step single-start
CARLA 512-step primary subset
CARLA 2048-step subset
```

Pass criteria:

```text
128: collision/offroad/stuck = 0
512: no systematic offroad/collision
2048: route progress no regression; goal if route is short enough
```

실패하면 fusion으로 가지 말고 여기서 멈춘다.

## 4. Stage 3 상세 작업표

Stage 3 목적:

```text
raw sensor world model을 privileged world model과 latent-aligned 되도록 학습한다.
```

구현 파일:

```text
code/rcs_ac_v2/raw_sensor_stream/encoder.py
code/rcs_ac_v2/raw_sensor_stream/world_model.py
code/rcs_ac_v2/world_model/alignment_losses.py
code/rcs_ac_v2/training/train_stage2_raw_world_model.py
```

Raw encoder 첫 구현 선택:

```text
Phase A: lightweight CNN multi-view encoder + BEV grid projector
Phase B: BEVFormer-style encoder replacement
```

주의:

```text
PDF 원형은 BEVFormer 계열이지만, Stage 3 smoke에서는 lightweight encoder로 shape/loss/guidance를 먼저 검증한다.
BEVFormer 교체는 contract가 안정된 뒤 수행한다.
```

Alignment losses:

```text
spatial-temporal encoder MSE
deterministic state MSE
stochastic posterior KL
decoder-only raw head loss
```

Sampling rule:

```text
raw stochastic distribution에서 한 번 sample
같은 sample을 raw/privileged rollout 양쪽에 넣음
```

Gate:

```text
alignment losses decrease
decoder output non-collapse
multi-step latent divergence bounded
raw reward/continue head 없음 확인
```

## 5. Stage 4 상세 작업표

Stage 4 목적:

```text
raw sensor policy를 head guidance로 학습하고 clean gate를 통과한다.
```

구현 파일:

```text
code/rcs_ac_v2/raw_sensor_stream/policy.py
code/rcs_ac_v2/policy/distillation.py
code/rcs_ac_v2/training/train_stage2_raw_policy.py
```

학습 신호:

```text
raw world model rollout latent
privileged world model reward head
privileged world model continue head
privileged policy action distribution distillation
actor-critic loss
```

금지:

```text
raw reward/continue head로 actor-critic target 만들기 금지
CARLA privileged BEV를 raw policy forward에 넣기 금지
```

Gate:

```text
raw policy inference input audit pass
128 clean pass
512 primary9 clean pass
2048 subset clean pass or documented partial pass
```

## 6. Stage 5 Fusion Ablation 상세 작업표

Fusion ablation은 Stage 4 clean gate 후 시작한다.

구현 파일:

```text
code/rcs_ac_v2/raw_sensor_stream/fusion.py
configs/ablation/fusion_uniform.yaml
configs/ablation/fusion_attention.yaml
configs/ablation/fusion_rcs_ac_shapley.yaml
configs/ablation/fusion_raw2drive_default.yaml
```

Fusion variants:

```text
uniform:
  source weights fixed equal or source-count normalized

learned_attention:
  MLP/Transformer attention over source features

rcs_ac_shapley:
  leave-one-source-out action/logit/latent contribution proxy
  damaged-source suppression only if corruption labels are in training target

raw2drive_default:
  no explicit RCS-AC Shapley; use raw encoder/world-model features as PDF baseline
```

동일하게 고정:

```text
action table
reward
continue
world model architecture except fusion insertion
training budget
eval starts
weather/corruption
checkpoint init
```

Report:

```text
experiments/05_fusion_ablation/result.md
```

필수 표:

```text
clean 512
clean 2048 subset
stress 512 primary9
source weight movement
latent alignment error
world model rollout error
action entropy
route progress
collision/offroad/stuck
```

## 7. Stage 6 Closed-loop Eval 상세 작업표

조건:

```text
clean
rgb_noise_heavy
lidar_sparsity
weather_sensor_severe
combined_severe
```

평가 길이:

```text
128 smoke
512 primary9
2048 subset
optional 4096 paper-scale
```

Result:

```text
experiments/06_closed_loop_eval/final_report.md
```

최종 판단 문장 template:

```text
Under the same Raw2Drive action/reward/world-model/eval contract, fusion variant X
is clean-capable and shows Y under corruption Z. This claim is bounded to
the tested starts/horizons and does not use privileged input at inference.
```

## 8. 매 turn 시작 체크리스트

Codex가 goal을 이어받으면 먼저 아래를 확인한다.

```text
pwd
git status --short --branch
find RCS-AC-v2 -maxdepth 3 -type f | sort
rg -n "TODO|Stage|PASS|FAIL|blocked|Next" RCS-AC-v2/docs RCS-AC-v2/experiments
```

그 다음 현재 stage의 `result.md`가 있으면 읽는다.

```text
RCS-AC-v2/experiments/<stage>/result.md
```

없으면 해당 stage README와 이 runbook을 읽고 시작한다.

## 9. 매 turn 종료 체크리스트

각 turn 종료 전:

```text
1. 수정 파일 목록 확인
2. 관련 테스트 실행
3. 실패한 테스트와 이유 기록
4. stage result.md 업데이트
5. GOAL_STATUS.md 업데이트
6. 다음 액션 한 줄 기록
```

종료 메시지에 반드시 포함:

```text
무엇을 구현했는지
어떤 테스트를 돌렸는지
현재 stage gate 상태
다음에 바로 할 작업
```

## 10. Stage result.md template

각 stage 결과 문서는 이 형식을 쓴다.

```markdown
# Stage N Result

Date:
Git status:
Config:

## Objective

## Sub-agent Findings

## Implementation Brief

## Escalation Decision

## Implemented

## Tests

## Metrics

## Gate Status

PASS / PARTIAL / FAIL / BLOCKED / ESCALATED

## Blockers

## Next Action
```

## 11. 첫 구현 goal의 정확한 범위

첫 구현 goal은 전체 Raw2Drive 완성이 아니다. 너무 크기 때문에 첫 goal은 다음으로 제한한다.

```text
Stage 0 contract implementation complete
Stage 1 world model skeleton ready
```

첫 goal 완료 산출물:

```text
discrete action table 구현
schema 구현
replay buffer 구현
reward/continue contract 구현
leakage audit 구현
Stage 0 tests pass
Stage 0 result.md 작성
Stage 1 skeleton/test stub 생성
```

이 산출물이 생기면 다음 goal은 Stage 1 privileged world model smoke로 분리한다.
