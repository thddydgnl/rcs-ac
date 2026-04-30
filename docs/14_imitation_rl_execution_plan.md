# RCS-AC-first Imitation + RL 상세 실행 계획서

작성일: 2026-04-27

## 0. 문서 목적

이 문서는 `RCS-AC-first`에서 imitation learning과 reinforcement learning을 결합해 clean driving 성능을 먼저 확보하고, 그 위에서 RCS-AC의 sensor corruption / adverse weather 강건성을 검증하기 위한 상세 실행 계획서다.

현재 `RCS-AC-first`는 Phase 3까지 구현되어 있다.

- Phase 1: RGB/LiDAR/ego/route 기반 recurrent PPO core
- Phase 1.1: RGB perception auxiliary task
- Phase 2: reliability + risk-aware fusion
- Phase 3: counterfactual Shapley contribution + dropout consistency

하지만 Phase 1~3의 현재 smoke validation은 주로 "학습 루프가 깨지지 않고 도는가"를 검증한다. 실제 논문 목표인 아래 주장을 만들려면 clean 환경에서 기본 주행 능력이 먼저 충분해야 한다.

```text
RCS-AC는 일반 주행 환경에서는 PPO/SAC baseline과 유사한 clean driving 성능을 유지하면서,
악천후 및 센서 노이즈 상황에서는 더 높은 robustness retention과 낮은 safety degradation을 보인다.
```

이를 위해 imitation learning을 먼저 사용해 기본 운전 policy를 만들고, 이후 PPO fine-tuning으로 RCS-AC의 reliability/risk/Shapley/dropout consistency 구조가 악조건에서 강건성을 높이는지 검증한다.

## 1. 핵심 결론

`imitation + RL`은 RCS-AC 논문 목표와 충돌하지 않는다. 단, 공정성 조건을 반드시 지켜야 한다.

필수 원칙:

- RCS-AC에만 imitation warmup을 주면 안 된다.
- PPO, SAC, RCS-AC 비교군 모두 동일 expert dataset으로 pretrain해야 한다.
- 동일 observation, action space, reward, route set, training step budget을 사용해야 한다.
- clean 성능은 "압도적으로 이기는가"보다 "baseline 대비 손상되지 않는가"를 본다.
- 강건성 주장은 corrupted/weather performance가 clean 대비 얼마나 유지되는지로 평가한다.

추천 논문 표현:

```text
All compared methods are initialized with the same expert dataset through behavior cloning
and fine-tuned using the same RL budget. RCS-AC improves robustness retention under
unseen sensor corruption and adverse weather while maintaining comparable clean-driving performance.
```

한국어 표현:

```text
모든 비교 모델은 동일한 expert dataset으로 imitation pretraining을 수행하고,
동일한 강화학습 step budget으로 fine-tuning하였다. RCS-AC는 clean 주행 성능을
유지하면서 unseen sensor corruption 및 adverse weather에서 더 높은 성능 보존율을 보인다.
```

## 2. 전체 단계

`RCS-AC-first` 기준 imitation + RL은 아래 6단계로 진행한다.

```text
Phase I0: Expert Dataset 수집
Phase I1: Behavior Cloning 학습
Phase I2: BC Policy 검증
Phase I3: BC checkpoint -> PPO fine-tuning 연결
Phase I4: Robust RL fine-tuning
Phase I5: PPO/SAC/RCS-AC 강건성 비교 실험
```

각 단계의 역할은 명확히 분리한다.

- I0는 데이터 생성 단계다.
- I1은 supervised pretraining 단계다.
- I2는 imitation policy 단독 검증 단계다.
- I3는 기존 Phase 3 PPO loop와 imitation checkpoint를 연결하는 단계다.
- I4는 clean driving policy를 RCS-AC robustness objective로 fine-tuning하는 단계다.
- I5는 논문용 비교 실험 단계다.

## 3. 현재 코드 기준

### 3.1 유지할 기존 contract

Observation contract:

```text
flat observation order:
rgb -> lidar -> ego -> route

rgb:   (3, 90, 160)
lidar: (1, 64, 64)
ego:   (5,)
route: (10,)
```

Action contract:

```text
action = [steer, longitudinal]

steer:        [-1, 1]
longitudinal: [-1, 1]

env conversion:
throttle = max(longitudinal, 0)
brake    = max(-longitudinal, 0)
```

Auxiliary contract:

```text
quality_stats -> risk_stats -> route_commitment_stats -> sensor_dropout_mask

quality_stats:           15
risk_stats:               6
route_commitment_stats:   5
sensor_dropout_mask:      4
total auxiliary dim:     30
```

Modality order:

```text
rgb, lidar, ego, route
```

### 3.2 활용할 기존 구현

기존 주요 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\models\rcs_ac_model.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\rl\ppo_agent.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\smoke_check.py
C:\Users\thddy\Documents\code\shared_core\env\carla_env.py
```

새로 추가할 주요 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\data\expert_dataset.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\collect_expert_dataset.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train_bc.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\evaluate_bc.py
C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_clean_smoke.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_clean_train.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_finetune_smoke.yaml
```

나중에 SAC baseline을 구현할 때 추가할 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\rl\sac_agent.py
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train_sac.py
C:\Users\thddy\Documents\code\RCS-AC-first\configs\sac_baseline_smoke.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_sac_finetune_smoke.yaml
```

## 4. Phase I0: Expert Dataset 수집

### 4.1 목표

CARLA expert/autopilot 또는 기존 route-following controller가 주행한 transition을 저장한다.

처음 목표:

```text
clean_smoke_2000.npz
clean_20000.npz
```

장기 목표:

```text
clean_100000.npz
clean_mild_weather_100000.npz
recovery_clean_50000.npz
```

처음에는 clean environment만 수집한다. heavy corruption/weather를 expert dataset에 처음부터 넣으면 baseline도 해당 상황을 imitation으로 직접 학습하게 되어 RCS-AC의 unseen robustness 차이가 줄어들 수 있다.

추천 데이터 분리:

```text
BC pretraining:
clean 또는 clean + very mild weather

RL fine-tuning:
clean + mild corruption

Final evaluation:
unseen heavy corruption + adverse weather
```

### 4.2 Expert source

우선순위:

1. CARLA Traffic Manager / autopilot control
2. shared_core 내부 route follower 또는 low-level controller
3. PID waypoint follower

초기 구현은 CARLA autopilot 기반으로 시작한다.

Expert action 변환:

```text
expert_steer = control.steer
expert_longitudinal = control.throttle - control.brake
expert_action = [expert_steer, expert_longitudinal]
```

주의:

- throttle과 brake가 동시에 0이거나 동시에 큰 값이면 기록한다.
- CARLA control range가 action contract와 맞는지 clamp한다.
- reverse gear 상황은 Phase I0에서는 제외하거나 metadata로 기록한다.

### 4.3 저장할 데이터

필수 arrays:

```text
obs_flat:         [N, 47311], float32
auxiliary:        [N, 30], float32
expert_action:   [N, 2], float32
reward:          [N], float32
done:            [N], bool
route_progress:  [N], float32
```

권장 arrays:

```text
rgb:              [N, 3, 90, 160], uint8 또는 float32
lidar:            [N, 1, 64, 64], float32
ego:              [N, 5], float32
route:            [N, 10], float32
expert_control:   [N, 3], float32  # steer, throttle, brake
value_hint:        [N], float32     # optional discounted return
episode_id:        [N], int32
step_in_episode:   [N], int32
```

Metadata arrays:

```text
done_reason:          [N], string 또는 int code
weather_id:           [N], int32
route_id:             [N], int32
town_id:              [N], int32
corruption_mode:      [N], string 또는 int code
corruption_level:     [N], float32
drop_rgb:             [N], bool
drop_lidar:           [N], bool
drop_ego:             [N], bool
drop_route:           [N], bool
```

처음 smoke dataset에서는 `obs_flat`, `auxiliary`, `expert_action`, `done`, `route_progress`, `done_reason`만 있어도 된다.

### 4.4 Dataset 파일 위치

```text
C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_smoke_2000.npz
C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_20000.npz
C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_100000.npz
```

Large dataset은 `.npz`가 너무 커질 수 있다. 이후 필요하면 shard 방식으로 바꾼다.

추천 shard 구조:

```text
data\expert\clean_100000\
  metadata.json
  shard_00000.npz
  shard_00001.npz
  shard_00002.npz
```

### 4.5 collect_expert_dataset.py CLI

추가할 스크립트:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\collect_expert_dataset.py
```

추천 CLI:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_expert_dataset `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_full_shapley_smoke.yaml `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_smoke_2000.npz `
  --num-transitions 2000 `
  --run-name expert_clean_smoke_2000 `
  --host 127.0.0.1 `
  --port 2000 `
  --weather clear_noon `
  --corruption none
```

필수 옵션:

```text
--config
--output
--num-transitions
--run-name
--host
--port
--weather
--corruption
```

권장 옵션:

```text
--max-episode-steps
--route-set
--town
--seed
--save-rgb-lidar
--compress
--expert-source autopilot|route_follower|pid
--min-route-progress
--skip-bad-transitions
```

### 4.6 Bad transition 필터링

Imitation dataset에 나쁜 expert 행동이 많이 들어가면 BC policy가 그대로 배운다.

필터링 후보:

- spawn 직후 센서 warmup이 안 된 transition
- expert action이 NaN/Inf인 transition
- observation에 NaN/Inf가 있는 transition
- route_progress가 일정 시간 동안 0인 transition
- expert가 stuck 상태인 transition
- collision 직전/직후 transition은 별도 recovery dataset으로 분리

처음 구현에서는 bad transition을 완전히 버리기보다 `bad_transition_mask`로 저장하는 것도 좋다.

```text
bad_transition_mask: [N], bool
bad_transition_reason: [N], string/int
```

### 4.7 I0 Smoke 통과 기준

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_expert_dataset `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_full_shapley_smoke.yaml `
  --output C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_smoke_2000.npz `
  --num-transitions 2000 `
  --run-name expert_clean_smoke_2000 `
  --device auto
```

Pass criteria:

- CARLA API import 성공
- env reset 성공
- expert action 생성 성공
- `obs_flat` shape `[2000, 47311]`
- `auxiliary` shape `[2000, 30]`
- `expert_action` shape `[2000, 2]`
- action range가 `[-1, 1]`
- NaN/Inf 없음
- route_progress 기록
- done_reason 기록
- dataset reload 성공

## 5. Phase I1: Behavior Cloning 학습

### 5.1 목표

`RCSACFirstModel`이 expert action을 예측하도록 supervised pretraining을 수행한다.

입력:

```text
obs_flat
auxiliary
optional perception targets
```

출력:

```text
action_mean = [steer_mean, longitudinal_mean]
```

기본 loss:

```text
bc_loss = MSE(action_mean, expert_action)
```

### 5.2 BC dataset loader

추가할 파일:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\data\expert_dataset.py
```

역할:

- `.npz` 파일 로드
- shard directory 로드
- shape validation
- dtype normalization
- optional action smoothing target 생성
- train/val split
- PyTorch Dataset 반환

권장 class:

```python
class ExpertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        val_fraction: float = 0.1,
        normalize_obs: bool = False,
        load_rgb_lidar: bool = False,
    ):
        ...
```

반환 dict:

```text
obs:              Tensor[47311]
auxiliary:        Tensor[30]
expert_action:   Tensor[2]
route_progress:  Tensor[]
done:            Tensor[]
metadata:         optional
```

### 5.3 train_bc.py CLI

추가할 스크립트:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\train_bc.py
```

추천 command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_clean_smoke.yaml `
  --dataset C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_smoke_2000.npz `
  --run-name bc_clean_smoke `
  --device auto
```

필수 CLI:

```text
--config
--dataset
--run-name
--device
```

권장 CLI:

```text
--epochs
--batch-size
--learning-rate
--val-fraction
--resume
--save-every
--num-workers
```

### 5.4 BC config

추가할 config:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_clean_smoke.yaml
```

초기 값:

```yaml
model:
  feature_dim: 128
  fusion_dim: 128
  gru_hidden_dim: 256
  enable_reliability: true
  enable_risk_context: true
  enable_shapley: true
  enable_dropout_consistency: true

bc:
  epochs: 5
  batch_size: 32
  learning_rate: 0.0003
  weight_decay: 0.00001
  val_fraction: 0.1
  grad_clip_norm: 0.5
  bc_loss_weight: 1.0
  steer_loss_weight: 1.0
  longitudinal_loss_weight: 1.0
  action_smoothness_loss_weight: 0.0
  perception_aux_loss_weight: 0.1
  reliability_loss_weight: 0.0
  risk_loss_weight: 0.0
  use_auxiliary: true
  use_shapley: false

logging:
  log_interval: 50
  checkpoint_interval: 1
```

중요:

- BC stage에서는 `enable_shapley=true`인 모델 구조를 사용하더라도, online Shapley 계산은 비활성화해도 된다.
- 이유: BC는 expert action을 맞추는 것이 목적이고, Shapley subset value sweep은 학습 비용을 크게 늘린다.
- Phase 3 구조의 checkpoint 호환성은 유지하되, `contribution`은 uniform fallback 또는 cheap estimate로 둔다.

### 5.5 BC loss 구성

기본:

```text
action_mse = mean((action_mean - expert_action)^2)
```

분리 로깅:

```text
steer_mse = mean((steer_mean - expert_steer)^2)
longitudinal_mse = mean((longitudinal_mean - expert_longitudinal)^2)
```

optional smoothness:

같은 episode 안의 연속 transition을 batch로 가져올 수 있을 때만 적용한다.

```text
smoothness_loss = mean(abs(action_mean_t - action_mean_t_minus_1))
```

초기에는 비활성화한다.

### 5.6 BC checkpoint format

저장 위치:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_clean_smoke\last.pt
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_clean_20000\last.pt
```

필수 checkpoint key:

```text
model_state_dict
optimizer_state_dict
config
epoch
global_step
train_loss
val_loss
dataset_path
model_type
model_variant
```

권장 metadata:

```text
expert_dataset_hash
num_transitions
action_mse
steer_mse
longitudinal_mse
created_at
```

### 5.7 BC 로그

로그 위치:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\logs\bc_clean_smoke\bc_training_log.csv
```

필수 필드:

```text
run_name
epoch
global_step
split
bc_loss
steer_mse
longitudinal_mse
action_l1
steer_abs_error
longitudinal_abs_error
policy_std
fusion_rgb
fusion_lidar
fusion_ego
fusion_route
reliability_rgb
reliability_lidar
reliability_ego
reliability_route
risk_scalar
contribution_rgb
contribution_lidar
contribution_ego
contribution_route
```

### 5.8 I1 Smoke 통과 기준

Command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_clean_smoke.yaml `
  --dataset C:\Users\thddy\Documents\code\RCS-AC-first\data\expert\clean_smoke_2000.npz `
  --run-name bc_clean_smoke `
  --device auto
```

Pass criteria:

- dataset load 성공
- train/val split 성공
- model build 성공
- forward 성공
- `action_mean` shape `[B, 2]`
- `bc_loss` finite
- backward 성공
- checkpoint save/load 성공
- validation loss 기록
- NaN/Inf 없음

## 6. Phase I2: BC Policy 검증

### 6.1 목표

BC policy가 RL 없이 CARLA clean 환경에서 최소한 forward driving을 하는지 확인한다.

이 단계의 목적은 BC가 완벽한지 보는 것이 아니다. PPO fine-tuning 시작점으로 쓸 수 있을 정도의 정책인지 확인한다.

### 6.2 evaluate_bc.py CLI

추가할 스크립트:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\evaluate_bc.py
```

추천 command:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\bc_clean_smoke.yaml `
  --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_clean_smoke\last.pt `
  --run-name bc_clean_eval_512 `
  --num-steps 512 `
  --device auto
```

### 6.3 평가 로그

저장 위치:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\<run_name>\bc_eval_log.csv
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\<run_name>\summary.json
```

필수 필드:

```text
step
episode
steer
longitudinal
throttle
brake
reward
done
done_reason
route_progress
speed
collision
offroad
stuck
fusion_rgb
fusion_lidar
fusion_ego
fusion_route
reliability_rgb
reliability_lidar
reliability_ego
reliability_route
risk_scalar
contribution_rgb
contribution_lidar
contribution_ego
contribution_route
```

### 6.4 BC 평가 기준

Smoke 기준:

```text
512 step run
- crash 없음
- action finite
- action range 정상
- route_progress > 0
- checkpoint reload 성공
- env close 정상
```

초기 pass 기준:

```text
route_progress_mean > 0
stuck_rate <= 0.50
collision_rate <= 0.50
action_jitter finite
```

실제 BC checkpoint로 PPO fine-tuning을 시작하기 전 권장 기준:

```text
route_progress_mean > random policy
stuck_rate <= 0.30
collision_rate <= 0.30
steer_jitter <= 0.30
longitudinal_jitter <= 0.35
```

너무 엄격하게 잡지 않는다. BC는 최종 policy가 아니라 RL fine-tuning의 시작점이다.

## 7. Phase I3: BC checkpoint에서 PPO fine-tuning 연결

### 7.1 목표

기존 `scripts/train.py`가 random initialization 대신 BC checkpoint에서 모델 weight를 로드해 PPO fine-tuning을 시작하도록 한다.

추가할 CLI:

```text
--init-checkpoint
--init-strict
--load-model-only
```

권장 기본값:

```text
--init-strict false
--load-model-only true
```

이유:

- BC checkpoint와 PPO checkpoint는 optimizer state나 training metadata가 다를 수 있다.
- PPO fine-tuning에서는 model weights만 가져오고 optimizer는 PPO용으로 새로 만드는 것이 안전하다.

### 7.2 train.py command

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_finetune_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_clean_smoke\last.pt `
  --total-timesteps 2048 `
  --run-name bc_phase3_finetune_smoke_2048 `
  --device auto
```

### 7.3 PPO fine-tuning config

추가할 config:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_finetune_smoke.yaml
```

초기 추천:

```yaml
train:
  total_timesteps: 2048
  rollout_steps: 128
  max_episode_steps: 384
  device: auto
  learning_rate: 0.0001
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.15
  entropy_coef: 0.005
  value_loss_coef: 0.5
  grad_clip_norm: 0.5

loss:
  scalar_aux_loss_weight: 0.05
  perception_aux_loss_weight: 0.05
  reliability_loss_weight: 0.05
  risk_supervision_loss_weight: 0.05
  risk_regularization_weight: 0.02
  shapley_alignment_loss_weight: 0.01
  dropout_consistency_loss_weight: 0.03
  bc_residual_loss_weight: 0.0

rl_finetune:
  init_from_bc: true
  freeze_encoders_steps: 0
  bc_residual_steps: 0
  bc_residual_decay: linear
```

중요한 차이:

- BC에서 시작하면 PPO learning rate를 random init보다 낮게 잡는다.
- entropy coefficient도 너무 높게 두지 않는다.
- PPO가 BC policy를 갑자기 망가뜨리지 않도록 clip range를 조금 보수적으로 둔다.

### 7.4 Optional: residual BC loss

PPO fine-tuning 중 BC policy가 너무 빨리 깨지면 residual BC loss를 일부 유지한다.

```text
total_loss = ppo_loss + lambda_bc(t) * MSE(action_mean, expert_action_batch)
```

단, on-policy PPO rollout과 offline expert dataset을 동시에 쓰는 구현 복잡도가 늘어난다.

초기 구현에서는 비활성화한다.

나중에 필요하면:

```text
bc_residual_loss_weight: 0.05
bc_residual_steps: 50000
bc_residual_decay: linear
```

### 7.5 I3 Smoke 통과 기준

Pass criteria:

- BC checkpoint load 성공
- missing/unexpected keys 로그 출력
- PPO optimizer 새로 생성
- env rollout 시작
- `training_log.csv` 생성
- `env_step_log.csv` 생성
- PPO update 최소 1회
- actor/value/std/fusion/reliability/risk/contribution finite
- checkpoint save/load 성공

## 8. Phase I4: Robust RL fine-tuning

### 8.1 목표

BC로 얻은 clean driving policy를 RCS-AC 구조로 fine-tuning하여 sensor corruption / weather shift에서 덜 무너지게 만든다.

추천 단계:

```text
I4-A: clean PPO fine-tuning
I4-B: mild sensor corruption fine-tuning
I4-C: mild weather fine-tuning
I4-D: mixed mild robustness fine-tuning
I4-E: unseen heavy corruption/weather evaluation
```

### 8.2 Curriculum

처음부터 heavy corruption과 악천후를 학습에 넣지 않는다.

권장 curriculum:

```text
Stage 1:
clear noon, no corruption

Stage 2:
clear noon, mild RGB/LiDAR/route noise

Stage 3:
cloudy/wet, mild sensor corruption

Stage 4:
rain/fog light, mixed mild corruption

Evaluation only:
heavy rain, dense fog, night, strong mixed corruption
```

이유:

- clean driving이 먼저 안정되어야 robust behavior가 의미 있다.
- heavy setting을 학습에 넣으면 unseen robustness 평가가 약해질 수 있다.
- mild에서 학습하고 heavy에서 평가하면 일반화 주장이 더 명확하다.

### 8.3 Robust fine-tuning loss weight 초기값

초기 추천:

```text
shapley_alignment_loss_weight: 0.01 ~ 0.03
dropout_consistency_loss_weight: 0.03 ~ 0.08
risk_regularization_weight: 0.02 ~ 0.05
reliability_loss_weight: 0.05
risk_supervision_loss_weight: 0.05
perception_aux_loss_weight: 0.05
entropy_coef: 0.003 ~ 0.01
```

주의:

- Shapley alignment가 너무 크면 contribution이 critic noise를 과하게 따라갈 수 있다.
- dropout consistency가 너무 크면 policy가 sensor input 차이를 무시하는 방향으로 뭉개질 수 있다.
- risk regularization이 너무 크면 특정 modality fusion weight가 과도하게 억제될 수 있다.

### 8.4 Stability monitor

Robust fine-tuning 중 아래를 자동 분석해야 한다.

```text
finite_check:
  action, value, log_prob, loss, fusion, reliability, risk, contribution

ppo_health:
  actor_loss
  critic_loss
  entropy
  policy_std
  approx_kl optional

driving_quality:
  route_progress
  done_reason
  collision_rate
  offroad_rate
  stuck_rate
  timeout_rate
  action_jitter

robustness_response:
  reliability drop under sensor dropout
  fusion weight shift under dropout
  risk increase under corruption
  contribution entropy
  contribution collapse
```

권장 script:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\scripts\analyze_stability.py
```

입력:

```powershell
--training-log
--env-log
--checkpoint
--mode clean|mild|heavy|weather
```

출력:

```text
PASS / WARN / FAIL
summary.json
stability_report.md
```

## 9. Phase I5: PPO/SAC/RCS-AC 비교 실험

### 9.1 비교군

최소 비교군:

```text
BC-PPO
BC-RCS-AC-PPO
```

논문 권장 비교군:

```text
BC-PPO
BC-SAC
BC-RCS-AC Phase 1
BC-RCS-AC Phase 2
BC-RCS-AC Phase 3
```

가능하면 ablation:

```text
Full RCS-AC-PPO
RCS-AC without reliability/risk
RCS-AC without Shapley
RCS-AC without dropout consistency
RCS-AC without risk-aware regularization
```

### 9.2 명명 규칙

논문과 로그에서는 아래 이름을 사용한다.

```text
BC-PPO:
  same observation/action/reward, simple fusion, PPO fine-tuning

BC-SAC:
  same observation/action/reward, simple fusion, SAC fine-tuning

BC-RCS-AC-PPO:
  RCS-AC architecture, PPO fine-tuning
```

주의:

`RCS-AC vs PPO vs SAC`라고만 쓰면 RCS-AC가 PPO로 학습된 것인지 SAC로 학습된 것인지 불명확해진다. 정확히는:

```text
RCS-AC-PPO vs PPO vs SAC
```

또는:

```text
PPO-optimized RCS-AC compared against PPO and SAC baselines
```

### 9.3 공정성 조건

모든 비교군은 아래 조건을 맞춘다.

```text
same expert dataset
same BC epochs or same BC gradient steps
same observation set
same action contract
same reward
same route set
same weather train set
same corruption train set
same RL total timesteps
same evaluation episodes
same random seeds
```

달라질 수 있는 것:

```text
policy architecture
fusion mechanism
RL optimizer type, PPO or SAC
RCS-AC auxiliary losses
```

### 9.4 평가 환경

Train:

```text
clear noon
clean
mild sensor corruption optional
light weather optional
```

Validation:

```text
clean
mild RGB noise/dropout
mild LiDAR dropout
mild route noise
light rain
```

Test:

```text
heavy rain
dense fog
night
wet road
mixed strong corruption
RGB blackout intervals
LiDAR dropout intervals
route perturbation
```

### 9.5 핵심 metric

Clean driving:

```text
route_completion
route_progress
collision_rate
offroad_rate
stuck_rate
timeout_rate
average_speed
action_jitter
```

Robustness:

```text
robustness_retention = corrupted_score / clean_score
safety_degradation = corrupted_safety_event_rate - clean_safety_event_rate
jitter_increase = corrupted_jitter - clean_jitter
recovery_time_after_corruption
```

RCS-AC internal response:

```text
reliability_response = clean_reliability - corrupted_reliability for affected modality
fusion_response = clean_fusion_weight - corrupted_fusion_weight for affected modality
risk_response = corrupted_risk_scalar - clean_risk_scalar
contribution_shift
contribution_entropy
dropout_consistency_action_l1
dropout_consistency_value_l1
```

### 9.6 논문용 pass gate

Clean pass gate:

```text
RCS-AC clean route_progress >= baseline clean route_progress * 0.95
RCS-AC collision_rate <= baseline collision_rate * 1.10 + epsilon
RCS-AC offroad_rate <= baseline offroad_rate * 1.10 + epsilon
RCS-AC stuck_rate <= baseline stuck_rate * 1.10 + epsilon
```

Robustness pass gate:

```text
RCS-AC robustness_retention >= baseline robustness_retention + 0.10
RCS-AC safety_degradation <= baseline safety_degradation
RCS-AC jitter_increase <= baseline jitter_increase
```

강한 주장 기준:

```text
RCS-AC robustness_retention >= baseline robustness_retention + 0.15 ~ 0.20
```

반복 수:

```text
minimum: 3 seeds
recommended: 5 seeds
```

## 10. 구현 순서

### Step 1: 문서 및 config 준비

생성:

```text
docs/14_imitation_rl_execution_plan.md
configs/bc_clean_smoke.yaml
configs/bc_clean_train.yaml
configs/phase3_bc_finetune_smoke.yaml
```

### Step 2: dataset utility 구현

생성:

```text
rcs_ac_first/data/__init__.py
rcs_ac_first/data/expert_dataset.py
```

구현:

- `.npz` save/load
- shape validation
- dtype validation
- train/val split
- PyTorch Dataset
- dataset summary

### Step 3: expert dataset collection 구현

생성:

```text
rcs_ac_first/scripts/collect_expert_dataset.py
```

기능:

- CARLA env bootstrap
- expert action 추출
- observation/auxiliary/action 저장
- metadata 저장
- bad transition filtering
- summary 출력

### Step 4: BC training 구현

생성:

```text
rcs_ac_first/scripts/train_bc.py
```

기능:

- config load
- dataset load
- model build
- supervised action loss
- validation
- CSV logging
- checkpoint save/load

### Step 5: BC evaluation 구현

생성:

```text
rcs_ac_first/scripts/evaluate_bc.py
```

기능:

- checkpoint load
- CARLA rollout
- deterministic action
- action jitter 계산
- route/safety summary 저장

### Step 6: PPO init checkpoint 연결

수정:

```text
rcs_ac_first/scripts/train.py
rcs_ac_first/rl/ppo_agent.py
```

기능:

- `--init-checkpoint`
- model-only load
- strict/non-strict option
- BC checkpoint compatibility
- init metadata logging

### Step 7: BC -> PPO smoke

실행:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_finetune_smoke.yaml `
  --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\bc_clean_smoke\last.pt `
  --total-timesteps 2048 `
  --run-name bc_phase3_finetune_smoke_2048 `
  --device auto
```

### Step 8: stability analyzer 구현

생성:

```text
rcs_ac_first/scripts/analyze_stability.py
```

기능:

- training log 분석
- env log 분석
- finite check
- route/safety/action jitter 계산
- contribution collapse 계산
- robustness response 계산
- PASS/WARN/FAIL 출력

### Step 9: baseline 확장

순서:

1. Simple fusion PPO baseline
2. BC-PPO baseline
3. SAC baseline
4. BC-SAC baseline
5. RCS-AC ablation configs

## 11. 주요 위험과 대응

### 11.1 Expert가 나쁜 행동을 가르치는 문제

위험:

- autopilot이 특정 route에서 stuck
- 불필요하게 급정지
- 교차로에서 이상 행동

대응:

- bad transition mask 저장
- stuck episode 제거
- clean route subset부터 시작
- recovery dataset은 별도 분리

### 11.2 BC policy가 distribution shift에 약한 문제

위험:

- expert가 지나간 상태에서는 잘하지만, 차선에서 벗어나면 복구 못함

대응:

- recovery dataset 추가
- noise augmentation
- DAgger-style data aggregation은 후속 단계로 고려
- RL fine-tuning에서 route deviation penalty와 recovery reward 강화

### 11.3 PPO fine-tuning이 BC policy를 망가뜨리는 문제

위험:

- PPO 초반 entropy/exploration 때문에 steering 흔들림
- reward 설계가 불완전해 BC 주행 습관 붕괴

대응:

- PPO learning rate 낮춤
- entropy coefficient 낮춤
- clip range 보수적으로 설정
- action smoothness penalty 추가
- 필요 시 residual BC loss 사용

### 11.4 Imitation 때문에 강건성 비교가 불공정해지는 문제

위험:

- RCS-AC만 BC warmup을 받으면 비교 불공정

대응:

- 모든 비교군에 같은 expert dataset 적용
- BC training budget 동일화
- BC-only 성능도 별도 보고
- RL fine-tuning budget 동일화

### 11.5 Clean 성능이 포화되어 차이가 안 보이는 문제

위험:

- 모든 모델이 clean에서 95~100% route completion 달성

대응:

- clean은 performance preservation 기준으로 사용
- 핵심 비교는 unseen heavy corruption/weather retention으로 수행
- action jitter, safety degradation, recovery time을 함께 측정

## 12. 최종 산출물

Imitation + RL phase가 완료되면 아래 산출물이 있어야 한다.

문서:

```text
docs/14_imitation_rl_execution_plan.md
docs/15_imitation_rl_result.md
docs/16_robustness_comparison_protocol.md
```

데이터:

```text
data/expert/clean_smoke_2000.npz
data/expert/clean_20000.npz
```

스크립트:

```text
rcs_ac_first/scripts/collect_expert_dataset.py
rcs_ac_first/scripts/train_bc.py
rcs_ac_first/scripts/evaluate_bc.py
rcs_ac_first/scripts/analyze_stability.py
```

체크포인트:

```text
checkpoints/bc_clean_smoke/last.pt
checkpoints/bc_clean_20000/last.pt
checkpoints/bc_phase3_finetune_smoke_2048/last.pt
```

로그:

```text
logs/bc_clean_smoke/bc_training_log.csv
outputs/bc_eval/<run_name>/bc_eval_log.csv
logs/<bc_finetune_run>/training_log.csv
outputs/phase1_core_env/<bc_finetune_run>/env_step_log.csv
```

## 13. 추천 최초 실행 순서

가장 작은 단위의 첫 성공 경로:

```text
1. collect_expert_dataset.py 구현
2. clean_smoke_2000.npz 수집
3. expert_dataset.py 구현 및 dataset reload smoke
4. train_bc.py 구현
5. bc_clean_smoke.yaml로 5 epoch BC 학습
6. evaluate_bc.py로 512-step clean eval
7. train.py에 --init-checkpoint 추가
8. BC checkpoint에서 Phase 3 PPO 2048-step fine-tuning
9. analyze_stability.py로 clean/mild 결과 자동 판정
```

최초 성공 기준:

```text
BC dataset 수집 성공
BC loss 감소
BC checkpoint reload 성공
BC policy CARLA forward step 성공
BC checkpoint 기반 PPO update 성공
NaN/Inf 없음
route_progress > 0
```

## 14. Phase 완료 기준

Imitation + RL phase 완료 기준:

```text
I0:
  clean expert dataset 20000 transitions 이상 수집

I1:
  BC train/val loss finite
  checkpoint save/load 성공

I2:
  BC policy clean 512 또는 2048 step 평가 성공
  route_progress > random policy

I3:
  BC checkpoint에서 Phase 3 PPO fine-tuning 시작 가능
  PPO update 최소 1회 성공

I4:
  clean fine-tuning과 mild corruption fine-tuning 모두 crash 없이 통과
  contribution/reliability/risk/fusion finite

I5:
  최소 BC-PPO vs BC-RCS-AC-PPO 비교 가능
  clean 성능 보존과 corrupted retention을 모두 계산 가능
```

Phase가 끝났다고 말할 수 있는 기준:

```text
BC-RCS-AC-PPO가 clean driving을 유지하면서,
mild 또는 heavy corruption에서 BC-PPO보다 높은 robustness retention을 보인다.
```

논문 실험으로 확장 가능한 기준:

```text
3 seeds 이상
clean/mild/heavy/weather 평가
baseline 대비 retention 향상
ablation 결과 확보
```

## 15. 다음 작업

이 문서 이후 바로 진행할 구현 작업은 다음 순서가 가장 좋다.

```text
1. configs/bc_clean_smoke.yaml 작성
2. rcs_ac_first/data/expert_dataset.py 작성
3. rcs_ac_first/scripts/collect_expert_dataset.py 작성
4. clean_smoke_2000.npz 수집
5. rcs_ac_first/scripts/train_bc.py 작성
6. BC smoke 학습
7. rcs_ac_first/scripts/evaluate_bc.py 작성
8. train.py --init-checkpoint 연결
9. BC -> Phase 3 PPO fine-tuning smoke
```

이 순서대로 가면 "imitation을 사용할 수 있는가"와 "imitation checkpoint가 기존 RCS-AC Phase 3 PPO loop에 실제로 연결되는가"를 가장 빠르게 검증할 수 있다.
