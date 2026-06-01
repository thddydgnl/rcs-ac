# Raw2Drive Exact Implementation Plan for RCS-AC v2

작성일: 2026-05-24

원본 PDF:

```text
RCS-AC-v2/docs/reference/2505.16394v2.pdf
Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving
```

실행용 companion 문서:

```text
RCS-AC-v2/docs/01_goal_driven_implementation_runbook.md
```

이 문서는 설계 원칙과 stage 구조를 정의한다. 실제 Codex goal 진행에서는 companion runbook의 순서, gate, 체크리스트를 우선 따른다.

## 0. 이번 v2 리셋의 목적

기존 `RCS-AC-v2` 폴더는 R2S / RCS-AC 확장 설계 문서만 있었고 실제 구현은 없었다. 이번 리셋에서는 그 내용을 전부 폐기하고, PDF의 Raw2Drive 구조를 기준으로 새 `RCS-AC-v2`를 시작한다.

목표는 다음이다.

1. Raw2Drive의 dual-stream model-based RL 구조를 최대한 원형 그대로 구현한다.
2. 기존 RCS-AC v1/v1.1의 waypoint-controller 안정화 편법을 v2 최종 구조에 섞지 않는다.
3. clean 성능을 먼저 잡는다.
4. clean-capable 공통 backbone 위에서 fusion 방식만 바꾼 ablation을 수행한다.
5. GT/privileged input은 training guidance에만 사용하고 inference policy input에는 넣지 않는다.

중요한 판단:

```text
clean 성능이 잡히지 않은 모델끼리 fusion을 비교하면 실험 의미가 약해진다.
따라서 fusion 비교는 Stage 1/2 Raw2Drive clean gate 이후에만 승격한다.
```

## 1. PDF에서 구현해야 하는 핵심 구조

Raw2Drive는 단순히 PPO나 SAC에 perception encoder를 붙인 구조가 아니다. 핵심은 두 개의 stream이다.

```text
Privileged stream:
  privileged observation
  -> privileged world model
  -> privileged policy

Raw sensor stream:
  raw sensor observation
  -> raw sensor world model
  -> raw sensor policy

Guidance mechanism:
  privileged world model이 raw sensor world model/policy 학습을 안내
```

### 1.1 Privileged stream

PDF 기준 privileged input은 다음이다.

```text
BEV semantic segmentation masks
ego vehicle vector
```

Appendix C.1 기준 privileged BEV mask는 `H x W x C`이고, `C=43` semantic channels를 둔다. 이 채널은 road, lane, ego vehicle, traffic sign/light, pedestrian, vehicle, obstacle 등 static/dynamic object를 표현한다.

v2 구현에서는 이를 다음 contract로 고정한다.

```text
privileged_bev: [B, T, 43, H, W]
privileged_ego: [B, T, D_ego]
action: [B, T] discrete action id
reward: [B, T]
continue: [B, T]
```

Privileged world model은 DreamerV3 계열 RSSM이다.

```text
Encoder:
  e_t = Enc(o_t)

RSSM:
  deterministic state h_t = f(h_{t-1}, a_{t-1}, s_{t-1})
  stochastic state train q(s_t | h_t, e_t)
  stochastic state infer p(s_t | h_t)

Heads:
  decoder head
  reward head
  continue head
```

Privileged policy는 world model rollout 안에서 actor-critic으로 학습한다.

### 1.2 Raw sensor stream

PDF 기준 raw input은 multi-view images + IMU다.

```text
raw_images: [B, T, N_cam, C, H_img, W_img]
raw_imu: [B, T, D_imu]
```

Raw sensor encoder는 BEVFormer 계열을 사용해 grid-shaped BEV feature를 만든다.

```text
raw_bev_feature = RawEncoder(raw_images, raw_imu)
```

Raw sensor world model은 privileged world model과 유사하지만, 중요한 차이가 있다.

1. encoder는 raw sensor용이다.
2. raw world model head는 decoder head만 사용한다.
3. reward/continue head를 raw stream에서 직접 학습하지 않는다.
4. raw policy 학습 시 reward/continue는 privileged world model head guidance로 얻는다.

PDF ablation상 raw stream에서 reward/continue head를 직접 학습하면 수렴이 나빠진다. 따라서 v2에서도 raw world model의 reward/continue head 직접 학습은 금지한다.

### 1.3 Guidance mechanism

Guidance는 두 부분이다.

```text
Rollout Guidance
Head Guidance
```

Rollout Guidance는 raw world model rollout이 privileged world model과 같은 latent dynamics를 유지하게 만든다.

정렬 대상:

```text
encoded state e_t
deterministic state h_t
stochastic state s_t
```

Loss:

```text
L_rollout =
  beta_e * sum_t sum_grid MSE(e_t, ehat_t)
  + sum_t beta_s * KL(s_t, shat_t)
  + sum_t beta_h * MSE(h_t, hhat_t)
```

PDF Appendix C.3 기본 weight:

```text
beta_e = 10
beta_h = 5
beta_s = 10
```

중요 구현 포인트:

```text
stochastic state sampling은 raw stream distribution에서 한 번만 한다.
그 sampled stochastic state를 raw/privileged 양쪽 rollout에 같이 넣어 sampling randomness 누적 오차를 제거한다.
```

Head Guidance는 raw policy training 때 사용한다.

```text
raw policy action
-> raw world model rollout
-> privileged world model도 같은 action으로 rollout
-> privileged reward/continue head에서 reward/continue를 얻음
-> raw policy actor-critic 학습에 사용
```

## 2. Action space는 PDF대로 discrete 39 actions

기존 RCS-AC v1.1은 waypoint + target_speed + pure-pursuit controller였다. Raw2Drive 원형 구현에서는 이 contract를 최종 v2 action으로 쓰지 않는다.

PDF Appendix C.2는 continuous control을 39개 discrete action으로 분해한다. v2는 다음을 구현한다.

```text
Action id -> throttle, brake, steer, reverse
```

필수 파일:

```text
code/rcs_ac_v2/env/discrete_actions.py
tests/test_discrete_actions.py
```

초기 테스트:

1. action 수가 39인지 확인
2. throttle/brake/steer/reverse 범위 확인
3. reverse action이 명시적으로 구분되는지 확인
4. CARLA control 변환이 deterministic인지 확인

## 3. Reward는 Think2Drive/Raw2Drive 계열로 분리

PDF는 reward design을 Think2Drive에서 가져온다고 한다. 현재 저장소에 Think2Drive reward 구현은 없다. 따라서 v2에서는 reward를 다음처럼 관리한다.

```text
code/rcs_ac_v2/env/reward.py
```

구현 순서:

1. CARLA route progress reward
2. collision/offroad/red-light/stuck penalty
3. comfort/efficiency shaping
4. continue flag definition
5. reward audit logger

주의:

```text
reward가 바뀌면 clean 성능과 fusion 비교가 동시에 흔들린다.
따라서 reward contract는 Stage 0에서 freeze하고,
fusion ablation 단계에서는 절대 바꾸지 않는다.
```

## 4. 폴더 구조

현재 v2 폴더는 다음 구현 단위를 전제로 정리한다.

```text
RCS-AC-v2/
  README.md

  docs/
    00_raw2drive_exact_implementation_plan.md

  code/rcs_ac_v2/
    privileged_stream/
      encoder.py
      dataset.py
      world_model.py
      policy.py

    raw_sensor_stream/
      encoder.py
      dataset.py
      world_model.py
      policy.py

    world_model/
      rssm.py
      losses.py
      heads.py
      rollout.py

    policy/
      actor_critic.py
      behavior_loss.py
      distillation.py

    training/
      train_stage1_privileged.py
      train_stage2_raw_world_model.py
      train_stage2_raw_policy.py

    evaluation/
      evaluate_clean.py
      evaluate_fusion_ablation.py
      summarize_runs.py

    data/
      schema.py
      replay_buffer.py
      sequence_sampler.py

    env/
      carla_adapter.py
      discrete_actions.py
      reward.py
      route_scenarios.py

    utils/
      seed.py
      logging.py
      checkpoint.py

  configs/
    stage1_privileged/
    stage2_raw_sensor/
    ablation/
    eval/

  experiments/
    00_preflight_contract/
    01_privileged_world_model/
    02_privileged_policy_clean/
    03_raw_world_model_alignment/
    04_raw_policy_clean/
    05_fusion_ablation/
    06_closed_loop_eval/

  scripts/windows/
  tests/
```

## 5. Stage별 구현 계획

### Stage 0. Contract preflight

목표:

```text
PDF 그대로 구현할 수 있는 observation/action/reward/dataset contract를 먼저 고정한다.
```

작업:

1. discrete 39 action table 구현
2. privileged BEV semantic mask schema 정의
3. raw sensor schema 정의
4. CARLA adapter output schema 정의
5. reward/continue flag definition 문서화 및 테스트
6. synthetic sequence batch 생성
7. replay buffer + sampler smoke test

산출물:

```text
experiments/00_preflight_contract/
configs/eval/preflight_contract.yaml
tests/test_discrete_actions.py
tests/test_data_schema.py
tests/test_reward_contract.py
```

통과 기준:

```text
python -m pytest RCS-AC-v2/tests -q
```

가 통과하고, synthetic batch가 다음 shape를 만족한다.

```text
privileged_bev: [B,T,43,H,W]
raw_images: [B,T,N_cam,3,H_img,W_img]
action: [B,T]
reward: [B,T]
continue: [B,T]
```

### Stage 1. Privileged world model

목표:

```text
GT/privileged BEV semantic mask 기반 world model을 먼저 안정적으로 학습한다.
```

작업:

1. privileged encoder 구현
2. RSSM 구현
3. decoder/reward/continue heads 구현
4. prediction/dynamics/representation loss 구현
5. CARLA privileged trajectory collection
6. local synthetic overfit
7. tiny CARLA dataset overfit

Loss:

```text
L_WM =
  sum_t beta_pred * L_pred
  + beta_dyn * L_dyn
  + beta_rep * L_rep
```

산출물:

```text
configs/stage1_privileged/world_model_smoke.yaml
experiments/01_privileged_world_model/
```

통과 기준:

1. synthetic sequence overfit 성공
2. decoder reconstruction이 collapse하지 않음
3. reward/continue prediction이 logged metric상 finite
4. rollout horizon을 늘려도 latent NaN 없음

### Stage 2. Privileged policy clean

목표:

```text
Privileged world model 안에서 actor-critic policy를 학습해 clean driving seed를 만든다.
```

작업:

1. privileged actor-critic 구현
2. lambda return / critic loss 구현
3. actor entropy regularization 구현
4. world model rollout 기반 policy update 구현
5. CARLA closed-loop clean eval

PDF Appendix E의 actor/critic 구조를 따른다.

산출물:

```text
configs/stage1_privileged/policy_clean.yaml
experiments/02_privileged_policy_clean/
```

Clean gate:

```text
128-step single-start: collision/offroad/stuck = 0
512-step primary subset: collision/offroad/stuck = 0
2048-step subset: goal 또는 route progress 안정
```

주의:

```text
privileged policy는 final inference model이 아니다.
raw stream을 학습시키기 위한 teacher/guide다.
```

### Stage 3. Raw sensor world model with rollout guidance

목표:

```text
raw sensor world model이 privileged world model latent dynamics와 정렬되도록 학습한다.
```

작업:

1. raw sensor encoder 구현
2. BEVFormer-style grid BEV feature output 구현
3. privileged encoder feature와 spatial-temporal alignment
4. deterministic state alignment
5. stochastic state KL alignment
6. single-sample stochastic rollout 구현
7. raw world model decoder-only head 구현

금지:

```text
raw reward head 직접 학습 금지
raw continue head 직접 학습 금지
```

산출물:

```text
configs/stage2_raw_sensor/raw_world_model_alignment.yaml
experiments/03_raw_world_model_alignment/
```

통과 기준:

1. encoder MSE 감소
2. deterministic state MSE 감소
3. stochastic KL 안정
4. decoder output collapse 없음
5. multi-step rollout에서 privileged/raw latent divergence가 관리됨

### Stage 4. Raw sensor policy with head guidance

목표:

```text
raw sensor policy를 raw world model rollout 안에서 학습하되,
reward/continue는 privileged world model head guidance를 사용한다.
```

작업:

1. raw actor-critic 구현
2. privileged policy checkpoint 로드
3. action distribution distillation 구현
4. head guidance reward/continue 연결
5. raw world model rollout actor-critic 학습
6. CARLA clean closed-loop eval

산출물:

```text
configs/stage2_raw_sensor/raw_policy_clean.yaml
experiments/04_raw_policy_clean/
```

Clean gate:

```text
128-step clean: 0/0/0
512 primary9 clean: 9/9 safety pass
2048 subset clean: goal 또는 route progress no-regression
```

여기까지 통과해야 fusion ablation으로 넘어간다.

### Stage 5. Fusion-only ablation

목표:

```text
같은 Raw2Drive dual-stream training/eval contract에서 fusion 방식만 바꿔 비교한다.
```

비교 대상:

1. `uniform_fusion`
2. `learned_attention_fusion`
3. `rcs_ac_shapley_style_fusion`
4. `raw2drive_default_fusion`

Fusion을 바꿀 수 있는 위치:

```text
raw sensor encoder의 source feature aggregation
raw BEV grid feature aggregation
policy latent input aggregation
```

바꾸면 안 되는 것:

```text
observation schema
action table
reward
continue definition
world model rollout horizon
training budget
eval starts/seeds/weather
controller/control application
```

산출물:

```text
configs/ablation/fusion_uniform.yaml
configs/ablation/fusion_attention.yaml
configs/ablation/fusion_rcs_ac_shapley.yaml
configs/ablation/fusion_raw2drive_default.yaml
experiments/05_fusion_ablation/
```

비교 metric:

```text
clean safety
route completion/progress
success rate
infraction score
driving score
mean speed
raw action jitter
steer jitter
fusion weight by source
fusion weight change under corruption
latent alignment error
world model rollout loss
```

### Stage 6. Closed-loop eval and stress

목표:

```text
clean-capable fusion variants만 CARLA closed-loop stress로 승격한다.
```

조건:

1. clean
2. RGB noise/blur
3. LiDAR sparsity/dropout
4. weather sensor severe
5. combined severe

산출물:

```text
configs/eval/clean_512_primary9.yaml
configs/eval/clean_2048_subset.yaml
configs/eval/stress_512_primary9.yaml
experiments/06_closed_loop_eval/
```

## 6. Clean 성능을 잡기 위한 운영 원칙

이 프로젝트에서 clean 성능이 어려웠던 이유는 model-free RL 또는 raw sensor policy를 직접 최적화할 때 perception/control/reward/dynamics 문제가 동시에 섞였기 때문이다.

Raw2Drive 방식은 이를 다음처럼 분리한다.

```text
privileged structured BEV로 world model 먼저 안정화
privileged policy를 world model 안에서 먼저 학습
raw world model은 privileged latent를 따라가게 학습
raw policy는 privileged reward/continue head guidance로 학습
```

따라서 v2 clean 성능 실험에서 우선순위는 fusion이 아니다.

우선순위:

1. action table과 reward가 정확한가
2. privileged policy가 clean을 통과하는가
3. raw world model이 privileged world model과 latent alignment 되는가
4. raw policy가 clean을 통과하는가
5. 그 다음 fusion을 비교하는가

Clean이 안 잡히면 확인 순서:

1. discrete action table이 CARLA control에 잘 매핑되는지
2. reward가 route progress를 제대로 밀어주는지
3. continue flag가 episode termination과 일치하는지
4. privileged BEV semantic mask 좌표계가 맞는지
5. privileged world model decoder/reward/continue heads가 collapse하지 않는지
6. raw/privileged stochastic state sampling이 동일 sample을 공유하는지
7. raw reward/continue head를 실수로 학습하고 있지 않은지
8. policy가 privileged policy distillation 없이 처음부터 흔들리고 있지 않은지

## 7. 구현 가능성과 리스크

가능성:

```text
구현은 가능하다.
현재 v1.1의 CARLA adapter, BEV rasterization, clean/stress eval 경험을 재사용하면
Raw2Drive 데이터/평가 인프라를 빠르게 만들 수 있다.
```

하지만 PDF 그대로 구현은 작지 않다.

주요 리스크:

1. BEVFormer-style raw encoder 구현/학습 비용
2. privileged 43-channel semantic BEV 생성 정확도
3. DreamerV3/RSSM 안정화 비용
4. CARLA trajectory collection 시간
5. reward/continue head quality
6. raw-policy world-model rollout의 train/eval gap
7. discrete action policy가 기존 waypoint-controller보다 clean에서 불안정할 가능성

따라서 실험은 반드시 작은 gate부터 한다.

```text
synthetic overfit
tiny CARLA replay overfit
128-step closed-loop
512-step primary subset
2048-step subset
```

## 8. 현재 RCS-AC-v1.1과의 관계

v1.1은 참고용이다. 다음은 재사용 가능하다.

```text
CARLA adapter 경험
BEV coordinate convention
route/start/seed eval protocol
stress condition definitions
logging/summarization scripts
```

다음은 v2 최종 구현에 그대로 가져오지 않는다.

```text
waypoint + target_speed action contract
pure-pursuit controller를 policy output 후단으로 두는 구조
r8u Shapley-only checkpoint
RCS-AC v1.1 specific sensor-health shortcut
```

단, preflight에서 sanity baseline으로 v1.1 결과를 비교표에 둘 수는 있다. 이 경우 문서에 반드시 `reference only`라고 표시한다.

## 9. 첫 코딩 턴 권장 작업

다음 코딩 턴에서는 Stage 0만 한다.

작업 목록:

1. `code/rcs_ac_v2/env/discrete_actions.py`
2. `code/rcs_ac_v2/data/schema.py`
3. `code/rcs_ac_v2/data/replay_buffer.py`
4. `code/rcs_ac_v2/env/reward.py`
5. `tests/test_discrete_actions.py`
6. `tests/test_data_schema.py`
7. `tests/test_reward_contract.py`
8. `configs/eval/preflight_contract.yaml`

Stage 0 통과 후에만 RSSM/world model 구현으로 넘어간다.

## 10. 최종 완료 정의

RCS-AC-v2 1차 완료는 다음 조건을 만족해야 한다.

1. PDF의 dual-stream training pipeline이 코드로 존재한다.
2. privileged world model/policy가 clean gate를 통과한다.
3. raw sensor world model이 rollout guidance loss로 학습된다.
4. raw sensor policy가 head guidance로 학습된다.
5. final inference는 raw sensor input만 사용한다.
6. discrete 39 action contract를 사용한다.
7. fusion ablation은 clean-capable variants에 대해서만 수행된다.
8. 각 fusion variant는 동일한 action/reward/eval/training-budget 조건으로 비교된다.
9. 결과 문서는 clean과 stress를 분리해 보고한다.
10. GT/privileged leakage audit가 통과한다.

## 11. 실험 기록 원칙

모든 run은 다음 정보를 남긴다.

```text
run_name
git commit
config path
checkpoint path
CARLA version
town/route/start/seed/weather
action table version
reward contract version
world model checkpoint
policy checkpoint
fusion mode
training budget
clean metrics
stress metrics
failure reason
```

실험 결과는 각 stage 폴더에 둔다.

```text
experiments/00_preflight_contract/
experiments/01_privileged_world_model/
experiments/02_privileged_policy_clean/
experiments/03_raw_world_model_alignment/
experiments/04_raw_policy_clean/
experiments/05_fusion_ablation/
experiments/06_closed_loop_eval/
```

## 12. 한 줄 결론

이번 v2는 기존 RCS-AC-v2 계획을 버리고 Raw2Drive 원형 구현으로 간다. clean 성능은 fusion이 아니라 privileged world model, rollout guidance, head guidance로 먼저 잡는다. fusion 비교는 clean-capable Raw2Drive 공통 backbone 위에서만 수행한다.
