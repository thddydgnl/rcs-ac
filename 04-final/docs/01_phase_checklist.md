# RCS-AC-first 단계별 구현 체크리스트

작성일: 2026-04-26

이 체크리스트는 `00_three_phase_master_plan.md`를 실제 작업 순서로 쪼갠 문서다.

현재 읽는 법:

- 각 Phase의 "공식 완료 상태"와 하단 "공식 완료 기록"이 현재 판단의 기준이다.
- 중간의 긴 세부 체크리스트는 초기 구현 당시의 작업 분해를 보존한 archival log다.
- 세부 항목이 비어 있어도 공식 완료 상태가 우선한다. 최신 실행 방법은 루트 `README.md`를 기준으로 한다.

## Phase 1: RCS-AC Core

### 0. Phase 1 공식 완료 상태

- [x] package scaffold, runtime contract, action contract 구현
- [x] RGB/LiDAR/ego/route encoder 구현
- [x] fusion gate와 ego/route safety floor 구현
- [x] GRU actor-critic, 2D action mean, critic value, state-independent std 구현
- [x] scalar auxiliary head와 Phase 2/3 placeholder loss 구현
- [x] Phase 1.1 RGB perception auxiliary head와 dense mask loss 구현
- [x] bounded/squashed normal distribution 구현
- [x] rollout buffer, GAE, PPO update, checkpoint save/load 구현
- [x] recurrent hidden state를 rollout buffer에 저장하고 PPO update에서 재사용
- [x] `evaluate_actions()` 경로 추가
- [x] local compile, smoke, checkpoint 검증 통과
- [x] CARLA 128-step probe 통과
- [x] CARLA 512-step smoke 통과
- [x] CARLA 2048-step smoke 통과
- [x] `training_log.csv`, `env_step_log.csv`, `last.pt` 생성 확인
- [x] Phase 1 결과 문서 갱신 및 Phase 2 진입 가능 판정

아래 세부 체크리스트는 초기 구현 항목 기록으로 남긴다. 공식 완료 판단은 위 항목과 `docs/10_phase1_core_result.md`를 기준으로 한다.

### 1. 폴더와 패키지

- [ ] `RCS-AC-first/rcs_ac_first` 생성
- [ ] `rcs_ac_first/__init__.py` 생성
- [ ] `models`, `rl`, `runtime`, `scripts`, `configs`, `tests` 하위 폴더 생성
- [ ] 각 하위 폴더에 `__init__.py` 생성
- [ ] `README.md`에 실행 방법 추가

### 2. Config

- [ ] `runtime/config.py` 생성
- [ ] `RCSACFirstConfig` dataclass 추가
- [ ] model config 필드 추가
  - [ ] `obs_dim`
  - [ ] `action_dim`
  - [ ] `rgb_shape`
  - [ ] `lidar_shape`
  - [ ] `ego_dim`
  - [ ] `route_dim`
  - [ ] `feature_dim`
  - [ ] `fusion_dim`
  - [ ] `recurrent_hidden_dim`
  - [ ] `recurrent_layers`
  - [ ] `min_std`
  - [ ] `max_std`
- [ ] loss config 필드 추가
- [ ] fusion config 필드 추가
- [ ] shapley config 필드는 Phase 1에서 0/off 기본값으로 추가

### 3. Observation contract

- [ ] `runtime/observation.py` 생성
- [ ] `ObservationSliceSpec` 추가
- [ ] `PolicyObservationSpec` 추가
- [ ] `build_policy_observation_spec()` 구현
- [ ] `split_flat_observation()` 구현
- [ ] NaN/inf 검사
- [ ] shape mismatch 에러 메시지 정리
- [ ] dummy split test 작성

### 4. Action contract

- [ ] `runtime/action.py` 생성
- [ ] `stage4_action_v2` 정의
- [ ] action order `("steer", "longitudinal")` 정의
- [ ] action bounds `[(-1, 1), (-1, 1)]` 정의
- [ ] optional 3-action contract placeholder 추가
- [ ] action dim validation 추가

### 5. Encoders

- [ ] `models/encoders.py` 생성
- [ ] `RGBEncoder` 구현
- [ ] `LiDAREncoder` 구현
- [ ] `VectorEncoder` 구현
- [ ] output feature dim 통일
- [ ] encoder diagnostics 추가
  - [ ] input mean/std
  - [ ] feature L1 mean
  - [ ] feature finite check
- [ ] dummy encoder forward test

### 6. Fusion gate skeleton

- [ ] `models/fusion_gate.py` 생성
- [ ] `FusionGateConfig` 추가
- [ ] `FusionGateOutput` 추가
- [ ] simple gate MLP 구현
- [ ] `reliability=None`일 때 all ones 처리
- [ ] `contribution=None`일 때 uniform 처리
- [ ] `risk_context=None`일 때 zero 처리
- [ ] softmax temperature 적용
- [ ] ego floor 적용
- [ ] route floor 적용
- [ ] floor 후 재정규화
- [ ] fusion diagnostics 추가
- [ ] weight sum test

### 7. Recurrent actor-critic

- [ ] `models/recurrent_actor_critic.py` 생성
- [ ] `GRU` backbone 구현
- [ ] hidden state init 구현
- [ ] hidden state reset 구현
- [ ] actor head 구현
- [ ] critic head 구현
- [ ] `log_std` parameter 구현
- [ ] `get_std()` 구현
- [ ] action mean tanh 적용
- [ ] value squeeze shape 확인
- [ ] dummy recurrent forward test

### 8. Top-level model

- [ ] `models/rcs_ac_model.py` 생성
- [ ] `RCSACFirstModel` 구현
- [ ] observation split 연결
- [ ] encoder 호출 연결
- [ ] fusion gate 연결
- [ ] recurrent actor-critic 연결
- [ ] `forward_policy_outputs()` 구현
- [ ] `forward()` 구현
- [ ] `predict_auxiliary_targets()` 구현
- [ ] `get_diagnostics()` 구현
- [ ] `get_stage_b_warmup_modules()` 구현
- [ ] `get_auxiliary_losses()` 구현

### 9. Phase 1 losses

- [ ] `rl/losses.py` 생성
- [ ] PPO clipped actor loss 구현
- [ ] value loss 구현
- [ ] entropy loss 구현
- [ ] lane offset aux loss 구현
- [ ] heading error aux loss 구현
- [ ] placeholder loss keys 추가
  - [ ] `reliability_loss`
  - [ ] `risk_regularization_loss`
  - [ ] `shapley_alignment_loss`
  - [ ] `dropout_consistency_loss`
- [ ] loss finite test

### 10. Distribution

- [ ] `rl/distributions.py` 생성
- [ ] bounded/squashed normal 구현
- [ ] sample 구현
- [ ] log_prob 구현
- [ ] entropy 구현
- [ ] action clamp 확인
- [ ] distribution shape test

### 11. Rollout/PPO

- [ ] `rl/rollout_buffer.py` 생성 또는 existing buffer adapter 작성
- [ ] observation storage
- [ ] action storage
- [ ] log_prob storage
- [ ] value storage
- [ ] reward/done storage
- [ ] aux target storage
- [ ] GAE 계산
- [ ] minibatch 생성
- [ ] `rl/ppo_agent.py` 생성
- [ ] `act()` 구현
- [ ] `evaluate_actions()` 구현
- [ ] `update()` 구현
- [ ] checkpoint save/load 연결

### 12. Scripts

- [ ] `scripts/smoke_check.py` 생성
- [ ] config load
- [ ] model build
- [ ] dummy batch forward
- [ ] dummy backward
- [ ] checkpoint save/load
- [ ] `scripts/train.py` 생성
- [ ] 기존 `attention` train loop wrapper 또는 adapter 선택
- [ ] CLI arg 추가
- [ ] logs/checkpoints/output path 생성

### 13. Phase 1 검증

- [ ] `py -3.12 -m compileall` 통과
- [ ] dummy model forward 통과
- [ ] dummy loss backward 통과
- [ ] smoke_check 통과
- [ ] CARLA 없이 가능한 PPOAgent 생성 통과
- [ ] CARLA smoke 512 step 통과
- [ ] CARLA smoke 2048 step 통과
- [ ] checkpoint 생성 확인
- [ ] training log 생성 확인
- [ ] NaN 없음 확인

## Phase 2: Reliability + Risk-aware Fusion

상세 실행계획: `docs/04_phase2_reliability_risk_execution_plan.md`
결과 문서: `docs/11_phase2_reliability_risk_result.md`
공식 상태: **완료, Phase 3 진입 가능**

### 1. Auxiliary schema

- [x] `quality_stats` schema 정리
- [x] `risk_stats` schema 정리
- [x] `route_commitment_stats` schema 정리
- [x] `sensor_dropout_mask` schema 정리
- [x] auxiliary observation adapter 구현
- [x] missing auxiliary default 처리

### 2. Reliability heads

- [x] `models/reliability.py` 생성
- [x] 모달별 reliability prior/head 구현
- [x] `RCSACReliabilityModule` 구현
- [x] RGB quality slice 정의
- [x] LiDAR quality slice 정의
- [x] Ego quality slice 정의
- [x] Route quality slice 정의
- [x] q shape `[B, 4]` 확인
- [x] q range `[0, 1]` 확인

### 3. Risk/context

- [x] `models/risk_context.py` 생성
- [x] `RiskContextHead` 구현
- [x] context embedding shape `[B, context_dim]`
- [x] risk scalar shape `[B]`
- [x] risk scalar range `[0, 1]`
- [x] risk diagnostics 추가

### 4. Corruption

- [x] corruption config 추가
- [x] RGB dropout 구현
- [x] RGB blur/downscale placeholder 또는 실제 구현
- [x] LiDAR point drop 구현
- [x] Ego noise 구현
- [x] Route corruption 구현
- [x] corruption metadata 생성
- [x] pseudo-label 생성

### 5. Fusion 연결

- [x] fusion gate에 reliability 연결
- [x] fusion gate에 risk scalar 연결
- [x] fusion gate에 risk context 연결
- [x] contribution은 uniform 유지
- [x] floor 적용 유지
- [x] diagnostics 기록

### 6. Phase 2 losses

- [x] reliability supervision loss 구현
- [x] risk-aware regularization 구현
- [x] config weight 연결
- [x] loss log field 추가
- [x] NaN guard 추가

### 7. Phase 2 검증

- [x] clean dummy forward 통과
- [x] corrupted dummy forward 통과
- [x] reliability loss finite
- [x] risk regularization finite
- [x] weak corruption smoke run 통과
- [x] clean run 성능 붕괴 없음 확인
- [x] RGB corruption 시 RGB q 하락 확인
- [x] LiDAR corruption 시 LiDAR q 하락 확인
- [x] route floor 유지 확인

## Phase 3: Counterfactual Shapley Full RCS-AC

### 1. Masked critic

- [ ] `models/masked_critic.py` 생성
- [ ] modality mask format 정의
- [ ] mask embedding 구현
- [ ] masked feature builder 구현
- [ ] masked critic head 구현
- [ ] value shape 확인

### 2. Shapley estimator

- [ ] `rl/shapley.py` 생성
- [ ] subset sampler 구현
- [ ] empty/full subset option 구현
- [ ] `V(S union m) - V(S)` 계산
- [ ] contribution 평균 계산
- [ ] contribution softmax normalization
- [ ] temperature 적용
- [ ] detached contribution option 구현

### 3. Full fusion

- [ ] contribution을 fusion gate에 연결
- [ ] reliability + contribution + risk 동시 사용
- [ ] contribution diagnostics 추가
- [ ] contribution collapse guard 검토

### 4. Phase 3 losses

- [ ] masked critic loss 구현
- [ ] Shapley alignment MSE 구현
- [ ] optional KL 구현
- [ ] dropout consistency batch 생성
- [ ] action mean consistency 구현
- [ ] optional distribution KL 구현
- [ ] loss weight config 연결

### 5. Evaluation

- [ ] `scripts/evaluate_shapley.py` 생성
- [ ] 16 subset sweep 구현
- [ ] exact-ish Shapley 계산
- [ ] contribution timeseries 저장
- [ ] fusion vs contribution 저장
- [ ] scenario summary 저장
- [ ] plot script 연결

### 6. Phase 3 검증

- [ ] masked critic dummy forward 통과
- [ ] approximate Shapley dummy batch 통과
- [ ] contribution shape `[B, 4]`
- [ ] contribution sum ~= 1
- [ ] Shapley loss finite
- [ ] dropout consistency finite
- [ ] full smoke run 통과
- [ ] exact-ish eval 통과
- [ ] contribution이 uniform 고정이 아닌지 확인
- [ ] contribution이 단일 모달로 붕괴하지 않는지 확인

## 최종 통합 체크

- [x] Phase 1 core 통과
- [x] Phase 2 reliability/risk 통과
- [x] Phase 3 Shapley/full robustness 통과
- [ ] clean weather baseline 비교
- [ ] mild weather 비교
- [ ] heavy weather 비교
- [ ] sensor dropout 비교
- [ ] contribution report 생성
- [x] 최종 README 업데이트
- [ ] 실험 결과 문서화

위 미완료 항목은 구현 wiring이 아니라 안정화/논문용 비교 실험 범위다.

## Phase 3 공식 완료 기록

상세 실행계획: `docs/05_phase3_counterfactual_shapley_execution_plan.md`
결과 문서: `docs/12_phase3_counterfactual_shapley_result.md`
공식 상태: **구현 완료, 안정화 라운드 진입 가능**

- [x] `runtime/modality_mask.py` 구현
- [x] `models/masked_critic.py` 구현 및 dummy forward 검증
- [x] `rl/shapley.py` 구현
- [x] 16 subset exact-ish Shapley 계산 구현
- [x] current critic 기반 `V(S union m) - V(S)` 계산 구현
- [x] contribution softmax normalization 및 row sum 검증
- [x] fusion gate에 contribution 및 modality mask 연결
- [x] Shapley alignment MSE loss 구현 및 PPO update 연결
- [x] representation-level dropout consistency loss 구현 및 PPO update 연결
- [x] `scripts/evaluate_shapley.py` 구현
- [x] Phase 3 local smoke 통과
- [x] Phase 2 regression smoke 통과
- [x] CARLA clean 512-step smoke 통과
- [x] CARLA mild corruption 2048-step smoke 통과
- [x] exact-ish Shapley eval 256-step 통과
- [x] `training_log.csv`, `env_step_log.csv`, `last.pt` 생성 확인
- [x] checkpoint reload 확인
