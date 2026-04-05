# Toy RCS-AC Master Plan

작성일: 2026-03-27

## 1. 전체 목표

이 toy env에서 먼저 확인할 것은 절대 성능보다도 아래 구조적 동작이다.

- reliability head가 망가진 모달을 낮게 평가하는가
- fusion weight가 정상 모달 쪽으로 이동하는가
- dropout consistency가 정책 흔들림을 줄이는가
- risk가 높을 때 더 안전한 모달 조합을 쓰는가

즉 이 플랜의 목적은 "차를 잘 모느냐"보다 "RCS-AC 구조가 의도대로 작동하느냐"를 작은 환경에서 먼저 검증하는 것이다.

## 2. 전체 실행 로드맵

### Step 1. Toy 환경 목표와 상태 정의 고정

- 2D 목표 도달 환경
- 성공: goal 도달
- 실패: obstacle 충돌
- 종료: success / collision / boundary out / max step

정책 입력용 4개 모달:

- vision proxy
- distance proxy
- ego proxy
- route proxy

보조 헤드 입력용 상태:

- vision 품질 점수
- distance 품질 점수
- ego noise 여부
- route corruption 여부
- speed
- heading error
- obstacle proximity
- goal distance

### Step 2. Reward / 종료 조건 단순 설계

- goal 접근 보상
- goal 도달 큰 보상
- collision 큰 패널티
- boundary out 패널티
- action jerk 작은 패널티
- idle 패널티

### Step 3. Stage 0 환경 검증

- random policy
- heuristic policy
- observation range 확인
- corruption on/off 안정성 확인
- 로그 저장 검증

필수 로그:

- episode reward
- success / collision / timeout
- min goal distance
- modality corruption flag
- quality stats
- risk stats

### Step 4. Baseline 먼저 구축

#### 4-1. Baseline A

- ego + route only

#### 4-2. Baseline B

- 4모달 early fusion PPO

#### 4-3. Baseline C

- 4모달 attention fusion PPO

### Step 5. RCS-AC Version 1

- 4 encoder
- simple attention fusion
- recurrent actor-critic

### Step 6. Stage 1 Warm-up

- actor / value / entropy 중심
- reliability loss는 약하게
- shapley / consistency / risk는 off 또는 매우 낮게

### Step 7. RCS-AC Version 2

- reliability head 추가
- dropout consistency 추가

여기서 확인할 것:

- corrupted 모달의 reliability가 실제로 낮아지는가
- full vs corrupted 정책 차이가 줄어드는가
- 망가진 모달 대신 다른 모달 weight가 올라가는가

### Step 8. Stage 2 Mild Robustness Learning

- actor / value / entropy / reliability / consistency 활성화
- shapley / risk는 낮게 시작
- mild corruption 중심 학습

### Step 9. RCS-AC Version 3

- counterfactual Shapley estimator
- Shapley alignment loss

### Step 10. RCS-AC Version 4

- risk-aware regularization
- high-risk 상황에서 더 보수적 fusion

### Step 11. Stage 3 Full Learning

- actor
- value
- entropy
- reliability
- shapley
- consistency
- risk

중간~강한 corruption과 복수 모달 손상을 포함한 full 학습

### Step 12. 최종 평가 / Ablation

필수 비교:

- Baseline A
- Baseline B
- Baseline C
- RCS-AC full

필수 ablation:

- w/o Reliability
- w/o Shapley
- w/o Risk
- w/o Consistency
- w/o Recurrent

핵심 평가지표:

- success rate
- collision rate
- timeout rate
- avg return
- avg progress to goal
- normal-to-adverse degradation

## 3. 현재 프로젝트 위치

현재 위치는 "구현 기준"과 "검증 기준"을 나눠서 봐야 한다.

### 3-1. 구현 기준 현재 위치

이미 리포지토리에는 아래 단계에 해당하는 구현과 실행 스크립트가 존재한다.

- Stage 0 환경 검증:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/stage0_env_validation.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/test_rcs_toy_nav_env.py`
- Baseline 구축:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/train_baseline.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_baseline_comparison.py`
- RCS-AC v1:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/rcs_ac_v1`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/train_rcs_ac_v1.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_rcs_ac_v1_multi_seed.py`
- Stage 1 / reliability warm-up:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/stage1_warmup`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_transition_to_stage1.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_stage1_reliability_calibration.py`
- Stage 2 mild robustness:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_stage2_mild_pilot.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_stage2_full.py`
- Version 3 Shapley 계열:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_shapley_pilot.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1_shapley_pilot.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1b_shapley_pilot.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/v3_shapley_helpers.py`
- late fusion / selective gate calibration 계열:
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_7_reliability_bias_pilot.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_7c_dual_teacher_gated_bias_pilot.py`
  - `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_7f2_13_anchor_cancel_selector_rescue_pilot.py`

즉 구현 스택만 보면 현재 프로젝트는 Step 9~11 이후의 late calibration 단계까지 이미 들어와 있다.

### 3-2. 검증 기준 현재 위치

반면 "문서의 핵심 차별점이 toy env에서 실제로 증명됐는가" 기준으로 보면 현재 위치는 더 앞선다.

현재까지의 최신 신뢰 가능한 기준점은:

- `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_7f2_13_anchor_cancel_selector_rescue_pilot_smoke/summary/v3_7f2_13_anchor_cancel_selector_rescue_pilot_score_summary.csv`

그리고 이 기준점은 f2.13 wiring bug 수정 이후의 결과만 신뢰한다.

현재 상태를 문서 단계로 매핑하면:

- Step 1~6:
  - 구현됨
  - 대체로 통과한 것으로 볼 수 있음
- Step 7~8:
  - reliability / consistency는 구현되어 있음
  - 그러나 "corrupted 모달이 실제로 낮아지고, fusion이 healthy modal로 이동한다"는 구조적 증명은 아직 부족함
- Step 9~10:
  - shapley / gated fusion / risk-aware 계열 코드는 깊게 들어와 있음
  - 하지만 selective gate actuation이 아직 구조적으로 증명되지 않아 해석 가능 완성 단계로 보긴 어려움
- Step 11~12:
  - full stack와 ablation의 뼈대는 있음
  - 그러나 현재는 full 평가보다 앞서, fusion control의 작동 증명이 더 우선임

따라서 "구현 위치"는 Step 9~11에 가깝지만,
"검증 위치"는 실질적으로 Step 7~9 사이, 정확히는 다음 질문을 아직 완전히 해결하지 못한 상태다.

- reliability가 실제로 corruption을 읽는가
- final fusion/gate가 실제로 그 신호를 사용해 weight를 이동시키는가
- consistency가 policy 흔들림을 줄이는가

## 4. 버그 수정이 과거 실험 해석에 미친 영향

f2.13 bugfix 전에는 아래 문제가 있었다.

- `cancel_temperature`, `selector_temperature`, `rescue_selector_floor`가 config에서 fusion runtime으로 주입되지 않음
- `mild_gate_min`이 candidate별로 forward에 반영되지 않음
- smoke에서 `vision_gate_clean_near_source_ratio`가 사실상 0/1처럼 거칠게 동작함

이 영향으로:

- f2.13 수정 전 결론은 신뢰하지 않는다
- f2.11~f2.12의 rescue-heavy vs balanced 비교 해석도 일부 오염됐을 가능성이 있다
- 다만 f2.3~f2.10에서 반복 관찰된 큰 방향성
  - 단순 residual만으로는 약함
  - adapter readout은 약함
  - output scale은 wrong-sign transfer를 만들 수 있음
  - common cancel 분리는 유의미함
  는 여전히 참고 가치가 있다

## 5. 최신 신뢰 가능한 기준점에서 본 현재 병목

bugfix 이후 f2.13 기준:

- balanced
  - `attention_gate_cancel_temperature = 4.0`
  - `attention_gate_selector_temperature = 4.0`
  - `attention_gate_rescue_selector_floor = 0.45`
  - `attention_gate_mild_gate_min = 0.08`
  - `attention_bias_gate_selector_prob_mean_vision ≈ 0.355`
  - `attention_bias_gate_selector_active_mean_vision = 0.0`
- sharp
  - `attention_gate_cancel_temperature = 5.0`
  - `attention_gate_selector_temperature = 6.0`
  - `attention_gate_rescue_selector_floor = 0.55`
  - `attention_gate_mild_gate_min = 0.10`
  - `attention_bias_gate_selector_prob_mean_vision ≈ 0.282`
  - `attention_bias_gate_selector_active_mean_vision = 0.0`

현재 핵심 병목은 아래로 정리된다.

1. clean을 source 근처로 보내는 것 자체는 절대 주병목이 아니다
- `vision_gate_clean_source_margin`이 최신 smoke에서 양수다
- 즉 clean 정렬은 적어도 기준선 아래로 들어올 수 있다

2. 현재 가장 큰 병목은 selector separation 부재다
- `vision_selector_prob_mild_minus_clean`가 사실상 0에 가깝다
- `selector_active_clean == 0`, `selector_active_mild == 0`이라 rescue가 아예 작동하지 않는다

3. 그래서 지금 단계는 "gate magnitude" 문제가 아니라 "mild-only decision boundary" 문제다

## 6. 다음 실험에서 목표로 해야 할 것

다음 실험의 목표는 더 큰 구조를 여는 것이 아니라, Step 7~9의 핵심 검증 질문으로 다시 정렬해야 한다.

### 6-1. 1차 목표

다음 실험은 "selector proof round"가 되어야 한다.

즉 먼저 아래를 증명해야 한다.

- corrupted / mild sample에서만 selector가 켜질 수 있는가
- clean에서는 selector가 거의 0에 머무는가
- selector gap이 생긴 뒤에야 rescue/value를 논할 수 있는가

권장 1차 성공 기준:

- `vision_selector_prob_clean_mean < 0.15`
- `vision_selector_prob_mild_minus_clean > 0.10`
- `vision_selector_active_clean_mean == 0`
- `vision_selector_active_mild_mean > 0`
- `vision_mild_rescue_logit_mild_minus_clean > 0.02`

### 6-2. 2차 목표

selector가 분리되면 그다음에야 아래를 본다.

- final gate가 실제로 mild 방향으로 벌어지는가
- normal 성능을 깨지 않고 corruption 시 degradation을 줄이는가

권장 2차 성공 기준:

- `vision_final_gate_mild_minus_clean > 0`
- `normal_to_mild_success_drop`가 baseline보다 작거나 비슷
- action consistency drift가 baseline/ablation보다 낮음

### 6-3. 지금 목표로 잡지 말아야 할 것

다음 라운드에서 아래를 주목표로 잡으면 안 된다.

- cancel을 더 세게 만드는 것
- rescue alpha만 키우는 것
- floor만 낮춰서 억지로 active를 켜는 것
- 또 다른 큰 branch를 추가하는 것

이유는 현재 문제의 본질이 magnitude 부족이 아니라 selector non-separability이기 때문이다.

## 7. 현 시점의 한 줄 위치 판정

현재 프로젝트는:

- 구현 관점: `Step 9~11` 수준까지 와 있음
- 검증 관점: `Step 7~9` 사이에서 멈춰 있음

조금 더 정확히 말하면:

> "reliability / contribution / risk-aware fusion을 모두 구현해 본 상태이지만, toy env에서 그 fusion control이 실제로 selective하게 작동한다는 증명은 아직 부족하다."

## 8. 현 시점의 한 줄 목표 판정

다음 실험의 목표는:

> "final gate 성능을 더 키우는 것"이 아니라, "mild-only selector가 실제로 분리 가능한지"를 먼저 증명하는 것이다.

그 증명이 되기 전에는 Step 10~12의 risk-aware / full ablation 해석을 밀어붙이지 않는다.

## 9. 앞으로의 8단계

아래 8단계는 "완성까지 가는 남은 순서"를 현재 상태 기준으로 다시 정리한 것이다.

### Phase 1. Selector Proof

목표:

- clean과 mild corruption을 selector가 실제로 구분하는지 증명한다

핵심 확인:

- clean에서는 selector가 낮게 유지되는가
- mild에서는 selector가 높아지는가
- selector active가 clean에는 거의 안 켜지고 mild에만 켜지는가

이 단계가 끝났다고 보는 기준:

- `vision_selector_prob_clean_mean`이 낮다
- `vision_selector_prob_mild_minus_clean`가 의미 있게 양수다
- `vision_selector_active_clean_mean == 0`
- `vision_selector_active_mild_mean > 0`

### Phase 2. Fusion Shift Proof

목표:

- selector가 켜졌을 때 실제로 손상 모달 weight가 줄고 healthy modal weight가 늘어나는지 증명한다

핵심 확인:

- corrupted vision이나 distance weight가 내려가는가
- ego / route 같은 안전 모달 weight가 보정되는가
- final fusion이 reliability/selector 변화와 같은 방향으로 움직이는가

이 단계가 끝났다고 보는 기준:

- corruption 시 fused weight 분포가 healthy modal 쪽으로 이동한다
- 이 이동이 단순 offset이 아니라 selector/reliability 신호와 같이 발생한다

### Phase 3. Consistency Proof

목표:

- 같은 상태에서 full input과 corrupted input을 넣었을 때 정책이 덜 흔들리는지 증명한다

핵심 확인:

- action mean 차이가 줄어드는가
- hidden state drift가 과도하지 않은가
- consistency loss가 성능을 완전히 망가뜨리지는 않는가

이 단계가 끝났다고 보는 기준:

- consistency on 모델이 off 모델보다 same-state action drift가 작다
- normal 성능이 과도하게 희생되지 않는다

### Phase 4. Reliability-Fusion Coupling Proof

목표:

- reliability score가 단순 logging용이 아니라 실제 fusion 결정에 영향을 미친다는 걸 증명한다

핵심 확인:

- reliability가 낮은 모달이 실제 final gate / fusion에서도 덜 반영되는가
- reliability가 좋아질 때 같은 모달 weight도 회복되는가

이 단계가 끝났다고 보는 기준:

- quality/corruption 라벨과 reliability가 대체로 일치한다
- reliability 변화와 fusion weight 변화가 같은 방향으로 연결된다

### Phase 5. Shapley Proof

목표:

- counterfactual contribution이 실제 중요한 모달을 가리키는지 증명한다

핵심 확인:

- attention weight와 Shapley score가 완전히 따로 놀지 않는가
- route/ego가 안전 바닥 역할을 하는가
- 중요 모달이 실제 contribution에서도 올라오는가

이 단계가 끝났다고 보는 기준:

- contribution score가 직관과 크게 어긋나지 않는다
- shapley loss를 켰을 때 해석 가능성이 증가한다

### Phase 6. Risk-Aware Proof

목표:

- 위험 상황에서 더 보수적인 모달 조합을 쓰는지 증명한다

핵심 확인:

- obstacle proximity, heading error, multi-corruption에서 안전 모달 비중이 올라가는가
- ego/route가 완전히 무시되지 않고 바닥 weight를 유지하는가

이 단계가 끝났다고 보는 기준:

- high-risk state에서 reliability 낮은 모달 weight가 줄어든다
- normal 구간 성능을 크게 깨지 않으면서 위험 구간이 더 보수적으로 바뀐다

### Phase 7. Full Training

목표:

- reliability / consistency / shapley / risk를 함께 켠 full RCS-AC를 안정적으로 학습한다

핵심 확인:

- normal 성능이 유지되는가
- adverse corruption에서 baseline보다 덜 무너지는가
- 여러 loss를 동시에 켰을 때 학습이 폭주하지 않는가

이 단계가 끝났다고 보는 기준:

- baseline 대비 adverse degradation이 작다
- 학습이 안정적이고 반복 seed에서도 방향성이 유지된다

### Phase 8. Final Evaluation + Ablation

목표:

- "왜 좋아졌는지"를 구조적으로 설명 가능한 최종 결과를 만든다

핵심 확인:

- baseline A/B/C 대비 full RCS-AC 차이
- `w/o Reliability`
- `w/o Shapley`
- `w/o Risk`
- `w/o Consistency`
- `w/o Recurrent`

이 단계가 끝났다고 보는 기준:

- absolute 성능뿐 아니라 `normal -> adverse degradation`가 개선된다
- reliability / contribution / final weight 3개가 함께 해석 가능하다
- ablation으로 각 구성요소의 기여를 설명할 수 있다

## 10. 현 시점에서 바로 다음에 할 일

현재 즉시 수행해야 하는 건 `Phase 1. Selector Proof`다.

즉 앞으로의 가장 가까운 목표는 아래 한 줄로 정리된다.

> "망가진 상황에서만 selector가 켜지고, clean에서는 꺼진다는 것을 먼저 증명한다."

이게 되기 전에는:

- gate magnitude를 더 키우는 실험
- rescue alpha를 더 키우는 실험
- risk-aware full training
- 최종 ablation 해석

을 주력으로 밀지 않는다.
