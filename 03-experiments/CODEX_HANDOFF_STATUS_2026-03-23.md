# Codex Handoff Status - 2026-03-23

이 문서는 이 repo를 다른 컴퓨터의 Codex에서 이어받기 위한 상세 handoff 문서다.

목표는 다음 세 가지다.
- 지금까지 무엇을 했는지 한 번에 이해할 수 있게 한다.
- 현재 어디에서 멈췄는지 정확히 보여준다.
- 다음 Codex가 무엇을 먼저 해야 하는지 결정 없이 바로 이어갈 수 있게 한다.

## 1. 한 줄 요약

현재 프로젝트는 다음 단계까지 왔다.
- Stage 1 calibration: 완료
- Stage 2 mild pilot / confirm / full: 완료, PASS 확보
- V3 counterfactual Shapley alignment pilot: 구현 및 pilot 완료, `medium` 후보 PASS
- V3.1 clean-referenced Shapley + targeted raw drop: 구현 및 pilot 완료, 그러나 전부 WARN
- 현재 병목: `vision` 쪽 contribution directionality가 여전히 복구되지 않음
- 현재 다음 작업: `V3.1b target-logit Shapley pilot`

핵심 결론은 이렇다.
- Stage 2까지는 성공적으로 안정화되었다.
- V3에서는 alignment는 좋아졌지만 `vision_shapley_clean_minus_mild`가 음수였다.
- V3.1에서도 `vision_raw_shapley_clean_minus_mild`가 여전히 음수였다.
- 그래서 현재는 `fused-feature Shapley`가 아니라 `target reliability logit` 기준 Shapley로 vision 방향성을 직접 복구하는 V3.1b를 구현한 상태다.
- V3.1b 코드는 붙었지만, 공식 3-seed pilot은 아직 완료하지 못했다.

---

## 2. 지금까지의 단계별 진행 내용

### Stage 1

다음 문제들이 순차적으로 해결되었다.
- corruption path bug 수정
- weak/mild probe 집계 버그 수정
- vision/ego separability debug 완료
- 결론: signal은 `raw obs -> encoder -> head input -> pre-sigmoid logit`까지 전달됨
- 병목은 feature separability가 아니라 objective / head calibration임

이후 아래 실험을 통해 해결했다.
- backbone freeze
- reliability head only calibration
- same-state clean/weak/mild paired calibration
- pre-sigmoid logit triplet loss

최종적으로 Stage 1 기준 후보는 다음으로 고정되었다.
- `reference_freeze_backbone_plus_vision_ego_logit_triplet`

관련 결과:
- [vision_ego_logit_triplet_final_recommendation.md](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/stage1_vision_ego_logit_triplet_calibration/summary/vision_ego_logit_triplet_final_recommendation.md)
- [vision_ego_logit_triplet_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/stage1_vision_ego_logit_triplet_calibration/summary/vision_ego_logit_triplet_score_summary.csv)

### Stage 2

Stage 2 mild pilot -> 5-seed confirm -> full Stage 2까지 진행했다.

최종 standard candidate:
- `reference_freeze_backbone_plus_vision_ego_logit_triplet`

backup candidate:
- `reference_freeze_backbone_plus_ego_logit_triplet`

Stage 2 confirm rescue에서 다음을 추가했다.
- vision/ego logit retention loss
- confirm train corruption을 vision/ego 쪽으로 재가중치
- learning rate 소폭 하향

최종 Full Stage 2 결과:
- candidate: `reference_freeze_backbone_plus_vision_ego_logit_triplet`
- pass_fail: `PASS`
- normal_success_rate: `0.791667`
- normal_return: `4.700968`
- mild_success_rate: `0.802778`
- normal_to_mild_success_drop: `-0.011111`
- normal_to_mild_return_drop: `-0.048956`
- consistency_gain_score: `0.139788`
- vision_reliability_clean_minus_mild: `+0.000577`
- vision_logit_clean_minus_mild: `+0.006932`
- saturation_warning_flag: `0`
- route_distance_side_effect_flag: `0`

관련 결과:
- [stage2_full_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/stage2_full/summary/stage2_full_score_summary.csv)
- [stage2_full_final_recommendation.md](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/stage2_full/summary/stage2_full_final_recommendation.md)

### V3

Version 3에서는 다음을 추가했다.
- exact counterfactual Shapley
- utility: fused feature 기준
- attention-Shapley alignment KL
- Stage 2 KPI 유지

pilot 결과:
- `v3_shapley_alignment_medium`: `PASS`
- `v3_shapley_alignment_small`: `WARN`
- `reference_stage2_full_locked_standard`: `WARN`

핵심 수치:
- `v3_shapley_alignment_medium`
  - normal_success_rate: `0.819444`
  - mild_success_rate: `0.821759`
  - attention_shapley_kl: `0.422133`
  - attention_shapley_rank_corr: `0.765147`
  - vision_reliability_clean_minus_mild: `+0.000367`
  - vision_logit_clean_minus_mild: `+0.006421`
  - pass_fail: `PASS`

하지만 병목이 남았다.
- `vision_shapley_clean_minus_mild = -0.001638`

즉 alignment는 좋아졌지만, `mild vision corruption 시 vision contribution이 실제로 내려간다`는 방향성은 확보하지 못했다.

관련 결과:
- [v3_shapley_pilot_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_shapley_pilot/summary/v3_shapley_pilot_score_summary.csv)
- [v3_shapley_pilot_final_recommendation.md](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_shapley_pilot/summary/v3_shapley_pilot_final_recommendation.md)

### V3.1

V3.1에서는 다음 문제를 해결하려 했다.
- corrupted full reference를 clean reference로 교체
- softmax distribution KL만 쓰던 것을 raw Shapley drop까지 분리
- mixed보다 single-target 중심으로 Shapley loss를 제한

추가된 개념:
- clean-referenced exact Shapley
- raw Shapley mean과 normalized target mean 분리 저장
- `vision/ego` raw Shapley drop loss
- attention / raw Shapley sign-match flag

pilot 결과:
- `v3_1_cleanref_alignment_plus_target_drop_small`: `WARN`
- `v3_1_cleanref_alignment_plus_target_drop_medium`: `WARN`
- `reference_v3_medium_locked`: `WARN`

핵심 수치:
- `v3_1_cleanref_alignment_plus_target_drop_small`
  - normal_success_rate: `0.833333`
  - mild_success_rate: `0.826389`
  - vision_reliability_clean_minus_mild: `+0.000319`
  - vision_logit_clean_minus_mild: `+0.007343`
  - ego_raw_shapley_clean_minus_mild: `+0.000050`
  - vision_raw_shapley_clean_minus_mild: `-0.001331`
  - pass_fail: `WARN`
- `v3_1_cleanref_alignment_plus_target_drop_medium`
  - normal_success_rate: `0.805556`
  - mild_success_rate: `0.814815`
  - vision_raw_shapley_clean_minus_mild: `-0.001140`
  - pass_fail: `WARN`

V3.1의 결론:
- ego 쪽은 아주 미세하게 개선되었지만
- 핵심 병목인 `vision_raw_shapley_clean_minus_mild`는 여전히 음수
- 즉 clean reference + fused-feature raw drop만으로는 vision directionality를 못 살렸다

관련 결과:
- [v3_1_shapley_pilot_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_score_summary.csv)
- [v3_1_shapley_pilot_alignment_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_alignment_summary.csv)
- [v3_1_shapley_pilot_raw_shapley_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_raw_shapley_summary.csv)
- [v3_1_shapley_pilot_final_recommendation.md](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_final_recommendation.md)

---

## 3. 현재 코드 상태

### 이미 구현되어 있는 핵심 파일

Stage 2 / V3 / V3.1 / V3.1b의 핵심 로직은 아래 파일에 모여 있다.

- [stage2_transition_helpers.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/stage2_transition_helpers.py)
- [v3_shapley_helpers.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/v3_shapley_helpers.py)

runner:
- [run_stage2_mild_pilot.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_stage2_mild_pilot.py)
- [run_stage2_confirm_5seed.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_stage2_confirm_5seed.py)
- [run_stage2_full.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_stage2_full.py)
- [run_v3_shapley_pilot.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_shapley_pilot.py)
- [run_v3_shapley_confirm_5seed.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_shapley_confirm_5seed.py)
- [run_v3_shapley_full.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_shapley_full.py)
- [run_v3_1_shapley_pilot.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1_shapley_pilot.py)
- [run_v3_1_shapley_confirm_5seed.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1_shapley_confirm_5seed.py)
- [run_v3_1_shapley_full.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1_shapley_full.py)
- [run_v3_1b_shapley_pilot.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1b_shapley_pilot.py)
- [run_v3_1b_shapley_confirm_5seed.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1b_shapley_confirm_5seed.py)
- [run_v3_1b_shapley_full.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1b_shapley_full.py)

config:
- [configs/stage2_full.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage2_full.yaml)
- [configs/v3_shapley_pilot.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_shapley_pilot.yaml)
- [configs/v3_1_shapley_pilot.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_1_shapley_pilot.yaml)
- [configs/v3_1b_shapley_pilot.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_1b_shapley_pilot.yaml)

### `v3_shapley_helpers.py` 안의 현재 중요한 클래스

- `Stage3ShapleyTrainer`
  - V3 기본 exact Shapley alignment
- `Stage31ShapleyTrainer`
  - clean-referenced fused-feature Shapley
  - raw Shapley drop loss
- `Stage31bShapleyTrainer`
  - V3.1b용
  - fused-feature alignment는 유지
  - target reliability logit 기준 exact Shapley drop loss 추가

### V3.1b에서 추가된 개념

`v3_shapley_helpers.py`에는 다음이 추가되었다.

- `utility_mode`가 있는 exact Shapley 계산
  - `clean_fused_l2`
  - `target_logit_clean_l1`
- `Stage31bShapleyTrainer`
  - `loss/vision_targetlogit_shapley_drop`
  - `loss/ego_targetlogit_shapley_drop`
- V3.1b summary helpers
  - `compute_v3_1b_targetlogit_metrics(...)`
  - `build_v3_1b_candidate_score_summary(...)`
  - `generate_v3_1b_report(...)`

---

## 4. 현재 정확히 어디에서 멈췄는가

현재 멈춘 지점은 **V3.1b official 3-seed pilot 전**이다.

상태는 다음과 같다.
- V3.1b 코드 추가 완료
- V3.1b runner/config 추가 완료
- `py_compile` 통과
- smoke 수준에서 train/eval 경로가 실제로 돌아가는 것 확인
- 하지만 공식 3-seed pilot은 아직 완료하지 않음

### V3.1b smoke에서 확인된 것

다음 파일에서 새 loss가 실제로 찍히는 것을 확인했다.
- [train_metrics.jsonl](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1b_shapley_pilot/v3_1b_targetlogit_drop_balanced/seed_0/train_metrics.jsonl)

확인된 로그:
- `loss/vision_targetlogit_shapley_drop`
- `loss/ego_targetlogit_shapley_drop`
- `train/corruption_mild_vision_ratio`
- `train/corruption_mild_ego_ratio`

즉 구현 wiring 자체는 맞다.

### V3.1b가 아직 끝나지 않은 이유

이 단계는 exact Shapley를 두 번 더 계산한다.
- fused-feature alignment용 exact Shapley
- vision target-logit exact Shapley
- ego target-logit exact Shapley

그래서 pilot eval 비용이 매우 크다.

실제로 smoke에서도:
- `eval_metrics.jsonl`의 update 2가 약 `354s`에 기록되었다

관련 파일:
- [eval_metrics.jsonl](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1b_shapley_pilot/v3_1b_targetlogit_drop_balanced/seed_0/eval_metrics.jsonl)

즉 현재 병목은 구현 오류가 아니라 **실험 속도**다.

---

## 5. 앞으로 해야 할 일

현재 가장 중요한 다음 액션은 하나다.

### 5.1 V3.1b pilot을 끝낼 수 있게 속도 최적화

다음 변경을 먼저 하는 것이 강하게 추천된다.

pilot 전용으로:
- `eval_interval_updates: 2 -> 4`
- `num_eval_episodes: 24 -> 12`
- `num_eval_episodes_smoke: 6 -> 4`
- `shapley_num_states_per_minibatch: 4 -> 2`

이유:
- 지금 구조는 맞다
- 현재 막힌 건 성능이 아니라 wall-clock이다
- official 3-seed pilot을 끝까지 돌려야 `vision_targetlogit_shapley_clean_minus_mild`의 sign을 확인할 수 있다

### 5.2 V3.1b official 3-seed pilot 실행

속도 최적화 후 아래 실행:

```bash
python3 /Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1b_shapley_pilot.py \
  --config /Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_1b_shapley_pilot.yaml
```

봐야 할 핵심 파일:
- `results/v3_1b_shapley_pilot/summary/v3_1b_shapley_pilot_score_summary.csv`
- `results/v3_1b_shapley_pilot/summary/v3_1b_shapley_pilot_targetlogit_shapley_summary.csv`
- `results/v3_1b_shapley_pilot/summary/v3_1b_shapley_pilot_final_recommendation.md`

### 5.3 V3.1b acceptance 기준

다음 조건을 만족하면 pilot 성공으로 본다.

- train 후보 중 최소 1개 `PASS`
- `vision_targetlogit_shapley_clean_minus_mild > 0`
- `vision_attention_targetlogit_sign_match_flag > 0`
- `vision_reliability_clean_minus_mild > 0`
- `vision_logit_clean_minus_mild > 0`
- `normal_success_rate >= 0.80`
- `normal_to_mild_success_drop <= 0.05`
- `route_distance_side_effect_flag == 0`
- `saturation_warning_flag == 0`

### 5.4 V3.1b pilot이 PASS면

그 다음 순서:
- `run_v3_1b_shapley_confirm_5seed.py`
- `run_v3_1b_shapley_full.py`

즉:
- pilot PASS -> confirm 5-seed
- confirm PASS -> full V3.1b

### 5.5 V3.1b pilot이 또 FAIL/WARN이면

그때는 다음 순서로 다시 본다.

1. `vision-heavy`가 `balanced`보다 실제로 `vision_targetlogit_shapley_clean_minus_mild`를 더 올리는가
2. `vision_targetlogit_shapley_drop_loss_coef`를 더 올릴 것인가
3. eval이 아니라 train-time target-logit drop direction 자체가 안 생기는가
4. 필요하면 target-logit utility를 `L1`에서 `squared` 쪽으로 바꿀 것인가

하지만 이건 **V3.1b official pilot 결과를 본 뒤**에만 판단하는 게 맞다.

---

## 6. 다음 Codex가 바로 봐야 할 우선순위 파일

### 결과 우선

1. [stage2_full_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/stage2_full/summary/stage2_full_score_summary.csv)
2. [v3_shapley_pilot_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_shapley_pilot/summary/v3_shapley_pilot_score_summary.csv)
3. [v3_1_shapley_pilot_score_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_score_summary.csv)
4. [v3_1_shapley_pilot_alignment_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_alignment_summary.csv)
5. [v3_1_shapley_pilot_raw_shapley_summary.csv](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_1_shapley_pilot/summary/v3_1_shapley_pilot_raw_shapley_summary.csv)

### 코드 우선

1. [v3_shapley_helpers.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/v3_shapley_helpers.py)
2. [run_v3_1_shapley_pilot.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1_shapley_pilot.py)
3. [run_v3_1b_shapley_pilot.py](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/run_v3_1b_shapley_pilot.py)
4. [configs/v3_1_shapley_pilot.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_1_shapley_pilot.yaml)
5. [configs/v3_1b_shapley_pilot.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_1b_shapley_pilot.yaml)

---

## 7. 현재 판단 요약

지금까지의 최종 해석은 다음과 같다.

- Stage 2는 성공했다.
- V3는 alignment에는 성공했다.
- V3.1은 clean reference와 raw drop까지 넣었지만, `vision raw directionality`는 복구하지 못했다.
- 그래서 현재 가장 가능성이 높은 다음 수는 `target reliability logit` 기준 Shapley로 vision contribution drop을 직접 학습하는 V3.1b다.
- V3.1b는 구현 완료 상태지만, 실험 속도가 느려서 아직 공식 결론은 없다.

즉 현재의 올바른 질문은:
- "V3.1b가 실제로 `vision_targetlogit_shapley_clean_minus_mild > 0`를 만들 수 있는가?"

이 질문에 답이 나오기 전까지는:
- Version 4
- risk-aware regularization
- 더 큰 구조 변경
- full V3.1b 이후 단계
로 넘어가면 안 된다.

---

## 8. 다음 Codex를 위한 추천 시작 순서

다른 컴퓨터에서 이 repo를 열었을 때, 아래 순서로 시작하는 걸 추천한다.

1. 이 파일 읽기
2. `results/stage2_full/...`와 `results/v3_1_shapley_pilot/...` summary 확인
3. `v3_shapley_helpers.py`에서 `Stage31ShapleyTrainer`, `Stage31bShapleyTrainer` 읽기
4. `configs/v3_1b_shapley_pilot.yaml`에서 pilot speed만 낮추기
5. `run_v3_1b_shapley_pilot.py` 공식 3-seed 실행
6. `vision_targetlogit_shapley_clean_minus_mild` 기준으로 PASS/WARN/FAIL 판단

이 순서를 어기지 않는 것이 좋다.

