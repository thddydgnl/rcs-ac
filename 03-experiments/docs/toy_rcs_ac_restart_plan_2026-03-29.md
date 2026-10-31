# Toy RCS-AC Restart Plan

작성일: 2026-03-29

이 문서는 toy env 기준 RCS-AC 재시작 계획을 다시 고정해 두기 위한 문서다.
목표는 자율주행 성능 자체보다도 아래 구조가 의도대로 작동하는지를 먼저 검증하는 것이다.

- reliability head가 망가진 모달을 낮게 평가하는가
- fusion weight가 정상 모달 쪽으로 이동하는가
- dropout consistency가 정책 흔들림을 줄이는가
- risk가 높을 때 더 안전한 모달 조합을 쓰는가

## 현재 재시작 기준점

- 이전 진행에서 9단계에서 반복 실패가 발생했다.
- 그래서 기초부터 다시 검증하는 흐름으로 되돌린다.
- 현재 상태:
  - 1단계 PASS
  - 2단계 PASS
  - 3단계 PASS
  - 4단계 PASS
  - 5단계 미통과
- 따라서 다음 우선순위는 "5단계 RCS-AC Version 1을 PASS 기준으로 안정화"하는 것이다.

## 1단계. Toy 환경 목표와 상태 정의 고정

먼저 자율주행처럼 복잡하게 만들지 않고, 작은 2D 목표 도달 환경으로 시작한다.

예시 환경:

- 에이전트가 2D 평면에서 움직임
- 목표점까지 가면 성공
- 장애물과 충돌하면 실패
- 최대 step을 넘기면 종료

정책 입력용 4개 모달:

- vision proxy: 전방 장애물 / 목표 방향 관련 간단한 격자 또는 벡터
- distance proxy: 장애물 거리나 주변 occupancy 요약
- ego proxy: 현재 속도, heading, 최근 action
- route proxy: 목표 방향, 목표 거리, 다음 waypoint 방향

보조 헤드 입력용 상태:

- vision 품질 점수
- distance 품질 점수
- ego noise 여부
- route corruption 여부
- speed
- heading error
- obstacle proximity
- goal distance

이 단계 통과 기준:

- observation shape 고정 완료
- action space 고정 완료
- reward / done 조건 정의 완료

## 2단계. Reward와 종료 조건 단순 설계

reward는 과도하게 복잡하게 만들지 않는다.

reward 구성:

- 목표에 가까워지면 +보상
- 목표 도착 시 큰 +보상
- 장애물 충돌 시 큰 -보상
- 영역 이탈 시 -보상
- action 변화가 너무 크면 작은 -보상
- 가만히 있으면 작은 -보상

종료 조건:

- goal 도달
- obstacle 충돌
- boundary 이탈
- max step 도달

이 단계 통과 기준:

- random policy로 reward, done이 정상 동작
- heuristic policy로 일부 episode에서 goal 도달 가능
- reward가 지나치게 폭주하지 않음

## 3단계. Stage 0 환경 검증

문서 기준으로 가장 먼저 해야 하는 것은 Stage 0 환경 검증이다.

확인 항목:

- random policy 실행
- simple heuristic policy 실행
- observation이 step마다 정상 생성되는지 확인
- 각 모달 값 범위가 이상하지 않은지 확인
- corruption을 넣어도 env가 죽지 않는지 확인
- 로그 저장이 제대로 되는지 확인

반드시 저장할 로그:

- episode reward
- success / collision / timeout
- min goal distance
- 모달별 corruption flag
- quality stats
- risk stats

이 단계 통과 기준:

- random, heuristic 모두 20~50 episode 이상 오류 없이 실행
- 로그 파일 저장 성공
- corruption on / off 둘 다 정상 동작

## 4단계. Baseline 먼저 만들기

RCS-AC 전에 baseline을 먼저 만든다.
순서는 single -> early fusion -> attention fusion -> RCS-AC 순서를 따른다.

### 4-1. Baseline A: ego + route only

입력:

- ego proxy
- route proxy

목적:

- 환경 자체가 학습 가능한지 확인

### 4-2. Baseline B: early fusion PPO

입력:

- 4개 모달 encoder 출력 후 concat

목적:

- 멀티모달만으로 성능이 오르는지 확인

### 4-3. Baseline C: attention fusion PPO

입력:

- 4개 모달 feature에 attention weight를 주고 fused feature 생성

목적:

- learned importance만으로도 어느 정도 되는지 확인

이 단계 통과 기준:

- 세 baseline 모두 최소 1회 이상 학습 곡선 저장
- early가 single보다 낫거나 비슷함
- attention이 early보다 최소한 불안정하지 않음

## 5단계. RCS-AC Version 1 구현

문서의 버전 분해를 toy env에 그대로 적용한다.

버전 1 구성:

- 4개 encoder
- simple attention fusion
- recurrent actor-critic

즉 아직은 reliability, shapley, risk를 넣지 않는다.

여기서 확인할 것:

- RNN이 붙어도 학습이 되는가
- 4모달 fusion이 baseline보다 망가지지 않는가
- episode가 길어져도 hidden state 관리가 안정적인가

이 단계 통과 기준:

- baseline C와 비슷한 수준으로라도 수렴
- hidden state 관련 버그 없음
- attention weight 로그 저장 가능

## 6단계. Stage 1 학습

Stage 1 warm-up에서는 아래 loss를 우선 사용한다.

- actor loss
- value loss
- entropy loss

설정:

- reliability loss는 약하게 켠다
- shapley / consistency / risk는 끄거나 매우 낮게 시작한다
- 정상 환경만 사용한다
- corruption은 거의 없거나 매우 약하게만 사용한다

목적:

- 정책이 기본 task를 배우는가
- fusion 구조가 이상하지 않은가
- learning curve가 올라가는가

이 단계 통과 기준:

- success rate 상승
- collision / timeout 감소
- reliability head가 최소한 NaN 없이 학습됨

## 7단계. RCS-AC Version 2 구현

버전 2 구성:

- reliability head 추가
- dropout consistency 추가

toy env corruption 예시:

- vision proxy: 랜덤 dropout, blur-like noise, scale distortion
- distance proxy: 거리값 dropout, noise, sparsity
- ego proxy: Gaussian noise
- route proxy: 방향값 일부 corruption, lookahead 누락

여기서 확인할 것:

- corrupted 모달의 reliability가 실제로 낮아지는가
- consistency loss가 full vs corrupted 정책 차이를 줄이는가
- 망가진 모달 대신 다른 모달 weight가 올라가는가

이 단계 통과 기준:

- corruption label과 reliability score가 대체로 일치
- 같은 state에서 full / corrupted action 차이가 줄어듦
- 한 모달 손상 시 나머지 모달 weight 보정 발생

## 8단계. Stage 2 학습

문서의 Mild robustness learning 단계다.

활성화 loss:

- actor
- value
- entropy
- reliability
- consistency

초기 설정:

- shapley, risk는 낮게 시작
- 에피소드마다 1~2개 모달만 약하게 손상
- 위험도 높은 상황은 가끔만 등장

목적:

- reliability head가 의미 있는 값을 내는지 확인
- corrupted input에서도 정책이 덜 흔들리는지 확인

이 단계 통과 기준:

- normal 환경 성능이 크게 안 무너짐
- mild corruption에서 baseline보다 degradation 감소
- consistency loss 없는 모델보다 안정적

## 9단계. RCS-AC Version 3 구현

버전 3 구성:

- counterfactual Shapley estimator 추가
- Shapley alignment loss 추가

toy env에서는 모달이 4개뿐이므로 처음에는 일부 subset만 샘플링해 근사 Shapley부터 시작한다.

여기서 확인할 것:

- attention weight와 contribution score가 완전히 따로 놀지 않는가
- 실제로 중요한 모달이 Shapley score에서 올라오는가
- route / ego가 안전 바닥 역할을 하는가

이 단계 통과 기준:

- contribution score가 직관과 크게 어긋나지 않음
- shapley loss를 켰을 때 fusion weight 해석 가능성 증가
- 학습이 지나치게 느려지지 않음

## 10단계. RCS-AC Version 4 구현

버전 4 구성:

- risk-aware regularization 추가
- 최종 RCS-AC 완성

toy env risk 예시:

- obstacle proximity 높음
- speed 높음
- heading error 큼
- goal 직전의 좁은 통로
- boundary 근접
- 다중 모달 corruption 동시 발생

여기서 확인할 것:

- risk가 높은 상황에서 reliability 낮은 모달 weight가 줄어드는가
- ego / route 최소 가중치 바닥이 유지되는가

이 단계 통과 기준:

- 고위험 구간에서 더 보수적인 fusion
- normal 성능 유지하면서 adverse 성능 감소폭 완화

## 11단계. Stage 3 Full 학습

모든 loss를 켠다.

- actor
- value
- entropy
- reliability
- shapley
- consistency
- risk

학습 환경:

- 중간~강한 corruption
- 일부 에피소드에서는 복수 모달 손상
- risk 높은 구간 자주 등장

목적:

- 악조건에서 덜 무너지는 정책 확보
- contribution-aware fusion 안정화

이 단계 통과 기준:

- baseline 대비 adverse 성능 감소폭이 더 작음
- reliability / contribution / final weight 3개가 해석 가능함

## 12단계. 최종 평가와 Ablation

필수 비교:

- baseline A: ego + route
- baseline B: early fusion PPO
- baseline C: attention fusion PPO
- RCS-AC full

필수 ablation:

- w/o Reliability
- w/o Shapley
- w/o Risk
- w/o Consistency
- w/o Recurrent

최종 평가 지표:

- success rate
- collision rate
- timeout rate
- avg return
- avg progress to goal
- normal-to-adverse degradation

핵심 해석 포인트:

- 절대 성능보다 정상 -> 악조건 감소폭을 우선 본다
- reliability / contribution / final fusion weight가 함께 해석 가능해야 한다

## 아주 짧게 압축한 실행 순서

1. toy 상태 / 행동 / 보상 설계
2. Stage 0 환경 검증
3. baseline 3종 구현
4. RCS-AC Version 1 구현
5. Stage 1 warm-up
6. Version 2 구현
7. Stage 2 robustness 학습
8. Version 3 구현
9. Version 4 구현
10. Stage 3 full 학습
11. baseline 비교
12. ablation 및 해석

## 현재 작업 지시문

재시작 기준에서 다음 작업은 아래 한 줄로 요약된다.

- 4단계 PASS 상태를 고정한 뒤, 5단계 RCS-AC Version 1이 baseline C 수준으로 수렴하도록 먼저 안정화한다.
