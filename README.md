# RCS-AC-first

작성일: 2026-04-26

이 폴더는 `RCS-AC_1-12_상세계획서.pdf`를 기준으로 RCS-AC 완성본을 새로 구현하기 위한 첫 작업 공간이다.

목표는 기존 `RCS-AC/rcs_ac`의 smoke scaffold를 그대로 복사하는 것이 아니라, 상세계획서가 말하는 최종 RCS-AC 구조를 처음부터 단계적으로 구현하는 것이다.

## 기준 문서

- 원문 PDF: `C:/Users/thddy/Documents/code/RCS-AC/docs/RCS-AC_1-12_상세계획서.pdf`
- 텍스트 추출본: `C:/Users/thddy/Documents/code/RCS-AC/docs/RCS-AC_1-12_상세계획서.txt`
- 기존 smoke 구현 참고: `C:/Users/thddy/Documents/code/RCS-AC/rcs_ac`
- shared runtime 참고: `C:/Users/thddy/Documents/code/shared_core`
- PPO 학습 루프 참고: `C:/Users/thddy/Documents/code/attention`

## 3단계 구현 원칙

1. Phase 1: RCS-AC Core
   - 모달별 encoder, fusion interface, recurrent actor-critic, PPO 연결을 먼저 완성한다.
   - reliability, risk, Shapley는 자리만 만들고 loss weight는 꺼둔다.

2. Phase 2: Reliability + Risk-aware Fusion
   - reliability head, risk/context head, sensor corruption, reliability supervision, risk regularization을 켠다.
   - 이 단계부터 RCS-AC가 attention baseline과 구조적으로 달라지기 시작한다.

3. Phase 3: Counterfactual Shapley Full RCS-AC
   - masked critic, approximate Shapley estimator, contribution-aware fusion, Shapley alignment, dropout consistency를 완성한다.
   - 평가 시 exact-ish subset sweep으로 해석 자료까지 만든다.

## 문서 구성

- `docs/00_three_phase_master_plan.md`
  - 전체 구현 철학, 폴더 구조, 모듈 설계, 단계별 구현 범위, 성공 기준을 정리한 마스터 플랜
- `docs/01_phase_checklist.md`
  - 실제 구현할 때 체크박스로 따라갈 수 있는 단계별 작업 목록
- `docs/02_phase1_core_execution_validation.md`
  - Phase 1 Core를 실제로 진행하고 검증하는 상세 절차 문서

## 폴더 생성 후 첫 구현 추천

다음 작업은 Phase 1의 코드 scaffold 생성이다.

권장 시작 파일:

```text
RCS-AC-first/
  rcs_ac_first/
    __init__.py
    models/
      __init__.py
      encoders.py
      fusion_gate.py
      recurrent_actor_critic.py
      rcs_ac_model.py
    rl/
      __init__.py
      losses.py
      rollout_buffer.py
      ppo_agent.py
    runtime/
      __init__.py
      observation.py
      config.py
    scripts/
      train.py
      smoke_check.py
```
