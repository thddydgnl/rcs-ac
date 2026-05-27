# RCS-AC v2 PDF-Core Clean-First Master Plan

작성일: 2026-05-27

이 문서는 RCS-AC v2의 새 기준 문서다. 이전 문서들은 폐기하지 않고
근거/세부 운영 문서로 유지하되, Codex goal 진행에서 목표와 플랜은 이 문서를
우선한다.

## 0. 결정

기존 목표는 Raw2Drive PDF를 최대한 그대로 구현하는 것이었다. 하지만 사용자의
실제 연구 목표는 PDF 재현 자체가 아니라, CARLA 환경에서 강화학습으로 clean
안정성을 먼저 확보한 뒤 fusion 방식만 바꿔 강건성을 비교하는 것이다. 따라서
이 문서는 "PDF에 명시된 계약은 고정하고, PDF가 명시하지 않은 구현 세부는
clean-first 관점에서 채운다"는 기준으로 진행한다.

Stage 2 privileged clean 진행 중 현재 입력/정책 구조만으로 clean 성능을 안정적으로
잡기 어렵다는 증거도 쌓였다. 특히 `target22` 계열 replay probe에서 특정 action으로
policy가 쏠리는 collapse가 반복됐다.

따라서 진행 방향을 다음처럼 수정한다.

```text
PDF에 확실히 명시된 구조와 계약은 그대로 따른다.
PDF가 명시하지 않은 해상도, 운영, 보조 진단, 일부 학습 안정화 세부만 채운다.
단, 최종 claim은 PDF-explicit core와 implementation-fill variant를 분리해서 기록한다.
Fusion ablation은 privileged/raw clean gate 이후에만 시작한다.
```

짧게 부르는 이름:

```text
PDF-explicit core:
  PDF에 명시된 input/action/model/guidance 계약을 그대로 따르는 기준선.

Clean-stability fills:
  PDF가 명시하지 않은 BEV 해상도, route/lane coverage probe,
  Windows gate/probe 운영, 보조 loss/diagnostic을 clean 안정화용으로 추가한 variant.
  V1.1 요소는 이 범위에서만 참고한다.
```

## 1. 병합된 기존 문서

이 문서는 아래 문서의 핵심 결정을 하나로 합친다.

```text
RCS-AC-v2/docs/00_raw2drive_exact_implementation_plan.md
RCS-AC-v2/docs/01_goal_driven_implementation_runbook.md
RCS-AC-v2/docs/02_persistent_subagent_goal_protocol.md
RCS-AC-v2/docs/03_windows_carla_remote_execution_workflow.md
RCS-AC-v2/docs/04_git_sync_artifact_cleanup_policy.md
RCS-AC-v2/docs/05_clean_performance_goal_prompt.md
RCS-AC-v2/docs/06_stage2_clean_progress_log.md
RCS-AC-v2/docs/07_pdf_stage1_privileged_strict_rework.md
RCS-AC-v1.1/docs/v1_1/rcs_ac_v1_1_input_contract.md
RCS-AC-v1.1/docs/v1_1/rcs_ac_v1_1_train_eval_input_pipeline.md
RCS-AC-v1.1/docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
shared/docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
shared/docs/operating_guides/02_integrated_subagent_remote_stabilization_workflow.md
```

우선순위:

```text
1. 이 문서
2. RCS-AC-v2/GOAL_STATUS.md
3. 현재 코드와 테스트
4. Raw2Drive PDF
5. 위의 세부 문서들
```

충돌이 있을 때는 이 문서가 우선한다. 단, Raw2Drive PDF가 명시한 입력, action,
world model, guidance, raw inference boundary와 충돌하는 구현-fill은 허용하지
않는다. 기존 `00`/`07` 문서는 PDF-explicit core 기준선으로 유지한다.

## 2. PDF 명시 계약과 V1.1 참고점

PDF에 명시된 고정 계약:

```text
privileged BEV semantic masks {0,1}^{H x W x 43} + ego vehicle info
-> privileged world model
-> privileged actor-critic policy
-> action space is 39 discrete actions
-> raw sensor world model with rollout guidance
-> raw sensor policy with head guidance
-> raw input is multi-view RGB images + IMU, encoded into BEV-shaped features
-> raw world model directly trains decoder head only
-> raw reward/continue are obtained by head guidance from privileged world model
-> inference uses raw sensor only
```

V1.1에서 참고할 수 있는 clean 안정화 요소:

```text
high-resolution BEV practice
route/lane/obstacle coverage probes
Windows CARLA remote gate/probe workflow
selected route geometry diagnostics
optional waypoint/speed teacher labels for auxiliary-only experiments
RGB + LiDAR BEV + route BEV + ego BEV + route/ego scalars + sensor health as later fusion variants
-> source-level BEV fusion
-> waypoint + target speed prediction
-> pure pursuit controller
-> steer/throttle/brake
```

핵심 차이:

| 항목 | Raw2Drive PDF 명시 | RCS-AC V1.1 참고 |
| --- | --- | --- |
| 학습 철학 | Model-based RL | BC + controller 중심 |
| clean 안정화 | world model rollout에서 actor-critic | waypoint/speed + pure pursuit |
| action | 39 discrete action id -> throttle/brake/steer/reverse | waypoints + target speed |
| route 사용 | semantic BEV의 road/line/lane 계열과 ego info, 구체 필드는 미명시 | route BEV + route scalar 명시 |
| BEV 해상도 | H/W 미명시, C=43만 명시 | 128x128 사용 경험 |
| raw input | multi-view RGB + IMU, BEVFormer-style encoder | RGB/LiDAR/route/ego/source fusion |
| fusion | clean 이후 raw/fusion 변경 비교 가능 | source-level fusion이 core |
| inference | raw sensor only, privileged leak 금지 | sensor-derived route/ego 포함 |

결론:

```text
PDF 명시 계약은 흔들지 않는다.
PDF가 비워둔 구현 세부는 clean 안정성 확보를 위해 보수적으로 채운다.
V1.1 요소는 auxiliary/diagnostic/운영 workflow 또는 fusion-stage variant로만 편입한다.
```

## 3. 구현-fill 허용 범위

PDF 명시 계약으로 고정:

```text
39 discrete action policy를 최종 제어 계약으로 사용한다.
continuous steer/throttle/brake 회귀를 최종 policy 출력으로 바꾸지 않는다.
privileged input은 43-channel BEV semantic masks + ego vector를 기본 계약으로 둔다.
raw input은 multi-view RGB + IMU를 기본 계약으로 둔다.
raw world model은 decoder head만 직접 학습하고 reward/continue 직접 학습은 금지한다.
raw policy reward/continue는 privileged world model head guidance에서 얻는다.
raw inference에는 privileged BEV, GT boxes, simulator-only state를 넣지 않는다.
```

PDF가 비워둔 세부로 허용:

```text
BEV 해상도 32x32 이상으로 상향
road/lane/obstacle semantic BEV 보강
route path / target waypoint / route heading / route curvature coverage probe
lane centerline, left/right boundary target을 43-channel schema 내부 의미로 검증
route geometry scalar 일부를 diagnostic 또는 auxiliary vector 후보로 사용
waypoint + target_speed auxiliary head는 main policy output이 아닌 보조 실험으로만 사용
waypoint/speed imitation auxiliary loss는 PDF-explicit core와 분리된 fill variant에서만 사용
pure pursuit debug baseline 또는 teacher label generator
sensor_health / modality availability logging
V1.1 Windows CARLA remote workflow, smoke/probe/gate 운영 방식
```

조건부 허용:

```text
route_progress, lane_offset, target_distance:
  privileged debug에는 사용 가능.
  raw inference input으로 쓰려면 raw stage에서도 GPS/map/localization으로
  제공되는 task contract인지 명시해야 한다.

CARLA map 기반 exact lane/route truth:
  privileged training target 또는 diagnostic에는 사용 가능.
  raw policy inference에는 금지.

pure pursuit controller:
  debug baseline, teacher, fallback-rate audit에는 사용 가능.
  PDF-style final policy 성능으로 claim하면 안 된다.

LiDAR BEV, route BEV, ego BEV, sensor_health:
  raw input PDF-explicit core에는 넣지 않는다.
  clean-capable raw baseline 이후 fusion-only ablation variant에서만 사용한다.
```

금지:

```text
raw policy inference에 privileged BEV, GT BEV, CARLA actor boxes를 넣지 않는다.
route fallback을 몰래 최종 clean 성능으로 합산하지 않는다.
clean gate 이전에 fusion ablation으로 목표를 옮기지 않는다.
reward/action/eval contract를 실험 중 조용히 바꾸지 않는다.
waypoint/target-speed 또는 pure pursuit를 최종 제어 출력으로 바꾸지 않는다.
Windows scratch run을 official clean evidence로 쓰지 않는다.
root .git을 삭제하지 않는다.
```

## 4. 새 목표

Codex goal objective는 다음처럼 읽는다.

```text
RCS-AC-v2를 Raw2Drive PDF 기반으로 구현하되,
PDF에 확실히 명시된 input/action/model/guidance 계약은 그대로 따른다.
clean 성능 확보를 위해 PDF가 명시하지 않은 해상도, coverage probe,
route/lane diagnostic, 보조 loss 실험, Windows CARLA 운영 방식을 명시적으로 채운다.
먼저 privileged clean gate를 통과시키고, 이후 raw sensor clean gate를 통과시킨다.
PDF-explicit core와 clean-stability fill variant는 결과와 claim을 분리한다.
Privileged/raw clean gates 통과 전 fusion ablation은 시작하지 않는다.
```

성공 조건:

```text
1. Stage 0 contract/preflight tests pass.
2. Privileged BEV/ego observation이 43-channel schema 안에서 route/lane/obstacle 정보를 충분히 담는지 검증된다.
3. PDF-explicit core privileged baseline 결과가 기록된다.
4. Clean-stability fill privileged variant가 clean gate를 목표로 훈련/평가된다.
5. Privileged clean pass 또는 실패 원인이 evidence로 좁혀진다.
6. Raw sensor stage는 privileged clean gate 이후 시작한다.
7. Fusion ablation은 raw/privileged clean-capable baseline 이후 시작한다.
```

## 5. Stage Plan

### Stage H0 - PDF-Core Contract Rebase

목적:

```text
기존 PDF-only goal 문서와 현재 clean-first implementation-fill 결정을 맞춘다.
현재 WIP code가 PDF-explicit core인지 fill/diagnostic인지 분류한다.
```

작업:

```text
문서 기준을 이 파일로 변경한다.
GOAL_STATUS.md의 next objective를 PDF-core clean-first로 바꾼다.
현재 route_priority_components WIP는 diagnostic sampler feature로 분류한다.
PDF-explicit strict run에는 route_priority_components를 금지한다.
테스트와 wrapper 전파가 끝나기 전에는 Windows official run을 시작하지 않는다.
```

Gate:

```text
새 기준 문서 존재
README/goal prompt가 새 기준 문서를 가리킴
git diff에서 문서와 코드 WIP가 분리되어 설명 가능
```

### Stage H1 - Privileged Observation Upgrade

목적:

```text
현재 16x16 semantic43 privileged BEV가 clean 제어에 충분하지 않은 문제를 검증하고,
PDF의 43-channel semantic mask 계약을 깨지 않는 범위에서 해상도와 coverage를 보강한다.
```

작업:

```text
BEVGridSpec을 config-driven으로 만든다.
기본 후보를 32x32로 올리고, 필요하면 64x64/128x128까지 비교한다.
route_path, route_target, route_left/right/straight 채널을 고해상도에서 검증한다.
lane centerline, lane boundary left/right, drivable, obstacle occupancy를 보강한다.
route geometry scalar 후보는 PDF-explicit core에서 제외하고 fill/diagnostic vector로 분리한다.
```

Raw-stage 재현성 원칙:

```text
raw sensor가 나중에 예측 가능한 semantic target 위주로 보강한다.
planner/navigation으로 제공 가능한 route vector는 허용한다.
CARLA-only exact truth는 training target/diagnostic으로만 둔다.
```

Gate:

```text
privileged_obs_probe가 nonzero 채널, route availability, actor availability를 기록한다.
committed Windows probe config는 configs/eval/clean_128_stage2_privileged_32x32.yaml를 사용한다.
replay manifest가 BEV shape/schema/hash를 기록한다.
BEV channel coverage report가 result.md에 남는다.
```

### Stage H2 - PDF-Faithful Privileged Baseline

목적:

```text
논문식 baseline을 계속 유지해 비교 기준을 만든다.
PDF에 명시된 계약 밖의 fill은 모두 꺼진 상태에서 결과를 기록한다.
```

허용:

```text
world model prediction/dynamics/representation loss
imagined rollout actor-critic behavior loss
39 discrete actions
reward/continue heads
43-channel semantic BEV + ego vector
```

금지:

```text
imitation loss
route action override
action cap/throttle ceiling
component weighting
waypoint auxiliary loss
pure pursuit final control
route geometry scalar input
LiDAR/route/ego BEV fusion input
```

Gate:

```text
--pdf-stage1-strict run passes trainer guard.
offline action/latent probes are sane enough for official eval.
official CARLA clean result is recorded as PDF-explicit core, pass or fail.
```

### Stage H3 - Clean-Stability Fill Privileged Variant

목적:

```text
clean 성능을 실제로 잡기 위해 PDF가 명시하지 않은 안정화 세부를 auxiliary/diagnostic으로 추가한다.
```

추가 요소:

```text
high-resolution semantic BEV
route geometry scalar diagnostic
waypoint + target_speed auxiliary head, main output 아님
waypoint/speed imitation auxiliary loss, fill variant 전용
optional pure pursuit teacher baseline
sensor_health and source availability logging
```

주의:

```text
최종 action policy는 39 discrete action을 유지한다.
waypoint/speed는 auxiliary target으로만 시작한다.
pure pursuit는 "debug/teacher"로만 표기한다.
clean PASS claim은 pure pursuit fallback 없이 산정한다.
```

Gate:

```text
target22 action collapse가 offline probe에서 줄어든다.
steer2 gate가 유지된다.
128 clean smoke에서 collision/offroad/stuck이 없다.
512 clean candidate가 route_progress regression 없이 진행된다.
```

### Stage H4 - Raw Sensor World Model

목적:

```text
privileged clean-capable variant를 teacher/guidance source로 사용해 raw sensor world model을 학습한다.
```

작업:

```text
raw_images + raw_imu input contract 확정
BEVFormer-style raw encoder가 BEV-shaped feature를 출력
decoder head는 semantic BEV target만 학습
reward/continue raw head 직접 학습 금지
rollout guidance로 encoder/deterministic/stochastic state align
LiDAR/route BEV/sensor_health는 PDF-explicit raw baseline에서 제외
```

Gate:

```text
raw inference boundary에 privileged leak 없음
alignment loss finite
decoder BEV reconstruction sanity pass
rollout guidance smoke pass
```

### Stage H5 - Raw Sensor Policy

목적:

```text
raw world model 안에서 head guidance를 사용해 raw policy clean을 잡는다.
```

작업:

```text
raw policy actor-critic training
privileged world model reward/continue head guidance
privileged policy distillation은 auxiliary로만 사용
39 discrete action policy 유지
```

Gate:

```text
128 clean pass
512 primary subset clean pass
2048 subset result recorded
failure class가 infra/adapter/model/driving으로 분류됨
```

### Stage H6 - Fusion Ablation

목적:

```text
clean-capable baseline 위에서 fusion 방식만 바꿔 비교한다.
PDF-explicit raw baseline을 먼저 고정하고, 이후 사용자의 fusion 방식만 바꾼다.
```

Gate:

```text
동일 action/reward/eval/train budget
동일 route/raw input contract, 단 fusion ablation에서 바꾸기로 명시한 modality/fusion block 제외
동일 clean/stress protocol
fusion 방식 외 차이 없음
```

## 6. Current Stage 2 판단

현재까지 evidence:

```text
steer2 offline gate는 비교적 살아남.
target22에서는 action15 collapse가 반복됨.
route progress 0.70 전후에서 정책이 다양한 제어를 못 고름.
현재 16x16 semantic43 BEV는 route/lane/obstacle 채널은 있으나 제어 정보가 거칠 가능성이 큼.
```

따라서 다음 우선순위:

```text
1. route_priority_components는 clean-stability fill/diagnostic으로 유지하고 PDF-explicit strict에는 쓰지 않는다.
2. H1 privileged observation upgrade를 시작한다.
3. PDF-explicit strict 결과와 clean-stability fill 결과를 분리해서 기록한다.
4. clean pass 전 fusion은 계속 보류한다.
```

## 7. Sub-Agent Protocol

서브에이전트 사용은 계속 유지한다.

Lite mode 기본:

```text
Agent A: Goal/Status Evidence Analyst
Agent C: PDF-Explicit Contract and Fill-Boundary Verifier
Agent B: Code/Test/Wiring Auditor, code/config/test/logging/checkpoint 편집 시 추가
```

트리거:

```text
stage entry
major edit
test failure
Windows CARLA run 전후
gate review
PDF-explicit vs implementation-fill claim 경계가 흐려질 때
```

spawn 실패 시:

```text
completed/obsolete agent를 닫는다.
기존 agent 재사용을 우선한다.
fork_context=true와 agent_type/model/reasoning override를 같이 쓰지 않는다.
그래도 실패하면 GOAL_STATUS.md에 에러를 기록하고 manual role simulation으로 진행한다.
```

## 8. Windows CARLA Workflow

역할 분리:

```text
Mac:
  Codex, 문서, 코드 수정, 테스트, git commit/push, 로그 분석

Windows:
  CARLA, CUDA training/probe/eval, heavy replay/checkpoint 보관
```

공식 run 순서:

```text
Mac local tests
commit/push
Windows git pull --ff-only
CUDA guard
CARLA readiness guard
one bounded train/probe/eval
stop CARLA after eval/collection
pull text/table logs only
result.md/GOAL_STATUS.md update
commit/push result note
```

Windows가 과열/재부팅될 때:

```text
CARLA quality/resolution을 낮춘다.
eval/collection 후 CARLA를 반드시 종료한다.
가능한 training/probe는 CUDA로 돌린다.
remote 연결 실패는 model failure로 기록하지 않는다.
```

## 9. Git And Artifact Hygiene

정책:

```text
official evidence는 commit/push된 코드에서만 생성한다.
checkpoint/replay/raw frame/video는 git에 올리지 않는다.
root .git은 삭제하지 않는다.
RCS-AC-v2 내부 nested .git만 accidental metadata일 때 검사 후 제거 가능하다.
```

커밋 전 확인:

```bash
git status --short -- RCS-AC-v2
bash RCS-AC-v2/scripts/check_git_hygiene.sh
```

현재 repo에는 RCS-AC-v2 밖의 unrelated dirty worktree가 많을 수 있다.
RCS-AC-v2 작업에서는 관련 파일만 staging/commit한다.

## 10. Goal Resume Prompt

새 기준으로 goal을 이어갈 때는 아래 프롬프트를 사용한다.

```text
RCS-AC-v2 clean-first goal을 RCS-AC-v2/docs/08_raw2drive_pdf_core_clean_first_master_plan.md 기준으로 이어서 진행해줘.

목표는 Raw2Drive PDF에 명시된 input/action/model/guidance 계약을 고정하고,
PDF가 명시하지 않은 구현 세부만 clean-first 관점에서 채우는 것이다.
PDF-explicit core baseline과 clean-stability fill variant의 claim을 분리하고,
privileged/raw clean gate 전에는 fusion ablation을 시작하지 않는다.

먼저 GOAL_STATUS.md와 새 master plan을 읽고 현재 stage를 판정해줘.
그 다음 sub-agent protocol을 적용하고, Implementation Brief -> edit -> validate -> docs/status update 순서로 진행해줘.
Windows CARLA official run은 commit/push된 코드에서만 실행하고, heavy artifact는 git에 올리지 마.
root .git은 삭제하지 마.
```

## 11. 다음 즉시 작업

다음 turn에서 바로 해야 할 일:

```text
1. H1 observation upgrade 설계 브리프를 쓴다.
2. BEVGridSpec config-driven화와 32x32 smoke test를 작은 단위로 시작한다.
3. route/lane/obstacle coverage report를 privileged_obs_probe 또는 별도 probe에 기록한다.
4. replay/checkpoint manifest에 BEV shape/schema/hash를 명시적으로 남긴다.
5. route geometry scalar 중 PDF-explicit core에서 제외할 diagnostic/fill 필드와 raw-stage 재현 가능한 필드를 선별한다.
```

2026-05-27 H0 note:

```text
route_priority_components WIP는 clean-stability fill/diagnostic sampler로
분류했고, Python trainer + Windows foreground/detached/mixed wrapper에 전파했다.
--pdf-stage1-strict에서는 금지된다.
검증: python -m pytest RCS-AC-v2/tests -q -> 151 passed.
```
