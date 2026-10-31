# Persistent Subagent Protocol for Raw2Drive Goal Runs

작성일: 2026-05-24

목적:

```text
RCS-AC-v2 Raw2Drive 구현 goal이 길어질 때 Codex가 초반 10분만 서브에이전트를 쓰고 잊지 않도록,
stage마다 반복적으로 서브에이전트를 호출하고 결과를 통합하는 규칙을 고정한다.
```

이 문서는 다음 파일과 함께 읽는다.

```text
RCS-AC-v2/GOAL_STATUS.md
RCS-AC-v2/docs/01_goal_driven_implementation_runbook.md
shared/docs/operating_guides/03_lite_subagent_stabilization_workflow.md
shared/docs/operating_guides/04_full_subagent_roles_and_legacy_rules.md
```

## 0. 핵심 원칙

서브에이전트 사용은 일회성 허가가 아니다. RCS-AC-v2 goal에서는 다음과 같이 해석한다.

```text
사용자가 RCS-AC-v2 goal에서 서브에이전트 사용을 허용했으면,
Codex는 stage entry, major edit, test failure, gate review 때 반복적으로 서브에이전트 사용 여부를 재평가한다.
```

서브에이전트는 다음 역할이다.

```text
main agent:
  critical path 구현, 통합 판단, 최종 patch 책임

subagent:
  독립 검토, read-only audit, disjoint worker implementation, test/failure 분석
```

절대 규칙:

```text
서브에이전트 결과를 기다리느라 main critical path를 멈추지 않는다.
main agent와 subagent가 같은 파일을 동시에 고치지 않는다.
subagent에게 모호한 전체 구현을 던지지 않는다.
subagent 결과를 무비판적으로 merge하지 않는다.
```

기본 흐름:

```text
Sub-agents first
Implementation Brief second
Edit third
Validate fourth
Document fifth
```

Pre-edit barrier:

```text
필요한 subagent findings와 Implementation Brief가 준비되기 전에는
파일 편집, 새 config 복사/생성, dataset/CARLA/train/eval 실행,
stage PASS 기록, checkpoint promotion을 하지 않는다.
```

Pre-edit barrier 전에 허용되는 작업:

```text
기존 docs/code/tests 읽기
경로 확인
현재 실패/현황 재현
non-binding draft 작성
subagent spawn
```

## 0.1 Lite / Full mode

기본값은 `Lite Subagent Workflow`다.

Lite 기본 조합:

```text
Agent A - Goal/Status Evidence Analyst
Agent C - Raw2Drive Contract/Gate Verifier
Optional Agent B - Code/Test Wiring Auditor
```

Agent B를 추가하는 경우:

```text
code/config/test/logging/checkpoint wiring을 건드릴 때
runtime import path 또는 config path가 불명확할 때
테스트가 반복 실패할 때
```

Full로 승격하는 조건:

```text
action/reward/observation contract 변경
model/RSSM/PPO/recurrent/env semantics 변경
dataset 생성/필터링/relabel/target semantics 변경
privileged input 또는 leakage 위험
Agent A와 Agent C 의견 불일치
baseline/config/log/gate 불명확
checkpoint promotion, 4096, paper claim 관련 판단
하나 이상의 plausible primary knob이 있어 실험 선택이 불명확
```

Full mode에서는 최소 다음 역할을 둔다.

```text
Agent A - evidence/failure analyst
Agent B - code/config/wiring auditor
Agent C - contract/gate verifier
Agent D - experiment protocol reviewer
Agent J - claim/baseline boundary reviewer
```

## 1. Goal 시작 시 mandatory subagent pulse

긴 goal을 시작하거나 재개하면 반드시 먼저 `Subagent Pulse`를 수행한다.

Subagent Pulse란:

```text
1. 현재 stage 확인
2. 병렬로 맡길 수 있는 sidecar 질문/작업 확인
3. explorer/worker 중 적합한 agent 1-3개 spawn
4. main agent는 즉시 겹치지 않는 local 작업 진행
5. 결과 도착 시 integration note 작성
6. GOAL_STATUS.md에 last_subagent_pulse 갱신
```

서브에이전트 도구가 현재 context에 없으면:

```text
tool_search로 "multi-agent spawn manage sub-agents"를 검색한다.
그래도 없으면 GOAL_STATUS.md에 "subagent unavailable; manual role simulation used"라고 기록하고,
문서의 역할 구조를 main agent가 수동으로 수행한다.
```

### 1.1 Subagent spawn failure recovery

서브에이전트 도구가 있는데도 spawn이 실패할 수 있다. 2026-05-26 goal
resume 중 실제로 다음 에러가 발생했다.

```text
collab spawn failed: agent thread limit reached
```

또한 같은 resume 중 다음 호출 제약도 확인했다.

```text
fork_context=true로 full-history fork를 만들 때는 agent_type/model/reasoning_effort를 같이 지정하지 않는다.
필요하면 fork_context=true만 쓰거나, fork_context=false에서 agent_type을 지정한다.
```

가능한 원인:

```text
thread limit:
  이전 goal turn의 completed/obsolete/running agent thread가 slot을 점유한다.

tool unavailable:
  compaction/resume 후 multi-agent tool이 아직 현재 context에 로드되지 않았다.

state mismatch:
  GOAL_STATUS.md의 Active agents 기록과 실제 tool 상태가 어긋났다.

over-spawn:
  한 번에 너무 많은 agent를 만들거나 같은 질문으로 반복 spawn했다.

invalid spawn request:
  full-history fork와 agent_type/model/reasoning override를 함께 지정했다.

overbroad delegation:
  작업이 너무 넓어 재사용 가능한 agent 대신 새 agent만 계속 만들게 된다.
```

이 경우 모델 성능 문제가 아니라 운영 문제인 경우가 많다. 처리 순서는 다음과 같다.

```text
1. 실패 메시지를 그대로 기록한다.
   예: agent thread limit reached, tool unavailable, invalid spawn arguments, spawn timeout.

2. multi-agent tool이 현재 context에 있는지 확인한다.
   없으면 tool_search로 "multi-agent spawn manage sub-agents"를 검색한다.

3. GOAL_STATUS.md, environment_context, tool 결과에 남은 agent 상태를 대조한다.

4. completed/obsolete agent를 먼저 close_agent로 닫는다.

5. 아직 유효한 agent가 있으면 새로 만들지 말고 resume_agent/send_input으로 재사용한다.

6. 새 agent는 한 번에 1-2개만 만든다.
   Stage entry 기본은 Agent A + Agent C이고,
   code/config/test/logging/checkpoint wiring을 편집할 때만 Agent B를 추가한다.

7. 같은 질문으로 반복 spawn하지 않는다.
   질문이 같으면 기존 agent 재사용 또는 manual role simulation으로 전환한다.

8. fork_context=true를 쓰는 full-history fork에서는 agent_type/model/reasoning_effort를 생략한다.

9. close/reuse 후에도 spawn이 실패하면 GOAL_STATUS.md에 다음 형식으로 남긴다.
   "subagent spawn failed: <error>; cleanup attempted; manual role simulation used"

10. sub-agent 실패가 있어도 goal은 멈추지 않는다.
    main agent가 Agent A/C/B 역할을 수동으로 나누어 read-only audit checklist를 수행한 뒤
    Implementation Brief -> edit -> validate 순서로 계속 진행한다.
```

Full-history fork 예시:

```text
OK:
  spawn_agent(fork_context=true, message="<bounded audit task>")

Avoid:
  spawn_agent(fork_context=true, agent_type="explorer", model="...", reasoning_effort="...", message="...")
```

권장 운영:

```text
goal turn 시작:
  기존 Active agents 확인
  completed/obsolete agent close
  필요한 agent만 1-2개 spawn

major edit 전:
  새 agent를 만들기보다 기존 explorer/worker 재사용 가능성 먼저 확인

Windows/Tailscale/CARLA 연결 장애 중:
  remote 실행은 보류하되, 로컬 문서/로그/스크립트 audit용 subagent pulse는 계속 수행

final 전:
  더 이상 필요 없는 agent close
  GOAL_STATUS.md Active agents 갱신
```

에이전트를 계속 쓰기 위한 핵심은 "많이 만들기"가 아니라 "닫고 재사용하기"다.
Thread limit에 걸리면 새로 만드는 대신 오래된 완료 thread를 정리한다.

## 2. 반복 사용 트리거

Codex는 아래 조건 중 하나라도 발생하면 서브에이전트를 다시 고려해야 한다.

### 2.1 Stage entry trigger

새 stage를 시작할 때:

```text
Stage 0 -> contract/schema/reward/leakage audit
Stage 1 -> RSSM/world-model/loss shape audit
Stage 2 -> actor-critic/clean gate audit
Stage 3 -> raw/privileged alignment/sampling audit
Stage 4 -> raw policy/head guidance/leakage audit
Stage 5 -> fusion fairness/ablation protocol audit
Stage 6 -> closed-loop metrics/report audit
```

최소 1개 explorer를 사용한다. stage가 code-heavy면 worker 1개를 추가할 수 있다.

### 2.2 Major edit trigger

다음 중 하나를 수정하면 subagent audit를 고려한다.

```text
action contract
reward/continue contract
data schema
replay buffer sampling
RSSM prior/posterior
world model losses
actor-critic loss
rollout guidance sampling
privileged/raw inference boundary
fusion comparison protocol
CARLA eval script
```

권장:

```text
main이 구현하는 동안 explorer에게 "이 파일 세트의 edge case와 테스트 누락"을 맡긴다.
```

### 2.3 Test failure trigger

같은 테스트가 2회 이상 실패하거나 failure 원인이 불명확하면 subagent를 쓴다.

역할:

```text
Agent A: failure log analyst
Agent B: code path auditor
```

main agent는 기다리지 말고 재현 command 정리, failing fixture 축소, 로그 수집을 진행한다.

### 2.4 Gate review trigger

stage gate를 PASS로 표시하기 전 반드시 read-only subagent gate review를 고려한다.

Gate review 질문:

```text
이 stage의 PASS 판단이 문서 기준과 일치하는가?
privileged leakage나 action/reward contract drift가 있는가?
다음 stage로 넘어가면 위험한 미검증 항목은 무엇인가?
```

### 2.5 Time/cadence trigger

긴 goal에서는 시간이 지나면 자연히 서브에이전트를 잊는다. 이를 막기 위해 cadence를 둔다.

```text
매 30분 또는 큰 파일 5개 이상 수정 후:
  GOAL_STATUS.md의 Subagent Ledger 확인
  마지막 pulse 이후 상황 변화가 있으면 새 pulse 실행
```

## 3. Stage별 추천 subagent 역할

### Stage 0. Contract preflight

Agent A - Goal/Status Evidence Analyst:

```text
Read GOAL_STATUS, current stage README/result, relevant tests/docs.
Identify current stage, blocked/failed evidence, regressions, metrics/contracts to preserve,
and one narrow next implementation step. State Full escalation yes/no.
Do not edit files.
```

Agent C - Raw2Drive Contract/Gate Verifier:

```text
Verify the proposed step against Raw2Drive contracts/gates.
Identify mechanism/gate metrics, leakage/false-win risks, stop conditions,
and whether existing tests prove the gate. State Full escalation yes/no.
Do not edit files.
```

Agent B - Code/Test Wiring Auditor:

```text
Inspect only planned files/tests/configs.
Confirm runtime path, wiring risk, missing tests, validation commands, blockers,
and Full escalation yes/no.
Do not edit files.
```

Optional Test Planner:

```text
Inspect planned Stage 0 files and propose the minimal pytest cases needed before RSSM work starts.
Do not edit files. Return test names, assertions, and expected failure modes.
```

Optional Worker - Disjoint Implementation:

```text
Own only tests/test_discrete_actions.py and no production files.
Write tests for the action table contract.
Do not modify files outside that write scope.
```

### Stage 1. Privileged world model

Agent A - RSSM Shape Auditor:

```text
Inspect rssm.py/world_model.py/losses.py.
Check prior/posterior state shapes, time-major vs batch-major convention, action embedding, and KL free-nats handling.
Return shape risks and missing tests.
```

Agent B - Loss Auditor:

```text
Inspect world model losses.
Verify prediction/dynamics/representation losses match Raw2Drive/Dreamer-style contract.
Check detach/stopgrad placement.
Return exact concerns with file/line references.
```

Worker 가능:

```text
Worker owns only tests/test_rssm_shapes.py.
Main owns production implementation.
```

### Stage 2. Privileged policy clean

Agent A - Actor-Critic Auditor:

```text
Review actor_critic.py and behavior_loss.py.
Check lambda return, entropy, value target, action distribution over 39 actions, and reward/continue usage.
```

Agent C - Clean Gate Analyst:

```text
Review eval summaries/logs.
Identify failure mode ranking: action saturation, reward mismatch, route progress, collision/offroad/stuck, action jitter.
Return whether clean gate is PASS/PARTIAL/FAIL.
```

### Stage 3. Raw world model alignment

Agent A - Alignment Auditor:

```text
Review raw/privileged alignment losses.
Check encoder MSE, deterministic MSE, stochastic KL, temporal rollout alignment, and beta weights.
```

Agent B - Sampling Auditor:

```text
Verify single-sample stochastic rollout rule:
sample from raw stream distribution once and feed the same sampled state to both raw and privileged rollout.
Find any duplicate independent sampling.
```

### Stage 4. Raw policy clean

Agent A - Leakage Auditor:

```text
Audit raw policy inference inputs.
Confirm privileged_bev, GT actor boxes, reward, continue, and future state are not consumed by inference path.
```

Agent B - Head Guidance Auditor:

```text
Verify raw policy training uses privileged reward/continue heads and does not train/use raw reward/continue heads.
```

Agent C - Clean Eval Analyst:

```text
Summarize clean 128/512/2048 logs.
Rank failure modes and state whether fusion ablation may start.
```

### Stage 5. Fusion ablation

Agent A - Fairness Auditor:

```text
Compare fusion variant configs.
Verify only fusion mode changes and action/reward/eval/training budget stay fixed.
Flag any confound.
```

Agent B - Metrics Auditor:

```text
Inspect result tables.
Check clean safety, route progress, source weight movement, latent alignment error, and action entropy.
Return missing metrics.
```

### Stage 6. Closed-loop eval

Agent A - Result Analyst:

```text
Read all eval summaries and produce a condition-by-condition PASS/PARTIAL/FAIL table.
```

Agent B - Claim Boundary Auditor:

```text
Review final_report.md.
Flag overclaims, missing start-set qualifiers, privileged leakage ambiguity, and unfair baseline comparison.
```

## 4. Explorer vs Worker 선택 기준

Explorer를 쓴다:

```text
코드/로그/문서를 읽고 독립 판단이 필요할 때
stage gate를 검토할 때
테스트 누락/edge case를 찾을 때
failure mode를 분석할 때
```

Worker를 쓴다:

```text
write scope가 명확히 분리될 때
테스트 파일 하나 또는 독립 모듈 하나만 맡길 때
main agent가 동시에 다른 파일을 구현할 수 있을 때
```

Worker에게 반드시 적는다:

```text
You are not alone in the codebase.
Do not revert changes made by others.
Only edit the files listed in your ownership scope.
List changed files in your final response.
```

## 5. Subagent Ledger

`GOAL_STATUS.md`에는 항상 아래 ledger를 유지한다.

```markdown
## Sub-Agent Protocol State

Mode:
Required agents:
Last findings summary:
Pending brief:
Escalation triggers checked:
Next sub-agent trigger:

## Subagent Ledger

Last pulse:
Active agents:
Latest integrated findings:
Next subagent trigger:
Unavailable/fallback:
```

Ledger 갱신 시점:

```text
subagent spawn 직후
subagent 결과 통합 직후
stage gate review 직후
turn 종료 전
```

## 6. Main-agent integration protocol

Subagent 결과가 오면 main agent는 다음을 수행한다.

```text
1. 결과를 그대로 붙여넣지 말고 action item으로 변환
2. 이미 main이 처리한 항목과 중복 제거
3. 타당한 지적은 코드/테스트/문서에 반영
4. 반영하지 않는 지적은 이유 기록
5. GOAL_STATUS.md의 latest integrated findings 갱신
```

Subagent 결과가 main plan과 충돌하면:

```text
1. 구현을 잠시 멈춘다.
2. 충돌 원인을 문서화한다.
3. 더 보수적인 gate를 선택한다.
4. 필요하면 두 번째 read-only auditor를 spawn한다.
```

## 6.1 Implementation Brief

각 work packet의 편집 전에는 짧은 Implementation Brief를 남긴다. 위치는 현재 stage result 문서 또는 `GOAL_STATUS.md`의 Pending brief다.

템플릿:

```markdown
## Implementation Brief

Work packet:
Stage:
Mode: Lite / Full
Current gate:
Baseline/status evidence:
Primary hypothesis:
Sub-agent summary:
Allowed changes:
Rejected changes:
Files to edit:
Contract/leakage risks:
Validation commands:
Stop conditions:
Escalation check:
```

편집 후에는 Brief를 result.md에 보존하고, GOAL_STATUS의 `Pending brief`를 갱신 또는 비운다.

## 7. Prompt templates

### 7.1 Stage entry explorer

```text
You are Agent A, Goal/Status Evidence Analyst for RCS-AC-v2 Raw2Drive Stage <N>.
Read only:
- RCS-AC-v2/GOAL_STATUS.md
- current stage README/result
- relevant tests/docs

Task:
Identify current stage, blocked/failed evidence, regressions,
metrics/contracts to preserve, and one narrow next implementation step.
Do not edit files.
Return:
1. evidence summary
2. current gate status
3. preserved contracts/metrics
4. narrow next step
5. Full escalation yes/no
```

### 7.1.1 Contract/gate verifier

```text
You are Agent C, Raw2Drive Contract/Gate Verifier for Stage <N>.
Read only:
- RCS-AC-v2/docs/00_raw2drive_exact_implementation_plan.md
- RCS-AC-v2/docs/01_goal_driven_implementation_runbook.md
- RCS-AC-v2/docs/02_persistent_subagent_goal_protocol.md
- proposed work packet or planned files

Task:
Verify the proposed step against Raw2Drive action/observation/reward/buffer/leakage/gate contracts.
Identify false-win risks, stop conditions, missing proof, and whether the existing tests prove the gate.
Do not edit files.
Return Full escalation yes/no.
```

### 7.2 Focused code auditor

```text
You are Agent B, Code/Test Wiring Auditor.
Scope:
<files>

Task:
Inspect only planned files/tests/configs.
Confirm runtime path, wiring risk, missing tests, validation commands, blockers,
contract drift, leakage, shape mismatch, dtype mismatch, and Full escalation yes/no.
Do not edit files.
Return file/line references where possible.
```

### 7.3 Worker with disjoint write scope

```text
You are Worker <name> for RCS-AC-v2.
You are not alone in the codebase. Do not revert others' changes.

Ownership:
You may edit only:
<file list>

Task:
Implement <bounded task>.

Validation:
Run <specific test command> if feasible.

Final response:
List changed files, tests run, and any blockers.
```

### 7.4 Gate reviewer

```text
You are Gate Reviewer for Stage <N>.
Read the stage result, tests, and changed files.
Do not edit files.

Question:
Does this stage satisfy the documented gate?

Return:
PASS/PARTIAL/FAIL
Evidence
Risks
Required fixes before next stage
```

## 8. Anti-patterns

피해야 할 패턴:

```text
처음에 explorer 하나만 띄우고 이후 stage에서 다시 안 씀
subagent에게 "전체 구현해줘"라고 맡김
main과 worker가 같은 파일을 동시에 수정
subagent 결과를 기다리느라 main이 멈춤
test failure를 main 혼자 추측으로 고침
gate PASS 전에 read-only review 없이 다음 stage로 이동
GOAL_STATUS.md에 subagent 사용 기록을 남기지 않음
```

## 9. Minimal cadence summary

짧게 외우는 규칙:

```text
Start stage -> spawn explorer.
Edit risky contract -> spawn auditor.
Fail twice -> spawn failure analyst.
Before PASS -> spawn gate reviewer.
Every 30 min -> check ledger and pulse again if context changed.
```

## 10. Stage result.md additions

각 stage 결과 문서는 기존 result template에 아래 필드를 포함한다.

```markdown
## Sub-agent Findings

## Implementation Brief

## Escalation Decision

## Gate Status

PASS / PARTIAL / FAIL / BLOCKED / ESCALATED
```

결과 문서에서는 개선 사항보다 regressions, blockers, gate risks를 먼저 적는다. 다음 stage 진입은 `PASS`일 때만 허용하며, `PARTIAL`은 명시적인 narrowed scope가 있을 때만 허용한다.
