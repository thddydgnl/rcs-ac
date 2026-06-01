# Lite Subagent Stabilization Workflow

Created: 2026-05-01

## 0. Purpose

This guide defines the lightweight default sub-agent workflow for ordinary RCS-AC-first stabilization rounds.

The goal is to keep the accuracy benefits of independent diagnosis and gate checking while avoiding the full token and coordination cost of the complete A/B/C/D/J structure.

Read this document together with:

```text
docs/operating_guides/00_agent_workflow_rules.md
```

Read these only when needed:

```text
docs/operating_guides/02_integrated_subagent_remote_stabilization_workflow.md
  - Windows/CARLA execution, readiness guard, log pullback

docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
  - SSH/Tailscale/SCP/CARLA troubleshooting reference

docs/operating_guides/04_full_subagent_roles_and_legacy_rules.md
  - Full workflow role details after escalation
```

If details conflict, use the stricter safety rule.

## 1. Default Mode

The default stabilization trigger:

```text
안정화 라운드 진행해줘
```

means:

```text
Start in Lite Subagent Workflow mode.
Use Mac-Windows remote CARLA execution when a CARLA run is required.
Escalate to Full Subagent Workflow if any escalation condition appears.
```

The main agent should say in the first progress update:

```text
이번 라운드는 기본값인 Lite Subagent Workflow로 시작하고, 고위험 조건이 보이면 Full workflow로 승격하겠습니다.
```

If the round is already high-risk from the initial request or latest docs, the main agent should start Full immediately and state the reason.

## 2. When Lite Is Allowed

Lite is allowed when all of these are true:

```text
1. The previous result document gives a clear next action or the latest failure mode is already narrow.
2. The round has one primary hypothesis.
3. The first CARLA step is a 512 eval, BC eval, dataset-free train, or 2048 gate, not automatic 4096 promotion.
4. The round does not change action contract, reward logic, PPO logic, model architecture, recurrent state handling, or CARLA environment semantics.
5. The round does not create a new expert dataset, relabel targets, or materially change dataset composition.
6. The round does not introduce a new controller assist, route/lane based action correction, hardcoded steering rule, or privileged runtime signal.
7. The expected result can be judged by already logged metrics or one small logging/config addition.
```

Typical Lite rounds:

```text
small config adjustment inside an existing mechanism
repeat eval with corrected gate thresholds
BC eval gate using an existing dataset and existing target semantics
minor logging addition for an already planned diagnostic
2048 smoke after a previous document explicitly names the next config
documentation-only result synthesis after logs are already available
```

## 3. Escalate To Full

Escalate from Lite to Full immediately when any condition below is true:

```text
1. The latest logs and previous result document disagree about the failure mode.
2. Agent A and Agent C disagree on the next hypothesis or stop gate.
3. The round needs reward, control, action-contract, PPO, recurrent, model, Shapley, or environment-code changes.
4. The round creates, filters, relabels, mixes, or audits a new dataset that will affect training targets.
5. The round changes BC target semantics, trainable module scope, or checkpoint loading behavior.
6. The round could promote a checkpoint to current best, paper-relevant result, or 4096 formal run.
7. The mechanism may depend on privileged route/lane/simulator information or controller assist.
8. A new safety regression appears during analysis: collision, stuck, severe early offroad, route collapse, NaN/Inf, or action-contract violation.
9. More than one plausible knob or ablation path is active.
10. The main agent cannot identify the baseline, checkpoint, config, logs, and gate with confidence.
```

Full means the current full workflow:

```text
Agent A - Log Analyst
Agent B - Code Auditor
Agent C - Mechanism Verifier
Agent D - Safety Regression Analyst
Agent J - Hardcoding / Privileged Controller Auditor
Agent F - Imitation/Data Auditor when new data/BC target work is involved
Agent E/G when ablation or control/RL uncertainty requires it
```

Do not ask the user for confirmation before escalating. State the reason and continue.

Do not downgrade back to Lite during the same round after Full has been triggered.

## 4. Lite Agent Set

Default Lite agents:

```text
Agent A - Log / Result Analyst
Agent C - Mechanism / Gate Verifier
```

Conditional Lite agent:

```text
Agent B - Focused Code / Config Auditor
```

Use Agent B when:

```text
1. Any code file is edited.
2. A config field is new, renamed, or has uncertain wiring.
3. Logging is changed or the gate depends on a newly logged field.
4. Checkpoint load, init checkpoint, trainable module scope, or strictness changes.
5. The main agent is not sure whether the proposed config can actually affect runtime behavior.
```

Do not spawn Lite Agent B for a pure rerun or pure documentation round if the relevant code path has already been validated and no files will be edited.

## 5. Lite Agent Prompts

### 5.1 Agent A - Log / Result Analyst

Assign Agent A a narrow read-only task:

```text
Read the latest relevant result document and logs for this stabilization round.
Identify the current baseline, latest failed or blocked run, primary failure mode, regressions, and the smallest next experiment implied by the evidence.
Return numeric evidence only. Include file paths and line numbers or JSON keys.
Do not propose broad rewrites.
```

Required Agent A output:

```text
Run identity:
Baseline:
Latest blocked/failed run:
Failure mode ranking:
Key regressions:
Metrics to preserve:
One narrow next experiment:
Uncertainty / missing logs:
Escalation needed: yes/no and reason
```

### 5.2 Agent C - Mechanism / Gate Verifier

Assign Agent C a narrow read-only task:

```text
Given the proposed next experiment, define the mechanism metrics and stop/continue gates.
Check whether existing logs can measure the mechanism.
Identify false-win risks where one metric improves but safety or causality regresses.
Return concrete gate thresholds when possible.
```

Required Agent C output:

```text
Mechanism that must activate:
Existing metrics that verify activation:
Missing metrics:
512/eval gate:
2048 gate:
4096/promotion gate:
Stop conditions:
False-win risks:
Escalation needed: yes/no and reason
```

### 5.3 Agent B - Focused Code / Config Auditor

Use only when needed. Assign Agent B a bounded read-only task before edits:

```text
Inspect only the files related to the planned config/code/logging change.
Confirm whether the planned change can affect the runtime path and whether existing tests or smoke checks cover it.
Identify blockers, missing config copies, package config copies, or logging mismatches.
Do not broaden the experiment.
```

Required Agent B output:

```text
Files inspected:
Runtime path confirmed: yes/no
Config/package copy risk:
Logging/gate risk:
Tests or smoke checks to run:
Blockers:
Escalation needed: yes/no and reason
```

## 6. Lite Required Order

The Lite workflow must follow this order:

```text
1. Main agent reads the latest docs and identifies candidate baseline/run paths.
2. Main agent decides Lite vs immediate Full using Section 3.
3. Spawn Agent A and Agent C in parallel.
4. Spawn Agent B only if Section 4 says it is needed.
5. While waiting, the main agent may read logs/docs/code but must not edit files or start CARLA.
6. Integrate agent findings.
7. Re-check escalation conditions.
8. Write the Lite Experiment Brief.
9. Edit only files listed in the brief.
10. Run local validation.
11. Commit/push if remote Windows execution needs updated code.
12. Run Windows pull and CARLA readiness guard when CARLA is needed.
13. Run exactly the planned CARLA or BC step.
14. Pull complete text/table logs.
15. Analyze regressions before improvements.
16. Decide BLOCK, WARN, PASS, continue, or escalate.
17. Write or update the round result document.
```

Short form:

```text
Lite agents first
Lite brief second
Edit third
Validate fourth
One planned run fifth
Gate and document sixth
```

## 7. Lite Pre-Edit Barrier

Before the Lite findings are integrated, do not:

```text
create a new config
copy an old config into a new experiment
patch runtime/control/reward/model code
start dataset collection
start CARLA eval
start PPO training
promote a checkpoint
```

Allowed while waiting:

```text
read existing logs
read existing docs
inspect existing code
verify file paths and run identities
draft non-binding diagnosis notes
prepare a tentative command list without running it
```

If an urgent infrastructure check is needed, it may only verify connectivity or file existence. It must not start CARLA experiments or modify results.

## 8. Lite Experiment Brief

The main agent must write this before editing files:

```text
Round name:
Mode: Lite
Baseline:
Latest blocked/failed run:
Primary hypothesis:
Sub-agent summary:
Allowed changes:
Rejected changes:
Files to edit:
Mechanism metrics:
Gate metrics:
Stop conditions:
Validation commands:
Remote run plan:
Escalation check:
```

Rules:

```text
1. Allowed changes must be narrow and explicit.
2. Rejected changes must name tempting changes that are intentionally excluded.
3. Gate metrics must include safety and mechanism metrics, not only reward or loss.
4. Stop conditions must say when not to run PPO, 4096, or promotion.
5. Escalation check must state why Lite is still valid.
```

Minimum Lite gate metrics:

```text
route progress mean/max/final
done reason counts
collision/offroad/stuck counts
early offroad count when relevant
action jitter when policy/action behavior changes
mechanism-specific activation metric
regression comparison against baseline
```

For visual/RGB actor rounds, include:

```text
mean_visual_zero_action_l1
mean_visual_zero_steer_delta
mean_visual_zero_longitudinal_delta
corrective_need_count when relevant
corrective_sign_rate when relevant
```

For speed/control guard rounds, include:

```text
guard active rate
guard coverage on risk rows
false guard rate
pre-failure speed/lane/heading windows
jitter and throttle/brake saturation
```

## 9. Local Validation

Run validation appropriate to the files changed.

Default for code changes:

```bash
python -m compileall -q rcs_ac_first tests
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests -q
```

For config changes:

```bash
python -m rcs_ac_first.scripts.smoke_check --config configs/<active_config>.yaml --device cpu
```

For script CLI changes:

```bash
python -m rcs_ac_first.scripts.<script_name> --help
```

If validation is skipped, the result document must state why. Do not skip validation just because Windows will run CARLA.

## 10. Remote Execution

Lite does not change the Mac-Windows split.

Mac owns:

```text
sub-agent analysis
code/config/docs edits
local validation
commits/pushes
log pullback
result synthesis
```

Windows owns:

```text
CARLA server
dataset collection
BC eval requiring CARLA
PPO training
raw logs/checkpoints/outputs
```

Before every CARLA-dependent command, follow `02_integrated_subagent_remote_stabilization_workflow.md`:

```text
1. Mac tunnel preflight before the first Windows command.
2. Windows git pull --ff-only after Mac push.
3. ensure_carla_ready.ps1 before every CARLA command.
4. ensure_carla_ready.ps1 -ForceRestart after crash, fatal dialog, streaming timeout, load_world timeout, reset timeout, or contaminated process state.
```

Lite rounds should run one planned remote step at a time. Do not chain collect -> BC -> eval -> PPO unless the Lite brief explicitly listed each gate and each previous gate has passed.

## 11. Result Document Requirements

Every Lite round must write or update:

```text
docs/stabilization_rounds/<NN_round_name>.md
```

Required sections:

```text
Status:
Mode: Lite
Baseline:
Lite Subagent Findings:
Experiment Brief:
Local Validation:
Remote Runs:
Result:
Regressions:
Improvements:
Mechanism Activation:
Gate Decision:
Escalation Decision:
Next Recommendation:
```

The result must summarize regressions before improvements.

Use these labels:

```text
PASS:
  The planned gate passed and continuation/promotion is allowed by the brief.

WARN:
  Some metrics improved but at least one gate remains weak, ambiguous, or below promotion standard.

BLOCK:
  Do not continue to PPO, 4096, or promotion from this checkpoint/config.

FAIL:
  The round regressed or did not activate the mechanism.

ESCALATED:
  Lite was insufficient and the round moved to Full workflow.
```

## 12. Error Handling

If a Lite sub-agent is delayed or unavailable:

```text
1. The main agent may proceed only if the previous result document already names the exact next experiment.
2. The main agent must manually fill the missing role's required output.
3. The Lite Experiment Brief must say which role was unavailable.
4. The result document must mark the round as higher-risk.
```

If Agent A or C reports uncertainty:

```text
Escalate to Full unless the uncertainty is only a missing non-critical metric and the stop gate remains conservative.
```

If CARLA infrastructure fails before metrics are written:

```text
Classify it as infrastructure contamination.
Force-restart CARLA.
Retry only the same planned command.
Do not change the experiment based on an invalid run.
```

If metrics are written but CARLA crashes afterward:

```text
Keep the metrics.
Force-restart before the next CARLA command.
Document the infrastructure event separately from model behavior.
```

## 13. Practical Prompt Template

Use this for ordinary rounds:

```text
이번 안정화 라운드는 Lite Subagent Workflow로 진행한다.
최신 결과 문서와 로그를 확인하고, Agent A는 실패 모드와 baseline을,
Agent C는 mechanism/gate를 검증한다.
고위험 조건이 보이면 Full workflow로 승격한다.
Experiment Brief 전에는 파일을 수정하거나 CARLA를 시작하지 않는다.
```

Agent A prompt:

```text
Read-only. Inspect the latest stabilization result document and relevant pulled logs.
Identify baseline, latest blocked run, failure mode ranking, regressions, metrics to preserve,
and one narrow next experiment. Include file paths and line numbers or JSON keys.
State whether Full escalation is needed.
```

Agent C prompt:

```text
Read-only. For the proposed next experiment, define mechanism activation metrics,
512/eval gate, 2048 gate, 4096/promotion gate, stop conditions, and false-win risks.
Check whether existing logs can measure the mechanism.
State whether Full escalation is needed.
```

Optional Agent B prompt:

```text
Read-only. Inspect only the files related to the planned code/config/logging change.
Confirm runtime path, config/package copy risk, logging/gate risk, and validation commands.
State blockers and whether Full escalation is needed.
```

## 14. Non-Negotiable Lite Rules

```text
1. Lite is the default, not permission to skip sub-agents.
2. Lite still requires Agent A and Agent C before experiment design.
3. Lite still requires an Experiment Brief before file edits.
4. Lite must escalate to Full when high-risk conditions appear.
5. Lite may not create new datasets or relabel targets without Full escalation.
6. Lite may not change reward/control/action-contract/model/PPO core behavior without Full escalation.
7. Lite may not promote a checkpoint without explicit gate comparison.
8. Lite may not treat a 512 eval alone as proof of stability.
9. Lite may not run 4096 automatically after 2048.
10. Lite must document failed, blocked, and infrastructure-invalid runs.
```
