# Core Stabilization Workflow Rules

Created: 2026-05-01

## 0. Purpose

This is the short core rule document for every RCS-AC-first stabilization round.

Read this file every round. Read longer references only when the routing table below says they are needed.

## 1. Reading Router

Default ordinary round:

```text
1. Read this file.
2. Read 03_lite_subagent_stabilization_workflow.md.
3. Read only the relevant sections of 02_integrated_subagent_remote_stabilization_workflow.md when Windows/CARLA execution is needed.
```

Escalated high-risk round:

```text
1. Read this file.
2. Read 03_lite_subagent_stabilization_workflow.md escalation rules.
3. Read 04_full_subagent_roles_and_legacy_rules.md for Full agent role details.
4. Read 02_integrated_subagent_remote_stabilization_workflow.md when Windows/CARLA execution is needed.
```

Infrastructure failure or remote setup issue:

```text
Read 01_remote_mac_windows_stabilization_workflow.md only for detailed SSH, Tailscale, SCP, CARLA startup, or failure-handling reference.
```

Do not read all operating guides by default. That defeats the Lite workflow's token-saving purpose.

## 2. Default Trigger

When the user says:

```text
안정화 라운드 진행해줘
라운드 안정화 진행해줘
서브에이전트로 안정화 진행해줘
서브에이전트 구조로 안정화 진행해줘
```

interpret it as:

```text
Start the stabilization round in Lite Subagent Workflow mode.
Use Mac-Windows remote CARLA execution when a CARLA run is required.
Escalate to Full Subagent Workflow if any high-risk condition appears.
```

First progress update should state:

```text
이번 라운드는 기본값인 Lite Subagent Workflow로 시작하고, 고위험 조건이 보이면 Full workflow로 승격하겠습니다.
```

If the user explicitly asks for Full, or the latest evidence already matches a Full escalation condition, start Full immediately and state the reason.

## 3. Non-Negotiable Order

Every stabilization round follows this order:

```text
1. Identify current state and latest relevant logs/results.
2. Choose Lite or Full mode.
3. Spawn required sub-agents.
4. Wait for required findings.
5. Integrate findings.
6. Write the Experiment Brief.
7. Edit code/config/docs.
8. Validate locally.
9. Run the planned Windows/CARLA or BC step if needed.
10. Pull text/table logs.
11. Analyze regressions before improvements.
12. Document PASS/WARN/BLOCK/FAIL and next recommendation.
```

Short form:

```text
Sub-agents first
Brief second
Edit third
Validate fourth
Run fifth
Document sixth
```

## 4. Lite vs Full

Lite is the default.

```text
Lite:
  Main + Agent A + Agent C
  Add Agent B only when code/config/logging/checkpoint wiring needs audit.

Full:
  Main + Agent A + Agent B + Agent C + Agent D + Agent J
  Add Agent F for new or changed imitation/data/BC target work.
  Add Agent E/G only when ablation or control/RL uncertainty requires it.
```

Escalate from Lite to Full immediately when any of these appear:

```text
unclear or contradictory failure mode
Agent A and Agent C disagree on hypothesis or gate
reward/control/action-contract/PPO/model/recurrent/Shapley/environment code changes
new dataset collection, filtering, relabeling, target semantics, or material remixing
checkpoint promotion, paper-relevant claim, or 4096 formal run risk
privileged route/lane/simulator signal or controller-assist risk
new severe safety regression or action-contract violation
more than one plausible primary knob
missing baseline, checkpoint, config, log, or gate identity
```

Do not downgrade back to Lite within the same round after Full is triggered.

## 5. Pre-Edit Barrier

Before required Lite or Full findings are integrated, do not:

```text
create or copy a new experiment config
patch runtime/control/reward/model/PPO code
start dataset collection
start CARLA eval or PPO training
promote a checkpoint
```

Allowed while waiting:

```text
read existing logs/docs/code
verify file paths and run identities
draft non-binding diagnosis notes
prepare command drafts without running experiments
```

## 6. Experiment Brief Barrier

The main agent must write an Experiment Brief before the first file edit.

Minimum fields:

```text
Round name:
Mode: Lite or Full
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

Allowed changes must be narrow. Rejected changes should name tempting changes intentionally excluded to keep the experiment interpretable.

## 7. Gate Rules

Do not run 4096 automatically.

```text
Run 512/eval or 2048 first when applicable.
Analyze the gate before 4096.
Continue only if the primary mechanism activated and safety did not regress.
```

Stop before PPO, 4096, or promotion when:

```text
primary failure is unchanged
failure preconditions are unchanged
new severe failure appears
route progress collapses
mechanism did not activate
BC init eval regresses severely
NaN/Inf/checkpoint/action-contract failure occurs
```

Never promote on deterministic eval alone if training distribution regresses severely.

## 8. Required Metrics

Minimum result metrics:

```text
route_progress mean/max/final
done reason counts
collision/offroad/stuck counts
early offroad count when relevant
action jitter when policy/control changes
mechanism-specific activation metric
baseline comparison
```

For imitation/data/BC rounds:

```text
dataset distribution
target correctness
BC strength
BC eval before PPO
whether PPO should be allowed
```

For controller-assisted rounds:

```text
assist active rate
assist parity
assist-off or same-assist ablation requirement
paper-relevance label
```

## 9. Documentation Rules

Every round must create or update:

```text
docs/stabilization_rounds/<NN_round_name>.md
```

The result must include:

```text
Status:
Mode:
Baseline:
Sub-agent findings or summary:
Experiment Brief:
Local validation:
Remote runs:
Regressions:
Improvements:
Mechanism activation:
Gate decision:
Escalation decision:
Next recommendation:
```

Always document failed, blocked, and infrastructure-invalid runs.

## 10. Global Safety Rules

```text
1. Never overwrite or delete old run logs.
2. Never run two CARLA jobs at the same time.
3. Never claim stabilization success without comparing against a baseline.
4. Never optimize only for reward.
5. Never ignore collision increases just because route progress improves.
6. Never combine multiple primary hypotheses in one round.
7. Always preserve action contract: [steer, longitudinal].
8. Always verify that the new mechanism actually activated in logs.
9. Never hide rule-based route/lane/steering correction inside a stabilization result.
10. CARLA execution and best-checkpoint promotion remain Main Agent responsibilities.
```
