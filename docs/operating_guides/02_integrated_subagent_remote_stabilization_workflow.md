# Integrated Sub-Agent + Remote CARLA Stabilization Workflow

Created: 2026-05-01

## 0. Purpose

This guide defines the combined stabilization workflow when both of these are true:

```text
1. The round should follow the sub-agent stabilization structure.
2. CARLA execution happens on the Windows desktop through Mac-to-Windows SSH.
```

This document connects:

```text
docs/operating_guides/00_agent_workflow_rules.md
docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
```

It does not replace them. If details conflict, use the stricter rule.

## 1. Trigger Phrase

For this repository, any request to run a stabilization round activates both workflows by default.

The short default trigger is:

```text
안정화 라운드 진행해줘
```

It is equivalent to:

```text
다음 라운드 서브에이전트 구조로, Mac-Windows 원격 실행 방식까지 포함해서 안정화 진행해줘.
```

Equivalent acceptable phrases:

```text
라운드 안정화 진행해줘
안정화 진행해줘
서브에이전트 구조로 Mac-Windows 연결해서 안정화 진행해줘
다음 라운드 서브에이전트로 안정화 진행하고 Windows CARLA에서 실행해줘
라운드 안정화 진행해줘, 서브에이전트 분석과 Windows 원격 실행 방식 둘 다 적용해줘
```

When the user says:

```text
안정화 라운드 진행해줘
```

or:

```text
라운드 안정화 진행해줘
```

the main agent must use this integrated workflow without asking for confirmation. State in the first progress update that the short trigger is being treated as the full sub-agent + Mac-Windows remote stabilization workflow.

## 2. Role Split

Mac/Codex owns thinking, editing, and reporting.

```text
Mac:
  - main agent orchestration
  - sub-agent analysis
  - previous-log inspection
  - code/config/docs edits
  - local static validation and smoke checks
  - commits and pushes
  - result synthesis
```

Windows owns CARLA and heavy runtime artifacts.

```text
Windows:
  - CARLA server
  - Python CARLA collect/eval/train scripts
  - raw logs
  - checkpoints
  - outputs
  - datasets
```

Do not run Codex sub-agents on the Windows CARLA machine during experiments.

## 3. Required Order

The combined workflow must follow this order:

```text
1. Identify current state.
2. Pull or locate the latest relevant Windows text/table logs.
3. Spawn required sub-agents on Mac.
4. Wait for required findings.
5. Integrate findings.
6. Write the Experiment Brief.
7. Edit code/config/docs.
8. Run local validation.
9. Commit and push.
10. Windows git pull --ff-only.
11. Run CARLA readiness guard.
12. Run one CARLA step: collect, eval, train, or PPO.
13. Re-run CARLA readiness guard before every next CARLA step.
14. Pull complete text/table logs for the round.
15. Analyze results and apply 2048/4096 gates.
16. Write result document.
17. Commit and push the report.
```

Short form:

```text
Sub-agents first
Experiment Brief second
Edit third
Validate fourth
Windows CARLA run fifth
Document sixth
```

## 4. Sub-Agent Phase

### 4.1 Mandatory Core Agents

For a triggered sub-agent stabilization round, spawn these by default:

```text
Agent A - Log Analyst
Agent B - Code Auditor
Agent C - Mechanism Verifier
Agent D - Safety Regression Analyst
Agent J - Hardcoding / Privileged Controller Auditor
```

### 4.2 Mandatory Conditional Agent

Add this whenever the round uses imitation/data:

```text
Agent F - Imitation/Data Auditor
```

This is mandatory for:

```text
expert dataset collection
behavior cloning
sequence BC
frame BC
recovery datasets
target relabeling
DAgger-style data
policy-generated targets
```

### 4.3 Optional Agents

Use only when needed:

```text
Agent E - Ablation Planner
  Use when several experiment variants or knobs are plausible.

Agent G - Control/RL Specialist
  Use when the next intervention is unclear or touches PPO/control/reward logic.

Agent H - Experiment Runner
  Use rarely. CARLA execution normally stays with the main agent.

Agent I - Documentation Auditor
  Use for large direction changes or paper-relevant result cleanup.
```

### 4.4 Pre-Edit Barrier

Before sub-agent findings are integrated, do not:

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
```

## 5. Experiment Brief Barrier

The main agent must write an Experiment Brief before editing files.

Required fields:

```text
Round name:
Starting checkpoint:
Baseline and blocked run:
Primary hypothesis:
Sub-agent consensus:
Allowed changes:
Rejected changes:
Files to edit:
Mechanism metrics:
Data/BC checks, if relevant:
Hardcoding / privileged-assist risk:
Assist parity plan:
2048 gate:
4096/eval gate:
Stop conditions:
```

The brief can be in a progress update, work note, or the top section of a round result document.

## 6. Local Mac Validation

Before using Windows CARLA, run local checks appropriate to the changed files.

Default checks:

```bash
python -m pytest tests
python -m rcs_ac_first.scripts.smoke_check --config configs/<active_config>.yaml --device cpu
```

For script/config changes, also run targeted help/import checks where useful.

Do not skip local validation just because Windows will run the real experiment.

## 7. Windows Sync

After Mac commits and pushes:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "Set-Location C:\Users\thddy\Documents\code\RCS-AC-first; git pull --ff-only; git log -1 --oneline"'
```

If the Mac/Codex environment lacks system Tailscale routing, start userspace Tailscale and use:

```bash
-o 'ProxyCommand=nc -x localhost:1055 %h %p'
```

Before the first Windows command in every stabilization round, perform the Mac-side remote tunnel preflight from `01_remote_mac_windows_stabilization_workflow.md`:

```bash
TS_SOCKET="$HOME/.local/run/tailscaled-codex.sock"

lsof -nP -iTCP:1055 -sTCP:LISTEN || true
tailscale --socket="$TS_SOCKET" status
rcswinssh song@100.87.219.58 'hostname'
```

Required tunnel behavior:

```text
1. Reuse `$HOME/.local/run/tailscaled-codex.sock`.
2. Reuse `$HOME/.local/share/tailscale-codex`.
3. Use SSH keepalive/ControlPersist wrappers for long Windows commands.
4. Do not create a fresh `/tmp` Tailscale state during a round.
5. If Tailscale reports `Logged out` or `NeedsLogin`, fix the Mac tunnel before debugging Windows SSH, git, CARLA, or Python.
```

The previous `Connection closed by UNKNOWN port 65535` failure was caused by the Mac-side userspace SOCKS/Tailscale session disappearing or restarting with fresh state. It was not a CARLA or RCS-AC model failure. Future rounds should classify this as remote-tunnel infrastructure and recover before starting or judging experiments.

The Windows working tree may contain untracked generated artifacts such as dataset summaries. These are acceptable if `git pull --ff-only` still succeeds.

## 8. CARLA Readiness Guard

Before every CARLA-dependent command, run:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\ensure_carla_ready.ps1"'
```

The guard must pass before dataset collection, CARLA evaluation, and PPO training.

Behavior cloning training with `rcs_ac_first.scripts.train_bc` does not require CARLA after the dataset exists, so it does not need the CARLA readiness guard. It still needs normal local/Windows Python validation and log capture.

It checks:

```text
1. `CarlaUE4-Win64-Shipping` process exists.
2. TCP port 2000 accepts connections.
3. Python can import `carla`, connect to the server, and read the current world/map.
4. The healthy state remains stable for a short settle window.
```

If the checks fail, the guard restarts CARLA through:

```text
scripts/windows/start_carla_interactive.ps1
```

Do not rely on port-open alone. A port can briefly open while CARLA is not ready for `client.load_world()` or streaming.

## 9. CARLA Execution Rules

Run only one CARLA job at a time.

Before each CARLA-dependent item below, run `ensure_carla_ready.ps1`:

```text
dataset collection
deterministic BC eval
2048 PPO smoke
4096 PPO smoke
final checkpoint eval
```

Do not interpret these as mandatory steps for every round. A plain PPO stabilization round may only need eval -> 2048 PPO -> final eval. An imitation/data round may additionally need collect -> BC train -> BC eval before PPO. The guard rule applies to the CARLA steps, not to every possible script in every round.

Default gate:

```text
Do not run 4096 automatically.
Run 2048 first.
Analyze the 2048 gate.
Continue to 4096 only if the 2048 gate passes or clearly warrants continuation.
```

Stop after 2048 when:

```text
primary failure is unchanged
failure preconditions are unchanged
new severe failure appears
route progress collapses
mechanism did not activate
BC init eval regresses severely
NaN/Inf/checkpoint failure occurs
```

## 10. Log Pullback Rule

After a round, pull complete text/table logs for every run in the experiment.

Create one round folder:

```bash
DEST="$HOME/code/windows_results/rcs-ac-first/experiments/<round_name>"
mkdir -p "$DEST"/{logs,outputs,summary,remote_runs}
```

Pull by default:

```text
logs/<run_name>*/
outputs/**/<run_name>*/summary.json
outputs/**/<run_name>*/bc_eval_log.csv
outputs/**/<run_name>*/env_step_log.csv
data/expert/*<dataset_name>*.summary.json
remote_runs/logs/<run_name>* if used
```

Do not pull by default:

```text
checkpoints/
*.pt
data/**/*.npz
full rgb/semantic/lidar frame trees
```

Exception:

```text
Pull selected sample images when visual diagnosis is needed.
```

## 11. Result Analysis

The main agent must report regressions before improvements.

Minimum result sections:

```text
1. What failed or regressed.
2. What improved.
3. What stayed unchanged.
4. Whether the mechanism activated.
5. Whether the result depends on hardcoded or privileged control assist.
6. PASS / WARN / FAIL.
7. Next ablation.
```

For imitation/data rounds, include:

```text
dataset distribution
target correctness
BC strength
BC eval before PPO
whether PPO should have been allowed
```

For controller-assisted rounds, include:

```text
assist active rate
assist parity
assist-off or same-assist ablation requirement
paper-relevance label
```

## 12. Result Documentation

Each round must create or update:

```text
docs/stabilization_rounds/<NN_round_name>.md
```

The document must include:

```text
sub-agent findings or a summary of them
Experiment Brief
commands/runs used
metrics
gates
operational issues
log pullback location
next recommendation
```

If any required sub-agent was unavailable, state:

```text
Sub-agent execution unavailable; main agent followed the same role structure manually.
```

## 13. Practical Command Template

```text
User:
  다음 라운드 서브에이전트 구조로, Mac-Windows 원격 실행 방식까지 포함해서 안정화 진행해줘.

Main Agent:
  1. Read 00/01/02 operating guides.
  2. Identify latest logs/checkpoints/configs.
  3. Spawn A/B/C/D/J and F if data/BC is involved.
  4. Integrate findings.
  5. Write Experiment Brief.
  6. Edit/validate/commit/push.
  7. Run Mac-side Tailscale/SSH tunnel preflight.
  8. Windows pull.
  9. Run ensure_carla_ready.ps1 before every CARLA command.
  10. Pull complete text/table logs.
  11. Document regression-first result and push.
```

## 14. Non-Negotiable Rules

```text
1. Sub-agent findings come before experiment design when the sub-agent trigger is used.
2. Experiment Brief comes before file edits.
3. Mac does analysis and edits; Windows runs CARLA.
4. Never run two CARLA jobs in parallel.
5. Run Mac-side Tailscale/SSH tunnel preflight before the first Windows command.
6. Run CARLA readiness guard before every CARLA-dependent command.
7. Never promote a checkpoint without baseline comparison.
8. Never claim RCS-AC algorithm superiority from controller-assisted results unless assist parity or assist-off ablation supports it.
9. Always document failed experiments.
```
