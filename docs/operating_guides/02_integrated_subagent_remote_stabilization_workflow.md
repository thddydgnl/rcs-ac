# Integrated Sub-Agent + Remote CARLA Stabilization Workflow

Created: 2026-05-01

## 0. Purpose

This guide defines the combined stabilization workflow when both of these are true:

```text
1. The round should follow the sub-agent stabilization structure.
2. CARLA execution happens on the Windows desktop through Mac-to-Windows SSH.
```

This document connects the short core workflow with Windows/CARLA execution:

```text
docs/operating_guides/00_agent_workflow_rules.md
docs/operating_guides/03_lite_subagent_stabilization_workflow.md
```

Use `01_remote_mac_windows_stabilization_workflow.md` only when detailed remote setup or troubleshooting is needed.
Use `04_full_subagent_roles_and_legacy_rules.md` only after Full escalation.
If details conflict, use the stricter safety rule.

## 1. Trigger Phrase

For this repository, any request to run a stabilization round activates both workflows by default.

The short default trigger is:

```text
안정화 라운드 진행해줘
```

It is equivalent to:

```text
다음 라운드를 Lite Subagent Workflow로 시작하고, Mac-Windows 원격 실행 방식까지 포함해서 안정화 진행해줘.
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

the main agent must use this integrated workflow without asking for confirmation. State in the first progress update that the short trigger is being treated as Lite sub-agent + Mac-Windows remote stabilization workflow by default.
The default sub-agent mode is Lite. State whether the round starts in Lite mode or is escalated to Full mode, and cite the escalation reason when Full is used.

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
3. Choose Lite or Full mode using `03_lite_subagent_stabilization_workflow.md`.
4. Spawn required sub-agents on Mac.
5. Wait for required findings.
6. Integrate findings.
7. Write the Experiment Brief.
8. Edit code/config/docs.
9. Run local validation.
10. Commit and push.
11. Windows git pull --ff-only.
12. Run CARLA readiness guard.
13. Run one CARLA step: collect, eval, train, or PPO.
14. Re-run CARLA readiness guard before every next CARLA step.
15. Pull complete text/table logs for the round.
16. Analyze results and apply 2048/4096 gates.
17. Write result document.
18. Commit and push the report.
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

### 4.1 Default Lite Agents

For an ordinary stabilization round, use the Lite workflow from:

```text
docs/operating_guides/03_lite_subagent_stabilization_workflow.md
```

Default Lite agents:

```text
Agent A - Log / Result Analyst
Agent C - Mechanism / Gate Verifier
```

Add Lite Agent B only when the round edits code, non-trivial config wiring, logging, or checkpoint loading behavior:

```text
Agent B - Focused Code / Config Auditor
```

### 4.2 Mandatory Full Core Agents

For a high-risk stabilization round, spawn these:

```text
Agent A - Log Analyst
Agent B - Code Auditor
Agent C - Mechanism Verifier
Agent D - Safety Regression Analyst
Agent J - Hardcoding / Privileged Controller Auditor
```

### 4.3 Mandatory Conditional Full Agent

Add this whenever the round creates, modifies, relabels, or remixes imitation/data:

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

Existing dataset reuse without new targets may remain Lite unless another escalation condition is true.

### 4.4 Optional Full Agents

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

### 4.5 Pre-Edit Barrier

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

Full required fields:

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

Lite rounds may use the shorter Lite Experiment Brief from `03_lite_subagent_stabilization_workflow.md`, but they must still include:

```text
Round name:
Mode: Lite
Baseline:
Primary hypothesis:
Sub-agent summary:
Allowed changes:
Rejected changes:
Gate metrics:
Stop conditions:
Files to edit:
Escalation check:
```

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

For the first CARLA command of a stabilization round, and after any fatal dialog, `TaskGraphThreadNP` crash, `CrashReportClient`, streaming failure, `load_world()` timeout, reset timeout, or duplicate CARLA process state, use:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\ensure_carla_ready.ps1" -ForceRestart'
```

Do not treat a normal `CarlaUE4` wrapper plus one `CarlaUE4-Win64-Shipping` process as two CARLAs. That pair is expected. Treat more than one `CarlaUE4-Win64-Shipping`, more than one port-2000 listener, or any `CrashReportClient` as contaminated state that requires `-ForceRestart`.

Behavior cloning training with `rcs_ac_first.scripts.train_bc` does not require CARLA after the dataset exists, so it does not need the CARLA readiness guard. It still needs normal local/Windows Python validation and log capture.

It checks:

```text
1. `CarlaUE4-Win64-Shipping` process exists.
2. TCP port 2000 accepts connections.
3. Python can import `carla`, connect to the server, and load the target town, default `Town04`.
4. The target town remains stable for a short settle window.
```

The target-town probe matters. CARLA can report a healthy process, open port, and readable default world such as `Town10HD_Opt`, then still fail on the experiment's immediate `client.load_world("Town04")` with streaming refused or timeout. A guard PASS is valid only after the target town has loaded successfully.

If the checks fail, the guard restarts CARLA through:

```text
scripts/windows/start_carla_interactive.ps1
```

Do not rely on port-open or `get_world()` alone. A port can briefly open while CARLA is not ready for `client.load_world()` or streaming.

If the target-town probe fails repeatedly, the guard restarts CARLA internally after the configured consecutive health-failure threshold. Keep the default unless debugging infrastructure:

```text
TargetTown=Town04
ProbeTimeoutSeconds=120
StableSeconds=15
HealthFailureRestartThreshold=2
```

If a crash or timeout happens after the guard passes, classify the current CARLA state as contaminated. Retry only after `ensure_carla_ready.ps1 -ForceRestart`; do not proceed to PPO or the next eval from that simulator instance.

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

Crash/fatal handling is non-negotiable:

```text
1. `Fatal error! Crash in runnable thread TaskGraphThreadNP ...`, CrashReportClient, streaming timeout, `load_world()` timeout, or reset timeout is infrastructure contamination first, not model failure.
2. If the Python command did not complete and write metrics, discard that run and retry after `ensure_carla_ready.ps1 -ForceRestart`.
3. If metrics were fully written before the crash appeared, keep the metrics but force-restart before the next CARLA command.
4. Never continue from a contaminated CARLA state into 2048 PPO, 4096 PPO, or checkpoint promotion.
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
  안정화 라운드 진행해줘.

Main Agent:
  1. Read 00 and 03.
  2. Read relevant 02 sections if Windows/CARLA execution is needed.
  3. Identify latest logs/checkpoints/configs.
  4. Start Lite by default: spawn A/C, plus B only if needed.
  5. Escalate to Full if any high-risk condition is present.
  6. Integrate findings.
  7. Write Lite or Full Experiment Brief.
  8. Edit/validate/commit/push.
  9. Run Mac-side Tailscale/SSH tunnel preflight.
  10. Windows pull.
  11. Run ensure_carla_ready.ps1 before every CARLA command.
  12. Pull complete text/table logs.
  13. Document regression-first result and push.
```

## 14. CARLA Crash/Timeout Guardrails

When CARLA shows `Fatal error! Crash in runnable thread TaskGraphThreadNP` or a Python timeout while waiting for the simulator:

```text
1. Treat it as infrastructure until proven otherwise.
2. Confirm no second CARLA job is running.
3. Run ensure_carla_ready.ps1 with TargetTown Town04; use ForceRestart after a failed load/eval.
4. Verify the guard prints PYTHON_PROBE_TARGET_ALREADY_LOADED=Town04 before launching train/eval.
5. Verify shared_core/env/carla_env.py does not call client.load_world(Town04) when the current world is already Town04.
6. Do not overwrite Windows shared_core/env/carla_env.py with an older standalone env/carla_env.py layout.
7. If shared_core/env/carla_env.py must be patched remotely, preserve shared_core imports, action_contract_version, target-speed longitudinal control, action_smoothing, maneuver_command route hints, semantic aux masks, and advanced StateBuilder input objects.
```

Before the first CARLA command after a Windows pull, run this drift check:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\check_shared_core_carla_env.ps1
```

The check must print:

```text
SHARED_CORE_CARLA_ENV_CHECK=PASS
```

If it fails, stop the round and fix the Windows `shared_core/env/carla_env.py` patch before launching train/eval. A failed check means the repo wrapper may still be correct while the runtime CARLA environment is using a stale external dependency.

The readiness guard and the env must work together:

```text
guard responsibility:
  start CARLA, wait for port, target-load Town04, confirm stable get_world.

env responsibility:
  reuse current Town04 world when already loaded.
  only call load_world when the current map is missing or not the requested town.
```

If the guard loads Town04 and the env immediately reloads Town04, the next command can still timeout or crash even though the guard reported healthy.

## 15. Non-Negotiable Rules

```text
1. Lite or Full sub-agent findings come before experiment design when the stabilization trigger is used.
2. Experiment Brief comes before file edits.
3. Mac does analysis and edits; Windows runs CARLA.
4. Never run two CARLA jobs in parallel.
5. Run Mac-side Tailscale/SSH tunnel preflight before the first Windows command.
6. Run CARLA readiness guard before every CARLA-dependent command.
7. Never promote a checkpoint without baseline comparison.
8. Never claim RCS-AC algorithm superiority from controller-assisted results unless assist parity or assist-off ablation supports it.
9. Always document failed experiments.
```
