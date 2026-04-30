# RCS-AC-first Stabilization Round Agent Workflow Rules

Created: 2026-04-29

## 0. Purpose

This document defines the standard operating workflow for future RCS-AC-first stabilization rounds.

The trigger phrase is:

```text
서브 에이전트 구조로 안정화 진행해줘
```

Canonical trigger phrases:

```text
서브에이전트로 안정화 진행해줘
서브에이전트 구조로 안정화 진행해줘
다음 라운드 서브 에이전트로 안정화 진행해줘
```

When any trigger phrase is used, the stabilization round should be executed with a main-agent orchestration structure and specialized sub-agents.

Correct trigger phrases, if the older Korean text above appears corrupted by encoding:

```text
서브에이전트로 안정화 진행해줘
서브에이전트 구조로 안정화 진행해줘
다음 라운드 서브 에이전트로 안정화 진행해줘
다음 단계를 서브에이전트로 안정화 진행해줘
```

The goal is not to generate many configs quickly. The goal is to identify the real failure mode, verify whether it is caused by code, configuration, control, reward, data, or policy behavior, then run the smallest defensible experiment that can move the system forward.

## 1. Current Stabilization Context

Current project state:

```text
Project root:
  C:\Users\thddy\Documents\code\RCS-AC-first

Runtime dependency:
  C:\Users\thddy\Documents\code\shared_core

Current main checkpoint family:
  BC sequence checkpoint
  BC initialized PPO checkpoint
  clean stabilization checkpoints
  control_v2b checkpoint
```

Current known bottleneck after recovery_imitation_v2_lateral_target:

```text
Resolved or improved:
  - PPO std / exploration was reduced.
  - GRU sequence BC was added.
  - target-speed control was added.
  - longitudinal zero-speed stuck collapse was reduced.
  - collision did not dominate the latest PPO training runs.
  - recovery imitation improved clean deterministic eval.
  - recovery_imitation_v2_lateral_target reached goal in deterministic 512-step eval.

Remaining bottleneck:
  - offroad is the dominant failure mode.
  - training-time offroad still appears even when deterministic eval succeeds.
  - lane_offset and heading_error can still grow before termination.
  - clean eval can be strong while PPO training distribution remains unstable.

Current default next target:
  reduce training offroad while preserving recovery_imitation_v2_lateral_target goal-reaching eval behavior
```

## 2. Required Operating Principle

Each stabilization round must follow this rule:

```text
Sub-agents first, experiment design second, modify third, run fourth, document fifth.
```

Do not start by editing reward weights blindly.

Do not start by copying or creating a new config before the sub-agent findings are integrated.

Do not run many configs at once unless each config tests a clearly different hypothesis.

Do not treat a single 512-step eval as proof of stability. It is only a probe.

Do not treat PPO training reward as the main success metric. The core metrics are route progress, done reason, collision/offroad/stuck rates, and failure-precondition diagnostics.

## 3. Main Agent Role

The main agent is the orchestrator and final decision-maker.

Responsibilities:

```text
1. Identify the latest relevant runs.
2. Select logs/checkpoints/configs for analysis.
3. Spawn specialized sub-agents only for parallel, well-scoped analysis.
4. Prevent duplicate work between sub-agents.
5. Integrate findings into one diagnosis.
6. Produce one explicit Experiment Brief.
7. Choose one next experiment from the Experiment Brief.
8. Apply code/config changes only after the Experiment Brief exists.
9. Run validation and smoke tests.
10. Update result documentation.
11. State whether the round is PASS, WARN, or FAIL.
```

The main agent must own final judgment. Sub-agent outputs are evidence, not automatic decisions.

### 3.1 Pre-Edit Barrier

The default stabilization workflow has a hard barrier:

```text
Do not create or edit experiment configs before the core analysis agents report.
Do not edit runtime/control/reward code before the core analysis agents report.
Do not start CARLA training before the main agent writes the Experiment Brief.
```

Core analysis agents are:

```text
Agent A - Log Analyst
Agent B - Code Auditor
Agent C - Mechanism Verifier
Agent D - Safety Regression Analyst
Agent J - Hardcoding / Privileged Controller Auditor
```

Minimum evidence required before config/code edits:

```text
1. Agent A identifies the latest numeric failure mode or confirms the previous result document is enough.
2. Agent B confirms the code/config/logging path for the proposed mechanism or identifies blockers.
3. Agent C defines activation metrics for the proposed mechanism.
4. Agent D defines 2048 stop/continue gates.
5. Agent J confirms whether the proposed or existing mechanism creates hardcoded or privileged-control advantage.
```

If an agent is delayed or unavailable, the main agent may proceed only when all are true:

```text
the previous result document already names the exact next experiment
the code path has been inspected locally by the main agent
the main agent writes the missing-agent assumption in the Experiment Brief
the round is marked as higher-risk in the result document
```

This exception is for continuation after tool failure or context interruption, not for routine speed.

### 3.2 Experiment Brief Requirement

Before the first code/config edit, the main agent must write a concise Experiment Brief in the working notes or user update.

Required fields:

```text
Round name:
  stable unique name for configs/runs/docs

Starting checkpoint:
  exact checkpoint path

Baseline:
  current best metrics and latest failed/blocked metrics

Primary hypothesis:
  one sentence

Allowed changes:
  exact config/code/data changes allowed this round

Rejected changes:
  changes intentionally excluded to keep the experiment interpretable

Mechanism metrics:
  activation, coverage, timing, false activation, precondition metrics

2048 gate:
  PASS/WARN/FAIL rules and stop conditions

4096/eval gate:
  promotion rules

Implementation plan:
  files to edit and commands to run
```

After the Experiment Brief is written, the main agent may implement the planned edits.

### 3.3 No Config-First Rule

The following pattern is not allowed as the default:

```text
spawn sub-agents
immediately create config
run 2048 while waiting for sub-agents
use sub-agent outputs only afterward
```

Correct default pattern:

```text
spawn sub-agents
wait for at least A/B/C/D or enough justified evidence
integrate findings
write Experiment Brief
then edit config/code
then validate
then run 2048
```

Reason:

```text
The sub-agent structure exists to improve diagnosis before the experiment is designed.
If configs are created first, the round becomes ordinary trial-and-error with delayed commentary.
```

## 4. Standard Sub-Agent Structure

Core analysis agents:

```text
Agent A: Log Analyst
Agent B: Code Auditor
Agent C: Mechanism Verifier
Agent D: Safety Regression Analyst
Agent J: Hardcoding / Privileged Controller Auditor
```

Conditional specialist agents:

```text
Agent E: Ablation Planner
Agent F: Imitation/Data Auditor
Agent G: Control/RL Specialist
```

Optional execution/documentation agents:

```text
Agent H: Experiment Runner
Agent I: Documentation Auditor
```

Recommended structures:

```text
Small round:
  Main + A + B + C + D + J

Ambiguous round:
  Main + A + B + C + D + J + E

Imitation/data round:
  Main + A + B + C + D + J + F

Multi-variable or unclear round:
  Main + A + B + C + D + J + E + F or G as needed
```

Rationale:

```text
1. Log Analyst and Code Auditor identify what failed and whether the proposed fix can actually work.
2. Mechanism Verifier checks whether the intended intervention actually activated in runtime logs.
3. Safety Regression Analyst prevents false wins where one metric improves but driving quality regresses.
4. Imitation/Data Auditor is mandatory whenever BC, expert data, recovery datasets, or target relabeling are involved.
5. Control/RL Specialist is useful when the next intervention is unclear or touches PPO/control/reward logic.
6. Ablation Planner is useful when several plausible knobs exist and the round needs strict experimental ordering.
7. Hardcoding / Privileged Controller Auditor is mandatory because stabilization must not silently become rule-based driving.
```

Do not add Experiment Runner or Documentation Auditor unless the round is large enough to justify them.
CARLA execution and best-checkpoint promotion should normally stay with the Main Agent.

## 4.1 Model And Reasoning Policy

The default stabilization-round model policy is:

```text
Main Agent:
  model: GPT-5.5
  reasoning_effort: high

Sub Agents:
  model: GPT-5.5
  reasoning_effort: medium
```

Rationale:

```text
1. The Main Agent owns final diagnosis, implementation decisions, and tradeoff integration.
   It should use high reasoning when available.

2. Sub-agents are assigned narrower analysis tasks.
   Medium reasoning is the default because it is efficient while still strong enough for log analysis, code audit, and control/RL proposal work.

3. Use high reasoning for a sub-agent only when the assigned task is unusually high risk, for example:
   - auditing a suspected safety-critical action-contract bug
   - redesigning recurrent PPO logic
   - changing reward/control code that affects all experiments
```

Execution rule:

```text
When spawning sub-agents for this workflow, explicitly request:
  model = GPT-5.5
  reasoning_effort = medium

The main agent should remain on GPT-5.5 with high reasoning when the environment supports it.
```

Fallback rule:

```text
If GPT-5.5 or explicit reasoning control is unavailable in the active environment:
  - use the strongest available model for the Main Agent
  - use the default available model for Sub Agents
  - state the fallback clearly in the stabilization-round summary
```

## 5. Agent A - Log Analyst

### 5.1 Objective

Find what actually failed in the latest runs.

This agent must analyze logs numerically and avoid unsupported explanations.

### 5.2 Inputs

Possible inputs:

```text
logs/<run_name>/training_log.csv
outputs/<env_output_root>/<run_name>/env_step_log.csv
outputs/bc_eval/<run_name>/bc_eval_log.csv
outputs/bc_eval/<run_name>/summary.json
checkpoints/<run_name>/last.pt metadata if needed
```

### 5.3 Required Analysis

The Log Analyst must report:

```text
1. Run identity
   - run name
   - config path if identifiable
   - checkpoint path if identifiable
   - total rows
   - total updates
   - episode count

2. Done reason distribution
   - collision
   - offroad
   - stuck
   - timeout
   - goal
   - other

3. Route progress
   - mean
   - max
   - final per episode
   - first half vs second half
   - best episode summary

4. Failure precondition windows
   - last 5 steps before done
   - last 10 steps before done
   - last 20 steps before done when available

5. Control/action diagnostics
   - policy_action_steer
   - policy_action_longitudinal
   - applied steer/throttle/brake
   - target_speed_kmh
   - speed_error_kmh
   - smoothed_steer/throttle/brake
   - action jitter

6. Lateral diagnostics
   - lane_offset mean/p90/max
   - heading_error mean/p90/max
   - offroad_candidate
   - offroad_steps
   - lane invasion if available

7. Longitudinal diagnostics
   - speed mean/p50/p90/max
   - low_speed_steps
   - throttle/brake saturation
   - target speed too high/too low

8. Reward component diagnostics
   - progress reward
   - dense progress reward
   - lane offset penalty
   - heading error penalty
   - speed reward
   - curve/turn speed penalty
   - offroad/collision/stuck penalties

9. Model diagnostics
   - policy std
   - fusion weights
   - reliability
   - risk scalar
   - contribution
```

### 5.4 Required Output Format

The Log Analyst must return:

```text
Failure mode ranking:
  1. <most likely failure mode> - numeric evidence
  2. <second likely failure mode> - numeric evidence
  3. <third likely failure mode> - numeric evidence

Keep:
  - settings that worked

Change:
  - settings that should be modified

Do not change:
  - settings that are not implicated by the logs

Recommended next experiment:
  - one clear experiment
  - expected metric movement
```

## 6. Agent B - Code Auditor

### 6.1 Objective

Determine whether the observed failure could be caused by code, config, logging, contract mismatch, or environment behavior.

This agent prevents wasting time tuning reward when the actual issue is an implementation mismatch.

### 6.2 Required Audit Areas

The Code Auditor must inspect relevant files only. Typical files:

```text
shared_core/env/carla_env.py
shared_core/env/log_schema.py
RCS-AC-first/rcs_ac_first/scripts/train.py
RCS-AC-first/rcs_ac_first/scripts/evaluate_bc.py
RCS-AC-first/rcs_ac_first/runtime/action.py
RCS-AC-first/rcs_ac_first/runtime/observation.py
RCS-AC-first/rcs_ac_first/rl/ppo_agent.py
RCS-AC-first/rcs_ac_first/models/rcs_ac_model.py
RCS-AC-first/configs/<active_config>.yaml
```

Audit checklist:

```text
1. Action contract
   - action = [steer, longitudinal]
   - steer range [-1, 1]
   - longitudinal range [-1, 1]
   - throttle/brake or target-speed conversion correct

2. Train/eval config consistency
   - train.py and evaluate_bc.py use the same env config fields
   - checkpoint loading does not silently ignore important keys
   - log_std override is applied only when intended

3. Control layer
   - target-speed mapping is correct
   - min/max target speed are applied
   - smoothing does not lag steering too much
   - throttle/brake caps are logged

4. Offroad and stuck logic
   - offroad threshold is interpretable
   - offroad grace/min steps are applied
   - stuck_min_steps and low_speed_steps are correct
   - junction/route edge cases are handled

5. Reward logic
   - reward keys exist in logs
   - signs are correct
   - scaling is not accidentally overwhelming progress
   - terminal penalties are applied once

6. Observation and auxiliary contract
   - flatten order is stable
   - auxiliary dim is correct
   - reliability/risk fields are available when enabled

7. Logging integrity
   - fields used for analysis are actually populated
   - no wrong column names are being used
   - eval log and env log differences are explicit
```

### 6.3 Required Output Format

The Code Auditor must return:

```text
Code risks:
  - [risk] file/function/evidence/impact

Confirmed contracts:
  - [contract] evidence

Possible implementation bug:
  - yes/no
  - if yes, patch recommendation

Config mismatch:
  - yes/no
  - if yes, exact fields

Before next experiment:
  - required code change
  - required logging fix
  - or "no blocking code issue found"
```

## 7. Agent C - Mechanism Verifier

### 7.1 Objective

Verify that the mechanism introduced by the round actually activated and moved the intended internal variables.

This agent exists because a run can look better or worse for the wrong reason. A config value, loss weight, gate, recovery correction, or data target must be proven active in logs or model outputs before the round can be interpreted.

The Mechanism Verifier is mandatory for every stabilization round.

### 7.2 Required Checks

```text
1. Config-to-runtime activation
   - active config contains the new fields
   - train/eval scripts pass those fields into runtime
   - logs contain the expected fields
   - values are not all defaults unless defaults are expected

2. Intervention-specific activation
   - speed gate:
     speed_gate_active, target speed, throttle cap, brake request, final throttle/brake
   - lateral recovery:
     raw steer, corrected steer, correction magnitude, correction sign, activation rate
   - reward tuning:
     reward component columns exist and move in the expected direction
   - PPO tuning:
     log_std, entropy, learning rate, clip range, grad norm, actor/value losses
   - Shapley/dropout:
     contribution, shapley raw/normalized values, dropout consistency loss
   - imitation:
     target action differs from policy action only where intended
     BC action_mean moves toward target without destroying clean eval

3. Mechanism-to-metric connection
   - the mechanism activated before the failure, not only after the failure was inevitable
   - the target precondition metric improved
   - no new severe precondition appeared
```

### 7.3 Required Output Format

The Mechanism Verifier must return:

```text
Mechanism:
  - what was supposed to activate

Activation evidence:
  - columns / metrics / before-after values

Correctness check:
  - activated as intended: yes/no/partial
  - sign/direction correct: yes/no/unknown
  - activation timing correct: yes/no/unknown

Metric connection:
  - precondition improved/regressed/unchanged

Interpretation:
  - success
  - false win
  - false failure
  - implementation issue
```

## 7.5 Agent D - Safety Regression Analyst

### 7.5.1 Objective

Identify what became worse, even if the headline metric improved.

This agent exists because autonomous-driving stabilization often creates false wins:

```text
offroad decreases but stuck increases
route progress improves but collision appears
eval passes but PPO training distribution worsens
speed gate helps turns but creates no-progress episodes
steering correction improves lane recovery but increases action jitter
imitation improves one scenario but reduces general driving ability
```

The Safety Regression Analyst must be used by default for stabilization rounds.

### 7.5.2 Inputs

Required inputs:

```text
previous baseline training env_step_log.csv
previous baseline training_log.csv
previous baseline final eval summary.json
new candidate training env_step_log.csv if available
new candidate training_log.csv if available
new candidate eval summary.json if available
active config diff or description
```

### 7.5.3 Required Analysis

The Safety Regression Analyst must compare baseline vs candidate on:

```text
1. Safety failures
   - collision count/rate
   - offroad count/rate
   - stuck count/rate
   - lane invasion if meaningful

2. Driving usefulness
   - route_progress_mean
   - route_progress_max
   - episode length mean/max
   - max_steps episode count
   - no-progress or early-deadlock signals

3. Control stability
   - steer jitter
   - longitudinal jitter
   - throttle/brake oscillation
   - brake-hold behavior
   - target speed vs actual speed

4. Failure migration
   - whether the old failure reduced
   - whether a new failure appeared
   - whether failure moved to a later route segment

5. Evaluation/training mismatch
   - deterministic eval improvement but training regression
   - training improvement but eval regression
   - short-horizon success that does not survive 4096
```

### 7.5.4 Required Output Format

```text
Regression summary:
  Improved:
    - metric / baseline / candidate
  Regressed:
    - metric / baseline / candidate / severity
  Unchanged:
    - metric / baseline / candidate

False-win risk:
  LOW / MEDIUM / HIGH
  reason:

Go/no-go recommendation:
  - continue to 4096
  - stop at 2048 and analyze
  - revert or narrow the change
```

### 7.5.5 Severity Rules

Use these default severity labels:

```text
CRITICAL:
  collision appears when baseline had zero
  route progress collapses near zero
  NaN/Inf or checkpoint failure

HIGH:
  stuck appears repeatedly
  offroad increases by more than 25%
  max_steps episodes disappear
  deterministic eval fails after training

MEDIUM:
  action jitter increases by more than 25%
  route progress mean drops by more than 10%
  target failure improves but another failure increases mildly

LOW:
  minor metric movement within expected noise
```

## 7.6 Agent J - Hardcoding / Privileged Controller Auditor

### 7.6.1 Objective

Detect whether a stabilization win is caused by the learned RCS-AC policy or by rule-based / privileged runtime assistance.

This agent is mandatory for every stabilization round.

The purpose is not to ban every safety layer. Real autonomous driving stacks often use low-level controllers and safety guards. The purpose is to prevent the project from accidentally claiming an algorithmic RCS-AC improvement when the measured gain actually comes from a hand-coded controller, route oracle, privileged simulator signal, or baseline-asymmetric runtime assist.

### 7.6.2 Definitions

Use these labels:

```text
LEARNED POLICY EFFECT:
  Improvement is mainly from model weights, learned fusion, reliability, risk, Shapley contribution,
  BC/RL policy behavior, or learned perception auxiliaries.

LOW-LEVEL CONTROL ASSIST:
  A generic controller converts a learned command into stable vehicle control.
  It is acceptable only if the same controller is available to every compared algorithm.

SAFETY GUARD:
  A narrow intervention that prevents extreme unsafe action or actuator conflict.
  It is acceptable only if it is logged, bounded, and applied equally across baselines.

HARDCODED DRIVING:
  Rule-based logic directly performs the driving task, such as steering toward route center,
  choosing lane/turn behavior, braking for route geometry, or recovering from offroad conditions
  using privileged route/lane signals.

PRIVILEGED SIGNAL:
  Any signal not produced by the policy's declared observation contract or not equally available
  to PPO/SAC/RCS-AC at inference time, for example route-manager lane offset, exact waypoint geometry,
  simulator-only offroad oracle, or expert/autopilot control during policy evaluation.
```

### 7.6.3 Required Audit Areas

The Hardcoding Auditor must inspect:

```text
1. Config assist flags
   - env.lateral_recovery.enabled
   - target-speed controller settings
   - speed gates
   - action smoothing
   - low-speed brake suppression
   - throttle/brake caps
   - offroad/junction special cases

2. Runtime code paths
   - shared_core/env/carla_env.py control resolution
   - any route-manager based correction
   - any rule that changes steer/throttle/brake after policy output
   - any policy_diagnostics field used to alter env control

3. Evaluation fairness
   - whether PPO/SAC baselines get the same controller/guard
   - whether RCS-AC receives extra route/lane/perception information
   - whether evaluation uses autopilot/expert route corrections
   - whether deterministic eval is benefiting from a different runtime than training

4. Log evidence
   - policy_action_steer / policy_action_longitudinal
   - requested_action_steer/throttle/brake
   - smoothed_steer/throttle/brake
   - final steer/throttle/brake
   - requested_applied_action_l1
   - lateral_recovery_active
   - lateral_recovery_correction
   - speed_gate_active / effective_active
   - throttle_cap_applied / brake_cap_applied
```

### 7.6.4 Required Metrics

Every round must report these hardcoding metrics when relevant columns exist:

```text
runtime_assist_active_rate:
  fraction of steps where any non-policy control assist changed final control

lateral_recovery_active_rate:
  fraction of steps where lateral recovery was active

mean_abs_lateral_recovery_correction:
  mean abs steering correction over all steps and active steps

max_abs_lateral_recovery_correction:
  maximum abs steering correction

speed_gate_active_rate:
  fraction of steps where a speed gate changed longitudinal behavior

requested_applied_action_l1_mean:
  mean absolute difference between requested policy-derived control and applied control

assist_share_of_steer:
  mean_abs_lateral_recovery_correction / mean_abs_policy_steer, when calculable

baseline_assist_parity:
  YES if PPO/SAC/RCS-AC use the same assist stack, otherwise NO

privileged_signal_used:
  YES if route/lane/offroad oracle signals directly alter action after the policy output
```

Suggested interpretation:

```text
LOW hardcoding risk:
  assist active rate < 5%, small action deltas, same assist stack across all algorithms.

MEDIUM hardcoding risk:
  assist active rate 5% to 25%, bounded safety-only intervention, or same controller across baselines.

HIGH hardcoding risk:
  assist active rate > 25%, steering correction frequently changes policy action,
  or RCS-AC alone receives the assist.

DISQUALIFY FOR ALGORITHM-ONLY CLAIM:
  privileged route/lane/controller logic is necessary for the win and is not shared by baselines.
```

### 7.6.5 Output Format

The Hardcoding Auditor must return:

```text
Hardcoding risk:
  LOW / MEDIUM / HIGH / DISQUALIFY-FOR-ALGORITHM-ONLY-CLAIM

Evidence:
  - metrics and log columns

Privileged signals:
  - yes/no
  - exact source

Assist parity:
  - same for PPO/SAC/RCS-AC: yes/no/unknown

Allowed interpretation:
  - algorithm-only improvement
  - policy + shared low-level controller improvement
  - controller-assisted result only
  - not suitable for paper comparison yet

Required ablation:
  - assist off
  - same assist for all baselines
  - assist-only / heuristic baseline
  - learned-policy-only eval
```

### 7.6.6 Hardcoding Gate

Before promoting a checkpoint as a paper-relevant best, one of these must be true:

```text
1. assist disabled:
   The candidate passes the relevant clean/corruption eval with rule-based assists disabled.

2. assist parity:
   The exact same low-level controller and safety guards are applied to RCS-AC, PPO, and SAC.

3. explicit ablation:
   The result is labeled as controller-assisted, and the document includes assist-off and assist-on comparisons.
```

If none are true, the checkpoint can be used as an engineering stepping stone, but must not be described as RCS-AC algorithm superiority.

### 7.6.7 Banned Stabilization Pattern

The following pattern is not allowed:

```text
observe failure
add a route/lane/speed rule that directly fixes the failure
promote the result as RCS-AC robustness
repeat until the route is solved
```

Acceptable alternatives:

```text
1. Use the rule only as a temporary diagnostic tool and document it as such.
2. Convert the failure into data and train the policy to handle it.
3. Apply the same rule to every baseline and compare RCS-AC-specific robustness on top.
4. Run assist-off ablations before any algorithmic claim.
```

## 7.7 Agent E - Ablation Planner

### 7.7.1 Objective

Prevent unstable, hard-to-interpret experiments by enforcing one primary hypothesis per round and defining the next ablation queue.

This agent is used when:

```text
1. More than one plausible fix exists.
2. A proposed experiment changes reward and control together.
3. A proposed experiment changes imitation data and PPO settings together.
4. A 2048 smoke did not improve and the next step is unclear.
5. A setting has several possible values and needs an ordered search.
```

### 7.7.2 Required Output

The Ablation Planner must produce:

```text
Primary hypothesis:
  - one sentence

Allowed changes this round:
  - change 1
  - change 2 only if necessary

Rejected changes this round:
  - change / reason

2048 gate:
  - exact pass criteria
  - exact stop criteria

4096 gate:
  - exact pass criteria
  - exact stop criteria

Ablation queue:
  1. next experiment if current fails
  2. next experiment if current partially works
  3. next experiment if current passes
```

### 7.7.3 One-Hypothesis Rule

Each stabilization round must have one primary hypothesis.

Examples:

```text
Good:
  Hypothesis: branch-turn failures happen because speed remains too high after lane error grows.
  Change: add branch-turn speed gate only.

Good:
  Hypothesis: recurrent policy lacks recovery behavior in high lane-offset states.
  Change: collect recovery imitation data and run sequence BC only.

Bad:
  Hypothesis: policy is unstable.
  Change: speed gate + reward rewrite + PPO LR change + new BC dataset.
```

### 7.7.4 2048 Gate Rule

Before running a 4096 formal smoke, the 2048 smoke must show at least one useful signal.

Default 2048 go criteria:

```text
no NaN/Inf
checkpoint created
collision does not appear if baseline had zero
stuck does not appear repeatedly
primary failure count decreases, or
primary failure precondition improves clearly
```

Default 2048 stop criteria:

```text
primary failure count is unchanged and precondition metrics are unchanged
new severe failure appears
route progress collapses
action jitter increases sharply
the new mechanism is not actually activating in logs
```

If the 2048 gate fails, do not run 4096 by default. Analyze the 2048 log first.

### 7.7.5 Regression-First Summary Rule

When reporting a round, summarize in this order:

```text
1. What regressed or failed.
2. What improved.
3. What stayed unchanged.
4. Whether the mechanism actually activated.
5. PASS / WARN / FAIL.
6. Next ablation.
```

This prevents overvaluing a single improved metric while ignoring safety or stability regression.

## 7.8 Agent F - Imitation/Data Auditor

### 7.8.1 Objective

Audit imitation learning, recovery datasets, expert targets, and BC checkpoint selection.

This agent is mandatory whenever a round uses:

```text
expert dataset collection
behavior cloning
sequence BC
frame BC
recovery relabeling
DAgger-style data
policy-generated target data
```

The goal is to prevent imitation from quietly overwriting a good driving policy or teaching the wrong target distribution.

### 7.8.2 Required Audit Areas

```text
1. Dataset distribution
   - route_progress min/mean/max
   - accepted / attempted count
   - acceptance rate
   - lane_offset and heading_error distribution
   - speed distribution
   - done/bad transition count

2. Target correctness
   - expert_action order is [steer, longitudinal]
   - steer target changes only when intended
   - longitudinal target is preserved when the round is steering-only
   - target_corrected_steer differs from raw policy steer where recovery is intended
   - no speed-gated throttle/brake target is accidentally mixed into steering-only BC

3. Temporal validity
   - sequence BC uses truly continuous windows, or
   - frame BC is used when accepted samples are sparse
   - episode_id and valid_mask are meaningful

4. BC strength
   - learning rate
   - epochs
   - steer_loss_weight / longitudinal_loss_weight
   - action_smooth_loss_weight
   - checkpoint epoch selection

5. BC regression risk
   - BC eval route progress
   - safety events
   - action jitter
   - whether later epochs overwrite clean behavior
```

### 7.8.3 Required Output Format

```text
Dataset quality:
  - usable / risky / invalid
  - numeric evidence

Target quality:
  - correct / partial / wrong
  - evidence

BC strength recommendation:
  - strong / gentle / too strong
  - checkpoint epoch recommendation

Go/no-go:
  - proceed to PPO
  - reduce BC strength
  - recollect dataset
  - change target design
```

### 7.8.4 Lessons From Previous Rounds

```text
1. Strong BC can reduce supervised loss while damaging clean driving.
2. Epoch 001 can outperform later epochs for policy-preserving recovery imitation.
3. Sparse accepted recovery samples should not automatically be treated as continuous sequence data.
4. Steering-only recovery requires the BC loss weights to be active in code, not only written in YAML.
5. A BC checkpoint should pass deterministic clean eval before PPO fine-tuning unless the round explicitly accepts higher risk.
```

## 7.9 Agent G - Control/RL Specialist

### 7.9.1 Objective

Propose the smallest technically defensible control, reward, PPO, or RL intervention when the next mechanism is unclear.

This agent is conditional, not mandatory. Use it when the logs and mechanism verification do not clearly point to a single next intervention.

### 7.9.2 Decision Scope

Possible intervention categories:

```text
1. PPO stabilization
   - learning rate
   - clip coefficient
   - entropy weight
   - log std
   - update epochs
   - minibatch size

2. Reward tuning
   - progress reward
   - lane offset penalty
   - heading error penalty
   - speed reward
   - curve speed penalty
   - terminal penalties

3. Control wrapper
   - target speed mapping
   - throttle/brake caps
   - action smoothing
   - lateral recovery steering correction

4. Imitation / data intervention
   - frame or sequence BC
   - high lane-offset expert relabel
   - DAgger-style data collection
   - recovery dataset

5. Evaluation changes
   - deterministic eval
   - random baseline
   - longer route eval
   - scenario split
```

### 7.9.3 Intervention Rules

```text
1. Change one primary mechanism per round.
2. Allow at most two secondary parameter changes.
3. Do not combine reward overhaul + controller overhaul + PPO overhaul in one run.
4. Preserve settings that solved previous failure modes.
5. State expected metric movement before running.
6. State failure criteria before running.
```

### 7.9.4 Required Output Format

```text
Recommended experiment:
  name:
  primary mechanism:
  config file:
  checkpoint source:

Changes:
  - exact fields or code modules

Expected improvement:
  - metric and expected direction

Risk:
  - likely downside

Acceptance criteria:
  PASS:
  WARN:
  FAIL:
```

## 8. Optional Agent H - Experiment Runner

Use this agent only when:

```text
1. The code/config decision is already made.
2. CARLA server is available.
3. Only one experiment should run at a time.
4. The main agent does not need to inspect intermediate output manually.
```

The Experiment Runner must not run multiple CARLA jobs in parallel.

Standard command sequence:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core

py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config <config> `
  --checkpoint <checkpoint> `
  --run-name <eval_probe_name> `
  --num-steps 512 `
  --device auto

py -3.12 -B -m rcs_ac_first.scripts.train `
  --config <config> `
  --init-checkpoint <checkpoint> `
  --init-strict true `
  --total-timesteps 2048 `
  --run-name <run_2048> `
  --device auto

py -3.12 -B -m rcs_ac_first.scripts.train `
  --config <config> `
  --init-checkpoint <checkpoint> `
  --init-strict true `
  --total-timesteps 4096 `
  --run-name <run_4096> `
  --device auto

py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config <config> `
  --checkpoint <new_checkpoint> `
  --run-name <final_eval_name> `
  --num-steps 512 `
  --device auto
```

Runner output:

```text
Commands executed
Exit status
Run names
Log paths
Checkpoint paths
Summary metrics
Any failed command
```

## 9. Optional Agent I - Documentation Auditor

Use this agent when the stabilization round changes project direction or produces important results.

Responsibilities:

```text
1. Check that experiment result docs are updated.
2. Ensure failed experiments are recorded.
3. Ensure configs and checkpoints are named consistently.
4. Create a concise comparison table.
5. Identify what should be cited later in paper/ablation writing.
```

## 10. Main-Agent Round Procedure

Every triggered stabilization round should use this sequence.

### Step 1 - Identify Current State

Main agent must identify:

```text
latest relevant checkpoint
latest relevant training run
latest relevant eval run
active config
known failure mode from previous round
current target, for example lateral_recovery_v1
```

If multiple latest runs exist, prefer:

```text
1. most recent formal 4096 run
2. its final deterministic eval
3. previous successful baseline for comparison
```

### Step 2 - Spawn Analysis Agents

Default parallel tasks:

```text
Agent A:
  Analyze latest logs and compare against previous baseline.

Agent B:
  Audit code/config contracts related to the suspected failure mode.

Agent C:
  Verify that the proposed mechanism activates in logs/model outputs.

Agent D:
  Identify regressions and false-win risk.

Agent J:
  Audit whether the latest/proposed improvement is caused by hardcoded driving, privileged signals, or baseline-asymmetric control assist.
```

Agents must have non-overlapping responsibilities.

Do not ask all agents the same broad question.

Conditional parallel tasks:

```text
Agent E:
  Use when several knobs or experiment variants are possible.
  Create the ablation queue and 2048/4096 gate criteria.

Agent F:
  Use whenever imitation, BC, expert data, recovery datasets, or target relabeling are involved.
  Audit dataset distribution, target correctness, and BC checkpoint risk.

Agent G:
  Use when the next intervention is still unclear after A/B/C/D/J.
  Propose one minimal control/RL/reward/PPO experiment.
```

### Step 3 - Wait For Required Agent Findings

Before designing the next experiment, the main agent must collect the required core-agent findings.

Default required outputs:

```text
Agent A:
  numeric failure summary and candidate root cause ranking

Agent B:
  code/config/logging feasibility and blockers

Agent C:
  mechanism activation metrics and interpretation rules

Agent D:
  2048 and 4096 pass/warn/fail gates

Agent J:
  hardcoding risk, privileged-signal risk, assist parity, and required ablation
```

The main agent should not create a new config, patch code, or start CARLA training during this waiting phase.

Allowed work while waiting:

```text
read existing logs
read existing docs
inspect existing code
verify current best paths
prepare a draft diagnosis without editing files
```

Not allowed while waiting:

```text
copy or create a new config
patch control/reward/model/runtime code
start 2048 or 4096 CARLA training
start a new dataset collection
promote a checkpoint
```

If a sub-agent result contradicts the initial plan, the main agent must pause implementation and resolve the contradiction in the Experiment Brief.

### Step 4 - Integrate Findings

Main agent must produce:

```text
Diagnosis:
  - most likely root cause
  - evidence
  - confidence

Not root cause:
  - issues that were checked and deprioritized

Regression check:
  - metrics that worsened
  - safety regressions
  - false-win risk

Mechanism check:
  - whether the intervention activated
  - whether activation timing/sign/direction were correct
  - whether precondition metrics moved as expected

Data check when imitation is involved:
  - whether dataset distribution matches the failure mode
  - whether targets are correct
  - whether BC strength is safe enough for PPO init

Hardcoding check:
  - whether any rule-based controller or privileged route/lane signal is responsible for the gain
  - whether the same assist is available to PPO/SAC baselines
  - whether assist-off or same-assist ablation is required before promotion

Next experiment:
  - exact change
  - why it is the smallest useful change
  - expected metrics

Gate decision:
  - 2048 pass criteria
  - 2048 stop criteria
  - whether 4096 is allowed after 2048
```

### Step 5 - Write Experiment Brief

The main agent must write an Experiment Brief before editing files.

Template:

```text
Experiment Brief

Round name:

Starting checkpoint:

Baseline and blocked run:

Primary hypothesis:

Sub-agent consensus:

Allowed changes:

Rejected changes:

Files to edit:

Mechanism metrics:

Hardcoding / privileged-assist risk:

Assist parity plan:

2048 gate:

4096/eval gate:

Stop conditions:
```

This brief can be written as a user update, a temporary work note, or the first section of a planned result document.

The important rule:

```text
No code/config edit should happen before this brief exists.
```

### Step 6 - Implement

Implementation should follow this rule:

```text
One round = one main hypothesis.
```

Examples:

```text
Good:
  Add lateral recovery steering correction and keep target-speed control unchanged.

Bad:
  Add lateral recovery, change PPO LR, change entropy, rewrite reward, change max speed, and add new dataset in one round.
```

Implementation order:

```text
1. create/update config files
2. patch code only if Agent B or main inspection found a code/logging gap
3. keep package config copy in sync when required
4. do not start CARLA until static validation and config load pass
```

### Step 7 - Validate

Minimum validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

If CARLA is available:

```text
512 deterministic eval probe
2048 PPO smoke
2048 gate analysis
4096 PPO smoke only if the 2048 gate passes
512 deterministic checkpoint eval
```

2048 gate rule:

```text
Do not run 4096 automatically.
If 2048 shows no improvement in the primary failure or its precondition metrics, stop and analyze.
If 2048 introduces a new severe regression, stop and analyze.
If the new mechanism did not actually activate, stop and fix code/config/logging before 4096.
If imitation was used and deterministic BC eval shows safety events or major route collapse, do not start PPO from that BC checkpoint.
If 2048 shows useful signal without severe regression, continue to 4096.
```

If CARLA is not available:

```text
static validation only
local smoke if relevant
document CARLA validation as blocked
```

### Step 8 - Document

Each round should append or create a result document under:

```text
docs/stabilization_rounds/
```

Recommended file naming:

```text
01_lateral_recovery_v1_plan.md
02_lateral_recovery_v1_result.md
03_clean_stabilization_comparison.md
```

## 11. Standard Stabilization Metrics

### 11.1 Primary Metrics

These decide whether a round worked:

```text
done_reason distribution
collision_rate
offroad_rate
stuck_rate
route_progress_mean
route_progress_max
episode_length_mean
```

### 11.2 Failure-Mode Metrics

For offroad:

```text
pre_done_lane_abs_mean
pre_done_lane_abs_p90
pre_done_heading_abs_mean
pre_done_heading_abs_p90
offroad_candidate_rate
offroad_steps_max
speed_before_offroad
steer_before_offroad
```

For collision:

```text
speed_before_collision
brake_before_collision
steer_before_collision
risk_scalar_before_collision
target_distance_before_collision
```

For stuck:

```text
low_speed_steps_max
speed_mean_before_stuck
target_speed_before_stuck
policy_longitudinal_before_stuck
throttle_before_stuck
brake_before_stuck
```

For unstable PPO:

```text
policy_std_mean
log_std_dim0
log_std_dim1
actor_loss
critic_loss
entropy
grad_norm
approx_kl if available
```

### 11.3 RCS-AC Specific Metrics

These must be preserved for robustness analysis:

```text
fusion_rgb/lidar/ego/route
reliability_rgb/lidar/ego/route
risk_scalar
contribution_rgb/lidar/ego/route
shapley_alignment_loss
dropout_consistency_loss
corruption_level
drop_rgb/drop_lidar/drop_ego/drop_route
```

### 11.4 Mechanism Activation Metrics

Every round must define the mechanism activation metrics before running.

Examples:

```text
lateral recovery:
  lateral_recovery_active
  lateral_recovery_raw_steer
  lateral_recovery_corrected_steer
  lateral_recovery_correction
  target_corrected_steer

speed gate:
  lateral_recovery_speed_gate_active
  lateral_recovery_speed_gate_target_speed_kmh
  lateral_recovery_speed_gate_throttle_cap
  final throttle
  final brake

imitation:
  policy_action_steer
  target_corrected_steer
  expert_action
  BC action_mean before/after
  BC eval route progress and safety events

reward:
  reward_component_<changed_component>
  terminal penalty count
  progress reward before/after
```

If these fields are missing, the first action is a logging/code fix, not another training run.

### 11.5 Imitation/Data Metrics

For imitation rounds, record:

```text
dataset path
num_transitions
attempted_transitions
acceptance_rate
route_progress min/mean/max
lane_offset abs mean/p90/max if available
heading_error abs mean/p90/max if available
speed mean/p50/p90 if available
bad_transition_count
done_count
steer_loss_weight
longitudinal_loss_weight
BC epoch selected
BC eval summary
```

### 11.6 Hardcoding / Assist Metrics

For every stabilization round, record these if the columns exist:

```text
lateral_recovery_enabled
lateral_recovery_active_rate
lateral_recovery_correction mean/active_mean/max
lateral_recovery_turn_boost_active_rate
lateral_recovery_speed_gate_active_rate
lateral_recovery_speed_gate_effective_active_rate
throttle_cap_applied_rate
brake_cap_applied_rate
requested_applied_action_l1 mean/p95/max
policy_action_steer abs_mean
requested_action_steer abs_mean
smoothed_steer abs_mean
final steer abs_mean
assist_share_of_steer
privileged_signal_used
baseline_assist_parity
assist_off_eval_result, if available
same_assist_baseline_result, if available
```

Interpretation rule:

```text
If runtime assist is active often enough to explain the win, the round is controller-assisted.
It may remain useful for engineering, but it is not an algorithm-only RCS-AC result.
```

## 12. PASS / WARN / FAIL Criteria

### 12.1 Static PASS

```text
compileall passes for RCS-AC-first
compileall passes for shared_core
config loads
checkpoint loads if used
dummy or eval forward is finite
```

### 12.2 512 Eval PASS

```text
completed_steps == 512
actions finite
actions in range
env close succeeds
summary.json created
route_progress_max > 0
collision_rate <= previous baseline or zero
stuck_rate <= previous baseline
```

### 12.3 2048 PPO Smoke PASS

```text
at least 16 PPO updates
no NaN/Inf
checkpoint created
training_log.csv created
env_step_log.csv created
done_reason recorded
```

### 12.4 4096 Formal Smoke PASS

```text
at least 32 PPO updates
no crash
checkpoint reloadable
primary failure mode improves or does not regress severely
secondary failure modes do not explode
```

### 12.5 WARN

Use WARN when:

```text
training completes but target metric does not improve
one failure mode improves while another worsens mildly
eval passes but training distribution is not clearly better
results are inconclusive due to short horizon
```

### 12.6 FAIL

Use FAIL when:

```text
compile or import fails
CARLA reset fails when server is expected available
PPO update fails
NaN/Inf appears
checkpoint is missing
route_progress collapses near zero
collision/offroad/stuck greatly worsen
```

### 12.7 Paper-Relevance Gate

Use a separate paper-relevance label in addition to PASS/WARN/FAIL:

```text
ALGORITHM-ONLY:
  The candidate works without rule-based driving assist or privileged action correction.

SHARED-ASSIST:
  The candidate uses a low-level controller or safety guard, but PPO/SAC/RCS-AC baselines use the exact same assist stack.

CONTROLLER-ASSISTED-ONLY:
  The candidate is useful engineering progress, but the gain may be caused by a controller or privileged route/lane signal.

NOT-PAPER-COMPARABLE:
  RCS-AC has a runtime assist or privileged signal that PPO/SAC do not have.
```

Promotion rule:

```text
A checkpoint may be called "best engineering checkpoint" even if controller-assisted.
A checkpoint may not be called "best RCS-AC algorithm checkpoint" unless it is ALGORITHM-ONLY or SHARED-ASSIST.
```

## 13. Decision Matrix For Common Failure Modes

### 13.1 Offroad Dominant

Likely causes:

```text
lane_offset correction too weak
heading correction too weak
steering smoothing too slow
target speed too high for curves
route target/local waypoint misalignment
reward does not penalize drift early enough
BC data lacks recovery examples
```

Preferred next actions:

```text
1. Analyze pre-offroad 20-step window.
2. Audit lane_offset and heading_error sign/units.
3. Check applied steer vs requested steer.
4. If code is correct, test lateral_recovery_v1.
5. If recovery still fails, collect recovery imitation data.
```

Avoid:

```text
only increasing offroad terminal penalty
large PPO LR changes
raising target speed
changing many reward weights at once
```

### 13.2 Collision Dominant

Likely causes:

```text
target speed too high
brake response too weak
risk context not affecting policy/control
traffic/object handling missing
route target encourages unsafe forward progress
```

Preferred next actions:

```text
1. Analyze speed/brake/risk before collision.
2. Lower target speed or increase brake gain only if evidence supports it.
3. Add risk-aware speed cap if risk_scalar is meaningful.
4. Separate traffic collision from static/world collision.
```

### 13.3 Stuck Dominant

Likely causes:

```text
longitudinal output too negative
target-speed floor too low
brake hold
no progress reward too weak
route target not advancing
spawn/route mismatch
```

Preferred next actions:

```text
1. Check policy_longitudinal before stuck.
2. Check target_speed_kmh before stuck.
3. Check throttle/brake before stuck.
4. Keep min_target_speed_kmh floor if it already solved collapse.
5. Avoid increasing speed too much if offroad is also present.
```

### 13.4 PPO Collapse

Likely causes:

```text
learning rate too high
clip coefficient too high
entropy too high
std too high
value loss unstable
reward scale too large
hidden-state mismatch
```

Preferred next actions:

```text
1. Check policy_std/log_std.
2. Check actor/value losses and grad norm.
3. Lower LR/clip/entropy only one at a time if possible.
4. Confirm recurrent hidden states are stored and used.
5. Prefer sequence BC checkpoint for recurrent policy.
```

## 14. Lateral Recovery v1 Default Plan

Because the current known bottleneck is offroad/lateral recovery, this is the default next stabilization target unless new logs contradict it.

### 14.1 Hypothesis

```text
The policy can drive forward, but it does not correct lane_offset and heading_error early enough.
Small residual steering correction can reduce offroad without replacing the learned policy.
```

### 14.2 Preferred Mechanism

Use a residual correction:

```text
corrected_steer = policy_steer + correction
correction = lane_gain * lane_offset_error + heading_gain * heading_error
corrected_steer = clamp(corrected_steer, -1, 1)
```

Important:

```text
The exact sign must be verified from code/logs before implementation.
The correction should be small.
The correction should be logged separately.
The raw policy action should still be logged.
```

### 14.3 Activation Rule

Do not always apply strong correction.

Start with:

```text
activate when:
  abs(lane_offset) > threshold
  or abs(heading_error) > threshold

recommended first thresholds:
  lane_offset threshold: 0.4 to 0.7 m
  heading_error threshold: 5 to 8 deg
```

### 14.4 Safety Clamp

Recommended first clamp:

```text
max_abs_correction: 0.12 to 0.20
```

Do not exceed this until logs show under-correction.

### 14.5 Required Logs

Add or verify:

```text
lateral_recovery_enabled
lateral_recovery_active
lateral_recovery_lane_error
lateral_recovery_heading_error
lateral_recovery_correction
lateral_recovery_raw_steer
lateral_recovery_corrected_steer
```

### 14.6 Success Criteria

PASS:

```text
offroad count decreases versus control_v2b_4096
stuck remains zero or near zero
collision does not increase
route_progress_max >= control_v2b baseline
deterministic 512 eval passes
```

WARN:

```text
offroad decreases but collision appears
route progress is flat
correction is active too often
```

FAIL:

```text
steering oscillation increases
collision increases sharply
route progress collapses
correction sign is wrong
```

## 15. Documentation Requirements Per Round

Each round must record:

```text
1. Hypothesis
2. Baseline run paths
3. Baseline metrics
4. Code/config changes
5. Commands executed
6. New run paths
7. New metrics
8. Comparison table
9. Regression-first summary
10. Mechanism activation check
11. Hardcoding / privileged-assist audit
12. Baseline assist parity or required assist-off ablation
13. 2048 gate decision
14. PASS/WARN/FAIL judgment
15. Paper-relevance label
16. Next recommendation
```

Recommended document template:

```markdown
# <Round Name> Result

## Hypothesis

## Baseline

## Changes

## Commands

## Results

## Comparison

## Regressions

## Mechanism Activation

## Hardcoding / Assist Audit

## Paper-Relevance Label

## Gate Decision

## Judgment

## Next Step
```

## 16. Standard Run Naming

Use names that encode mechanism and horizon.

Examples:

```text
lateral_recovery_v1_eval_512
lateral_recovery_v1_2048
lateral_recovery_v1_4096
lateral_recovery_v1_final_eval_512
```

Avoid vague names:

```text
test1
new_run
fix_reward
try_again
```

## 17. How The Trigger Should Be Interpreted

When the user says:

```text
서브 에이전트 구조로 안정화 진행해줘
```

The assistant should interpret it as:

```text
1. Use this document as the workflow source.
2. Identify the latest relevant logs/checkpoints.
3. Spawn Log Analyst, Code Auditor, Mechanism Verifier, Safety Regression Analyst, and Hardcoding / Privileged Controller Auditor if available.
4. Add Imitation/Data Auditor when BC, expert data, recovery datasets, or target relabeling are involved.
5. Add Ablation Planner or Control/RL Specialist only when needed.
6. Integrate their findings.
7. Propose one next experiment.
8. Implement the selected code/config change unless blocked.
9. Run static validation and 2048 smoke if CARLA is available.
10. Apply the 2048 gate before any 4096 smoke.
11. Document the result with regression-first summary, mechanism activation evidence, hardcoding/assist audit, and paper-relevance label.
```

If sub-agents are unavailable in a future environment, follow the same role structure manually and state that sub-agent execution was unavailable.

## 18. Current Default Baselines

These are the current useful comparison points:

```text
current best clean eval candidate:
  checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt
  outputs/bc_eval/recovery_imitation_v2_lateral_target_4096_eval_512_20260429T231537/summary.json
  result: goal reached, max_route_progress=1.0, collision/offroad/stuck=0/0/0

current conservative fallback:
  checkpoints/recovery_imitation_v1_ppo_4096_20260429T164950/last.pt
  outputs/bc_eval/recovery_imitation_v1_ppo_eval_512_20260429T165905/summary.json
  result: max_route_progress=0.9231, collision/offroad/stuck=0/0/0

current unresolved training issue:
  recovery_imitation_v2_lateral_target improved eval and route max,
  but training offroad remained nonzero and slightly increased versus v1.

control_v1 training:
  outputs/phase3_bc_clean_control_env/phase3_bc_clean_control_4096_20260427T170917/env_step_log.csv
  logs/phase3_bc_clean_control_4096_20260427T170917/training_log.csv

control_v1 eval:
  outputs/bc_eval/phase3_bc_clean_control_eval_512_20260427T171817/summary.json

control_v2b training:
  outputs/phase3_bc_clean_control_v2_env/phase3_bc_clean_control_v2b_4096_20260427T174353/env_step_log.csv
  logs/phase3_bc_clean_control_v2b_4096_20260427T174353/training_log.csv

control_v2b checkpoint:
  checkpoints/phase3_bc_clean_control_v2b_4096_20260427T174353/last.pt

control_v2b eval:
  outputs/bc_eval/phase3_bc_clean_control_v2b_eval_512_20260427T175252/summary.json
```

If newer runs exist, use them instead but preserve these as historical baselines.

## 19. Non-Negotiable Rules

```text
1. Never overwrite or delete old run logs.
2. Never run two CARLA training jobs at the same time.
3. Never claim stabilization success without comparing against a baseline.
4. Never optimize only for reward.
5. Never ignore collision increases just because route progress improves.
6. Never use robustness/weather/corruption evaluation until clean driving is stable enough.
7. Never modify original RCS-AC source when working in RCS-AC-first unless explicitly requested.
8. Always preserve action contract: [steer, longitudinal].
9. Always document failed experiments.
10. Always keep the next experiment narrow.
11. Always use the 2048 gate before a 4096 formal smoke.
12. Always summarize regressions before improvements.
13. Always verify that the new mechanism actually activated in logs.
14. Never combine multiple primary hypotheses in one round.
15. For imitation rounds, always audit dataset distribution, target correctness, and BC checkpoint eval before PPO.
16. Do not promote a checkpoint on deterministic eval alone if training distribution regresses severely.
17. CARLA experiment execution and best-checkpoint promotion remain Main Agent responsibilities by default.
18. Never claim RCS-AC algorithm superiority from a checkpoint that depends on an assist not shared by PPO/SAC baselines.
19. Always run or explicitly schedule assist-off or same-assist ablation when a controller-assisted checkpoint is considered best.
20. Never hide rule-based route/lane/steering correction inside a stabilization result; label it as controller-assisted.
```

## 20. Practical Next Command

The next recommended user command is:

```text
다음 라운드 서브 에이전트로 안정화 진행해줘.
목표는 recovery_imitation_v2_lateral_target의 goal-reaching eval을 유지하면서 training offroad를 줄이는 것이다.
```

Older encoded command, preserved for history:

```text
서브 에이전트 구조로 안정화 진행해줘. lateral_recovery_v1부터 시작해줘.
```

Expected behavior:

```text
1. Analyze recovery_imitation_v2_lateral_target 4096 and eval logs.
2. Audit whether the latest mechanism activated and whether logging is sufficient.
3. Audit whether the latest/best result depends on hardcoded or privileged controller assistance.
4. Audit data/BC only if the next round uses imitation or recovery datasets.
5. Propose one narrow experiment to reduce training offroad without losing goal-reaching eval.
6. Implement only the smallest code/config change needed.
7. Run compileall.
8. Run 2048 smoke first and apply the gate.
9. Run 4096 only if the 2048 gate passes.
10. Run deterministic eval and compare against current best/fallback.
11. Write result documentation with hardcoding/assist audit and paper-relevance label.
```
