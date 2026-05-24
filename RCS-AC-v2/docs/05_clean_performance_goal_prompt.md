# RCS-AC v2 Clean Performance Goal Prompt

작성일: 2026-05-24

목적:

```text
Codex goal mode에서 RCS-AC-v2를 Raw2Drive PDF 기준으로 끝까지 구현하고,
fusion ablation 이전에 clean 성능을 먼저 잡도록 지시하는 복붙용 프롬프트를 제공한다.
```

사용법:

```text
새 Codex thread 또는 goal continuation에서 아래 프롬프트 전체를 붙여넣는다.
이미 active goal이 있으면 "현재 goal을 이어서 이 프롬프트 기준으로 진행해줘"라고 붙인다.
active goal이 없으면 "goal 기능을 사용해서 아래 objective를 생성하고 진행해줘"라고 붙인다.
```

## Copy-Paste Prompt

```text
goal 기능을 사용해서 RCS-AC-v2 clean 성능을 잡는 방향으로 끝까지 진행해줘.

Objective:
RCS-AC-v2를 /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2 안에서 처음부터 Raw2Drive PDF 그대로 구현한다. 목표는 fusion ablation이 아니라 clean 성능 확보가 먼저다. PDF 기반 dual-stream MBRL 구조를 구현하고, privileged world model/policy clean gate와 raw sensor policy clean gate를 순서대로 통과시킨 뒤에만 fusion 방식 ablation으로 넘어간다.

Workspace:
/Users/song-yonghwi/code/RCS-AC-first

Primary reference:
/Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/docs/reference/2505.16394v2.pdf

Required reading before edits:
1. /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/GOAL_STATUS.md
2. /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/docs/01_goal_driven_implementation_runbook.md
3. /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/docs/02_persistent_subagent_goal_protocol.md
4. /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/docs/03_windows_carla_remote_execution_workflow.md
5. /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/docs/04_git_sync_artifact_cleanup_policy.md
6. /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/docs/00_raw2drive_exact_implementation_plan.md

Core rule:
Do not optimize fusion first. Do not start fusion ablation until clean-capable Raw2Drive baselines exist. Clean performance must be diagnosed through contract, privileged stream, world model, rollout guidance, head guidance, and raw policy in that order.

Do not carry over old RCS-AC-v2 or v1.1 final runtime assumptions:
- Do not use v1.1 waypoint + pure-pursuit as the final v2 action contract.
- Use Raw2Drive 39 discrete actions.
- Do not expose privileged BEV/GT tensors to raw policy inference.
- Do not train raw reward/continue heads directly.
- Do not change action/reward/eval contract silently between experiments.

Execution mode:
Work autonomously stage by stage. Do not stop at a plan if implementation is possible. At the start of each turn, inspect GOAL_STATUS.md and the current stage result.md. At the end of each turn, update result.md and GOAL_STATUS.md with stage status, test status, gate status, and next action.

Sub-agent requirement:
Use persistent sub-agents throughout the goal, not only at the beginning.
If multi-agent tools are not loaded, use tool_search for "multi-agent spawn manage sub-agents".
At stage entry or work-packet entry, spawn at least:
- Agent A: Goal/Status Evidence Analyst
- Agent C: Raw2Drive Contract/Gate Verifier
Add Agent B whenever editing code/config/test/logging/checkpoint wiring.
Escalate to Full mode if action/reward/schema/leakage/model/dataset/baseline/gate uncertainty appears.
Re-trigger sub-agents at stage entry, major edit batch, repeated failure, first Windows CARLA run for a stage, and gate review.

Mandatory order for each work packet:
1. Read GOAL_STATUS.md and current stage docs.
2. Decide Lite or Full subagent mode.
3. Spawn required sub-agents.
4. Integrate findings.
5. Write an Implementation Brief before edits.
6. Implement the smallest useful slice.
7. Run the relevant tests or smoke checks.
8. Update experiments/<stage>/result.md.
9. Update GOAL_STATUS.md.
10. Decide next stage or next failure fix.

Stage progression:

Stage 0 - Contract preflight:
Implement and test the fixed contracts before any model work:
- code/rcs_ac_v2/env/discrete_actions.py
- code/rcs_ac_v2/data/schema.py
- code/rcs_ac_v2/data/replay_buffer.py
- code/rcs_ac_v2/env/reward.py
- leakage audit tests
Gate:
python -m pytest RCS-AC-v2/tests -q
must pass for Stage 0 contract tests.

Stage 1 - Privileged world model:
Implement privileged BEV + ego RSSM world model, reconstruction/reward/continue heads, sequence sampler, training script, checkpointing, and smoke configs. Use tiny synthetic and tiny CARLA replay smoke first.
Gate:
synthetic overfit, tiny replay overfit, finite losses, finite rollout, checkpoint reload, result note complete.

Stage 2 - Privileged policy clean:
Train actor-critic inside the privileged world model and evaluate clean closed-loop on Windows CARLA.
After privileged observation probe passes, collect simulator-state semantic43/ego8
Windows CARLA replay, train checkpoints with
source=windows_carla_privileged_replay, and reject synthetic smoke checkpoints
for official clean eval.
Gate sequence:
128-step clean smoke: collision/offroad/stuck = 0
512-step primary subset: no systematic collision/offroad/stuck
2048-step subset: route progress no regression, goal success when route length allows
If clean fails, classify whether the failure is infrastructure, CARLA, adapter, contract, model, or driving before changing algorithms.

Stage 3 - Raw sensor world model with rollout guidance:
Implement multi-view image + IMU raw stream, BEVFormer-style or staged lightweight encoder, raw RSSM, privileged alignment, and rollout guidance from the privileged world model.
Gate:
raw observation shapes correct, alignment loss finite, rollout guidance finite, no privileged leakage at raw inference boundary.

Stage 4 - Raw sensor policy with head guidance:
Implement raw policy training with head guidance from the privileged policy. Final raw policy inference must use only raw sensor inputs allowed by the PDF plan.
Gate:
128 clean pass, 512 primary9 clean pass, 2048 subset clean pass or documented partial pass with failure class and next fix.

Stage 5 - Fusion-only ablation:
Start only after clean-capable baselines exist. Keep action/reward/eval/training-budget contracts fixed and change only the fusion mechanism under test.
Gate:
fusion comparison reports clean and stress separately, with identical evaluation protocol and explicit commit/config/checkpoint metadata.

Stage 6 - Closed-loop clean/stress report:
Write final report with clean metrics, stress metrics, route progress, success rate, collision/offroad/stuck, and leakage audit result.

Windows CARLA execution:
Use Mac-Codex / Windows-CARLA split:
- Mac: Codex, sub-agents, edits, tests, analysis, docs, git commit/push
- Windows: CARLA, Python CARLA execution, heavy artifacts
Official Windows runs must come from committed/pushed code:
Mac commit/push -> Windows git pull --ff-only -> CARLA readiness guard -> one bounded run -> pull text/table logs -> result.md update -> commit/push.
Run the CARLA readiness guard before every Windows CARLA command.
Scratch Windows runs are allowed only for infrastructure probing and must not be used as clean performance evidence.

Git/artifact hygiene:
Do not delete the root .git directory on Mac or Windows.
Use git maintenance/gc only for repository maintenance.
Do not commit checkpoints, replay buffers, raw frames, videos, or large per-step logs.
Before committing Windows results, run:
/Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/scripts/check_git_hygiene.sh

Every official experiment result must record:
commit, branch, Windows command, config, CARLA version, CARLA town, stage, gate, output path, pulled files, failure class, and next action.

Completion condition:
Do not mark the full goal complete until:
1. Stage 0 contract tests pass.
2. Privileged world model/policy reaches clean gate or a documented clean bottleneck with evidence-backed fixes exhausted.
3. Raw sensor world model/policy reaches clean gate or a documented clean bottleneck with evidence-backed fixes exhausted.
4. Privileged leakage audit passes.
5. Fusion ablation, if run, is only on clean-capable variants.
6. Final clean/stress report is written.

If blocked:
Only mark blocked after the same blocking condition repeats across the required blocked-audit turns and no meaningful progress is possible without user input or an external state change. Otherwise keep making measurable progress through implementation, tests, instrumentation, docs, or smaller smoke gates.
```

## Short Resume Prompt

Use this when the goal already exists and the next Codex session needs a compact
instruction:

```text
RCS-AC-v2 clean 성능 goal을 이어서 진행해줘. 먼저 /Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/GOAL_STATUS.md와 docs/05_clean_performance_goal_prompt.md를 읽고, 현재 stage에서 sub-agent protocol을 적용한 뒤 Implementation Brief -> edit -> validate -> result.md/GOAL_STATUS.md update 순서로 계속 진행해줘. Fusion ablation은 clean-capable baseline 전까지 금지하고, Windows CARLA 공식 run은 commit/push된 코드에서만 실행해줘. Root .git은 삭제하지 말고 check_git_hygiene.sh로 artifact만 확인해줘.
```
