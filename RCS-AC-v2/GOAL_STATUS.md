# RCS-AC v2 Goal Status

Updated: 2026-05-24

## Active Objective

Implement RCS-AC-v2 from scratch following the Raw2Drive PDF:

```text
docs/reference/2505.16394v2.pdf
```

## Current Stage

```text
Stage 2: Privileged policy clean
```

Status:

```text
V2_BRANCH_PUSHED__LOCAL_DEPENDENCY_AND_OBS_PROBE_READY__LOCAL_CHECKPOINT_EXPORT_PASS__WINDOWS_PRIVILEGED_OBS_VALIDATION_PENDING__CARLA_CLEAN_PENDING
```

Last completed stage:

```text
Stage 0: Contract preflight PASS
Stage 1: Privileged world model PASS
```

## Required Reading Before Coding

1. `docs/01_goal_driven_implementation_runbook.md`
2. `docs/02_persistent_subagent_goal_protocol.md`
3. `docs/03_windows_carla_remote_execution_workflow.md`
4. `docs/04_git_sync_artifact_cleanup_policy.md`
5. `docs/05_clean_performance_goal_prompt.md`
6. `docs/00_raw2drive_exact_implementation_plan.md`
7. `docs/reference/2505.16394v2.pdf`

## Sub-Agent Protocol State

Mode:

```text
Lite by default. Escalate to Full if contract/model/dataset/leakage/baseline/gate uncertainty appears.
```

Required agents:

```text
Stage entry and work-packet start: Agent A + Agent C.
Add Agent B when editing code/config/test/logging/checkpoint wiring.
```

Last findings summary:

```text
Explorer Darwin recommended persistent Lite workflow, Agent A/C/B role split,
Full escalation triggers, pre-edit barrier, Implementation Brief, and result template fields.
Integrated into docs/02_persistent_subagent_goal_protocol.md and runbook.
```

Pending brief:

```text
Stage 2 local smoke, Windows dry-run prep, fake-env adapter-loop contract, Stage 2 shared-core CARLA adapter factory, local checkpoint export/manifest path, v2 branch push, v2-local readiness scripts, dependency probe, and privileged obs probe are complete. Next work packet must run the Windows dependency/obs probes, promote real trained checkpoints, and verify privileged BEV/ego keys on Windows CARLA.
```

Escalation triggers checked:

```text
Stage 0 contract implementation passed local tests.
Stage 1 privileged world-model smoke passed local tests.
Stage 2 local actor-critic/world-model rollout smoke passed local tests.
Stage 2 Windows clean-eval config/CLI/runner dry-run contract passed local tests.
Stage 2 clean-eval fake adapter loop, metrics CSV validator, and non-dry-run classified failure path passed local tests.
Stage 2 shared-core CARLA adapter factory, checkpoint loader, policy wrapper, stage4_v2/direct_3d control bridges, reverse-action guard, and model-failure classification passed local tests.
Stage 2 checkpoint export produces paired world_model/policy checkpoints plus sha256 manifest and config patch; synthetic export remains non-CARLA/non-clean evidence.
Stage 2 checkpoint loader tests now cover missing sha/path, bad sha, missing config/state_dict, unreadable checkpoint path, invalid policy config, policy latent mismatch, bad policy state_dict load, and 8x8/16x16 world-model mismatch.
RCS-AC-v2/.gitignore now makes v2 artifact policy self-contained; check_git_hygiene.sh now fails nonzero if v2 nested .git, heavy artifacts, or >50MB files are found.
Stage 2 runner now has v2-local CARLA readiness scripts and a Windows dependency probe; remaining temporary dependency is RCS-AC-v1.1 shared-core env builder plus sibling shared_core.
Stage 2 clean config now uses the temporary shared-core stage4_v2 action bridge for v1.1 env compatibility; Raw2Drive 39-action policy output remains the v2 policy contract.
Stage 2 privileged-observation probe writes privileged_obs_probe.json with clean_gate_evidence=false before any clean PASS attempt.
Stage 2 local tests now include dependency-probe static contract, v2-local readiness scripts, stage4_v2 action bridge, and privileged_obs_probe.json non-clean evidence checks.
First Windows CARLA run must follow docs/03_windows_carla_remote_execution_workflow.md.
GitHub/Windows sync and .git cleanup rules must follow docs/04_git_sync_artifact_cleanup_policy.md.
Clean-performance goal start/resume prompt is docs/05_clean_performance_goal_prompt.md.
```

Next sub-agent trigger:

```text
Before Stage 2 checkpoint promotion, Windows privileged-observation validation, or official Windows execution.
```

## Subagent Ledger

Last pulse:

```text
2026-05-24 Stage 2 CARLA factory prep: Agents A/B/C audited current status, false-PASS risks, checkpoint enforcement, policy interface mismatch, v1.1 shared_core env builder, action/control format risks, and Windows import-path requirements.
```

Active agents:

```text
None active. At Stage 2 checkpoint promotion, Windows privileged-observation validation, or official Windows execution, spawn Agent A + Agent C; add Agent B for adapter/eval/logging/checkpoint wiring.
```

Latest integrated findings:

```text
Persistent cadence is now documented: stage entry, major edit, repeated failure, gate review, and every 30 minutes/large edit batch.
Lite default now requires Agent A + Agent C at work-packet/stage entry, with Agent B added for code/config/test/logging/checkpoint wiring.
Pre-edit barrier and Implementation Brief are documented.
Stage 0 action table order is locally canonicalized as raw2drive_table12_row_major_v1 because PDF Table 12 has no explicit action IDs.
Stage 0 config now pins action hash, reward contract version, continue truncation policy, and leakage allowlist/denylist.
Stage 1 uses a compact Gaussian RSSM smoke; this is documented as an engineering smoke approximation before later exactness passes.
Stage 1 binary synthetic BEV targets, stop-gradient KL tests, previous-action convention tests, ego influence tests, finite rollout, synthetic overfit, and checkpoint roundtrip all pass.
Stage 2 local smoke uses policy-driven RSSM prior imagination with reward/continue heads; scalar critic and batch-normalized advantage are documented smoke approximations.
Stage 2 local tests pass, but CARLA clean gate is still pending and must not be treated as full clean PASS.
Stage 2 dry-run summary validation prevents LOCAL_SMOKE_ONLY or scratch evidence from becoming PASS.
Official PASS now requires source windows_carla, steps reaching horizon, done_reason horizon_reached, zero collision/offroad/stuck, failure_class none, finite metrics, pulled_files evidence, and commit/branch/command/config/CARLA metadata.
Stage 2 adapter loop now writes metrics.csv and summary.json in a finally-style path, validates fake PASS/FAIL cases locally, and requires metrics.csv evidence for any official PASS.
Windows runner real mode now invokes evaluate_clean --run-carla after CARLA readiness.
Stage 2 config now points at rcs_ac_v2.evaluation.carla_adapter:create_stage2_carla_adapter.
The adapter factory lazily builds the v1.1-style shared_core CARLA env, bridges Raw2Drive controls through temporary stage4_v2 for v1.1 compatibility, keeps a direct_3d reverse-loss guard, extracts strict privileged_bev [1,1,43,16,16] and privileged_ego [1,1,8], and wraps world-model+policy checkpoints into policy.act(obs, info) logits [1,1,39].
Real mode with missing checkpoints is now classified as failure_class model / done_reason world_model_checkpoint_missing before CARLA env construction.
Checkpoint loader now requires path and sha256, validates payload config/state_dict, rejects 8x8 world-model checkpoints for the 16x16 clean config, rejects missing policy config, and keeps checkpoint failures in failure_class model.
Windows runner PYTHONPATH now includes the v1.1 support_code, v1.1 code path, and workspace root needed by the shared-core env builder.
Windows scripts README now reflects the implemented Stage 2 runner, v2-local readiness scripts, dependency probe, and privileged obs probe. It warns that v1.1/shared_core dependencies are temporary and must be available or ported before official clean claims.
RCS-AC-v2 subtree was committed and pushed to origin/codex/rcs-ac-v2-raw2drive. Unrelated root worktree changes were not staged.
```

Next subagent trigger:

```text
Before Stage 2 checkpoint promotion, Windows privileged-observation validation, or Windows execution, spawn at least Agent A + Agent C; add Agent B for code/test/logging/checkpoint wiring.
```

Unavailable/fallback:

```text
If multi-agent tools are not loaded, use tool_search for "multi-agent spawn manage sub-agents".
If still unavailable, manually simulate Agent A/B/C roles and record fallback here.
```

## Next Concrete Action

Validate Windows privileged observations and promote real trained Stage 2 checkpoints before official Windows execution:

```text
current local branch is codex/rcs-ac-v2-raw2drive
pull origin/codex/rcs-ac-v2-raw2drive on Windows with git pull --ff-only before any official Windows clean gate
replace synthetic smoke checkpoints with real Stage 1 world-model and Stage 2 privileged-policy checkpoints
fill Windows-local clean config checkpoints.world_model/path+sha256 and checkpoints.policy/path+sha256
run scripts/windows/check_stage2_windows_dependencies.ps1 or run_stage2_privileged_obs_probe.ps1 to verify Windows dependency chain: RCS-AC-v1.1 shared-core env builder and sibling shared_core are available, or port the minimum env builder into RCS-AC-v2
run scripts/windows/run_stage2_privileged_obs_probe.ps1 -StepNeutral and record privileged_obs_probe.json before clean eval
run scripts/windows/run_stage2_clean_eval.ps1 official execution after git pull --ff-only
verify Windows shared_core env exposes/adapter can build privileged_bev [1,1,43,16,16] and privileged_ego [1,1,8]
experiments/02_privileged_policy_clean/windows_128_result.md official run update with summary+metrics pulled files
commit/push RCS-AC-v2 work before Windows git pull --ff-only
```

## Stage 0 Gate

Status:

```text
PASS
```

Command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
25 passed in 1.12s
```

## Stage 1 Gate

Status:

```text
PASS
```

Command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
37 passed in 2.00s
```

Synthetic overfit:

```text
initial_loss 4.801789283752441
final_loss 2.732663631439209
checkpoint_path /tmp/rcs_ac_v2_stage1_smoke/stage1_privileged_world_model_smoke.pt
```

## Stage 2 Gate

Local status:

```text
LOCAL_SMOKE_PASS
DRY_RUN_CONTRACT_PASS
LOCAL_ADAPTER_LOOP_PASS
LOCAL_CARLA_FACTORY_PASS
LOCAL_CHECKPOINT_EXPORT_PASS
```

Local command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Local result:

```text
89 passed in 2.25s
```

Local policy smoke:

```text
initial_critic_loss 0.001165103050880134
final_critic_loss 0.00015270980657078326
valid_action_ids True
rollout_continue_mean 0.5067161917686462
```

Dry-run contract:

```text
DRY_RUN_SUMMARY=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
SUMMARY_VALID=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
generated dry-run summary removed after validation
```

Full clean status:

```text
REAL_TRAINED_CHECKPOINTS_PENDING
WINDOWS_PRIVILEGED_OBS_VALIDATION_PENDING
CARLA_CLEAN_PENDING
```

Do not start Stage 3 raw sensor work or any fusion ablation until:

```text
CARLA 128-step single-start clean gate is run and classified
CARLA 512-step primary subset clean gate is run or explicitly blocked
CARLA 2048-step subset clean gate is run or explicitly blocked
collision/offroad/stuck/route-progress evidence is recorded
failure class is infrastructure / CARLA / adapter / contract / model / driving
```

## Important Constraints

```text
Use Raw2Drive 39 discrete actions.
Do not use v1.1 waypoint + pure-pursuit as final v2 action contract.
Do not expose privileged BEV/GT data to raw policy inference.
Do not train raw reward/continue heads.
Do not run fusion ablation before clean gates.
Use persistent subagent pulse at stage entry, major edits, repeated failures, and gate reviews.
Use Mac-Codex / Windows-CARLA split for all real CARLA runs.
Run CARLA readiness guard before every Windows CARLA command.
Do not delete root .git during goal execution; use git maintenance/gc only.
Delete accidental nested .git only after the documented checks pass.
```
