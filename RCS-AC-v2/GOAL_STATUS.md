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
FAST35_128_CLEAN_PASS__512_PRIMARY_FAIL_COLLISION
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
Stage 2 local smoke, Windows dry-run prep, fake-env adapter-loop contract, Stage 2 shared-core CARLA adapter factory, local checkpoint export/manifest path, v2 branch push, v2-local readiness scripts, dependency probe, simulator-state privileged observation builder, Windows privileged obs probe, privileged replay collector, replay trainer, checkpoint provenance gate, official Windows clean-clone privileged-observation probe, replay-source checkpoint promotion, first 128-step clean classification, BC checkpoint action probe, fast35 replay relabeling, fast35 replay retraining, checkpoint action probe, official 128-step clean rerun, 512 runtime config prep, runner output-path hardening, and official 512-step primary clean run are complete. The first clean run failed as driving/stuck after 64/128 steps, and the first BC checkpoint probe failed because it collapsed to action_id=34 for all 512 replay observations. The fast35 replay fixed the zero-throttle label collapse; the b4c8e58/37a0875 checkpoint probe covered 512 replay actions with action_ids 7/15, reverse_action_count=0, nonzero_throttle_fraction=1.0, and recommended_for_clean_rerun=true. The official clean clone then reached 128/128 steps with source=windows_carla, status=PASS, done_reason=horizon_reached, zero collision/offroad/stuck, and metrics.csv rows=128. The same checkpoint failed the official 512 primary gate as a driving/collision failure at step 310/512 with route_progress=0.5641025641025641, action_counts={7:308,15:3}, no adapter/runtime errors, offroad=0, and stuck=0. Next work packet should patch the smallest collision cause, most likely by collecting/retraining on a longer or failure-window privileged replay that includes steering/turn labels before rerunning 512.
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
Stage 2 runner now has v2-local CARLA readiness scripts, a Windows dependency probe, and a v2-local minimal shared_core env builder; remaining temporary external dependency is sibling shared_core.
Stage 2 clean config now uses the temporary shared-core stage4_v2 action bridge for v1.1 env compatibility; Raw2Drive 39-action policy output remains the v2 policy contract.
Stage 2 privileged-observation probe writes privileged_obs_probe.json with clean_gate_evidence=false before any clean PASS attempt.
Stage 2 local tests now include dependency-probe static contract, v2-local readiness scripts, stage4_v2 action bridge, and privileged_obs_probe.json non-clean evidence checks.
Stage 2 Windows raw-key diagnostic probe at commit 6dacd8d showed the current shared-core env exposes rgb/lidar/route/aux masks but not privileged_bev or privileged_ego.
Stage 2 now builds the pinned 43-channel semantic BEV and 8-field ego vector from CARLA simulator state only; raw-derived/debug privileged tensors are not clean-gate eligible.
Stage 2 Windows archive probe at commit 5b00347 passed dependency check, CARLA Town04 readiness, simulator-state privileged provenance, BEV/ego shape/finite checks, and neutral-step safety checks. The probe remains clean_gate_evidence=false.
Stage 2 replay path now exists: collect_stage2_privileged_replay.ps1 writes simulator-state semantic43/ego8 .npz episodes, train_stage2_privileged_from_replay.ps1 exports replay-source world_model/policy checkpoints and manifest, and clean eval rejects synthetic smoke checkpoints by provenance.
Stage 2 offline diagnosis now includes replay_latent_probe.py and probe_stage2_replay_latents.ps1. The steer2 replay candidates all scored linear_probe_test_accuracy=0.7384615540504456 against majority_action_fraction=0.6831530139103554, so the current world-model latent only weakly separates action labels and the next patch should address representation/objective collapse before another CARLA run.
Stage 2 replay path Windows scratch check at commit e1c9c86 passed dependency probe, replay module imports, 1x8-step privileged replay collection, 2-step replay checkpoint training, and strict replay-source checkpoint loading. This remains clean_gate_evidence=false and is not clean performance evidence.
Windows official sync lane now uses SSH GitHub access, not HTTPS. Clean clone path C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90 was created at commit ac11c90 after enabling core.longpaths. The v1.1 env config is now copied under RCS-AC-v2 and the adapter owns the minimal shared_core env builder, so the clean clone should only need sibling shared_core as an external runtime dependency.
Official Windows clean clone was pulled to commit e42dc6a after fixing optional PowerShell dependency arguments. From that clean clone, dependency check, CARLA Town04 readiness, simulator-state privileged observation provenance, BEV/ego shape/finite checks, and neutral-step safety checks all passed with clean_gate_evidence=false.
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
2026-05-24 Stage 2 privileged obs Windows validation: Agents A/B/C audited false-PASS risk, raw-derived privileged leakage risk, simulator-state provenance requirements, clean-gate eligibility, and v2-only adapter scope.
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
The adapter now falls back to a v2-owned simulator-state builder when the shared-core env lacks privileged keys. It uses CARLA world/map/vehicle/route-manager state, pins schema carla_simulator_semantic43_v1, pins channel hash 382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3, and records clean-gate-eligible provenance.
The clean loop validates privileged provenance and rejects missing, schema-mismatched, hash-mismatched, or derived_debug_from_raw_v1 observations before any clean PASS can be claimed.
Windows archive probe result for commit 5b00347 is PROBE_PASS with source carla_simulator_state, BEV [1,1,43,16,16], ego [1,1,8], finite tensors, nonzero drivable_road/route_path/ego_vehicle/static_obstacle/junction/route_straight channels, and neutral step collision/offroad/stuck false.
The existing Windows checkout was dirty and GitHub credential fetch failed, so this round used a scratch archive runtime instead of switching the Windows repo. That is valid infrastructure/probe evidence, but the official clean run still needs a clean git-pull lane or clean clone.
Stage 2 training is no longer limited to synthetic smoke plumbing: v2 has replay IO/sampler, Windows privileged replay collection, replay-based world-model/policy training, and checkpoint manifests with source=windows_carla_privileged_replay. Synthetic smoke checkpoints now carry source=local_synthetic_smoke_not_carla and are rejected by the official clean config.
Windows archive scratch check for commit e1c9c86 confirmed replay modules import under Python 3.12, tiny CARLA privileged replay collection writes source=windows_carla_privileged_replay manifest, tiny replay training writes world_model/policy sha256 manifest, and load_stage2_policy_runner accepts those replay-source checkpoints under the strict provenance gate.
HTTPS GitHub access on Windows still fails due credential prompt, but SSH git@github.com works for origin/codex/rcs-ac-v2-raw2drive. A clean official clone is available after core.longpaths=true. Stage 2 scripts no longer require untracked RCS-AC-v1.1 support/code; they still support EnvConfigPathOverride/SupportCodePath/V11CodePath/SharedCorePath for local overrides.
Official clean clone probe at commit e42dc6a passed without override paths: source=carla_simulator_state, schema=carla_simulator_semantic43_v1, hash=382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3, BEV [1,1,43,16,16], ego [1,1,8], finite tensors, neutral step running/collision=false/offroad=false/stuck=false. This remains infrastructure evidence, not clean performance evidence.
Real mode with missing checkpoints is now classified as failure_class model / done_reason world_model_checkpoint_missing before CARLA env construction.
Checkpoint loader now requires path and sha256, validates payload config/state_dict, rejects 8x8 world-model checkpoints for the 16x16 clean config, rejects missing policy config, and keeps checkpoint failures in failure_class model.
Replay provenance was hardened at commit 48c0543: clean/replay now require privileged_observation_source=carla_simulator_state, existing env privileged keys are clean-ineligible, replay trainer requires collector manifest/file sha validation, and checkpoint payloads must carry replay lineage, role, trainer, non-clean evidence, data_files, replay manifest, and valid_action_ids.
Reverse-action masking was added at commit db1e8b6 for direct_3d/stage4_v2 when allow_reverse_direct_3d=false. The first replay checkpoint then loaded successfully and had REVERSE_ACTION_COUNT=0 on 64 replay observations.
First official 128-step clean eval with promoted replay checkpoint ran from the clean clone at commit db1e8b6 and produced source=windows_carla, status=FAIL, failure_class=driving, done_reason=stuck, steps=64/128, collision_count=0, offroad_count=0, stuck_count=1, route_progress=0.02564102564102564, metrics_rows=64, adapter_errors=0, runtime_errors=0.
BC warm-start for replay policy training was added at commit b20d8eb after the first clean failure showed action collapse to action_id=8 for all 64 clean steps. Windows BC retraining completed into windows_replay_b20d8eb_bc_pilot, but action probe and clean rerun are pending because the Windows Tailscale peer became temporarily unreachable.
Stage 2 now has a reproducible checkpoint action probe wrapper: scripts/windows/probe_stage2_checkpoint_actions.ps1. It creates the checkpoint-filled runtime config from a checkpoint manifest and writes stage2_checkpoint_action_probe.json with action counts, reverse_action_count, nonzero_throttle_action_count, moving_steer_action_count, single_action_collapse, and recommended_for_clean_rerun.
The checkpoint action probe now rejects both under-steering and over-steering candidates before clean eval by requiring moving/nonzero steering between the configured lower bound and 0.5 fraction.
The checkpoint action probe also records dominant_action_fraction and rejects dominant-action collapse above 0.85 before CARLA reruns. Replay checkpoint manifests now record sampler priority-window stats, and trainer class balancing can be softened with imitation_balance_power for offline steering sweeps.
Windows runner PYTHONPATH now includes v2 code, optional legacy support/code overrides, and workspace root needed by the shared-core env builder.
Windows clean eval runner now accepts -RunOutputDir to copy validated summary.json, metrics.csv, and the exact input config into a run-id evidence directory after summary validation. This is transport-only mirroring and does not change clean gate status or PASS criteria.
BC checkpoint action probe on windows_replay_b20d8eb_bc_pilot failed the pre-clean gate: probed_actions=512, action_counts={34:512}, reverse_action_count=0, nonzero_throttle_fraction=0.0, single_action_collapse=true, recommended_for_clean_rerun=false. No clean rerun was attempted for that checkpoint.
Replay label diagnosis found the old official replay had 388/512 zero-throttle action_id=34 labels, and TargetSpeedKmh overrides were ignored because env info.target_speed_kmh took precedence. Commit 30b7313 fixed explicit TargetSpeedKmh precedence; commit 9c24ff7 removed waypoint target_distance<2 as a zero-throttle label source. The 9c24ff7 fast35 replay collected on Windows has action_counts={7:264,15:248}, nonzero_throttle_fraction=1.0, reverse_fraction=0.0, and clean_gate_evidence=false.
Fast35 detached replay training now works through the scheduled-task wrapper. The b4c8e58/37a0875 checkpoint manifest has source=windows_carla_privileged_replay, valid_action_ids=true, policy_final_imitation=0.7903282642364502, and policy_final_total=1.0499897003173828.
Fast35 checkpoint action probe passed the pre-clean gate with probed_actions=512, action_counts={7:500,15:12}, reverse_action_count=0, single_action_collapse=false, nonzero_throttle_fraction=1.0, previous_stuck_action8_count=0, and recommended_for_clean_rerun=true.
Official 128-step clean rerun from the clean clone at commit 37a087516f7767bb297b8b5567590d7a316eb644 passed: source=windows_carla, status=PASS, clean_gate_pass=true, done_reason=horizon_reached, steps=128/128, collision/offroad/stuck counts all zero, route_progress=0.1794871794871795, episode_return=13.545934572121052, metrics_rows=128, action_counts={7:125,15:3}. This is the first Stage 2 clean PASS, not full clean-performance completion.
Official 512-step primary clean run from the clean clone at commit 0b10ff5628b2b9616759bd4c6f9f60cc441c9b0d failed with source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=311/512, collision_count=1, offroad_count=0, stuck_count=0, route_progress=0.5641025641025641, episode_return=19.066852226147766, metrics_rows=311, action_counts={7:308,15:3}, adapter_errors=0, runtime_errors=0. This validates the 512 runner path but does not satisfy the 512 clean gate.
Offline steer2 replay sweep at commit d4c5b70 rejected four checkpoint candidates before CARLA: balance_power=0.5 -> {7:647}, balance_power=0.625 -> {7:647}, balance_power=0.75 -> {4:639,7:8}, no class balance -> {7:647}. The replay sampler reported priority_window_count=212/617, so steering windows exist; the current replay-imitation policy objective is collapsing to one dominant action instead of learning state-dependent straight/steer decisions.
Offline latent separability probe at commit 7902e44 compared the same four rejected steer2 candidates. Each candidate used 647 replay samples, latent_dim=160, majority_action_fraction=0.6831530139103554, linear_probe_train_accuracy=0.7698259353637695, and linear_probe_test_accuracy=0.7384615540504456. This is above baseline but too weak to justify another clean run; patch latent/action separability or the sequence-aware policy objective before rerunning CARLA.
Windows scripts README now reflects the implemented Stage 2 runner, v2-local readiness scripts, dependency probe, privileged obs probe, replay collector, replay trainer, detached replay training start/check wrappers, and v2-local minimal shared_core env builder bridge.
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

Patch the 512-step collision failure before raw/fusion work:

```text
current local branch is codex/rcs-ac-v2-raw2drive
use official clean clone C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90
pull latest origin/codex/rcs-ac-v2-raw2drive before every Windows run
use the 512 failure evidence in RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\b4c8e58_fast35_primary512
diagnose why the fast35 policy stayed on straight actions 7/15 through the collision window
do not rerun CARLA only because a longer/failure-window replay exists; the steer2 replay already has 212/617 steering-priority windows and still produced collapsed candidates
patch representation/objective collapse first: improve latent/action separability or train a sequence-aware balanced policy head that keeps straight/steer decisions state-dependent
rerun replay_latent_probe.py and checkpoint_action_probe.py before any new 512 clean gate; require action probe to pass and latent probe to show materially stronger separation than majority baseline
do not treat the 128 PASS as full clean completion
use SSH remote on Windows; HTTPS prompts for credentials
use the v2-local minimal shared_core env builder by default; pass explicit dependency overrides only for local experiments
keep the archive-runtime lane only for scratch infrastructure probes, not official clean PASS evidence
run scripts/windows/check_stage2_windows_dependencies.ps1 to verify the v2-local env config and sibling shared_core remain available
rerun scripts/windows/run_stage2_privileged_obs_probe.ps1 -StepNeutral if checkpoints/config/env wiring changes; clean clone commit e42dc6a already passed the current probe
run scripts/windows/run_stage2_clean_eval.ps1 official execution after git pull --ff-only; use -RunOutputDir for summary/metrics/config mirroring, not PASS classification
verify clean summary source=windows_carla and metrics.csv rows equal the configured horizon before any PASS claim
experiments/02_privileged_policy_clean/windows_128_result.md and windows_512_result.md official run updates with summary+metrics pulled files
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
WINDOWS_DEPENDENCY_PROBE_PASS
OFFICIAL_CLEAN_CLONE_PRIVILEGED_OBS_PROBE_PASS
REPLAY_TRAINING_PATH_READY
SYNTHETIC_CHECKPOINT_REJECTION_PASS
```

Local command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Local result:

```text
106 passed in 3.03s
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
WINDOWS_PRIVILEGED_REPLAY_COLLECTION_PASS
REPLAY_CHECKPOINT_PROMOTED
BC_COLLAPSE_REJECTED_BY_ACTION_PROBE
FAST35_REPLAY_CHECKPOINT_PROMOTED
FAST35_ACTION_PROBE_PASS
FAST35_128_CLEAN_PASS
CARLA_512_PRIMARY_FAIL_COLLISION
```

Do not start Stage 3 raw sensor work or any fusion ablation until:

```text
CARLA 128-step single-start clean gate is PASS
CARLA 512-step primary subset clean gate is PASS or explicitly blocked after classified retry attempts
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
