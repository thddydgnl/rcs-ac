# Stage 2 Result

Date: 2026-05-24
Git status: dirty working tree; RCS-AC-v2 is untracked during initial implementation
Config: local Stage 2 privileged policy smoke defaults in `train_stage1_privileged_policy.py`

## Objective

Start privileged policy clean work by adding the local actor-critic and
world-model rollout smoke that must pass before any Windows CARLA clean eval.

## Sub-agent Findings

Agents A, B, and C were spawned at Stage 2 entry according to the persistent
sub-agent protocol.

Agent A confirmed Stage 2 deliverables, local readiness gate, and full clean
gate split. Important warning integrated: do not claim Stage 2 clean PASS after
local smoke only; CARLA 128/512/2048 evidence is still required.

Agent B confirmed the actor-critic/lambda-return path existed but flagged that
the first smoke used posterior latents plus logged synthetic actions/rewards
instead of true policy-driven world-model imagination. Integrated fix:
`imagine_policy_rollout()` now loops policy latent -> action -> RSSM imagine ->
reward/continue heads, and tests verify finite rollout tensors, valid action
ids, deterministic rollout repeatability, and lambda-return edge cases.

Agent C confirmed the same contract gap and noted PDF exactness differences:
the current critic is scalar value regression rather than a return distribution
NLL, and advantage normalization is batch normalization rather than Appendix E's
EMA percentile range. This is recorded as a local smoke approximation, not a
PDF-exact final policy implementation.

## Implementation Brief

Implement only the local Stage 2 policy smoke. Do not run Windows CARLA yet and
do not start raw sensor stream or fusion ablation.

Use Stage 1 latent/world-model outputs as the policy input. Add an actor-critic
with 39-action logits, scalar value prediction, entropy, lambda-return utility,
critic regression, actor policy-gradient style loss with detached advantage,
and entropy regularization. Add a privileged policy wrapper that can act from
RSSM latent, plus a synthetic rollout/training smoke proving finite losses and
valid action IDs.

Recorded Stage 2 local smoke approximations:

```text
Critic: scalar value regression for smoke; PDF-exact return distribution NLL is pending.
Advantage scaling: batch advantage normalization for smoke; Appendix E EMA percentile scaling is pending.
Rollout source: policy-driven RSSM prior imagination with reward/continue heads.
Clean evidence: local smoke only; not a CARLA clean pass.
```

The local Stage 2 smoke is not a clean-driving claim. It only prepares the
policy code path for later Windows CARLA 128/512/2048 clean gates.

## Escalation Decision

Lite sub-agent mode is active with Agent A, Agent B, and Agent C. No Full
escalation yet. Escalate if the policy loss or rollout contract conflicts with
Raw2Drive/Dreamer-style behavior training.

## Implemented Work

Completed local smoke:

- Added actor-critic with logits `[B,T,39]`, scalar value `[B,T]`, entropy,
  deterministic argmax action, stochastic sampled action, and Raw2Drive action
  id compatibility.
- Added lambda-return and behavior loss with critic regression, detached
  normalized advantage, and entropy regularization.
- Added privileged policy wrapper and policy-driven world-model imagination
  helper.
- Added local policy smoke training over frozen privileged world-model rollout
  rewards/continues.
- Added local clean summary contract that explicitly marks itself as non-CARLA
  evidence.
- Added tests for action/value/entropy shapes, deterministic argmax actions,
  reproducible sampled actions, lambda edge cases, behavior loss finiteness,
  entropy sign, imagined rollout shapes, deterministic rollout repeatability,
  and local policy smoke metrics.

Completed Windows clean-eval preparation:

- Added 128-step Stage 2 clean eval config with action/reward hashes,
  continue policy, route/weather/sync settings, required metrics, pass criteria,
  failure classes, and per-step log schema.
- Extended `evaluate_clean.py` with dry-run summary generation, summary
  validation, official PASS metadata checks, finite metric checks, `steps`
  versus `horizon`, and local-smoke anti-PASS safeguards.
- Added a locally testable clean-eval adapter loop. The loop validates Stage 2
  privileged observations, finite `[1,1,39]` policy logits, deterministic
  Raw2Drive action selection, action-to-control conversion, reward/continue
  contracts, per-step metrics CSV, summary JSON, failure classes, and
  `env.close()` cleanup.
- Added PASS protection that cross-checks `summary.json` against `metrics.csv`.
  A Windows `PASS` now requires source `windows_carla`, horizon rows in
  `metrics.csv`, `done_reason=horizon_reached`, zero collision/offroad/stuck
  rows, no adapter/runtime error flags, pulled-file evidence, and official
  metadata.
- Updated Windows `run_stage2_clean_eval.ps1`. `-DryRun` still skips CARLA
  readiness; real mode runs optional `git pull --ff-only`, dependency probe,
  the v2-local CARLA readiness guard, then calls `evaluate_clean.py --run-carla`.
- Added v2-local `ensure_carla_ready.ps1` and `start_carla_interactive.ps1`
  so Stage 2 no longer depends on untracked `shared/scripts/windows` for CARLA
  readiness.
- Added `check_stage2_windows_dependencies.ps1`, which checks v2 code, env
  config, v2 readiness scripts, workspace parent, and sibling `shared_core`
  before CARLA starts.
- Added `run_stage2_privileged_obs_probe.ps1` and
  `rcs_ac_v2.evaluation.probe_stage2_windows_env`, a shape-only probe that
  writes `privileged_obs_probe.json` with `clean_gate_evidence=false`.
- Added `rcs_ac_v2.evaluation.carla_adapter:create_stage2_carla_adapter` and
  set it in the clean-128 config. The factory lazily uses the v1.1-style
  `rcs_ac_first.scripts.train._build_env()` shared-core CARLA builder, wraps
  Stage 2 privileged observations, loads world-model/policy checkpoints,
  exposes `policy.act(obs, info) -> logits [1,1,39]`, and bridges Raw2Drive
  controls to the temporary shared-core `stage4_v2 [steer, throttle-brake]`
  action contract.
- Added a v2-owned simulator-state privileged observation builder. When the
  shared-core env does not expose `privileged_bev`/`privileged_ego`, the adapter
  builds semantic43 BEV `[1,1,43,16,16]` and ego8 `[1,1,8]` from CARLA
  world/map/vehicle/route-manager state. It records schema
  `carla_simulator_semantic43_v1`, channel hash
  `382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3`, source
  `carla_simulator_state`, and `privileged_observation_clean_gate_eligible`.
- Hardened clean eval so derived/raw-debug privileged tensors cannot become
  clean evidence. A clean run now rejects missing provenance, schema mismatch,
  channel-hash mismatch, and `derived_debug_from_raw_v1` style observations.
- Added the first real-checkpoint path after the Windows privileged probe:
  `privileged_replay.py` for simulator-state semantic43/ego8 replay IO,
  `collect_stage2_privileged_replay.py` plus a Windows runner for CARLA replay
  collection, and `train_stage2_privileged_from_replay.py` plus a Windows
  runner for replay-trained world-model/policy checkpoint export.
- Added checkpoint provenance enforcement. The base clean config now accepts
  only checkpoint payloads with `source=windows_carla_privileged_replay` and
  rejects `local_synthetic_smoke_not_carla` before clean eval can start.
- Added a direct_3d reverse-action guard. If the policy selects a Raw2Drive
  action with `reverse=True`, the adapter fails as `contract` instead of
  silently dropping reverse in the shared env bridge.
- Real mode now classifies missing checkpoint paths as `failure_class=model`
  before constructing the CARLA env.
- Added Stage 2 checkpoint export from the privileged policy smoke. The export
  writes paired world-model/policy `.pt` files, sha256 values, and a
  manifest/config patch. These synthetic checkpoints are plumbing evidence
  only, not CARLA clean-performance evidence.
- Checkpoint loading now requires both path and sha256, validates payload
  config/state_dict, rejects 8x8 world-model checkpoints for the 16x16 clean
  config, rejects missing policy config, and keeps checkpoint failures in
  `failure_class=model`.
- Added direct tests for unreadable checkpoint paths, invalid policy config,
  policy latent-dim mismatch, and bad policy state-dict load failures.
- Windows runner `PYTHONPATH` now includes v2 code and the workspace root needed
  by the shared-core env builder. Legacy v1.1 support/code paths can still be
  supplied as overrides for local experiments.
- Added `RCS-AC-v2/.gitignore` so v2 artifact policy is self-contained even if
  root `.gitignore` changes are not part of the v2 commit.
- Hardened `check_git_hygiene.sh` so v2 nested `.git`, v2 heavy artifacts, or
  v2 files larger than 50 MB fail with a nonzero exit.
- Updated `scripts/windows/README.md` to reflect that Stage 2 real-mode runner,
  privileged obs probe, replay collector, replay trainer, and v2-local minimal
  shared_core env builder bridge exist.
- Created and pushed the dedicated v2 branch
  `codex/rcs-ac-v2-raw2drive`. The commit scope was limited to the
  `RCS-AC-v2/` subtree; unrelated root worktree changes were not staged.
- Added `windows_128_result.md` template for the first official Windows clean
  result.
- Fixed the Windows Stage 2 wrappers so optional dependency overrides are passed
  only when non-empty. This lets the official clean clone run the dependency
  probe and privileged-observation probe without local override arguments.
- Pulled the official clean clone
  `C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90` to commit
  `e42dc6a` and reran `run_stage2_privileged_obs_probe.ps1 -StepNeutral
  -SkipGitPull` with no override paths. The clean clone passed dependency
  checks, CARLA Town04 readiness, simulator-state privileged provenance,
  BEV/ego shape and finite checks, and neutral-step safety checks. The result
  remains `clean_gate_evidence=false`.
- Hardened replay/checkpoint provenance before clean promotion. Clean/replay
  now require `privileged_observation_source=carla_simulator_state`; existing
  env privileged keys are clean-ineligible; replay training requires collector
  manifest/file sha validation; checkpoint payloads must record role, trainer,
  replay manifest, data files, non-clean evidence, and valid action ids.
- Collected the first official clean-clone privileged replay at commit
  `ef3b530`: 4 episodes, 128 steps each, total 512 steps, source
  `windows_carla_privileged_replay`, collector
  `collect_stage2_privileged_replay`, privileged source
  `carla_simulator_state`.
- Trained replay-source checkpoints from that replay and ran the first official
  128-step clean eval from the clean clone. The run reached `source=windows_carla`
  but failed as `failure_class=driving` / `done_reason=stuck` after 64/128
  steps. Collision/offroad were zero, reverse actions were masked out, and
  adapter/runtime errors were zero.
- Added replay imitation warm-start after the first failure showed action
  collapse to action id 8 for all 64 clean steps. Windows retraining completed
  into `windows_replay_b20d8eb_bc_pilot`; action probe and clean rerun are
  pending because the Windows Tailscale peer became temporarily unreachable.
- Rejected the `windows_replay_b20d8eb_bc_pilot` checkpoint before clean rerun:
  its action probe covered 512 replay observations but collapsed to action
  id 34 with `nonzero_throttle_fraction=0.0` and
  `recommended_for_clean_rerun=false`.
- Fixed the replay-label path with explicit `TargetSpeedKmh` precedence and
  removal of waypoint-distance stop labels, then collected the
  `windows_privileged_9c24ff7_fast35_pilot` replay. It contains 4x128
  simulator-state privileged episodes with action counts `{7:264,15:248}` and
  `nonzero_throttle_fraction=1.0`; this remains non-clean training evidence.
- Retrained from the fast35 replay through the scheduled-task detached wrapper,
  probed the resulting checkpoint over all 512 replay actions
  (`{7:500,15:12}`, no reverse actions, no single-action collapse), and ran the
  official 128-step clean rerun from the clean clone at commit `37a0875`.
  Result: `source=windows_carla`, `status=PASS`, `steps=128/128`,
  `done_reason=horizon_reached`, zero collision/offroad/stuck,
  `route_progress=0.1794871794871795`, and 128 metric rows. This is the first
  Stage 2 128-step clean PASS, not the end of clean-performance validation.
- Added offline replay-latent separability diagnosis for the steer2 checkpoint
  failures. Across the four rejected `d4c5b70` candidates, a frozen world-model
  latent linear probe reached test accuracy `0.7384615540504456` against replay
  majority baseline `0.6831530139103554`, which means action information is
  present but weakly separated. The next Stage 2 patch should improve
  latent/action separability or the sequence-aware policy objective before any
  new CARLA 512 clean run.
- Added `WorldModelActionAuxWeight` to replay training as a world-model-only
  action-prediction auxiliary loss. Candidate
  `7f87656_steer2_seq010_aux1_balpow05_iw5` passed offline gates with
  `recommended_for_clean_rerun=true`,
  `moving_steer_fraction=0.14837712519319937`,
  `dominant_action_fraction=0.6738794435857806`, and latent probe test accuracy
  `0.9076923131942749`. Its official 512 clean eval still failed as a driving
  collision at `steps=178/512`, `route_progress=0.6410256410256411`, with clean
  run action counts `{1:2,4:15,7:76,10:3,15:82}`.

## Tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
109 passed in 2.61s
```

Additional policy-driven rollout smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python - <<'PY'
from rcs_ac_v2.training.train_stage1_privileged_policy import run_privileged_policy_smoke
result = run_privileged_policy_smoke(steps=30, seed=13)
print(result)
PY
```

Result:

```text
initial_critic_loss: 0.001165103050880134
final_critic_loss: 0.00015270980657078326
final_total_loss: -0.0037799356505274773
final_entropy: 3.655482053756714
rollout_reward_mean: -0.06795604526996613
rollout_continue_mean: 0.5067161917686462
valid_action_ids: True
actor: -0.0027716318145394325
critic: 0.00015270980657078326
entropy: 3.655482053756714
lambda_return_mean: -0.10952942073345184
total: -0.0037799356505274773
```

Clean eval dry-run contract:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m rcs_ac_v2.evaluation.evaluate_clean \
  --config RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml \
  --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m rcs_ac_v2.evaluation.evaluate_clean \
  --validate-summary RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
```

Result:

```text
DRY_RUN_SUMMARY=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
SUMMARY_VALID=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
```

The generated dry-run summary was removed after validation so it cannot be
mistaken for official Windows evidence.

Local adapter-loop and CARLA-factory contract:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests/test_evaluate_clean_contract.py RCS-AC-v2/tests/test_stage2_carla_adapter.py RCS-AC-v2/tests/test_privileged_policy_smoke.py -q -p no:cacheprovider
```

Result:

```text
31 passed in 2.23s
```

The adapter tests cover local loop PASS, collision-as-driving failure, adapter
failure, continue-contract failure, non-finite-logit model failure, missing
adapter factory failure summary, metrics row-count/header/aggregate validation,
Windows runner static behavior, checkpoint sha validation, Stage 2 policy
wrapper logits, shared-core env wrapper, direct_3d reverse-action rejection,
checkpoint export manifest generation, exported checkpoint policy loading,
unreadable checkpoint path rejection, invalid policy config rejection, policy
latent mismatch rejection, policy state-dict load failure classification,
stage4_v2 action bridge, dependency-probe static contract, v2-local readiness
scripts, and non-clean privileged-observation probe output.

Stage 2 checkpoint export CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m rcs_ac_v2.training.train_stage1_privileged_policy \
  --steps 4 \
  --seed 19 \
  --bev-height 16 \
  --bev-width 16 \
  --ego-dim 8 \
  --hidden-dim 32 \
  --output-dir /tmp/rcs_ac_v2_stage2_export_smoke \
  --export-checkpoints
```

Result:

```text
stage2_privileged_checkpoint_manifest.json
stage2_privileged_policy.pt
stage2_privileged_world_model.pt
```

The `/tmp/rcs_ac_v2_stage2_export_smoke` directory was removed after the
export smoke verified the files.

Real-mode null-checkpoint classification probe:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m rcs_ac_v2.evaluation.evaluate_clean \
  --config RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml \
  --run-carla \
  --commit LOCAL_CHECKPOINT_NULL_PROBE \
  --branch codex/local-probe \
  --windows-command local-null-checkpoint-probe \
  --carla-version NOT_RUN
```

Result:

```text
EVAL_EXIT_CODE=2
SUMMARY_VALID=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
failure_class=model
done_reason=world_model_checkpoint_missing
clean_gate_pass=False
```

The generated failure summary was removed after validation so it cannot be
mistaken for official Windows evidence.

## Gate

Status: `FAST35_128_CLEAN_PASS__512_PRIMARY_PENDING`

Required local evidence:

```text
actor-critic forward shapes pass
lambda-return/behavior losses finite
world-model rollout policy smoke returns valid action ids
synthetic policy optimization produces stable finite metrics
CARLA 128-step clean eval PASS from official Windows clean clone
CARLA 512-step primary subset remains pending
```

Evidence:

```text
actor-critic forward shapes pass: pytest actor_critic tests
lambda-return/behavior losses finite: pytest behavior_loss tests
world-model rollout policy smoke returns valid action ids: pytest privileged_policy_smoke tests
synthetic policy optimization stable/improves: critic 0.001165 -> 0.000153
CARLA 128-step single-start: PASS at commit 37a0875 from official Windows clean clone
clean eval dry-run: summary contract validates but source is LOCAL_SMOKE_ONLY
adapter-loop fake env: PASS/FAIL/failure-class/metrics contracts pass locally
shared-core CARLA factory: checkpoint loader/policy wrapper/env wrapper tests pass locally
checkpoint export: synthetic world_model/policy `.pt` files plus sha256 manifest are created and load locally
real-mode missing checkpoint: classified as model/world_model_checkpoint_missing with a valid failure summary
artifact hygiene: v2 heavy file negative probe fails; clean hygiene pass has no v2 heavy files, no v2 nested .git, and no >50MB v2 files
git sync: origin/codex/rcs-ac-v2-raw2drive exists and contains the RCS-AC-v2 subtree for Windows pull
Windows dependency probe: PASS on scratch archive runtime at commit 5b00347 and official clean clone at commit e42dc6a; v2-local readiness scripts, v2-local env config, and sibling shared_core were present
privileged obs probe: PASS on Windows CARLA Town04 at commit 5b00347 scratch archive and commit e42dc6a official clean clone with clean_gate_evidence=false before any clean PASS attempt
privileged provenance: source=carla_simulator_state, schema=carla_simulator_semantic43_v1, hash=382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3
privileged tensors: BEV [1,1,43,16,16] finite, ego [1,1,8] finite, neutral step collision/offroad/stuck false
replay checkpoint path: replay IO/sampler, fake collector test, two-step replay training export, and strict checkpoint loader all pass locally
checkpoint provenance gate: local_synthetic_smoke_not_carla checkpoints are rejected by the official clean config
Windows replay path scratch: commit e1c9c86 archive dependency probe passed, replay modules imported under Windows Python 3.12, 1x8-step privileged replay collection produced source=windows_carla_privileged_replay manifest, 2-step replay training produced world_model/policy sha256 manifest, and strict checkpoint loader accepted the replay-source checkpoints
official sync lane: Windows SSH git access works, clean clone path C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90 is at e42dc6a after core.longpaths=true, and the adapter now owns the minimal shared_core env builder; sibling shared_core remains external
official clean clone probe: source=carla_simulator_state, schema=carla_simulator_semantic43_v1, hash=382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3, BEV [1,1,43,16,16], ego [1,1,8], finite tensors, neutral step running/collision=false/offroad=false/stuck=false
official replay collection: commit ef3b530 clean clone, 4x128 episodes, total_steps=512, source=windows_carla_privileged_replay, privileged_observation_source=carla_simulator_state, collector manifest sha/file checks enforced
first promoted replay checkpoint: world_model final_total 2.0421366691589355, policy final_total 0.058986030519008636, valid_action_ids=True, strict loader accepted checkpoint lineage, reverse action probe after mask had reverse_count=0 on 64 replay observations
first 128 clean eval: commit db1e8b6, source=windows_carla, status=FAIL, failure_class=driving, done_reason=stuck, steps=64/128, collision_count=0, offroad_count=0, stuck_count=1, route_progress=0.02564102564102564, metrics_rows=64, adapter_errors=0, runtime_errors=0
failure diagnosis: clean policy selected action_id=8 for all 64 steps; action 8 has throttle=0.0, brake=0.0, steer=0.6, so the car stopped with no route progress and triggered stuck
BC warm-start: commit b20d8eb adds replay imitation loss to policy training; Windows retraining completed to windows_replay_b20d8eb_bc_pilot, but action probe/clean rerun are pending Windows SSH recovery
BC checkpoint probe: windows_replay_b20d8eb_bc_pilot rejected before clean rerun, probed_actions=512, action_counts={34:512}, nonzero_throttle_fraction=0.0, recommended_for_clean_rerun=false
fast35 replay relabeling: commits 30b7313 and 9c24ff7 produced windows_privileged_9c24ff7_fast35_pilot with action_counts={7:264,15:248}, nonzero_throttle_fraction=1.0, clean_gate_evidence=false
fast35 checkpoint action probe: b4c8e58/37a0875 checkpoint passed pre-clean gate with probed_actions=512, action_counts={7:500,15:12}, reverse_action_count=0, single_action_collapse=false, previous_stuck_action8_count=0, recommended_for_clean_rerun=true
fast35 128 clean eval: commit 37a0875, source=windows_carla, status=PASS, clean_gate_pass=true, done_reason=horizon_reached, steps=128/128, collision/offroad/stuck counts all zero, route_progress=0.1794871794871795, episode_return=13.545934572121052, metrics_rows=128, adapter_errors=0, runtime_errors=0
fast35 512 primary eval: commit 0b10ff5, source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=311/512, collision_count=1, offroad_count=0, stuck_count=0, route_progress=0.5641025641025641, episode_return=19.066852226147766, metrics_rows=311, adapter_errors=0, runtime_errors=0, action_counts={7:308,15:3}
offline steer2 replay sweep: commit d4c5b70 added sampler priority-window stats, imitation_balance_power, and dominant-action probe gating. Four steer2 candidates were rejected before CARLA: balpow05 {7:647}, balpow0625 {7:647}, balpow075 {4:639,7:8}, and no-balance {7:647}. The replay has priority_window_count=212/617, so the failures point to policy collapse rather than missing steering windows.
corr050 component guard: after BF7 failed the 512 clean gate and corr050 t12/t16 variants produced identical steer2 distributions, the trainer now records component_sampler_stats plus sampler included/dropped episode lists and fails fast if an explicit positive component weight has no sequence_length-eligible episodes. Local test: PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py -> 29 passed. Windows clean clone pulled commit 32a65cb and confirmed both mixed_t22_steer2_corr050_t12 and mixed_t22_steer2_corr050_t16 have corr050 episodes length 10, below sequence_length=16, so the corr050 component was never entering training.
corr048 min16 branch: Windows CARLA collection at route_progress>=0.48 produced 4 eligible correction episodes with lengths [17,17,17,17] and action_counts={2:16,5:32,8:20}. The mixed replay component_sampler_stats confirmed corr048 included_episodes=4, dropped_short_episodes=0, window_count=8, configured probability 0.2. CUDA training completed for 33d1705_corr048_t16_min16_w025_bf7base, but cross-replay gate rejected before CARLA: target22 dominant_action_fraction=0.9272030651 and steer2 dominant_action_fraction=0.9119010819, even though both latent probes passed.
```

## Blockers / Next Gate

Windows CARLA clean eval has now produced the first Stage 2 128-step PASS from
the official clean clone. The same checkpoint lineage failed the 512-step
primary subset as a driving/collision failure at step 310, after mostly straight
actions 7/15. Stage 2 still cannot be treated as fully clean-stable until a
patched checkpoint passes the 512-step primary subset and later broader clean
gates, or those gates are explicitly blocked and classified.

The generic Stage 2 adapter loop, shared-core CARLA factory, checkpoint loader,
local synthetic checkpoint export path, simulator-state privileged observation
builder, Windows privileged-observation probe, replay collector, replay trainer,
and checkpoint provenance gate are implemented. The base clean config remains
null-checkpoint for source control; runtime checkpoint-filled configs are
generated from replay checkpoint manifests and kept as run artifacts.

The current Stage 2 runner still has a temporary reproducibility dependency on
sibling `shared_core`. The adapter now owns the minimal shared_core env builder
and the v1.1 env config is copied under `RCS-AC-v2`, so untracked
`RCS-AC-v1.1` support/code is no longer needed for the clean clone lane.
Official Windows execution should continue using the clean clone/git pull lane
before claiming performance evidence. The existing Windows repo was dirty and
GitHub HTTPS credential fetch failed, so it was not used as the official sync
source in this round.

## Next Action

Patch the 512 collision by keeping the component eligibility guard, but do not
promote the first corr048 min16 candidate. It entered training correctly and
then over-corrected into dominant straight action 7 on both target22 and
steer2. The next candidate must reduce this straight-action collapse before
any official 512 clean eval is attempted. Do not start raw/fusion work until
the 512-step result is PASS or explicitly blocked after classified retry
attempts.
