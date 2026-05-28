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

Status: `H3_32X32_CLEAN_STABILITY_FILL_IN_PROGRESS__OFFLINE_ACTION_COLLAPSE`

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
corr048 w005 sweep: subagent sidecar recommended corr048=0.05. Three CUDA candidates were trained/probed and all rejected before CARLA. ssf005 passed steer2 but failed target22 with action15 dominant_action_fraction=0.8812260536. ssf010 failed target22 at 0.8735632184 and steer2 moving_steer_fraction=0.3137557960. ssf015 failed target22 at 0.8620689655 and steer2 moving_steer_fraction=0.3338485317. This shows corr048 weight/ssf sweeps shifted the failure from action7 collapse to action15/right-turn dominance; official 512 clean eval remains blocked by offline gates.
corr048 cap15 direct-control candidate: 8b78015_corr048_w005_ssf005_cap15w1c82 reused the ssf005/corr048=0.05 setting and added ActionMassCapWeight=1.0 with ActionMassCaps=15=0.82. Windows CUDA training completed and the cross-replay gate rejected before CARLA. target22 still failed with action_counts={15:460,4:34,10:6,13:2,19:2,1:2,7:16}, dominant_action_fraction=0.8812260536, moving_steer_fraction=0.0881226054. steer2 remained healthy with recommended_for_clean_rerun=true, action_counts={15:438,7:141,4:55,1:10,12:3}, moving_steer_fraction=0.1051004637, dominant_action_fraction=0.6769706337. The direct cap did not yet move target22 below the 0.85 dominant-action gate, so no official CARLA eval was run.
corr048 cap15 stronger candidate: 8b78015_corr048_w005_ssf005_cap15w4c80 increased ActionMassCapWeight to 4.0 and tightened ActionMassCaps=15=0.80. It produced the same target22 and steer2 action distributions as the weaker cap, so it was rejected before CARLA. This suggests global action-mass capping is too indirect for the action15 argmax failure.
corr048 logit15 candidate: 8b78015_corr048_w005_ssf005_logit15w05m0 added RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=15, RouteActionLogitCapMargin=0.0 over route_progress 0.0-1.0. It passed the cross-replay gate: target22 recommended_for_clean_rerun=true with action_counts={15:442,4:40,7:38,19:2}, dominant_action_fraction=0.8467432950; steer2 recommended_for_clean_rerun=true with action_counts={15:420,7:167,4:49,1:8,12:3}, moving_steer_fraction=0.0927357032; both latent margins stayed above 0.15. Official Windows 512 clean eval then ran after CARLA readiness and failed as source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=146/512, route_progress=0.6666666667, collision_count=1, offroad_count=0, stuck_count=0, action_counts={15:117,7:17,4:10,12:2}. The last two steps selected action12 (throttle=0.7, steer=-0.1), and the collision occurred on the second action12.
corr048 logit12/15 candidate: 8b78015_corr048_w005_ssf005_logit1215w05m0 expanded RouteActionLogitCapIds to 12,15. It again passed cross-replay gate: target22 recommended_for_clean_rerun=true with action_counts={15:442,4:40,7:38,19:2}, dominant_action_fraction=0.8467432950; steer2 recommended_for_clean_rerun=true with action_counts={15:419,7:170,4:48,1:8,12:2}, moving_steer_fraction=0.0896445131. Official 512 clean eval still failed as source=windows_carla, status=FAIL, done_reason=collision, steps=146/512, route_progress=0.6666666667, collision_count=1, offroad_count=0, stuck_count=0, action_counts={15:116,7:18,4:10,1:1,12:1}. The only action12 occurred at step145 and collided immediately.
corr048 logit12/15 bracket: increasing the same full-route 12,15 logit cap to weight 2.0 was rejected offline because target22 collapsed to action7 with dominant_action_fraction=0.9195402299 and steer2 collapsed to action7 with dominant_action_fraction=0.9103554869. The midpoint weight 1.0 passed offline with balanced target22 action_counts={7:262,15:220,4:38,19:2} and steer2 action_counts={7:383,15:208,4:49,1:7}, but official 512 failed earlier at route_progress=0.5384615385 with collision_count=1 and offroad_count=1 after action19 repeated at steps 144-147. The full-route logit cap bracket therefore trades late action12 collision for earlier right-oversteer when strengthened.
corr048 late throttle ceiling: 8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4 kept the full-route action15 logit cap and added RouteThrottleCeilingWeight=4.0 with RouteThrottleCeiling=0.3 only around route_progress 0.64-0.70. It passed the cross-replay gate: target22 action_counts={15:442,4:40,7:38,19:2}, dominant_action_fraction=0.8467432950; steer2 action_counts={15:414,4:48,7:177,1:8}, moving_steer_fraction=0.0865533230. Official 512 still failed as source=windows_carla, status=FAIL, done_reason=collision, steps=146/512, route_progress=0.6666666667, collision_count=1, offroad_count=0, stuck_count=0, action_counts={15:116,7:19,4:10,1:1}. The high-throttle action12 tail disappeared, but the final collision happened on slow straight action7 at step145. This narrows the next patch to late-window geometry/action selection rather than throttle-only suppression.
late056 replay follow-up: two warmup correction replays were collected from the lateceil checkpoint-filled config at record_after_route_progress=0.56. The t16 replay produced 4 episodes / 144 steps with action_counts={2:4,28:68,31:24,34:48}; the t24 replay produced 4 episodes / 142 steps with action_counts={2:4,28:48,30:18,31:25,34:47}. Both are eligible by min16 length but are dominated by zero-throttle steering labels. A conservative offline-only mixed candidate, 4e9f4b3_corr048_w005_late056t24_w002_logit15_lateceil, mixed t24 at ComponentSampleWeights target22=1,steer2=1,corr048=0.05,late056t24=0.02 while preserving action15 logit cap and late throttle ceiling. CUDA training completed, steer2 passed, but target22 failed with action_counts={15:462,4:42,7:12,12:4,1:2} and dominant_action_fraction=0.8850574713. No official CARLA eval was run. This rejects the current late056 teacher-label route unless the labeling heuristic is changed to avoid zero-throttle steering labels.
late045 min-throttle replay follow-up: commit 5bbca2a added `min_recorded_throttle` to the privileged replay collector and Windows wrapper. Collection at route_progress>=0.56 and >=0.50 with target_speed_kmh=24, min_recorded_throttle=0.3, and min_recorded_steps=16 did not produce enough successful eligible episodes, but route_progress>=0.45 produced 4 eligible episodes / 64 steps with action_counts={7:8,10:12,13:16,16:16,19:12}. The mixed candidate 5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil passed the target22/steer2 cross-replay gate, then official Windows 512 clean eval reached route_progress=0.7692307692 before collision after a right-steer tail ending in action19. A mask19 cross probe also passed, but the official mask19 run again failed at route_progress=0.7692307692; the final tail shifted to action10 -> action13 -> action16. This is the best Stage 2 clean progress so far, but not a clean pass. The next patch should target the late 0.74-0.77 right-steer transition rather than repeating broad action19 masks.
late mask diagnostics: c2a7767 documented the late045 min-throttle branch and then ran two mask-only diagnostics from the Windows clean clone. `ForbiddenActionIds=16,19` passed cross replay and a 128-step smoke run, but the real 512 run failed at route_progress=0.7692307692 with final action8 (throttle=0.0, steer=0.6). `ForbiddenActionIds=8,16,19` also passed cross replay and improved official route_progress to 0.7948717949, but failed with final action25 (throttle=0.0, steer=-1.0). These runs confirm that the late failure is a zero-throttle sharp-steer fallback family, not one bad action id. Stop broad mask escalation and patch the late-window objective or replay labels so moving mild steering is preferred over zero-throttle sharp steering.
PDF-core H0 rebase update on 2026-05-27 KST: the active master plan is now `RCS-AC-v2/docs/08_raw2drive_pdf_core_clean_first_master_plan.md`. `route_priority_components` and `priority_episode_mask` are classified as clean-stability fill/diagnostic sampler controls, not PDF-explicit core. `--pdf-stage1-strict` rejects `route_priority_components` and the surrounding non-PDF diagnostic policy settings. The component priority mask is wired through the Python trainer, Windows foreground trainer, detached trainer, and mixed-candidate cycle wrapper. Validation: `python -m pytest RCS-AC-v2/tests/test_privileged_replay_training.py -q` -> 33 passed in 2.99s; `python -m pytest RCS-AC-v2/tests -q` -> 151 passed in 2.86s. This is not clean performance evidence; it only closes the H0 WIP classification/guarding gap before future Windows official runs.
H1 local implementation update on 2026-05-27 KST: simulator-state privileged observations now report BEV shape/grid spec and route/lane/obstacle coverage diagnostics; replay collection accepts `[43,H,W]` instead of hard-coded `[43,16,16]`; replay episode metadata and replay/checkpoint manifests record BEV/ego shapes plus schema/hash; replay trainer derives `PrivilegedWorldModelConfig.bev_height/bev_width` from replay shape and rejects mixed H/W replay sets. Local 32x32 tests cover simulator-state observation, replay roundtrip, replay-trained world model config, matching 32x32 policy runner acceptance, and 16x16 checkpoint vs 32x32 contract rejection. Validation: `python -m pytest RCS-AC-v2/tests/test_stage2_carla_adapter.py RCS-AC-v2/tests/test_privileged_replay_training.py -q` -> 61 passed in 3.51s; `python -m pytest RCS-AC-v2/tests -q` -> 157 passed in 3.16s. This is still not Windows clean performance evidence; Windows 32x32 privileged obs probe and tiny replay collection remain pending.
H1 Windows readiness update on 2026-05-28 KST: added the committed 32x32 probe config `RCS-AC-v2/configs/eval/clean_128_stage2_privileged_32x32.yaml`, preserving semantic43+ego, Raw2Drive 39-action, reward, and clean-gate contracts while changing only privileged BEV H/W to `[1,1,43,32,32]` and separating outputs under `windows_128_32x32`. The Windows privileged observation probe wrapper now separates default output when the config path contains `32x32`, and the Windows workflow document now points H1 probe/tiny-replay commands to the 32x32 config and replay directory. This is still not Windows clean performance evidence; the next evidence must come from the clean Windows clone after pull.
H1 Windows probe/replay evidence on 2026-05-28 KST: diagnosed Windows access failure as Mac-side codex userspace Tailscale stale socket/daemon exit, not CARLA/model failure. Restarting userspace `tailscaled` restored SOCKS SSH to Windows host `Song`. The official clean clone pulled branch `codex/rcs-ac-v2-raw2drive` to commit `27a3450ca270047d43f15f3812e4e2b7d4c0ab6a`, with tracked files clean under `--untracked-files=no`. CUDA/dependency guard passed on NVIDIA GeForce RTX 3080. The 32x32 privileged observation probe passed: output `windows_128_32x32/privileged_obs_probe.json`, `clean_gate_evidence=false`, status `PROBE_PASS`, source `carla_simulator_state`, schema `carla_simulator_semantic43_v1`, BEV `[1,1,43,32,32]`, ego `[1,1,8]`, route/lane/obstacle availability true, neutral step collision/offroad/stuck false. Empty critical channels were `vehicle`, `pedestrian`, and `dynamic_vehicle`, while actor near/far and static obstacle channels were nonzero. Tiny replay collection also passed: manifest `replay/windows_privileged_h1_32x32_tiny/stage2_privileged_replay_manifest.json`, 1 episode, 8 total steps, `privileged_bev_shapes=[[8,43,32,32]]`, `privileged_ego_shapes=[[8,8]]`, source `windows_carla_privileged_replay`. CARLA was stopped afterward; RTX 3080 was 58C / 0% util / 685 MiB. This closes H1 Windows shape/coverage/replay-smoke evidence, but it is not clean performance evidence.
H2 strict 32x32 trainer smoke on 2026-05-28 KST: Windows CUDA training from the tiny H1 replay passed with `-PdfStage1Strict`, sequence_length=4, batch_size=2, world_model_steps=2, policy_steps=2. The checkpoint manifest in `checkpoints/windows_replay_h1_32x32_tiny_strict_smoke` records `pdf_stage1_strict=true`, source `windows_carla_privileged_replay`, replay shape `[[8,43,32,32]]`, world model `bev_height=32`, `bev_width=32`, `action_dim=39`, no imitation, no world-model action auxiliary, and no route/action override/cap/fill weights. Checkpoint action probe loaded the strict smoke checkpoint with `clean_128_stage2_privileged_32x32.yaml` and generated `clean_128_stage2_privileged_32x32_with_checkpoints.yaml`; it predicted action15 for all 8 tiny steps, so `recommended_for_clean_rerun=false`. This is a plumbing/shape smoke only, not clean evidence and not a checkpoint for official clean rerun. The next H2 step is real 32x32 privileged replay collection and strict PDF-faithful training/probing before any official clean claim.
H2 strict 32x32 real replay candidate on 2026-05-28 KST: the Windows clean clone at commit d6e45343 collected `windows_privileged_h2_32x32_real_4x128_seed20260524`, with 4 episodes, 512 total steps, lengths [128,128,128,128], failed_attempt_count=0, source `windows_carla_privileged_replay`, BEV shape [128,43,32,32] per episode, and ego shape [128,8]. CARLA was stopped afterward and RTX 3080 was 59C / 0% util / 685 MiB. Strict training from this replay passed on CUDA with `-PdfStage1Strict`, sequence_length=16, batch_size=8, world_model_steps=200, policy_steps=200; manifest records world model 32x32/action_dim39, `imitation_weight=0`, `world_model_action_aux_weight=0`, `route_priority_sample_fraction=0`, `route_action_override_weight=0`, final world_model total 2.050567626953125, final policy total -0.0028065170627087355, and valid_action_ids=true. The action probe rejected the checkpoint before official CARLA eval: action_counts={15:4,16:4,19:504}, dominant_action_fraction=0.984375, moving_steer_fraction=0.9921875, predicted_moving_on_label_nonmoving_fraction=0.9921875, reverse_action_count=0, previous_stuck_action8_fraction=0, recommended_for_clean_rerun=false. This records the H2 PDF-faithful baseline failure mode and is not clean evidence; next work should move to H3 clean-stability fill rather than running official clean eval from this known-bad checkpoint.
H3 32x32 clean-stability fill start on 2026-05-28 KST: Windows access was rechecked through the documented userspace Tailscale SOCKS path. SSH public-key auth to `song@100.87.219.58` worked, the clean clone path `C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656` existed, CARLA was not running, and RTX 3080 was idle at 43C / 0% util / 685 MiB. The H3 replay `windows_privileged_h3_32x32_real_2x512_seed20260524` contains 2 episodes, 530 total steps, lengths [265,265], route_progress about 0.025641 to 0.512821, and label_action_counts={7:432,10:26,13:12,15:38,16:14,19:8}. The first long fill candidate `windows_replay_h3_32x32_long_iw1_cb05_msfpa1_msnm1_seed20260524` was rejected offline before CARLA: action_counts={7:530}, single_action_collapse=true, recommended_for_clean_rerun=false. This shows the strict H2 action19 collapse was replaced by action7 majority collapse under weak class-balanced imitation. Three no-new-code follow-ups were then trained/probed on Windows CUDA and all stayed below the official clean-eval bar. `iw1_cb1_msb2_msfpa05_msnm05_rprio50` collapsed to action8 for all 530 observations, with zero_throttle_fraction=1.0 and previous_stuck_action8_fraction=1.0. `iw5_cb05_ssf005_aux1_cap7w1c70_logit7w05` improved to action_counts={7:501,10:10,13:19}, zero_throttle_fraction=0, and single_action_collapse=false, but dominant_action_fraction=0.9452830189 kept recommended_for_clean_rerun=false. Stronger `iw5_cb1_ssf010_aux1_cap7w2c65_logit7w1_rprio25` overcorrected to action_counts={10:526,15:4}, moving_steer_fraction=0.9924528302, predicted_moving_on_label_nonmoving_fraction=0.8792452830, and recommended_for_clean_rerun=false. The midpoint `iw5_cb075_ssf0075_aux1_cap7w15c67_logit7w075_rprio10` also collapsed to action10 for all 530 observations. No official CARLA clean eval was run from any H3 candidate because every offline action probe returned recommended_for_clean_rerun=false. This narrows the next H3 step away from further blind no-code cap sweeps and toward either a longer/more diverse replay set or a small trainer change that can regularize between action7 majority and action10 oversteer collapse.
H3 route-action-distribution regularizer on 2026-05-28 KST: commit 9aee5109 added a fill-only route-window action-distribution regularizer to the replay trainer and Windows wrappers. It preserves the Raw2Drive 39-action policy head, records manifest/metric fields, and is rejected by `-PdfStage1Strict`, keeping H2 PDF-core separated from H3 fill. Local validation passed: `test_privileged_replay_training.py` 37 passed, `test_evaluate_clean_contract.py` 41 passed, and full `RCS-AC-v2/tests` 159 passed. The Windows clean clone pulled 9aee5109 and ran two CUDA offline candidates on `windows_privileged_h3_32x32_real_2x512_seed20260524`. `iw5_cb05_ssf005_aux1_dist2_cap7w05` produced action_counts={7:514,10:10,13:6}, dominant_action_fraction=0.9698113208, moving_steer_fraction=0.0301886792, zero_throttle_fraction=0, and recommended_for_clean_rerun=false. `iw5_cb05_ssf005_aux1_dist10_msfpa025` collapsed to action_counts={7:530}, moving_steer_fraction=0, single_action_collapse=true, and recommended_for_clean_rerun=false. No official CARLA clean eval was run. The current H3 replay remains insufficient for clean-rerun-ready policy training; next work should collect a longer/more diverse 32x32 privileged replay before further H3 offline gates.
H3 diverse replay and mid045 correction update on 2026-05-28 KST: Windows access was restored through the documented userspace Tailscale SOCKS SSH path. The clean clone `C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656` fast-forwarded to commit 0fcb5906, CARLA readiness passed on Town04, and CARLA was stopped after each collection. New 32x32 replay components were collected from committed code: `windows_privileged_h3_32x32_diverse_t22_signm_3x512_seed20260524` with 3 episodes / 784 steps, route_progress 0.025641-0.512821, labels {7:621,10:38,13:12,15:73,16:30,19:10}; and `windows_privileged_h3_32x32_diverse_t22_signp_3x512_seed20260524` with 3 episodes / 978 steps, route_progress 0.025641-0.666667, labels {4:69,7:837,15:72}. A mixed signpm+base replay had 8 episodes / 2292 steps, but `windows_replay_h3_32x32_mixdiv_iw5_cb05_ssf005_aux1_dist2_cap7w05_seed20260524` collapsed to action_counts={7:2292}, dominant_action_fraction=1.0, recommended_for_clean_rerun=false. Two route-window correction components were then collected with RecordAfterRouteProgress=0.45 and MinRecordedThrottle=0.3: signp produced 4 episodes / 404 steps with labels {4:92,7:312}, route_progress 0.461538-0.666667; signm produced 4 episodes / 144 steps with labels {7:24,10:52,13:16,16:40,19:12}, route_progress 0.461538-0.512821. A four-component mixed replay had 14 episodes / 2310 steps, but the first mid-weighted candidate `windows_replay_h3_32x32_mixdivmid_iw5_cb05_ssf005_aux1_dist2_cap7w05_seed20260524` overcorrected to action_counts={10:2310}, moving_steer_fraction=1.0, predicted_moving_on_label_nonmoving_fraction=0.8393939394, recommended_for_clean_rerun=false. No official CARLA clean eval was run. The new data proves route coverage beyond 0.51 is available and mid045 correction labels can break action7 collapse, but the next H3 candidate must rebalance the action7/action10 bracket before any clean rerun.
H3 midm bracket and moving-steer FP probe on 2026-05-28 KST: the Windows clean clone pulled commit 3b5d9bee and ran additional CUDA offline candidates on `windows_privileged_h3_32x32_mixed_diverse_t22_signpm_mid045_seed20260524`. `mid1` with ComponentSampleWeights signp=1,signm=1,midp=1,midm=1 still collapsed to action_counts={10:2310}, moving_steer_fraction=1.0, predicted_moving_on_label_nonmoving_fraction=0.8393939394. Lowering only midm to 0.5 (`mid105`) collapsed to action_counts={7:2310}; midm=0.75 (`mid1075`) also collapsed to action_counts={7:2310}; midm=0.875 (`mid10875`) flipped back to action_counts={10:2310}. This shows the component-weight bracket is discontinuous between 0.75 and 0.875 rather than smoothly balanced. A follow-up `mid1_msfpa025` kept weights at 1,1,1,1 but changed SteerSequenceFraction to 0.025 and added MovingSteerFalsePositiveWeight=0.25. It broke single-action action10 collapse but shifted to action_counts={10:104,13:28,16:2178}, dominant_action_fraction=0.9428571429, moving_steer_fraction=1.0, and recommended_for_clean_rerun=false. No official CARLA eval was run. Next H3 work should avoid more broad component-weight-only sweeps and instead reduce or route-localize the moving-steer false-positive/steer-sequence pressure so action7/action10/action16 attractors are controlled without changing the Raw2Drive 39-action contract.
H3 route-local sampling gate and official clean eval update on 2026-05-28 KST: weak global MovingSteerFalsePositiveWeight=0.05 still overcorrected to action16 dominance and was rejected offline with action_counts={10:98,13:28,16:2184}, dominant_action_fraction=0.9454545455, moving_steer_fraction=1.0, and recommended_for_clean_rerun=false. The next route-local sampler candidate `windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524` removed global moving-steer pressure and used RoutePrioritySampleFraction=0.025 for actions 10,13,16,19 over route_progress 0.45-0.67. It passed the full offline gate with probed_actions=2310, action_counts={4:327,7:1435,10:56,13:32,16:460}, dominant_action_fraction=0.6212121212, moving_steer_fraction=0.3787878788, predicted_moving_on_label_nonmoving_fraction=0.2272727273, zero_throttle_fraction=0.0, and recommended_for_clean_rerun=true. Official Windows 128 clean eval from this checkpoint lineage passed at commit feb91059 with source=windows_carla, status=PASS, clean_gate_pass=true, steps=128/128, done_reason=horizon_reached, route_progress=0.1794871795, episode_return=13.2430181420, and collision/offroad/stuck counts all zero. Commit 342935c added `RCS-AC-v2/configs/eval/clean_512_stage2_privileged_32x32.yaml`; official Windows 512 eval from the same checkpoint lineage failed as source=windows_carla, status=FAIL, clean_gate_pass=false, failure_class=driving, done_reason=offroad, steps=338/512, route_progress=0.6153846154, episode_return=22.7896805126, collision_count=0, offroad_count=1, stuck_count=0, and action_counts={4:67,7:271}. The final tail repeated action4 at about 18 km/h from route_progress 0.5897435897 to 0.6153846154 before offroad. This is the first H3 32x32 lineage to pass offline plus official 128, but it is not a 512 clean PASS; next work should target the route 0.59-0.62 action4/offroad window and avoid broad global moving-steer pressure.
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
steer2. Lowering corr048 weight fixed the action7 collapse but exposed action15
dominance / right-turn false positives. The first global action15 cap preserved
steer2 but left target22 at dominant_action_fraction=0.8812260536, still above
the 0.85 offline gate; a stronger global cap behaved the same. A route-local
action15 logit cap cleared offline gates and moved the official collision later
to route_progress=0.6666666667. Adding action12 to the same weak logit cap
reduced clean action12 from two steps to one but did not clear the collision.
Strengthening the full-route 12,15 cap is not the right next move because
weight 1.0 reintroduced an earlier action19 right-oversteer collision and
weight 2.0 failed offline. The late throttle ceiling then removed the
high-throttle action12 tail but still collided at the same route-progress under
slow straight action7. The first late056 replay attempt was rejected offline
because the teacher labels were mostly zero-throttle steering actions and the
low-weight candidate worsened target22 action15 dominance. The next patch
should preserve the w0.5 full-route action15 behavior and the min-throttle late
replay fix, but move the correction to the newly classified route_progress
0.74-0.79 failure. Mask-only diagnostics show that action19, action16, and
action8 removal just exposes action25-style zero-throttle sharp steering, so the
next work should add a late-window training constraint/replay correction against
zero-throttle sharp-steer fallbacks before another official 512 eval. Do not
start raw/fusion work until the 512-step result is PASS or explicitly blocked
after classified retry attempts.
