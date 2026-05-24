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
  readiness; real mode runs optional `git pull --ff-only`, the CARLA readiness
  guard, then calls `evaluate_clean.py --run-carla`.
- Added `rcs_ac_v2.evaluation.carla_adapter:create_stage2_carla_adapter` and
  set it in the clean-128 config. The factory lazily uses the v1.1-style
  `rcs_ac_first.scripts.train._build_env()` shared-core CARLA builder, wraps
  Stage 2 privileged observations, loads world-model/policy checkpoints,
  exposes `policy.act(obs, info) -> logits [1,1,39]`, and converts Raw2Drive
  controls to the shared direct_3d `[steer, throttle, brake]` bridge.
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
- Windows runner `PYTHONPATH` now includes v2 code plus the v1.1 support/code
  paths needed by the shared-core env builder.
- Added `RCS-AC-v2/.gitignore` so v2 artifact policy is self-contained even if
  root `.gitignore` changes are not part of the v2 commit.
- Hardened `check_git_hygiene.sh` so v2 nested `.git`, v2 heavy artifacts, or
  v2 files larger than 50 MB fail with a nonzero exit.
- Updated `scripts/windows/README.md` to reflect that Stage 2 real-mode runner
  exists, but still depends temporarily on v1.1/shared Windows runtime pieces.
- Created and pushed the dedicated v2 branch
  `codex/rcs-ac-v2-raw2drive`. The commit scope was limited to the
  `RCS-AC-v2/` subtree; unrelated root worktree changes were not staged.
- Added `windows_128_result.md` template for the first official Windows clean
  result.

## Tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
84 passed in 2.55s
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
latent mismatch rejection, and policy state-dict load failure classification.

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

Status: `V2_BRANCH_PUSHED__LOCAL_CHECKPOINT_EXPORT_PASS__WINDOWS_PRIVILEGED_OBS_VALIDATION_PENDING__CARLA_CLEAN_PENDING`

Required local evidence:

```text
actor-critic forward shapes pass
lambda-return/behavior losses finite
world-model rollout policy smoke returns valid action ids
synthetic policy optimization produces stable finite metrics
CARLA 128-step clean eval remains blocked until Windows readiness guard
```

Evidence:

```text
actor-critic forward shapes pass: pytest actor_critic tests
lambda-return/behavior losses finite: pytest behavior_loss tests
world-model rollout policy smoke returns valid action ids: pytest privileged_policy_smoke tests
synthetic policy optimization stable/improves: critic 0.001165 -> 0.000153
CARLA 128-step single-start: not run yet; must use Windows workflow
clean eval dry-run: summary contract validates but source is LOCAL_SMOKE_ONLY
adapter-loop fake env: PASS/FAIL/failure-class/metrics contracts pass locally
shared-core CARLA factory: checkpoint loader/policy wrapper/env wrapper tests pass locally
checkpoint export: synthetic world_model/policy `.pt` files plus sha256 manifest are created and load locally
real-mode missing checkpoint: classified as model/world_model_checkpoint_missing with a valid failure summary
artifact hygiene: v2 heavy file negative probe fails; clean hygiene pass has no v2 heavy files, no v2 nested .git, and no >50MB v2 files
git sync: origin/codex/rcs-ac-v2-raw2drive exists and contains the RCS-AC-v2 subtree for Windows pull
```

## Blockers

Windows CARLA clean eval is intentionally not run in this local smoke. Stage 2
cannot be promoted to full clean PASS until committed/pushed code is pulled on
Windows, CARLA readiness guard passes, and 128/512/2048 clean evidence is
recorded.

The generic Stage 2 adapter loop, shared-core CARLA factory, checkpoint loader,
and local synthetic checkpoint export path are implemented. The official
checkpoint paths are still intentionally null in the base clean config, and the
Windows shared-core env must still be verified to expose or be adaptable to
strict `privileged_bev [1,1,43,16,16]` and `privileged_ego [1,1,8]`. A
non-dry-run with current null checkpoint paths exits nonzero with
`failure_class=model` and a valid failure summary.

The current Stage 2 runner still has a temporary reproducibility dependency on
`RCS-AC-v1.1` shared-core env builder code and `shared/scripts/windows`
readiness scripts. Official Windows execution must either verify those
dependencies are present in the Windows checkout or port the minimum
runner/env-builder pieces into `RCS-AC-v2`.

## Next Action

Produce/promote real Stage 1 world-model and Stage 2 privileged-policy
checkpoints, fill checkpoint paths and sha256 in the clean-128 config,
commit/push v2 work, then run the official Windows sequence: 128-step
single-start first, then 512 primary subset, then 2048 subset if safe.
