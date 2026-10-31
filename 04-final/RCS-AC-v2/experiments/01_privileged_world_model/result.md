# Stage 1 Result

Date: 2026-05-24
Git status: dirty working tree; RCS-AC-v2 is untracked during initial implementation
Config: synthetic Stage 1 smoke defaults in `train_stage1_privileged_world_model.py`

## Objective

Implement the first privileged-stream world-model smoke for Raw2Drive v2:
privileged BEV encoder, RSSM prior/posterior dynamics, decoder/reward/continue
heads, prediction/KL losses, one-step rollout, synthetic overfit smoke, and
checkpoint roundtrip.

## Sub-agent Findings

Agents A, B, and C were spawned at Stage 1 entry according to the persistent
sub-agent protocol.

Agent A confirmed Stage 1 deliverables and gate criteria: RSSM/heads/losses/
rollout/privileged encoder/world model/training smoke files, plus tests for
RSSM shapes, world-model losses, privileged forward, synthetic overfit,
rollout finiteness, and checkpoint roundtrip.

Agent B flagged package risks and test requirements. Integrated fixes include
explicit `__init__.py` files, a v2 `pyproject.toml`, a `.gitignore` exception
so `rcs_ac_v2/env` source is not hidden by the global virtualenv `env/` rule,
tmp-path checkpoint tests, previous-action convention tests, and a full pytest
gate.

Agent C confirmed the local Stage 1 split is PDF-compatible as a privileged WM
smoke, while noting divergences to record: Gaussian stochastic state is used for
this smoke although the paper discusses categorical/one-hot variants, and BEV
decoder supervision is used as the local prediction target. It also required
binary BEV smoke targets, explicit stop-gradient KL tests, finite prior rollout,
and proof that `privileged_ego` affects the latent.

## Implementation Brief

Implement only Stage 1 privileged world-model smoke. Do not start privileged
actor-critic policy, raw sensor stream, Windows CARLA, or fusion ablation.

Use the Stage 0 contracts as input: privileged BEV `[B,T,43,H,W]`, action ids
`[B,T]` over 39 Raw2Drive discrete actions, reward `[B,T]`, and continue
`[B,T]`. Build a compact Gaussian RSSM with deterministic GRU state, stochastic
state, prior/posterior distributions, and heads for BEV reconstruction logits,
reward regression, and continue logits. Tests must prove forward shapes, finite
losses, free-nats KL handling, finite rollout, and checkpoint save/load.

Recorded Stage 1 smoke approximations:

```text
RSSM stochastic state: Gaussian for local smoke; categorical can be swapped in a later exactness pass.
Prediction target: privileged BEV reconstruction logits plus reward/continue heads.
Synthetic BEV targets: binary masks generated from random Bernoulli samples.
Transition convention: t=0 uses zero start action; t>0 uses previous action id.
```

This is a local smoke gate only. CARLA data collection and clean policy
evaluation remain blocked until Stage 1 passes.

## Escalation Decision

Lite sub-agent mode is active with Agent A, Agent B, and Agent C. No Full
escalation yet. Escalate if Agent C identifies a PDF-vs-code mismatch in RSSM
structure, loss definitions, or reward/continue use.

## Implemented Work

Completed:

- Added compact Gaussian RSSM with deterministic GRU state, prior/posterior,
  previous-action transition convention, and prior imagination.
- Added BEV decoder, reward head, and continue head.
- Added prediction, dynamics KL, and representation KL losses with free-nats
  clamp and stop-gradient directions.
- Added privileged BEV+ego encoder and combined privileged world model.
- Added synthetic overfit smoke training helper.
- Added v2 `pyproject.toml` and `.gitignore` exception for `rcs_ac_v2/env`.
- Added tests for RSSM shapes, prior rollout, previous-action convention,
  loss finiteness, KL stop-gradient direction, synthetic overfit,
  ego influence, and checkpoint roundtrip.

## Tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
37 passed in 2.00s
```

Additional synthetic overfit smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python - <<'PY'
from pathlib import Path
from rcs_ac_v2.training.train_stage1_privileged_world_model import run_synthetic_overfit_smoke
result = run_synthetic_overfit_smoke(output_dir=Path('/tmp/rcs_ac_v2_stage1_smoke'), steps=30, seed=7)
print(result)
PY
```

Result:

```text
initial_loss: 4.801789283752441
final_loss: 2.732663631439209
checkpoint_path: /tmp/rcs_ac_v2_stage1_smoke/stage1_privileged_world_model_smoke.pt
decoder: 0.5957834720611572
reward: 0.11754953861236572
continue: 0.019330652430653572
dynamics: 1.0
representation: 1.0
total: 2.732663631439209
```

## Gate

Status: `PASS`

Required evidence:

```text
synthetic one-episode overfit loss decreases
forward shapes pass
one rollout step has finite latent/reward/continue
checkpoint save/load roundtrip works
```

Evidence:

```text
synthetic one-episode overfit loss decreases: 4.801789 -> 2.732664
forward shapes pass: pytest forward/shape tests
one rollout step has finite latent/reward/continue: pytest rollout tests
checkpoint save/load roundtrip works: pytest tmp_path checkpoint test and /tmp smoke checkpoint
```

## Blockers

None currently. Tiny CARLA replay overfit was not run in this local Stage 1
smoke; Windows CARLA execution remains governed by the later Windows workflow.

## Next Action

Start Stage 2 privileged policy clean work. Before edits, re-trigger the
persistent sub-agent protocol for Stage 2 entry, write a Stage 2 Implementation
Brief, then implement actor-critic and local world-model rollout smoke before
any Windows CARLA clean eval.
