# Stage 0 Result

Date: 2026-05-24
Git status: dirty working tree; RCS-AC-v2 is untracked during initial implementation
Config: `RCS-AC-v2/configs/eval/preflight_contract.yaml`

## Objective

Freeze Raw2Drive v2 contracts before any world-model or policy implementation:
39 discrete actions, tensor schemas, replay sequence sampling, reward/continue,
and raw-policy privileged leakage boundaries.

## Sub-agent Findings

Agent A confirmed current stage is `Stage 0: Contract preflight`, status
`NOT STARTED`, and identified the runbook as the broadest Stage 0 file list.
Agent A also flagged that `GOAL_STATUS.md` had a narrower file list than the
runbook, and that the PDF Table 12 action list needed transcription into tests
to prevent action-order drift.

Agent B confirmed the package should use the `RCS-AC-v2/code` layout with
explicit package initializers, pytest path setup, and tests covering action,
schema, replay, reward, leakage, and config contracts.

Agent C confirmed Stage 0 was not gate-passable before implementation and
flagged three required hardening points: freeze a local canonical action order
because PDF Table 12 has no action IDs, add exact action tuple tests, and make
continue truncation plus leakage allowlist explicit in config/tests.

## Implementation Brief

Implement only Stage 0 contracts. Do not start RSSM, world model, policy, CARLA
training, or fusion ablation. Use PDF Table 12 as the fixed action source and
pin the selected row-major table order with tests. Add validation helpers and
synthetic batch factories so later Stage 1/2 work cannot silently change shapes,
action dtype/range, continue values, or raw-policy leakage boundaries.

## Escalation Decision

Lite sub-agent mode is active with Agent A, Agent B, and Agent C. No Full
escalation yet; if Agent C finds a PDF-vs-code mismatch in action order or
leakage contract, escalate before Stage 0 gate closeout.

## Implemented Work

Completed:

- Added package initializers.
- Added Raw2Drive Table 12 39-action contract with canonical row-major order.
- Added action table SHA-256 fingerprint.
- Added reward/continue contract.
- Added tensor schema validation and raw-policy input view.
- Added in-memory sequence replay buffer and sampler helper.
- Added seed/checkpoint utilities needed by later Stage 1 smoke work.
- Expanded preflight config with contract versions, action hash, continue
  policy, and leakage allowlist/denylist.
- Added Stage 0 pytest coverage for actions, schema, replay, reward, leakage,
  and config.

## Tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
25 passed in 1.12s
```

## Gate

Status: `PASS`

Required command:

```bash
python -m pytest RCS-AC-v2/tests -q
```

Equivalent no-cache command passed.

## Blockers

None currently. Action table order is now pinned as
`raw2drive_table12_row_major_v1`; if later evidence from author code shows a
different ID mapping, update the table version and rerun Stage 0.

## Next Action

Start Stage 1 privileged world model smoke. Before edits, re-trigger the
persistent sub-agent protocol for Stage 1 entry, write a Stage 1 Implementation
Brief, then implement RSSM/world-model shape tests and synthetic overfit smoke.
