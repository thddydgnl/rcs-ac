# PDF Stage I Strict Privileged Rework

This note supersedes the route-local Stage 2 diagnostic direction for the clean
baseline. The current goal is to implement Raw2Drive Algorithm 1 Stage I as
faithfully as possible before adding raw sensors or fusion ablations.

## Current Decision

Proceed with a PDF Stage I strict profile:

1. Train the privileged world model from privileged replay.
2. Freeze/use the privileged world model for imagined rollout.
3. Train the privileged policy with actor-critic loss on imagined latent
   trajectories.
4. Do not use imitation loss, route-local action overrides, action mass caps,
   throttle ceilings, steering false-positive penalties, or component sampling
   weights in the clean baseline.

The previous route-0.64 to route-0.70 action overrides and throttle ceilings are
kept only as diagnostics. They must not be promoted as the clean baseline unless
we explicitly label the run as a non-PDF diagnostic ablation.

## PDF Explicit Contract

Raw2Drive Algorithm 1 Stage I specifies the following privileged-side loop:

- sample a privileged trajectory from the privileged replay buffer;
- train the privileged world model with prediction loss, dynamics loss, and
  representation loss;
- roll out trajectories in the privileged world model;
- obtain rewards and continues from the world model heads during rollout;
- train the privileged policy with actor-critic based on the imagined rollout.

The PDF does not specify route-specific corrective losses, supervised action
imitation for the privileged policy, action-distribution caps, or hard action
overrides. Those can be useful diagnostics, but they are outside the strict
Stage I baseline.

## Current Code Mapping

The implementation already has the PDF-shaped core:

- privileged replay sampler:
  `code/rcs_ac_v2/training/privileged_replay.py`
- privileged world model:
  `code/rcs_ac_v2/privileged_stream/world_model.py`
- RSSM losses:
  `code/rcs_ac_v2/world_model/losses.py`
- imagined rollout and privileged policy:
  `code/rcs_ac_v2/privileged_stream/policy.py`
- actor-critic behavior loss:
  `code/rcs_ac_v2/policy/behavior_loss.py`
- replay training entry point:
  `code/rcs_ac_v2/training/train_stage2_privileged_from_replay.py`

The same training entry point previously also accepted diagnostic controls:
imitation, moving-steer losses, action mass caps, route-conditioned throttle
ceilings, route-conditioned action caps, route action override, component sample
weights, and world-model action auxiliary loss.

## Strict Guard

`train_stage2_privileged_from_replay.py` now supports:

```bash
--pdf-stage1-strict
```

When enabled, the trainer rejects non-PDF Stage I settings before loading replay
or starting optimization. The strict profile requires:

- `behavior_weight > 0`
- `imitation_weight == 0`
- `imitation_class_balance == false`
- `steer_sequence_fraction == 0`
- all moving-steer diagnostic weights set to `0`
- `world_model_action_aux_weight == 0`
- no component sample weights
- no action mass caps
- no route action ids for route-conditioned caps
- all route-conditioned cap/ceiling/override weights set to `0`
- `route_action_override_id == -1`

The default Python and Windows trainer imitation weight is now `0.0`, matching
the strict actor-critic baseline direction.

## Windows GPU Command

For the next clean-baseline training run on Windows, use the strict profile and
CUDA guard:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\start_stage2_replay_training_detached.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first `
  -ReplayPath RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\pdf_stage1_strict_actorcritic `
  -TaskName RCSV2Stage2PdfStage1Strict `
  -PdfStage1Strict `
  -Device cuda `
  -SequenceLength 16 `
  -BatchSize 8 `
  -WorldModelSteps 200 `
  -PolicySteps 200 `
  -BehaviorWeight 1.0 `
  -ImitationWeight 0.0
```

Do not add route override, throttle ceiling, action cap, or component weighting
arguments to this run.

## Gate Order

1. Run unit tests for the strict guard and replay training smoke path.
2. Start strict Stage I training on Windows GPU.
3. Run replay action probe and latent probe.
4. Run official clean eval only if probes are sane.
5. If clean still fails, diagnose the PDF-aligned components first:
   reward/continue heads, RSSM rollout horizon, behavior loss scale, replay
   coverage, privileged observation schema, and action discretization.
6. Only after a privileged clean-capable baseline exists, proceed to PDF Stage II
   raw world model and raw policy with rollout guidance and head guidance.

## Sub-Agent Use During Goal Runs

Keep sub-agents active beyond the first setup burst:

- Every 20 to 30 minutes, ask one agent to audit current code against this strict
  document and `00_raw2drive_exact_implementation_plan.md`.
- Ask another agent to inspect the latest manifest/logs for signs that a
  diagnostic option leaked into a strict run.
- Ask a third agent to compare clean-eval failures against replay probes and
  summarize the next PDF-aligned intervention.
- Do not ask agents to design fusion changes until privileged and raw clean
  gates have passed.

## Status

As of this rework, Stage 2 is still not clean-passed. The best prior diagnostic
candidate reached route progress about `0.6667` before collision. That result is
evidence that the environment/action path executes, not evidence that the PDF
Stage I policy is solved.

The next work item is a strict actor-critic Stage I run, followed by probe and
official clean evaluation.
