# H2 PDF-Faithful 32x32 Privileged Baseline Brief

Date: 2026-05-28 KST

## Purpose

Enter Stage H2 from the completed H1 32x32 privileged observation lane without
claiming clean performance early. H2 keeps the PDF-explicit privileged baseline
separate from later clean-stability fill variants.

## Fixed Contracts

- Input: 43-channel simulator-state semantic BEV plus ego8.
- BEV size for this lane: 32x32.
- Action output: Raw2Drive 39 discrete actions.
- Final v2 control does not use waypoint, target-speed, or pure-pursuit output.
- Training source for clean eval checkpoints must be Windows CARLA privileged replay.
- Heavy replay/checkpoint artifacts stay on Windows and out of Git.

## Strict Baseline Settings

Use `-PdfStage1Strict` for the PDF-faithful baseline. The strict trainer lane
must keep these clean-stability fill knobs disabled:

- imitation loss
- class-balanced imitation
- route-priority sampling/components
- moving-steer false-positive or margin losses
- world-model action auxiliary loss
- action mass caps
- route-local action/logit/throttle caps
- route action override

The strict lane may use world-model prediction/dynamics/representation losses
and imagined rollout actor-critic behavior loss.

## Completed Smoke Evidence

Windows clean clone:

```text
C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656
```

Tiny replay source:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_h1_32x32_tiny
```

Smoke command class:

```text
train_stage2_privileged_from_replay.ps1
  -SequenceLength 4
  -BatchSize 2
  -WorldModelSteps 2
  -PolicySteps 2
  -PdfStage1Strict
  -Device cuda
```

Result:

```text
PASS as plumbing smoke only.
Not clean evidence.
Not a checkpoint for official clean rerun.
```

Important manifest facts:

```text
pdf_stage1_strict=true
source=windows_carla_privileged_replay
replay privileged_bev_shapes=[[8,43,32,32]]
world_model bev_height=32, bev_width=32, action_dim=39
imitation_weight=0
world_model_action_aux_weight=0
route/action override/cap/fill weights=0
policy valid_action_ids=true
```

Action probe:

```text
Loaded the strict smoke checkpoint with configs/eval/clean_128_stage2_privileged_32x32.yaml.
Generated a checkpoint-filled 32x32 eval config.
Predicted action15 for all 8 tiny replay steps.
recommended_for_clean_rerun=false.
```

The action15 collapse is expected from an 8-step tiny replay and is not a clean
failure. It only proves that checkpoint loading, 32x32 shape propagation, and
action table plumbing work on Windows CUDA.

Do not use the tiny smoke checkpoint for official clean rerun. Its action probe
sets `recommended_for_clean_rerun=false`, so it cannot support a clean claim.

## Next Gates

1. Collect a real 32x32 privileged replay set from Windows CARLA.
   - Use committed code and `clean_128_stage2_privileged_32x32.yaml`.
   - Record replay manifest shape/schema/hash.
   - Keep `.npz` artifacts out of Git.

2. Train H2 strict PDF-faithful checkpoints from that replay.
   - Use `-PdfStage1Strict`.
   - Do not enable imitation, route-priority, action caps, throttle ceilings,
     action overrides, waypoint auxiliary, or pure-pursuit control.

3. Run checkpoint action/latent probes.
   - Use the 32x32 base config.
   - Require finite logits, valid 39-action outputs, more than one predicted
     action, no reverse collapse, and no previous stuck action8 collapse.
   - Treat dominant-action collapse as a block for official clean rerun unless
     explicitly documented as a PDF-faithful failure result.

4. Record an official Windows CARLA clean result as PDF-explicit core.
   - 128-step single-start first.
   - Then 512-step primary subset.
   - Then 2048-step subset or explicit classified block.
   - Record pass or fail; do not hide a PDF-faithful failure with fill knobs.
   - Do not infer a 512-step claim from 128-step, replay probe, or tiny trainer
     smoke evidence.

5. Only after H2 is recorded, start H3 clean-stability fill variants if needed.
   - H3 may add auxiliary/diagnostic/fill knobs for clean performance.
   - H2 and H3 results must stay separate in result/status docs.
