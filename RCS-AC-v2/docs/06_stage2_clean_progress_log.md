# Stage 2 Clean Progress Log

This log records only offline candidate decisions and restart points. It is not
clean PASS evidence.

## Current gate state

- Stage 0 contract preflight: PASS.
- Stage 1 privileged world model smoke: PASS.
- Stage 2 privileged clean:
  - official 128 clean: PASS with fast35 checkpoint.
  - official 512 clean: failed by collision.

## Rejected candidates

### target24 replay candidate

- Primary target24 action/latent probes passed.
- Cross replay on steer2 rejected:
  - moving steer fraction: `0.3292117465`, gate max `0.20`.
  - latent margin: `0.116846998`, gate min `0.15`.
- Decision: reject, try target22 fallback.

### target22 replay candidate

- Target22 replay action/latent probes passed.
- Cross replay on steer2 rejected:
  - action probe recommended: `false`.
  - moving steer fraction: `0.3292117465`, gate max `0.20`.
  - latent margin: `0.0937700465`, gate min `0.15`.
- Decision: reject, do not run official 512 clean eval.

### mixed target22 + steer2, ssf025

- Mixed replay manifest:
  - replay dir:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_mixed_target22_steer2`
  - episodes: `4`
  - total steps: `1169`
  - git commit: `083d23b70fba72301f4073d1994a8ae9e49df73c`
- Training output:
  - `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5`
  - `SteerSequenceFraction=0.25`
  - `ImitationWeight=5.0`
  - `ImitationBalancePower=0.5`
  - `WorldModelActionAuxWeight=1.0`
- Cross replay gate:
  - target22 action/latent probes passed.
  - steer2 latent probe passed:
    - test accuracy: `0.8923076987`
    - majority: `0.6831530139`
    - margin: `0.2091546848`
  - steer2 action probe rejected:
    - moving steer fraction: `0.2225656878`, gate max `0.20`.
    - dominant action fraction: `0.6414219474`.
- Decision: close but reject, do not run official 512 clean eval.

## Active restart point

The next candidate lowers steer oversampling while keeping the mixed replay and
class balancing:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\start_stage2_replay_training_detached.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -ReplayPath RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_mixed_target22_steer2 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015 `
  -SequenceLength 16 `
  -BatchSize 8 `
  -WorldModelSteps 1500 `
  -PolicySteps 1500 `
  -LearningRate 0.0003 `
  -ImitationWeight 5.0 `
  -BehaviorWeight 0.0 `
  -ImitationClassBalance `
  -ImitationBalancePower 0.5 `
  -SteerSequenceFraction 0.15 `
  -WorldModelActionAuxWeight 1.0 `
  -HiddenDim 128 `
  -Device cpu `
  -TaskName RCSV2Stage2MixedT22Steer2Ssf015
```

After Windows reconnects, use the mixed wrapper first. It checks the detached
training state and only probes when the checkpoint manifest is ready:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015 `
  -CrossProbeOutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015_cross_probe `
  -TaskName RCSV2Stage2MixedT22Steer2Ssf015 `
  -SteerSequenceFraction 0.15 `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady
```

If `TRAIN_READY_FOR_PROBE=True`, run cross replay probe against original target22
and original steer2 manifests. Promote to official 512 clean eval only if
`stage2_candidate_cross_replay_gate.json` has `ready_for_clean_eval=true`.
