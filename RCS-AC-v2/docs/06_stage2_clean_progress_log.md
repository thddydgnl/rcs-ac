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

### mixed target22 + steer2, ssf015

- Training output:
  - requested:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015`
  - active rerun:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015_rerun_20260525T071329`
  - `SteerSequenceFraction=0.15`
  - `ImitationWeight=5.0`
  - `ImitationBalancePower=0.5`
  - `WorldModelActionAuxWeight=1.0`
- Training completed:
  - `TRAIN_EXIT_CODE=0`
  - checkpoint manifest sha256:
    `33a3598409240770b7e4687a1177ab64689e277b96fecb6bd239f331d0a2a3db`
- Cross replay gate evidence received before Windows went offline:
  - target22 action probe passed:
    - moving steer fraction: `0.1187739464`
    - dominant action fraction: `0.5019157088`
    - recommended: `true`
  - target22 latent probe passed:
    - test accuracy: `0.9904761910`
    - majority: `0.7931034483`
    - margin: `0.1973727428`
  - steer2 action probe rejected:
    - moving steer fraction: `0.3323029366`, gate max `0.20`.
    - dominant action fraction: `0.3786707883`
    - recommended: `false`
- The SSH session dropped while steer2 latent probe was running, so the final
  gate JSON status was not observed. The action probe alone already rejects this
  candidate; do not run official 512 clean eval from ssf015.
- Decision: reject. Lowering steer oversampling from `0.25` to `0.15` did not
  reduce steer2 over-steer; it made it worse. Next offline candidate should keep
  mixed replay but reduce class-balance strength instead of only changing
  `SteerSequenceFraction`.

### mixed target22 + steer2, balpow025 + ssf025

- Training output:
  - `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow025_iw5_ssf025`
  - `SteerSequenceFraction=0.25`
  - `ImitationWeight=5.0`
  - `ImitationBalancePower=0.25`
  - `WorldModelActionAuxWeight=1.0`
- Training completed:
  - `TRAIN_EXIT_CODE=0`
  - checkpoint manifest sha256:
    `bc1628f2d7a2ad8ff5057f6ef17a1e35759d5276ff001942c6113d4b1869ab2f`
- Cross replay gate:
  - target22 action/latent probes passed:
    - action moving steer fraction: `0.1149425287`
    - action dominant fraction: `0.7662835249`
    - latent test accuracy: `0.9904761910`
    - latent margin: `0.1973727428`
  - steer2 latent probe passed:
    - test accuracy: `0.8999999762`
    - majority: `0.6831530139`
    - margin: `0.2168469622`
  - steer2 action probe rejected:
    - moving steer fraction: `0.2132921175`, gate max `0.20`.
    - dominant action fraction: `0.6862442040`
    - recommended: `false`
- Decision: close but reject, do not run official 512 clean eval. Reducing
  class-balance power from `0.5` to `0.25` improved steer2 action balance from
  `0.2225656878` to `0.2132921175`, but not enough.

## Active restart point

The next candidate keeps `SteerSequenceFraction=0.25`, lowers class-balance power
again from `0.25` to `0.125`, and was already launched as a detached Windows
task before Windows went offline. This is still a CARLA-free replay
training/probe step. Stage 2 replay training/probe now defaults to `Device=cuda`;
after Windows reconnects, verify CUDA first and then check this candidate:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\check_stage2_windows_dependencies.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -RequireCuda

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow0125_iw5_ssf025 `
  -CrossProbeOutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_mixed_target22_steer2_aux1_balpow0125_iw5_ssf025_cross_probe `
  -TaskName RCSV2Stage2MixedT22Steer2Bal0125Ssf025 `
  -SteerSequenceFraction 0.25 `
  -ImitationBalancePower 0.125 `
  -Device cuda `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady
```

Promote to official 512 clean eval only if `stage2_candidate_cross_replay_gate.json`
has `ready_for_clean_eval=true`.
