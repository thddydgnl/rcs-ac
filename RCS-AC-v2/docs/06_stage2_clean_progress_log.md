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
training/probe step. Stage 2 replay training/probe/official clean eval now defaults to `Device=cuda`;
direct Python probe entrypoints default to `--device auto`, which chooses CUDA
when available. After Windows reconnects, verify CUDA first and then check this
candidate:

Update 2026-05-25: the Windows Stage 2 replay training/probe/clean eval wrappers now invoke
`assert_stage2_cuda_available.ps1` before scheduling, probing, or official clean
eval. Any `Device=cuda` or `Device=auto` run fails fast if PyTorch CUDA is
unavailable. Checkpoint action probe also writes the selected device into the
generated clean eval config, so the standard promotion path should leave
`device: cuda` in `clean_stage2_candidate_with_checkpoints.yaml`. Use explicit
`Device=cpu` only for diagnostic fallback runs; do not count CPU fallback runs as
official clean-performance candidates.

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

## CUDA follow-up candidates

After the GPU default update, Windows CUDA readiness passed on RTX 3080 and the
following CARLA-free replay candidates were checked. None is clean PASS evidence.

- `balpow0125_iw5_ssf025_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2720247295`.
- `balpow025_iw5_ssf025_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2132921175`.
- `balpow025_iw5_ssf020_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2982998454`.
- `balpow025_iw4_ssf025_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2148377125`.
- `balpow025_iw5_ssf025_seed20260525_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2411128284`.
- `balpow025_iw5_ssf025_seed20260523_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2689335394`.

Current best offline candidate remains `balpow025_iw5_ssf025_cuda`, close to the
`0.20` moving steer gate. The wrapper now exposes `-Seed`, but two nearby seed
checks moved the steer2 action gate in the wrong direction. The next step should
be a structural action diagnostic or a narrower regularization candidate, not
official 512 clean eval.

Observed steer2 failure pattern:

- `balpow025_iw5_ssf025_cuda` overproduced moving steer by only `0.0132921175`
  above gate.
- Lowering `SteerSequenceFraction` to `0.20`, lowering `ImitationWeight` to `4`,
  or changing seed did not improve the gate.
- The failure is mostly argmax action distribution on steer2 replay while latent
  separability remains healthy. Diagnose policy action logits/replay-label
  agreement before promoting any candidate.

Label diagnostic on `balpow025_iw5_ssf025_cuda` steer2 replay:

- output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\diagnostics\balpow025_iw5_ssf025_cuda_steer2_label_action_probe.json`
- action match fraction: `0.7697063369`
- teacher label moving steer fraction: `0.1298299845`
- predicted moving steer fraction: `0.2132921175`
- predicted moving on label-nonmoving fraction: `0.0896445131`
- predicted nonmoving on label-moving fraction: `0.0061823802`

Interpretation: the near-best candidate is not mainly missing steering labels.
It is over-predicting moving steer on teacher non-moving frames, especially on
the steer2 replay. The next structural fix should penalize predicted moving
steer on non-moving replay labels or add a non-moving/steer agreement diagnostic
to training, then re-run the offline cross replay gate before any 512 clean eval.

False-positive penalty follow-up:

- `balpow025_iw5_ssf025_msfpa05_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2102009274`.
- `balpow025_iw5_ssf025_msfpa10_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected, moving steer fraction `0.2086553323`.
- `balpow025_iw5_ssf025_msfpa20_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected by one action bucket, moving steer fraction
    `0.2009273570`, gate max `0.20`.
- `balpow025_iw5_ssf025_msfpa25_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action still rejected, moving steer fraction `0.2009273570`.

Decision: false-positive-only regularization improved the gate but saturated just
above the threshold. Do not relax the gate and do not run official 512 clean eval
from these candidates. The next implementation adds an optional moving-steer
binary auxiliary loss, default `0.0`, so future CUDA candidates can directly
supervise whether replay frames should be moving-steer while preserving the
existing action imitation loss.

Moving-steer binary follow-up:

- `balpow025_iw5_ssf025_msfpa20_msb05_cuda`:
  - target22 action/latent: pass.
  - target22 action match fraction: `0.9693486590`.
  - target22 moving steer fraction: `0.1149425287`.
  - target22 latent test accuracy: `0.9904761910`.
  - steer2 latent: pass, test accuracy `0.8999999762`, margin over majority
    `0.2168469622`.
  - steer2 action rejected, moving steer fraction `0.2024729521`, gate max
    `0.20`.
  - steer2 predicted moving on label-nonmoving fraction `0.0834621329`;
    predicted nonmoving on label-moving fraction `0.0108191654`.

Decision: binary auxiliary did not clear the argmax action gate. It slightly
worsened the best false-positive-only moving steer fraction, while target22
became strong enough. The next structural fix adds
`moving_steer_nonmoving_margin_weight`, a non-moving-frame margin loss that
directly penalizes moving-steer logits that approach or exceed the best
non-moving logit. Continue with CUDA replay training/probe only; official 512
clean eval remains blocked until `ready_for_clean_eval=true`.

Non-moving margin follow-up:

- `balpow025_iw5_ssf025_msfpa20_msmm05_cuda`:
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action still rejected by one action bucket, moving steer fraction
    `0.2009273570`, gate max `0.20`.
- `balpow025_iw5_ssf025_msfpa20_msmm10_cuda`:
  - target22 action/latent: pass.
  - steer2 action/latent: pass.
  - steer2 moving steer fraction `0.1978361669`.
  - cross replay gate: `ready_for_clean_eval=true`.

Official 512 clean eval for `msmm10`:

- run output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_mixed_target22_steer2_aux1_balpow025_iw5_ssf025_msfpa20_msmm10_cuda_official_clean512`
- source: `windows_carla`
- status: `FAIL`
- done reason: `offroad`
- steps: `257` / `512`
- collision count: `0`
- offroad count: `1`
- route progress: `0.5128205128`
- tail actions: mostly action `16` (`throttle=0.3`, `steer=0.3`), then action
  `19` (`throttle=0.3`, `steer=0.5`) immediately before offroad.

Decision: Stage 2 finally produced an offline-promotable candidate, but the
official CARLA clean gate still fails by late offroad rather than collision. Do
not mark clean PASS and do not start raw/fusion. The next smallest patch is a
configurable CARLA safety action mask using the existing policy runner
`forbidden_action_ids` path, then re-run offline probe and official 512 clean
with action `19` masked. This keeps the experiment in privileged clean recovery
and targets the observed failure tail directly.

Safety mask follow-up:

- Config support added: `carla.forbidden_action_ids` is now merged with the
  existing reverse-action mask in the Stage 2 CARLA policy runner.
- `action19_mask`:
  - offline cross replay gate: pass, `ready_for_clean_eval=true`.
  - official 512 clean: fail by offroad.
  - steps: `257` / `512`.
  - route progress: `0.5128205128`.
  - collision count: `0`.
  - tail action changed from action `19` to repeated action `16`
    (`throttle=0.3`, `steer=0.3`), then offroad.
- `action16_19_mask`:
  - offline cross replay gate: pass, `ready_for_clean_eval=true`.
  - official 512 clean: fail by offroad.
  - steps: `349` / `512`.
  - route progress: `0.7179487179`.
  - collision count: `0`.
  - tail actions mostly action `7` (`throttle=0.3`, `steer=0.0`), then action
    `10` (`throttle=0.3`, `steer=0.1`) immediately before offroad.

Decision: masking high right-steer actions improves route progress but does not
solve the clean gate. The failure moved later on the route, which suggests the
policy lacks supervised coverage for the post-0.7 route geometry rather than
only overusing one action id. The next aligned step is to collect a new Windows
privileged replay from the later/offroad-prone clean segment, mix it with the
target22 and steer2 replays, then retrain/probe before another official 512
run. Do not keep masking progressively weaker steering actions without a new
replay signal.
