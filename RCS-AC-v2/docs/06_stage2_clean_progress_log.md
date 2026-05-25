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

### PDF Stage I strict actor-critic, 200/200

- Commit: `f9a7f44`.
- Training output:
  - `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\f9a7f44_pdf_stage1_strict_actorcritic`
  - `PdfStage1Strict=true`
  - `WorldModelSteps=200`
  - `PolicySteps=200`
  - `ImitationWeight=0.0`
  - no route override/action cap/throttle ceiling/component sampling diagnostics.
- Cross replay gate:
  - target22 action probe rejected:
    - action collapse: `{24: 522}`
    - dominant action fraction: `1.0`
    - moving steer fraction: `1.0`
  - target22 latent probe rejected:
    - test accuracy: `0.7619047761`
    - majority: `0.7931034483`
    - margin: `-0.0311986722`
  - steer2 action probe rejected:
    - action collapse: `{24: 647}`
    - dominant action fraction: `1.0`
    - moving steer fraction: `1.0`
  - steer2 latent probe rejected:
    - test accuracy: `0.6692307591`
    - majority: `0.6831530139`
    - margin: `-0.0139222548`
- Decision: reject, do not run official 512 clean eval. The strict path worked
  mechanically, but this short smoke was undertrained and collapsed.

### PDF Stage I strict actor-critic, wm2000/pi1000

- Commit: `f9a7f44`.
- Training output:
  - `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\f9a7f44_pdf_stage1_strict_actorcritic_wm2000_pi1000`
  - `PdfStage1Strict=true`
  - `WorldModelSteps=2000`
  - `PolicySteps=1000`
  - `ImitationWeight=0.0`
  - no route override/action cap/throttle ceiling/component sampling diagnostics.
- Cross replay gate:
  - target22 latent probe passed:
    - test accuracy: `0.9809523821`
    - majority: `0.7931034483`
    - margin: `0.1878489338`
  - target22 action probe rejected:
    - action collapse: `{13: 522}`
    - dominant action fraction: `1.0`
    - moving steer fraction: `1.0`
  - steer2 latent probe still below margin:
    - test accuracy: `0.8153846264`
    - majority: `0.6831530139`
    - margin: `0.1322316125`, gate min `0.15`
  - steer2 action probe rejected:
    - action collapse: `{13: 647}`
    - dominant action fraction: `1.0`
    - moving steer fraction: `1.0`
- Decision: reject, do not run official 512 clean eval. More world-model
  training improved latent separability, but the actor-critic policy still
  exploits one high-reward steering action. Next PDF-aligned patch should tune
  actor-critic entropy/bootstrap/objective settings, not add route-local losses.

### PDF Stage I strict actor-critic, entropy 0.05

- Commit: `22a374f`.
- Training output:
  - `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\22a374f_pdf_stage1_strict_entropy005_wm2000_pi1000`
  - `PdfStage1Strict=true`
  - `WorldModelSteps=2000`
  - `PolicySteps=1000`
  - `BehaviorEntropyWeight=0.05`
  - `ImitationWeight=0.0`
  - no route override/action cap/throttle ceiling/component sampling diagnostics.
- Cross replay gate:
  - target22 latent probe passed:
    - test accuracy: `0.9809523821`
    - majority: `0.7931034483`
    - margin: `0.1878489338`
  - target22 action probe rejected:
    - action collapse: `{7: 522}`
    - dominant action fraction: `1.0`
    - moving steer fraction: `0.0`
  - steer2 latent probe still below margin:
    - test accuracy: `0.8153846264`
    - majority: `0.6831530139`
    - margin: `0.1322316125`, gate min `0.15`
  - steer2 action probe rejected:
    - action collapse: `{7: 647}`
    - dominant action fraction: `1.0`
    - moving steer fraction: `0.0`
- Decision: reject, do not run official 512 clean eval. Entropy changed the
  collapsed action from `13` to `7` but did not produce a state-dependent policy.
  The next strict PDF Stage I intervention should inspect imagined reward/action
  returns and policy logit calibration, not official clean eval.

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
direct Python training/probe entrypoints also default to `--device cuda`, so
they fail instead of silently falling back to CPU. After Windows reconnects,
verify CUDA first and then check this candidate:

Update 2026-05-25: the Windows Stage 2 replay training/probe/clean eval wrappers now invoke
`assert_stage2_cuda_available.ps1` before scheduling, probing, or official clean
eval. Any standard `Device=cuda` or direct `--device cuda` run fails fast if PyTorch CUDA is
unavailable. Checkpoint action probe also writes the selected device into the
generated clean eval config, so the standard promotion path should leave
`device: cuda` in `clean_stage2_candidate_with_checkpoints.yaml`. Use explicit
`Device=cpu` or `--device cpu` only for diagnostic fallback runs; do not count CPU fallback runs as
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

Late replay + reproducible mask follow-up:

- `5fa68f0_late_route050_min8` replay:
  - collected with `record_after_route_progress=0.50`, `min_recorded_steps=8`.
  - 4 episodes, 44 total recorded steps.
  - Attempts at `record_after_route_progress=0.65` produced 0 recorded steps;
    `0.50` produced only 11 recorded steps per episode. This means the current
    hand-coded replay collector does not reliably reach the post-0.7 failure
    segment on its own.
- `5fa68f0_mixed_t22_steer2_late050_msmm10_cuda`:
  - CUDA training passed checkpoint validation.
  - target22 action/latent: pass.
  - steer2 latent: pass.
  - steer2 action rejected by one action bucket, moving steer fraction
    `0.2009273570`, gate max `0.20`.
- `2113360_mixed_t22_steer2_late050_msmm125_cuda`:
  - CUDA training passed checkpoint validation.
  - Increasing `moving_steer_nonmoving_margin_weight` from `1.0` to `1.25`
    did not change the argmax action distribution enough.
  - steer2 action rejected again at moving steer fraction `0.2009273570`.
- `mask10_16_19` cross-probe on the late050 mixed checkpoint:
  - `probe_stage2_candidate_cross_replays.ps1` now supports reproducible
    `-ForbiddenActionIds "10,16,19"` mask configs rather than hand-edited YAML.
  - target22 action/latent: pass.
  - steer2 action/latent: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
- official 512 clean eval for `mask10_16_19`:
  - run output:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\80af7bc_mixed_t22_steer2_late050_msmm125_mask10_16_19_official_clean512`
  - source: `windows_carla`
  - status: `FAIL`
  - done reason: `offroad`
  - steps: `351` / `512`
  - collision count: `0`
  - offroad count: `1`
  - route progress: `0.7179487179`
  - tail actions: repeated action `7` (`throttle=0.3`, `steer=0.0`) through
    the final offroad step.

Decision: the late replay and action mask moved the policy away from high
right-steer/offroad, but the clean failure remains at the same post-0.7 route
progress and is now a straight-line offroad. Do not keep adding more steering
masks. The next structural fix is a policy-warmup late replay collector: use the
current gated clean candidate to drive into the post-0.7 segment, then record
that segment with teacher/privileged labels so Stage 2 can learn the missing
late-turn behavior before another official clean run.

Policy-warmup late replay follow-up:

- `a2f5145_policy_warmup_route068_min8` replay:
  - collected with warmup policy driving until `record_after_route_progress=0.68`.
  - 4 episodes, 90 recorded steps.
  - retry/skip support is now in the collector, but this run needed no failed
    attempts.
- Full late replay mixed into target22/steer2 caused offline collapse:
  - `late068warm_msmm125_cuda`: target22 and steer2 action probes both rejected,
    with dominant action `7` above gate.
  - `late068warm_msmm10_cuda`: still rejected, target22 moving steer improved
    but action `7` remained dominant.
- Single weak late replay mixed into target22/steer2 shifted collapse:
  - `late068warm1_msmm10_cuda`: steer2 passed, but target22 rejected with
    dominant action `15`.
  - seed `20260525` worsened target22 action `15` dominance, so do not run
    official clean from that seed.

Decision: the late segment is useful but too strong when copied into the mixed
replay at file-count weight. The next patch adds component sample weights to
Stage 2 replay training so `target22`, `steer2`, and `late068warm1` can be
mixed at controlled training probabilities while retaining the same source
artifacts and cross-replay gate. This is the next CUDA-only offline candidate
before any new official CARLA clean eval:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -PrimaryReplayDir RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_7f87656_target22_failure_window `
  -PrimaryReplayName target22 `
  -SecondaryReplayDir RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6e74d9e_long512_t25_steer2_probe `
  -SecondaryReplayName steer2 `
  -TertiaryReplayDir RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_8e4c45d_policy_warmup_route068_probe `
  -TertiaryReplayName late068warm1 `
  -MixedReplayDir RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_5ae274f_mixed_target22_steer2_late068warm1_weighted `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_5ae274f_mixed_t22_steer2_late068warm1_w025_msmm10_cuda `
  -CrossProbeOutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5ae274f_mixed_t22_steer2_late068warm1_w025_msmm10_cuda_cross_probe `
  -TaskName RCSV2Stage2MixedT22Steer2Late068Warm1W025Msmm10 `
  -Seed 20260524 `
  -SteerSequenceFraction 0.25 `
  -ImitationBalancePower 0.25 `
  -ImitationWeight 5.0 `
  -MovingSteerFalsePositiveWeight 2.0 `
  -MovingSteerBinaryWeight 0.0 `
  -MovingSteerNonmovingMarginWeight 1.0 `
  -MovingSteerNonmovingMargin 0.25 `
  -ComponentSampleWeights "target22=1,steer2=1,late068warm1=0.25" `
  -Device cuda `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady
```

Promote only if the generated `stage2_candidate_cross_replay_gate.json` says
`ready_for_clean_eval=true`; otherwise continue offline component-weight tuning.

Weighted late/collision replay results:

- `fc0d0ad_mixed_t22_steer2_late068warm1_w025_msmm10_cuda`:
  - component sample weights: `target22=1,steer2=1,late068warm1=0.25`.
  - CUDA training manifest/checkpoint validation: pass.
  - target22 action/latent: pass.
  - steer2 action/latent: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\fc0d0ad_mixed_t22_steer2_late068warm1_w025_msmm10_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `304` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
- `fc0d0ad_policy_warmup_route062_collision_min8` replay:
  - collected after the weighted-late collision failure, using the weighted-late
    candidate as warmup policy.
  - `record_after_route_progress=0.62`, `min_recorded_steps=8`.
  - 4 episodes, episode lengths `[19,19,19,19]`, total steps `76`.
  - failed attempts: `0`.
- `fc0d0ad_mixed_t22_steer2_collision062_w025_msmm10_cuda`:
  - component sample weights: `target22=1,steer2=1,collision062=0.25`.
  - CUDA training manifest/checkpoint validation: pass.
  - target22 action/latent: pass.
  - steer2 action/latent: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\fc0d0ad_mixed_t22_steer2_collision062_w025_msmm10_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `offroad`.
    - steps: `357` / `512`.
    - route progress: `0.7179487179`.
    - collision count: `0`, offroad count: `1`.
    - tail actions include action `12`/`15` throttle `0.7`, then action `7`
      at final offroad.

Decision: component weighting fixed the offline collapse and produced two
officially promotable candidates. The route062 collision replay removed the
collision and restored the previous post-0.7 offroad failure. This is progress,
but still not clean PASS. Do not start raw/fusion. The next aligned step is to
collect another policy-warmup replay from the current collision062 candidate at
`record_after_route_progress=0.70`, because the remaining failure is again the
post-0.7 offroad segment.

Route070/offroad and route064/collision balancing results:

- `338d8d5_policy_warmup_route070_offroad_min8` replay:
  - collected with the current best collision062 candidate as warmup policy.
  - `record_after_route_progress=0.70`, `min_recorded_steps=8`.
  - 4 episodes, episode lengths `[10,11,18,12]`, total steps `51`.
  - failed attempts: `0`.
- `338d8d5_mixed_t22_steer2_offroad070_w025_msmm10_cuda`:
  - component sample weights: `target22=1,steer2=1,offroad070=0.25`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `183` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
- `338d8d5_mixed_t22_steer2_offroad070_w010_msmm10_cuda`:
  - component sample weights: `target22=1,steer2=1,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `305` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
- `71b3e51_mixed_t22_steer2_collision062_w025_offroad070_w010_msmm10_cuda`:
  - four-component replay support was added so target22, steer2,
    collision062, and offroad070 can be mixed without rebuilding source
    replays.
  - mixed replay:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_71b3e51_mixed_t22_steer2_collision062_offroad070_weighted`
  - mixed replay has 12 episodes and 1296 total steps.
  - component sample weights:
    `target22=1,steer2=1,collision062=0.25,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\71b3e51_mixed_t22_steer2_collision062_w025_offroad070_w010_msmm10_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `307` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - tail was mostly action `7` at throttle `0.3`.
- `71b3e51_policy_warmup_route064_collision_min8` replay:
  - collected from the four-component collision failure with the same policy as
    warmup.
  - `record_after_route_progress=0.64`, `min_recorded_steps=8`.
  - 4 episodes, episode lengths `[20,20,20,20]`, total steps `80`.
  - failed attempts: `0`.
- `71b3e51_mixed_t22_steer2_collision064_w025_offroad070_w010_msmm10_cuda`:
  - component sample weights:
    `target22=1,steer2=1,collision064=0.25,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=false`.
  - rejection cause:
    - target22 dominant action `7` fraction `0.9348659004`.
    - steer2 dominant action `7` fraction `0.9134466770`.
  - official clean was intentionally skipped because the offline gate rejected
    the candidate.
- `71b3e51_mixed_t22_steer2_collision064_w010_offroad070_w010_msmm10_cuda`:
  - component sample weights:
    `target22=1,steer2=1,collision064=0.10,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"12":6,"15":412,"4":22,"7":82}`.
    - dominant action fraction `0.7892720307`.
    - moving steer fraction `0.0536398467`.
  - steer2 action probe:
    - action counts `{"1":3,"15":305,"4":60,"7":279}`.
    - dominant action fraction `0.4714064915`.
    - moving steer fraction `0.0973724884`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\71b3e51_mixed_t22_steer2_collision064_w010_offroad070_w010_msmm10_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `offroad`.
    - steps: `148` / `512`.
    - route progress: `0.6410256410`.
    - collision count: `0`, offroad count: `1`.
    - tail was action `7` at throttle `0.3`.

Decision: route070 replay helped expose that the current Stage 2 policy is
balanced on a narrow failure band rather than clean behavior. A strong
collision064 weight collapses back to action `7`; a weak collision064/offroad070
mix passes offline but shifts official failure to earlier offroad at
`route_progress=0.6410`. Keep Stage 2 active and do not start raw/fusion. The
next aligned move is to collect/inspect a route064 offroad correction replay
from the weak candidate or add a smaller low-throttle/steer regularizer before
another CUDA candidate, because pure replay weighting is oscillating between
collision at `0.6667` and offroad at `0.6410-0.7179`.

Action mass cap regularizer results:

- Commit `403e927` added a Stage 2 replay-training action mass cap regularizer.
  - This does not change the Raw2Drive action table, CARLA adapter, or clean
    gate contract.
  - It adds optional training arguments:
    `--action-mass-cap-weight` and repeatable `--action-mass-cap`.
  - Windows wrappers expose `-ActionMassCapWeight` and `-ActionMassCaps`.
  - Local verification:
    `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py RCS-AC-v2/tests/test_stage2_carla_adapter.py`
    passed: `80 passed in 2.82s`.
- `403e927_mixed_t22_steer2_collision064_w010_offroad070_w010_cap7_70_cap15_55_cuda`:
  - component sample weights:
    `target22=1,steer2=1,collision064=0.10,offroad070=0.10`.
  - action mass caps: `7=0.70,15=0.55`.
  - action mass cap weight: `0.5`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - offline action probes were effectively identical to the uncapped weak
    candidate:
    - target22 action counts `{"12":6,"15":412,"4":22,"7":82}`.
    - steer2 action counts `{"1":3,"15":305,"4":60,"7":279}`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\403e927_mixed_t22_steer2_collision064_w010_offroad070_w010_cap7_70_cap15_55_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `offroad`.
    - steps: `148` / `512`.
    - route progress: `0.6410256410`.
    - collision count: `0`, offroad count: `1`.
- `403e927_mixed_t22_steer2_collision064_w010_offroad070_w010_cap7_60_cap15_35_w5_cuda`:
  - component sample weights:
    `target22=1,steer2=1,collision064=0.10,offroad070=0.10`.
  - action mass caps: `7=0.60,15=0.35`.
  - action mass cap weight: `5.0`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"12":6,"15":436,"4":22,"7":58}`.
    - dominant action fraction `0.8352490421`.
    - moving steer fraction `0.0536398467`.
  - steer2 action probe:
    - action counts `{"1":3,"15":401,"4":60,"7":183}`.
    - dominant action fraction `0.6197836167`.
    - moving steer fraction `0.0973724884`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\403e927_mixed_t22_steer2_collision064_w010_offroad070_w010_cap7_60_cap15_35_w5_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `134` / `512`.
    - route progress: `0.5897435897`.
    - collision count: `1`, offroad count: `0`.
    - tail shows action `15` high-throttle straight dominance before recovery
      attempts and final collision.

Decision: action mass caps are technically wired and verified, but they are not
the right clean-performance lever by themselves. A weak cap does not move the
policy; a stronger cap mostly shifts action `7` mass into action `15`, causing
earlier collision. The next Stage 2 step should not be another plain replay
weight/cap sweep. Prefer a route-window/speed-aware objective or route064
failure-window replay inspection that separates lateral correction from
longitudinal throttle, then rerun CUDA offline gates before any official clean
eval. Raw sensor stream and fusion ablation remain blocked.

Route/speed action cap results:

- Commit `d79cf4d` added a route/speed-conditioned Stage 2 action mass cap.
  - The regularizer only applies inside a configured route-progress window and
    above a configured normalized speed threshold.
  - This keeps the objective aligned with the observed failure band instead of
    globally suppressing useful actions.
  - Optional training arguments:
    `--route-speed-action-mass-cap-weight`,
    `--route-speed-action-ids`, `--route-speed-action-mass-cap`,
    `--route-speed-min-speed-norm`, `--route-speed-min-route-progress`, and
    `--route-speed-max-route-progress`.
  - Windows wrappers expose the same knobs.
  - Local verification:
    `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py RCS-AC-v2/tests/test_stage2_carla_adapter.py`
    passed: `81 passed in 3.20s`.
- Commits `b20d1e8`, `a6cc767`, and `f4c30c2` fixed Windows optional argument
  handling before the CUDA candidate could run end to end.
  - `b20d1e8`: mixed-candidate wrapper now uses an argument array and omits
    empty optional list arguments.
  - `a6cc767`: detached wrapper string quoting accepts empty strings.
  - `f4c30c2`: detached train command omits empty optional train arguments
    rather than passing `-ActionMassCaps ''`.
  - Local verification after the fixes:
    `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py RCS-AC-v2/tests/test_stage2_carla_adapter.py`
    passed: `81 passed in 2.38s`.
- `d79cf4d_route055075_speed08_straightcap_cuda`:
  - Windows commit used for the successful run: `f4c30c2`.
  - CUDA guard: pass on `NVIDIA GeForce RTX 3080`.
  - active checkpoint output:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_d79cf4d_route055075_speed08_straightcap_cuda_rerun_20260525T203750`.
  - component sample weights:
    `target22=1,steer2=1,collision064=0.10,offroad070=0.10`.
  - route/speed cap:
    - action ids: `7,15,37`.
    - mass cap: `0.35`.
    - weight: `5.0`.
    - speed threshold: `0.8`.
    - route window: `0.55-0.75`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"12":6,"15":412,"4":22,"7":82}`.
    - dominant action fraction `0.7892720307`.
    - moving steer fraction `0.0536398467`.
    - latent test accuracy `0.9904761910`.
  - steer2 action probe:
    - action counts `{"1":3,"15":305,"4":60,"7":279}`.
    - dominant action fraction `0.4714064915`.
    - moving steer fraction `0.0973724884`.
    - latent test accuracy `0.8999999762`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\d79cf4d_route055075_speed08_straightcap_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `135` / `512`.
    - route progress: `0.5897435897`.
    - collision count: `1`, offroad count: `0`.
    - tail changed from high-throttle action `15` to several action `4` and
      action `7` steps, but still collided at the same route band as the strong
      global cap candidate.

Decision: the route/speed cap is implemented and runs on the intended GPU path,
but this exact cap does not recover clean performance. It avoids the earlier
offroad failure mode of the weak replay mix, while regressing to the collision
mode around `route_progress=0.5897`. The next Stage 2 move should target the
collision band directly with a route-window correction replay or a localized
longitudinal throttle penalty/relabel around route `0.56-0.60`; another broad
cap sweep is unlikely to solve the oscillation. Stage 3 raw sensor stream and
fusion ablation remain blocked until privileged clean passes.

Route-window throttle ceiling results:

- Commit `3c234dd` added a localized expected-throttle ceiling loss for Stage 2
  replay training.
  - The loss uses the Raw2Drive action table throttle values and only activates
    inside a configured speed/route-progress window.
  - This was intended to suppress high-throttle straight dominance in the
    collision band without globally forbidding recovery actions.
  - Local verification:
    `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py RCS-AC-v2/tests/test_stage2_carla_adapter.py`
    passed: `82 passed in 3.25s`.
- `3c234dd_route056060_throttle035_speedcap_cuda`:
  - CUDA guard: pass on `NVIDIA GeForce RTX 3080`.
  - checkpoint output:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_3c234dd_route056060_throttle035_speedcap_cuda`.
  - component sample weights:
    `target22=1,steer2=1,collision064=0.10,offroad070=0.10`.
  - route/speed cap:
    - action ids: `7,15,37`.
    - mass cap: `0.35`.
    - weight: `5.0`.
    - speed threshold: `0.8`.
    - route window: `0.55-0.75`.
  - route throttle ceiling:
    - ceiling: `0.35`.
    - weight: `5.0`.
    - speed threshold: `0.0`.
    - route window: `0.56-0.60`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"12":6,"15":412,"4":22,"7":82}`.
    - dominant action fraction `0.7892720307`.
    - moving steer fraction `0.0536398467`.
  - steer2 action probe:
    - action counts `{"1":3,"15":379,"4":60,"7":205}`.
    - dominant action fraction `0.5857805255`.
    - moving steer fraction `0.0973724884`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\3c234dd_route056060_throttle035_speedcap_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `134` / `512`.
    - route progress: `0.5897435897`.
    - collision count: `1`, offroad count: `0`.
    - tail shifted toward action `4` and action `7` in the collision band, but
      the vehicle still collided at the same route progress.

Decision: a localized throttle ceiling changes the tail behavior slightly but
does not solve the collision band. The policy likely needs positive corrective
supervision in that band rather than another negative regularizer. The next
Stage 2-aligned move is to collect a short post-route `0.56` privileged
correction replay from the current failure window, or to inspect the replay
state/action labels around route `0.56-0.60` and add a route-local label
override. Raw sensor stream and fusion ablation remain blocked.

Correction replay results:

- Route-band inspection after `3c234dd`:
  - target22 replay has no labels in route `0.54-0.62`.
  - collision064/offroad070 replays also have no labels in route `0.54-0.62`.
  - steer2 has labels in this band, mostly action `7` and action `4`.
  - official `3c234dd` failed while executing action `4/7` in this band, so
    merely pushing existing labels harder is insufficient.
- `windows_privileged_3c234dd_route056_collision_window_t12_min5`:
  - collected with `3c234dd` policy warmup to route `0.56`.
  - `target_speed_kmh=12`, `record_after_route_progress=0.56`.
  - initial `min_recorded_steps=8` collection failed 8/8 attempts because every
    attempt collided after only 5 recorded steps.
  - rerun with `min_recorded_steps=5` succeeded.
  - 4 episodes, episode lengths `[5,5,5,5]`, total steps `20`.
  - labels: action `2` x16 and action `34` x4, both zero-throttle steering.
- `9be6b2f_corr056_w4_speedcap_cuda`:
  - mixed replay components:
    `target22`, `steer2`, `correction056`, `offroad070`.
  - component weights: `target22=1,steer2=1,correction056=4,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"10":18,"15":64,"4":22,"7":418}`.
    - dominant action fraction `0.8007662835`.
    - moving steer fraction `0.0766283525`.
  - steer2 action probe:
    - action counts `{"10":10,"12":3,"15":68,"4":68,"7":498}`.
    - dominant action fraction `0.7697063369`.
    - moving steer fraction `0.1251931994`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\9be6b2f_corr056_w4_speedcap_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `305` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
  - interpretation:
    - This is meaningful progress: the candidate cleared the previous
      `route_progress=0.5897` collision band and moved the failure to
      `route_progress=0.6667`.
- `windows_privileged_9be6b2f_route064_collision_window_t12_min8`:
  - collected with `9be6b2f_corr056_w4_speedcap_cuda` policy warmup to route
    `0.64`.
  - `target_speed_kmh=12`, `record_after_route_progress=0.64`.
  - 4 episodes, episode lengths `[67,62,22,68]`, total steps `219`.
  - labels are mostly low/zero throttle and left correction:
    action `34` x57, `33` x31, `36` x28, `1` x26, `30` x21, `2` x17,
    `31` x15, `4` x12, `28` x7, `10` x5.
- `9be6b2f_corr_combined_w3_speedcap_cuda`:
  - combined `correction056 + correction064` into one correction component.
  - component weights: `target22=1,steer2=1,correction=3,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=false`.
  - rejection cause:
    - target22 dominant action `7` fraction `0.9501915709`.
    - steer2 dominant action `7` fraction `0.9582689335`.
  - official clean intentionally skipped.
- `9be6b2f_corr_combined_w1_speedcap_cuda`:
  - component weights: `target22=1,steer2=1,correction=1,offroad070=0.10`.
  - CUDA training/checkpoint validation: pass.
  - cross replay gate: `ready_for_clean_eval=false`.
  - rejection cause:
    - target22 dominant action `7` fraction `0.9578544061`.
    - steer2 dominant action `7` fraction `0.9675425039`.
  - official clean intentionally skipped.

Decision: route056 positive correction is useful and moved the clean failure
from route `0.5897` to `0.6667`. Route064 correction with `target_speed_kmh=12`
is too strong as a generic mixed component and collapses offline action
diversity toward action `7`. The next Stage 2 step should recollect route064
with a higher target speed, likely `18 km/h`, so the correction carries
stronger steering labels without teaching a broad slow/straight bias. Raw sensor
stream and fusion ablation remain blocked.

Higher-speed route064 correction retry:

- `windows_privileged_9be6b2f_route064_collision_window_t18_min8`:
  - collected with `9be6b2f_corr056_w4_speedcap_cuda` policy warmup to route
    `0.64`.
  - `target_speed_kmh=18`, `record_after_route_progress=0.64`.
  - 4 episodes, episode lengths `[20,20,19,19]`, total steps `78`.
  - labels: action `7` x64, action `10` x13, action `13` x1.
  - interpretation:
    - Compared with the `target_speed_kmh=12` collection, this replay avoids the
      heavy zero-throttle steering bias.
    - It is still mostly action `7`, so it is not a strong standalone steering
      correction source.
- `6cdb5ca_corr056w4_corr064t18w1_cuda`:
  - mixed replay components:
    `target22`, `steer2`, `correction056`, `correction064_t18`.
  - component weights:
    `target22=1,steer2=1,correction056=4,correction064_t18=1`.
  - mixed replay:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_6cdb5ca_mixed_t22_steer2_corr056_corr064t18_weighted`.
  - 12 episodes, total steps `1267`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=false`.
  - rejection cause:
    - target22 dominant action `7` fraction `0.9310344828`;
      moving steer fraction `0.0689655172`.
    - steer2 dominant action `7` fraction `0.9041731066`;
      moving steer fraction `0.0958268934`.
  - official clean intentionally skipped because the offline gate did not pass.

Decision: simply mixing the route064 correction replay, even at
`target_speed_kmh=18`, does not solve the Stage 2 collapse. The next Stage 2
move should stop treating route064 as another global replay-weight component and
instead implement a route-local corrective target or label override for the
post-`0.64` collision window. Fusion ablation remains blocked until a privileged
clean pass exists.

Route-local override implementation:

- Added a Stage 2 replay-training option for a route-window corrective action
  target:
  - `route_action_override_weight`
  - `route_action_override_id`
  - `route_action_override_min_speed_norm`
  - `route_action_override_min_route_progress`
  - `route_action_override_max_route_progress`
- The override is implemented as an additional cross-entropy loss only on
  replay samples whose privileged ego state is inside the configured speed/route
  window. It does not rewrite replay artifacts, so replay provenance and clean
  evidence boundaries remain intact.
- Windows wrappers now pass the option through:
  - `train_stage2_privileged_from_replay.ps1`
  - `start_stage2_replay_training_detached.ps1`
  - `run_stage2_mixed_candidate_cycle.ps1`
- Validation:
  - `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py`
    passed: `61 passed`.
  - `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests`
    passed: `139 passed`.

Next candidate should use the existing useful route056 correction plus a
post-`0.64` action override instead of adding more global route064 replay mass.
A conservative first probe is action `10` (`throttle=0.3`, `steer=0.1`) in the
`route_progress=0.64-0.70` window with a moderate override weight, then apply
the existing target22/steer2 offline gate before any official clean eval.

Route-local override candidate results:

- `5f77791_corr056w4_override10_cuda`:
  - route-local override: action `10`, route window `0.64-0.70`, weight `2.0`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=false`.
  - target22 passed, but steer2 failed because moving steer was too high:
    `0.3292117465`.
  - official clean intentionally skipped.
- `5f77791_corr056w4_override10w1_cuda`:
  - same as above with override weight `1.0`.
  - cross replay gate: `ready_for_clean_eval=false`.
  - target22 passed, but steer2 again failed with moving steer `0.3307573416`.
  - official clean intentionally skipped.
  - interpretation: this pair used different surrounding hyperparameters from
    the previous `9be6b2f_corr056_w4_speedcap_cuda` candidate, so the excessive
    steer2 moving-steer rate was not a clean test of override weight alone.
- `5f77791_corr056w4_override10orig_cuda`:
  - restored the previous `9be6b2f_corr056_w4_speedcap_cuda` hyperparameters:
    `imitation_balance_power=0.25`, `steer_sequence_fraction=0.25`,
    `moving_steer_false_positive_weight=2.0`,
    `moving_steer_nonmoving_margin_weight=1.0`.
  - added route-local override: action `10`, route window `0.64-0.70`,
    weight `1.0`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"13":4,"15":110,"16":24,"19":4,"4":16,"7":364}`.
    - dominant action fraction `0.6973180077`.
    - moving steer fraction `0.0919540230`.
  - steer2 action probe:
    - action counts `{"10":59,"15":84,"4":59,"7":445}`.
    - dominant action fraction `0.6877897991`.
    - moving steer fraction `0.1823802164`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5f77791_corr056w4_override10orig_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `297` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `10` dominated from route `0.6410` through
      collision at route `0.6667`.

Decision: Stage 2 is not complete, but it is now more diagnosable. The
route-local override can pass the offline target22/steer2 gate when combined
with the original corr056 hyperparameters, but action `10` is insufficient to
clear the route `0.6667` collision. The next Stage 2 candidate should keep the
same hyperparameters and test a stronger/rightward post-`0.64` target action
or a short action schedule rather than continuing to increase global replay
weights.

Stronger route-local override retry:

- `8477ec6_corr056w4_override13orig_cuda`:
  - same original corr056 hyperparameters as
    `5f77791_corr056w4_override10orig_cuda`.
  - route-local override: action `13` (`throttle=0.3`, `steer=0.2`), route
    window `0.64-0.70`, weight `1.0`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"13":6,"15":220,"16":22,"19":4,"4":16,"7":254}`.
    - dominant action fraction `0.4865900383`.
    - moving steer fraction `0.0919540230`.
  - steer2 action probe:
    - action counts `{"13":62,"15":188,"16":1,"4":59,"7":337}`.
    - dominant action fraction `0.5208655332`.
    - moving steer fraction `0.1885625966`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8477ec6_corr056w4_override13orig_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `165` / `512`.
    - route progress: `0.6153846154`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `13` appeared from route `0.5897`, followed by
      action `16` at route `0.6154` and collision.

Decision: action `13` is too strong when trained with the same `0.64-0.70`
override window. It passes offline gates but generalizes into the earlier
`0.5897-0.6154` band and regresses clean performance. The next candidate should
avoid simply increasing steering magnitude. A better next test is a later,
narrower override window, e.g. action `13` or a staged target only from
`route_progress >= 0.66`, so the policy keeps the known-good approach behavior
from the action `10` candidate and only changes the final collision band.

Late stronger override retry:

- `cb1abf9_corr056w4_override13late66_cuda`:
  - same original corr056 hyperparameters.
  - route-local override: action `13`, route window `0.66-0.70`, weight `1.0`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"13":10,"15":226,"16":10,"19":2,"4":24,"7":250}`.
    - dominant action fraction `0.4789272031`.
    - moving steer fraction `0.0881226054`.
  - steer2 action probe:
    - action counts `{"13":38,"15":188,"16":1,"4":65,"7":355}`.
    - dominant action fraction `0.5486862442`.
    - moving steer fraction `0.1607418856`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\cb1abf9_corr056w4_override13late66_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `177` / `512`.
    - route progress: `0.6410256410`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `13` started at route `0.6154`, then action
      `16` appeared before collision at route `0.6410`.

Decision: narrowing the action `13` override to `0.66-0.70` still leaks into
the earlier route band and regresses clean performance. The next useful test is
not a stronger steer or later window at the same loss strength, but a weaker
action `13` override weight, or a code-level scheduled/masked override that
prevents action `13/16` from being learned before route `0.66`.

Weaker late stronger-override retry:

- `9533880_corr056w4_override13late66w025_cuda`:
  - same original corr056 hyperparameters.
  - route-local override: action `13`, route window `0.66-0.70`,
    weight `0.25`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"10":18,"15":80,"4":22,"7":402}`.
    - dominant action fraction `0.7701149425`.
    - moving steer fraction `0.0766283525`.
  - steer2 action probe:
    - action counts `{"10":10,"12":4,"13":1,"15":88,"4":66,"7":478}`.
    - dominant action fraction `0.7387944359`.
    - moving steer fraction `0.1251931994`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\9533880_corr056w4_override13late66w025_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `307` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: mostly action `7` from route `0.6410` through
      collision at route `0.6667`.

Decision: `action13` override strength is bracketed. Weight `1.0` leaks too
much and causes earlier route `0.6154-0.6410` collisions; weight `0.25` is too
weak and returns to the route `0.6667` straight-line collision. The next Stage 2
move should stop searching steering magnitude alone and test a speed/throttle
reduction after route `0.64`, or add an explicit route-local action schedule
that can keep the approach stable while changing only the final collision band.

Route-local action cap support:

- Added `route_action_mass_cap_*` training parameters to Stage 2 replay
  training and Windows wrappers.
- Purpose: keep approach behavior stable by capping high-steer action mass
  before the intended late correction window, while preserving the existing
  `route_speed_action_mass_cap_*` speed/action cap used by the corr056 baseline.
- Targeted use for the next CUDA candidate:
  - keep the original `9be6b2f_corr056_w4` hyperparameters.
  - keep route speed cap on actions `7,15,37` for route `0.55-0.75`.
  - keep early throttle ceiling for route `0.56-0.60`.
  - add `route_action_mass_cap_ids="13,16"`.
  - add `route_action_mass_cap=0.05-0.10`.
  - apply the cap on route `0.55-0.66`, speed norm `>=0.0`.
  - then retry a late action `13` override only from route `0.66-0.70` with
    moderate weight, e.g. `0.5`.
- Local verification:
  - targeted Stage 2 tests: `61 passed`.
  - full `RCS-AC-v2/tests`: `139 passed`.

Late throttle plus moderate steering overrides:

- `c112325_corr056w4_late_thr020_override10w050_cuda`:
  - same original corr056 hyperparameters.
  - kept early route throttle ceiling:
    - ceiling `0.35`, route `0.56-0.60`, weight `5.0`.
  - kept late route throttle ceiling:
    - ceiling `0.20`, route `0.64-0.70`, weight `6.0`.
  - added route-local override:
    - action `10` (`throttle=0.3`, `steer=0.1`).
    - route window `0.66-0.70`.
    - weight `0.5`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\c112325_corr056w4_late_thr020_override10w050_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `304` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `10` was selected at steps `299-303`, but
      the turn was still insufficient at about `18 km/h`.
- `c112325_corr056w4_late_thr020_override13w050_cuda`:
  - same as above, but route-local override action `13`
    (`throttle=0.3`, `steer=0.2`) with weight `0.5`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\c112325_corr056w4_late_thr020_override13w050_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `306` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `13` appeared near steps `300` and
      `303-305`, but speed stayed near `18 km/h` and the collision remained.
- `c112325_corr056w4_late_thr010_override13w050_cuda`:
  - same original corr056 hyperparameters.
  - tightened late route throttle ceiling:
    - ceiling `0.10`, route `0.64-0.70`, weight `8.0`.
  - kept route-local override action `13`, route `0.66-0.70`, weight `0.5`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\c112325_corr056w4_late_thr010_override13w050_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `307` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `13` appeared at steps `302` and `304-306`,
      but actual speed still stayed near `18 km/h`.

Decision: action `10` and `13` can now be made to execute in the final band,
but they still carry `throttle=0.3`. The Raw2Drive Table 12 action set has no
brake action, so the next PDF-faithful correction should test zero-throttle
steering actions rather than adding non-table braking. The primary candidate is
action `5` (`throttle=0.0`, `steer=0.3`) in the same route `0.66-0.70` window;
action `8` (`throttle=0.0`, `steer=0.6`) is the stronger follow-up if action
`5` reaches the collision band but still understeers.

Zero-throttle steering override in progress:

- `c112325_corr056w4_late_thr020_override5w050_cuda`:
  - first launch failed before training because the SSH/PowerShell invocation
    passed literal quote characters into `ComponentSampleWeights` and
    `RouteSpeedActionIds`.
  - no checkpoint was produced; do not evaluate this failed launch.
- `c112325_corr056w4_late_thr020_override5w050_retry_cuda`:
  - same original corr056 hyperparameters.
  - kept early route throttle ceiling:
    - ceiling `0.35`, route `0.56-0.60`, weight `5.0`.
  - kept late route throttle ceiling:
    - ceiling `0.20`, route `0.64-0.70`, weight `6.0`.
  - added route-local override:
    - action `5` (`throttle=0.0`, `steer=0.3`).
    - route window `0.66-0.70`.
    - weight `0.5`.
  - CUDA guard passed on `NVIDIA GeForce RTX 3080`.
  - detached Windows task:
    - `RCSV2Stage2LateThr020Override5w050Retry`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"15":80,"16":12,"4":20,"5":12,"7":398}`.
    - dominant action fraction `0.7624521073`.
    - moving steer fraction `0.0613026820`.
    - zero-throttle steer fraction `0.0229885057`.
  - steer2 action probe:
    - action counts `{"10":6,"15":84,"4":63,"5":25,"7":469}`.
    - dominant action fraction `0.7248840804`.
    - moving steer fraction `0.1066460587`.
    - zero-throttle steer fraction `0.0386398764`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\c112325_corr056w4_late_thr020_override5w050_retry_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `306` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `5` executed at steps `300-301` and
      `303-305`; speed dropped from about `18 km/h` to `5.67 km/h` at
      collision, but the turn still arrived too late and/or with insufficient
      steering authority.

Decision: zero-throttle steering is the right direction because it reduced
final speed without creating offroad. The next candidate should start earlier
at route `0.64` and use the stronger zero-throttle action `8`
(`throttle=0.0`, `steer=0.6`) instead of action `5`.

- `c112325_corr056w4_late_thr020_override8w050_r064_cuda`:
  - same original corr056 hyperparameters.
  - kept early route throttle ceiling:
    - ceiling `0.35`, route `0.56-0.60`, weight `5.0`.
  - kept late route throttle ceiling:
    - ceiling `0.20`, route `0.64-0.70`, weight `6.0`.
  - added route-local override:
    - action `8` (`throttle=0.0`, `steer=0.6`).
    - route window `0.64-0.70`.
    - weight `0.5`.
  - CUDA guard passed on `NVIDIA GeForce RTX 3080`.
  - detached Windows task:
    - `RCSV2Stage2LateThr020Override8w050R064`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=false`.
  - target22 action probe:
    - action counts `{"10":24,"13":8,"15":80,"16":20,"19":6,"7":384}`.
    - dominant action fraction `0.7356321839`.
    - moving steer fraction `0.1111111111`.
    - zero-throttle steer fraction `0.0`; action `8` did not appear.
  - steer2 action probe:
    - action counts `{"10":87,"15":84,"4":29,"7":398,"8":49}`.
    - dominant action fraction `0.6151468315`.
    - moving steer fraction `0.1792890263`.
    - zero-throttle steer fraction `0.0757341577`.
    - `recommended_for_clean_rerun=false` because the action `8` fraction
      exceeded the previous-stuck guardrail.
  - status: rejected by cross replay gate; no official clean eval.

Decision: action `8` is too strong for the steer2 replay and does not reliably
map to the target22 final band. Continue with action `5`, but move its route
window start from `0.66` to `0.64` so the zero-throttle steering begins before
the final route jump.

- `c112325_corr056w4_late_thr020_override5w050_r064_cuda`:
  - same original corr056 hyperparameters.
  - kept early route throttle ceiling:
    - ceiling `0.35`, route `0.56-0.60`, weight `5.0`.
  - kept late route throttle ceiling:
    - ceiling `0.20`, route `0.64-0.70`, weight `6.0`.
  - added route-local override:
    - action `5` (`throttle=0.0`, `steer=0.3`).
    - route window `0.64-0.70`.
    - weight `0.5`.
  - CUDA guard passed on `NVIDIA GeForce RTX 3080`.
  - detached Windows task:
    - `RCSV2Stage2LateThr020Override5w050R064`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"10":24,"13":8,"15":80,"16":20,"19":6,"7":384}`.
    - dominant action fraction `0.7356321839`.
    - moving steer fraction `0.1111111111`.
    - zero-throttle steer fraction `0.0`; action `5` did not appear in the
      target22 probe.
  - steer2 action probe:
    - action counts `{"10":86,"15":84,"4":29,"5":50,"7":398}`.
    - dominant action fraction `0.6151468315`.
    - moving steer fraction `0.1777434312`.
    - zero-throttle steer fraction `0.0772797527`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\c112325_corr056w4_late_thr020_override5w050_r064_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `148` / `512`.
    - route progress: `0.5384615385`.
    - collision count: `1`, offroad count: `1`.
    - final action band: action `10/13/16/19` leaked far earlier around route
      `0.48-0.54` while speed was still about `30-34 km/h`; the official run
      failed before the intended route `0.64` zero-throttle correction.

Decision: widening the override window to route `0.64` is unsafe because it
induces early right-steer leakage before the known `0.6667` bottleneck. Keep the
late start near `0.66`, but add a second guard against action `10/13/16/19`
leakage before `0.60`, or collect a replay slice closer to the exact collision
state so zero-throttle action `5` can be learned without changing the earlier
approach.

Late throttle-only candidate:

- `0308f26_corr056w4_late_throttle020_cuda`:
  - same original corr056 hyperparameters.
  - no route action override.
  - kept early route throttle ceiling:
    - ceiling `0.35`, route `0.56-0.60`.
  - added late route throttle ceiling:
    - ceiling `0.20`, route `0.64-0.70`.
    - weight `6.0`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"10":18,"15":64,"4":22,"7":418}`.
    - dominant action fraction `0.8007662835`.
    - moving steer fraction `0.0766283525`.
  - steer2 action probe:
    - action counts `{"10":10,"15":64,"4":71,"7":502}`.
    - dominant action fraction `0.7758887172`.
    - moving steer fraction `0.1251931994`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\0308f26_corr056w4_late_throttle020_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `306` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `7` dominated from route `0.6410` through
      collision at route `0.6667`.
    - late speed dropped to around `18 km/h`, compared with about `25 km/h` in
      the cap-plus-override failure.

Decision: late throttle worked as a speed reducer but did not create the final
turn. The next candidate should keep late throttle and add a weak late steering
target, preferably action `13` or `10` only from route `0.66-0.70`, because
throttle-only returns to the straight-line route `0.6667` collision while
stronger steering without enough speed control regressed earlier.

Late throttle plus weak action13:

- `0308f26_corr056w4_late_thr020_override13w025_cuda`:
  - same original corr056 hyperparameters.
  - kept early route throttle ceiling:
    - ceiling `0.35`, route `0.56-0.60`.
  - kept late route throttle ceiling:
    - ceiling `0.20`, route `0.64-0.70`.
    - weight `6.0`.
  - added weak route-local override:
    - action `13`.
    - route window `0.66-0.70`.
    - weight `0.25`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"10":18,"15":80,"4":22,"7":402}`.
    - dominant action fraction `0.7701149425`.
    - moving steer fraction `0.0766283525`.
  - steer2 action probe:
    - action counts `{"10":10,"13":4,"15":84,"4":68,"7":481}`.
    - dominant action fraction `0.7434312210`.
    - moving steer fraction `0.1267387944`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\0308f26_corr056w4_late_thr020_override13w025_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `304` / `512`.
    - route progress: `0.6666666667`.
    - collision count: `1`, offroad count: `0`.
    - final action band: action `7` dominated through the final route band;
      no action `13` was selected in the final metrics tail.

Decision: weak action `13` is too weak even with late throttle; it does not
change the final executed action sequence. The useful bracket is now:

- action `13` weight `0.25`: no effect in official final band.
- action `13` weight `0.5-1.0` without enough masking/scheduling: leaks too
  early and regresses to route `0.6410` or `0.6154` collisions.
- late throttle ceiling: reduces speed from about `25 km/h` to `18 km/h`, but
  still needs a reliable final turn.

Next Stage 2 move should test either action `10` with late throttle, or add a
route-phase schedule that changes the target action only after route progress
has actually crossed `0.666`, because the current route-local CE target still
does not reliably alter the executed action in the exact final collision band.

Decision: this is the next Stage 2 direction because the last two official
clean runs bracketed action `13` strength but did not solve the route `0.64+`
collision. The cap lets the candidate learn a late correction without leaking
action `13/16` into the already fragile approach band.

Route-local cap plus moderate late override:

- `bc47fd5_corr056w4_cap1316_override13w05_cuda`:
  - same original corr056 hyperparameters.
  - added route action mass cap:
    - action ids `13,16`.
    - cap `0.08`.
    - route window `0.55-0.66`, speed norm `>=0.0`.
    - weight `4.0`.
  - kept late route-local override:
    - action `13`.
    - route window `0.66-0.70`.
    - weight `0.5`.
  - CUDA training/checkpoint validation: pass on `NVIDIA GeForce RTX 3080`.
  - cross replay gate: `ready_for_clean_eval=true`.
  - target22 action probe:
    - action counts `{"10":18,"15":110,"4":22,"7":372}`.
    - dominant action fraction `0.7126436782`.
    - moving steer fraction `0.0766283525`.
  - steer2 action probe:
    - action counts `{"10":2,"12":2,"13":16,"15":87,"4":66,"7":474}`.
    - dominant action fraction `0.7326120556`.
    - moving steer fraction `0.1329211747`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\bc47fd5_corr056w4_cap1316_override13w05_cuda_official_clean512`
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `189` / `512`.
    - route progress: `0.6410256410`.
    - collision count: `1`, offroad count: `1`.
    - final action band: action `12` appeared at route `0.6154`, then action
      `15/7`, then action `13` at route `0.6410`; collision happened while
      still moving around `25 km/h`.

Decision: action capping reduced early `13/16` leakage in offline probes, but
official driving still failed earlier than the `9be6b2f` baseline because the
vehicle stayed too fast through the late correction band. The next Stage 2
change should keep the early `0.56-0.60` throttle ceiling and add a second,
late route throttle ceiling for `0.64-0.70`; reusing only the single existing
throttle ceiling would remove the early constraint that helped reach route
`0.6667`.

Late throttle ceiling support:

- Added `route_late_throttle_*` training parameters to Stage 2 replay training
  and Windows wrappers.
- Purpose: preserve the early `route_throttle_*` ceiling at `0.56-0.60` while
  adding an independent late speed/throttle reduction in the collision band.
- Targeted next CUDA candidate:
  - keep original `9be6b2f_corr056_w4` hyperparameters.
  - keep early `route_throttle_ceiling=0.35` on route `0.56-0.60`.
  - add `route_late_throttle_ceiling=0.20-0.25` on route `0.64-0.70`.
  - start without an action `13` override, or keep action `13` override very
    weak, because stronger steering already regressed to route `0.6154-0.6410`
    collisions.
  - keep CUDA guard mandatory.
- Local verification:
  - targeted Stage 2 tests: `61 passed`.
  - full `RCS-AC-v2/tests`: `139 passed`.

2026-05-26 clean-first resume after sub-agent thread-limit recovery:

- Sub-agent creation failure root cause:
  - `agent thread limit reached` was caused by completed legacy sub-agent
    threads still occupying the available thread slots.
  - Completed/obsolete threads were closed, then two bounded explorer agents
    were spawned successfully for this work packet.
  - The persistent sub-agent protocol now documents close/reuse/retry recovery.
- Strict PDF-internal actor-collapse diagnostic:
  - commit: `b5a12e1`.
  - diagnostic: `strict_actor_collapse_probe.py`.
  - rejected strict entropy candidate:
    `22a374f_pdf_stage1_strict_entropy005_wm2000_pi1000`.
  - target22 and steer2 probes both produced:
    - actor action counts: `{"7":256}`.
    - counterfactual best action counts: `{"7":256}`.
    - actor equals counterfactual best fraction: `1.0`.
  - decision: collapse is not only actor logits; the world model's
    counterfactual returns also globally prefer action `7`. Pure PDF Stage I
    strict actor-critic remains blocked until the reward/continue/imagination
    side can produce state-dependent action returns.
- Non-route-local fallback candidate:
  - checkpoint:
    `b5a12e1_mixed_t22_steer2_aux1_iw5_ssf005_nobal`.
  - training source:
    `windows_privileged_7f87656_mixed_target22_steer2`.
  - CUDA: `NVIDIA GeForce RTX 3080`.
  - objective:
    - `WorldModelActionAuxWeight=1.0`.
    - `ImitationWeight=5.0`.
    - `BehaviorWeight=0.0`.
    - `SteerSequenceFraction=0.05`.
    - no imitation class balancing.
  - cross replay gate:
    - `ready_for_clean_eval=true`.
    - target22 action counts: `{"12":2,"15":82,"4":28,"7":410}`.
    - target22 moving steer fraction: `0.0574712644`.
    - target22 latent test accuracy: `0.9904761910`.
    - steer2 action counts: `{"15":78,"4":34,"7":535}`.
    - steer2 moving steer fraction: `0.0525502318`.
    - steer2 latent test accuracy: `0.9153845906`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\b5a12e1_mixed_t22_steer2_aux1_iw5_ssf005_nobal_official512`.
    - source: `windows_carla`.
    - commit: `b5a12e1280e489f1a0aabc5e07e514781628b330`.
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `305` / `512`.
    - metrics rows: `305`.
    - route progress: `0.6666666667`.
    - collision/offroad/stuck: `1/0/0`.
    - final action band: speed was reduced to about `18 km/h`, but action
      `7` straight still dominated at route `0.6410-0.6667`; collision occurred
      at step `304`.
- Late action10 override candidate:
  - checkpoint:
    `b5a12e1_mixed_t22_steer2_aux1_iw5_ssf005_nobal_late10w075_r0666`.
  - route override:
    - action `10` (`throttle=0.3`, `steer=0.1`).
    - route window `0.666-0.70`.
    - weight `0.75`.
  - CUDA detached training completed with valid checkpoint manifest.
  - cross replay gate:
    - `ready_for_clean_eval=true`.
    - target22 action counts:
      `{"10":16,"13":6,"15":64,"16":2,"19":4,"4":18,"7":412}`.
    - target22 moving steer fraction: `0.0881226054`.
    - target22 latent test accuracy: `0.9904761910`.
    - steer2 action counts: `{"10":51,"13":2,"15":64,"4":52,"7":478}`.
    - steer2 moving steer fraction: `0.1622874807`.
    - steer2 latent test accuracy: `0.9153845906`.
  - official 512 clean:
    - run output:
      `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\b5a12e1_mixed_t22_steer2_aux1_iw5_ssf005_nobal_late10w075_r0666_official512`.
    - source: `windows_carla`.
    - commit: `b5a12e1280e489f1a0aabc5e07e514781628b330`.
    - status: `FAIL`, clean gate pass: `false`.
    - done reason: `collision`.
    - steps: `325` / `512`.
    - metrics rows: `325`.
    - route progress: `0.6666666667`.
    - collision/offroad/stuck: `1/0/0`.
    - final action band: action `10` appeared from step `309` through `324`;
      this proves the late turn target now affects executed actions, but the
      weak `steer=0.1` correction is still insufficient to clear the turn.

Decision: this is still Stage 2 privileged clean FAIL, not clean PASS. The
important improvement is that the failure moved from "no final steering effect"
to "late steering effect exists but is too weak." Raw and fusion remain blocked.
The next clean-first patch is a route-windowed late moving-steer margin loss
that applies only after route progress `>=0.666`, so it can encourage a moving
steer family without hard-coding one action or leaking the correction into the
fragile approach band.

Route-late moving-steer margin candidate:

- code commit: `bbea3a5`.
- checkpoint:
  `bbea3a5_mixed_t22_steer2_aux1_iw5_ssf005_nobal_latemove_w1_r0666`.
- training:
  - CUDA: `NVIDIA GeForce RTX 3080`.
  - `WorldModelActionAuxWeight=1.0`.
  - `ImitationWeight=5.0`.
  - `BehaviorWeight=0.0`.
  - `SteerSequenceFraction=0.05`.
  - `RouteLateMovingSteerMarginWeight=1.0`.
  - `RouteLateMovingSteerMargin=0.25`.
  - route window `0.666-0.70`.
- local verification before Windows pull:
  - full `RCS-AC-v2/tests`: `143 passed`.
  - `scripts/check_git_hygiene.sh`: pass for v2.
- cross replay gate:
  - `ready_for_clean_eval=true`.
  - target22 action counts:
    `{"10":8,"13":6,"15":64,"16":20,"19":2,"4":10,"7":412}`.
  - target22 moving steer fraction: `0.0881226054`.
  - target22 latent test accuracy: `0.9904761910`.
  - steer2 action counts: `{"15":64,"4":96,"7":487}`.
  - steer2 moving steer fraction: `0.1483771252`.
  - steer2 latent test accuracy: `0.9153845906`.
- official 512 clean:
  - run output:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\bbea3a5_mixed_t22_steer2_aux1_iw5_ssf005_nobal_latemove_w1_r0666_official512`.
  - source: `windows_carla`.
  - commit: `bbea3a5e5f586c0143992007818a068be9db85b0`.
  - status: `FAIL`, clean gate pass: `false`.
  - done reason: `collision`.
  - steps: `327` / `512`.
  - metrics rows: `327`.
  - route progress: `0.6666666667`.
  - collision/offroad/stuck: `1/0/0`.
  - final action band: action `4` left steer was selected repeatedly from
    route `0.6410`, then action `7`, then action `4` again at `0.6667`.

Decision: route-windowing prevented early global steering collapse, but the
generic moving-steer family can choose the wrong direction. The next smallest
clean-first patch should distinguish right-moving steering from left-moving
steering in the late route window, or use an action-family/right-steer margin
instead of a generic moving-steer margin. This is still Stage 2 privileged
clean FAIL; raw and fusion remain blocked.

Route-late moving-steer plus weak action13 candidate:

- checkpoint:
  `cc8ac46_mixed_t22_steer2_aux1_iw5_ssf005_latemove_w1_late13w05_r0666`.
- training:
  - CUDA: `NVIDIA GeForce RTX 3080`.
  - same `bbea3a5` route-late moving-steer margin setup.
  - added `RouteActionOverrideWeight=0.5`.
  - added `RouteActionOverrideId=13`.
  - route window `0.666-0.70`.
- cross replay gate:
  - `ready_for_clean_eval=true`.
  - target22 action counts:
    `{"10":6,"13":6,"15":64,"16":22,"19":2,"4":12,"7":410}`.
  - target22 moving steer fraction: `0.0919540230`.
  - steer2 action counts: `{"13":30,"15":64,"16":27,"4":55,"7":471}`.
  - steer2 moving steer fraction: `0.1731066461`.
- official 512 clean:
  - run output:
    `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\cc8ac46_mixed_t22_steer2_aux1_iw5_ssf005_latemove_w1_late13w05_r0666_official512`.
  - source: `windows_carla`.
  - commit: `cc8ac46c6d33c62ee1047108778d97ec33a18da3`.
  - status: `FAIL`, clean gate pass: `false`.
  - done reason: `collision`.
  - steps: `302` / `512`.
  - metrics rows: `302`.
  - route progress: `0.6410256410`.
  - collision/offroad/stuck: `1/0/0`.
  - final action band: action `13` appeared at route `0.6154`, then action
    `13/16` dominated at `0.6410`; collision happened before reaching the
    previous `0.6667` failure point.

Decision: the clean failure is now bracketed. Action `10` right steer is too
weak at route `0.6667`, while action `13/16` pressure is too strong or leaks
too early around route `0.6154-0.6410`. The next smallest patch should add a
late-right-steer family margin with an explicit pre-`0.666` cap for actions
`13/16/19`, or reduce action13 pressure while adding a late throttle ceiling
below `18 km/h`. Do not start raw/fusion.

## 2026-05-26 Late Right-Steer Margin Patch

- Added a route-windowed `route_late_right_steer_margin_*` loss to Stage 2
  replay training.
- Purpose:
  - avoid the generic moving-steer failure where the policy selected left action
    `4` in the final route band.
  - encourage only forward right-moving steer actions (`throttle > 0`,
    `steer > 0`, non-reverse) after the configured route-progress threshold.
  - keep the existing `route_action_mass_cap_*` available to cap strong-right
    actions `13/16/19` before route `0.666`.
- Windows wrappers now expose the same parameters through:
  - `train_stage2_privileged_from_replay.ps1`
  - `start_stage2_replay_training_detached.ps1`
  - `run_stage2_mixed_candidate_cycle.ps1`
- Strict PDF Stage I mode rejects this diagnostic loss, so strict actor-critic
  experiments remain separated from clean-recovery candidates.
- Local verification:
  - `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py`
    passed: `66 passed`.
  - `PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests`
    passed: `144 passed`.
- Next CUDA candidate after Windows reconnect:
  - run the canonical resume wrapper first:
    `RCS-AC-v2\scripts\windows\resume_stage2_late_right_margin_candidate.ps1`.
  - this wrapper runs git pull, dependency/CUDA check, late-right `-PlanOnly`,
    the candidate wrapper, and gate summary, but never runs official 512 clean
    eval.
  - if detached training is still running, it reports
    `STAGE2_LATE_RIGHT_RESUME_CANDIDATE_RUNNING=True` and exits `3` so the same
    command can be retried later without treating the running task as a failed
    candidate.
  - fallback/manual candidate wrapper:
    `RCS-AC-v2\scripts\windows\run_stage2_late_right_margin_candidate.ps1`.
  - use target22 + steer2 mixed replay first so the new diagnostic loss is
    tested without additional correction/offroad component confounds.
  - add late right-steer margin on route `0.666-0.70`.
  - add pre-window cap for strong-right actions `13,16,19` on route
    `0.55-0.66`.
  - keep CUDA mandatory and run target22/steer2 cross replay gate before any
    official 512 clean eval.
  - the wrapper prints
    `OFFICIAL_512_CLEAN_EVAL_BLOCKED_UNTIL_READY_FOR_CLEAN_EVAL=True` and a
    `summarize_stage2_candidate_gate.ps1` command with explicit `-ProjectRoot`
    and `-CandidateName late_right_margin` so reconnect/resume runs do not
    accidentally skip the offline gate or summarize under an older default
    clone/candidate name.
  - `-PlanOnly` prints resolved runner arguments, output dirs, gate summary
    command, and official-eval blocker text without git pull, CUDA guard,
    scheduled-task launch, replay creation, or probing.
- Recommended Windows command after reconnect:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\resume_stage2_late_right_margin_candidate.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady `
  -Device cuda
```

Manual fallback if the resume wrapper cannot be used:

```powershell
git pull --ff-only

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_late_right_margin_candidate.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady `
  -Device cuda

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\summarize_stage2_candidate_gate.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -GatePath RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\f7d903a2_mixed_t22_steer2_aux1_iw5_ssf005_rightmargin_w1_cap131619_pre0666_cross_probe\stage2_candidate_cross_replay_gate.json `
  -CandidateName late_right_margin `
  -FallbackCandidateName target22
```

Only run official 512 clean eval if the gate summary reports
`ready_for_clean_eval=true`.

- Windows connectivity note:
  - local Tailscale userspace restart failed to provide a usable socket.
  - log evidence included:
    `policy requires hardware attestation, but device does not support it`.
  - no Windows CUDA/CARLA run was attempted from this patch in this turn.

## 2026-05-26 Windows Remote Resume And Late-Right Offline Gates

Windows remote execution was resumed through the documented Mac -> userspace
Tailscale SOCKS -> Windows SSH path:

- Windows node: `100.87.219.58`.
- Windows clean clone:
  `C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656`.
- CARLA readiness guard passed with Town04 loaded.
- CUDA guard passed on `NVIDIA GeForce RTX 3080`; Stage 2 training/probe
  commands used `Device=cuda`.

Infrastructure fixes committed and pushed before/while resuming:

- `2ee1b4c`: fixed empty `ComponentSampleWeights` propagation in
  `run_stage2_mixed_candidate_cycle.ps1`.
- `5b7ee99`: preserved gate summaries when a candidate is rejected during
  `resume_stage2_late_right_margin_candidate.ps1`.
- `b462386`: added route-progress-bin action probe diagnostics so rejected
  candidates show where moving-steer false positives concentrate.

Late-right canonical candidate:

- checkpoint:
  `f7d903a2_mixed_t22_steer2_aux1_iw5_ssf005_rightmargin_w1_cap131619_pre0666`.
- target22 cross replay:
  - action/latent probes passed.
  - moving steer fraction: `0.1149425287`.
  - latent test accuracy: `0.9904761910`.
- steer2 cross replay:
  - latent probe passed.
  - action probe rejected.
  - moving steer fraction: `0.2380216383`, gate max `0.20`.
  - decision summary: `run_target22_fallback`.
- Route-bin diagnostic showed the remaining false positives concentrated around
  route `0.50-0.60` and `0.66-0.70`, not in the early straight segment.
- Decision: reject, do not run official 512 clean eval.

Target22 fallback replay candidate:

- checkpoint:
  `windows_replay_7f87656_target22_aux1_balpow05_iw5_rerun_20260525T064214`.
- target22 cross replay passed.
- steer2 cross replay rejected:
  - moving steer fraction: `0.3292117465`.
  - latent test accuracy: `0.7769230604`, below gate `0.80`.
- Route-bin diagnostic showed near-total predicted moving steer after route
  `0.50`, so this fallback is worse than the late-right candidate on steer2.
- Decision: reject, do not run official 512 clean eval.

All-steer route action cap candidate:

- checkpoint:
  `b462386_mixed_t22_steer2_lateright_allsteercap012_pre050066`.
- route action mass cap:
  - ids: `1,4,10,12,13,16,19,21`.
  - cap: `0.12`.
  - route window: `0.50-0.66`.
  - weight: `4`.
- target22 cross replay passed:
  - moving steer fraction: `0.1187739464`.
  - latent test accuracy: `0.9904761910`.
- steer2 cross replay:
  - latent probe passed.
  - action probe rejected.
  - moving steer fraction: `0.2194744977`, gate max `0.20`.
  - latent margin over majority: `0.2399239138`.
- Route-bin diagnostic showed this candidate improved over the late-right
  baseline, but remaining false positives persist in route `0.50-0.60` and
  especially `0.66-0.70`.
- Decision: close but reject, do not run official 512 clean eval.

Current active CUDA offline candidate:

- checkpoint/output dir:
  `b462386_mixed_t22_steer2_allsteercap010_pre050070`.
- task:
  `RCSV2Stage2B462AllSteerCap010Pre050070`.
- route action mass cap:
  - ids: `1,4,10,12,13,16,19,21`.
  - cap: `0.10`.
  - route window: `0.50-0.70`.
  - weight: `5`.
- started as detached Windows CUDA training.
- latest check:
  - task state: `Running`.
  - manifest exists: `False`.
  - ready for probe: `False`.

Resume command for the active candidate:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\check_stage2_replay_training_detached.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\b462386_mixed_t22_steer2_allsteercap010_pre050070 `
  -TaskName RCSV2Stage2B462AllSteerCap010Pre050070
```

When `TRAIN_READY_FOR_PROBE=True`, rerun
`run_stage2_mixed_candidate_cycle.ps1` with the same output/probe dirs and
`-ProbeWhenReady`. Promote to official 512 clean eval only if
`stage2_candidate_cross_replay_gate.json` reports `ready_for_clean_eval=true`.
