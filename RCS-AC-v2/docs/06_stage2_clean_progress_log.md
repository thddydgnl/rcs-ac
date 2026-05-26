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

All-steer wider route action cap candidate:

- checkpoint:
  `b462386_mixed_t22_steer2_allsteercap010_pre050070`.
- task:
  `RCSV2Stage2B462AllSteerCap010Pre050070`.
- route action mass cap:
  - ids: `1,4,10,12,13,16,19,21`.
  - cap: `0.10`.
  - route window: `0.50-0.70`.
  - weight: `5`.
- training completed on Windows CUDA with valid checkpoint manifest.
- target22 cross replay passed:
  - moving steer fraction: `0.1187739464`.
  - latent test accuracy: `0.9904761910`.
- steer2 cross replay:
  - latent probe passed.
  - action probe rejected.
  - moving steer fraction: `0.2117465224`, gate max `0.20`.
  - latent margin over majority: `0.2399239138`.
- Route-bin diagnostic showed the wide cap improved the previous
  `0.2194744977` result but still over-produced moving steer around route
  `0.66-0.70`, where predicted moving-steer fraction remained
  `0.9117647059`.
- Decision: close but reject, do not run official 512 clean eval.

Late-window cap candidate:

- checkpoint:
  `fdf50ab_mixed_t22_steer2_latecap045_r066070`.
- task:
  `RCSV2Stage2FDFLateCap045R066070`.
- route action mass cap:
  - ids: `1,4,10,12,13,16,18,19,21`.
  - cap: `0.45`.
  - route window: `0.666-0.70`.
  - weight: `5`.
- rationale: the rejected wide-cap candidate was already close overall, and the
  remaining false positives were concentrated in the late route window. This
  candidate leaves the earlier `0.50-0.66` window less constrained and caps only
  the final late-turn band.
- training completed on Windows CUDA with valid checkpoint manifest.
- target22 cross replay passed:
  - moving steer fraction: `0.1187739464`.
  - latent test accuracy: `0.9904761910`.
- steer2 cross replay:
  - latent probe passed.
  - action probe rejected.
  - moving steer fraction: `0.2117465224`, gate max `0.20`.
  - late route `0.66-0.70` predicted moving-steer fraction improved to
    `0.5294117647`.
- Decision: reject, do not run official 512 clean eval. The late cap fixed the
  final route-bin excess but left the mid-route `0.50-0.60` false positives
  high enough that the total moving-steer gate still failed.

Current active CUDA offline candidate:

- checkpoint/output dir:
  `bf7f201_mixed_t22_steer2_latecap045_msfpa20_msmm10`.
- task:
  `RCSV2Stage2BF7LateCap045Msfpa20Msmm10`.
- objective additions over the late-window cap:
  - `MovingSteerFalsePositiveWeight=2.0`.
  - `MovingSteerNonmovingMarginWeight=1.0`.
  - `MovingSteerNonmovingMargin=0.25`.
- rationale: the late cap handled route `0.66-0.70`; the remaining gate failure
  is mid-route predicted moving steer on non-moving labels, so reuse the
  previous false-positive/non-moving margin levers that had already produced an
  offline-promotable candidate in an earlier sweep.
- latest check:
  - detached task started.
  - manifest exists: `False`.
  - ready for probe: `False`.

Resume command for the active candidate:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\check_stage2_replay_training_detached.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\bf7f201_mixed_t22_steer2_latecap045_msfpa20_msmm10 `
  -TaskName RCSV2Stage2BF7LateCap045Msfpa20Msmm10
```

When `TRAIN_READY_FOR_PROBE=True`, rerun
`run_stage2_mixed_candidate_cycle.ps1` with the same output/probe dirs and
`-ProbeWhenReady`. Promote to official 512 clean eval only if
`stage2_candidate_cross_replay_gate.json` reports `ready_for_clean_eval=true`.

## 2026-05-26 BF7 Offline Gate Pass And Official Mask Diagnostics

The active CUDA candidate completed on the Windows clean clone:

- checkpoint/output dir:
  `bf7f201_mixed_t22_steer2_latecap045_msfpa20_msmm10`.
- objective additions:
  - late action-mass cap at route `0.666-0.70`.
  - `MovingSteerFalsePositiveWeight=2.0`.
  - `MovingSteerNonmovingMarginWeight=1.0`.
  - `MovingSteerNonmovingMargin=0.25`.
- target22 replay gate passed:
  - action counts `{10:26,13:8,15:88,16:20,19:6,7:374}`.
  - moving steer fraction `0.1149425287`.
  - latent test accuracy `0.9904761910`.
- steer2 replay gate passed:
  - action counts `{1:5,10:71,12:7,15:93,21:19,4:26,7:426}`.
  - moving steer fraction `0.1978361669`.
  - dominant action 7 fraction `0.6584234930`.
  - latent test accuracy `0.9230769277`.
- cross replay gate result: `ready_for_clean_eval=true`.

Official 512 clean eval then ran after the CARLA readiness guard:

- output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\bf7f201_mixed_t22_steer2_latecap045_msfpa20_msmm10_official_clean512`.
- result:
  - `source=windows_carla`.
  - `status=FAIL`.
  - `done_reason=collision`.
  - `steps=147/512`.
  - `route_progress=0.5384615385`.
  - `collision_count=1`.
  - `offroad_count=1`.
- tail diagnosis:
  - final sequence used actions `10 -> 13 -> 16 -> 19`.
  - final action 19 was `throttle=0.3`, `steer=0.5`.
  - speed was already high, about `24 km/h`.

Three forbidden-action probes were then used as diagnostics only. They are not
new trained policies and should not become the main optimization path:

- `ForbiddenActionIds=19`:
  - cross replay gate still passed.
  - official 512 failed at the same route progress `0.5384615385`.
  - final repeated action 16 caused collision/offroad at about `27.6 km/h`.
- `ForbiddenActionIds=16,19`:
  - cross replay gate still passed.
  - official 512 failed at the same route progress `0.5384615385`.
  - failure changed to offroad without collision.
  - final action 21 was `throttle=0.7`, `steer=0.2`, speed about `34.4 km/h`.
- `ForbiddenActionIds=16,19,21`:
  - cross replay gate still passed.
  - official 512 failed by offroad at `148/512`.
  - route progress stayed `0.5384615385`.
  - final tail used action 15 straight at `throttle=0.7`, then action 13
    at `throttle=0.3`, `steer=0.2`; speed was about `32.8 km/h`.

Decision:

- The offline gate is now capable of promoting a candidate, but official clean
  still fails in the route `0.49-0.54` band.
- Masking high-right actions shifts the failure from collision to offroad but
  does not improve route progress, so do not continue by only adding more
  forbidden actions.
- The immediate bottleneck is speed/longitudinal control before and during the
  right-correction band. The next smallest trained candidate should add a
  route-window throttle ceiling around route `0.48-0.56` while preserving the
  BF7 false-positive/non-moving margin objective.
- Continue to use CUDA for all replay training/probes and run the cross replay
  gate before any new official CARLA clean eval.

Next candidate command:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\5455ac1_mixed_t22_steer2_latecap045_msfpa20_msmm10_thr030_r048056 `
  -CrossProbeOutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5455ac1_mixed_t22_steer2_latecap045_msfpa20_msmm10_thr030_r048056_cross_probe `
  -TaskName RCSV2Stage2Throttle030R048056 `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady `
  -WorldModelSteps 1500 `
  -PolicySteps 1500 `
  -ImitationWeight 5.0 `
  -ImitationBalancePower 0.5 `
  -SteerSequenceFraction 0.15 `
  -WorldModelActionAuxWeight 1.0 `
  -RouteActionMassCapWeight 5.0 `
  -RouteActionMassCapIds "1,4,10,12,13,16,18,19,21" `
  -RouteActionMassCap 0.45 `
  -RouteActionMassCapMinRouteProgress 0.666 `
  -RouteActionMassCapMaxRouteProgress 0.70 `
  -MovingSteerFalsePositiveWeight 2.0 `
  -MovingSteerNonmovingMarginWeight 1.0 `
  -MovingSteerNonmovingMargin 0.25 `
  -RouteThrottleCeilingWeight 4.0 `
  -RouteThrottleCeiling 0.30 `
  -RouteThrottleMinRouteProgress 0.48 `
  -RouteThrottleMaxRouteProgress 0.56 `
  -Device cuda
```

If that candidate passes the cross replay gate, run CARLA readiness first and
then a single official 512 eval. If it fails offline, reject it without CARLA
and inspect route-bin action diagnostics before choosing another candidate.

Follow-up:

- `5455ac1_mixed_t22_steer2_latecap045_msfpa20_msmm10_thr030_r048056` trained
  on Windows CUDA and was rejected offline:
  - target22 passed.
  - steer2 action probe failed with moving steer fraction `0.2117465224`.
- `5455ac1_mixed_t22_steer2_latecap045_msfpa20_msmm10_thr025_hicap_r048056`
  added route `0.48-0.56` high-throttle action mass cap for actions `15,21`,
  but produced the same steer2 moving steer fraction `0.2117465224`.
- BF7 with diagnostic `ForbiddenActionIds=15` was also rejected offline:
  - target22 dominant action 7 fraction increased to `0.8850574713`.
  - steer2 moving steer fraction stayed above gate at `0.2040185471`.
- Conclusion: soft action-mass/throttle losses and post-hoc action 15 masking do
  not solve the route `0.49-0.54` clean failure. The next implementation patch
  adds a route-window action-logit cap loss so selected high-speed actions are
  pushed below the best non-capped action at the logit/argmax level.

## 2026-05-26 - Route-window action-logit cap candidates

Commit `78fa94b` added a route-window action-logit cap loss to Stage 2 replay
training and the Windows wrappers. Commit `0382d07` fixed the detached Windows
training command so the route override max-progress argument is always passed
before the new logit-cap arguments.

The strong logit-cap candidate
`78fa94b_mixed_t22_steer2_latecap045_msfpa20_msmm10_logitcap15161921_r048056`
used BF7 plus:

- `RouteActionLogitCapWeight=4.0`.
- `RouteActionLogitCapIds=15,16,19,21`.
- `RouteActionLogitCapMargin=0.20`.
- route window `0.48-0.56`.

Result: rejected offline, no official clean eval.

- target22 action probe passed:
  - action counts `{1:2,10:6,13:8,15:390,19:2,4:24,7:90}`.
  - moving steer fraction `0.0804597701`.
  - dominant action 15 fraction `0.7471264368`.
  - latent test accuracy `0.9904761910`.
- steer2 action probe failed:
  - action counts `{1:4,10:91,12:2,15:374,4:41,7:135}`.
  - moving steer fraction `0.2132921175`.
  - latent test accuracy `0.8923076987`.
- diagnosis: the loss was too strong and pushed the policy toward action 15
  globally instead of only removing the clean-failure action tail.

The weak logit-cap candidate
`0382d07_mixed_t22_steer2_latecap045_msfpa20_msmm10_logitcap15161921_w1m005_r048056`
used BF7 plus:

- `RouteActionLogitCapWeight=1.0`.
- `RouteActionLogitCapIds=15,16,19,21`.
- `RouteActionLogitCapMargin=0.05`.
- route window `0.48-0.56`.

It trained on Windows CUDA and was also rejected offline:

- target22 action probe passed:
  - action counts `{1:2,10:6,13:4,15:244,16:2,19:4,4:24,7:236}`.
  - moving steer fraction `0.0804597701`.
  - dominant action 15 fraction `0.4674329502`.
  - latent test accuracy `0.9904761910`.
- steer2 action probe failed:
  - action counts `{1:4,10:91,12:2,15:224,4:40,7:286}`.
  - moving steer fraction `0.2117465224`.
  - latent test accuracy `0.8923076987`.

Decision:

- Do not run official CARLA eval for either logit-cap candidate.
- The route `0.48-0.56` family repeatedly fails the steer2 offline gate around
  moving-steer fraction `0.211-0.213`, while BF7 without the added cap already
  passed steer2 at `0.1978361669`.
- The clean bottleneck is not solved by suppressing individual high-right or
  high-throttle actions after BF7. That search tends to shift the policy toward
  action 15 or slightly increase mid-route false-positive steering.
- The next candidate should be a narrower, less global intervention: either
  re-collect the route `0.49-0.56` failure-window replay at a lower target speed
  so labels encode the slower correction directly, or use a very narrow
  route-progress action override/cap that is checked offline before any CARLA
  command.

## 2026-05-26 - BF7 route 0.50 correction replay branch

The narrow action-override probe
`0382d07_mixed_t22_steer2_latecap045_msfpa20_msmm10_override5w050_r048056`
kept BF7 and added:

- `RouteActionOverrideWeight=0.5`.
- `RouteActionOverrideId=5`.
- route window `0.48-0.56`.

Result: rejected offline, no official clean eval.

- target22 passed:
  - action counts `{1:2,10:8,12:4,13:2,15:408,16:2,19:6,4:20,7:70}`.
  - moving steer fraction `0.0842911877`.
  - dominant action 15 fraction `0.7816091954`.
  - latent test accuracy `0.9904761910`.
- steer2 failed:
  - action counts `{1:2,10:89,12:2,15:238,4:42,5:6,7:268}`.
  - moving steer fraction `0.2086553323`.
  - zero-throttle action 5 appeared only 6/647 times.
  - latent test accuracy `0.8923076987`.

Decision: a direct route override in the existing mixed replay is not enough.
It does not teach the official failure band strongly enough and still fails the
steer2 action gate.

A new BF7-warmup correction replay was collected from the official failure
band:

- output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\replay\bf7_route050_collision_window_t12_min5`.
- collection config:
  - BF7 checkpoint-filled config as both runtime and warmup policy.
  - target speed `12 km/h`.
  - record after route progress `0.50`.
  - `4` episodes, `256` max steps, `12` max attempts, skip failed episodes.
- manifest written with clean_gate_evidence=false.

Two BF7-base mixed-replay candidates used this replay as `corr050` with
target22 and steer2:

- `0382d07_corr050_t12_w025_bf7base`:
  - mixed replay: `8` episodes, `1209` steps.
  - component weights: `target22=1,steer2=1,corr050=0.25`.
  - target22 passed:
    - action counts `{1:2,10:8,13:2,15:244,16:4,19:4,4:22,7:236}`.
    - moving steer fraction `0.0804597701`.
    - latent test accuracy `0.9904761910`.
  - steer2 failed:
    - action counts `{1:4,10:91,12:2,15:213,4:40,7:297}`.
    - moving steer fraction `0.2117465224`.
    - latent test accuracy `0.8923076987`.
- `0382d07_corr050_t12_w010_bf7base`:
  - component weights: `target22=1,steer2=1,corr050=0.10`.
  - target22 and steer2 probes matched the w0.25 action distributions.
  - steer2 again failed with moving steer fraction `0.2117465224`.

Decision:

- Do not run official CARLA eval for the t12 correction candidates.
- Lowering the correction component weight did not change the offline action
  distribution, so t12 correction labels are not resolving the steer2
  false-positive steering boundary.
- Continue with a medium-speed correction replay at target speed `16 km/h`
  before changing the training objective again.

The medium-speed BF7-warmup correction replay was then collected:

- output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\replay\bf7_route050_collision_window_t16_min5`.
- collection config:
  - BF7 checkpoint-filled config as both runtime and warmup policy.
  - target speed `16 km/h`.
  - record after route progress `0.50`.
  - `4` episodes, `256` max steps, `12` max attempts, skip failed episodes.

Two t16 candidates were trained:

- `0382d07_corr050_t16_w025_bf7base`:
  - component weights: `target22=1,steer2=1,corr050=0.25`.
  - target22 passed with the same healthy distribution as t12:
    `{1:2,10:8,13:2,15:244,16:4,19:4,4:22,7:236}`.
  - steer2 failed:
    - action counts `{1:4,10:91,12:2,15:213,4:40,7:297}`.
    - moving steer fraction `0.2117465224`.
    - latent test accuracy `0.8923076987`.
- `0382d07_corr050_t16_w025_msfpa30_bf7base`:
  - same t16 mixed replay and component weights.
  - raised `MovingSteerFalsePositiveWeight` from `2.0` to `3.0`.
  - target22 again passed with the same distribution.
  - steer2 again failed with the same action counts and moving steer fraction
    `0.2117465224`.

Decision:

- Do not run official CARLA eval for the t16 or msfpa30 candidates.
- The corr050 branch is not moving the policy distribution in the problematic
  steer2 bins, even when the correction replay target speed changes from `12`
  to `16` and the false-positive weight increases to `3.0`.

### 2026-05-26 - corr050 short-window diagnosis and corr048 min16 retry

The identical corr050 candidate distributions were traced to replay component
eligibility, not to an effective correction signal:

- The trainer now writes `component_sampler_stats` and sampler included/dropped
  episode lists.
- If an explicitly positive component weight has zero episodes with
  `length >= sequence_length`, training raises before spending CUDA time.
- Windows clean clone diagnostic confirmed both
  `mixed_t22_steer2_corr050_t12` and `mixed_t22_steer2_corr050_t16` had
  corr050 lengths `[10,10,10,10]` while `sequence_length=16`.

The route-progress trigger was moved from `0.50` to `0.48` so the collision
window would be long enough:

- corrected replay:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\replay\bf7_route048_collision_window_t16_min16`.
- episodes: `4`.
- total_steps: `68`.
- episode_lengths: `[17,17,17,17]`.
- action_counts: `{2:16,5:32,8:20}`.
- clean_gate_evidence: `false`.

The corrected mixed replay was:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\replay\mixed_t22_steer2_corr048_t16_min16
```

Component sampler stats confirmed the correction component now enters training:

```text
corr048 included_episodes=4
corr048 dropped_short_episodes=0
corr048 window_count=8
configured_episode_sample_prob_by_component={target22:0.4, steer2:0.4, corr048:0.2}
```

CUDA training completed for:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\33d1705_corr048_t16_min16_w025_bf7base
```

Cross-replay gate rejected before any official CARLA eval:

- target22 action probe:
  - `recommended_for_clean_rerun=false`.
  - action_counts `{7:484,4:22,10:8,16:4,19:4}`.
  - moving_steer_fraction `0.0727969349`.
  - dominant_action_fraction `0.9272030651`.
  - latent test accuracy `0.9904761910`.
- steer2 action probe:
  - `recommended_for_clean_rerun=false`.
  - action_counts `{7:590,4:57}`.
  - moving_steer_fraction `0.0880989181`.
  - dominant_action_fraction `0.9119010819`.
  - latent test accuracy `0.8999999762`.

Decision:

- Do not run official 512 clean eval for
  `33d1705_corr048_t16_min16_w025_bf7base`.
- The corrected component now participates in training, but the first weight
  setting over-corrects toward dominant straight action 7.
- The next branch should keep the min16 replay guard and adjust the objective or
  component probability to restore BF7-like action diversity before CARLA.

### 2026-05-26 - corr048 weight 0.05 / steer-sequence sweep

A sub-agent sidecar audit recommended the smallest next CUDA change: keep the
corrected min16 replay, lower corr048 component weight from `0.25` to `0.05`,
and do not run official CARLA unless the cross-replay gate passes.

Three CUDA candidates were trained from the Windows clean clone:

1. `095edec_corr048_t16_min16_w005_bf7params`
   - `ComponentSampleWeights=target22=1,steer2=1,corr048=0.05`.
   - `SteerSequenceFraction=0.05`.
   - target22:
     - `recommended_for_clean_rerun=false`.
     - action_counts `{15:460,4:34,10:6,13:2,19:2,1:2,7:16}`.
     - moving_steer_fraction `0.0881226054`.
     - dominant_action_fraction `0.8812260536`.
     - latent test accuracy `0.9904761910`.
   - steer2:
     - `recommended_for_clean_rerun=true`.
     - action_counts `{15:438,7:141,4:55,1:10,12:3}`.
     - moving_steer_fraction `0.1051004637`.
     - dominant_action_fraction `0.6769706337`.
     - latent test accuracy `0.8999999762`.
   - Decision: reject before CARLA because target22 action probe failed.

2. `095edec_corr048_t16_min16_w005_ssf010`
   - `ComponentSampleWeights=target22=1,steer2=1,corr048=0.05`.
   - `SteerSequenceFraction=0.10`.
   - target22:
     - `recommended_for_clean_rerun=false`.
     - action_counts `{15:456,10:32,16:20,13:8,19:6}`.
     - moving_steer_fraction `0.1264367816`.
     - dominant_action_fraction `0.8735632184`.
     - latent test accuracy `0.9904761910`.
   - steer2:
     - `recommended_for_clean_rerun=false`.
     - action_counts `{15:425,10:203,7:19}`.
     - moving_steer_fraction `0.3137557960`.
     - dominant_action_fraction `0.6568778980`.
     - latent test accuracy `0.8769230843`.
   - Decision: reject before CARLA because target22 dominance remains too high
     and steer2 over-steers.

3. `095edec_corr048_t16_min16_w005_ssf015`
   - `ComponentSampleWeights=target22=1,steer2=1,corr048=0.05`.
   - `SteerSequenceFraction=0.15`.
   - target22:
     - `recommended_for_clean_rerun=false`.
     - action_counts `{15:450,10:28,16:20,13:8,19:6,7:10}`.
     - moving_steer_fraction `0.1187739464`.
     - dominant_action_fraction `0.8620689655`.
     - latent test accuracy `0.9904761910`.
   - steer2:
     - `recommended_for_clean_rerun=false`.
     - action_counts `{15:421,10:153,16:58,13:5,7:10}`.
     - moving_steer_fraction `0.3338485317`.
     - dominant_action_fraction `0.6506955178`.
     - latent test accuracy `0.9076923132`.
   - Decision: reject before CARLA because target22 dominance remains too high
     and steer2 over-steers.

Interpretation:

- Lowering corr048 weight solved the previous action7 collapse, but the policy
  shifted into action15/right-turn dominance.
- Increasing `SteerSequenceFraction` from `0.05` to `0.10/0.15` reduces target22
  action15 dominance only slightly and breaks steer2 by over-predicting moving
  right steering.
- Plain corr048 weight/steer-sequence sweeps are no longer the best next move.
  The next patch should directly control action15/right-moving dominance while
  preserving the steer2 false-positive gate.
- The current repeated failure is now more structural than a single route-local
  replay weight: corr048 weight/ssf changes can move the collapse target, but
  they do not reliably produce a route-dependent policy distribution that passes
  target22 and steer2 together.
- Next work should directly constrain action15/right-turn dominance and still
  treat steer2 moving-steer false positives as a hard pre-CARLA gate.

### 2026-05-26 - first direct action15 cap

The first direct action15-control candidate reused the best corr048 low-weight
setting and added the existing action-mass cap hook:

```text
candidate: 8b78015_corr048_w005_ssf005_cap15w1c82
ComponentSampleWeights: target22=1, steer2=1, corr048=0.05
SteerSequenceFraction: 0.05
ActionMassCapWeight: 1.0
ActionMassCaps: 15=0.82
```

Windows CUDA training completed successfully and the cross-replay probe ran
from the clean clone. The gate rejected the candidate before official CARLA:

- target22:
  - `recommended_for_clean_rerun=false`.
  - action_counts `{15:460,4:34,10:6,13:2,19:2,1:2,7:16}`.
  - moving_steer_fraction `0.0881226054`.
  - dominant_action_fraction `0.8812260536`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:438,7:141,4:55,1:10,12:3}`.
  - moving_steer_fraction `0.1051004637`.
  - dominant_action_fraction `0.6769706337`.
  - latent test accuracy `0.8999999762`.

Decision:

- Do not run official 512 clean eval for
  `8b78015_corr048_w005_ssf005_cap15w1c82`.
- The action15 cap did not change the target22 distribution enough to clear
  the `dominant_action_fraction <= 0.85` pre-CARLA gate.
- The useful signal is that steer2 stayed healthy, so a stronger or
  target22/route-aware action15 control is the next smallest candidate.

The stronger mass-cap follow-up,
`8b78015_corr048_w005_ssf005_cap15w4c80`, used
`ActionMassCapWeight=4.0` and `ActionMassCaps=15=0.80`. It produced the same
target22 and steer2 action counts as `cap15w1c82`, so it was also rejected
before CARLA. The failure mode is argmax-level action15 dominance, not merely
softmax mass above a cap.

### 2026-05-26 - route action15 logit cap and official 512 rerun

The next candidate used the existing route action logit cap hook:

```text
candidate: 8b78015_corr048_w005_ssf005_logit15w05m0
RouteActionLogitCapWeight: 0.5
RouteActionLogitCapIds: 15
RouteActionLogitCapMargin: 0.0
RouteActionLogitCapMinRouteProgress: 0.0
RouteActionLogitCapMaxRouteProgress: 1.0
```

Cross-replay gate:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:442,4:40,7:38,19:2}`.
  - moving_steer_fraction `0.0804597701`.
  - dominant_action_fraction `0.8467432950`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:420,7:167,4:49,1:8,12:3}`.
  - moving_steer_fraction `0.0927357032`.
  - dominant_action_fraction `0.6491499227`.
  - latent test accuracy `0.8999999762`.
- gate: `ready_for_clean_eval=true`.

Official 512 clean eval:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `146/512`.
- route_progress: `0.6666666667`.
- collision_count/offroad_count/stuck_count: `1/0/0`.
- clean action_counts: `{15:117,7:17,4:10,12:2}`.
- failure tail:
  - step144: action12, throttle `0.7`, steer `-0.1`,
    route_progress `0.6666666667`.
  - step145: action12, throttle `0.7`, steer `-0.1`, collision `true`.

Decision:

- The route action15 logit cap materially improved clean progress: it cleared
  offline gates and moved the official failure from the earlier route-progress
  0.53-0.54 region to 0.6667.
- It is still not a clean PASS. Do not start raw/fusion work.
- The next candidate should patch the late `0.64-0.70` high-throttle-left
  action12 collision, preferably with a route-local logit/throttle control that
  is checked offline before another official 512 eval.

### 2026-05-26 - route action12/15 logit cap

The next candidate expanded the same logit cap to include the collision-tail
action12:

```text
candidate: 8b78015_corr048_w005_ssf005_logit1215w05m0
RouteActionLogitCapWeight: 0.5
RouteActionLogitCapIds: 12,15
RouteActionLogitCapMargin: 0.0
RouteActionLogitCapMinRouteProgress: 0.0
RouteActionLogitCapMaxRouteProgress: 1.0
```

Cross-replay gate:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:442,4:40,7:38,19:2}`.
  - moving_steer_fraction `0.0804597701`.
  - dominant_action_fraction `0.8467432950`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:419,7:170,4:48,1:8,12:2}`.
  - moving_steer_fraction `0.0896445131`.
  - dominant_action_fraction `0.6476043277`.
  - latent test accuracy `0.8999999762`.
- gate: `ready_for_clean_eval=true`.

Official 512 clean eval:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `146/512`.
- route_progress: `0.6666666667`.
- collision_count/offroad_count/stuck_count: `1/0/0`.
- clean action_counts: `{15:116,7:18,4:10,1:1,12:1}`.
- failure tail:
  - step145: action12, throttle `0.7`, steer `-0.1`,
    route_progress `0.6666666667`, collision `true`.

Decision:

- Adding action12 to the weak full-route logit cap reduced clean action12 from
  two steps to one, but the remaining action12 still caused the same collision.
- The next candidate should not run CARLA until it passes target22/steer2
  cross replay. Prefer a stronger action12/15 logit cap or a route-local
  throttle/logit control around route_progress `0.64-0.70`.

### 2026-05-26 - route action12/15 logit-cap bracket

Two bracket candidates tested whether the full-route 12/15 logit cap should be
stronger.

`8b78015_corr048_w005_ssf005_logit1215w2m0` used:

```text
RouteActionLogitCapWeight: 2.0
RouteActionLogitCapIds: 12,15
```

It was rejected before CARLA:

- target22 action_counts `{7:480,4:40,19:2}`.
- target22 dominant_action_fraction `0.9195402299`.
- steer2 action_counts `{7:589,4:51,1:7}`.
- steer2 dominant_action_fraction `0.9103554869`.
- gate: `ready_for_clean_eval=false`.

`8b78015_corr048_w005_ssf005_logit1215w1m0` used:

```text
RouteActionLogitCapWeight: 1.0
RouteActionLogitCapIds: 12,15
```

Cross-replay gate passed:

- target22 action_counts `{7:262,15:220,4:38,19:2}`.
- target22 dominant_action_fraction `0.5019157088`.
- steer2 action_counts `{7:383,15:208,4:49,1:7}`.
- steer2 dominant_action_fraction `0.5919629057`.
- steer2 moving_steer_fraction `0.0865533230`.
- gate: `ready_for_clean_eval=true`.

Official 512 clean eval failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `148/512`.
- route_progress: `0.5384615385`.
- collision_count/offroad_count/stuck_count: `1/1/0`.
- clean action_counts: `{15:80,7:47,10:10,16:5,19:4,13:2}`.
- failure tail:
  - steps 144-147: action19, throttle `0.3`, steer `0.5`,
    route_progress `0.5384615385`.
  - step147: collision `true`, offroad `true`.

Decision:

- Full-route 12/15 logit-cap weight is now bracketed:
  - `0.5` reaches route_progress `0.6667` but leaves one action12 collision.
  - `1.0` shifts into earlier action19 right-oversteer.
  - `2.0` fails offline by action7 dominance.
- Stop full-route logit-weight sweeps.
- Next candidate should preserve the w0.5 full-route action15 behavior and add
  a late route-local action12/throttle control around route_progress
  `0.64-0.70`.

### 2026-05-26 - late route throttle ceiling

The next candidate kept the useful full-route action15 logit cap and added a
route-local throttle ceiling in the failure window:

```text
candidate: 8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4
RouteActionLogitCapWeight: 0.5
RouteActionLogitCapIds: 15
RouteThrottleCeilingWeight: 4.0
RouteThrottleCeiling: 0.3
RouteThrottleCeilingMinRouteProgress: 0.64
RouteThrottleCeilingMaxRouteProgress: 0.70
```

Cross-replay gate:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:442,4:40,7:38,19:2}`.
  - moving_steer_fraction `0.0804597701`.
  - dominant_action_fraction `0.8467432950`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:414,4:48,7:177,1:8}`.
  - moving_steer_fraction `0.0865533230`.
  - dominant_action_fraction `0.6398763524`.
  - latent test accuracy `0.8999999762`.
- gate: `ready_for_clean_eval=true`.

Official 512 clean eval:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `146/512`.
- route_progress: `0.6666666667`.
- collision_count/offroad_count/stuck_count: `1/0/0`.
- clean action_counts: `{15:116,7:19,4:10,1:1}`.
- failure tail:
  - steps 140-145 stayed at throttle `0.3` with action7 straight controls.
  - step145: action7, throttle `0.3`, steer `0.0`,
    route_progress `0.6666666667`, collision `true`.

Decision:

- The route-local throttle ceiling removed the high-throttle action12 tail, so
  the previous immediate action12 failure has been patched.
- The same route-progress collision remains under slow straight action7,
  meaning the next smallest patch should not continue throttle-only tuning.
- Next work should collect or reuse a late collision-window replay around
  route_progress `0.64-0.70`, then train a route-local correction that changes
  late-window action selection without broad full-route logit caps. Do not
  start raw/fusion work before this privileged clean gate passes.

### 2026-05-26 - late056 warmup replay and low-weight offline reject

Following the lateceil failure, two short correction replays were collected
from the Windows clean clone with the lateceil checkpoint-filled config as both
runtime and warmup policy:

- `windows_privileged_4e9f4b3_late056_collision_window_t16_min16`
  - `RecordAfterRouteProgress=0.56`.
  - `TargetSpeedKmh=16`.
  - `MinRecordedSteps=16`.
  - episodes/steps: `4` / `144`.
  - action_counts `{2:4,28:68,31:24,34:48}`.
- `windows_privileged_4e9f4b3_late056_collision_window_t24_min16`
  - `RecordAfterRouteProgress=0.56`.
  - `TargetSpeedKmh=24`.
  - `MinRecordedSteps=16`.
  - episodes/steps: `4` / `142`.
  - action_counts `{2:4,28:48,30:18,31:25,34:47}`.

Interpretation:

- Both replays satisfy the min16 eligibility requirement.
- Both are poor correction labels for the current clean failure because they are
  dominated by zero-throttle steering actions. The t24 variant contains only 18
  throttle-positive action30 labels and still has many action34-style
  zero-throttle labels, which previously caused stuck/collapse failures.

One conservative offline-only candidate tested whether the t24 replay could be
used at a tiny component weight without damaging target22/steer2:

```text
candidate: 4e9f4b3_corr048_w005_late056t24_w002_logit15_lateceil
mixed replay: target22 + steer2 + corr048 + late056t24
ComponentSampleWeights: target22=1, steer2=1, corr048=0.05, late056t24=0.02
RouteActionLogitCapWeight: 0.5
RouteActionLogitCapIds: 15
RouteThrottleCeilingWeight: 4.0
RouteThrottleCeiling: 0.3
RouteThrottle window: 0.64-0.70
```

CUDA training completed and cross replay rejected before CARLA:

- target22:
  - `recommended_for_clean_rerun=false`.
  - action_counts `{15:462,4:42,7:12,12:4,1:2}`.
  - moving_steer_fraction `0.0919540230`.
  - dominant_action_fraction `0.8850574713`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{15:445,7:117,4:57,12:16,1:12}`.
  - moving_steer_fraction `0.1313755796`.
  - dominant_action_fraction `0.6877897991`.
  - latent test accuracy `0.9076923132`.
- gate: `ready_for_clean_eval=false`.

Decision:

- Do not run official 512 clean eval for this candidate.
- The late056 teacher-label route currently worsens target22 action15
  dominance, even at low component weight.
- Next work should change the late replay labeling/action-selection heuristic so
  the correction component is not dominated by zero-throttle steering labels, or
  use a different local correction mechanism. Keep the cross-replay gate as the
  hard pre-CARLA filter.

### 2026-05-26 - min-throttle late045 replay and progress 0.769 failure

The replay collector was patched at commit `5bbca2a` with
`min_recorded_throttle`. This keeps late correction labels moving when the
target speed says the vehicle should still progress, while preserving the
default behavior at `0.0`.

Collection attempts:

- `record_after_route_progress=0.56`, `target_speed_kmh=24`,
  `min_recorded_throttle=0.3`, `min_recorded_steps=16`:
  rejected for low volume. Only 1 eligible episode was produced across 12
  attempts.
- `record_after_route_progress=0.50`, same target/min settings:
  rejected for low volume. The collector produced 0 eligible episodes.
- `record_after_route_progress=0.45`, same target/min settings:
  accepted as a correction replay with 4 episodes, 64 total steps,
  `episode_lengths=[16,16,16,16]`, and
  `action_counts={7:8,10:12,13:16,16:16,19:12}`.

Mixed candidate:

```text
candidate: 5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil
mixed replay: target22 + steer2 + corr048 + late045t24mth03
ComponentSampleWeights: target22=1, steer2=1, corr048=0.05, late045t24mth03=0.02
RouteActionLogitCapWeight: 0.5
RouteActionLogitCapIds: 15
RouteThrottleCeilingWeight: 4.0
RouteThrottleCeiling: 0.3
RouteThrottle window: 0.64-0.70
```

Cross-replay gate passed:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:354,15:110,4:50,19:4,13:2,1:2}`.
  - moving_steer_fraction `0.1111111111`.
  - dominant_action_fraction `0.6781609195`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:457,15:84,4:75,10:18,1:9,12:4}`.
  - moving_steer_fraction `0.1638330757`.
  - dominant_action_fraction `0.7063369397`.
  - latent test accuracy `0.9230769277`.
- gate: `ready_for_clean_eval=true`.

Official 512 clean eval without action mask:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `361/512`.
- route_progress: `0.7692307692`.
- collision/offroad/stuck: `1/0/0`.
- action_counts `{7:249,4:52,15:41,1:7,10:6,19:3,13:2,16:1}`.
- failure tail: action10 mild-right, then action13/action16, then action19
  strong-right at route_progress `0.7692307692`.

The same checkpoint was then probed with `ForbiddenActionIds=19`.
The offline gate still passed, but official 512 clean eval failed again:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `350/512`.
- route_progress: `0.7692307692`.
- collision/offroad/stuck: `1/0/0`.
- action_counts `{1:7,4:42,7:248,10:7,13:3,15:41,16:2}`.
- failure tail: action10 mild-right, then action13, then action16; collision
  occurred on action16 after action19 was masked.

Decision:

- `min_recorded_throttle` fixed the zero-throttle correction-label problem.
- The late045 replay improved official clean progress from the previous
  route_progress `0.6667` collision to `0.7692`.
- A simple action19 mask is not enough; it changes the last collision action
  to action16.
- Next work should target the route_progress `0.74-0.77` transition with a
  more precise late right-steer correction before another official 512 clean
  run.

### 2026-05-26 - mask16/19 and mask8/16/19 diagnostics

Two additional mask-only diagnostics were run on the same checkpoint. These are
not considered the final algorithmic fix; they were used to identify whether a
single bad action id was responsible for the late collision.

`ForbiddenActionIds=16,19`:

- Cross-replay gate: `ready_for_clean_eval=true`.
- target22 action_counts `{7:354,15:110,4:50,13:4,1:2,8:2}`.
- steer2 action_counts `{7:457,15:84,4:75,10:18,1:9,12:4}`.
- A mistaken 128-step smoke run passed with `source=windows_carla`,
  `horizon=128`, `steps=128`, and zero collision/offroad/stuck. This is useful
  smoke evidence only, not the 512 clean gate.
- The real 512 run failed:
  - status `FAIL`, failure_class `driving`, done_reason `collision`.
  - steps `361/512`.
  - route_progress `0.7692307692`.
  - collision/offroad/stuck `1/0/0`.
  - action_counts `{7:244,4:53,15:41,10:8,1:8,13:4,8:3}`.
  - failure tail ended in action8, `throttle=0.0`, `steer=0.6`.

`ForbiddenActionIds=8,16,19`:

- Cross-replay gate: `ready_for_clean_eval=true`.
- target22 action_counts `{7:354,15:110,4:50,13:4,1:2,25:2}`.
- steer2 action_counts `{7:457,15:84,4:75,10:18,1:9,12:4}`.
- The real 512 run failed:
  - status `FAIL`, failure_class `driving`, done_reason `collision`.
  - steps `361/512`.
  - route_progress `0.7948717949`.
  - collision/offroad/stuck `1/0/0`.
  - action_counts `{7:244,4:53,15:41,10:8,1:8,13:6,25:1}`.
  - failure tail ended in action25, `throttle=0.0`, `steer=-1.0`.

Decision:

- Masking individual sharp-right actions exposes other zero-throttle
  sharp-steer alternatives.
- The 0.77-0.79 failure is now classified as a late-window zero-throttle
  sharp-steer fallback problem, not a single action-id problem.
- Stop broad mask escalation as the main path. The next patch should train or
  constrain the late window so it keeps moving mild steering instead of
  falling to zero-throttle sharp steering.

### 2026-05-26 - late action logit cap and ssf010 official regression

Commit `620a897` added an optional route-local late action logit cap to the
Stage 2 replay trainer and Windows wrappers. The hook is disabled by default
and is rejected by `pdf_stage1_strict`; it is a diagnostic clean-first control,
not a PDF Stage I strict setting.

The first two late-cap candidates used the late045 mixed replay with
`ComponentSampleWeights=target22=1,steer2=1,corr048=0.05,late045t24mth03=0.02`,
full-route `RouteActionLogitCapIds=15`, late-window
`RouteLateActionLogitCapIds=8,25` for route `0.74-0.80`, and the existing
route `0.64-0.70` throttle ceiling.

- `620a897_late045_logit15_latecap0825_w05`:
  - target22 action gate passed, but steer2 failed.
  - steer2 moving_steer_fraction was `0.2132921175`, just above the `0.20`
    gate.
  - no official CARLA eval was run.
- `620a897_late045_logit15_latecap0825_w025`:
  - target22 action gate passed, but steer2 failed with the same
    moving_steer_fraction `0.2132921175`.
  - no official CARLA eval was run.

A follow-up `SteerSequenceFraction=0.10` run was first launched with the wrong
mixed replay directory. The command used component weights for `corr048` and
`late045t24mth03` while leaving `-MixedReplayDir` at the default target22+steer2
manifest, so training exited before checkpoint creation. This is classified as
run-argument mismatch, not model/CUDA failure. The fixed run was:

```text
candidate: 620a897_late045_logit15_latecap0825_w025_ssf010_fixed
mixed replay: mixed_t22_steer2_corr048_late045t24mth03_w002
SteerSequenceFraction: 0.10
RouteLateActionLogitCapWeight: 0.25
```

Cross-replay gate passed:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:240,15:222,10:48,16:6,19:6}`.
  - moving_steer_fraction `0.1149425287`.
  - dominant_action_fraction `0.4597701149`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:326,15:215,4:52,10:50,1:4}`.
  - moving_steer_fraction `0.1638330757`.
  - dominant_action_fraction `0.5038639876`.
  - latent test accuracy `0.8846153617`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 eval was then run once from the clean clone and
failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `131/512`.
- route_progress: `0.5641025641`.
- collision/offroad/stuck: `1/0/0`.
- action_counts `{7:4,10:12,15:112,16:2,19:1}`.
- failure tail: action10 mild-right through route `0.5385`, then action16 at
  steps 128-129, then action19 at step 130 with collision.

Decision:

- `SteerSequenceFraction=0.10` fixed the offline steer2 gate but regressed the
  real clean route from the previous `0.7692/0.7949` bracket back to `0.5641`.
- The useful region is likely between `0.10` and `0.15`, or requires a more
  route-local right-steer cap around route `0.50-0.57` without destroying the
  late `0.74-0.80` behavior.
- Next candidate is an offline-only midpoint
  `620a897_late045_logit15_latecap0825_w025_ssf0125`; official 512 should be
  run only if its cross-replay gate passes.

The midpoint `620a897_late045_logit15_latecap0825_w025_ssf0125` was rejected
offline:

- target22:
  - `recommended_for_clean_rerun=false`.
  - action_counts `{15:458,10:28,16:22,13:8,19:6}`.
  - dominant_action_fraction `0.8773946360`, above the `0.85` gate.
  - moving_steer_fraction `0.1226053640`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=false`.
  - action_counts `{15:431,10:148,7:38,4:21,12:9}`.
  - moving_steer_fraction `0.2751159196`, above the `0.20` gate.
  - dominant_action_fraction `0.6661514683`.
  - latent test accuracy `0.8999999762`.
- gate: `ready_for_clean_eval=false`.

Decision:

- Do not run CARLA for `ssf0125`.
- The midpoint overshoots both action15 dominance and steer2 moving-steer
  false positives, while `ssf010` passes offline but collides early in real
  CARLA at route `0.5641` after action16/action19.
- The next offline-only candidate is
  `620a897_late045_logit15_midcap1619_w025_ssf010`, reusing `ssf010` but
  moving the second route-local logit cap to actions `16,19` over route
  `0.50-0.57`.

`620a897_late045_logit15_midcap1619_w025_ssf010` passed the cross-replay gate:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:240,15:222,10:48,19:6,13:4,16:2}`.
  - moving_steer_fraction `0.1149425287`.
  - dominant_action_fraction `0.4597701149`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:321,15:228,4:53,10:42,1:3}`.
  - moving_steer_fraction `0.1514683153`.
  - dominant_action_fraction `0.4961360124`.
  - latent test accuracy `0.8846153617`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 was run once and failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `131/512`.
- route_progress: `0.5641025641`.
- collision/offroad/stuck: `1/0/0`.
- action_counts `{15:112,10:12,7:4,13:2,19:1}`.
- failure tail: action10 through route `0.5385`, then action13 at steps
  128-129, then action19 at step 130 with collision.

Decision:

- The midcap removed action16 from the official failure tail, but the policy
  still escalated to action19 at the same route bracket.
- Next offline-only candidate is
  `620a897_late045_logit15_midcap1319_w025_ssf010`, replacing the mid-route cap
  ids with `13,19` over route `0.50-0.57`.

`620a897_late045_logit15_midcap1319_w025_ssf010` also passed cross replay:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:240,15:222,10:48,16:8,19:4}`.
  - moving_steer_fraction `0.1149425287`.
  - dominant_action_fraction `0.4597701149`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:321,15:228,4:54,10:41,1:3}`.
  - moving_steer_fraction `0.1514683153`.
  - dominant_action_fraction `0.4961360124`.
  - latent test accuracy `0.8846153617`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 was run once and failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `131/512`.
- route_progress: `0.5641025641`.
- collision/offroad/stuck: `1/0/0`.
- action_counts `{15:112,10:12,7:4,16:2,19:1}`.
- failure tail: action10 through route `0.5385`, then action16 at steps
  128-129, then action19 at step 130 with collision.

Decision:

- Swapping mid-route cap ids among `13,16,19` at weight `0.25` does not change
  the real CARLA failure bracket enough.
- The next offline-only candidate is
  `620a897_late045_logit15_midcap131619_w1_ssf010`, a stronger route
  `0.50-0.57` cap over `13,16,19`.

`620a897_late045_logit15_midcap131619_w1_ssf010` passed cross replay:

- target22:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:242,15:222,10:54,19:4}`.
  - moving_steer_fraction `0.1111111111`.
  - dominant_action_fraction `0.4636015326`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - `recommended_for_clean_rerun=true`.
  - action_counts `{7:336,15:224,4:62,10:19,1:6}`.
  - moving_steer_fraction `0.1344667697`.
  - dominant_action_fraction `0.5193199382`.
  - latent test accuracy `0.8846153617`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 was run once and failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `offroad`.
- steps: `149/512`.
- route_progress: `0.5384615385`.
- collision/offroad/stuck: `0/1/0`.
- action_counts `{15:80,7:47,10:22}`.
- failure tail: action10 repeated through route `0.5385`, then offroad at
  step 148 without collision.

Decision:

- Stronger mid-route cap removed the action16/action19 collision tail, but
  converted the bracket into a slow mild-right offroad at route `0.5385`.
- Stop route-local cap-only escalation for this bracket. The next useful work
  is to collect or synthesize a route `0.50-0.57` correction replay/label window
  that teaches the policy the needed moving turn trajectory, then run the same
  cross-replay gate before another official 512.

## 2026-05-26: route `0.45-0.56` privileged correction replay attempts

`354c0b6_t22_steer2_late045_mid048_w002_ssf010` was rejected offline:

- mixed replay: `mixed_t22_steer2_late045_mid048_w002`.
- component weights:
  `target22=1,steer2=1,late045t24mth03=0.02,mid048t24mth03=0.02`.
- target22 passed, but steer2 failed:
  - moving_steer_fraction `0.227202473`, above the `0.20` gate.
  - gate: `ready_for_clean_eval=false`.

`354c0b6_t22_steer2_late045_mid048_w001_ssf010` passed offline:

- component weights:
  `target22=1,steer2=1,late045t24mth03=0.02,mid048t24mth03=0.01`.
- target22:
  - action_counts `{1:2,4:22,7:58,10:8,12:2,15:420,16:4,19:6}`.
  - dominant_action_fraction `0.8045977011`.
  - moving_steer_fraction `0.0842911877`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - action_counts `{1:9,4:73,7:227,10:13,12:1,15:322,16:2}`.
  - dominant_action_fraction `0.4976816074`.
  - moving_steer_fraction `0.1514683153`.
  - latent test accuracy `0.8923076987`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 was run once and failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `130/512`.
- route_progress: `0.5384615385`.
- collision/offroad/stuck: `1/0/0`.
- failure tail: high speed `58-61 km/h` approach, action15/action7 straight
  throttle through route `0.46`, then action10/action13/action16/action19.
  Collision occurred on action19 at step 129.

Decision:

- The weak `mid048` replay kept the offline gate healthy but did not fix the
  real bottleneck; the car still reached the route `0.5385` bracket too fast.
- Do not continue cap-only changes for this bracket.
- Collect PDF Stage I style privileged replay earlier in the approach and
  verify offline before another official 512.

Replay collection attempts:

- `windows_privileged_354c0b6_corr048_midroute_t18_minthr00_min16` failed:
  - all 12 attempts ended with collision.
  - each attempt produced only 14 recorded steps after route `0.48`, below
    `MinRecordedSteps=16`.
  - max_route_progress `0.5641025641`.
- `windows_privileged_354c0b6_corr045_midroute_t10_minthr00_min12` succeeded:
  - episodes `4`.
  - episode_lengths `[15,15,15,15]`.
  - total_steps `60`.
  - action_counts `{2:24,5:29,8:7}`.
  - route window `0.4615-0.5385`.
  - clean_gate_evidence `false`.

`354c0b6_t22_steer2_late045_corr045t10_w003_seq12_ssf010` was rejected
offline:

- component weights:
  `target22=1,steer2=1,late045t24mth03=0.02,corr045t10mth00=0.03`.
- sequence_length `12`, because the new correction episodes are 15 steps long.
- target22 failed:
  - action_counts `{2:16,5:38,8:4,15:464}`.
  - dominant_action_fraction `0.8888888889`, above the `0.85` gate.
  - moving_steer_fraction `0.0`, below the gate.
  - zero_throttle_fraction `0.1111111111`.
- steer2 passed action probe but showed aggressive late moving steer.
- gate: `ready_for_clean_eval=false`.

`354c0b6_t22_steer2_late045_corr045t10_w001_seq12_ssf010` was also rejected
offline:

- component weights:
  `target22=1,steer2=1,late045t24mth03=0.02,corr045t10mth00=0.01`.
- target22 passed:
  - action_counts `{7:34,10:26,13:8,15:428,16:20,19:6}`.
  - dominant_action_fraction `0.8199233716`.
  - moving_steer_fraction `0.1149425287`.
  - zero_throttle_fraction `0.0`.
- steer2 failed:
  - action_counts `{1:3,4:35,7:78,10:103,12:11,15:416,16:1}`.
  - moving_steer_fraction `0.2364760433`, above the `0.20` gate.
  - latent test accuracy `0.9230769277`.
- gate: `ready_for_clean_eval=false`.

Decision:

- The zero-throttle `t10/min0` correction replay is useful diagnostically but
  too sharp for the current mixed BC recipe: at weight `0.03` it collapses
  target22 moving steer; at weight `0.01` it preserves target22 but creates
  steer2 false-positive moving steer.
- Do not run official CARLA for either `corr045t10` candidate.
- Next clean-first direction should keep the PDF Stage I privileged replay path
  but avoid all-zero-throttle labels: either collect a longer earlier window
  with a moving throttle floor, or add an offline label/window filter before
  mixing, then re-run target22/steer2 cross replay gate.

## 2026-05-26: moving-throttle route `0.45-0.72` correction replay attempts

`windows_privileged_9398470_corr045_midroute_t14_minthr03_min16` was collected
from the latest `mid048_w001` checkpoint:

- target_speed_kmh `14`.
- min_recorded_throttle `0.3`.
- record_after_route_progress `0.45`.
- episodes `4`.
- episode_lengths `[16,16,16,16]`.
- action_counts `{7:8,10:16,13:16,16:16,19:8}`.
- route window `0.4615-0.5385`.
- clean_gate_evidence `false`.

`9398470_t22_steer2_late045_corr045t14_w001_ssf010_msfpa2` passed the
target22/steer2 offline gate:

- component weights:
  `target22=1,steer2=1,late045t24mth03=0.02,corr045t14mth03=0.01`.
- sequence_length `16`.
- imitation_balance_power `0.25`.
- moving_steer_false_positive_weight `2.0`.
- moving_steer_nonmoving_margin_weight `1.0`.
- target22:
  - action_counts `{1:2,4:20,7:394,10:8,12:2,13:2,15:86,19:8}`.
  - dominant_action_fraction `0.7547892720`.
  - moving_steer_fraction `0.0804597701`.
  - zero_throttle_fraction `0.0`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - action_counts `{1:5,4:59,7:491,10:8,15:82,16:2}`.
  - dominant_action_fraction `0.7588871716`.
  - moving_steer_fraction `0.1143740340`.
  - zero_throttle_fraction `0.0`.
  - latent test accuracy `0.9076923132`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 was run once and failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `collision`.
- steps: `307/512`.
- route_progress: `0.6666666667`.
- collision/offroad/stuck: `1/0/0`.
- episode_return `17.9571528668`.
- failure tail: route `0.59-0.6667`, speed around `18 km/h`, mostly action7
  straight throttle with occasional action4, then collision on action7.

Decision:

- This was a real improvement over the previous route `0.5385` failure: the
  policy reached route `0.6667` while keeping speed near `18 km/h`.
- The next bottleneck moved to the later straight-hold region. Collect a
  moving-throttle correction replay starting around route `0.60`.

`windows_privileged_9398470_corr060_late_t14_minthr03_min16` was then
collected from the `corr045t14` candidate:

- target_speed_kmh `14`.
- min_recorded_throttle `0.3`.
- record_after_route_progress `0.60`.
- episodes `4`.
- episode_lengths `[29,24,24,24]`.
- action_counts `{7:24,10:51,13:20,16:6}`.
- route window `0.6154-0.6667`.
- clean_gate_evidence `false`.

`9398470_t22_steer2_corr045t14_corr060t14_w001_ssf010_msfpa2` also passed
the target22/steer2 offline gate:

- component weights:
  `target22=1,steer2=1,corr045t14mth03=0.01,corr060t14mth03=0.01`.
- sequence_length `16`.
- imitation_balance_power `0.25`.
- moving_steer_false_positive_weight `2.0`.
- moving_steer_nonmoving_margin_weight `1.0`.
- target22:
  - action_counts `{4:26,7:422,12:2,15:64,16:6,19:2}`.
  - dominant_action_fraction `0.8084291188`.
  - moving_steer_fraction `0.0689655172`.
  - zero_throttle_fraction `0.0`.
  - latent test accuracy `0.9904761910`.
- steer2:
  - action_counts `{1:1,4:66,7:512,12:2,15:66}`.
  - dominant_action_fraction `0.7913446677`.
  - moving_steer_fraction `0.1066460587`.
  - zero_throttle_fraction `0.0`.
  - latent test accuracy `0.9153845906`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 was run once and failed:

- source: `windows_carla`.
- status: `FAIL`.
- failure_class: `driving`.
- done_reason: `offroad`.
- steps: `339/512`.
- route_progress: `0.7179487179`.
- collision/offroad/stuck: `0/1/0`.
- episode_return `25.0445291091`.
- failure tail: route `0.6667-0.7179`, speed around `18-19.5 km/h`, long
  action7 straight holds with action4 corrections, then action12/action15 high
  throttle reappeared near route `0.692-0.718`; offroad occurred on action7.

Decision:

- The `corr060` replay moved the failure again, from route `0.6667` collision
  to route `0.7179` offroad. This is progress but not clean pass.
- Do not start raw/fusion.
- Next clean-first direction is route `0.69-0.72` privileged correction or a
  route-local high-throttle/straight-hold regularizer that specifically reduces
  action12/action15 and excessive action7 straight holding in this later window
  while preserving target22/steer2 gates.

## 2026-05-26: route `0.64-0.72` correction mix attempts

`windows_privileged_c6b03e7_corr069_late_t14_minthr03_min16` was attempted
from the `corr045+corr060` candidate:

- target_speed_kmh `14`.
- min_recorded_throttle `0.3`.
- record_after_route_progress `0.69`.
- result: only `1/4` successful episodes within `12` attempts.
- saved episode length `[17]`.
- action_counts `{4:1,7:16}`.
- route window `0.6923-0.7179`.
- decision: keep as diagnostic only; too few episodes for the main mix.

`windows_privileged_c6b03e7_corr064_late_t14_minthr03_min16` was then
collected from the same warmup:

- target_speed_kmh `14`.
- min_recorded_throttle `0.3`.
- record_after_route_progress `0.64`.
- episodes `4`.
- episode_lengths `[19,19,19,19]`.
- action_counts `{7:24,10:52}`.
- route window `0.6410-0.6667`.
- clean_gate_evidence `false`.

`c6b03e7_t22_steer2_corr060_corr064_w001_ssf010_msfpa2` passed the target22
and steer2 offline gate after replacing `corr045` with `corr064`:

- component weights:
  `target22=1,steer2=1,corr060t14mth03=0.01,corr064t14mth03=0.01`.
- target22 action probe:
  - action_counts `{1:4,4:20,7:78,10:4,13:8,15:404,16:4}`.
  - dominant_action_fraction `0.7739463602`.
  - moving_steer_fraction `0.0766283525`.
- steer2 action probe:
  - action_counts `{1:3,4:56,7:193,15:395}`.
  - dominant_action_fraction `0.6105100464`.
  - moving_steer_fraction `0.0911901082`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 failed:

- done_reason `collision`.
- steps `144/512`.
- route_progress `0.5897435897`.
- collision/offroad/stuck `1/0/0`.
- action_counts `{4:20,7:124}` in the clean run tail region.
- decision: reject as a real regression. Dropping `corr045` reopened the
  earlier mid-route geometry.

To keep `corr045` while adding late correction, a nested correction mix was
created:

- `windows_privileged_c6b03e7_corrmix_045_060_064`.
- components: `corr045`, `corr060`, `corr064`.
- episodes `12`.
- total_steps `241`.

`c6b03e7_t22_steer2_corrmix045060064_w002_ssf010_msfpa2` used
`corrmix=0.02` and was rejected offline:

- target22 dominant_action_fraction `0.9272030651`.
- steer2 dominant_action_fraction `0.8995363215`.
- failure: action7 dominance. The component weight applies per episode, so
  `0.02` over-weighted the 12-episode correction mix.
- decision: do not run official CARLA.

`c6b03e7_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2` corrected the
episode-level weight:

- component weights: `target22=1,steer2=1,corrmix=0.0067`.
- target22 action probe:
  - action_counts `{4:30,7:412,15:72,16:4,19:4}`.
  - dominant_action_fraction `0.7892720307`.
  - moving_steer_fraction `0.0727969349`.
- steer2 action probe:
  - action_counts `{4:81,7:488,15:78}`.
  - dominant_action_fraction `0.7542503864`.
  - moving_steer_fraction `0.1251931994`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 failed:

- done_reason `offroad`.
- steps `339/512`.
- route_progress `0.7179487179`.
- collision/offroad/stuck `0/1/0`.
- action_counts `{4:46,7:254,15:39}`.
- tail: route `0.6667-0.7179`, speed around `18 km/h`, mostly action7 with
  short action4 runs. action12/high-throttle failure did not reappear.
- decision: this recovered the previous best route but did not solve the
  late straight-hold/offroad bottleneck.

`windows_privileged_c6b03e7_corr066_late2_t14_minthr03_min16` was collected
from the `corrmix045060064_w00067` candidate:

- target_speed_kmh `14`.
- min_recorded_throttle `0.3`.
- record_after_route_progress `0.66`.
- result: only `2/4` successful episodes within `12` attempts.
- saved episode_lengths `[32,32]`.
- action_counts `{4:7,7:57}`.
- route window `0.6667-0.7179`.
- clean_gate_evidence `false`.

`c6b03e7_t22_steer2_corrmix045060066_w0008_ssf010_msfpa2` replaced `corr064`
with the direct late `corr066` component and passed offline:

- correction mix:
  - `windows_privileged_c6b03e7_corrmix_045_060_066`.
  - components: `corr045`, `corr060`, `corr066`.
  - episodes `10`.
  - total_steps `229`.
- component weights: `target22=1,steer2=1,corrmix=0.008`.
- target22 action probe:
  - action_counts `{4:30,7:414,15:72,16:2,19:4}`.
  - dominant_action_fraction `0.7931034483`.
  - moving_steer_fraction `0.0689655172`.
- steer2 action probe:
  - action_counts `{1:2,4:66,7:501,15:78}`.
  - dominant_action_fraction `0.7743431221`.
  - moving_steer_fraction `0.1051004637`.
- gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 failed:

- done_reason `collision`.
- steps `311/512`.
- route_progress `0.6666666667`.
- collision/offroad/stuck `1/0/0`.
- action_counts `{1:1,4:29,7:242,15:39}`.
- tail: route `0.6154-0.6667`, mostly action7 straight at about `18 km/h`,
  collision on action7 at route `0.6667`.
- decision: reject as a regression relative to `corrmix045060064_w00067`.
  The direct `corr066` tail component shifts the failure back to the `0.6667`
  collision bracket.

Current best restart point:

- Best recent official behavior is
  `c6b03e7_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2`.
- It reaches route `0.7179` and fails by offroad, with high-throttle action12
  removed.
- Next clean-first candidate should preserve that checkpoint recipe and target
  the route `0.66-0.72` action7 straight-hold/offroad issue directly, without
  replacing `corr064` by `corr066` or dropping `corr045`.

## 2026-05-26 Late Action7/4 Cap Resume

Windows remote execution resumed through the documented Mac -> userspace
Tailscale SOCKS -> Windows SSH path, using the official clean clone:

- Windows node: `100.87.219.58`.
- Windows clean clone:
  `C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656`.
- CUDA guard passed on `NVIDIA GeForce RTX 3080`, torch `2.11.0+cu126`.
- CARLA was stopped immediately after each official eval; final GPU status
  after cleanup was `57-58 C`, `0 %` utilization, `562 MiB` memory.

`8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap7w010_r066072`
preserved the current best `corrmix045060064_w00067` recipe and added a narrow
route-window action-logit cap on action `7`:

- `RouteLateActionLogitCapWeight=0.10`.
- `RouteLateActionLogitCapIds=7`.
- `RouteLateActionLogitCapMinRouteProgress=0.66`.
- `RouteLateActionLogitCapMaxRouteProgress=0.72`.
- `RouteThrottleCeilingWeight=4.0`, `RouteThrottleCeiling=0.3`,
  route window `0.64-0.72`.
- target22/steer2 cross replay gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 failed:

- run output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap7w010_r066072_official512`.
- done_reason `offroad`.
- steps `342/512`.
- route_progress `0.7435897436`.
- collision/offroad/stuck `0/1/0`.
- action_counts `{4:46,7:258,15:38}`.
- tail: speed stayed around `18 km/h`; route `0.6667-0.7436` was still mostly
  action `7` with intermittent action `4`; final offroad happened on action
  `7`.

Decision: this improved over the prior `0.7179` offroad, but did not solve the
late straight-hold/offroad failure. A broader right-steer-margin variant was
briefly launched, then stopped before completion after subagent review because
it was a less conservative next move.

`8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap47w010_r066076`
then kept the same replay/checkpoint recipe and changed only the local late cap:

- `RouteLateActionLogitCapWeight=0.10`.
- `RouteLateActionLogitCapIds=4,7`.
- `RouteLateActionLogitCapMinRouteProgress=0.66`.
- `RouteLateActionLogitCapMaxRouteProgress=0.76`.
- `RouteThrottleCeilingWeight=4.0`, `RouteThrottleCeiling=0.3`,
  route window `0.64-0.70`.

Offline gate passed:

- target22 action probe:
  - action_counts `{4:26,7:412,10:4,13:4,15:70,19:6}`.
  - dominant_action_fraction `0.7892720307`.
  - moving_steer_fraction `0.0766283525`.
  - latent test accuracy `0.9904761910`, majority `0.7931034483`.
- steer2 action probe:
  - action_counts `{4:81,7:490,15:76}`.
  - dominant_action_fraction `0.7573415765`.
  - moving_steer_fraction `0.1251931994`.
  - latent test accuracy `0.9153845906`, majority `0.6831530139`.
- cross replay gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 failed:

- run output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap47w010_r066076_official512`.
- done_reason `offroad`.
- steps `344/512`.
- route_progress `0.7435897436`.
- collision/offroad/stuck `0/1/0`.
- action_counts `{4:47,7:259,15:38}`.
- tail: final band still alternated action `7` straight and action `4` left;
  the terminal offroad was on action `4` at route `0.7435897436`.

Decision: reject as not materially better than `latecap7w010_r066072`. The
failure has now moved from a generic route `0.66-0.72` straight-hold issue to a
more specific post-`0.72` left/straight tail. The next clean-first CUDA
candidate should be a narrow post-`0.72` correction and should reject any
official result that fails at or below route `0.7436` with the same action
`4/7` tail.

Subagent review recommended one final narrow post-`0.72` probe before moving to
new replay collection: keep `latecap47w010_r066076` and add a weak route-local
action override toward action `10` only in route `0.72-0.755`. Action `10` is
the mild forward-right action (`throttle=0.3`, `steer=+0.1`), so it was less
risky than broad right-steer margin or strong-right actions `13/16/19`.

`8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap47w010_r066076_o10w025_r0720755`
trained on Windows CUDA and passed the cross replay gate:

- action override:
  - `RouteActionOverrideWeight=0.25`.
  - `RouteActionOverrideId=10`.
  - `RouteActionOverrideMinRouteProgress=0.72`.
  - `RouteActionOverrideMaxRouteProgress=0.755`.
- target22 action probe:
  - action_counts `{4:26,7:412,10:4,13:4,15:70,19:6}`.
  - dominant_action_fraction `0.7892720307`.
  - moving_steer_fraction `0.0766283525`.
  - latent test accuracy `0.9904761910`, majority `0.7931034483`.
- steer2 action probe:
  - action_counts `{4:81,7:490,15:76}`.
  - dominant_action_fraction `0.7573415765`.
  - moving_steer_fraction `0.1251931994`.
  - latent test accuracy `0.9153845906`, majority `0.6831530139`.
- cross replay gate: `ready_for_clean_eval=true`.

Official Windows CARLA 512 failed:

- run output:
  `RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap47w010_r066076_o10w025_r0720755_official512`.
- done_reason `offroad`.
- steps `339/512`.
- route_progress `0.7179487179`.
- collision/offroad/stuck `0/1/0`.
- action_counts `{4:46,7:255,15:38}`.
- tail: action `10` did not appear in closed-loop; route `0.6923-0.7179`
  remained action `7` plus action `4`, and terminal offroad happened on action
  `7`.

Decision: reject as a regression relative to `latecap47w010_r066076`. The
route-local override was too weak to change the closed-loop action sequence and
returned to the older `0.7179` offroad bracket. Current best official behavior
remains `latecap47w010_r066076` at route `0.7435897436`; next work should
collect or build a post-`0.72` correction component instead of adding another
small replay-only override to the same mixed replay.
