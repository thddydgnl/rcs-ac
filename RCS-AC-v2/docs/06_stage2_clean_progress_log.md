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
