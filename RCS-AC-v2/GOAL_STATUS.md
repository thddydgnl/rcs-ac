# RCS-AC v2 Goal Status

Updated: 2026-05-28

## Active Objective

Implement RCS-AC-v2 as a clean-first Raw2Drive PDF-core system:

```text
Primary goal document:
docs/08_raw2drive_pdf_core_clean_first_master_plan.md

Base paper:
docs/reference/2505.16394v2.pdf
```

The active direction fixes the Raw2Drive contracts that are explicit in the PDF:
43-channel privileged semantic BEV plus ego info, 39 discrete actions, raw
multi-view RGB plus IMU baseline input, rollout guidance, head guidance, no raw
reward/continue direct heads, and no privileged leakage at raw inference.
Unspecified engineering details such as BEV resolution, coverage probes,
route/lane diagnostics, auxiliary-only supervision, and Windows CARLA probe/gate
workflow may be filled for clean-first progress. PDF-explicit core results and
clean-stability fill results must be recorded separately.

## Current Stage

```text
Stage H3: 32x32 privileged clean-stability fill after H2 PDF-core rejection
```

Status:

```text
H3_32X32_CLEAN_STABILITY_FILL_IN_PROGRESS__ROUTE0667_CORRECTION_BALANCE_SEARCH
```

Last completed stage:

```text
Stage H0: PDF-core clean-first rebase PASS
Stage H1: 32x32 privileged observation/replay-smoke PASS
Stage H2: strict 32x32 PDF-faithful replay baseline OFFLINE_REJECTED
```

## Required Reading Before Coding

1. `docs/08_raw2drive_pdf_core_clean_first_master_plan.md`
2. `docs/02_persistent_subagent_goal_protocol.md`
3. `docs/03_windows_carla_remote_execution_workflow.md`
4. `docs/04_git_sync_artifact_cleanup_policy.md`
5. `docs/05_clean_performance_goal_prompt.md`
6. `docs/01_goal_driven_implementation_runbook.md`
7. `docs/00_raw2drive_exact_implementation_plan.md`
8. `docs/reference/2505.16394v2.pdf`

## Sub-Agent Protocol State

Mode:

```text
Lite by default. Escalate to Full if contract/model/dataset/leakage/baseline/gate uncertainty appears.
```

Required agents:

```text
Stage entry and work-packet start: Agent A + Agent C.
Add Agent B when editing code/config/test/logging/checkpoint wiring.
```

Last findings summary:

```text
2026-05-26 Windows remote resumed through the documented userspace Tailscale
SOCKS SSH path to 100.87.219.58. CARLA Town04 readiness and RTX 3080 CUDA guard
passed. Commits 2ee1b4c, 5b7ee99, and b462386 fixed mixed-candidate argument
handling, rejected-gate summarization, and route-bin action diagnostics. The
late-right candidate and target22 fallback were rejected offline. Two all-steer
route cap candidates improved steer2 moving-steer fraction to 0.2194744977 and
then 0.2117465224, but still failed the 0.20 action gate. A late-window-only
cap also stayed at 0.2117465224, proving the remaining excess is mid-route
false-positive steering. BF7 combined the late cap with false-positive and
non-moving margin losses and passed the target22/steer2 cross-replay gate, but
official 512 clean failed at route_progress=0.5384615385. Diagnostic forbidden
action runs for 19, then 16/19, then 16/19/21 shifted collision to offroad but
did not improve route progress. The subsequent throttle-ceiling, high-throttle
mass-cap, action15-mask, and route-window action-logit-cap candidates were all
rejected offline before official clean eval; their common failure mode is
steer2 moving-steer fraction just above the 0.20 gate. No raw/fusion work should
start before privileged 512 clean is stable. The corr050 replay branch then
showed identical steer2 distributions across target-speed/weight variants. A
component-sampler guard was added so replay training now records included and
dropped short episodes per component and fails fast when a positive-weight
component has no episodes long enough for the configured sequence_length.
After pulling commit 32a65cb on the Windows clean clone, the diagnostic guard
confirmed both mixed_t22_steer2_corr050_t12 and mixed_t22_steer2_corr050_t16
have corr050 episodes of length 10 while sequence_length=16, so corr050 was
filtered out of training entirely. The next clean-first action is to recollect
the collision-window component with MinRecordedSteps >= 16 or longer contiguous
windows before training another corr050 candidate. A corrected route-progress
0.48 component was then collected at commit 33d1705 with 4 episodes, 68 total
steps, episode_lengths=[17,17,17,17], and action_counts={2:16,5:32,8:20}.
The mixed replay included corr048 with no dropped short episodes, but the
candidate `33d1705_corr048_t16_min16_w025_bf7base` was rejected offline:
target22 and steer2 latent probes passed, while both action probes failed
recommended_for_clean_rerun due dominant action 7 fractions above 0.85. No
official 512 clean eval was run for this candidate. A follow-up corr048=0.05
sweep reduced action 7 collapse but shifted the failure to action 15/right-turn
dominance: ssf005 passed steer2 but failed target22 with dominant_action_fraction
0.8812260536; ssf010 failed target22 at 0.8735632184 and steer2 moving_steer
0.3137557960; ssf015 failed target22 at 0.8620689655 and steer2 moving_steer
0.3338485317. No official CARLA eval was run for these offline-rejected
candidates. Next work should stop plain corr048 weight/ssf sweeps and add or
use an action15/right-turn dominance control that preserves steer2 false-positive
limits. The first direct control candidate,
`8b78015_corr048_w005_ssf005_cap15w1c82`, trained on Windows CUDA with
`ActionMassCapWeight=1.0` and `ActionMassCaps=15=0.82`. It preserved the
steer2 action gate (`recommended_for_clean_rerun=true`,
moving_steer_fraction=0.1051004637, dominant_action_fraction=0.6769706337),
but target22 stayed just over the dominant-action gate
(`action_counts={15:460,4:34,10:6,13:2,19:2,1:2,7:16}`,
dominant_action_fraction=0.8812260536). The cross-replay gate rejected the
candidate before official CARLA eval with failure
`target22: action probe recommended_for_clean_rerun is not true`.
Stronger `ActionMassCapWeight=4.0, ActionMassCaps=15=0.80` produced the same
action distributions, so global action-mass capping is not an effective argmax
control in this setting. The next candidate,
`8b78015_corr048_w005_ssf005_logit15w05m0`, used the existing
`RouteActionLogitCap` hook on action 15 with weight 0.5 and margin 0.0. It
passed the target22/steer2 cross-replay gate with `ready_for_clean_eval=true`:
target22 dominant_action_fraction=0.8467432950, steer2
moving_steer_fraction=0.0927357032, and both latent margins above 0.15.
Official Windows CARLA 512 eval was then run from the clean clone after CARLA
readiness. It failed as `source=windows_carla`, `status=FAIL`,
`failure_class=driving`, `done_reason=collision`, steps=146/512,
route_progress=0.6666666667, collision_count=1, offroad_count=0,
stuck_count=0. Clean action_counts were {15:117,7:17,4:10,12:2}; the last two
steps selected action12 (`throttle=0.7`, `steer=-0.1`) and the collision
occurred on the second action12. CARLA was stopped after the run. A follow-up
`8b78015_corr048_w005_ssf005_logit1215w05m0` candidate expanded the same
argmax-level logit cap to actions 12 and 15. It preserved the offline gate
(`ready_for_clean_eval=true`; target22 dominant_action_fraction=0.8467432950,
steer2 moving_steer_fraction=0.0896445131), but official 512 still failed as
collision at steps=146/512 and route_progress=0.6666666667. Clean action_counts
were {15:116,7:18,4:10,1:1,12:1}; the only action12 occurred at step145 and
collided immediately. CARLA was stopped after the run. A stronger
`logit1215w2m0` candidate was rejected offline because both target22 and steer2
collapsed back to dominant action7. The bracket midpoint
`logit1215w1m0` passed offline, but official 512 failed earlier at
route_progress=0.5384615385 with collision+offroad after action19
(`throttle=0.3`, `steer=0.5`) repeated at steps 144-147. This means simply
raising the full-route 12/15 logit cap trades the late action12 collision for
earlier right-oversteer. Stop full-route logit-weight sweeps; next candidate
should keep the offline-passing w0.5 action15 control and add a route-local
late throttle/action12 control around route_progress 0.64-0.70. The first
route-local throttle-ceiling follow-up,
`8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4`, kept the full-route
action15 logit cap and added `RouteThrottleCeilingWeight=4.0`,
`RouteThrottleCeiling=0.3` for route_progress 0.64-0.70. It passed the
target22/steer2 cross-replay gate and removed clean-run action12 from the
failure tail, but official Windows CARLA 512 still failed as
`done_reason=collision`, steps=146/512, route_progress=0.6666666667,
collision_count=1, offroad_count=0, stuck_count=0. Clean action_counts were
{15:116,7:19,4:10,1:1}; the final step selected action7
(`throttle=0.3`, `steer=0.0`) at route_progress=0.6667 and collided. CARLA was
stopped after the run and RTX 3080 reported 65C/0% utilization. This narrows
the next patch away from high-throttle action12 and toward a late-route
collision-window correction/replay or local action selection change that
handles the 0.64-0.70 geometry without reintroducing action19 right-oversteer.
Following that decision, two late056 warmup correction replays were collected
from the Windows clean clone with the lateceil checkpoint-filled config as both
runtime and warmup policy: `late056_collision_window_t16_min16` and
`late056_collision_window_t24_min16`. Both produced sufficient min16 episodes,
but their labels were mostly zero-throttle steering actions; t24 had 142 total
steps with action_counts {2:4,28:48,30:18,31:25,34:47}. To avoid a blind CARLA
run, the t24 replay was mixed at a very low component weight
(`late056t24=0.02`) with target22, steer2, and corr048 while preserving the
action15 logit cap plus late throttle ceiling. Windows CUDA training completed
and steer2 passed offline, but target22 failed with action15 dominance
(`action_counts={15:462,4:42,7:12,12:4,1:2}`,
dominant_action_fraction=0.8850574713). The gate returned
`ready_for_clean_eval=false`, so no official 512 clean eval was run. This
rejects the current late056 teacher-label route; next work should either change
the late replay labeling heuristic to avoid zero-throttle steering labels or
use a different local action-selection correction, then pass cross replay before
CARLA.

Commit 5bbca2a added `min_recorded_throttle` to the privileged replay
collector so late correction windows do not record zero-throttle steering labels
when the chosen target-speed action family should remain moving. Windows
collection attempts at `record_after_route_progress=0.56` and `0.50` with
`target_speed_kmh=24`, `min_recorded_throttle=0.3`, and `min_recorded_steps=16`
failed to produce enough successful episodes before collision, but
`record_after_route_progress=0.45` succeeded with 4 episodes, 64 total steps,
episode_lengths=[16,16,16,16], and action_counts={7:8,10:12,13:16,16:16,19:12}.
The mixed candidate
`5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil` passed the
target22/steer2 cross-replay gate. Official Windows CARLA 512 without an action
mask improved the clean route progress to `0.7692307692` but failed by
collision at step 360 after a right-steer tail ending in action19. A follow-up
offline mask19 probe also passed cross replay; its official 512 run again
failed at `route_progress=0.7692307692`, steps=350/512, collision_count=1,
offroad_count=0, stuck_count=0, with final actions action10 -> action13 ->
action16. CARLA was stopped after the run and RTX 3080 was idle at 65C.
This is progress over the previous 0.6667 collision but does not close Stage 2.
The next clean-first patch should target the later `0.74-0.77` route window,
especially the transition from mild right steer to action16/right-steer, before
another official 512 run.

Follow-up action-mask probes showed that broad forbidden-action masking is only
a diagnostic, not yet the fix. Masking actions `16,19` passed cross replay and
also passed a 128-step smoke run, but the real 512 run still failed at
`route_progress=0.7692307692`, steps=361/512, with the final collision on
action8 (`throttle=0.0`, `steer=0.6`). Masking `8,16,19` again passed cross
replay and improved the official 512 failure to `route_progress=0.7948717949`,
but it failed at step 360 on action25 (`throttle=0.0`, `steer=-1.0`). This
confirms the remaining issue is not one specific action id; the policy is
falling into zero-throttle sharp-steer alternatives in the late route window.
Stop extending broad masks as the main strategy. The next implementation should
add a late-window correction/training constraint that suppresses zero-throttle
sharp-steer fallback actions while preserving moving mild steering.
Commit 620a897 added a route-local late action logit cap for diagnostic
clean-first training. Late-cap weights 0.5 and 0.25 with
SteerSequenceFraction=0.15 were rejected offline because steer2 moving-steer
fraction stayed at 0.2132921175, slightly above the 0.20 gate. A corrected
SteerSequenceFraction=0.10 run on the proper
mixed_t22_steer2_corr048_late045t24mth03_w002 replay passed target22/steer2
cross replay but official Windows CARLA 512 regressed to
route_progress=0.5641025641 and failed by collision after action10 -> action16
-> action19. The next clean-first action is an offline-only midpoint
SteerSequenceFraction=0.125 candidate; do not run another official 512 unless
the cross-replay gate is true.
The midpoint SteerSequenceFraction=0.125 candidate was rejected offline:
target22 dominant_action_fraction=0.8773946360 and steer2
moving_steer_fraction=0.2751159196. The active next candidate is
`620a897_late045_logit15_midcap1619_w025_ssf010`, which keeps the offline-passing
ssf010 setting but moves the second route-local logit cap to actions 16 and 19
over route_progress 0.50-0.57. This is still offline-only until the gate passes.
That midcap1619 candidate passed cross replay but official 512 again failed at
route_progress=0.5641025641; action16 disappeared and the tail became action10
-> action13 -> action19. The active next candidate is
`620a897_late045_logit15_midcap1319_w025_ssf010`, offline-only until its gate
passes.
The midcap1319 candidate also passed cross replay but official 512 returned to
the same action10 -> action16 -> action19 collision at route_progress
0.5641025641. The active next candidate is
`620a897_late045_logit15_midcap131619_w1_ssf010`, a stronger route 0.50-0.57
cap over actions 13,16,19. It is offline-only until its cross-replay gate
passes.
The stronger midcap131619 candidate passed cross replay, but official 512
failed as offroad at route_progress=0.5384615385 with no collision; the tail was
repeated action10 mild-right. Stop cap-only escalation for this bracket. Next
work should collect or build a route 0.50-0.57 correction replay/label window
that teaches the required moving turn trajectory, then gate it offline before
another official clean run.
```

Pending brief:

```text
Stage 2 local smoke, Windows dry-run prep, fake-env adapter-loop contract, Stage 2 shared-core CARLA adapter factory, local checkpoint export/manifest path, v2 branch push, v2-local readiness scripts, dependency probe, simulator-state privileged observation builder, Windows privileged obs probe, privileged replay collector, replay trainer, checkpoint provenance gate, official Windows clean-clone privileged-observation probe, replay-source checkpoint promotion, first 128-step clean classification, BC checkpoint action probe, fast35 replay relabeling, fast35 replay retraining, checkpoint action probe, official 128-step clean rerun, 512 runtime config prep, runner output-path hardening, and official 512-step primary clean run are complete. The first clean run failed as driving/stuck after 64/128 steps, and the first BC checkpoint probe failed because it collapsed to action_id=34 for all 512 replay observations. The fast35 replay fixed the zero-throttle label collapse; the b4c8e58/37a0875 checkpoint probe covered 512 replay actions with action_ids 7/15, reverse_action_count=0, nonzero_throttle_fraction=1.0, and recommended_for_clean_rerun=true. The official clean clone then reached 128/128 steps with source=windows_carla, status=PASS, done_reason=horizon_reached, zero collision/offroad/stuck, and metrics.csv rows=128. The same checkpoint failed the official 512 primary gate as a driving/collision failure at step 310/512 with route_progress=0.5641025641025641, action_counts={7:308,15:3}, no adapter/runtime errors, offroad=0, and stuck=0. Next work packet should patch the smallest collision cause, most likely by collecting/retraining on a longer or failure-window privileged replay that includes steering/turn labels before rerunning 512.
```

Escalation triggers checked:

```text
Stage 0 contract implementation passed local tests.
Stage 1 privileged world-model smoke passed local tests.
Stage 2 local actor-critic/world-model rollout smoke passed local tests.
Stage 2 Windows clean-eval config/CLI/runner dry-run contract passed local tests.
Stage 2 clean-eval fake adapter loop, metrics CSV validator, and non-dry-run classified failure path passed local tests.
Stage 2 shared-core CARLA adapter factory, checkpoint loader, policy wrapper, stage4_v2/direct_3d control bridges, reverse-action guard, and model-failure classification passed local tests.
Stage 2 checkpoint export produces paired world_model/policy checkpoints plus sha256 manifest and config patch; synthetic export remains non-CARLA/non-clean evidence.
Stage 2 checkpoint loader tests now cover missing sha/path, bad sha, missing config/state_dict, unreadable checkpoint path, invalid policy config, policy latent mismatch, bad policy state_dict load, and 8x8/16x16 world-model mismatch.
RCS-AC-v2/.gitignore now makes v2 artifact policy self-contained; check_git_hygiene.sh now fails nonzero if v2 nested .git, heavy artifacts, or >50MB files are found.
Stage 2 runner now has v2-local CARLA readiness scripts, a Windows dependency probe, and a v2-local minimal shared_core env builder; remaining temporary external dependency is sibling shared_core.
Stage 2 clean config now uses the temporary shared-core stage4_v2 action bridge for v1.1 env compatibility; Raw2Drive 39-action policy output remains the v2 policy contract.
Stage 2 privileged-observation probe writes privileged_obs_probe.json with clean_gate_evidence=false before any clean PASS attempt.
Stage 2 local tests now include dependency-probe static contract, v2-local readiness scripts, stage4_v2 action bridge, and privileged_obs_probe.json non-clean evidence checks.
Stage 2 Windows raw-key diagnostic probe at commit 6dacd8d showed the current shared-core env exposes rgb/lidar/route/aux masks but not privileged_bev or privileged_ego.
Stage 2 now builds the pinned 43-channel semantic BEV and 8-field ego vector from CARLA simulator state only; raw-derived/debug privileged tensors are not clean-gate eligible.
Stage 2 Windows archive probe at commit 5b00347 passed dependency check, CARLA Town04 readiness, simulator-state privileged provenance, BEV/ego shape/finite checks, and neutral-step safety checks. The probe remains clean_gate_evidence=false.
Stage 2 replay path now exists: collect_stage2_privileged_replay.ps1 writes simulator-state semantic43/ego8 .npz episodes, train_stage2_privileged_from_replay.ps1 exports replay-source world_model/policy checkpoints and manifest, and clean eval rejects synthetic smoke checkpoints by provenance.
Stage 2 offline diagnosis now includes replay_latent_probe.py and probe_stage2_replay_latents.ps1. The initial steer2 replay candidates all scored linear_probe_test_accuracy=0.7384615540504456 against majority_action_fraction=0.6831530139103554, so the world-model latent only weakly separated action labels. The 7f87656 action-aux candidate improved latent test accuracy to 0.9076923131942749 and passed checkpoint action probe, but still failed the official 512 clean gate by collision at step 177/178.
Stage 2 replay path Windows scratch check at commit e1c9c86 passed dependency probe, replay module imports, 1x8-step privileged replay collection, 2-step replay checkpoint training, and strict replay-source checkpoint loading. This remains clean_gate_evidence=false and is not clean performance evidence.
PDF Stage I strict replay training was added at commit f9a7f44. The trainer now has `pdf_stage1_strict=true`, default `imitation_weight=0.0`, rejects route-local diagnostic losses/action caps/overrides/component weighting in strict mode, and records the strict flag in checkpoint manifests.
Two PDF Stage I strict Windows CUDA candidates were run from clean clone `C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656` on RTX 3080: `f9a7f44_pdf_stage1_strict_actorcritic` with 200/200 steps and `f9a7f44_pdf_stage1_strict_actorcritic_wm2000_pi1000` with 2000/1000 steps. Both are rejected offline and are not clean evidence. The short run collapsed to action 24 on target22/steer2; the long run improved latent separability but collapsed to action 13 on both replays.
The next PDF-aligned patch exposed behavior actor-critic hyperparameters and removed forced zero bootstrap so the policy can use value bootstrap through `compute_behavior_loss`. This remains inside Algorithm 1 Stage I actor-critic optimization and does not add imitation or route-local correction.
Entropy strict candidate `22a374f_pdf_stage1_strict_entropy005_wm2000_pi1000` trained on Windows CUDA and was rejected offline: target22 latent margin passed, steer2 latent margin remained below gate, and action probes collapsed to action 7 on both target22 and steer2. Entropy changed the collapsed action but did not produce a state-dependent actor. Strict actor-collapse probe at commit b5a12e1 then showed both actor actions and world-model counterfactual best actions collapse to action 7 on target22 and steer2, so pure PDF Stage I actor-critic remains blocked by state-independent imagined returns rather than only actor logits.
Windows official sync lane now uses SSH GitHub access, not HTTPS. Clean clone path C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90 was created at commit ac11c90 after enabling core.longpaths. The v1.1 env config is now copied under RCS-AC-v2 and the adapter owns the minimal shared_core env builder, so the clean clone should only need sibling shared_core as an external runtime dependency.
Official Windows clean clone was pulled to commit e42dc6a after fixing optional PowerShell dependency arguments. From that clean clone, dependency check, CARLA Town04 readiness, simulator-state privileged observation provenance, BEV/ego shape/finite checks, and neutral-step safety checks all passed with clean_gate_evidence=false.
First Windows CARLA run must follow docs/03_windows_carla_remote_execution_workflow.md.
GitHub/Windows sync and .git cleanup rules must follow docs/04_git_sync_artifact_cleanup_policy.md.
Clean-performance goal start/resume prompt is docs/05_clean_performance_goal_prompt.md.
```

Next sub-agent trigger:

```text
Before Stage 2 checkpoint promotion, Windows privileged-observation validation, or official Windows execution.
```

## Subagent Ledger

Last pulse:

```text
2026-05-28 Stage H3 resume pulse: Agent A (Schrodinger) completed a read-only
evidence/failure audit after Windows SSH recovery. Agent C spawn hit
`agent thread limit reached`, so the contract/gate role was manually simulated
per the persistent subagent protocol. Agent A classified the latest 512 failure
as under-correction: repeated action7 straight after only brief action4, ending
in collision at route_progress=0.6666666667. The recommended next clean-first
direction is a narrow route 0.615-0.667 closed-loop privileged correction
replay with min_recorded_throttle>=0.3 and very low component weight, not
another broad override/cap sweep.
```

Active agents:

```text
Schrodinger completed the 2026-05-28 H3 evidence/failure audit and should be
closed after integration. One parallel Agent C spawn failed with
`agent thread limit reached`; cleanup/reuse should be attempted before adding
more agents. Ohm completed the 2026-05-26 Stage 2 component-sampler audit and found that
component weights are episode-level, priority sampling can dilute their effect,
unmatched component names silently default to 1.0, and the trainer lacked
realized per-component visibility. The integrated patch adds component
sampler stats and a positive-weight dropped-component guard before more blind
Windows sweeps.
Feynman completed the 2026-05-26 late-right resume/candidate/probe audit.
James was closed after the 2026-05-26 blocked audit.
Epicurus was closed after the 2026-05-26 resume-wrapper
audit. Turing was closed after the 2026-05-26 late-right resume
path audit. Confucius was closed after the 2026-05-26 spawn-failure
documentation audit. Raman and Linnaeus were closed after their 2026-05-26
audits. Helmholtz and Fermat completed earlier Stage 2 route-late steering and
false-PASS sidecar audits. Close completed/obsolete agents before spawning more.
```

Latest integrated findings:

```text
Persistent cadence is now documented: stage entry, major edit, repeated failure, gate review, and every 30 minutes/large edit batch.
Lite default now requires Agent A + Agent C at work-packet/stage entry, with Agent B added for code/config/test/logging/checkpoint wiring.
Pre-edit barrier and Implementation Brief are documented.
Stage 0 action table order is locally canonicalized as raw2drive_table12_row_major_v1 because PDF Table 12 has no explicit action IDs.
Stage 0 config now pins action hash, reward contract version, continue truncation policy, and leakage allowlist/denylist.
Stage 1 uses a compact Gaussian RSSM smoke; this is documented as an engineering smoke approximation before later exactness passes.
Stage 1 binary synthetic BEV targets, stop-gradient KL tests, previous-action convention tests, ego influence tests, finite rollout, synthetic overfit, and checkpoint roundtrip all pass.
Stage 2 local smoke uses policy-driven RSSM prior imagination with reward/continue heads; scalar critic and batch-normalized advantage are documented smoke approximations.
Stage 2 local tests pass, but CARLA clean gate is still pending and must not be treated as full clean PASS.
Stage 2 dry-run summary validation prevents LOCAL_SMOKE_ONLY or scratch evidence from becoming PASS.
Official PASS now requires source windows_carla, steps reaching horizon, done_reason horizon_reached, zero collision/offroad/stuck, failure_class none, finite metrics, pulled_files evidence, and commit/branch/command/config/CARLA metadata.
Stage 2 adapter loop now writes metrics.csv and summary.json in a finally-style path, validates fake PASS/FAIL cases locally, and requires metrics.csv evidence for any official PASS.
Windows runner real mode now invokes evaluate_clean --run-carla after CARLA readiness.
Stage 2 config now points at rcs_ac_v2.evaluation.carla_adapter:create_stage2_carla_adapter.
The adapter factory lazily builds the v1.1-style shared_core CARLA env, bridges Raw2Drive controls through temporary stage4_v2 for v1.1 compatibility, keeps a direct_3d reverse-loss guard, extracts strict privileged_bev [1,1,43,16,16] and privileged_ego [1,1,8], and wraps world-model+policy checkpoints into policy.act(obs, info) logits [1,1,39].
The adapter now falls back to a v2-owned simulator-state builder when the shared-core env lacks privileged keys. It uses CARLA world/map/vehicle/route-manager state, pins schema carla_simulator_semantic43_v1, pins channel hash 382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3, and records clean-gate-eligible provenance.
The clean loop validates privileged provenance and rejects missing, schema-mismatched, hash-mismatched, or derived_debug_from_raw_v1 observations before any clean PASS can be claimed.
Windows archive probe result for commit 5b00347 is PROBE_PASS with source carla_simulator_state, BEV [1,1,43,16,16], ego [1,1,8], finite tensors, nonzero drivable_road/route_path/ego_vehicle/static_obstacle/junction/route_straight channels, and neutral step collision/offroad/stuck false.
The existing Windows checkout was dirty and GitHub credential fetch failed, so this round used a scratch archive runtime instead of switching the Windows repo. That is valid infrastructure/probe evidence, but the official clean run still needs a clean git-pull lane or clean clone.
Stage 2 training is no longer limited to synthetic smoke plumbing: v2 has replay IO/sampler, Windows privileged replay collection, replay-based world-model/policy training, and checkpoint manifests with source=windows_carla_privileged_replay. Synthetic smoke checkpoints now carry source=local_synthetic_smoke_not_carla and are rejected by the official clean config.
Windows archive scratch check for commit e1c9c86 confirmed replay modules import under Python 3.12, tiny CARLA privileged replay collection writes source=windows_carla_privileged_replay manifest, tiny replay training writes world_model/policy sha256 manifest, and load_stage2_policy_runner accepts those replay-source checkpoints under the strict provenance gate.
HTTPS GitHub access on Windows still fails due credential prompt, but SSH git@github.com works for origin/codex/rcs-ac-v2-raw2drive. A clean official clone is available after core.longpaths=true. Stage 2 scripts no longer require untracked RCS-AC-v1.1 support/code; they still support EnvConfigPathOverride/SupportCodePath/V11CodePath/SharedCorePath for local overrides.
Official clean clone probe at commit e42dc6a passed without override paths: source=carla_simulator_state, schema=carla_simulator_semantic43_v1, hash=382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3, BEV [1,1,43,16,16], ego [1,1,8], finite tensors, neutral step running/collision=false/offroad=false/stuck=false. This remains infrastructure evidence, not clean performance evidence.
Real mode with missing checkpoints is now classified as failure_class model / done_reason world_model_checkpoint_missing before CARLA env construction.
Checkpoint loader now requires path and sha256, validates payload config/state_dict, rejects 8x8 world-model checkpoints for the 16x16 clean config, rejects missing policy config, and keeps checkpoint failures in failure_class model.
Replay provenance was hardened at commit 48c0543: clean/replay now require privileged_observation_source=carla_simulator_state, existing env privileged keys are clean-ineligible, replay trainer requires collector manifest/file sha validation, and checkpoint payloads must carry replay lineage, role, trainer, non-clean evidence, data_files, replay manifest, and valid_action_ids.
Reverse-action masking was added at commit db1e8b6 for direct_3d/stage4_v2 when allow_reverse_direct_3d=false. The first replay checkpoint then loaded successfully and had REVERSE_ACTION_COUNT=0 on 64 replay observations.
First official 128-step clean eval with promoted replay checkpoint ran from the clean clone at commit db1e8b6 and produced source=windows_carla, status=FAIL, failure_class=driving, done_reason=stuck, steps=64/128, collision_count=0, offroad_count=0, stuck_count=1, route_progress=0.02564102564102564, metrics_rows=64, adapter_errors=0, runtime_errors=0.
BC warm-start for replay policy training was added at commit b20d8eb after the first clean failure showed action collapse to action_id=8 for all 64 clean steps. Windows BC retraining completed into windows_replay_b20d8eb_bc_pilot, but action probe and clean rerun are pending because the Windows Tailscale peer became temporarily unreachable.
Stage 2 now has a reproducible checkpoint action probe wrapper: scripts/windows/probe_stage2_checkpoint_actions.ps1. It creates the checkpoint-filled runtime config from a checkpoint manifest and writes stage2_checkpoint_action_probe.json with action counts, reverse_action_count, nonzero_throttle_action_count, moving_steer_action_count, single_action_collapse, and recommended_for_clean_rerun.
The checkpoint action probe now rejects both under-steering and over-steering candidates before clean eval by requiring moving/nonzero steering between the configured lower bound and 0.5 fraction.
The checkpoint action probe also records dominant_action_fraction and rejects dominant-action collapse above 0.85 before CARLA reruns. Replay checkpoint manifests now record sampler priority-window stats, and trainer class balancing can be softened with imitation_balance_power for offline steering sweeps.
Windows runner PYTHONPATH now includes v2 code, optional legacy support/code overrides, and workspace root needed by the shared-core env builder.
Windows clean eval runner now accepts -RunOutputDir to copy validated summary.json, metrics.csv, and the exact input config into a run-id evidence directory after summary validation. This is transport-only mirroring and does not change clean gate status or PASS criteria.
BC checkpoint action probe on windows_replay_b20d8eb_bc_pilot failed the pre-clean gate: probed_actions=512, action_counts={34:512}, reverse_action_count=0, nonzero_throttle_fraction=0.0, single_action_collapse=true, recommended_for_clean_rerun=false. No clean rerun was attempted for that checkpoint.
Replay label diagnosis found the old official replay had 388/512 zero-throttle action_id=34 labels, and TargetSpeedKmh overrides were ignored because env info.target_speed_kmh took precedence. Commit 30b7313 fixed explicit TargetSpeedKmh precedence; commit 9c24ff7 removed waypoint target_distance<2 as a zero-throttle label source. The 9c24ff7 fast35 replay collected on Windows has action_counts={7:264,15:248}, nonzero_throttle_fraction=1.0, reverse_fraction=0.0, and clean_gate_evidence=false.
Fast35 detached replay training now works through the scheduled-task wrapper. The b4c8e58/37a0875 checkpoint manifest has source=windows_carla_privileged_replay, valid_action_ids=true, policy_final_imitation=0.7903282642364502, and policy_final_total=1.0499897003173828.
Fast35 checkpoint action probe passed the pre-clean gate with probed_actions=512, action_counts={7:500,15:12}, reverse_action_count=0, single_action_collapse=false, nonzero_throttle_fraction=1.0, previous_stuck_action8_count=0, and recommended_for_clean_rerun=true.
Official 128-step clean rerun from the clean clone at commit 37a087516f7767bb297b8b5567590d7a316eb644 passed: source=windows_carla, status=PASS, clean_gate_pass=true, done_reason=horizon_reached, steps=128/128, collision/offroad/stuck counts all zero, route_progress=0.1794871794871795, episode_return=13.545934572121052, metrics_rows=128, action_counts={7:125,15:3}. This is the first Stage 2 clean PASS, not full clean-performance completion.
Official 512-step primary clean run from the clean clone at commit 0b10ff5628b2b9616759bd4c6f9f60cc441c9b0d failed with source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=311/512, collision_count=1, offroad_count=0, stuck_count=0, route_progress=0.5641025641025641, episode_return=19.066852226147766, metrics_rows=311, action_counts={7:308,15:3}, adapter_errors=0, runtime_errors=0. This validates the 512 runner path but does not satisfy the 512 clean gate.
Offline steer2 replay sweep at commit d4c5b70 rejected four checkpoint candidates before CARLA: balance_power=0.5 -> {7:647}, balance_power=0.625 -> {7:647}, balance_power=0.75 -> {4:639,7:8}, no class balance -> {7:647}. The replay sampler reported priority_window_count=212/617, so steering windows exist; the current replay-imitation policy objective is collapsing to one dominant action instead of learning state-dependent straight/steer decisions.
Offline latent separability probe at commit 7902e44 compared the same four rejected steer2 candidates. Each candidate used 647 replay samples, latent_dim=160, majority_action_fraction=0.6831530139103554, linear_probe_train_accuracy=0.7698259353637695, and linear_probe_test_accuracy=0.7384615540504456. This is above baseline but too weak to justify another clean run; patch latent/action separability or the sequence-aware policy objective before rerunning CARLA.
Action-aux replay training at commit 7f87656 added WorldModelActionAuxWeight and produced the first post-collision candidate that passed offline gates: action_counts={1:8,4:73,7:436,10:14,12:1,15:115}, moving_steer_fraction=0.14837712519319937, dominant_action_fraction=0.6738794435857806, recommended_for_clean_rerun=true, and latent linear_probe_test_accuracy=0.9076923131942749. Official 512 eval from clean clone C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 still failed with source=windows_carla, status=FAIL, done_reason=collision, steps=178/512, route_progress=0.6410256410256411, metrics_rows=178, clean action_counts={1:2,4:15,7:76,10:3,15:82}. The next patch should address speed/turn timing around route_progress 0.59-0.64, not raw/fusion.
Commit b5a12e1 non-route-local fallback `b5a12e1_mixed_t22_steer2_aux1_iw5_ssf005_nobal` passed offline cross replay gates, but official 512 clean failed with source=windows_carla, status=FAIL, clean_gate_pass=false, done_reason=collision, steps=305/512, metrics_rows=305, route_progress=0.6666666666666666, collision/offroad/stuck=1/0/0. Final tail still selected action 7 straight, so speed was reduced but final turning did not occur.
Late action10 override candidate `b5a12e1_mixed_t22_steer2_aux1_iw5_ssf005_nobal_late10w075_r0666` passed offline cross replay gates and official 512 clean failed with source=windows_carla, status=FAIL, clean_gate_pass=false, done_reason=collision, steps=325/512, metrics_rows=325, route_progress=0.6666666666666666, collision/offroad/stuck=1/0/0. Final tail selected action 10 from step 309 onward, proving the late turn can affect execution, but steer=0.1 is too weak to clear the turn. Commit bbea3a5 added route_late_moving_steer_margin so only route_progress>=0.666 pushes the moving-steer family without early leakage. The first bbea3a5 candidate passed offline cross replay gates but official 512 clean still failed with source=windows_carla, status=FAIL, clean_gate_pass=false, done_reason=collision, steps=327/512, metrics_rows=327, route_progress=0.6666666666666666, collision/offroad/stuck=1/0/0. Final tail selected action 4 left steer, so the next patch must distinguish right-moving steering from generic moving steering. The cc8ac46 action13 mix passed offline cross replay gates but official 512 clean failed earlier with source=windows_carla, status=FAIL, clean_gate_pass=false, done_reason=collision, steps=302/512, metrics_rows=302, route_progress=0.6410256410256411, collision/offroad/stuck=1/0/0. Final tail selected action 13/16 too early, bracketing the failure between weak action10 and over-strong action13/16.
Target-24 failure-window replay was collected from the 7f87656 clean clone with 2 episodes / 520 steps / target_speed_kmh=24.0 and action_counts={7:386,10:26,13:8,15:74,16:20,19:6}. Windows/Tailscale disconnected during the first target-24 action-aux checkpoint training attempt, so checkpoint/probe status is pending; resume by checking whether windows_replay_7f87656_t24_aux1_balpow05_iw5 has a manifest, then run action+latent probes before any new 512 clean eval.
Detached replay training resume support was hardened after the Windows disconnect: start_stage2_replay_training_detached.ps1 now writes train_stage2_privileged_from_replay.launch.json in the checkpoint output directory, and check_stage2_replay_training_detached.ps1 can recover LogPath/TaskName from that launch manifest using only OutputDir. Local RCS-AC-v2 tests still pass with 109 passed.
Candidate cross-replay probing was added so target-24 candidates must pass action+latent probes on both the target-24 replay and the existing steer2 replay before any new 512 clean eval.
Cross-replay probe promotion is now gated by candidate_cross_replay_gate.py. A candidate is 512-eligible only when every replay action probe has recommended_for_clean_rerun=true and every latent probe has linear_probe_test_accuracy>=0.80 with >=0.15 margin as a fraction of the remaining accuracy headroom over the majority baseline. The gate writes stage2_candidate_cross_replay_gate.json and exits nonzero when rejected.
Target-24 candidate resume orchestration was added: run_stage2_target24_candidate_cycle.ps1 checks the interrupted OutputDir first, starts a fresh detached rerun OutputDir if needed, writes `${OutputDir}.latest.json` pointing to the active rerun, and can run cross replay probes automatically once TRAIN_READY_FOR_PROBE=True. The same runner now accepts CandidateName/PrimaryReplayName overrides so the next target-22 candidate can reuse the same check/restart/probe path if target-24 remains unsafe.
Target-speed replay orchestration was added for the next speed-timing sweep: run_stage2_target_speed_replay_cycle.ps1 defaults to target-22, collects replay only with -CollectIfMissing, then hands off to the candidate cycle runner for detached training and cross replay probes.
Candidate gate decision summarization was added: summarize_stage2_candidate_gate.ps1 reads stage2_candidate_cross_replay_gate.json and emits either run_official_512_clean_eval, run_target22_fallback, or wait_for_cross_replay_gate while preserving clean_gate_evidence=false so offline gates cannot be recorded as clean PASS.
Target-24 candidate training resumed on Windows after reconnect. The rerun checkpoint windows_replay_7f87656_t24_aux1_balpow05_iw5_rerun_20260525T021559 trained successfully and passed the target24 replay action/latent probes, but cross-replay promotion was rejected on steer2: action recommended_for_clean_rerun=false due moving_steer_fraction=0.32921174652241114 > 0.20 and latent margin=0.11684699801057352 < 0.15. Decision summary is run_target22_fallback, not clean evidence. Target22 replay collection then completed and detached training started as RCSV2Stage2target22Aux1BalPow05, but Windows SSH/Tailscale became unreachable before completion/probe status could be checked.
Windows scripts README now reflects the implemented Stage 2 runner, v2-local readiness scripts, dependency probe, privileged obs probe, replay collector, replay trainer, detached replay training start/check wrappers, and v2-local minimal shared_core env builder bridge.
RCS-AC-v2 subtree was committed and pushed to origin/codex/rcs-ac-v2-raw2drive. Unrelated root worktree changes were not staged.
```

Next subagent trigger:

```text
Before Stage 2 checkpoint promotion, Windows privileged-observation validation, or Windows execution, spawn at least Agent A + Agent C; add Agent B for code/test/logging/checkpoint wiring.
```

Unavailable/fallback:

```text
If multi-agent tools are not loaded, use tool_search for "multi-agent spawn manage sub-agents".
If still unavailable, manually simulate Agent A/B/C roles and record fallback here.
```

## Next Concrete Action

Patch the current 512-step H3 route 0.6667 collision before raw/fusion work:

```text
current true stage is H3 32x32 privileged clean-stability fill
current local branch is codex/rcs-ac-v2-raw2drive
canonical Windows clean clone for the next Stage 2 candidate is C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656
pull latest origin/codex/rcs-ac-v2-raw2drive before every Windows run
latest best H3 lineage is windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524
it passed offline action probe and official 128 clean, but failed official 512 as offroad at route_progress=0.6153846154 after repeated action4 mild-left tail
route-local action4 cap probes at weights 0.25 and 0.10 were rejected offline before CARLA because action4 suppression leaked into action7 dominance on signp/signm cross probes
late058 correction replay collection succeeded with 4 episodes / 160 steps and label_action_counts {7:24,10:24,13:20,16:12,19:24,22:56}; CARLA was stopped afterward
late058w005 mixed candidate fixed action probes on signp/signm; the previous cross gate rejection was traced to an impossible absolute latent-margin rule on signp (majority=0.8558282209 made majority+0.15 exceed 1.0)
candidate_cross_replay_gate.py now uses headroom-normalized latent margin while keeping the absolute latent test accuracy and action probe gates unchanged
Windows clean clone pulled commit 783fe7e and reran the late058w005 cross gate from existing artifacts; ready_for_clean_eval=true
official 128 clean eval from committed/pushed code passed: source=windows_carla, status=PASS, clean_gate_pass=true, steps=128/128, done_reason=horizon_reached, route_progress=0.2820512821, collision/offroad/stuck=0/0/0
the generated 512 checkpoint-filled config from clean_512_stage2_privileged_32x32.yaml passed full signp checkpoint_action_probe with recommended_for_clean_rerun=true
official 512 clean eval still failed: source=windows_carla, status=FAIL, clean_gate_pass=false, done_reason=collision, steps=310/512, route_progress=0.6666666667, collision/offroad/stuck=1/0/0, clean action_counts={7:248,15:37,4:25}
failure tail: route 0.615-0.667 reverted to mostly action7 straight after only brief action4 mild-left; next clean-first action is a narrow route 0.615-0.667 late turn/right-moving correction, not raw/fusion
follow-up `late058w005_action10override_r0615067_w005` trained on Windows CUDA,
passed cross replay on signp/signm, and passed official 128
(`steps=128/128`, route_progress=0.2564102564, collision/offroad/stuck=0/0/0),
but official 512 still failed at route_progress=0.6666666667 with collision
after a repeated action7 straight tail (`steps=326/512`,
collision/offroad/stuck=1/0/0). CARLA was stopped afterward and RTX 3080 was
idle at about 50C/0%/695MiB.
Agent A recommendation after this failure: prefer a narrow closed-loop
`corr0615_0667` privileged correction replay using the latest 128-pass/512-fail
checkpoint as warmup, min_recorded_throttle>=0.3, labels constrained to the
Raw2Drive 39-action table, and low component weight around 0.003-0.005. Before
official CARLA, require cross replay plus route-bin evidence that action10/13
does not leak before route 0.60 and action7 dominance is reduced inside
0.615-0.667.
offline-only exploratory candidate
`windows_replay_h3_32x32_late058w005_action10override_r0615067_w010_seed20260524`
was launched on Windows CUDA to test whether a slightly stronger action10-only
local override can reduce the straight-tail failure without broad right-steer
pressure. Do not run official CARLA from it unless cross replay passes and
route-bin distributions are sane.
latest route-local correction diagnostics:
`corr0615_0667_t18_signp` was valid but too weak
(`action_counts={7:116,10:6}`) and its low-weight candidate collapsed to
action7 on signp. `corr0615_0667_t16_signp3` produced usable route-window
labels (`action_counts={1:6,4:18,7:47,10:34}`); the low/medium mix still
underfit the correction and failed cross replay, while the high-weight
diagnostic learned route 0.64-0.70 action10 but still failed the global
signp/signm promotion gate. Follow-up interpolation confirmed the threshold:
`corr0615p3=0.5` still collapsed to action7 on the correction replay, while
`corr0615p3=0.75` learned the route 0.64-0.70 correction
(`self_probe action_counts={1:1,4:6,7:68,10:30}`) but failed cross replay with
signp/signm dominant action7 fractions 0.966/0.983. Do not run official CARLA
from these diagnostic checkpoints. Next clean-first work should either add a
route-window-aware promotion gate for the known route 0.6667 collision bracket,
or adjust the objective so signm mid-route diversity is preserved while the
route 0.64-0.70 action10 correction remains active.
route-window supplemental gate implementation is now in progress as H3
clean-stability fill. It must remain `clean_gate_evidence=false`, must not
replace official Windows CARLA clean evidence, and must not be reported as
PDF-explicit H2 core. Use it to inspect known collision-window correction and
early leakage before deciding whether the next step is objective/data repair.
Windows clean clone pulled commit `1c05267` and applied the supplemental gate
to existing `w075` artifacts. Generic gate remains false, but route-window
gate is true for signp: route 0.64-0.70 action10 fraction 33/45=0.7333333333,
early route <0.60 action10 leakage 0/897, zero-throttle steer 0.0. This
confirms the correction is localized; next objective/data repair should preserve
signm diversity while keeping that route-window behavior.
Signm duplicate-boost follow-ups did not solve this: signmboost=1 suppressed
the correction to corr self-probe `{7:100,10:5}`, and signmboost=0.25 still
failed with `{7:93,10:12}`, dominant action7 0.8857142857. Do not run official
CARLA from either. Simple signm replay duplication is therefore not the right
repair; use a targeted objective/gate repair or collect a correction replay
with signm-compatible mid-route diversity.
do not run another official 512 until a new candidate passes cross replay and a 512 checkpoint action probe from committed/pushed code
do not continue broad/global moving-steer pressure; it already overcorrected into action16 dominance
do not start raw/fusion; privileged 512 clean is still failing
commit/push RCS-AC-v2 work before Windows git pull --ff-only
```

## Stage 0 Gate

Status:

```text
PASS
```

Command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
25 passed in 1.12s
```

## Stage 1 Gate

Status:

```text
PASS
```

Command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Result:

```text
37 passed in 2.00s
```

Synthetic overfit:

```text
initial_loss 4.801789283752441
final_loss 2.732663631439209
checkpoint_path /tmp/rcs_ac_v2_stage1_smoke/stage1_privileged_world_model_smoke.pt
```

## Stage 2 Gate

Local status:

```text
LOCAL_SMOKE_PASS
DRY_RUN_CONTRACT_PASS
LOCAL_ADAPTER_LOOP_PASS
LOCAL_CARLA_FACTORY_PASS
LOCAL_CHECKPOINT_EXPORT_PASS
WINDOWS_DEPENDENCY_PROBE_PASS
OFFICIAL_CLEAN_CLONE_PRIVILEGED_OBS_PROBE_PASS
REPLAY_TRAINING_PATH_READY
SYNTHETIC_CHECKPOINT_REJECTION_PASS
```

Local command:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
```

Local result:

```text
106 passed in 3.03s
```

Local policy smoke:

```text
initial_critic_loss 0.001165103050880134
final_critic_loss 0.00015270980657078326
valid_action_ids True
rollout_continue_mean 0.5067161917686462
```

Dry-run contract:

```text
DRY_RUN_SUMMARY=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
SUMMARY_VALID=RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/summary.json
generated dry-run summary removed after validation
```

Full clean status:

```text
WINDOWS_PRIVILEGED_REPLAY_COLLECTION_PASS
REPLAY_CHECKPOINT_PROMOTED
BC_COLLAPSE_REJECTED_BY_ACTION_PROBE
FAST35_REPLAY_CHECKPOINT_PROMOTED
FAST35_ACTION_PROBE_PASS
FAST35_128_CLEAN_PASS
CARLA_512_PRIMARY_FAIL_COLLISION
B5A12E1_512_FAIL_COLLISION_ROUTE0666
B5A12E1_LATE10_512_FAIL_COLLISION_ROUTE0666
BBEA3A5_LATEMOVE_512_FAIL_COLLISION_ROUTE0666
CC8AC46_LATEMOVE_ACTION13_512_FAIL_COLLISION_ROUTE0641
LATE_RIGHT_STEER_MARGIN_LOCAL_TEST_PASS
354C0B6_MID048_W001_OFFLINE_GATE_PASS
354C0B6_MID048_W001_512_FAIL_COLLISION_ROUTE0538
354C0B6_CORR045_T10_REPLAY_COLLECTION_PASS
354C0B6_CORR045_T10_W003_OFFLINE_GATE_REJECT_TARGET22
354C0B6_CORR045_T10_W001_OFFLINE_GATE_REJECT_STEER2
9398470_CORR045_T14_REPLAY_COLLECTION_PASS
9398470_CORR045_T14_W001_OFFLINE_GATE_PASS
9398470_CORR045_T14_W001_512_FAIL_COLLISION_ROUTE0666
9398470_CORR060_T14_REPLAY_COLLECTION_PASS
9398470_CORR045_CORR060_T14_W001_OFFLINE_GATE_PASS
9398470_CORR045_CORR060_T14_W001_512_FAIL_OFFROAD_ROUTE0718
C6B03E7_CORR069_T14_PARTIAL_REPLAY_ONLY
C6B03E7_CORR064_T14_REPLAY_COLLECTION_PASS
C6B03E7_CORR060_CORR064_OFFLINE_GATE_PASS
C6B03E7_CORR060_CORR064_512_FAIL_COLLISION_ROUTE0590
C6B03E7_CORRMIX045060064_W002_OFFLINE_GATE_REJECT_ACTION7
C6B03E7_CORRMIX045060064_W00067_OFFLINE_GATE_PASS
C6B03E7_CORRMIX045060064_W00067_512_FAIL_OFFROAD_ROUTE0718
C6B03E7_CORR066_T14_PARTIAL_REPLAY_ONLY
C6B03E7_CORRMIX045060066_W0008_OFFLINE_GATE_PASS
C6B03E7_CORRMIX045060066_W0008_512_FAIL_COLLISION_ROUTE0666
8859F1D_LATECAP7_R066072_OFFLINE_GATE_PASS
8859F1D_LATECAP7_R066072_512_FAIL_OFFROAD_ROUTE0744
8859F1D_LATECAP47_R066076_OFFLINE_GATE_PASS
8859F1D_LATECAP47_R066076_512_FAIL_OFFROAD_ROUTE0744
8859F1D_LATECAP47_O10W025_R0720755_OFFLINE_GATE_PASS
8859F1D_LATECAP47_O10W025_R0720755_512_FAIL_OFFROAD_ROUTE0718
3ED89EB_POST072_MIN16_REPLAY_COLLECTION_FAIL_TOO_SPARSE
3ED89EB_POST070P_MIN8_REPLAY_COLLECTION_PASS
3ED89EB_POST070P_W001_SEQ8_OFFLINE_GATE_REJECT_STEER2
3ED89EB_POST070P_W0003_SEQ8_OFFLINE_GATE_REJECT_STEER2
3ED89EB_POST070P_W0003_CAP10PRE072_OFFLINE_GATE_PASS
3ED89EB_POST070P_W0003_CAP10PRE072_512_FAIL_COLLISION_ROUTE0667
5B12B8F_POST070P_CAP10PRE070W1_O10TAILW02_OFFLINE_GATE_PASS
5B12B8F_POST070P_CAP10PRE070W1_O10TAILW02_512_FAIL_COLLISION_ROUTE0615
5B12B8F_POST070P_CAP10PRE067W05_O10TAILW002_OFFLINE_GATE_PASS_STRICT_REJECT
```

Current Stage 2 position:

```text
Privileged clean is not solved yet.
Latest official 512: FAIL collision at route_progress 0.6666666667, 326/512 steps, from `late058w005_action10override_r0615067_w005`.
Best recent official 512 evidence remains failure-only; no privileged 512 clean PASS exists.
Latest successful full replay collections: windows_privileged_9398470_corr045_midroute_t14_minthr03_min16, windows_privileged_9398470_corr060_late_t14_minthr03_min16, and windows_privileged_c6b03e7_corr064_late_t14_minthr03_min16.
Latest partial replay: windows_privileged_c6b03e7_corr066_late2_t14_minthr03_min16, 2 saved episodes, route 0.6667-0.7179.
Latest offline candidates passed target22/steer2; latecap47 reached route 0.7436, but the action10 override regressed to route 0.7179.
Do not start raw/fusion.
Post-0.72 correction work using the latecap47 warmup policy found that a strict
post-0.72 min16 replay is too sparse: one run reached route_progress 0.7436 but
only yielded 2 eligible steps, so no usable episode was saved. A post-0.70
positive-steer min8 component was collected with 4 episodes, 37 total steps, and
action_counts {10:28,4:8,7:1}. Two CUDA seq8 candidates mixed that component
with target22, steer2, and corrmix045060064. Both were rejected before official
CARLA eval: w0.01 failed steer2 with moving_steer_fraction 0.2395672334, and
w0.003 failed steer2 with moving_steer_fraction 0.2472952087 plus action10
over-application through route bins 0.55-0.70. Best official 512 remains
latecap47 route_progress 0.7435897436, 344/512, offroad FAIL. The remaining
gap is a narrow late closed-loop tail, not an early-route or whole-system
failure. Next direction: stop global post070p weight sweeps and use a
route-conditioned correction mechanism or stricter replay filtering that
localizes the action10/rightward correction to the true post-0.72 tail.
Follow-up cap10pre072 then added a route-conditioned action10 logit cap before
route 0.72. This fixed the offline steer2 leak and passed the cross-replay gate
(steer2 moving_steer_fraction 0.1622874807), but official 512 regressed to
collision at route_progress 0.6666666667 after action10 repeated through
route 0.6154-0.6410 and one action13 at the terminal step. Best official 512
therefore remains latecap47 route_progress 0.7435897436. Next direction: do not
run more official CARLA from weak action10 pre-0.72 caps; use stronger/narrower
offline-only suppression or implement route-conditioned filtering/loss that
separates mid-route 0.60-0.67 from the true post-0.70/0.72 tail.
The stronger cap10pre070w1 + o10tailw02 branch passed the generic offline gate
and removed action10 from steer2 replay, but official 512 regressed further to
route_progress 0.6153846154 collision after action10 repeated through
route 0.5641-0.6154 and terminal action13. The subagent-recommended conservative
cap10pre067w05 + o10tailw002 branch also passed the generic gate but failed the
stricter promotion rule: steer2 moving_steer_fraction was 0.1622874807 and
action10 count remained 63, so no official run was launched. `checkpoint_action_probe.py`
now records route-bin action_counts and label_action_counts so future promotion
checks can directly reject mid-route action10/action13 leakage. Next meaningful
work should stop global action10 cap/override sweeps and move to route-conditioned
replay filtering/loss or closed-loop failure-window collection.
Avoid zero-throttle-only correction components unless filtered or strongly gated offline.
Route-priority replay sampling is now implemented for the next clean-first
candidate: `route_priority_sample_fraction`, optional
`route_priority_action_ids`, and min/max route-progress bounds are wired through
the Python trainer plus Windows train/detached/mixed candidate wrappers. The
first route-priority candidate
`6a0cc43_t22_steer2_corrmix_post070p_routeprio10_0700755_w0003_seq8`
trained successfully on Windows CUDA with sampler priority_window_count=14 and
effective_priority_sample_fraction=0.25. Offline cross replay rejected it:
steer2 passed, but target22 collapsed too strongly to action15
(dominant_action_fraction=0.9157088123, recommended_for_clean_rerun=false).
Do not run official CARLA for this candidate. The next candidate should keep
the route-priority mechanism available but reduce target22 action15 dominance
before another official clean attempt.
The no-latecap follow-up
`9bd3f7f_t22_steer2_corrmix_post070p_routeprio10_0700755_nolatecap_seq8`
produced effectively the same offline rejection: target22
dominant_action_fraction=0.9157088123 with action15 dominance, while steer2
remained gate-passing. This rules out the retained latecap controls as the
primary cause. Next work should reduce post070p/route-priority strength or add
a target22 action15 dominance guard before any official CARLA run.
The route-priority fraction `0.05` candidate and the same candidate plus
`RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=15` were also rejected
offline. Both kept target22 action15 dominance around 0.9195, while steer2
remained gate-passing. The action15 cap was recorded in the checkpoint manifest
but did not move this branch. Do not run official CARLA from this branch. Next
meaningful Stage 2 work should either make route-priority sampling/loss
component-aware enough not to dominate target22 early-route logits, or collect
a cleaner closed-loop tail component.
After the late058w005 latent-gate fix, official 128 clean passed but official
512 failed at route 0.6667 by collision. The conservative action10 override
candidate `late058w005_action10override_r0615067_w005` also passed official 128
and failed official 512 at the same route bracket, with repeated action7
straight from route 0.6154 through collision. This supersedes older offroad
route 0.7436 text as the latest evidence. The next primary direction is a
route 0.615-0.667 closed-loop correction replay; any further override/cap
candidate is offline-only unless route-bin probes prove it does not leak
action10/13 before route 0.60 or overcorrect the signp/signm cross replays.
PDF-core H0 rebase then classified `route_priority_components` and
`priority_episode_mask` as clean-stability fill/diagnostic sampler controls,
not PDF-explicit core. Subagent Agent A and Agent C agreed with this
classification. The WIP is now guarded and tested: Python trainer, Windows
foreground trainer, detached trainer, and mixed-candidate cycle wrapper all
accept `route_priority_components`; `--pdf-stage1-strict` rejects it; manifests
record whether component priority masks are effective. Local validation on
2026-05-27 KST: `python -m pytest RCS-AC-v2/tests/test_privileged_replay_training.py -q`
-> 33 passed, and `python -m pytest RCS-AC-v2/tests -q` -> 151 passed. This is
not clean performance evidence. Next work is H1 privileged observation upgrade:
config-driven BEVGridSpec, 32x32+ smoke, route/lane/obstacle coverage report,
and replay manifest BEV shape/schema/hash recording before another official
Windows run. The H1 implementation brief is recorded at
`RCS-AC-v2/experiments/02_privileged_policy_clean/h1_privileged_observation_upgrade_brief.md`.
H1 local implementation then made the simulator-state BEV path shape-aware:
diagnostics now record BEV shape/grid spec and route/lane/obstacle coverage,
replay collection accepts `[43,H,W]`, replay episode metadata and replay/checkpoint
manifests record BEV/ego shapes plus schema/hash, and replay training derives
world-model H/W from replay shape while rejecting mixed H/W replay sets. Local
validation on 2026-05-27 KST: `python -m pytest RCS-AC-v2/tests/test_stage2_carla_adapter.py RCS-AC-v2/tests/test_privileged_replay_training.py -q`
-> 61 passed, and `python -m pytest RCS-AC-v2/tests -q` -> 157 passed. This is
not Windows clean performance evidence. Next required evidence is a committed
Windows 32x32 privileged observation probe and tiny 32x32 replay collection
from the clean clone.
H1 Windows execution readiness update on 2026-05-28 KST: a committed
32x32-specific clean config now exists at
`RCS-AC-v2/configs/eval/clean_128_stage2_privileged_32x32.yaml`. It preserves
the PDF-explicit privileged semantic43+ego contract and Raw2Drive 39-action
contract while changing only BEV H/W to 32x32 and separating output paths under
`windows_128_32x32`. The Windows observation probe wrapper now keeps 32x32
probe output separate by default when the config path contains `32x32`, and
the Windows workflow document points the H1 probe/tiny-replay lane at the
32x32 config. This is still not Windows clean performance evidence; the clean
clone must pull the committed code and run the probe plus tiny replay before
H1 can be closed.
H1 Windows probe update on 2026-05-28 KST: Windows SSH recovery first diagnosed
a Mac-side userspace Tailscale stale-socket failure. The codex `tailscaled`
process had exited, `tailscaled-codex.sock` remained, `lsof -iTCP:1055` showed
no SOCKS listener, normal SSH to 100.87.219.58 timed out, SOCKS SSH returned
`Connection closed by UNKNOWN port 65535`, and documented LAN IPs were stale
or refused SSH. Restarting the codex userspace daemon restored local Tailscale
IP 100.72.72.122, Windows peer 100.87.219.58, and SSH to host `Song`.
The clean clone
`C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656` then pulled
branch `codex/rcs-ac-v2-raw2drive` fast-forward to commit
`27a3450ca270047d43f15f3812e4e2b7d4c0ab6a`. Tracked files were clean with
`git status --short --branch --untracked-files=no`; old and new run artifacts
remain untracked under experiments. CUDA/dependency guard passed with RTX 3080.
The H1 32x32 privileged observation probe passed and wrote
`RCS-AC-v2\experiments\02_privileged_policy_clean\windows_128_32x32\privileged_obs_probe.json`
with `clean_gate_evidence=false`, status `PROBE_PASS`, source
`carla_simulator_state`, schema `carla_simulator_semantic43_v1`, BEV shape
`[1,1,43,32,32]`, ego shape `[1,1,8]`, route/lane/obstacle availability true,
and neutral step collision/offroad/stuck false. The expected empty critical
channels were `vehicle`, `pedestrian`, and `dynamic_vehicle`; actor near/far
and static obstacle channels were nonzero. A tiny replay collection also
passed at the same commit with 1 episode, 8 total steps,
`privileged_bev_shapes=[[8,43,32,32]]`,
`privileged_ego_shapes=[[8,8]]`, source
`windows_carla_privileged_replay`, and manifest path
`RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_h1_32x32_tiny\stage2_privileged_replay_manifest.json`.
CARLA was stopped after the probe/replay; RTX 3080 reported 58C, 0% GPU use,
and 685 MiB memory. This closes the H1 Windows shape/coverage/replay-smoke
evidence, but it is not clean performance evidence. Next work should decide
whether to train a tiny 32x32 replay checkpoint purely as a shape smoke or move
to the real privileged clean replay/training lane before official 128/512/2048
clean gates.
H2 stage-entry update on 2026-05-28 KST: the strict 32x32 trainer plumbing
smoke passed on Windows CUDA. It used the tiny H1 32x32 replay only as a shape
smoke with `SequenceLength=4`, `BatchSize=2`, `WorldModelSteps=2`,
`PolicySteps=2`, `-PdfStage1Strict`, and `Device=cuda`. The checkpoint manifest
under
`RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_h1_32x32_tiny_strict_smoke`
recorded `pdf_stage1_strict=true`, source `windows_carla_privileged_replay`,
replay shape `[[8,43,32,32]]`, world model `bev_height=32`,
`bev_width=32`, `action_dim=39`, `imitation_weight=0`,
`world_model_action_aux_weight=0`, and route/action override/cap/fill weights
at zero. The action probe loaded this checkpoint with
`configs/eval/clean_128_stage2_privileged_32x32.yaml` and generated a
checkpoint-filled 32x32 eval config. It predicted action15 for all 8 tiny
steps, so `recommended_for_clean_rerun=false`; this is expected from an
8-step tiny replay and is not clean evidence. The H2 brief is recorded at
`RCS-AC-v2/experiments/02_privileged_policy_clean/h2_pdf_faithful_32x32_baseline_brief.md`.
Next required H2 evidence is a real 32x32 privileged replay set, strict
PDF-faithful training from that replay, offline probes, and then official
128/512/2048 clean results recorded as PDF-explicit core.
H2 real-replay update on 2026-05-28 KST: the Windows clean clone at commit
`d6e45343abde88079da3b15f10b7a4c27ce0d9e9` collected a real 32x32 privileged
replay set at
`RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_h2_32x32_real_4x128_seed20260524`.
The replay manifest records 4 episodes, 512 total steps, episode lengths
`[128,128,128,128]`, failed_attempt_count=0, source
`windows_carla_privileged_replay`, `clean_gate_evidence=false`, BEV shape
`[128,43,32,32]` per episode, and ego shape `[128,8]` per episode. CARLA was
stopped after collection; RTX 3080 reported 59C, 0% GPU use, and 685 MiB
memory. Strict H2 training from that replay completed on Windows CUDA with
`-PdfStage1Strict`, `SequenceLength=16`, `BatchSize=8`, `WorldModelSteps=200`,
and `PolicySteps=200`. The checkpoint manifest under
`RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_h2_32x32_strict_4x128_seed20260524`
records `pdf_stage1_strict=true`, world model `bev_height=32`,
`bev_width=32`, `action_dim=39`, `imitation_weight=0`,
`world_model_action_aux_weight=0`, `route_priority_sample_fraction=0`,
`route_action_override_weight=0`, world model final_total=2.050567626953125,
policy final_total=-0.0028065170627087355, and valid_action_ids=true.
The checkpoint action probe rejected the candidate before official CARLA eval:
512 replay actions probed, action_counts `{15:4,16:4,19:504}`,
dominant_action_id=19, dominant_action_fraction=0.984375,
moving_steer_fraction=0.9921875,
predicted_moving_on_label_nonmoving_fraction=0.9921875,
reverse_action_count=0, previous_stuck_action8_fraction=0, and
`recommended_for_clean_rerun=false`. This is a PDF-faithful privileged baseline
failure mode, not clean evidence. Do not run official clean eval from this
checkpoint unless intentionally recording a known-bad PDF-core failure. Next
work should move to H3 clean-stability fill while keeping this H2 result
separate from H3 clean-performance claims.
```

Do not start Stage 3 raw sensor work or any fusion ablation until:

```text
CARLA 128-step single-start clean gate is PASS
CARLA 512-step primary subset clean gate is PASS or explicitly blocked after classified retry attempts
CARLA 2048-step subset clean gate is run or explicitly blocked
collision/offroad/stuck/route-progress evidence is recorded
failure class is infrastructure / CARLA / adapter / contract / model / driving
```

## Current H3 Snapshot

```text
Current true stage:
H3 clean-stability fill for privileged 32x32 clean performance.

Latest completed H3 evidence:
windows_privileged_h3_32x32_real_2x512_seed20260524 replay exists with
2 episodes, 530 total steps, lengths [265,265], and labels
{7:432, 10:26, 13:12, 15:38, 16:14, 19:8}.

The first H3 long fill candidate,
windows_replay_h3_32x32_long_iw1_cb05_msfpa1_msnm1_seed20260524,
was rejected offline before official CARLA eval because the action probe
predicted action7 for all 530 replay observations:
single_action_collapse=true and recommended_for_clean_rerun=false.

Additional H3 no-code fill candidates were trained/probed on Windows CUDA and
rejected before official clean eval:
- iw1_cb1_msb2_msfpa05_msnm05_rprio50 collapsed to action8, with
  zero_throttle_fraction=1.0 and previous_stuck_action8_fraction=1.0.
- iw5_cb05_ssf005_aux1_cap7w1c70_logit7w05 improved to
  action_counts={7:501,10:10,13:19} with zero_throttle_fraction=0, but still
  failed dominant_action_fraction=0.9452830189.
- iw5_cb1_ssf010_aux1_cap7w2c65_logit7w1_rprio25 overcorrected to
  action_counts={10:526,15:4}, moving_steer_fraction=0.9924528302.
- iw5_cb075_ssf0075_aux1_cap7w15c67_logit7w075_rprio10 collapsed to
  action10 for all 530 observations.

No official CARLA clean eval was run from these rejected H3 checkpoints.
```

This H3 lane is clean-stability fill, not PDF-explicit core. Official clean
eval may only run from an H3 checkpoint if offline action probes recommend a
clean rerun. H3 failure or pass must not be merged into the H2 PDF-faithful
baseline claim.

Next H3 action:

```text
The route-window action-distribution regularizer is implemented and validated.
The current 2x512 replay is no longer the only H3 source. Windows clean clone
commit 0fcb5906 collected new 32x32 privileged replay components:

- diverse_t22_signm_3x512: 3 episodes / 784 steps, route max 0.512821,
  labels {7:621,10:38,13:12,15:73,16:30,19:10}.
- diverse_t22_signp_3x512: 3 episodes / 978 steps, route max 0.666667,
  labels {4:69,7:837,15:72}.
- mid045_t22_signp_mth03_4x512: 4 episodes / 404 steps, route
  0.461538-0.666667, labels {4:92,7:312}.
- mid045_t22_signm_mth03_4x512: 4 episodes / 144 steps, route
  0.461538-0.512821, labels {7:24,10:52,13:16,16:40,19:12}.

Two mixed CUDA candidates were rejected offline before CARLA:

- mixdiv signpm+base collapsed to action7 for all 2292 probed observations.
- mixdivmid signpm+mid045 overcorrected to action10 for all 2310 probed
  observations, with moving_steer_fraction=1.0 and
  predicted_moving_on_label_nonmoving_fraction=0.8393939394.

Do not run official CARLA clean eval from either checkpoint. The next H3
candidate should keep the new route>0.51 and mid045 components, but lower or
rebalance the midm correction pressure so the policy does not jump from action7
collapse to action10 collapse. Continue with offline action probes until
recommended_for_clean_rerun=true.

Follow-up Windows CUDA probes at clean clone commit 3b5d9bee narrowed the
bracket:

- midp=1,midm=1 collapsed to action10 for all 2310 observations.
- midp=1,midm=0.5 collapsed to action7 for all 2310 observations.
- midp=1,midm=0.75 also collapsed to action7 for all 2310 observations.
- midp=1,midm=0.875 flipped back to action10 for all 2310 observations.
- adding MovingSteerFalsePositiveWeight=0.25 with SteerSequenceFraction=0.025
  broke the single action10 collapse but still failed with
  action_counts={10:104,13:28,16:2178}, dominant_action_fraction=0.9428571429,
  moving_steer_fraction=1.0.

Do not run official CARLA from these checkpoints. Component weight only is too
discontinuous; the next offline-only H3 action should make the moving-steer
false-positive/steer-sequence pressure weaker or route-local so it does not
turn the whole replay into action10/action16 moving-steer dominance.

Latest continuation on 2026-05-28 KST:

- The `w075` route-window correction candidate with
  `RouteActionDistributionWeight=0.05` was cross-probed on Windows CUDA.
  Generic promotion remained false: signp predicted `{7:954,10:24}` with
  dominant action fraction `0.9754601227`, and signm predicted `{7:763,10:21}`
  with dominant action fraction `0.9732142857`.
- The supplemental route-window gate also stayed false for this follow-up:
  signp still had route `0.64-0.70` action10 fraction `24/45=0.5333333333`,
  but signm early action10 leakage was `21/784=0.0267857143`, above the
  `0.02` limit.
- Stronger route-action-distribution candidates were trained from committed
  code with CUDA only, no CARLA:
  `w1`, `w025`, and `w010`.
- All three stronger candidates were rejected by the correction self-probe
  before cross/official promotion. `w1` produced `{1:4,10:53,16:48}`,
  `w025` produced `{1:4,10:55,16:46}`, and `w010` produced
  `{1:4,10:58,16:42,19:1}` on `corr0615p3`; all had
  `moving_steer_fraction=1.0` and `recommended_for_clean_rerun=false`.
- Decision: no official CARLA clean eval is authorized from these checkpoints.
  The clean-first H3 direction should move away from simply increasing
  route-action-distribution weight on the same replay mixture. The next
  candidate should change the data/objective so signp/signm diversity and the
  route `0.64-0.70` correction coexist in the same learned distribution, while
  explicitly penalizing false moving-steer in the correction window.
- Follow-up implementation added an H3-only route-local moving-steer
  false-positive penalty:
  `route_moving_steer_false_positive_weight` plus speed/route window bounds.
  It preserves Raw2Drive 39 discrete actions, adds no inference-time
  controller/fallback, and is rejected by `pdf_stage1_strict`. Local focused
  validation passed:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code pytest -q RCS-AC-v2/tests/test_privileged_replay_training.py RCS-AC-v2/tests/test_evaluate_clean_contract.py -q`
  -> 82 passed.
- Next Windows CUDA candidate should pair a conservative
  `RouteActionDistributionWeight=0.10` with route-local false-positive
  pressure around route `0.60-0.67`, then run correction self-probe and
  signp/signm cross gate before any official CARLA eval.
- Windows CUDA follow-up at commit `ad32ee3` tested that route-local
  false-positive objective without CARLA. `rfp025` still overcorrected on the
  correction replay with `{1:4,10:58,16:42,19:1}` and
  `moving_steer_fraction=1.0`. `rfp2` swung too far back to action7 with
  `{7:102,16:3}`, `dominant_action_fraction=0.9714285714`, and
  `recommended_for_clean_rerun=false`. Neither checkpoint is authorized for
  cross/official promotion. The next H3 candidate should bracket the route
  false-positive weight between `0.25` and `2.0` and/or reduce late-steer
  margin pressure before any official CARLA eval.

Latest H3 route-local evidence:

- Weak global `MovingSteerFalsePositiveWeight=0.05` still overcorrected to
  action16 dominance and was rejected offline.
- The route-local sampler candidate
  `windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524`
  passed the full offline probe with action_counts
  `{4:327,7:1435,10:56,13:32,16:460}`,
  dominant_action_fraction=0.6212121212, moving_steer_fraction=0.3787878788,
  zero_throttle_fraction=0.0, and recommended_for_clean_rerun=true.
- Official Windows 128 clean eval from that checkpoint lineage passed:
  source=windows_carla, commit=feb91059, status=PASS, clean_gate_pass=true,
  steps=128/128, done_reason=horizon_reached, route_progress=0.1794871795,
  and collision/offroad/stuck counts all zero.
- Commit 342935c added the committed 512-step 32x32 clean eval config. Official
  Windows 512 eval from the same checkpoint lineage failed as a driving/offroad
  case: source=windows_carla, status=FAIL, clean_gate_pass=false, steps=338/512,
  done_reason=offroad, route_progress=0.6153846154, collision_count=0,
  offroad_count=1, stuck_count=0, action_counts=`{4:67,7:271}`.
- The failure tail repeated action4 at about 18 km/h around route_progress
  0.59-0.62. The next H3 work should target that window before another
  official 512 claim, while avoiding broad/global moving-steer pressure because
  it previously produced action16 dominance.
```

## Important Constraints

```text
Use Raw2Drive 39 discrete actions.
Do not use v1.1 waypoint + pure-pursuit as final v2 action contract.
Do not expose privileged BEV/GT data to raw policy inference.
Do not train raw reward/continue heads.
Do not run fusion ablation before clean gates.
Use persistent subagent pulse at stage entry, major edits, repeated failures, and gate reviews.
If subagent spawn fails with "agent thread limit reached", close completed old agents before spawning new sidecars.
Use Mac-Codex / Windows-CARLA split for all real CARLA runs.
Run CARLA readiness guard before every Windows CARLA command.
Do not delete root .git during goal execution; use git maintenance/gc only.
Delete accidental nested .git only after the documented checks pass.
```
