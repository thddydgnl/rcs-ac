# Stage 2 Windows 512-Step Primary Clean Result

Date: 2026-05-24
Status: `FAIL_DRIVING_COLLISION_AT_STEP_310`

## Objective

Scale the fast35 Stage 2 128-step clean PASS to a 512-step primary subset before
starting raw-sensor or fusion work.

## Input Evidence

Use the same fast35 checkpoint lineage that passed the official 128-step clean
gate:

```text
128 run id: b4c8e58_fast35_task_pilot
128 evidence: RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/b4c8e58_fast35_task_pilot/
checkpoint source: windows_carla_privileged_replay
action probe: probed_actions=512, action_counts={7:500,15:12}, reverse_action_count=0, single_action_collapse=false, recommended_for_clean_rerun=true
128 clean summary: source=windows_carla, status=PASS, steps=128/128, done_reason=horizon_reached, collision/offroad/stuck=0
```

## Runtime Config

Prepared config:

```text
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/b4c8e58_fast35_primary512/clean_512_stage2_privileged_fast35_with_checkpoints.yaml
```

Only these fields should differ from the fast35 128-step config:

```text
horizon: 512
carla.run_name: rcs_ac_v2_stage2_clean_512_fast35_b4c8e58
output_dir: RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512
summary_path: RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/summary.json
metrics_path: RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/metrics.csv
gate.pass_criteria.steps: 512
```

Checkpoint paths and sha256 hashes must remain unchanged.

## Windows Command

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90
git pull --ff-only

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_clean_eval.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90 `
  -ConfigPath RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\b4c8e58_fast35_primary512\clean_512_stage2_privileged_fast35_with_checkpoints.yaml `
  -RunOutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\b4c8e58_fast35_primary512
```

## PASS Criteria

```text
source = windows_carla
status = PASS
horizon = 512
steps = 512
done_reason = horizon_reached
failure_class = none
collision_count = 0
offroad_count = 0
stuck_count = 0
metrics.csv has exactly 512 data rows
adapter/runtime errors are absent
summary and metrics are mirrored into the run-id evidence directory
```

Do not treat the 128-step PASS as full clean-performance completion. If this
gate fails, keep the summary and metrics, classify the failure, and patch the
smallest cause before trying raw/fusion work.

## Official 512-Step Result

This run executed successfully from the official clean clone and produced a
validated failure summary. It is not a PASS.

```text
commit: 0b10ff5628b2b9616759bd4c6f9f60cc441c9b0d
branch: codex/rcs-ac-v2-raw2drive
source: windows_carla
status: FAIL
clean_gate_pass: false
horizon: 512
steps: 311
done_reason: collision
failure_class: driving
collision_count: 1
offroad_count: 0
stuck_count: 0
route_progress: 0.5641025641025641
episode_return: 19.066852226147766
metrics_rows: 311
adapter_errors: 0
runtime_errors: 0
policy action_counts: {7: 308, 15: 3}
```

Failure window:

```text
collision row: step 310
last action: action_id=7, throttle=0.3, brake=0.0, steer=0.0
last speed_kmh: 2.633119882897091
last route_progress: 0.5641025641025641
last 20 actions: all action_id=7
last 20 route_progress_delta values: mostly 0.0 with two 0.025641 increments
```

Local evidence:

```text
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/b4c8e58_fast35_primary512/summary.json
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/b4c8e58_fast35_primary512/metrics.csv
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/b4c8e58_fast35_primary512/clean_eval.log
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/b4c8e58_fast35_primary512/clean_512_stage2_privileged_fast35_with_checkpoints.yaml
```

Interpretation:

```text
The 512 infrastructure path is now validated: dependencies, CARLA readiness,
configured windows_512 output paths, summary validation, and RunOutputDir
mirroring all worked. The failure is a clean-driving collision, not adapter,
model-loading, or CARLA infrastructure.
```

## Next Step

The fast35 checkpoint learned mostly straight throttle actions from the 4x128
replay. The next experiment should collect a longer or failure-window privileged
replay that reaches this segment and includes steering labels, retrain from that
replay, rerun the checkpoint action probe, and rerun the 512 clean gate.

## Offline Steering Replay Sweep

After the 512 collision, a longer steer2 replay was collected on Windows with
2 episodes and 647 total steps. The replay reaches beyond the collision segment
and contains forward-steer labels, so several replay-only checkpoint candidates
were trained and rejected before any new CARLA run.

Strict offline probe criteria for these candidates:

```text
reverse_action_count = 0
nonzero_throttle_fraction >= 0.5
moving_steer_count >= 12
0.02 <= moving_steer_fraction <= 0.20
dominant_action_fraction <= 0.85
recommended_for_clean_rerun = true
```

Results:

```text
d4c5b70_steer2_seq010_balpow05_iw5:
  train: ImitationClassBalance=true, ImitationBalancePower=0.5, SteerSequenceFraction=0.10
  sampler: priority_window_count=212/617, priority_window_fraction=0.34359805510534847
  probe: action_counts={7:647}, moving_steer_fraction=0.0, dominant_action_fraction=1.0
  decision: reject before CARLA

d4c5b70_steer2_seq010_balpow0625_iw5:
  train: ImitationClassBalance=true, ImitationBalancePower=0.625, SteerSequenceFraction=0.10
  sampler: priority_window_count=212/617, priority_window_fraction=0.34359805510534847
  probe: action_counts={7:647}, moving_steer_fraction=0.0, dominant_action_fraction=1.0
  decision: reject before CARLA

d4c5b70_steer2_seq010_balpow075_iw5:
  train: ImitationClassBalance=true, ImitationBalancePower=0.75, SteerSequenceFraction=0.10
  sampler: priority_window_count=212/617, priority_window_fraction=0.34359805510534847
  probe: action_counts={4:639,7:8}, moving_steer_fraction=0.9876352395672334, dominant_action_fraction=0.9876352395672334
  decision: reject before CARLA

d4c5b70_steer2_seq010_nobal_iw5:
  train: ImitationClassBalance=false, SteerSequenceFraction=0.10
  sampler: priority_window_count=212/617, priority_window_fraction=0.34359805510534847
  probe: action_counts={7:647}, moving_steer_fraction=0.0, dominant_action_fraction=1.0
  decision: reject before CARLA
```

Interpretation:

```text
The replay has enough steering-priority windows, so the latest failures are not
explained by missing steering windows. The current replay-imitation policy is
bistable: weak balancing collapses to straight action 7, while stronger
balancing collapses to left-steer action 4. No candidate reached the offline
gate for a new 512 CARLA run.
```

## Offline Latent Separability Probe

A replay-latent linear probe was added after the offline action probes rejected
all four candidates. The probe freezes each candidate world model, extracts
RSSM latents for the same 647-step steer2 replay, trains a small linear
classifier from latent to discrete action label, and compares test accuracy
against the replay majority-action baseline.

Results:

```text
d4c5b70_steer2_seq010_balpow05_iw5:
  samples=647, latent_dim=160
  majority_action_fraction=0.6831530139103554
  linear_probe_train_accuracy=0.7698259353637695
  linear_probe_test_accuracy=0.7384615540504456

d4c5b70_steer2_seq010_balpow0625_iw5:
  samples=647, latent_dim=160
  majority_action_fraction=0.6831530139103554
  linear_probe_train_accuracy=0.7698259353637695
  linear_probe_test_accuracy=0.7384615540504456

d4c5b70_steer2_seq010_balpow075_iw5:
  samples=647, latent_dim=160
  majority_action_fraction=0.6831530139103554
  linear_probe_train_accuracy=0.7698259353637695
  linear_probe_test_accuracy=0.7384615540504456

d4c5b70_steer2_seq010_nobal_iw5:
  samples=647, latent_dim=160
  majority_action_fraction=0.6831530139103554
  linear_probe_train_accuracy=0.7698259353637695
  linear_probe_test_accuracy=0.7384615540504456
```

Interpretation:

```text
The latent contains some action information, but only weakly above the majority
baseline. This points to a representation/objective problem rather than a pure
"not enough steering replay windows" problem. The next patch should keep CARLA
off until an offline probe shows state-dependent straight/steer separation:
for example, add a contrastive/action-prediction auxiliary loss to the world
model latent or train the policy head with a sequence-aware balanced objective
that explicitly preserves straight/steer conditionality instead of only changing
global class weights.
```

Local evidence:

```text
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/d4c5b70_steer2_seq010_balpow05_iw5/
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/d4c5b70_steer2_seq010_balpow0625_iw5/
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/d4c5b70_steer2_seq010_balpow075_iw5/
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/d4c5b70_steer2_seq010_nobal_iw5/
```

## Action-Aux Candidate 512 Result

After the latent probe showed weak action separation, the replay trainer was
patched to support a world-model action-prediction auxiliary loss:

```text
commit: 7f876568a9c9a42e38e520f1e2661993805c15ec
candidate: 7f87656_steer2_seq010_aux1_balpow05_iw5
WorldModelActionAuxWeight: 1.0
ImitationClassBalance: true
ImitationBalancePower: 0.5
SteerSequenceFraction: 0.10
ImitationWeight: 5.0
BehaviorWeight: 0.0
```

Offline gates:

```text
checkpoint action probe:
  probed_actions: 647
  action_counts: {1:8, 4:73, 7:436, 10:14, 12:1, 15:115}
  reverse_action_count: 0
  nonzero_throttle_fraction: 1.0
  moving_steer_fraction: 0.14837712519319937
  dominant_action_fraction: 0.6738794435857806
  recommended_for_clean_rerun: true

latent probe:
  majority_action_fraction: 0.6831530139103554
  linear_probe_train_accuracy: 0.9168278574943542
  linear_probe_test_accuracy: 0.9076923131942749
```

Official 512 clean eval:

```text
clean clone: C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656
source: windows_carla
status: FAIL
clean_gate_pass: false
horizon: 512
steps: 178
done_reason: collision
failure_class: driving
collision_count: 1
offroad_count: 0
stuck_count: 0
route_progress: 0.6410256410256411
episode_return: 5.680538470046445
metrics_rows: 178
policy action_counts in clean run: {1:2, 4:15, 7:76, 10:3, 15:82}
```

Failure window:

```text
collision row: step 177
last action: action_id=4, throttle=0.3, brake=0.0, steer=-0.1
last speed_kmh: 22.579665788706205
last route_progress: 0.6410256410256411
last 20 actions: 7,7,7,7,7,7,4,4,4,7,7,7,7,7,10,10,10,4,4,4
```

Interpretation:

```text
The action-aux patch fixed the offline collapse: the policy now emits a
state-dependent mixture of straight, left, and right forward actions and the
latent probe is far above the majority baseline. The 512 clean gate still fails
as a real driving collision. The next patch should not revert to raw/fusion; it
should tune the privileged policy for safer speed/turn timing around the
0.59-0.64 route-progress segment, likely by collecting or relabeling a local
failure-window replay with lower target speed / earlier braking or stronger
turn timing, then rerunning the same offline probes before another 512 attempt.
```

Local evidence:

```text
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_512/7f87656_steer2_seq010_aux1_balpow05_iw5/
```

## Target-24 Failure-Window Replay

After the action-aux candidate failed at route progress `0.6410256410256411`,
a lower target-speed replay was collected to prepare the next candidate. This
is training data only, not clean evidence.

```text
replay: RCS-AC-v2/experiments/02_privileged_policy_clean/replay/windows_privileged_7f87656_t24_failure_window/
episodes: 2
total_steps: 520
target_speed_kmh: 24.0
steer_sign: -1.0
action_counts: {7:386, 10:26, 13:8, 15:74, 16:20, 19:6}
```

Interpretation:

```text
The target-24 replay produces a lower-speed/right-steer label mix instead of
the earlier left/right oscillation seen in the failed clean run. The intended
next experiment is to train an action-aux checkpoint from this replay, run the
same checkpoint action probe and replay latent probe, and attempt another 512
clean eval only if the offline gates pass. The Windows peer disconnected during
the first target-24 checkpoint training attempt, so checkpoint/probe status for
this replay is still pending.
```

## Target-24 Decision Log

This log exists to prevent an offline probe result from being mistaken for
clean performance. `stage2_candidate_cross_replay_gate.json` can only promote a
candidate to "ready for 512 clean eval"; it is not a clean PASS.

```text
2026-05-24:
  state: pending_windows_resume
  replay: windows_privileged_7f87656_t24_failure_window
  checkpoint_output_dir: windows_replay_7f87656_t24_aux1_balpow05_iw5
  latest_pointer: windows_replay_7f87656_t24_aux1_balpow05_iw5.latest.json
  decision_summary: summarize_stage2_candidate_gate.ps1 after cross replay gate exists
  clean_gate_evidence: false
  next_command: run_stage2_target24_candidate_cycle.ps1 -StartIfMissing -ProbeWhenReady

2026-05-25:
  state: rejected_by_cross_replay_gate
  commit: 610a590
  checkpoint_output_dir: windows_replay_7f87656_t24_aux1_balpow05_iw5_rerun_20260525T021559
  target24_action_probe: recommended_for_clean_rerun=true, action_counts={7:398,10:26,13:8,15:62,16:20,19:6}, moving_steer_fraction=0.11538461538461539, dominant_action_fraction=0.7653846153846153
  target24_latent_probe: linear_probe_test_accuracy=0.9807692170143127, majority_action_fraction=0.7423076923076923, margin=0.2384615247066204
  steer2_action_probe: recommended_for_clean_rerun=false, action_counts={7:377,10:134,13:19,15:57,16:57,19:3}, moving_steer_fraction=0.32921174652241114, dominant_action_fraction=0.5826893353941267
  steer2_latent_probe: linear_probe_test_accuracy=0.800000011920929, majority_action_fraction=0.6831530139103554, margin=0.11684699801057352
  gate_failures: steer2 action probe recommended_for_clean_rerun is not true; steer2 latent margin below 0.15
  decision_summary: run_target22_fallback
  clean_gate_evidence: false
  target22_status: replay collected and detached training started as RCSV2Stage2target22Aux1BalPow05; Windows SSH/Tailscale became unreachable before completion check

2026-05-26:
  state: corr050_branch_rejected_offline_and_short_episode_cause_confirmed
  commit: 32a65cb
  cause_under_test: BF7 512 clean collision window near route_progress 0.50
  replay_branches: bf7_route050_collision_window_t12_min5 and bf7_route050_collision_window_t16_min5
  candidate_result: corr050 t12/t16 weight and false-positive variants were rejected before official CARLA eval because steer2 moving-steer false positives stayed above the 0.20 gate
  diagnostic_finding: identical steer2 distributions across corr050 variants can happen when short component episodes are filtered out by sequence_length or when episode-level weights are not visible in realized batches
  integrated_patch: replay training now records component_sampler_stats and sampler included/dropped episode lists, and raises before training if an explicitly positive component weight has no sequence_length-eligible episodes
  windows_diagnostic: both mixed_t22_steer2_corr050_t12 and mixed_t22_steer2_corr050_t16 reported lengths [10,10,10,10,314,333,261,261] and raised "component 'corr050' has positive sample weight but no episodes long enough for sequence_length=16"
  clean_gate_evidence: false
  next_action: recollect corr050 with MinRecordedSteps >= sequence_length, rebuild the mixed replay manifest, verify component_sampler_stats includes corr050 windows, then train/probe before any official 512 clean eval

2026-05-26:
  state: corr048_min16_candidate_rejected_by_cross_replay_gate
  commit: 33d1705
  corrected_replay: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\bf7_route048_collision_window_t16_min16
  corrected_replay_summary: episodes=4, total_steps=68, episode_lengths=[17,17,17,17], action_counts={2:16,5:32,8:20}
  mixed_replay: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\mixed_t22_steer2_corr048_t16_min16
  component_sampler_stats: corr048 included_episodes=4, dropped_short_episodes=0, window_count=8, configured_episode_sample_prob_by_component={target22:0.4,steer2:0.4,corr048:0.2}
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\33d1705_corr048_t16_min16_w025_bf7base
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\33d1705_corr048_t16_min16_w025_bf7base_cross_probe
  target22_action_probe: recommended_for_clean_rerun=false, action_counts={7:484,4:22,10:8,16:4,19:4}, moving_steer_fraction=0.0727969349, dominant_action_fraction=0.9272030651
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=false, action_counts={7:590,4:57}, moving_steer_fraction=0.0880989181, dominant_action_fraction=0.9119010819
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  gate_failures: target22 action probe recommended_for_clean_rerun is not true; steer2 action probe recommended_for_clean_rerun is not true
  decision_summary: reject_before_carla
  clean_gate_evidence: false
  next_action: do not run official 512 for w0.25; if continuing this branch, reduce straight-action collapse before CARLA, for example lower corr048 probability/priority or restore BF7 steer balance while keeping the min16 replay guard

2026-05-26:
  state: corr048_w005_sweep_rejected_by_cross_replay_gate
  commit: 095edec
  subagent_finding: Jason recommended lowering corr048 component weight to 0.05 and keeping BF7-like settings; official CARLA remained forbidden unless cross-replay gate passed
  candidates:
    - name: 095edec_corr048_t16_min16_w005_bf7params
      steer_sequence_fraction: 0.05
      component_weights: target22=1,steer2=1,corr048=0.05
      target22_action_probe: recommended_for_clean_rerun=false, action_counts={15:460,4:34,10:6,13:2,19:2,1:2,7:16}, moving_steer_fraction=0.0881226054, dominant_action_fraction=0.8812260536
      target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
      steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:438,7:141,4:55,1:10,12:3}, moving_steer_fraction=0.1051004637, dominant_action_fraction=0.6769706337
      steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
      decision_summary: reject_before_carla_target22_dominance
    - name: 095edec_corr048_t16_min16_w005_ssf010
      steer_sequence_fraction: 0.10
      component_weights: target22=1,steer2=1,corr048=0.05
      target22_action_probe: recommended_for_clean_rerun=false, action_counts={15:456,10:32,16:20,13:8,19:6}, moving_steer_fraction=0.1264367816, dominant_action_fraction=0.8735632184
      target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
      steer2_action_probe: recommended_for_clean_rerun=false, action_counts={15:425,10:203,7:19}, moving_steer_fraction=0.3137557960, dominant_action_fraction=0.6568778980
      steer2_latent_probe: linear_probe_test_accuracy=0.8769230843, majority_action_fraction=0.6831530139, margin=0.1937700703
      decision_summary: reject_before_carla_target22_dominance_and_steer2_oversteer
    - name: 095edec_corr048_t16_min16_w005_ssf015
      steer_sequence_fraction: 0.15
      component_weights: target22=1,steer2=1,corr048=0.05
      target22_action_probe: recommended_for_clean_rerun=false, action_counts={15:450,10:28,16:20,13:8,19:6,7:10}, moving_steer_fraction=0.1187739464, dominant_action_fraction=0.8620689655
      target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
      steer2_action_probe: recommended_for_clean_rerun=false, action_counts={15:421,10:153,16:58,13:5,7:10}, moving_steer_fraction=0.3338485317, dominant_action_fraction=0.6506955178
      steer2_latent_probe: linear_probe_test_accuracy=0.9076923132, majority_action_fraction=0.6831530139, margin=0.2245392993
      decision_summary: reject_before_carla_target22_dominance_and_steer2_oversteer
  clean_gate_evidence: false
  next_action: stop plain corr048 weight/ssf sweeps; next patch should control action15/right-turn dominance directly while preserving steer2 false-positive limits

2026-05-26:
  state: corr048_w005_cap15_rejected_by_cross_replay_gate
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_cap15w1c82
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\8b78015_corr048_w005_ssf005_cap15w1c82
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_cap15w1c82_cross_probe
  control_change: ActionMassCapWeight=1.0, ActionMassCaps=15=0.82
  target22_action_probe: recommended_for_clean_rerun=false, action_counts={15:460,4:34,10:6,13:2,19:2,1:2,7:16}, moving_steer_fraction=0.0881226054, dominant_action_fraction=0.8812260536
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:438,7:141,4:55,1:10,12:3}, moving_steer_fraction=0.1051004637, dominant_action_fraction=0.6769706337
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  gate_failures: target22 action probe recommended_for_clean_rerun is not true
  decision_summary: reject_before_carla_target22_action15_dominance
  clean_gate_evidence: false
  next_action: do not run official 512 clean eval for cap15w1c82; if continuing this line, tighten action15 control enough to move target22 dominant_action_fraction below 0.85 while watching that steer2 moving_steer_fraction stays below 0.20

2026-05-26:
  state: corr048_w005_cap15w4_rejected_by_cross_replay_gate
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_cap15w4c80
  control_change: ActionMassCapWeight=4.0, ActionMassCaps=15=0.80
  target22_action_probe: recommended_for_clean_rerun=false, action_counts={15:460,4:34,10:6,13:2,19:2,1:2,7:16}, moving_steer_fraction=0.0881226054, dominant_action_fraction=0.8812260536
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:438,7:141,4:55,1:10,12:3}, moving_steer_fraction=0.1051004637, dominant_action_fraction=0.6769706337
  gate_failures: target22 action probe recommended_for_clean_rerun is not true
  decision_summary: reject_before_carla_same_as_cap15w1
  clean_gate_evidence: false

2026-05-26:
  state: corr048_w005_logit15_official512_fail_collision
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_logit15w05m0
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\8b78015_corr048_w005_ssf005_logit15w05m0
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit15w05m0_cross_probe
  official_run_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit15w05m0_official_clean512
  control_change: RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=15, RouteActionLogitCapMargin=0.0, RouteActionLogitCapMinRouteProgress=0.0, RouteActionLogitCapMaxRouteProgress=1.0
  target22_action_probe: recommended_for_clean_rerun=true, action_counts={15:442,4:40,7:38,19:2}, moving_steer_fraction=0.0804597701, dominant_action_fraction=0.8467432950
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:420,7:167,4:49,1:8,12:3}, moving_steer_fraction=0.0927357032, dominant_action_fraction=0.6491499227
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  cross_replay_gate: ready_for_clean_eval=true, failures=[]
  official_512_summary: source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=146/512, route_progress=0.6666666667, collision_count=1, offroad_count=0, stuck_count=0, clean_gate_pass=false
  official_512_action_counts: {15:117,7:17,4:10,12:2}
  official_512_failure_tail: step144 action12 throttle=0.7 steer=-0.1 route_progress=0.6666666667; step145 action12 throttle=0.7 steer=-0.1 collision=true
  clean_gate_evidence: false
  next_action: patch the late route_progress 0.64-0.70 high-throttle-left collision, likely by using a route-local logit/throttle control on action12 before another official 512 eval

2026-05-26:
  state: corr048_w005_logit1215_official512_fail_collision
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_logit1215w05m0
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\8b78015_corr048_w005_ssf005_logit1215w05m0
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit1215w05m0_cross_probe
  official_run_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit1215w05m0_official_clean512
  control_change: RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=12,15, RouteActionLogitCapMargin=0.0, RouteActionLogitCapMinRouteProgress=0.0, RouteActionLogitCapMaxRouteProgress=1.0
  target22_action_probe: recommended_for_clean_rerun=true, action_counts={15:442,4:40,7:38,19:2}, moving_steer_fraction=0.0804597701, dominant_action_fraction=0.8467432950
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:419,7:170,4:48,1:8,12:2}, moving_steer_fraction=0.0896445131, dominant_action_fraction=0.6476043277
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  cross_replay_gate: ready_for_clean_eval=true, failures=[]
  official_512_summary: source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=146/512, route_progress=0.6666666667, collision_count=1, offroad_count=0, stuck_count=0, clean_gate_pass=false
  official_512_action_counts: {15:116,7:18,4:10,1:1,12:1}
  official_512_failure_tail: step145 action12 throttle=0.7 steer=-0.1 route_progress=0.6666666667 collision=true
  clean_gate_evidence: false
  next_action: action12 remains the immediate collision action despite the weak 12,15 logit cap; try a stronger 12,15 logit cap or route-local throttle/logit control offline before another official 512 eval

2026-05-26:
  state: corr048_w005_logit1215w2_rejected_by_cross_replay_gate
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_logit1215w2m0
  control_change: RouteActionLogitCapWeight=2.0, RouteActionLogitCapIds=12,15, RouteActionLogitCapMargin=0.0
  target22_action_probe: recommended_for_clean_rerun=false, action_counts={7:480,4:40,19:2}, moving_steer_fraction=0.0804597701, dominant_action_fraction=0.9195402299
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=false, action_counts={7:589,4:51,1:7}, moving_steer_fraction=0.0896445131, dominant_action_fraction=0.9103554869
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  gate_failures: target22 action probe recommended_for_clean_rerun is not true; steer2 action probe recommended_for_clean_rerun is not true
  decision_summary: reject_before_carla_full_route_logit_cap_too_strong
  clean_gate_evidence: false

2026-05-26:
  state: corr048_w005_logit1215w1_official512_fail_collision_offroad
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_logit1215w1m0
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\8b78015_corr048_w005_ssf005_logit1215w1m0
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit1215w1m0_cross_probe
  official_run_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit1215w1m0_official_clean512
  control_change: RouteActionLogitCapWeight=1.0, RouteActionLogitCapIds=12,15, RouteActionLogitCapMargin=0.0
  target22_action_probe: recommended_for_clean_rerun=true, action_counts={7:262,15:220,4:38,19:2}, moving_steer_fraction=0.0766283525, dominant_action_fraction=0.5019157088
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={7:383,15:208,4:49,1:7}, moving_steer_fraction=0.0865533230, dominant_action_fraction=0.5919629057
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  cross_replay_gate: ready_for_clean_eval=true, failures=[]
  official_512_summary: source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=148/512, route_progress=0.5384615385, collision_count=1, offroad_count=1, stuck_count=0, clean_gate_pass=false
  official_512_action_counts: {15:80,7:47,10:10,16:5,19:4,13:2}
  official_512_failure_tail: steps144-147 action19 throttle=0.3 steer=0.5 route_progress=0.5384615385; step147 collision=true offroad=true
  clean_gate_evidence: false
  next_action: stop full-route 12,15 logit-weight sweeps; w0.5 reaches route_progress 0.6667 while w1.0 reintroduces earlier right-oversteer, so next use route-local late action12/throttle control while preserving action15 w0.5 behavior

2026-05-26:
  state: corr048_w005_logit15_late_throttle_ceiling_official512_fail_collision
  commit: 8b78015
  candidate: 8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4_cross_probe
  official_run_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\8b78015_corr048_w005_ssf005_logit15w05_lateceil03w4_official_clean512
  control_change: RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=15, RouteActionLogitCapMargin=0.0, RouteThrottleCeilingWeight=4.0, RouteThrottleCeiling=0.3, RouteThrottleCeilingMinRouteProgress=0.64, RouteThrottleCeilingMaxRouteProgress=0.70
  target22_action_probe: recommended_for_clean_rerun=true, action_counts={15:442,4:40,7:38,19:2}, moving_steer_fraction=0.0804597701, dominant_action_fraction=0.8467432950
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:414,4:48,7:177,1:8}, moving_steer_fraction=0.0865533230, dominant_action_fraction=0.6398763524
  steer2_latent_probe: linear_probe_test_accuracy=0.8999999762, majority_action_fraction=0.6831530139, margin=0.2168469622
  cross_replay_gate: ready_for_clean_eval=true, failures=[]
  official_512_summary: source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=146/512, route_progress=0.6666666667, collision_count=1, offroad_count=0, stuck_count=0, clean_gate_pass=false
  official_512_action_counts: {15:116,7:19,4:10,1:1}
  official_512_failure_tail: steps140-145 mostly action7 throttle=0.3 steer=0.0; step145 action7 throttle=0.3 steer=0.0 route_progress=0.6666666667 collision=true
  clean_gate_evidence: false
  next_action: stop throttle-only late-window tuning; action12 high-throttle tail is gone, but the same geometry still collides under slow straight action7, so collect or use late collision-window replay/correction around route_progress 0.64-0.70 before another official 512 eval

2026-05-26:
  state: late056_warmup_replay_low_weight_rejected_by_cross_replay_gate
  commit: 4e9f4b3
  replay_t16: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_4e9f4b3_late056_collision_window_t16_min16
  replay_t16_summary: episodes=4, total_steps=144, action_counts={2:4,28:68,31:24,34:48}, target_speed_kmh=16, record_after_route_progress=0.56, min_recorded_steps=16
  replay_t24: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_4e9f4b3_late056_collision_window_t24_min16
  replay_t24_summary: episodes=4, total_steps=142, action_counts={2:4,28:48,30:18,31:25,34:47}, target_speed_kmh=24, record_after_route_progress=0.56, min_recorded_steps=16
  candidate: 4e9f4b3_corr048_w005_late056t24_w002_logit15_lateceil
  mixed_replay_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\mixed_t22_steer2_corr048_late056t24_w002
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\4e9f4b3_corr048_w005_late056t24_w002_logit15_lateceil
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\4e9f4b3_corr048_w005_late056t24_w002_logit15_lateceil_cross_probe
  control_change: ComponentSampleWeights=target22=1,steer2=1,corr048=0.05,late056t24=0.02, RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=15, RouteThrottleCeilingWeight=4.0, RouteThrottleCeiling=0.3, RouteThrottleMinRouteProgress=0.64, RouteThrottleMaxRouteProgress=0.70
  target22_action_probe: recommended_for_clean_rerun=false, action_counts={15:462,4:42,7:12,12:4,1:2}, moving_steer_fraction=0.0919540230, dominant_action_fraction=0.8850574713
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910, majority_action_fraction=0.7931034483, margin=0.1973727428
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={15:445,7:117,4:57,12:16,1:12}, moving_steer_fraction=0.1313755796, dominant_action_fraction=0.6877897991
  steer2_latent_probe: linear_probe_test_accuracy=0.9076923132, majority_action_fraction=0.6831530139, margin=0.2245392993
  cross_replay_gate: ready_for_clean_eval=false, failures=[target22 action probe recommended_for_clean_rerun is not true]
  clean_gate_evidence: false
  next_action: do not run official 512; late056 teacher replay is dominated by zero-throttle steering labels and worsens target22 action15 dominance, so change the late labeling/action-selection heuristic or use a different local correction before retrying offline

2026-05-26:
  state: late045_min_recorded_throttle_official512_fail_progress077
  commit: 5bbca2a
  code_change: privileged replay collector supports min_recorded_throttle so recorded correction labels keep at least throttle=0.3 when requested
  failed_collection_late056: record_after_route_progress=0.56, target_speed_kmh=24, min_recorded_throttle=0.3, min_recorded_steps=16, result=1 eligible episode after 12 attempts, rejected_for_training_volume
  failed_collection_late050: record_after_route_progress=0.50, target_speed_kmh=24, min_recorded_throttle=0.3, min_recorded_steps=16, result=0 eligible episodes, rejected_for_training_volume
  replay_late045: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_5bbca2a_late045_t24_minthr03_min16
  replay_late045_summary: episodes=4, total_steps=64, episode_lengths=[16,16,16,16], action_counts={7:8,10:12,13:16,16:16,19:12}, target_speed_kmh=24, min_recorded_throttle=0.3, record_after_route_progress=0.45, min_recorded_steps=16
  candidate: 5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil
  mixed_replay_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\replay\mixed_t22_steer2_corr048_late045t24mth03_w002
  checkpoint_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil
  cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil_cross_probe
  control_change: ComponentSampleWeights=target22=1,steer2=1,corr048=0.05,late045t24mth03=0.02, RouteActionLogitCapWeight=0.5, RouteActionLogitCapIds=15, RouteThrottleCeilingWeight=4.0, RouteThrottleCeiling=0.3, RouteThrottle window=0.64-0.70
  target22_action_probe: recommended_for_clean_rerun=true, action_counts={7:354,15:110,4:50,19:4,13:2,1:2}, moving_steer_fraction=0.1111111111, dominant_action_fraction=0.6781609195
  target22_latent_probe: linear_probe_test_accuracy=0.9904761910
  steer2_action_probe: recommended_for_clean_rerun=true, action_counts={7:457,15:84,4:75,10:18,1:9,12:4}, moving_steer_fraction=0.1638330757, dominant_action_fraction=0.7063369397
  steer2_latent_probe: linear_probe_test_accuracy=0.9230769277
  cross_replay_gate: ready_for_clean_eval=true, failures=[]
  official_run_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil_official_clean512
  official_512_summary: source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=361/512, route_progress=0.7692307692, collision_count=1, offroad_count=0, stuck_count=0, clean_gate_pass=false
  official_512_action_counts: {7:249,4:52,15:41,1:7,10:6,19:3,13:2,16:1}
  official_512_failure_tail: route_progress=0.7692307692; steps349-354 action10 throttle=0.3 steer=0.1; steps355-356 action13 steer=0.2; step357 action16 steer=0.3; steps358-360 action19 steer=0.5; collision at step360
  mask19_cross_probe_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil_mask19_cross_probe
  mask19_cross_replay_gate: ready_for_clean_eval=true, ForbiddenActionIds=19
  mask19_official_run_output_dir: RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\5bbca2a_corr048_w005_late045t24mth03_w002_logit15_lateceil_mask19_official_clean512
  mask19_official_512_summary: source=windows_carla, status=FAIL, failure_class=driving, done_reason=collision, steps=350/512, route_progress=0.7692307692, collision_count=1, offroad_count=0, stuck_count=0, clean_gate_pass=false
  mask19_official_512_action_counts: {1:7,4:42,7:248,10:7,13:3,15:41,16:2}
  mask19_official_512_failure_tail: steps338-344 action10 throttle=0.3 steer=0.1 at route_progress=0.7435897436; steps345-347 action13 steer=0.2; steps348-349 action16 steer=0.3; collision at step349
  clean_gate_evidence: false
  decision_summary: progress improved from 0.6667 to 0.7692, but masking action19 only shifts the collision to action16; next patch must target the 0.74-0.77 late right-steer transition before another official 512 eval
```

Decision rules:

```text
if target-24 detached training is still running:
  leave status as running and do not start target-22 yet

if target-24 cross replay gate rejects:
  record the action/latent probe paths and rejection reason here
  run summarize_stage2_candidate_gate.ps1 and record decision=run_target22_fallback
  then run target-22 with run_stage2_target_speed_replay_cycle.ps1

if target-24 cross replay gate passes:
  record ready_for_clean_eval=true here
  run summarize_stage2_candidate_gate.ps1 and record decision=run_official_512_clean_eval
  run exactly one official 512 clean eval from the clean clone
  classify the resulting windows_carla summary.json and metrics.csv

only source=windows_carla, clean_gate_pass=true, 512/512 metrics rows,
horizon_reached, and zero collision/offroad/stuck can close the clean gate
```
