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
