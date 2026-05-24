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
