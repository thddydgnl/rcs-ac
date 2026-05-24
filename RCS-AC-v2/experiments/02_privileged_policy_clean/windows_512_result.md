# Stage 2 Windows 512-Step Primary Clean Result

Date: 2026-05-24
Status: `PENDING_OFFICIAL_WINDOWS_RUN`

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
