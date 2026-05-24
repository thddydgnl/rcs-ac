# Stage 2 Windows 128-Step Clean Result

Date: 2026-05-24
Status: `NOT_RUN`

## Objective

Run the first official Stage 2 privileged policy clean gate on Windows CARLA for
128 steps after committed/pushed code is pulled on Windows and the CARLA
readiness guard passes.

## Required Command Shape

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_clean_eval.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml
```

The config now contains the default Stage 2 adapter factory:

```text
rcs_ac_v2.evaluation.carla_adapter:create_stage2_carla_adapter
```

Use `-AdapterFactory module:function` only when intentionally overriding that
default for a Windows-local probe.

Dry-run contract check:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_clean_eval.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml `
  -DryRun
```

## Gate

This result is not a PASS until the summary source is `windows_carla`.

PASS requires:

```text
source = windows_carla
status = PASS
steps = horizon = 128
done_reason = horizon_reached
collision_count = 0
offroad_count = 0
stuck_count = 0
route_progress finite
episode_return finite
action_table_version = raw2drive_table12_row_major_v1
reward_contract_version = stage0_route_progress_penalty_v1
failure_class = none
summary.json exists
metrics.csv exists
metrics.csv has exactly 128 step rows
metrics.csv required fields match config
metrics.csv has no adapter_error/runtime_error flags
commit, branch, Windows command, config, CARLA version, CARLA town, output path populated
pulled files include summary.json and metrics.csv
```

## Metrics

```text
commit:
branch:
Windows command:
config:
CARLA version:
CARLA town:
stage:
gate:
output path:
pulled files:
failure class:
next action:
CARLA readiness result:
summary.json:
steps:
episode_return:
done_reason:
collision_count:
offroad_count:
stuck_count:
route_progress:
failure_class:
```

## Current Blocker

The locally testable Stage 2 clean-eval adapter loop, metrics CSV writer,
summary validator, Windows runner invocation path, and shared-core CARLA
adapter factory are implemented. The local synthetic checkpoint export path is
also implemented and verified, but those checkpoints are plumbing evidence
only. The official run is still `NOT_RUN` because real world-model/policy
checkpoint paths are still null and the Windows shared-core env has not yet
been verified to expose or be adaptable to strict Stage 2 privileged tensors.

A local real-mode probe with the base null-checkpoint config exits with code 2,
writes a valid failure summary, and classifies the setup as
`failure_class=model` / `done_reason=world_model_checkpoint_missing`. That is
the expected setup failure before real checkpoints are promoted.

One reproducibility blocker remains before an official run: the current runner
uses `RCS-AC-v1.1` shared-core env builder code and `shared/scripts/windows`
readiness scripts as temporary Windows dependencies. The official run must
either confirm those paths exist on Windows after `git pull --ff-only`, or move
the minimum required runner/env-builder pieces into `RCS-AC-v2`.

Local adapter-loop tests are not clean evidence. This file can move to PASS
only after Windows runs the command above after `git pull --ff-only`, the CARLA
readiness guard passes, and the resulting `summary.json` validates with
`source=windows_carla`.
