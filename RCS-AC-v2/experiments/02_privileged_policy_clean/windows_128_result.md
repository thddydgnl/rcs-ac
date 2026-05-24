# Stage 2 Windows 128-Step Clean Result

Date: 2026-05-24
Status: `PRIVILEGED_OBS_PROBE_PASS__OFFICIAL_CLEAN_NOT_RUN`

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
commit: 5b00347722c161c41bed62db731230d163513a1b
branch: codex/rcs-ac-v2-raw2drive
Windows command: scratch archive runtime; run_stage2_privileged_obs_probe.ps1 -StepNeutral -SkipGitPull
config: RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml
CARLA version: 0.9.16
CARLA town: Town04
stage: stage2_privileged_policy_clean
gate: privileged observation probe only; clean_gate_evidence=false
output path: RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/privileged_obs_probe.json
pulled files: privileged_obs_probe.json
failure class: none
next action: promote real checkpoints and run official 128-step clean eval from a clean Windows sync lane
CARLA readiness result: PASS, target already loaded
summary.json: not produced by probe
steps: neutral reset + one neutral step only
episode_return: not applicable to probe
done_reason: privileged_obs_probe_pass
collision_count: 0 during neutral probe
offroad_count: 0 during neutral probe
stuck_count: 0 during neutral probe
route_progress: neutral_step_route_progress=0.02564102564102564
failure_class: none
```

## Privileged Observation Probe

This is non-clean infrastructure evidence, not a clean-performance PASS.

Local archive:

```text
/Users/song-yonghwi/code/windows_results/rcs-ac-v2/stage2_archive_probe_20260524_160752_5b003477/
```

Windows runtime:

```text
C:\Users\thddy\Documents\code\RCS-AC-v2-runtime-20260524_160752-5b003477
```

Result:

```text
status=PROBE_PASS
clean_gate_evidence=false
failure_class=none
done_reason=privileged_obs_probe_pass
privileged_observation_source=carla_simulator_state
privileged_observation_schema=carla_simulator_semantic43_v1
privileged_observation_clean_gate_eligible=True
privileged_bev_channel_names_sha256=382e53122998198ad8f392c7298465057d364b23cf1fedb32465872ae43320a3
privileged_bev_shape=[1, 1, 43, 16, 16]
privileged_ego_shape=[1, 1, 8]
privileged_bev_finite=True
privileged_ego_finite=True
semantic_nonzero_channel_names=drivable_road,lane_marking,route_path,route_target,ego_vehicle,static_obstacle,junction,route_straight,ego_footprint_tail,actor_near_field,actor_far_field
neutral_step_done_reason=running
neutral_step_collision=False
neutral_step_offroad=False
neutral_step_stuck=False
```

The probe confirmed that the current shared-core raw observation lacks
`privileged_bev`/`privileged_ego`, but the v2 adapter can now construct the
pinned simulator-state semantic43/ego8 observation from CARLA state. Raw-derived
or debug privileged tensors remain ineligible for clean PASS.

## Current Blocker

The locally testable Stage 2 clean-eval adapter loop, metrics CSV writer,
summary validator, Windows runner invocation path, shared-core CARLA adapter
factory, simulator-state privileged observation builder, and Windows privileged
obs probe are implemented. The replay collection/training path is also present:
collect privileged CARLA replay, train replay-source world-model/policy
checkpoints, and fill config paths/sha from the generated manifest. The local
synthetic checkpoint export path remains plumbing evidence only, and the clean
config rejects it by provenance. The official run is still not run because real
world-model/policy checkpoint paths are still null.

A local real-mode probe with the base null-checkpoint config exits with code 2,
writes a valid failure summary, and classifies the setup as
`failure_class=model` / `done_reason=world_model_checkpoint_missing`. That is
the expected setup failure before real checkpoints are promoted.

The v2 readiness scripts now live under `RCS-AC-v2/scripts/windows`, and the
Stage 2 runner has a dependency probe. One reproducibility blocker remains
before an official run: the current runner still uses `RCS-AC-v1.1` shared-core
env builder code plus a sibling `shared_core` package as temporary Windows
dependencies. The scratch archive probe confirmed these paths are available on
Windows, but official performance evidence should still come from a clean
checkout/git-pull lane or a documented clean clone.

The existing Windows checkout was not used for this probe because it was on an
unrelated dirty branch and `git fetch` hit interactive GitHub credential
failure. The archive lane avoided touching that worktree and is acceptable for
infrastructure validation, but not for final clean-performance claims.

Before running clean eval, run the non-clean shape probe:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_privileged_obs_probe.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml `
  -StepNeutral
```

This must produce `privileged_obs_probe.json` with `clean_gate_evidence=false`,
strict privileged BEV/ego shapes, finite tensors, raw obs/info keys,
simulator-state provenance, schema/hash evidence, and neutral-step info. It
must not be interpreted as clean PASS.

Local adapter-loop tests are not clean evidence. This file can move to PASS
only after Windows runs the command above after `git pull --ff-only`, the CARLA
readiness guard passes, and the resulting `summary.json` validates with
`source=windows_carla`.
