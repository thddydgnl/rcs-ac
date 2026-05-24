# Stage 2 Windows 128-Step Clean Result

Date: 2026-05-24
Status: `FIRST_128_CLEAN_FAIL_STUCK__BC_RETRAIN_PENDING_VERIFY`

## Objective

Run the first official Stage 2 privileged policy clean gate on Windows CARLA for
128 steps after committed/pushed code is pulled on Windows and the CARLA
readiness guard passes.

## Required Command Shape

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90
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

## First 128-Step Clean Eval

This is an official clean-clone run, but it is **not** a PASS.

```text
commit: db1e8b6193202e122cc2a8a4c3d0de606d5c428e
branch: codex/rcs-ac-v2-raw2drive
source: windows_carla
status: FAIL
clean_gate_pass: false
horizon: 128
steps: 64
done_reason: stuck
failure_class: driving
collision_count: 0
offroad_count: 0
stuck_count: 1
route_progress: 0.02564102564102564
episode_return: 3.3699999988079017
metrics_rows: 64
adapter_errors: 0
runtime_errors: 0
reverse_actions: 0
```

Local evidence:

```text
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/db1e8b6_pilot/summary.json
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/db1e8b6_pilot/metrics.csv
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/db1e8b6_pilot/stage2_privileged_replay_manifest.json
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/db1e8b6_pilot/stage2_privileged_replay_checkpoint_manifest.json
RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/db1e8b6_pilot/clean_128_stage2_privileged_with_checkpoints.yaml
```

Artifact copy policy:

```text
run_stage2_clean_eval.ps1 -RunOutputDir may mirror validated Windows outputs
into a run-id evidence folder. The mirrored set is transport evidence only:
summary.json, metrics.csv, and the exact runtime config retain the authoritative
Windows-generated metadata. A copied artifact set does not change status unless
summary.json validates with source=windows_carla, steps=horizon, done_reason
horizon_reached, zero collision/offroad/stuck, failure_class=none, and
metrics.csv has exactly the expected step rows.
```

Diagnosis:

```text
policy action_counts: action_id=8 for all 64 steps
action_id=8: throttle=0.0, brake=0.0, steer=0.6
speed_kmh mean: 0.29480263839785836
route_progress_delta: 0.0 for all rows
result: no forward progress, stuck at step 63
```

Follow-up already implemented:

```text
db1e8b6: masked reverse logits for direct_3d/stage4_v2 when allow_reverse_direct_3d=false
b20d8eb: added replay imitation warm-start to policy training
Windows retraining completed to windows_replay_b20d8eb_bc_pilot
pending: action probe and clean rerun after Windows SSH/Tailscale recovers
```

## Privileged Probe Metrics

```text
commit: e42dc6a8861146dd40c671c439061e1de622f918
branch: codex/rcs-ac-v2-raw2drive
Windows command: official clean clone; run_stage2_privileged_obs_probe.ps1 -StepNeutral -SkipGitPull
config: RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml
CARLA version: 0.9.16
CARLA town: Town04
stage: stage2_privileged_policy_clean
gate: privileged observation probe only; clean_gate_evidence=false
output path: RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/privileged_obs_probe.json
pulled files: clean_clone_e42dc6a/privileged_obs_probe.json, clean_clone_e42dc6a/privileged_obs_probe.log
failure class: none
next action: promote real checkpoints and run official 128-step clean eval from the clean clone
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

Local evidence:

```text
/Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128/clean_clone_e42dc6a/
```

Windows runtime:

```text
C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90
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

Earlier scratch archive evidence exists at commit `5b003477`, but the current
reproducibility reference is the official clean clone at commit `e42dc6a`.

## Current Blocker

The locally testable Stage 2 clean-eval adapter loop, metrics CSV writer,
summary validator, Windows runner invocation path, shared-core CARLA adapter
factory, simulator-state privileged observation builder, Windows privileged
obs probe, replay collection, replay checkpoint training, and checkpoint
promotion path are implemented. The first official 128-step clean run has been
classified as driving/stuck, not PASS.

The immediate driving issue was action collapse to throttle-zero action id 8.
BC warm-start has been added and Windows retraining completed, but the
post-retrain action probe and clean rerun are pending because Windows SSH became
temporarily unreachable through Tailscale.

The v2 readiness scripts now live under `RCS-AC-v2/scripts/windows`, and the
Stage 2 runner has a dependency probe. The remaining reproducibility dependency
is sibling `shared_core`. The adapter now owns the minimal shared_core env
builder and the v1.1 env config is copied under `RCS-AC-v2`. Official
performance evidence should continue to come from the documented clean clone.

The existing Windows checkout was not used for this probe because it was on an
unrelated dirty branch and `git fetch` hit interactive GitHub credential
failure. The archive lane avoided touching that worktree and is acceptable for
infrastructure validation, but not for final clean-performance claims.

An official clean clone is now available at:

```text
C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-ac11c90
```

Windows HTTPS GitHub access still prompts for credentials, but SSH
`git@github.com:thddydgnl/RCS-AC-first.git` works. The clean clone needed
`core.longpaths=true` to finish checkout and is now at commit `e42dc6a`. The
remaining external dependency for this lane is sibling `shared_core`.

If checkpoint/config/env wiring changes, rerun the non-clean shape probe:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_privileged_obs_probe.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml `
  -StepNeutral
```

The current clean clone already produces `privileged_obs_probe.json` with
`clean_gate_evidence=false`, strict privileged BEV/ego shapes, finite tensors,
raw obs/info keys, simulator-state provenance, schema/hash evidence, and
neutral-step info. It must not be interpreted as clean PASS.

Local adapter-loop tests and privileged observation probes are not clean
evidence. This file can move to PASS only after Windows runs the clean eval
command after `git pull --ff-only`, the CARLA readiness guard passes, and the
resulting `summary.json` validates with `source=windows_carla`,
`done_reason=horizon_reached`, and zero collision/offroad/stuck.
