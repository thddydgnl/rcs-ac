# Round90 Route Contract Diagnosis

Date: 2026-05-04

## Question

The previous Round90 and Round90 shared-reward probe logs showed 2048-step clean
eval with offroad `0`, but the fresh revalidation produced late offroad events.
This diagnosis checks whether the revalidation differed from the stored
offroad-zero runs.

## Main Finding

The current Windows `shared_core` runtime no longer exposes the same route debug
fields in `info` / environment logs that were present in the stored offroad-zero
logs.

Stored PASS logs include route geometry:

- `env_step_log.csv`
- `route_representation`
- `target_x_local`, `target_y_local`
- `lookahead_turn_delta_rad`
- `route_waypoint_count`
- `route_total_length_m`
- `route_heading_mismatch`

Fresh revalidation logs include only `logs.csv` and omit those detailed route
geometry fields. As a result, `evaluate_bc.py` records route debug fields as
`0.0` in `bc_eval_log.csv`.

This is a logging/contract drift problem, not proof that policy route input is
zero.

## Direct Observation Probe

`tools/route_contract_probe.py` was run on Windows in:

`C:\Users\thddy\Documents\code\RCS-AC-first-round90-shared-reward-reverify`

Config:

`configs\bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`

Probe result at reset:

- `obs["route"]` shape: `[10]`
- `obs["route"]` nonzero count: `9`
- `obs["route"]` abs sum: `12.350019`
- route vector:
  `[2.999997, -0.000018, 2.999997, -0.000006, 5.999994, -0.000001, -0.000003, 0.000003, 0.0, 0.35]`
- route metadata from `route_manager`: `route_waypoint_count=40`, `route_total_length_m=119.727730`

So the actor still receives a nonzero route vector. The zero route fields in
fresh `bc_eval_log.csv` are caused by missing debug/info fields.

## Runtime Drift Evidence

Current Windows `shared_core` is imported from the parent workspace path:

`C:\Users\thddy\Documents\code\shared_core`

The current `CarlaEnv` writes `logs.csv`, while the stored PASS runs had
`env_step_log.csv` with the full route schema. The parent Git state reported the
shared-core files as untracked under the parent repo, so the runtime dependency is
not cleanly pinned by the RCS-AC worktree.

This makes old and new CARLA evals weaker as apples-to-apples comparisons unless
the `shared_core` revision/schema is frozen and recorded.

## Driving Difference

The current late-offroad failures are not explained by shared reward alone:

- Fresh shared-reward 2048 revalidation: `{"max_steps": 2, "offroad": 2}`
- Fresh original Round90 2048 revalidation: first retry `{"max_steps": 3, "offroad": 1}`

Both use the same checkpoint and no lateral recovery. Therefore the problem is
not only the reward shaping block.

Compared with stored PASS logs, fresh failing runs show:

- more negative mean lane offset over the full 2048 run
- lower RGB and route fusion weights in diagnostics
- later route offroad around route progress `0.30` to `0.41`, not severe early offroad
- route debug fields missing from the env log, preventing full turn/route
  geometry diagnosis from the fresh logs

## Conclusion

The immediate issue is not that the route tensor fed to the policy is all zero.
The route tensor is alive.

The real blocker is that the Windows CARLA/shared-core runtime used for current
revalidation is not the same observable contract as the stored offroad-zero runs,
and Round90 clean safety is not repeatably passing 2048 under the current runtime.

Before treating Round90 or shared-reward Round90 as a locked baseline:

1. Restore or pin the full `env_step_log.csv` route/debug schema.
2. Record the exact `shared_core` version or copy/hash in each eval manifest.
3. Re-run original Round90 and shared-reward Round90 under that pinned runtime.
4. Only call the baseline fixed if repeated 2048 runs pass with offroad `0`.

## Follow-Up: Schema Restored

The Windows `shared_core/env/carla_env.py` runtime was patched on 2026-05-04 to
restore the full route/debug env log contract:

- `logs.csv` -> `env_step_log.csv`
- `ENV_STEP_LOG_FIELDS` is used as the CSV schema
- route geometry is placed back into `info` and the env log:
  - `route_representation`
  - `target_x_local`, `target_y_local`
  - `lookahead_turn_delta_rad`
  - `route_waypoint_count`
  - `route_total_length_m`
  - `route_heading_mismatch`

Verification:

- Probe output: `outputs/route_contract_probe_round90_after_schema_restore`
- `obs["route"]` remains nonzero
- `env_step_log.csv` is created
- route geometry columns are populated

The helper script that applied the remote shared-core patch is stored in:

`tools/patch_shared_core_route_log.py`

The Windows guard script was also tightened:

`scripts/windows/check_shared_core_carla_env.ps1`

It now checks for `env_step_log.csv`, `ENV_STEP_LOG_FIELDS`,
`target_x_local`, `route_waypoint_count`, and `lookahead_turn_delta_rad`.

## Follow-Up: Original Round90 Revalidation After Schema Restore

Artifacts:

`round90_results/schema_restored_reverify_20260504/windows_outputs/`

512-step original Round90:

- run: `rgb_perception_actor_features_v2_round90_original_schema_restored_512_20260504T140729`
- done reasons: `{"max_steps": 1}`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- `env_step_log.csv`: present
- `route_target_x`: nonzero for all `512` rows
- `route_waypoint_count`: nonzero for all `512` rows
- mean route target heading: `-0.561988`

2048-step original Round90:

- run: `rgb_perception_actor_features_v2_round90_original_schema_restored_2048_20260504T140844`
- done reasons: `{"max_steps": 2, "offroad": 2}`
- collision/offroad/severe early offroad/stuck: `0/>0/0/0`
- `env_step_log.csv`: present
- `route_target_x`: nonzero for all `2048` rows
- `route_waypoint_count`: nonzero for all `2048` rows
- mean route target heading: `-0.549803`

Interpretation:

The logging/route-debug contract is now restored closely enough for old/new log
comparison. However, restoring the schema did not restore 2048 clean safety.
Original Round90 still produced late offroad under the current CARLA/shared-core
runtime. Therefore the remaining difference is not missing route observation or
missing route logging; it is long-horizon rollout variability or policy/runtime
behavior under the currently pinned environment.
