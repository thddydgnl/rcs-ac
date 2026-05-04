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

