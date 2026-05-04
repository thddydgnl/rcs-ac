# Round90 Runtime Drift Analysis

Date: 2026-05-04

## Purpose

After PPO stabilization, Round90 revalidation started producing late offroad
despite earlier offroad-zero 2048 logs. This analysis checks remaining causes
after restoring route/debug log schema.

## What Is Not The Main Difference

The following matched between the old PASS worktree and current reverify
worktree:

- Round90 checkpoint SHA256:
  `0CDBB66FC867FB55CEAF664B2C6A99F96A8A04DEB03476A159994C4CEA32149E`
- Round90 config SHA256:
  `E594AD879B20752DE1B71FE8FD687925E12681374973764D780D82394BA10016`
- `rcs_ac_first/scripts/train.py` SHA256:
  `1C56CFC70782C37CBB0E976F574887116CDD805BD56A1CD0D3C1740CBFC5FBC0`

So the current late offroad is not explained by a changed Round90 checkpoint,
changed Round90 config, or changed environment builder in RCS-AC `train.py`.

## Cross-Checks

### Old evaluator, current shared_core

Worktree:

`C:\Users\thddy\Documents\code\RCS-AC-first-shared-reward-eval`

Run:

`rgb_perception_actor_features_v2_round90_old_eval_script_current_sharedcore_2048_20260504T152246`

Result:

- done reasons: `{"max_steps": 3, "offroad": 1}`
- collision/stuck/severe early offroad: `0/0/0`
- max route: `0.461538`
- mean route: `0.198555`

### Current evaluator, current shared_core repeat

Worktree:

`C:\Users\thddy\Documents\code\RCS-AC-first-round90-shared-reward-reverify`

Run:

`rgb_perception_actor_features_v2_round90_current_eval_script_current_sharedcore_repeat_2048_20260504T152659`

Result:

- done reasons: `{"max_steps": 2, "offroad": 2}`
- collision/stuck/severe early offroad: `0/0/0`
- max route: `0.461538`
- mean route: `0.197428`

Interpretation:

Changing back to the old `evaluate_bc.py` did not restore the offroad-zero
result. The evaluator/logging script is therefore unlikely to be the primary
cause of the current offroad.

## Remaining Runtime Differences

### Full metadata is still not fully restored

The schema-restored current logs now have `314` columns and populated route
geometry, but many runtime metadata fields are still empty compared with the
stored PASS env logs:

- `episode_id`
- `weather`
- `traffic_mode`
- `fixed_delta_seconds`
- `speed_mps`
- `action_contract_version`
- `policy_action_space`
- `target_speed_kmh`
- `raw_requested_throttle`
- `raw_requested_brake`
- `smoothed_steer`
- `lateral_recovery_enabled`
- `action_smoothing_enabled`
- `drop_rgb`, `drop_lidar`
- `rgb_brightness`
- `lidar_valid_ratio`

This means the route/debug schema is restored, but the full shared-core logging
contract is still not exactly the old PASS contract.

### CARLA world lifecycle changed during PPO work

The PPO stabilization log records that shared-core was changed to reuse the
current Town04 world instead of always calling `client.load_world(self.town)`.
This likely reduced CARLA crashes/timeouts, but it is still a runtime lifecycle
difference from some earlier eval conditions.

## Driving Difference

The current runtime consistently fails in late left/turn-heavy route windows,
especially around route `0.30-0.34`.

Compared with stored PASS shared-reward 2048:

| Metric, route 0.30-0.34 | Stored PASS | Current runtime |
| --- | ---: | ---: |
| speed mean km/h | `6.411` | `9.576-9.874` |
| brake mean | `0.739` | `0.432-0.440` |
| fusion RGB mean | `0.0162` | `0.0042-0.0046` |
| fusion LiDAR mean | `0.0777` | `0.0185-0.0199` |
| fusion ego mean | `0.654` | `0.837-0.839` |
| fusion route mean | `0.252` | `0.139` |
| route visual-center agreement | `0.233` | `0.065-0.120` |

The current runtime is therefore faster in the same bottleneck window and the
fusion diagnostics are much more ego-dominant, with weaker RGB/route contribution.
This makes accumulated lateral drift more likely.

Terminal failures under current runtime:

- old evaluator/current shared_core: offroad at route `0.307692`, lane offset
  `-2.612152`, heading error `38.203517`, speed `8.906 km/h`
- current evaluator/current shared_core repeat: offroad at route `0.307692`,
  lane offset `-2.981325`, heading error `37.144702`, speed `11.948 km/h`
- current evaluator/current shared_core repeat: offroad at route `0.410256`,
  lane offset `-2.357132`, heading error `11.282885`, speed `12.349 km/h`

## Current Best Explanation

PPO stabilization most likely did not alter the RCS-AC Round90 checkpoint or
config. The strongest remaining explanation is shared runtime drift:

1. common Windows `shared_core` changed during PPO stabilization,
2. current logs prove the same checkpoint now runs faster and more ego-dominant
   in the left branch window,
3. Round90 has a narrow clean-safety margin, so this runtime drift is enough to
   turn old offroad-zero 2048 runs into late offroad runs.

## Next Verification

To fully pin this down, restore the full old shared-core logging/action metadata
contract, then run a 2048 repeat matrix. If speed/fusion stats remain current-like
and offroad persists, the remaining issue is policy margin under the current
runtime. If speed/fusion stats return to PASS-like values and offroad disappears,
the culprit is shared-core action/runtime drift.

