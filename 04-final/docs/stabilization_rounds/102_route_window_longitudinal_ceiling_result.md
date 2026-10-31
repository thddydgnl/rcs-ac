# Round102 Route-Window Longitudinal Ceiling

## Motivation

Round90 remains the clean baseline reference, but the 2026-05-04 re-verification exposed a current-runtime failure mode around route progress `0.25-0.35`: the global target-speed wrapper can remove offroad when capped to `16-18 km/h`, but that collapses route progress and should not be promoted as the final RCS-AC baseline.

The route-bin diagnostic in `round90_results/action_log_contract_fix_20260504/target_speed_route_bins.json` points to a narrower fix: preserve the Round90 steering policy and train only the longitudinal actor row so this route window is less aggressive under the normal `max_target_speed_kmh: 28.0` contract.

## Change

Added a training-time `route_progress_longitudinal_ceiling_*` BC loss in `rcs_ac_first/scripts/train_bc.py`.

This is offline privileged supervision because it uses dataset `route_progress` as a mask during BC training. It is not a runtime steering rule, target-speed override, lateral recovery assist, or lane-offset based control rule.

Round102 config:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_round102_route_window_longitudinal_ceiling.yaml`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Trainable module: `recurrent_actor_critic.actor_head.2`
- Trainable action row: `[1]` only, preserving steering row `[0]`
- Added loss:
  - `route_progress_longitudinal_ceiling_loss_weight: 0.25`
  - `route_progress_longitudinal_ceiling_route_min: 0.25`
  - `route_progress_longitudinal_ceiling_route_max: 0.35`
  - `route_progress_longitudinal_ceiling: -0.68`

## Local Verification

- `python -m py_compile rcs_ac_first/scripts/train_bc.py`: PASS
- Config load for Round102: PASS
- Tensor-level loss probe: PASS, only rows in the route window were penalized.

## CARLA Results

Windows `git pull --ff-only` was blocked by Git Credential Manager:

```text
fatal: Unable to persist credentials with the 'wincredman' credential store.
fatal: could not read Username for 'https://github.com': No such file or directory
```

To keep the round moving, the pushed Round102 files were copied to the Windows worktree by `scp`. This is a workflow issue, not a model result.

BC train:

- Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`
- Train output: `checkpoints/rgb_perception_actor_features_v2_route_window_longitudinal_ceiling_bc_round102/last.pt`
- Trainable parameters: `514`
- Trainable action row: `[1]`
- Train loss: `0.322666`
- Val loss: `0.319902`
- Val route-window ceiling loss: `0.318035`
- Val route-window ceiling rate: `0.089344`
- Val route-window ceiling mask rate: `0.556338`

512 clean eval:

- Run: `rgb_perception_actor_features_v2_route_window_longitudinal_ceiling_eval_512_round102_20260504T190142`
- Completed steps: `512`
- Done reasons: `{"max_steps": 1}`
- Collision: `0`
- Offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- Mean route progress: `0.220954`
- Mean action jitter L1: `0.127200`
- Visual-center correction rate: `0.500000`
- PPO gate: `BLOCK`

2048 clean eval:

- Run: `rgb_perception_actor_features_v2_route_window_longitudinal_ceiling_eval_2048_round102_20260504T190259`
- Completed steps: `2048`
- Done reasons: `{"max_steps": 3, "offroad": 1}`
- Collision: `0`
- Offroad: `1`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- Mean route progress: `0.195187`
- Mean action jitter L1: `0.140018`
- Visual-center correction rate: `0.560847`
- PPO gate: `BLOCK`

2048 route-bin detail:

| route bin | rows | speed mean | target mean | policy longitudinal mean | done |
| --- | ---: | ---: | ---: | ---: | --- |
| `0.25-0.30` | 255 | `7.455` | `9.381` | `-0.693` | none |
| `0.30-0.35` | 193 | `7.884` | `10.280` | `-0.611` | `{"offroad": 1}` |
| `0.35-0.40` | 178 | `7.555` | `10.049` | `-0.632` | none |

Compared with the current Round90 max28 metadata recheck, Round102 reduced the `0.30-0.35` offroad count from `2` to `1`, but did not eliminate it. It also worsened action jitter, so it is not a clean baseline replacement.

## Verdict

BLOCK.

The direction is partially informative but not successful. A mild route-window longitudinal ceiling can reduce the failure pressure, but the one-epoch actor-row update left the policy too inconsistent: 512 clean safety passed, while 2048 still produced offroad and jitter exceeded the gate.

Do not promote Round102 over Round90.

## Next Step

Avoid a broader global target-speed cap. The next useful probe should target smoother longitudinal behavior rather than only a lower ceiling:

- keep steering row `[0]` frozen,
- keep `max_target_speed_kmh: 28.0`,
- add a route-window longitudinal smoothness or teacher-consistency term for `route_progress 0.25-0.40`,
- reduce the route-window ceiling weight or make it speed-conditioned,
- require 512 and 2048 safety PASS before any PPO resume.
