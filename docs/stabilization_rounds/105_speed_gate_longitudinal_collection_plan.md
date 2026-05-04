# Round105 Speed-Gate Longitudinal Collection Plan

## Motivation

Rounds 102-104 failed to eliminate the current clean 2048 offroad using actor-head losses alone. The important audit result is that the terminal pre-offroad state requires both stronger lateral recovery and lower target speed:

- Round103 terminal route: `0.282051`
- lane offset: `-2.804958`
- heading error: `42.213886`
- applied steer: `0.133588`
- policy longitudinal: `-0.230216`
- target speed: `14.467625 km/h`

Existing `collect_recovery_dataset.py` could store corrected steer from lateral recovery, but it always kept policy longitudinal as the longitudinal expert label. That made speed-gated recovery collection unable to teach the actor the lower target-speed action.

## Code Change

`rcs_ac_first/scripts/collect_recovery_dataset.py` now supports:

```text
--use-speed-gate-longitudinal-target
```

When this flag is enabled and `info["lateral_recovery_speed_gate_active"]` is true, the collected `expert_action[1]` is converted from the configured `lateral_recovery.speed_gate_target_speed_kmh` into the normalized target-speed longitudinal action.

Additional dataset fields:

- `target_corrected_longitudinal`
- `target_longitudinal_available`

Additional summary fields:

- `use_speed_gate_longitudinal_target`
- `target_longitudinal_available_rate`
- `target_corrected_longitudinal_mean`
- `target_corrected_longitudinal_delta_abs_mean`

This is privileged offline collection supervision. It is not a runtime override and cannot be reported as raw RCS-AC inference performance.

## Verification

- `python -m py_compile rcs_ac_first/scripts/collect_recovery_dataset.py`: PASS
- Unit-style helper probe:
  - min target speed `6 km/h`, max target speed `28 km/h`, speed-gate target `10 km/h`
  - normalized longitudinal target: `-0.636364`

## Round105 Collection Fixes

Two collection blockers were found before generating the Round105 dataset:

- `info["speed_kmh"]` was not always present in the CARLA env info, while `env_step_log.csv` records speed as `speed`. The collector now falls back from `speed_kmh` to `speed` so `--speed-kmh-min` does not silently reject every row.
- The Windows `shared_core` currently exposes speed-gate log columns, but the observed env info did not set `lateral_recovery_speed_gate_active`. The collector now computes speed-gate label eligibility from the collection config when generating offline longitudinal targets. This remains privileged offline supervision, not a runtime steering or speed override.

The original collection config had `speed_gate_min_speed_kmh: 16.0`, but Round90’s route-window failures and candidates often occur below that speed. Round105 therefore uses:

```text
configs/bc_frame_rgb_perception_actor_features_v2_recovery_collect_round105_speed_gate_labels.yaml
```

with `speed_gate_min_speed_kmh: 8.0` for offline label collection only.

## Next Run

Collect a small high-risk recovery dataset before another BC round:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.collect_recovery_dataset `
  --config configs/bc_frame_rgb_perception_actor_features_v2_recovery_collect_round105_speed_gate_labels.yaml `
  --checkpoint checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt `
  --output data/expert/rgb_perception_actor_features_v2_speed_gate_longitudinal_round105.npz `
  --num-transitions 256 `
  --max-steps 12000 `
  --run-name rgb_perception_actor_features_v2_speed_gate_longitudinal_collect_round105 `
  --route-progress-min 0.25 `
  --route-progress-max 0.35 `
  --lane-offset-threshold 1.50 `
  --heading-error-threshold 8.0 `
  --speed-kmh-min 8.0 `
  --lane-offset-sign left `
  --accept-lateral-branch-only `
  --require-speed-gate-active `
  --require-corrective-sign-aligned `
  --use-speed-gate-longitudinal-target `
  --device cuda
```

Acceptance criteria before BC:

- `target_corrective_delta_sign_rate >= 0.98`
- `target_longitudinal_available_rate >= 0.95`
- collected route range overlaps `0.25-0.35`
- collected rows include `abs(lane_offset) >= 1.5` and `abs(heading_error) >= 8`
