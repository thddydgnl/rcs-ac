# Rounds 99-100 Longitudinal Floor Probes

## Goal

Continue from Round90, which remains the clean-safety baseline, and test whether the raw longitudinal/brake jitter blocker can be reduced without runtime hardcoding or steering overrides.

The preceding Rounds 93-98 showed that broad sequence smoothing and longitudinal row reopening preserve safety but do not beat Round90. These rounds used the existing `route_progress_longitudinal_floor_*` training-time loss to target excessive low-speed braking in narrow route windows.

## Round99

Config: `configs/bc_frame_rgb_perception_actor_features_v2_longitudinal_floor_round99.yaml`

Change:

- Init from Round90.
- Frame BC.
- Trained only actor output row `[1]`; steering row stayed frozen.
- Disabled the early-severe longitudinal ceiling loss.
- Added route/speed masked longitudinal floor:
  - route window: `0.08-0.28`
  - speed max: `3.0 m/s`
  - floor: `-0.65`
  - weight: `0.8`

Result:

- BC completed: `train_loss=0.090364`, `val_loss=0.089495`.
- Offline `early_any` audit: PASS.
  - all axes rate: `1.0`
  - min abs delta: `0.030364`
  - max abs zero steer: `0.024747`
- 512 clean eval: PASS safety, but not an upgrade.
  - collision/offroad/severe early offroad/stuck: `0/0/0/0`
  - max/final route: `0.358974`
  - mean action jitter L1: `0.140056`
  - steer jitter: `0.086441`
  - longitudinal jitter: `0.053615`
  - PPO gate: BLOCK

Decision: no 2048 promotion. The floor reduced some longitudinal amplitude but hurt route progress.

## Round100

Config: `configs/bc_sequence_rgb_perception_actor_features_v2_conservative_longitudinal_floor_round100.yaml`

Change:

- Init from Round90.
- Sequence BC.
- Trained only actor output row `[1]`; steering row stayed frozen.
- Low LR `0.000002`, sequence length `16`, stride `4`.
- Used moderate action smoothness `0.04`.
- Added weaker route/speed masked longitudinal floor:
  - route window: `0.05-0.25`
  - speed max: `1.8 m/s`
  - floor: `-0.55`
  - weight: `0.25`
- Kept a reduced early-severe longitudinal ceiling loss.

Result:

- BC completed: `train_loss=0.192134`, `val_loss=0.116939`.
- Offline `early_any` audit: PASS.
  - all axes rate: `1.0`
  - min abs delta: `0.030364`
  - max abs zero steer: `0.024747`
- 512 clean eval: PASS safety, but not an upgrade.
  - collision/offroad/severe early offroad/stuck: `0/0/0/0`
  - max/final route: `0.333333`
  - mean action jitter L1: `0.152231`
  - steer jitter: `0.084435`
  - longitudinal jitter: `0.067797`
  - PPO gate: BLOCK

Decision: no 2048 promotion. The conservative sequence-floor combination also reduced route progress and increased jitter versus Round90.

## Comparison

| Round | 512 max route | action jitter | steer jitter | longitudinal jitter | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 90 | 0.435897 | 0.132107 | 0.085229 | 0.046877 | current baseline |
| 99 | 0.358974 | 0.140056 | 0.086441 | 0.053615 | blocked |
| 100 | 0.333333 | 0.152231 | 0.084435 | 0.067797 | blocked |

## Interpretation

The longitudinal floor loss is not a useful upgrade path in its current form. It can be applied without runtime hardcoding and without changing steering, but it still disturbs the learned speed/brake rhythm enough to reduce route progress.

Round90 remains the best clean baseline. The current remaining issue appears less like a simple per-row BC target problem and more like a mismatch between:

- raw longitudinal output scale,
- target-speed action wrapper dynamics,
- PPO gate jitter threshold,
- and the conservative route progress achieved by the clean-safe policy.

## Next Direction

Do not continue broad longitudinal row tuning from Round90.

The next useful step is to freeze Round90 as the clean-safe RCS-AC baseline and separate two tracks:

1. Clean baseline confirmation for fair robustness comparison.
2. PPO route-progress probe with strict safety regression checks, reported separately from the BC baseline.

If the project goal is robustness versus PPO/SAC under weather and sensor dropout, Round90 is strong enough to serve as the first RCS-AC clean-safe candidate, while later PPO improvements should be treated as optional route-progress refinement.
