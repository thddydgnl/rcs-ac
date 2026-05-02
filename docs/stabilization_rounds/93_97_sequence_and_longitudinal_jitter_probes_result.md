# Rounds 93-97 Sequence Smooth And Longitudinal Jitter Probes

## Goal

Continue from the current clean-safety baseline, Round90, without adding runtime steering overrides or lane-offset hardcoding.

Round90 remains safety-clean at 512 and 2048 steps, but it is conservative and PPO-gated by raw action jitter plus weak visual-center correction rate. These rounds tested whether the remaining blocker could be reduced through small BC-only output-head updates.

## Round93

Config: `configs/bc_sequence_rgb_perception_actor_features_v2_smooth_round93.yaml`

Change:

- Init from Round90.
- Switched BC to sequence mode.
- Trained actor output rows `[0, 1]`.
- Used `action_smooth_loss_weight: 0.03`.

Result:

- BC completed: `train_loss=0.139467`, `val_loss=0.089695`.
- Offline `early_any` actor sign audit: BLOCK.
  - all axes rate: `1.0`
  - min abs delta: `0.031333`
  - max abs zero steer: `0.033768`
  - sample pass rate: `0.9375`

Decision: blocked before CARLA. The zero-steer bias exceeded the strict `0.03` target.

## Round94

Config: `configs/bc_sequence_rgb_perception_actor_features_v2_soft_smooth_round94.yaml`

Change:

- Kept sequence BC but softened the smooth loss to `0.01`.
- Lowered LR to `0.000005`.
- Trained actor output rows `[0, 1]`.

Result:

- BC completed: `train_loss=0.136553`, `val_loss=0.087228`.
- Offline `early_any` audit: PASS.
  - all axes rate: `1.0`
  - min abs delta: `0.030845`
  - max abs zero steer: `0.029334`
- 512 clean eval: PASS safety, but not an upgrade.
  - collision/offroad/severe early offroad/stuck: `0/0/0/0`
  - max/final route: `0.358974`
  - mean action jitter L1: `0.182872`
  - steer jitter: `0.088306`
  - longitudinal jitter: `0.094566`
  - PPO gate: BLOCK

Decision: no 2048 promotion. The extra jitter came mostly from longitudinal/brake flip-flop, not steering.

## Round95

Config: `configs/bc_sequence_rgb_perception_actor_features_v2_steer_only_smooth_round95.yaml`

Change:

- Same sequence setup as Round94, but trained only actor output row `[0]`.
- Longitudinal row stayed frozen from Round90.

Result:

- BC completed: `train_loss=0.134339`, `val_loss=0.085495`.
- Offline `early_any` audit: PASS.
  - all axes rate: `1.0`
  - min abs delta: `0.030840`
  - max abs zero steer: `0.029318`
- 512 clean eval: PASS safety, but not an upgrade.
  - collision/offroad/severe early offroad/stuck: `0/0/0/0`
  - max/final route: `0.333333`
  - mean action jitter L1: `0.164354`
  - steer jitter: `0.090065`
  - longitudinal jitter: `0.074289`
  - PPO gate: BLOCK

Decision: no 2048 promotion. Steering-only smoothing did not improve route or jitter versus Round90.

## Round96

Config: `configs/bc_sequence_rgb_perception_actor_features_v2_longitudinal_smooth_round96.yaml`

Change:

- Returned to Round90 checkpoint.
- Trained only actor output row `[1]`.
- Used sequence BC with stronger smooth loss `0.08`, lower LR `0.000002`, and a reduced longitudinal ceiling loss.

Result:

- BC completed: `train_loss=0.138897`, `val_loss=0.086375`.
- Offline `early_any` audit: PASS.
  - all axes rate: `1.0`
  - min abs delta: `0.030364`
  - max abs zero steer: `0.024747`
- First CARLA eval attempt failed during `load_world()` with simulator timeout and was not counted as a model result.
- Retry 512 clean eval: PASS safety, but not an upgrade.
  - collision/offroad/severe early offroad/stuck: `0/0/0/0`
  - max/final route: `0.410256`
  - mean action jitter L1: `0.139629`
  - steer jitter: `0.087802`
  - longitudinal jitter: `0.051826`
  - PPO gate: BLOCK

Decision: no 2048 promotion. It recovered much of Round94's longitudinal regression, but still did not beat Round90.

## Round97

Config: `configs/bc_frame_rgb_perception_actor_features_v2_round90_longitudinal_reopen_round97.yaml`

Change:

- Returned to frame BC.
- Trained only actor output row `[1]`.
- Reused the milder Round82-style longitudinal target pressure on top of Round90.

Result:

- BC completed: `train_loss=0.273426`, `val_loss=0.271852`.
- Offline `early_any` audit: PASS.
  - all axes rate: `1.0`
  - min abs delta: `0.030364`
  - max abs zero steer: `0.024747`
- First CARLA eval attempt failed during `load_world()` with simulator timeout and was not counted as a model result.
- Retry 512 clean eval: PASS safety, but not an upgrade.
  - collision/offroad/severe early offroad/stuck: `0/0/0/0`
  - max/final route: `0.435897`
  - mean action jitter L1: `0.147085`
  - steer jitter: `0.081393`
  - longitudinal jitter: `0.065692`
  - PPO gate: BLOCK

Decision: no 2048 promotion. It matched Round90 route at 512 but increased overall jitter.

## Comparison To Round90

Round90 is still the clean-safety baseline:

- 512: route `0.435897`, action jitter `0.132107`, collision/offroad/severe early offroad/stuck `0/0/0/0`.
- 2048: route max `0.410256`, final `0.384615`, collision/offroad/severe early offroad/stuck `0/0/0/0`.

Rounds 94-97 all preserved 512 safety, but none improved both route progress and jitter:

| Round | 512 max route | action jitter | steer jitter | longitudinal jitter | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 90 | 0.435897 | 0.132107 | 0.085229 | 0.046877 | current baseline |
| 94 | 0.358974 | 0.182872 | 0.088306 | 0.094566 | blocked |
| 95 | 0.333333 | 0.164354 | 0.090065 | 0.074289 | blocked |
| 96 | 0.410256 | 0.139629 | 0.087802 | 0.051826 | blocked |
| 97 | 0.435897 | 0.147085 | 0.081393 | 0.065692 | blocked |

## Interpretation

The sequence smooth direction was not enough. It can preserve offline sign response, but when both output rows are trained it makes longitudinal control unstable. Freezing steering or training only longitudinal reduces the damage, but still does not beat Round90.

The current bottleneck is no longer the original early-left sign bug. It is the tradeoff between:

- sustained lateral recovery strong enough to avoid late offroad,
- longitudinal/brake behavior smooth enough for PPO promotion,
- route progress high enough to justify moving beyond clean BC eval.

## Next Direction

Keep Round90 as the promoted clean-safety baseline.

Next useful step is not another broad sequence-smooth run. The likely better path is to add an offline diagnostic for per-route-bin target speed/brake consistency and then try a narrowly masked longitudinal objective that reduces brake flip-flop only in the problematic route windows, while leaving the Round90 steering recovery unchanged. Any runtime target-speed wrapper or smoothing result must continue to be reported separately from raw policy output.
