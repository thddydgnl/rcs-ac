# Round106 Route-Window Steer Delta Probe Result

## Purpose

Round105 showed that a speed-gate longitudinal label can slow the car in the route `0.25-0.31` failure window, but the 2048 clean eval still failed from lateral/heading drift. Round106 therefore tested a narrow steering-only probe:

- initialize from the Round105 checkpoint
- freeze longitudinal actor output row `[1]`
- train only steering actor output row `[0]`
- avoid global absolute steer imitation by setting `bc_loss_weight: 0.0`
- apply corrective-delta/visual-center losses only around route `0.25-0.31`

This is privileged offline supervision because route/lane/heading/target-corrective-delta fields are used for loss masks and labels. It is not a runtime steering override or lateral recovery assist.

## Config and Training

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_round106_route_window_steer_delta_probe.yaml`
- Dataset: `data/expert/rgb_perception_actor_features_v2_round105_speed_gate_mix_4776.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_speed_gate_longitudinal_bc_round105/last.pt`
- Output checkpoint: `checkpoints/rgb_perception_actor_features_v2_route_window_steer_delta_bc_round106/last.pt`
- Trainable parameters: `514`
- Trainable actor head rows: `[0]`
- Train loss: `0.008336`
- Val loss: `0.021599`

## Clean Eval

512-step clean eval:

- `collision=0`, `offroad=0`, `severe_early_offroad=0`, `stuck=0`
- Done reasons: `{"max_steps": 1}`
- Max route: `0.461538`
- Mean action jitter L1: `0.125466`
- Visual-center correction rate: `0.823529`
- Corrective sign rate: `0.823529`
- Initial mean steer: `-0.061532`
- Initial left steer rate: `0.6375`

2048-step clean eval:

- `collision=0`, `offroad=1`, `severe_early_offroad=0`, `stuck=0`
- Done reasons: `{"max_steps": 3, "offroad": 1}`
- Max route: `0.461538`
- Mean action jitter L1: `0.143488`
- Visual-center correction rate: `0.503067`
- Corrective sign rate: `0.723926`
- Initial mean steer: `-0.060729`
- Initial left steer rate: `0.6175`

The offroad episode ended at:

- route progress: `0.307692`
- lane offset: `-2.589771`
- heading error: `40.371372`
- final longitudinal action: `-0.217010`
- final speed: `7.848389 km/h`

## Comparison With Round105

Round106 improved some lateral indicators relative to Round105:

- 512 action jitter improved from `0.134265` to `0.125466`
- 2048 action jitter improved from `0.147721` to `0.143488`
- 2048 corrective sign rate improved from `0.620482` to `0.723926`
- offroad shifted from route `0.282051` to `0.307692`
- terminal heading error decreased from `52.321483` to `40.371372`

But Round106 did not preserve Round105's strong terminal slowdown:

- Round105 terminal longitudinal: `-0.988157`, speed `4.792938 km/h`
- Round106 terminal longitudinal: `-0.217010`, speed `7.848389 km/h`

## Decision

Round106 is BLOCKED for promotion. Steering-only learning helped but was not sufficient, and the remaining failure appears to require keeping both lateral correction and low target-speed behavior active in the same route window.

Next direction: train a combined narrow route-window probe with both actor rows `[0, 1]` open, but keep `bc_loss_weight=0.0` and use only explicit corrective steering delta plus speed-gate longitudinal target losses in route `0.25-0.31`. Do not broaden to global steering imitation or PPO until 512/2048 clean safety are both zero.
