# Round105 Speed-Gate Longitudinal Probe Result

## Purpose

Round105 tested whether the repeated 2048 clean offroad around route progress `0.28` could be reduced by teaching the actor a lower target-speed longitudinal action in the risky recovery window, without changing runtime steering rules or enabling lateral recovery assist.

This round is still privileged offline supervision: route/lane/heading/speed signals were used to collect and label recovery rows, but the trained policy still receives the normal RCS-AC sensor/action inputs at inference.

## Code and Data

- Collector fixes:
  - fallback from `info["speed_kmh"]` to `info["speed"]`
  - collection-time speed-gate label computation from config when Windows `shared_core` does not emit `lateral_recovery_speed_gate_active`
- Collection config: `configs/bc_frame_rgb_perception_actor_features_v2_recovery_collect_round105_speed_gate_labels.yaml`
- BC config: `configs/bc_frame_rgb_perception_actor_features_v2_round105_speed_gate_longitudinal_probe.yaml`
- Collected dataset: `data/expert/rgb_perception_actor_features_v2_speed_gate_longitudinal_round105.npz`
- Oversampled mix: `data/expert/rgb_perception_actor_features_v2_round105_speed_gate_mix_4776.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Trained checkpoint: `checkpoints/rgb_perception_actor_features_v2_speed_gate_longitudinal_bc_round105/last.pt`

The actor trainable scope was restricted to `recurrent_actor_critic.actor_head.2` output row `[1]`, so steering row `[0]` was frozen.

## Collection Result

- Attempted transitions: `12000`
- Accepted transitions: `65`
- Acceptance rate: `0.00542`
- Route range: `0.2564` to `0.3077`
- Corrective delta sign rate: `1.0`
- Target longitudinal available rate: `1.0`
- Target longitudinal mean: `-0.63636`
- Target longitudinal delta abs mean: `0.30345`

The original `speed_gate_min_speed_kmh: 16.0` produced zero candidates in the current Round90 failure window, so the Round105 collection-only config lowered the label threshold to `8.0 km/h`.

## BC Result

- Dataset size: `4776` rows (`4256` base + `65 * 8` speed-gate rows)
- Trainable parameters: `514`
- Train loss: `0.371574`
- Val loss: `0.266810`

## Clean Eval

512-step clean eval:

- `collision=0`, `offroad=0`, `severe_early_offroad=0`, `stuck=0`
- Done reasons: `{"max_steps": 1}`
- Max route: `0.461538`
- Mean action jitter L1: `0.134265`
- Mean longitudinal jitter: `0.054579`
- Visual-center correction rate: `1.0`

2048-step clean eval:

- `collision=0`, `offroad=1`, `severe_early_offroad=0`, `stuck=0`
- Done reasons: `{"max_steps": 3, "offroad": 1}`
- Max route: `0.461538`
- Mean action jitter L1: `0.147721`
- Mean longitudinal jitter: `0.067221`
- Visual-center correction rate: `0.397590`

The offroad episode ended at:

- route progress: `0.282051`
- lane offset: `-2.762044`
- heading error: `52.321483`
- final longitudinal action: `-0.988157`
- final speed: `4.792938 km/h`

## Decision

Round105 is BLOCKED for promotion. It improved the longitudinal behavior in the failure window, but the 2048 failure still occurs because lateral/heading recovery remains insufficient. The terminal speed is already low, so further speed-only BC is unlikely to solve this specific failure.

Next direction: keep the speed-gate dataset as useful evidence, but the next probe should target lateral heading recovery in the route `0.25-0.31` window without reopening the general steering row broadly. A narrow route-window steering correction head/loss or a small high-confidence steering-only dataset should be tested before PPO is resumed.
