# Round 18 - Bridge-Window Visual-Steer BC Gate

Date: 2026-05-01

Status: planned / running

## Baseline

Round 17 is the current best-but-blocked baseline:

- Max route: `0.282051`
- Mean route: `0.132762`
- Offroad: `2/512`
- Early offroad before route `0.20`: `0`
- Mean action jitter L1: `0.057810`
- Mean visual-zero action L1: `0.007283`
- Mean visual-zero steer delta: `0.001037`
- Corrective-window mean visual-zero steer delta: `0.000121`
- Corrective sign rate: `0.625`

Round 17 improved safety/progress, but PPO remains blocked because RGB perception actor features do not yet show enough steering-relevant control authority in rollout.

## Subagent Consensus

- The remaining failure is later left-drift/offroad, around route `0.256-0.282`.
- The old corrective eval window `0.12-0.23` undercounts the new failure frontier.
- Generic visual-zero action margins mostly increase longitudinal dependence.
- Round 18 should repair the `0.205-0.282` bridge window with targeted data and a drift-window-gated steering visual-zero loss.
- Direct line-following, lateral recovery, turn boost, speed gates, and post-policy route/lane action correction remain disallowed.

## Intervention

Round 18 adds:

1. BC batch access to optional `lane_offset` and `heading_error` fields from expert NPZs.
2. Optional visual-zero corrective mask:
   - route window: `0.205-0.282`
   - lane risk: `abs(lane_offset) >= 1.8`
   - heading risk: `abs(heading_error) >= 6.0`
3. Optional signed corrective steering loss inside that mask.
4. Configurable BC eval corrective window and explicit PPO gate status.
5. A new BC config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_bridge_window_round18.yaml
```

## Planned Remote Steps

1. Push Mac code and pull on Windows.
2. Force-restart CARLA before the first CARLA command.
3. Collect bridge-window data:
   - route window: `0.205-0.282`
   - lane offset threshold: `0.45`
   - heading threshold: `5.0`
   - target size: `224-256` transitions
4. Mix a masked frame dataset:
   - clean early: `384`
   - Round 17 drift: `192`
   - Round 18 bridge: `224-256`
   - recovery tail: `96-128`
5. Train Round 18 BC with conservative trainable scope:
   - `perception_aux_head`
   - `rgb_perception_actor_features`
   - `actor_input_adapter`
6. Run 512-step no-lateral eval.
7. Pull logs/summaries.
8. Decide whether PPO 2048 is allowed.

## Gate

PPO 2048 may run only if all pass:

- Early offroad before route `0.20`: `0`
- Total offroad: `<= 2/512`, preferred `<= 1/512`
- Max route: `>= 0.282051`, preferred `>= 0.307692`
- Mean route: `>= 0.132762`, preferred `>= 0.14`
- Collision/stuck: `0`
- Mean action jitter L1: `<= 0.07`, preferred `<= 0.06`
- Rollout mean visual-zero action L1: `>= 0.012`
- Rollout mean visual-zero steer delta: `>= 0.003`
- Corrective-window count: `>= 32`
- Corrective-window mean visual-zero steer delta: `>= 0.004`
- Corrective sign rate: `>= 0.70`

If safety/progress improves but steering causality remains weak, stop before PPO and continue BC/data iteration.
