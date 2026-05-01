# Round 17 - Drift-Window Visual-Steer BC Gate

Date: 2026-05-01

Status: planned

## Baseline

Round 16 is the current blocked run:

- BC val loss: `0.0204317`
- Eval max route: `0.205128`
- Eval mean route: `0.091096`
- Offroad: `3/512`
- Early offroad: route `0.153846`
- Mean visual-zero action L1: `0.006131`
- Mean visual-zero steer delta: `0.001017`
- Mean visual-zero longitudinal delta: `0.005114`
- Mean action jitter L1: `0.069248`

The failure window is route `0.15-0.23`, left drift lane offset about `-2.2` to `-2.75 m`, heading error about `5.3` to `16.3 deg`.

## Subagent Consensus

- The problem is still lateral drift plus weak steering-relevant RGB causality.
- Round 16's generic visual-zero margin mostly increased longitudinal dependence, not steering dependence.
- PPO remains blocked.
- The smallest useful next step is targeted drift-window data, not another loss-only tweak.
- Direct line-follow steering control is not allowed; RGB lane/road information must remain inside the learned policy path.

## Intervention

Round 17 combines:

1. Collect `128-256` masked drift-window transitions around route `0.15-0.23`.
2. Mix drift-window data with Round 15 clean/recovery data.
3. Train conservative frame BC using:
   - `perception_aux_head`
   - `rgb_perception_actor_features`
   - `actor_input_adapter`
4. Add a small BC-only steering-specific visual-zero margin.
5. Add eval logging for corrective-window visual steering metrics.

Config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_drift_window_round17.yaml
```

No direct action override, no lateral recovery in final eval, and no PPO before the BC/eval gate passes.

## Planned Remote Steps

1. Windows pull latest code.
2. Run `ensure_carla_ready.ps1`.
3. Collect drift-window data with `collect_recovery_dataset.py`:
   - config: `configs/phase3_bc_clean_recovery_imitation_v2_lateral_target_smoke.yaml`
   - checkpoint: `checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt`
   - route window: `0.15-0.23`
   - lane offset threshold: `0.45`
   - heading threshold: `5.0`
4. Mix:
   - Round 15 clean early: `384`
   - Round 17 drift-window: `192` if available
   - Round 15 recovery: `128`
5. Train Round 17 BC.
6. Run 512-step no-lateral eval.
7. Pull logs and summaries.
8. Judge gate before any PPO.

## Gate

PPO 2048 may run only if all pass:

- early offroad before route `0.20`: `0`
- total offroad: `<= 2/512`
- max route: `>= 0.230769`, preferred `>= 0.256410`
- collision/stuck: `0`
- mean action jitter L1: `<= 0.07`
- rollout mean visual-zero action L1: `>= 0.012`
- rollout mean visual-zero steer delta: `>= 0.003`
- corrective-window mean visual-zero steer delta: `>= 0.004`
- corrective sign rate: `>= 0.65`

If BC improves but rollout steering causality remains weak, stop before PPO.
