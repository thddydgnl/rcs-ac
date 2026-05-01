# Round 16 - Visual-Zero Margin BC Gate

Date: 2026-05-01

Status: in progress

## Baseline

Round 15 recovered the Round 14 regression but did not beat the Round 13/15 plateau.

- Dataset: `rgb_perception_actor_features_v2_masked_mix_512_round15.npz`
- BC val: `bc_loss=0.0212305`, `aux_road_mask_iou=0.606665`, `aux_lane_edge_mask_iou=0.0`
- Eval: max route `0.179487`, mean route `0.083033`, offroad `3/512`
- Visual-zero action effect: action L1 `0.004148`, steer delta `0.000719`, longitudinal delta `0.003429`
- Failure shape: all offroad rows occur near route `0.1795` with lane offset about `-2.6 m` and heading error about `11.6 deg`

## Subagent Verdict

- Log analysis: the current failure is lane drift with weak action-level RGB causality, not a generic BC loss or jitter failure.
- Code audit: the RGB perception actor path exists, but plain BC does not directly require the actor output to depend on the compact visual features.
- Mechanism verifier: internal `visual_actor_delta_l1_mean` is not sufficient. PPO is blocked unless visual-zero action and steer deltas rise in rollout.
- Safety verifier: Round 16 must beat max route `0.179487`, reduce early offroad, and keep jitter bounded before PPO.
- Data/BC audit: keep the Round 15 mixed masked dataset and conservative trainable scope. Do not unfreeze the GRU/actor head yet.
- Hardcoding audit: any improvement must disclose simulator semantic mask supervision, route/auxiliary channels, target-speed/action-smoothing wrappers, and controller-assisted expert data lineage.

## Intervention

Round 16 adds a BC-only optional visual-zero margin:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_visual_margin_round16.yaml`
- Dataset: reuse Round 15 mixed masked `512` rows
- Init checkpoint: use the original v2 base checkpoint, not the Round 15 adapter checkpoint, so adapter initialization remains controlled by config
- Epochs: `5`
- Trainable modules remain:
  - `perception_aux_head`
  - `rgb_perception_actor_features`
  - `actor_input_adapter`
- New BC keys:
  - `visual_zero_margin: 0.02`
  - `visual_zero_margin_loss_weight: 0.05`

The loss compares clean action output with a second forward pass where `zero_rgb_perception_actor_features=True`. It logs:

- `visual_zero_action_l1`
- `visual_zero_steer_delta`
- `visual_zero_longitudinal_delta`
- `visual_zero_margin_loss`

This is an intervention on BC pretraining only. It does not change reward, CARLA control wrappers, lateral recovery, PPO, or evaluation logic.

## Local Validation

- `python -m compileall -q rcs_ac_first tests`: PASS
- `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests -q`: PASS, `15 passed`
- `python -m rcs_ac_first.scripts.smoke_check --config configs/bc_frame_rgb_perception_actor_features_v2_visual_margin_round16.yaml --device cpu`: PASS

## BC/Eval Gate

Run PPO 2048 only if BC/eval passes all core gates:

- `mean_visual_zero_action_l1 >= 0.012`
- `mean_visual_zero_steer_delta >= 0.003`
- max route progress `>= 0.230769`
- early offroad before route `0.20` is `0`
- total offroad `<= 2`
- collision `0`, stuck `0`
- mean action jitter `<= 0.07`
- road mask IoU `>= 0.58`
- lane-edge loss finite and nonzero

If visual-zero action effect rises but route/safety does not improve, stop before PPO. If route improves but visual-zero rollout ablation does not regress, do not credit the RGB perception actor path.

## Required Remote Procedure

1. Push Mac changes.
2. Windows `git pull --ff-only`.
3. Run `scripts/windows/ensure_carla_ready.ps1` before every CARLA-dependent command.
4. Train BC with the Round 16 config and Round 15 mixed masked dataset.
5. Evaluate `epoch_005.pt` for `512` steps with lateral recovery disabled.
6. Pull `bc_training_log.csv`, `summary.json`, `bc_eval_log.csv`, and `env_step_log.csv` back to Mac.
7. Judge BC/eval gate before any PPO 2048.
