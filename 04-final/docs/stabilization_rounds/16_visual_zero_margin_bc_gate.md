# Round 16 - Visual-Zero Margin BC Gate

Date: 2026-05-01

Status: BC/eval complete - PPO blocked

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

## Remote Result

Commit pushed to Windows: `d0fb1f0`.

Windows repo fast-forwarded successfully, and `ensure_carla_ready.ps1` passed before eval and after eval. The first eval attempt used the Anaconda Python environment and failed before CARLA startup because that environment could not import the CARLA Python API. The valid rollout was rerun with `py -3.12`, which has both `carla` and CUDA Torch available.

BC run:

- Run: `rgb_perception_actor_features_v2_visual_margin_bc_round16`
- Init: `checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt`
- Dataset: `data/expert/rgb_perception_actor_features_v2_masked_mix_512_round15.npz`
- Selected trainable params: `130418 / 954944`
- Epoch 5 val `bc_loss`: `0.0204317`
- Epoch 5 val `action_l1`: `0.124544`
- Epoch 5 val `perception_aux_loss`: `0.294706`
- Epoch 5 val `aux_road_mask_iou`: `0.623960`
- Epoch 5 val `aux_lane_edge_mask_iou`: `0.0`
- Epoch 5 val `visual_actor_delta_l1_mean`: `0.063234`
- Epoch 5 val `visual_zero_action_l1`: `0.018529`
- Epoch 5 val `visual_zero_steer_delta`: `0.000521`
- Epoch 5 val `visual_zero_longitudinal_delta`: `0.018009`

Rollout eval:

- Run: `rgb_perception_actor_features_v2_visual_margin_eval_512_round16_20260501T115331`
- Completed steps: `512`
- Script status: `PASS`
- Max route progress: `0.205128`
- Mean route progress: `0.091096`
- Final route progress: `0.025641`
- Done reasons: `{"offroad": 3}`
- Collision rate: `0.0`
- Stuck rate: `0.0`
- Mean action jitter L1: `0.069248`
- Mean steer jitter: `0.006854`
- Mean longitudinal jitter: `0.062394`
- Mean visual-zero action L1: `0.006131`
- Mean visual-zero steer delta: `0.001017`
- Mean visual-zero longitudinal delta: `0.005114`

Offroad rows:

- Step `140`, episode `0`: route `0.153846`, lane offset `-2.7470`, heading error `16.2786`
- Step `314`, episode `1`: route `0.205128`, lane offset `-2.2330`, heading error `5.3163`
- Step `508`, episode `2`: route `0.205128`, lane offset `-2.1864`, heading error `5.6770`

Lane/heading envelope:

- `abs(lane_offset)` mean `1.0276`, p95 `2.1170`, max `2.7470`
- `abs(heading_error)` mean `4.3621`, p95 `10.6676`, max `16.2786`

## Judgment

Round 16 is a partial improvement but does not pass the PPO gate.

Improved:

- BC val loss improved from Round 15 `0.0212305` to `0.0204317`.
- Road mask IoU improved from Round 15 `0.606665` to `0.623960`.
- Eval max route improved from Round 15 `0.179487` to `0.205128`.

Blocked:

- Total offroad remains `3/512`, above the gate of `<= 2`.
- There is still early offroad before route `0.20` at step `140`.
- Rollout visual-zero action L1 is only `0.006131`, below the PPO gate `0.012`.
- Rollout visual-zero steer delta is only `0.001017`, below the PPO gate `0.003`.
- The BC margin mostly increased longitudinal dependence, not steering dependence.

Decision: do not run PPO 2048 from this checkpoint. The next round should target steering-relevant visual causality or fill the route `0.15-0.23` lateral drift window before PPO.
