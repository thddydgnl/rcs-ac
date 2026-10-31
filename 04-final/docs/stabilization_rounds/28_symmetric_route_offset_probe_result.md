# Round 28 - Symmetric Route-Offset Probe

Date: 2026-05-02
Branch: `codex-stabilization-rgb-v2`
Code commit: `0b7c166 Add symmetric route offset round 28 probe`

## Goal

Round27c used a broad near-cluster route-offset loss and removed collisions in the 512 retry, but regressed into repeated early offroad around `route_progress=0.076923`, `lane_offset≈-3m`, `heading_error=+21~27deg`.

This round checked whether the Round27 loss had a code/design issue: it only trained positive lane offsets. Round28 changed the near-cluster loss to support symmetric `abs` lane/heading masks and added metrics/gates that make this failure easier to detect.

## Code Changes

- `collision_cluster_route_offset_agreement_loss` now supports:
  - `collision_cluster_lane_offset_mode: abs`
  - `collision_cluster_heading_error_mode: abs`
  - desired steer sign `-sign(lane_offset)`
- Added corrected and side-split BC metrics:
  - `collision_cluster_route_offset_agreement_corrected_delta_mean`
  - `collision_cluster_route_offset_agreement_positive_sign_rate`
  - `collision_cluster_route_offset_agreement_negative_sign_rate`
  - positive/negative mask rates
- Added eval summary gate fields:
  - `severe_early_offroad_count`
  - `severe_early_terminal_offroad_count`
  - `ppo_gate_early_offroad_route_max`
- Added config:
  - `configs/bc_frame_rgb_perception_actor_features_v2_symmetric_route_offset_round28.yaml`

## Local Validation

```bash
python -m py_compile rcs_ac_first/scripts/train_bc.py rcs_ac_first/scripts/evaluate_bc.py tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
```

Result:

```text
25 passed in 1.00s
```

## BC Training

Windows training completed:

- Dataset: `data/expert/rgb_perception_actor_features_v2_near_cluster_mix_3744_round27b.npz`
- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt`
- Run: `rgb_perception_actor_features_v2_symmetric_route_offset_bc_round28`
- Final val loss: `0.009898`

Final val correction metrics:

- Overall sign rate: `0.860795`
- Corrected delta mean: `+0.025233`
- Positive-lane sign rate: `0.199643`
- Negative-lane sign rate: `1.000000`
- Positive mask rate: `0.053476`
- Negative mask rate: `0.197861`

Interpretation: the symmetric loss successfully trained the negative-lane side, but the positive-lane side remained weak.

## 512 Eval

CARLA was force-restarted and readiness passed before eval.

Run:

- `rgb_perception_actor_features_v2_symmetric_route_offset_eval_512_round28_20260502T022549`

Summary:

- `completed_steps`: `512`
- `pass_status`: `PASS`
- `ppo_gate_status`: `BLOCK`
- `collision_count`: `0`
- `done_reason_counts`: `{"offroad": 7}`
- `early_offroad_count`: `7`
- `severe_early_offroad_count`: `7`
- `severe_early_terminal_offroad_count`: `7`
- `max_route_progress`: `0.076923`
- `mean_route_progress`: `0.035006`
- `mean_action_jitter_l1`: `0.079138`

Terminal offroad rows were again concentrated at:

- `route_progress`: `0.076923`
- `lane_offset`: roughly `-3.0m`
- `heading_error`: roughly `+23` to `+29` degrees
- `collision`: `False`

## Conclusion

Round28 confirmed that the positive-only loss was a real code/design issue and fixed its symmetry, but the actual early offroad failure persisted. The model reaches the same early failure route before the near-cluster training window (`0.10-0.32`) can help.

Do not run 2048 or PPO from Round28.

## Next Direction

1. Add or collect early-route recovery rows around `route_progress <= 0.08`, `lane_offset≈-3m`, `heading_error≈+25deg`.
2. Add a narrow early-route recovery loss or training window that covers `route_progress 0.025-0.08`.
3. Keep side-split metrics; do not use raw steer delta mean as a symmetric gate.
4. Rebalance positive-lane rows because Round28 val positive sign rate was only `0.199643`.
