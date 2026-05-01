# Round 27 - Collision Cluster Route-Offset BC Loss

Date: 2026-05-02
Branch: `codex-stabilization-rgb-v2`
Code commit: `1842aff Add collision cluster route offset BC loss`

## Goal

Round25 showed repeated collisions around the exact cluster:

- `0.16 <= route_progress <= 0.20`
- `lane_offset > +1.5`
- `heading_error < -8`

In that cluster, visual-center correction looked active, but route-offset corrective sign was `0.0`. This round adds a narrow BC loss that forces RGB contribution to create a negative steer delta relative to zero-RGB output for that exact cluster.

## Changes

- Added `collision_cluster_route_offset_agreement_loss` to `rcs_ac_first/scripts/train_bc.py`.
- Added train/val log metrics:
  - `collision_cluster_route_offset_agreement_loss`
  - `collision_cluster_route_offset_agreement_sign_rate`
  - `collision_cluster_route_offset_agreement_mask_rate`
  - `collision_cluster_route_offset_agreement_signed_delta_mean`
- Added focused tests for exact signed mask, sequence valid mask, and missing-field zero fallback.
- Added config:
  - `configs/bc_frame_rgb_perception_actor_features_v2_collision_cluster_route_offset_round27.yaml`

## Validation

Local validation passed:

```bash
python -m py_compile rcs_ac_first/scripts/train_bc.py tests/test_train_bc_perception.py
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_train_bc_perception.py tests/test_evaluate_bc_gate.py tests/test_expert_dataset_sequence.py -q
```

Result:

```text
23 passed in 1.20s
```

## Remote Execution Status

Code was pushed to GitHub, but Windows pull/training did not start because Mac-side remote access failed before reaching Windows:

```text
Connection closed by UNKNOWN port 65535
```

Tailscale userspace daemon was restarted with the documented persistent socket/state path, but it did not stay reachable:

```text
failed to connect to local Tailscale service; is Tailscale running?
```

Direct Tailscale SSH and documented LAN SSH both timed out. This is classified as remote-tunnel infrastructure, not a CARLA/model failure.

## Resume Update - 2026-05-02

Mac-side Tailscale was recovered by keeping `tailscaled` alive in a foreground session with the persistent socket/state paths. Windows SSH then worked and `git pull --ff-only` fast-forwarded to `4ff2ca7`.

Round27 exact-cluster BC training completed on Windows:

- Dataset: `data/expert/rgb_perception_actor_features_v2_corrective_delta_mix_928_round19.npz`
- Checkpoint init: `checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt`
- Run: `rgb_perception_actor_features_v2_collision_cluster_route_offset_bc_round27`
- Final val loss: `0.005251`

However, the exact-cluster loss did not activate:

- `collision_cluster_route_offset_agreement_mask_rate = 0.0`
- Dataset scan showed `lane_offset > 1.5` count was `0`
- Therefore the exact cluster had no trainable rows in the Round19 dataset

Windows dataset scan found no exact cluster rows in any available expert npz. A near-cluster fallback was used:

- New config: `configs/bc_frame_rgb_perception_actor_features_v2_near_cluster_route_offset_round27b.yaml`
- Mixed dataset: `data/expert/rgb_perception_actor_features_v2_near_cluster_mix_3744_round27b.npz`
- Inputs: Round19 corrective mix plus available recovery datasets
- Near mask: `0.10 <= route_progress <= 0.32`, `lane_offset > 0.75`, `heading_error < -3`
- Near-mask rows: `218 / 3744` (`0.058226`)

Round27b training result:

- Run: `rgb_perception_actor_features_v2_near_cluster_route_offset_bc_round27b`
- Final val loss: `0.010155`
- Near-cluster sign rate: `0.614973`
- Near-cluster signed delta mean: `-0.007567`

Round27c stronger loss result:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_near_cluster_route_offset_round27c.yaml`
- Loss weight: `0.60`
- Margin: `0.018`
- Run: `rgb_perception_actor_features_v2_near_cluster_route_offset_bc_round27c`
- Final val loss: `0.011121`
- Near-cluster sign rate: `0.636364`
- Near-cluster signed delta mean: `-0.013691`

CARLA was force-restarted successfully before the 512-step eval:

```text
CARLA_HEALTH=OK
CARLA_READY=True
```

The 512-step eval command for Round27c then lost SSH with exit `255` and produced no command output. Immediately afterward, Windows was unreachable:

```text
tailscale ping 100.87.219.58 timed out
nc: connection failed, SOCKS error 1
Connection closed by UNKNOWN port 65535
```

Tailscale status reported the Windows peer offline/last seen around the eval interruption. LAN SSH to `192.168.0.13` also timed out. This second interruption is classified as Windows-side offline/crash/sleep/network loss during eval, not a Mac SOCKS daemon failure.

## Current Interpretation

The original exact-cluster idea was valid but blocked by missing training data. Available datasets do not contain the `lane_offset > 1.5` exact collision state. Near-cluster training did make the average RGB steer correction negative, but sign consistency remained below the desired `0.80` threshold. The next useful step is to recover Windows, inspect whether the interrupted 512 eval wrote partial logs, and if not rerun the 512 eval from the Round27c checkpoint after a forced CARLA restart.

## 512 Retry Result - 2026-05-02

Windows came back online and Mac-side Tailscale/SSH was restored. The interrupted Round27c eval was confirmed partial:

- Run: `rgb_perception_actor_features_v2_near_cluster_route_offset_eval_512_round27c_20260502T010125`
- Last `bc_eval_log.csv` step: `78`
- No summary JSON

After another `ensure_carla_ready.ps1 -ForceRestart`, the 512-step retry completed:

- Run: `rgb_perception_actor_features_v2_near_cluster_route_offset_eval_512_retry_round27c_20260502T020912`
- Completed steps: `512`
- `pass_status`: `PASS`
- `ppo_gate_status`: `BLOCK`
- `collision_count`: `0`
- `done_reason_counts`: `{"offroad": 7}`
- `early_offroad_count`: `7`
- `max_route_progress`: `0.076923`
- `mean_route_progress`: `0.036058`
- `mean_action_jitter_l1`: `0.071855`
- `near_collision_cluster_row_count`: `0`

PPO gate fail reasons:

- `offroad_count 7 > 2`
- `early_offroad_count 7 > 0`
- `max_route_progress 0.076923 < 0.282051`
- `mean_route_progress 0.036058 < 0.132762`
- `mean_action_jitter_l1 0.071855 > 0.070000`
- `corrective_need_count 0 < 32`
- `mean_corrective_visual_zero_steer_delta 0.000000 < 0.004000`
- `visual_center_correction_rate 0.000000 < 0.700000`

Terminal offroad rows were consistent across episodes:

- Route progress at terminal: `0.076923`
- Lane offset: roughly `-2.93` to `-3.14`
- Heading error: roughly `+21` to `+27` degrees
- Collision: `False`

Interpretation: Round27c removed the observed collision failure in this 512 retry, but it regressed into repeated early offroad before the previous collision cluster. The near-cluster signed correction is too broad or biased for early route behavior. Do not run 2048 or PPO from Round27c.

Recommended next direction:

1. Do not keep broad near-cluster loss as-is.
2. Collect or synthesize exact-cluster recovery rows for `lane_offset > +1.5`, `heading_error < -8`, `route_progress 0.16-0.20`.
3. Add a symmetric guard so positive-lane-offset correction does not destroy negative-lane-offset / early-route behavior.
4. Add a 512 gate that explicitly blocks if terminal offroad rows cluster at `route_progress <= 0.08`.

## Resume Commands

After Mac-side Tailscale/SSH is healthy, run on Windows:

```powershell
Set-Location 'C:\Users\thddy\Documents\code\RCS-AC-first'
git pull --ff-only
```

Then train BC:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train_bc `
  --config configs/bc_frame_rgb_perception_actor_features_v2_collision_cluster_route_offset_round27.yaml `
  --dataset data/expert/rgb_perception_actor_features_v2_corrective_delta_mix_928_round19.npz `
  --init-checkpoint checkpoints/rgb_perception_actor_features_v2_route_offset_agreement_bc_round24/last.pt `
  --init-strict true `
  --run-name rgb_perception_actor_features_v2_collision_cluster_route_offset_bc_round27 `
  --device cuda
```

Before eval, force a clean CARLA state:

```powershell
.\scripts\windows\ensure_carla_ready.ps1 -ForceRestart
```

Then run the 512-step gate:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config configs/bc_frame_rgb_perception_actor_features_v2_collision_cluster_route_offset_round27.yaml `
  --checkpoint checkpoints/rgb_perception_actor_features_v2_collision_cluster_route_offset_bc_round27/last.pt `
  --run-name rgb_perception_actor_features_v2_collision_cluster_route_offset_eval_512_round27 `
  --num-steps 512 `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000
```

Only run 2048 if 512 passes with:

- `collision_count = 0`
- `offroad_count = 0`
- `early_offroad_count = 0`
- exact-cluster signed delta mean `< 0`
- exact-cluster sign rate `>= 0.80`
- no runtime assist
