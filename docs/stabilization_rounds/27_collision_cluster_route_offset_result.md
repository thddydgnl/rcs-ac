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
