# RCS-AC v1.1 2048 Goal Round 0001

Date: 2026-05-05
Branch: `codex/rcs-ac-v1-1-2048-stabilization`
Commit: `2e8fa5c`

## Goal

Start the RCS-AC v1.1 clean 2048 stabilization goal from a new branch and prepare the Windows CARLA 64/128 smoke path.

Required promotion order:

```text
Mac py_compile/pytest
Windows git pull
CARLA health
tiny CARLA BEV dataset
train_bev_bc tiny smoke
64 smoke
128 smoke
256 clean
512 clean
2048 clean
```

## Completed

Mac-side implementation and verification completed.

Implemented/pushed:

```text
RCS-AC v1.1 package
CARLA observation/route/LiDAR/map-actor adapter scaffold
collect_bev_expert_dataset.py
evaluate_bev_pure_pursuit.py --mode carla
full_bev_smoke.yaml
64/128 smoke plan
handoff/current/full implementation docs
adapter contract tests
```

Adapter compatibility fixes added after explorer review:

```text
obs["route"] 10-d vector fallback to target proxy
heading_error / route_heading_error treated as degree
*_rad keys treated as radian
legacy obs["lidar"] [C,H,W] BEV detected instead of treated as raw point cloud
previous control fallback from top-level steer/throttle/brake/smoothed/applied fields
CARLA eval action format restricted to STAGE4_ACTION_V2 [steer, longitudinal]
expert step result normalization in v1.1 dataset collection
```

Validation:

```text
python -m py_compile $(find rcs_ac_v1_1 -name '*.py' -not -path '*/__pycache__/*')
python -m pytest rcs_ac_v1_1/tests -q
47 passed
```

Git:

```text
git push -u origin codex/rcs-ac-v1-1-2048-stabilization
success
```

## Blocker

Windows remote execution did not start because Mac -> Windows SSH is unavailable.

Observed:

```text
Tailscale userspace SOCKS listener on localhost:1055 was not stable.
tailscale --socket=$HOME/.local/run/tailscaled-codex.sock status failed with:
  failed to connect to local Tailscale service; is Tailscale running?

SSH through ProxyCommand failed with:
  Connection closed by UNKNOWN port 65535

Direct SSH failed with:
  ssh: connect to host 100.87.219.58 port 22: Operation timed out

nc direct probe failed with:
  connectx to 100.87.219.58 port 22 failed: Operation timed out
```

Classification:

```text
remote infrastructure failure
not CARLA crash
not adapter failure
not model failure
not controller failure
not driving failure
```

## Next Command Sequence After Windows SSH Recovers

On Windows:

```powershell
Set-Location 'C:\Users\thddy\Documents\code\RCS-AC-first'
git fetch origin
git switch codex/rcs-ac-v1-1-2048-stabilization
git pull --ff-only
.\scripts\windows\ensure_carla_ready.ps1
```

Then tiny dataset:

```powershell
py -3.12 -B -m rcs_ac_v1_1.scripts.collect_bev_expert_dataset `
  --config configs/v1_1/full_bev_smoke.yaml `
  --env-config configs/<existing-clean-carla-config>.yaml `
  --output data/v1_1/bev_expert_tiny_carla_smoke.npz `
  --num-steps 128 `
  --host 127.0.0.1 `
  --port 2000
```

Then train smoke:

```powershell
py -3.12 -B -m rcs_ac_v1_1.scripts.train_bev_bc `
  --config configs/v1_1/full_bev_smoke.yaml `
  --dataset data/v1_1/bev_expert_tiny_carla_smoke.npz `
  --output-dir checkpoints/v1_1_bev_bc_tiny_carla_smoke `
  --epochs 1 `
  --batch-size 8 `
  --device cuda
```

Then 64 smoke:

```powershell
py -3.12 -B -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit `
  --config configs/v1_1/full_bev_smoke.yaml `
  --env-config configs/<existing-clean-carla-config>.yaml `
  --checkpoint checkpoints/v1_1_bev_bc_tiny_carla_smoke/last.pt `
  --run-name v1_1_carla_eval_64_smoke `
  --num-steps 64 `
  --device cuda `
  --mode carla
```

Do not promote to 128/256/512/2048 until the previous step has logs and `summary.json`.

Convenience runner added after the initial remote blocker:

```powershell
.\scripts\windows\run_v1_1_smoke.ps1 `
  -EnvConfig configs\<existing-clean-carla-config>.yaml `
  -Stage all
```

Individual stages:

```powershell
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage health
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage collect
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage train
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage eval64
.\scripts\windows\run_v1_1_smoke.ps1 -EnvConfig configs\<existing-clean-carla-config>.yaml -Stage eval128
```

## Fairness Checklist

```text
[x] No runtime steering override added
[x] No lane_offset/heading_error direct steering rule added
[x] No CARLA privileged signal added to policy inference input
[x] GT BEV/actor boxes remain target/debug only
[x] Controller output remains separate from raw policy output in logs
[x] Stage4 action wrapper preserved for shared_core env
[ ] Windows CARLA 64 smoke executed
[ ] Windows CARLA 128 smoke executed
[ ] 256/512/2048 clean eval executed
```
