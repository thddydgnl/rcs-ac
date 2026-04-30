# RCS-AC-first Remote Setup

This repository contains the `RCS-AC-first` experiment code only.

Large artifacts are intentionally ignored:
- `checkpoints/`
- `logs/`
- `outputs/`
- `remote_runs/`
- `data/**/*.npz`

## Runtime Dependency

The CARLA PPO and dataset collection scripts import `shared_core` from the workspace parent directory.
On the Windows CARLA machine, keep this layout:

```text
C:\Users\thddy\Documents\code\
  RCS-AC-first\
  shared_core\
```

Mac can edit this repo independently, but Windows execution should still happen with `shared_core` beside it unless `shared_core` is later packaged or added as a submodule.

## Windows Remote Execution

Use the runner from the Codex setup folder:

```powershell
C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1
```

Example from Mac:

```bash
ssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode ppo -Config "configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml" -RunName "remote_test"'
```

If CARLA does not reliably start from the SSH runner, use the interactive scheduled-task fallback in:

```text
docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
```

This is the verified path for cases where the runner times out waiting for port `2000` or Python later fails at `client.load_world()`.

Shortcut after Windows has pulled the latest repo:

```bash
ssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\ensure_carla_ready.ps1"'
```

Use `start_carla_interactive.ps1` directly only when you intentionally want to kill and relaunch the CARLA process. `ensure_carla_ready.ps1` is safer between round steps because it restarts only when the process, port, or Python CARLA probe fails.

Current verified details:

```text
SSH login: song@100.87.219.58
Windows project path: C:\Users\thddy\Documents\code\RCS-AC-first
Git remote on Windows: git@github.com:thddydgnl/RCS-AC-first.git
```

If the Mac/Codex session uses userspace Tailscale instead of a system Tailscale route, add:

```bash
-o 'ProxyCommand=nc -x localhost:1055 %h %p'
```

## Stabilization Workflow

For the full Mac analysis/edit/report and Windows CARLA/Python execution workflow, see:

```text
docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
```
