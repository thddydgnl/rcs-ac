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
ssh thddy@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode ppo -Config "configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml" -RunName "remote_test"'
```

## Stabilization Workflow

For the full Mac analysis/edit/report and Windows CARLA/Python execution workflow, see:

```text
docs/stabilization_rounds/13_remote_mac_windows_stabilization_workflow.md
```
