# Stage 2 Checkpoint Export

Date: 2026-05-24
Status: `SYNTHETIC_EXPORT_REJECTED_FOR_CLEAN__REPLAY_TRAINING_PATH_READY__REAL_CARLA_CHECKPOINTS_PENDING`

## Purpose

Stage 2 Windows clean eval needs two checkpoint files:

```text
world_model: PrivilegedWorldModel checkpoint
policy: PrivilegedPolicy checkpoint
```

The files themselves are heavy artifacts and must not be committed. They should
live in a local or Windows artifact folder, while `summary.json`, `metrics.csv`,
commands, sha256 hashes, and result notes remain the reproducibility evidence.

## Local Smoke Export Command

This creates a synthetic smoke pair that proves checkpoint payloads and sha256
manifest generation. It is not clean performance evidence, and the official
clean config now rejects it unless the checkpoint provenance gate is explicitly
disabled for a local loader-only test.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m rcs_ac_v2.training.train_stage1_privileged_policy \
  --steps 30 \
  --seed 13 \
  --bev-height 16 \
  --bev-width 16 \
  --ego-dim 8 \
  --hidden-dim 64 \
  --output-dir RCS-AC-v2/experiments/02_privileged_policy_clean/checkpoints/local_synthetic_smoke \
  --export-checkpoints
```

Expected files:

```text
stage2_privileged_world_model.pt
stage2_privileged_policy.pt
stage2_privileged_checkpoint_manifest.json
```

The `.pt` files are ignored by git. The manifest contains:

```text
world_model.path
world_model.sha256
policy.path
policy.sha256
clean_eval_config_patch.checkpoints
clean_gate_evidence = false
```

Verification run:

```text
2026-05-24 local CLI export smoke passed with --steps 4 --seed 19
created stage2_privileged_world_model.pt, stage2_privileged_policy.pt, and stage2_privileged_checkpoint_manifest.json
temporary /tmp/rcs_ac_v2_stage2_export_smoke directory removed after validation
```

## Config Wiring

For an official Windows run, fill these fields in a Windows-local config or a
committed experiment config that points to Windows-local checkpoint paths:

```yaml
checkpoints:
  world_model:
    path: <WINDOWS_PATH_OR_REPO_RELATIVE_PATH>
    sha256: <SHA256>
    commit: <PRODUCING_COMMIT_OR_NULL>
  policy:
    path: <WINDOWS_PATH_OR_REPO_RELATIVE_PATH>
    sha256: <SHA256>
    commit: <PRODUCING_COMMIT_OR_NULL>
```

The checkpoint payloads must also carry:

```text
stage = stage2_privileged_policy_clean
source = windows_carla_privileged_replay
```

The base clean config rejects `local_synthetic_smoke_not_carla` checkpoints and
checkpoints with missing/unknown source.

## Replay-Based Real Checkpoint Path

After the privileged observation probe passes, collect Windows CARLA privileged
replay:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\collect_stage2_privileged_replay.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged.yaml `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged `
  -Episodes 4 `
  -MaxSteps 256
```

This writes `.npz` episodes with semantic43 BEV, ego8, action, reward, continue,
and simulator-state provenance. Replay `.npz` files are heavy training inputs
and should stay out of git.

Then train loader-compatible real checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\train_stage2_privileged_from_replay.ps1 `
  -ReplayPath RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay `
  -SequenceLength 16 `
  -BatchSize 8 `
  -WorldModelSteps 200 `
  -PolicySteps 200
```

Expected real-training files:

```text
stage2_privileged_world_model_replay.pt
stage2_privileged_policy_replay.pt
stage2_privileged_replay_checkpoint_manifest.json
```

Scratch verification:

```text
2026-05-24 Windows archive runtime at commit e1c9c86 collected 1 episode x 8 steps into replay/tiny_probe.
Replay manifest source was windows_carla_privileged_replay and clean_gate_evidence=false.
2-step replay training exported world_model/policy checkpoints plus sha256 manifest.
Strict load_stage2_policy_runner accepted the replay-source checkpoints.
This was path validation only, not clean-performance evidence.
```

Do not claim PASS from synthetic smoke checkpoints. PASS requires the Windows
CARLA clean gate summary and metrics to validate with `source=windows_carla`,
`steps=horizon`, `done_reason=horizon_reached`, zero safety violations,
`failure_class=none`, and pulled `summary.json` plus `metrics.csv`.

## Next Real Step

Collect enough Windows CARLA privileged replay, train replay-source checkpoints,
fill the clean config checkpoint paths/sha from
`stage2_privileged_replay_checkpoint_manifest.json`, then run the Windows
128-step clean gate after commit/push and a clean Windows sync lane.
