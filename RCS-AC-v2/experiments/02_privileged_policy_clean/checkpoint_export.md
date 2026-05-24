# Stage 2 Checkpoint Export

Date: 2026-05-24
Status: `LOCAL_EXPORT_PATH_READY__REAL_CARLA_CHECKPOINTS_PENDING`

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

This creates a synthetic smoke pair that proves checkpoint payloads, sha256
manifest generation, and `evaluate_clean` policy loading. It is not clean
performance evidence.

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

Do not claim PASS from synthetic smoke checkpoints. PASS requires the Windows
CARLA clean gate summary and metrics to validate with `source=windows_carla`,
`steps=horizon`, `done_reason=horizon_reached`, zero safety violations,
`failure_class=none`, and pulled `summary.json` plus `metrics.csv`.

## Next Real Step

Replace the synthetic smoke checkpoint pair with checkpoints trained from the
Raw2Drive Stage 1/2 data path, then run the Windows 128-step clean gate after
commit/push and `git pull --ff-only`.
