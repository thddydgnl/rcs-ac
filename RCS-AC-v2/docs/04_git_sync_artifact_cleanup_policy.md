# RCS-AC v2 Git Sync, Artifact, And Cleanup Policy

작성일: 2026-05-24

목표:

```text
Windows CARLA 실행을 GitHub 기반으로 동기화하되,
불필요한 왕복과 대용량 artifact 유입을 줄이고,
goal 진행 중 .git 삭제 여부를 명확히 판단한다.
```

## 1. Source Of Truth

공식 v2 진행 상태의 source of truth는 GitHub branch다.

```text
Mac/Codex edits -> commit/push -> Windows git pull --ff-only -> CARLA run
-> text/table result pullback -> result.md update -> commit/push
```

GitHub에 남길 것:

```text
source code
tests
configs
Windows runner scripts
README/runbook/status docs
experiment result.md
small summary.json
small metrics.csv
exact command
commit hash
failure classification
```

GitHub에 남기지 않을 것:

```text
checkpoint weights
replay buffers
raw sensor frames
large videos
large per-step logs
private SSH keys/tokens
unredacted machine-specific secrets
```

## 2. Efficiency Changes

Avoid one commit per tiny thought. Use these lanes instead.

Lane A: official experiment lane

```text
Use for any run that will be documented as evidence.
Always run from committed and pushed code.
Every result.md must include the commit hash.
Windows must use git pull --ff-only before running.
```

Lane B: local implementation lane

```text
Use for Mac-only tests, unit tests, formatting, and quick refactors.
No Windows run is required.
Commit only when the work packet has a coherent checkpoint.
```

Lane C: scratch Windows lane

```text
Use only for infrastructure probing or throwaway CARLA connectivity checks.
Do not claim clean performance from scratch runs.
Do not write official result.md from scratch code.
If scratch run reveals a needed code change, commit/push it before the official rerun.
```

Recommended cadence:

```text
1. Implement a small work packet on Mac.
2. Run local tests.
3. Commit/push once for that work packet.
4. Windows pulls and runs one bounded smoke/eval.
5. Pull only text/table outputs.
6. Commit/push result note.
```

This keeps GitHub readable while still making every official Windows result
reproducible.

## 3. Branch Policy

Use a dedicated branch for the long v2 implementation:

```text
codex/rcs-ac-v2-raw2drive
```

Use stage/work-packet commits:

```text
stage0-contract-preflight
stage1-privileged-smoke
stage2-clean-eval
stage3-raw-alignment
stage4-raw-clean
stage5-fusion-ablation
```

WIP commits are acceptable on the working branch when they represent a
reproducible state. Before a public PR, summarize the story in docs/results
rather than relying on commit messages alone.

## 4. Artifact Policy

`RCS-AC-v2/.gitignore` blocks v2 heavy outputs inside the v2 subtree, so the
v2 branch remains self-contained even if the repository root `.gitignore` has
unrelated local changes:

```text
data/**/*.npz
data/**/*.pt
data/**/*.pth
data/**/*.ckpt
data/**/*.jsonl
experiments/**/*.npz
experiments/**/*.pt
experiments/**/*.pth
experiments/**/*.ckpt
experiments/**/*.jsonl
experiments/**/frames/
experiments/**/rgb/
experiments/**/videos/
experiments/**/replay_buffers/
remote_runs/**/frames/
remote_runs/**/checkpoints/
```

The root `.gitignore` may also contain broader project artifact rules, but do
not rely on unrelated root changes for the v2 official branch.

Small summaries remain trackable:

```text
result.md
summary.json
metrics.csv
config yaml
runner ps1/sh
small diagnostic json/csv
```

Before committing a Windows result, run:

```bash
git status --short -- RCS-AC-v2 .gitignore
find RCS-AC-v2 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.npz' -o -name '*.npy' -o -name '*.jsonl' \) -print
find RCS-AC-v2 -type f -size +50M -print
```

If the second or third command prints a file that should not be public, move it
to Windows local artifact storage or `_archive_local_not_git/`.

Preferred guard:

```bash
bash RCS-AC-v2/scripts/check_git_hygiene.sh
```

The guard is intentionally strict for the v2 subtree. It exits nonzero if
`RCS-AC-v2` contains nested `.git` metadata, generated checkpoint/replay files,
JSONL streams, or files larger than 50 MB. Repo-wide nested `.git` outside
`RCS-AC-v2` is printed as informational so unrelated work is visible but does
not block v2 commits.

## 5. `.git` Deletion Policy

Do not periodically delete the root `.git` directory.

```text
DO NOT DELETE:
/Users/song-yonghwi/code/RCS-AC-first/.git
C:\Users\thddy\Documents\code\RCS-AC-first\.git
```

Deleting the root `.git` breaks:

```text
branch tracking
commit history
remote origin
Windows git pull
Mac/Windows source-of-truth sync
ability to prove which commit produced a result
```

Allowed periodic maintenance:

```bash
git maintenance run --auto
git gc --auto
```

Disk-pressure maintenance, only when no git command is running:

```bash
git reflog expire --expire=30.days --all
git gc --prune=30.days
```

Do not use aggressive cleanup like `rm -rf .git` or scripted deletion of `.git`
during goal execution.

## 6. Nested `.git` Policy

Nested `.git` directories are different from the root `.git`.

They can appear when a tool, template, or external repo is copied into this repo.
For v2, nested `.git` directories should not be kept inside `RCS-AC-v2` unless
there is an intentional submodule or vendored repository decision.

Before deleting a nested `.git`, all checks must pass:

```text
1. It is not the repository root .git.
2. It is not declared in .gitmodules.
3. It is inside generated/vendor/scratch content, not user work.
4. Its inner `git status --short` has no valuable uncommitted work.
5. Removing it will not hide the source/version needed for attribution.
```

Inspection commands:

```bash
find . -name .git -type d -prune -print
find . -name .git -type f -print
git submodule status --recursive
```

Safe v2 direction:

```text
If a nested `.git` appears under RCS-AC-v2 by accident, remove only that nested
metadata after the checks above. Do not remove the root repository `.git`.
```

Current repo observation on 2026-05-24:

```text
Root .git exists and must be kept.
An unrelated nested .git exists at shared/tools/future-slide-skill/.git.
Do not touch unrelated nested .git during RCS-AC-v2 work unless a separate cleanup task asks for it.
```

## 7. Official Result Minimum Metadata

Every official v2 Windows result must include:

```text
commit:
branch:
Windows command:
config:
CARLA version:
CARLA town:
stage:
gate:
output path:
pulled files:
failure class:
next action:
```

Without commit and command, the result is scratch evidence only.
