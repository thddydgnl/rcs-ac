# Remote Mac/Windows Stabilization Workflow

Created: 2026-04-30
Updated: 2026-05-01

## 0. Purpose

This document defines the operating workflow for RCS-AC-first stabilization rounds when:

- MacBook runs Codex, analysis, code review, code edits, commits, and reports.
- Windows desktop runs CARLA, Python experiment execution, heavy sensor IO, checkpoints, and raw outputs.
- GitHub transfers code.
- SSH/SCP transfers commands and selected result files.

The goal is to keep Codex/sub-agent CPU load away from the Windows CARLA machine while still letting Codex inspect results and drive the next experiment.

## 1. Machine Roles

### MacBook

MacBook is the control and analysis machine.

MacBook responsibilities:

```text
1. Clone and edit the RCS-AC-first GitHub repo.
2. Run Codex and any sub-agent analysis.
3. Inspect the complete text/table logs for each experiment, plus selected images when needed.
4. Modify code/config/docs.
5. Commit and push code changes to GitHub.
6. Trigger Windows experiments over SSH.
7. Pull back the experiment's complete text/table logs over SCP.
8. Write stabilization reports and next-step recommendations.
```

MacBook should not:

```text
1. Run CARLA.
2. Pull full sensor image folders by default.
3. Pull checkpoints by default.
4. Pull `.npz` datasets or other large binary artifacts by default.
5. Run heavy raw sensor processing unless explicitly requested.
```

### Windows Desktop

Windows is the execution machine.

Windows responsibilities:

```text
1. Keep CARLA installed and runnable.
2. Keep Python 3.12 and the CARLA 0.9.16 Python API available.
3. Keep RCS-AC-first and shared_core side by side.
4. Pull latest code from GitHub before each remote experiment.
5. Run CARLA and RCS-AC-first scripts locally.
6. Store logs, checkpoints, outputs, and raw sensor artifacts locally.
7. Expose SSH access for MacBook.
```

Windows should not:

```text
1. Run Codex sub-agents during CARLA experiments.
2. Push logs, checkpoints, datasets, or sensor outputs to GitHub.
3. Send full raw output trees to Mac unless explicitly requested.
```

## 2. Repository Layout

GitHub repo:

```text
https://github.com/thddydgnl/RCS-AC-first
```

Windows layout:

```text
C:\Users\thddy\Documents\code\
  RCS-AC-first\
  shared_core\
```

Remote access identity:

```text
SSH login: song@100.87.219.58
Windows project path: C:\Users\thddy\Documents\code\RCS-AC-first
```

The SSH account name and the Windows profile path are intentionally different.

Mac recommended layout:

```text
~/code/
  RCS-AC-first/
  windows_results/
```

Runtime dependency:

```text
RCS-AC-first/rcs_ac_first/scripts/train.py imports shared_core from the parent workspace.
On Windows, shared_core must remain beside RCS-AC-first unless it is later packaged or added as a submodule.
```

Ignored heavy artifacts:

```text
checkpoints/
logs/
outputs/
remote_runs/
data/**/*.npz
```

## 3. One-Time Setup

### 3.1 Windows SSH

Run once on Windows in elevated PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\admin_enable_ssh.ps1
```

Add the Mac public key to:

```text
C:\Users\thddy\.ssh\authorized_keys
```

Test from Mac:

```bash
ssh song@100.87.219.58 'hostname'
```

Use LAN address when Mac and Windows are on the same local network:

```text
192.168.0.13
```

Use Tailscale address when not on the same LAN:

```text
100.87.219.58
```

If the Mac/Codex environment does not have a system Tailscale route, run userspace Tailscale and connect through its SOCKS proxy:

```bash
/opt/homebrew/opt/tailscale/bin/tailscaled \
  --tun=userspace-networking \
  --socks5-server=localhost:1055 \
  --socket=$HOME/.local/run/tailscaled-codex.sock \
  --statedir=$HOME/.local/share/tailscale-codex
```

In another shell:

```bash
rcswinssh() { ssh -o 'ProxyCommand=nc -x localhost:1055 %h %p' "$@"; }
rcswinscp() { scp -o 'ProxyCommand=nc -x localhost:1055 %h %p' "$@"; }
rcswinssh song@100.87.219.58 'hostname'
```

### 3.2 Mac Clone

On Mac:

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/thddydgnl/RCS-AC-first.git
mkdir -p ~/code/windows_results/rcs-ac-first
```

### 3.3 Windows Git Remote

On Windows:

```powershell
cd C:\Users\thddy\Documents\code\RCS-AC-first
git remote -v
git status --short --branch
```

Expected:

```text
origin  git@github.com:thddydgnl/RCS-AC-first.git
main...origin/main
```

Windows uses a read-only GitHub deploy key for `git pull --ff-only`. If the remote is still HTTPS, fix it before running remote experiments:

```powershell
cd C:\Users\thddy\Documents\code\RCS-AC-first
git remote set-url origin git@github.com:thddydgnl/RCS-AC-first.git
git pull --ff-only
```

## 4. Stabilization Round Lifecycle

Every stabilization round should follow this order:

```text
1. Mac gathers latest Windows logs.
2. Mac Codex analyzes failure mode.
3. Mac Codex writes Experiment Brief.
4. Mac Codex edits code/config/docs.
5. Mac commits and pushes changes.
6. Mac triggers Windows run over SSH.
7. Windows pulls latest code and runs CARLA/Python.
8. Mac pulls the experiment's complete text/table logs over SCP.
9. Mac Codex analyzes results.
10. Mac Codex writes a report.
11. Mac commits/pushes report and any follow-up code.
```

This preserves the existing stabilization rule:

```text
Sub-agents first, experiment design second, modify third, run fourth, document fifth.
```

The difference is location:

```text
Sub-agents/analysis/edit/report: Mac
CARLA/Python execution/raw artifacts: Windows
```

## 5. Pull Experiment Logs From Windows To Mac

Create local result folders on Mac:

```bash
mkdir -p ~/code/windows_results/rcs-ac-first/logs
mkdir -p ~/code/windows_results/rcs-ac-first/remote_runs
mkdir -p ~/code/windows_results/rcs-ac-first/metrics
mkdir -p ~/code/windows_results/rcs-ac-first/experiments
```

Principle:

```text
For each experiment, pull all text/table logs needed to reconstruct the run.
Do not pull large binary or raw sensor artifacts by default.
```

For a run named `RUN_NAME`, create an experiment bundle:

```bash
RUN_NAME="YOUR_RUN_NAME"
DEST="$HOME/code/windows_results/rcs-ac-first/experiments/$RUN_NAME"
mkdir -p "$DEST"/{logs,remote_runs,outputs,data_summaries}
```

Pull all training logs for that run:

```bash
rcswinscp -r "song@100.87.219.58:C:/Users/thddy/Documents/code/RCS-AC-first/logs/${RUN_NAME}*" \
  "$DEST/logs/"
```

Pull all remote runner transcripts for that run:

```bash
rcswinscp -r "song@100.87.219.58:C:/Users/thddy/Documents/code/RCS-AC-first/remote_runs/logs/${RUN_NAME}*" \
  "$DEST/remote_runs/"
```

Pull all `.csv` / `.json` outputs for that run without copying image folders:

```bash
rcswinssh song@100.87.219.58 "powershell -NoProfile -Command \"Get-ChildItem C:\Users\thddy\Documents\code\RCS-AC-first\outputs -Recurse -Include *.csv,*.json | Where-Object { \\\$_.FullName -like '*${RUN_NAME}*' } | ForEach-Object { \\\$relative = \\\$_.FullName.Substring('C:\Users\thddy\Documents\code\RCS-AC-first\'.Length); \\\$target = Join-Path 'C:\Users\thddy\Documents\code\RCS-AC-first\remote_runs\export_text_logs\${RUN_NAME}' \\\$relative; New-Item -ItemType Directory -Force -Path (Split-Path \\\$target) | Out-Null; Copy-Item \\\$_.FullName \\\$target -Force }\""
rcswinscp -r "song@100.87.219.58:C:/Users/thddy/Documents/code/RCS-AC-first/remote_runs/export_text_logs/${RUN_NAME}" \
  "$DEST/outputs/"
```

Pull dataset summary JSON files for that run, if any:

```bash
rcswinscp "song@100.87.219.58:C:/Users/thddy/Documents/code/RCS-AC-first/data/expert/*${RUN_NAME}*.summary.json" \
  "$DEST/data_summaries/"
```

Pull these by default for each experiment:

```text
logs/<RUN_NAME>*/
remote_runs/logs/<RUN_NAME>*
outputs/**/*<RUN_NAME>*.csv
outputs/**/*<RUN_NAME>*.json
data/expert/*<RUN_NAME>*.summary.json
selected sample images only when a visual diagnosis needs them
```

Do not pull these by default:

```text
outputs/**/rgb/
outputs/**/semantic/
outputs/**/lidar/
checkpoints/
data/**/*.npz
*.pt
raw frame dumps
```

## 6. Mac Edit And Push

Before editing:

```bash
cd ~/code/RCS-AC-first
git pull --ff-only
git status --short --branch
```

After code/config/doc changes:

```bash
git diff --stat
git diff
git add PATHS_YOU_CHANGED
git commit -m "Describe stabilization change"
git push
```

Mac Codex should report:

```text
Commit:
Changed files:
Hypothesis:
Expected metric movement:
Windows command to run:
```

## 7. Run PPO/CARLA On Windows From Mac

Default PPO run:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode ppo -Config "configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml" -RunName "remote_test"'
```

Short smoke run:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode ppo -Config "configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml" -RunName "smoke_remote" -TotalTimesteps 512'
```

2048 stabilization gate:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode ppo -Config "configs\YOUR_CONFIG.yaml" -RunName "YOUR_ROUND_2048" -TotalTimesteps 2048'
```

Windows runner behavior:

```text
1. cd C:\Users\thddy\Documents\code\RCS-AC-first
2. git pull --ff-only
3. install/check CARLA Python wheel when requested
4. start CARLA if port 2000 is not open
5. wait for CARLA RPC port
6. run py -3.12 rcs_ac_first\scripts\train.py
7. save transcript in remote_runs/logs
```

## 8. Run Dataset Collection On Windows From Mac

Collect expert transitions:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode collect -Config "configs\phase3_bc_clean_rgb_perception_actor_features_v1_smoke.yaml" -Dataset "data\expert\remote_collect_1024.npz" -TotalTimesteps 1024 -RunName "remote_collect_1024"'
```

Generated `.npz` files stay on Windows unless explicitly copied.

Copy only summary JSON by default:

```bash
rcswinscp 'song@100.87.219.58:C:/Users/thddy/Documents/code/RCS-AC-first/data/expert/remote_collect_1024.summary.json' \
  ~/code/windows_results/rcs-ac-first/metrics/
```

## 9. Run BC On Mac Or Windows

BC training can run without CARLA if the dataset exists locally.

Recommended default:

```text
Run BC on Windows when the dataset is large or already lives on Windows.
Run BC on Mac only for light analysis or when the dataset has intentionally been copied.
```

Windows BC from Mac:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\Codex\2026-04-30\new-chat\carla-remote-runner\run_rcs_ac_first.ps1" -Mode bc -Config "configs\bc_frame_rgb_perception_actor_features_v2_pretrain_smoke.yaml" -Dataset "data\expert\clean_probe_256.npz" -RunName "bc_probe_remote"'
```

Mac BC only after copying the selected dataset:

```bash
rcswinscp 'song@100.87.219.58:C:/Users/thddy/Documents/code/RCS-AC-first/data/expert/clean_probe_256.npz' \
  ~/code/windows_results/rcs-ac-first/data/
```

## 10. Codex Report Template

Every stabilization experiment report should include:

```text
Round name:
Date:
Machine split:
  Analysis/edit/report: Mac
  CARLA/Python/raw outputs: Windows

Input commit:
Windows command:
Config:
Run name:
Starting checkpoint:

Hypothesis:
Allowed changes:
Rejected changes:

Pulled artifacts:
  logs:
  remote runner transcript:
  csv/json:
  sample images:

Primary metrics:
  route_progress:
  done_reason:
  collision:
  offroad:
  stuck:
  average_reward:
  max_route_progress:

Mechanism metrics:
  activation:
  coverage:
  timing:
  false activation:
  precondition metrics:

Diagnosis:
PASS/WARN/FAIL:
Next action:
```

## 11. Minimal Codex Prompt For Mac

Use this prompt when starting a Mac Codex session:

```text
You are working on RCS-AC-first from Mac.

Follow docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md.
Use Mac for analysis, code edits, commits, and reports.
Use Windows only through SSH for CARLA/Python execution.

GitHub repo:
https://github.com/thddydgnl/RCS-AC-first

Windows SSH:
song@100.87.219.58

Windows project:
C:/Users/thddy/Documents/code/RCS-AC-first

Windows runner:
C:/Users/thddy/Documents/Codex/2026-04-30/new-chat/carla-remote-runner/run_rcs_ac_first.ps1

For each experiment, pull all text/table logs: logs, remote runner transcripts, csv/json outputs, and summary json files.
Do not copy checkpoints, npz datasets, or raw sensor/image folders unless explicitly requested.
Before each experiment, write an Experiment Brief.
After each experiment, write a PASS/WARN/FAIL report with metrics and next action.
```

## 12. Failure Handling

If SSH fails:

```bash
rcswinssh -v song@100.87.219.58 'hostname'
```

If Git pull fails on Windows:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "cd C:\Users\thddy\Documents\code\RCS-AC-first; git status --short --branch; git remote -v; git pull --ff-only"'
```

If CARLA port does not open:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "Get-Process CarlaUE4 -ErrorAction SilentlyContinue; netstat -ano | findstr :2000"'
```

If Windows reboots or powers off:

```text
1. Stop running Codex/sub-agents on Windows.
2. Keep Codex on Mac only.
3. Reduce CARLA quality or timestep count.
4. Pull Event Viewer Kernel-Power logs before rerunning.
5. Treat the run as infrastructure FAIL, not model FAIL.
```

## 13. Golden Rule

Use GitHub for code.
Use SSH for commands.
Use SCP for each experiment's complete text/table logs.
Keep raw CARLA sensor outputs, checkpoints, and datasets on Windows by default.
Keep Codex reasoning and reports on Mac.
