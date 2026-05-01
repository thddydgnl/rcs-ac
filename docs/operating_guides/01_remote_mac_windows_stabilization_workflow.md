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
mkdir -p "$HOME/.local/run" "$HOME/.local/share/tailscale-codex"
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

Important:

```text
Always reuse the same socket and statedir above.
Do not use `/tmp` for Tailscale state during stabilization rounds unless a fresh login is acceptable.
If a fresh or empty state directory is used, Tailscale treats it as a new machine/session and asks for a browser auth URL.
```

Before every stabilization round, run this Mac-side tunnel preflight:

```bash
TS_SOCKET="$HOME/.local/run/tailscaled-codex.sock"

pgrep -af 'tailscaled.*tailscale-codex' || true
lsof -nP -iTCP:1055 -sTCP:LISTEN || true
tailscale --socket="$TS_SOCKET" status
ssh -o 'ServerAliveInterval=30' \
  -o 'ServerAliveCountMax=6' \
  -o 'ControlMaster=auto' \
  -o 'ControlPersist=8h' \
  -o 'ControlPath=~/.ssh/rcs-win-%r@%h:%p' \
  -o 'ProxyCommand=nc -x localhost:1055 %h %p' \
  song@100.87.219.58 'hostname'
```

Use these keepalive wrappers for long remote rounds:

```bash
rcswinssh() {
  ssh \
    -o 'ServerAliveInterval=30' \
    -o 'ServerAliveCountMax=6' \
    -o 'ControlMaster=auto' \
    -o 'ControlPersist=8h' \
    -o 'ControlPath=~/.ssh/rcs-win-%r@%h:%p' \
    -o 'ProxyCommand=nc -x localhost:1055 %h %p' \
    "$@"
}

rcswinscp() {
  scp \
    -o 'ServerAliveInterval=30' \
    -o 'ServerAliveCountMax=6' \
    -o 'ControlMaster=auto' \
    -o 'ControlPersist=8h' \
    -o 'ControlPath=~/.ssh/rcs-win-%r@%h:%p' \
    -o 'ProxyCommand=nc -x localhost:1055 %h %p' \
    "$@"
}
```

The SSH keepalive options keep long commands from being closed silently and make broken links fail clearly. They do not replace the Tailscale daemon; `tailscaled` must remain running and authenticated.

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

If one stabilization round contains multiple related runs, prefer one round folder with run-specific subfolders instead of copying everything into the root. This avoids overwriting common filenames such as `summary.json`, `env_step_log.csv`, and `bc_eval_log.csv`.

Example:

```bash
ROUND="rgb_perception_actor_features_v2_masked_clean_round13"
DEST="$HOME/code/windows_results/rcs-ac-first/experiments/$ROUND"
mkdir -p "$DEST"/{dataset,collect_env,bc_pretrain,pretrain_eval,ppo_lowstd,lowstd_eval,remote_runs}
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

When running multiple CARLA scripts in one round, re-check the simulator between scripts with the guard script:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\ensure_carla_ready.ps1"'
```

The guard checks all three conditions before the next collect/train/eval command:

```text
1. `CarlaUE4-Win64-Shipping` process exists.
2. TCP port 2000 accepts connections.
3. Python can import `carla`, connect to the server, and read the current world/map.
```

If any check fails, the guard restarts CARLA through the interactive scheduled-task helper and waits for a stable healthy state. This matters after dataset collection, BC evaluation, and PPO training because CARLA may exit between scripts even when the previous Python command completed successfully.

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

Observed limitation:

```text
The runner can start CARLA over SSH with Start-Process, but GUI/UE4 startup may be unstable from a non-interactive SSH session.
If CARLA opens port 2000 and then dies, or if the runner times out waiting for port 2000, treat this as infrastructure FAIL and use the interactive scheduled-task fallback below.
```

### 7.1 CARLA Interactive Scheduled-Task Fallback

Use this when the runner reports:

```text
CARLA did not open port 2000 within N seconds
```

or when Python fails at `client.load_world()` / streaming connection after a runner-started CARLA.

Preferred command after pulling the latest repo on Windows:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\ensure_carla_ready.ps1"'
```

Use the lower-level restart helper only when you intentionally want to kill and relaunch CARLA:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\scripts\windows\start_carla_interactive.ps1"'
```

Equivalent manual fallback, if the script is not present yet:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "
Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
$taskName = \"RCSCarlaStart\"
$exe = \"C:\Users\thddy\Documents\CARLA_0.9.16\CarlaUE4.exe\"
$arguments = \"-carla-rpc-port=2000 -quality-level=Low -nosound -windowed -ResX=800 -ResY=600\"
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute $exe -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$principal = New-ScheduledTaskPrincipal -UserId \"song\song\" -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"'
```

Then wait for the port and confirm the real UE4 process is alive:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "
for ($i = 0; $i -lt 180; $i++) {
  $ok = (Test-NetConnection -ComputerName 127.0.0.1 -Port 2000 -WarningAction SilentlyContinue).TcpTestSucceeded
  if ($ok) { Write-Host PORT_OPEN_AT_SECOND=$i; break }
  Start-Sleep -Seconds 1
}
Get-Process CarlaUE4,CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,Responding
"'
```

After CARLA is up, run Python scripts directly from SSH instead of using the runner:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config configs\YOUR_CONFIG.yaml --checkpoint checkpoints\YOUR_RUN\last.pt --run-name YOUR_EVAL_512 --num-steps 512 --device auto
"'
```

Recommended config setting for remote CARLA eval/train on Windows:

```yaml
env:
  client_timeout: 120.0
```

`30.0` seconds can be too short after CARLA has just loaded or after a previous run has stressed streaming.

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

For RGB/line perception rounds, verify the dataset really contains usable semantic mask targets before training BC:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -B -c \"import numpy as np; d=np.load(r'\''data/expert/YOUR_DATASET.npz'\''); print(d.files); print(float(d['\''aux_mask_targets_available'\''].mean()), float(d['\''aux_road_mask_target'\''].mean()), float(d['\''aux_lane_edge_target'\''].mean()))\"
"'
```

Expected for a useful RGB/line pretrain dataset:

```text
aux_mask_targets_available mean > 0
aux_road_mask_target mean > 0
aux_lane_edge_target mean > 0
```

If `aux_mask_targets_available` is `0.0`, do not interpret RGB/line pretrain results; the perception loss was inactive.

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

For perception BC, inspect the last train/val rows before moving to eval/PPO:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "
Import-Csv C:\Users\thddy\Documents\code\RCS-AC-first\logs\YOUR_BC_RUN\bc_training_log.csv |
  Select-Object -Last 4 |
  Format-Table epoch,split,bc_loss,perception_aux_loss,aux_mask_available_rate,aux_road_mask_loss,aux_lane_edge_mask_loss,visual_actor_delta_l1_mean -AutoSize
"'
```

Required gate:

```text
aux_mask_available_rate > 0
perception_aux_loss > 0
visual_actor_delta_l1_mean > 0
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

If SSH fails with `Connection closed by UNKNOWN port 65535`, or if `nc -x localhost:1055` closes immediately:

```bash
TS_SOCKET="$HOME/.local/run/tailscaled-codex.sock"

lsof -nP -iTCP:1055 -sTCP:LISTEN || true
pgrep -af 'tailscaled.*tailscale-codex' || true
tailscale --socket="$TS_SOCKET" status
```

Interpretation:

```text
1. No listener on `localhost:1055` means the userspace SOCKS daemon is not running.
2. `Logged out` or `NeedsLogin` means the daemon is running but not authenticated.
3. A different or temporary state directory usually means a new Tailscale machine/session was created.
4. Windows SSH should not be debugged until the Mac-side SOCKS listener and Tailscale auth state are healthy.
```

Recovery:

```bash
mkdir -p "$HOME/.local/run" "$HOME/.local/share/tailscale-codex"

/opt/homebrew/opt/tailscale/bin/tailscaled \
  --tun=userspace-networking \
  --socks5-server=localhost:1055 \
  --socket="$HOME/.local/run/tailscaled-codex.sock" \
  --statedir="$HOME/.local/share/tailscale-codex"

tailscale --socket="$HOME/.local/run/tailscaled-codex.sock" status
```

If the status prints a login URL, open it and approve the same session. After approval:

```bash
tailscale --socket="$HOME/.local/run/tailscaled-codex.sock" status
rcswinssh song@100.87.219.58 'hostname'
```

Do not start a second daemon with a `/tmp` state path while another round is in progress. That is the failure mode that caused the previous round to require a new auth URL before Windows SSH could continue.

If Git pull fails on Windows:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "cd C:\Users\thddy\Documents\code\RCS-AC-first; git status --short --branch; git remote -v; git pull --ff-only"'
```

If Mac push over HTTPS asks for a username or fails in a non-interactive shell:

```bash
gh auth status
gh auth setup-git
git push
```

If Mac push over SSH fails on GitHub host verification:

```bash
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
ssh -T git@github.com
```

If SSH then says `Permission denied (publickey)`, use the authenticated HTTPS remote through `gh auth setup-git` unless a write-capable SSH key is intentionally configured on Mac.

If CARLA port does not open:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "Get-Process CarlaUE4 -ErrorAction SilentlyContinue; netstat -ano | findstr :2000"'
```

If CARLA port opens but `load_world()` or streaming fails:

```text
1. Stop CarlaUE4 and CarlaUE4-Win64-Shipping.
2. Restart CARLA through the interactive scheduled-task fallback in Section 7.1.
3. Retry only the failed Python command.
4. Do not classify the model as failed unless the Python script reaches rollout/eval and writes metrics.
```

If Git history was amended after Windows already pulled the branch:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -Command "Set-Location C:\Users\thddy\Documents\code\RCS-AC-first; git fetch origin YOUR_BRANCH; git reset --hard origin/YOUR_BRANCH"'
```

Use this only for code branches where Windows has no local source edits. It does not delete ignored logs, outputs, datasets, or checkpoints.

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
