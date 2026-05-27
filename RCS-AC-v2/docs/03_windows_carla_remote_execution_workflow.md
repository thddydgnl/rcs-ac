# RCS-AC v2 Windows CARLA Remote Execution Workflow

작성일: 2026-05-24

목표:

```text
RCS-AC v1.1에서 사용했던 Mac-Codex / Windows-CARLA 분리 실행 방식을
RCS-AC v2 Raw2Drive 구현에도 그대로 적용할 수 있게 정리한다.
```

이 문서는 다음 v1.1/공용 문서를 v2용으로 옮긴 것이다.

```text
shared/docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
shared/docs/operating_guides/02_integrated_subagent_remote_stabilization_workflow.md
RCS-AC-v1.1/docs/v1_1/rcs_ac_v1_1_carla_64_128_smoke_plan.md
shared/docs/rcs_ac_project_handoff_for_new_codex_chat.md
```

v1.1과 같은 점:

```text
MacBook은 Codex, sub-agent, 분석, 문서, 코드 수정, 로컬 테스트를 담당한다.
Windows desktop은 CARLA, Python CARLA 실행, heavy sensor IO, checkpoint, raw artifact를 담당한다.
Mac에서 SSH/SCP로 Windows 실행을 트리거하고 text/table log만 회수한다.
CARLA readiness guard를 매 CARLA 실행 전 반복한다.
```

v2에서 달라지는 점:

```text
v2 final action contract는 Raw2Drive PDF의 39 discrete actions다.
v1.1 waypoint + pure-pursuit controller는 v2 final policy로 가져오지 않는다.
v2 clean gate가 잡히기 전 fusion ablation을 시작하지 않는다.
첫 Windows CARLA run은 성능 평가가 아니라 adapter/action/reward/logging 연결 smoke다.
```

## 1. Machine Role Split

MacBook 역할:

```text
1. Codex goal orchestration
2. sub-agent 분석과 findings 통합
3. Raw2Drive PDF/구현 문서 확인
4. 코드, 설정, 테스트, 문서 수정
5. Mac에서 가능한 unit/static/smoke test 실행
6. Windows 실험 명령 SSH 실행
7. Windows text/table log 회수
8. result note와 GOAL_STATUS.md 업데이트
```

Windows 역할:

```text
1. CARLA 0.9.16 실행
2. CARLA Python API import/probe
3. v2 CARLA collect/eval/train script 실행
4. raw sensor frame, replay buffer, checkpoint, heavy output 보관
5. experiment log, csv, json summary 생성
```

금지:

```text
Windows에서 Codex sub-agent를 돌리지 않는다.
Mac으로 checkpoint, .npz, raw image dump 전체를 기본 회수하지 않는다.
Windows에서 생성된 heavy artifact를 GitHub에 올리지 않는다.
Mac 로컬 테스트만 통과한 코드를 Windows CARLA 검증 없이 clean 성능으로 간주하지 않는다.
```

## 2. Paths And Access

Mac path:

```text
/Users/song-yonghwi/code/RCS-AC-first
/Users/song-yonghwi/code/RCS-AC-first/RCS-AC-v2
/Users/song-yonghwi/code/windows_results/rcs-ac-v2
```

Windows path:

```text
C:\Users\thddy\Documents\code\RCS-AC-first
C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v2
C:\Users\thddy\Documents\code\shared_core
```

Current official v2 clean clone:

```text
C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656
```

If a newer clean clone path is used, keep the same rule: the Windows run must
come from a committed/pushed checkout of branch `codex/rcs-ac-v2-raw2drive`,
after `git pull --ff-only`, with `git status --short RCS-AC-v2` empty. Record
`git rev-parse --short HEAD` with every probe, replay, training, and clean eval
result.

Remote identity:

```text
SSH login: song@100.87.219.58
LAN address when available: 192.168.0.13
Tailscale address: 100.87.219.58
```

Windows user name and Windows profile path can differ. Keep commands explicit.

## 3. Mac SSH/Tailscale Preflight

If Mac cannot reach the Windows Tailscale IP through the normal network route,
start userspace Tailscale on Mac:

```bash
mkdir -p "$HOME/.local/run" "$HOME/.local/share/tailscale-codex"
/opt/homebrew/opt/tailscale/bin/tailscaled \
  --tun=userspace-networking \
  --socks5-server=localhost:1055 \
  --socket=$HOME/.local/run/tailscaled-codex.sock \
  --statedir=$HOME/.local/share/tailscale-codex
```

Use the same socket/statedir for future rounds. A fresh state directory may
require a new browser login.

Before each Windows CARLA round:

```bash
TS_SOCKET="$HOME/.local/run/tailscaled-codex.sock"

pgrep -af 'tailscaled.*tailscale-codex' || true
lsof -nP -iTCP:1055 -sTCP:LISTEN || true
tailscale --socket="$TS_SOCKET" status
```

Use keepalive wrappers for long Windows runs:

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

Connectivity check:

```bash
rcswinssh song@100.87.219.58 'hostname'
```

If a previous SSH control socket becomes stale after Windows reboot or thermal
shutdown, bypass multiplexing for the next recovery command:

```bash
ssh \
  -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o ControlMaster=no \
  -o 'ProxyCommand=nc -x localhost:1055 %h %p' \
  song@100.87.219.58 'hostname'
```

Use the same `ControlMaster=no` form for immediate post-reboot `git pull`,
CUDA guard, CARLA stop, and result-read commands when the socket state is
uncertain.

If LAN is available and stable, the same workflow can use `song@192.168.0.13`
without the SOCKS `ProxyCommand`.

After Windows reboot, thermal shutdown, router changes, or Tailscale reconnect,
do not treat the documented LAN IP as authoritative. First verify the current
address with Windows `ipconfig` or Tailscale status, then run the post-reboot
`ControlMaster=no` SSH probe before any CUDA/CARLA work.

Hardware-attestation policy failure:

```text
If userspace Tailscale exits or becomes unreachable with:
  policy requires hardware attestation, but device does not support it

classify the issue as remote-tunnel infrastructure, not CARLA/model failure.
Do not start or judge Windows experiments from this state.
```

Recovery order:

```text
1. Try system Tailscale / normal OS route instead of the userspace daemon.
2. If Mac and Windows are on the same LAN, try direct LAN SSH:
   ssh song@192.168.0.13 'hostname'
3. If both are unavailable, fix the tailnet device policy or Windows/Mac
   Tailscale authorization before starting CUDA/CARLA work.
4. Once connectivity returns, run git pull --ff-only, CUDA guard, dependency
   probe, and then the intended candidate wrapper.
```

## 4. Required Goal Order

Every v2 implementation/experiment round that touches CARLA must follow this
order:

```text
1. Read RCS-AC-v2/GOAL_STATUS.md.
2. Read current stage in docs/01_goal_driven_implementation_runbook.md.
3. Apply docs/02_persistent_subagent_goal_protocol.md.
4. Spawn at least Agent A + Agent C at stage/work-packet entry.
5. Add Agent B for code/config/test/logging/checkpoint wiring.
6. Integrate sub-agent findings.
7. Write Implementation Brief.
8. Edit code/config/tests/docs on Mac.
9. Run Mac-side unit/static validation.
10. Sync the exact code state to Windows.
11. Run CARLA readiness guard on Windows.
12. Run one CARLA collect/eval/train/smoke step.
13. Run CARLA readiness guard again before the next CARLA step.
14. Pull complete text/table logs to Mac.
15. Analyze logs against the stage gate.
16. Update result note and GOAL_STATUS.md.
```

Short form:

```text
Sub-agents first
Implementation Brief second
Edit third
Validate fourth
Windows CARLA run fifth
Document sixth
```

Do not let sub-agent use stop after the first ten minutes. Re-trigger it at:

```text
stage entry
major edit batch
first Windows CARLA run for a stage
repeated failure
gate review
action/reward/schema/leakage uncertainty
clean baseline failure that could be implementation-related
```

If a spawn attempt fails with:

```text
collab spawn failed: agent thread limit reached
```

then the problem is not Windows connectivity. It means the Codex sub-agent
thread slots are full. Close completed old agents first, then spawn only the
next bounded sidecars needed for the current critical path. Keep the main
rollout moving locally while sidecars analyze non-blocking questions.

## 5. Syncing Code To Windows

Preferred sync path:

```text
Mac commit/push -> Windows git pull --ff-only -> Windows run
```

For efficiency, do not commit every tiny Mac-only edit. Batch local work into a
coherent work packet, run local validation, then commit/push once before the
official Windows run.

Scratch Windows runs are allowed only for connectivity/infrastructure probing.
Do not record scratch runs as clean-performance evidence.

Windows pull command:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\RCS-AC-first; git pull --ff-only"'
```

If a change is not committed/pushed yet, do not run Windows CARLA as if it is
testing the Mac state. Either sync intentionally or delay the Windows run.

Detailed GitHub sync, artifact, and `.git` cleanup policy:

```text
RCS-AC-v2/docs/04_git_sync_artifact_cleanup_policy.md
```

## 6. CARLA Readiness Guard

Run the guard before every CARLA command:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\shared\scripts\windows\ensure_carla_ready.ps1" -ProjectRoot "C:\Users\thddy\Documents\code\RCS-AC-first" -StartScript "C:\Users\thddy\Documents\code\RCS-AC-first\shared\scripts\windows\start_carla_interactive.ps1"'
```

Force restart after crash, timeout, duplicate server, or unclear state:

```bash
rcswinssh song@100.87.219.58 'powershell -ExecutionPolicy Bypass -File "C:\Users\thddy\Documents\code\RCS-AC-first\shared\scripts\windows\ensure_carla_ready.ps1" -ProjectRoot "C:\Users\thddy\Documents\code\RCS-AC-first" -StartScript "C:\Users\thddy\Documents\code\RCS-AC-first\shared\scripts\windows\start_carla_interactive.ps1" -ForceRestart'
```

The guard must verify:

```text
CarlaUE4 / CarlaUE4-Win64-Shipping process exists
TCP port 2000 is open
Python 3.12 can import carla
carla.Client can connect to 127.0.0.1:2000
world/map can be read, and target town can be loaded when requested
```

Safe process state:

```text
One launcher process plus one CarlaUE4-Win64-Shipping process can be normal.
No process is unsafe.
Duplicate CarlaUE4-Win64-Shipping processes are unsafe.
CrashReportClient is unsafe.
Port 2000 closed is unsafe.
Python probe failure is unsafe.
```

If Python never reaches the v2 rollout/eval script or metrics are not written,
classify it as infrastructure failure, not model failure.

## 7. v2 Stage-Specific Windows Usage

Stage 0: Contract preflight

```text
Normally Mac-only.
Implement action/schema/reward/replay/leakage tests first.
Windows CARLA is optional and only used to verify CARLA Python availability.
```

Optional Windows environment probe:

```bash
rcswinssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\RCS-AC-first; py -3.12 -B -c \"import carla; c=carla.Client(''127.0.0.1'',2000); c.set_timeout(30); print(c.get_server_version()); print(c.get_world().get_map().name)\""'
```

Stage 1: Privileged world model / data path

```text
Use Windows to collect tiny privileged CARLA trajectories after Stage 0 passes.
First run is 64/128-step smoke, not clean performance.
Required checks: 43-channel BEV shape, ego vector shape, action id in 0..38,
reward finite, cont in {0,1}, replay sequence sample readable on Mac.
```

Future command contract after scripts exist:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first
$env:PYTHONPATH="C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v2\code;$env:PYTHONPATH"
py -3.12 -B -m rcs_ac_v2.training.collect_privileged_rollouts `
  --config RCS-AC-v2\configs\collect\privileged_carla_smoke.yaml `
  --steps 128 `
  --output RCS-AC-v2\experiments\01_privileged_world_model\windows_smoke
```

Stage 2: Privileged policy clean gate

```text
Use Windows for closed-loop clean eval.
Clean performance must be diagnosed before raw/fusion work.
No v1.1 pure-pursuit final controller should be inserted as a hidden helper.
```

Stage 3: Raw stream and alignment

```text
Use Windows for multi-view image + IMU collection and short alignment eval.
Raw policy inference must not receive privileged BEV/GT tensors.
Privileged stream can guide training/rollout as documented, not inference leakage.
```

Stage 4: Raw policy clean gate

```text
Use Windows clean closed-loop eval.
Do not start fusion ablation until the raw/privileged clean gate is understood.
```

Stage 5: Fusion-only ablation

```text
Only after clean-capable baselines exist.
Keep input/action/reward/eval contract fixed.
Change only the fusion mechanism under test.
```

Stage 6: Stress eval/report

```text
Use Windows for weather/noise/stress runs.
Mac pulls text/table logs and writes final report.
```

## 8. 64/128-Step Smoke Gate

The first CARLA run for a new v2 stage should use 64 or 128 steps.

64-step smoke pass:

```text
CARLA process does not crash
Python script exits normally
action id is finite int in 0..38
to_carla_control output is in CARLA range
reward is finite
cont is 0 or 1
env.step progresses until 64 steps or normal done
step log csv/jsonl is written
summary json is written
```

128-step smoke pass:

```text
All 64-step checks pass
done_reason is logged
collision/offroad/stuck are separated from adapter/runtime errors
sensor health fields are logged when raw sensors are involved
episode return/length are finite
Raw2Drive contract version is recorded
```

Smoke is not a performance gate. A poor driving outcome can still be useful if
adapter/action/reward/logging are valid.

## 9. Log Pullback Policy

Mac result root:

```text
/Users/song-yonghwi/code/windows_results/rcs-ac-v2
```

Default pullback:

```text
logs/*.log
logs/*.txt
logs/*.csv
logs/*.json
remote_runs/**/*.json
remote_runs/**/*.csv
experiments/**/result.md
experiments/**/summary.json
experiments/**/*.csv
experiments/**/*.json
```

Do not pull by default:

```text
checkpoints
.pt/.pth/.ckpt
.npz replay buffers
raw camera frame folders
large videos
```

Example pull command after a run named `stage1_privileged_128`:

```bash
mkdir -p /Users/song-yonghwi/code/windows_results/rcs-ac-v2/stage1_privileged_128
rcswinscp -r \
  'song@100.87.219.58:C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v2\experiments\01_privileged_world_model\windows_smoke\*.json' \
  /Users/song-yonghwi/code/windows_results/rcs-ac-v2/stage1_privileged_128/
rcswinscp -r \
  'song@100.87.219.58:C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v2\experiments\01_privileged_world_model\windows_smoke\*.csv' \
  /Users/song-yonghwi/code/windows_results/rcs-ac-v2/stage1_privileged_128/
```

If Windows OpenSSH/SCP wildcard handling is unreliable, compress only selected
text/table files on Windows and pull the small archive.

## 10. Failure Classification

Always classify failures before patching:

```text
infrastructure failure:
  SSH disconnect, Tailscale stale state, CARLA port closed, Python cannot import carla

CARLA failure:
  Unreal crash, CrashReportClient, duplicate server, RPC timeout, map load failure

adapter failure:
  missing obs/info key, shape mismatch, coordinate mismatch, invalid dtype

contract failure:
  invalid action id, reward NaN, cont outside {0,1}, privileged leakage

model failure:
  loss NaN, bad checkpoint load, policy logits invalid, RSSM rollout invalid

driving failure:
  collision, offroad, stuck, low route progress, bad return with valid runtime
```

Only model/driving failures should drive algorithm changes. Infrastructure,
CARLA, adapter, and contract failures must be fixed before interpreting clean
performance.

## 11. Result Note Template

Every Windows CARLA v2 run should leave a short result note:

```text
# Result

Run name:
Stage:
Date:
Windows command:
CARLA guard result:
Sub-agent mode:
Commit/hash or sync source:

## Gate

PASS / PARTIAL / FAIL / BLOCKED

## Key Metrics

steps:
episode_return:
done_reason:
collision:
offroad:
stuck:
action_contract:
observation_contract:
reward_contract:
leakage_check:

## Failure Class

infrastructure / CARLA / adapter / contract / model / driving / none

## Evidence

List csv/json/log files pulled to Mac.

## Next Action

One concrete next step.
```

## 12. v2 Runner Status

Use v2-specific Windows runners instead of overloading v1.1 scripts:

```text
RCS-AC-v2/scripts/windows/run_v2_carla_probe.ps1
RCS-AC-v2/scripts/windows/run_stage1_privileged_smoke.ps1
RCS-AC-v2/scripts/windows/check_stage2_windows_dependencies.ps1
RCS-AC-v2/scripts/windows/run_stage2_privileged_obs_probe.ps1
RCS-AC-v2/scripts/windows/collect_stage2_privileged_replay.ps1
RCS-AC-v2/scripts/windows/train_stage2_privileged_from_replay.ps1
RCS-AC-v2/scripts/windows/run_stage2_clean_eval.ps1
RCS-AC-v2/scripts/windows/run_stage3_raw_alignment_smoke.ps1
RCS-AC-v2/scripts/windows/run_stage4_raw_clean_eval.ps1
RCS-AC-v2/scripts/windows/run_stage5_fusion_ablation.ps1
```

`run_stage2_clean_eval.ps1` now exists. Its `-DryRun` path intentionally skips
CARLA readiness and writes `LOCAL_SMOKE_ONLY`; its real path runs `git pull
--ff-only` unless `-SkipGitPull` is set, runs
`check_stage2_windows_dependencies.ps1`, runs v2-local
`ensure_carla_ready.ps1`, then calls:

```text
python -m rcs_ac_v2.evaluation.evaluate_clean --run-carla
```

The runner sets `PYTHONPATH` to include:

```text
RCS-AC-v2\code
RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code
RCS-AC-v1.1\code
<workspace root containing sibling shared_core>
```

This is required because the v2 Stage 2 adapter factory reuses the v1.1-style
shared-core CARLA env builder for the first clean gate.

Before the first clean eval, run the shape-only probe. For the H1 observation
upgrade lane, use the committed 32x32 probe config so the evidence does not
overwrite or mix with the older 16x16 lane:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_privileged_obs_probe.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged_32x32.yaml `
  -StepNeutral
```

This writes `privileged_obs_probe.json`, not `summary.json`. It must record
`clean_gate_evidence=false`, exact `privileged_bev [1,1,43,32,32]`,
`privileged_ego [1,1,8]`, finite tensors, raw obs/info keys, coverage
diagnostics, and optional neutral-step info. It cannot be used as clean PASS
evidence.

After the probe passes and before official clean eval, collect privileged replay
and train replay-source checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\collect_stage2_privileged_replay.ps1 `
  -ConfigPath RCS-AC-v2\configs\eval\clean_128_stage2_privileged_32x32.yaml `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_h1_32x32_tiny `
  -Episodes 1 `
  -MaxSteps 8 `
  -TargetSpeedKmh 18 `
  -MinRecordedSteps 1 `
  -SkipFailedEpisodes

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\train_stage2_privileged_from_replay.ps1 `
  -ReplayPath RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_h1_32x32_tiny `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_h1_32x32_tiny
```

For longer replays where steering labels are sparse, add
`-SteerSequenceFraction 0.75` to oversample sequences that contain forward
steering actions. Still require the checkpoint action probe to pass before any
clean eval rerun.

Replay `.npz` and checkpoint `.pt` files are heavy local artifacts. Do not put
them in GitHub. Commit only small manifests/result notes and use sha256 values
from the manifest to fill the clean config.

The clean-128 config now points at the default factory:

```text
rcs_ac_v2.evaluation.carla_adapter:create_stage2_carla_adapter
```

Use `-AdapterFactory module:function` only to override the config for a
Windows-local probe. With the current null checkpoint paths, real mode must exit
nonzero with `failure_class=model` before CARLA env construction. Synthetic
smoke checkpoints are also rejected by provenance. Both are classified setup
failures, not clean evidence.

The v2 runner now carries its own readiness scripts:

```text
RCS-AC-v2/scripts/windows/ensure_carla_ready.ps1
RCS-AC-v2/scripts/windows/start_carla_interactive.ps1
```

Default CARLA launch is intentionally low-load for the Windows workstation:

```text
-quality-level=Low
-nosound
-windowed
-ResX=640
-ResY=360
-benchmark -fps=10
```

This follows the v1.1 remote workflow, which already used `Low`, `800x600`,
`-nosound`, and `-windowed`, but lowers the resolution further and caps FPS to
reduce heat during repeated replay collection/eval. Do not use `-nullrhi` in
the official clean lane because it can invalidate rendered sensor paths.

The remaining temporary dependency is the shared-core env runtime:

```text
<workspace root>\shared_core
```

`check_stage2_windows_dependencies.ps1` must pass before CARLA starts. Official
clean PASS should not be claimed until this dependency is verified. The adapter
now owns the minimal shared_core env builder and the v1.1 env config is copied
under v2 for the clean clone lane. The sibling `shared_core` package must still
be available next to the checkout or replaced by v2-local env code.

If a local run needs to override the v2 defaults, keep the v2 code clean-cloned
from GitHub and pass the temporary dependency paths explicitly:

```powershell
-EnvConfigPathOverride C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v1.1\configs\v1_1\bc_clean_v1_1_2048_eval.yaml `
-SupportCodePath C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v1.1\goal_achieved_round_0002_execution\support_code `
-V11CodePath C:\Users\thddy\Documents\code\RCS-AC-first\RCS-AC-v1.1\code `
-SharedCorePath C:\Users\thddy\Documents\code\shared_core
```

This keeps the official v2 source tied to the GitHub branch while making the
temporary runtime dependency explicit. It is still a temporary bridge; replace
sibling shared_core with v2-local env code before final reproducibility claims
if possible.

If a candidate wrapper exits nonzero after creating
`stage2_candidate_cross_replay_gate.json`, treat it as a possible gate reject
before assuming an infrastructure failure. Run the corresponding summary-only
path, or `summarize_stage2_candidate_gate.ps1` with the gate path, so the reject
reason and pullback targets are recorded before starting another candidate.
