# Goal Bootstrap Summary

Created: 2026-05-05

이 문서는 장시간 `/goal` 실행에서 매번 긴 operating guide 전문을 읽지 않기 위한 짧은 부트스트랩 문서다.

기본 원칙:

```text
1. 이 파일을 먼저 읽는다.
2. 원격 연결, Tailscale, SSH, SCP, CARLA health, Windows 실행 문제가 생겼을 때만
   docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md의 관련 섹션을 부분적으로 읽는다.
3. subagent workflow 세부 역할이 필요할 때만 03/04 문서를 부분적으로 읽는다.
4. 모든 긴 원문 문서를 매번 처음부터 끝까지 읽지 않는다.
```

---

## 1. 이번 프로젝트 목표 요약

RCS-AC 최종 목표:

```text
RCS-AC가 PPO/SAC baseline보다 악천후, 센서드랍, 센서노이즈 환경에서 더 강건하다는 것을
공정한 clean/robustness 실험으로 입증한다.
```

현재 v1.1 목표:

```text
RCS-AC v1.1을 Windows CARLA clean 2048-step run까지 안정화한다.
먼저 64/128 smoke를 통과시키고, tiny CARLA dataset, BC smoke train, 256/512/2048 clean eval 순서로 승격한다.
```

---

## 2. 기본 읽기 규칙

항상 읽을 문서:

```text
docs/operating_guides/00_goal_bootstrap_summary.md
docs/v1_1/rcs_ac_v1_1_carla_64_128_smoke_plan.md
docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
```

필요할 때만 부분적으로 읽을 문서:

```text
docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
  SSH/Tailscale/SCP/CARLA startup/remote troubleshooting 필요 시

docs/operating_guides/03_lite_subagent_stabilization_workflow.md
  Lite subagent 역할/절차가 불명확할 때

docs/operating_guides/04_full_subagent_roles_and_legacy_rules.md
  Full workflow로 승격했을 때

docs/operating_guides/02_integrated_subagent_remote_stabilization_workflow.md
  원격 실행 + subagent workflow를 함께 조율해야 할 때
```

---

## 3. Mac/Windows 역할

Mac:

```text
Codex 실행
코드 분석/수정/문서화
pytest/py_compile 등 lightweight 검증
git push
Windows 원격 명령 실행
text/table logs pullback
결과 분석 및 다음 단계 결정
```

Windows:

```text
CARLA 실행
Python 3.12 + CARLA 0.9.16 API 실행
RCS-AC-first와 shared_core side-by-side 유지
git pull
CARLA health check
dataset collection / eval / training smoke 실행
logs/checkpoints/data는 로컬 보관
```

Windows 기본 경로:

```text
C:\Users\thddy\Documents\code\RCS-AC-first
C:\Users\thddy\Documents\code\shared_core
```

SSH:

```text
song@100.87.219.58
```

---

## 4. 원격 연결 최소 규칙

원격 실행 전 Mac preflight:

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

권장 SSH 옵션:

```text
ServerAliveInterval=30
ServerAliveCountMax=6
ControlMaster=auto
ControlPersist=8h
ControlPath=~/.ssh/rcs-win-%r@%h:%p
ProxyCommand=nc -x localhost:1055 %h %p
```

Tailscale 주의:

```text
같은 socket/statedir를 재사용한다.
임시 /tmp statedir를 쓰지 않는다.
새 statedir를 쓰면 새 로그인/auth URL이 필요할 수 있다.
```

자세한 SSH/Tailscale 문제는:

```text
docs/operating_guides/01_remote_mac_windows_stabilization_workflow.md
```

관련 섹션만 찾아 읽는다.

---

## 5. Windows 실행 순서

Windows에서 매 run 전:

```powershell
Set-Location 'C:\Users\thddy\Documents\code\RCS-AC-first'
git pull --ff-only
.\scripts\windows\ensure_carla_ready.ps1
```

CARLA 중복 실행 금지:

```text
CARLA 프로세스가 2개 이상이면 TaskGraphThread crash 위험이 커진다.
ensure_carla_ready.ps1를 먼저 실행한다.
CARLA crash/timeout과 모델 주행 실패를 분리 기록한다.
```

---

## 6. v1.1 2048 안정화 승격 순서

반드시 이 순서로만 승격한다:

```text
1. Mac pytest/py_compile pass
2. Windows git pull
3. CARLA health check pass
4. tiny CARLA dataset collection
5. train_bev_bc real tiny dataset 1 epoch smoke
6. CARLA 64-step smoke
7. CARLA 128-step smoke
8. CARLA 256-step clean eval
9. CARLA 512-step clean eval
10. CARLA 2048-step clean eval
```

금지:

```text
64/128 smoke 실패 상태에서 256/512/2048로 승격하지 않는다.
512 pass를 2048 pass로 간주하지 않는다.
random policy smoke를 학습된 모델 성능으로 기록하지 않는다.
```

---

## 7. 실패 분류

실패는 반드시 분리한다:

```text
CARLA crash:
  Unreal/CARLA process crash, TaskGraphThread, RPC timeout

adapter failure:
  shape mismatch, obs_flat dim mismatch, coordinate mismatch, NaN, missing key

model failure:
  checkpoint mismatch, forward NaN/inf, invalid output shape

controller failure:
  invalid waypoint, target speed cap stuck at 0, invalid control range

driving failure:
  collision, offroad, severe early offroad, stuck, poor route progress
```

이 분류를 `summary.json`, docs 기록, final report에 반영한다.

---

## 8. 공정 비교 보호 규칙

PPO/SAC와 나중에 비교해야 하므로 다음을 지킨다:

```text
RCS-AC에만 유리한 runtime steering override 금지
lane_offset/heading_error/route y 기반 직접 steer rule 금지
특정 route/seed/town 전용 steering constant 금지
CARLA privileged signal을 policy inference input에 넣기 금지
GT BEV/actor boxes/lane_offset/heading_error는 target/debug/offline audit 전용
controller safety cap 결과와 raw policy 결과를 분리 보고
shared reward/action wrapper 변경 시 PPO/SAC에도 동일 적용 가능하게 문서화
clean 2048 안정화 전 robustness 비교로 넘어가지 않기
```

---

## 9. 하드코딩 금지

금지:

```text
offroad 방지를 위한 lane_offset runtime override
route/lane 기반 직접 steer rule
특정 실패 지점만 우회하는 if문
GT BEV를 inference input에 concat
CARLA actor position/bounding box를 policy input에 concat
결과가 나쁜 로그 삭제/은폐
사용자 변경 git reset/checkout으로 되돌리기
```

허용:

```text
shape validation
NaN/inf guard
invalid waypoint stop cap
pure pursuit controller safety cap
debug-only GT BEV rendering
offline supervised target 생성을 위한 privileged CARLA signal
```

---

## 10. 필수 검증 명령

Mac:

```bash
python -m py_compile $(find rcs_ac_v1_1 -name '*.py' -not -path '*/__pycache__/*')
python -m pytest rcs_ac_v1_1/tests -q
```

Windows examples:

```powershell
py -3.12 -B -m rcs_ac_v1_1.scripts.collect_bev_expert_dataset `
  --config configs/v1_1/full_bev_smoke.yaml `
  --env-config <existing_rcs_ac_first_carla_config> `
  --output data/v1_1/bev_expert_tiny_carla_smoke.npz `
  --num-steps 128 `
  --host 127.0.0.1 `
  --port 2000

py -3.12 -B -m rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit `
  --config configs/v1_1/full_bev_smoke.yaml `
  --env-config <existing_rcs_ac_first_carla_config> `
  --checkpoint checkpoints/v1_1_bev_bc_tiny_carla_smoke/last.pt `
  --run-name v1_1_carla_eval_64_smoke `
  --num-steps 64 `
  --device cuda `
  --mode carla
```

---

## 11. 로그 회수 원칙

Mac으로 기본 회수할 것:

```text
logs/**/*.csv
logs/**/summary.json
remote_runs/summary.json
remote_runs/metrics.csv
작은 .txt 실행 로그
```

기본 회수하지 않을 것:

```text
full image folders
checkpoints
large .npz datasets
large binary artifacts
```

필요할 때만 명시적으로 가져온다.

---

## 12. 문서화 위치

v1.1 구현/실행 문서:

```text
docs/v1_1/
```

라운드 결과:

```text
docs/stabilization_rounds/
```

운영 규칙:

```text
docs/operating_guides/
```

운영 규칙 폴더에는 one-off 실험 결과를 넣지 않는다.
