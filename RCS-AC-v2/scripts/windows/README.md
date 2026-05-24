# RCS-AC v2 Windows Scripts

작성일: 2026-05-24

이 폴더는 v2 전용 Windows CARLA runner를 둘 자리다.

현재 상태:

```text
Stage 2 clean eval runner는 있다.
CARLA readiness/start는 v2-local script를 사용한다.
Stage 2 privileged replay collection/training runner도 있다.
```

공용 script:

```text
RCS-AC-v2/scripts/windows/ensure_carla_ready.ps1
RCS-AC-v2/scripts/windows/start_carla_interactive.ps1
```

CARLA는 Windows 과열/다운을 줄이기 위해 v2-local start script에서 기본
저부하 모드로 시작한다:

```text
quality-level: Low
resolution: 640x360
audio: disabled
window mode: windowed
fps cap: 10 via -benchmark -fps=10
```

이전 v1.1 방식도 `Low`, `800x600`, `-nosound`, `-windowed`였고, v2는 여기에
더 낮은 해상도와 FPS cap을 추가한다. `-nullrhi`처럼 sensor/render 경로를
깨뜨릴 수 있는 옵션은 clean evidence lane에서 쓰지 않는다.

v2 CARLA 실행 절차 문서:

```text
RCS-AC-v2/docs/03_windows_carla_remote_execution_workflow.md
RCS-AC-v2/docs/04_git_sync_artifact_cleanup_policy.md
```

나중에 추가할 v2 runner:

```text
run_v2_carla_probe.ps1
run_stage1_privileged_smoke.ps1
run_stage3_raw_alignment_smoke.ps1
run_stage4_raw_clean_eval.ps1
run_stage5_fusion_ablation.ps1
```

현재 추가된 v2 runner:

```text
check_stage2_windows_dependencies.ps1
run_stage2_privileged_obs_probe.ps1
collect_stage2_privileged_replay.ps1
train_stage2_privileged_from_replay.ps1
start_stage2_replay_training_detached.ps1
check_stage2_replay_training_detached.ps1
probe_stage2_checkpoint_actions.ps1
probe_stage2_candidate_cross_replays.ps1
summarize_stage2_candidate_gate.ps1
run_stage2_target24_candidate_cycle.ps1
run_stage2_target_speed_replay_cycle.ps1
run_stage2_clean_eval.ps1
ensure_carla_ready.ps1
start_carla_interactive.ps1
```

주의:

```text
현재 Stage 2 runner는 dry-run, CARLA readiness guard, real-mode evaluate_clean
호출, summary validation까지 제공한다.
run_stage2_privileged_obs_probe.ps1는 clean PASS를 만들지 않는
privileged_obs_probe.json 전용 shape/action probe다.
collect_stage2_privileged_replay.ps1는 Windows CARLA에서 simulator-state
semantic43/ego8 replay .npz를 수집한다. replay는 학습 입력이지 clean PASS
증거가 아니다.
train_stage2_privileged_from_replay.ps1는 replay로 world_model/policy
checkpoint와 sha256 manifest를 만든다. 이 checkpoint는 clean eval 입력으로만
사용하고, PASS 여부는 run_stage2_clean_eval.ps1 summary/metrics로만 판정한다.
긴 straight 구간 때문에 steering label이 묻히면 `-SteerSequenceFraction`으로
전진 조향 action이 포함된 sequence를 oversample한 후보를 만들고, action probe가
moving steer gate를 통과할 때만 clean eval로 보낸다.
rare steering class가 너무 강해져 over-steer collapse가 나면
`-ImitationBalancePower`를 0과 1 사이로 낮춘 후보를 먼저 offline probe한다.
replay에 steering window가 있는데도 정책이 straight/steer 한쪽으로 무너지면
`-WorldModelActionAuxWeight`로 world-model latent에 action-prediction 보조
손실을 추가한 후보를 먼저 만든다. 이 head는 runtime checkpoint에 쓰지 않고
latent가 action 조건을 보존하도록 훈련 중에만 사용한다.
checkpoint manifest의 `sampler` 필드는 priority window 수와 effective sampling
fraction을 기록하므로, replay 자체에 steering window가 충분한지 먼저 확인한다.
start_stage2_replay_training_detached.ps1는 긴 replay training을 Windows
scheduled task로 띄우고 train log path, task name, launch manifest path를
출력한다. SSH가 끊길 수 있는 원격 실행에서는 이 wrapper를 우선 사용하고,
완료 여부는 log의 TRAIN_EXIT_CODE와 checkpoint manifest 검증 결과로
확인한다. 같은 TaskName의 기존 task가 아직 Running이면 새 task를 덮어쓰지
않고 실패한다.
이 wrapper는 checkpoint output dir에
`train_stage2_privileged_from_replay.launch.json`도 남긴다. 재연결 후에는
TaskName/LogPath를 기억하지 못해도 OutputDir만으로 check script가 launch
manifest를 읽어 log/task 상태를 복구할 수 있다.
check_stage2_replay_training_detached.ps1는 detached training log, process id,
checkpoint manifest를 확인하고 TRAIN_READY_FOR_PROBE=True일 때만 action
probe 단계로 넘어가게 한다. 준비 판정은 manifest 존재, TRAIN_EXIT_CODE=0,
world_model/policy checkpoint 파일 존재, manifest sha256 일치까지 포함한다.
run_stage2_clean_eval.ps1의 `-RunOutputDir` 옵션은 validated Windows run의
summary.json, metrics.csv, 실제 입력 config를 지정한 증거 폴더로 복사한다.
이 옵션은 artifact mirror 편의 기능일 뿐 PASS/FAIL을 바꾸거나 summary 필드를
수정하지 않는다. stale artifact 착시를 막기 위해 runner는 eval 시작 직전에
canonical summary.json/metrics.csv만 지운 뒤 새 결과만 검증/복사한다.
probe_stage2_checkpoint_actions.ps1는 checkpoint manifest를 base clean config에
적용한 runtime config를 만들고, replay observation에서 deterministic action
분포를 점검한다. reverse_count=0, throttle 있는 action, 전진 throttle이 있는
조향 action, 이전 stuck 실패의 action_id=8 반복 여부를 clean rerun 전에
확인한다. `-MaxStepsPerFile 128`로 전체 4x128 replay를 확인하고,
`recommended_for_clean_rerun=true`는 reverse 0, single-action collapse 없음,
nonzero-throttle fraction 0.5 이상, moving/nonzero steer fraction 0.02 이상,
moving/nonzero steer count 12 이상, moving/nonzero steer fraction 0.5 이하,
dominant action fraction 0.85 이하, action_id=8 fraction 0.05 이하일 때만
나온다.
probe_stage2_candidate_cross_replays.ps1는 후보 checkpoint를 primary replay와
secondary replay 양쪽에서 action probe + latent probe로 교차 검증한다.
target-24 후보는 target-24 replay만 통과했다고 512 clean으로 보내지 말고,
기존 steer2 replay에서도 collapse하지 않는지 이 wrapper로 확인한다. probe가
끝나면 `stage2_candidate_cross_replay_gate.json`을 만들고, 모든 replay에서
action probe `recommended_for_clean_rerun=true` 및 latent probe 기준
`linear_probe_test_accuracy >= 0.80`, majority 대비 margin `>= 0.15`를
만족할 때만 512 clean 후보로 본다.
summarize_stage2_candidate_gate.ps1는 `stage2_candidate_cross_replay_gate.json`을
읽고 다음 행동을 기계적으로 출력한다. `ready_for_clean_eval=true`면
`run_official_512_clean_eval`까지만 승격하고, reject면 target-22 fallback을
지시한다. 이 summary 역시 `clean_gate_evidence=false`이며 clean PASS 기록으로
쓰면 안 된다.
run_stage2_target24_candidate_cycle.ps1는 target-24 후보 재개용 orchestration
script이며, `-CandidateName`/`-PrimaryReplayName`을 바꿔 target-22 후보에도
재사용할 수 있다. 먼저 기존 OutputDir을 check_stage2_replay_training_detached.ps1로
검사하고, 준비되지 않았으면 fresh rerun OutputDir으로 detached training을
시작한다. fresh rerun을 만든 경우 `${OutputDir}.latest.json`에 active
OutputDir을 기록하고, 다음 재개 명령은 이 포인터를 먼저 읽는다. 따라서
Windows/Tailscale 중단 후 같은 명령을 반복해도 이미 시작한 rerun manifest를
우선 검사하고 불필요한 rerun 폴더 증식을 피한다. `-ProbeWhenReady`를 붙이면
TRAIN_READY_FOR_PROBE=True일 때 cross replay probe까지 이어서 실행한다.
run_stage2_target_speed_replay_cycle.ps1는 target speed별 replay 수집과
candidate cycle을 한 번에 이어주는 wrapper다. 기본값은 target-22이며,
replay manifest가 없으면 `-CollectIfMissing`일 때만 CARLA replay를 수집하고,
그 뒤 `run_stage2_target24_candidate_cycle.ps1`로 detached training/cross probe
단계를 넘긴다.

Target-24 재개 명령:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656
git pull --ff-only

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_target24_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -StartIfMissing `
  -ProbeWhenReady
```

Target-24가 cross replay gate에서 reject되거나 여전히 512 후보로 불안정하면
target-22를 다음 speed-timing 후보로 수집/학습/probe한다:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_target_speed_replay_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -CandidateName target22 `
  -TargetSpeedKmh 22 `
  -CollectIfMissing `
  -StartIfMissing `
  -ProbeWhenReady
```

두 명령 모두 offline gate만 만든다. 512 clean eval은
`stage2_candidate_cross_replay_gate.json`의 `ready_for_clean_eval=true`를 확인한
뒤 별도 `run_stage2_clean_eval.ps1`로 실행한다.

Gate 요약 명령:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\summarize_stage2_candidate_gate.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -GatePath RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_t24_aux1_balpow05_iw5_cross_probe\stage2_candidate_cross_replay_gate.json `
  -CandidateName target24 `
  -FallbackCandidateName target22
```
check_stage2_windows_dependencies.ps1는 CARLA 실행 전에 v2 code,
shared_core, env config, readiness script 의존성을 검사한다.
base config의 checkpoint path/sha256은 아직 null이므로 그대로 real mode를
실행하면 model/world_model_checkpoint_missing 실패로 끝나는 것이 정상이다.
synthetic checkpoint export는 plumbing 증거일 뿐 clean PASS 증거가 아니다.
clean eval config는 local_synthetic_smoke_not_carla checkpoint를 거부하고
windows_carla_privileged_replay source checkpoint만 official clean 입력으로
허용한다.
real trained checkpoint, Windows privileged obs validation, source=windows_carla
summary.json, metrics.csv가 모두 갖춰지기 전까지 clean PASS를 기록하지 않는다.
현재 Stage 2 runner는 v2-local minimal shared_core env builder bridge를
사용한다. readiness script도 v2-local로 옮겼다. 공식 Windows run 전에는
sibling shared_core가 Windows workspace에 존재하는지 확인하거나 v2 전용
env builder로 완전히 옮긴다. 필요할 때만 -EnvConfigPathOverride,
-SupportCodePath, -V11CodePath, -SharedCorePath를 명시해서 기존 Windows
runtime dependency 위치를 고정한다.
```

원칙:

```text
Windows는 CARLA와 heavy runtime만 담당한다.
Codex, sub-agent, 분석, 문서화는 Mac에서 진행한다.
각 runner는 실행 command, output path, summary json path를 명시적으로 출력해야 한다.
artifact auto-copy는 Windows clean-clone 실행 결과를 run-id 폴더로 운반하는
용도로만 사용하고, 공식 판정은 항상 source=windows_carla summary.json과
metrics.csv validation으로 한다.
```
