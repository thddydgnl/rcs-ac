# RCS-AC v2 Windows Scripts

작성일: 2026-05-24

이 폴더는 v2 전용 Windows CARLA runner를 둘 자리다.

현재 상태:

```text
Stage 2 clean eval runner는 있다.
CARLA readiness/start는 v2-local script를 사용한다.
Stage 2 privileged replay collection/training runner도 있다.
Stage 2 replay 학습, offline probe, official clean eval은 기본 Device=cuda로
실행한다.
2026-05-26 기준 현재 Stage 2 clean은 아직 미해결이다. 최신 best official
512는
`8859f1d_t22_steer2_corrmix045060064_w00067_ssf010_msfpa2_latecap47w010_r066076`
에서 route_progress=0.7435897436 offroad FAIL이다. 이 후보는 offline
target22/steer2 gate를 통과했지만 official tail이 여전히 action `7` 직진과
action `4` 왼쪽 조향으로 끝났다. action `10` route-local override 후보는
offline gate를 통과했지만 official 512가 route_progress=0.7179487179로
regressed했다. 다음 clean-first 작업은 `corrmix045060064_w00067` 레시피를
보존하면서 post-0.72 correction replay/component를 만들거나 붙이는 방향이다.
post-0.72 min16 수집은 너무 sparse해서 실패했고, post-0.70 positive-steer
min8 component는 수집됐지만 seq8 mixed training에서 steer2 moving-steer
gate를 깨뜨렸다. `post070p=0.01`은 steer2 moving_steer_fraction
0.2395672334, `post070p=0.003`은 0.2472952087로 rejected였으므로 official
CARLA 512는 실행하지 않았다. 다음 후보는 global component weight sweep이
아니라 route-conditioned correction 또는 더 엄격한 post-0.72 replay
filtering을 우선한다.
그 직후 `cap10pre072` 후보는 offline cross-replay gate를 통과했지만 official
512에서 route_progress=0.6666666667 collision으로 회귀했다. 이 run의 tail은
route 0.6154-0.6410에서 action `10`이 반복되고 마지막 action `13`에서
충돌했다. 따라서 weak action10 pre-0.72 cap은 더 이상 official로 바로 보내지
말고, mid-route 0.60-0.67과 post-0.70/0.72 tail을 분리하는 offline-only
후보나 코드 레벨 route-conditioned filtering/loss를 먼저 쓴다.
추가 확인에서 `cap10pre070w1_o10tailw02`는 generic offline gate를 통과했지만
official 512가 route_progress=0.6153846154 collision으로 더 회귀했다. 더
보수적인 `cap10pre067w05_o10tailw002`도 generic gate는 통과했으나 stricter
promotion 기준에서 steer2 moving_steer_fraction=0.1622874807, action10 count=63
으로 rejected였고 official은 실행하지 않았다. 이제 weak/global action10
cap/override sweep은 중단하고, route-bin action_counts를 확인한 뒤
route-conditioned filtering/loss 또는 closed-loop failure-window collection으로
넘어간다.
raw/fusion은 아직 시작하지 않는다.
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
make_stage2_mixed_replay_manifest.ps1
run_stage2_mixed_candidate_cycle.ps1
resume_stage2_late_right_margin_candidate.ps1
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
2026-05-26 이후 action probe의 `route_progress_bins`에는 bin별 predicted
`action_counts`와 replay `label_action_counts`도 포함된다. post070p/action10
계열 후보는 generic `recommended_for_clean_rerun=true`만 보지 말고 route
`0.55-0.67`의 action10/action13 누수를 확인한 뒤 official 512로 승격한다.
probe_stage2_candidate_cross_replays.ps1는 후보 checkpoint를 primary replay와
secondary replay 양쪽에서 action probe + latent probe로 교차 검증한다.
target-24 후보는 target-24 replay만 통과했다고 512 clean으로 보내지 말고,
기존 steer2 replay에서도 collapse하지 않는지 이 wrapper로 확인한다. probe가
끝나면 `stage2_candidate_cross_replay_gate.json`을 만들고, 모든 replay에서
action probe `recommended_for_clean_rerun=true` 및 latent probe 기준
`linear_probe_test_accuracy >= 0.80`, majority baseline을 이길 수 있는
남은 accuracy headroom 대비 margin fraction `>= 0.15`를 만족할 때만
512 clean 후보로 본다. 단순 majority 대비 절대 margin 0.15는
`majority_action_fraction > 0.85`일 때 100% 초과 정확도를 요구할 수 있으므로
사용하지 않는다.
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
target-24와 target-22가 각각 자기 replay에는 맞지만 steer2 replay에서
무너지는 경우에는 CARLA를 더 켜기 전에
make_stage2_mixed_replay_manifest.ps1로 기존 target replay와 steer2 replay를
작은 mixed replay 디렉터리에 복사하고 새 manifest를 만든다. 이 manifest는
`git_commit`, source component, sha256을 다시 기록하며
train_stage2_privileged_from_replay의 `--replay` 파일 목록 검증과 1:1로 맞도록
episode 파일을 mixed 디렉터리 안에 둔다. mixed 후보도 clean 증거가 아니며,
원본 target replay와 원본 steer2 replay 양쪽 cross replay gate가 true일 때만
512 clean eval로 승격한다.
run_stage2_mixed_candidate_cycle.ps1는 이 mixed 후보 전용 재개 wrapper다.
학습 replay는 mixed manifest를 쓰되, probe는 원본 target replay와 원본 steer2
replay manifest를 따로 넘긴다. 따라서 target-22/steer2 양쪽 일반화 gate를
깨지 않고 Windows 재부팅 후 같은 명령으로 check, start, probe를 이어갈 수
있다.

run_stage2_late_right_margin_candidate.ps1는 Stage 2 route `0.666` collision
bracket을 위한 다음 clean-first 후보 wrapper다. target22 + steer2 mixed replay
위에서 `RouteLateRightSteerMargin*`로 late right-moving steer family를 밀고,
`RouteActionMassCap*`로 route `0.55-0.66` 구간의 strong-right action
`13,16,19` 선행 누출을 막는다. 이 wrapper도 offline cross replay gate만
생성한다. official 512 clean eval은 `stage2_candidate_cross_replay_gate.json`의
`ready_for_clean_eval=true`를 확인한 뒤 별도 실행한다.
재연결 직후에는 먼저 `-PlanOnly`를 붙여 resolved runner args, output dir,
gate path, official-eval blocker 문구를 확인한다. `-PlanOnly`는 git pull,
CUDA guard, scheduled task launch, replay 생성, probe를 실행하지 않는다.

resume_stage2_late_right_margin_candidate.ps1는 현재 canonical Stage 2
clean-first resume wrapper다. Windows가 다시 붙으면 이 wrapper를 먼저 쓴다.
순서는 git pull, dependency/CUDA check, late-right `-PlanOnly`, candidate
check/start/probe, gate summary다. 이 wrapper도 official 512 clean eval을
실행하지 않으며, gate summary가 `ready_for_clean_eval=true`를 보고할 때까지
512 clean eval은 계속 별도 보류한다. detached training이 아직 실행 중이면
`STAGE2_LATE_RIGHT_RESUME_CANDIDATE_RUNNING=True`를 출력하고 exit `3`으로
끝내며, 같은 명령을 나중에 다시 실행해 check/probe를 이어간다.

Canonical late-right resume command:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\resume_stage2_late_right_margin_candidate.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady `
  -Device cuda
```

Target-24 재개 명령:

```powershell
Set-Location C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656
git pull --ff-only

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\check_stage2_windows_dependencies.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -RequireCuda

powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_target24_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -Device cuda `
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
  -Device cuda `
  -CollectIfMissing `
  -StartIfMissing `
  -ProbeWhenReady
```

Target-22도 steer2 cross replay에서 reject되면, CARLA 부하를 늘리기 전에 기존
리플레이 두 개를 섞은 offline 후보를 만든다:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -SteerSequenceFraction 0.25 `
  -Device cuda `
  -MakeReplayIfMissing `
  -StartIfMissing `
  -ProbeWhenReady
```

`SteerSequenceFraction=0.25`가 steer2 action gate에서 `0.20`을 살짝 넘으면
같은 wrapper로 `0.15` 후보를 이어간다:

```powershell
powershell -ExecutionPolicy Bypass -File RCS-AC-v2\scripts\windows\run_stage2_mixed_candidate_cycle.ps1 `
  -ProjectRoot C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656 `
  -OutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015 `
  -CrossProbeOutputDir RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512\7f87656_mixed_target22_steer2_aux1_balpow05_iw5_ssf015_cross_probe `
  -TaskName RCSV2Stage2MixedT22Steer2Ssf015 `
  -SteerSequenceFraction 0.15 `
  -Device cuda `
  -MakeReplayIfMissing `
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
GPU 사용을 보장해야 하는 replay 학습/probe/clean eval 재개 전에는
`-RequireCuda`를 붙여 `torch.cuda.is_available()`와 CUDA device name을 먼저
확인한다. Stage 2 replay 학습, action probe, latent probe, candidate cycle
wrapper, official clean eval wrapper의 기본값은 모두 `Device=cuda`이며, 직접
Python training/probe entrypoint를 실행할 때도 기본 `--device cuda`로 GPU를
요구한다. GPU가 불안정하거나 PyTorch CUDA가 깨진 원인 분석 때만
명시적으로 `-Device cpu` 또는 `--device cpu`를 붙여 후퇴한다.
CPU 후퇴 run은 clean 성능 확보용 공식 후보가 아니라 진단용으로 기록한다.
2026-05-25 이후 Stage 2 replay 학습/probe/clean eval wrapper는 시작 시
`assert_stage2_cuda_available.ps1`를 호출한다. `Device=cuda`인데 PyTorch CUDA가
잡히지 않으면 run을 시작하지 않고 실패한다. 따라서
Windows 재부팅 뒤에는 별도 확인을 잊어도 wrapper 로그에
`STAGE2_CUDA_GUARD_AVAILABLE=True`, `STAGE2_CUDA_GUARD_DEVICE_NAME=...`,
`STAGE2_CUDA_GUARD_TENSOR_DEVICE=cuda:0`가 먼저 남아야 정상이다. `Device=cpu`는
명시적 진단 override로만 허용하며 로그에
`STAGE2_CUDA_GUARD_CPU_DIAGNOSTIC_OVERRIDE=True`가 남는다.
checkpoint action probe가 official clean eval용 config를 생성할 때도 probe
wrapper의 `-Device` 값을 `device:` 필드에 기록한다. 따라서 현재 표준 경로의
official clean 후보 config는 `device: cuda`를 명시적으로 포함해야 한다.
mask 후보는 YAML을 손으로 고치지 않고 `probe_stage2_checkpoint_actions.ps1`,
`probe_stage2_candidate_cross_replays.ps1`, 또는
`run_stage2_mixed_candidate_cycle.ps1`의 `-ForbiddenActionIds` 옵션으로
생성한다. 이렇게 해야 offline gate와 official clean eval에 같은
`carla.forbidden_action_ids`가 남고, 어떤 action mask를 썼는지 config artifact로
추적할 수 있다. 원격 SSH 문자열에서는 PowerShell 배열이 뭉칠 수 있으므로
`-ForbiddenActionIds "10,16,19"`처럼 쉼표 문자열로 넘긴다.
candidate cycle wrapper는 `-Seed`도 그대로 replay training까지 전달한다.
가장 가까운 후보가 action gate 근처에서 흔들릴 때는 새 OutputDir/TaskName과
함께 seed sweep을 먼저 돌려, 이미 실패한 artifact를 덮어쓰지 않고 비교한다.
late replay를 추가한 mixed 후보가 target22 또는 steer2 action distribution을
무너뜨리면 `-ComponentSampleWeights "target22=1,steer2=1,late068warm1=0.25"`
처럼 component별 training sample weight를 낮춰서 먼저 offline gate를 확인한다.
이 값은 checkpoint manifest의 `component_sample_weights`와
`sampler.episode_sample_probabilities`에 남아야 하며, `ready_for_clean_eval=true`
전에는 official clean eval로 승격하지 않는다.
component별 weight를 지정할 때는 `-MixedReplayDir`도 해당 component를 실제로
포함하는 manifest로 명시한다. 예를 들어 `corr048` 또는 `late045t24mth03`에
양수 weight를 주면서 기본 target22+steer2 mixed replay를 쓰면 guard가 실패해야
정상이다. 이런 실패는 모델/CUDA 문제가 아니라 run-argument mismatch로 기록한다.
late correction replay를 수집할 때 teacher action이 zero-throttle steering으로
무너지면 `collect_stage2_privileged_replay.ps1 -MinRecordedThrottle 0.3`을
사용한다. 기본값은 `0.0`이라 기존 동작은 보존된다. 2026-05-26 실험에서
`RecordAfterRouteProgress=0.45`, `TargetSpeedKmh=24`, `MinRecordedThrottle=0.3`,
`MinRecordedSteps=16` 조합은 4x16 step eligible replay를 만들었고, official
512 progress를 0.769까지 늘렸다. `0.56`과 `0.50`은 eligible episode 수가
부족했으므로 같은 조건을 반복하지 않는다.
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

Sub-agent 운영 메모:

```text
Windows가 켜져 있어도 sub-agent 생성은 Codex thread slot 제한 때문에 실패할 수 있다.
"agent thread limit reached"가 나오면 Windows 문제가 아니라 열린 agent가 너무 많은 상태다.
완료된 agent를 close한 뒤, 실패 분석/코드 확인처럼 서로 겹치지 않는 sidecar만 다시 띄운다.
CARLA official run 여부는 sub-agent 판단이 아니라 cross replay gate와 windows_carla summary로 결정한다.
```
