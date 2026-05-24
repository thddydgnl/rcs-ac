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
probe_stage2_checkpoint_actions.ps1
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
run_stage2_clean_eval.ps1의 `-RunOutputDir` 옵션은 validated Windows run의
summary.json, metrics.csv, 실제 입력 config를 지정한 증거 폴더로 복사한다.
이 옵션은 artifact mirror 편의 기능일 뿐 PASS/FAIL을 바꾸거나 summary 필드를
수정하지 않는다. stale artifact 착시를 막기 위해 runner는 eval 시작 직전에
canonical summary.json/metrics.csv만 지운 뒤 새 결과만 검증/복사한다.
probe_stage2_checkpoint_actions.ps1는 checkpoint manifest를 base clean config에
적용한 runtime config를 만들고, replay observation에서 deterministic action
분포를 점검한다. reverse_count=0, throttle 있는 action, 이전 stuck 실패의
action_id=8 단일 반복 여부를 clean rerun 전에 확인한다. `-MaxStepsPerFile 128`
로 전체 4x128 replay를 확인하고, `recommended_for_clean_rerun=true`는
reverse 0, single-action collapse 없음, nonzero-throttle fraction 0.5 이상,
action_id=8 fraction 0.05 이하일 때만 나온다.
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
