# PPO Matched-r7j Goal Prompt - Fast Validation Workflow

## Goal

PPO matched-r7j baseline을 RCS-AC v1.1 r7j와 논문 비교 가능한 clean matched baseline으로 안정화한다. PPO는 r7j와 동일한 sensor setup, observation preprocessing, BEV spec, ego/route scalar contract, waypoint 4x2 + target_speed_kmh action contract, shared pure-pursuit controller, target-speed cap, curvature speed cap, steer lowpass/rate limit, target_y_lowpass_tau_s, reward, termination, eval protocol, logging schema를 유지한다.

PPO는 RCS-AC 전용 reliability, source-level reliability fusion, Shapley, risk context module을 사용하지 않는다. PPO-only steering override, privileged inference input, spawn/seed/Town04-specific rule도 사용하지 않는다.

## Current Candidate

Use the current best candidate as the clean baseline candidate:

```text
label: imitation-initialized PPO matched-r7j
config: PPO/configs/matched_r7j/ppo_matched_r7j_train_seed7_imitation_teacher.yaml
checkpoint: PPO/results/windows_pullback/ppo_matched_r7j_seed7_imitation_teacher_train/last.pt
```

Current achieved gates:

```text
primary 512 9/9 safety pass
primary 2048 9/9 safety pass and 9/9 goal
generalization 512 36/36 safety pass
```

## Fast Validation Rule

Do not rerun expensive completed gates by default. If a code/config/checkpoint change does not affect policy inference, controller, adapter, reward, termination, or eval logging, preserve existing gate results and continue from the next pending gate.

Completed gates should only be rerun when one of these changes:

```text
policy checkpoint
PPO model architecture
observation adapter
runtime config
pure-pursuit controller
action wrapper
reward or termination
CARLA/shared env behavior
eval logging schema
```

This is the default execution policy. The purpose is not to give PPO any easier
condition than RCS-AC r7j, but to avoid rerunning evidence that is already valid
under the unchanged matched-r7j contract.

Current clean evidence must be treated as reusable unless invalidated:

```text
primary 512: 9/9 safety pass
primary 2048: 9/9 safety pass, 9/9 route goal
generalization 512: 36/36 safety pass
```

Therefore, after this point, do not restart from primary gates unless the
checkpoint, inference path, controller/action wrapper, runtime config, CARLA env
behavior, or logging schema changed.

## Reduced Round Workflow

When the user says `/goal 진행` or asks to continue stabilization:

1. Check current process state and stop stale Windows eval/train processes before starting new runs.
2. Confirm the latest completed gate from PPO docs/results.
3. Prefer the next pending gate, not a full rerun.
4. If a quick gate fails, stop and analyze that failed start only. Do not run the full gate.
5. Use quick gates before full gates:
   - hidden quick: seed2 6-start only
   - if pass, hidden full: seed2 + seed5 + seed9, 18-start
   - if pass, generalization 2048 subset: 6-start
6. Use full expensive gates only for final candidate confirmation.

## CARLA Time Saving

Default to fewer readiness checks:

```text
For quick gates:
- do not use -EnsureReadyBetweenRuns unless CARLA just crashed
- use one ensure_carla_ready.ps1 before the batch

For full final gates:
- use -EnsureReadyBetweenRuns only when crash risk is high
- otherwise check readiness every group/seed, not every start
```

If CARLA crash/timeout occurs, record it separately from model failure, restart CARLA, and rerun only the failed start or failed seed group.

If a model driving failure occurs, do not hide it by deleting the start or
changing the start set. Record the exact seed/spawn/run directory and analyze
the local step log before changing training or evaluation settings.

## RCS-AC r7j Equivalence Meaning

The goal is not to make PPO use RCS-AC-specific modules. The goal is to make PPO
reach RCS-AC r7j-level clean driving under the same sensor/action/controller/eval
conditions:

```text
same CARLA/Town04/clean weather/fixed-delta conditions
same route/start/seed sets
same sensor and observation preprocessing contract
same waypoint+target_speed output contract
same shared pure-pursuit direct_3d action path
same reward/termination/eval/logging protocol
no RCS-AC reliability/fusion/Shapley/risk module in PPO
no PPO-only runtime steering override or privileged inference input
```

When this document says "same as RCS-AC r7j", it means comparable clean
performance under the matched contract, not identical algorithm internals.

## Pending Gates

The next gates after the current candidate state are:

1. Hidden 512 quick gate: seed2, spawn indices 24, 163, 177, 190, 230, 364.
2. Hidden 512 full gate: seed2/seed5/seed9, total 18 starts.
3. Generalization 2048 subset: 6 starts.
4. Final PPO clean matched-r7j baseline manifest.

## Completion Criteria

The goal is complete only when:

```text
contract audit PASS
primary 512 9/9 safety pass
primary 2048 9/9 safety pass and route goal
generalization 512 36/36 safety pass
hidden 512 18/18 safety pass
generalization 2048 subset 6/6 safety pass
validity/fallback clean
speed/jitter within target or explicitly documented as candidate-only
manifest written under PPO/docs
results stored under PPO only
git commit/push complete
```

If hidden/generalization 2048 is pending, the checkpoint is a strong clean candidate but not a final paper-ready baseline.
