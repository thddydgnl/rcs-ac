# Comparison Training Budget And Protocol

Status: **comparison execution guide**

Purpose: define how to move from the clean-safe RCS-AC Round90/v6 stabilization
state to a fair PPO/SAC/RCS-AC robustness comparison.

The intended final claim is not just clean driving. The comparison claim is:

> RCS-AC preserves clean-driving safety and retains more route progress under
> weather, corruption, and sensor-dropout shifts than PPO and SAC trained under
> the same external environment, sensor, route, and action-wrapper conditions.

## Current RCS-AC Starting Point

Two RCS-AC clean references exist:

1. `Round90 oracle scalar`
   - Manifest: `docs/baselines/rcs_ac_clean_baseline_round90.md`
   - Uses privileged scalar actor features at inference.
   - Clean 512/2048 safety: PASS.

2. `Round90 estimated scalar v6`
   - Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_estimated_scalar_v6.yaml`
   - Checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v6/last.pt`
   - Evidence: `round90_results/estimated_scalar_v6_carla_eval/README.md`
   - Inference replaces privileged `lane_offset` and `heading_error` with RGB/route estimated scalars.
   - Clean 512/2048 safety: PASS.

For robustness comparison, prefer `Round90 estimated scalar v6` as the RCS-AC
candidate because it avoids privileged scalar inference. Keep Round90 oracle
scalar as the historical control/fallback.

## Training Budget Recommendation

Do not use Round90 length as the final comparison budget. Round90 is a clean
stabilization checkpoint, not a saturated algorithm-comparison run.

Use three budget tiers.

### Phase A: Viability

Goal: prove each algorithm can train/evaluate in clean driving without obvious
collapse.

Recommended budget:

| Algorithm | Seeds | Env steps per seed | Purpose |
| --- | ---: | ---: | --- |
| RCS-AC | 1-3 | 50k | PPO fine-tune viability from clean-safe checkpoint |
| PPO | 1-3 | 50k | Clean baseline viability |
| SAC | 1-3 | 50k | Clean baseline viability |

Gate:

- 512 clean eval PASS.
- collision/offroad/severe early offroad/stuck all zero.
- no simulator or logging contamination.
- no runtime lateral recovery assist.

### Phase B: Main Comparison

Goal: generate defensible comparison checkpoints.

Recommended budget:

| Algorithm | Seeds | Env steps per seed |
| --- | ---: | ---: |
| RCS-AC | 3 | 300k |
| PPO | 3 | 300k |
| SAC | 3 | 300k |

This is the first serious comparison budget. It is long enough to avoid
accidentally comparing a stabilized BC checkpoint against undertrained PPO/SAC,
but still practical enough to iterate on CARLA failures.

### Phase C: Paper-Strength Extension

Goal: strengthen the claim if Phase B shows a real robustness gap.

Recommended budget:

| Algorithm | Seeds | Env steps per seed |
| --- | ---: | ---: |
| RCS-AC | 3 | 1M |
| PPO | 3 | 1M |
| SAC | 3 | 1M |

Only run this after Phase B confirms the setup is stable and the result is worth
scaling.

## Fairness Rules

All compared algorithms must share:

- Town04 clean route protocol unless a new shared protocol is explicitly defined.
- same route length, route spacing, maneuver schedule, episode length, and done rules.
- same RGB/LiDAR/ego/route observation availability.
- same 2D action contract.
- same target-speed longitudinal wrapper.
- same action smoothing.
- same traffic setting.
- same clean/weather/corruption evaluation matrix.
- same checkpoint selection rule.

Not allowed:

- RCS-AC-only runtime steering override.
- PPO/SAC-only handicaps.
- privileged simulator scalar input at inference.
- selecting checkpoints on robustness test conditions.
- changing route/environment settings for one algorithm without re-running all algorithms.

## Checkpoint Selection Rule

Use clean validation eval only.

For each algorithm and seed:

1. Run periodic clean eval at fixed intervals.
2. Discard checkpoints with any collision/offroad/severe early offroad/stuck.
3. Among remaining checkpoints, select the checkpoint with best clean score:

```text
clean_score = max_route_progress
            + 0.25 * mean_route_progress
            - 0.10 * mean_action_jitter_l1
```

If scores are close, prefer the checkpoint with lower safety risk and lower
action jitter.

Do not use weather, corruption, or sensor-dropout results for checkpoint
selection. Those are final test conditions.

## Clean Gates Before Robustness

Before any robustness comparison, each algorithm checkpoint must pass:

| Gate | Required |
| --- | --- |
| 512 clean eval | collision/offroad/severe early offroad/stuck all zero |
| 2048 clean eval | collision/offroad/severe early offroad/stuck all zero |
| completed steps | equals requested eval steps |
| route progress | positive and not clearly collapsed |
| action range | finite and in-range |
| runtime assist report | raw policy and wrapper/guard behavior separated |

If a checkpoint fails collision/offroad/severe early offroad/stuck, it cannot be
used for robustness comparison.

## Robustness Eval Matrix

Minimum matrix after clean gates pass:

| Condition | Required variants |
| --- | --- |
| clean | default |
| weather | rain, fog |
| RGB corruption | dropout or blackout, noise/blur |
| LiDAR corruption | dropout/sparsity |
| combined corruption | RGB + LiDAR mild/medium |

Recommended first matrix:

```text
Algorithms: RCS-AC estimated scalar v6, PPO, SAC
Conditions: clean, rain, fog, RGB dropout, LiDAR dropout, RGB+LiDAR corruption
Seeds: 3
Eval horizon: 2048
Total: 3 algorithms * 6 conditions * 3 seeds = 54 eval runs
```

Use 512-step eval only as a preflight. Use 2048-step eval for reported results.

## Metrics To Report

Safety:

- collision count/rate
- offroad rate
- severe early offroad count
- stuck rate
- completed steps

Progress:

- max route progress
- mean route progress
- final route progress

Behavior:

- mean action jitter L1
- steer jitter
- longitudinal jitter
- corrective sign rate when available
- visual-center correction rate when available

Robustness:

```text
retention(condition) = metric(condition) / metric(clean)
```

Use retention mainly for progress metrics. Safety failures should be reported as
absolute counts/rates, not hidden inside a score.

## Recommended Immediate Next Actions

1. Treat `Round90 estimated scalar v6` as the RCS-AC clean comparison candidate.
2. Keep `Round90 oracle scalar` as a historical control/fallback, not the main
   non-privileged comparison checkpoint.
3. Run PPO and SAC Phase A viability under the same external environment and
   action wrapper.
4. If PPO/SAC Phase A pass, run Phase B `300k x 3 seeds` for all three
   algorithms.
5. Freeze clean-selected checkpoints.
6. Run the 2048-step robustness matrix.

## Current Completion State

Done:

- RCS-AC Round90 oracle scalar 512/2048 clean safety PASS.
- RCS-AC estimated scalar v6 512/2048 clean safety PASS.
- v6 removes privileged scalar inference by estimating lane/heading from
  RGB/route features trained with privileged offline supervision.

Not done:

- PPO Phase A clean-safe baseline accepted.
- SAC baseline implemented/accepted.
- equal-budget RCS-AC/PPO/SAC Phase B training.
- robustness matrix.
- statistical summary across seeds.

The goal is therefore not complete yet. The next concrete blocker is preparing
PPO/SAC clean-safe baselines under the shared protocol.
