# RCS-AC Clean Baseline: Round90

Status: **fixed clean-safe RCS-AC baseline**

Fixed on: 2026-05-02

## Baseline Identity

- Baseline name: `rcs_ac_clean_baseline_round90`
- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Training source: Round90 BC from Round89/Round90 stabilization branch
- Model family: RCS-AC Phase 1-3 architecture with RGB perception actor features and scalar actor features
- Runtime assist status:
  - `env.lateral_recovery.enabled: false`
  - no runtime lane-offset steering override
  - no handcoded steering rule
  - target-speed longitudinal wrapper and action smoothing are the shared environment/action interface, not an RCS-AC-only claim

## Fixed Evidence

Offline actor sign audit:

- Source: `round90_results/audit_early_any_summary.json`
- pass status: `PASS`
- all axes rate: `1.0`
- min abs delta: `0.030364`
- max abs zero steer: `0.024747`
- sample pass rate: `1.0`

512-step clean eval:

- Source: `round90_results/eval_512_summary.json`
- pass status: `PASS`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- max/final route progress: `0.435897`
- mean action jitter L1: `0.132107`
- steer jitter: `0.085229`
- longitudinal jitter: `0.046877`
- PPO gate: `BLOCK`

2048-step clean eval:

- Source: `round90_results/eval_2048_summary.json`
- pass status: `PASS`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- max route progress: `0.410256`
- final route progress: `0.384615`
- mean action jitter L1: `0.146305`
- route-offset corrective sign rate: `0.815951`
- visual-center correction rate: `0.490798`
- PPO gate: `BLOCK`

2048-step clean confirmation:

- Source: `round101_results/eval_2048_summary.json`
- pass status: `PASS`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- max route progress: `0.384615`
- final route progress: `0.358974`
- mean action jitter L1: `0.138907`
- route-offset corrective sign rate: `0.833333`
- visual-center correction rate: `0.462963`
- PPO gate: `BLOCK`

## Interpretation

Round90 is fixed as the RCS-AC clean-safe baseline because it repeatedly satisfies the clean safety criteria:

- no collision
- no offroad
- no severe early offroad
- no stuck
- 512 and 2048 clean eval complete with finite in-range actions

Round90 is **not** the final completed RCS-AC result:

- route progress is conservative
- PPO gate remains blocked by action jitter and visual-center correction rate
- robustness comparisons against PPO/SAC are not complete

## Baseline Lock Rule

For future stabilization or comparison work:

1. Do not replace this baseline unless a candidate passes offline audit, 512 clean eval, and 2048 clean eval with zero collision/offroad/severe early offroad/stuck.
2. Any replacement candidate must be compared against this manifest and documented with matching result files.
3. Rounds 93-100 are blocked probes and must not be used as the clean baseline.
4. PPO fine-tuned candidates must be reported separately until they preserve this baseline's clean safety.
5. Weather, corruption, and sensor-dropout robustness experiments may use Round90 as the fixed RCS-AC clean checkpoint.

## Next Comparison Use

Use this baseline when preparing fair comparisons:

- RCS-AC clean baseline: Round90
- PPO baseline: to be trained/evaluated under the same route, weather, sensor, wrapper, and eval protocol
- SAC baseline: to be implemented or selected, then trained/evaluated under the same protocol

Report both absolute performance and degradation from each algorithm's own clean baseline.
