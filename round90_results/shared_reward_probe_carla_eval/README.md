# Round90 Shared Reward Shaping Probe

Date: 2026-05-04

Status: **clean safety preserved in eval**

## Purpose

This probe checks whether the proposed shared reward contract damages the fixed RCS-AC Round90 clean baseline before applying the same contract to PPO matched-baseline training.

The checkpoint is unchanged. The only intended config change is adding the shared reward shaping block to the Round90 eval config.

## Identity

- Base clean baseline: `rcs_ac_clean_baseline_round90`
- Base config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Probe config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_shared_reward_probe.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Windows eval worktree: `C:\Users\thddy\Documents\code\RCS-AC-first-shared-reward-eval`
- Runtime assist status:
  - `env.lateral_recovery.enabled: false`
  - no new steering rule
  - no lane-offset runtime steering override
  - same target-speed/action wrapper as Round90

## Reward Contract Candidate

```yaml
reward:
  lane_offset_penalty_weight: 0.04
  heading_error_penalty_weight: 0.005
  heading_recovery_reward_weight: 0.003
  turn_speed_penalty_weight: 0.06
  turn_speed_delta_threshold_rad: 0.10
  turn_speed_free_kmh: 10.0
  turn_speed_use_junction_only: true
  control_cost_scale: 0.12
```

The config was validated on Windows against `shared_core.env.reward.RewardConfig`; all keys were accepted.

## Eval Results

512-step clean eval:

- Output: `windows_outputs/rgb_perception_actor_features_v2_round90_shared_reward_eval_512_20260504T014903/summary.json`
- pass status: `PASS`
- completed steps: `512`
- done reasons: `{"max_steps": 1}`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- max/final route progress: `0.384615/0.384615`
- mean route progress: `0.192458`
- mean action jitter L1: `0.133210`
- steer jitter: `0.088925`
- longitudinal jitter: `0.044285`

2048-step clean eval:

- Output: `windows_outputs/rgb_perception_actor_features_v2_round90_shared_reward_eval_2048_retry1_20260504T015412/summary.json`
- pass status: `PASS`
- completed steps: `2048`
- done reasons: `{"max_steps": 4}`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- max/final route progress: `0.435897/0.384615`
- mean route progress: `0.192558`
- mean action jitter L1: `0.151773`
- steer jitter: `0.083238`
- longitudinal jitter: `0.068535`

One first 2048 attempt failed before rollout because CARLA timed out while loading the world. That was classified as an infrastructure failure, not a model driving failure. CARLA was force-restarted and the retry completed successfully.

## Round90 Comparison

| Metric | Round90 512 | Shaped 512 | Round90 2048 | Shaped 2048 |
|---|---:|---:|---:|---:|
| collision/offroad/severe early/stuck | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| max route progress | 0.435897 | 0.384615 | 0.410256 | 0.435897 |
| final route progress | 0.435897 | 0.384615 | 0.384615 | 0.384615 |
| mean route progress | 0.213792 | 0.192458 | 0.183681 | 0.192558 |
| mean action jitter L1 | 0.132107 | 0.133210 | 0.146305 | 0.151773 |
| steer jitter | 0.085229 | 0.088925 | 0.084437 | 0.083238 |
| longitudinal jitter | 0.046877 | 0.044285 | 0.061867 | 0.068535 |

Interpretation:

- Clean safety is preserved at both 512 and 2048 steps.
- Route progress does not collapse. The 512 shaped run is slightly lower than the original single-episode Round90 eval, while the 2048 shaped run is slightly higher on max/mean route and equal on final route.
- Action jitter is close to Round90. 2048 action jitter is modestly higher (`+0.005468` L1), steer jitter is slightly lower, and longitudinal jitter is higher (`+0.006667`).
- The probe should remain separate from the fixed Round90 clean baseline. Do not merge these results into the original Round90 baseline identity.

## Reward Breakdown Observed

From shaped `env_step_log.csv`:

- 2048 mean reward: `0.043958`
- mean lane offset penalty: `0.009672`
- mean heading error penalty: `0.015147`
- mean heading recovery reward: `0.000629`
- mean turn speed penalty: `0.008982`
- mean control cost: `0.000526`
- target speed mean/min/max: `9.096/6.000/19.633` km/h
- left/turn-heavy rows: `1719`
- left/turn-heavy target speed mean: `8.669` km/h
- left/turn-heavy brake mean: `0.016867`
- left/turn-heavy turn-speed penalty mean: `0.010225`

There is no evidence that the turn-speed shaping made the policy over-brake in the severe-left/turn-heavy portions of this eval. The actual applied brake mean stayed low in the environment step log.

## Eval-Only Effect

`evaluate_bc.py` runs the fixed checkpoint policy and logs the environment reward. The reward config is passed into `CarlaEnv` through `train._build_env()`, so the reward breakdown is real, but deterministic BC evaluation does not consume reward to choose actions or update weights.

Therefore this probe is a clean-safety and reward-contract wiring check, not proof that shaping improves learning. To test learning impact, run a short RCS-AC PPO smoke/fine-tuning pass with this reward config and verify that shaped rewards enter rollout returns/advantages before launching a long training round.

## Decision

Shared reward shaping candidate is **safe to carry forward as a common RCS-AC/PPO reward contract candidate**, with two constraints:

1. Keep the original Round90 baseline fixed and separately reported.
2. Before using this in PPO matched-baseline experiments, run a short PPO/RCS-AC training smoke to confirm the reward is actually driving advantages and does not introduce safety regression.
