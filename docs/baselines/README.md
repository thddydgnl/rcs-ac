# Baselines

This directory records fixed experiment baselines that future stabilization and comparison rounds must reference.

## Current Fixed Baseline

- RCS-AC clean-safe baseline: `rcs_ac_clean_baseline_round90`
- Manifest: `docs/baselines/rcs_ac_clean_baseline_round90.md`

Round90 is fixed as the current RCS-AC clean baseline for weather, corruption, and sensor-dropout comparison work. It should not be replaced by later probes unless they pass the same offline, 512-step, and 2048-step clean safety checks and are explicitly documented as a baseline replacement.

## Preparation Guides

- PPO clean baseline preparation: `docs/baselines/ppo_baseline_preparation_plan.md`
- Estimated scalar actor features plan: `docs/baselines/estimated_scalar_actor_features_plan.md`
- Comparison training budget and protocol: `docs/baselines/comparison_training_budget_and_protocol.md`

## PPO Baseline Candidates

- PPO same-env/sensor/action seed7 config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml`
- PPO matched estimated-scalar seed7 config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_matched_estimated_scalar.yaml`
- PPO matched estimated-scalar seed9 config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_matched_estimated_scalar_rollout512_mb256.yaml`
- Implementation record: `docs/baselines/ppo_clean_same_env_sensor_action_seed7_implementation.md`
- PPO clean baseline rounds: `docs/baselines/ppo_clean_baseline_rounds.md`
