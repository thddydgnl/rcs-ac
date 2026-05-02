# Baselines

This directory records fixed experiment baselines that future stabilization and comparison rounds must reference.

## Current Fixed Baseline

- RCS-AC clean-safe baseline: `rcs_ac_clean_baseline_round90`
- Manifest: `docs/baselines/rcs_ac_clean_baseline_round90.md`

Round90 is fixed as the current RCS-AC clean baseline for weather, corruption, and sensor-dropout comparison work. It should not be replaced by later probes unless they pass the same offline, 512-step, and 2048-step clean safety checks and are explicitly documented as a baseline replacement.

## Preparation Guides

- PPO clean baseline preparation: `docs/baselines/ppo_baseline_preparation_plan.md`
