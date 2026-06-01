# PPO Clean Baseline Implementation

Status: **configured, not yet accepted**

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml`

Reference plan: `docs/baselines/ppo_baseline_preparation_plan.md`

Reference RCS-AC clean baseline: `docs/baselines/rcs_ac_clean_baseline_round90.md`

## Baseline Identity

This config implements the first PPO clean baseline candidate:

`ppo_same_env_sensor_action_baseline`

The current repo trains it through `RCSACFirstModel`, so this is documented as a same-observation architecture PPO baseline, not as a pure vanilla PPO network. RCS-AC Phase 2/3 mechanisms and auxiliary policy features are disabled.

## Shared Conditions With Round90

- Town: `Town04`
- Seed: `7`
- Max episode steps: `512`
- Policy observations: `rgb`, `lidar`, `ego`, `route`
- Action dimension: `2`
- Route representation: `target_xy_plus_two_distance_heading_plus_third_heading_plus_turn_delta_plus_maneuver_hint_v1`
- Traffic: disabled
- Clean corruption state: disabled
- Lateral recovery assist: disabled
- Target-speed longitudinal wrapper: matched to Round90
- Action smoothing: matched to Round90

## Disabled RCS-AC-Specific Features

- RGB perception auxiliary head disabled
- RGB perception actor features disabled
- Scalar actor features disabled
- Reliability module disabled
- Risk context disabled
- Shapley disabled
- Dropout consistency disabled
- Auxiliary, reliability, risk, Shapley, and dropout-consistency losses set to zero

The environment can still expose the semantic auxiliary camera for diagnostics, but the PPO policy contract records that the baseline does not consume auxiliary inputs as policy inputs.

## PPO Training Settings

Initial smoke budget:

- `total_timesteps: 2048`
- `rollout_steps: 128`
- `learning_rate: 0.0001`
- `gamma: 0.99`
- `gae_lambda: 0.95`
- `clip_coef: 0.2`
- `update_epochs: 1`
- `minibatch_size: 64`
- `max_grad_norm: 0.5`

These are intentionally conservative smoke settings. Longer PPO candidates should use the same config and override only the training budget or permitted PPO knobs documented in the preparation plan.

## Acceptance State

This implementation is not a fixed PPO clean baseline yet. It becomes one only after:

- 512 clean eval: collision `0`, offroad `0`, severe early offroad `0`, stuck `0`
- 2048 clean eval: collision `0`, offroad `0`, severe early offroad `0`, stuck `0`
- max route progress is positive and preferably at least `0.30`
- a final accepted manifest is added under `docs/baselines/`

Until then, this file is an implementation record for the PPO baseline candidate.
