# Rounds 86-90 Deterministic PPO Probe and Sustained Late Recovery

## Intent

Rounds 83-85 showed PPO repeatedly offroading near route progress `0.333` even with very low policy std. Rounds 86-87 isolated whether this was actor/trunk drift or rollout sampling. Rounds 88-90 then returned to BC because the 2048 clean baseline was not reproducible under rerun.

This work does not add a runtime steering rule. All BC changes are privileged offline supervision through loss masks and trainable scope. The only PPO code change is a config-driven rollout action selection option for diagnosis.

## Code

`rcs_ac_first/scripts/train.py`

- Added `ppo.trainable_module_names` support so PPO can update only selected parameter prefixes.
- Added `ppo.deterministic_rollout` so PPO collection can use the actor mean through config rather than a hardcoded edit.
- Logged `deterministic_rollout` in `training_log.csv`.

## Round86

Config: `configs/ppo_round86_critic_head_warmup_from_longitudinal_row_mid_brake.yaml`

Change:

- Round82 checkpoint init.
- Ultra-low std retained.
- Trainable scope: `recurrent_actor_critic.critic_head` only.

Result:

- PPO training still produced one offroad at route progress `0.333333`.
- Because actor/trunk were frozen, this pointed away from actor mean drift and toward stochastic rollout/reset-margin issues.

## Round87

Config: `configs/ppo_round87_deterministic_critic_head_warmup_from_longitudinal_row_mid_brake.yaml`

Change:

- Same as Round86, plus `ppo.deterministic_rollout: true`.

PPO collection result:

- 2048 steps completed with all episodes ending by `max_steps`.
- collision/offroad/stuck: `0/0/0`.
- This confirmed that stochastic action sampling was enough to break the narrow safety margin.

Post-training eval:

- 2048 deterministic eval on the Round87 checkpoint still had one offroad.
- Since actor parameters were unchanged, this indicated the underlying Round82/Round78 clean BC margin was not robust enough.

## Round88 Probe

Re-ran existing baselines under the current CARLA/reset conditions:

- Round82 2048 rerun: offroad `1`.
- Round78 2048 rerun: offroad `1`.

Both failures happened around route progress `0.333`, with lane offset already beyond roughly `-2.4` before terminal offroad. This invalidated treating Round78/Round82 as stable long-horizon baselines.

## Round89

Config: `configs/bc_frame_rgb_perception_actor_features_v2_late_margin_speed_round89.yaml`

Change:

- Init from Round78.
- Train only `recurrent_actor_critic.actor_head.2`, output rows `[0, 1]`.
- Increased late recovery steer and longitudinal ceiling losses.

Result:

- Offline actor sign audit: PASS.
- 512 eval: clean PASS.
- 2048 eval: offroad `1`.

Failure analysis:

- Recovery became stronger, but was still discontinuous. At route `0.307-0.333`, the policy braked/steered strongly near lane offset `-1.0`, then relaxed, drifted to about `-2.2`, and only then applied strong steering/braking again.

## Round90

Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`

Change:

- Init from Round89.
- Kept actor-head output-row scope only.
- Expanded late recovery route window to `0.28-0.36`.
- Lowered late recovery mask threshold to lane offset `0.85` and heading error `6.0`.
- Increased longitudinal ceiling pressure in the same window.

Offline audit:

- `early_any` PASS.
- all axes rate: `1.0`
- min abs delta: `0.030364`
- max abs zero steer: `0.024747`

CARLA eval:

- 512: collision `0`, offroad `0`, severe early offroad `0`, stuck `0`, max route `0.435897`.
- 2048: collision `0`, offroad `0`, severe early offroad `0`, stuck `0`, max route `0.410256`, final route `0.384615`.

Round90 restores strict clean safety for 512/2048 in the current rerun conditions, but it is conservative and slow. PPO gate remains BLOCK because action jitter is high and visual-center correction rate is below threshold.

## Decision

Round90 is the current clean-safety baseline. It should not be reported as final RCS-AC completion because route progress is low and PPO/robustness are not complete.

Next step:

- Do not run broad stochastic PPO from this checkpoint.
- If PPO is attempted, start with deterministic rollout and a very narrow trainable scope, and treat it as a route-progress probe.
- Alternatively, improve BC gate metrics first by reducing action jitter and raising visual-center correction without losing Round90 2048 safety.
