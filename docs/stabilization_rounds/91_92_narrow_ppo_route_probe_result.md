# Rounds 91-92 Narrow PPO Route Probe

## Intent

Round90 restored 512/2048 clean safety under current rerun conditions, but route progress remained low and PPO gate stayed blocked by action jitter and visual-center correction. Rounds 91-92 tested whether a very narrow PPO update could improve route progress without regressing safety.

No runtime steering rule or lane-offset override was added.

## Round91

Config: `configs/ppo_round91_deterministic_actor_head_from_sustain_late_recovery.yaml`

Setup:

- Init checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- Rollout: `ppo.deterministic_rollout: true`
- Trainable scope:
  - `recurrent_actor_critic.actor_head.2`
  - `recurrent_actor_critic.critic_head`
- Learning rate: `1e-7`
- Loaded log std: `[-5.0, -5.0]`

Result:

- 2048 PPO collection safety: collision `0`, offroad `0`, stuck `0`.
- Episode route progress: `0.435897`, `0.384615`, `0.384615`, `0.384615`.
- Actor loss stayed effectively zero.

Interpretation:

Deterministic PPO collection is safe here, but it is not an effective actor-improvement mechanism. Because the collected action equals the policy mean, the normal log-prob gradient with respect to the mean is nearly zero at that action. It is useful for safety diagnosis or critic warmup, not for route-progress improvement.

## Round92

Config: `configs/ppo_round92_ultralow_stochastic_actor_head_from_sustain_late_recovery.yaml`

Setup:

- Same Round90 init checkpoint.
- Rollout: `ppo.deterministic_rollout: false`
- Ultra-low std retained with loaded log std `[-5.0, -5.0]`.
- Same narrow actor-head and critic-head trainable scope.

Result:

- 2048 PPO collection safety: collision `0`, offroad `0`, stuck `0`.
- Episode route progress: `0.435897`, `0.410256`, `0.333333`, `0.358974`.
- Actor gradients were nonzero but route progress did not improve over Round90.

## Decision

PPO no longer immediately breaks safety when starting from Round90 and using ultra-low std with narrow trainable scope. However, it also does not yet improve route progress.

Next work should not escalate PPO scope or std. The more useful path is to improve the BC policy/eval gate first:

- reduce action jitter,
- raise visual-center correction rate,
- preserve Round90 2048 clean safety,
- then retry a narrow PPO route probe.
