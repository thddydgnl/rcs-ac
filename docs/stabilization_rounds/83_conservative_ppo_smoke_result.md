# Round83 Conservative PPO Smoke

## Intent

Round82 restored 512/2048 clean safety and improved max route progress, but the configured PPO gate remained BLOCK because of combined action jitter and visual-center correction rate. Round83 tested whether a very conservative PPO update could start from Round82 without safety regression.

This round was a PPO smoke, not a promotion. The seed checkpoint was:

`checkpoints/rgb_perception_actor_features_v2_longitudinal_row_mid_brake_bc_round82/last.pt`

## Config

Config: `configs/ppo_round83_from_longitudinal_row_mid_brake.yaml`

Key changes from Round82:

- `ppo.learning_rate: 0.000001`
- `ppo.clip_coef: 0.03`
- total timesteps: `2048`
- loaded log std from config remained `[-2.0, -2.0]`

## Result

PPO training completed 2048 steps, but one training episode terminated offroad:

| episode | total_steps | length | done_reason | route_progress | collision | offroad | stuck |
| ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | 512 | 512 | max_steps | 0.384615 | False | False | False |
| 1 | 940 | 428 | offroad | 0.333333 | False | True | False |
| 2 | 1452 | 512 | max_steps | 0.435897 | False | False | False |
| 3 | 1964 | 512 | max_steps | 0.461538 | False | False | False |

Final update metrics:

- actor_loss: `0.000063`
- critic_loss: `0.578963`
- policy_std_dim0: `0.135335`
- policy_std_dim1: `0.135335`

## Decision

Round83 is blocked and must not be promoted.

The failure is a PPO safety regression relative to Round82:

- Round82 BC 2048: offroad `0`
- Round83 PPO training: offroad `1`

The most likely immediate issue is exploration noise, not update magnitude. The PPO actor loaded std is about `0.135` per action dimension, which is large compared with the narrow late-recovery safety margin. Even with learning rate `1e-6` and clip `0.03`, stochastic PPO rollout can leave the road before the policy has a chance to improve.

Next direction:

- keep Round82 as the BC seed
- do not evaluate or promote Round83 checkpoint as a candidate
- try PPO only with lower loaded log std, for example `[-3.0, -3.0]` or lower, and an even smaller learning rate
- require PPO training itself to have collision/offroad/stuck all zero before running post-PPO 512/2048 eval
