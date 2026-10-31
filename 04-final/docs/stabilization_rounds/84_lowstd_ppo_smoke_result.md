# Round84 Low-Std PPO Smoke

## Intent

Round83 showed an offroad during PPO training from the Round82 BC seed, even with a small learning rate. Round84 tested whether reducing PPO exploration noise would prevent that safety regression.

## Config

Config: `configs/ppo_round84_lowstd_from_longitudinal_row_mid_brake.yaml`

Key changes from Round83:

- `ppo.learning_rate: 0.0000005`
- `ppo.loaded_log_std_by_dim: [-3.0, -3.0]`
- effective policy std about `0.0498` per action dimension
- total timesteps: `2048`

Seed checkpoint:

`checkpoints/rgb_perception_actor_features_v2_longitudinal_row_mid_brake_bc_round82/last.pt`

## Result

PPO training completed, but one episode still terminated offroad:

| episode | total_steps | length | done_reason | route_progress | collision | offroad | stuck |
| ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | 512 | 512 | max_steps | 0.435897 | False | False | False |
| 1 | 828 | 316 | offroad | 0.333333 | False | True | False |
| 2 | 1340 | 512 | max_steps | 0.384615 | False | False | False |
| 3 | 1852 | 512 | max_steps | 0.512821 | False | False | False |

Final update metrics:

- actor_loss: `-0.000439`
- critic_loss: `1.384828`
- policy_std_dim0: `0.049787`
- policy_std_dim1: `0.049787`

## Decision

Round84 is blocked and must not be promoted.

Lowering exploration std from about `0.135` to `0.050` did not eliminate the same late route offroad mode around route `0.333`. The issue is therefore not only excessive stochastic exploration. PPO rollout/update is still operating on a policy whose late recovery margin is thin.

Next direction:

- keep Round82 as the BC safety candidate
- do not promote Round83 or Round84 PPO checkpoints
- before more actor PPO, test either:
  - a critic-only/value warmup if supported, or
  - a stricter minimum std near the configured `min_std_by_dim`, plus much shorter PPO smoke, or
  - a BC-side late recovery margin improvement that does not rely on heavy braking
