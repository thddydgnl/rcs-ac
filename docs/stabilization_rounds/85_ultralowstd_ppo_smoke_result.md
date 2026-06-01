# Round85 Ultra-Low-Std PPO Smoke

## Intent

Round84 still produced a PPO training offroad with action std about `0.05`. Round85 tested whether pushing std close to deterministic would keep PPO rollout safe.

## Config

Config: `configs/ppo_round85_ultralowstd_from_longitudinal_row_mid_brake.yaml`

Key changes from Round84:

- `model.min_std_by_dim: [0.005, 0.005]`
- `ppo.loaded_log_std_by_dim: [-5.0, -5.0]`
- effective policy std about `0.00674`
- `ppo.learning_rate: 0.0000002`

Seed checkpoint:

`checkpoints/rgb_perception_actor_features_v2_longitudinal_row_mid_brake_bc_round82/last.pt`

## Result

PPO training completed, but one episode still terminated offroad:

| episode | total_steps | length | done_reason | route_progress | collision | offroad | stuck |
| ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | 512 | 512 | max_steps | 0.564103 | False | False | False |
| 1 | 797 | 285 | offroad | 0.333333 | False | True | False |
| 2 | 1309 | 512 | max_steps | 0.512821 | False | False | False |
| 3 | 1821 | 512 | max_steps | 0.564103 | False | False | False |

Final update metrics:

- actor_loss: `0.001520`
- critic_loss: `1.187313`
- policy_std_dim0: `0.006738`
- policy_std_dim1: `0.006738`

## Decision

Round85 is blocked and must not be promoted.

The first episode matched the Round82 clean route range, but after updates began the next episode went offroad at route `0.333333`. Since exploration std was already only about `0.0067`, this is not primarily an exploration-noise problem.

Most likely issue:

- PPO updates all model parameters.
- Critic/value loss can update shared encoder/fusion/recurrent features.
- Those shared changes alter actor means even when the actor loss and learning rate are tiny.
- The Round82 late recovery margin is thin, so small actor-mean drift is enough to reintroduce offroad.

Next direction:

- add PPO trainable-scope support
- run a critic-head-only warmup from Round82
- require PPO training offroad/collision/stuck all zero before any actor PPO
