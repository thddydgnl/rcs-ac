"""Training entrypoint placeholder for future RL experiments on the strategy environment."""

from __future__ import annotations

import logging

from config import EnvConfig
from carla_strategy_env.envs import CarlaF1StrategyEnv


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    env = CarlaF1StrategyEnv(EnvConfig())
    try:
        observation, info = env.reset()
        logging.info("Environment ready for RL training. Observation keys: %s", list(observation))
        logging.info("Initial episode info: %s", info)
        logging.info("TODO: Plug SB3/RLlib/SVMM learner into this entrypoint.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
