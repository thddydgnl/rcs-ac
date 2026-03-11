"""Sample script to smoke-test the CARLA F1 strategy environment with random actions."""

from __future__ import annotations

import argparse
import logging
import random

from config import EnvConfig
from carla_strategy_env.envs import CarlaF1StrategyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test for the CARLA F1 strategy environment.")
    parser.add_argument("--laps", type=int, default=3, help="Number of laps to configure for the episode.")
    parser.add_argument("--steps", type=int, default=20, help="Maximum number of strategy steps to run.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    config = EnvConfig()
    config.race.total_laps = args.laps
    env = CarlaF1StrategyEnv(config)

    try:
        observation, info = env.reset()
        logging.info("Initial observation keys: %s", observation.keys())
        logging.info("Initial strategy info: %s", info)

        for step in range(args.steps):
            _, reward, terminated, truncated, info = env.step(random.randrange(env.action_space.n))
            logging.info(
                "step=%s reward=%.3f lap=%s remaining=%s tire=%s pit_count=%s",
                step,
                reward,
                info["current_lap"],
                info["laps_remaining"],
                info["tire_compound"],
                info["pit_count"],
            )
            if terminated or truncated:
                logging.info("Episode ended: terminated=%s truncated=%s", terminated, truncated)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
