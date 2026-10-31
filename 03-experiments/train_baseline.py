"""Entry point for multimodal PPO baselines."""

from __future__ import annotations

import argparse

from ppo_baselines.config import load_experiment_config
from ppo_baselines.ppo.trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO baseline for RCSToyNavEnv.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    trainer = PPOTrainer(config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
