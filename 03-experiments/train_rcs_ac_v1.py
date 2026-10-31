"""Train RCS-AC v1 recurrent PPO."""

from __future__ import annotations

import argparse

from rcs_ac_v1.config import load_experiment_config
from rcs_ac_v1.trainer import RecurrentPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCS-AC v1 recurrent PPO.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    trainer = RecurrentPPOTrainer(config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
