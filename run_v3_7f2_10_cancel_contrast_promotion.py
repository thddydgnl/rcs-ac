"""Run Version 3.7f2.10 common-mode cancel + contrastive vision shift promotion check."""

from __future__ import annotations

import argparse

from run_v3_7f2_10_cancel_contrast_pilot import run_v3_7f2_10_experiment
from v3_shapley_helpers import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.7f2.10 common-mode cancel + contrastive vision shift promotion check.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_7f2_10_cancel_contrast_promotion.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_v3_7f2_10_experiment(config, smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
