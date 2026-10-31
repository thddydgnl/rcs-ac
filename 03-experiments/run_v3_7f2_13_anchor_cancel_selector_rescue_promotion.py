"""Run Version 3.7f2.13 anchor cancel + thresholded selector rescue promotion check."""

from __future__ import annotations

import argparse

from run_v3_7f2_13_anchor_cancel_selector_rescue_pilot import run_v3_7f2_13_experiment
from v3_shapley_helpers import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.7f2.13 anchor cancel + thresholded selector rescue promotion check.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_7f2_13_anchor_cancel_selector_rescue_promotion.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_v3_7f2_13_experiment(config, smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
