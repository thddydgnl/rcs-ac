"""Run Version 3.5 confirm on the single winning pilot candidate."""

from __future__ import annotations

import argparse

from confirm_utils import build_confirm_config, resolve_confirm_winner
from run_v3_5_adapter_pilot import run_v3_5_adapter_pilot
from v3_shapley_helpers import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.5 confirm on the winning pilot candidate.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_5_adapter_confirm_5seed.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    winner = resolve_confirm_winner(config["pilot_score_summary_path"], config["baseline_candidate"], "V3.5")
    confirm_config = build_confirm_config(config, winner)
    run_v3_5_adapter_pilot(confirm_config, smoke=args.smoke)


if __name__ == "__main__":
    main()
