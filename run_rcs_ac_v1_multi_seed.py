"""Draft multi-seed runner for RCS-AC v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rcs_ac_v1.config import load_experiment_config
from rcs_ac_v1.trainer import RecurrentPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed RCS-AC v1 experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-root", type=str, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in args.seeds:
        config = load_experiment_config(args.config)
        config.seed = seed
        config.logging.output_dir = str(output_root / f"seed_{seed}")
        trainer = RecurrentPPOTrainer(config)
        trainer.train()

        eval_rows = read_jsonl(Path(config.logging.output_dir) / "eval_metrics.jsonl")
        train_rows = read_jsonl(Path(config.logging.output_dir) / "train_metrics.jsonl")
        final_eval = eval_rows[-1] if eval_rows else {}
        final_train = train_rows[-1] if train_rows else {}
        summaries.append(
            {
                "seed": seed,
                "eval_return": final_eval.get("eval/return_mean", 0.0),
                "eval_success_rate": final_eval.get("eval/success_rate", 0.0),
                "eval_avg_progress_to_goal": final_eval.get("eval/avg_progress_to_goal", 0.0),
                "hidden_norm_mean": final_eval.get("eval/hidden_norm_mean", 0.0),
                "nan_detected": final_train.get("nan_detected", 0),
                "crashed_flag": final_train.get("crashed_flag", 0),
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_root / "rcs_ac_v1_multi_seed_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\nMean results")
    print(summary_df.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
