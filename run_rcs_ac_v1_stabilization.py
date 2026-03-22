"""Conservative stabilization runner for RCS-AC v1.

Runs a small set of ablations focused on recurrent PPO stabilization without
changing the overall model family beyond optimization-safe options.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rcs_ac_v1.config import load_experiment_config
from rcs_ac_v1.trainer import RecurrentPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCS-AC v1 stabilization ablations.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/rcs_ac_v1_stabilization_ablation.yaml",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_overrides(config, overrides: dict[str, Any]) -> None:
    for section_name, section_values in overrides.items():
        target = getattr(config, section_name)
        for key, value in section_values.items():
            setattr(target, key, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    for row in reversed(rows):
        if key in row:
            value = row[key]
            if isinstance(value, (int, float)):
                return float(value)
    return default


def run_single_ablation(ablation_name: str, seed: int, config_path: str, output_root: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    apply_overrides(config, overrides)
    config.seed = seed
    run_dir = ensure_dir(output_root / ablation_name / f"seed_{seed}")
    config.logging.output_dir = str(run_dir)
    config.logging.run_name = f"{ablation_name}_seed_{seed}"
    trainer = RecurrentPPOTrainer(config)
    trainer.train()

    train_rows = read_jsonl(run_dir / "train_metrics.jsonl")
    eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    return {
        "ablation_name": ablation_name,
        "seed": seed,
        "eval_return": last_metric(eval_rows, "eval/return_mean", 0.0),
        "eval_success_rate": last_metric(eval_rows, "eval/success_rate", 0.0),
        "eval_avg_progress_to_goal": last_metric(eval_rows, "eval/avg_progress_to_goal", 0.0),
        "hidden_norm_mean": last_metric(eval_rows, "eval/hidden_norm_mean", last_metric(train_rows, "train/hidden_norm_mean", 0.0)),
        "route_attention_dominance_ratio": last_metric(eval_rows, "eval_attention/route_dominance_ratio", last_metric(train_rows, "train_attention/route_dominance_ratio", 0.0)),
        "masked_timestep_ratio": last_metric(train_rows, "train/masked_timestep_ratio", 0.0),
        "nan_detected": int(status.get("nan_detected", 0)),
        "crashed_flag": int(status.get("crashed_flag", 0)),
    }


def main() -> None:
    args = parse_args()
    runner_cfg = load_yaml(args.config)
    output_root = ensure_dir(runner_cfg["output_root"])
    rows = []

    for ablation_name, overrides in runner_cfg["ablations"].items():
        for seed in runner_cfg["seeds"]:
            rows.append(
                run_single_ablation(
                    ablation_name=ablation_name,
                    seed=seed,
                    config_path=runner_cfg["base_config_path"],
                    output_root=output_root,
                    overrides=overrides,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "stabilization_summary.csv", index=False)
    print(df.to_string(index=False))
    print("\nGrouped mean")
    print(df.groupby("ablation_name").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
