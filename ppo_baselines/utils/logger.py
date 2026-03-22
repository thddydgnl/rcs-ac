"""Simple JSONL/CSV-friendly logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.train_metrics_path = self.output_dir / "train_metrics.jsonl"
        self.eval_metrics_path = self.output_dir / "eval_metrics.jsonl"

    def log(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    def log_train(self, metrics: dict[str, Any]) -> None:
        with self.train_metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    def log_eval(self, metrics: dict[str, Any]) -> None:
        with self.eval_metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    def save_config(self, config_dict: dict[str, Any]) -> None:
        with (self.output_dir / "config_snapshot.json").open("w", encoding="utf-8") as file:
            json.dump(config_dict, file, ensure_ascii=True, indent=2)

    def save_status(self, status_dict: dict[str, Any]) -> None:
        with (self.output_dir / "run_status.json").open("w", encoding="utf-8") as file:
            json.dump(status_dict, file, ensure_ascii=True, indent=2)
