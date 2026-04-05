"""Shared helpers for confirm-stage configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def resolve_confirm_winner(score_summary_path: str | Path, baseline_candidate: str, label: str) -> str:
    score_summary = Path(score_summary_path)
    if not score_summary.exists():
        raise FileNotFoundError(f"Pilot score summary missing: {score_summary}")
    score_df = pd.read_csv(score_summary)
    pass_rows = score_df[
        (score_df["pass_fail"] == "PASS") & (score_df["candidate_name"] != baseline_candidate)
    ]
    if pass_rows.empty:
        raise RuntimeError(f"No PASS winner available from {label} pilot; confirm is gated closed.")
    return str(pass_rows.iloc[0]["candidate_name"])


def build_confirm_config(config: dict[str, Any], winner: str) -> dict[str, Any]:
    baseline_candidate = str(config["baseline_candidate"])
    confirm_config = dict(config)
    baseline_spec = config["candidates"].get(baseline_candidate)
    if baseline_spec is None:
        checkpoint_root = config.get("reference_checkpoint_root") or config.get("source_checkpoint_root")
        if not checkpoint_root:
            raise KeyError(f"Missing baseline candidate spec for {baseline_candidate} and no reference/source checkpoint root")
        baseline_spec = {
            "checkpoint_root": checkpoint_root,
            "source_experiment_name": config.get("source_experiment_name", baseline_candidate),
            "eval_only": True,
            "loss_overrides": {},
        }
    confirm_config["evaluation_candidate"] = winner
    confirm_config["comparison_baseline_candidate"] = baseline_candidate
    confirm_config["candidate_order"] = [baseline_candidate, winner]
    confirm_config["candidates"] = {
        baseline_candidate: baseline_spec,
        winner: config["candidates"][winner],
    }
    confirm_config["baseline_candidate"] = baseline_candidate
    return confirm_config
