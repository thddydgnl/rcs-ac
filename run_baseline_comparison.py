"""Automated baseline comparison runner for RCSToyNavEnv PPO baselines.

This script runs baseline A/B/C under identical protocols, aggregates results,
plots learning curves, writes summary tables, and emits PASS/WARN/FAIL reports.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ppo_baselines.config import ExperimentConfig, load_experiment_config
from ppo_baselines.ppo.trainer import PPOTrainer


BASELINE_ORDER = ["a_ego_route", "b_early_fusion", "c_attention_fusion"]
BASELINE_TO_MODEL_TYPE = {
    "a_ego_route": "baseline_a",
    "b_early_fusion": "baseline_b",
    "c_attention_fusion": "baseline_c",
}
COMMON_METRICS = [
    "eval_return",
    "eval_success_rate",
    "eval_collision_rate",
    "eval_timeout_rate",
    "eval_avg_progress_to_goal",
    "avg_episode_length",
    "actor_loss",
    "critic_loss",
    "entropy",
]
ATTENTION_METRICS = [
    "mean_attention_vision",
    "mean_attention_distance",
    "mean_attention_ego",
    "mean_attention_route",
    "std_attention_vision",
    "std_attention_distance",
    "std_attention_ego",
    "std_attention_route",
    "success_attention_vision",
    "success_attention_distance",
    "success_attention_ego",
    "success_attention_route",
    "failure_attention_vision",
    "failure_attention_distance",
    "failure_attention_ego",
    "failure_attention_route",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed baseline comparison for RCSToyNavEnv.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/compare_baselines.yaml",
        help="Path to comparison YAML config.",
    )
    parser.add_argument(
        "--override-total-steps",
        type=int,
        default=None,
        help="Optional short override for smoke or quick experiments.",
    )
    return parser.parse_args()


def load_runner_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_mode_to_corruption_mode(env_mode: str) -> str:
    if env_mode == "normal":
        return "clean"
    if env_mode in ("mild", "adverse"):
        return "corrupted"
    raise ValueError(f"Unsupported env_mode: {env_mode}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_last_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not rows:
        return default
    for row in reversed(rows):
        if key in row and isinstance(row[key], (int, float)):
            value = float(row[key])
            if np.isfinite(value):
                return value
    return default


def _extract_best_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return max(values) if values else default


def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(y[0]) if len(y) == 1 else 0.0
    trapz = getattr(np, "trapezoid", None)
    if trapz is None:
        trapz = np.trapz
    return float(trapz(y, x) / max(x[-1] - x[0], 1.0))


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_single_experiment(
    model_name: str,
    base_config_path: str,
    seed: int,
    output_root: str | Path,
    env_mode: str = "normal",
    override_total_steps: int | None = None,
) -> dict[str, Any]:
    model_dir = ensure_dir(Path(output_root) / model_name / f"seed_{seed}")
    run_result: dict[str, Any] = {
        "model_name": model_name,
        "seed": seed,
        "env_mode": env_mode,
        "run_dir": str(model_dir),
        "finished_flag": 0,
        "nan_detected": 0,
        "crashed_flag": 0,
        "error_message": "",
    }

    try:
        config = load_experiment_config(base_config_path)
        config.seed = seed
        config.model.baseline_type = BASELINE_TO_MODEL_TYPE[model_name]
        config.env.corruption_mode = env_mode_to_corruption_mode(env_mode)
        config.logging.run_name = f"{model_name}_seed_{seed}"
        config.logging.output_dir = str(model_dir)
        if override_total_steps is not None:
            config.ppo.total_steps = override_total_steps
            config.ppo.rollout_steps = min(config.ppo.rollout_steps, config.ppo.total_steps)
            config.ppo.minibatch_size = min(config.ppo.minibatch_size, config.ppo.rollout_steps)
            config.logging.eval_interval_updates = 1
            config.logging.save_interval_updates = max(1, config.logging.save_interval_updates)

        trainer = PPOTrainer(config)
        trainer.train()

        train_rows = read_jsonl(model_dir / "train_metrics.jsonl")
        eval_rows = read_jsonl(model_dir / "eval_metrics.jsonl")
        status = {}
        status_path = model_dir / "run_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))

        train_df = pd.DataFrame(train_rows)
        eval_df = pd.DataFrame(eval_rows)
        if not train_df.empty:
            train_df["model_name"] = model_name
            train_df["seed"] = seed
            train_df.to_csv(model_dir / "train_metrics.csv", index=False)
        if not eval_df.empty:
            eval_df["model_name"] = model_name
            eval_df["seed"] = seed
            eval_df.to_csv(model_dir / "eval_metrics.csv", index=False)

        finished_flag = int(status.get("finished_flag", 0))
        nan_detected = int(status.get("nan_detected", 0))
        crashed_flag = int(status.get("crashed_flag", 0))
        wall_clock_time = float(status.get("wall_clock_time", 0.0))

        if not train_df.empty:
            for col in train_df.columns:
                if train_df[col].dtype.kind in "fc":
                    if not np.all(np.isfinite(train_df[col].fillna(0.0))):
                        nan_detected = 1
        if not eval_df.empty:
            for col in eval_df.columns:
                if eval_df[col].dtype.kind in "fc":
                    if not np.all(np.isfinite(eval_df[col].fillna(0.0))):
                        nan_detected = 1

        attention_defaults = {key: 0.0 for key in ATTENTION_METRICS}
        if not eval_df.empty and model_name == "c_attention_fusion":
            attention_defaults = {
                "mean_attention_vision": float(_extract_last_metric(eval_rows, "eval_attention_mean/vision")),
                "mean_attention_distance": float(_extract_last_metric(eval_rows, "eval_attention_mean/distance")),
                "mean_attention_ego": float(_extract_last_metric(eval_rows, "eval_attention_mean/ego")),
                "mean_attention_route": float(_extract_last_metric(eval_rows, "eval_attention_mean/route")),
                "std_attention_vision": float(_extract_last_metric(eval_rows, "eval_attention_std/vision")),
                "std_attention_distance": float(_extract_last_metric(eval_rows, "eval_attention_std/distance")),
                "std_attention_ego": float(_extract_last_metric(eval_rows, "eval_attention_std/ego")),
                "std_attention_route": float(_extract_last_metric(eval_rows, "eval_attention_std/route")),
                "success_attention_vision": float(_extract_last_metric(eval_rows, "eval_attention_success/vision")),
                "success_attention_distance": float(_extract_last_metric(eval_rows, "eval_attention_success/distance")),
                "success_attention_ego": float(_extract_last_metric(eval_rows, "eval_attention_success/ego")),
                "success_attention_route": float(_extract_last_metric(eval_rows, "eval_attention_success/route")),
                "failure_attention_vision": float(_extract_last_metric(eval_rows, "eval_attention_failure/vision")),
                "failure_attention_distance": float(_extract_last_metric(eval_rows, "eval_attention_failure/distance")),
                "failure_attention_ego": float(_extract_last_metric(eval_rows, "eval_attention_failure/ego")),
                "failure_attention_route": float(_extract_last_metric(eval_rows, "eval_attention_failure/route")),
            }

        eval_steps = eval_df["train_step"].to_numpy(dtype=float) if not eval_df.empty else np.array([], dtype=float)
        eval_success = eval_df["eval/success_rate"].to_numpy(dtype=float) if (not eval_df.empty and "eval/success_rate" in eval_df) else np.array([], dtype=float)

        run_result.update(
            {
                "finished_flag": finished_flag,
                "nan_detected": nan_detected,
                "crashed_flag": crashed_flag,
                "wall_clock_time": wall_clock_time,
                "train_log_path": str(model_dir / "train_metrics.jsonl"),
                "eval_log_path": str(model_dir / "eval_metrics.jsonl"),
                "final_train_step": int(_extract_last_metric(train_rows, "train_step", 0.0)),
                "episode_return": float(_extract_last_metric(train_rows, "train/episode_return", 0.0)),
                "avg_episode_length": float(_extract_last_metric(train_rows, "train/episode_length", 0.0)),
                "actor_loss": float(_extract_last_metric(train_rows, "loss/actor", 0.0)),
                "critic_loss": float(_extract_last_metric(train_rows, "loss/critic", 0.0)),
                "entropy": float(_extract_last_metric(train_rows, "loss/entropy", 0.0)),
                "eval_return": float(_extract_last_metric(eval_rows, "eval/return_mean", 0.0)),
                "eval_success_rate": float(_extract_last_metric(eval_rows, "eval/success_rate", 0.0)),
                "eval_collision_rate": float(_extract_last_metric(eval_rows, "eval/collision_rate", 0.0)),
                "eval_timeout_rate": float(_extract_last_metric(eval_rows, "eval/timeout_rate", 0.0)),
                "eval_avg_progress_to_goal": float(_extract_last_metric(eval_rows, "eval/avg_progress_to_goal", 0.0)),
                "best_eval_success_rate": float(_extract_best_metric(eval_rows, "eval/success_rate", 0.0)),
                "best_eval_return": float(_extract_best_metric(eval_rows, "eval/return_mean", 0.0)),
                "learning_curve_auc_success": float(_safe_auc(eval_steps, eval_success)) if len(eval_steps) > 0 else 0.0,
                "learning_curve_saved": int((model_dir / "learning_curve.png").exists()),
            }
        )
        run_result.update(attention_defaults)

        stability_penalty = (
            1.0 * run_result["nan_detected"]
            + 1.0 * run_result["crashed_flag"]
            + 0.25 * float(run_result["critic_loss"] > 10.0)
            + 0.25 * float(abs(run_result["actor_loss"]) > 10.0)
        )
        run_result["training_stability_score"] = float(max(0.0, 1.0 - stability_penalty))
        (model_dir / "run_summary.json").write_text(json.dumps(run_result, indent=2), encoding="utf-8")

    except Exception as exc:
        run_result["crashed_flag"] = 1
        run_result["error_message"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        (model_dir / "crash_trace.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status = {
            "finished_flag": 0,
            "nan_detected": int(run_result["nan_detected"]),
            "crashed_flag": 1,
            "error_message": run_result["error_message"],
        }
        (model_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    return run_result


def run_multi_seed_experiments(
    runner_config: dict[str, Any],
    override_total_steps: int | None = None,
) -> pd.DataFrame:
    output_root = ensure_dir(runner_config["output_root"])
    seeds = list(runner_config["seeds"])
    env_mode = runner_config.get("env_mode", "normal")

    results: list[dict[str, Any]] = []
    for model_name in BASELINE_ORDER:
        base_config_path = runner_config["baselines"][model_name]["config_path"]
        for seed in seeds:
            result = run_single_experiment(
                model_name=model_name,
                base_config_path=base_config_path,
                seed=seed,
                output_root=output_root,
                env_mode=env_mode,
                override_total_steps=override_total_steps,
            )
            results.append(result)

    df = pd.DataFrame(results)
    return df


def _mean_std_str(series: pd.Series) -> str:
    return f"{series.mean():.3f} +/- {series.std(ddof=0):.3f}"


def aggregate_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for model_name, group in results_df.groupby("model_name", sort=False):
        row = {
            "model_name": model_name,
            "num_runs": int(len(group)),
            "finished_rate_mean": float(group["finished_flag"].mean()),
            "crashed_rate_mean": float(group["crashed_flag"].mean()),
            "nan_rate_mean": float(group["nan_detected"].mean()),
            "final_eval_return_mean": float(group["eval_return"].mean()),
            "final_eval_return_std": float(group["eval_return"].std(ddof=0)),
            "final_eval_success_rate_mean": float(group["eval_success_rate"].mean()),
            "final_eval_success_rate_std": float(group["eval_success_rate"].std(ddof=0)),
            "final_eval_collision_rate_mean": float(group["eval_collision_rate"].mean()),
            "final_eval_timeout_rate_mean": float(group["eval_timeout_rate"].mean()),
            "final_eval_avg_progress_mean": float(group["eval_avg_progress_to_goal"].mean()),
            "final_eval_avg_progress_std": float(group["eval_avg_progress_to_goal"].std(ddof=0)),
            "best_eval_success_rate_mean": float(group["best_eval_success_rate"].mean()),
            "best_eval_return_mean": float(group["best_eval_return"].mean()),
            "avg_episode_length_mean": float(group["avg_episode_length"].mean()),
            "actor_loss_mean": float(group["actor_loss"].mean()),
            "critic_loss_mean": float(group["critic_loss"].mean()),
            "entropy_mean": float(group["entropy"].mean()),
            "wall_clock_time_mean": float(group["wall_clock_time"].mean()),
            "learning_curve_auc_success_mean": float(group["learning_curve_auc_success"].mean()),
            "training_stability_score_mean": float(group["training_stability_score"].mean()),
        }
        for key in ATTENTION_METRICS:
            if key in group.columns:
                row[f"{key}_mean"] = float(group[key].mean())
                row[f"{key}_std"] = float(group[key].std(ddof=0))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    pretty_rows = []
    for _, row in summary_df.iterrows():
        pretty_rows.append(
            {
                "model_name": row["model_name"],
                "eval_success_rate": f"{row['final_eval_success_rate_mean']:.3f} +/- {row['final_eval_success_rate_std']:.3f}",
                "eval_return": f"{row['final_eval_return_mean']:.3f} +/- {row['final_eval_return_std']:.3f}",
                "eval_avg_progress": f"{row['final_eval_avg_progress_mean']:.3f} +/- {row['final_eval_avg_progress_std']:.3f}",
                "stability_score": f"{row['training_stability_score_mean']:.3f}",
                "crashed_rate": f"{row['crashed_rate_mean']:.3f}",
                "nan_rate": f"{row['nan_rate_mean']:.3f}",
            }
        )
    pretty_df = pd.DataFrame(pretty_rows)
    return summary_df, pretty_df


def plot_learning_curves(results_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    metric_specs = [
        ("eval/return_mean", "Eval Return"),
        ("eval/success_rate", "Eval Success Rate"),
        ("train/episode_return", "Train Episode Return"),
        ("train/avg_progress_to_goal", "Train Avg Progress"),
    ]

    for axis, (metric_key, title) in zip(axes, metric_specs):
        has_any_line = False
        for model_name in BASELINE_ORDER:
            run_frames = []
            model_runs = results_df[results_df["model_name"] == model_name]
            for _, run in model_runs.iterrows():
                run_dir = Path(run["run_dir"])
                log_file = run_dir / ("eval_metrics.jsonl" if metric_key.startswith("eval/") else "train_metrics.jsonl")
                rows = read_jsonl(log_file)
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                if metric_key not in frame.columns:
                    continue
                run_frames.append(frame[["train_step", metric_key]].rename(columns={metric_key: "metric"}))

            if not run_frames:
                continue

            merged = pd.concat(run_frames, keys=range(len(run_frames)), names=["run_idx"]).reset_index(level=0)
            grouped = merged.groupby("train_step")["metric"]
            mean = grouped.mean()
            std = grouped.std(ddof=0).fillna(0.0)
            axis.plot(mean.index, mean.values, label=model_name)
            axis.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.2)
            has_any_line = True

        axis.set_title(title)
        axis.set_xlabel("Train Step")
        axis.grid(True, alpha=0.3)

    if has_any_line:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "learning_curve_overview.png", dpi=200)
    plt.close(fig)

    for model_name in BASELINE_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        metric_specs = [("eval/return_mean", "Eval Return"), ("eval/success_rate", "Eval Success Rate")]
        has_any_line = False
        for axis, (metric_key, title) in zip(axes, metric_specs):
            model_runs = results_df[results_df["model_name"] == model_name]
            for _, run in model_runs.iterrows():
                rows = read_jsonl(Path(run["run_dir"]) / "eval_metrics.jsonl")
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                if metric_key not in frame.columns:
                    continue
                axis.plot(frame["train_step"], frame[metric_key], alpha=0.4, label=f"seed_{int(run['seed'])}")
                has_any_line = True
            axis.set_title(f"{model_name} - {title}")
            axis.set_xlabel("Train Step")
            axis.grid(True, alpha=0.3)
        if has_any_line:
            axes[0].legend()
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"{model_name}_learning_curve.png", dpi=200)
        plt.close(fig)


def _grade_from_comparison(
    mean_diff: float,
    std_ratio: float,
    crash_or_nan_rate: float,
    pass_threshold: float,
    warn_threshold: float,
    max_std_ratio_pass: float,
    max_std_ratio_warn: float,
) -> str:
    if crash_or_nan_rate > 0.34:
        return "FAIL"
    if mean_diff >= pass_threshold and std_ratio <= max_std_ratio_pass:
        return "PASS"
    if mean_diff >= warn_threshold and std_ratio <= max_std_ratio_warn:
        return "WARN"
    return "FAIL"


def evaluate_pass_fail(summary_df: pd.DataFrame, results_df: pd.DataFrame, summary_dir: str | Path) -> dict[str, Any]:
    summary_dir = ensure_dir(summary_dir)
    summary_by_model = {row["model_name"]: row for _, row in summary_df.iterrows()}

    def _curve_exists(model_name: str) -> bool:
        return (Path(summary_dir) / f"{model_name}_learning_curve.png").exists()

    curve_checks = {
        model_name: _curve_exists(model_name)
        for model_name in BASELINE_ORDER
    }
    eval_file_checks = {
        model_name: bool((results_df[results_df["model_name"] == model_name]["eval_log_path"].apply(lambda p: Path(p).exists()).all()))
        for model_name in BASELINE_ORDER
    }

    a = summary_by_model["a_ego_route"]
    b = summary_by_model["b_early_fusion"]
    c = summary_by_model["c_attention_fusion"]

    early_success_diff = b["final_eval_success_rate_mean"] - a["final_eval_success_rate_mean"]
    early_return_diff = b["final_eval_return_mean"] - a["final_eval_return_mean"]
    early_progress_diff = b["final_eval_avg_progress_mean"] - a["final_eval_avg_progress_mean"]
    early_std_ratio = max(
        b["final_eval_success_rate_std"] / max(a["final_eval_success_rate_std"], 1e-6),
        b["final_eval_return_std"] / max(a["final_eval_return_std"], 1e-6),
    )
    early_crash_nan = max(b["crashed_rate_mean"], b["nan_rate_mean"])
    early_grade = _grade_from_comparison(
        mean_diff=min(early_success_diff + 0.05, early_return_diff + 0.5, early_progress_diff + 0.01),
        std_ratio=early_std_ratio,
        crash_or_nan_rate=early_crash_nan,
        pass_threshold=0.0,
        warn_threshold=-0.05,
        max_std_ratio_pass=1.5,
        max_std_ratio_warn=2.0,
    )

    attention_success_diff = c["final_eval_success_rate_mean"] - b["final_eval_success_rate_mean"]
    attention_return_diff = c["final_eval_return_mean"] - b["final_eval_return_mean"]
    attention_progress_diff = c["final_eval_avg_progress_mean"] - b["final_eval_avg_progress_mean"]
    attention_std_ratio = max(
        c["final_eval_success_rate_std"] / max(b["final_eval_success_rate_std"], 1e-6),
        c["final_eval_return_std"] / max(b["final_eval_return_std"], 1e-6),
    )
    attention_crash_nan = max(c["crashed_rate_mean"], c["nan_rate_mean"])
    attention_grade = _grade_from_comparison(
        mean_diff=min(attention_success_diff + 0.05, attention_return_diff + 0.5, attention_progress_diff + 0.01),
        std_ratio=attention_std_ratio,
        crash_or_nan_rate=attention_crash_nan,
        pass_threshold=0.0,
        warn_threshold=-0.05,
        max_std_ratio_pass=1.6,
        max_std_ratio_warn=2.3,
    )

    overall_grade = "PASS"
    if not all(curve_checks.values()) or not all(eval_file_checks.values()):
        overall_grade = "FAIL"
    elif "FAIL" in (early_grade, attention_grade):
        overall_grade = "FAIL"
    elif "WARN" in (early_grade, attention_grade):
        overall_grade = "WARN"

    report = {
        "overall_grade": overall_grade,
        "curve_checks": curve_checks,
        "eval_file_checks": eval_file_checks,
        "single_vs_early": {
            "grade": early_grade,
            "success_diff": float(early_success_diff),
            "return_diff": float(early_return_diff),
            "progress_diff": float(early_progress_diff),
            "std_ratio": float(early_std_ratio),
        },
        "early_vs_attention": {
            "grade": attention_grade,
            "success_diff": float(attention_success_diff),
            "return_diff": float(attention_return_diff),
            "progress_diff": float(attention_progress_diff),
            "std_ratio": float(attention_std_ratio),
        },
    }

    lines = [
        "# Stage 4 Pass/Fail Report",
        "",
        f"- Overall: **{overall_grade}**",
        "",
        "## File Checks",
    ]
    for model_name in BASELINE_ORDER:
        lines.append(
            f"- {model_name}: curve_saved={curve_checks[model_name]} eval_logs_saved={eval_file_checks[model_name]}"
        )
    lines.extend(
        [
            "",
            "## Single vs Early",
            f"- Grade: **{early_grade}**",
            f"- Success diff: {early_success_diff:.3f}",
            f"- Return diff: {early_return_diff:.3f}",
            f"- Progress diff: {early_progress_diff:.3f}",
            f"- Std ratio: {early_std_ratio:.3f}",
            "",
            "## Early vs Attention",
            f"- Grade: **{attention_grade}**",
            f"- Success diff: {attention_success_diff:.3f}",
            f"- Return diff: {attention_return_diff:.3f}",
            f"- Progress diff: {attention_progress_diff:.3f}",
            f"- Std ratio: {attention_std_ratio:.3f}",
        ]
    )
    (Path(summary_dir) / "pass_fail_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def save_comparison_table(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pretty_df: pd.DataFrame,
    pass_fail_report: dict[str, Any],
    runner_config: dict[str, Any],
    summary_dir: str | Path,
) -> None:
    summary_dir = ensure_dir(summary_dir)
    results_df.to_csv(Path(summary_dir) / "all_run_results.csv", index=False)
    summary_df.to_csv(Path(summary_dir) / "comparison_metrics.csv", index=False)
    pretty_df.to_csv(Path(summary_dir) / "comparison_metrics_pretty.csv", index=False)
    with (Path(summary_dir) / "comparison_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary_df.to_dict(orient="records"), file, ensure_ascii=True, indent=2)
    with (Path(summary_dir) / "pass_fail_report.json").open("w", encoding="utf-8") as file:
        json.dump(pass_fail_report, file, ensure_ascii=True, indent=2)
    with (Path(summary_dir) / "comparison_metrics.md").open("w", encoding="utf-8") as file:
        file.write(_dataframe_to_markdown(pretty_df))
    with (Path(summary_dir) / "experiment_config_snapshot.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(runner_config, file, sort_keys=False)


def save_attention_summary(results_df: pd.DataFrame, summary_dir: str | Path) -> None:
    summary_dir = ensure_dir(summary_dir)
    attention_df = results_df[results_df["model_name"] == "c_attention_fusion"][
        ["model_name", "seed"] + ATTENTION_METRICS
    ].copy()
    attention_df.to_csv(Path(summary_dir) / "attention_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    runner_config = load_runner_config(args.config)

    output_root = ensure_dir(runner_config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")

    results_df = run_multi_seed_experiments(
        runner_config=runner_config,
        override_total_steps=args.override_total_steps,
    )
    summary_df, pretty_df = aggregate_results(results_df)
    plot_learning_curves(results_df, summary_dir)
    pass_fail_report = evaluate_pass_fail(summary_df, results_df, summary_dir)
    save_attention_summary(results_df, summary_dir)
    save_comparison_table(results_df, summary_df, pretty_df, pass_fail_report, runner_config, summary_dir)

    print("\nBaseline comparison completed.")
    print(pretty_df.to_string(index=False))
    print(f"\nOverall grade: {pass_fail_report['overall_grade']}")
    print(f"Summary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
