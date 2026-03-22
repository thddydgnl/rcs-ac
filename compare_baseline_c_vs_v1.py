"""Compare baseline C (feedforward attention PPO) vs RCS-AC v1 (recurrent attention PPO)."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ppo_baselines.config import load_experiment_config as load_ff_config
from ppo_baselines.ppo.trainer import PPOTrainer
from rcs_ac_v1.config import load_experiment_config as load_rec_config
from rcs_ac_v1.trainer import RecurrentPPOTrainer


MODEL_NAMES = ["baseline_c_attention", "rcs_ac_v1"]
COMMON_RESULT_KEYS = [
    "model_name",
    "seed",
    "train_step",
    "eval_return",
    "eval_success_rate",
    "eval_collision_rate",
    "eval_timeout_rate",
    "eval_avg_progress_to_goal",
    "actor_loss",
    "critic_loss",
    "entropy",
    "hidden_norm_mean",
    "hidden_norm_std",
    "hidden_reset_count",
    "sequence_length",
    "masked_ratio",
    "nan_detected",
    "crashed_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline C vs RCS-AC v1.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override-total-steps", type=int, default=None)
    return parser.parse_args()


def load_compare_config(path: str | Path) -> dict[str, Any]:
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


def _last_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    for row in reversed(rows):
        if key in row:
            value = row[key]
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                return float(value)
    return default


def _best_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return max(vals) if vals else default


def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(y[0]) if len(y) == 1 else 0.0
    return float(np.trapezoid(y, x) / max(x[-1] - x[0], 1.0))


def _trainer_and_config(model_name: str, base_config_path: str):
    if model_name == "baseline_c_attention":
        config = load_ff_config(base_config_path)
        config.model.baseline_type = "baseline_c"
        trainer_cls = PPOTrainer
    elif model_name == "rcs_ac_v1":
        config = load_rec_config(base_config_path)
        trainer_cls = RecurrentPPOTrainer
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return trainer_cls, config


def run_single_experiment(
    model_name: str,
    base_config_path: str,
    seed: int,
    output_root: str | Path,
    env_mode: str,
    override_total_steps: int | None = None,
) -> dict[str, Any]:
    run_dir = ensure_dir(Path(output_root) / model_name / f"seed_{seed}")
    result = {
        "model_name": model_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "finished_flag": 0,
        "nan_detected": 0,
        "crashed_flag": 0,
        "error_message": "",
    }
    try:
        trainer_cls, config = _trainer_and_config(model_name, base_config_path)
        config.seed = seed
        config.env.corruption_mode = env_mode_to_corruption_mode(env_mode)
        config.logging.output_dir = str(run_dir)
        config.logging.run_name = f"{model_name}_seed_{seed}"
        if override_total_steps is not None:
            config.ppo.total_steps = override_total_steps
            config.ppo.rollout_steps = min(config.ppo.rollout_steps, config.ppo.total_steps)
            if hasattr(config.ppo, "minibatch_size"):
                config.ppo.minibatch_size = min(config.ppo.minibatch_size, config.ppo.rollout_steps)
            if hasattr(config.ppo, "minibatch_num_sequences"):
                max_num_sequences = max(1, config.ppo.rollout_steps // max(config.ppo.sequence_length, 1))
                config.ppo.minibatch_num_sequences = min(config.ppo.minibatch_num_sequences, max_num_sequences)
            config.logging.eval_interval_updates = 1
            config.logging.save_interval_updates = max(1, config.logging.save_interval_updates)

        trainer = trainer_cls(config)
        trainer.train()

        train_rows = read_jsonl(run_dir / "train_metrics.jsonl")
        eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
        train_df = pd.DataFrame(train_rows)
        eval_df = pd.DataFrame(eval_rows)
        if not train_df.empty:
            train_df.to_csv(run_dir / "train_metrics.csv", index=False)
        if not eval_df.empty:
            eval_df.to_csv(run_dir / "eval_metrics.csv", index=False)

        status = {}
        status_path = run_dir / "run_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))

        result.update(
            {
                "finished_flag": int(status.get("finished_flag", 0)),
                "nan_detected": int(status.get("nan_detected", 0)),
                "crashed_flag": int(status.get("crashed_flag", 0)),
                "train_step": int(_last_metric(train_rows, "train_step", 0.0)),
                "eval_return": float(_last_metric(eval_rows, "eval/return_mean", 0.0)),
                "eval_success_rate": float(_last_metric(eval_rows, "eval/success_rate", 0.0)),
                "eval_collision_rate": float(_last_metric(eval_rows, "eval/collision_rate", 0.0)),
                "eval_timeout_rate": float(_last_metric(eval_rows, "eval/timeout_rate", 0.0)),
                "eval_avg_progress_to_goal": float(_last_metric(eval_rows, "eval/avg_progress_to_goal", 0.0)),
                "actor_loss": float(_last_metric(train_rows, "loss/actor", 0.0)),
                "critic_loss": float(_last_metric(train_rows, "loss/critic", 0.0)),
                "entropy": float(_last_metric(train_rows, "loss/entropy", 0.0)),
                "best_eval_success_rate": float(_best_metric(eval_rows, "eval/success_rate", 0.0)),
                "best_eval_return": float(_best_metric(eval_rows, "eval/return_mean", 0.0)),
                "learning_curve_auc_success": float(
                    _safe_auc(
                        eval_df["train_step"].to_numpy(dtype=float) if not eval_df.empty else np.array([], dtype=float),
                        eval_df["eval/success_rate"].to_numpy(dtype=float) if (not eval_df.empty and "eval/success_rate" in eval_df) else np.array([], dtype=float),
                    )
                ),
                "wall_clock_time": float(status.get("wall_clock_time", 0.0)),
                "hidden_norm_mean": 0.0,
                "hidden_norm_std": 0.0,
                "hidden_reset_count": 0.0,
                "sequence_length": 0.0,
                "masked_ratio": 0.0,
                "route_attention_dominance_ratio": float(_last_metric(eval_rows, "eval_attention/route_dominance_ratio", 0.0)),
                "hidden_zero_init_ok": 1.0,
                "hidden_reset_ok": 1.0,
                "hidden_leakage_detected": 0.0,
                "hidden_shape_mismatch_detected": 0.0,
                "hidden_nan_detected": 0.0,
                "mean_attention_vision": float(_last_metric(eval_rows, "eval_attention_mean/vision", 0.0)),
                "mean_attention_distance": float(_last_metric(eval_rows, "eval_attention_mean/distance", 0.0)),
                "mean_attention_ego": float(_last_metric(eval_rows, "eval_attention_mean/ego", 0.0)),
                "mean_attention_route": float(_last_metric(eval_rows, "eval_attention_mean/route", 0.0)),
                "std_attention_vision": float(_last_metric(eval_rows, "eval_attention_std/vision", 0.0)),
                "std_attention_distance": float(_last_metric(eval_rows, "eval_attention_std/distance", 0.0)),
                "std_attention_ego": float(_last_metric(eval_rows, "eval_attention_std/ego", 0.0)),
                "std_attention_route": float(_last_metric(eval_rows, "eval_attention_std/route", 0.0)),
                "eval_log_path": str(run_dir / "eval_metrics.jsonl"),
                "train_log_path": str(run_dir / "train_metrics.jsonl"),
            }
        )

        if model_name == "rcs_ac_v1":
            result["hidden_norm_mean"] = float(_last_metric(eval_rows, "eval/hidden_norm_mean", _last_metric(train_rows, "train/hidden_norm_mean", 0.0)))
            result["hidden_norm_std"] = float(_last_metric(eval_rows, "eval/hidden_norm_std", _last_metric(train_rows, "train/hidden_norm_std", 0.0)))
            result["hidden_reset_count"] = float(_last_metric(train_rows, "train/hidden_reset_count", 0.0))
            result["sequence_length"] = float(_last_metric(train_rows, "train/sequence_length", float(config.ppo.sequence_length)))
            result["masked_ratio"] = float(_last_metric(train_rows, "train/masked_timestep_ratio", 0.0))
            result["route_attention_dominance_ratio"] = float(
                _last_metric(eval_rows, "eval_attention/route_dominance_ratio", _last_metric(train_rows, "train_attention/route_dominance_ratio", 0.0))
            )
            hidden_validation = status.get("hidden_validation", {})
            result["hidden_zero_init_ok"] = float(hidden_validation.get("hidden_zero_init_ok", True))
            result["hidden_reset_ok"] = float(hidden_validation.get("hidden_reset_ok", True))
            result["hidden_leakage_detected"] = float(hidden_validation.get("hidden_leakage_detected", False))
            result["hidden_shape_mismatch_detected"] = float(hidden_validation.get("hidden_shape_mismatch_detected", False))
            result["hidden_nan_detected"] = float(hidden_validation.get("hidden_nan_detected", False))

        (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        result["crashed_flag"] = 1
        result["error_message"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        (run_dir / "crash_trace.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return result


def run_multi_seed_experiments(compare_config: dict[str, Any], override_total_steps: int | None = None) -> pd.DataFrame:
    output_root = ensure_dir(compare_config["output_root"])
    rows = []
    for model_name in MODEL_NAMES:
        config_path = compare_config["models"][model_name]["config_path"]
        for seed in compare_config["seeds"]:
            rows.append(
                run_single_experiment(
                    model_name=model_name,
                    base_config_path=config_path,
                    seed=seed,
                    output_root=output_root,
                    env_mode=compare_config.get("env_mode", "normal"),
                    override_total_steps=override_total_steps,
                )
            )
    return pd.DataFrame(rows)


def aggregate_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for model_name, group in results_df.groupby("model_name", sort=False):
        row = {
            "model_name": model_name,
            "num_runs": int(len(group)),
            "eval_return_mean": float(group["eval_return"].mean()),
            "eval_return_std": float(group["eval_return"].std(ddof=0)),
            "eval_success_rate_mean": float(group["eval_success_rate"].mean()),
            "eval_success_rate_std": float(group["eval_success_rate"].std(ddof=0)),
            "eval_collision_rate_mean": float(group["eval_collision_rate"].mean()),
            "eval_timeout_rate_mean": float(group["eval_timeout_rate"].mean()),
            "eval_avg_progress_mean": float(group["eval_avg_progress_to_goal"].mean()),
            "eval_avg_progress_std": float(group["eval_avg_progress_to_goal"].std(ddof=0)),
            "actor_loss_mean": float(group["actor_loss"].mean()),
            "critic_loss_mean": float(group["critic_loss"].mean()),
            "entropy_mean": float(group["entropy"].mean()),
            "hidden_norm_mean": float(group["hidden_norm_mean"].mean()),
            "hidden_norm_std_mean": float(group["hidden_norm_std"].mean()),
            "hidden_reset_count_mean": float(group["hidden_reset_count"].mean()),
            "sequence_length_mean": float(group["sequence_length"].mean()),
            "masked_ratio_mean": float(group["masked_ratio"].mean()),
            "nan_rate": float(group["nan_detected"].mean()),
            "crashed_rate": float(group["crashed_flag"].mean()),
            "attention_variance_mean": float(
                group[["std_attention_vision", "std_attention_distance", "std_attention_ego", "std_attention_route"]]
                .mean(axis=1)
                .mean()
            ),
            "hidden_bug_rate": float(
                (
                    group["hidden_leakage_detected"]
                    + group["hidden_shape_mismatch_detected"]
                    + group["hidden_nan_detected"]
                    + (1.0 - group["hidden_zero_init_ok"])
                    + (1.0 - group["hidden_reset_ok"])
                ).mean()
            ),
            "learning_curve_auc_success_mean": float(group["learning_curve_auc_success"].mean()),
        }
        for key in [
            "mean_attention_vision",
            "mean_attention_distance",
            "mean_attention_ego",
            "mean_attention_route",
            "std_attention_vision",
            "std_attention_distance",
            "std_attention_ego",
            "std_attention_route",
        ]:
            row[f"{key}_mean"] = float(group[key].mean())
            row[f"{key}_std"] = float(group[key].std(ddof=0))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    pretty_df = pd.DataFrame(
        [
            {
                "model_name": row["model_name"],
                "eval_success_rate": f"{row['eval_success_rate_mean']:.3f} +/- {row['eval_success_rate_std']:.3f}",
                "eval_return": f"{row['eval_return_mean']:.3f} +/- {row['eval_return_std']:.3f}",
                "eval_avg_progress": f"{row['eval_avg_progress_mean']:.3f} +/- {row['eval_avg_progress_std']:.3f}",
                "attention_variance": f"{row['attention_variance_mean']:.3f}",
                "hidden_bug_rate": f"{row['hidden_bug_rate']:.3f}",
            }
            for _, row in summary_df.iterrows()
        ]
    )
    return summary_df, pretty_df


def plot_learning_curves(results_df: pd.DataFrame, summary_dir: str | Path) -> None:
    summary_dir = ensure_dir(summary_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    metric_specs = [("eval/success_rate", "Eval Success Rate"), ("eval/return_mean", "Eval Return")]

    for axis, (metric_key, title) in zip(axes, metric_specs):
        for model_name in MODEL_NAMES:
            run_frames = []
            model_runs = results_df[results_df["model_name"] == model_name]
            for _, run in model_runs.iterrows():
                rows = read_jsonl(Path(run["eval_log_path"]))
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
        axis.set_title(title)
        axis.set_xlabel("Train Step")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(summary_dir) / "learning_curve_overview.png", dpi=200)
    plt.close(fig)


def save_comparison_table(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pretty_df: pd.DataFrame,
    summary_dir: str | Path,
    compare_config: dict[str, Any],
) -> None:
    summary_dir = ensure_dir(summary_dir)
    results_df.to_csv(Path(summary_dir) / "all_run_results.csv", index=False)
    summary_df.to_csv(Path(summary_dir) / "comparison_metrics.csv", index=False)
    pretty_df.to_csv(Path(summary_dir) / "comparison_metrics_pretty.csv", index=False)
    with (Path(summary_dir) / "comparison_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary_df.to_dict(orient="records"), file, ensure_ascii=True, indent=2)
    with (Path(summary_dir) / "experiment_config_snapshot.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(compare_config, file, sort_keys=False)


def save_attention_hidden_summary(results_df: pd.DataFrame, summary_dir: str | Path) -> None:
    summary_dir = ensure_dir(summary_dir)
    columns = [
        "model_name",
        "seed",
        "mean_attention_vision",
        "mean_attention_distance",
        "mean_attention_ego",
        "mean_attention_route",
        "std_attention_vision",
        "std_attention_distance",
        "std_attention_ego",
        "std_attention_route",
        "hidden_norm_mean",
        "hidden_norm_std",
        "hidden_reset_count",
        "masked_ratio",
        "hidden_zero_init_ok",
        "hidden_reset_ok",
        "hidden_leakage_detected",
        "hidden_shape_mismatch_detected",
        "hidden_nan_detected",
    ]
    results_df[columns].to_csv(Path(summary_dir) / "attention_hidden_summary.csv", index=False)


def evaluate_pass_fail(summary_df: pd.DataFrame, results_df: pd.DataFrame, summary_dir: str | Path) -> dict[str, Any]:
    summary_dir = ensure_dir(summary_dir)
    by_name = {row["model_name"]: row for _, row in summary_df.iterrows()}
    base = by_name["baseline_c_attention"]
    rec = by_name["rcs_ac_v1"]

    performance_close = (
        rec["eval_success_rate_mean"] >= base["eval_success_rate_mean"] - 0.05
        and rec["eval_return_mean"] >= base["eval_return_mean"] - 0.5
        and rec["eval_avg_progress_mean"] >= base["eval_avg_progress_mean"] - 0.01
    )
    std_improved_or_close = (
        rec["eval_success_rate_std"] <= base["eval_success_rate_std"] * 1.10
        and rec["eval_return_std"] <= base["eval_return_std"] * 1.10
    )
    hidden_bug_free = bool(
        rec["hidden_bug_rate"] == 0.0
        and rec["nan_rate"] == 0.0
        and rec["crashed_rate"] == 0.0
    )

    if performance_close and std_improved_or_close and hidden_bug_free:
        grade = "PASS"
    elif performance_close and hidden_bug_free:
        grade = "WARN"
    else:
        grade = "FAIL"

    report = {
        "overall_grade": grade,
        "performance_close": performance_close,
        "std_improved_or_close": std_improved_or_close,
        "hidden_bug_free": hidden_bug_free,
        "baseline_c": {
            "eval_success_rate_mean": float(base["eval_success_rate_mean"]),
            "eval_return_mean": float(base["eval_return_mean"]),
            "eval_avg_progress_mean": float(base["eval_avg_progress_mean"]),
        },
        "rcs_ac_v1": {
            "eval_success_rate_mean": float(rec["eval_success_rate_mean"]),
            "eval_return_mean": float(rec["eval_return_mean"]),
            "eval_avg_progress_mean": float(rec["eval_avg_progress_mean"]),
            "hidden_bug_rate": float(rec["hidden_bug_rate"]),
        },
    }

    lines = [
        "# Baseline C vs RCS-AC v1 Report",
        "",
        f"- Overall: **{grade}**",
        f"- Performance close: {performance_close}",
        f"- Std improved or close: {std_improved_or_close}",
        f"- Hidden bug free: {hidden_bug_free}",
        "",
        "## Baseline C",
        f"- eval_success_rate_mean: {base['eval_success_rate_mean']:.3f}",
        f"- eval_return_mean: {base['eval_return_mean']:.3f}",
        f"- eval_avg_progress_mean: {base['eval_avg_progress_mean']:.3f}",
        "",
        "## RCS-AC v1",
        f"- eval_success_rate_mean: {rec['eval_success_rate_mean']:.3f}",
        f"- eval_return_mean: {rec['eval_return_mean']:.3f}",
        f"- eval_avg_progress_mean: {rec['eval_avg_progress_mean']:.3f}",
        f"- hidden_bug_rate: {rec['hidden_bug_rate']:.3f}",
    ]
    (Path(summary_dir) / "pass_fail_report.md").write_text("\n".join(lines), encoding="utf-8")
    with (Path(summary_dir) / "pass_fail_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=True, indent=2)
    return report


def validate_hidden_specific(results_df: pd.DataFrame) -> pd.DataFrame:
    recurrent_rows = results_df[results_df["model_name"] == "rcs_ac_v1"].copy()
    recurrent_rows["hidden_validation_pass"] = (
        (recurrent_rows["hidden_zero_init_ok"] == 1.0)
        & (recurrent_rows["hidden_reset_ok"] == 1.0)
        & (recurrent_rows["hidden_leakage_detected"] == 0.0)
        & (recurrent_rows["hidden_shape_mismatch_detected"] == 0.0)
        & (recurrent_rows["hidden_nan_detected"] == 0.0)
    )
    return recurrent_rows


def main() -> None:
    args = parse_args()
    compare_config = load_compare_config(args.config)
    output_root = ensure_dir(compare_config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")

    results_df = run_multi_seed_experiments(compare_config, override_total_steps=args.override_total_steps)
    summary_df, pretty_df = aggregate_results(results_df)
    plot_learning_curves(results_df, summary_dir)
    save_comparison_table(results_df, summary_df, pretty_df, summary_dir, compare_config)
    save_attention_hidden_summary(results_df, summary_dir)
    hidden_validation_df = validate_hidden_specific(results_df)
    hidden_validation_df.to_csv(summary_dir / "hidden_validation.csv", index=False)
    report = evaluate_pass_fail(summary_df, results_df, summary_dir)

    print("\nComparison completed.")
    print(pretty_df.to_string(index=False))
    print(f"\nOverall grade: {report['overall_grade']}")
    print(f"Summary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
