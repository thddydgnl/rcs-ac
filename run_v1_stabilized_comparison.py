"""Round-based fair comparison runner for baseline C vs stabilized RCS-AC v1 candidates."""

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


BASELINE_NAME = "baseline_c_attention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stabilized v1 candidates against baseline C.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/compare_v1_stabilized_vs_baseline.yaml",
    )
    parser.add_argument("--resume", action="store_true", help="Skip completed runs if summaries exist.")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def last_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    for row in reversed(rows):
        if key in row:
            value = row[key]
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                return float(value)
    return default


def best_metric(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float)) and np.isfinite(float(row[key]))]
    return max(vals) if vals else default


def safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(y[0]) if len(y) == 1 else 0.0
    return float(np.trapezoid(y, x) / max(x[-1] - x[0], 1.0))


def env_mode_to_corruption_mode(env_mode: str) -> str:
    if env_mode == "normal":
        return "clean"
    if env_mode in ("mild", "adverse"):
        return "corrupted"
    raise ValueError(f"Unsupported env_mode: {env_mode}")


def deep_apply_overrides(config_obj: Any, overrides: dict[str, Any]) -> None:
    for section_name, section_values in overrides.items():
        target = getattr(config_obj, section_name)
        for key, value in section_values.items():
            setattr(target, key, value)


def build_trainer_and_config(model_type: str, config_path: str):
    if model_type == "baseline_c_attention":
        config = load_ff_config(config_path)
        config.model.baseline_type = "baseline_c"
        trainer_cls = PPOTrainer
    elif model_type == "rcs_ac_v1":
        config = load_rec_config(config_path)
        trainer_cls = RecurrentPPOTrainer
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return trainer_cls, config


def maybe_load_existing_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def run_single_experiment(
    config_name: str,
    model_spec: dict[str, Any],
    seed: int,
    output_root: str | Path,
    env_mode: str,
    fair_total_steps: int,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir = ensure_dir(Path(output_root) / config_name / f"seed_{seed}")
    status_path = run_dir / "run_status.json"
    train_log_path = run_dir / "train_metrics.jsonl"
    eval_log_path = run_dir / "eval_metrics.jsonl"
    if resume:
        existing = maybe_load_existing_summary(run_dir)
        if existing is not None and int(existing.get("finished_flag", 0)) == 1:
            return existing

    result = {
        "model_name": model_spec["model_type"],
        "config_name": config_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "finished_flag": 0,
        "nan_detected": 0,
        "crashed_flag": 0,
        "error_message": "",
        "train_step": 0,
        "episode_return": 0.0,
        "eval_return": 0.0,
        "eval_success_rate": 0.0,
        "eval_collision_rate": 0.0,
        "eval_timeout_rate": 0.0,
        "eval_avg_progress_to_goal": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
        "wall_clock_time": 0.0,
        "hidden_norm_mean": 0.0,
        "hidden_norm_std": 0.0,
        "hidden_reset_count": 0.0,
        "sequence_length": 0.0,
        "masked_timestep_ratio": 0.0,
        "hidden_nan_detected": 0.0,
        "hidden_shape_mismatch_detected": 0.0,
        "hidden_leakage_suspected": 0.0,
        "attention_mean_vision": 0.0,
        "attention_mean_distance": 0.0,
        "attention_mean_ego": 0.0,
        "attention_mean_route": 0.0,
        "attention_std_vision": 0.0,
        "attention_std_distance": 0.0,
        "attention_std_ego": 0.0,
        "attention_std_route": 0.0,
        "route_attention_dominance_ratio": 0.0,
        "best_eval_success_rate": 0.0,
        "best_eval_return": 0.0,
        "learning_curve_auc_success": 0.0,
        "train_log_path": str(train_log_path),
        "eval_log_path": str(eval_log_path),
    }
    try:
        trainer_cls, config = build_trainer_and_config(model_spec["model_type"], model_spec["config_path"])
        config.seed = seed
        config.env.corruption_mode = env_mode_to_corruption_mode(env_mode)
        config.logging.output_dir = str(run_dir)
        config.logging.run_name = f"{config_name}_seed_{seed}"

        # Force fair baseline-vs-v1 total steps by default.
        config.ppo.total_steps = fair_total_steps
        deep_apply_overrides(config, model_spec.get("overrides", {}))

        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            status = {}

        # If training already finished and logs exist, reconstruct summary without rerunning.
        if not (status_path.exists() and int(status.get("finished_flag", 0)) == 1 and train_log_path.exists()):
            trainer = trainer_cls(config)
            trainer.train()
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

        train_rows = read_jsonl(train_log_path)
        eval_rows = read_jsonl(eval_log_path)
        train_df = pd.DataFrame(train_rows)
        eval_df = pd.DataFrame(eval_rows)
        if not train_df.empty:
            train_df.to_csv(run_dir / "train_metrics.csv", index=False)
        if not eval_df.empty:
            eval_df.to_csv(run_dir / "eval_metrics.csv", index=False)

        recurrent = model_spec["model_type"] == "rcs_ac_v1"
        result.update(
            {
                "train_step": int(last_metric(train_rows, "train_step", 0.0)),
                "episode_return": float(last_metric(train_rows, "train/episode_return", 0.0)),
                "eval_return": float(last_metric(eval_rows, "eval/return_mean", 0.0)),
                "eval_success_rate": float(last_metric(eval_rows, "eval/success_rate", 0.0)),
                "eval_collision_rate": float(last_metric(eval_rows, "eval/collision_rate", 0.0)),
                "eval_timeout_rate": float(last_metric(eval_rows, "eval/timeout_rate", 0.0)),
                "eval_avg_progress_to_goal": float(last_metric(eval_rows, "eval/avg_progress_to_goal", 0.0)),
                "actor_loss": float(last_metric(train_rows, "loss/actor", 0.0)),
                "critic_loss": float(last_metric(train_rows, "loss/critic", 0.0)),
                "entropy": float(last_metric(train_rows, "loss/entropy", 0.0)),
                "grad_norm": float(last_metric(train_rows, "loss/grad_norm", 0.0)),
                "wall_clock_time": float(status.get("wall_clock_time", 0.0)),
                "nan_detected": int(status.get("nan_detected", 0)),
                "crashed_flag": int(status.get("crashed_flag", 0)),
                "finished_flag": int(status.get("finished_flag", 0)),
                "best_eval_success_rate": float(best_metric(eval_rows, "eval/success_rate", 0.0)),
                "best_eval_return": float(best_metric(eval_rows, "eval/return_mean", 0.0)),
                "learning_curve_auc_success": float(
                    safe_auc(
                        eval_df["train_step"].to_numpy(dtype=float) if not eval_df.empty else np.array([], dtype=float),
                        eval_df["eval/success_rate"].to_numpy(dtype=float) if (not eval_df.empty and "eval/success_rate" in eval_df) else np.array([], dtype=float),
                    )
                ),
                "hidden_norm_mean": float(last_metric(eval_rows, "eval/hidden_norm_mean", last_metric(train_rows, "train/hidden_norm_mean", 0.0))) if recurrent else 0.0,
                "hidden_norm_std": float(last_metric(eval_rows, "eval/hidden_norm_std", last_metric(train_rows, "train/hidden_norm_std", 0.0))) if recurrent else 0.0,
                "hidden_reset_count": float(last_metric(train_rows, "train/hidden_reset_count", 0.0)) if recurrent else 0.0,
                "sequence_length": float(last_metric(train_rows, "train/sequence_length", 0.0)) if recurrent else 0.0,
                "masked_timestep_ratio": float(last_metric(train_rows, "train/masked_timestep_ratio", 0.0)) if recurrent else 0.0,
                "hidden_nan_detected": 0.0,
                "hidden_shape_mismatch_detected": 0.0,
                "hidden_leakage_suspected": 0.0,
                "attention_mean_vision": float(last_metric(eval_rows, "eval_attention_mean/vision", 0.0)),
                "attention_mean_distance": float(last_metric(eval_rows, "eval_attention_mean/distance", 0.0)),
                "attention_mean_ego": float(last_metric(eval_rows, "eval_attention_mean/ego", 0.0)),
                "attention_mean_route": float(last_metric(eval_rows, "eval_attention_mean/route", 0.0)),
                "attention_std_vision": float(last_metric(eval_rows, "eval_attention_std/vision", 0.0)),
                "attention_std_distance": float(last_metric(eval_rows, "eval_attention_std/distance", 0.0)),
                "attention_std_ego": float(last_metric(eval_rows, "eval_attention_std/ego", 0.0)),
                "attention_std_route": float(last_metric(eval_rows, "eval_attention_std/route", 0.0)),
                "route_attention_dominance_ratio": float(
                    last_metric(eval_rows, "eval_attention/route_dominance_ratio", last_metric(train_rows, "train_attention/route_dominance_ratio", 0.0))
                ),
                "train_log_path": str(run_dir / "train_metrics.jsonl"),
                "eval_log_path": str(run_dir / "eval_metrics.jsonl"),
            }
        )

        if recurrent:
            hidden_validation = status.get("hidden_validation", {})
            result["hidden_nan_detected"] = float(hidden_validation.get("hidden_nan_detected", False))
            result["hidden_shape_mismatch_detected"] = float(hidden_validation.get("hidden_shape_mismatch_detected", False))
            result["hidden_leakage_suspected"] = float(hidden_validation.get("hidden_leakage_detected", False))

        (run_dir / "run_summary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception as exc:
        result["crashed_flag"] = 1
        result["error_message"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        (run_dir / "crash_trace.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (run_dir / "run_summary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def run_multi_seed_experiments(
    round_name: str,
    candidate_names: list[str],
    runner_config: dict[str, Any],
    resume: bool = False,
) -> pd.DataFrame:
    output_root = ensure_dir(Path(runner_config["output_root"]) / round_name)
    rows = []
    for config_name in candidate_names:
        model_spec = runner_config["models"][config_name]
        for seed in runner_config["seeds"]:
            rows.append(
                run_single_experiment(
                    config_name=config_name,
                    model_spec=model_spec,
                    seed=seed,
                    output_root=output_root,
                    env_mode=runner_config["env_mode"],
                    fair_total_steps=runner_config["fair_total_steps"],
                    resume=resume,
                )
            )
    return pd.DataFrame(rows)


def aggregate_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = [
        "finished_flag",
        "nan_detected",
        "crashed_flag",
        "eval_success_rate",
        "eval_return",
        "eval_avg_progress_to_goal",
        "eval_collision_rate",
        "eval_timeout_rate",
        "actor_loss",
        "critic_loss",
        "entropy",
        "grad_norm",
        "hidden_norm_mean",
        "hidden_norm_std",
        "hidden_reset_count",
        "sequence_length",
        "masked_timestep_ratio",
        "hidden_nan_detected",
        "hidden_shape_mismatch_detected",
        "hidden_leakage_suspected",
        "route_attention_dominance_ratio",
        "attention_std_vision",
        "attention_std_distance",
        "attention_std_ego",
        "attention_std_route",
        "wall_clock_time",
    ]
    results_df = results_df.copy()
    for column in required_columns:
        if column not in results_df.columns:
            results_df[column] = 0.0

    summary_rows = []
    for config_name, group in results_df.groupby("config_name", sort=False):
        row = {
            "config_name": config_name,
            "model_name": group["model_name"].iloc[0],
            "num_runs": int(len(group)),
            "finished_rate": float(group["finished_flag"].mean()),
            "nan_rate": float(group["nan_detected"].mean()),
            "crashed_rate": float(group["crashed_flag"].mean()),
            "mean_eval_success_rate": float(group["eval_success_rate"].mean()),
            "std_eval_success_rate": float(group["eval_success_rate"].std(ddof=0)),
            "mean_eval_return": float(group["eval_return"].mean()),
            "std_eval_return": float(group["eval_return"].std(ddof=0)),
            "mean_eval_avg_progress_to_goal": float(group["eval_avg_progress_to_goal"].mean()),
            "std_eval_avg_progress_to_goal": float(group["eval_avg_progress_to_goal"].std(ddof=0)),
            "mean_eval_collision_rate": float(group["eval_collision_rate"].mean()),
            "mean_eval_timeout_rate": float(group["eval_timeout_rate"].mean()),
            "mean_actor_loss": float(group["actor_loss"].mean()),
            "mean_critic_loss": float(group["critic_loss"].mean()),
            "mean_entropy": float(group["entropy"].mean()),
            "mean_grad_norm": float(group["grad_norm"].mean()),
            "mean_hidden_norm": float(group["hidden_norm_mean"].mean()),
            "std_hidden_norm": float(group["hidden_norm_std"].mean()),
            "mean_hidden_reset_count": float(group["hidden_reset_count"].mean()),
            "mean_sequence_length": float(group["sequence_length"].mean()),
            "mean_masked_timestep_ratio": float(group["masked_timestep_ratio"].mean()),
            "hidden_bug_rate": float(
                (
                    group["hidden_nan_detected"]
                    + group["hidden_shape_mismatch_detected"]
                    + group["hidden_leakage_suspected"]
                ).mean()
            ),
            "mean_route_attention_dominance_ratio": float(group["route_attention_dominance_ratio"].mean()),
            "mean_attention_std": float(
                group[["attention_std_vision", "attention_std_distance", "attention_std_ego", "attention_std_route"]]
                .mean(axis=1)
                .mean()
            ),
            "mean_wall_clock_time": float(group["wall_clock_time"].mean()),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    pretty_df = pd.DataFrame(
        [
            {
                "config_name": row["config_name"],
                "mean_eval_success_rate": f"{row['mean_eval_success_rate']:.3f} +/- {row['std_eval_success_rate']:.3f}",
                "mean_eval_return": f"{row['mean_eval_return']:.3f} +/- {row['std_eval_return']:.3f}",
                "mean_eval_progress": f"{row['mean_eval_avg_progress_to_goal']:.3f} +/- {row['std_eval_avg_progress_to_goal']:.3f}",
                "collision_rate": f"{row['mean_eval_collision_rate']:.3f}",
                "timeout_rate": f"{row['mean_eval_timeout_rate']:.3f}",
                "hidden_bug_rate": f"{row['hidden_bug_rate']:.3f}",
            }
            for _, row in summary_df.iterrows()
        ]
    )
    return summary_df, pretty_df


def plot_learning_curves(results_df: pd.DataFrame, summary_dir: str | Path) -> None:
    summary_dir = ensure_dir(summary_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    metric_specs = [("eval/return_mean", "Eval Return"), ("eval/success_rate", "Eval Success Rate")]

    for axis, (metric_key, title) in zip(axes, metric_specs):
        for config_name, group in results_df.groupby("config_name", sort=False):
            frames = []
            for _, row in group.iterrows():
                eval_rows = read_jsonl(row["eval_log_path"])
                if not eval_rows:
                    continue
                frame = pd.DataFrame(eval_rows)
                if metric_key not in frame.columns:
                    continue
                frames.append(frame[["train_step", metric_key]].rename(columns={metric_key: "metric"}))
            if not frames:
                continue
            merged = pd.concat(frames, keys=range(len(frames)), names=["seed_idx"]).reset_index(level=0)
            grouped_frame = merged.groupby("train_step")["metric"]
            mean = grouped_frame.mean()
            std = grouped_frame.std(ddof=0).fillna(0.0)
            axis.plot(mean.index, mean.values, label=config_name)
            axis.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Train Step")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(summary_dir) / "learning_curve_overview.png", dpi=200)
    plt.close(fig)


def compare_against_baseline(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline_row = summary_df[summary_df["config_name"] == BASELINE_NAME].iloc[0]
    rows = []
    for _, row in summary_df.iterrows():
        if row["config_name"] == BASELINE_NAME:
            continue
        rows.append(
            {
                "config_name": row["config_name"],
                "success_diff_vs_baseline": float(row["mean_eval_success_rate"] - baseline_row["mean_eval_success_rate"]),
                "return_diff_vs_baseline": float(row["mean_eval_return"] - baseline_row["mean_eval_return"]),
                "progress_diff_vs_baseline": float(row["mean_eval_avg_progress_to_goal"] - baseline_row["mean_eval_avg_progress_to_goal"]),
                "success_std_ratio_vs_baseline": float(row["std_eval_success_rate"] / max(baseline_row["std_eval_success_rate"], 1e-6)),
                "return_std_ratio_vs_baseline": float(row["std_eval_return"] / max(baseline_row["std_eval_return"], 1e-6)),
                "hidden_bug_rate": float(row["hidden_bug_rate"]),
                "nan_rate": float(row["nan_rate"]),
                "crashed_rate": float(row["crashed_rate"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_pass_fail(summary_df: pd.DataFrame, summary_dir: str | Path, round_name: str) -> pd.DataFrame:
    summary_dir = ensure_dir(summary_dir)
    baseline_row = summary_df[summary_df["config_name"] == BASELINE_NAME].iloc[0]
    verdict_rows = []

    for _, row in summary_df.iterrows():
        if row["config_name"] == BASELINE_NAME:
            verdict_rows.append({"config_name": row["config_name"], "verdict": "BASELINE", "reason": "reference"})
            continue

        success_diff = row["mean_eval_success_rate"] - baseline_row["mean_eval_success_rate"]
        return_diff = row["mean_eval_return"] - baseline_row["mean_eval_return"]
        progress_diff = row["mean_eval_avg_progress_to_goal"] - baseline_row["mean_eval_avg_progress_to_goal"]
        success_std_ratio = row["std_eval_success_rate"] / max(baseline_row["std_eval_success_rate"], 1e-6)
        return_std_ratio = row["std_eval_return"] / max(baseline_row["std_eval_return"], 1e-6)
        hidden_bug = row["hidden_bug_rate"] > 0.0 or row["nan_rate"] > 0.0 or row["crashed_rate"] > 0.0 or row["finished_rate"] < 1.0

        if (
            not hidden_bug
            and success_diff >= -0.05
            and return_diff >= -0.5
            and progress_diff >= -0.01
            and success_std_ratio <= 1.2
            and return_std_ratio <= 1.2
        ):
            verdict = "PASS"
            reason = "performance close or better, variance controlled, no hidden/runtime issues"
        elif (
            not hidden_bug
            and success_diff >= -0.10
            and return_diff >= -1.0
            and progress_diff >= -0.02
        ):
            verdict = "WARN"
            reason = "performance roughly competitive but variance or one metric remains weaker"
        else:
            verdict = "FAIL"
            reason = "performance materially below baseline or stability/runtime issue detected"

        verdict_rows.append({"config_name": row["config_name"], "verdict": verdict, "reason": reason})

    verdict_df = pd.DataFrame(verdict_rows)
    with (Path(summary_dir) / "pass_fail_report.md").open("w", encoding="utf-8") as file:
        file.write(f"# {round_name} Pass/Fail Report\n\n")
        for _, row in verdict_df.iterrows():
            file.write(f"- {row['config_name']}: **{row['verdict']}** - {row['reason']}\n")
    with (Path(summary_dir) / "pass_fail_report.json").open("w", encoding="utf-8") as file:
        json.dump(verdict_df.to_dict(orient="records"), file, ensure_ascii=True, indent=2)
    return verdict_df


def select_best_config(summary_df: pd.DataFrame, verdict_df: pd.DataFrame) -> str:
    merged = summary_df.merge(verdict_df, on="config_name", how="left")
    merged = merged[merged["config_name"] != BASELINE_NAME].copy()
    merged["verdict_rank"] = merged["verdict"].map({"PASS": 0, "WARN": 1, "FAIL": 2}).fillna(3)
    merged["selection_score"] = (
        100.0 * merged["mean_eval_success_rate"]
        + 10.0 * merged["mean_eval_return"]
        + 50.0 * merged["mean_eval_avg_progress_to_goal"]
        - 20.0 * merged["std_eval_success_rate"]
        - 5.0 * merged["std_eval_return"]
        - 20.0 * merged["mean_eval_collision_rate"]
        - 10.0 * merged["mean_eval_timeout_rate"]
        - 100.0 * merged["hidden_bug_rate"]
        - 2.0 * np.maximum(merged["mean_route_attention_dominance_ratio"] - 8.0, 0.0)
    )
    merged = merged.sort_values(["verdict_rank", "selection_score"], ascending=[True, False])
    return str(merged.iloc[0]["config_name"])


def save_round_summary(
    round_name: str,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pretty_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    verdict_df: pd.DataFrame,
    runner_config: dict[str, Any],
) -> None:
    round_summary_dir = ensure_dir(Path(runner_config["output_root"]) / round_name / "summary")
    results_df.to_csv(round_summary_dir / "all_run_results.csv", index=False)
    summary_df.to_csv(round_summary_dir / "comparison_metrics.csv", index=False)
    compare_df.to_csv(round_summary_dir / "comparison_against_baseline.csv", index=False)
    pretty_df.to_csv(round_summary_dir / "comparison_metrics_pretty.csv", index=False)
    results_df[
        [
            "config_name",
            "seed",
            "attention_mean_vision",
            "attention_mean_distance",
            "attention_mean_ego",
            "attention_mean_route",
            "attention_std_vision",
            "attention_std_distance",
            "attention_std_ego",
            "attention_std_route",
            "route_attention_dominance_ratio",
            "hidden_norm_mean",
            "hidden_norm_std",
            "hidden_reset_count",
            "masked_timestep_ratio",
            "hidden_nan_detected",
            "hidden_shape_mismatch_detected",
            "hidden_leakage_suspected",
        ]
    ].to_csv(round_summary_dir / "attention_hidden_summary.csv", index=False)
    with (round_summary_dir / "comparison_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary_df.to_dict(orient="records"), file, ensure_ascii=True, indent=2)
    with (round_summary_dir / "experiment_config_snapshot.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(runner_config, file, sort_keys=False)
    with (round_summary_dir / "comparison_metrics.md").open("w", encoding="utf-8") as file:
        file.write(pretty_df.to_markdown(index=False) if hasattr(pretty_df, "to_markdown") else pretty_df.to_string(index=False))
    plot_learning_curves(results_df, round_summary_dir)
    evaluate_pass_fail(summary_df, round_summary_dir, round_name)


def generate_markdown_report(
    runner_config: dict[str, Any],
    round_summaries: dict[str, dict[str, Any]],
    final_winner: str,
) -> None:
    summary_root = ensure_dir(Path(runner_config["output_root"]) / "summary")
    lines = [
        "# Final V1 Stabilized Comparison Report",
        "",
        f"- Final recommended config: **{final_winner}**",
        "",
    ]
    for round_name, payload in round_summaries.items():
        lines.append(f"## {round_name}")
        lines.append("")
        lines.append(payload["pretty_df"].to_string(index=False))
        lines.append("")
        lines.append(f"- winner: **{payload['winner']}**")
        lines.append("")

    lines.append("## Stage 1 Warm-up Decision")
    if final_winner == BASELINE_NAME:
        lines.append("- Do not move to Stage 1 warm-up with recurrent v1 yet. Keep baseline C as reference and continue v1 stabilization.")
    else:
        lines.append(f"- Proceed to Stage 1 warm-up with **{final_winner}** as the recurrent v1 standard candidate.")
    (summary_root / "final_recommendation_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_round_candidates(runner_config: dict[str, Any], round_name: str, previous_winners: dict[str, str]) -> list[str]:
    round_cfg = runner_config["rounds"][round_name]
    if round_name == "round_1":
        return list(round_cfg["candidates"])

    candidates: list[str] = []
    if round_name == "round_2":
        round1_summary = previous_winners["_round_1_summary"]
        top_configs = list(round1_summary["summary_df"][round1_summary["summary_df"]["config_name"] != BASELINE_NAME]
                           .sort_values(["mean_eval_success_rate", "mean_eval_return"], ascending=[False, False])
                           .head(round_cfg["auto_select_top_k_from_round_1"])["config_name"])
        candidates.extend(top_configs)
        candidates.extend(round_cfg["fixed_candidates"])
    elif round_name == "round_3":
        if round_cfg.get("include_baseline", False):
            candidates.append(BASELINE_NAME)
        if round_cfg.get("include_round_2_winner", False):
            candidates.append(previous_winners["round_2"])
        candidates.extend(round_cfg.get("optional_candidates", []))
    seen = set()
    deduped = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def main() -> None:
    args = parse_args()
    runner_config = load_yaml(args.config)
    ensure_dir(runner_config["output_root"])

    round_summaries: dict[str, dict[str, Any]] = {}
    previous_winners: dict[str, str] = {}

    for round_name in ["round_1", "round_2", "round_3"]:
        candidate_names = build_round_candidates(runner_config, round_name, previous_winners)
        results_df = run_multi_seed_experiments(round_name, candidate_names, runner_config, resume=args.resume)
        summary_df, pretty_df = aggregate_results(results_df)
        compare_df = compare_against_baseline(summary_df)
        round_summary_dir = ensure_dir(Path(runner_config["output_root"]) / round_name / "summary")
        verdict_df = evaluate_pass_fail(summary_df, round_summary_dir, round_name)
        save_round_summary(round_name, results_df, summary_df, pretty_df, compare_df, verdict_df, runner_config)
        winner = select_best_config(summary_df, verdict_df)
        previous_winners[round_name] = winner
        previous_winners[f"_{round_name}_summary"] = {
            "summary_df": summary_df,
        }
        round_summaries[round_name] = {
            "results_df": results_df,
            "summary_df": summary_df,
            "pretty_df": pretty_df,
            "verdict_df": verdict_df,
            "winner": winner,
        }
        print(f"\n{round_name} completed. winner={winner}")
        print(pretty_df.to_string(index=False))

    final_winner = previous_winners["round_3"]
    generate_markdown_report(runner_config, round_summaries, final_winner)
    print(f"\nFinal recommended config: {final_winner}")


if __name__ == "__main__":
    main()
