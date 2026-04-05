"""Freeze standard v1, 5-seed revalidation, and Stage 1 warm-up transition runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ppo_baselines.config import load_experiment_config as load_ff_config
from ppo_baselines.ppo.trainer import PPOTrainer
from rcs_ac_v1.config import load_experiment_config as load_rec_config
from rcs_ac_v1.trainer import RecurrentPPOTrainer
from stage1_warmup.config import load_experiment_config as load_stage1_config
from stage1_warmup.trainer import Stage1WarmupTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard v1 freeze -> revalidation -> Stage 1 warm-up.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/transition_to_stage1.yaml",
    )
    parser.add_argument("--resume", action="store_true")
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


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def latest_checkpoint(seed_dir: Path) -> Path:
    checkpoint_final = seed_dir / "checkpoint_final.pt"
    if checkpoint_final.exists():
        return checkpoint_final
    checkpoints = sorted(seed_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No checkpoint found in {seed_dir}")


def register_standard_v1_config(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    standard_spec = config["standard_v1"]
    base_cfg = load_rec_config(standard_spec["base_config_path"])
    deep_apply_overrides(base_cfg, standard_spec["overrides"])
    snapshot = asdict(base_cfg)
    out_path = Path(standard_spec["registered_config_path"])
    ensure_dir(out_path.parent)
    save_yaml(out_path, snapshot)
    return out_path, snapshot


def _build_baseline_trainer(
    config_path: str,
    seed: int,
    output_dir: str,
    env_mode: str,
    total_steps: int,
    eval_interval: int,
    num_eval_episodes: int | None = None,
):
    config = load_ff_config(config_path)
    config.model.baseline_type = "baseline_c"
    config.seed = seed
    config.env.corruption_mode = env_mode_to_corruption_mode(env_mode)
    config.ppo.total_steps = total_steps
    if num_eval_episodes is not None:
        config.ppo.num_eval_episodes = int(num_eval_episodes)
    config.logging.eval_interval_updates = eval_interval
    config.logging.output_dir = output_dir
    config.logging.run_name = f"baseline_c_attention_seed_{seed}"
    return PPOTrainer, config


def _build_standard_v1_trainer(
    config_path: str,
    seed: int,
    output_dir: str,
    env_mode: str,
    total_steps: int,
    eval_interval: int,
    num_eval_episodes: int | None = None,
    warm_start_checkpoint_path: str | None = None,
):
    config = load_rec_config(config_path)
    config.seed = seed
    config.env.corruption_mode = env_mode_to_corruption_mode(env_mode)
    config.ppo.total_steps = total_steps
    if num_eval_episodes is not None:
        config.ppo.num_eval_episodes = int(num_eval_episodes)
    if warm_start_checkpoint_path:
        config.model.warm_start_baseline_c_checkpoint = str(warm_start_checkpoint_path)
    config.logging.eval_interval_updates = eval_interval
    config.logging.output_dir = output_dir
    config.logging.run_name = f"standard_v1_seed_{seed}"
    return RecurrentPPOTrainer, config


def _build_stage1_trainer(
    config_path: str,
    seed: int,
    output_dir: str,
    env_mode: str,
    init_checkpoint_path: str | None = None,
):
    config = load_stage1_config(config_path)
    config.seed = seed
    config.env.corruption_mode = env_mode_to_corruption_mode(env_mode)
    config.logging.output_dir = output_dir
    config.logging.run_name = f"stage1_warmup_seed_{seed}"
    if init_checkpoint_path:
        config.warmup.init_checkpoint_path = init_checkpoint_path
    return Stage1WarmupTrainer, config


def _read_status(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        return json.loads(status_path.read_text(encoding="utf-8"))
    return {}


def _summarize_revalidation_run(
    model_name: str,
    config_name: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    train_rows = read_jsonl(run_dir / "train_metrics.jsonl")
    eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
    status = _read_status(run_dir)
    hidden_validation = status.get("hidden_validation", {})
    summary = {
        "model_name": model_name,
        "config_name": config_name,
        "seed": seed,
        "train_step": int(last_metric(eval_rows or train_rows, "train_step", 0.0)),
        "episode_return": float(last_metric(train_rows, "train/episode_return", 0.0)),
        "eval_return": float(last_metric(eval_rows, "eval/return_mean", 0.0)),
        "eval_success_rate": float(last_metric(eval_rows, "eval/success_rate", 0.0)),
        "eval_collision_rate": float(last_metric(eval_rows, "eval/collision_rate", 0.0)),
        "eval_timeout_rate": float(last_metric(eval_rows, "eval/timeout_rate", 0.0)),
        "eval_avg_progress_to_goal": float(last_metric(eval_rows, "eval/avg_progress_to_goal", 0.0)),
        "actor_loss": float(last_metric(train_rows, "loss/actor", 0.0)),
        "critic_loss": float(last_metric(train_rows, "loss/critic", 0.0)),
        "entropy": float(last_metric(train_rows, "loss/entropy", 0.0)),
        "grad_norm": float(last_metric(train_rows, "loss/grad_norm", last_metric(train_rows, "train/grad_norm", 0.0))),
        "wall_clock_time": float(status.get("wall_clock_time", 0.0)),
        "nan_detected": int(status.get("nan_detected", 0)),
        "crashed_flag": int(status.get("crashed_flag", 0)),
        "finished_flag": int(status.get("finished_flag", 0)),
        "hidden_norm_mean": float(last_metric(eval_rows, "eval/hidden_norm_mean", 0.0)),
        "hidden_norm_std": float(last_metric(eval_rows, "eval/hidden_norm_std", 0.0)),
        "hidden_reset_count": float(hidden_validation.get("hidden_reset_count", last_metric(train_rows, "train/hidden_reset_count", 0.0))),
        "sequence_length": float(last_metric(train_rows, "train/sequence_length", 0.0)),
        "masked_timestep_ratio": float(last_metric(train_rows, "train/masked_timestep_ratio", 0.0)),
        "hidden_nan_detected": float(hidden_validation.get("hidden_nan_detected", False)),
        "hidden_shape_mismatch_detected": float(hidden_validation.get("hidden_shape_mismatch_detected", False)),
        "hidden_leakage_suspected": float(hidden_validation.get("hidden_leakage_detected", False)),
        "attention_mean_vision": float(last_metric(eval_rows, "eval_attention_mean/vision", 0.0)),
        "attention_mean_distance": float(last_metric(eval_rows, "eval_attention_mean/distance", 0.0)),
        "attention_mean_ego": float(last_metric(eval_rows, "eval_attention_mean/ego", 0.0)),
        "attention_mean_route": float(last_metric(eval_rows, "eval_attention_mean/route", 0.0)),
        "attention_std_vision": float(last_metric(eval_rows, "eval_attention_std/vision", 0.0)),
        "attention_std_distance": float(last_metric(eval_rows, "eval_attention_std/distance", 0.0)),
        "attention_std_ego": float(last_metric(eval_rows, "eval_attention_std/ego", 0.0)),
        "attention_std_route": float(last_metric(eval_rows, "eval_attention_std/route", 0.0)),
        "route_attention_dominance_ratio": float(last_metric(eval_rows, "eval_attention/route_dominance_ratio", 0.0)),
        "best_eval_success_rate": float(best_metric(eval_rows, "eval/success_rate", 0.0)),
        "best_eval_return": float(best_metric(eval_rows, "eval/return_mean", 0.0)),
        "learning_curve_auc_success": safe_auc(
            np.asarray([row.get("train_step", 0.0) for row in eval_rows], dtype=np.float32),
            np.asarray([row.get("eval/success_rate", 0.0) for row in eval_rows], dtype=np.float32),
        ) if eval_rows else 0.0,
        "train_log_path": str(run_dir / "train_metrics.jsonl"),
        "eval_log_path": str(run_dir / "eval_metrics.jsonl"),
    }
    return summary


def run_revalidation_experiment(
    model_name: str,
    model_spec: dict[str, Any],
    seed: int,
    output_root: str | Path,
    env_mode: str,
    total_steps: int,
    eval_interval: int,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir = ensure_dir(Path(output_root) / model_name / f"seed_{seed}")
    run_summary_path = run_dir / "run_summary.json"
    status_path = run_dir / "run_status.json"
    if resume and run_summary_path.exists():
        return json.loads(run_summary_path.read_text(encoding="utf-8"))

    try:
        if model_name == "baseline_c_attention":
            trainer_cls, trainer_config = _build_baseline_trainer(
                config_path=model_spec["config_path"],
                seed=seed,
                output_dir=str(run_dir),
                env_mode=env_mode,
                total_steps=total_steps,
                eval_interval=eval_interval,
                num_eval_episodes=model_spec.get("num_eval_episodes"),
            )
        else:
            warm_start_checkpoint_path = None
            baseline_checkpoint_root = model_spec.get("baseline_checkpoint_root")
            if baseline_checkpoint_root:
                warm_start_checkpoint_path = str(
                    latest_checkpoint(Path(str(baseline_checkpoint_root)) / f"seed_{seed}")
                )
            trainer_cls, trainer_config = _build_standard_v1_trainer(
                config_path=model_spec["config_path"],
                seed=seed,
                output_dir=str(run_dir),
                env_mode=env_mode,
                total_steps=total_steps,
                eval_interval=eval_interval,
                num_eval_episodes=model_spec.get("num_eval_episodes"),
                warm_start_checkpoint_path=warm_start_checkpoint_path,
            )
        trainer = trainer_cls(trainer_config)
        trainer.train()
        summary = _summarize_revalidation_run(
            model_name=model_name,
            config_name=model_name,
            seed=seed,
            run_dir=run_dir,
        )
    except Exception as exc:
        error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        summary = {
            "model_name": model_name,
            "config_name": model_name,
            "seed": seed,
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
            "nan_detected": 0,
            "crashed_flag": 1,
            "finished_flag": 0,
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
            "error_message": error_message,
        }
        status_payload = {
            "finished_flag": 0,
            "nan_detected": 0,
            "crashed_flag": 1,
            "error_message": error_message,
        }
        status_path.write_text(json.dumps(status_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def run_multi_seed_revalidation(config: dict[str, Any], standard_config_path: Path, resume: bool = False) -> list[dict[str, Any]]:
    output_root = ensure_dir(config["output_root"]) / "revalidation"
    seeds = list(config["revalidation"]["seeds"])
    env_mode = config["revalidation"]["env_mode"]
    total_steps = int(config["revalidation"]["total_steps"])
    eval_interval = int(config["revalidation"]["eval_interval_updates"])
    num_eval_episodes = int(config["revalidation"].get("num_eval_episodes", 10))
    baseline_spec = {
        "config_path": config["revalidation"]["baseline_config_path"],
        "num_eval_episodes": num_eval_episodes,
    }
    standard_spec = {
        "config_path": str(standard_config_path),
        "num_eval_episodes": num_eval_episodes,
        "baseline_checkpoint_root": config["revalidation"].get("baseline_checkpoint_root"),
    }
    rows = []
    for model_name, model_spec in (
        ("baseline_c_attention", baseline_spec),
        ("standard_v1", standard_spec),
    ):
        for seed in seeds:
            rows.append(
                run_revalidation_experiment(
                    model_name=model_name,
                    model_spec=model_spec,
                    seed=seed,
                    output_root=output_root,
                    env_mode=env_mode,
                    total_steps=total_steps,
                    eval_interval=eval_interval,
                    resume=resume,
                )
            )
    return rows


def aggregate_revalidation_results(rows: list[dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["model_name", "seed"]).reset_index(drop=True)
    df.to_csv(summary_dir / "all_run_results.csv", index=False)
    grouped = []
    for model_name, group in df.groupby("model_name"):
        grouped.append(
            {
                "model_name": model_name,
                "mean_eval_success_rate": float(group["eval_success_rate"].mean()),
                "std_eval_success_rate": float(group["eval_success_rate"].std(ddof=0)),
                "mean_eval_return": float(group["eval_return"].mean()),
                "std_eval_return": float(group["eval_return"].std(ddof=0)),
                "mean_eval_avg_progress_to_goal": float(group["eval_avg_progress_to_goal"].mean()),
                "std_eval_avg_progress_to_goal": float(group["eval_avg_progress_to_goal"].std(ddof=0)),
                "collision_rate": float(group["eval_collision_rate"].mean()),
                "timeout_rate": float(group["eval_timeout_rate"].mean()),
                "hidden_bug_rate": float(
                    ((group["hidden_nan_detected"] + group["hidden_shape_mismatch_detected"] + group["hidden_leakage_suspected"]) > 0).mean()
                ),
                "nan_rate": float(group["nan_detected"].mean()),
                "crashed_rate": float(group["crashed_flag"].mean()),
                "training_completion_rate": float(group["finished_flag"].mean()),
            }
        )
    agg_df = pd.DataFrame(grouped)
    agg_df.to_csv(summary_dir / "revalidation_comparison_metrics.csv", index=False)
    agg_df.to_json(summary_dir / "revalidation_comparison_metrics.json", orient="records", indent=2)
    return agg_df


def save_attention_reliability_logs(df: pd.DataFrame, output_path: Path) -> None:
    keep_cols = [
        "model_name",
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
        "sequence_length",
        "masked_timestep_ratio",
        "hidden_nan_detected",
        "hidden_shape_mismatch_detected",
        "hidden_leakage_suspected",
    ]
    df[keep_cols].to_csv(output_path, index=False)


def compare_standard_v1_vs_baseline_c(agg_df: pd.DataFrame) -> pd.DataFrame:
    baseline = agg_df[agg_df["model_name"] == "baseline_c_attention"].iloc[0]
    standard = agg_df[agg_df["model_name"] == "standard_v1"].iloc[0]
    compare = pd.DataFrame(
        [
            {
                "model_name": "standard_v1",
                "success_diff_vs_baseline": float(standard["mean_eval_success_rate"] - baseline["mean_eval_success_rate"]),
                "return_diff_vs_baseline": float(standard["mean_eval_return"] - baseline["mean_eval_return"]),
                "progress_diff_vs_baseline": float(
                    standard["mean_eval_avg_progress_to_goal"] - baseline["mean_eval_avg_progress_to_goal"]
                ),
                "success_std_ratio_vs_baseline": float(
                    standard["std_eval_success_rate"] / max(baseline["std_eval_success_rate"], 1e-6)
                ),
                "return_std_ratio_vs_baseline": float(
                    standard["std_eval_return"] / max(baseline["std_eval_return"], 1e-6)
                ),
                "hidden_bug_rate": float(standard["hidden_bug_rate"]),
                "nan_rate": float(standard["nan_rate"]),
                "crashed_rate": float(standard["crashed_rate"]),
            }
        ]
    )
    return compare


def evaluate_revalidation_pass_fail(compare_df: pd.DataFrame) -> tuple[str, str]:
    row = compare_df.iloc[0]
    hidden_issue = row["hidden_bug_rate"] > 0 or row["nan_rate"] > 0 or row["crashed_rate"] > 0
    if hidden_issue or row["success_diff_vs_baseline"] < -0.05 or row["return_diff_vs_baseline"] < -0.5:
        return "FAIL", "standard_v1 is clearly worse or has hidden/runtime issues"
    if row["success_diff_vs_baseline"] >= 0.0 and row["return_diff_vs_baseline"] >= 0.0 and row["success_std_ratio_vs_baseline"] <= 1.5 and row["return_std_ratio_vs_baseline"] <= 1.5:
        return "PASS", "standard_v1 is competitive or better with acceptable seed variance"
    return "WARN", "performance is roughly competitive but variance or one metric still needs monitoring"


def plot_learning_curves(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    run_dirs = [Path(row["train_log_path"]).parent for row in summary_rows if "train_log_path" in row]
    plt.figure(figsize=(9, 5))
    for run_dir in run_dirs:
        model_name = run_dir.parent.name
        seed_name = run_dir.name
        eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
        if not eval_rows:
            continue
        x = [row["train_step"] for row in eval_rows]
        y = [row["eval/success_rate"] for row in eval_rows]
        plt.plot(x, y, alpha=0.35, label=f"{model_name}-{seed_name}")
    plt.xlabel("Train Step")
    plt.ylabel("Eval Success Rate")
    plt.title("Revalidation Learning Curves")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_stage1_warmup(config: dict[str, Any], resume: bool = False) -> list[dict[str, Any]]:
    output_root = ensure_dir(config["output_root"]) / "stage1_warmup" / "standard_v1"
    seeds = list(config["stage1_warmup"]["seeds"])
    env_mode = config["stage1_warmup"]["env_mode"]
    config_path = config["stage1_warmup"]["config_path"]
    init_checkpoint_root = config["stage1_warmup"].get("init_checkpoint_root")
    rows = []
    for seed in seeds:
        run_dir = ensure_dir(output_root / f"seed_{seed}")
        run_summary_path = run_dir / "run_summary.json"
        if resume and run_summary_path.exists():
            rows.append(json.loads(run_summary_path.read_text(encoding="utf-8")))
            continue
        try:
            init_checkpoint_path = None
            if init_checkpoint_root:
                init_checkpoint_path = str(latest_checkpoint(Path(init_checkpoint_root) / f"seed_{seed}"))
            trainer_cls, trainer_config = _build_stage1_trainer(
                config_path=config_path,
                seed=seed,
                output_dir=str(run_dir),
                env_mode=env_mode,
                init_checkpoint_path=init_checkpoint_path,
            )
            trainer = trainer_cls(trainer_config)
            trainer.train()
            train_rows = read_jsonl(run_dir / "train_metrics.jsonl")
            eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
            status = _read_status(run_dir)
            hidden_validation = status.get("hidden_validation", {})
            row = {
                "model_name": "standard_v1_stage1_warmup",
                "config_name": "standard_v1_stage1_warmup",
                "seed": seed,
                "train_step": int(last_metric(eval_rows or train_rows, "train_step", 0.0)),
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
                "hidden_norm_mean": float(last_metric(eval_rows, "eval/hidden_norm_mean", 0.0)),
                "hidden_norm_std": float(last_metric(eval_rows, "eval/hidden_norm_std", 0.0)),
                "hidden_reset_count": float(hidden_validation.get("hidden_reset_count", last_metric(train_rows, "train/hidden_reset_count", 0.0))),
                "sequence_length": float(last_metric(train_rows, "train/sequence_length", 0.0)),
                "masked_timestep_ratio": float(last_metric(train_rows, "train/masked_timestep_ratio", 0.0)),
                "hidden_nan_detected": float(hidden_validation.get("hidden_nan_detected", False)),
                "hidden_shape_mismatch_detected": float(hidden_validation.get("hidden_shape_mismatch_detected", False)),
                "hidden_leakage_suspected": float(hidden_validation.get("hidden_leakage_detected", False)),
                "reliability_mean_vision": float(last_metric(eval_rows, "eval_reliability_mean/vision", 0.0)),
                "reliability_mean_distance": float(last_metric(eval_rows, "eval_reliability_mean/distance", 0.0)),
                "reliability_mean_ego": float(last_metric(eval_rows, "eval_reliability_mean/ego", 0.0)),
                "reliability_mean_route": float(last_metric(eval_rows, "eval_reliability_mean/route", 0.0)),
                "reliability_std_vision": float(last_metric(eval_rows, "eval_reliability_std/vision", 0.0)),
                "reliability_std_distance": float(last_metric(eval_rows, "eval_reliability_std/distance", 0.0)),
                "reliability_std_ego": float(last_metric(eval_rows, "eval_reliability_std/ego", 0.0)),
                "reliability_std_route": float(last_metric(eval_rows, "eval_reliability_std/route", 0.0)),
                "reliability_nan_detected": float(last_metric(eval_rows, "eval/reliability_nan_detected", 0.0)),
                "reliability_loss": float(last_metric(train_rows, "loss/reliability", 0.0)),
                "weak_corruption_flag": float(last_metric(train_rows, "train/weak_corruption_flag", 0.0)),
                "quality_label_if_available": float(last_metric(eval_rows, "eval/quality_label_if_available", 0.0)),
                "attention_mean_vision": float(last_metric(eval_rows, "eval_attention_mean/vision", 0.0)),
                "attention_mean_distance": float(last_metric(eval_rows, "eval_attention_mean/distance", 0.0)),
                "attention_mean_ego": float(last_metric(eval_rows, "eval_attention_mean/ego", 0.0)),
                "attention_mean_route": float(last_metric(eval_rows, "eval_attention_mean/route", 0.0)),
                "attention_std_vision": float(last_metric(eval_rows, "eval_attention_std/vision", 0.0)),
                "attention_std_distance": float(last_metric(eval_rows, "eval_attention_std/distance", 0.0)),
                "attention_std_ego": float(last_metric(eval_rows, "eval_attention_std/ego", 0.0)),
                "attention_std_route": float(last_metric(eval_rows, "eval_attention_std/route", 0.0)),
                "route_attention_dominance_ratio": float(last_metric(eval_rows, "eval_attention/route_dominance_ratio", 0.0)),
                "attention_reliability_correlation": float(last_metric(eval_rows, "eval_attention_reliability/correlation", 0.0)),
                "success_episode_attention_summary_route": float(last_metric(eval_rows, "eval_attention_success/route", 0.0)),
                "failure_episode_attention_summary_route": float(last_metric(eval_rows, "eval_attention_failure/route", 0.0)),
                "success_episode_reliability_summary_route": float(last_metric(eval_rows, "eval_reliability_success/route", 0.0)),
                "failure_episode_reliability_summary_route": float(last_metric(eval_rows, "eval_reliability_failure/route", 0.0)),
                "init_checkpoint_path": init_checkpoint_path or "",
            }
        except Exception as exc:
            error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            row = {
                "model_name": "standard_v1_stage1_warmup",
                "config_name": "standard_v1_stage1_warmup",
                "seed": seed,
                "train_step": 0,
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
                "nan_detected": 0,
                "crashed_flag": 1,
                "finished_flag": 0,
                "error_message": error_message,
                "init_checkpoint_path": init_checkpoint_path or "",
            }
            (run_dir / "run_status.json").write_text(
                json.dumps({"finished_flag": 0, "crashed_flag": 1, "nan_detected": 0, "error_message": error_message}, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        run_summary_path.write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")
        rows.append(row)
    return rows


def aggregate_stage1_results(rows: list[dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["seed"]).reset_index(drop=True)
    df.to_csv(summary_dir / "stage1_seed_results.csv", index=False)
    grouped = {
        "mean_eval_success_rate": float(df["eval_success_rate"].mean()),
        "std_eval_success_rate": float(df["eval_success_rate"].std(ddof=0)),
        "mean_eval_return": float(df["eval_return"].mean()),
        "std_eval_return": float(df["eval_return"].std(ddof=0)),
        "mean_eval_avg_progress_to_goal": float(df["eval_avg_progress_to_goal"].mean()),
        "std_eval_avg_progress_to_goal": float(df["eval_avg_progress_to_goal"].std(ddof=0)),
        "collision_rate": float(df["eval_collision_rate"].mean()),
        "timeout_rate": float(df["eval_timeout_rate"].mean()),
        "mean_reliability_loss": float(df["reliability_loss"].mean()),
        "reliability_nan_rate": float(df["reliability_nan_detected"].mean()),
        "mean_attention_reliability_correlation": float(df["attention_reliability_correlation"].mean()),
        "finished_rate": float(df["finished_flag"].mean()),
    }
    metrics_df = pd.DataFrame([grouped])
    metrics_df.to_csv(summary_dir / "stage1_warmup_metrics.csv", index=False)
    metrics_df.to_json(summary_dir / "stage1_warmup_metrics.json", orient="records", indent=2)
    keep_cols = [
        "seed",
        "attention_mean_vision",
        "attention_mean_distance",
        "attention_mean_ego",
        "attention_mean_route",
        "reliability_mean_vision",
        "reliability_mean_distance",
        "reliability_mean_ego",
        "reliability_mean_route",
        "attention_reliability_correlation",
        "success_episode_attention_summary_route",
        "failure_episode_attention_summary_route",
        "success_episode_reliability_summary_route",
        "failure_episode_reliability_summary_route",
        "reliability_nan_detected",
        "reliability_loss",
        "weak_corruption_flag",
        "quality_label_if_available",
    ]
    df[keep_cols].to_csv(summary_dir / "stage1_attention_reliability_summary.csv", index=False)
    return metrics_df


def generate_final_transition_report(
    standard_snapshot_path: Path,
    reval_agg: pd.DataFrame,
    reval_compare: pd.DataFrame,
    reval_grade: str,
    reval_reason: str,
    stage1_metrics: pd.DataFrame | None,
    stage1_executed: bool,
    output_path: Path,
) -> None:
    baseline = reval_agg[reval_agg["model_name"] == "baseline_c_attention"].iloc[0]
    standard = reval_agg[reval_agg["model_name"] == "standard_v1"].iloc[0]
    lines = [
        "# Transition Report",
        "",
        f"- Standard v1 config frozen: **Yes** ([standard_v1.yaml]({standard_snapshot_path}))",
        f"- Revalidation grade: **{reval_grade}**",
        f"- Revalidation decision: {reval_reason}",
        "",
        "## 5-Seed Revalidation",
        "",
        f"- baseline_c_attention: success={baseline['mean_eval_success_rate']:.3f} +/- {baseline['std_eval_success_rate']:.3f}, return={baseline['mean_eval_return']:.3f} +/- {baseline['std_eval_return']:.3f}, progress={baseline['mean_eval_avg_progress_to_goal']:.3f} +/- {baseline['std_eval_avg_progress_to_goal']:.3f}",
        f"- standard_v1: success={standard['mean_eval_success_rate']:.3f} +/- {standard['std_eval_success_rate']:.3f}, return={standard['mean_eval_return']:.3f} +/- {standard['std_eval_return']:.3f}, progress={standard['mean_eval_avg_progress_to_goal']:.3f} +/- {standard['std_eval_avg_progress_to_goal']:.3f}",
        f"- Comparison vs baseline: success_diff={reval_compare.iloc[0]['success_diff_vs_baseline']:.3f}, return_diff={reval_compare.iloc[0]['return_diff_vs_baseline']:.3f}, progress_diff={reval_compare.iloc[0]['progress_diff_vs_baseline']:.3f}",
        "",
        "## Stage 1 Warm-up",
        "",
        f"- Executed: **{'Yes' if stage1_executed else 'No'}**",
    ]
    if stage1_executed and stage1_metrics is not None:
        row = stage1_metrics.iloc[0]
        lines.extend(
            [
                f"- mean_eval_success_rate={row['mean_eval_success_rate']:.3f}",
                f"- mean_eval_return={row['mean_eval_return']:.3f}",
                f"- mean_eval_avg_progress_to_goal={row['mean_eval_avg_progress_to_goal']:.3f}",
                f"- mean_reliability_loss={row['mean_reliability_loss']:.6f}",
                f"- reliability_nan_rate={row['reliability_nan_rate']:.3f}",
                f"- attention_reliability_correlation={row['mean_attention_reliability_correlation']:.3f}",
                "- Reliability head learning stability: NaN-free if reliability_nan_rate == 0",
                "- Stage 2 gate: check weak reliability values are non-collapsed and attention/reliability logs are stored",
            ]
        )
    else:
        lines.append("- Not executed because revalidation did not PASS.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])

    standard_path, standard_snapshot = register_standard_v1_config(config)
    summary_root = ensure_dir(output_root / "revalidation" / "summary")
    save_yaml(summary_root / "experiment_config_snapshot.yaml", config)

    revalidation_rows = run_multi_seed_revalidation(config, standard_path, resume=args.resume)
    reval_agg = aggregate_revalidation_results(revalidation_rows, summary_root)
    reval_compare = compare_standard_v1_vs_baseline_c(reval_agg)
    reval_compare.to_csv(summary_root / "comparison_against_baseline.csv", index=False)
    grade, reason = evaluate_revalidation_pass_fail(reval_compare)
    (summary_root / "revalidation_pass_fail_report.md").write_text(
        "\n".join(
            [
                "# Revalidation Pass/Fail Report",
                "",
                f"- Grade: **{grade}**",
                f"- Reason: {reason}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    revalidation_df = pd.DataFrame(revalidation_rows)
    save_attention_reliability_logs(revalidation_df, summary_root / "attention_hidden_summary.csv")
    plot_learning_curves(revalidation_rows, summary_root / "revalidation_learning_curve_overview.png")

    stage1_metrics = None
    stage1_executed = False
    if grade == "PASS":
        stage1_rows = run_stage1_warmup(config, resume=args.resume)
        stage1_summary_dir = ensure_dir(output_root / "stage1_warmup" / "summary")
        save_yaml(stage1_summary_dir / "experiment_config_snapshot.yaml", config)
        stage1_metrics = aggregate_stage1_results(stage1_rows, stage1_summary_dir)
        stage1_executed = True

    final_summary_dir = ensure_dir(output_root / "summary")
    # Copy canonical outputs for convenience.
    for src_name in [
        "revalidation_comparison_metrics.csv",
        "revalidation_comparison_metrics.json",
        "revalidation_pass_fail_report.md",
        "revalidation_learning_curve_overview.png",
        "attention_hidden_summary.csv",
        "experiment_config_snapshot.yaml",
    ]:
        src = summary_root / src_name
        if src.exists():
            (final_summary_dir / src_name).write_bytes(src.read_bytes())
    if stage1_executed:
        stage1_summary_dir = output_root / "stage1_warmup" / "summary"
        for src_name in [
            "stage1_warmup_metrics.csv",
            "stage1_warmup_metrics.json",
            "stage1_attention_reliability_summary.csv",
        ]:
            src = stage1_summary_dir / src_name
            if src.exists():
                (final_summary_dir / src_name).write_bytes(src.read_bytes())
    save_yaml(final_summary_dir / "experiment_config_snapshot.yaml", config)
    generate_final_transition_report(
        standard_snapshot_path=standard_path,
        reval_agg=reval_agg,
        reval_compare=reval_compare,
        reval_grade=grade,
        reval_reason=reason,
        stage1_metrics=stage1_metrics,
        stage1_executed=stage1_executed,
        output_path=final_summary_dir / "final_transition_report.md",
    )
    print(f"Standard v1 config: {standard_path}")
    print(f"Revalidation grade: {grade}")
    if stage1_executed:
        print("Stage 1 warm-up executed.")
    else:
        print("Stage 1 warm-up skipped.")


if __name__ == "__main__":
    main()
