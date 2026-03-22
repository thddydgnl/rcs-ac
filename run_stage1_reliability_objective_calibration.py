"""Objective-focused reliability calibration sweep after corruption-path fix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from run_very_weak_probe import (
    build_model,
    build_very_weak_corruption_configs,
    compare_clean_vs_corrupted_attention,
    compare_clean_vs_corrupted_reliability,
    ensure_dir,
    generate_weak_probe_report,
    latest_checkpoint,
    plot_probe_summary,
    run_clean_eval,
    run_weak_corruption_eval,
    save_yaml,
)
from stage1_warmup.config import load_experiment_config
from stage1_warmup.trainer import Stage1WarmupTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 reliability objective calibration sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage1_reliability_objective_calibration.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def deep_apply_overrides(config_obj: Any, overrides: dict[str, Any]) -> None:
    for section_name, section_values in overrides.items():
        target = getattr(config_obj, section_name)
        for key, value in section_values.items():
            setattr(target, key, value)


def build_calibration_objective_candidates(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(config["candidates"])


def fine_tune_with_targeted_calibration_objective(
    candidate_name: str,
    candidate_spec: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if candidate_spec.get("use_existing_checkpoint", False):
        return
    output_root = ensure_dir(Path(config["output_root"]) / candidate_name)
    base_checkpoint_root = Path(config["base_checkpoint_root"])
    for seed in config["seeds"]:
        run_dir = ensure_dir(output_root / f"seed_{seed}")
        if (run_dir / "run_status.json").exists():
            continue
        cfg = load_experiment_config(config["base_stage1_config_path"])
        cfg.seed = seed
        cfg.device = config.get("device", "cpu")
        cfg.env.corruption_mode = "clean"
        cfg.logging.output_dir = str(run_dir)
        cfg.logging.run_name = f"{candidate_name}_seed_{seed}"
        deep_apply_overrides(cfg, config.get("base_config_overrides", {}))
        cfg.ppo.total_steps = int(config["train"]["total_steps"])
        cfg.logging.eval_interval_updates = int(config["train"]["eval_interval_updates"])
        cfg.logging.save_interval_updates = int(config["train"]["save_interval_updates"])
        cfg.warmup.init_checkpoint_path = str(latest_checkpoint(base_checkpoint_root / f"seed_{seed}"))
        deep_apply_overrides(cfg, candidate_spec.get("overrides", {}))
        trainer = Stage1WarmupTrainer(cfg)
        trainer.train()
        (run_dir / "candidate_config_snapshot.json").write_text(
            json.dumps(asdict(cfg), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


def run_targeted_probe_suite(
    candidate_name: str,
    candidate_spec: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    output_root = ensure_dir(Path(config["output_root"]) / candidate_name)
    checkpoint_root = Path(config["base_checkpoint_root"]) if candidate_spec.get("use_existing_checkpoint", False) else output_root
    probes = build_very_weak_corruption_configs(config)
    rows = []
    device = config.get("device", "cpu")
    for probe_name, probe_cfg in probes.items():
        probe_dir = ensure_dir(output_root / probe_name)
        for seed in config["seeds"]:
            seed_dir = ensure_dir(probe_dir / f"seed_{seed}")
            summary_path = seed_dir / "eval_summary.json"
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            checkpoint = latest_checkpoint(checkpoint_root / f"seed_{seed}")
            torch_device = torch.device(device)
            config_snapshot_path = checkpoint_root / f"seed_{seed}" / "candidate_config_snapshot.json"
            if config_snapshot_path.exists():
                stage1_config_source = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
            else:
                stage1_config_source = config["base_stage1_config_path"]
            model = build_model(stage1_config_source, checkpoint, device=torch_device)
            if probe_name == "clean":
                row = run_clean_eval(model, torch_device, seed, int(config.get("num_eval_episodes", 20)))
            else:
                row = run_weak_corruption_eval(
                    model=model,
                    device=torch_device,
                    seed=seed,
                    probe_name=probe_name,
                    probe_cfg=probe_cfg,
                    num_eval_episodes=int(config.get("num_eval_episodes", 20)),
                )
            row["candidate_name"] = candidate_name
            row["config_name"] = candidate_name
            row["probe_name"] = probe_name
            row["round"] = candidate_spec.get("round", "")
            (seed_dir / "eval_summary.json").write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")
            rows.append(row)
    return rows


def compute_target_drop_margin_metrics(reliability_cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corruption_type in sorted(reliability_cmp["corruption_type"].unique()):
        if corruption_type == "clean":
            continue
        target_modality = corruption_type.split("_")[-1]
        sub = reliability_cmp[reliability_cmp["corruption_type"] == corruption_type].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("delta_clean_minus_weak", ascending=False).reset_index(drop=True)
        target_row = sub[sub["modality"] == target_modality]
        if target_row.empty:
            continue
        target_delta = float(target_row.iloc[0]["delta_clean_minus_weak"])
        rank = int(sub["delta_clean_minus_weak"].rank(ascending=False, method="min")[sub["modality"] == target_modality].iloc[0])
        rows.append(
            {
                "corruption_type": corruption_type,
                "target_modality": target_modality,
                "target_delta_clean_minus_probe": target_delta,
                "target_relative_drop_ratio": float(target_row.iloc[0]["relative_drop_ratio"]),
                "rank_among_modalities": rank,
                "target_positive_flag": int(target_delta > 0.0),
                "target_top1_flag": int(rank == 1),
                "target_top2_flag": int(rank <= 2),
            }
        )
    return pd.DataFrame(rows)


def compute_target_vs_nontarget_ordering_metrics(reliability_cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corruption_type in sorted(reliability_cmp["corruption_type"].unique()):
        if corruption_type == "clean":
            continue
        target_modality = corruption_type.split("_")[-1]
        sub = reliability_cmp[reliability_cmp["corruption_type"] == corruption_type]
        if sub.empty:
            continue
        target = float(sub[sub["modality"] == target_modality]["delta_clean_minus_weak"].iloc[0])
        non_target_mean = float(sub[sub["modality"] != target_modality]["delta_clean_minus_weak"].mean())
        rows.append(
            {
                "corruption_type": corruption_type,
                "target_modality": target_modality,
                "target_minus_nontarget_mean_delta": target - non_target_mean,
            }
        )
    return pd.DataFrame(rows)


def compute_monotonicity_metrics(target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    modalities = sorted(set(target_df["target_modality"]))
    for modality in modalities:
        weak_name = f"weak_{modality}"
        mild_name = f"mild_{modality}"
        weak_row = target_df[target_df["corruption_type"] == weak_name]
        mild_row = target_df[target_df["corruption_type"] == mild_name]
        if weak_row.empty or mild_row.empty:
            continue
        weak_drop = float(weak_row.iloc[0]["target_delta_clean_minus_probe"])
        mild_drop = float(mild_row.iloc[0]["target_delta_clean_minus_probe"])
        rows.append(
            {
                "target_modality": modality,
                "weak_drop": weak_drop,
                "mild_drop": mild_drop,
                "weak_to_mild_monotonicity_flag": int(mild_drop > weak_drop and mild_drop > 0.0),
            }
        )
    return pd.DataFrame(rows)


def aggregate_candidate_scores(
    candidate_name: str,
    candidate_df: pd.DataFrame,
    reliability_cmp: pd.DataFrame,
    reference_clean: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    target_df = compute_target_drop_margin_metrics(reliability_cmp)
    ordering_df = compute_target_vs_nontarget_ordering_metrics(reliability_cmp)
    monotonicity_df = compute_monotonicity_metrics(target_df)
    clean = candidate_df[candidate_df["corruption_type"] == "clean"]
    clean_success = float(clean["eval_success_rate"].mean()) if not clean.empty else 0.0
    clean_return = float(clean["eval_return"].mean()) if not clean.empty else 0.0
    clean_progress = float(clean["eval_avg_progress_to_goal"].mean()) if not clean.empty else 0.0
    clean_route_dom = float(clean["route_attention_dominance_ratio"].mean()) if "route_attention_dominance_ratio" in clean else 0.0
    clean_success_drop = max(0.0, reference_clean["eval_success_rate"] - clean_success)
    clean_return_drop = max(0.0, reference_clean["eval_return"] - clean_return)
    clean_progress_drop = max(0.0, reference_clean["eval_avg_progress_to_goal"] - clean_progress)
    nan_or_crash = float(candidate_df["reliability_nan_detected"].max() > 0)
    positive_ratio = float(target_df["target_positive_flag"].mean()) if not target_df.empty else 0.0
    top1_ratio = float(target_df["target_top1_flag"].mean()) if not target_df.empty else 0.0
    top2_ratio = float(target_df["target_top2_flag"].mean()) if not target_df.empty else 0.0
    monotonicity_ratio = float(monotonicity_df["weak_to_mild_monotonicity_flag"].mean()) if not monotonicity_df.empty else 0.0
    ordering_ratio = float((ordering_df["target_minus_nontarget_mean_delta"] > 0.0).mean()) if not ordering_df.empty else 0.0
    score = (
        5.0 * positive_ratio
        + 2.0 * monotonicity_ratio
        + 1.5 * ordering_ratio
        + 1.0 * top1_ratio
        + 0.5 * top2_ratio
        - 2.0 * clean_success_drop
        - 1.0 * clean_return_drop
        - 1.0 * clean_progress_drop
        - 0.05 * clean_route_dom
        - 10.0 * nan_or_crash
    )
    summary = {
        "candidate_name": candidate_name,
        "clean_success_rate": clean_success,
        "clean_return": clean_return,
        "clean_progress": clean_progress,
        "clean_route_dominance_ratio": clean_route_dom,
        "target_positive_ratio": positive_ratio,
        "target_top1_ratio": top1_ratio,
        "target_top2_ratio": top2_ratio,
        "weak_to_mild_monotonicity_ratio": monotonicity_ratio,
        "target_vs_nontarget_ordering_ratio": ordering_ratio,
        "clean_success_drop_vs_reference": clean_success_drop,
        "clean_return_drop_vs_reference": clean_return_drop,
        "clean_progress_drop_vs_reference": clean_progress_drop,
        "nan_or_crash": nan_or_crash,
        "calibration_score": score,
    }
    return summary, target_df, monotonicity_df


def evaluate_pass_fail(summary: dict[str, Any]) -> str:
    if summary["nan_or_crash"] > 0:
        return "FAIL"
    if summary["target_positive_ratio"] == 0.0:
        return "FAIL"
    if summary["target_positive_ratio"] >= 0.25 and summary["weak_to_mild_monotonicity_ratio"] >= 0.5 and summary["clean_success_drop_vs_reference"] <= 0.10:
        return "PASS"
    return "WARN"


def generate_final_calibration_report(
    score_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
    secondary: str,
) -> None:
    lines = [
        "# Stage 1 Reliability Objective Calibration Report",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        score_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "calibration_objective_config_snapshot.yaml", config)

    candidates = build_calibration_objective_candidates(config)
    all_probe_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    reference_name = config["reference_candidate_name"]
    reference_spec = candidates[reference_name]
    reference_rows = run_targeted_probe_suite(reference_name, reference_spec, config)
    all_probe_rows.extend(reference_rows)
    reference_df = pd.DataFrame(reference_rows)
    reference_clean = (
        reference_df[reference_df["corruption_type"] == "clean"][
            ["eval_success_rate", "eval_return", "eval_avg_progress_to_goal"]
        ]
        .mean()
        .to_dict()
    )

    ordered_candidates = list(config.get("candidate_order", candidates.keys()))
    if reference_name not in ordered_candidates:
        ordered_candidates = [reference_name, *ordered_candidates]

    for candidate_name in ordered_candidates:
        candidate_spec = candidates[candidate_name]
        fine_tune_with_targeted_calibration_objective(candidate_name, candidate_spec, config)
        probe_rows = reference_rows if candidate_name == reference_name else run_targeted_probe_suite(candidate_name, candidate_spec, config)
        if candidate_name != reference_name:
            all_probe_rows.extend(probe_rows)
        candidate_df = pd.DataFrame(probe_rows)
        reliability_cmp = compare_clean_vs_corrupted_reliability(candidate_df)
        attention_cmp = compare_clean_vs_corrupted_attention(candidate_df)
        perf_df = (
            candidate_df.groupby("corruption_type", as_index=False)[
                ["eval_return", "eval_success_rate", "eval_collision_rate", "eval_timeout_rate", "eval_avg_progress_to_goal"]
            ]
            .mean()
            .sort_values("corruption_type")
        )
        summary, target_df, monotonicity_df = aggregate_candidate_scores(
            candidate_name=candidate_name,
            candidate_df=candidate_df,
            reliability_cmp=reliability_cmp,
            reference_clean=reference_clean,
        )
        summary["round"] = candidate_spec.get("round", "")
        summary["pass_fail"] = evaluate_pass_fail(summary)
        candidate_summaries.append(summary)

        target_df = target_df.copy()
        target_df["candidate_name"] = candidate_name
        monotonicity_df = monotonicity_df.copy()
        monotonicity_df["candidate_name"] = candidate_name
        ordering_df = compute_target_vs_nontarget_ordering_metrics(reliability_cmp).copy()
        ordering_df["candidate_name"] = candidate_name
        comparison_rows.extend(target_df.to_dict("records"))
        comparison_rows.extend(monotonicity_df.to_dict("records"))
        comparison_rows.extend(ordering_df.to_dict("records"))

        candidate_summary_dir = ensure_dir(output_root / candidate_name / "summary")
        candidate_df.to_csv(candidate_summary_dir / "probe_metrics.csv", index=False)
        reliability_cmp.to_csv(candidate_summary_dir / "reliability_delta.csv", index=False)
        attention_cmp.to_csv(candidate_summary_dir / "attention_delta.csv", index=False)
        target_df.to_csv(candidate_summary_dir / "target_drop_metrics.csv", index=False)
        monotonicity_df.to_csv(candidate_summary_dir / "monotonicity_metrics.csv", index=False)
        pd.DataFrame([summary]).to_csv(candidate_summary_dir / "score_summary.csv", index=False)
        plot_probe_summary(perf_df, candidate_summary_dir / "probe_summary.png")
        generate_weak_probe_report(
            perf_df=perf_df,
            reliability_df=reliability_cmp,
            attention_df=attention_cmp,
            ranking_df=target_df,
            verdict=summary["pass_fail"],
            reason="objective calibration candidate result",
            output_path=candidate_summary_dir / "probe_report.md",
        )

    all_probe_df = pd.DataFrame(all_probe_rows).sort_values(["candidate_name", "corruption_type", "seed"]).reset_index(drop=True)
    all_probe_df.to_csv(summary_dir / "calibration_objective_sweep_metrics.csv", index=False)
    all_probe_df.to_json(summary_dir / "calibration_objective_sweep_metrics.json", orient="records", indent=2)

    score_df = pd.DataFrame(candidate_summaries).sort_values(
        ["pass_fail", "target_positive_ratio", "calibration_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    # Enforce explicit PASS > WARN > FAIL sort order.
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        ["_order", "target_positive_ratio", "weak_to_mild_monotonicity_ratio", "calibration_score"],
        ascending=[True, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    score_df.to_csv(summary_dir / "calibration_objective_score_summary.csv", index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(summary_dir / "calibration_attention_reliability_comparison.csv", index=False)

    recommended = score_df.iloc[0]["candidate_name"]
    secondary = score_df.iloc[1]["candidate_name"] if len(score_df) > 1 else recommended

    pass_report_lines = [
        "# Calibration Objective PASS/WARN/FAIL Report",
        "",
        score_df.to_markdown(index=False),
        "",
    ]
    (summary_dir / "calibration_objective_pass_fail_report.md").write_text("\n".join(pass_report_lines) + "\n", encoding="utf-8")
    generate_final_calibration_report(
        score_df=score_df,
        output_path=summary_dir / "calibration_objective_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
    )

    print("Recommended:", recommended)
    print("Secondary:", secondary)


if __name__ == "__main__":
    main()
