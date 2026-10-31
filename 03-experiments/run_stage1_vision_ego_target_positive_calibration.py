"""Vision/Ego target-positive calibration sweep on top of the ranking reference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from run_stage1_targeted_monotonicity_calibration import deep_apply_overrides
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
    parser = argparse.ArgumentParser(description="Run Stage 1 vision/ego target-positive calibration sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage1_vision_ego_target_positive_calibration.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_target_positive_candidates(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(config["candidates"])


def fine_tune_reference_calibration(candidate_name: str, candidate_spec: dict[str, Any], config: dict[str, Any]) -> None:
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


def run_probe_suite(candidate_name: str, candidate_spec: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = ensure_dir(Path(config["output_root"]) / candidate_name)
    checkpoint_root = Path(config["base_checkpoint_root"]) if candidate_spec.get("use_existing_checkpoint", False) else output_root
    probes = build_very_weak_corruption_configs(config)
    rows: list[dict[str, Any]] = []
    device = config.get("device", "cpu")
    for probe_name, probe_cfg in probes.items():
        probe_dir = ensure_dir(output_root / probe_name)
        for seed in config["seeds"]:
            seed_dir = ensure_dir(probe_dir / f"seed_{seed}")
            summary_path = seed_dir / "eval_summary.json"
            if summary_path.exists():
                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                if cached.get("probe_name") == probe_name and cached.get("corruption_type") == probe_name:
                    rows.append(cached)
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


def compute_positive_drop_metrics(reliability_cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probe_name in sorted(reliability_cmp["corruption_type"].unique()):
        if probe_name == "clean":
            continue
        target_modality = probe_name.split("_")[-1]
        sub = reliability_cmp[reliability_cmp["corruption_type"] == probe_name].copy()
        sub = sub.sort_values("delta_clean_minus_weak", ascending=False).reset_index(drop=True)
        target_row = sub[sub["modality"] == target_modality]
        if target_row.empty:
            continue
        delta = float(target_row.iloc[0]["delta_clean_minus_weak"])
        non_target_mean = float(sub[sub["modality"] != target_modality]["delta_clean_minus_weak"].mean())
        rows.append(
            {
                "corruption_type": probe_name,
                "target_modality": target_modality,
                "target_delta_clean_minus_probe": delta,
                "target_delta_clean_minus_weak": delta if probe_name.startswith("weak_") else 0.0,
                "target_delta_clean_minus_mild": delta if probe_name.startswith("mild_") else 0.0,
                "weak_to_mild_delta": 0.0,
                "target_positive_flag": int(delta > 0.0),
                "weak_to_mild_directional_flag": 0,
                "weak_to_mild_margin_flag": 0,
                "vision_specific_positive_flag": int(target_modality == "vision" and delta > 0.0),
                "ego_specific_positive_flag": int(target_modality == "ego" and delta > 0.0),
                "target_vs_nontarget_gap": float(delta - non_target_mean),
                "saturation_warning_flag": 0,
            }
        )
    return pd.DataFrame(rows)


def compute_directional_monotonicity_metrics(target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality in sorted(set(target_df["target_modality"])):
        weak_row = target_df[target_df["corruption_type"] == f"weak_{modality}"]
        mild_row = target_df[target_df["corruption_type"] == f"mild_{modality}"]
        if weak_row.empty or mild_row.empty:
            continue
        weak_drop = float(weak_row.iloc[0]["target_delta_clean_minus_probe"])
        mild_drop = float(mild_row.iloc[0]["target_delta_clean_minus_probe"])
        rows.append(
            {
                "target_modality": modality,
                "target_delta_clean_minus_weak": weak_drop,
                "target_delta_clean_minus_mild": mild_drop,
                "weak_to_mild_delta": float(mild_drop - weak_drop),
                "weak_to_mild_directional_flag": int(mild_drop > weak_drop),
            }
        )
    return pd.DataFrame(rows)


def compute_margin_monotonicity_metrics(direction_df: pd.DataFrame, margin: float) -> pd.DataFrame:
    if direction_df.empty:
        return pd.DataFrame(columns=["target_modality", "weak_to_mild_margin_flag"])
    out = direction_df[["target_modality", "weak_to_mild_delta"]].copy()
    out["weak_to_mild_margin_flag"] = (out["weak_to_mild_delta"] > margin).astype(int)
    return out


def _ratio_by_modality(df: pd.DataFrame, value_col: str, modalities: list[str]) -> dict[str, float]:
    result = {}
    for modality in modalities:
        sub = df[df["target_modality"] == modality]
        result[modality] = float(sub[value_col].mean()) if not sub.empty else 0.0
    return result


def aggregate_candidate_scores(
    candidate_name: str,
    candidate_df: pd.DataFrame,
    reliability_cmp: pd.DataFrame,
    config: dict[str, Any],
    reference_clean: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    target_df = compute_positive_drop_metrics(reliability_cmp)
    direction_df = compute_directional_monotonicity_metrics(target_df)
    margin_df = compute_margin_monotonicity_metrics(direction_df, float(config["scoring"]["weak_to_mild_margin_threshold"]))
    merged = target_df.merge(direction_df, on="target_modality", how="left", suffixes=("", "_dir"))
    merged = merged.merge(margin_df[["target_modality", "weak_to_mild_margin_flag"]], on="target_modality", how="left", suffixes=("", "_margin"))
    if "weak_to_mild_margin_flag_margin" in merged.columns:
        merged["weak_to_mild_margin_flag"] = merged["weak_to_mild_margin_flag_margin"].fillna(merged["weak_to_mild_margin_flag"])
        merged = merged.drop(columns=["weak_to_mild_margin_flag_margin"])
    if "weak_to_mild_delta_dir" in merged.columns:
        merged["weak_to_mild_delta"] = merged["weak_to_mild_delta_dir"]
        merged = merged.drop(columns=["weak_to_mild_delta_dir", "target_delta_clean_minus_weak_dir", "target_delta_clean_minus_mild_dir"], errors="ignore")
    merged["weak_to_mild_directional_flag"] = merged["weak_to_mild_directional_flag"].fillna(0).astype(int)
    merged["weak_to_mild_margin_flag"] = merged["weak_to_mild_margin_flag"].fillna(0).astype(int)

    modalities = ["vision", "distance", "ego", "route"]
    positive_ratios = _ratio_by_modality(merged, "target_positive_flag", modalities)
    directional_ratios = _ratio_by_modality(merged, "weak_to_mild_directional_flag", modalities)
    margin_ratios = _ratio_by_modality(merged, "weak_to_mild_margin_flag", modalities)
    target_positive_ratio = float(merged["target_positive_flag"].mean()) if not merged.empty else 0.0
    weak_to_mild_directional_ratio = float(merged["weak_to_mild_directional_flag"].mean()) if not merged.empty else 0.0
    weak_to_mild_margin_ratio = float(merged["weak_to_mild_margin_flag"].mean()) if not merged.empty else 0.0

    clean = candidate_df[candidate_df["probe_name"] == "clean"]
    clean_success = float(clean["eval_success_rate"].mean()) if not clean.empty else 0.0
    clean_return = float(clean["eval_return"].mean()) if not clean.empty else 0.0
    clean_progress = float(clean["eval_avg_progress_to_goal"].mean()) if not clean.empty else 0.0
    clean_rel_max = float(clean[[f"reliability_mean_{m}" for m in modalities]].mean().max()) if not clean.empty else 0.0
    saturation_warning_flag = float(clean_rel_max >= 0.98)
    clean_success_drop = max(0.0, reference_clean["eval_success_rate"] - clean_success)
    clean_return_drop = max(0.0, reference_clean["eval_return"] - clean_return)
    clean_progress_drop = max(0.0, reference_clean["eval_avg_progress_to_goal"] - clean_progress)
    score = (
        5.0 * positive_ratios["vision"]
        + 5.0 * positive_ratios["ego"]
        + 3.0 * target_positive_ratio
        + 2.0 * weak_to_mild_directional_ratio
        + 1.0 * weak_to_mild_margin_ratio
        + 2.0 * directional_ratios["vision"]
        + 2.0 * directional_ratios["ego"]
        - 2.0 * clean_success_drop
        - 1.0 * clean_return_drop
        - 1.0 * clean_progress_drop
        - 1.5 * saturation_warning_flag
    )
    summary = {
        "candidate_name": candidate_name,
        "target_positive_ratio": target_positive_ratio,
        "weak_to_mild_directional_ratio": weak_to_mild_directional_ratio,
        "weak_to_mild_margin_ratio": weak_to_mild_margin_ratio,
        "vision_positive_ratio": positive_ratios["vision"],
        "distance_positive_ratio": positive_ratios["distance"],
        "ego_positive_ratio": positive_ratios["ego"],
        "route_positive_ratio": positive_ratios["route"],
        "vision_directional_monotonicity_ratio": directional_ratios["vision"],
        "distance_directional_monotonicity_ratio": directional_ratios["distance"],
        "ego_directional_monotonicity_ratio": directional_ratios["ego"],
        "route_directional_monotonicity_ratio": directional_ratios["route"],
        "vision_margin_monotonicity_ratio": margin_ratios["vision"],
        "distance_margin_monotonicity_ratio": margin_ratios["distance"],
        "ego_margin_monotonicity_ratio": margin_ratios["ego"],
        "route_margin_monotonicity_ratio": margin_ratios["route"],
        "clean_success_rate": clean_success,
        "clean_return": clean_return,
        "clean_progress": clean_progress,
        "clean_success_drop_vs_reference": clean_success_drop,
        "clean_return_drop_vs_reference": clean_return_drop,
        "clean_progress_drop_vs_reference": clean_progress_drop,
        "saturation_warning_flag": saturation_warning_flag,
        "nan_or_crash": float(candidate_df["reliability_nan_detected"].max() > 0),
        "calibration_score": score,
    }
    return summary, merged


def evaluate_pass_fail(summary: dict[str, Any]) -> str:
    if summary["nan_or_crash"] > 0:
        return "FAIL"
    if summary["vision_positive_ratio"] == 0.0 and summary["ego_positive_ratio"] == 0.0:
        return "FAIL"
    if summary["weak_to_mild_directional_ratio"] <= 0.0:
        return "FAIL"
    if (
        (summary["vision_positive_ratio"] > 0.0 or summary["ego_positive_ratio"] > 0.0)
        and summary["weak_to_mild_directional_ratio"] > 0.0
        and (
            summary["vision_directional_monotonicity_ratio"] > 0.0
            or summary["ego_directional_monotonicity_ratio"] > 0.0
        )
        and summary["clean_success_drop_vs_reference"] <= 0.10
    ):
        return "PASS"
    return "WARN"


def generate_final_recommendation_report(score_df: pd.DataFrame, output_path: Path, recommended: str, secondary: str) -> None:
    lines = [
        "# Stage 1 Vision/Ego Target-Positive Calibration Report",
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
    save_yaml(summary_dir / "vision_ego_target_positive_config_snapshot.yaml", config)

    candidates = build_target_positive_candidates(config)
    ordered_candidates = list(config.get("candidate_order", candidates.keys()))
    reference_name = config["reference_candidate_name"]
    if reference_name not in ordered_candidates:
        ordered_candidates = [reference_name, *ordered_candidates]

    all_probe_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    reference_rows = run_probe_suite(reference_name, candidates[reference_name], config)
    all_probe_rows.extend(reference_rows)
    reference_df = pd.DataFrame(reference_rows)
    reference_clean = (
        reference_df[reference_df["probe_name"] == "clean"][
            ["eval_success_rate", "eval_return", "eval_avg_progress_to_goal"]
        ].mean().to_dict()
    )

    for candidate_name in ordered_candidates:
        candidate_spec = candidates[candidate_name]
        fine_tune_reference_calibration(candidate_name, candidate_spec, config)
        probe_rows = reference_rows if candidate_name == reference_name else run_probe_suite(candidate_name, candidate_spec, config)
        if candidate_name != reference_name:
            all_probe_rows.extend(probe_rows)
        candidate_df = pd.DataFrame(probe_rows)
        reliability_cmp = compare_clean_vs_corrupted_reliability(candidate_df)
        attention_cmp = compare_clean_vs_corrupted_attention(candidate_df)
        perf_df = (
            candidate_df.groupby("probe_name", as_index=False)[
                ["eval_return", "eval_success_rate", "eval_collision_rate", "eval_timeout_rate", "eval_avg_progress_to_goal"]
            ].mean().sort_values("probe_name").rename(columns={"probe_name": "corruption_type"})
        )
        summary, comparison_df = aggregate_candidate_scores(candidate_name, candidate_df, reliability_cmp, config, reference_clean)
        summary["round"] = candidate_spec.get("round", "")
        summary["pass_fail"] = evaluate_pass_fail(summary)
        candidate_summaries.append(summary)

        comparison_df = comparison_df.copy()
        comparison_df["candidate_name"] = candidate_name
        comparison_rows.extend(comparison_df.to_dict("records"))

        candidate_summary_dir = ensure_dir(output_root / candidate_name / "summary")
        candidate_df.to_csv(candidate_summary_dir / "probe_metrics.csv", index=False)
        reliability_cmp.to_csv(candidate_summary_dir / "reliability_delta.csv", index=False)
        attention_cmp.to_csv(candidate_summary_dir / "attention_delta.csv", index=False)
        comparison_df.to_csv(candidate_summary_dir / "vision_ego_target_positive_metrics.csv", index=False)
        pd.DataFrame([summary]).to_csv(candidate_summary_dir / "score_summary.csv", index=False)
        plot_probe_summary(perf_df, candidate_summary_dir / "probe_summary.png")
        generate_weak_probe_report(
            perf_df=perf_df,
            reliability_df=reliability_cmp,
            attention_df=attention_cmp,
            ranking_df=comparison_df,
            verdict=summary["pass_fail"],
            reason="vision/ego target-positive calibration candidate result",
            output_path=candidate_summary_dir / "probe_report.md",
        )

    all_probe_df = pd.DataFrame(all_probe_rows).sort_values(["candidate_name", "probe_name", "seed"]).reset_index(drop=True)
    all_probe_df.to_csv(summary_dir / "vision_ego_target_positive_sweep_metrics.csv", index=False)
    all_probe_df.to_json(summary_dir / "vision_ego_target_positive_sweep_metrics.json", orient="records", indent=2)

    score_df = pd.DataFrame(candidate_summaries)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "vision_positive_ratio",
            "ego_positive_ratio",
            "target_positive_ratio",
            "weak_to_mild_directional_ratio",
            "weak_to_mild_margin_ratio",
            "vision_directional_monotonicity_ratio",
            "ego_directional_monotonicity_ratio",
            "calibration_score",
        ],
        ascending=[True, False, False, False, False, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    score_df.to_csv(summary_dir / "vision_ego_target_positive_score_summary.csv", index=False)

    comparison_all_df = pd.DataFrame(comparison_rows)
    comparison_all_df.to_csv(summary_dir / "vision_ego_target_positive_attention_reliability_comparison.csv", index=False)

    recommended = score_df.iloc[0]["candidate_name"]
    secondary = score_df.iloc[1]["candidate_name"] if len(score_df) > 1 else recommended
    (summary_dir / "vision_ego_target_positive_pass_fail_report.md").write_text(
        "# Vision/Ego Target-Positive PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    generate_final_recommendation_report(
        score_df=score_df,
        output_path=summary_dir / "vision_ego_target_positive_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
    )
    print("Recommended:", recommended)
    print("Secondary:", secondary)


if __name__ == "__main__":
    main()
