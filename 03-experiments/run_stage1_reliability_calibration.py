"""Small calibration sweep for reliability sensitivity before Stage 2."""

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

from debug_corruption_eval_path import (
    compare_train_vs_eval_corruption_path,
    evaluate_corruption_debug_pass_fail,
    first_episode_metric_sanity,
    trace_probe_config_to_env,
    validate_corruption_metadata_consistency,
    compare_clean_vs_probe_observations,
    dump_raw_observations_for_probe,
)
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
    parser = argparse.ArgumentParser(description="Calibrate Stage 1 reliability sensitivity.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage1_reliability_calibration.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_corruption_debug_gate(config: dict[str, Any]) -> dict[str, Any]:
    debug_cfg_path = config.get(
        "corruption_debug_config_path",
        "/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/corruption_debug.yaml",
    )
    debug_cfg = load_yaml(debug_cfg_path)
    trace_df = trace_probe_config_to_env(debug_cfg)
    clean_dump = dump_raw_observations_for_probe(
        probe_name="clean",
        probe_cfg=debug_cfg["probes"]["clean"],
        seed=int(debug_cfg["debug_seed"]),
        num_steps=int(debug_cfg.get("debug_steps", 4)),
    )
    diff_frames = []
    for probe_name, probe_cfg in debug_cfg["probes"].items():
        if probe_name == "clean":
            continue
        probe_dump = dump_raw_observations_for_probe(
            probe_name=probe_name,
            probe_cfg=probe_cfg,
            seed=int(debug_cfg["debug_seed"]),
            num_steps=int(debug_cfg.get("debug_steps", 4)),
        )
        diff_frames.append(compare_clean_vs_probe_observations(clean_dump, probe_dump))
    obs_diff_df = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame()
    consistency_df = validate_corruption_metadata_consistency(trace_df, obs_diff_df)
    metric_df = first_episode_metric_sanity(debug_cfg, trace_df)
    train_eval_df = compare_train_vs_eval_corruption_path(debug_cfg, int(debug_cfg["debug_seed"]))
    verdict, reason, go_no_go = evaluate_corruption_debug_pass_fail(
        consistency_df=consistency_df,
        train_eval_df=train_eval_df,
        metric_df=metric_df,
        train_eval_l2_tolerance=float(debug_cfg.get("train_eval_l2_tolerance", 1e-6)),
    )
    return {
        "verdict": verdict,
        "reason": reason,
        "go_no_go_for_calibration": int(go_no_go),
        "mean_train_eval_l2": float(train_eval_df["train_eval_path_diff_l2"].mean()) if not train_eval_df.empty else 0.0,
    }


def deep_apply_overrides(config_obj: Any, overrides: dict[str, Any]) -> None:
    for section_name, section_values in overrides.items():
        target = getattr(config_obj, section_name)
        for key, value in section_values.items():
            setattr(target, key, value)


def run_candidate_finetune(
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
        cfg.ppo.total_steps = int(config["train"]["total_steps"])
        cfg.logging.eval_interval_updates = int(config["train"]["eval_interval_updates"])
        cfg.logging.save_interval_updates = int(config["train"]["save_interval_updates"])
        cfg.warmup.init_checkpoint_path = str(latest_checkpoint(base_checkpoint_root / f"seed_{seed}"))
        deep_apply_overrides(cfg, candidate_spec.get("overrides", {}))
        trainer = Stage1WarmupTrainer(cfg)
        trainer.train()
        (run_dir / "candidate_config_snapshot.json").write_text(json.dumps(asdict(cfg), ensure_ascii=True, indent=2), encoding="utf-8")


def run_probe_for_candidate(candidate_name: str, candidate_spec: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
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
                    model,
                    torch_device,
                    seed,
                    probe_name,
                    probe_cfg,
                    int(config.get("num_eval_episodes", 20)),
                )
            row["candidate_name"] = candidate_name
            row["round"] = candidate_spec.get("round", "")
            (seed_dir / "eval_summary.json").write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")
            rows.append(row)
    return rows


def sensitivity_ranking(reliability_cmp: pd.DataFrame) -> pd.DataFrame:
    target_map = {
        "vision": "vision",
        "distance": "distance",
        "ego": "ego",
        "route": "route",
    }
    rows = []
    for corruption_type in reliability_cmp["corruption_type"].unique():
        if corruption_type == "clean":
            continue
        target = corruption_type.split("_")[-1]
        if target not in target_map:
            continue
        sub = reliability_cmp[reliability_cmp["corruption_type"] == corruption_type]
        target_row = sub[sub["modality"] == target]
        if target_row.empty:
            continue
        rows.append(
            {
                "corruption_type": corruption_type,
                "target_modality": target,
                "target_delta_clean_minus_probe": float(target_row.iloc[0]["delta_clean_minus_weak"]),
                "target_relative_drop_ratio": float(target_row.iloc[0]["relative_drop_ratio"]),
                "rank_among_modalities": int(
                    sub["delta_clean_minus_weak"].rank(ascending=False, method="min")[sub["modality"] == target].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_delta_clean_minus_probe"], ascending=False).reset_index(drop=True)


def calibration_score_summary(candidate_df: pd.DataFrame, reliability_cmp: pd.DataFrame) -> dict[str, float]:
    clean = candidate_df[candidate_df["corruption_type"] == "clean"]
    ranks = sensitivity_ranking(reliability_cmp)
    targeted_positive = float((ranks["target_delta_clean_minus_probe"] > 0.005).mean()) if not ranks.empty else 0.0
    targeted_top2 = float((ranks["rank_among_modalities"] <= 2).mean()) if not ranks.empty else 0.0
    clean_success = float(clean["eval_success_rate"].mean()) if not clean.empty else 0.0
    clean_route_dom = float(clean["attention_mean_route"].mean() / max(clean[["attention_mean_vision", "attention_mean_distance", "attention_mean_ego"]].mean(axis=1).mean(), 1e-6)) if not clean.empty else 0.0
    nan_or_crash = float((candidate_df["reliability_nan_detected"].max() > 0) or False)
    score = targeted_positive * 2.0 + targeted_top2 * 2.0 + clean_success - 0.1 * clean_route_dom - 5.0 * nan_or_crash
    return {
        "clean_success_rate": clean_success,
        "targeted_positive_ratio": targeted_positive,
        "targeted_top2_ratio": targeted_top2,
        "clean_route_dominance_ratio": clean_route_dom,
        "nan_or_crash": nan_or_crash,
        "calibration_score": score,
    }


def evaluate_candidate_pass_fail(summary: dict[str, float]) -> str:
    if summary["nan_or_crash"] > 0:
        return "FAIL"
    if summary["targeted_positive_ratio"] >= 0.6 and summary["targeted_top2_ratio"] >= 0.6 and summary["clean_success_rate"] >= 0.7:
        return "PASS"
    if summary["targeted_positive_ratio"] >= 0.3:
        return "WARN"
    return "FAIL"


def generate_calibration_report(rank_df: pd.DataFrame, output_path: Path, recommended: str, secondary: str) -> None:
    lines = [
        "# Stage 1 Reliability Calibration Report",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        rank_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "experiment_config_snapshot.yaml", config)

    debug_gate = run_corruption_debug_gate(config)
    (summary_dir / "corruption_debug_gate.json").write_text(
        json.dumps(debug_gate, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    if debug_gate["go_no_go_for_calibration"] != 1:
        pd.DataFrame([debug_gate]).to_csv(summary_dir / "candidate_calibration_score_summary.csv", index=False)
        generate_calibration_report(
            pd.DataFrame([debug_gate]),
            summary_dir / "final_calibration_report.md",
            recommended="none",
            secondary="none",
        )
        print("Calibration blocked by corruption debug gate.")
        print(debug_gate["reason"])
        return

    def evaluate_candidate(candidate_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidate_spec = config["candidates"][candidate_name]
        run_candidate_finetune(candidate_name, candidate_spec, config)
        probe_rows = run_probe_for_candidate(candidate_name, candidate_spec, config)
        candidate_df = pd.DataFrame(probe_rows)
        reliability_cmp = compare_clean_vs_corrupted_reliability(candidate_df)
        attention_cmp = compare_clean_vs_corrupted_attention(candidate_df)
        ranking_df = sensitivity_ranking(reliability_cmp)
        perf_df = (
            candidate_df.groupby("corruption_type", as_index=False)[
                ["eval_return", "eval_success_rate", "eval_collision_rate", "eval_timeout_rate", "eval_avg_progress_to_goal"]
            ]
            .mean()
            .sort_values("corruption_type")
        )
        candidate_summary = calibration_score_summary(candidate_df, reliability_cmp)
        candidate_summary["candidate_name"] = candidate_name
        candidate_summary["round"] = candidate_spec.get("round", "")
        candidate_summary["pass_fail"] = evaluate_candidate_pass_fail(candidate_summary)
        candidate_summary["corruption_debug_verdict"] = debug_gate["verdict"]
        candidate_summary["go_no_go_for_calibration"] = debug_gate["go_no_go_for_calibration"]

        candidate_summary_dir = ensure_dir(output_root / candidate_name / "summary")
        candidate_df.to_csv(candidate_summary_dir / "weak_probe_metrics.csv", index=False)
        reliability_cmp.to_csv(candidate_summary_dir / "clean_vs_probe_reliability.csv", index=False)
        attention_cmp.to_csv(candidate_summary_dir / "clean_vs_probe_attention.csv", index=False)
        ranking_df.to_csv(candidate_summary_dir / "corruption_sensitivity_ranking.csv", index=False)
        pd.DataFrame([candidate_summary]).to_csv(candidate_summary_dir / "calibration_score_summary.csv", index=False)
        plot_probe_summary(perf_df, candidate_summary_dir / "probe_summary.png")
        generate_weak_probe_report(
            perf_df=perf_df,
            reliability_df=reliability_cmp,
            attention_df=attention_cmp,
            ranking_df=ranking_df,
            verdict=candidate_summary["pass_fail"],
            reason="candidate calibration sweep result",
            output_path=candidate_summary_dir / "probe_report.md",
        )
        return probe_rows, candidate_summary

    all_probe_rows = []
    candidate_summaries = []
    initial_candidates = list(config.get("initial_candidate_order", []))
    deferred_candidates = list(config.get("deferred_candidate_order", []))
    if not initial_candidates:
        initial_candidates = list(config["candidates"].keys())

    pass_found = False
    for candidate_name in initial_candidates:
        probe_rows, candidate_summary = evaluate_candidate(candidate_name)
        all_probe_rows.extend(probe_rows)
        candidate_summaries.append(candidate_summary)
        if candidate_summary["pass_fail"] == "PASS":
            pass_found = True

    if not pass_found:
        for candidate_name in deferred_candidates:
            if candidate_name in initial_candidates:
                continue
            probe_rows, candidate_summary = evaluate_candidate(candidate_name)
            all_probe_rows.extend(probe_rows)
            candidate_summaries.append(candidate_summary)

    all_probe_df = pd.DataFrame(all_probe_rows)
    all_probe_df.to_csv(summary_dir / "all_candidate_probe_metrics.csv", index=False)
    rank_df = pd.DataFrame(candidate_summaries).sort_values(["calibration_score", "clean_success_rate"], ascending=False).reset_index(drop=True)
    rank_df.to_csv(summary_dir / "candidate_calibration_score_summary.csv", index=False)
    recommended = rank_df.iloc[0]["candidate_name"]
    secondary = rank_df.iloc[1]["candidate_name"] if len(rank_df) > 1 else recommended
    generate_calibration_report(rank_df, summary_dir / "final_calibration_report.md", recommended, secondary)
    print("Recommended:", recommended)
    print("Secondary:", secondary)


if __name__ == "__main__":
    main()
