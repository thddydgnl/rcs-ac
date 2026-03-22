"""Run separability debug first, then a single path-A or path-B calibration sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from run_stage1_vision_ego_feature_debug import load_yaml, run_feature_debug
from run_stage1_vision_ego_target_positive_calibration import (
    aggregate_candidate_scores,
    evaluate_pass_fail,
    fine_tune_reference_calibration,
    generate_final_recommendation_report,
    run_probe_suite,
)
from run_very_weak_probe import (
    compare_clean_vs_corrupted_attention,
    compare_clean_vs_corrupted_reliability,
    ensure_dir,
    generate_weak_probe_report,
    plot_probe_summary,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve Stage 1 vision/ego bottleneck by debug -> single path sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage1_vision_ego_bottleneck_resolution.yaml",
    )
    return parser.parse_args()


def _path_candidates(config: dict[str, Any], selected_path: str) -> dict[str, dict[str, Any]]:
    if selected_path == "A":
        return dict(config["path_a_candidates"])
    return dict(config["path_b_candidates"])


def _ordered_candidates(config: dict[str, Any], selected_path: str, candidates: dict[str, dict[str, Any]]) -> list[str]:
    key = "path_a_candidate_order" if selected_path == "A" else "path_b_candidate_order"
    ordered = list(config.get(key, list(candidates.keys())))
    reference_name = config["reference_candidate_name"]
    if reference_name not in ordered:
        ordered = [reference_name, *ordered]
    return ordered


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "vision_ego_bottleneck_resolution_config_snapshot.yaml", config)

    debug_config = load_yaml(config["feature_debug_config_path"])
    if "output_root" in config:
        debug_config["output_root"] = str(output_root / "feature_debug")
    summary_df, _head_input_df, _trace_json, decision = run_feature_debug(debug_config)
    selected_path = str(decision["selected_path"])
    candidates = _path_candidates(config, selected_path)
    ordered_candidates = _ordered_candidates(config, selected_path, candidates)

    all_probe_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    reference_name = config["reference_candidate_name"]

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
            ]
            .mean()
            .sort_values("probe_name")
            .rename(columns={"probe_name": "corruption_type"})
        )
        summary, comparison_df = aggregate_candidate_scores(candidate_name, candidate_df, reliability_cmp, config, reference_clean)
        summary["selected_path"] = selected_path
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
        comparison_df.to_csv(candidate_summary_dir / "comparison_metrics.csv", index=False)
        pd.DataFrame([summary]).to_csv(candidate_summary_dir / "score_summary.csv", index=False)
        plot_probe_summary(perf_df, candidate_summary_dir / "probe_summary.png")
        generate_weak_probe_report(
            perf_df=perf_df,
            reliability_df=reliability_cmp,
            attention_df=attention_cmp,
            ranking_df=comparison_df,
            verdict=summary["pass_fail"],
            reason=f"vision/ego bottleneck path {selected_path} candidate result",
            output_path=candidate_summary_dir / "probe_report.md",
        )

    all_probe_df = pd.DataFrame(all_probe_rows).sort_values(["candidate_name", "probe_name", "seed"]).reset_index(drop=True)
    all_probe_df.to_csv(summary_dir / "vision_ego_bottleneck_probe_metrics.csv", index=False)
    all_probe_df.to_json(summary_dir / "vision_ego_bottleneck_probe_metrics.json", orient="records", indent=2)

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
            "calibration_score",
        ],
        ascending=[True, False, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    score_df.to_csv(summary_dir / "vision_ego_bottleneck_score_summary.csv", index=False)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(summary_dir / "vision_ego_bottleneck_attention_reliability_comparison.csv", index=False)

    recommended = score_df.iloc[0]["candidate_name"]
    secondary = score_df.iloc[1]["candidate_name"] if len(score_df) > 1 else recommended
    generate_final_recommendation_report(
        score_df=score_df,
        output_path=summary_dir / "vision_ego_bottleneck_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
    )
    (summary_dir / "vision_ego_bottleneck_pass_fail_report.md").write_text(
        "# Vision/Ego Bottleneck PASS/WARN/FAIL Report\n\n"
        + f"- Selected path: **{selected_path}**\n"
        + f"- Reason: {decision['reason']}\n\n"
        + score_df.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    (summary_dir / "vision_ego_bottleneck_decision.json").write_text(
        json.dumps(
            {
                "selected_path": selected_path,
                "reason": decision["reason"],
                "recommended": recommended,
                "secondary": secondary,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
