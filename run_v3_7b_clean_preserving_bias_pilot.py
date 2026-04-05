"""Run Version 3.7b clean-preserving bias pilot."""

from __future__ import annotations

import argparse

from v3_shapley_helpers import (
    Stage37bCleanPreservingBiasTrainer,
    aggregate_stage2_results,
    build_v3_1b_targetlogit_summary,
    build_v3_7b_candidate_score_summary,
    choose_v3_recommendations,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    compute_v3_7b_clean_preserving_metrics,
    ensure_dir,
    generate_v3_7b_report,
    infer_experiment_mode,
    is_calibration_only,
    load_yaml,
    run_v3_candidate,
    save_v3_handoff_snapshot,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.7b clean-preserving bias pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_7b_clean_preserving_bias_pilot.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run_v3_7b_clean_preserving_bias_pilot(config: dict, smoke: bool = False) -> None:
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "v3_7b_clean_preserving_bias_pilot_config_snapshot.yaml", config)
    save_v3_handoff_snapshot(config, summary_dir, "v3_7b_clean_preserving_bias_pilot")

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    if smoke:
        ordered_candidates = ordered_candidates[:2]
    all_rows = []
    for candidate_name in ordered_candidates:
        rows = run_v3_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="v3_7b_pilot_eval_rows.json",
            smoke=smoke,
            trainer_cls=Stage37bCleanPreservingBiasTrainer,
        )
        all_rows.extend(rows)

    df = aggregate_stage2_results(all_rows, summary_dir, "v3_7b_clean_preserving_bias_pilot")
    degradation_df = compute_degradation_metrics(df)
    consistency_df = compute_consistency_metrics(df)
    tiebreak_df = compute_tiebreak_metrics(df)
    targetlogit_df = build_v3_1b_targetlogit_summary(df)
    clean_anchor_df = compute_v3_7b_clean_preserving_metrics(df)

    degradation_df.to_csv(summary_dir / "v3_7b_clean_preserving_bias_pilot_degradation_summary.csv", index=False)
    consistency_df.to_csv(summary_dir / "v3_7b_clean_preserving_bias_pilot_consistency_summary.csv", index=False)
    tiebreak_df.to_csv(summary_dir / "v3_7b_clean_preserving_bias_pilot_tiebreak_summary.csv", index=False)
    targetlogit_df.to_csv(summary_dir / "v3_7b_clean_preserving_bias_pilot_targetlogit_shapley_summary.csv", index=False)
    clean_anchor_df.to_csv(summary_dir / "v3_7b_clean_preserving_bias_pilot_clean_anchor_summary.csv", index=False)

    score_df = build_v3_7b_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        clean_anchor_df=clean_anchor_df,
        baseline_candidate=config["baseline_candidate"],
        tiebreak_weights=config["tiebreak_weights"],
    )
    score_df["experiment_mode"] = infer_experiment_mode(config["losses"])
    score_df["calibration_only_flag"] = float(is_calibration_only(config["losses"]))
    score_df.to_csv(summary_dir / "v3_7b_clean_preserving_bias_pilot_score_summary.csv", index=False)
    (summary_dir / "v3_7b_clean_preserving_bias_pilot_pass_fail_report.md").write_text(
        "# Version 3.7b Clean-Preserving Bias Pilot PASS/WARN/FAIL Report\n\n"
        + "> This run is calibration-only when `calibration_only_flag=1`.\n\n"
        + score_df.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    recommended, secondary = choose_v3_recommendations(
        score_df=score_df,
        baseline_candidate=config["baseline_candidate"],
    )
    generate_v3_7b_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        clean_anchor_df=clean_anchor_df,
        output_path=summary_dir / "v3_7b_clean_preserving_bias_pilot_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
        title="Version 3.7b Clean-Preserving Bias Pilot Report (Calibration-Aware)",
    )
    print("Recommended:", recommended)
    print("Secondary:", secondary)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_v3_7b_clean_preserving_bias_pilot(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
