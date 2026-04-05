"""Run Version 3.5 score-adapter pilot."""

from __future__ import annotations

import argparse

from v3_shapley_helpers import (
    Stage35AttentionAdapterTrainer,
    aggregate_stage2_results,
    build_v3_1b_targetlogit_summary,
    build_v3_5_candidate_score_summary,
    choose_v3_recommendations,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    compute_v3_5_adapter_metrics,
    ensure_dir,
    generate_v3_5_report,
    load_yaml,
    run_v3_candidate,
    save_v3_handoff_snapshot,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.5 score-adapter pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_5_adapter_pilot.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run_v3_5_adapter_pilot(config: dict, smoke: bool = False) -> None:
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "v3_5_adapter_pilot_config_snapshot.yaml", config)
    save_v3_handoff_snapshot(config, summary_dir, "v3_5_adapter_pilot")

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    if smoke:
        ordered_candidates = ordered_candidates[:2]
    all_rows = []
    for candidate_name in ordered_candidates:
        rows = run_v3_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="v3_5_pilot_eval_rows.json",
            smoke=smoke,
            trainer_cls=Stage35AttentionAdapterTrainer,
        )
        all_rows.extend(rows)

    df = aggregate_stage2_results(all_rows, summary_dir, "v3_5_adapter_pilot")
    degradation_df = compute_degradation_metrics(df)
    consistency_df = compute_consistency_metrics(df)
    tiebreak_df = compute_tiebreak_metrics(df)
    targetlogit_df = build_v3_1b_targetlogit_summary(df)
    adapter_df = compute_v3_5_adapter_metrics(df)

    degradation_df.to_csv(summary_dir / "v3_5_adapter_pilot_degradation_summary.csv", index=False)
    consistency_df.to_csv(summary_dir / "v3_5_adapter_pilot_consistency_summary.csv", index=False)
    tiebreak_df.to_csv(summary_dir / "v3_5_adapter_pilot_tiebreak_summary.csv", index=False)
    targetlogit_df.to_csv(summary_dir / "v3_5_adapter_pilot_targetlogit_shapley_summary.csv", index=False)
    adapter_df.to_csv(summary_dir / "v3_5_adapter_pilot_adapter_summary.csv", index=False)

    score_df = build_v3_5_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        adapter_df=adapter_df,
        baseline_candidate=config["baseline_candidate"],
        tiebreak_weights=config["tiebreak_weights"],
    )
    score_df.to_csv(summary_dir / "v3_5_adapter_pilot_score_summary.csv", index=False)
    (summary_dir / "v3_5_adapter_pilot_pass_fail_report.md").write_text(
        "# Version 3.5 Adapter Pilot PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    recommended, secondary = choose_v3_recommendations(
        score_df=score_df,
        baseline_candidate=config["baseline_candidate"],
    )
    generate_v3_5_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        adapter_df=adapter_df,
        output_path=summary_dir / "v3_5_adapter_pilot_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
        title="Version 3.5 Adapter Pilot Report",
    )
    print("Recommended:", recommended)
    print("Secondary:", secondary)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_v3_5_adapter_pilot(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
