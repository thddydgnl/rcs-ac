"""Run full Stage 2 after confirm gate passes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stage2_transition_helpers import (
    aggregate_stage2_results,
    build_candidate_score_summary,
    choose_recommendations,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    ensure_dir,
    generate_stage2_pilot_report,
    load_yaml,
    run_stage2_candidate,
    save_stage2_handoff_snapshot,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Stage 2 after confirm.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage2_full.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _confirm_gate_passed(config: dict) -> tuple[bool, str]:
    confirm_summary = Path(config["confirm_gate"]["score_summary_path"])
    if not confirm_summary.exists():
        return False, f"Confirm summary not found: {confirm_summary}"
    df = pd.read_csv(confirm_summary)
    standard = config["standard_stage2_candidate"]
    row = df[df["candidate_name"] == standard]
    if row.empty:
        return False, f"Standard candidate missing in confirm summary: {standard}"
    passed = str(row.iloc[0]["pass_fail"]).upper() == "PASS"
    if not passed:
        return False, f"Standard candidate did not PASS confirm: {standard}"
    return True, "PASS"


def run_stage2_full(config: dict, smoke: bool = False) -> None:
    okay, reason = _confirm_gate_passed(config)
    if not okay:
        raise RuntimeError(reason)

    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "stage2_full_config_snapshot.yaml", config)
    save_stage2_handoff_snapshot(config, summary_dir)

    ordered_candidates = list(config.get("candidate_order", [config["standard_stage2_candidate"]]))
    all_rows = []
    for candidate_name in ordered_candidates:
        rows = run_stage2_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="full_eval_rows.json",
            smoke=smoke,
        )
        all_rows.extend(rows)

    df = aggregate_stage2_results(all_rows, summary_dir, "stage2_full")
    degradation_df = compute_degradation_metrics(df)
    consistency_df = compute_consistency_metrics(df)
    tiebreak_df = compute_tiebreak_metrics(df)
    degradation_df.to_csv(summary_dir / "stage2_full_degradation_summary.csv", index=False)
    consistency_df.to_csv(summary_dir / "stage2_full_consistency_summary.csv", index=False)
    tiebreak_df.to_csv(summary_dir / "stage2_full_tiebreak_summary.csv", index=False)

    weights = {
        "normal_success": config["tiebreak_weights"]["normal_success"],
        "degradation": config["tiebreak_weights"]["degradation"],
        "return_degradation": config["tiebreak_weights"]["return_degradation"],
        "vision_delta": config["tiebreak_weights"]["vision_delta"],
        "ego_delta": config["tiebreak_weights"]["ego_delta"],
        "vision_logit_delta": config["tiebreak_weights"]["vision_logit_delta"],
        "ego_logit_delta": config["tiebreak_weights"]["ego_logit_delta"],
        "consistency_gain": config["tiebreak_weights"]["consistency_gain"],
        "saturation_penalty": config["tiebreak_weights"]["saturation_penalty"],
        "side_effect_penalty": config["tiebreak_weights"]["side_effect_penalty"],
    }
    score_df = build_candidate_score_summary(df, degradation_df, consistency_df, tiebreak_df, weights)
    score_df.to_csv(summary_dir / "stage2_full_score_summary.csv", index=False)
    (summary_dir / "stage2_full_pass_fail_report.md").write_text(
        "# Stage 2 Full PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    recommended, secondary = choose_recommendations(
        score_df=score_df,
        standard_candidate=config["standard_stage2_candidate"],
        backup_candidate=config["backup_candidate"],
    )
    generate_stage2_pilot_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        output_path=summary_dir / "stage2_full_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
        title="Stage 2 Full Report",
    )
    print("Recommended:", recommended)
    print("Secondary:", secondary)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_stage2_full(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
