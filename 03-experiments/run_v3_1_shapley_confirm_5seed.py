"""Run Version 3.1 Shapley 5-seed confirm from the pilot winner."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v3_shapley_helpers import (
    Stage31ShapleyTrainer,
    aggregate_stage2_results,
    build_v3_1_candidate_score_summary,
    build_v3_1_raw_shapley_summary,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    compute_v3_1_alignment_metrics,
    ensure_dir,
    generate_v3_1_report,
    load_yaml,
    maybe_override_checkpoint_root,
    resolve_v3_winner,
    run_v3_candidate,
    save_v3_handoff_snapshot,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.1 Shapley 5-seed confirm.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_1_shapley_confirm_5seed.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _pilot_gate_passed(config: dict, winner: str) -> tuple[bool, str]:
    pilot_summary = Path(config["pilot_gate"]["score_summary_path"])
    if not pilot_summary.exists():
        return False, f"Pilot summary not found: {pilot_summary}"
    df = pd.read_csv(pilot_summary)
    row = df[df["candidate_name"] == winner]
    if row.empty:
        return False, f"Winner candidate missing in pilot summary: {winner}"
    passed = str(row.iloc[0]["pass_fail"]).upper() == "PASS"
    if not passed:
        return False, f"Winner candidate did not PASS pilot: {winner}"
    return True, "PASS"


def run_v3_1_shapley_confirm(config: dict, smoke: bool = False) -> None:
    winner = resolve_v3_winner(
        score_summary_path=config["pilot_gate"]["score_summary_path"],
        fallback_candidate=config["fallback_candidate"],
    )
    okay, reason = _pilot_gate_passed(config, winner)
    if not okay:
        raise RuntimeError(reason)

    runtime_config = maybe_override_checkpoint_root(
        config=config,
        candidate_name=winner,
        inherited_output_root=config.get("pilot_output_root"),
        source_experiment_name="v3_1_shapley_pilot",
    )
    ordered_candidates = [runtime_config["baseline_candidate"]]
    if winner != runtime_config["baseline_candidate"]:
        ordered_candidates.append(winner)
    runtime_config["candidate_order"] = ordered_candidates

    output_root = ensure_dir(runtime_config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "v3_1_shapley_confirm_5seed_config_snapshot.yaml", runtime_config)
    save_v3_handoff_snapshot(runtime_config, summary_dir, "v3_1_shapley_confirm_5seed")

    rows = []
    for candidate_name in runtime_config["candidate_order"]:
        rows.extend(
            run_v3_candidate(
                candidate_name=candidate_name,
                config=runtime_config,
                output_root=output_root,
                metrics_filename="v3_1_confirm_eval_rows.json",
                smoke=smoke,
                trainer_cls=Stage31ShapleyTrainer,
            )
        )

    df = aggregate_stage2_results(rows, summary_dir, "v3_1_shapley_confirm_5seed")
    degradation_df = compute_degradation_metrics(df)
    consistency_df = compute_consistency_metrics(df)
    tiebreak_df = compute_tiebreak_metrics(df)
    alignment_df = compute_v3_1_alignment_metrics(df)
    raw_shapley_df = build_v3_1_raw_shapley_summary(df)
    degradation_df.to_csv(summary_dir / "v3_1_shapley_confirm_5seed_degradation_summary.csv", index=False)
    consistency_df.to_csv(summary_dir / "v3_1_shapley_confirm_5seed_consistency_summary.csv", index=False)
    tiebreak_df.to_csv(summary_dir / "v3_1_shapley_confirm_5seed_tiebreak_summary.csv", index=False)
    alignment_df.to_csv(summary_dir / "v3_1_shapley_confirm_5seed_alignment_summary.csv", index=False)
    raw_shapley_df.to_csv(summary_dir / "v3_1_shapley_confirm_5seed_raw_shapley_summary.csv", index=False)

    score_df = build_v3_1_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        alignment_df=alignment_df,
        baseline_candidate=runtime_config["baseline_candidate"],
        tiebreak_weights=runtime_config["tiebreak_weights"],
    )
    score_df.to_csv(summary_dir / "v3_1_shapley_confirm_5seed_score_summary.csv", index=False)
    (summary_dir / "v3_1_shapley_confirm_5seed_pass_fail_report.md").write_text(
        "# Version 3.1 Shapley Confirm 5-Seed PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    generate_v3_1_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        alignment_df=alignment_df,
        raw_df=raw_shapley_df,
        output_path=summary_dir / "v3_1_shapley_confirm_5seed_final_recommendation.md",
        recommended=winner,
        secondary=runtime_config["baseline_candidate"],
        title="Version 3.1 Shapley Confirm 5-Seed Report",
    )
    print("Recommended:", winner)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_v3_1_shapley_confirm(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
