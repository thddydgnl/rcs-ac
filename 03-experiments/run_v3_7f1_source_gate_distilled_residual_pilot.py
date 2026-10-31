"""Run Version 3.7f1 source-gate distilled residual pilot."""

from __future__ import annotations

import argparse
import copy
import json

from v3_shapley_helpers import (
    Stage37f1SourceGateDistilledResidualTrainer,
    aggregate_stage2_results,
    build_candidate_gate_thresholds,
    build_v3_1b_targetlogit_summary,
    build_v3_7f1_candidate_score_summary,
    choose_v3_recommendations,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    compute_v3_7f1_source_gate_distilled_residual_metrics,
    ensure_dir,
    generate_v3_7f1_report,
    infer_experiment_mode,
    is_calibration_only,
    load_yaml,
    resolve_experiment_output_root,
    run_v3_candidate,
    run_teacher_reproduction_preflight,
    save_v3_handoff_snapshot,
    save_yaml,
    validate_run_completeness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.7f1 source-gate distilled residual pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_7f1_source_gate_distilled_residual_pilot.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _validate_v3_7f1_config(config: dict) -> None:
    if str(config["baseline_candidate"]) != "reference_v3_6_medium_locked":
        raise ValueError("V3.7f1 baseline_candidate must be reference_v3_6_medium_locked")
    if not config.get("sign_teacher_checkpoint_root"):
        raise ValueError("V3.7f1 requires sign_teacher_checkpoint_root")
    if not config.get("clean_teacher_checkpoint_root"):
        raise ValueError("V3.7f1 requires clean_teacher_checkpoint_root")
    losses_cfg = config["losses"]
    if str(losses_cfg.get("gate_mode", "absolute_v1")) != "source_residual_v1":
        raise ValueError("V3.7f1 requires gate_mode=source_residual_v1")
    if str(losses_cfg.get("gate_input_mode", "legacy")) != "source_residual_bias_aware_v1":
        raise ValueError("V3.7f1 requires gate_input_mode=source_residual_bias_aware_v1")
    if not bool(losses_cfg.get("gate_only_finetune", False)):
        raise ValueError("V3.7f1 requires gate_only_finetune=true")


def run_v3_7f1_source_gate_distilled_residual_pilot(config: dict, smoke: bool = False) -> None:
    _validate_v3_7f1_config(config)
    output_root = resolve_experiment_output_root(config["output_root"], smoke)
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_config_snapshot.yaml", config)
    save_v3_handoff_snapshot(config, summary_dir, "v3_7f1_source_gate_distilled_residual_pilot")

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    if smoke:
        ordered_candidates = ordered_candidates[:3]
    all_rows = []
    for candidate_name in ordered_candidates:
        rows = run_v3_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="v3_7f1_eval_rows.json",
            smoke=smoke,
            trainer_cls=Stage37f1SourceGateDistilledResidualTrainer,
        )
        all_rows.extend(rows)
        if candidate_name == str(config.get("source_gate_teacher_candidate", "reference_v3_7c_medium_locked")):
            run_teacher_reproduction_preflight(
                all_rows,
                config=config,
                score_builder_fn=build_v3_7f1_candidate_score_summary,
                gate_metric_fn=compute_v3_7f1_source_gate_distilled_residual_metrics,
            )

    shadow_seeds = list(config.get("shadow_eval_seeds", []))
    if shadow_seeds and not smoke:
        shadow_config = copy.deepcopy(config)
        shadow_config["seeds"] = shadow_seeds
        for candidate_name in ordered_candidates:
            rows = run_v3_candidate(
                candidate_name=candidate_name,
                config=shadow_config,
                output_root=output_root,
                metrics_filename="v3_7f1_eval_rows.json",
                smoke=False,
                trainer_cls=Stage37f1SourceGateDistilledResidualTrainer,
            )
            all_rows.extend(rows)

    manifest_path = summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_summary_manifest.json"
    try:
        df = aggregate_stage2_results(all_rows, summary_dir, "v3_7f1_source_gate_distilled_residual_pilot")
        if not smoke:
            expected_seeds = list(config["seeds"]) + list(config.get("shadow_eval_seeds", []))
            validate_run_completeness(
                df,
                expected_candidates=ordered_candidates,
                expected_seeds=expected_seeds,
                expected_probe_names=[str(probe["name"]) for probe in config["eval_probes"]],
            )
        degradation_df = compute_degradation_metrics(df)
        consistency_df = compute_consistency_metrics(df)
        tiebreak_df = compute_tiebreak_metrics(df)
        targetlogit_df = build_v3_1b_targetlogit_summary(df)
        gate_df = compute_v3_7f1_source_gate_distilled_residual_metrics(
            df,
            candidate_thresholds=build_candidate_gate_thresholds(config),
        )

        degradation_df.to_csv(
        summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_degradation_summary.csv",
        index=False,
        )
        consistency_df.to_csv(
        summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_consistency_summary.csv",
        index=False,
        )
        tiebreak_df.to_csv(
        summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_tiebreak_summary.csv",
        index=False,
        )
        targetlogit_df.to_csv(
        summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_targetlogit_shapley_summary.csv",
        index=False,
        )
        gate_df.to_csv(
        summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_gate_summary.csv",
        index=False,
        )

        score_df = build_v3_7f1_candidate_score_summary(
        df=df,
        degradation_df=degradation_df,
        consistency_df=consistency_df,
        tiebreak_df=tiebreak_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        baseline_candidate=config["baseline_candidate"],
        tiebreak_weights=config["tiebreak_weights"],
    )
        score_df["experiment_mode"] = infer_experiment_mode(config["losses"])
        score_df["calibration_only_flag"] = float(is_calibration_only(config["losses"]))
        score_df.to_csv(summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_score_summary.csv", index=False)
        (summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_pass_fail_report.md").write_text(
        "# Version 3.7f1 Source-Gate Distilled Residual Pilot PASS/WARN/FAIL Report\n\n"
        + "> This run is calibration-only when `calibration_only_flag=1`.\n\n"
        + score_df.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
        )
        recommended, secondary = choose_v3_recommendations(
        score_df=score_df,
        baseline_candidate=config["baseline_candidate"],
        )
        generate_v3_7f1_report(
        score_df=score_df,
        degradation_df=degradation_df,
        tiebreak_df=tiebreak_df,
        consistency_df=consistency_df,
        targetlogit_df=targetlogit_df,
        gate_df=gate_df,
        output_path=summary_dir / "v3_7f1_source_gate_distilled_residual_pilot_final_recommendation.md",
        recommended=recommended,
        secondary=secondary,
        title="Version 3.7f1 Source-Gate Distilled Residual Pilot Report",
        )
        manifest_path.write_text(
            json.dumps({"summary_complete_flag": 1, "status": "ok"}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print("Recommended:", recommended)
        print("Secondary:", secondary)
    except Exception as exc:
        manifest_path.write_text(
            json.dumps({"summary_complete_flag": 0, "status": "failed", "error": str(exc)}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        raise


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_v3_7f1_source_gate_distilled_residual_pilot(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
