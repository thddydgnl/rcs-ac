"""Run Version 3.7f2.2 residual activation debug/promotion experiments."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from v3_shapley_helpers import (
    Stage37f2SourceGateResidualLogitSelectiveTrainer,
    aggregate_stage2_results,
    build_candidate_gate_thresholds,
    build_v3_1b_targetlogit_summary,
    build_v3_7f2_2_candidate_score_summary,
    build_v3_7f2_2_pass_distance_dashboard,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_tiebreak_metrics,
    compute_v3_7f2_2_activation_metrics,
    ensure_dir,
    generate_v3_7f2_2_report,
    infer_experiment_mode,
    is_calibration_only,
    load_yaml,
    normalize_tiebreak_weights,
    resolve_experiment_output_root,
    run_teacher_reproduction_preflight,
    run_v3_candidate,
    save_v3_handoff_snapshot,
    save_yaml,
    validate_run_completeness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 3.7f2.2 residual activation pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_7f2_2_residual_activation_pilot.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _validate_config(config: dict) -> None:
    losses_cfg = config["losses"]
    if str(losses_cfg.get("gate_mode", "absolute_v1")) != "source_residual_logit_v2":
        raise ValueError("V3.7f2.2 requires gate_mode=source_residual_logit_v2")
    if str(losses_cfg.get("gate_input_mode", "legacy")) != "source_residual_bias_aware_v1":
        raise ValueError("V3.7f2.2 requires gate_input_mode=source_residual_bias_aware_v1")
    if str(losses_cfg.get("residual_alpha_mode", "fixed")) != "learnable":
        raise ValueError("V3.7f2.2 requires residual_alpha_mode=learnable")
    if not bool(losses_cfg.get("gate_only_finetune", False)):
        raise ValueError("V3.7f2.2 requires gate_only_finetune=true")


def _choose_train_candidate_recommendations(
    score_df: pd.DataFrame,
    *,
    baseline_candidate: str,
    source_candidate: str,
) -> tuple[str, str]:
    eligible = score_df[
        ~score_df["candidate_name"].isin([baseline_candidate, source_candidate])
    ].reset_index(drop=True)
    if eligible.empty:
        return source_candidate, baseline_candidate
    recommended = str(eligible.iloc[0]["candidate_name"])
    secondary = str(eligible.iloc[1]["candidate_name"]) if len(eligible) > 1 else source_candidate
    return recommended, secondary


def _collect_residual_alpha_trajectory(
    output_root: Path,
    ordered_candidates: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for candidate_name in ordered_candidates:
        for seed in seeds:
            metrics_path = output_root / candidate_name / f"seed_{seed}" / "train_metrics.jsonl"
            if not metrics_path.exists():
                continue
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                rows.append(
                    {
                        "candidate_name": candidate_name,
                        "seed": int(seed),
                        "update": int(payload.get("update", len(rows))),
                        "residual_alpha": float(payload.get("train/residual_alpha", 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def run_v3_7f2_2_experiment(config: dict, smoke: bool = False) -> None:
    _validate_config(config)
    config = copy.deepcopy(config)
    config["tiebreak_weights"] = normalize_tiebreak_weights(config.get("tiebreak_weights"))
    experiment_name = str(config["experiment_name"])
    output_root = resolve_experiment_output_root(config["output_root"], smoke)
    summary_dir = ensure_dir(output_root / "summary")
    prefix = experiment_name
    save_yaml(summary_dir / f"{prefix}_config_snapshot.yaml", config)
    save_v3_handoff_snapshot(config, summary_dir, prefix)

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    all_rows = []
    teacher_warn_only = bool(smoke and config.get("smoke_teacher_reproduction_warn_only", False))
    teacher_audit_report = summary_dir / f"{prefix}_teacher_reproduction_audit_report.md"
    teacher_audit_report.write_text("# Teacher Reproduction Audit\n\n- Status: pending\n", encoding="utf-8")
    source_candidate = str(config.get("source_gate_teacher_candidate", "reference_v3_7c_medium_locked"))

    for candidate_name in ordered_candidates:
        rows = run_v3_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="v3_7f2_2_eval_rows.json",
            smoke=smoke,
            trainer_cls=Stage37f2SourceGateResidualLogitSelectiveTrainer,
        )
        all_rows.extend(rows)
        if candidate_name == source_candidate:
            try:
                run_teacher_reproduction_preflight(
                    all_rows,
                    config=config,
                    score_builder_fn=build_v3_7f2_2_candidate_score_summary,
                    gate_metric_fn=compute_v3_7f2_2_activation_metrics,
                )
                teacher_audit_report.write_text("# Teacher Reproduction Audit\n\n- Status: pass\n", encoding="utf-8")
            except Exception as exc:
                teacher_audit_report.write_text(
                    f"# Teacher Reproduction Audit\n\n- Status: {'warn' if teacher_warn_only else 'fail'}\n- Error: `{exc}`\n",
                    encoding="utf-8",
                )
                if not teacher_warn_only:
                    raise

    shadow_seeds = list(config.get("shadow_eval_seeds", []))
    if shadow_seeds and not smoke:
        shadow_config = copy.deepcopy(config)
        shadow_config["seeds"] = shadow_seeds
        for candidate_name in ordered_candidates:
            rows = run_v3_candidate(
                candidate_name=candidate_name,
                config=shadow_config,
                output_root=output_root,
                metrics_filename="v3_7f2_2_eval_rows.json",
                smoke=False,
                trainer_cls=Stage37f2SourceGateResidualLogitSelectiveTrainer,
            )
            all_rows.extend(rows)

    manifest_path = summary_dir / f"{prefix}_summary_manifest.json"
    try:
        df = aggregate_stage2_results(all_rows, summary_dir, prefix)
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
        gate_df = compute_v3_7f2_2_activation_metrics(
            df,
            candidate_thresholds=build_candidate_gate_thresholds(config),
        )
        score_df = build_v3_7f2_2_candidate_score_summary(
            df=df,
            degradation_df=degradation_df,
            consistency_df=consistency_df,
            tiebreak_df=tiebreak_df,
            targetlogit_df=targetlogit_df,
            gate_df=gate_df,
            baseline_candidate=str(config["baseline_candidate"]),
            source_candidate=source_candidate,
            tiebreak_weights=config["tiebreak_weights"],
        )
        score_df["experiment_mode"] = infer_experiment_mode(config["losses"])
        score_df["calibration_only_flag"] = float(is_calibration_only(config["losses"]))
        pass_distance_df = build_v3_7f2_2_pass_distance_dashboard(
            score_df,
            baseline_candidate=str(config["baseline_candidate"]),
            source_candidate=source_candidate,
        )
        feature_df = gate_df[
            [
                "candidate_name",
                "clean_vs_mild_gate_feature_auc",
                "clean_vs_mild_gate_feature_delta_norm",
                "source_gate_feature_onoff_auc_delta",
                "teacher_delta_feature_onoff_auc_delta",
            ]
        ].copy()
        gate_compare_df = gate_df[
            [
                "candidate_name",
                "vision_source_gate_clean_mean",
                "vision_source_gate_mild_mean",
                "vision_residual_logit_clean_mean",
                "vision_residual_logit_mild_mean",
                "vision_residual_logit_mild_minus_clean",
                "vision_final_gate_clean_mean",
                "vision_final_gate_mild_mean",
                "vision_final_gate_mild_minus_clean",
                "residual_alpha",
            ]
        ].copy()
        alpha_df = _collect_residual_alpha_trajectory(output_root, ordered_candidates, list(config["seeds"]))

        degradation_df.to_csv(summary_dir / f"{prefix}_degradation_summary.csv", index=False)
        consistency_df.to_csv(summary_dir / f"{prefix}_consistency_summary.csv", index=False)
        tiebreak_df.to_csv(summary_dir / f"{prefix}_tiebreak_summary.csv", index=False)
        targetlogit_df.to_csv(summary_dir / f"{prefix}_targetlogit_shapley_summary.csv", index=False)
        gate_df.to_csv(summary_dir / f"{prefix}_gate_summary.csv", index=False)
        score_df.to_csv(summary_dir / f"{prefix}_score_summary.csv", index=False)
        pass_distance_df.to_csv(summary_dir / f"{prefix}_pass_distance_dashboard.csv", index=False)
        feature_df.to_csv(summary_dir / f"{prefix}_gate_feature_separability_report.csv", index=False)
        gate_compare_df.to_csv(summary_dir / f"{prefix}_gate_axis_comparison.csv", index=False)
        alpha_df.to_csv(summary_dir / f"{prefix}_residual_alpha_trajectory.csv", index=False)
        (summary_dir / f"{prefix}_pass_fail_report.md").write_text(
            "# Residual Activation Debug Report\n\n" + score_df.to_markdown(index=False) + "\n",
            encoding="utf-8",
        )
        recommended, secondary = _choose_train_candidate_recommendations(
            score_df,
            baseline_candidate=str(config["baseline_candidate"]),
            source_candidate=source_candidate,
        )
        generate_v3_7f2_2_report(
            score_df=score_df,
            degradation_df=degradation_df,
            tiebreak_df=tiebreak_df,
            consistency_df=consistency_df,
            targetlogit_df=targetlogit_df,
            gate_df=gate_df,
            output_path=summary_dir / f"{prefix}_final_recommendation.md",
            recommended=recommended,
            secondary=secondary,
            title=f"{experiment_name} Report",
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
    run_v3_7f2_2_experiment(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
