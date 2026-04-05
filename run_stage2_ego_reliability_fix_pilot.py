"""Pilot runner for repairing ego reliability responsiveness in the Stage 2 trunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_stage2_reliability_proof import (
    build_reliability_proof_score_summary,
    compute_candidate_reliability_delta,
    compute_reliability_target_metrics,
)
from stage2_transition_helpers import ensure_dir, load_yaml, run_stage2_candidate, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 2 ego reliability fix pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage2_ego_reliability_fix_pilot.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_ego_fix_recommendation(score_df: pd.DataFrame) -> str:
    simple_name = "ego_raw_observable_adapter"
    positive_name = "ego_raw_observable_adapter_plus_ego_positive"
    if simple_name in set(score_df["candidate_name"]) and str(
        score_df.loc[score_df["candidate_name"] == simple_name, "pass_fail"].iloc[0]
    ) == "PASS":
        return simple_name
    if positive_name in set(score_df["candidate_name"]) and str(
        score_df.loc[score_df["candidate_name"] == positive_name, "pass_fail"].iloc[0]
    ) == "PASS":
        return positive_name
    return str(score_df.iloc[0]["candidate_name"])


def generate_ego_fix_report(score_df: pd.DataFrame, output_path: Path, recommended: str) -> None:
    top = score_df.iloc[0]
    lines = [
        "# Stage 2 Ego Reliability Fix Pilot Report",
        "",
        f"- Recommended: **{recommended}**",
        f"- Top verdict: **{top['pass_fail']}**",
        f"- Top target positive ratio: `{top['target_positive_ratio']:.3f}`",
        f"- Top target lowest ratio: `{top['target_lowest_ratio']:.3f}`",
        f"- Top mean target-vs-nontarget gap: `{top['mean_target_vs_nontarget_gap']:.6f}`",
        f"- Top ego positive ratio: `{top['ego_positive_ratio']:.3f}`",
        f"- Top clean success rate: `{top['clean_success_rate']:.3f}`",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "- 다음 단계 판단:",
        f"  - {'Fusion Shift Proof로 넘어갈 수 있음' if str(score_df.loc[score_df['candidate_name'] == recommended, 'pass_fail'].iloc[0]) == 'PASS' else '먼저 ego reliability 반응성부터 더 고정해야 함'}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "stage2_ego_reliability_fix_pilot_config_snapshot.yaml", config)

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    all_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for candidate_name in ordered_candidates:
        rows = run_stage2_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="ego_fix_eval_rows.json",
            smoke=args.smoke,
        )
        candidate_root = ensure_dir(output_root / candidate_name)
        (candidate_root / "eval_rows.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
        candidate_df = pd.DataFrame(rows)
        if "probe_name" not in candidate_df.columns and "corruption_type" in candidate_df.columns:
            candidate_df["probe_name"] = candidate_df["corruption_type"]
        candidate_df = candidate_df.sort_values(["probe_name", "seed"]).reset_index(drop=True)
        all_rows.extend(candidate_df.to_dict(orient="records"))
        reliability_cmp = compute_candidate_reliability_delta(candidate_df)
        target_df = compute_reliability_target_metrics(reliability_cmp)
        reliability_cmp.to_csv(candidate_root / "reliability_delta.csv", index=False)
        target_df.to_csv(candidate_root / "target_positive_metrics.csv", index=False)
        score_rows.append(build_reliability_proof_score_summary(candidate_name, candidate_df, target_df, config))

    metrics_df = pd.DataFrame(all_rows).sort_values(["candidate_name", "probe_name", "seed"]).reset_index(drop=True)
    metrics_df.to_csv(summary_dir / "stage2_ego_reliability_fix_pilot_metrics.csv", index=False)
    metrics_df.to_json(summary_dir / "stage2_ego_reliability_fix_pilot_metrics.json", orient="records", indent=2)
    score_df = pd.DataFrame(score_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        ["_order", "target_positive_ratio", "target_lowest_ratio", "mean_target_vs_nontarget_gap", "clean_success_rate"],
        ascending=[True, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    score_df.to_csv(summary_dir / "stage2_ego_reliability_fix_pilot_score_summary.csv", index=False)
    (summary_dir / "stage2_ego_reliability_fix_pilot_pass_fail_report.md").write_text(
        "# Stage 2 Ego Reliability Fix Pilot PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    recommended = choose_ego_fix_recommendation(score_df)
    generate_ego_fix_report(
        score_df=score_df,
        output_path=summary_dir / "stage2_ego_reliability_fix_pilot_final_recommendation.md",
        recommended=recommended,
    )
    print("Recommended:", recommended)


if __name__ == "__main__":
    main()
