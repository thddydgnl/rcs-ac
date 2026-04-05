"""Evaluation-only reliability audit for the current Stage 2 trunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from rcs_ac_v1.model import MODALITY_NAMES
from run_very_weak_probe import (
    build_model,
    compare_clean_vs_corrupted_reliability,
    run_clean_eval,
    run_weak_corruption_eval,
)
from stage2_transition_helpers import (
    build_run_cache_metadata,
    ensure_dir,
    latest_checkpoint,
    load_yaml,
    save_yaml,
    should_reuse_run_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation-only reliability audit for the Stage 2 trunk.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage2_reliability_proof.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _candidate_seed_dir(checkpoint_root: Path, seed: int, smoke: bool = False) -> Path:
    candidate_names = [f"seed_{seed}_smoke", f"seed_{seed}"] if smoke else [f"seed_{seed}", f"seed_{seed}_smoke"]
    for seed_dir_name in candidate_names:
        seed_dir = checkpoint_root / seed_dir_name
        if seed_dir.exists():
            return seed_dir
    raise FileNotFoundError(f"Missing seed directory under {checkpoint_root} for seed {seed}")


def _run_probe(
    model: Any,
    device: torch.device,
    seed: int,
    probe: dict[str, Any],
    num_eval_episodes: int,
    run_mode: str,
) -> dict[str, Any]:
    probe_name = str(probe["name"])
    if probe_name == "normal_eval":
        row = run_clean_eval(
            model=model,
            device=device,
            seed=seed,
            num_eval_episodes=num_eval_episodes,
            run_mode=run_mode,
        )
    else:
        row = run_weak_corruption_eval(
            model=model,
            device=device,
            seed=seed,
            probe_name=probe_name,
            probe_cfg={
                "probe_name": probe_name,
                "corruption_type": str(probe["modality"]),
                "corruption_strength": float(probe["strength"]),
            },
            num_eval_episodes=num_eval_episodes,
            run_mode=run_mode,
        )
    row["probe_name"] = probe_name
    row["corruption_type"] = probe_name
    row["env_mode"] = "normal" if probe_name == "normal_eval" else "mild"
    row["corruption_strength"] = float(probe.get("strength", 0.0))
    return row


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.empty:
        return 0.0
    if "num_eval_episodes" not in frame.columns:
        return float(values.mean())
    weights = pd.to_numeric(frame["num_eval_episodes"], errors="coerce").fillna(0.0)
    valid = values.notna() & weights.gt(0.0)
    if not valid.any():
        return float(values.mean())
    return float(
        (values[valid].to_numpy(dtype=float) * weights[valid].to_numpy(dtype=float)).sum()
        / weights[valid].to_numpy(dtype=float).sum()
    )


def run_reliability_proof_candidate(
    candidate_name: str,
    candidate_spec: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_root = ensure_dir(output_root / candidate_name)
    device = torch.device(config.get("device", "cpu"))
    seeds = list(config["seeds"][:1] if smoke else config["seeds"])
    num_eval_episodes = int(config["num_eval_episodes_smoke"] if smoke else config["num_eval_episodes"])
    mode = "smoke" if smoke else "full"
    probes = list(config["eval_probes"])
    checkpoint_root = Path(candidate_spec["checkpoint_root"])
    base_stage1_config_path = str(config["base_stage1_config_path"])

    for seed in seeds:
        seed_dir_name = f"seed_{seed}_smoke" if smoke else f"seed_{seed}"
        seed_output_dir = ensure_dir(candidate_root / seed_dir_name)
        seed_rows_path = seed_output_dir / "eval_rows.json"
        metadata_path = seed_output_dir / "run_metadata.json"
        expected_metadata = build_run_cache_metadata(
            config=config,
            candidate_name=candidate_name,
            metrics_filename="eval_rows.json",
            mode=mode,
            num_eval_episodes=num_eval_episodes,
            step_name=str(config.get("experiment_name", "stage2_reliability_proof")),
            upstream_checkpoint_root=checkpoint_root,
        )
        if seed_rows_path.exists() and should_reuse_run_cache(metadata_path, expected_metadata):
            seed_rows = json.loads(seed_rows_path.read_text(encoding="utf-8"))
            rows.extend(seed_rows)
            continue
        seed_dir = _candidate_seed_dir(checkpoint_root, seed, smoke=smoke)
        checkpoint_path = latest_checkpoint(seed_dir)
        candidate_snapshot_path = seed_dir / "candidate_config_snapshot.json"
        if candidate_snapshot_path.exists():
            candidate_snapshot = json.loads(candidate_snapshot_path.read_text(encoding="utf-8"))
            model = build_model(candidate_snapshot, checkpoint_path, device)
        else:
            model = build_model(base_stage1_config_path, checkpoint_path, device)
        seed_rows: list[dict[str, Any]] = []
        for probe in probes:
            row = _run_probe(
                model=model,
                device=device,
                seed=seed,
                probe=probe,
                num_eval_episodes=num_eval_episodes,
                run_mode=mode,
            )
            row["candidate_name"] = candidate_name
            row["config_name"] = candidate_name
            row["seed"] = seed
            seed_rows.append(row)
        seed_rows_path.write_text(json.dumps(seed_rows, ensure_ascii=True, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(expected_metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        rows.extend(seed_rows)

    (candidate_root / f"eval_rows_{mode}.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    if not smoke:
        (candidate_root / "eval_rows.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    return rows


def aggregate_reliability_proof_results(rows: list[dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["candidate_name", "probe_name", "seed"]).reset_index(drop=True)
    df.to_csv(summary_dir / "stage2_reliability_proof_metrics.csv", index=False)
    df.to_json(summary_dir / "stage2_reliability_proof_metrics.json", orient="records", indent=2)
    return df


def compute_candidate_reliability_delta(candidate_df: pd.DataFrame) -> pd.DataFrame:
    compare_df = candidate_df.copy()
    if "probe_name" in compare_df.columns:
        compare_df["probe_name"] = compare_df["probe_name"].replace({"normal_eval": "clean"})
    return compare_clean_vs_corrupted_reliability(compare_df)


def compute_reliability_target_metrics(reliability_cmp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for corruption_type in sorted(reliability_cmp["corruption_type"].unique()):
        if corruption_type == "normal_eval":
            continue
        if not corruption_type.startswith("mild_"):
            continue
        target_modality = corruption_type.replace("mild_", "", 1)
        if target_modality not in MODALITY_NAMES:
            continue
        sub = reliability_cmp[reliability_cmp["corruption_type"] == corruption_type].copy()
        if sub.empty:
            continue
        target_row = sub[sub["modality"] == target_modality]
        if target_row.empty:
            continue
        target_delta = float(target_row.iloc[0]["delta_clean_minus_weak"])
        rank = int(
            sub["delta_clean_minus_weak"].rank(ascending=False, method="min")[sub["modality"] == target_modality].iloc[0]
        )
        nontarget_mean = float(sub[sub["modality"] != target_modality]["delta_clean_minus_weak"].mean())
        rows.append(
            {
                "corruption_type": corruption_type,
                "target_modality": target_modality,
                "target_delta_clean_minus_mild": target_delta,
                "target_relative_drop_ratio": float(target_row.iloc[0]["relative_drop_ratio"]),
                "rank_among_modalities": rank,
                "target_lowest_flag": int(rank == 1),
                "target_positive_flag": int(target_delta > 0.0),
                "target_vs_nontarget_gap": float(target_delta - nontarget_mean),
                "nontarget_mean_delta": nontarget_mean,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "corruption_type",
                "target_modality",
                "target_delta_clean_minus_mild",
                "target_relative_drop_ratio",
                "rank_among_modalities",
                "target_lowest_flag",
                "target_positive_flag",
                "target_vs_nontarget_gap",
                "nontarget_mean_delta",
            ]
        )
    return pd.DataFrame(rows).sort_values(["corruption_type"]).reset_index(drop=True)


def _modality_positive_ratio(target_df: pd.DataFrame, modality: str) -> float:
    sub = target_df[target_df["target_modality"] == modality]
    if sub.empty:
        return 0.0
    return float(sub["target_positive_flag"].mean())


def _load_baseline_clean_reliability_means(config: dict[str, Any]) -> dict[str, float] | None:
    baseline_eval_rows_path = config.get("baseline_eval_rows_path")
    if not baseline_eval_rows_path:
        return None
    path = Path(str(baseline_eval_rows_path))
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    if "probe_name" not in df.columns and "corruption_type" in df.columns:
        df["probe_name"] = df["corruption_type"]
    clean = df[df["probe_name"] == "normal_eval"]
    if clean.empty:
        return None
    summary: dict[str, float] = {}
    for modality in MODALITY_NAMES:
        key = f"reliability_mean_{modality}"
        if key in clean.columns:
            summary[key] = _weighted_mean(clean, key)
    return summary


def evaluate_reliability_proof_summary(summary: dict[str, Any], scoring_cfg: dict[str, Any]) -> str:
    required_modalities = int(scoring_cfg.get("required_target_modalities", 4))
    if int(summary.get("mixed_eval_episode_count_flag", 0)) == 1:
        return "WARN"
    if int(summary["observed_target_modalities"]) < required_modalities:
        return "WARN"
    pass_clean_floor = float(scoring_cfg.get("clean_success_rate_floor", 0.76))
    pass_target_positive = float(scoring_cfg.get("target_positive_ratio_pass", 0.75))
    pass_target_lowest = float(scoring_cfg.get("target_lowest_ratio_pass", 0.50))
    pass_target_gap = float(scoring_cfg.get("mean_target_vs_nontarget_gap_pass", 0.0))
    vision_positive_pass = float(scoring_cfg.get("vision_positive_ratio_pass", 1.0e-8))
    distance_positive_pass = float(scoring_cfg.get("distance_positive_ratio_pass", 0.0))
    ego_positive_pass = float(scoring_cfg.get("ego_positive_ratio_pass", 1.0e-8))
    route_positive_pass = float(scoring_cfg.get("route_positive_ratio_pass", 0.0))
    max_non_ego_clean_reliability_drift_pass = float(
        scoring_cfg.get("max_non_ego_clean_reliability_drift_pass", float("inf"))
    )
    vision_residual_delta_mild_minus_clean_max = float(
        scoring_cfg.get("vision_residual_delta_mild_minus_clean_max", float("inf"))
    )
    ego_residual_delta_mild_minus_clean_max = float(
        scoring_cfg.get("ego_residual_delta_mild_minus_clean_max", float("inf"))
    )
    if (
        summary["target_positive_ratio"] >= pass_target_positive
        and summary["target_lowest_ratio"] >= pass_target_lowest
        and summary["mean_target_vs_nontarget_gap"] > pass_target_gap
        and summary["vision_positive_ratio"] >= vision_positive_pass
        and summary["distance_positive_ratio"] >= distance_positive_pass
        and summary["ego_positive_ratio"] >= ego_positive_pass
        and summary["route_positive_ratio"] >= route_positive_pass
        and summary["clean_success_rate"] >= pass_clean_floor
        and summary.get("max_non_ego_clean_reliability_drift", 0.0) <= max_non_ego_clean_reliability_drift_pass
        and summary.get("vision_residual_delta_mild_minus_clean", 0.0) <= vision_residual_delta_mild_minus_clean_max
        and summary.get("ego_residual_delta_mild_minus_clean", 0.0) <= ego_residual_delta_mild_minus_clean_max
    ):
        return "PASS"
    if summary["target_positive_ratio"] > 0.0 and summary["mean_target_vs_nontarget_gap"] >= 0.0:
        return "WARN"
    return "FAIL"


def build_reliability_proof_score_summary(
    candidate_name: str,
    candidate_df: pd.DataFrame,
    target_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    scoring_cfg = dict(config.get("scoring", {}))
    clean = candidate_df[candidate_df["probe_name"] == "normal_eval"]
    mild_vision = candidate_df[candidate_df["probe_name"] == "mild_vision"]
    mild_ego = candidate_df[candidate_df["probe_name"] == "mild_ego"]
    if "num_eval_episodes" in candidate_df.columns and not candidate_df["num_eval_episodes"].dropna().empty:
        eval_counts = pd.to_numeric(candidate_df["num_eval_episodes"], errors="coerce").dropna()
        min_eval_episodes_per_row = int(eval_counts.min()) if not eval_counts.empty else 0
        max_eval_episodes_per_row = int(eval_counts.max()) if not eval_counts.empty else 0
    else:
        min_eval_episodes_per_row = 0
        max_eval_episodes_per_row = 0
    clean_success_rate = _weighted_mean(clean, "eval_success_rate") if not clean.empty else 0.0
    clean_return = _weighted_mean(clean, "eval_return") if not clean.empty else 0.0
    observed_target_modalities = int(target_df["target_modality"].nunique()) if not target_df.empty else 0
    summary = {
        "candidate_name": candidate_name,
        "clean_success_rate": clean_success_rate,
        "clean_return": clean_return,
        "min_eval_episodes_per_row": min_eval_episodes_per_row,
        "max_eval_episodes_per_row": max_eval_episodes_per_row,
        "mixed_eval_episode_count_flag": int(
            min_eval_episodes_per_row > 0 and max_eval_episodes_per_row > 0 and min_eval_episodes_per_row != max_eval_episodes_per_row
        ),
        "observed_target_modalities": observed_target_modalities,
        "target_positive_ratio": float(target_df["target_positive_flag"].mean()) if not target_df.empty else 0.0,
        "target_lowest_ratio": float(target_df["target_lowest_flag"].mean()) if not target_df.empty else 0.0,
        "mean_target_vs_nontarget_gap": float(target_df["target_vs_nontarget_gap"].mean()) if not target_df.empty else 0.0,
        "vision_positive_ratio": _modality_positive_ratio(target_df, "vision"),
        "distance_positive_ratio": _modality_positive_ratio(target_df, "distance"),
        "ego_positive_ratio": _modality_positive_ratio(target_df, "ego"),
        "route_positive_ratio": _modality_positive_ratio(target_df, "route"),
        "vision_residual_delta_clean_mean": (
            _weighted_mean(clean, "reliability_vision_residual_delta_mean")
            if not clean.empty and "reliability_vision_residual_delta_mean" in clean.columns
            else 0.0
        ),
        "vision_residual_delta_mild_mean": (
            _weighted_mean(mild_vision, "reliability_vision_residual_delta_mean")
            if not mild_vision.empty and "reliability_vision_residual_delta_mean" in mild_vision.columns
            else 0.0
        ),
        "ego_residual_delta_clean_mean": (
            _weighted_mean(clean, "reliability_ego_residual_delta_mean")
            if not clean.empty and "reliability_ego_residual_delta_mean" in clean.columns
            else 0.0
        ),
        "ego_residual_delta_mild_mean": (
            _weighted_mean(mild_ego, "reliability_ego_residual_delta_mean")
            if not mild_ego.empty and "reliability_ego_residual_delta_mean" in mild_ego.columns
            else 0.0
        ),
    }
    summary["ego_residual_delta_mild_minus_clean"] = (
        summary["ego_residual_delta_mild_mean"] - summary["ego_residual_delta_clean_mean"]
    )
    summary["vision_residual_delta_mild_minus_clean"] = (
        summary["vision_residual_delta_mild_mean"] - summary["vision_residual_delta_clean_mean"]
    )
    baseline_clean_means = _load_baseline_clean_reliability_means(config)
    for modality in MODALITY_NAMES:
        summary[f"{modality}_clean_reliability_drift"] = 0.0
    summary["max_non_ego_clean_reliability_drift"] = 0.0
    if baseline_clean_means is not None and not clean.empty:
        drift_values: dict[str, float] = {}
        for modality in MODALITY_NAMES:
            key = f"reliability_mean_{modality}"
            baseline_value = float(baseline_clean_means.get(key, 0.0))
            current_value = _weighted_mean(clean, key) if key in clean.columns else baseline_value
            drift = current_value - baseline_value
            summary[f"{modality}_clean_reliability_drift"] = drift
            drift_values[modality] = drift
        summary["max_non_ego_clean_reliability_drift"] = max(
            abs(drift_values["vision"]),
            abs(drift_values["distance"]),
            abs(drift_values["route"]),
        )
    if target_df.empty:
        summary["weakest_target_modality"] = "none"
    else:
        modality_ratios = {
            modality: _modality_positive_ratio(target_df, modality)
            for modality in MODALITY_NAMES
            if not target_df[target_df["target_modality"] == modality].empty
        }
        summary["weakest_target_modality"] = min(modality_ratios, key=modality_ratios.get)
    summary["pass_fail"] = evaluate_reliability_proof_summary(summary, scoring_cfg)
    return summary


def generate_reliability_proof_report(
    score_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
) -> None:
    row = score_df.iloc[0]
    weak_modality = row["weakest_target_modality"]
    lines = [
        "# Stage 2 Reliability Proof Report",
        "",
        f"- Recommended: **{recommended}**",
        f"- Verdict: **{row['pass_fail']}**",
        f"- Target positive ratio: `{row['target_positive_ratio']:.3f}`",
        f"- Target lowest ratio: `{row['target_lowest_ratio']:.3f}`",
        f"- Mean target-vs-nontarget gap: `{row['mean_target_vs_nontarget_gap']:.6f}`",
        f"- Clean success rate: `{row['clean_success_rate']:.3f}`",
        f"- Weakest modality: `{weak_modality}`",
        f"- Vision residual delta (mild-clean): `{row.get('vision_residual_delta_mild_minus_clean', 0.0):.6f}`",
        f"- Ego residual delta (mild-clean): `{row.get('ego_residual_delta_mild_minus_clean', 0.0):.6f}`",
        f"- Mixed eval episode count flag: `{int(row.get('mixed_eval_episode_count_flag', 0))}`",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "- 다음 단계 판단:",
        f"  - {'fusion shift proof로 넘어갈 수 있음' if row['pass_fail'] == 'PASS' else '먼저 reliability 반응성부터 더 확인해야 함'}",
    ]
    if int(row.get("mixed_eval_episode_count_flag", 0)) == 1:
        lines.extend(
            [
                "",
                "- 진단:",
                "  - mixed eval episode counts detected; rerun on clean output root",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "stage2_reliability_proof_config_snapshot.yaml", config)

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    all_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for candidate_name in ordered_candidates:
        candidate_spec = dict(config["candidates"][candidate_name])
        candidate_root = ensure_dir(output_root / candidate_name)
        rows = run_reliability_proof_candidate(
            candidate_name=candidate_name,
            candidate_spec=candidate_spec,
            config=config,
            output_root=output_root,
            smoke=args.smoke,
        )
        all_rows.extend(rows)
        candidate_df = pd.DataFrame(rows).sort_values(["probe_name", "seed"]).reset_index(drop=True)
        reliability_cmp = compute_candidate_reliability_delta(candidate_df)
        target_df = compute_reliability_target_metrics(reliability_cmp)
        reliability_cmp.to_csv(candidate_root / "reliability_delta.csv", index=False)
        target_df.to_csv(candidate_root / "target_positive_metrics.csv", index=False)
        score_rows.append(build_reliability_proof_score_summary(candidate_name, candidate_df, target_df, config))

    aggregate_reliability_proof_results(all_rows, summary_dir)
    score_df = pd.DataFrame(score_rows).sort_values(
        ["pass_fail", "target_positive_ratio", "target_lowest_ratio", "mean_target_vs_nontarget_gap", "clean_success_rate"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        ["_order", "target_positive_ratio", "target_lowest_ratio", "mean_target_vs_nontarget_gap", "clean_success_rate"],
        ascending=[True, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)
    score_df.to_csv(summary_dir / "stage2_reliability_proof_score_summary.csv", index=False)
    (summary_dir / "stage2_reliability_proof_pass_fail_report.md").write_text(
        "# Stage 2 Reliability Proof PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    recommended = str(score_df.iloc[0]["candidate_name"])
    generate_reliability_proof_report(
        score_df=score_df,
        output_path=summary_dir / "stage2_reliability_proof_final_recommendation.md",
        recommended=recommended,
    )
    print("Recommended:", recommended)


if __name__ == "__main__":
    main()
