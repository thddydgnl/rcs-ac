"""Vision-first same-state fusion shift proof runner for the Stage 2 trunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rcs_ac_v1.model import MODALITY_NAMES
from stage2_transition_helpers import (
    build_stage2_candidate_effective_config,
    ensure_dir,
    load_yaml,
    run_stage2_candidate,
    run_stage2_eval_only_candidate,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 2 vision-first fusion shift proof pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage2_fusion_shift_proof_pilot.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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


def _load_baseline_clean_frame(
    config: dict[str, Any],
    candidate_df: pd.DataFrame,
    *,
    smoke: bool,
) -> tuple[pd.DataFrame | None, bool, bool, bool]:
    baseline_eval_rows_path = (
        config.get("baseline_eval_rows_path_smoke")
        if smoke and config.get("baseline_eval_rows_path_smoke")
        else config.get("baseline_eval_rows_path")
    )
    if not baseline_eval_rows_path:
        return None, False, False, False
    path = Path(str(baseline_eval_rows_path))
    if not path.exists():
        return None, True, True, False
    rows = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    if "probe_name" not in df.columns and "corruption_type" in df.columns:
        df["probe_name"] = df["corruption_type"]
    clean = df[df["probe_name"] == "normal_eval"]
    if clean.empty:
        return None, True, False, True
    candidate_clean = candidate_df[candidate_df["probe_name"] == "normal_eval"]
    if candidate_clean.empty:
        return None, True, False, True
    filtered = clean.copy()
    for column in ("seed", "run_mode", "num_eval_episodes"):
        if column not in filtered.columns or column not in candidate_clean.columns:
            return None, True, False, True
        values = candidate_clean[column].dropna().unique().tolist()
        if not values:
            return None, True, False, True
        filtered = filtered[filtered[column].isin(values)]
        if filtered.empty:
            return None, True, False, True
    return filtered, True, False, False


def _resolve_baseline_eval_rows_path(config: dict[str, Any], *, smoke: bool) -> Path | None:
    baseline_eval_rows_path = (
        config.get("baseline_eval_rows_path_smoke")
        if smoke and config.get("baseline_eval_rows_path_smoke")
        else config.get("baseline_eval_rows_path")
    )
    if not baseline_eval_rows_path:
        return None
    return Path(str(baseline_eval_rows_path))


def _candidate_effective_config(config: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    try:
        return build_stage2_candidate_effective_config(config, candidate_name)
    except Exception:
        return dict(config)


def _teacher_effect_distill_enabled(config: dict[str, Any], candidate_name: str) -> bool:
    effective = _candidate_effective_config(config, candidate_name)
    losses = dict(effective.get("losses", {}))
    return (
        float(losses.get("teacher_gate_effect_distill_loss_coef", 0.0)) > 0.0
        or float(losses.get("teacher_clean_gate_effect_preserve_loss_coef", 0.0)) > 0.0
        or float(losses.get("teacher_nonvision_gate_effect_preserve_loss_coef", 0.0)) > 0.0
    )


def _scoring_threshold(
    scoring_cfg: dict[str, Any],
    key: str,
    default: float,
    *,
    smoke: bool,
) -> float:
    if smoke:
        smoke_key = f"smoke_{key}"
        if smoke_key in scoring_cfg:
            return float(scoring_cfg[smoke_key])
    return float(scoring_cfg.get(key, default))


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def evaluate_fusion_shift_proof_summary(
    summary: dict[str, Any],
    scoring_cfg: dict[str, Any],
    *,
    smoke: bool = False,
) -> str:
    if int(summary.get("baseline_clean_missing_flag", 0)) == 1:
        return "WARN"
    if int(summary.get("baseline_clean_match_missing_flag", 0)) == 1:
        return "WARN"
    if int(summary.get("mixed_eval_episode_count_flag", 0)) == 1:
        return "WARN"
    clean_success_rate_floor = _scoring_threshold(
        scoring_cfg,
        "clean_success_rate_standalone_floor",
        float(scoring_cfg.get("clean_success_rate_floor", 0.79)),
        smoke=smoke,
    )
    vision_reliability_delta_pass = _scoring_threshold(scoring_cfg, "vision_reliability_delta_pass", 0.0, smoke=smoke)
    vision_gate_effect_delta_pass = _scoring_threshold(
        scoring_cfg,
        "vision_gate_effect_delta_pass",
        float(scoring_cfg.get("vision_gate_delta_pass", 0.0)),
        smoke=smoke,
    )
    vision_gate_effect_absmax_pass = _scoring_threshold(
        scoring_cfg,
        "vision_gate_effect_absmax_pass",
        0.0,
        smoke=smoke,
    )
    vision_bias_score_delta_pass = _scoring_threshold(scoring_cfg, "vision_bias_score_delta_pass", 0.0, smoke=smoke)
    vision_prebias_score_delta_pass = _scoring_threshold(
        scoring_cfg,
        "vision_prebias_score_delta_pass",
        0.0,
        smoke=smoke,
    )
    vision_attention_delta_pass = _scoring_threshold(scoring_cfg, "vision_attention_delta_pass", 0.0, smoke=smoke)
    vision_final_score_delta_pass = _scoring_threshold(scoring_cfg, "vision_final_score_delta_pass", 0.0, smoke=smoke)
    vision_reliability_gate_effect_alignment_ratio_pass = _scoring_threshold(
        scoring_cfg,
        "vision_reliability_gate_effect_alignment_ratio_pass",
        float(
            scoring_cfg.get(
            "vision_reliability_gate_effect_alignment_ratio_pass",
            scoring_cfg.get("vision_alignment_ratio_pass", 0.70),
            )
        ),
        smoke=smoke,
    )
    vision_gate_effect_attention_alignment_ratio_pass = _scoring_threshold(
        scoring_cfg,
        "vision_gate_effect_attention_alignment_ratio_pass",
        float(
            scoring_cfg.get(
            "vision_gate_effect_attention_alignment_ratio_pass",
            scoring_cfg.get("vision_alignment_ratio_pass", 0.70),
            )
        ),
        smoke=smoke,
    )
    vision_gate_effect_score_alignment_ratio_pass = _scoring_threshold(
        scoring_cfg,
        "vision_gate_effect_score_alignment_ratio_pass",
        float(scoring_cfg.get("vision_gate_effect_score_alignment_ratio_pass", 0.70)),
        smoke=smoke,
    )
    vision_prebias_score_final_score_alignment_ratio_pass = _scoring_threshold(
        scoring_cfg,
        "vision_prebias_score_final_score_alignment_ratio_pass",
        float(scoring_cfg.get("vision_gate_effect_score_alignment_ratio_pass", 0.70)),
        smoke=smoke,
    )
    nontarget_rescue_delta_pass = _scoring_threshold(scoring_cfg, "nontarget_rescue_delta_pass", 0.0, smoke=smoke)
    max_non_target_clean_attention_drift_pass = _scoring_threshold(
        scoring_cfg,
        "max_non_target_clean_attention_drift_standalone_pass",
        float(
            scoring_cfg.get(
            "max_non_target_clean_attention_drift_standalone_pass",
            scoring_cfg.get("max_non_target_clean_attention_drift_pass", 0.03),
            )
        ),
        smoke=smoke,
    )
    max_non_ego_clean_reliability_drift_pass = _scoring_threshold(
        scoring_cfg,
        "max_non_ego_clean_reliability_drift_standalone_pass",
        float(
            scoring_cfg.get(
            "max_non_ego_clean_reliability_drift_standalone_pass",
            scoring_cfg.get("max_non_ego_clean_reliability_drift_pass", 0.02),
            )
        ),
        smoke=smoke,
    )
    clean_score_anchor_error_pass = _scoring_threshold(
        scoring_cfg,
        "clean_score_anchor_error_pass",
        float("inf"),
        smoke=smoke,
    )
    clean_action_anchor_error_pass = _scoring_threshold(
        scoring_cfg,
        "clean_action_anchor_error_pass",
        float("inf"),
        smoke=smoke,
    )
    clean_bias_score_anchor_error_pass = _scoring_threshold(
        scoring_cfg,
        "clean_bias_score_anchor_error_pass",
        float("inf"),
        smoke=smoke,
    )
    clean_prebias_score_anchor_error_pass = _scoring_threshold(
        scoring_cfg,
        "clean_prebias_score_anchor_error_pass",
        float("inf"),
        smoke=smoke,
    )
    if (
        summary["vision_reliability_delta_clean_minus_corrupt"] > vision_reliability_delta_pass
        and summary["vision_gate_effect_delta_clean_minus_corrupt"] > vision_gate_effect_delta_pass
        and summary.get("vision_gate_effect_absmax", 0.0) >= vision_gate_effect_absmax_pass
        and summary["vision_bias_score_delta_clean_minus_corrupt"] > vision_bias_score_delta_pass
        and summary.get("vision_prebias_score_delta_clean_minus_corrupt", 0.0) > vision_prebias_score_delta_pass
        and summary["vision_attention_delta_clean_minus_corrupt"] > vision_attention_delta_pass
        and summary["vision_final_score_delta_clean_minus_corrupt"] > vision_final_score_delta_pass
        and summary["vision_reliability_gate_effect_alignment_ratio"] >= vision_reliability_gate_effect_alignment_ratio_pass
        and summary["vision_gate_effect_attention_alignment_ratio"] >= vision_gate_effect_attention_alignment_ratio_pass
        and summary["vision_gate_effect_score_alignment_ratio"] >= vision_gate_effect_score_alignment_ratio_pass
        and summary.get("vision_prebias_score_final_score_alignment_ratio", 0.0)
        >= vision_prebias_score_final_score_alignment_ratio_pass
        and summary["nontarget_rescue_delta"] >= nontarget_rescue_delta_pass
        and summary["clean_success_rate_standalone"] >= clean_success_rate_floor
        and summary["max_non_target_clean_attention_drift_standalone"] <= max_non_target_clean_attention_drift_pass
        and summary["max_non_ego_clean_reliability_drift_standalone"] <= max_non_ego_clean_reliability_drift_pass
        and summary.get("clean_score_anchor_error", 0.0) <= clean_score_anchor_error_pass
        and summary.get("clean_action_anchor_error", 0.0) <= clean_action_anchor_error_pass
        and summary.get("clean_bias_score_anchor_error", 0.0) <= clean_bias_score_anchor_error_pass
        and summary.get("clean_prebias_score_anchor_error", 0.0) <= clean_prebias_score_anchor_error_pass
    ):
        return "PASS"
    if (
        summary["vision_reliability_delta_clean_minus_corrupt"] > 0.0
        or summary["vision_gate_effect_delta_clean_minus_corrupt"] > 0.0
        or summary.get("vision_bias_score_delta_clean_minus_corrupt", 0.0) > 0.0
        or summary.get("vision_prebias_score_delta_clean_minus_corrupt", 0.0) > 0.0
        or summary["vision_attention_delta_clean_minus_corrupt"] > 0.0
        or summary["vision_final_score_delta_clean_minus_corrupt"] > 0.0
    ):
        return "WARN"
    return "FAIL"


def build_fusion_shift_proof_score_summary(
    candidate_name: str,
    candidate_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    scoring_cfg = dict(config.get("scoring", {}))
    clean = candidate_df[candidate_df["probe_name"] == "normal_eval"]
    mild_vision = candidate_df[candidate_df["probe_name"] == "mild_vision"]
    if "num_eval_episodes" in candidate_df.columns and not candidate_df["num_eval_episodes"].dropna().empty:
        eval_counts = pd.to_numeric(candidate_df["num_eval_episodes"], errors="coerce").dropna()
        min_eval_episodes_per_row = int(eval_counts.min()) if not eval_counts.empty else 0
        max_eval_episodes_per_row = int(eval_counts.max()) if not eval_counts.empty else 0
    else:
        min_eval_episodes_per_row = 0
        max_eval_episodes_per_row = 0
    summary = {
        "candidate_name": candidate_name,
        "teacher_effect_distill_enabled": int(_teacher_effect_distill_enabled(config, candidate_name)),
        "clean_success_rate": _weighted_mean(clean, "eval_success_rate") if not clean.empty else 0.0,
        "clean_return": _weighted_mean(clean, "eval_return") if not clean.empty else 0.0,
        "clean_success_rate_standalone": _weighted_mean(clean, "eval_success_rate") if not clean.empty else 0.0,
        "clean_return_standalone": _weighted_mean(clean, "eval_return") if not clean.empty else 0.0,
        "min_eval_episodes_per_row": min_eval_episodes_per_row,
        "max_eval_episodes_per_row": max_eval_episodes_per_row,
        "mixed_eval_episode_count_flag": int(
            min_eval_episodes_per_row > 0 and max_eval_episodes_per_row > 0 and min_eval_episodes_per_row != max_eval_episodes_per_row
        ),
        "vision_reliability_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_reliability_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_attention_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_attention_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_final_score_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_final_score_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_gate_effect_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_gate_effect_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_bias_score_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_bias_score_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_prebias_score_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_prebias_score_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_gate_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_gate_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_source_gate_base_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_source_gate_base_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_source_gate_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_source_gate_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_teacher_gate_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_teacher_gate_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_teacher_gate_effect_delta_clean_minus_corrupt": (
            _weighted_mean(mild_vision, "target_teacher_gate_effect_delta_clean_minus_corrupt")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_reliability_gate_effect_alignment_ratio": (
            _weighted_mean(mild_vision, "reliability_gate_effect_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_gate_effect_attention_alignment_ratio": (
            _weighted_mean(mild_vision, "gate_effect_attention_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_gate_effect_score_alignment_ratio": (
            _weighted_mean(mild_vision, "gate_effect_score_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_prebias_score_final_score_alignment_ratio": (
            _weighted_mean(mild_vision, "prebias_score_final_score_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_reliability_gate_alignment_ratio": (
            _weighted_mean(mild_vision, "reliability_gate_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_gate_attention_alignment_ratio": (
            _weighted_mean(mild_vision, "gate_attention_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "vision_reliability_attention_alignment_ratio": (
            _weighted_mean(mild_vision, "reliability_attention_alignment_ratio")
            if not mild_vision.empty
            else 0.0
        ),
        "nontarget_rescue_delta": _weighted_mean(mild_vision, "nontarget_rescue_delta") if not mild_vision.empty else 0.0,
        "clean_score_anchor_error": _weighted_mean(clean, "clean_score_anchor_error") if not clean.empty else 0.0,
        "clean_action_anchor_error": _weighted_mean(clean, "clean_action_anchor_error") if not clean.empty else 0.0,
        "clean_route_attention_anchor_error": _weighted_mean(clean, "clean_route_attention_anchor_error") if not clean.empty else 0.0,
        "clean_reliability_anchor_error": _weighted_mean(clean, "clean_reliability_anchor_error") if not clean.empty else 0.0,
        "clean_bias_score_anchor_error": _weighted_mean(clean, "clean_bias_score_anchor_error") if not clean.empty else 0.0,
        "clean_prebias_score_anchor_error": _weighted_mean(clean, "clean_prebias_score_anchor_error") if not clean.empty else 0.0,
        "attention_bias_score_mean_vision_clean": _weighted_mean(mild_vision, "attention_bias_score_mean_vision_clean") if not mild_vision.empty else 0.0,
        "attention_bias_score_mean_vision_corrupt": _weighted_mean(mild_vision, "attention_bias_score_mean_vision_corrupt") if not mild_vision.empty else 0.0,
        "attention_prebias_score_mean_vision_clean": _weighted_mean(mild_vision, "attention_prebias_score_mean_vision_clean") if not mild_vision.empty else 0.0,
        "attention_prebias_score_mean_vision_corrupt": _weighted_mean(mild_vision, "attention_prebias_score_mean_vision_corrupt") if not mild_vision.empty else 0.0,
        "attention_bias_gate_effect_mean_vision_clean": _weighted_mean(mild_vision, "attention_bias_gate_effect_mean_vision_clean") if not mild_vision.empty else 0.0,
        "attention_bias_gate_effect_mean_vision_corrupt": _weighted_mean(mild_vision, "attention_bias_gate_effect_mean_vision_corrupt") if not mild_vision.empty else 0.0,
        "teacher_source": (
            str(mild_vision["teacher_source"].iloc[0])
            if not mild_vision.empty and "teacher_source" in mild_vision.columns
            else "disabled"
        ),
    }
    teacher_gate_effect_absmax = 0.0
    teacher_bias_score_absmax = 0.0
    if not mild_vision.empty and "teacher_preserve_signal_absmax" in mild_vision.columns:
        teacher_preserve_signal_absmax = float(
            pd.to_numeric(mild_vision["teacher_preserve_signal_absmax"], errors="coerce").fillna(0.0).max()
        )
    else:
        teacher_preserve_signal_absmax = abs(float(summary["vision_teacher_gate_effect_delta_clean_minus_corrupt"]))
    if not mild_vision.empty and "teacher_effect_nonflat_preserve_flag" in mild_vision.columns:
        teacher_effect_nonflat_preserve_flag = int(
            (
                pd.to_numeric(mild_vision["teacher_effect_nonflat_preserve_flag"], errors="coerce").fillna(0.0) > 0.0
            ).any()
        )
    else:
        teacher_effect_nonflat_preserve_flag = int(teacher_preserve_signal_absmax >= 1.0e-5)
    if not mild_vision.empty and "teacher_effect_nonflat_vision_flag" in mild_vision.columns:
        teacher_effect_nonflat_vision_flag = int(
            (
                pd.to_numeric(mild_vision["teacher_effect_nonflat_vision_flag"], errors="coerce").fillna(0.0) > 0.0
            ).any()
        )
    else:
        teacher_effect_nonflat_vision_flag = int(
            abs(float(summary["vision_teacher_gate_effect_delta_clean_minus_corrupt"])) >= 1.0e-5
        )
    if not mild_vision.empty and "teacher_gate_effect_absmax" in mild_vision.columns:
        teacher_gate_effect_absmax = float(
            pd.to_numeric(mild_vision["teacher_gate_effect_absmax"], errors="coerce").fillna(0.0).max()
        )
    if not mild_vision.empty and "teacher_bias_score_absmax" in mild_vision.columns:
        teacher_bias_score_absmax = float(
            pd.to_numeric(mild_vision["teacher_bias_score_absmax"], errors="coerce").fillna(0.0).max()
        )
    if not mild_vision.empty and "teacher_actuation_regime_mismatch_flag" in mild_vision.columns:
        teacher_actuation_regime_mismatch_flag = int(
            (
                pd.to_numeric(mild_vision["teacher_actuation_regime_mismatch_flag"], errors="coerce").fillna(0.0)
                > 0.0
            ).any()
        )
    else:
        teacher_actuation_regime_mismatch_flag = 0
    if not mild_vision.empty and "teacher_preserve_active_flag" in mild_vision.columns:
        teacher_preserve_active_flag = int(
            (
                pd.to_numeric(mild_vision["teacher_preserve_active_flag"], errors="coerce").fillna(0.0)
                > 0.0
            ).any()
        )
    else:
        teacher_preserve_active_flag = int(
            teacher_effect_nonflat_preserve_flag == 1 and teacher_actuation_regime_mismatch_flag == 0
        )
    summary["teacher_preserve_signal_absmax"] = teacher_preserve_signal_absmax
    summary["teacher_effect_nonflat_preserve_flag"] = teacher_effect_nonflat_preserve_flag
    summary["teacher_gate_effect_nonflat_preserve_flag"] = teacher_effect_nonflat_preserve_flag
    summary["teacher_effect_nonflat_vision_flag"] = teacher_effect_nonflat_vision_flag
    summary["teacher_effect_nonflat_flag"] = teacher_effect_nonflat_preserve_flag
    summary["teacher_gate_effect_absmax"] = teacher_gate_effect_absmax
    summary["teacher_bias_score_absmax"] = teacher_bias_score_absmax
    summary["teacher_actuation_regime_mismatch_flag"] = teacher_actuation_regime_mismatch_flag
    summary["teacher_preserve_active_flag"] = teacher_preserve_active_flag
    summary["vision_gate_effect_absmax"] = max(
        abs(float(summary["attention_bias_gate_effect_mean_vision_clean"])),
        abs(float(summary["attention_bias_gate_effect_mean_vision_corrupt"])),
    )
    summary["vision_bias_score_absmax"] = max(
        abs(float(summary["attention_bias_score_mean_vision_clean"])),
        abs(float(summary["attention_bias_score_mean_vision_corrupt"])),
    )
    summary["teacher_gate_effect_flat_flag"] = int(1 - summary["teacher_effect_nonflat_flag"])
    baseline_clean, baseline_clean_expected, baseline_clean_file_missing, baseline_clean_match_missing = _load_baseline_clean_frame(
        config,
        candidate_df,
        smoke=smoke,
    )
    summary["baseline_clean_file_missing_flag"] = int(baseline_clean_file_missing)
    summary["baseline_clean_match_missing_flag"] = int(baseline_clean_match_missing)
    summary["baseline_clean_missing_flag"] = int(
        baseline_clean_expected and (baseline_clean_file_missing or baseline_clean_match_missing)
    )
    for modality in MODALITY_NAMES:
        summary[f"{modality}_clean_reliability_drift"] = 0.0
    for modality in ["distance", "ego", "route"]:
        summary[f"{modality}_clean_attention_drift"] = 0.0
    summary["max_non_target_clean_attention_drift"] = 0.0
    summary["max_non_ego_clean_reliability_drift"] = 0.0
    summary["max_non_target_clean_attention_drift_standalone"] = 0.0
    summary["max_non_ego_clean_reliability_drift_standalone"] = 0.0
    if baseline_clean is not None and not clean.empty:
        attention_drifts = []
        for modality in ["distance", "ego", "route"]:
            baseline_key = f"attention_mean_{modality}"
            current_key = f"attention_mean_{modality}_clean"
            if baseline_key in baseline_clean.columns and current_key in clean.columns:
                drift = _weighted_mean(clean, current_key) - _weighted_mean(baseline_clean, baseline_key)
                summary[f"{modality}_clean_attention_drift"] = drift
                attention_drifts.append(abs(drift))
        if attention_drifts:
            summary["max_non_target_clean_attention_drift"] = max(attention_drifts)
            summary["max_non_target_clean_attention_drift_standalone"] = max(attention_drifts)
        reliability_drifts = []
        for modality in MODALITY_NAMES:
            key = f"reliability_mean_{modality}"
            if key in baseline_clean.columns and key in clean.columns:
                drift = _weighted_mean(clean, key) - _weighted_mean(baseline_clean, key)
                summary[f"{modality}_clean_reliability_drift"] = drift
                if modality != "ego":
                    reliability_drifts.append(abs(drift))
        if reliability_drifts:
            summary["max_non_ego_clean_reliability_drift"] = max(reliability_drifts)
            summary["max_non_ego_clean_reliability_drift_standalone"] = max(reliability_drifts)
    summary["pass_fail"] = evaluate_fusion_shift_proof_summary(summary, scoring_cfg, smoke=smoke)
    return summary


def choose_fusion_shift_proof_recommendation(score_df: pd.DataFrame) -> str:
    pool = score_df.copy()
    distill_enabled = _numeric_series(pool, "teacher_effect_distill_enabled") > 0.0
    teacher_nonflat = _numeric_series(pool, "teacher_effect_nonflat_preserve_flag") > 0.0
    teacher_mismatch = _numeric_series(pool, "teacher_actuation_regime_mismatch_flag") > 0.0
    teacher_preserve_active = _numeric_series(pool, "teacher_preserve_active_flag") > 0.0
    valid_distilled = distill_enabled & teacher_nonflat & (~teacher_mismatch) & teacher_preserve_active
    rank = pd.Series(5, index=pool.index, dtype=int)
    pass_mask = pool["pass_fail"].astype(str) == "PASS"
    warn_mask = pool["pass_fail"].astype(str) == "WARN"
    rank.loc[pass_mask & (~distill_enabled)] = 0
    rank.loc[pass_mask & valid_distilled] = 1
    rank.loc[pass_mask] = rank.loc[pass_mask].where(rank.loc[pass_mask] < 5, 2)
    rank.loc[warn_mask & (~distill_enabled)] = 3
    rank.loc[warn_mask & valid_distilled] = 4
    pool["_recommend_rank"] = rank
    pool["_vision_final_score_delta"] = _numeric_series(pool, "vision_final_score_delta_clean_minus_corrupt")
    pool["_vision_attention_delta"] = _numeric_series(pool, "vision_attention_delta_clean_minus_corrupt")
    pool["_vision_gate_effect_delta"] = _numeric_series(pool, "vision_gate_effect_delta_clean_minus_corrupt")
    pool["_clean_success_rate"] = _numeric_series(pool, "clean_success_rate_standalone")
    pool = pool.sort_values(
        [
            "_recommend_rank",
            "_vision_final_score_delta",
            "_vision_attention_delta",
            "_vision_gate_effect_delta",
            "_clean_success_rate",
        ],
        ascending=[True, False, False, False, False],
    )
    return str(pool.iloc[0]["candidate_name"])


def generate_fusion_shift_proof_report(
    score_df: pd.DataFrame,
    output_path: Path,
    recommended: str,
) -> None:
    row = score_df.loc[score_df["candidate_name"] == recommended].iloc[0]
    lines = [
        "# Stage 2 Fusion Shift Proof Report",
        "",
        f"- Recommended: **{recommended}**",
        f"- Verdict: **{row['pass_fail']}**",
        f"- Vision reliability delta (clean-corrupt): `{row['vision_reliability_delta_clean_minus_corrupt']:.6f}`",
        f"- Vision gate-effect delta (clean-corrupt): `{row['vision_gate_effect_delta_clean_minus_corrupt']:.6f}`",
        f"- Vision gate-effect abs max: `{row.get('vision_gate_effect_absmax', 0.0):.6f}`",
        f"- Vision bias-score delta (clean-corrupt): `{row.get('vision_bias_score_delta_clean_minus_corrupt', 0.0):.6f}`",
        f"- Vision pre-bias score delta (clean-corrupt): `{row.get('vision_prebias_score_delta_clean_minus_corrupt', 0.0):.6f}`",
        f"- Vision raw gate delta (debug): `{row.get('vision_gate_delta_clean_minus_corrupt', 0.0):.6f}`",
        f"- Vision teacher gate-effect delta (clean-corrupt): `{row.get('vision_teacher_gate_effect_delta_clean_minus_corrupt', 0.0):.6f}`",
        f"- Vision attention delta (clean-corrupt): `{row['vision_attention_delta_clean_minus_corrupt']:.6f}`",
        f"- Vision final score delta (clean-corrupt): `{row['vision_final_score_delta_clean_minus_corrupt']:.6f}`",
        f"- Vision bias score abs max: `{row.get('vision_bias_score_absmax', 0.0):.6f}`",
        f"- Vision base source gate delta (debug): `{row.get('vision_source_gate_base_delta_clean_minus_corrupt', 0.0):.6f}`",
        f"- Vision reliability->gate-effect alignment ratio: `{row['vision_reliability_gate_effect_alignment_ratio']:.3f}`",
        f"- Vision gate-effect->attention alignment ratio: `{row['vision_gate_effect_attention_alignment_ratio']:.3f}`",
        f"- Vision gate-effect->score alignment ratio: `{row.get('vision_gate_effect_score_alignment_ratio', 0.0):.3f}`",
        f"- Vision pre-bias->score alignment ratio: `{row.get('vision_prebias_score_final_score_alignment_ratio', 0.0):.3f}`",
        f"- Vision reliability->attention alignment ratio (debug): `{row['vision_reliability_attention_alignment_ratio']:.3f}`",
        f"- Nontarget rescue delta: `{row['nontarget_rescue_delta']:.6f}`",
        f"- Max non-target clean attention drift (standalone): `{row['max_non_target_clean_attention_drift_standalone']:.6f}`",
        f"- Max non-ego clean reliability drift (standalone): `{row['max_non_ego_clean_reliability_drift_standalone']:.6f}`",
        f"- Clean success rate (standalone): `{row['clean_success_rate_standalone']:.3f}`",
        f"- Clean route attention anchor error: `{row.get('clean_route_attention_anchor_error', 0.0):.6f}`",
        f"- Clean reliability anchor error: `{row.get('clean_reliability_anchor_error', 0.0):.6f}`",
        f"- Clean bias-score anchor error: `{row.get('clean_bias_score_anchor_error', 0.0):.6f}`",
        f"- Clean pre-bias score anchor error: `{row.get('clean_prebias_score_anchor_error', 0.0):.6f}`",
        f"- Teacher-effect distill enabled: `{int(row.get('teacher_effect_distill_enabled', 0))}`",
        f"- Teacher preserve signal abs max: `{row.get('teacher_preserve_signal_absmax', 0.0):.6f}`",
        f"- Teacher-effect nonflat preserve flag: `{int(row.get('teacher_effect_nonflat_preserve_flag', 0))}`",
        f"- Teacher-effect nonflat vision flag: `{int(row.get('teacher_effect_nonflat_vision_flag', 0))}`",
        f"- Teacher gate-effect abs max: `{row.get('teacher_gate_effect_absmax', 0.0):.6f}`",
        f"- Teacher bias-score abs max: `{row.get('teacher_bias_score_absmax', 0.0):.6f}`",
        f"- Teacher actuation regime mismatch flag: `{int(row.get('teacher_actuation_regime_mismatch_flag', 0))}`",
        f"- Teacher preserve active flag: `{int(row.get('teacher_preserve_active_flag', 0))}`",
        f"- Teacher source: `{row.get('teacher_source', 'disabled')}`",
        f"- Baseline clean missing flag: `{int(row.get('baseline_clean_missing_flag', 0))}`",
        f"- Baseline clean file missing flag: `{int(row.get('baseline_clean_file_missing_flag', 0))}`",
        f"- Baseline clean exact-match missing flag: `{int(row.get('baseline_clean_match_missing_flag', 0))}`",
        f"- Mixed eval episode count flag: `{int(row.get('mixed_eval_episode_count_flag', 0))}`",
        "",
        "## Final Summary",
        "",
        score_df.to_markdown(index=False),
        "",
        "- 다음 단계 판단:",
        f"  - {'vision-first Reliability -> Fusion Shift 증명이 완료됨' if row['pass_fail'] == 'PASS' else '아직 fusion shift proof를 더 고정해야 함'}",
    ]
    if int(row.get("mixed_eval_episode_count_flag", 0)) == 1:
        lines.extend(
            [
                "",
                "- 진단:",
                "  - mixed eval episode counts detected; rerun on clean output root",
            ]
        )
    if int(row.get("baseline_clean_missing_flag", 0)) == 1:
        lines.extend(
            [
                "",
                "- 진단:",
                "  - baseline clean frame was unavailable for this run",
                "  - clean drift metrics may be understated because baseline comparison could not be computed",
            ]
        )
    if int(row.get("baseline_clean_match_missing_flag", 0)) == 1:
        lines.extend(
            [
                "",
                "- 진단:",
                "  - smoke baseline file exists, but no exact-match clean row was found for seed/run_mode/num_eval_episodes",
                "  - cross-seed or cross-mode drift fallback is disabled, so this run cannot be treated as PASS",
            ]
        )
    lines.extend(
        [
            "",
            "- 참고:",
            "  - legacy `reference_same_state_action_diff_clean_vs_corrupt` alias has been removed from official telemetry",
            "  - verdicts use only `same_state_action_diff_reference_clean_vs_corrupt` and `same_state_action_diff_reference_clean_vs_current_corrupt` when needed",
        ]
    )
    if "vision_gate_effect_absmax" in score_df.columns and bool(
        (pd.to_numeric(score_df["vision_gate_effect_absmax"], errors="coerce").fillna(0.0) < 1.0e-4).all()
    ):
        lines.extend(
            [
                "",
                "- 진단:",
                "  - 모든 후보에서 vision gate-effect amplitude가 매우 작음 (`< 1e-4`)",
                "  - 다음 단계는 output-scale 확대 또는 adapter-readout 강화를 우선 검토해야 함",
            ]
        )
    if {"teacher_effect_distill_enabled", "teacher_effect_nonflat_preserve_flag"}.issubset(score_df.columns):
        distilled = score_df[
            pd.to_numeric(score_df["teacher_effect_distill_enabled"], errors="coerce").fillna(0.0) > 0.0
        ]
        invalid_distilled = distilled
        if "teacher_actuation_regime_mismatch_flag" in invalid_distilled.columns:
            invalid_distilled = invalid_distilled[
                (_numeric_series(invalid_distilled, "teacher_effect_nonflat_preserve_flag") <= 0.0)
                | (_numeric_series(invalid_distilled, "teacher_actuation_regime_mismatch_flag") > 0.0)
                | (_numeric_series(invalid_distilled, "teacher_preserve_active_flag") <= 0.0)
            ]
        if not distilled.empty and len(invalid_distilled) == len(distilled):
            lines.extend(
                [
                    "",
                    "- 진단:",
                    "  - distilled 후보 전부에서 preserve 대상 teacher signal이 flat하거나 actuation regime mismatch가 있음",
                    "  - 현재 smoke 해석은 anchor-only 후보 결과를 중심으로 보는 편이 안전함",
                ]
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_candidate_rows(
    candidate_name: str,
    config: dict[str, Any],
    output_root: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    if bool(config.get("evaluation_only", False)):
        return run_stage2_eval_only_candidate(
            candidate_name=candidate_name,
            config=config,
            output_root=output_root,
            metrics_filename="fusion_shift_eval_rows.json",
            smoke=smoke,
        )
    return run_stage2_candidate(
        candidate_name=candidate_name,
        config=config,
        output_root=output_root,
        metrics_filename="fusion_shift_eval_rows.json",
        smoke=smoke,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    experiment_name = str(config["experiment_name"])
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / f"{experiment_name}_config_snapshot.yaml", config)
    baseline_path = _resolve_baseline_eval_rows_path(config, smoke=args.smoke)
    if args.smoke and baseline_path is not None and not baseline_path.exists():
        message = "\n".join(
            [
                f"# {experiment_name} Smoke Baseline Missing",
                "",
                f"- Expected smoke baseline eval rows at `{baseline_path}`",
                "- Generate the reference smoke baseline before running this pilot/check smoke.",
            ]
        )
        (summary_dir / f"{experiment_name}_final_recommendation.md").write_text(message + "\n", encoding="utf-8")
        (summary_dir / f"{experiment_name}_pass_fail_report.md").write_text(message + "\n", encoding="utf-8")
        print(f"Smoke baseline missing: {baseline_path}")
        return

    ordered_candidates = list(config.get("candidate_order", config["candidates"].keys()))
    all_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for candidate_name in ordered_candidates:
        rows = _run_candidate_rows(candidate_name, config, output_root, smoke=args.smoke)
        candidate_root = ensure_dir(output_root / candidate_name)
        mode = "smoke" if args.smoke else "full"
        (candidate_root / f"eval_rows_{mode}.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
        if not args.smoke:
            (candidate_root / "eval_rows.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
        candidate_df = pd.DataFrame(rows)
        if "probe_name" not in candidate_df.columns and "corruption_type" in candidate_df.columns:
            candidate_df["probe_name"] = candidate_df["corruption_type"]
        candidate_df = candidate_df.sort_values(["probe_name", "seed"]).reset_index(drop=True)
        all_rows.extend(candidate_df.to_dict(orient="records"))
        score_rows.append(build_fusion_shift_proof_score_summary(candidate_name, candidate_df, config, smoke=args.smoke))

    metrics_df = pd.DataFrame(all_rows).sort_values(["candidate_name", "probe_name", "seed"]).reset_index(drop=True)
    metrics_df.to_csv(summary_dir / f"{experiment_name}_metrics.csv", index=False)
    metrics_df.to_json(
        summary_dir / f"{experiment_name}_metrics.json",
        orient="records",
        indent=2,
    )

    score_df = pd.DataFrame(score_rows)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        [
            "_order",
            "teacher_effect_distill_enabled",
            "vision_reliability_delta_clean_minus_corrupt",
            "vision_bias_score_delta_clean_minus_corrupt",
            "vision_prebias_score_delta_clean_minus_corrupt",
            "vision_gate_effect_delta_clean_minus_corrupt",
            "vision_attention_delta_clean_minus_corrupt",
            "vision_final_score_delta_clean_minus_corrupt",
            "vision_reliability_gate_effect_alignment_ratio",
            "vision_gate_effect_attention_alignment_ratio",
            "vision_gate_effect_score_alignment_ratio",
            "vision_prebias_score_final_score_alignment_ratio",
            "nontarget_rescue_delta",
            "max_non_target_clean_attention_drift_standalone",
            "max_non_ego_clean_reliability_drift_standalone",
            "clean_success_rate_standalone",
            "clean_route_attention_anchor_error",
            "clean_reliability_anchor_error",
            "clean_bias_score_anchor_error",
            "clean_prebias_score_anchor_error",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
        ],
    ).drop(columns="_order").reset_index(drop=True)
    score_df.to_csv(summary_dir / f"{experiment_name}_score_summary.csv", index=False)
    (summary_dir / f"{experiment_name}_pass_fail_report.md").write_text(
        f"# {experiment_name} PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    recommended = choose_fusion_shift_proof_recommendation(score_df)
    generate_fusion_shift_proof_report(
        score_df=score_df,
        output_path=summary_dir / f"{experiment_name}_final_recommendation.md",
        recommended=recommended,
    )
    print("Recommended:", recommended)


if __name__ == "__main__":
    main()
