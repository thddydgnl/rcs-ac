from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_ROOT = Path(
    "/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_stage1_to_stage8_audit_2026-03-29"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _try_read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status_rank(status: str) -> int:
    order = {
        "PASS": 0,
        "WARN": 1,
        "FAIL": 2,
        "BLOCKED": 3,
        "INCOMPLETE": 4,
    }
    return order.get(status, 9)


def _parse_preflight(summary_dir: Path) -> dict[str, Any]:
    path = summary_dir / "preflight_pytest.txt"
    if not path.exists():
        return {
            "path": str(path),
            "result_line": "missing",
            "status": "INCOMPLETE",
        }
    text = path.read_text(encoding="utf-8").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    result_line = next((line for line in reversed(lines) if "passed" in line or "failed" in line), "unknown")
    status = "PASS" if "passed" in result_line and "failed" not in result_line else "FAIL"
    return {
        "path": str(path),
        "result_line": result_line,
        "status": status,
    }


def _manifest(summary_dir: Path) -> dict[str, Any]:
    return _try_read_json(summary_dir / "dependency_manifest.json") or {}


def _locked_baseline(summary_dir: Path) -> dict[str, Any] | None:
    payload = _try_read_json(summary_dir / "step8_locked_baseline.json")
    if not payload:
        return None
    if payload.get("status") != "LOCKED":
        return payload
    baseline_manifest_path = Path(str(payload.get("baseline_manifest_path", "")))
    if not baseline_manifest_path.exists():
        return {
            "status": "FAILED",
            "error": f"baseline manifest missing: {baseline_manifest_path}",
        }
    return payload


def _step1(summary_dir: Path) -> dict[str, Any]:
    schema_path = summary_dir / "step1_env_schema.json"
    if not schema_path.exists():
        return {
            "step": "1",
            "name": "환경/상태 정의",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "vision/distance/ego/route/aux 구조와 trainer/model 입력 shape 일치",
            "notes": "step1_env_schema.json missing",
        }
    schema = _read_json(schema_path)
    expected = ["vision", "distance", "ego", "route", "aux"]
    expected_set = set(expected)
    observed_keys = schema.get("observation_keys") or []
    logical_expected_keys = schema.get("stage1_expected_modality_keys") or []
    observed_shapes = schema.get("observation_shapes") or {}
    observed_set = set(observed_keys)
    logical_expected_set = set(logical_expected_keys)
    keys_ok = observed_set == expected_set
    expected_keys_ok = logical_expected_set == expected_set
    shapes_ok = all(key in observed_shapes for key in expected)
    action_ok = schema.get("action_shape") == [2]
    aux_ok = schema.get("aux_dim") == 8
    risk_ok = bool(schema.get("has_risk_stats_in_info")) and bool(schema.get("has_quality_stats_in_info"))
    status = "PASS" if keys_ok and expected_keys_ok and shapes_ok and action_ok and aux_ok and risk_ok else "FAIL"
    if status == "PASS":
        notes = "schema keys/shapes are valid; Dict key order is non-canonical but accepted"
    else:
        failure_reasons = []
        if not keys_ok:
            failure_reasons.append("observation key set mismatch")
        if not expected_keys_ok:
            failure_reasons.append("logical expected key set mismatch")
        if not shapes_ok:
            failure_reasons.append("missing observation shape entry")
        if not action_ok:
            failure_reasons.append("action shape mismatch")
        if not aux_ok:
            failure_reasons.append("aux dim mismatch")
        if not risk_ok:
            failure_reasons.append("risk/quality info flags missing")
        notes = "; ".join(failure_reasons) if failure_reasons else "env schema mismatch"
    return {
        "step": "1",
        "name": "환경/상태 정의",
        "status": status,
        "evidence": [str(schema_path)],
        "metrics": (
            f"obs_keys={schema.get('observation_keys')} "
            f"obs_keys_sorted={schema.get('observation_keys_sorted')} "
            f"shapes={schema.get('observation_shapes')} "
            f"action_shape={schema.get('action_shape')} aux_dim={schema.get('aux_dim')}"
        ),
        "pass_criteria": "vision/distance/ego/route/aux 구조와 trainer/model 입력 shape 일치",
        "notes": notes,
    }


def _step2_and_3(audit_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage0_dir = audit_root / "stage0_logs"
    csv_path = stage0_dir / "stage0_episode_summaries.csv"
    jsonl_path = stage0_dir / "stage0_episode_summaries.jsonl"
    rows = _try_read_csv_rows(csv_path)
    if not rows:
        incomplete = {
            "step": "2",
            "name": "reward/done",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "heuristic goal 접근 가능, random에서도 reward/done 안정, 종료 사유 일치",
            "notes": "stage0 logs missing",
        }
        return incomplete, {
            **incomplete,
            "step": "3",
            "name": "Stage0 환경 검증",
            "pass_criteria": "random/heuristic x clean/corrupted 4조합 실행과 csv/jsonl 저장",
        }

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["policy_type"], row["corruption_mode"])
        groups.setdefault(key, []).append(row)
    required_groups = {
        ("random", "clean"),
        ("random", "corrupted"),
        ("heuristic", "clean"),
        ("heuristic", "corrupted"),
    }
    groups_ok = all(groups.get(key) for key in required_groups)
    reasons_ok = all(row.get("terminated_reason") for row in rows)
    heuristic_success = (
        sum(_safe_float(row["success"]) for row in groups.get(("heuristic", "clean"), []))
        / max(len(groups.get(("heuristic", "clean"), [])), 1)
    )
    random_rows = len(groups.get(("random", "clean"), [])) + len(groups.get(("random", "corrupted"), []))
    step2_status = "PASS" if groups_ok and reasons_ok and heuristic_success > 0.0 and random_rows >= 40 else "FAIL"

    def _group_summary(key: tuple[str, str]) -> str:
        group = groups.get(key, [])
        if not group:
            return f"{key[0]}/{key[1]}:missing"
        n = len(group)
        reward = sum(_safe_float(row["total_reward"]) for row in group) / n
        success = sum(_safe_float(row["success"]) for row in group) / n
        collision = sum(_safe_float(row["collision"]) for row in group) / n
        timeout = sum(_safe_float(row["timeout"]) for row in group) / n
        return (
            f"{key[0]}/{key[1]}:n={n},reward={reward:.3f},success={success:.3f},"
            f"collision={collision:.3f},timeout={timeout:.3f}"
        )

    step2 = {
        "step": "2",
        "name": "reward/done",
        "status": step2_status,
        "evidence": [str(csv_path), str(jsonl_path)],
        "metrics": " | ".join(_group_summary(key) for key in sorted(required_groups)),
        "pass_criteria": "heuristic goal 접근 가능, random에서도 reward/done 안정, 종료 사유 일치",
        "notes": (
            "heuristic success는 낮지만 0보다 크고 종료 사유/로그는 안정적"
            if step2_status == "PASS"
            else "heuristic success 또는 stage0 termination logging 불충분"
        ),
    }
    step3 = {
        "step": "3",
        "name": "Stage0 환경 검증",
        "status": "PASS" if step2_status == "PASS" and jsonl_path.exists() else "FAIL",
        "evidence": [str(csv_path), str(jsonl_path)],
        "metrics": f"episodes={len(rows)} groups={sorted(required_groups)}",
        "pass_criteria": "random/heuristic x clean/corrupted 4조합 실행과 csv/jsonl 저장",
        "notes": "stage0 smoke 포함 전 조합 로그 저장 완료" if jsonl_path.exists() else "stage0 artifact missing",
    }
    return step2, step3


def _step4(audit_root: Path) -> dict[str, Any]:
    summary_dir = audit_root / "baseline_comparison" / "summary"
    pass_fail = _try_read_json(summary_dir / "pass_fail_report.json")
    rows = _try_read_csv_rows(summary_dir / "comparison_metrics.csv")
    if pass_fail is None or not rows:
        return {
            "step": "4",
            "name": "baseline A/B/C",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "baseline B >= A, baseline C가 B보다 현저히 불안정하지 않음",
            "notes": "baseline comparison summary missing",
        }
    rows_by_name = {row["model_name"]: row for row in rows}

    def _metric(name: str) -> str:
        row = rows_by_name[name]
        return (
            f"{name}:success={_safe_float(row['final_eval_success_rate_mean']):.3f}±{_safe_float(row['final_eval_success_rate_std']):.3f},"
            f"return={_safe_float(row['final_eval_return_mean']):.3f}±{_safe_float(row['final_eval_return_std']):.3f},"
            f"progress={_safe_float(row['final_eval_avg_progress_mean']):.3f}"
        )

    status = str(pass_fail.get("overall_grade", "INCOMPLETE"))
    notes = (
        "baseline ordering이 계획 기준을 만족"
        if status == "PASS"
        else "baseline ordering이 계획 기준을 만족하지 않음"
    )
    return {
        "step": "4",
        "name": "baseline A/B/C",
        "status": status,
        "evidence": [
            str(summary_dir / "comparison_metrics.csv"),
            str(summary_dir / "pass_fail_report.md"),
            str(summary_dir / "attention_summary.csv"),
        ],
        "metrics": " | ".join(
            _metric(name) for name in ("a_ego_route", "b_early_fusion", "c_attention_fusion") if name in rows_by_name
        ),
        "pass_criteria": "baseline B >= A, baseline C가 B보다 현저히 불안정하지 않음",
        "notes": notes,
    }


def _parse_grade_from_markdown(path: Path) -> str:
    text = _read_text(path)
    match = re.search(r"Grade:\s*\*\*(PASS|WARN|FAIL)\*\*", text)
    if match:
        return match.group(1)
    return "INCOMPLETE"


def _step5(audit_root: Path, prev_status: str) -> dict[str, Any]:
    if prev_status != "PASS":
        return {
            "step": "5",
            "name": "RCS-AC v1",
            "status": "BLOCKED",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "revalidation grade가 PASS이고 standard_v1이 baseline_c보다 materially worse가 아님",
            "notes": "Step 4 PASS 이후에만 실행",
        }
    reval_dir = audit_root / "transition_to_stage1" / "revalidation" / "summary"
    report_path = reval_dir / "revalidation_pass_fail_report.md"
    compare_path = reval_dir / "comparison_against_baseline.csv"
    if not report_path.exists() or not compare_path.exists():
        return {
            "step": "5",
            "name": "RCS-AC v1",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "revalidation grade가 PASS이고 standard_v1이 baseline_c보다 materially worse가 아님",
            "notes": "transition_to_stage1 revalidation outputs missing",
        }
    rows = _try_read_csv_rows(compare_path)
    row = rows[0] if rows else {}
    grade = _parse_grade_from_markdown(report_path)
    return {
        "step": "5",
        "name": "RCS-AC v1",
        "status": grade,
        "evidence": [str(report_path), str(compare_path)],
        "metrics": (
            f"success_diff={_safe_float(row.get('success_diff_vs_baseline')):.3f} "
            f"return_diff={_safe_float(row.get('return_diff_vs_baseline')):.3f} "
            f"progress_diff={_safe_float(row.get('progress_diff_vs_baseline')):.3f}"
        ),
        "pass_criteria": "revalidation grade가 PASS이고 standard_v1이 baseline_c보다 materially worse가 아님",
        "notes": "standard_v1 -> Stage1 warm-up transition gate",
    }


def _step6(audit_root: Path, prev_status: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if prev_status != "PASS":
        return {
            "step": "6",
            "name": "Stage1 warm-up",
            "status": "BLOCKED",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "fixed evaluator 기준 reliability가 sane하고 downstream winner가 명확",
            "notes": "Step 5 PASS 이후에만 실행",
        }
    summary_dir = audit_root / "stage1_vision_ego_logit_triplet_calibration" / "summary"
    score_path = summary_dir / "vision_ego_logit_triplet_score_summary.csv"
    if not score_path.exists():
        return {
            "step": "6",
            "name": "Stage1 warm-up",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "fixed evaluator 기준 reliability가 sane하고 downstream winner가 명확",
            "notes": "Stage1 calibration summary missing",
        }
    rows = _try_read_csv_rows(score_path)
    row = rows[0] if rows else {}
    winner = manifest.get("step6_winner_name") or row.get("candidate_name", "unknown")
    return {
        "step": "6",
        "name": "Stage1 warm-up",
        "status": str(row.get("pass_fail", "INCOMPLETE")),
        "evidence": [str(score_path), str(summary_dir / "vision_ego_logit_triplet_final_recommendation.md")],
        "metrics": (
            f"winner={winner} "
            f"calibration_score={_safe_float(row.get('calibration_score')):.3f} "
            f"clean_success={_safe_float(row.get('clean_success_rate')):.3f} "
            f"target_positive_ratio={_safe_float(row.get('target_positive_ratio')):.3f}"
        ),
        "pass_criteria": "fixed evaluator 기준 reliability가 sane하고 downstream winner가 명확",
        "notes": "Stage1 calibration winner and secondary are recorded in dependency manifest",
    }


def _step7(audit_root: Path, prev_status: str) -> dict[str, Any]:
    if prev_status != "PASS":
        return {
            "step": "7",
            "name": "RCS-AC v2 진단",
            "status": "BLOCKED",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "mild_pilot PASS + reliability_proof PASS + fusion_shift_proof PASS/WARN",
            "notes": "Step 6 PASS 이후에만 실행",
        }
    mild_path = audit_root / "stage2_mild_pilot" / "summary" / "stage2_mild_pilot_score_summary.csv"
    proof_path = audit_root / "stage2_reliability_proof" / "summary" / "stage2_reliability_proof_score_summary.csv"
    fusion_path = audit_root / "stage2_fusion_shift_proof_pilot" / "summary" / "stage2_fusion_shift_proof_pilot_score_summary.csv"
    if not mild_path.exists() or not proof_path.exists() or not fusion_path.exists():
        return {
            "step": "7",
            "name": "RCS-AC v2 진단",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "mild_pilot PASS + reliability_proof PASS + fusion_shift_proof PASS/WARN",
            "notes": "one or more Stage2 diagnostic summaries missing",
        }
    mild_row = (_try_read_csv_rows(mild_path) or [{}])[0]
    proof_row = (_try_read_csv_rows(proof_path) or [{}])[0]
    fusion_row = (_try_read_csv_rows(fusion_path) or [{}])[0]
    mild_status = str(mild_row.get("pass_fail", "INCOMPLETE"))
    proof_status = str(proof_row.get("pass_fail", "INCOMPLETE"))
    fusion_status = str(fusion_row.get("pass_fail", "INCOMPLETE"))
    gate_pass = mild_status == "PASS" and proof_status == "PASS" and fusion_status in {"PASS", "WARN"}
    status = "PASS" if gate_pass else "FAIL"
    return {
        "step": "7",
        "name": "RCS-AC v2 진단",
        "status": status,
        "evidence": [str(mild_path), str(proof_path), str(fusion_path)],
        "metrics": (
            f"mild={mild_row.get('candidate_name', 'n/a')}:{mild_status} "
            f"proof={proof_row.get('candidate_name', 'n/a')}:{proof_status} "
            f"fusion={fusion_row.get('candidate_name', 'n/a')}:{fusion_status}"
        ),
        "pass_criteria": "mild_pilot PASS + reliability_proof PASS + fusion_shift_proof PASS/WARN",
        "notes": "fusion-shift는 diagnostic이므로 WARN까지 허용",
    }


def _row_for_candidate(path: Path, candidate_name: str | None) -> dict[str, str]:
    rows = _try_read_csv_rows(path)
    if not rows:
        return {}
    if candidate_name is None:
        return rows[0]
    for row in rows:
        if row.get("candidate_name") == candidate_name:
            return row
    return rows[0]


def _step8(audit_root: Path, prev_status: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if prev_status != "PASS":
        return {
            "step": "8",
            "name": "Stage2 mild robustness",
            "status": "BLOCKED",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "confirm PASS + full PASS",
            "notes": "Step 7 gate 통과 이후에만 실행",
        }
    confirm_path = audit_root / "stage2_confirm_5seed" / "summary" / "stage2_confirm_5seed_score_summary.csv"
    full_path = audit_root / "stage2_full" / "summary" / "stage2_full_score_summary.csv"
    if not confirm_path.exists() or not full_path.exists():
        return {
            "step": "8",
            "name": "Stage2 mild robustness",
            "status": "INCOMPLETE",
            "evidence": [],
            "metrics": "",
            "pass_criteria": "confirm PASS + full PASS",
            "notes": "confirm/full summaries missing",
        }
    winner = manifest.get("step6_winner_name")
    confirm_row = _row_for_candidate(confirm_path, winner)
    full_row = _row_for_candidate(full_path, winner)
    confirm_status = str(confirm_row.get("pass_fail", "INCOMPLETE"))
    full_status = str(full_row.get("pass_fail", "INCOMPLETE"))
    status = "PASS" if confirm_status == "PASS" and full_status == "PASS" else "FAIL"
    return {
        "step": "8",
        "name": "Stage2 mild robustness",
        "status": status,
        "evidence": [str(confirm_path), str(full_path)],
        "metrics": (
            f"confirm={confirm_row.get('candidate_name', 'n/a')}:{confirm_status} "
            f"full={full_row.get('candidate_name', 'n/a')}:{full_status}"
        ),
        "pass_criteria": "confirm PASS + full PASS",
        "notes": "Step 9는 confirm과 full이 모두 PASS일 때만 GO",
    }


def _blocked_step(step: str, name: str, reason: str) -> dict[str, Any]:
    return {
        "step": step,
        "name": name,
        "status": "BLOCKED",
        "evidence": [],
        "metrics": "",
        "pass_criteria": "상위 단계 PASS 이후 실행",
        "notes": reason,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["step", "name", "status", "evidence", "metrics", "pass_criteria", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "evidence": " | ".join(row["evidence"])})


def _compute_verdict(rows: list[dict[str, Any]], locked_baseline: dict[str, Any] | None) -> str:
    statuses = {row["step"]: row["status"] for row in rows}
    if any(status == "FAIL" for status in statuses.values()):
        return "NO-GO"
    all_steps_pass = all(statuses.get(str(step)) == "PASS" for step in range(1, 9))
    if all_steps_pass and locked_baseline and locked_baseline.get("status") == "LOCKED":
        return "GO"
    return "INCOMPLETE"


def _write_md(
    path: Path,
    rows: list[dict[str, Any]],
    preflight: dict[str, Any],
    verdict: str,
    audit_root: Path,
    locked_baseline: dict[str, Any] | None,
) -> None:
    lock_status = locked_baseline.get("status", "missing") if locked_baseline else "missing"
    lines = [
        "# Pre-V9 Step 1-8 Audit Scorecard",
        "",
        f"- Audit root: `{audit_root}`",
        f"- Preflight: **{preflight['status']}** ({preflight['result_line']})",
        f"- Final decision: **{verdict}**",
        f"- Step8 locked baseline: **{lock_status}**",
        "",
        "| Step | Name | Status | Metrics | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = row["metrics"].replace("|", "\\|") if row["metrics"] else "-"
        notes = row["notes"].replace("|", "\\|") if row["notes"] else "-"
        lines.append(f"| {row['step']} | {row['name']} | {row['status']} | {metrics} | {notes} |")
    lines.extend(["", "## Evidence"])
    for row in rows:
        evidence = ", ".join(f"`{item}`" for item in row["evidence"]) if row["evidence"] else "none"
        lines.append(f"- Step {row['step']}: {evidence}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_handoff(
    path: Path,
    rows: list[dict[str, Any]],
    preflight: dict[str, Any],
    audit_root: Path,
    verdict: str,
    manifest: dict[str, Any],
    locked_baseline: dict[str, Any] | None,
) -> None:
    passed_steps = [row["step"] for row in rows if row["status"] == "PASS"]
    locked_status = locked_baseline.get("status", "missing") if locked_baseline else "missing"
    lines = [
        "# Pre-V9 Audit Handoff",
        "",
        "## Decision",
        f"- Verdict: **{verdict}**",
        f"- Preflight: `{preflight['result_line']}`",
        f"- Passed steps: {', '.join(passed_steps) if passed_steps else 'none'}",
        f"- Locked baseline: `{locked_status}`",
        "",
    ]
    if verdict == "GO":
        lines.extend(
            [
                "## Outcome",
                "- Step 9 design can proceed on top of the current audited Stage2 chain.",
                "- Step9 must boot from locked baseline only.",
                f"- Winner: `{manifest.get('step6_winner_name', 'unknown')}`",
                f"- Locked baseline root: `{locked_baseline.get('locked_baseline_root', 'unknown')}`",
                f"- Step9 base checkpoint: `{locked_baseline.get('step8_primary_checkpoint_path', 'unknown')}`",
            ]
        )
    elif verdict == "NO-GO":
        failed = [row for row in rows if row["status"] == "FAIL"]
        lines.extend(
            [
                "## Blocking Finding",
                f"- Blocking steps: {', '.join(row['step'] for row in failed) if failed else 'unknown'}",
                "- Fix the failed gate before treating this audit chain as a basis for Step 9.",
            ]
        )
    else:
        incomplete = [row for row in rows if row["status"] in {"INCOMPLETE", "BLOCKED"}]
        lines.extend(
            [
                "## Remaining Work",
                f"- Incomplete steps: {', '.join(row['step'] for row in incomplete) if incomplete else 'none'}",
                "- This audit chain is not failed, but it is not complete enough to justify a Step 9 redesign handoff yet.",
            ]
        )
        if all(row["status"] == "PASS" for row in rows) and locked_status != "LOCKED":
            lines.append("- Step8 passed, but locked baseline is missing or failed, so Step9 must not start yet.")
    lines.extend(
        [
            "",
            "## Key Paths",
            f"- Audit root: `{audit_root}`",
            f"- Dependency manifest: `{audit_root / 'summary' / 'dependency_manifest.json'}`",
            f"- Scorecard CSV: `{audit_root / 'summary' / 'step_1_to_8_scorecard.csv'}`",
            f"- Scorecard MD: `{audit_root / 'summary' / 'step_1_to_8_scorecard.md'}`",
            f"- Locked baseline summary: `{audit_root / 'summary' / 'step8_locked_baseline.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    args = parser.parse_args()

    audit_root = args.audit_root
    summary_dir = audit_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(summary_dir)
    locked_baseline = _locked_baseline(summary_dir)
    preflight = _parse_preflight(summary_dir)

    rows = [_step1(summary_dir)]
    step2, step3 = _step2_and_3(audit_root)
    rows.extend([step2, step3])
    step4 = _step4(audit_root)
    rows.append(step4)
    step5 = _step5(audit_root, step4["status"])
    rows.append(step5)
    step6 = _step6(audit_root, step5["status"], manifest)
    rows.append(step6)
    step7 = _step7(audit_root, step6["status"])
    rows.append(step7)
    step8 = _step8(audit_root, step7["status"], manifest)
    rows.append(step8)

    verdict = _compute_verdict(rows, locked_baseline)
    _write_csv(summary_dir / "step_1_to_8_scorecard.csv", rows)
    _write_md(summary_dir / "step_1_to_8_scorecard.md", rows, preflight, verdict, audit_root, locked_baseline)
    _write_handoff(summary_dir / "pre_v9_handoff.md", rows, preflight, audit_root, verdict, manifest, locked_baseline)


if __name__ == "__main__":
    main()
