from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcs_toy_nav_env import RCSToyNavEnv
from run_pre_v9_stage_audit import build_generated_configs, lock_step8_baseline
from stage2_transition_helpers import build_run_cache_metadata


def _write_stage0_logs(audit_root: Path) -> None:
    log_dir = audit_root / "stage0_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    episode_id = 0
    for policy_type in ("random", "heuristic"):
        for corruption_mode in ("clean", "corrupted"):
            for _ in range(20):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "policy_type": policy_type,
                        "corruption_mode": corruption_mode,
                        "total_reward": 1.0 if policy_type == "heuristic" else -1.0,
                        "episode_length": 10,
                        "success": 1 if policy_type == "heuristic" and corruption_mode == "clean" else 0,
                        "collision": 0,
                        "out_of_bounds": 0,
                        "timeout": 0,
                        "min_goal_distance": 0.1,
                        "final_goal_distance": 0.2,
                        "vision_corrupted": 1 if corruption_mode == "corrupted" else 0,
                        "distance_corrupted": 1 if corruption_mode == "corrupted" else 0,
                        "ego_corrupted": 1 if corruption_mode == "corrupted" else 0,
                        "route_corrupted": 1 if corruption_mode == "corrupted" else 0,
                        "avg_vision_quality": 0.8,
                        "avg_distance_quality": 0.8,
                        "avg_risk": 0.2,
                        "terminated_reason": "goal" if policy_type == "heuristic" and corruption_mode == "clean" else "timeout",
                    }
                )
                episode_id += 1
    fieldnames = list(rows[0].keys())
    with (log_dir / "stage0_episode_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (log_dir / "stage0_episode_summaries.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_step1_schema(summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "observation_keys": ["vision", "distance", "ego", "route", "aux"],
        "observation_keys_sorted": ["aux", "distance", "ego", "route", "vision"],
        "observation_shapes": {
            "vision": [8],
            "distance": [9],
            "ego": [5],
            "route": [5],
            "aux": [8],
        },
        "stage1_expected_modality_keys": ["vision", "distance", "ego", "route", "aux"],
        "action_shape": [2],
        "aux_dim": 8,
        "has_risk_stats_in_info": True,
        "has_quality_stats_in_info": True,
    }
    (summary_dir / "step1_env_schema.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_step4_pass(audit_root: Path) -> None:
    summary_dir = audit_root / "baseline_comparison" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = [
        {
            "model_name": "a_ego_route",
            "final_eval_success_rate_mean": "0.85",
            "final_eval_success_rate_std": "0.04",
            "final_eval_return_mean": "4.65",
            "final_eval_return_std": "0.38",
            "final_eval_avg_progress_mean": "0.086",
        },
        {
            "model_name": "b_early_fusion",
            "final_eval_success_rate_mean": "0.95",
            "final_eval_success_rate_std": "0.04",
            "final_eval_return_mean": "5.34",
            "final_eval_return_std": "0.33",
            "final_eval_avg_progress_mean": "0.091",
        },
        {
            "model_name": "c_attention_fusion",
            "final_eval_success_rate_mean": "0.90",
            "final_eval_success_rate_std": "0.04",
            "final_eval_return_mean": "4.87",
            "final_eval_return_std": "0.22",
            "final_eval_avg_progress_mean": "0.088",
        },
    ]
    with (summary_dir / "comparison_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    (summary_dir / "pass_fail_report.json").write_text(
        json.dumps({"overall_grade": "PASS"}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (summary_dir / "pass_fail_report.md").write_text("# PASS\n", encoding="utf-8")
    (summary_dir / "attention_summary.csv").write_text("candidate,attention\n", encoding="utf-8")


def _write_preflight(summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "preflight_pytest.txt").write_text("122 passed, 69 warnings in 2.50s\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_step5_pass(audit_root: Path) -> None:
    reval_dir = audit_root / "transition_to_stage1" / "revalidation" / "summary"
    reval_dir.mkdir(parents=True, exist_ok=True)
    (reval_dir / "revalidation_pass_fail_report.md").write_text(
        "# Revalidation Pass/Fail Report\n\n- Grade: **PASS**\n",
        encoding="utf-8",
    )
    _write_csv(
        reval_dir / "comparison_against_baseline.csv",
        [
            {
                "success_diff_vs_baseline": "0.05",
                "return_diff_vs_baseline": "0.10",
                "progress_diff_vs_baseline": "0.01",
            }
        ],
    )


def _write_step6_pass(audit_root: Path, candidate_name: str = "winner") -> None:
    summary_dir = audit_root / "stage1_vision_ego_logit_triplet_calibration" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        summary_dir / "vision_ego_logit_triplet_score_summary.csv",
        [
            {
                "candidate_name": candidate_name,
                "pass_fail": "PASS",
                "calibration_score": "0.91",
                "clean_success_rate": "0.88",
                "target_positive_ratio": "0.42",
            }
        ],
    )
    (summary_dir / "vision_ego_logit_triplet_final_recommendation.md").write_text("# winner\n", encoding="utf-8")


def _write_step7_pass(audit_root: Path, candidate_name: str = "winner") -> None:
    _write_csv(
        audit_root / "stage2_mild_pilot" / "summary" / "stage2_mild_pilot_score_summary.csv",
        [{"candidate_name": candidate_name, "pass_fail": "PASS"}],
    )
    _write_csv(
        audit_root / "stage2_reliability_proof" / "summary" / "stage2_reliability_proof_score_summary.csv",
        [{"candidate_name": candidate_name, "pass_fail": "PASS"}],
    )
    _write_csv(
        audit_root / "stage2_fusion_shift_proof_pilot" / "summary" / "stage2_fusion_shift_proof_pilot_score_summary.csv",
        [{"candidate_name": candidate_name, "pass_fail": "WARN"}],
    )


def _write_step8_pass(audit_root: Path, candidate_name: str = "winner") -> None:
    _write_csv(
        audit_root / "stage2_confirm_5seed" / "summary" / "stage2_confirm_5seed_score_summary.csv",
        [{"candidate_name": candidate_name, "pass_fail": "PASS"}],
    )
    _write_csv(
        audit_root / "stage2_full" / "summary" / "stage2_full_score_summary.csv",
        [{"candidate_name": candidate_name, "pass_fail": "PASS"}],
    )


def _write_step8_candidate_dirs(audit_root: Path, candidate_name: str = "winner") -> None:
    confirm_dir = audit_root / "stage2_confirm_5seed" / candidate_name
    full_dir = audit_root / "stage2_full" / candidate_name
    confirm_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    (confirm_dir / "checkpoint_final.pt").write_text("confirm", encoding="utf-8")
    (full_dir / "checkpoint_final.pt").write_text("full", encoding="utf-8")


def test_build_generated_configs_uses_audit_local_paths(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    temp_root = tmp_path / "tmp_cfg"
    paths = build_generated_configs(audit_root, temp_root, env_overrides={})

    transition_cfg = yaml_safe_load(Path(paths["transition_to_stage1_full"]))
    standard_v1_cfg_path = Path(paths["standard_v1_full"])
    assert standard_v1_cfg_path.exists()
    assert transition_cfg["standard_v1"]["registered_config_path"].startswith(str(temp_root))
    assert transition_cfg["standard_v1"]["registered_config_path"] == str(standard_v1_cfg_path)
    assert transition_cfg["stage1_warmup"]["init_checkpoint_root"] == str(
        audit_root / "transition_to_stage1" / "revalidation" / "standard_v1"
    )

    triplet_cfg = yaml_safe_load(Path(paths["stage1_triplet_full"]))
    assert triplet_cfg["base_checkpoint_root"] == str(
        audit_root / "transition_to_stage1" / "stage1_warmup" / "standard_v1"
    )

    manifest = json.loads((audit_root / "summary" / "dependency_manifest.json").read_text(encoding="utf-8"))
    assert manifest["step4_standard_v1_config"].startswith(str(temp_root))


def test_build_generated_configs_rewires_stage2_from_step6_winner(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    temp_root = tmp_path / "tmp_cfg"
    paths = build_generated_configs(
        audit_root,
        temp_root,
        env_overrides={},
        smoke_stage6_winner="smoke_best",
        smoke_stage6_secondary="smoke_backup",
        full_stage6_winner="best",
        full_stage6_secondary="backup",
    )

    mild_cfg = yaml_safe_load(Path(paths["stage2_mild_pilot_full"]))
    assert mild_cfg["standard_stage2_candidate"] == "best"
    assert mild_cfg["backup_candidate"] == "backup"
    assert mild_cfg["candidates"]["best"]["checkpoint_root"] == str(
        audit_root / "stage1_vision_ego_logit_triplet_calibration" / "best"
    )

    proof_cfg = yaml_safe_load(Path(paths["stage2_reliability_proof_full"]))
    assert proof_cfg["candidate_order"] == ["best"]
    assert proof_cfg["candidates"]["best"]["checkpoint_root"] == str(
        audit_root / "stage2_mild_pilot" / "best"
    )

    fusion_cfg = yaml_safe_load(Path(paths["stage2_fusion_shift_proof_pilot_full"]))
    assert fusion_cfg["baseline_eval_rows_path"].endswith("/stage2_reliability_proof/best/eval_rows_full.json")
    assert fusion_cfg["baseline_eval_rows_path_smoke"].endswith("/stage2_reliability_proof/best/eval_rows_smoke.json")

    full_cfg = yaml_safe_load(Path(paths["stage2_full_smoke"]))
    assert full_cfg["confirm_gate"]["score_summary_path"].endswith(
        "/stage2_confirm_5seed_smoke/summary/stage2_confirm_5seed_score_summary.csv"
    )


def test_build_run_cache_metadata_includes_step_and_upstream_root() -> None:
    metadata = build_run_cache_metadata(
        config={"experiment_name": "stage2_reliability_proof"},
        candidate_name="cand",
        metrics_filename="eval_rows.json",
        mode="smoke",
        num_eval_episodes=6,
        step_name="stage2_reliability_proof",
        upstream_checkpoint_root="/tmp/checkpoints/cand",
    )
    assert metadata["step_name"] == "stage2_reliability_proof"
    assert metadata["upstream_checkpoint_root"] == "/tmp/checkpoints/cand"
    assert metadata["mode"] == "smoke"
    assert "config_hash" in metadata


def test_route_waypoint_env_override_changes_route_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RCS_TOY_ROUTE_INCLUDE_WAYPOINT", "0")
    env = RCSToyNavEnv()
    assert env.observation_space["route"].shape == (3,)


def test_summary_generator_marks_incomplete_after_step4_pass(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_step1_schema(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    (summary_dir / "dependency_manifest.json").write_text(json.dumps({}, ensure_ascii=True, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )

    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "Final decision: **INCOMPLETE**" in scorecard
    assert "| 1 | 환경/상태 정의 | PASS |" in scorecard


def test_summary_generator_marks_no_go_on_step5_fail(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_step1_schema(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    reval_dir = audit_root / "transition_to_stage1" / "revalidation" / "summary"
    reval_dir.mkdir(parents=True, exist_ok=True)
    (reval_dir / "revalidation_pass_fail_report.md").write_text(
        "# Revalidation Pass/Fail Report\n\n- Grade: **FAIL**\n",
        encoding="utf-8",
    )
    with (reval_dir / "comparison_against_baseline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["success_diff_vs_baseline", "return_diff_vs_baseline", "progress_diff_vs_baseline"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "success_diff_vs_baseline": "-0.1",
                "return_diff_vs_baseline": "-1.0",
                "progress_diff_vs_baseline": "-0.02",
            }
        )
    (summary_dir / "dependency_manifest.json").write_text(json.dumps({}, ensure_ascii=True, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )

    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "Final decision: **NO-GO**" in scorecard


def test_summary_generator_accepts_non_canonical_observation_key_order(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    payload = {
        "observation_keys": ["aux", "distance", "ego", "route", "vision"],
        "observation_keys_sorted": ["aux", "distance", "ego", "route", "vision"],
        "observation_shapes": {
            "aux": [8],
            "distance": [9],
            "ego": [5],
            "route": [5],
            "vision": [8],
        },
        "stage1_expected_modality_keys": ["vision", "distance", "ego", "route", "aux"],
        "action_shape": [2],
        "aux_dim": 8,
        "has_risk_stats_in_info": True,
        "has_quality_stats_in_info": True,
    }
    (summary_dir / "step1_env_schema.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (summary_dir / "dependency_manifest.json").write_text(json.dumps({}, ensure_ascii=True, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )

    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "| 1 | 환경/상태 정의 | PASS |" in scorecard


def test_summary_generator_fails_when_step1_key_missing(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    payload = {
        "observation_keys": ["aux", "distance", "ego", "route"],
        "observation_keys_sorted": ["aux", "distance", "ego", "route"],
        "observation_shapes": {
            "aux": [8],
            "distance": [9],
            "ego": [5],
            "route": [5],
        },
        "stage1_expected_modality_keys": ["vision", "distance", "ego", "route", "aux"],
        "action_shape": [2],
        "aux_dim": 8,
        "has_risk_stats_in_info": True,
        "has_quality_stats_in_info": True,
    }
    (summary_dir / "step1_env_schema.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (summary_dir / "dependency_manifest.json").write_text(json.dumps({}, ensure_ascii=True, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )

    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "| 1 | 환경/상태 정의 | FAIL |" in scorecard


def test_summary_generator_fails_when_step1_shape_missing(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    payload = {
        "observation_keys": ["aux", "distance", "ego", "route", "vision"],
        "observation_keys_sorted": ["aux", "distance", "ego", "route", "vision"],
        "observation_shapes": {
            "aux": [8],
            "distance": [9],
            "ego": [5],
            "route": [5],
        },
        "stage1_expected_modality_keys": ["vision", "distance", "ego", "route", "aux"],
        "action_shape": [2],
        "aux_dim": 8,
        "has_risk_stats_in_info": True,
        "has_quality_stats_in_info": True,
    }
    (summary_dir / "step1_env_schema.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (summary_dir / "dependency_manifest.json").write_text(json.dumps({}, ensure_ascii=True, indent=2), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )

    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "| 1 | 환경/상태 정의 | FAIL |" in scorecard


def test_lock_step8_baseline_creates_frozen_snapshot(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "step_1_to_8_scorecard.csv").write_text("step,name,status\n", encoding="utf-8")
    (summary_dir / "step_1_to_8_scorecard.md").write_text("# scorecard\n", encoding="utf-8")
    (summary_dir / "pre_v9_handoff.md").write_text("# handoff\n", encoding="utf-8")
    (summary_dir / "dependency_manifest.json").write_text(
        json.dumps(
            {
                "step6_winner_name": "winner",
                "config_hashes": {"stage2_full.yaml": "abc123"},
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_step8_pass(audit_root, candidate_name="winner")
    _write_step8_candidate_dirs(audit_root, candidate_name="winner")

    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    config_paths = {
        "stage1_warmup_runtime_full": str(config_root / "stage1_warmup_runtime.yaml"),
        "stage2_confirm_5seed_full": str(config_root / "stage2_confirm_5seed.yaml"),
        "stage2_full_full": str(config_root / "stage2_full.yaml"),
    }
    for path in config_paths.values():
        Path(path).write_text("dummy: true\n", encoding="utf-8")

    payload = lock_step8_baseline(
        audit_root,
        config_paths,
        env_overrides={"RCS_TOY_ROUTE_INCLUDE_WAYPOINT": "0"},
    )

    assert payload["status"] == "LOCKED"
    assert (audit_root / "step8_locked_baseline" / "baseline_manifest.json").exists()
    assert (audit_root / "step8_locked_baseline" / "stage2_full_winner" / "checkpoint_final.pt").exists()
    assert (audit_root / "step8_locked_baseline" / "stage2_confirm_winner" / "checkpoint_final.pt").exists()
    assert (audit_root / "step8_locked_baseline" / "configs" / "stage1_warmup_runtime.yaml").exists()
    assert (summary_dir / "step8_locked_baseline.json").exists()

    baseline_manifest = json.loads(
        (audit_root / "step8_locked_baseline" / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    dependency_manifest = json.loads((summary_dir / "dependency_manifest.json").read_text(encoding="utf-8"))
    assert baseline_manifest["step9_base_candidate_name"] == "winner"
    assert baseline_manifest["step9_base_stage1_config_path"].endswith("/step8_locked_baseline/configs/stage1_warmup_runtime.yaml")
    assert dependency_manifest["step8_primary_candidate_name"] == "winner"
    assert dependency_manifest["step9_base_candidate_name"] == "winner"


def test_summary_generator_requires_locked_baseline_for_go(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_step1_schema(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    _write_step5_pass(audit_root)
    _write_step6_pass(audit_root, candidate_name="winner")
    _write_step7_pass(audit_root, candidate_name="winner")
    _write_step8_pass(audit_root, candidate_name="winner")
    (summary_dir / "dependency_manifest.json").write_text(
        json.dumps({"step6_winner_name": "winner"}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )
    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "Final decision: **INCOMPLETE**" in scorecard

    locked_root = audit_root / "step8_locked_baseline"
    locked_root.mkdir(parents=True, exist_ok=True)
    baseline_manifest_path = locked_root / "baseline_manifest.json"
    baseline_manifest_path.write_text(json.dumps({"ok": True}, ensure_ascii=True, indent=2), encoding="utf-8")
    (summary_dir / "step8_locked_baseline.json").write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "locked_baseline_root": str(locked_root),
                "baseline_manifest_path": str(baseline_manifest_path),
                "step8_primary_checkpoint_path": str(locked_root / "stage2_full_winner" / "checkpoint_final.pt"),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )
    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    handoff = (summary_dir / "pre_v9_handoff.md").read_text(encoding="utf-8")
    assert "Final decision: **GO**" in scorecard
    assert "Step9 must boot from locked baseline only." in handoff


def test_summary_generator_stays_incomplete_when_lock_failed(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    summary_dir = audit_root / "summary"
    _write_preflight(summary_dir)
    _write_step1_schema(summary_dir)
    _write_stage0_logs(audit_root)
    _write_step4_pass(audit_root)
    _write_step5_pass(audit_root)
    _write_step6_pass(audit_root, candidate_name="winner")
    _write_step7_pass(audit_root, candidate_name="winner")
    _write_step8_pass(audit_root, candidate_name="winner")
    (summary_dir / "dependency_manifest.json").write_text(
        json.dumps({"step6_winner_name": "winner"}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (summary_dir / "step8_locked_baseline.json").write_text(
        json.dumps({"status": "FAILED", "error": "copy failed"}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        check=True,
        cwd=str(ROOT),
    )

    scorecard = (summary_dir / "step_1_to_8_scorecard.md").read_text(encoding="utf-8")
    assert "Final decision: **INCOMPLETE**" in scorecard


def yaml_safe_load(path: Path) -> dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
