from __future__ import annotations

import numpy as np

from rcs_ac_first.scripts.audit_corrective_delta_dataset import audit_arrays


def _base_arrays(*, wrong_right_sign: bool = False) -> dict[str, np.ndarray]:
    n = 260
    route_progress = np.linspace(0.03, 0.095, n, dtype=np.float32)
    lane_offset = np.concatenate(
        [
            np.full(n // 2, -0.8, dtype=np.float32),
            np.full(n - n // 2, 0.8, dtype=np.float32),
        ]
    )
    heading_error = np.concatenate(
        [
            np.full(n // 2, 4.0, dtype=np.float32),
            np.full(n - n // 2, -4.0, dtype=np.float32),
        ]
    )
    target_delta = np.where(lane_offset < 0.0, 0.05, -0.05).astype(np.float32)
    if wrong_right_sign:
        target_delta[lane_offset > 0.0] = 0.05
    policy_steer = np.zeros(n, dtype=np.float32)
    return {
        "obs_flat": np.zeros((n, 47311), dtype=np.float32),
        "auxiliary": np.zeros((n, 30), dtype=np.float32),
        "expert_action": np.stack([target_delta, np.zeros(n, dtype=np.float32)], axis=1),
        "expert_control": np.zeros((n, 3), dtype=np.float32),
        "reward": np.zeros(n, dtype=np.float32),
        "done": np.zeros(n, dtype=np.bool_),
        "route_progress": route_progress,
        "episode_id": np.zeros(n, dtype=np.int32),
        "step_in_episode": np.arange(n, dtype=np.int32),
        "bad_transition_mask": np.zeros(n, dtype=np.bool_),
        "lane_offset": lane_offset,
        "heading_error": heading_error,
        "policy_action_steer": policy_steer,
        "target_corrected_steer": policy_steer + target_delta,
        "target_corrective_delta": target_delta,
        "target_corrective_delta_available": np.ones(n, dtype=np.float32),
    }


def _audit(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return audit_arrays(
        arrays,
        route_min=0.025,
        route_max=0.10,
        abs_lane_offset_min=0.55,
        abs_heading_error_min=3.0,
        min_severe_count=200,
        min_side_count=40,
        min_sign_rate=0.98,
        min_side_sign_rate=0.98,
        min_abs_delta_mean=0.035,
        min_abs_delta_p10=0.010,
        max_target_steer_saturation_rate=0.05,
        max_target_steer_consistency_mae=1.0e-5,
        max_wrong_sign_rows=500,
    )


def _with_source_metadata(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = int(arrays["obs_flat"].shape[0])
    arrays["source_label"] = np.full((n,), "synthetic")
    arrays["source_dataset"] = np.full((n,), "synthetic.npz")
    arrays["source_dataset_index"] = np.zeros((n,), dtype=np.int32)
    arrays["source_row_index"] = np.arange(n, dtype=np.int32)
    arrays["source_episode_id"] = arrays["episode_id"].astype(np.int32, copy=True)
    arrays["source_step_in_episode"] = arrays["step_in_episode"].astype(np.int32, copy=True)
    return arrays


def test_corrective_delta_dataset_audit_passes_clean_balanced_targets() -> None:
    summary = _audit(_with_source_metadata(_base_arrays()))

    assert summary["gate_status"] == "PASS"
    assert summary["semantic_gate_status"] == "PASS"
    assert summary["strict_lane_gate_status"] == "PASS"
    assert summary["eligible_severe_count"] == 260
    assert summary["left_count"] == 130
    assert summary["right_count"] == 130
    assert summary["aligned_rate_in_eligible"] == 1.0
    assert summary["left_sign_rate_vs_neg_lane"] == 1.0
    assert summary["right_sign_rate_vs_neg_lane"] == 1.0
    assert np.isclose(float(summary["target_delta_abs_mean"]), 0.05)
    assert len(summary["route_bins"]) == 3
    assert summary["wrong_sign_left_count"] == 0
    assert summary["wrong_sign_left_rows"] == []
    assert summary["missing_source_metadata"] is False


def test_corrective_delta_dataset_audit_blocks_wrong_side_targets() -> None:
    summary = _audit(_with_source_metadata(_base_arrays(wrong_right_sign=True)))

    assert summary["gate_status"] == "BLOCK"
    assert summary["semantic_gate_status"] == "BLOCK"
    assert summary["strict_lane_gate_status"] == "BLOCK"
    assert "semantic_lane_dominant_alignment_rate" in summary["gate_failures"]
    assert "strict_lane_sign_rate" in summary["strict_lane_gate_failures"]
    assert "strict_lane_right_sign_rate" in summary["strict_lane_gate_failures"]
    assert summary["right_sign_rate_vs_neg_lane"] == 0.0


def test_corrective_delta_dataset_semantic_gate_requires_source_metadata() -> None:
    summary = _audit(_base_arrays())

    assert summary["strict_lane_gate_status"] == "PASS"
    assert summary["heading_aware_gate_status"] == "PASS"
    assert summary["semantic_gate_status"] == "BLOCK"
    assert summary["gate_status"] == "BLOCK"
    assert summary["gate_failures"] == ["source_metadata"]
    assert summary["missing_source_metadata"] is True


def test_corrective_delta_dataset_audit_exports_wrong_left_rows() -> None:
    arrays = _base_arrays()
    left_indices = np.flatnonzero(arrays["lane_offset"] < 0.0)
    arrays["target_corrective_delta"][left_indices[:3]] = -0.05
    arrays["target_corrected_steer"] = arrays["policy_action_steer"] + arrays["target_corrective_delta"]
    arrays["source_label"] = np.full((arrays["obs_flat"].shape[0],), "round29_failure")
    arrays["source_dataset"] = np.full((arrays["obs_flat"].shape[0],), "failure.npz")
    arrays["source_dataset_index"] = np.ones((arrays["obs_flat"].shape[0],), dtype=np.int32)
    arrays["source_row_index"] = np.arange(arrays["obs_flat"].shape[0], dtype=np.int32) + 100
    arrays["source_episode_id"] = np.full((arrays["obs_flat"].shape[0],), 12, dtype=np.int32)
    arrays["source_step_in_episode"] = np.arange(arrays["obs_flat"].shape[0], dtype=np.int32) + 50

    summary = _audit(arrays)

    assert summary["strict_lane_gate_status"] == "BLOCK"
    assert summary["wrong_sign_left_count"] == 3
    assert np.isclose(float(summary["wrong_sign_left_rate_in_left"]), 3 / 130)
    rows = summary["wrong_sign_left_rows"]
    assert len(rows) == 3
    assert rows[0]["classification"] == "unresolved"
    assert rows[0]["episode_id"] == 0
    assert rows[0]["source_label"] == "round29_failure"
    assert rows[0]["source_dataset"] == "failure.npz"
    assert rows[0]["source_row_index"] == 100
    assert rows[0]["source_episode_id"] == 12
    assert rows[0]["source_step_in_episode"] == 50
    assert rows[0]["route_bin"] == "0.025-0.050"
    assert summary["missing_source_metadata"] is False
    assert summary["wrong_sign_left_by_source_label_top"] == [{"value": "round29_failure", "count": 3}]
    assert summary["wrong_sign_left_by_source_dataset_top"] == [{"value": "failure.npz", "count": 3}]
    assert summary["wrong_sign_left_by_source_episode_top"] == [{"value": "12", "count": 3}]


def test_corrective_delta_dataset_audit_reports_heading_dominant_alignment() -> None:
    arrays = _with_source_metadata(_base_arrays())
    lane_offset = np.concatenate(
        [
            np.full(130, -0.2, dtype=np.float32),
            np.full(130, 0.2, dtype=np.float32),
        ]
    )
    heading_error = np.concatenate(
        [
            np.full(130, -8.0, dtype=np.float32),
            np.full(130, 8.0, dtype=np.float32),
        ]
    )
    target_delta = np.where(heading_error < 0.0, -0.05, 0.05).astype(np.float32)
    arrays["lane_offset"] = lane_offset
    arrays["heading_error"] = heading_error
    arrays["target_corrective_delta"] = target_delta
    arrays["target_corrected_steer"] = arrays["policy_action_steer"] + target_delta
    arrays["expert_action"][:, 0] = target_delta

    summary = _audit(arrays)

    assert summary["gate_status"] == "PASS"
    assert summary["semantic_gate_status"] == "PASS"
    assert summary["strict_lane_gate_status"] == "BLOCK"
    assert "strict_lane_sign_rate" in summary["strict_lane_gate_failures"]
    assert summary["heading_aware_gate_status"] == "PASS"
    assert summary["heading_aware_lane_dominant_count"] == 0
    assert summary["heading_aware_heading_dominant_count"] == 260
    assert summary["heading_aware_heading_dominant_alignment_rate"] == 1.0
    assert summary["heading_aware_lane_heading_conflict_count"] == 260
    assert summary["heading_aware_conflict_alignment_vs_heading_rate"] == 1.0
    assert summary["heading_aware_conflict_alignment_vs_lane_rate"] == 0.0


def test_corrective_delta_dataset_audit_reports_ambiguous_heading_rows() -> None:
    arrays = _with_source_metadata(_base_arrays())
    arrays["lane_offset"][:] = 0.2
    arrays["heading_error"][:] = 1.0

    summary = _audit(arrays)

    assert summary["gate_status"] == "BLOCK"
    assert summary["semantic_gate_status"] == "BLOCK"
    assert "semantic_ambiguous_rate" in summary["semantic_gate_failures"]
    assert summary["heading_aware_gate_status"] == "BLOCK"
    assert summary["heading_aware_ambiguous_count"] == 260
    assert summary["heading_aware_ambiguous_rate"] == 1.0
