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
    )


def test_corrective_delta_dataset_audit_passes_clean_balanced_targets() -> None:
    summary = _audit(_base_arrays())

    assert summary["gate_status"] == "PASS"
    assert summary["eligible_severe_count"] == 260
    assert summary["left_count"] == 130
    assert summary["right_count"] == 130
    assert summary["aligned_rate_in_eligible"] == 1.0
    assert summary["left_sign_rate_vs_neg_lane"] == 1.0
    assert summary["right_sign_rate_vs_neg_lane"] == 1.0
    assert np.isclose(float(summary["target_delta_abs_mean"]), 0.05)
    assert len(summary["route_bins"]) == 3


def test_corrective_delta_dataset_audit_blocks_wrong_side_targets() -> None:
    summary = _audit(_base_arrays(wrong_right_sign=True))

    assert summary["gate_status"] == "BLOCK"
    assert "sign_rate" in summary["gate_failures"]
    assert "right_sign_rate" in summary["gate_failures"]
    assert summary["right_sign_rate_vs_neg_lane"] == 0.0
