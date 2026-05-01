from pathlib import Path

import numpy as np

from rcs_ac_first.scripts.evaluate_bc import _build_log_row, _build_summary


def test_log_row_splits_route_offset_and_visual_center_signs() -> None:
    row = _build_log_row(
        step=0,
        episode=0,
        action=np.array([0.1, 0.0], dtype=np.float32),
        previous_action=None,
        reward=0.0,
        done=False,
        info={"route_progress": 0.24, "lane_offset": 2.0, "heading_error": -12.0},
        diagnostics={"visual_lane_center_error_mean": -1.5},
        visual_zero_deltas={"action_l1": 0.02, "steer": 0.006, "signed_steer": 0.006, "longitudinal": 0.0},
        corrective_route_min=0.20,
        corrective_route_max=0.32,
        corrective_abs_lane_offset_min=1.8,
        corrective_abs_heading_error_min=6.0,
        visual_center_error_min=0.05,
    )

    assert row["corrective_need_active"] is True
    assert row["visual_center_corrective_sign"] is True
    assert row["route_offset_corrective_sign"] is False
    assert row["route_visual_center_sign_agreement"] is False
    assert row["visual_zero_corrective_sign"] is False


def test_summary_can_gate_on_visual_center_without_route_offset_sign() -> None:
    rows = [
        _build_log_row(
            step=step,
            episode=0,
            action=np.array([0.1, 0.0], dtype=np.float32),
            previous_action=np.array([0.09, 0.0], dtype=np.float32),
            reward=0.0,
            done=False,
            info={"route_progress": 0.24, "lane_offset": 2.0, "heading_error": -12.0},
            diagnostics={"visual_lane_center_error_mean": -1.5},
            visual_zero_deltas={"action_l1": 0.02, "steer": 0.006, "signed_steer": 0.006, "longitudinal": 0.0},
            corrective_route_min=0.20,
            corrective_route_max=0.32,
            corrective_abs_lane_offset_min=1.8,
            corrective_abs_heading_error_min=6.0,
            visual_center_error_min=0.05,
        )
        for step in range(4)
    ]
    ppo_gate = {
        "max_offroad": 2,
        "min_max_route": 0.20,
        "min_mean_route": 0.0,
        "max_action_jitter": 0.07,
        "min_visual_zero_action_l1": 0.012,
        "min_visual_zero_steer_delta": 0.003,
        "min_corrective_count": 1,
        "min_corrective_visual_zero_steer_delta": 0.004,
        "min_corrective_sign_rate": 0.70,
        "min_visual_center_correction_rate": 0.70,
        "corrective_sign_source": "visual_center",
    }

    summary = _build_summary(
        run_name="unit",
        checkpoint_path=Path("checkpoint.pt"),
        num_steps=4,
        rows=rows,
        output_csv=Path("eval.csv"),
        ppo_gate=ppo_gate,
    )

    assert summary["route_offset_corrective_sign_rate"] == 0.0
    assert summary["visual_center_correction_rate"] == 1.0
    assert summary["corrective_sign_source"] == "visual_center"
    assert summary["ppo_gate_status"] == "PASS"
