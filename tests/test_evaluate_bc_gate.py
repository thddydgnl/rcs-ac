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


def test_summary_records_collision_context_window() -> None:
    rows = []
    previous_action = None
    for step in range(40):
        action = np.array([0.05 + step * 0.001, 0.1], dtype=np.float32)
        rows.append(
            _build_log_row(
                step=step,
                episode=0,
                action=action,
                previous_action=previous_action,
                reward=0.0,
                done=step == 39,
                info={
                    "route_progress": 0.1794871794871795,
                    "lane_offset": 1.7,
                    "heading_error": -16.0,
                    "speed_kmh": 5.0,
                    "collision": step == 39,
                    "done_reason": "collision" if step == 39 else "running",
                },
                diagnostics={"visual_lane_center_error_mean": -1.5, "risk_scalar": 0.2},
                visual_zero_deltas={
                    "action_l1": 0.03,
                    "steer": 0.02,
                    "signed_steer": 0.02,
                    "longitudinal": 0.01,
                },
                corrective_route_min=0.10,
                corrective_route_max=0.60,
                corrective_abs_lane_offset_min=1.5,
                corrective_abs_heading_error_min=6.0,
                visual_center_error_min=0.05,
            )
        )
        previous_action = action

    ppo_gate = {
        "max_offroad": 2,
        "min_max_route": 0.10,
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
        num_steps=40,
        rows=rows,
        output_csv=Path("eval.csv"),
        ppo_gate=ppo_gate,
    )

    assert summary["collision_count"] == 1
    assert summary["collision_cluster_counts"] == {"0179487": 1}
    assert summary["near_collision_cluster_row_count"] == 40
    assert len(summary["collision_events"]) == 1
    event = summary["collision_events"][0]
    assert event["cluster"] == "0179487"
    assert event["window_step_count"] == 32
    assert event["window_visual_center_correction_rate"] == 1.0


def test_summary_blocks_severe_early_offroad_route_cluster() -> None:
    rows = [
        _build_log_row(
            step=0,
            episode=0,
            action=np.array([0.1, 0.0], dtype=np.float32),
            previous_action=None,
            reward=0.0,
            done=True,
            info={
                "route_progress": 0.076923,
                "lane_offset": -3.0,
                "heading_error": 24.0,
                "offroad": True,
                "done_reason": "offroad",
            },
            diagnostics={"visual_lane_center_error_mean": 1.5},
            visual_zero_deltas={"action_l1": 0.02, "steer": 0.006, "signed_steer": -0.006, "longitudinal": 0.0},
            corrective_route_min=0.20,
            corrective_route_max=0.32,
            corrective_abs_lane_offset_min=1.8,
            corrective_abs_heading_error_min=6.0,
            visual_center_error_min=0.05,
        )
    ]
    ppo_gate = {
        "max_offroad": 2,
        "min_max_route": 0.0,
        "min_mean_route": 0.0,
        "max_action_jitter": 0.07,
        "min_visual_zero_action_l1": 0.0,
        "min_visual_zero_steer_delta": 0.0,
        "min_corrective_count": 0,
        "min_corrective_visual_zero_steer_delta": 0.0,
        "min_corrective_sign_rate": 0.0,
        "min_visual_center_correction_rate": 0.0,
        "corrective_sign_source": "visual_center",
        "early_offroad_route_max": 0.08,
    }

    summary = _build_summary(
        run_name="unit",
        checkpoint_path=Path("checkpoint.pt"),
        num_steps=1,
        rows=rows,
        output_csv=Path("eval.csv"),
        ppo_gate=ppo_gate,
    )

    assert summary["early_offroad_count"] == 1
    assert summary["severe_early_offroad_count"] == 1
    assert summary["severe_early_terminal_offroad_count"] == 1
    assert summary["ppo_gate_status"] == "BLOCK"
    assert any("severe_early_offroad_count" in reason for reason in summary["ppo_gate_fail_reasons"])
    assert any("severe_early_terminal_offroad_count" in reason for reason in summary["ppo_gate_fail_reasons"])
