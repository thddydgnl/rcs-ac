from __future__ import annotations

from rcs_ac_first.scripts.audit_actor_sign_response import _axis_summary, _gate_status


def _rows() -> list[dict[str, float | str]]:
    return [
        {"axis": "lane_offset", "probe_value": -1.0, "steer": 0.10},
        {"axis": "lane_offset", "probe_value": 0.0, "steer": 0.0},
        {"axis": "lane_offset", "probe_value": 1.0, "steer": -0.10},
        {"axis": "heading_error", "probe_value": -1.0, "steer": -0.10},
        {"axis": "heading_error", "probe_value": 0.0, "steer": 0.0},
        {"axis": "heading_error", "probe_value": 1.0, "steer": 0.10},
        {"axis": "route_target_y", "probe_value": -1.0, "steer": -0.10},
        {"axis": "route_target_y", "probe_value": 0.0, "steer": 0.0},
        {"axis": "route_target_y", "probe_value": 1.0, "steer": 0.10},
    ]


def test_actor_sign_axis_summary_uses_semantic_heading_and_route_direction() -> None:
    summary = _axis_summary(_rows(), min_delta=0.005, sign_convention="semantic")

    assert summary["lane_offset"]["sign_pass"] is True
    assert summary["lane_offset"]["expected_direction"] == -1
    assert summary["heading_error"]["sign_pass"] is True
    assert summary["heading_error"]["expected_direction"] == 1
    assert summary["route_target_y"]["sign_pass"] is True
    assert summary["route_target_y"]["expected_direction"] == 1


def test_actor_sign_axis_summary_preserves_legacy_all_negative_direction() -> None:
    summary = _axis_summary(_rows(), min_delta=0.005, sign_convention="legacy")

    assert summary["lane_offset"]["sign_pass"] is True
    assert summary["heading_error"]["sign_pass"] is False
    assert summary["route_target_y"]["sign_pass"] is False


def test_actor_sign_gate_blocks_positive_zero_bias() -> None:
    summary = _axis_summary(_rows(), min_delta=0.005, sign_convention="semantic")
    for axis_summary in summary.values():
        axis_summary["zero_steer"] = 0.16

    gate = _gate_status(summary, min_delta=0.005, max_zero_bias=0.03)

    assert gate["actor_sign_response_max_abs_zero_steer"] == 0.16
    assert gate["pass_status"] == "BLOCK"
