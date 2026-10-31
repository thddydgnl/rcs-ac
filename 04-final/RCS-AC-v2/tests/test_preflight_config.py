from __future__ import annotations

from pathlib import Path

import yaml

from rcs_ac_v2.env.discrete_actions import ACTION_TABLE_SHA256, ACTION_TABLE_VERSION


def test_preflight_config_pins_stage0_contracts() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "eval" / "preflight_contract.yaml"
    config = yaml.safe_load(path.read_text())

    assert config["stage"] == "preflight_contract"
    assert config["contract_version"] == "stage0_raw2drive_contract_v1"
    assert config["action_space"] == "raw2drive_discrete_39"
    assert config["action_table_version"] == ACTION_TABLE_VERSION
    assert config["action_table_sha256"] == ACTION_TABLE_SHA256
    assert config["reward_contract_version"] == "stage0_route_progress_penalty_v1"
    assert config["privileged_bev_channels"] == 43
    assert config["raw_sensor_input"] == {"multi_view_images": True, "imu": True}
    assert config["continue_contract"] == {
        "terminal_done": 0.0,
        "running": 1.0,
        "time_limit_truncated": 1.0,
    }

    allowed = set(config["leakage"]["raw_policy_inference_allowed"])
    forbidden = set(config["leakage"]["raw_policy_inference_forbidden"])
    assert allowed == {"raw_images", "raw_imu", "route_command", "past_action"}
    assert allowed.isdisjoint(forbidden)
    assert "privileged_bev" in forbidden
    assert "reward" in forbidden
    assert "cont" in forbidden
