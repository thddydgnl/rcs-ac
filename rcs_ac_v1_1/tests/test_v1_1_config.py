from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from rcs_ac_v1_1.runtime.config import build_v11_runtime_config, load_v11_runtime_config


CONFIG_PATH = Path("configs/v1_1/full_bev_smoke.yaml")


def _load_raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_full_bev_smoke_config_builds_runtime_objects() -> None:
    config = load_v11_runtime_config(CONFIG_PATH)
    assert config.version == "rcs_ac_v1_1"
    assert config.bev_spec.shape_hw == (128, 128)
    assert config.observation_spec.total_dim == 239839
    assert config.target_spec.gt_bev_shape == (5, 128, 128)
    assert config.controller_config.invalid_path_target_speed_kmh == 0.0

    model = config.build_policy()
    outputs = model(torch.zeros((1, config.observation_spec.total_dim), dtype=torch.float32))
    assert tuple(outputs["waypoints"].shape) == (1, 4, 2)

    controller = config.build_controller()
    control = controller.control(
        [(-1.0, 0.0)],
        target_speed_kmh=12.0,
        current_speed_kmh=8.0,
        dt=0.05,
    )
    assert control["debug"]["used_synthetic_fallback"] is True


def test_config_rejects_model_bev_spec_mismatch() -> None:
    raw = _load_raw_config()
    raw["model"]["bev_height"] = 127
    with pytest.raises(ValueError, match="model.bev_height"):
        build_v11_runtime_config(raw)


def test_config_rejects_declared_policy_input_shape_mismatch() -> None:
    raw = _load_raw_config()
    raw["policy_inputs"]["lidar_bev"] = [5, 127, 128]
    with pytest.raises(ValueError, match="policy_inputs.lidar_bev"):
        build_v11_runtime_config(raw)


def test_config_rejects_unsupported_controller_type() -> None:
    raw = _load_raw_config()
    raw["controller"] = copy.deepcopy(raw["controller"])
    raw["controller"]["type"] = "pid"
    with pytest.raises(ValueError, match="controller.type"):
        build_v11_runtime_config(raw)
