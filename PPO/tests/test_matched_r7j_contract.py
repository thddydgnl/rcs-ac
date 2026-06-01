from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
V11_CODE = ROOT / "RCS-AC-v1.1" / "code"
for candidate in (ROOT, V11_CODE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from PPO.ppo_r7j.models import PPOMatchedR7JPolicy
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config


def test_ppo_matched_r7j_output_contract() -> None:
    runtime_config = load_v11_runtime_config(
        ROOT / "RCS-AC-v1.1" / "configs" / "v1_1" / "full_bev_smoke_shared_agile_midbalanced_route_geometry_cc4_r7j_target_y_lowpass_light.yaml"
    )
    model = PPOMatchedR7JPolicy(runtime_config)
    obs = torch.zeros((1, runtime_config.observation_spec.total_dim), dtype=torch.float32)
    sensor_health = runtime_config.observation_spec.modal("sensor_health")
    obs[:, sensor_health.start : sensor_health.stop] = 1.0
    outputs = model.act(obs, deterministic=True)
    assert tuple(outputs["waypoints"].shape) == (1, 4, 2)
    assert tuple(outputs["target_speed_kmh"].shape) == (1, 1)
    assert tuple(outputs["raw_action"].shape) == (1, 9)
    assert tuple(outputs["bev_fusion_weights"].shape) == (1, 4)
    assert torch.isfinite(outputs["waypoints"]).all()
    assert torch.isfinite(outputs["target_speed_kmh"]).all()
