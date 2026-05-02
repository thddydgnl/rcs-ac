import pytest
import torch

from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.runtime.auxiliary import RISK_STATS_FIELDS, build_auxiliary_observation_spec, default_flat_auxiliary
from rcs_ac_first.runtime.config import RCSACFirstModelConfig


def test_rgb_perception_actor_adapter_init_std_controls_visual_weights() -> None:
    torch.manual_seed(7)
    config = RCSACFirstModelConfig(
        use_rgb_perception_actor_features=True,
        rgb_perception_actor_feature_dim=16,
        rgb_perception_actor_adapter_init_std=0.05,
    )
    model = RCSACFirstModel(config)

    assert model.actor_input_adapter is not None
    fusion_dim = int(config.fusion_dim)
    fusion_weights = model.actor_input_adapter.weight[:, :fusion_dim]
    visual_weights = model.actor_input_adapter.weight[:, fusion_dim:]

    assert torch.allclose(fusion_weights, torch.eye(fusion_dim))
    assert float(visual_weights.detach().abs().mean()) > 0.01


def test_rgb_perception_actor_adapter_init_std_rejects_negative() -> None:
    with pytest.raises(ValueError, match="rgb_perception_actor_adapter_init_std"):
        RCSACFirstModelConfig(
            use_rgb_perception_actor_features=True,
            rgb_perception_actor_adapter_init_std=-0.01,
        )


def test_scalar_actor_adapter_starts_as_zero_residual() -> None:
    config = RCSACFirstModelConfig(use_scalar_actor_features=True)
    model = RCSACFirstModel(config)

    assert model.scalar_actor_adapter is not None
    assert tuple(model.scalar_actor_adapter.weight.shape) == (int(config.fusion_dim), 3)
    assert torch.allclose(model.scalar_actor_adapter.weight, torch.zeros_like(model.scalar_actor_adapter.weight))


def test_scalar_actor_features_extract_lane_heading_and_route_y() -> None:
    config = RCSACFirstModelConfig(use_scalar_actor_features=True)
    model = RCSACFirstModel(config)
    spec = build_auxiliary_observation_spec()
    aux = default_flat_auxiliary(1, spec=spec)
    risk = spec.item("risk_stats")
    aux[:, risk.start + RISK_STATS_FIELDS.index("lane_offset")] = -0.8
    aux[:, risk.start + RISK_STATS_FIELDS.index("heading_error")] = 12.0
    route = torch.zeros((1, int(config.route_dim)), dtype=torch.float32)
    route[:, 1] = 0.25
    reference = torch.zeros((1, int(config.fusion_dim)), dtype=torch.float32)

    features = model._scalar_actor_features(aux, route_input=route, reference=reference)

    assert torch.allclose(features, torch.tensor([[-0.8, 1.2, 0.25]], dtype=torch.float32))


def test_scalar_actor_adapter_can_change_actor_input_without_changing_base_adapter() -> None:
    config = RCSACFirstModelConfig(
        use_rgb_perception_actor_features=True,
        use_scalar_actor_features=True,
        scalar_actor_adapter_init_std=0.0,
    )
    model = RCSACFirstModel(config)
    assert model.scalar_actor_adapter is not None
    with torch.no_grad():
        model.scalar_actor_adapter.weight[:, 0].fill_(0.5)
    fused = torch.zeros((1, int(config.fusion_dim)), dtype=torch.float32)
    visual = torch.zeros((1, int(config.rgb_perception_actor_feature_dim)), dtype=torch.float32)
    spec = build_auxiliary_observation_spec()
    aux = default_flat_auxiliary(1, spec=spec)
    risk = spec.item("risk_stats")
    aux[:, risk.start + RISK_STATS_FIELDS.index("lane_offset")] = 2.0
    route = torch.zeros((1, int(config.route_dim)), dtype=torch.float32)

    actor_input = model._build_actor_input(fused, visual, auxiliary_tensor=aux, route_input=route)

    assert torch.allclose(actor_input, torch.full_like(actor_input, 1.0))


def test_scalar_actor_adapter_init_std_rejects_negative() -> None:
    with pytest.raises(ValueError, match="scalar_actor_adapter_init_std"):
        RCSACFirstModelConfig(use_scalar_actor_features=True, scalar_actor_adapter_init_std=-0.01)
