import pytest
import torch

from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
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
