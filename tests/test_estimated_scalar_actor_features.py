import torch

from rcs_ac_first.models.rcs_ac_model import RCSACFirstModel
from rcs_ac_first.runtime.config import RCSACFirstFusionConfig, RCSACFirstModelConfig


def test_existing_scalar_configs_default_to_oracle_auxiliary() -> None:
    config = RCSACFirstModelConfig(use_scalar_actor_features=True)
    assert config.scalar_actor_feature_source == "oracle_auxiliary"

    model = RCSACFirstModel(config, RCSACFirstFusionConfig())
    assert model.scalar_actor_adapter is not None
    assert model.estimated_scalar_head is None


def test_estimated_scalar_source_uses_estimator_without_oracle_actor_input() -> None:
    config = RCSACFirstModelConfig(
        enable_rgb_perception_aux=True,
        use_scalar_actor_features=True,
        scalar_actor_feature_source="estimated",
    )
    model = RCSACFirstModel(config, RCSACFirstFusionConfig())
    assert model.scalar_actor_adapter is not None
    assert model.estimated_scalar_head is not None

    obs = torch.zeros((2, int(config.obs_dim)), dtype=torch.float32)
    auxiliary = torch.zeros((2, int(config.auxiliary_dim)), dtype=torch.float32)
    outputs = model.forward_policy_outputs(obs, auxiliary=auxiliary, track_diagnostics=True)

    assert isinstance(outputs["estimated_lane_offset"], torch.Tensor)
    assert isinstance(outputs["estimated_heading_error"], torch.Tensor)
    assert tuple(outputs["estimated_lane_offset"].shape) == (2,)
    assert tuple(outputs["estimated_heading_error"].shape) == (2,)
    diagnostics = model.get_diagnostics()
    assert diagnostics["scalar_actor_feature_source_estimated"] == 1.0
    assert diagnostics["scalar_actor_feature_source_oracle_auxiliary"] == 0.0


def test_disabled_scalar_source_disables_scalar_actor_features() -> None:
    config = RCSACFirstModelConfig(
        use_scalar_actor_features=True,
        scalar_actor_feature_source="disabled",
    )
    assert config.use_scalar_actor_features is False
    model = RCSACFirstModel(config, RCSACFirstFusionConfig())
    assert model.scalar_actor_adapter is None
    assert model.estimated_scalar_head is None
