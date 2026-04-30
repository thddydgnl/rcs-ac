import torch

from rcs_ac_first.models.fusion_gate import FusionGate, FusionGateConfig, MODALITY_ORDER


def _features(batch_size: int = 2, feature_dim: int = 3) -> dict[str, torch.Tensor]:
    return {
        key: torch.full((batch_size, feature_dim), float(index + 1))
        for index, key in enumerate(MODALITY_ORDER)
    }


def test_fusion_weights_sum_to_one_and_mask_drops_modality() -> None:
    gate = FusionGate(FusionGateConfig(feature_dim=3, fusion_dim=4, ego_weight_floor=0.1, route_weight_floor=0.2))
    mask = torch.ones((2, 4))
    mask[:, MODALITY_ORDER.index("route")] = 0.0

    output = gate(_features(), modality_mask=mask)

    torch.testing.assert_close(output.weights.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(output.weights[:, MODALITY_ORDER.index("route")], torch.zeros(2))


def test_reliability_scales_ego_route_floor_down() -> None:
    gate = FusionGate(
        FusionGateConfig(
            feature_dim=3,
            fusion_dim=4,
            ego_weight_floor=0.2,
            route_weight_floor=0.3,
        )
    )
    reliability = torch.ones((1, 4))
    reliability[:, MODALITY_ORDER.index("route")] = 0.0

    output = gate(_features(batch_size=1), reliability=reliability)

    torch.testing.assert_close(output.weights.sum(dim=-1), torch.ones(1))
    assert float(output.weights[:, MODALITY_ORDER.index("route")].item()) < 0.3
