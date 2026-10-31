import torch

from rcs_ac_first.rl.shapley import (
    ShapleyEstimatorConfig,
    build_shapley_subset_masks,
    estimate_exact_shapley_current_critic,
)


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_masked_values(
        self,
        projected_feature_stack: torch.Tensor,
        fusion_weights: torch.Tensor,
        modality_masks: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del fusion_weights, hidden_state
        self.batch_sizes.append(int(projected_feature_stack.shape[0]))
        return modality_masks.sum(dim=-1).to(dtype=projected_feature_stack.dtype)


def test_shapley_respects_batch_fraction_and_subset_limit() -> None:
    evaluator = _RecordingEvaluator()
    projected = torch.zeros((4, 4, 3), dtype=torch.float32)
    fusion_weights = torch.full((4, 4), 0.25, dtype=torch.float32)

    estimate = estimate_exact_shapley_current_critic(
        evaluator,
        projected_feature_stack=projected,
        fusion_weights=fusion_weights,
        hidden_state=torch.zeros((1, 4, 5), dtype=torch.float32),
        config=ShapleyEstimatorConfig(
            subsets_per_sample=2,
            batch_fraction=0.5,
            include_empty_subset=True,
            include_full_subset=False,
        ),
    )

    assert estimate.subset_count == 5
    assert evaluator.batch_sizes == [2, 2, 2, 2, 2]
    torch.testing.assert_close(estimate.contribution[2:], fusion_weights[2:])


def test_shapley_subset_limit_keeps_all_singleton_pairs() -> None:
    masks = build_shapley_subset_masks(
        ShapleyEstimatorConfig(
            subsets_per_sample=4,
            include_empty_subset=True,
            include_full_subset=True,
        ),
        device=torch.device("cpu"),
    )
    keys = {tuple(int(item) for item in mask.tolist()) for mask in masks}

    assert (0, 0, 0, 0) in keys
    for modality in range(4):
        assert tuple(1 if index == modality else 0 for index in range(4)) in keys
