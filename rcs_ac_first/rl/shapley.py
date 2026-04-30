from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Protocol

import torch

from rcs_ac_first.runtime.modality_mask import NUM_MODALITIES


class CurrentCriticValueEvaluator(Protocol):
    def evaluate_masked_values(
        self,
        projected_feature_stack: torch.Tensor,
        fusion_weights: torch.Tensor,
        modality_masks: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


@dataclass
class ShapleyEstimatorConfig:
    subsets_per_sample: int = 4
    batch_fraction: float = 0.25
    contribution_temperature: float = 1.0
    min_contribution: float = 0.02
    include_empty_subset: bool = True
    include_full_subset: bool = True
    detach_contribution_target: bool = True


@dataclass
class ShapleyEstimate:
    raw: torch.Tensor
    contribution: torch.Tensor
    subset_count: int


def build_all_subset_masks(
    *,
    num_modalities: int = NUM_MODALITIES,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    masks = []
    for value in range(2 ** int(num_modalities)):
        masks.append([(value >> index) & 1 for index in range(int(num_modalities))])
    return torch.as_tensor(masks, dtype=dtype, device=device)


def normalize_contributions(
    raw: torch.Tensor,
    *,
    temperature: float = 1.0,
    min_contribution: float = 0.02,
) -> torch.Tensor:
    if raw.ndim != 2 or int(raw.shape[1]) != NUM_MODALITIES:
        raise ValueError(f"raw Shapley tensor must have shape [B, 4], got {tuple(raw.shape)}")
    if not torch.isfinite(raw).all():
        raise ValueError("raw Shapley tensor contains NaN or inf values")
    contribution = torch.softmax(raw / float(temperature), dim=-1)
    floor = float(min_contribution)
    if floor > 0.0:
        contribution = contribution * (1.0 - floor * NUM_MODALITIES) + floor
    return contribution / contribution.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def estimate_exact_shapley_current_critic(
    evaluator: CurrentCriticValueEvaluator,
    *,
    projected_feature_stack: torch.Tensor,
    fusion_weights: torch.Tensor,
    hidden_state: torch.Tensor | None,
    config: ShapleyEstimatorConfig,
) -> ShapleyEstimate:
    batch_size = int(projected_feature_stack.shape[0])
    device = projected_feature_stack.device
    dtype = projected_feature_stack.dtype
    masks = build_all_subset_masks(device=device, dtype=dtype)
    values: dict[tuple[int, ...], torch.Tensor] = {}
    for mask in masks:
        expanded = mask.unsqueeze(0).expand(batch_size, -1)
        key = tuple(int(item) for item in mask.detach().cpu().tolist())
        values[key] = evaluator.evaluate_masked_values(
            projected_feature_stack,
            fusion_weights,
            expanded,
            hidden_state=hidden_state,
        )

    raw = projected_feature_stack.new_zeros((batch_size, NUM_MODALITIES))
    n_fact = float(factorial(NUM_MODALITIES))
    for modality in range(NUM_MODALITIES):
        for key, subset_value in values.items():
            if key[modality] == 1:
                continue
            subset_size = sum(key)
            with_modality = list(key)
            with_modality[modality] = 1
            weight = float(factorial(subset_size) * factorial(NUM_MODALITIES - subset_size - 1)) / n_fact
            raw[:, modality] = raw[:, modality] + weight * (values[tuple(with_modality)] - subset_value)

    contribution = normalize_contributions(
        raw,
        temperature=float(config.contribution_temperature),
        min_contribution=float(config.min_contribution),
    )
    if bool(config.detach_contribution_target):
        contribution = contribution.detach()
        raw = raw.detach()
    return ShapleyEstimate(raw=raw, contribution=contribution, subset_count=int(masks.shape[0]))
