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


def build_shapley_subset_masks(
    config: ShapleyEstimatorConfig,
    *,
    num_modalities: int = NUM_MODALITIES,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    all_masks = build_all_subset_masks(num_modalities=num_modalities, device=device, dtype=dtype)
    sizes = all_masks.sum(dim=-1)
    keep = torch.ones((int(all_masks.shape[0]),), dtype=torch.bool, device=device)
    if not bool(config.include_empty_subset):
        keep = torch.logical_and(keep, sizes > 0)
    if not bool(config.include_full_subset):
        keep = torch.logical_and(keep, sizes < int(num_modalities))
    candidates = all_masks[keep]
    if int(candidates.shape[0]) <= 0:
        raise ValueError("Shapley subset selection produced no masks")

    requested_calls = max(1, int(config.subsets_per_sample))
    minimum_pair_complete_calls = 1
    if bool(config.include_empty_subset) or bool(config.include_full_subset):
        minimum_pair_complete_calls = int(num_modalities) + 1
    target_calls = max(requested_calls, minimum_pair_complete_calls)
    target_calls = min(target_calls, int(candidates.shape[0]))
    selected: list[tuple[int, ...]] = []
    candidate_keys = {tuple(int(item) for item in mask.detach().cpu().tolist()) for mask in candidates}

    def add_key(key: tuple[int, ...]) -> None:
        if key in candidate_keys and key not in selected:
            selected.append(key)

    empty_key = tuple(0 for _ in range(int(num_modalities)))
    full_key = tuple(1 for _ in range(int(num_modalities)))
    if bool(config.include_empty_subset):
        add_key(empty_key)
        for modality in range(int(num_modalities)):
            key = tuple(1 if index == modality else 0 for index in range(int(num_modalities)))
            add_key(key)
            if len(selected) >= target_calls:
                break
    if len(selected) < target_calls and bool(config.include_full_subset):
        add_key(full_key)
        for modality in range(int(num_modalities)):
            key = tuple(0 if index == modality else 1 for index in range(int(num_modalities)))
            add_key(key)
            if len(selected) >= target_calls:
                break
    for mask in candidates:
        if len(selected) >= target_calls:
            break
        add_key(tuple(int(item) for item in mask.detach().cpu().tolist()))

    return torch.as_tensor(selected, dtype=dtype, device=device)


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
    selected_count = max(1, int(round(batch_size * float(config.batch_fraction))))
    selected_count = min(selected_count, batch_size)
    row_indices = torch.arange(selected_count, device=device)
    projected_selected = projected_feature_stack[row_indices]
    weights_selected = fusion_weights[row_indices]
    hidden_selected = hidden_state
    if hidden_state is not None:
        hidden_selected = hidden_state[:, row_indices, :].contiguous()
    masks = build_shapley_subset_masks(config, device=device, dtype=dtype)

    full_mask = torch.ones((NUM_MODALITIES,), dtype=dtype, device=device)
    if bool(config.include_full_subset) and not any(bool(torch.equal(mask, full_mask)) for mask in masks):
        masks = torch.cat([masks, full_mask.unsqueeze(0)], dim=0)
    values: dict[tuple[int, ...], torch.Tensor] = {}
    for mask in masks:
        expanded = mask.unsqueeze(0).expand(selected_count, -1)
        key = tuple(int(item) for item in mask.detach().cpu().tolist())
        values[key] = evaluator.evaluate_masked_values(
            projected_selected,
            weights_selected,
            expanded,
            hidden_state=hidden_selected,
        )

    raw_selected = projected_feature_stack.new_zeros((selected_count, NUM_MODALITIES))
    n_fact = float(factorial(NUM_MODALITIES))
    for modality in range(NUM_MODALITIES):
        for key, subset_value in values.items():
            if key[modality] == 1:
                continue
            subset_size = sum(key)
            with_modality = list(key)
            with_modality[modality] = 1
            with_key = tuple(with_modality)
            if with_key not in values:
                continue
            weight = float(factorial(subset_size) * factorial(NUM_MODALITIES - subset_size - 1)) / n_fact
            raw_selected[:, modality] = raw_selected[:, modality] + weight * (values[with_key] - subset_value)

    contribution = fusion_weights.to(device=device, dtype=dtype).detach().clamp_min(1.0e-8)
    contribution = contribution / contribution.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    raw = projected_feature_stack.new_zeros((batch_size, NUM_MODALITIES))
    selected_contribution = normalize_contributions(
        raw_selected,
        temperature=float(config.contribution_temperature),
        min_contribution=float(config.min_contribution),
    )
    contribution[row_indices] = selected_contribution
    raw[row_indices] = raw_selected
    if bool(config.detach_contribution_target):
        contribution = contribution.detach()
        raw = raw.detach()
    return ShapleyEstimate(raw=raw, contribution=contribution, subset_count=int(masks.shape[0]))
