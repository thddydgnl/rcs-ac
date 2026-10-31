from __future__ import annotations

import torch

MODALITY_ORDER: tuple[str, ...] = ("rgb", "lidar", "ego", "route")
NUM_MODALITIES = len(MODALITY_ORDER)


def full_modality_mask(batch_size: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.ones((int(batch_size), NUM_MODALITIES), dtype=dtype, device=device)


def validate_modality_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    allow_empty: bool = False,
) -> torch.Tensor:
    if mask is None:
        return full_modality_mask(batch_size, device=device, dtype=dtype)
    result = mask.to(device=device, dtype=dtype)
    if result.ndim == 1:
        result = result.unsqueeze(0)
    expected = (int(batch_size), NUM_MODALITIES)
    if tuple(result.shape) != expected:
        raise ValueError(f"modality_mask shape mismatch: expected {expected}, got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError("modality_mask contains NaN or inf values")
    result = (result > 0.5).to(dtype=dtype)
    if not allow_empty and bool((result.sum(dim=-1) <= 0.0).any()):
        raise ValueError("modality_mask contains an empty row")
    return result


def single_modality_drop_masks(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    base = full_modality_mask(batch_size, device=device, dtype=dtype)
    rows = torch.arange(int(batch_size), device=device)
    base[rows, rows % NUM_MODALITIES] = 0.0
    return base
