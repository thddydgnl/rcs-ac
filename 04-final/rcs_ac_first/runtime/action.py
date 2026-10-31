from __future__ import annotations

from typing import Any

import torch

STAGE4_ACTION_V2 = "stage4_action_v2"

ACTION_CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    STAGE4_ACTION_V2: {
        "shape": (2,),
        "order": ("steer", "longitudinal"),
        "bounds": {
            "steer": (-1.0, 1.0),
            "longitudinal": (-1.0, 1.0),
        },
        "policy_action_space": "steer_longitudinal",
    }
}


def action_contract_spec(version: str = STAGE4_ACTION_V2) -> dict[str, Any]:
    if version not in ACTION_CONTRACT_SPECS:
        raise ValueError(f"Unsupported action contract version: {version!r}")
    return ACTION_CONTRACT_SPECS[version]


def action_dim_for_contract(version: str = STAGE4_ACTION_V2) -> int:
    return int(action_contract_spec(version)["shape"][0])


def _bounds_tensor(
    *,
    version: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    spec = action_contract_spec(version)
    order = tuple(spec["order"])
    bounds = spec["bounds"]
    lower = torch.as_tensor([float(bounds[key][0]) for key in order], dtype=dtype, device=device)
    upper = torch.as_tensor([float(bounds[key][1]) for key in order], dtype=dtype, device=device)
    return lower, upper


def validate_action_tensor(action: torch.Tensor, version: str = STAGE4_ACTION_V2) -> None:
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"action must be a torch.Tensor, got {type(action)!r}")
    expected_dim = action_dim_for_contract(version)
    if action.shape[-1] != expected_dim:
        raise ValueError(f"Expected action last dim {expected_dim}, got shape {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("Action contains NaN or inf values.")


def clamp_action_tensor(action: torch.Tensor, version: str = STAGE4_ACTION_V2) -> torch.Tensor:
    validate_action_tensor(action, version=version)
    lower, upper = _bounds_tensor(version=version, dtype=action.dtype, device=action.device)
    return torch.maximum(torch.minimum(action, upper), lower)


def decode_stage4_action_v2(action: torch.Tensor | list[float] | tuple[float, ...]) -> dict[str, float]:
    tensor = torch.as_tensor(action, dtype=torch.float32)
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    validate_action_tensor(tensor, version=STAGE4_ACTION_V2)
    tensor = clamp_action_tensor(tensor, version=STAGE4_ACTION_V2)
    steer = float(tensor[0].item())
    longitudinal = float(tensor[1].item())
    return {
        "steer": steer,
        "longitudinal": longitudinal,
        "throttle": max(longitudinal, 0.0),
        "brake": max(-longitudinal, 0.0),
    }
