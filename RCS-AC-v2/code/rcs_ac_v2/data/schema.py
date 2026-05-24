"""Stage 0 tensor schema and leakage boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rcs_ac_v2.env.discrete_actions import NUM_RAW2DRIVE_ACTIONS


@dataclass(frozen=True)
class SchemaConfig:
    privileged_bev_channels: int = 43
    image_channels: int = 3
    num_actions: int = NUM_RAW2DRIVE_ACTIONS
    ego_dim: int | None = None
    imu_dim: int | None = None
    num_cameras: int | None = None


@dataclass
class PrivilegedSequenceBatch:
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    next_privileged_bev: torch.Tensor | None = None


@dataclass
class RawPolicyInput:
    raw_images: torch.Tensor
    raw_imu: torch.Tensor
    route_command: torch.Tensor | None = None
    past_action: torch.Tensor | None = None


@dataclass
class RawSensorSequenceBatch:
    raw_images: torch.Tensor
    raw_imu: torch.Tensor
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    route_command: torch.Tensor | None = None

    def policy_input_view(self, *, include_past_action: bool = False) -> RawPolicyInput:
        return RawPolicyInput(
            raw_images=self.raw_images,
            raw_imu=self.raw_imu,
            route_command=self.route_command,
            past_action=self.action if include_past_action else None,
        )


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def _require_float(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be floating point, got {tensor.dtype}")


def _require_batch_time(name: str, tensor: torch.Tensor, batch_size: int, sequence_length: int) -> None:
    if tuple(tensor.shape[:2]) != (batch_size, sequence_length):
        raise ValueError(
            f"{name} must start with [B,T]=[{batch_size},{sequence_length}], "
            f"got {tuple(tensor.shape[:2])}"
        )


def _validate_action(action: torch.Tensor, cfg: SchemaConfig, batch_size: int, sequence_length: int) -> None:
    _require_rank("action", action, 2)
    _require_batch_time("action", action, batch_size, sequence_length)
    if action.dtype != torch.int64:
        raise ValueError(f"action must be torch.int64, got {action.dtype}")
    if action.numel() > 0:
        if int(action.min()) < 0:
            raise ValueError("action contains negative ids")
        if int(action.max()) >= cfg.num_actions:
            raise ValueError(f"action id must be < {cfg.num_actions}")


def _validate_reward_cont(
    reward: torch.Tensor,
    cont: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> None:
    for name, tensor in (("reward", reward), ("cont", cont)):
        _require_rank(name, tensor, 2)
        _require_batch_time(name, tensor, batch_size, sequence_length)
        _require_float(name, tensor)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    if not torch.logical_or(cont == 0.0, cont == 1.0).all():
        raise ValueError("cont must contain only 0 or 1")


def validate_privileged_batch(
    batch: PrivilegedSequenceBatch,
    cfg: SchemaConfig | None = None,
) -> None:
    cfg = cfg or SchemaConfig()

    privileged_bev = _require_tensor("privileged_bev", batch.privileged_bev)
    privileged_ego = _require_tensor("privileged_ego", batch.privileged_ego)
    action = _require_tensor("action", batch.action)
    reward = _require_tensor("reward", batch.reward)
    cont = _require_tensor("cont", batch.cont)

    _require_rank("privileged_bev", privileged_bev, 5)
    _require_float("privileged_bev", privileged_bev)
    batch_size, sequence_length, channels = privileged_bev.shape[:3]
    if channels != cfg.privileged_bev_channels:
        raise ValueError(
            f"privileged_bev channel count must be {cfg.privileged_bev_channels}, got {channels}"
        )
    if not torch.isfinite(privileged_bev).all():
        raise ValueError("privileged_bev contains non-finite values")

    _require_rank("privileged_ego", privileged_ego, 3)
    _require_float("privileged_ego", privileged_ego)
    _require_batch_time("privileged_ego", privileged_ego, batch_size, sequence_length)
    if cfg.ego_dim is not None and privileged_ego.shape[-1] != cfg.ego_dim:
        raise ValueError(f"privileged_ego dim must be {cfg.ego_dim}, got {privileged_ego.shape[-1]}")
    if not torch.isfinite(privileged_ego).all():
        raise ValueError("privileged_ego contains non-finite values")

    _validate_action(action, cfg, batch_size, sequence_length)
    _validate_reward_cont(reward, cont, batch_size, sequence_length)

    if batch.next_privileged_bev is not None:
        next_bev = _require_tensor("next_privileged_bev", batch.next_privileged_bev)
        if next_bev.shape != privileged_bev.shape:
            raise ValueError("next_privileged_bev must match privileged_bev shape")
        _require_float("next_privileged_bev", next_bev)


def validate_raw_policy_input(policy_input: RawPolicyInput, cfg: SchemaConfig | None = None) -> None:
    cfg = cfg or SchemaConfig()
    raw_images = _require_tensor("raw_images", policy_input.raw_images)
    raw_imu = _require_tensor("raw_imu", policy_input.raw_imu)

    _require_rank("raw_images", raw_images, 6)
    _require_float("raw_images", raw_images)
    if raw_images.shape[3] != cfg.image_channels:
        raise ValueError(f"raw_images channel count must be {cfg.image_channels}")
    if cfg.num_cameras is not None and raw_images.shape[2] != cfg.num_cameras:
        raise ValueError(f"raw_images camera count must be {cfg.num_cameras}")

    _require_rank("raw_imu", raw_imu, 3)
    _require_float("raw_imu", raw_imu)
    _require_batch_time("raw_imu", raw_imu, raw_images.shape[0], raw_images.shape[1])
    if cfg.imu_dim is not None and raw_imu.shape[-1] != cfg.imu_dim:
        raise ValueError(f"raw_imu dim must be {cfg.imu_dim}")


def validate_raw_sensor_batch(
    batch: RawSensorSequenceBatch,
    cfg: SchemaConfig | None = None,
) -> None:
    cfg = cfg or SchemaConfig()
    raw_images = _require_tensor("raw_images", batch.raw_images)
    raw_imu = _require_tensor("raw_imu", batch.raw_imu)

    _require_rank("raw_images", raw_images, 6)
    _require_float("raw_images", raw_images)
    batch_size, sequence_length, num_cameras, channels = raw_images.shape[:4]
    if channels != cfg.image_channels:
        raise ValueError(f"raw_images channel count must be {cfg.image_channels}, got {channels}")
    if cfg.num_cameras is not None and num_cameras != cfg.num_cameras:
        raise ValueError(f"raw_images camera count must be {cfg.num_cameras}, got {num_cameras}")
    if not torch.isfinite(raw_images).all():
        raise ValueError("raw_images contains non-finite values")

    _require_rank("raw_imu", raw_imu, 3)
    _require_float("raw_imu", raw_imu)
    _require_batch_time("raw_imu", raw_imu, batch_size, sequence_length)
    if cfg.imu_dim is not None and raw_imu.shape[-1] != cfg.imu_dim:
        raise ValueError(f"raw_imu dim must be {cfg.imu_dim}, got {raw_imu.shape[-1]}")
    if not torch.isfinite(raw_imu).all():
        raise ValueError("raw_imu contains non-finite values")

    privileged = PrivilegedSequenceBatch(
        privileged_bev=batch.privileged_bev,
        privileged_ego=batch.privileged_ego,
        action=batch.action,
        reward=batch.reward,
        cont=batch.cont,
    )
    validate_privileged_batch(privileged, cfg)
    if batch.privileged_bev.shape[:2] != raw_images.shape[:2]:
        raise ValueError("privileged and raw tensors must share [B,T]")
    validate_raw_policy_input(batch.policy_input_view(), cfg)


def make_synthetic_privileged_batch(
    *,
    batch_size: int = 2,
    sequence_length: int = 4,
    height: int = 16,
    width: int = 16,
    ego_dim: int = 8,
    seed: int = 0,
    binary_bev: bool = True,
    cfg: SchemaConfig | None = None,
) -> PrivilegedSequenceBatch:
    cfg = cfg or SchemaConfig(ego_dim=ego_dim)
    generator = torch.Generator().manual_seed(seed)
    privileged_bev = torch.rand(
        batch_size,
        sequence_length,
        cfg.privileged_bev_channels,
        height,
        width,
        generator=generator,
    )
    if binary_bev:
        privileged_bev = (privileged_bev > 0.5).float()
    privileged_ego = torch.randn(batch_size, sequence_length, ego_dim, generator=generator)
    action = torch.randint(0, cfg.num_actions, (batch_size, sequence_length), generator=generator)
    reward = torch.randn(batch_size, sequence_length, generator=generator)
    cont = torch.ones(batch_size, sequence_length, dtype=torch.float32)
    return PrivilegedSequenceBatch(
        privileged_bev=privileged_bev.float(),
        privileged_ego=privileged_ego.float(),
        action=action.to(torch.int64),
        reward=reward.float(),
        cont=cont,
    )


def make_synthetic_raw_sensor_batch(
    *,
    batch_size: int = 2,
    sequence_length: int = 4,
    num_cameras: int = 4,
    image_height: int = 24,
    image_width: int = 32,
    imu_dim: int = 6,
    bev_height: int = 16,
    bev_width: int = 16,
    ego_dim: int = 8,
    seed: int = 0,
    binary_bev: bool = True,
    cfg: SchemaConfig | None = None,
) -> RawSensorSequenceBatch:
    cfg = cfg or SchemaConfig(ego_dim=ego_dim, imu_dim=imu_dim, num_cameras=num_cameras)
    generator = torch.Generator().manual_seed(seed)
    privileged = make_synthetic_privileged_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        height=bev_height,
        width=bev_width,
        ego_dim=ego_dim,
        seed=seed,
        binary_bev=binary_bev,
        cfg=cfg,
    )
    raw_images = torch.rand(
        batch_size,
        sequence_length,
        num_cameras,
        cfg.image_channels,
        image_height,
        image_width,
        generator=generator,
    )
    raw_imu = torch.randn(batch_size, sequence_length, imu_dim, generator=generator)
    return RawSensorSequenceBatch(
        raw_images=raw_images.float(),
        raw_imu=raw_imu.float(),
        privileged_bev=privileged.privileged_bev,
        privileged_ego=privileged.privileged_ego,
        action=privileged.action,
        reward=privileged.reward,
        cont=privileged.cont,
    )
