from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.multimodal_encoders import ImageEncoder, VectorEncoder, prepare_modal_batch


@dataclass
class EarlyFusionConfig:
    obs_dim: int
    action_dim: int = 3
    hidden_dim: int = 256
    min_std: float = 0.05
    rgb_shape: tuple[int, int, int] = (3, 90, 160)
    lidar_shape: tuple[int, int, int] = (1, 64, 64)
    ego_dim: int = 7
    route_dim: int = 6
    rgb_feature_dim: int = 64
    lidar_feature_dim: int = 32
    ego_feature_dim: int = 32
    route_feature_dim: int = 32


class EarlyFusionActorCritic(nn.Module):
    def __init__(self, config: EarlyFusionConfig) -> None:
        super().__init__()
        self.config = config
        self._last_debug_info = {
            "encoder_feature_dims": "",
            "fused_dim": int(config.hidden_dim),
            "obs_keys_used": "rgb,lidar,ego,route",
            "action_dim": int(config.action_dim),
            "value_shape": "",
        }

        self.rgb_numel = int(config.rgb_shape[0] * config.rgb_shape[1] * config.rgb_shape[2])
        self.lidar_numel = int(config.lidar_shape[0] * config.lidar_shape[1] * config.lidar_shape[2])
        self.ego_numel = int(config.ego_dim)
        self.route_numel = int(config.route_dim)

        expected_dim = self.rgb_numel + self.lidar_numel + self.ego_numel + self.route_numel
        if expected_dim != int(config.obs_dim):
            raise ValueError(f"EarlyFusionConfig.obs_dim mismatch: expected {expected_dim}, got {config.obs_dim}")

        self.rgb_encoder = ImageEncoder(config.rgb_shape[0], config.rgb_feature_dim, (16, 32, 64))
        self.lidar_encoder = ImageEncoder(config.lidar_shape[0], config.lidar_feature_dim, (8, 16, 32))
        self.ego_encoder = VectorEncoder(config.ego_dim, config.ego_feature_dim)
        self.route_encoder = VectorEncoder(config.route_dim, config.route_feature_dim)

        self.concat_dim = (
            config.rgb_feature_dim
            + config.lidar_feature_dim
            + config.ego_feature_dim
            + config.route_feature_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.concat_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.critic_head = nn.Linear(config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def _split_modalities(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb_end = self.rgb_numel
        lidar_end = rgb_end + self.lidar_numel
        ego_end = lidar_end + self.ego_numel
        route_end = ego_end + self.route_numel

        rgb = obs[..., :rgb_end].reshape(-1, *self.config.rgb_shape)
        lidar = obs[..., rgb_end:lidar_end].reshape(-1, *self.config.lidar_shape)
        ego = obs[..., lidar_end:ego_end].reshape(-1, self.config.ego_dim)
        route = obs[..., ego_end:route_end].reshape(-1, self.config.route_dim)
        return {"rgb": rgb, "lidar": lidar, "ego": ego, "route": route}

    def encode_modalities(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        modal_batch = (
            self._split_modalities(obs)
            if isinstance(obs, torch.Tensor)
            else prepare_modal_batch(
                observation=obs,
                rgb_shape=self.config.rgb_shape,
                lidar_shape=self.config.lidar_shape,
                ego_dim=self.config.ego_dim,
                route_dim=self.config.route_dim,
                device=self.actor_head.weight.device,
            )
        )
        return {
            "rgb": self.rgb_encoder(modal_batch["rgb"]),
            "lidar": self.lidar_encoder(modal_batch["lidar"]),
            "ego": self.ego_encoder(modal_batch["ego"]),
            "route": self.route_encoder(modal_batch["route"]),
        }

    def fuse_features(self, feature_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(
            [feature_dict["rgb"], feature_dict["lidar"], feature_dict["ego"], feature_dict["route"]],
            dim=-1,
        )
        if fused.shape[-1] != self.concat_dim:
            raise ValueError(f"Expected fused concat dim {self.concat_dim}, got {fused.shape[-1]}")
        return self.fusion(fused)

    def forward_with_debug(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        feature_dict = self.encode_modalities(obs)
        fused_feature = self.fuse_features(feature_dict)
        policy_latent = self.actor_head(fused_feature)

        steer_mean = torch.tanh(policy_latent[..., 0:1])
        throttle_mean = torch.sigmoid(policy_latent[..., 1:2])
        brake_mean = torch.sigmoid(policy_latent[..., 2:3])
        action_mean = torch.cat([steer_mean, throttle_mean, brake_mean], dim=-1)

        value = self.critic_head(fused_feature).squeeze(-1)
        debug_info = {
            "encoder_feature_dims": {key: int(value_tensor.shape[-1]) for key, value_tensor in feature_dict.items()},
            "fused_dim": int(fused_feature.shape[-1]),
            "obs_keys_used": ["rgb", "lidar", "ego", "route"],
            "action_dim": int(action_mean.shape[-1]),
            "value_shape": tuple(value.shape),
        }
        self._last_debug_info = {
            "encoder_feature_dims": ",".join(f"{key}:{dim}" for key, dim in debug_info["encoder_feature_dims"].items()),
            "fused_dim": int(debug_info["fused_dim"]),
            "obs_keys_used": ",".join(debug_info["obs_keys_used"]),
            "action_dim": int(debug_info["action_dim"]),
            "value_shape": "x".join(str(part) for part in debug_info["value_shape"]),
        }
        return action_mean, value, debug_info

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        action_mean, value, _ = self.forward_with_debug(obs)
        return action_mean, value

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp_min(self.config.min_std)

    def get_debug_info(self) -> dict[str, object]:
        return dict(self._last_debug_info)
