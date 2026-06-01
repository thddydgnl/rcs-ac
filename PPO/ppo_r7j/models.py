from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from rcs_ac_v1_1.models.bev_policy import BEVSourceEncoder, RGBContextEncoder, VectorEncoder, parameterize_waypoints
from rcs_ac_v1_1.runtime.config import V11RuntimeConfig
from rcs_ac_v1_1.runtime.observation import V11PolicyObservationSpec, split_flat_observation


@dataclass(frozen=True)
class PPOMatchedR7JPolicyConfig:
    feature_dim: int = 128
    hidden_dim: int = 256
    log_std_init: float = -1.5
    min_log_std: float = -4.0
    max_log_std: float = 0.0


class PPOMatchedR7JPolicy(nn.Module):
    """PPO policy with the r7j observation and action contract.

    This policy deliberately avoids the RCS-AC reliability/fusion/Shapley/risk
    modules. It encodes the same r7j observation tensors, concatenates source
    features, emits raw waypoint and target-speed parameters, and relies on the
    shared r7j pure-pursuit controller for CARLA controls.
    """

    def __init__(
        self,
        runtime_config: V11RuntimeConfig,
        config: PPOMatchedR7JPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        self.runtime_config = runtime_config
        self.policy_config = runtime_config.policy_config
        self.config = config or PPOMatchedR7JPolicyConfig(feature_dim=int(self.policy_config.feature_dim))
        self.obs_spec: V11PolicyObservationSpec = runtime_config.observation_spec
        feature_dim = int(self.config.feature_dim)
        hidden_dim = int(self.config.hidden_dim)

        self.rgb_context_encoder = RGBContextEncoder(self.policy_config.rgb_shape[0], feature_dim)
        self.lidar_bev_encoder = BEVSourceEncoder(self.policy_config.bev_channels.lidar_channels, feature_dim)
        self.route_bev_encoder = BEVSourceEncoder(self.policy_config.bev_channels.route_channels, feature_dim)
        self.ego_bev_encoder = BEVSourceEncoder(self.policy_config.bev_channels.ego_channels, feature_dim)
        self.ego_scalar_encoder = VectorEncoder(self.policy_config.ego_scalar_dim, feature_dim)
        self.route_scalar_encoder = VectorEncoder(self.policy_config.route_scalar_dim, feature_dim)

        body_input_dim = 6 * feature_dim
        self.policy_body = nn.Sequential(
            nn.Linear(body_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        action_dim = int(self.policy_config.waypoint_count) * 2 + 1
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(self.config.log_std_init)))

    @property
    def action_dim(self) -> int:
        return int(self.policy_config.waypoint_count) * 2 + 1

    def forward(self, flat_observation: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, source_weights = self._encode(flat_observation)
        raw_action_mean = self.action_mean(hidden)
        outputs = self._decode_action(raw_action_mean)
        outputs.update(
            {
                "raw_action_mean": raw_action_mean,
                "value": self.value_head(hidden),
                "bev_fusion_weights": source_weights,
            }
        )
        return outputs

    def act(self, flat_observation: torch.Tensor, *, deterministic: bool = False) -> dict[str, torch.Tensor]:
        hidden, source_weights = self._encode(flat_observation)
        mean = self.action_mean(hidden)
        std = self._std(mean)
        dist = Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        outputs = self._decode_action(raw_action)
        outputs.update(
            {
                "raw_action": raw_action,
                "raw_action_mean": mean,
                "log_prob": log_prob,
                "entropy": entropy,
                "value": self.value_head(hidden),
                "bev_fusion_weights": source_weights,
            }
        )
        return outputs

    def evaluate_raw_action(
        self,
        flat_observation: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden, source_weights = self._encode(flat_observation)
        mean = self.action_mean(hidden)
        std = self._std(mean)
        dist = Normal(mean, std)
        outputs = self._decode_action(raw_action)
        outputs.update(
            {
                "raw_action_mean": mean,
                "log_prob": dist.log_prob(raw_action).sum(dim=-1, keepdim=True),
                "entropy": dist.entropy().sum(dim=-1, keepdim=True),
                "value": self.value_head(hidden),
                "bev_fusion_weights": source_weights,
            }
        )
        return outputs

    def _encode(self, flat_observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        split = split_flat_observation(flat_observation, self.obs_spec)
        encoded = {
            "rgb": self.rgb_context_encoder(split["rgb"]),
            "lidar_bev": self.lidar_bev_encoder(split["lidar_bev"]),
            "route_bev": self.route_bev_encoder(split["route_bev"]),
            "ego_bev": self.ego_bev_encoder(split["ego_bev"]),
            "ego_scalar": self.ego_scalar_encoder(split["ego_scalar"]),
            "route_scalar": self.route_scalar_encoder(split["route_scalar"]),
        }
        hidden = self.policy_body(
            torch.cat(
                [
                    encoded["rgb"],
                    encoded["lidar_bev"],
                    encoded["route_bev"],
                    encoded["ego_bev"],
                    encoded["ego_scalar"],
                    encoded["route_scalar"],
                ],
                dim=-1,
            )
        )
        source_weights = _availability_weights(split["sensor_health"])
        return hidden, source_weights

    def _decode_action(self, raw_action: torch.Tensor) -> dict[str, torch.Tensor]:
        waypoint_raw_dim = int(self.policy_config.waypoint_count) * 2
        waypoints = parameterize_waypoints(
            raw_action[:, :waypoint_raw_dim],
            distances_m=self.policy_config.waypoint_distances_m,
            gap_residual_scale_m=float(self.policy_config.waypoint_gap_residual_scale_m),
            min_gap_m=float(self.policy_config.waypoint_min_gap_m),
            max_lateral_m=float(self.policy_config.waypoint_max_lateral_m),
        )
        speed_unit = torch.sigmoid(raw_action[:, waypoint_raw_dim : waypoint_raw_dim + 1])
        target_speed = speed_unit * (
            float(self.policy_config.max_target_speed_kmh) - float(self.policy_config.min_target_speed_kmh)
        ) + float(self.policy_config.min_target_speed_kmh)
        return {"waypoints": waypoints, "target_speed_kmh": target_speed}

    def _std(self, mean: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, min=float(self.config.min_log_std), max=float(self.config.max_log_std))
        return torch.exp(log_std).reshape(1, -1).expand_as(mean)


def _availability_weights(sensor_health: torch.Tensor) -> torch.Tensor:
    if sensor_health.ndim != 2 or int(sensor_health.shape[-1]) < 12:
        return torch.full((int(sensor_health.shape[0]), 4), 0.25, dtype=sensor_health.dtype, device=sensor_health.device)
    availability = torch.stack(
        (
            sensor_health[:, 0],
            sensor_health[:, 3],
            sensor_health[:, 6],
            torch.maximum(sensor_health[:, 9], sensor_health[:, 10]),
        ),
        dim=-1,
    ).to(dtype=sensor_health.dtype)
    denom = availability.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return availability / denom
