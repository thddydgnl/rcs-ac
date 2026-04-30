from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from rcs_ac_first.runtime.observation import (
    DEFAULT_EGO_DIM,
    DEFAULT_LIDAR_SHAPE,
    DEFAULT_RGB_SHAPE,
    DEFAULT_ROUTE_DIM,
    build_policy_observation_spec,
)
from rcs_ac_first.runtime.auxiliary import (
    DEFAULT_AUXILIARY_DIM,
    DEFAULT_QUALITY_STATS_DIM,
    DEFAULT_RISK_STATS_DIM,
    DEFAULT_ROUTE_COMMITMENT_STATS_DIM,
    DEFAULT_SENSOR_DROPOUT_DIM,
)


def _tuple_ints(value: tuple[int, ...] | list[int], *, name: str, expected_len: int) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if len(result) != expected_len or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain {expected_len} positive ints, got {value!r}")
    return result


def _tuple_floats(value: tuple[float, ...] | list[float], *, name: str, expected_len: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {value!r}")
    return result


@dataclass
class RCSACFirstModelConfig:
    obs_dim: int | None = None
    action_dim: int = 2
    rgb_shape: tuple[int, int, int] | list[int] = DEFAULT_RGB_SHAPE
    lidar_shape: tuple[int, int, int] | list[int] = DEFAULT_LIDAR_SHAPE
    ego_dim: int = DEFAULT_EGO_DIM
    route_dim: int = DEFAULT_ROUTE_DIM
    feature_dim: int = 128
    fusion_dim: int = 128
    recurrent_hidden_dim: int = 256
    recurrent_layers: int = 1
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    aux_hidden_dim: int = 128
    enable_rgb_perception_aux: bool = True
    aux_mask_height: int = 45
    aux_mask_width: int = 80
    aux_perception_hidden_channels: int = 64
    use_rgb_perception_actor_features: bool = False
    rgb_perception_actor_feature_dim: int = 16
    auxiliary_dim: int = DEFAULT_AUXILIARY_DIM
    quality_stats_dim: int = DEFAULT_QUALITY_STATS_DIM
    risk_stats_dim: int = DEFAULT_RISK_STATS_DIM
    route_commitment_stats_dim: int = DEFAULT_ROUTE_COMMITMENT_STATS_DIM
    sensor_dropout_dim: int = DEFAULT_SENSOR_DROPOUT_DIM
    reliability_hidden_dim: int = 64
    risk_hidden_dim: int = 64
    risk_context_dim: int = 64
    reliability_min: float = 0.02
    reliability_max: float = 1.0
    log_std_init_by_dim: tuple[float, ...] | list[float] = (-0.2, -0.2)
    min_std_by_dim: tuple[float, ...] | list[float] = (0.05, 0.05)
    max_std_by_dim: tuple[float, ...] | list[float] = (0.8, 0.8)
    enable_reliability: bool = False
    enable_risk_context: bool = False
    enable_shapley: bool = False
    enable_dropout_consistency: bool = False

    def __post_init__(self) -> None:
        self.action_dim = int(self.action_dim)
        if self.action_dim != 2:
            raise ValueError(f"Phase 1 requires action_dim=2, got {self.action_dim}")
        self.rgb_shape = _tuple_ints(self.rgb_shape, name="rgb_shape", expected_len=3)
        self.lidar_shape = _tuple_ints(self.lidar_shape, name="lidar_shape", expected_len=3)
        self.ego_dim = int(self.ego_dim)
        self.route_dim = int(self.route_dim)
        self.feature_dim = int(self.feature_dim)
        self.fusion_dim = int(self.fusion_dim)
        self.recurrent_hidden_dim = int(self.recurrent_hidden_dim)
        self.recurrent_layers = int(self.recurrent_layers)
        self.actor_hidden_dim = int(self.actor_hidden_dim)
        self.critic_hidden_dim = int(self.critic_hidden_dim)
        self.aux_hidden_dim = int(self.aux_hidden_dim)
        self.enable_rgb_perception_aux = bool(self.enable_rgb_perception_aux)
        self.aux_mask_height = int(self.aux_mask_height)
        self.aux_mask_width = int(self.aux_mask_width)
        self.aux_perception_hidden_channels = int(self.aux_perception_hidden_channels)
        self.use_rgb_perception_actor_features = bool(self.use_rgb_perception_actor_features)
        self.rgb_perception_actor_feature_dim = int(self.rgb_perception_actor_feature_dim)
        self.auxiliary_dim = int(self.auxiliary_dim)
        self.quality_stats_dim = int(self.quality_stats_dim)
        self.risk_stats_dim = int(self.risk_stats_dim)
        self.route_commitment_stats_dim = int(self.route_commitment_stats_dim)
        self.sensor_dropout_dim = int(self.sensor_dropout_dim)
        self.reliability_hidden_dim = int(self.reliability_hidden_dim)
        self.risk_hidden_dim = int(self.risk_hidden_dim)
        self.risk_context_dim = int(self.risk_context_dim)
        self.reliability_min = float(self.reliability_min)
        self.reliability_max = float(self.reliability_max)
        for name in (
            "ego_dim",
            "route_dim",
            "feature_dim",
            "fusion_dim",
            "recurrent_hidden_dim",
            "recurrent_layers",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "aux_hidden_dim",
            "aux_mask_height",
            "aux_mask_width",
            "aux_perception_hidden_channels",
            "rgb_perception_actor_feature_dim",
            "auxiliary_dim",
            "quality_stats_dim",
            "risk_stats_dim",
            "route_commitment_stats_dim",
            "sensor_dropout_dim",
            "reliability_hidden_dim",
            "risk_hidden_dim",
            "risk_context_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        expected_auxiliary_dim = (
            self.quality_stats_dim
            + self.risk_stats_dim
            + self.route_commitment_stats_dim
            + self.sensor_dropout_dim
        )
        if self.auxiliary_dim != expected_auxiliary_dim:
            raise ValueError(
                f"auxiliary_dim mismatch: expected {expected_auxiliary_dim}, got {self.auxiliary_dim}"
            )
        if not 0.0 < self.reliability_min <= self.reliability_max <= 1.0:
            raise ValueError("reliability_min/max must satisfy 0 < min <= max <= 1")
        self.log_std_init_by_dim = _tuple_floats(
            self.log_std_init_by_dim,
            name="log_std_init_by_dim",
            expected_len=self.action_dim,
        )
        self.min_std_by_dim = _tuple_floats(
            self.min_std_by_dim,
            name="min_std_by_dim",
            expected_len=self.action_dim,
        )
        self.max_std_by_dim = _tuple_floats(
            self.max_std_by_dim,
            name="max_std_by_dim",
            expected_len=self.action_dim,
        )
        if any(min_v <= 0.0 for min_v in self.min_std_by_dim):
            raise ValueError("min_std_by_dim entries must be > 0")
        for min_v, max_v in zip(self.min_std_by_dim, self.max_std_by_dim):
            if max_v < min_v:
                raise ValueError("max_std_by_dim entries must be >= min_std_by_dim")

        spec = build_policy_observation_spec(
            rgb_shape=self.rgb_shape,
            lidar_shape=self.lidar_shape,
            ego_dim=self.ego_dim,
            route_dim=self.route_dim,
        )
        if self.obs_dim is None or int(self.obs_dim) <= 0:
            self.obs_dim = int(spec.total_dim)
        if int(self.obs_dim) != int(spec.total_dim):
            raise ValueError(f"obs_dim mismatch: expected {spec.total_dim}, got {self.obs_dim}")
        self.obs_dim = int(self.obs_dim)


@dataclass
class RCSACFirstFusionConfig:
    feature_dim: int = 128
    fusion_dim: int = 128
    hidden_dim: int = 128
    temperature: float = 1.0
    route_weight_floor: float = 0.08
    ego_weight_floor: float = 0.05
    contribution_bias_scale: float = 0.0
    risk_context_dim: int = 64
    reliability_logit_scale: float = 1.0
    risk_logit_scale: float = 0.5
    use_risk_context: bool = False

    def __post_init__(self) -> None:
        self.feature_dim = int(self.feature_dim)
        self.fusion_dim = int(self.fusion_dim)
        self.hidden_dim = int(self.hidden_dim)
        self.temperature = float(self.temperature)
        self.route_weight_floor = float(self.route_weight_floor)
        self.ego_weight_floor = float(self.ego_weight_floor)
        self.contribution_bias_scale = float(self.contribution_bias_scale)
        self.risk_context_dim = int(self.risk_context_dim)
        self.reliability_logit_scale = float(self.reliability_logit_scale)
        self.risk_logit_scale = float(self.risk_logit_scale)
        self.use_risk_context = bool(self.use_risk_context)
        if self.feature_dim <= 0 or self.fusion_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("feature_dim, fusion_dim, and hidden_dim must be > 0")
        if self.risk_context_dim <= 0:
            raise ValueError("risk_context_dim must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("fusion temperature must be > 0")
        if self.route_weight_floor < 0.0 or self.ego_weight_floor < 0.0:
            raise ValueError("fusion weight floors must be >= 0")
        if self.route_weight_floor + self.ego_weight_floor >= 1.0:
            raise ValueError("route_weight_floor + ego_weight_floor must be < 1")


@dataclass
class RCSACFirstLossConfig:
    value_loss_weight: float = 0.5
    entropy_loss_weight: float = 0.01
    aux_lane_offset_loss_weight: float = 1.0
    aux_heading_error_loss_weight: float = 1.0
    aux_road_mask_loss_weight: float = 0.25
    aux_lane_edge_mask_loss_weight: float = 1.0
    aux_lane_edge_pos_weight: float = 3.0
    reliability_loss_weight: float = 0.0
    risk_supervision_loss_weight: float = 0.0
    risk_regularization_weight: float = 0.0
    shapley_alignment_loss_weight: float = 0.0
    dropout_consistency_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, float(getattr(self, name)))
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class RCSACFirstShapleyConfig:
    enabled: bool = False
    estimator: str = "approximate"
    value_source: str = "current_critic"
    subsets_per_sample: int = 4
    batch_fraction: float = 0.25
    every_updates: int = 1
    contribution_temperature: float = 1.0
    include_empty_subset: bool = True
    include_full_subset: bool = True
    normalize: str = "softmax"
    detach_contribution_target: bool = True
    min_contribution: float = 0.02

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.estimator = str(self.estimator)
        self.value_source = str(self.value_source)
        self.subsets_per_sample = int(self.subsets_per_sample)
        self.batch_fraction = float(self.batch_fraction)
        self.every_updates = int(self.every_updates)
        self.contribution_temperature = float(self.contribution_temperature)
        self.include_empty_subset = bool(self.include_empty_subset)
        self.include_full_subset = bool(self.include_full_subset)
        self.normalize = str(self.normalize)
        self.detach_contribution_target = bool(self.detach_contribution_target)
        self.min_contribution = float(self.min_contribution)
        if self.estimator not in {"approximate", "exact"}:
            raise ValueError("shapley.estimator must be 'approximate' or 'exact'")
        if self.value_source != "current_critic":
            raise ValueError("Phase 3 v1 supports only shapley.value_source='current_critic'")
        if self.subsets_per_sample <= 0:
            raise ValueError("shapley.subsets_per_sample must be > 0")
        if not 0.0 < self.batch_fraction <= 1.0:
            raise ValueError("shapley.batch_fraction must be in (0, 1]")
        if self.every_updates <= 0:
            raise ValueError("shapley.every_updates must be > 0")
        if self.contribution_temperature <= 0.0:
            raise ValueError("shapley.contribution_temperature must be > 0")
        if self.normalize != "softmax":
            raise ValueError("Phase 3 v1 supports only shapley.normalize='softmax'")
        if not 0.0 <= self.min_contribution < 0.25:
            raise ValueError("shapley.min_contribution must be in [0, 0.25)")


@dataclass
class RCSACFirstDropoutConsistencyConfig:
    enabled: bool = False
    mode: str = "representation_mask"
    batch_fraction: float = 0.25
    max_masks_per_sample: int = 1
    action_mean_weight: float = 1.0
    value_weight: float = 0.25
    reliability_floor: float = 0.05
    detach_clean_policy: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.mode = str(self.mode)
        self.batch_fraction = float(self.batch_fraction)
        self.max_masks_per_sample = int(self.max_masks_per_sample)
        self.action_mean_weight = float(self.action_mean_weight)
        self.value_weight = float(self.value_weight)
        self.reliability_floor = float(self.reliability_floor)
        self.detach_clean_policy = bool(self.detach_clean_policy)
        if self.mode != "representation_mask":
            raise ValueError("Phase 3 v1 supports only dropout_consistency.mode='representation_mask'")
        if not 0.0 < self.batch_fraction <= 1.0:
            raise ValueError("dropout_consistency.batch_fraction must be in (0, 1]")
        if self.max_masks_per_sample <= 0:
            raise ValueError("dropout_consistency.max_masks_per_sample must be > 0")
        for name in ("action_mean_weight", "value_weight", "reliability_floor"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"dropout_consistency.{name} must be >= 0")


@dataclass
class RCSACFirstTrainConfig:
    seed: int = 7
    model_type: str = "rcs_ac_first"
    model_variant: str = "phase1_core"
    training_stage: str = "phase1_core_smoke"
    run_name: str = "rcs_ac_first_phase1_core_smoke"
    total_timesteps: int = 2048
    rollout_steps: int = 128
    max_episode_steps: int = 384
    device: str = "auto"
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 1
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    save_every_updates: int = 1
    host: str = "127.0.0.1"
    port: int = 2000
    town: str = "Town04"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"
    output_root: str = "outputs/phase1_core_env"

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self.total_timesteps = int(self.total_timesteps)
        self.rollout_steps = int(self.rollout_steps)
        self.max_episode_steps = int(self.max_episode_steps)
        self.learning_rate = float(self.learning_rate)
        self.gamma = float(self.gamma)
        self.gae_lambda = float(self.gae_lambda)
        self.clip_coef = float(self.clip_coef)
        self.update_epochs = int(self.update_epochs)
        self.minibatch_size = int(self.minibatch_size)
        self.max_grad_norm = float(self.max_grad_norm)
        self.save_every_updates = int(self.save_every_updates)
        self.port = int(self.port)
        for name in ("total_timesteps", "rollout_steps", "max_episode_steps", "update_epochs", "minibatch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of auto, cpu, cuda")

    def resolve_device(self, override: str | None = None) -> str:
        requested = override or self.device
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested


@dataclass
class Phase1CoreConfig:
    model: RCSACFirstModelConfig = field(default_factory=RCSACFirstModelConfig)
    fusion: RCSACFirstFusionConfig = field(default_factory=RCSACFirstFusionConfig)
    loss: RCSACFirstLossConfig = field(default_factory=RCSACFirstLossConfig)
    shapley: RCSACFirstShapleyConfig = field(default_factory=RCSACFirstShapleyConfig)
    dropout_consistency: RCSACFirstDropoutConsistencyConfig = field(default_factory=RCSACFirstDropoutConsistencyConfig)
    train: RCSACFirstTrainConfig = field(default_factory=RCSACFirstTrainConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "fusion": asdict(self.fusion),
            "loss": asdict(self.loss),
            "shapley": asdict(self.shapley),
            "dropout_consistency": asdict(self.dropout_consistency),
            "train": asdict(self.train),
            "raw": dict(self.raw),
        }


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value)!r}")
    return dict(value)


def load_phase1_config(config_path: str | Path) -> Phase1CoreConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Phase 1 config must be a mapping, got {type(data)!r}")

    model_raw = _mapping(data.get("model"), name="model")
    fusion_raw = _mapping(data.get("fusion"), name="fusion")
    loss_raw = _mapping(data.get("loss"), name="loss")
    shapley_raw = _mapping(data.get("shapley"), name="shapley")
    dropout_raw = _mapping(data.get("dropout_consistency"), name="dropout_consistency")
    ppo_raw = _mapping(data.get("ppo"), name="ppo")

    train_keys = {
        "seed",
        "model_type",
        "model_variant",
        "training_stage",
        "run_name",
        "total_timesteps",
        "rollout_steps",
        "max_episode_steps",
        "device",
        "save_every_updates",
        "host",
        "port",
        "town",
        "logs_dir",
        "checkpoints_dir",
        "output_root",
    }
    train_raw = {key: data[key] for key in train_keys if key in data}
    for key in ("learning_rate", "gamma", "gae_lambda", "clip_coef", "update_epochs", "minibatch_size", "max_grad_norm"):
        if key in ppo_raw:
            train_raw[key] = ppo_raw[key]

    model = RCSACFirstModelConfig(**model_raw)
    fusion_defaults = {
        "feature_dim": model.feature_dim,
        "fusion_dim": model.fusion_dim,
    }
    fusion = RCSACFirstFusionConfig(**{**fusion_defaults, **fusion_raw})
    loss = RCSACFirstLossConfig(**loss_raw)
    shapley_defaults = {"enabled": bool(model.enable_shapley)}
    dropout_defaults = {"enabled": bool(model.enable_dropout_consistency)}
    shapley = RCSACFirstShapleyConfig(**{**shapley_defaults, **shapley_raw})
    dropout_consistency = RCSACFirstDropoutConsistencyConfig(**{**dropout_defaults, **dropout_raw})
    train = RCSACFirstTrainConfig(**train_raw)
    return Phase1CoreConfig(
        model=model,
        fusion=fusion,
        loss=loss,
        shapley=shapley,
        dropout_consistency=dropout_consistency,
        train=train,
        raw=dict(data),
    )
