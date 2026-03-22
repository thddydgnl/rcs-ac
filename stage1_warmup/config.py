"""Config helpers for Stage 1 warm-up."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    corruption_mode: str = "clean"


@dataclass
class ModelConfig:
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    encoder_output_dim: int = 64
    attention_hidden_dim: int = 64
    recurrent_hidden_dim: int = 64
    attention_temperature: float = 1.5
    use_feature_layer_norm: bool = True
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])
    reliability_hidden_dims: list[int] = field(default_factory=lambda: [32])
    reliability_input_mode: str = "own"
    action_log_std_init: float = -0.5


@dataclass
class PPOConfig:
    total_steps: int = 30000
    rollout_steps: int = 1024
    sequence_length: int = 16
    num_eval_episodes: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    minibatch_num_sequences: int = 8
    learning_rate: float = 1e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.3


@dataclass
class WarmupConfig:
    reliability_loss_coef: float = 0.05
    weak_corruption_probability: float = 0.0
    weak_corruption_scale: float = 0.03
    mild_corruption_probability: float = 0.0
    mild_corruption_scale: float = 0.06
    use_aux_quality_labels: bool = True
    use_direct_corruption_targets: bool = False
    direct_target_drop_scale: float = 3.0
    init_checkpoint_path: str | None = None
    calibration_objective: str = "mse"
    target_margin_loss_coef: float = 0.0
    target_ranking_loss_coef: float = 0.0
    monotonicity_loss_coef: float = 0.0
    directional_monotonicity_loss_coef: float = 0.0
    vision_target_positive_loss_coef: float = 0.0
    ego_target_positive_loss_coef: float = 0.0
    vision_logit_positive_loss_coef: float = 0.0
    ego_logit_positive_loss_coef: float = 0.0
    target_margin_value: float = 0.05
    target_ranking_margin: float = 0.03
    monotonicity_margin: float = 0.04
    monotonicity_margin_clean_weak: float = 0.0
    monotonicity_margin_weak_mild: float = 0.0
    target_positive_margin_weak: float = 0.0
    target_positive_margin_mild: float = 0.0
    logit_positive_margin_weak: float = 0.0
    logit_positive_margin_mild: float = 0.0
    freeze_policy_backbone: bool = False
    train_reliability_head_only: bool = False
    train_reliability_projection: bool = True
    use_paired_calibration_batches: bool = False
    logit_triplet_loss_coef: float = 0.0
    weak_margin_vision: float = 0.0
    mild_margin_vision: float = 0.0
    weak_margin_ego: float = 0.0
    mild_margin_ego: float = 0.0
    calibration_only_vision_weak_strength: float = 0.03
    calibration_only_vision_mild_strength: float = 0.06
    calibration_only_ego_weak_strength: float = 0.025
    calibration_only_ego_mild_strength: float = 0.05
    vision_ego_margin_multiplier: float = 1.0
    vision_margin_multiplier: float = 1.0
    ego_margin_multiplier: float = 1.0
    vision_ego_weight_multiplier: float = 1.0
    vision_weight_multiplier: float = 1.0
    ego_weight_multiplier: float = 1.0
    focus_vision_ego_pairing: bool = False


@dataclass
class LoggingConfig:
    run_name: str = "stage1_warmup"
    output_dir: str = "./runs/stage1_warmup"
    eval_interval_updates: int = 2
    save_interval_updates: int = 10


@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dict(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_cfg = yaml.safe_load(file) or {}

    defaults = {
        "seed": ExperimentConfig.seed,
        "device": ExperimentConfig.device,
        "env": EnvConfig().__dict__,
        "model": ModelConfig().__dict__,
        "ppo": PPOConfig().__dict__,
        "warmup": WarmupConfig().__dict__,
        "logging": LoggingConfig().__dict__,
    }
    merged = _merge_dict(defaults, raw_cfg)
    return ExperimentConfig(
        seed=merged["seed"],
        device=merged["device"],
        env=EnvConfig(**merged["env"]),
        model=ModelConfig(**merged["model"]),
        ppo=PPOConfig(**merged["ppo"]),
        warmup=WarmupConfig(**merged["warmup"]),
        logging=LoggingConfig(**merged["logging"]),
    )
