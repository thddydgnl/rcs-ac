"""Config loading utilities for PPO baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    max_episode_steps: int = 200
    corruption_mode: str = "clean"


@dataclass
class ModelConfig:
    baseline_type: str = "baseline_a"
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    encoder_output_dim: int = 64
    trunk_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])
    attention_hidden_dim: int = 64
    action_log_std_init: float = -0.5


@dataclass
class PPOConfig:
    total_steps: int = 20000
    rollout_steps: int = 1024
    num_eval_episodes: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None


@dataclass
class LoggingConfig:
    run_name: str = "debug_run"
    output_dir: str = "./runs/debug_run"
    log_interval_updates: int = 1
    eval_interval_updates: int = 5
    save_interval_updates: int = 10


@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
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
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_cfg = yaml.safe_load(file) or {}

    defaults = {
        "seed": ExperimentConfig.seed,
        "device": ExperimentConfig.device,
        "env": EnvConfig().__dict__,
        "model": ModelConfig().__dict__,
        "ppo": PPOConfig().__dict__,
        "logging": LoggingConfig().__dict__,
    }
    merged = _merge_dict(defaults, raw_cfg)

    return ExperimentConfig(
        seed=merged["seed"],
        device=merged["device"],
        env=EnvConfig(**merged["env"]),
        model=ModelConfig(**merged["model"]),
        ppo=PPOConfig(**merged["ppo"]),
        logging=LoggingConfig(**merged["logging"]),
    )

