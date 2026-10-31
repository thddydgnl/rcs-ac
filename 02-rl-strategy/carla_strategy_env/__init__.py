"""Public package exports for the CARLA F1 strategy MVP project."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CarlaF1StrategyEnv",
    "F1StrategyModel",
    "StrategyAction",
    "StrategyState",
]


def __getattr__(name: str) -> Any:
    if name == "CarlaF1StrategyEnv":
        return import_module("carla_strategy_env.envs.f1_strategy_env").CarlaF1StrategyEnv
    if name in {"F1StrategyModel", "StrategyAction", "StrategyState"}:
        module = import_module("carla_strategy_env.strategy.models")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
