"""Unit tests for the virtual strategy model without requiring a CARLA runtime."""

from config import RaceConfig
from carla_strategy_env.strategy.models import F1StrategyModel, StrategyAction, TireCompound


def test_pit_stop_switches_compound_and_resets_wear() -> None:
    model = F1StrategyModel(RaceConfig())
    model.state.tire_wear = 0.8

    pit_loss = model.apply_strategy_action(StrategyAction.PIT_HARD)

    assert pit_loss == model.config.pit_stop_loss_seconds
    assert model.state.tire_compound == TireCompound.HARD
    assert model.state.tire_wear == 0.0
    assert model.state.pit_count == 1


def test_lap_completion_burns_fuel_and_increases_wear() -> None:
    model = F1StrategyModel(RaceConfig(total_laps=3), seed=1)
    initial_fuel = model.state.fuel_kg

    result = model.advance(1.0, StrategyAction.STAY_OUT)

    assert result.lap_completed is True
    assert model.state.fuel_kg < initial_fuel
    assert model.state.tire_wear > 0.0
    assert model.state.current_lap == 2
