"""Virtual race strategy model for tires, fuel, pit loss, and reward-friendly state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
import logging
import math
import random

from config import RaceConfig


LOGGER = logging.getLogger(__name__)


class TireCompound(StrEnum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"


class StrategyAction(IntEnum):
    STAY_OUT = 0
    PIT_SOFT = 1
    PIT_MEDIUM = 2
    PIT_HARD = 3


@dataclass(slots=True)
class StrategyState:
    current_lap: int
    laps_remaining: int
    tire_compound: TireCompound
    tire_wear: float
    fuel_kg: float
    last_lap_time_seconds: float
    best_lap_time_seconds: float
    total_elapsed_seconds: float
    pit_count: int
    weather_grip: float
    progress: float
    just_pitted: bool = False


@dataclass(slots=True)
class StrategyStepResult:
    state: StrategyState
    lap_completed: bool
    lap_time_delta_seconds: float
    pit_stop_seconds: float
    race_finished: bool


class F1StrategyModel:
    """Keeps virtual race state decoupled from low-level CARLA vehicle control."""

    _WEAR_RATE = {
        TireCompound.SOFT: 0.15,
        TireCompound.MEDIUM: 0.10,
        TireCompound.HARD: 0.07,
    }

    _BASE_COMPOUND_OFFSET = {
        TireCompound.SOFT: -1.4,
        TireCompound.MEDIUM: 0.0,
        TireCompound.HARD: 1.2,
    }

    def __init__(self, config: RaceConfig, seed: int = 7) -> None:
        self.config = config
        self._random = random.Random(seed)
        self._state = self._initial_state()

    @property
    def state(self) -> StrategyState:
        return self._state

    def reset(self) -> StrategyState:
        self._state = self._initial_state()
        LOGGER.info("Strategy model reset: %s", self._state)
        return self._state

    def _initial_state(self) -> StrategyState:
        base = self.config.base_lap_time_seconds
        return StrategyState(
            current_lap=1,
            laps_remaining=self.config.total_laps,
            tire_compound=TireCompound.MEDIUM,
            tire_wear=0.0,
            fuel_kg=self.config.initial_fuel_kg,
            last_lap_time_seconds=base,
            best_lap_time_seconds=base,
            total_elapsed_seconds=0.0,
            pit_count=0,
            weather_grip=self.config.weather_grip_baseline,
            progress=0.0,
            just_pitted=False,
        )

    def observe_vector(self) -> list[float]:
        compound_flags = {
            TireCompound.SOFT: [1.0, 0.0, 0.0],
            TireCompound.MEDIUM: [0.0, 1.0, 0.0],
            TireCompound.HARD: [0.0, 0.0, 1.0],
        }[self._state.tire_compound]
        return [
            float(self._state.current_lap),
            float(self._state.laps_remaining),
            self._state.tire_wear,
            self._state.fuel_kg,
            self._state.last_lap_time_seconds,
            self._state.best_lap_time_seconds,
            float(self._state.pit_count),
            self._state.weather_grip,
            self._state.progress,
            *compound_flags,
        ]

    def apply_strategy_action(self, action: StrategyAction | int) -> float:
        action = StrategyAction(action)
        self._state.just_pitted = False
        if action == StrategyAction.STAY_OUT:
            return 0.0

        next_compound = {
            StrategyAction.PIT_SOFT: TireCompound.SOFT,
            StrategyAction.PIT_MEDIUM: TireCompound.MEDIUM,
            StrategyAction.PIT_HARD: TireCompound.HARD,
        }[action]
        self._state.tire_compound = next_compound
        self._state.tire_wear = 0.0
        self._state.pit_count += 1
        self._state.just_pitted = True
        LOGGER.info("Pit stop executed: compound=%s count=%s", next_compound, self._state.pit_count)
        return self.config.pit_stop_loss_seconds

    def advance(self, progress_delta: float, action: StrategyAction | int) -> StrategyStepResult:
        pit_stop_seconds = self.apply_strategy_action(action)
        previous_lap_time = self._state.last_lap_time_seconds
        new_progress = min(1.0, max(0.0, self._state.progress + progress_delta))
        lap_completed = math.isclose(new_progress, 1.0, abs_tol=1e-6) or new_progress >= 1.0
        self._state.progress = 0.0 if lap_completed else new_progress

        if lap_completed:
            lap_time = self._estimate_lap_time() + pit_stop_seconds
            self._state.total_elapsed_seconds += lap_time
            self._state.last_lap_time_seconds = lap_time
            self._state.best_lap_time_seconds = min(self._state.best_lap_time_seconds, lap_time)
            self._consume_lap_resources()
            self._state.current_lap += 1
            self._state.laps_remaining = max(0, self.config.total_laps - (self._state.current_lap - 1))
            LOGGER.info(
                "Lap complete: lap=%s lap_time=%.3f total=%.3f wear=%.3f fuel=%.3f",
                self._state.current_lap - 1,
                lap_time,
                self._state.total_elapsed_seconds,
                self._state.tire_wear,
                self._state.fuel_kg,
            )
        else:
            tick_time = (self._estimate_lap_time() / max(self.config.route_length_hint_meters, 1.0)) * (
                progress_delta * self.config.route_length_hint_meters
            )
            self._state.total_elapsed_seconds += tick_time

        lap_time_delta = previous_lap_time - self._state.last_lap_time_seconds
        race_finished = self._state.current_lap > self.config.total_laps
        return StrategyStepResult(
            state=self._state,
            lap_completed=lap_completed,
            lap_time_delta_seconds=lap_time_delta,
            pit_stop_seconds=pit_stop_seconds,
            race_finished=race_finished,
        )

    def _estimate_lap_time(self) -> float:
        wear_penalty = 9.0 * self._state.tire_wear
        fuel_penalty = 0.04 * self._state.fuel_kg
        grip_penalty = max(0.0, (1.0 - self._state.weather_grip) * 12.0)
        compound_offset = self._BASE_COMPOUND_OFFSET[self._state.tire_compound]
        noise = self._random.uniform(-0.4, 0.4)
        return self.config.base_lap_time_seconds + wear_penalty + fuel_penalty + grip_penalty + compound_offset + noise

    def _consume_lap_resources(self) -> None:
        wear_rate = self._WEAR_RATE[self._state.tire_compound]
        stochastic_wear = self._random.uniform(0.0, 0.03)
        self._state.tire_wear = min(self.config.max_tire_wear, self._state.tire_wear + wear_rate + stochastic_wear)
        self._state.fuel_kg = max(0.0, self._state.fuel_kg - self.config.fuel_burn_per_lap_kg)

        grip_delta = self._random.uniform(-self.config.weather_grip_variance, self.config.weather_grip_variance)
        self._state.weather_grip = min(1.05, max(0.75, self._state.weather_grip + grip_delta))

        # TODO: Record per-modality contribution traces here for future SVMM analysis.
