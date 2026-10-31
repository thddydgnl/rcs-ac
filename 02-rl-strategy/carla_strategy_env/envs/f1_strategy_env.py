"""Gymnasium-style CARLA environment where the agent makes pit strategy decisions."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import EnvConfig
from carla_strategy_env.carla_integration.actors import spawn_ego_vehicle
from carla_strategy_env.carla_integration.client import CarlaSession
from carla_strategy_env.carla_integration.sensors import CollisionSensor, RgbCameraSensor
from carla_strategy_env.carla_integration.track import TrackManager, TrackRoute
from carla_strategy_env.drivers.base import BaseDriverController, DriverStepResult
from carla_strategy_env.drivers.behavior_agent import BehaviorAgentDriver
from carla_strategy_env.drivers.fallback import SimpleWaypointDriver
from carla_strategy_env.strategy.models import F1StrategyModel, StrategyAction, StrategyStepResult


LOGGER = logging.getLogger(__name__)


class CarlaF1StrategyEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self._strategy_model = F1StrategyModel(self.config.race, self.config.random_seed)
        self._session: CarlaSession | None = None
        self._track_manager = TrackManager(self.config.carla.map_name, self.config.carla.spa_xodr_path)
        self._track_route: TrackRoute | None = None
        self._vehicle: Any | None = None
        self._camera: RgbCameraSensor | None = None
        self._collision_sensor: CollisionSensor | None = None
        self._driver: BaseDriverController | None = None
        self._step_count = 0

        strategy_dim = len(self._strategy_model.observe_vector())
        self.action_space = spaces.Discrete(len(StrategyAction))
        self.observation_space = spaces.Dict(
            {
                "vision": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.config.sensors.image_height, self.config.sensors.image_width, 3),
                    dtype=np.uint8,
                ),
                "strategy": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(strategy_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.close()
        self._step_count = 0
        self._strategy_model.reset()
        self._session = CarlaSession(self.config.carla)
        self._session.connect()
        self._session.validate_map()
        self._track_manager.validate_runtime_map(self._session.road_map)
        self._track_route = self._track_manager.build_route(self._session.road_map)

        self._vehicle = spawn_ego_vehicle(self._session, self.config)
        self._camera = RgbCameraSensor(self._session, self._vehicle, self.config.sensors)
        self._camera.spawn()
        self._collision_sensor = CollisionSensor(self._session, self._vehicle, self.config.sensors)
        self._collision_sensor.spawn()
        self._driver = self._setup_driver()

        for _ in range(5):
            self._session.tick()

        observation = self._build_observation()
        info = self._build_info()
        LOGGER.info("Environment reset complete.")
        return observation, info

    def _setup_driver(self) -> BaseDriverController:
        if self._vehicle is None or self._session is None or self._track_route is None:
            raise RuntimeError("Vehicle and track must exist before driver setup.")

        if BehaviorAgentDriver.is_available():
            driver = BehaviorAgentDriver(self.config.race, self.config.carla.behavior_agent_mode)
            try:
                driver.setup(self._vehicle, self._session.world, self._session.road_map, self._track_route.waypoints)
                return driver
            except Exception:
                LOGGER.exception("BehaviorAgent setup failed; switching to fallback waypoint driver.")

        fallback = SimpleWaypointDriver(self.config.race)
        fallback.setup(self._vehicle, self._session.world, self._session.road_map, self._track_route.waypoints)
        return fallback

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._session is None or self._driver is None or self._collision_sensor is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self._step_count += 1

        try:
            strategy_action = StrategyAction(action)
        except ValueError:
            strategy_action = StrategyAction.STAY_OUT
            invalid_penalty = self.config.reward.invalid_action_penalty
        else:
            invalid_penalty = 0.0

        progress_delta = 0.0
        last_driver_result: DriverStepResult | None = None
        for _ in range(self.config.race.strategy_interval_ticks):
            self._session.tick()
            if hasattr(self._driver, "set_collision_state"):
                self._driver.set_collision_state(self._collision_sensor.has_collision())
            last_driver_result = self._driver.tick()
            progress_delta += last_driver_result.progress_delta
            if last_driver_result.collision or last_driver_result.stuck:
                break

        if last_driver_result is None:
            raise RuntimeError("Driver did not produce a step result.")

        progress_ratio = min(1.0, progress_delta)
        strategy_result = self._strategy_model.advance(progress_ratio, strategy_action)
        reward = self._compute_reward(strategy_result, last_driver_result, invalid_penalty)
        terminated = bool(last_driver_result.collision or last_driver_result.stuck or strategy_result.race_finished)
        truncated = self._step_count >= self.config.race.max_episode_steps

        observation = self._build_observation()
        info = self._build_info(
            reward=reward,
            progress_delta=progress_delta,
            speed_mps=last_driver_result.speed_mps,
            lap_completed=strategy_result.lap_completed,
            pit_stop_seconds=strategy_result.pit_stop_seconds,
        )

        if terminated or truncated:
            LOGGER.info("Episode finished: terminated=%s truncated=%s info=%s", terminated, truncated, info)

        return observation, reward, terminated, truncated, info

    def _compute_reward(self, strategy_result: StrategyStepResult, driver_result: DriverStepResult, invalid_penalty: float) -> float:
        reward = self.config.reward.alive_penalty + invalid_penalty
        reward += self.config.reward.progress_reward_scale * driver_result.progress_delta
        reward += self.config.reward.lap_improvement_scale * strategy_result.lap_time_delta_seconds
        reward += self.config.reward.time_penalty_scale * strategy_result.state.last_lap_time_seconds * float(
            strategy_result.lap_completed
        )
        reward += self.config.reward.pit_stop_penalty_scale * strategy_result.pit_stop_seconds

        if driver_result.collision:
            reward += self.config.reward.collision_penalty
        if driver_result.stuck:
            reward += self.config.reward.stuck_penalty
        if strategy_result.race_finished:
            reward += self.config.reward.completion_reward

        LOGGER.info(
            "Reward breakdown: total=%.3f progress=%.3f lap_delta=%.3f pit=%.3f collision=%s stuck=%s finished=%s",
            reward,
            driver_result.progress_delta,
            strategy_result.lap_time_delta_seconds,
            strategy_result.pit_stop_seconds,
            driver_result.collision,
            driver_result.stuck,
            strategy_result.race_finished,
        )
        return float(reward)

    def _build_observation(self) -> dict[str, np.ndarray]:
        if self._camera is None:
            raise RuntimeError("Camera sensor is not initialized.")

        return {
            "vision": self._camera.get_observation(),
            "strategy": np.asarray(self._strategy_model.observe_vector(), dtype=np.float32),
        }

    def _build_info(self, **extra: Any) -> dict[str, Any]:
        state = self._strategy_model.state
        info = {
            "current_lap": state.current_lap,
            "laps_remaining": state.laps_remaining,
            "tire_compound": state.tire_compound.value,
            "tire_wear": state.tire_wear,
            "fuel_kg": state.fuel_kg,
            "last_lap_time_seconds": state.last_lap_time_seconds,
            "best_lap_time_seconds": state.best_lap_time_seconds,
            "pit_count": state.pit_count,
            "weather_grip": state.weather_grip,
            "progress": state.progress,
        }
        info.update(extra)
        return info

    def render(self) -> np.ndarray:
        return self._build_observation()["vision"]

    def close(self) -> None:
        if self._driver is not None:
            self._driver.destroy()
            self._driver = None
        if self._session is not None:
            self._session.cleanup()
            self._session = None
        self._vehicle = None
        self._camera = None
        self._collision_sensor = None
        self._track_route = None


def rollout_random_policy(env: CarlaF1StrategyEnv, actions: Iterable[int]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    obs, info = env.reset()
    observations.append({"observation": obs, "info": info})

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append({"observation": obs, "reward": reward, "info": info})
        if terminated or truncated:
            break
    return observations
