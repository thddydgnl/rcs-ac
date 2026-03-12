"""Central configuration for the CARLA F1 strategy MVP environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class CarlaConfig:
    host: str = field(default_factory=lambda: os.environ.get("CARLA_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("CARLA_PORT", "2000")))
    timeout_seconds: float = 10.0
    map_name: str = "SpaCircuit"
    fixed_delta_seconds: float = 0.1
    synchronous_mode: bool = True
    traffic_manager_port: int = 8000
    behavior_agent_mode: str = "normal"
    spa_xodr_path: Path = PROJECT_ROOT / "data" / "maps" / "spa" / "Spa_Circuit_Base.xodr"
    spa_udatasmith_path: Path = PROJECT_ROOT / "data" / "maps" / "spa" / "Spa_Circuit_Base.udatasmith"
    spa_fbx_path: Path = PROJECT_ROOT / "data" / "maps" / "spa" / "Spa_Circuit_Base.fbx"


@dataclass(slots=True)
class SensorConfig:
    image_width: int = 160
    image_height: int = 90
    image_fov: float = 100.0
    camera_x: float = 1.8
    camera_z: float = 1.1
    collision_sensor_tick: float = 0.0


@dataclass(slots=True)
class RaceConfig:
    total_laps: int = 5
    max_episode_steps: int = 150
    strategy_interval_ticks: int = 20
    base_lap_time_seconds: float = 110.0
    pit_stop_loss_seconds: float = 20.0
    weather_grip_baseline: float = 1.0
    weather_grip_variance: float = 0.08
    max_tire_wear: float = 1.0
    initial_fuel_kg: float = 35.0
    fuel_burn_per_lap_kg: float = 1.8
    stuck_speed_threshold_mps: float = 1.0
    stuck_tick_limit: int = 30
    route_length_hint_meters: float = 7004.0


@dataclass(slots=True)
class RewardConfig:
    completion_reward: float = 100.0
    collision_penalty: float = -100.0
    stuck_penalty: float = -60.0
    invalid_action_penalty: float = -2.0
    time_penalty_scale: float = -0.05
    lap_improvement_scale: float = 0.5
    pit_stop_penalty_scale: float = -0.5
    progress_reward_scale: float = 2.0
    alive_penalty: float = -0.01


@dataclass(slots=True)
class EnvConfig:
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    vehicle_filter: str = "vehicle.*"
    preferred_vehicle_ids: tuple[str, ...] = (
        "vehicle.tesla.model3",
        "vehicle.audi.tt",
        "vehicle.dodge.charger_2020",
    )
    random_seed: int = 7
    enable_render: bool = False
