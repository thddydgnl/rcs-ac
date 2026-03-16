from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_carla() -> None:
    try:
        import carla  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    search_roots = []

    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        search_roots.append(Path(env_root))

    search_roots.extend(
        [
            Path.home() / "Downloads" / "CARLA_0.9.16",
            Path("C:/CARLA_0.9.16"),
        ]
    )

    for root in search_roots:
        dist_dir = root / "PythonAPI" / "carla" / "dist"
        if not dist_dir.exists():
            continue

        matching = sorted(dist_dir.glob(f"carla-*-{py_tag}-{py_tag}-win_amd64.whl"))
        if matching:
            sys.path.append(str(matching[-1]))
            import carla  # noqa: F401
            return

    raise RuntimeError("CARLA Python API was not found. Set CARLA_ROOT or use a matching Python environment.")


_bootstrap_carla()


REQUIRED_TOP_LEVEL_KEYS = {"obs", "reward", "done", "info"}
REQUIRED_OBS_KEYS = {"rgb", "lidar_bev", "ego", "route"}
REQUIRED_BREAKDOWN_KEYS = {
    "progress_reward",
    "lane_reward",
    "collision_penalty",
    "offroad_penalty",
    "lane_invasion_penalty",
    "control_cost",
    "goal_bonus",
    "stuck_penalty",
}
ALLOWED_DONE_REASONS = {"running", "goal", "collision", "offroad", "stuck", "max_steps"}
REQUIRED_LOG_COLUMNS = {
    "reward",
    "progress_reward",
    "lane_reward",
    "collision_penalty",
    "offroad_penalty",
    "lane_invasion_penalty",
    "control_cost",
    "goal_bonus",
    "stuck_penalty",
    "done",
    "done_reason",
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_obs(obs: dict[str, Any], info: dict[str, Any], expected_shapes: dict[str, tuple[int, ...]] | None = None) -> dict[str, tuple[int, ...]]:
    assert_true(REQUIRED_OBS_KEYS.issubset(obs.keys()), f"Observation keys missing: {REQUIRED_OBS_KEYS - set(obs.keys())}")

    rgb = obs["rgb"]
    lidar_bev = obs["lidar_bev"]
    ego = obs["ego"]
    route = obs["route"]

    for name, value in obs.items():
        assert_true(isinstance(value, np.ndarray), f"obs['{name}'] is not numpy.ndarray")
        assert_true(value.dtype == np.float32, f"obs['{name}'] dtype is not np.float32: {value.dtype}")

    assert_true(rgb.ndim == 3 and rgb.shape[0] == 3, f"rgb shape expected (3,H,W), got {rgb.shape}")
    assert_true(lidar_bev.ndim == 3 and lidar_bev.shape[0] == 1, f"lidar_bev shape expected (1,H,W), got {lidar_bev.shape}")
    assert_true(ego.shape == (7,), f"ego shape expected (7,), got {ego.shape}")
    assert_true(route.shape == (6,), f"route shape expected (6,), got {route.shape}")

    assert_true("observation_shapes" in info, "info['observation_shapes'] missing")
    for key in REQUIRED_OBS_KEYS:
        actual_shape = tuple(obs[key].shape)
        reported_shape = tuple(info["observation_shapes"][key])
        assert_true(actual_shape == reported_shape, f"Shape mismatch for {key}: actual={actual_shape}, reported={reported_shape}")

    if expected_shapes is not None:
        for key, expected_shape in expected_shapes.items():
            assert_true(tuple(obs[key].shape) == expected_shape, f"Shape drift detected for {key}: expected={expected_shape}, got={tuple(obs[key].shape)}")

    return {key: tuple(obs[key].shape) for key in REQUIRED_OBS_KEYS}


def validate_step_result(result: dict[str, Any], expected_shapes: dict[str, tuple[int, ...]] | None = None) -> dict[str, tuple[int, ...]]:
    assert_true(REQUIRED_TOP_LEVEL_KEYS.issubset(result.keys()), f"Top-level keys missing: {REQUIRED_TOP_LEVEL_KEYS - set(result.keys())}")
    assert_true(isinstance(result["reward"], float), f"reward is not float: {type(result['reward'])}")
    assert_true(result["reward"] is not None, "reward is None")
    assert_true(isinstance(result["done"], bool), f"done is not bool: {type(result['done'])}")
    assert_true(isinstance(result["info"], dict), "info is not dict")

    info = result["info"]
    assert_true("reward_breakdown" in info, "info['reward_breakdown'] missing")
    assert_true("done_reason" in info, "info['done_reason'] missing")
    assert_true(info["done_reason"] in ALLOWED_DONE_REASONS, f"Unexpected done_reason: {info['done_reason']}")
    if result["done"]:
        assert_true(info["done_reason"] != "running", "done=True but done_reason='running'")

    breakdown = info["reward_breakdown"]
    assert_true(REQUIRED_BREAKDOWN_KEYS.issubset(breakdown.keys()), f"reward_breakdown keys missing: {REQUIRED_BREAKDOWN_KEYS - set(breakdown.keys())}")
    for key in REQUIRED_BREAKDOWN_KEYS:
        float(breakdown[key])

    return validate_obs(result["obs"], info, expected_shapes)


def read_csv_header(csv_path: str) -> list[str]:
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def run_heuristic_episode(CarlaEnv) -> tuple[str, dict[str, tuple[int, ...]]]:
    output_dir = os.path.join("outputs", "verify_stage3_heuristic")
    env = CarlaEnv(output_dir=output_dir, max_steps=25)
    try:
        reset_result = env.reset()
        shapes = validate_step_result(reset_result)
        assert_true(reset_result["info"]["done_reason"] == "running", "reset done_reason must be 'running'")

        last_result = reset_result
        for _ in range(100):
            last_result = env.step(None)
            shapes = validate_step_result(last_result, shapes)
            if last_result["done"]:
                break

        assert_true(last_result["done"], "heuristic episode did not terminate")
        assert_true(os.path.exists(os.path.join(output_dir, "logs.csv")), "heuristic logs.csv was not created")
        return os.path.join(output_dir, "logs.csv"), shapes
    finally:
        env.close()


def run_random_episode(CarlaEnv, expected_shapes: dict[str, tuple[int, ...]]) -> str:
    output_dir = os.path.join("outputs", "verify_stage3_random")
    env = CarlaEnv(output_dir=output_dir, max_steps=15)
    rng = np.random.default_rng(42)
    try:
        reset_result = env.reset()
        validate_step_result(reset_result, expected_shapes)

        np_action = np.asarray([0.1, 0.3, 0.0], dtype=np.float32)
        np_result = env.step(np_action)
        validate_step_result(np_result, expected_shapes)

        list_action = [0.0, 0.2, 0.1]
        list_result = env.step(list_action)
        validate_step_result(list_result, expected_shapes)

        for _ in range(30):
            random_action = np.asarray(
                [
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(0.0, 1.0),
                    rng.uniform(0.0, 1.0),
                ],
                dtype=np.float32,
            )
            result = env.step(random_action)
            validate_step_result(result, expected_shapes)
            if result["done"]:
                break

        assert_true(os.path.exists(os.path.join(output_dir, "logs.csv")), "random logs.csv was not created")
        return os.path.join(output_dir, "logs.csv")
    finally:
        env.close()


def run_max_steps_check(CarlaEnv) -> None:
    env = CarlaEnv(output_dir=os.path.join("outputs", "verify_stage3_max_steps"), max_steps=1)
    try:
        env.reset()
        result = env.step(None)
        assert_true(result["done"] is True, "max_steps check did not terminate")
        assert_true(result["done_reason"] == "max_steps", f"Expected max_steps done_reason, got {result['done_reason']}")
    finally:
        env.close()


def main() -> int:
    print("[A] import verification")
    from env import carla_env, reward, route_manager, sensors, state_builder
    from env.carla_env import CarlaEnv
    from env.reward import RewardConfig
    from env.state_builder import ObservationConfig

    print("  imports: ok")

    print("[B] static structure verification")
    obs_cfg = ObservationConfig()
    reward_cfg = RewardConfig()
    assert_true(obs_cfg.rgb_width == 160, "ObservationConfig.rgb_width default mismatch")
    assert_true(obs_cfg.rgb_height == 90, "ObservationConfig.rgb_height default mismatch")
    assert_true(obs_cfg.route_lookahead_count == 3, "ObservationConfig.route_lookahead_count default mismatch")
    assert_true(reward_cfg.progress_weight == 5.0, "RewardConfig.progress_weight default mismatch")
    assert_true(reward_cfg.progress_delta_clip == 1.0, "RewardConfig.progress_delta_clip default mismatch")
    assert_true(hasattr(CarlaEnv, "reset"), "CarlaEnv.reset missing")
    assert_true(hasattr(CarlaEnv, "step"), "CarlaEnv.step missing")
    assert_true(hasattr(CarlaEnv, "close"), "CarlaEnv.close missing")
    print("  static config/interface: ok")

    print("[C] heuristic runtime verification")
    heuristic_csv, shapes = run_heuristic_episode(CarlaEnv)
    print(f"  heuristic episode: ok ({heuristic_csv})")

    print("[D] random action runtime verification")
    random_csv = run_random_episode(CarlaEnv, shapes)
    print(f"  random episode: ok ({random_csv})")

    print("[E] max_steps done_reason verification")
    run_max_steps_check(CarlaEnv)
    print("  max_steps done_reason: ok")

    print("[F] csv column verification")
    heuristic_header = set(read_csv_header(heuristic_csv))
    random_header = set(read_csv_header(random_csv))
    assert_true(REQUIRED_LOG_COLUMNS.issubset(heuristic_header), f"heuristic logs missing columns: {REQUIRED_LOG_COLUMNS - heuristic_header}")
    assert_true(REQUIRED_LOG_COLUMNS.issubset(random_header), f"random logs missing columns: {REQUIRED_LOG_COLUMNS - random_header}")
    print("  csv columns: ok")

    print("[RESULT] PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - runtime verification path
        print(f"[RESULT] FAIL: {exc}")
        raise
