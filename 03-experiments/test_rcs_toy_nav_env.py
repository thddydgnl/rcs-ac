"""Basic shape and NaN safety tests for RCSToyNavEnv."""

from __future__ import annotations

import numpy as np

from rcs_toy_nav_env import RCSToyNavEnv


def assert_obs_valid(env: RCSToyNavEnv, obs: dict[str, np.ndarray]) -> None:
    expected_shapes = {
        "vision": (8,),
        "distance": (9,),
        "ego": (5,),
        "route": (5,),
        "aux": (8,),
    }

    for key, expected_shape in expected_shapes.items():
        value = obs[key]
        assert value.shape == expected_shape, f"{key} shape mismatch: {value.shape}"
        assert value.dtype == np.float32, f"{key} dtype mismatch: {value.dtype}"
        assert np.all(np.isfinite(value)), f"{key} contains non-finite values"

    assert env.observation_space.contains(obs), "Observation violates observation_space"


def run_smoke_test() -> None:
    env = RCSToyNavEnv()

    obs, info = env.reset(seed=42)
    assert_obs_valid(env, obs)
    assert "true_state" in info

    # NaN / Inf robustness check on actions.
    weird_action = np.array([np.nan, np.inf], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(weird_action)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert_obs_valid(env, obs)

    # Multi-step random rollout shape check.
    for _ in range(32):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert_obs_valid(env, obs)
        if terminated or truncated:
            obs, info = env.reset()
            assert_obs_valid(env, obs)

    print("RCSToyNavEnv smoke test passed.")


def random_policy(env: RCSToyNavEnv) -> np.ndarray:
    return env.action_space.sample()


def heuristic_goal_policy(env: RCSToyNavEnv) -> np.ndarray:
    goal_vec = env.goal - np.array([env.agent_x, env.agent_y], dtype=np.float32)
    goal_bearing = float(np.arctan2(goal_vec[1], goal_vec[0]))
    rel_angle = env._wrap_angle(goal_bearing - env.agent_heading)

    turn = np.clip(rel_angle / np.pi, -1.0, 1.0)
    accel = 0.8 if abs(rel_angle) < (np.pi / 3.0) else 0.2
    return np.array([accel, turn], dtype=np.float32)


def rollout_episode(
    env: RCSToyNavEnv,
    policy_fn,
    seed: int,
    max_steps: int = 200,
) -> dict[str, float]:
    obs, info = env.reset(seed=seed)
    del obs, info

    episode_return = 0.0
    component_sums = {
        "reward/progress": 0.0,
        "reward/goal": 0.0,
        "reward/collision": 0.0,
        "reward/out_of_bounds": 0.0,
        "reward/action_change": 0.0,
        "reward/idle": 0.0,
    }

    for _ in range(max_steps):
        action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        del obs
        episode_return += float(reward)
        for key in component_sums:
            component_sums[key] += float(info[key])
        if terminated or truncated:
            break

    result = {
        "episode_return": episode_return,
        "steps": float(env.step_count),
        "goal_reached": float(info["goal_reached"]),
        "collision": float(info["collision"]),
        "out_of_bounds": float(info["out_of_bounds"]),
        "timeout": float(info["timeout"]),
    }
    result.update(component_sums)
    return result


def run_policy_comparison_test(num_episodes: int = 20) -> None:
    random_env = RCSToyNavEnv()
    heuristic_env = RCSToyNavEnv()

    random_returns = []
    heuristic_returns = []
    random_goals = 0.0
    heuristic_goals = 0.0

    for seed in range(num_episodes):
        random_result = rollout_episode(random_env, random_policy, seed=seed)
        heuristic_result = rollout_episode(heuristic_env, heuristic_goal_policy, seed=seed)
        random_returns.append(random_result["episode_return"])
        heuristic_returns.append(heuristic_result["episode_return"])
        random_goals += random_result["goal_reached"]
        heuristic_goals += heuristic_result["goal_reached"]

    print("Random policy avg return:", float(np.mean(random_returns)))
    print("Heuristic policy avg return:", float(np.mean(heuristic_returns)))
    print("Random policy goal rate:", random_goals / num_episodes)
    print("Heuristic policy goal rate:", heuristic_goals / num_episodes)


if __name__ == "__main__":
    run_smoke_test()
    run_policy_comparison_test()
