import numpy as np

from rcs_ac_first.rl.rollout_buffer import RolloutBuffer


def _add_transition(buffer: RolloutBuffer, *, reward: float, done: bool, value: float) -> None:
    buffer.add(
        observation=np.zeros((buffer.obs_dim,), dtype=np.float32),
        action=np.zeros((buffer.action_dim,), dtype=np.float32),
        log_prob=0.0,
        reward=reward,
        done=done,
        value=value,
    )


def test_gae_stops_at_terminal_transition_inside_rollout() -> None:
    buffer = RolloutBuffer(
        capacity=3,
        obs_dim=1,
        action_dim=1,
        gamma=1.0,
        gae_lambda=1.0,
    )
    _add_transition(buffer, reward=1.0, done=True, value=10.0)
    _add_transition(buffer, reward=1.0, done=False, value=10.0)
    _add_transition(buffer, reward=1.0, done=False, value=10.0)

    buffer.compute_returns_and_advantages(last_value=10.0, last_done=False)

    np.testing.assert_allclose(buffer.advantages[: buffer.ptr], [-9.0, 2.0, 1.0])
    np.testing.assert_allclose(buffer.returns[: buffer.ptr], [1.0, 12.0, 11.0])


def test_gae_uses_last_done_for_final_transition() -> None:
    buffer = RolloutBuffer(
        capacity=1,
        obs_dim=1,
        action_dim=1,
        gamma=1.0,
        gae_lambda=1.0,
    )
    _add_transition(buffer, reward=1.0, done=False, value=10.0)

    buffer.compute_returns_and_advantages(last_value=99.0, last_done=True)

    np.testing.assert_allclose(buffer.advantages[: buffer.ptr], [-9.0])
    np.testing.assert_allclose(buffer.returns[: buffer.ptr], [1.0])
