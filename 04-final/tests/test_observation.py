import numpy as np
import pytest
import torch

from rcs_ac_first.runtime.observation import (
    build_policy_observation_spec,
    flatten_observation_dict,
    split_flat_observation,
)


def test_flat_observation_round_trip_uses_contract_order() -> None:
    spec = build_policy_observation_spec(
        rgb_shape=(1, 2, 2),
        lidar_shape=(1, 1, 2),
        ego_dim=2,
        route_dim=3,
    )
    observation = {
        "rgb": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        "lidar": np.array([[[5.0, 6.0]]], dtype=np.float32),
        "ego": np.array([7.0, 8.0], dtype=np.float32),
        "route": np.array([9.0, 10.0, 11.0], dtype=np.float32),
    }

    flat = flatten_observation_dict(observation, spec)
    split = split_flat_observation(flat, spec)

    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    for key, expected in observation.items():
        torch.testing.assert_close(split[key].squeeze(0), torch.as_tensor(expected))


def test_split_flat_observation_rejects_bad_dim_and_non_finite() -> None:
    spec = build_policy_observation_spec(
        rgb_shape=(1, 1, 1),
        lidar_shape=(1, 1, 1),
        ego_dim=1,
        route_dim=1,
    )

    with pytest.raises(ValueError, match="Expected flat observation dim"):
        split_flat_observation(np.zeros((spec.total_dim + 1,), dtype=np.float32), spec)

    bad = np.zeros((spec.total_dim,), dtype=np.float32)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        split_flat_observation(bad, spec)
