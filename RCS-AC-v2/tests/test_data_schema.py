from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from rcs_ac_v2.data.schema import (
    RawPolicyInput,
    SchemaConfig,
    make_synthetic_privileged_batch,
    make_synthetic_raw_sensor_batch,
    validate_privileged_batch,
    validate_raw_sensor_batch,
)


def test_valid_synthetic_privileged_batch_passes() -> None:
    batch = make_synthetic_privileged_batch(batch_size=2, sequence_length=3, seed=1)
    validate_privileged_batch(batch)
    assert batch.privileged_bev.shape[:3] == (2, 3, 43)
    assert batch.action.dtype == torch.int64


def test_valid_synthetic_raw_sensor_batch_passes() -> None:
    batch = make_synthetic_raw_sensor_batch(batch_size=2, sequence_length=3, seed=2)
    validate_raw_sensor_batch(batch)
    assert batch.raw_images.shape[:4] == (2, 3, 4, 3)


def test_wrong_privileged_channel_count_fails() -> None:
    batch = make_synthetic_privileged_batch()
    batch.privileged_bev = batch.privileged_bev[:, :, :42]
    with pytest.raises(ValueError, match="channel"):
        validate_privileged_batch(batch)


def test_action_dtype_and_range_are_checked() -> None:
    batch = make_synthetic_privileged_batch()
    batch.action = batch.action.float()
    with pytest.raises(ValueError, match="int64"):
        validate_privileged_batch(batch)

    batch = make_synthetic_privileged_batch()
    batch.action[0, 0] = 39
    with pytest.raises(ValueError, match="< 39"):
        validate_privileged_batch(batch)


def test_continue_values_are_binary() -> None:
    batch = make_synthetic_privileged_batch()
    batch.cont[0, 0] = 0.5
    with pytest.raises(ValueError, match="0 or 1"):
        validate_privileged_batch(batch)


def test_schema_config_can_pin_dims() -> None:
    batch = make_synthetic_raw_sensor_batch(num_cameras=6, imu_dim=5, ego_dim=7)
    validate_raw_sensor_batch(batch, SchemaConfig(num_cameras=6, imu_dim=5, ego_dim=7))
    with pytest.raises(ValueError, match="camera count"):
        validate_raw_sensor_batch(batch, SchemaConfig(num_cameras=4, imu_dim=5, ego_dim=7))


def test_raw_policy_input_view_excludes_privileged_targets() -> None:
    batch = make_synthetic_raw_sensor_batch()
    view = batch.policy_input_view()
    view_fields = {field.name for field in fields(RawPolicyInput)}
    assert view_fields == {"raw_images", "raw_imu", "route_command", "past_action"}
    assert not any("privileged" in field for field in view_fields)
    assert not hasattr(view, "reward")
    assert not hasattr(view, "cont")
    assert view.raw_images is batch.raw_images
    assert view.raw_imu is batch.raw_imu
