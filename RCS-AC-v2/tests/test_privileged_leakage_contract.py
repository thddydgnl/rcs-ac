from __future__ import annotations

from dataclasses import fields

from rcs_ac_v2.data.schema import RawPolicyInput, make_synthetic_raw_sensor_batch


FORBIDDEN_RAW_POLICY_FIELDS = {
    "privileged_bev",
    "privileged_ego",
    "reward",
    "cont",
    "next_state",
    "actor_boxes",
    "hd_map",
    "route_completion_future",
}


def test_raw_policy_input_dataclass_has_no_privileged_fields() -> None:
    field_names = {field.name for field in fields(RawPolicyInput)}
    assert field_names.isdisjoint(FORBIDDEN_RAW_POLICY_FIELDS)
    assert all("privileged" not in name for name in field_names)


def test_raw_sensor_batch_policy_input_view_excludes_privileged_training_targets() -> None:
    batch = make_synthetic_raw_sensor_batch()
    view = batch.policy_input_view(include_past_action=True)
    assert view.raw_images is batch.raw_images
    assert view.raw_imu is batch.raw_imu
    assert view.past_action is batch.action
    for forbidden in FORBIDDEN_RAW_POLICY_FIELDS:
        assert not hasattr(view, forbidden)
