from __future__ import annotations

import pytest

from rcs_ac_v2.env.discrete_actions import (
    ACTION_TABLE_SHA256,
    ACTION_TABLE_VERSION,
    NUM_RAW2DRIVE_ACTIONS,
    RAW2DRIVE_ACTIONS,
    action_table_fingerprint,
    get_discrete_action,
    to_carla_control,
    validate_action_table,
)


EXPECTED_TABLE12_ROW_MAJOR = (
    (0.0, 0.0, 1.0, False),
    (0.3, 0.0, -0.2, False),
    (0.0, 0.0, 0.1, False),
    (0.7, 0.0, -0.5, False),
    (0.3, 0.0, -0.1, False),
    (0.0, 0.0, 0.3, False),
    (0.7, 0.0, -0.3, False),
    (0.3, 0.0, 0.0, False),
    (0.0, 0.0, 0.6, False),
    (0.7, 0.0, -0.2, False),
    (0.3, 0.0, 0.1, False),
    (0.0, 0.0, 1.0, False),
    (0.7, 0.0, -0.1, False),
    (0.3, 0.0, 0.2, False),
    (0.5, 0.0, -0.5, True),
    (0.7, 0.0, 0.0, False),
    (0.3, 0.0, 0.3, False),
    (0.5, 0.0, -0.3, True),
    (0.7, 0.0, 0.1, False),
    (0.3, 0.0, 0.5, False),
    (0.5, 0.0, -0.2, True),
    (0.7, 0.0, 0.2, False),
    (0.3, 0.0, 0.7, False),
    (0.5, 0.0, -0.1, True),
    (0.7, 0.0, 0.3, False),
    (0.0, 0.0, -1.0, False),
    (0.5, 0.0, 0.0, True),
    (0.7, 0.0, 0.5, False),
    (0.0, 0.0, -0.6, False),
    (0.5, 0.0, 0.1, True),
    (0.3, 0.0, -0.7, False),
    (0.0, 0.0, -0.3, False),
    (0.5, 0.0, 0.2, True),
    (0.3, 0.0, -0.5, False),
    (0.0, 0.0, -0.1, False),
    (0.5, 0.0, 0.3, True),
    (0.3, 0.0, -0.3, False),
    (1.0, 0.0, 0.0, False),
    (0.5, 0.0, 0.5, True),
)


def test_action_table_matches_pdf_table12_row_major() -> None:
    assert ACTION_TABLE_VERSION == "raw2drive_table12_row_major_v1"
    assert ACTION_TABLE_SHA256 == action_table_fingerprint()
    assert ACTION_TABLE_SHA256 == "a4c1efa534cc9d75065c533cbd9860043088432f184c2ffb27a20679257b294d"
    assert len(RAW2DRIVE_ACTIONS) == NUM_RAW2DRIVE_ACTIONS == 39
    actual = tuple(
        (action.throttle, action.brake, action.steer, action.reverse)
        for action in RAW2DRIVE_ACTIONS
    )
    assert actual == EXPECTED_TABLE12_ROW_MAJOR


def test_action_ids_are_contiguous_and_ranges_are_valid() -> None:
    validate_action_table()
    assert [action.action_id for action in RAW2DRIVE_ACTIONS] == list(range(39))
    for action in RAW2DRIVE_ACTIONS:
        assert 0.0 <= action.throttle <= 1.0
        assert 0.0 <= action.brake <= 1.0
        assert -1.0 <= action.steer <= 1.0
        assert isinstance(action.reverse, bool)


def test_action_lookup_rejects_out_of_range_ids() -> None:
    assert get_discrete_action(0) is RAW2DRIVE_ACTIONS[0]
    assert get_discrete_action(38) is RAW2DRIVE_ACTIONS[38]
    with pytest.raises(ValueError):
        get_discrete_action(-1)
    with pytest.raises(ValueError):
        get_discrete_action(39)
    with pytest.raises(TypeError):
        get_discrete_action(1.0)  # type: ignore[arg-type]


def test_to_carla_control_is_deterministic() -> None:
    action = get_discrete_action(14)
    assert to_carla_control(action) == to_carla_control(action)
    assert to_carla_control(action) == {
        "throttle": 0.5,
        "brake": 0.0,
        "steer": -0.5,
        "reverse": True,
    }
