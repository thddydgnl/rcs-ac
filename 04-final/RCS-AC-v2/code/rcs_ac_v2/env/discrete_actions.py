"""Raw2Drive Table 12 discrete action contract.

The PDF prints Table 12 as three action groups per row. This module preserves
the left-to-right, top-to-bottom table order so action logits have a stable
mapping throughout v2.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


ACTION_TABLE_VERSION = "raw2drive_table12_row_major_v1"
NUM_RAW2DRIVE_ACTIONS = 39


@dataclass(frozen=True)
class DiscreteAction:
    action_id: int
    throttle: float
    brake: float
    steer: float
    reverse: bool


_TABLE12_VALUES: tuple[tuple[float, float, float, bool], ...] = (
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


RAW2DRIVE_ACTIONS: tuple[DiscreteAction, ...] = tuple(
    DiscreteAction(action_id=index, throttle=throttle, brake=brake, steer=steer, reverse=reverse)
    for index, (throttle, brake, steer, reverse) in enumerate(_TABLE12_VALUES)
)


def validate_action_table(actions: tuple[DiscreteAction, ...] = RAW2DRIVE_ACTIONS) -> None:
    if len(actions) != NUM_RAW2DRIVE_ACTIONS:
        raise ValueError(f"expected {NUM_RAW2DRIVE_ACTIONS} actions, got {len(actions)}")

    for expected_id, action in enumerate(actions):
        if action.action_id != expected_id:
            raise ValueError(f"action id mismatch at index {expected_id}: {action.action_id}")
        if not isinstance(action.reverse, bool):
            raise ValueError(f"reverse must be bool for action {action.action_id}")
        for name, value in (
            ("throttle", action.throttle),
            ("brake", action.brake),
            ("steer", action.steer),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite for action {action.action_id}")
        if not 0.0 <= action.throttle <= 1.0:
            raise ValueError(f"throttle out of range for action {action.action_id}")
        if not 0.0 <= action.brake <= 1.0:
            raise ValueError(f"brake out of range for action {action.action_id}")
        if not -1.0 <= action.steer <= 1.0:
            raise ValueError(f"steer out of range for action {action.action_id}")
        if action.throttle > 0.0 and action.brake > 0.0:
            raise ValueError(f"throttle and brake both positive for action {action.action_id}")


def get_discrete_action(action_id: int) -> DiscreteAction:
    if not isinstance(action_id, int):
        raise TypeError(f"action_id must be int, got {type(action_id).__name__}")
    if action_id < 0 or action_id >= NUM_RAW2DRIVE_ACTIONS:
        raise ValueError(f"action_id must be in [0, {NUM_RAW2DRIVE_ACTIONS - 1}], got {action_id}")
    return RAW2DRIVE_ACTIONS[action_id]


def to_carla_control(action: DiscreteAction) -> dict[str, float | bool]:
    return {
        "throttle": float(action.throttle),
        "brake": float(action.brake),
        "steer": float(action.steer),
        "reverse": bool(action.reverse),
    }


def action_table_fingerprint(actions: tuple[DiscreteAction, ...] = RAW2DRIVE_ACTIONS) -> str:
    payload = [
        {
            "action_id": action.action_id,
            "throttle": action.throttle,
            "brake": action.brake,
            "steer": action.steer,
            "reverse": action.reverse,
        }
        for action in actions
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ACTION_TABLE_SHA256 = action_table_fingerprint()


validate_action_table()
