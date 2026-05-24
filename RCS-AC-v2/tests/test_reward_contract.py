from __future__ import annotations

import math

import pytest

from rcs_ac_v2.env.reward import RewardInputs, compute_continue, compute_reward


def base_inputs(**overrides: object) -> RewardInputs:
    values = {
        "route_progress_prev": 0.0,
        "route_progress_curr": 0.0,
        "speed_kmh": 20.0,
        "target_speed_kmh": 20.0,
        "lane_offset_m": 0.0,
        "collision": False,
        "offroad": False,
        "red_light": False,
        "stuck": False,
        "done": False,
    }
    values.update(overrides)
    return RewardInputs(**values)  # type: ignore[arg-type]


def test_route_progress_increases_reward() -> None:
    no_progress = compute_reward(base_inputs(route_progress_curr=0.0))
    progress = compute_reward(base_inputs(route_progress_curr=0.2))
    assert progress.total > no_progress.total
    assert progress.route_progress > 0.0


def test_penalties_are_negative() -> None:
    breakdown = compute_reward(
        base_inputs(collision=True, offroad=True, red_light=True, stuck=True, done=True)
    )
    assert breakdown.collision < 0.0
    assert breakdown.offroad < 0.0
    assert breakdown.red_light < 0.0
    assert breakdown.stuck < 0.0


def test_continue_contract() -> None:
    assert compute_continue(done=False, truncated=False) == 1.0
    assert compute_continue(done=True, truncated=False) == 0.0
    assert compute_continue(done=True, truncated=True) == 1.0


def test_reward_breakdown_sums_to_total() -> None:
    breakdown = compute_reward(base_inputs(route_progress_curr=0.2, lane_offset_m=1.0))
    assert math.isclose(sum(breakdown.components.values()), breakdown.total)


def test_nan_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_reward(base_inputs(speed_kmh=float("nan")))
