from __future__ import annotations

import pytest

from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, bev_row_col_to_ego_xy, ego_xy_to_bev_row_col


def test_bev_spec_extent_consistency_default() -> None:
    spec = BEVSpec()
    spec.validate()
    assert abs(spec.height * spec.resolution_m - (spec.front_m + spec.rear_m)) < 1.0e-4
    assert abs(spec.width * spec.resolution_m - (spec.left_m + spec.right_m)) < 1.0e-4


def test_ego_origin_maps_inside_grid() -> None:
    spec = BEVSpec()
    row, col, valid = ego_xy_to_bev_row_col(0.0, 0.0, spec)
    assert valid
    assert 0 <= row < spec.height
    assert 0 <= col < spec.width


def test_forward_point_has_smaller_row_than_ego() -> None:
    spec = BEVSpec()
    ego_row, _, _ = ego_xy_to_bev_row_col(0.0, 0.0, spec)
    fwd_row, _, _ = ego_xy_to_bev_row_col(10.0, 0.0, spec)
    assert fwd_row < ego_row


def test_left_point_has_smaller_col_than_ego() -> None:
    spec = BEVSpec()
    _, ego_col, _ = ego_xy_to_bev_row_col(0.0, 0.0, spec)
    _, left_col, _ = ego_xy_to_bev_row_col(0.0, 5.0, spec)
    assert left_col < ego_col


def test_bev_row_col_round_trip_near_cell_center() -> None:
    spec = BEVSpec()
    row, col, valid = ego_xy_to_bev_row_col(6.0, -3.0, spec)
    assert valid
    x_m, y_m = bev_row_col_to_ego_xy(row, col, spec)
    assert abs(x_m - 6.0) <= spec.resolution_m / 2.0
    assert abs(y_m + 3.0) <= spec.resolution_m / 2.0


def test_bev_boundaries_map_inside_grid_symmetrically() -> None:
    spec = BEVSpec()
    assert ego_xy_to_bev_row_col(spec.front_m, 0.0, spec) == (0, 64, True)
    assert ego_xy_to_bev_row_col(-spec.rear_m, 0.0, spec) == (127, 64, True)
    assert ego_xy_to_bev_row_col(0.0, spec.left_m, spec) == (106, 0, True)
    assert ego_xy_to_bev_row_col(0.0, -spec.right_m, spec) == (106, 127, True)


def test_invalid_extent_raises() -> None:
    with pytest.raises(ValueError, match="extent mismatch"):
        BEVSpec(
            height=128,
            width=128,
            resolution_m=0.35,
            front_m=40.0,
            rear_m=8.0,
            left_m=22.4,
            right_m=22.4,
            strict_extent_check=True,
        )
