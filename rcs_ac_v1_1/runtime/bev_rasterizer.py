from __future__ import annotations

import math

import numpy as np

from rcs_ac_v1_1.runtime.bev_spec import BEVSpec, bev_row_col_to_ego_xy, ego_xy_to_bev_row_col


def make_empty_bev(channels: int, spec: BEVSpec) -> np.ndarray:
    return np.zeros(spec.channel_shape(int(channels)), dtype=np.float32)


def draw_disk(channel: np.ndarray, row: int, col: int, radius_cells: int, value: float) -> None:
    radius = max(0, int(radius_cells))
    h, w = channel.shape
    r0, r1 = max(0, int(row) - radius), min(h - 1, int(row) + radius)
    c0, c1 = max(0, int(col) - radius), min(w - 1, int(col) + radius)
    rr, cc = np.ogrid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (rr - int(row)) ** 2 + (cc - int(col)) ** 2 <= radius**2
    patch = channel[r0 : r1 + 1, c0 : c1 + 1]
    if float(value) < 0.0:
        patch[mask] = np.minimum(patch[mask], np.float32(value))
    else:
        patch[mask] = np.maximum(patch[mask], np.float32(value))


def draw_polyline(
    channel: np.ndarray,
    points_ego: np.ndarray,
    spec: BEVSpec,
    *,
    radius_cells: int,
    value: float,
) -> None:
    points = np.asarray(points_ego, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        return
    if points.shape[0] == 1:
        row, col, valid = ego_xy_to_bev_row_col(float(points[0, 0]), float(points[0, 1]), spec)
        if valid:
            draw_disk(channel, row, col, radius_cells, value)
        return
    sample_step = max(float(spec.resolution_m) * 0.5, 1.0e-3)
    for start, end in zip(points[:-1], points[1:]):
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        length = float(np.linalg.norm(end - start))
        steps = max(1, int(math.ceil(length / sample_step)))
        for idx in range(steps + 1):
            point = start + (end - start) * (idx / steps)
            row, col, valid = ego_xy_to_bev_row_col(float(point[0]), float(point[1]), spec)
            if valid:
                draw_disk(channel, row, col, radius_cells, value)


def draw_gaussian(
    channel: np.ndarray,
    x_m: float,
    y_m: float,
    spec: BEVSpec,
    *,
    sigma_m: float,
    peak: float = 1.0,
) -> None:
    row, col, valid = ego_xy_to_bev_row_col(float(x_m), float(y_m), spec)
    if not valid:
        return
    sigma_cells = max(float(sigma_m) / float(spec.resolution_m), 1.0e-6)
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    h, w = channel.shape
    r0, r1 = max(0, row - radius), min(h - 1, row + radius)
    c0, c1 = max(0, col - radius), min(w - 1, col + radius)
    rr, cc = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
    dist2 = (rr - row) ** 2 + (cc - col) ** 2
    values = float(peak) * np.exp(-0.5 * dist2 / (sigma_cells**2))
    channel[r0 : r1 + 1, c0 : c1 + 1] = np.maximum(channel[r0 : r1 + 1, c0 : c1 + 1], values)


def rasterize_points(points_ego: np.ndarray, spec: BEVSpec) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_ego, dtype=np.float32)
    rows: list[int] = []
    cols: list[int] = []
    if points.ndim != 2 or points.shape[1] < 2:
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    for point in points:
        if not np.isfinite(point[:2]).all():
            continue
        row, col, valid = ego_xy_to_bev_row_col(float(point[0]), float(point[1]), spec)
        if valid:
            rows.append(row)
            cols.append(col)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def cell_center_grid(spec: BEVSpec) -> tuple[np.ndarray, np.ndarray]:
    xs = np.zeros(spec.shape_hw, dtype=np.float32)
    ys = np.zeros(spec.shape_hw, dtype=np.float32)
    for row in range(int(spec.height)):
        for col in range(int(spec.width)):
            x_m, y_m = bev_row_col_to_ego_xy(row, col, spec)
            xs[row, col] = x_m
            ys[row, col] = y_m
    return xs, ys
