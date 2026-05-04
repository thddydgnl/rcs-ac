from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BEVSpec:
    """Ego-centered BEV grid definition used by RCS-AC v1.1.

    Coordinates follow the CARLA ego convention used by the project docs:
    ``x`` is forward meters and ``y`` is left/right meters. The BEV tensor is
    represented as ``[C, H, W]``.
    """

    height: int = 128
    width: int = 128
    resolution_m: float = 0.375
    front_m: float = 40.0
    rear_m: float = 8.0
    left_m: float = 24.0
    right_m: float = 24.0
    strict_extent_check: bool = True
    extent_tolerance_m: float = 1.0e-4

    def __post_init__(self) -> None:
        for name in ("height", "width"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("resolution_m",):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        for name in ("front_m", "rear_m", "left_m", "right_m"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if float(self.extent_tolerance_m) < 0.0:
            raise ValueError("extent_tolerance_m must be >= 0")
        self.validate()

    @property
    def x_extent_m(self) -> float:
        return float(self.front_m) + float(self.rear_m)

    @property
    def y_extent_m(self) -> float:
        return float(self.left_m) + float(self.right_m)

    @property
    def grid_x_extent_m(self) -> float:
        return int(self.height) * float(self.resolution_m)

    @property
    def grid_y_extent_m(self) -> float:
        return int(self.width) * float(self.resolution_m)

    @property
    def ego_row_float(self) -> float:
        return float(self.front_m) / float(self.resolution_m)

    @property
    def ego_col_float(self) -> float:
        return float(self.left_m) / float(self.resolution_m)

    def validate(self) -> None:
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError(f"Invalid BEV shape: height={self.height}, width={self.width}")
        if float(self.resolution_m) <= 0.0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")
        if min(float(self.front_m), float(self.rear_m), float(self.left_m), float(self.right_m)) < 0.0:
            raise ValueError("BEV extents must be non-negative")
        if not bool(self.strict_extent_check):
            return

        x_err = abs(self.grid_x_extent_m - self.x_extent_m)
        y_err = abs(self.grid_y_extent_m - self.y_extent_m)
        tolerance = float(self.extent_tolerance_m)
        if x_err > tolerance:
            raise ValueError(
                "BEV x extent mismatch: "
                f"height * resolution_m = {self.grid_x_extent_m:.6f}, "
                f"front_m + rear_m = {self.x_extent_m:.6f}, "
                f"error = {x_err:.6f}"
            )
        if y_err > tolerance:
            raise ValueError(
                "BEV y extent mismatch: "
                f"width * resolution_m = {self.grid_y_extent_m:.6f}, "
                f"left_m + right_m = {self.y_extent_m:.6f}, "
                f"error = {y_err:.6f}"
            )

    @property
    def shape_hw(self) -> tuple[int, int]:
        return int(self.height), int(self.width)

    def channel_shape(self, channels: int) -> tuple[int, int, int]:
        if int(channels) <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        return int(channels), int(self.height), int(self.width)


def ego_xy_to_bev_row_col(x_m: float, y_m: float, spec: BEVSpec | None = None) -> tuple[int, int, bool]:
    spec = spec or BEVSpec()
    spec.validate()
    row_float = (float(spec.front_m) - float(x_m)) / float(spec.resolution_m)
    col_float = (float(spec.left_m) - float(y_m)) / float(spec.resolution_m)
    valid = (
        -float(spec.extent_tolerance_m) <= row_float <= float(spec.height) + float(spec.extent_tolerance_m)
        and -float(spec.extent_tolerance_m) <= col_float <= float(spec.width) + float(spec.extent_tolerance_m)
    )
    row = int(math.floor(row_float))
    col = int(math.floor(col_float))
    if valid:
        row = min(max(row, 0), int(spec.height) - 1)
        col = min(max(col, 0), int(spec.width) - 1)
    return row, col, bool(valid)


def bev_row_col_to_ego_xy(row: int, col: int, spec: BEVSpec | None = None) -> tuple[float, float]:
    spec = spec or BEVSpec()
    spec.validate()
    x_m = float(spec.front_m) - (int(row) + 0.5) * float(spec.resolution_m)
    y_m = float(spec.left_m) - (int(col) + 0.5) * float(spec.resolution_m)
    return x_m, y_m


@dataclass(frozen=True)
class BEVChannelSpec:
    rgb_channels: int = 4
    lidar_channels: int = 5
    route_channels: int = 4
    ego_channels: int = 3
    target_channels: int = 5

    def __post_init__(self) -> None:
        for name in (
            "rgb_channels",
            "lidar_channels",
            "route_channels",
            "ego_channels",
            "target_channels",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")


DEFAULT_BEV_SPEC = BEVSpec()
DEFAULT_BEV_CHANNELS = BEVChannelSpec()

SENSOR_BEV_INPUT_CHANNELS: tuple[str, ...] = (
    "rgb_drivable_area_pred",
    "rgb_lane_marking_pred",
    "rgb_road_boundary_pred",
    "rgb_bev_confidence",
    "lidar_occupancy",
    "lidar_free_space",
    "lidar_unknown",
    "lidar_height",
    "lidar_density",
    "route_path",
    "target_waypoint_heatmap",
    "turn_region",
    "speed_hint",
    "ego_mask",
    "past_ego_trajectory",
    "previous_control_trail",
)

GT_BEV_TARGET_CHANNELS: tuple[str, ...] = (
    "drivable_area_gt",
    "lane_marking_gt",
    "obstacle_occupancy_gt",
    "free_space_gt",
    "route_path_gt",
)
