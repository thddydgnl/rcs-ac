from __future__ import annotations

import numpy as np

from rcs_ac_v1_1.runtime.bev_spec import BEVSpec
from rcs_ac_v1_1.runtime.bev_rasterizer import make_empty_bev


def build_rgb_bev_unavailable(spec: BEVSpec) -> np.ndarray:
    return make_empty_bev(4, spec)


def build_rgb_bev_from_prediction(prediction: np.ndarray, spec: BEVSpec) -> np.ndarray:
    bev = np.asarray(prediction, dtype=np.float32)
    expected = spec.channel_shape(4)
    if tuple(bev.shape) != tuple(expected):
        raise ValueError(f"rgb BEV prediction expected shape {expected}, got {tuple(bev.shape)}")
    if not np.isfinite(bev).all():
        raise ValueError("rgb BEV prediction contains NaN or inf")
    return np.clip(bev, 0.0, 1.0).astype(np.float32, copy=False)
