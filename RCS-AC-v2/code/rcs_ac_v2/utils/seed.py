"""Reproducible seeding helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed
