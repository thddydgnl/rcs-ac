from __future__ import annotations

import numpy as np
import torch

from rcs_ac_v1_1.scripts.evaluate_bev_pure_pursuit import REQUIRED_LOG_COLUMNS, synthetic_eval_step
from rcs_ac_v1_1.scripts.train_bev_bc import train_bev_bc


def _write_tiny_dataset(path, n: int = 2) -> None:
    arrays = {
        "rgb": np.zeros((n, 3, 90, 160), dtype=np.float32),
        "lidar_bev": np.zeros((n, 5, 128, 128), dtype=np.float32),
        "route_bev": np.zeros((n, 4, 128, 128), dtype=np.float32),
        "ego_bev": np.zeros((n, 3, 128, 128), dtype=np.float32),
        "ego_scalar": np.zeros((n, 6), dtype=np.float32),
        "route_scalar": np.zeros((n, 10), dtype=np.float32),
        "sensor_health": np.ones((n, 15), dtype=np.float32),
        "gt_bev_target": np.zeros((n, 5, 128, 128), dtype=np.float32),
        "expert_waypoints": np.tile(np.asarray([[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]], dtype=np.float32), (n, 1, 1)),
        "expert_target_speed": np.ones((n, 1), dtype=np.float32) * 10.0,
        "bev_target_available": np.ones(n, dtype=np.float32),
        "waypoint_target_available": np.ones(n, dtype=np.float32),
    }
    np.savez(path, **arrays)


def test_train_bev_bc_smoke_one_epoch_cpu(tmp_path) -> None:
    dataset = tmp_path / "tiny.npz"
    output_dir = tmp_path / "ckpt"
    _write_tiny_dataset(dataset)
    metrics = train_bev_bc(
        config_path="configs/v1_1/full_bev_smoke.yaml",
        dataset_path=dataset,
        output_dir=output_dir,
        epochs=1,
        batch_size=1,
        device="cpu",
        lr=1.0e-4,
    )
    assert (output_dir / "last.pt").exists()
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_synthetic_eval_step_and_required_columns() -> None:
    result = synthetic_eval_step("configs/v1_1/full_bev_smoke.yaml", device="cpu")
    assert "control" in result
    assert {"target_speed_capped_kmh", "used_synthetic_fallback"}.issubset(result["control"]["debug"])
    assert "bev_fusion_weight_rgb" in REQUIRED_LOG_COLUMNS
