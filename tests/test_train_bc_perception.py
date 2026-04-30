import torch

from rcs_ac_first.scripts.train_bc import _compute_frame_perception_losses


def test_frame_perception_loss_accepts_dataset_lane_target_key() -> None:
    outputs = {
        "action_mean": torch.zeros((2, 2), dtype=torch.float32),
        "pred_road_mask_logits": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "pred_lane_edge_mask_logits": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
    }
    batch = {
        "aux_road_mask_target": torch.ones((2, 1, 4, 4), dtype=torch.float32),
        "aux_lane_edge_target": torch.ones((2, 1, 4, 4), dtype=torch.float32),
        "aux_mask_targets_available": torch.ones((2,), dtype=torch.float32),
    }

    losses = _compute_frame_perception_losses(
        outputs=outputs,
        batch=batch,
        perception_aux_loss_weight=0.5,
        road_mask_weight=0.25,
        lane_edge_weight=1.0,
        lane_edge_pos_weight=3.0,
    )

    assert float(losses["perception_aux_loss"]) > 0.0
    assert float(losses["aux_mask_available_rate"]) == 1.0
    assert float(losses["aux_lane_edge_mask_loss"]) > 0.0
