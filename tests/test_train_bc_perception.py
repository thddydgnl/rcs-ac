import torch

from rcs_ac_first.scripts.train_bc import (
    _compute_corrective_delta_losses,
    _compute_frame_perception_losses,
    _compute_sequence_perception_losses,
    _compute_visual_zero_losses,
)


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


def test_sequence_perception_loss_masks_invalid_timesteps() -> None:
    outputs = {
        "action_mean_sequence": torch.zeros((1, 3, 2), dtype=torch.float32),
        "pred_road_mask_logits_sequence": torch.zeros((1, 3, 1, 4, 4), dtype=torch.float32),
        "pred_lane_edge_mask_logits_sequence": torch.zeros((1, 3, 1, 4, 4), dtype=torch.float32),
    }
    batch = {
        "aux_road_mask_target": torch.ones((1, 3, 1, 4, 4), dtype=torch.float32),
        "aux_lane_edge_target": torch.ones((1, 3, 1, 4, 4), dtype=torch.float32),
        "aux_mask_targets_available": torch.ones((1, 3), dtype=torch.float32),
    }

    losses = _compute_sequence_perception_losses(
        outputs=outputs,
        batch=batch,
        valid_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        perception_aux_loss_weight=0.5,
        road_mask_weight=0.25,
        lane_edge_weight=1.0,
        lane_edge_pos_weight=3.0,
    )

    assert float(losses["perception_aux_loss"]) > 0.0
    assert float(losses["aux_mask_available_rate"]) == torch.tensor([1.0, 1.0, 0.0]).mean().item()


def test_visual_zero_loss_uses_action_delta_sum_and_margin() -> None:
    action_mean = torch.tensor([[0.10, 0.20], [0.30, -0.40]])
    zero_action_mean = torch.tensor([[0.05, 0.10], [0.10, -0.10]])

    losses = _compute_visual_zero_losses(action_mean, zero_action_mean, visual_zero_margin=0.40)

    assert torch.isclose(losses["visual_zero_action_l1"], torch.tensor(0.325))
    assert torch.isclose(losses["visual_zero_steer_delta"], torch.tensor(0.125))
    assert torch.isclose(losses["visual_zero_longitudinal_delta"], torch.tensor(0.20))
    assert torch.isclose(losses["visual_zero_margin_loss"], torch.tensor(0.075))
    assert torch.isclose(losses["visual_zero_steer_margin_loss"], torch.tensor(0.0))


def test_visual_zero_sequence_loss_respects_valid_mask() -> None:
    action_mean = torch.tensor([[[0.10, 0.20], [0.30, -0.40]]])
    zero_action_mean = torch.tensor([[[0.00, 0.10], [0.00, 0.00]]])
    valid_mask = torch.tensor([[1.0, 0.0]])

    losses = _compute_visual_zero_losses(
        action_mean,
        zero_action_mean,
        visual_zero_margin=0.30,
        visual_zero_steer_margin=0.20,
        valid_mask=valid_mask,
    )

    assert torch.isclose(losses["visual_zero_action_l1"], torch.tensor(0.20))
    assert torch.isclose(losses["visual_zero_steer_delta"], torch.tensor(0.10))
    assert torch.isclose(losses["visual_zero_longitudinal_delta"], torch.tensor(0.10))
    assert torch.isclose(losses["visual_zero_margin_loss"], torch.tensor(0.10))
    assert torch.isclose(losses["visual_zero_steer_margin_loss"], torch.tensor(0.10))


def test_visual_zero_corrective_loss_uses_signed_window_mask() -> None:
    action_mean = torch.tensor([[0.40, 0.00], [-0.20, 0.00], [0.50, 0.00]])
    zero_action_mean = torch.zeros_like(action_mean)
    route_progress = torch.tensor([0.24, 0.24, 0.10])
    lane_offset = torch.tensor([-2.0, -2.0, -2.0])
    heading_error = torch.tensor([7.0, 7.0, 7.0])

    losses = _compute_visual_zero_losses(
        action_mean,
        zero_action_mean,
        visual_zero_route_min=0.20,
        visual_zero_route_max=0.30,
        visual_zero_abs_lane_offset_min=1.8,
        visual_zero_abs_heading_error_min=6.0,
        visual_zero_corrective_steer_margin=0.30,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
    )

    assert torch.isclose(losses["visual_zero_corrective_mask_rate"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(losses["visual_zero_corrective_steer_delta"], torch.tensor(0.20))
    assert torch.isclose(losses["visual_zero_corrective_sign_loss"], torch.tensor(0.10))


def test_visual_zero_corrective_loss_is_zero_without_masked_rows() -> None:
    action_mean = torch.tensor([[0.40, 0.00]])
    zero_action_mean = torch.zeros_like(action_mean)
    route_progress = torch.tensor([0.10])
    lane_offset = torch.tensor([-2.0])
    heading_error = torch.tensor([7.0])

    losses = _compute_visual_zero_losses(
        action_mean,
        zero_action_mean,
        visual_zero_route_min=0.20,
        visual_zero_route_max=0.30,
        visual_zero_abs_lane_offset_min=1.8,
        visual_zero_abs_heading_error_min=6.0,
        visual_zero_corrective_steer_margin=0.30,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
    )

    assert torch.isclose(losses["visual_zero_corrective_mask_rate"], torch.tensor(0.0))
    assert torch.isclose(losses["visual_zero_corrective_steer_delta"], torch.tensor(0.0))
    assert torch.isclose(losses["visual_zero_corrective_sign_loss"], torch.tensor(0.0))


def test_corrective_delta_loss_matches_policy_and_visual_delta_targets() -> None:
    action_mean = torch.tensor([[0.20, 0.0], [-0.10, 0.0], [0.30, 0.0]])
    zero_action_mean = torch.tensor([[0.05, 0.0], [0.00, 0.0], [0.10, 0.0]])
    batch = {
        "policy_action_steer": torch.tensor([0.00, -0.20, 0.10]),
        "target_corrective_delta": torch.tensor([0.20, 0.10, 0.20]),
        "target_corrective_delta_available": torch.tensor([1.0, 1.0, 0.0]),
    }

    losses = _compute_corrective_delta_losses(action_mean, zero_action_mean, batch)

    assert torch.isclose(losses["target_corrective_delta_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["target_corrective_delta_l1"], torch.tensor(0.0))
    assert torch.isclose(losses["target_corrective_delta_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["visual_zero_target_delta_loss"], torch.tensor(0.02125))
    assert torch.isclose(losses["visual_zero_target_delta_sign_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["target_corrective_delta_mask_rate"], torch.tensor(2.0 / 3.0))
