import torch

from rcs_ac_first.models.perception_aux import RGBPerceptionActorFeatureExtractor
from rcs_ac_first.scripts.train_bc import (
    _compute_collision_cluster_route_offset_losses,
    _compute_corrective_delta_losses,
    _compute_early_severe_steer_losses,
    _compute_early_route_offset_losses,
    _compute_frame_perception_losses,
    _compute_route_progress_longitudinal_losses,
    _compute_route_aware_visual_center_losses,
    _compute_sequence_perception_losses,
    _compute_visual_center_corrective_losses,
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


def test_rgb_perception_raw_feature_indices_track_road_and_lane_x() -> None:
    extractor = RGBPerceptionActorFeatureExtractor()
    road_logits = torch.full((1, 1, 4, 5), -20.0)
    lane_logits = torch.full((1, 1, 4, 5), -20.0)
    road_logits[:, :, 2:, 0] = 20.0
    lane_logits[:, :, 2:, 4] = 20.0

    raw = extractor._build_raw_features(road_logits, lane_logits)

    assert raw.shape == (1, 14)
    assert raw[0, 10] < -0.99
    assert raw[0, 11] > 0.99


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


def test_visual_center_corrective_loss_aligns_visual_delta_to_target() -> None:
    action_mean = torch.tensor([[0.20, 0.0], [-0.20, 0.0], [0.10, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)
    outputs = {
        "rgb_perception_actor_raw_features": torch.tensor(
            [
                [0.0] * 10 + [0.0, -0.40, 0.0, 0.0],
                [0.0] * 10 + [0.0, 0.40, 0.0, 0.0],
                [0.0] * 10 + [0.0, 0.40, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }
    batch = {
        "target_corrective_delta": torch.tensor([0.20, -0.20, 0.20]),
        "target_corrective_delta_available": torch.ones((3,), dtype=torch.float32),
    }

    losses = _compute_visual_center_corrective_losses(
        action_mean,
        zero_action_mean,
        outputs,
        batch,
        visual_center_error_min=0.05,
        visual_center_corrective_margin=0.30,
    )

    assert torch.isclose(losses["visual_center_corrective_mask_rate"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(losses["visual_center_target_consistency_rate"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(losses["visual_center_target_consistency_mask_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["visual_center_corrective_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["visual_center_correction_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["visual_center_corrective_sign_loss"], torch.tensor(0.10))


def test_visual_center_corrective_loss_respects_sequence_valid_mask() -> None:
    action_mean = torch.tensor([[[0.20, 0.0], [0.20, 0.0]]])
    zero_action_mean = torch.zeros_like(action_mean)
    outputs = {
        "rgb_perception_actor_raw_features_sequence": torch.tensor(
            [[[0.0] * 10 + [0.0, -0.40, 0.0, 0.0], [0.0] * 10 + [0.0, -0.40, 0.0, 0.0]]],
            dtype=torch.float32,
        )
    }
    batch = {
        "target_corrective_delta": torch.tensor([[0.20, 0.20]]),
        "target_corrective_delta_available": torch.ones((1, 2), dtype=torch.float32),
    }

    losses = _compute_visual_center_corrective_losses(
        action_mean,
        zero_action_mean,
        outputs,
        batch,
        visual_center_error_min=0.05,
        visual_center_corrective_margin=0.30,
        valid_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    assert torch.isclose(losses["visual_center_corrective_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["visual_center_target_consistency_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["visual_center_target_consistency_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["visual_center_corrective_sign_rate"], torch.tensor(1.0))


def test_visual_center_corrective_loss_is_zero_without_raw_features() -> None:
    action_mean = torch.tensor([[0.20, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)
    batch = {
        "target_corrective_delta": torch.tensor([0.20]),
        "target_corrective_delta_available": torch.ones((1,), dtype=torch.float32),
    }

    losses = _compute_visual_center_corrective_losses(action_mean, zero_action_mean, {}, batch)

    assert torch.isclose(losses["visual_center_corrective_sign_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["visual_center_corrective_mask_rate"], torch.tensor(0.0))


def test_route_aware_visual_center_loss_uses_route_window() -> None:
    action_mean = torch.tensor([[0.20, 0.0], [0.20, 0.0], [0.20, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)
    outputs = {
        "rgb_perception_actor_raw_features": torch.tensor(
            [
                [0.0] * 10 + [0.0, -0.40, 0.0, 0.0],
                [0.0] * 10 + [0.0, -0.40, 0.0, 0.0],
                [0.0] * 10 + [0.0, -0.40, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }
    batch = {
        "route_progress": torch.tensor([0.24, 0.10, 0.40]),
        "target_corrective_delta": torch.tensor([0.20, 0.20, 0.20]),
        "target_corrective_delta_available": torch.ones((3,), dtype=torch.float32),
    }

    losses = _compute_route_aware_visual_center_losses(
        action_mean,
        zero_action_mean,
        outputs,
        batch,
        route_min=0.205,
        route_max=0.36,
        visual_center_error_min=0.05,
        margin=0.30,
    )

    assert torch.isclose(losses["route_aware_visual_center_mask_rate"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(losses["route_aware_visual_center_target_consistency_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["route_aware_visual_center_target_consistency_mask_rate"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(losses["route_aware_visual_center_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["route_aware_visual_center_correction_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["route_aware_visual_center_loss"], torch.tensor(0.10))


def test_route_aware_visual_center_loss_respects_sequence_valid_mask() -> None:
    action_mean = torch.tensor([[[0.20, 0.0], [0.20, 0.0]]])
    zero_action_mean = torch.zeros_like(action_mean)
    outputs = {
        "rgb_perception_actor_raw_features_sequence": torch.tensor(
            [[[0.0] * 10 + [0.0, -0.40, 0.0, 0.0], [0.0] * 10 + [0.0, -0.40, 0.0, 0.0]]],
            dtype=torch.float32,
        )
    }
    batch = {
        "route_progress": torch.tensor([[0.24, 0.24]]),
        "target_corrective_delta": torch.tensor([[0.20, 0.20]]),
        "target_corrective_delta_available": torch.ones((1, 2), dtype=torch.float32),
    }

    losses = _compute_route_aware_visual_center_losses(
        action_mean,
        zero_action_mean,
        outputs,
        batch,
        route_min=0.205,
        route_max=0.36,
        visual_center_error_min=0.05,
        margin=0.30,
        valid_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    assert torch.isclose(losses["route_aware_visual_center_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["route_aware_visual_center_target_consistency_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["route_aware_visual_center_target_consistency_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["route_aware_visual_center_loss"], torch.tensor(0.10))


def test_route_aware_visual_center_loss_reports_inconsistent_target_mask() -> None:
    action_mean = torch.tensor([[0.20, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)
    outputs = {
        "rgb_perception_actor_raw_features": torch.tensor(
            [[0.0] * 10 + [0.0, -0.40, 0.0, 0.0]],
            dtype=torch.float32,
        )
    }
    batch = {
        "route_progress": torch.tensor([0.18]),
        "target_corrective_delta": torch.tensor([-0.20]),
        "target_corrective_delta_available": torch.ones((1,), dtype=torch.float32),
    }

    losses = _compute_route_aware_visual_center_losses(
        action_mean,
        zero_action_mean,
        outputs,
        batch,
        route_min=0.16,
        route_max=0.20,
        visual_center_error_min=0.05,
        margin=0.30,
    )

    assert torch.isclose(losses["route_aware_visual_center_target_consistency_rate"], torch.tensor(0.0))
    assert torch.isclose(losses["route_aware_visual_center_target_consistency_mask_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["route_aware_visual_center_mask_rate"], torch.tensor(0.0))
    assert torch.isclose(losses["route_aware_visual_center_loss"], torch.tensor(0.0))


def test_collision_cluster_route_offset_loss_uses_exact_signed_mask() -> None:
    action_mean = torch.tensor([[0.02, 0.0], [-0.02, 0.0], [-0.03, 0.0], [0.02, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)
    route_progress = torch.tensor([0.18, 0.18, 0.22, 0.18])
    lane_offset = torch.tensor([1.7, 1.7, 1.7, -1.7])
    heading_error = torch.tensor([-16.0, -16.0, -16.0, -16.0])

    losses = _compute_collision_cluster_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=route_progress,
        lane_offset=lane_offset,
        heading_error=heading_error,
        route_min=0.16,
        route_max=0.20,
        lane_offset_min=1.5,
        heading_error_max=-8.0,
        margin=0.01,
    )

    assert torch.isclose(losses["collision_cluster_route_offset_agreement_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_sign_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_signed_delta_mean"], torch.tensor(0.0))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_loss"], torch.tensor(0.015))


def test_collision_cluster_route_offset_loss_can_use_symmetric_lane_and_heading_mask() -> None:
    action_mean = torch.tensor([[0.02, 0.0], [-0.02, 0.0], [-0.02, 0.0], [0.02, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)

    losses = _compute_collision_cluster_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=torch.tensor([0.18, 0.18, 0.18, 0.22]),
        lane_offset=torch.tensor([-1.7, -1.7, 1.7, -1.7]),
        heading_error=torch.tensor([16.0, 16.0, -16.0, 16.0]),
        route_min=0.16,
        route_max=0.20,
        lane_offset_min=1.5,
        heading_error_max=-8.0,
        margin=0.01,
        lane_offset_mode="abs",
        heading_error_mode="abs",
    )

    assert torch.isclose(losses["collision_cluster_route_offset_agreement_mask_rate"], torch.tensor(0.75))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_sign_rate"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(
        losses["collision_cluster_route_offset_agreement_signed_delta_mean"],
        torch.tensor(-0.02 / 3.0),
    )
    assert torch.isclose(
        losses["collision_cluster_route_offset_agreement_corrected_delta_mean"],
        torch.tensor(0.02 / 3.0),
    )
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_positive_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_negative_sign_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_positive_mask_rate"], torch.tensor(0.25))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_negative_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_loss"], torch.tensor(0.01))


def test_collision_cluster_route_offset_loss_respects_sequence_valid_mask() -> None:
    action_mean = torch.tensor([[[0.02, 0.0], [-0.02, 0.0]]])
    zero_action_mean = torch.zeros_like(action_mean)

    losses = _compute_collision_cluster_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=torch.tensor([[0.18, 0.18]]),
        lane_offset=torch.tensor([[1.7, 1.7]]),
        heading_error=torch.tensor([[-16.0, -16.0]]),
        route_min=0.16,
        route_max=0.20,
        lane_offset_min=1.5,
        heading_error_max=-8.0,
        margin=0.01,
        valid_mask=torch.tensor([[1.0, 0.0]]),
    )

    assert torch.isclose(losses["collision_cluster_route_offset_agreement_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_sign_rate"], torch.tensor(0.0))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_signed_delta_mean"], torch.tensor(0.02))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_loss"], torch.tensor(0.03))


def test_collision_cluster_route_offset_loss_is_zero_without_fields() -> None:
    action_mean = torch.tensor([[0.02, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)

    losses = _compute_collision_cluster_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=None,
        lane_offset=torch.tensor([1.7]),
        heading_error=torch.tensor([-16.0]),
        route_min=0.16,
        route_max=0.20,
        lane_offset_min=1.5,
        heading_error_max=-8.0,
        margin=0.01,
    )

    assert torch.isclose(losses["collision_cluster_route_offset_agreement_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["collision_cluster_route_offset_agreement_mask_rate"], torch.tensor(0.0))


def test_early_severe_steer_loss_uses_target_delta_and_balances_sides() -> None:
    action_mean = torch.tensor([[0.20, 0.0], [-0.10, 0.0], [0.00, 0.0], [0.05, 0.0]])
    batch = {
        "policy_action_steer": torch.zeros((4,), dtype=torch.float32),
        "target_corrective_delta": torch.tensor([0.30, -0.30, 0.30, -0.30]),
        "target_corrective_delta_available": torch.ones((4,), dtype=torch.float32),
    }

    losses = _compute_early_severe_steer_losses(
        action_mean,
        batch,
        route_progress=torch.tensor([0.05, 0.05, 0.05, 0.12]),
        lane_offset=torch.tensor([-1.2, 1.2, -1.2, 1.2]),
        heading_error=torch.tensor([12.0, -12.0, 12.0, -12.0]),
        route_min=0.025,
        route_max=0.10,
        abs_lane_offset_min=0.55,
        abs_heading_error_min=3.0,
        margin=0.05,
        side_balanced=True,
    )

    assert torch.isclose(losses["early_severe_steer_mask_rate"], torch.tensor(0.75))
    assert torch.isclose(losses["early_severe_steer_left_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_severe_steer_right_mask_rate"], torch.tensor(0.25))
    assert torch.isclose(losses["early_severe_steer_sign_rate"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(losses["early_severe_steer_left_sign_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_severe_steer_right_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["early_severe_steer_delta_l1"], torch.tensor(0.2))
    assert torch.isclose(losses["early_severe_absolute_steer_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["early_severe_absolute_steer_sign_rate"], torch.tensor(1.0))


def test_early_severe_steer_loss_respects_sequence_valid_mask() -> None:
    action_mean = torch.tensor([[[0.20, 0.0], [0.20, 0.0]]])
    batch = {
        "policy_action_steer": torch.zeros((1, 2), dtype=torch.float32),
        "target_corrective_delta": torch.tensor([[0.30, 0.30]]),
        "target_corrective_delta_available": torch.ones((1, 2), dtype=torch.float32),
    }

    losses = _compute_early_severe_steer_losses(
        action_mean,
        batch,
        route_progress=torch.tensor([[0.05, 0.05]]),
        lane_offset=torch.tensor([[-1.2, -1.2]]),
        heading_error=torch.tensor([[12.0, 12.0]]),
        route_min=0.025,
        route_max=0.10,
        abs_lane_offset_min=0.55,
        abs_heading_error_min=3.0,
        margin=0.05,
        valid_mask=torch.tensor([[1.0, 0.0]]),
    )

    assert torch.isclose(losses["early_severe_steer_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_severe_steer_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(losses["early_severe_steer_delta_l1"], torch.tensor(0.1))


def test_early_severe_steer_loss_semantic_mode_keeps_heading_dominant_rows() -> None:
    action_mean = torch.tensor([[-0.04, 0.0], [-0.04, 0.0], [0.04, 0.0]])
    batch = {
        "policy_action_steer": torch.zeros((3,), dtype=torch.float32),
        "target_corrective_delta": torch.tensor([-0.08, -0.08, 0.08]),
        "target_corrective_delta_available": torch.ones((3,), dtype=torch.float32),
    }
    kwargs = {
        "route_progress": torch.tensor([0.05, 0.05, 0.05]),
        "lane_offset": torch.tensor([-0.20, -0.20, 0.20]),
        "heading_error": torch.tensor([-8.0, -8.0, 8.0]),
        "route_min": 0.025,
        "route_max": 0.10,
        "abs_lane_offset_min": 0.55,
        "abs_heading_error_min": 3.0,
        "margin": 0.02,
    }

    lane_losses = _compute_early_severe_steer_losses(action_mean, batch, **kwargs)
    semantic_losses = _compute_early_severe_steer_losses(
        action_mean,
        batch,
        alignment_mode="semantic",
        **kwargs,
    )

    assert torch.isclose(lane_losses["early_severe_steer_mask_rate"], torch.tensor(0.0))
    assert torch.isclose(semantic_losses["early_severe_steer_mask_rate"], torch.tensor(1.0))
    assert torch.isclose(semantic_losses["early_severe_steer_sign_rate"], torch.tensor(1.0))
    assert torch.isclose(semantic_losses["early_severe_steer_delta_l1"], torch.tensor(0.04))


def test_early_severe_absolute_steer_loss_pushes_raw_actor_toward_recovery_side() -> None:
    action_mean = torch.tensor([[-0.06, 0.0], [0.04, 0.0], [0.02, 0.0], [-0.03, 0.0]])
    batch = {
        "policy_action_steer": torch.zeros((4,), dtype=torch.float32),
        "target_corrective_delta": torch.tensor([0.20, 0.20, -0.20, -0.20]),
        "target_corrective_delta_available": torch.ones((4,), dtype=torch.float32),
    }

    losses = _compute_early_severe_steer_losses(
        action_mean,
        batch,
        route_progress=torch.tensor([0.05, 0.05, 0.05, 0.12]),
        lane_offset=torch.tensor([-1.0, -1.0, 1.0, 1.0]),
        heading_error=torch.tensor([8.0, 8.0, -8.0, -8.0]),
        route_min=0.025,
        route_max=0.10,
        abs_lane_offset_min=0.55,
        abs_heading_error_min=3.0,
        margin=0.02,
        absolute_steer_margin=0.0,
        side_balanced=True,
        alignment_mode="semantic",
    )

    assert torch.isclose(losses["early_severe_steer_mask_rate"], torch.tensor(0.75))
    assert torch.isclose(losses["early_severe_absolute_steer_loss"], torch.tensor(0.025))
    assert torch.isclose(losses["early_severe_absolute_steer_sign_rate"], torch.tensor(1.0 / 3.0))


def test_early_route_offset_loss_targets_left_drift_right_correction() -> None:
    action_mean = torch.tensor([[0.02, 0.0], [-0.02, 0.0], [0.02, 0.0], [0.02, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)

    losses = _compute_early_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=torch.tensor([0.06, 0.06, 0.11, 0.06]),
        lane_offset=torch.tensor([-2.5, -2.5, -2.5, 2.5]),
        heading_error=torch.tensor([20.0, 20.0, 20.0, 20.0]),
        route_min=0.025,
        route_max=0.085,
        lane_offset_min=1.5,
        heading_error_min=12.0,
        margin=0.01,
        lane_offset_mode="negative",
        heading_error_mode="positive",
    )

    assert torch.isclose(losses["early_route_offset_agreement_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_route_offset_agreement_sign_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_route_offset_agreement_signed_delta_mean"], torch.tensor(0.0))
    assert torch.isclose(losses["early_route_offset_agreement_corrected_delta_mean"], torch.tensor(0.0))
    assert torch.isclose(losses["early_route_offset_agreement_negative_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_route_offset_agreement_loss"], torch.tensor(0.015))


def test_early_route_offset_loss_is_zero_after_early_window() -> None:
    action_mean = torch.tensor([[-0.02, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)

    losses = _compute_early_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=torch.tensor([0.10]),
        lane_offset=torch.tensor([-2.5]),
        heading_error=torch.tensor([20.0]),
        route_min=0.025,
        route_max=0.085,
        lane_offset_min=1.5,
        heading_error_min=12.0,
        margin=0.01,
        lane_offset_mode="negative",
        heading_error_mode="positive",
    )

    assert torch.isclose(losses["early_route_offset_agreement_loss"], torch.tensor(0.0))
    assert torch.isclose(losses["early_route_offset_agreement_mask_rate"], torch.tensor(0.0))


def test_route_offset_losses_keep_cluster_and_early_metrics_separate() -> None:
    action_mean = torch.tensor([[0.02, 0.0], [-0.02, 0.0]])
    zero_action_mean = torch.zeros_like(action_mean)
    batch = {
        "route_progress": torch.tensor([0.06, 0.18]),
        "lane_offset": torch.tensor([-2.5, 1.7]),
        "heading_error": torch.tensor([20.0, -16.0]),
    }

    losses = _compute_collision_cluster_route_offset_losses(
        action_mean,
        zero_action_mean,
        route_progress=batch["route_progress"],
        lane_offset=batch["lane_offset"],
        heading_error=batch["heading_error"],
        route_min=0.16,
        route_max=0.20,
        lane_offset_min=1.5,
        heading_error_max=-8.0,
        margin=0.01,
    )
    losses.update(
        _compute_early_route_offset_losses(
            action_mean,
            zero_action_mean,
            route_progress=batch["route_progress"],
            lane_offset=batch["lane_offset"],
            heading_error=batch["heading_error"],
            route_min=0.025,
            route_max=0.085,
            lane_offset_min=1.5,
            heading_error_min=12.0,
            margin=0.01,
            lane_offset_mode="negative",
            heading_error_mode="positive",
        )
    )

    assert torch.isclose(losses["collision_cluster_route_offset_agreement_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["early_route_offset_agreement_mask_rate"], torch.tensor(0.5))
    assert "collision_cluster_route_offset_agreement_loss" in losses
    assert "early_route_offset_agreement_loss" in losses


def test_route_progress_longitudinal_floor_loss_uses_route_and_speed_mask() -> None:
    action_mean = torch.tensor([[0.20, -0.20], [0.20, 0.10], [0.20, -0.40]])
    auxiliary = torch.zeros((3, 30), dtype=torch.float32)
    auxiliary[:, 15] = torch.tensor([2.0, 5.0, 2.0])
    batch = {
        "route_progress": torch.tensor([0.20, 0.20, 0.05]),
        "auxiliary": auxiliary,
    }

    losses = _compute_route_progress_longitudinal_losses(
        action_mean,
        batch,
        route_min=0.10,
        route_max=0.32,
        speed_max_mps=3.5,
        longitudinal_floor=0.05,
    )

    assert torch.isclose(losses["route_progress_longitudinal_mask_rate"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(losses["route_progress_longitudinal_floor_loss"], torch.tensor(0.25))
    assert torch.isclose(losses["route_progress_longitudinal_floor_rate"], torch.tensor(0.0))
    assert torch.isclose(losses["route_progress_longitudinal_mean"], torch.tensor(-0.20))


def test_route_progress_longitudinal_floor_loss_respects_sequence_valid_mask() -> None:
    action_mean = torch.tensor([[[0.20, -0.20], [0.20, 0.10]]])
    auxiliary = torch.zeros((1, 2, 30), dtype=torch.float32)
    auxiliary[..., 15] = 2.0
    batch = {
        "route_progress": torch.tensor([[0.20, 0.20]]),
        "auxiliary": auxiliary,
    }

    losses = _compute_route_progress_longitudinal_losses(
        action_mean,
        batch,
        route_min=0.10,
        route_max=0.32,
        speed_max_mps=3.5,
        longitudinal_floor=0.05,
        valid_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    assert torch.isclose(losses["route_progress_longitudinal_mask_rate"], torch.tensor(0.5))
    assert torch.isclose(losses["route_progress_longitudinal_floor_loss"], torch.tensor(0.25))
