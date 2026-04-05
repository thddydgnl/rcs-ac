from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confirm_utils import build_confirm_config
from ppo_baselines.models.fusion import AttentionFusion
from rcs_ac_v1.model import MODALITY_NAMES, RecurrentAttentionActorCritic
from run_stage2_reliability_proof import (
    build_reliability_proof_score_summary,
    compute_candidate_reliability_delta,
    compute_reliability_target_metrics,
    evaluate_reliability_proof_summary,
    run_reliability_proof_candidate,
)
from run_stage2_ego_reliability_anchor_pilot import choose_ego_anchor_recommendation
from run_stage2_ego_reliability_fix_pilot import choose_ego_fix_recommendation
from run_stage2_ego_reliability_residual_pilot import choose_ego_residual_recommendation
from run_stage2_fusion_shift_proof_pilot import (
    build_fusion_shift_proof_score_summary,
    choose_fusion_shift_proof_recommendation,
    evaluate_fusion_shift_proof_summary,
    generate_fusion_shift_proof_report,
)
from run_stage2_reliability_selective_residual_pilot import choose_selective_residual_recommendation
from run_very_weak_probe import compare_clean_vs_corrupted_reliability
from stage1_warmup.model import Stage1WarmupActorCritic
from stage2_transition_helpers import (
    Stage2TransitionTrainer,
    _probe_logging_fields,
    build_stage2_candidate_effective_config,
    build_run_cache_metadata,
    build_stage2_reference_anchored_targets,
    compute_attention_bias_gate_effects,
    compute_stage2_ego_positive_losses,
    compute_stage2_fusion_shift_losses,
    compute_stage2_fusion_shift_gate_effect_losses,
    compute_stage2_fusion_shift_prebias_losses,
    compute_stage2_teacher_gate_effect_losses,
    compute_stage2_teacher_actuation_regime_state,
    compute_stage2_teacher_preserve_signal_absmax,
    compute_stage2_local_residual_losses,
    compute_stage2_modality_positive_losses,
    compute_stage2_teacher_distilled_gate_losses,
    infer_experiment_mode,
    is_calibration_only,
    make_probe_rng,
    run_stage2_candidate,
    run_stage2_eval_only_candidate,
    select_stage2_gate_teacher_preserve_indices,
)
from v3_shapley_helpers import (
    apply_attention_loss_overrides_to_model,
    build_candidate_gate_thresholds,
    build_shapley_compute_kwargs,
    build_v3_1b_targetlogit_summary,
    build_v3_7f2_2_pass_distance_dashboard,
    build_v3_7f2_5_pass_distance_dashboard,
    build_v3_7f2_6_pass_distance_dashboard,
    build_v3_7f2_7_pass_distance_dashboard,
    build_v3_7f2_8_pass_distance_dashboard,
    build_v3_7f2_10_pass_distance_dashboard,
    build_v3_7f2_11_pass_distance_dashboard,
    build_v3_7f2_12_pass_distance_dashboard,
    build_v3_7f2_13_candidate_score_summary,
    build_v3_7f2_13_pass_distance_dashboard,
    canonicalize_gate_loss_keys,
    compute_cancel_gate_floor_loss,
    compute_consistency_metrics,
    compute_degradation_metrics,
    compute_gate_feature_separability_audit,
    compute_final_gate_target_loss,
    compute_projection_norm_floor_loss,
    compute_relative_gap_anchor_loss,
    compute_relative_gap_margin_loss,
    compute_score_adapter_anchor_loss,
    compute_adapter_delta_margin_loss,
    compute_adapter_gate_margin_loss,
    compute_adapter_gate_target_loss,
    compute_teacher_reproduction_metrics_from_rows,
    compute_paired_selective_gate_losses,
    compute_residual_logit_target_loss,
    compute_v3_7f2_5_gate_score_adapter_micro_unfreeze_metrics,
    compute_v3_7f2_6_adapter_gate_coupling_metrics,
    compute_v3_7f2_7_adapter_readout_metrics,
    compute_v3_7f2_8_output_scale_metrics,
    compute_v3_7f2_10_cancel_contrast_metrics,
    compute_v3_7f2_11_cancel_rescue_metrics,
    compute_v3_7f2_12_excess_cancel_selective_rescue_metrics,
    compute_v3_7f2_13_anchor_cancel_selector_rescue_metrics,
    compute_exact_counterfactual_shapley,
    compute_v3_7e_gate_only_metrics,
    compute_v3_7f1_source_gate_distilled_residual_metrics,
    compute_v3_7f_source_gate_residual_metrics,
    compute_selector_pair_margin_loss,
    compute_tiebreak_metrics,
    configure_trainable_gate_actuation_parameters,
    configure_trainable_gate_logit_offset,
    configure_trainable_gate_residual_alpha,
    generate_v3_7_report,
    merge_teacher_loss_overrides,
    normalize_tiebreak_weights,
    resolve_experiment_output_root,
    select_gate_teacher_preserve_indices,
    validate_run_completeness,
    validate_teacher_reproduction_metrics,
)
import v3_shapley_helpers as v3_helpers


def _build_stage1_model() -> Stage1WarmupActorCritic:
    return Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=False,
        reliability_ego_raw_adapter_dim=8,
        reliability_ego_residual_input_mode="raw_only",
        reliability_ego_residual_apply_to_provisional=True,
        action_dim=2,
        action_log_std_init=-0.5,
    )


def _sample_obs() -> dict[str, torch.Tensor]:
    return {
        "vision": torch.randn(1, 8),
        "distance": torch.randn(1, 9),
        "ego": torch.randn(1, 5),
        "route": torch.randn(1, 5),
    }


def _sample_fused(batch_size: int = 1, width: int = 64) -> torch.Tensor:
    return torch.randn(batch_size, width)


def test_import_canonicalization() -> None:
    model = _build_stage1_model()
    assert Stage1WarmupActorCritic.__mro__[1] is RecurrentAttentionActorCritic
    assert model.fusion_module.__class__ is AttentionFusion
    assert model.fusion_module.__class__.__module__ == "ppo_baselines.models.fusion"
    assert RecurrentAttentionActorCritic.__module__ == "rcs_ac_v1.model"


def test_shapley_return_semantics() -> None:
    model = _build_stage1_model()
    state_obs = _sample_obs()
    clean_obs = {key: value + 0.1 for key, value in state_obs.items()}
    raw_values, target = compute_exact_counterfactual_shapley(
        model,
        state_obs,
        clean_reference_obs=clean_obs,
        temperature=0.25,
        return_raw=True,
    )
    target_only = compute_exact_counterfactual_shapley(
        model,
        state_obs,
        clean_reference_obs=clean_obs,
        temperature=0.25,
        return_raw=False,
    )
    assert isinstance(target_only, np.ndarray)
    assert raw_values.shape == target.shape == target_only.shape == (4,)
    assert np.allclose(target, target_only)
    assert not np.allclose(raw_values, target)


def test_deterministic_probe_seed() -> None:
    rng_a = make_probe_rng(7, 9000, "mild_vision")
    rng_b = make_probe_rng(7, 9000, "mild_vision")
    seq_a = rng_a.integers(0, 100000, size=16)
    seq_b = rng_b.integers(0, 100000, size=16)
    assert np.array_equal(seq_a, seq_b)


def test_confirm_gating_preserves_baseline() -> None:
    config = {
        "baseline_candidate": "reference",
        "candidates": {
            "reference": {"eval_only": True},
            "winner": {"eval_only": False},
        },
        "candidate_order": ["reference", "winner"],
    }
    confirm = build_confirm_config(config, "winner")
    assert confirm["baseline_candidate"] == "reference"
    assert confirm["comparison_baseline_candidate"] == "reference"
    assert confirm["evaluation_candidate"] == "winner"
    assert confirm["candidate_order"] == ["reference", "winner"]


def test_frozen_teacher_like_freeze_scope() -> None:
    model = _build_stage1_model()
    model.freeze_attention_clean_preserving_bias_calibration_only(train_score_adapter=False)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all("fusion_module.reliability_bias_head" in name for name in trainable)


def test_config_wiring_for_shapley_kwargs() -> None:
    losses_cfg = {
        "shapley_reference_mode": "clean_full",
        "shapley_counterfactual_baseline": "zero",
        "shapley_utility": "fused_feature_l2_to_full",
        "shapley_estimator": "sampled",
        "shapley_num_subsets": 5,
        "shapley_temperature": 0.5,
    }
    kwargs = build_shapley_compute_kwargs(losses_cfg, rng_seed=123, return_raw=True)
    assert kwargs["reference_mode"] == "clean_full"
    assert kwargs["counterfactual_baseline"] == "zero"
    assert kwargs["utility_mode"] == "clean_fused_l2"
    assert kwargs["estimator"] == "sampled"
    assert kwargs["num_subsets"] == 5
    assert kwargs["rng_seed"] == 123


def test_relative_gap_modes() -> None:
    fusion = AttentionFusion(feature_dim=64, hidden_dim=32)
    stacked = torch.randn(2, 4, 64)
    fusion.use_relative_score_head = True
    fusion.relative_gap_residual_scale = 0.2

    fusion.relative_gap_mode = "vision_anchor"
    scores_a, _, _, gaps_a, *_ = fusion.compute_scores(stacked)
    assert scores_a.shape == (2, 4)
    assert gaps_a.shape == (2, 3)

    fusion.relative_gap_mode = "all_pairs"
    scores_b, _, _, gaps_b, *_ = fusion.compute_scores(stacked)
    assert scores_b.shape == (2, 4)
    assert gaps_b.shape == (2, 6)


def test_gated_bias_path_and_freeze_scope() -> None:
    model = _build_stage1_model()
    model.freeze_attention_gated_reliability_bias_calibration_only()
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert any("fusion_module.reliability_bias_head" in name for name in trainable)
    assert any("fusion_module.reliability_bias_gate_head" in name for name in trainable)
    assert all(
        ("fusion_module.reliability_bias_head" in name) or ("fusion_module.reliability_bias_gate_head" in name)
        for name in trainable
    )

    fusion = model.fusion_module
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.relative_gap_residual_scale = 0.2
    fusion.reliability_bias_residual_scale = 0.2
    fusion.reliability_bias_gate_temperature = 8.0
    fusion.reliability_bias_gate_threshold = 0.2
    stacked = torch.randn(2, 4, 64)
    bias_inputs = torch.randn(2, 4, 66)
    gate_inputs = torch.tensor(
        [
            [[0.9, 0.9, 0.1, 0.3], [0.8, 0.8, 0.1, 0.2], [0.8, 0.8, 0.1, 0.2], [0.8, 0.8, 0.1, 0.2]],
            [[0.1, -0.1, 0.5, -0.4], [0.2, -0.1, 0.4, -0.3], [0.2, -0.1, 0.4, -0.3], [0.2, -0.1, 0.4, -0.3]],
        ],
        dtype=torch.float32,
    )
    scores, _, _, _, rel_before_bias, _, gates, *_ = fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
    )
    assert scores.shape == (2, 4)
    assert rel_before_bias.shape == (2, 4)
    assert gates.shape == (2, 4)
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)
    assert gates[0, 0].item() != gates[1, 0].item()


def test_gate_only_freeze_scope() -> None:
    model = _build_stage1_model()
    model.freeze_attention_gate_only_selective_finetune(train_gate_input_adapter=False)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all("fusion_module.reliability_bias_gate_head" in name for name in trainable)
    frozen_bias = [name for name, param in model.named_parameters() if "fusion_module.reliability_bias_head" in name and not param.requires_grad]
    frozen_score = [name for name, param in model.named_parameters() if "fusion_module.score_adapter" in name and not param.requires_grad]
    frozen_relative = [name for name, param in model.named_parameters() if "fusion_module.relative_gap_head" in name and not param.requires_grad]
    frozen_reliability = [name for name, param in model.named_parameters() if "reliability_head" in name and not param.requires_grad]
    assert frozen_bias and frozen_score and frozen_relative and frozen_reliability


def test_source_residual_gate_freeze_scope() -> None:
    model = _build_stage1_model()
    model.freeze_attention_source_gate_residual_only(train_gate_input_adapter=False)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable
    assert all("fusion_module.reliability_bias_gate_residual_head" in name for name in trainable)
    frozen_bias = [
        name for name, param in model.named_parameters() if "fusion_module.reliability_bias_head" in name and not param.requires_grad
    ]
    assert frozen_bias


def test_adapter_readout_freeze_scope() -> None:
    model = _build_stage1_model()
    model.freeze_attention_source_gate_adapter_readout_micro_unfreeze(train_gate_input_adapter=True, train_score_adapter=False)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any("fusion_module.reliability_bias_gate_adapter_input_adapter" in name for name in trainable)
    assert any("fusion_module.reliability_bias_gate_adapter_coupler" in name for name in trainable)
    assert any("fusion_module.reliability_bias_gate_residual_head" in name for name in trainable)
    assert all("fusion_module.score_adapter" not in name for name in trainable)


def test_output_scale_freeze_scope() -> None:
    model = _build_stage1_model()
    model.fusion_module.gate_output_scale_mode = "learnable"
    configure_trainable_gate_actuation_parameters(
        model,
        {
            "gate_output_scale_mode": "learnable",
            "gate_output_scale_init": [1.0, 1.0, 1.0, 1.0],
        },
    )
    model.freeze_attention_source_gate_output_scale_micro_unfreeze(
        train_gate_input_adapter=True,
        train_score_adapter=False,
    )
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any("fusion_module.gate_output_scale" in name for name in trainable)
    assert any("fusion_module.reliability_bias_gate_adapter_input_adapter" in name for name in trainable)
    assert all("fusion_module.score_adapter" not in name for name in trainable)


def test_source_residual_gate_requires_source_context_and_matches_source_plus_delta() -> None:
    fusion = AttentionFusion(feature_dim=64, hidden_dim=32)
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_v1"
    fusion.relative_gap_residual_scale = 0.2
    fusion.reliability_bias_residual_scale = 0.2
    fusion.gate_residual_scale = 0.08
    stacked = torch.randn(2, 4, 64)
    bias_inputs = torch.randn(2, 4, 66)
    gate_inputs = torch.randn(2, 4, 70)
    try:
        fusion.compute_scores(stacked, bias_inputs=bias_inputs, gate_inputs=gate_inputs)
    except ValueError as exc:
        assert "source_gate_scores" in str(exc)
    else:
        raise AssertionError("source_residual_v1 should require source_gate_scores")

    last_layer = fusion.reliability_bias_gate_residual_head[-1]
    torch.nn.init.zeros_(last_layer.weight)
    torch.nn.init.zeros_(last_layer.bias)
    source_gate_scores = torch.rand(2, 4)
    _, _, _, _, _, _, final_gates, source_gates, gate_residuals, _gate_residual_logits, _gate_adapter_logits, *_ = fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
        source_gate_scores=source_gate_scores,
    )
    assert torch.allclose(source_gates, source_gate_scores)
    assert torch.allclose(gate_residuals, torch.zeros_like(gate_residuals), atol=1e-6)
    assert torch.allclose(final_gates, torch.clamp(source_gate_scores, 0.0, 1.0), atol=1e-6)


def test_source_residual_logit_v2_supports_vector_alpha_and_offset() -> None:
    fusion = AttentionFusion(feature_dim=64, hidden_dim=32)
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_logit_v2"
    fusion.relative_gap_residual_scale = 0.2
    fusion.reliability_bias_residual_scale = 0.2
    fusion.gate_residual_alpha = torch.tensor([4.0, 1.0, 1.0, 1.0])
    fusion.gate_logit_offset = torch.tensor([0.2, 0.0, 0.0, 0.0])
    fusion.gate_output_scale = torch.tensor([2.0, 1.0, 1.0, 1.0])
    stacked = torch.randn(1, 4, 64)
    bias_inputs = torch.randn(1, 4, 66)
    gate_inputs = torch.randn(1, 4, 70)
    source_gate_scores = torch.full((1, 4), 0.05)
    scores, _, _, _, _, _, gates, _, _, residual_logits, _gate_adapter_logits, _delta_logits, _scaled_delta_logits, *_ = fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
        source_gate_scores=source_gate_scores,
    )
    assert scores.shape == (1, 4)
    assert gates.shape == (1, 4)
    assert residual_logits.shape == (1, 4)
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)


def test_gate_loss_alias_canonicalization() -> None:
    canonical = canonicalize_gate_loss_keys(
        {
            "clean_gate_suppression_loss_coef": 0.11,
            "mild_gate_activation_loss_coef": 0.22,
            "gate_contrast_loss_coef": 0.33,
            "mild_gate_activation_margin": 0.44,
            "gate_contrast_margin": 0.55,
        }
    )
    assert canonical["clean_gate_cap_loss_coef"] == 0.11
    assert canonical["mild_gate_floor_loss_coef"] == 0.22
    assert canonical["gate_gap_margin_loss_coef"] == 0.33
    assert canonical["mild_gate_min"] == 0.44
    assert canonical["gate_gap_margin"] == 0.55


def test_gate_metric_threshold_wiring() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "env_mode": "normal",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.15,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.3,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.2,
                "attention_relative_gap_norm_mean": 0.4,
                "attention_relative_gap_norm_mean": 0.5,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.7,
                "reliability_mean_ego": 0.7,
                "reliability_mean_route": 0.7,
                "logit_mean_vision": 0.4,
                "logit_mean_vision_clean": 0.0,
                "logit_mean_vision_mild": 0.0,
                "attention_bias_mean_vision": 0.1,
                "attention_bias_mean_distance": 0.0,
                "attention_bias_mean_ego": 0.0,
                "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
                "attention_bias_gate_mean_vision": 0.01,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_score_adapter_delta_norm_mean_vision": 0.05,
                "attention_score_adapter_delta_norm_mean_distance": 0.04,
                "attention_score_adapter_delta_norm_mean_ego": 0.04,
                "attention_score_adapter_delta_norm_mean_route": 0.04,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "env_mode": "mild",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.1,
                "attention_score_mean_ego": 0.1,
                "attention_score_mean_route": 0.0,
                "attention_relative_gap_score_mean_vision_distance": 0.4,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.25,
                "attention_relative_gap_norm_mean": 0.6,
                "attention_mean_vision": 0.5,
                "attention_mean_distance": 0.18,
                "attention_mean_ego": 0.17,
                "attention_mean_route": 0.15,
                "targetlogit_shapley_mean_vision": 0.1,
                "reliability_mean_vision": 0.8,
                "reliability_mean_distance": 0.7,
                "reliability_mean_ego": 0.7,
                "reliability_mean_route": 0.7,
                "logit_mean_vision": 0.3,
                "logit_mean_vision_clean": 0.0,
                "logit_mean_vision_mild": 0.0,
                "attention_bias_mean_vision": 0.2,
                "attention_bias_mean_distance": 0.0,
                "attention_bias_mean_ego": 0.0,
                "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.2,
                "attention_bias_abs_mean": 0.2,
                "attention_bias_gate_mean_vision": 0.09,
                "attention_bias_gate_mean_distance": 0.02,
                "attention_bias_gate_mean_ego": 0.02,
                "attention_bias_gate_mean_route": 0.02,
                "attention_score_adapter_delta_norm_mean_vision": 0.06,
                "attention_score_adapter_delta_norm_mean_distance": 0.05,
                "attention_score_adapter_delta_norm_mean_ego": 0.05,
                "attention_score_adapter_delta_norm_mean_route": 0.05,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
        ]
    )
    metrics = compute_v3_7e_gate_only_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics.iloc[0]
    assert row["vision_gate_clean_below_cap_ratio"] == 1.0
    assert row["vision_gate_mild_above_floor_ratio"] == 1.0
    assert row["vision_gate_gap_margin_satisfied_ratio"] == 1.0


def _build_reliability_proof_eval_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "probe_name": "normal_eval",
                "corruption_type": "normal_eval",
                "seed": 0,
                "eval_success_rate": 0.82,
                "eval_return": 5.1,
                "reliability_mean_vision": 0.90,
                "reliability_std_vision": 0.01,
                "reliability_mean_distance": 0.80,
                "reliability_std_distance": 0.01,
                "reliability_mean_ego": 0.85,
                "reliability_std_ego": 0.01,
                "reliability_mean_route": 0.78,
                "reliability_std_route": 0.01,
                "reliability_vision_residual_delta_mean": 0.0,
                "reliability_ego_residual_delta_mean": 0.0,
            },
            {
                "candidate_name": "cand",
                "probe_name": "mild_vision",
                "corruption_type": "mild_vision",
                "seed": 0,
                "eval_success_rate": 0.80,
                "eval_return": 4.9,
                "reliability_mean_vision": 0.40,
                "reliability_std_vision": 0.01,
                "reliability_mean_distance": 0.76,
                "reliability_std_distance": 0.01,
                "reliability_mean_ego": 0.80,
                "reliability_std_ego": 0.01,
                "reliability_mean_route": 0.75,
                "reliability_std_route": 0.01,
                "reliability_vision_residual_delta_mean": -0.04,
                "reliability_ego_residual_delta_mean": 0.0,
            },
            {
                "candidate_name": "cand",
                "probe_name": "mild_distance",
                "corruption_type": "mild_distance",
                "seed": 0,
                "eval_success_rate": 0.79,
                "eval_return": 4.8,
                "reliability_mean_vision": 0.88,
                "reliability_std_vision": 0.01,
                "reliability_mean_distance": 0.30,
                "reliability_std_distance": 0.01,
                "reliability_mean_ego": 0.82,
                "reliability_std_ego": 0.01,
                "reliability_mean_route": 0.77,
                "reliability_std_route": 0.01,
                "reliability_vision_residual_delta_mean": 0.0,
                "reliability_ego_residual_delta_mean": 0.0,
            },
            {
                "candidate_name": "cand",
                "probe_name": "mild_ego",
                "corruption_type": "mild_ego",
                "seed": 0,
                "eval_success_rate": 0.80,
                "eval_return": 4.85,
                "reliability_mean_vision": 0.89,
                "reliability_std_vision": 0.01,
                "reliability_mean_distance": 0.79,
                "reliability_std_distance": 0.01,
                "reliability_mean_ego": 0.25,
                "reliability_std_ego": 0.01,
                "reliability_mean_route": 0.76,
                "reliability_std_route": 0.01,
                "reliability_vision_residual_delta_mean": 0.0,
                "reliability_ego_residual_delta_mean": -0.05,
            },
            {
                "candidate_name": "cand",
                "probe_name": "mild_route",
                "corruption_type": "mild_route",
                "seed": 0,
                "eval_success_rate": 0.78,
                "eval_return": 4.75,
                "reliability_mean_vision": 0.88,
                "reliability_std_vision": 0.01,
                "reliability_mean_distance": 0.79,
                "reliability_std_distance": 0.01,
                "reliability_mean_ego": 0.83,
                "reliability_std_ego": 0.01,
                "reliability_mean_route": 0.20,
                "reliability_std_route": 0.01,
                "reliability_vision_residual_delta_mean": 0.0,
                "reliability_ego_residual_delta_mean": 0.0,
            },
        ]
    )


def _build_fusion_shift_eval_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "probe_name": "normal_eval",
                "corruption_type": "normal_eval",
                "seed": 0,
                "run_mode": "smoke",
                "eval_success_rate": 0.82,
                "eval_return": 5.0,
                "num_eval_episodes": 6,
                "reliability_mean_vision": 0.90,
                "reliability_mean_distance": 0.80,
                "reliability_mean_ego": 0.85,
                "reliability_mean_route": 0.78,
                "clean_eval_variant": "standalone",
                "attention_mean_distance_clean": 0.22,
                "attention_mean_ego_clean": 0.18,
                "attention_mean_route_clean": 0.24,
                "clean_route_attention_anchor_error": 0.003,
                "clean_reliability_anchor_error": 0.004,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "clean_bias_score_anchor_error": 0.003,
                "clean_prebias_score_anchor_error": 0.004,
            },
            {
                "candidate_name": "cand",
                "probe_name": "mild_vision",
                "corruption_type": "mild_vision",
                "seed": 0,
                "run_mode": "smoke",
                "eval_success_rate": 0.80,
                "eval_return": 4.9,
                "num_eval_episodes": 6,
                "reliability_mean_vision": 0.60,
                "reliability_mean_distance": 0.80,
                "reliability_mean_ego": 0.85,
                "reliability_mean_route": 0.78,
                "proof_clean_eval_variant": "teacher_conditioned",
                "target_reliability_delta_clean_minus_corrupt": 0.25,
                "target_gate_delta_clean_minus_corrupt": 0.05,
                "target_gate_effect_delta_clean_minus_corrupt": 0.04,
                "target_bias_score_delta_clean_minus_corrupt": 0.25,
                "target_prebias_score_delta_clean_minus_corrupt": 0.20,
                "target_attention_delta_clean_minus_corrupt": 0.08,
                "target_final_score_delta_clean_minus_corrupt": 0.06,
                "target_source_gate_base_delta_clean_minus_corrupt": 0.04,
                "target_source_gate_delta_clean_minus_corrupt": 0.04,
                "target_teacher_gate_delta_clean_minus_corrupt": 0.03,
                "target_teacher_gate_effect_delta_clean_minus_corrupt": 0.02,
                "teacher_preserve_signal_absmax": 0.03,
                "teacher_effect_nonflat_preserve_flag": 1,
                "teacher_effect_nonflat_vision_flag": 1,
                "teacher_effect_nonflat_flag": 1,
                "reliability_gate_alignment_ratio": 0.92,
                "reliability_gate_effect_alignment_ratio": 0.85,
                "gate_attention_alignment_ratio": 0.88,
                "gate_effect_attention_alignment_ratio": 0.80,
                "gate_effect_score_alignment_ratio": 0.75,
                "prebias_score_final_score_alignment_ratio": 0.72,
                "reliability_attention_alignment_ratio": 0.9,
                "nontarget_rescue_delta": 0.01,
                "attention_bias_score_mean_vision_clean": 0.40,
                "attention_bias_score_mean_vision_corrupt": 0.15,
                "attention_prebias_score_mean_vision_clean": 0.35,
                "attention_prebias_score_mean_vision_corrupt": 0.15,
                "attention_bias_gate_effect_mean_vision_clean": 0.03,
                "attention_bias_gate_effect_mean_vision_corrupt": -0.01,
                "attention_mean_distance_clean": 0.22,
                "attention_mean_ego_clean": 0.18,
                "attention_mean_route_clean": 0.24,
                "clean_route_attention_anchor_error": 0.003,
                "clean_reliability_anchor_error": 0.004,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "clean_bias_score_anchor_error": 0.003,
                "clean_prebias_score_anchor_error": 0.004,
                "teacher_source": "external",
            },
        ]
    )


def test_reliability_proof_target_modality_mapping_and_gap() -> None:
    df = _build_reliability_proof_eval_rows()
    reliability_cmp = compute_candidate_reliability_delta(df)
    target_df = compute_reliability_target_metrics(reliability_cmp)
    vision_row = target_df[target_df["target_modality"] == "vision"].iloc[0]
    ego_row = target_df[target_df["target_modality"] == "ego"].iloc[0]
    assert vision_row["corruption_type"] == "mild_vision"
    assert vision_row["target_lowest_flag"] == 1
    assert vision_row["target_vs_nontarget_gap"] > 0.0
    assert ego_row["corruption_type"] == "mild_ego"
    assert ego_row["rank_among_modalities"] == 1


def test_reliability_proof_score_summary_contains_target_ratios() -> None:
    df = _build_reliability_proof_eval_rows()
    reliability_cmp = compute_candidate_reliability_delta(df)
    target_df = compute_reliability_target_metrics(reliability_cmp)
    summary = build_reliability_proof_score_summary(
        "cand",
        df,
        target_df,
        {
            "scoring": {
                "required_target_modalities": 4,
                "clean_success_rate_floor": 0.76,
                "target_positive_ratio_pass": 0.75,
                "target_lowest_ratio_pass": 0.50,
                "mean_target_vs_nontarget_gap_pass": 0.0,
            }
        },
    )
    assert summary["target_positive_ratio"] == 1.0
    assert summary["target_lowest_ratio"] == 1.0
    assert summary["mean_target_vs_nontarget_gap"] > 0.0
    assert summary["clean_success_rate"] == pytest.approx(0.82)
    assert summary["pass_fail"] == "PASS"


def test_compare_clean_vs_corrupted_reliability_uses_num_eval_episode_weighting() -> None:
    rows = []
    for probe_name, count, vision_mean in (
        ("clean", 6, 1.0),
        ("clean", 24, 0.0),
        ("mild_vision", 6, 0.5),
        ("mild_vision", 24, 0.25),
    ):
        rows.append(
            {
                "probe_name": probe_name,
                "num_eval_episodes": count,
                "reliability_mean_vision": vision_mean,
                "reliability_std_vision": 0.1,
                "reliability_mean_distance": 0.8,
                "reliability_std_distance": 0.1,
                "reliability_mean_ego": 0.7,
                "reliability_std_ego": 0.1,
                "reliability_mean_route": 0.6,
                "reliability_std_route": 0.1,
            }
        )
    cmp_df = compare_clean_vs_corrupted_reliability(pd.DataFrame(rows))
    vision_row = cmp_df[(cmp_df["corruption_type"] == "mild_vision") & (cmp_df["modality"] == "vision")].iloc[0]
    assert vision_row["clean_mean"] == pytest.approx((6 * 1.0 + 24 * 0.0) / 30.0)
    assert vision_row["weak_mean"] == pytest.approx((6 * 0.5 + 24 * 0.25) / 30.0)


def test_reliability_proof_missing_probe_becomes_warn_not_false_fail() -> None:
    df = _build_reliability_proof_eval_rows()
    df = df[df["probe_name"] != "mild_route"].reset_index(drop=True)
    reliability_cmp = compute_candidate_reliability_delta(df)
    target_df = compute_reliability_target_metrics(reliability_cmp)
    summary = build_reliability_proof_score_summary(
        "cand",
        df,
        target_df,
        {
            "scoring": {
                "required_target_modalities": 4,
                "clean_success_rate_floor": 0.76,
                "target_positive_ratio_pass": 0.75,
                "target_lowest_ratio_pass": 0.50,
                "mean_target_vs_nontarget_gap_pass": 0.0,
            }
        },
    )
    assert summary["observed_target_modalities"] == 3
    assert summary["pass_fail"] == "WARN"


def test_reliability_proof_explicit_fail_rule() -> None:
    summary = {
        "observed_target_modalities": 4,
        "target_positive_ratio": 0.0,
        "target_lowest_ratio": 0.0,
        "mean_target_vs_nontarget_gap": -0.1,
        "vision_positive_ratio": 0.0,
        "distance_positive_ratio": 0.0,
        "ego_positive_ratio": 0.0,
        "route_positive_ratio": 0.0,
        "clean_success_rate": 0.82,
    }
    verdict = evaluate_reliability_proof_summary(
        summary,
        {
            "required_target_modalities": 4,
            "clean_success_rate_floor": 0.76,
            "target_positive_ratio_pass": 0.75,
            "target_lowest_ratio_pass": 0.50,
            "mean_target_vs_nontarget_gap_pass": 0.0,
        },
    )
    assert verdict == "FAIL"


def test_fusion_shift_proof_summary_and_recommendation(tmp_path: Path) -> None:
    baseline_rows = pd.DataFrame(
        [
            {
                "probe_name": "normal_eval",
                "seed": 0,
                "run_mode": "smoke",
                "num_eval_episodes": 6,
                "reliability_mean_vision": 0.90,
                "reliability_mean_distance": 0.80,
                "reliability_mean_ego": 0.85,
                "reliability_mean_route": 0.78,
                "attention_mean_distance": 0.22,
                "attention_mean_ego": 0.18,
                "attention_mean_route": 0.24,
            },
            {
                "probe_name": "normal_eval",
                "seed": 1,
                "run_mode": "full",
                "num_eval_episodes": 24,
                "reliability_mean_vision": 0.10,
                "reliability_mean_distance": 0.10,
                "reliability_mean_ego": 0.10,
                "reliability_mean_route": 0.10,
                "attention_mean_distance": 0.90,
                "attention_mean_ego": 0.90,
                "attention_mean_route": 0.90,
            },
        ]
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_rows.to_dict(orient="records")), encoding="utf-8")
    summary = build_fusion_shift_proof_score_summary(
        "cand",
        _build_fusion_shift_eval_rows(),
        {
            "baseline_eval_rows_path": str(baseline_path),
            "baseline_eval_rows_path_smoke": str(baseline_path),
            "candidates": {
                "cand": {
                    "checkpoint_root": "/tmp/cand",
                }
            },
            "scoring": {
                "clean_success_rate_floor": 0.79,
                "vision_reliability_delta_pass": 0.0,
                "vision_gate_effect_delta_pass": 0.0,
                "vision_gate_effect_absmax_pass": 0.01,
                "vision_bias_score_delta_pass": 0.0,
                "vision_prebias_score_delta_pass": 0.0,
                "vision_attention_delta_pass": 0.0,
                "vision_final_score_delta_pass": 0.0,
                "vision_reliability_gate_effect_alignment_ratio_pass": 0.70,
                "vision_gate_effect_attention_alignment_ratio_pass": 0.70,
                "vision_gate_effect_score_alignment_ratio_pass": 0.70,
                "vision_prebias_score_final_score_alignment_ratio_pass": 0.70,
                "nontarget_rescue_delta_pass": 0.0,
                "max_non_target_clean_attention_drift_pass": 0.03,
                "max_non_ego_clean_reliability_drift_pass": 0.02,
                "clean_score_anchor_error_pass": 0.01,
                "clean_action_anchor_error_pass": 0.01,
                "clean_bias_score_anchor_error_pass": 0.01,
                "clean_prebias_score_anchor_error_pass": 0.01,
            },
        },
        smoke=True,
    )
    assert summary["vision_reliability_delta_clean_minus_corrupt"] == pytest.approx(0.25)
    assert summary["vision_gate_effect_delta_clean_minus_corrupt"] == pytest.approx(0.04)
    assert summary["vision_bias_score_delta_clean_minus_corrupt"] == pytest.approx(0.25)
    assert summary["vision_prebias_score_delta_clean_minus_corrupt"] == pytest.approx(0.20)
    assert summary["vision_gate_delta_clean_minus_corrupt"] == pytest.approx(0.05)
    assert summary["vision_attention_delta_clean_minus_corrupt"] == pytest.approx(0.08)
    assert summary["vision_final_score_delta_clean_minus_corrupt"] == pytest.approx(0.06)
    assert summary["vision_source_gate_base_delta_clean_minus_corrupt"] == pytest.approx(0.04)
    assert summary["vision_teacher_gate_effect_delta_clean_minus_corrupt"] == pytest.approx(0.02)
    assert summary["vision_reliability_gate_effect_alignment_ratio"] == pytest.approx(0.85)
    assert summary["vision_gate_effect_attention_alignment_ratio"] == pytest.approx(0.80)
    assert summary["vision_gate_effect_score_alignment_ratio"] == pytest.approx(0.75)
    assert summary["vision_prebias_score_final_score_alignment_ratio"] == pytest.approx(0.72)
    assert summary["teacher_preserve_signal_absmax"] == pytest.approx(0.03)
    assert summary["teacher_effect_nonflat_preserve_flag"] == 1
    assert summary["teacher_effect_nonflat_vision_flag"] == 1
    assert summary["vision_gate_effect_absmax"] == pytest.approx(0.03)
    assert summary["vision_bias_score_absmax"] == pytest.approx(0.40)
    assert summary["max_non_target_clean_attention_drift"] == pytest.approx(0.0)
    assert summary["max_non_ego_clean_reliability_drift"] == pytest.approx(0.0)
    assert summary["clean_success_rate_standalone"] == pytest.approx(0.82)
    assert summary["clean_route_attention_anchor_error"] == pytest.approx(0.003)
    assert summary["clean_reliability_anchor_error"] == pytest.approx(0.004)
    assert summary["clean_bias_score_anchor_error"] == pytest.approx(0.003)
    assert summary["clean_prebias_score_anchor_error"] == pytest.approx(0.004)
    assert summary["teacher_source"] == "external"
    assert evaluate_fusion_shift_proof_summary(summary, {"clean_success_rate_floor": 0.79}) == "PASS"
    score_df = pd.DataFrame(
        [
            {
                "candidate_name": "vision_prebias_output_distilled_scale005",
                "pass_fail": "PASS",
                "teacher_effect_distill_enabled": 1,
                "teacher_effect_nonflat_preserve_flag": 0,
            },
            {
                "candidate_name": "vision_prebias_output_anchor_scale005",
                "pass_fail": "PASS",
                "teacher_effect_distill_enabled": 0,
                "teacher_effect_nonflat_preserve_flag": 1,
            },
        ]
    )
    assert choose_fusion_shift_proof_recommendation(score_df) == "vision_prebias_output_anchor_scale005"


def test_fusion_shift_proof_pass_checks_gate_effect_absmax_threshold() -> None:
    summary = {
        "vision_reliability_delta_clean_minus_corrupt": 0.1,
        "vision_gate_effect_delta_clean_minus_corrupt": 0.1,
        "vision_gate_effect_absmax": 1.0e-5,
        "vision_bias_score_delta_clean_minus_corrupt": 0.1,
        "vision_prebias_score_delta_clean_minus_corrupt": 0.1,
        "vision_attention_delta_clean_minus_corrupt": 0.1,
        "vision_final_score_delta_clean_minus_corrupt": 0.1,
        "vision_reliability_gate_effect_alignment_ratio": 1.0,
        "vision_gate_effect_attention_alignment_ratio": 1.0,
        "vision_gate_effect_score_alignment_ratio": 1.0,
        "vision_prebias_score_final_score_alignment_ratio": 1.0,
        "nontarget_rescue_delta": 0.1,
        "clean_success_rate_standalone": 0.9,
        "max_non_target_clean_attention_drift_standalone": 0.0,
        "max_non_ego_clean_reliability_drift_standalone": 0.0,
        "clean_score_anchor_error": 0.0,
        "clean_action_anchor_error": 0.0,
        "clean_bias_score_anchor_error": 0.0,
        "clean_prebias_score_anchor_error": 0.0,
    }
    scoring = {
        "clean_success_rate_floor": 0.79,
        "vision_gate_effect_absmax_pass": 1.0e-4,
        "clean_score_anchor_error_pass": 0.01,
        "clean_action_anchor_error_pass": 0.01,
        "clean_bias_score_anchor_error_pass": 0.01,
        "clean_prebias_score_anchor_error_pass": 0.01,
    }
    assert evaluate_fusion_shift_proof_summary(summary, scoring) == "WARN"
    summary["vision_gate_effect_absmax"] = 2.0e-4
    assert evaluate_fusion_shift_proof_summary(summary, scoring) == "PASS"


def test_fusion_shift_proof_pass_checks_clean_score_and_action_anchor_errors() -> None:
    summary = {
        "vision_reliability_delta_clean_minus_corrupt": 0.1,
        "vision_gate_effect_delta_clean_minus_corrupt": 0.1,
        "vision_bias_score_delta_clean_minus_corrupt": 0.1,
        "vision_prebias_score_delta_clean_minus_corrupt": 0.1,
        "vision_attention_delta_clean_minus_corrupt": 0.1,
        "vision_final_score_delta_clean_minus_corrupt": 0.1,
        "vision_reliability_gate_effect_alignment_ratio": 1.0,
        "vision_gate_effect_attention_alignment_ratio": 1.0,
        "vision_gate_effect_score_alignment_ratio": 1.0,
        "vision_prebias_score_final_score_alignment_ratio": 1.0,
        "nontarget_rescue_delta": 0.1,
        "clean_success_rate_standalone": 0.9,
        "max_non_target_clean_attention_drift_standalone": 0.0,
        "max_non_ego_clean_reliability_drift_standalone": 0.0,
        "clean_score_anchor_error": 0.02,
        "clean_action_anchor_error": 0.0,
        "clean_bias_score_anchor_error": 0.0,
        "clean_prebias_score_anchor_error": 0.0,
    }
    scoring = {
        "clean_success_rate_floor": 0.79,
        "clean_score_anchor_error_pass": 0.01,
        "clean_action_anchor_error_pass": 0.01,
        "clean_bias_score_anchor_error_pass": 0.01,
        "clean_prebias_score_anchor_error_pass": 0.01,
    }
    assert evaluate_fusion_shift_proof_summary(summary, scoring) == "WARN"
    summary["clean_score_anchor_error"] = 0.0
    summary["clean_action_anchor_error"] = 0.02
    assert evaluate_fusion_shift_proof_summary(summary, scoring) == "WARN"
    summary["clean_action_anchor_error"] = 0.0
    summary["clean_prebias_score_anchor_error"] = 0.02
    assert evaluate_fusion_shift_proof_summary(summary, scoring) == "WARN"


def test_fusion_shift_proof_report_mentions_legacy_alias_removal(tmp_path: Path) -> None:
    score_df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "pass_fail": "WARN",
                "vision_reliability_delta_clean_minus_corrupt": 0.1,
                "vision_gate_effect_delta_clean_minus_corrupt": 0.0,
                "vision_gate_effect_absmax": 1.0e-5,
                "vision_bias_score_delta_clean_minus_corrupt": 0.1,
                "vision_prebias_score_delta_clean_minus_corrupt": 0.1,
                "vision_gate_delta_clean_minus_corrupt": 0.0,
                "vision_teacher_gate_effect_delta_clean_minus_corrupt": 0.0,
                "vision_attention_delta_clean_minus_corrupt": 0.0,
                "vision_final_score_delta_clean_minus_corrupt": 0.0,
                "vision_bias_score_absmax": 0.1,
                "vision_source_gate_base_delta_clean_minus_corrupt": 0.0,
                "vision_reliability_gate_effect_alignment_ratio": 0.0,
                "vision_gate_effect_attention_alignment_ratio": 0.0,
                "vision_gate_effect_score_alignment_ratio": 0.0,
                "vision_prebias_score_final_score_alignment_ratio": 0.0,
                "vision_reliability_attention_alignment_ratio": 0.0,
                "nontarget_rescue_delta": 0.0,
                "max_non_target_clean_attention_drift_standalone": 0.0,
                "max_non_ego_clean_reliability_drift_standalone": 0.0,
                "clean_success_rate_standalone": 0.8,
                "clean_route_attention_anchor_error": 0.0,
                "clean_reliability_anchor_error": 0.0,
                "clean_bias_score_anchor_error": 0.0,
                "clean_prebias_score_anchor_error": 0.0,
                "teacher_effect_distill_enabled": 0,
                "teacher_preserve_signal_absmax": 0.0,
                "teacher_effect_nonflat_preserve_flag": 0,
                "teacher_effect_nonflat_vision_flag": 0,
                "teacher_gate_effect_absmax": 0.0,
                "teacher_bias_score_absmax": 0.0,
                "teacher_actuation_regime_mismatch_flag": 0,
                "teacher_preserve_active_flag": 0,
                "teacher_source": "disabled",
                "baseline_clean_missing_flag": 0,
                "baseline_clean_file_missing_flag": 0,
                "baseline_clean_match_missing_flag": 0,
                "mixed_eval_episode_count_flag": 0,
            }
        ]
    )
    report_path = tmp_path / "report.md"
    generate_fusion_shift_proof_report(score_df, report_path, "cand")
    report = report_path.read_text(encoding="utf-8")
    assert "legacy `reference_same_state_action_diff_clean_vs_corrupt` alias has been removed" in report


def test_fusion_shift_proof_missing_baseline_clean_frame_becomes_warn(tmp_path: Path) -> None:
    baseline_path = tmp_path / "missing_baseline.json"
    summary = build_fusion_shift_proof_score_summary(
        "cand",
        _build_fusion_shift_eval_rows(),
        {
            "baseline_eval_rows_path": str(baseline_path),
            "baseline_eval_rows_path_smoke": str(baseline_path),
            "candidates": {"cand": {"checkpoint_root": "/tmp/cand"}},
            "scoring": {
                "clean_success_rate_floor": 0.79,
                "vision_reliability_delta_pass": 0.0,
                "vision_gate_effect_delta_pass": 0.0,
                "vision_bias_score_delta_pass": 0.0,
                "vision_prebias_score_delta_pass": 0.0,
                "vision_attention_delta_pass": 0.0,
                "vision_final_score_delta_pass": 0.0,
                "vision_reliability_gate_effect_alignment_ratio_pass": 0.70,
                "vision_gate_effect_attention_alignment_ratio_pass": 0.70,
                "vision_gate_effect_score_alignment_ratio_pass": 0.70,
                "vision_prebias_score_final_score_alignment_ratio_pass": 0.70,
                "nontarget_rescue_delta_pass": 0.0,
                "max_non_target_clean_attention_drift_pass": 0.03,
                "max_non_ego_clean_reliability_drift_pass": 0.02,
                "clean_score_anchor_error_pass": 0.01,
                "clean_action_anchor_error_pass": 0.01,
                "clean_bias_score_anchor_error_pass": 0.01,
                "clean_prebias_score_anchor_error_pass": 0.01,
            },
        },
        smoke=True,
    )
    assert summary["baseline_clean_missing_flag"] == 1
    assert summary["pass_fail"] == "WARN"


def test_fusion_shift_proof_exact_match_missing_baseline_becomes_warn(tmp_path: Path) -> None:
    baseline_rows = pd.DataFrame(
        [
            {
                "probe_name": "normal_eval",
                "seed": 99,
                "run_mode": "smoke",
                "num_eval_episodes": 6,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.9,
                "reliability_mean_ego": 0.9,
                "reliability_mean_route": 0.9,
                "attention_mean_distance": 0.9,
                "attention_mean_ego": 0.9,
                "attention_mean_route": 0.9,
            },
        ]
    )
    baseline_path = tmp_path / "baseline_mismatch.json"
    baseline_path.write_text(json.dumps(baseline_rows.to_dict(orient="records")), encoding="utf-8")
    summary = build_fusion_shift_proof_score_summary(
        "cand",
        _build_fusion_shift_eval_rows(),
        {
            "baseline_eval_rows_path": str(baseline_path),
            "baseline_eval_rows_path_smoke": str(baseline_path),
            "candidates": {"cand": {"checkpoint_root": "/tmp/cand"}},
            "scoring": {
                "clean_success_rate_floor": 0.79,
                "vision_reliability_delta_pass": 0.0,
                "vision_gate_effect_delta_pass": 0.0,
                "vision_bias_score_delta_pass": 0.0,
                "vision_prebias_score_delta_pass": 0.0,
                "vision_attention_delta_pass": 0.0,
                "vision_final_score_delta_pass": 0.0,
                "vision_reliability_gate_effect_alignment_ratio_pass": 0.70,
                "vision_gate_effect_attention_alignment_ratio_pass": 0.70,
                "vision_gate_effect_score_alignment_ratio_pass": 0.70,
                "vision_prebias_score_final_score_alignment_ratio_pass": 0.70,
                "nontarget_rescue_delta_pass": 0.0,
                "max_non_target_clean_attention_drift_pass": 0.03,
                "max_non_ego_clean_reliability_drift_pass": 0.02,
                "clean_score_anchor_error_pass": 0.01,
                "clean_action_anchor_error_pass": 0.01,
                "clean_bias_score_anchor_error_pass": 0.01,
                "clean_prebias_score_anchor_error_pass": 0.01,
            },
        },
        smoke=True,
    )
    assert summary["baseline_clean_missing_flag"] == 1
    assert summary["baseline_clean_match_missing_flag"] == 1
    assert summary["baseline_clean_file_missing_flag"] == 0
    assert summary["pass_fail"] == "WARN"


def test_choose_fusion_shift_proof_recommendation_prefers_anchor_warn_over_invalid_distilled_warn() -> None:
    score_df = pd.DataFrame(
        [
            {
                "candidate_name": "distilled_warn",
                "pass_fail": "WARN",
                "teacher_effect_distill_enabled": 1,
                "teacher_effect_nonflat_preserve_flag": 1,
                "teacher_actuation_regime_mismatch_flag": 1,
                "teacher_preserve_active_flag": 0,
                "vision_final_score_delta_clean_minus_corrupt": 0.2,
                "vision_attention_delta_clean_minus_corrupt": 0.2,
                "vision_gate_effect_delta_clean_minus_corrupt": 0.2,
                "clean_success_rate_standalone": 0.9,
            },
            {
                "candidate_name": "anchor_warn",
                "pass_fail": "WARN",
                "teacher_effect_distill_enabled": 0,
                "teacher_effect_nonflat_preserve_flag": 0,
                "teacher_actuation_regime_mismatch_flag": 0,
                "teacher_preserve_active_flag": 0,
                "vision_final_score_delta_clean_minus_corrupt": 0.1,
                "vision_attention_delta_clean_minus_corrupt": 0.1,
                "vision_gate_effect_delta_clean_minus_corrupt": 0.1,
                "clean_success_rate_standalone": 0.8,
            },
        ]
    )
    assert choose_fusion_shift_proof_recommendation(score_df) == "anchor_warn"


def test_reliability_input_mode_raw_builds_expected_inputs() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own_plus_fused_plus_raw",
        reliability_raw_adapter_dim=16,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    first_linear = next(module for module in model.reliability_head if isinstance(module, torch.nn.Linear))
    assert first_linear.in_features == 64 * 2 + 16
    assert set(model.reliability_raw_adapters.keys()) == set(["vision", "distance", "ego", "route"])
    output = model.step(
        obs_dict=_sample_obs(),
        hidden_state=model.initial_hidden(batch_size=1, device=torch.device("cpu")),
        episode_start=torch.ones(1),
        deterministic=True,
    )
    assert output.aux_outputs["reliability_scores"].shape == (1, 4)
    assert output.aux_outputs["attention_bias_gate_effects"].shape == (1, 4)


def test_ego_residual_zero_init_keeps_reliability_logits_unchanged() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=True,
        reliability_ego_raw_adapter_dim=8,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    logits = torch.randn(1, 4)
    adjusted, residuals = model._apply_ego_reliability_residual(logits.clone(), _sample_obs(), _sample_fused())
    assert torch.allclose(adjusted, logits)
    assert torch.count_nonzero(residuals) == 0


def test_ego_residual_only_modifies_ego_logit() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=True,
        reliability_ego_raw_adapter_dim=8,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    for parameter in model.reliability_ego_raw_adapter.parameters():
        torch.nn.init.constant_(parameter, 0.1)
    for parameter in model.reliability_ego_residual_head.parameters():
        torch.nn.init.constant_(parameter, 0.2)
    logits = torch.randn(1, 4)
    adjusted, residuals = model._apply_ego_reliability_residual(logits.clone(), _sample_obs(), _sample_fused())
    assert torch.allclose(adjusted[:, 0], logits[:, 0])
    assert torch.allclose(adjusted[:, 1], logits[:, 1])
    assert torch.allclose(adjusted[:, 3], logits[:, 3])
    assert not torch.allclose(adjusted[:, 2], logits[:, 2])
    assert torch.allclose(residuals[:, 0], torch.zeros_like(residuals[:, 0]))
    assert torch.allclose(residuals[:, 1], torch.zeros_like(residuals[:, 1]))
    assert torch.allclose(residuals[:, 3], torch.zeros_like(residuals[:, 3]))


def test_freeze_ego_reliability_residual_only_scope() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=True,
        reliability_ego_raw_adapter_dim=8,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    model.freeze_ego_reliability_residual_only()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        name.startswith("reliability_ego_raw_adapter.")
        or name.startswith("reliability_ego_residual_head.")
        or name.startswith("reliability_local_residual_adapters.ego.")
        or name.startswith("reliability_local_residual_heads.ego.")
        for name in trainable
    )


def test_stage2_candidate_effective_config_applies_overrides() -> None:
    config = {
        "candidates": {
            "cand": {
                "checkpoint_root": "/tmp/checkpoint",
                "overrides": {
                    "model": {
                        "reliability_input_mode": "own_plus_fused_plus_raw",
                        "reliability_raw_adapter_dim": 16,
                    },
                    "losses": {
                        "ego_logit_retention_loss_coef": 0.0,
                    },
                },
            }
        },
        "losses": {"ego_logit_retention_loss_coef": 0.03},
    }
    effective = build_stage2_candidate_effective_config(config, "cand")
    assert effective["model"]["reliability_input_mode"] == "own_plus_fused_plus_raw"
    assert effective["model"]["reliability_raw_adapter_dim"] == 16
    assert effective["losses"]["ego_logit_retention_loss_coef"] == pytest.approx(0.0)


def test_stage2_ego_positive_losses_only_apply_to_single_target_ego() -> None:
    reliability_scores = torch.tensor([[[0.8, 0.8, 0.9, 0.8]]], dtype=torch.float32)
    reliability_logits = torch.logit(torch.clamp(reliability_scores, min=1e-4, max=1.0 - 1e-4))
    clean_reliability_scores = torch.tensor([[[0.8, 0.8, 0.7, 0.8]]], dtype=torch.float32)
    clean_reliability_logits = torch.logit(torch.clamp(clean_reliability_scores, min=1e-4, max=1.0 - 1e-4))
    corrupted_mask = torch.tensor([[1.0]], dtype=torch.float32)

    ego_target_mask = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    ego_positive_loss, ego_logit_positive_loss, ego_active_ratio, ego_logit_active_ratio = compute_stage2_ego_positive_losses(
        reliability_scores=reliability_scores,
        reliability_logits=reliability_logits,
        clean_reliability_scores=clean_reliability_scores,
        clean_reliability_logits=clean_reliability_logits,
        target_mask=ego_target_mask,
        corrupted_mask=corrupted_mask,
        positive_margin=0.004,
        logit_margin=0.05,
    )
    assert ego_positive_loss.item() > 0.0
    assert ego_logit_positive_loss.item() > 0.0
    assert ego_active_ratio.item() > 0.0
    assert ego_logit_active_ratio.item() > 0.0

    vision_target_mask = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    vision_positive_loss, vision_logit_positive_loss, vision_active_ratio, vision_logit_active_ratio = compute_stage2_ego_positive_losses(
        reliability_scores=reliability_scores,
        reliability_logits=reliability_logits,
        clean_reliability_scores=clean_reliability_scores,
        clean_reliability_logits=clean_reliability_logits,
        target_mask=vision_target_mask,
        corrupted_mask=corrupted_mask,
        positive_margin=0.004,
        logit_margin=0.05,
    )
    assert vision_positive_loss.item() == pytest.approx(0.0)
    assert vision_logit_positive_loss.item() == pytest.approx(0.0)
    assert vision_active_ratio.item() == pytest.approx(0.0)
    assert vision_logit_active_ratio.item() == pytest.approx(0.0)


def test_reference_anchored_targets_lower_only_target_modalities() -> None:
    clean_scores = torch.tensor([[[0.9, 0.8, 0.7, 0.6]]], dtype=torch.float32)
    severity = torch.tensor([[[0.0, 0.0, 0.05, 0.0]]], dtype=torch.float32)
    target_mask = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    targets = build_stage2_reference_anchored_targets(
        clean_reliability_scores=clean_scores,
        severity=severity,
        target_mask=target_mask,
        direct_target_drop_scale=3.5,
    )
    assert targets[0, 0, 0].item() == pytest.approx(0.9)
    assert targets[0, 0, 1].item() == pytest.approx(0.8)
    assert targets[0, 0, 3].item() == pytest.approx(0.6)
    assert targets[0, 0, 2].item() < clean_scores[0, 0, 2].item()


def test_ego_residual_apply_to_provisional_false_skips_provisional_adjustment() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=True,
        reliability_ego_raw_adapter_dim=16,
        reliability_ego_residual_input_mode="raw_route_fused_v1",
        reliability_ego_residual_apply_to_provisional=False,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    output = model.step(
        obs_dict=_sample_obs(),
        hidden_state=model.initial_hidden(batch_size=1, device=torch.device("cpu")),
        episode_start=torch.ones(1),
        deterministic=True,
    )
    provisional = output.aux_outputs["provisional_reliability_ego_residual_deltas"]
    assert torch.allclose(provisional, torch.zeros_like(provisional))


def test_forward_sequence_returns_ego_residual_delta_sequences() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=True,
        reliability_ego_raw_adapter_dim=16,
        reliability_ego_residual_input_mode="raw_route_fused_v1",
        reliability_ego_residual_apply_to_provisional=False,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    obs_seq = {
        key: value.unsqueeze(1).repeat(1, 3, 1)
        for key, value in _sample_obs().items()
    }
    output = model.forward_sequence(
        obs_seq=obs_seq,
        episode_starts=torch.zeros(1, 3),
        initial_hidden=model.initial_hidden(batch_size=1, device=torch.device("cpu")),
        deterministic=True,
    )
    assert output["reliability_ego_residual_deltas"].shape == (1, 3, 4)
    assert output["provisional_reliability_ego_residual_deltas"].shape == (1, 3, 4)
    assert torch.allclose(
        output["provisional_reliability_ego_residual_deltas"],
        torch.zeros_like(output["provisional_reliability_ego_residual_deltas"]),
    )


def test_raw_route_fused_residual_inputs_exclude_aux_and_have_expected_width() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_ego_residual_enabled=True,
        reliability_ego_raw_adapter_dim=16,
        reliability_ego_residual_input_mode="raw_route_fused_v1",
        reliability_ego_residual_apply_to_provisional=False,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    obs = _sample_obs()
    obs["aux"] = torch.randn(1, 8)
    fused = _sample_fused()
    residual_inputs = model._build_ego_reliability_residual_inputs(obs, fused)
    expected_width = obs["ego"].shape[-1] + obs["route"].shape[-1] + 2 + 2 + fused.shape[-1]
    assert residual_inputs.shape == (1, expected_width)


def test_local_residual_only_modifies_vision_and_ego_and_skips_provisional() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_local_residual_enabled_modalities=["vision", "ego"],
        reliability_local_residual_adapter_dim=16,
        reliability_local_residual_input_mode="raw_route_fused_v2",
        reliability_local_residual_apply_to_provisional=False,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    output = model.step(
        obs_dict=_sample_obs(),
        hidden_state=model.initial_hidden(batch_size=1, device=torch.device("cpu")),
        episode_start=torch.ones(1),
        deterministic=True,
    )
    local = output.aux_outputs["reliability_local_residual_deltas"]
    provisional = output.aux_outputs["provisional_reliability_local_residual_deltas"]
    assert local.shape == (1, 4)
    assert provisional.shape == (1, 4)
    assert torch.allclose(provisional, torch.zeros_like(provisional))
    assert local[0, MODALITY_NAMES.index("distance")].item() == pytest.approx(0.0)
    assert local[0, MODALITY_NAMES.index("route")].item() == pytest.approx(0.0)


def test_forward_sequence_returns_local_residual_delta_sequences() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_local_residual_enabled_modalities=["vision", "ego"],
        reliability_local_residual_adapter_dim=16,
        reliability_local_residual_input_mode="raw_route_fused_v2",
        reliability_local_residual_apply_to_provisional=False,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    obs_seq = {key: value.unsqueeze(1).repeat(1, 3, 1) for key, value in _sample_obs().items()}
    output = model.forward_sequence(
        obs_seq=obs_seq,
        episode_starts=torch.zeros(1, 3),
        initial_hidden=model.initial_hidden(batch_size=1, device=torch.device("cpu")),
        deterministic=True,
    )
    assert output["reliability_local_residual_deltas"].shape == (1, 3, 4)
    assert output["provisional_reliability_local_residual_deltas"].shape == (1, 3, 4)
    assert torch.allclose(
        output["provisional_reliability_local_residual_deltas"],
        torch.zeros_like(output["provisional_reliability_local_residual_deltas"]),
    )


def test_raw_route_fused_v2_local_residual_inputs_exclude_aux_and_have_expected_width() -> None:
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=[32],
        encoder_output_dim=64,
        attention_hidden_dim=32,
        recurrent_hidden_dim=32,
        attention_temperature=1.0,
        use_feature_layer_norm=True,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        reliability_hidden_dims=[32],
        reliability_input_mode="own",
        reliability_raw_adapter_dim=16,
        reliability_local_residual_enabled_modalities=["vision", "ego"],
        reliability_local_residual_adapter_dim=16,
        reliability_local_residual_input_mode="raw_route_fused_v2",
        reliability_local_residual_apply_to_provisional=False,
        action_dim=2,
        action_log_std_init=-0.5,
    )
    obs = _sample_obs()
    obs["aux"] = torch.randn(1, 8)
    fused = _sample_fused()
    base_scores = torch.zeros(1, 4)
    base_logits = torch.zeros(1, 4)
    vision_inputs = model._build_local_reliability_residual_inputs(
        "vision",
        obs,
        fused,
        base_reliability_scores=base_scores,
        base_reliability_logits=base_logits,
    )
    ego_inputs = model._build_local_reliability_residual_inputs(
        "ego",
        obs,
        fused,
        base_reliability_scores=base_scores,
        base_reliability_logits=base_logits,
    )
    assert vision_inputs.shape == (1, obs["vision"].shape[-1] + fused.shape[-1] + 2)
    assert ego_inputs.shape == (1, obs["ego"].shape[-1] + obs["route"].shape[-1] + 2 + 2 + fused.shape[-1] + 2)


def test_modality_positive_and_selective_residual_losses_activate_on_synthetic_batches() -> None:
    reliability_scores = torch.tensor([[[0.55, 0.80, 0.56, 0.70], [0.80, 0.80, 0.57, 0.70]]], dtype=torch.float32)
    reliability_logits = torch.tensor([[[0.20, 0.10, 0.10, 0.30], [0.10, 0.10, 0.15, 0.30]]], dtype=torch.float32)
    clean_reliability_scores = torch.tensor([[[0.52, 0.80, 0.52, 0.70], [0.80, 0.80, 0.52, 0.70]]], dtype=torch.float32)
    clean_reliability_logits = torch.tensor([[[0.15, 0.10, 0.05, 0.30], [0.10, 0.10, 0.05, 0.30]]], dtype=torch.float32)
    target_mask = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    corrupted_mask = torch.ones(1, 2, dtype=torch.float32)
    vision_positive_loss, _vision_logit_positive_loss, _vision_ratio, _vision_logit_ratio = (
        compute_stage2_modality_positive_losses(
            reliability_scores=reliability_scores,
            reliability_logits=reliability_logits,
            clean_reliability_scores=clean_reliability_scores,
            clean_reliability_logits=clean_reliability_logits,
            target_mask=target_mask,
            corrupted_mask=corrupted_mask,
            modality_name="vision",
            positive_margin=0.002,
            logit_margin=0.02,
        )
    )
    assert vision_positive_loss.item() > 0.0
    assert (0.0 * vision_positive_loss).item() == pytest.approx(0.0)
    assert (0.08 * vision_positive_loss).item() > 0.0

    local_residual_deltas = torch.tensor([[[ -0.005, 0.0, 0.0, 0.0], [0.0, 0.0, -0.004, 0.0]]], dtype=torch.float32)
    current_clean_local_residual_deltas = torch.zeros_like(local_residual_deltas)
    clean_zero_loss, selectivity_loss = compute_stage2_local_residual_losses(
        local_residual_deltas=local_residual_deltas,
        current_clean_local_residual_deltas=current_clean_local_residual_deltas,
        target_mask=target_mask,
        corrupted_mask=corrupted_mask,
        valid_mask=torch.ones(1, 2, dtype=torch.float32),
        enabled_modalities=["vision", "ego"],
        selectivity_margin=0.02,
    )
    assert clean_zero_loss.item() == pytest.approx(0.0)
    assert selectivity_loss.item() > 0.0

    (
        target_gate_down_loss,
        target_attention_down_loss,
        target_final_score_down_loss,
        nontarget_rescue_loss,
        reliability_gate_alignment_loss,
        gate_attention_alignment_loss,
        reliability_attention_alignment_loss,
        gate_attention_alignment_ratio,
        reliability_gate_alignment_ratio,
        alignment_ratio,
    ) = compute_stage2_fusion_shift_losses(
        clean_attention_weights=torch.tensor([[[0.80, 0.10, 0.05, 0.05]]], dtype=torch.float32),
        corrupt_attention_weights=torch.tensor([[[0.74, 0.12, 0.07, 0.07]]], dtype=torch.float32),
        clean_final_scores=torch.tensor([[[0.40, 0.10, 0.10, 0.10]]], dtype=torch.float32),
        corrupt_final_scores=torch.tensor([[[0.35, 0.11, 0.11, 0.11]]], dtype=torch.float32),
        clean_gate_scores=torch.tensor([[[0.08, 0.02, 0.02, 0.02]]], dtype=torch.float32),
        corrupt_gate_scores=torch.tensor([[[0.05, 0.03, 0.03, 0.03]]], dtype=torch.float32),
        clean_reliability_scores=torch.tensor([[[0.90, 0.80, 0.70, 0.60]]], dtype=torch.float32),
        corrupt_reliability_scores=torch.tensor([[[0.84, 0.80, 0.70, 0.60]]], dtype=torch.float32),
        target_mask=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupted_mask=torch.ones(1, 1, dtype=torch.float32),
        valid_mask=torch.ones(1, 1, dtype=torch.float32),
        target_modality="vision",
        gate_down_margin=0.05,
        attention_down_margin=0.10,
        final_score_down_margin=0.08,
        nontarget_rescue_margin=0.02,
        alignment_margin=0.01,
    )
    assert target_gate_down_loss.item() > 0.0
    assert target_attention_down_loss.item() > 0.0
    assert target_final_score_down_loss.item() > 0.0
    assert nontarget_rescue_loss.item() > 0.0
    assert reliability_gate_alignment_loss.item() == pytest.approx(0.0)
    assert gate_attention_alignment_loss.item() == pytest.approx(0.0)
    assert reliability_attention_alignment_loss.item() == pytest.approx(0.0)
    assert gate_attention_alignment_ratio.item() == pytest.approx(1.0)
    assert reliability_gate_alignment_ratio.item() == pytest.approx(1.0)
    assert alignment_ratio.item() == pytest.approx(1.0)


def test_run_stage2_candidate_separates_smoke_and_full_seed_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stage2_transition_helpers as helpers

    calls: list[tuple[str, Path]] = []

    class FakeTrainer:
        def __init__(
            self,
            config: dict[str, object],
            candidate_name: str,
            seed: int,
            run_dir: Path,
            smoke: bool = False,
        ) -> None:
            self.seed = seed
            self.run_dir = run_dir
            calls.append(("init", run_dir))

        def train(self) -> None:
            calls.append(("train", self.run_dir))

        def run_eval_suite(self, num_eval_episodes: int, run_mode: str = "full") -> list[dict[str, object]]:
            return [
                {
                    "candidate_name": "cand",
                    "probe_name": "normal_eval",
                    "corruption_type": "normal_eval",
                    "seed": self.seed,
                    "eval_success_rate": 1.0,
                    "eval_return": 1.0,
                    "num_eval_episodes": num_eval_episodes,
                    "run_mode": run_mode,
                }
            ]

    monkeypatch.setattr(helpers, "Stage2TransitionTrainer", FakeTrainer)
    config = {"seeds": [0], "num_eval_episodes": 24, "num_eval_episodes_smoke": 6}

    run_stage2_candidate("cand", config, tmp_path, "rows.json", smoke=True)
    run_stage2_candidate("cand", config, tmp_path, "rows.json", smoke=False)

    assert (tmp_path / "cand" / "seed_0_smoke" / "rows.json").exists()
    assert (tmp_path / "cand" / "seed_0" / "rows.json").exists()
    assert (tmp_path / "cand" / "seed_0_smoke" / "run_metadata.json").exists()
    assert (tmp_path / "cand" / "seed_0" / "run_metadata.json").exists()
    assert len([entry for entry in calls if entry[0] == "train"]) == 2


def test_run_stage2_candidate_reruns_on_metadata_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stage2_transition_helpers as helpers

    train_calls = 0

    class FakeTrainer:
        def __init__(
            self,
            config: dict[str, object],
            candidate_name: str,
            seed: int,
            run_dir: Path,
            smoke: bool = False,
        ) -> None:
            self.seed = seed
            self.run_dir = run_dir

        def train(self) -> None:
            nonlocal train_calls
            train_calls += 1

        def run_eval_suite(self, num_eval_episodes: int, run_mode: str = "full") -> list[dict[str, object]]:
            return [
                {
                    "candidate_name": "cand",
                    "probe_name": "normal_eval",
                    "corruption_type": "normal_eval",
                    "seed": self.seed,
                    "eval_success_rate": 1.0,
                    "eval_return": 1.0,
                    "num_eval_episodes": num_eval_episodes,
                    "run_mode": run_mode,
                }
            ]

    monkeypatch.setattr(helpers, "Stage2TransitionTrainer", FakeTrainer)
    config = {"seeds": [0], "num_eval_episodes": 24, "num_eval_episodes_smoke": 6}

    run_stage2_candidate("cand", config, tmp_path, "rows.json", smoke=True)
    metadata_path = tmp_path / "cand" / "seed_0_smoke" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_hash"] = "badbadbadbadbadb"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    run_stage2_candidate("cand", config, tmp_path, "rows.json", smoke=True)

    assert train_calls == 2


def test_run_stage2_eval_only_candidate_separates_smoke_and_full_seed_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stage2_transition_helpers as helpers

    calls: list[str] = []

    class FakeTrainer:
        def __init__(
            self,
            config: dict[str, object],
            candidate_name: str,
            seed: int,
            run_dir: Path,
            smoke: bool = False,
        ) -> None:
            self.seed = seed

        def run_eval_suite(self, num_eval_episodes: int, run_mode: str = "full") -> list[dict[str, object]]:
            calls.append(run_mode)
            return [
                {
                    "candidate_name": "cand",
                    "probe_name": "normal_eval",
                    "corruption_type": "normal_eval",
                    "seed": self.seed,
                    "eval_success_rate": 1.0,
                    "eval_return": 1.0,
                    "num_eval_episodes": num_eval_episodes,
                    "run_mode": run_mode,
                }
            ]

    monkeypatch.setattr(helpers, "Stage2TransitionTrainer", FakeTrainer)
    config = {"seeds": [0], "num_eval_episodes": 24, "num_eval_episodes_smoke": 6}

    run_stage2_eval_only_candidate("cand", config, tmp_path, "rows.json", smoke=True)
    run_stage2_eval_only_candidate("cand", config, tmp_path, "rows.json", smoke=False)

    assert (tmp_path / "cand" / "seed_0_smoke" / "rows.json").exists()
    assert (tmp_path / "cand" / "seed_0" / "rows.json").exists()
    assert calls == ["smoke", "full"]


def test_run_reliability_proof_candidate_separates_smoke_and_full_seed_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_stage2_reliability_proof as proof

    checkpoint_root = tmp_path / "checkpoints" / "cand"
    seed_root = checkpoint_root / "seed_0"
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "checkpoint_final.pt").write_text("stub", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(proof, "build_model", lambda *args, **kwargs: object())

    def fake_run_probe(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["run_mode"]))
        return {
            "probe_name": "normal_eval",
            "corruption_type": "normal_eval",
            "eval_success_rate": 1.0,
            "eval_return": 1.0,
            "num_eval_episodes": int(kwargs["num_eval_episodes"]),
            "run_mode": str(kwargs["run_mode"]),
        }

    monkeypatch.setattr(proof, "_run_probe", fake_run_probe)
    config = {
        "device": "cpu",
        "seeds": [0],
        "num_eval_episodes": 24,
        "num_eval_episodes_smoke": 6,
        "eval_probes": [{"name": "normal_eval", "kind": "clean", "strength": 0.0}],
        "base_stage1_config_path": str(tmp_path / "base.yaml"),
    }
    candidate_spec = {"checkpoint_root": str(checkpoint_root)}

    run_reliability_proof_candidate("cand", candidate_spec, config, tmp_path / "out", smoke=True)
    run_reliability_proof_candidate("cand", candidate_spec, config, tmp_path / "out", smoke=False)

    assert (tmp_path / "out" / "cand" / "seed_0_smoke" / "eval_rows.json").exists()
    assert (tmp_path / "out" / "cand" / "seed_0" / "eval_rows.json").exists()
    assert calls == ["smoke", "full"]


def test_choose_ego_anchor_recommendation_prefers_simple_pass_candidate() -> None:
    score_df = pd.DataFrame(
        [
            {"candidate_name": "ego_anchor_target_residual", "pass_fail": "PASS"},
            {"candidate_name": "ego_anchor_target_residual_plus_margin", "pass_fail": "PASS"},
        ]
    )
    assert choose_ego_anchor_recommendation(score_df) == "ego_anchor_target_residual"

    score_df = pd.DataFrame(
        [
            {"candidate_name": "ego_anchor_target_residual", "pass_fail": "WARN"},
            {"candidate_name": "ego_anchor_target_residual_plus_margin", "pass_fail": "PASS"},
        ]
    )
    assert choose_ego_anchor_recommendation(score_df) == "ego_anchor_target_residual_plus_margin"


def test_choose_ego_fix_recommendation_prefers_simple_pass_candidate() -> None:
    score_df = pd.DataFrame(
        [
            {"candidate_name": "ego_raw_observable_adapter", "pass_fail": "PASS"},
            {"candidate_name": "ego_raw_observable_adapter_plus_ego_positive", "pass_fail": "PASS"},
        ]
    )
    assert choose_ego_fix_recommendation(score_df) == "ego_raw_observable_adapter"

    score_df = pd.DataFrame(
        [
            {"candidate_name": "ego_raw_observable_adapter", "pass_fail": "WARN"},
            {"candidate_name": "ego_raw_observable_adapter_plus_ego_positive", "pass_fail": "PASS"},
        ]
    )
    assert choose_ego_fix_recommendation(score_df) == "ego_raw_observable_adapter_plus_ego_positive"


def test_choose_ego_residual_recommendation_prefers_simple_pass_candidate() -> None:
    score_df = pd.DataFrame(
        [
            {"candidate_name": "ego_residual_only", "pass_fail": "PASS"},
            {"candidate_name": "ego_residual_plus_weak_positive", "pass_fail": "PASS"},
        ]
    )
    assert choose_ego_residual_recommendation(score_df) == "ego_residual_only"

    score_df = pd.DataFrame(
        [
            {"candidate_name": "ego_residual_only", "pass_fail": "WARN"},
            {"candidate_name": "ego_residual_plus_weak_positive", "pass_fail": "PASS"},
        ]
    )
    assert choose_ego_residual_recommendation(score_df) == "ego_residual_plus_weak_positive"


def test_choose_selective_residual_recommendation_prefers_simple_pass_candidate() -> None:
    score_df = pd.DataFrame(
        [
            {"candidate_name": "vision_ego_selective_residual", "pass_fail": "PASS"},
            {"candidate_name": "vision_ego_selective_residual_plus_weak_positive", "pass_fail": "PASS"},
        ]
    )
    assert choose_selective_residual_recommendation(score_df) == "vision_ego_selective_residual"

    score_df = pd.DataFrame(
        [
            {"candidate_name": "vision_ego_selective_residual", "pass_fail": "WARN"},
            {"candidate_name": "vision_ego_selective_residual_plus_weak_positive", "pass_fail": "PASS"},
        ]
    )
    assert choose_selective_residual_recommendation(score_df) == "vision_ego_selective_residual_plus_weak_positive"


def test_reliability_proof_summary_includes_clean_reliability_drift_metrics(tmp_path: Path) -> None:
    baseline_rows = _build_reliability_proof_eval_rows().to_dict(orient="records")
    baseline_path = tmp_path / "baseline_eval_rows.json"
    baseline_path.write_text(json.dumps(baseline_rows), encoding="utf-8")

    candidate_df = _build_reliability_proof_eval_rows().copy()
    candidate_df.loc[candidate_df["probe_name"] == "normal_eval", "reliability_mean_vision"] = 0.90
    candidate_df.loc[candidate_df["probe_name"] == "normal_eval", "reliability_mean_distance"] = 0.81
    candidate_df.loc[candidate_df["probe_name"] == "normal_eval", "reliability_mean_ego"] = 0.79
    candidate_df.loc[candidate_df["probe_name"] == "normal_eval", "reliability_mean_route"] = 0.74
    candidate_df.loc[candidate_df["probe_name"] == "normal_eval", "reliability_vision_residual_delta_mean"] = 0.02
    candidate_df.loc[candidate_df["probe_name"] == "normal_eval", "reliability_ego_residual_delta_mean"] = 0.01
    candidate_df.loc[candidate_df["probe_name"] == "mild_vision", "reliability_vision_residual_delta_mean"] = -0.02
    candidate_df.loc[candidate_df["probe_name"] == "mild_ego", "reliability_ego_residual_delta_mean"] = -0.03
    candidate_df["num_eval_episodes"] = 24

    reliability_cmp = compute_candidate_reliability_delta(candidate_df)
    target_df = compute_reliability_target_metrics(reliability_cmp)
    summary = build_reliability_proof_score_summary(
        "cand",
        candidate_df,
        target_df,
        {
            "baseline_eval_rows_path": str(baseline_path),
            "scoring": {
                "required_target_modalities": 4,
                "clean_success_rate_floor": 0.76,
                "target_positive_ratio_pass": 0.75,
                "target_lowest_ratio_pass": 0.50,
                "mean_target_vs_nontarget_gap_pass": 0.0,
                "vision_positive_ratio_pass": 1.0,
                "distance_positive_ratio_pass": 1.0,
                "ego_positive_ratio_pass": 1.0,
                "route_positive_ratio_pass": 1.0,
                "max_non_ego_clean_reliability_drift_pass": 0.02,
                "vision_residual_delta_mild_minus_clean_max": 0.0,
                "ego_residual_delta_mild_minus_clean_max": 0.0,
            },
        },
    )
    assert summary["vision_clean_reliability_drift"] == pytest.approx(0.0)
    assert summary["distance_clean_reliability_drift"] == pytest.approx(0.01)
    assert summary["ego_clean_reliability_drift"] == pytest.approx(-0.06)
    assert summary["route_clean_reliability_drift"] == pytest.approx(-0.04)
    assert summary["max_non_ego_clean_reliability_drift"] == pytest.approx(0.04)
    assert summary["vision_residual_delta_clean_mean"] == pytest.approx(0.02)
    assert summary["vision_residual_delta_mild_mean"] == pytest.approx(-0.02)
    assert summary["vision_residual_delta_mild_minus_clean"] == pytest.approx(-0.04)
    assert summary["ego_residual_delta_clean_mean"] == pytest.approx(0.01)
    assert summary["ego_residual_delta_mild_mean"] == pytest.approx(-0.03)
    assert summary["ego_residual_delta_mild_minus_clean"] == pytest.approx(-0.04)
    assert summary["min_eval_episodes_per_row"] == 24
    assert summary["max_eval_episodes_per_row"] == 24
    assert summary["mixed_eval_episode_count_flag"] == 0


def test_reliability_proof_summary_detects_mixed_eval_episode_counts() -> None:
    candidate_df = _build_reliability_proof_eval_rows().copy()
    candidate_df["num_eval_episodes"] = [6, 24, 24, 24, 24]
    reliability_cmp = compute_candidate_reliability_delta(candidate_df)
    target_df = compute_reliability_target_metrics(reliability_cmp)
    summary = build_reliability_proof_score_summary(
        "cand",
        candidate_df,
        target_df,
        {
            "scoring": {
                "required_target_modalities": 4,
                "clean_success_rate_floor": 0.76,
                "target_positive_ratio_pass": 0.75,
                "target_lowest_ratio_pass": 0.50,
                "mean_target_vs_nontarget_gap_pass": 0.0,
            }
        },
    )
    assert summary["mixed_eval_episode_count_flag"] == 1
    assert evaluate_reliability_proof_summary(summary, {"required_target_modalities": 4}) == "WARN"


def test_v3_7f_metric_threshold_wiring_and_source_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "env_mode": "normal",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.15,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.3,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.2,
                "attention_relative_gap_norm_mean": 0.5,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.7,
                "reliability_mean_ego": 0.7,
                "reliability_mean_route": 0.7,
                "logit_mean_vision": 0.4,
                "logit_mean_vision_clean": 0.0,
                "logit_mean_vision_mild": 0.0,
                "attention_bias_mean_vision": 0.1,
                "attention_bias_mean_distance": 0.0,
                "attention_bias_mean_ego": 0.0,
                "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
                "attention_bias_gate_mean_vision": 0.01,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_bias_gate_residual_mean_vision": -0.01,
                "attention_score_adapter_delta_norm_mean_vision": 0.05,
                "attention_score_adapter_delta_norm_mean_distance": 0.04,
                "attention_score_adapter_delta_norm_mean_ego": 0.04,
                "attention_score_adapter_delta_norm_mean_route": 0.04,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "source_sign_score_anchor_error": 0.003,
                "source_clean_gate_preserve_error": 0.004,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "env_mode": "mild",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.1,
                "attention_score_mean_ego": 0.1,
                "attention_score_mean_route": 0.0,
                "attention_relative_gap_score_mean_vision_distance": 0.4,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.25,
                "attention_relative_gap_norm_mean": 0.6,
                "attention_mean_vision": 0.5,
                "attention_mean_distance": 0.18,
                "attention_mean_ego": 0.17,
                "attention_mean_route": 0.15,
                "targetlogit_shapley_mean_vision": 0.1,
                "reliability_mean_vision": 0.8,
                "reliability_mean_distance": 0.7,
                "reliability_mean_ego": 0.7,
                "reliability_mean_route": 0.7,
                "logit_mean_vision": 0.3,
                "logit_mean_vision_clean": 0.0,
                "logit_mean_vision_mild": 0.0,
                "attention_bias_mean_vision": 0.2,
                "attention_bias_mean_distance": 0.0,
                "attention_bias_mean_ego": 0.0,
                "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.2,
                "attention_bias_abs_mean": 0.2,
                "attention_bias_gate_mean_vision": 0.09,
                "attention_bias_gate_mean_distance": 0.02,
                "attention_bias_gate_mean_ego": 0.02,
                "attention_bias_gate_mean_route": 0.02,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.06,
                "attention_score_adapter_delta_norm_mean_vision": 0.06,
                "attention_score_adapter_delta_norm_mean_distance": 0.05,
                "attention_score_adapter_delta_norm_mean_ego": 0.05,
                "attention_score_adapter_delta_norm_mean_route": 0.05,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "source_sign_score_anchor_error": 0.003,
                "source_clean_gate_preserve_error": 0.004,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
        ]
    )
    metrics = compute_v3_7f_source_gate_residual_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics.iloc[0]
    assert row["vision_gate_clean_below_cap_ratio"] == 1.0
    assert row["vision_gate_mild_above_floor_ratio"] == 1.0
    assert row["vision_gate_gap_margin_satisfied_ratio"] == 1.0
    assert row["vision_source_gate_clean_mean"] == 0.04
    assert row["vision_gate_residual_mild_mean"] == 0.06


def test_teacher_loss_overrides_apply_gate_settings() -> None:
    model = _build_stage1_model()
    apply_attention_loss_overrides_to_model(
        model,
        {
            "use_relative_score_head": True,
            "use_reliability_bias_head": True,
            "use_reliability_bias_gate": True,
            "gate_mode": "absolute_v1",
            "gate_input_mode": "legacy",
            "reliability_bias_gate_temperature": 8.0,
            "reliability_bias_gate_threshold": 0.52,
            "reliability_bias_residual_scale": 0.22,
        },
    )
    fusion = model.fusion_module
    assert fusion.use_relative_score_head is True
    assert fusion.use_reliability_bias_head is True
    assert fusion.use_reliability_bias_gate is True
    assert fusion.gate_mode == "absolute_v1"
    assert fusion.gate_input_mode == "legacy"
    assert fusion.reliability_bias_gate_temperature == 8.0
    assert fusion.reliability_bias_gate_threshold == 0.52
    assert fusion.reliability_bias_residual_scale == 0.22


def test_v3_7f1_metric_includes_reconstruction_and_source_gate_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "env_mode": "normal",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.15,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.3,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.2,
                "attention_relative_gap_norm_mean": 0.5,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.7,
                "reliability_mean_ego": 0.7,
                "reliability_mean_route": 0.7,
                "logit_mean_vision": 0.4,
                "logit_mean_vision_clean": 0.0,
                "logit_mean_vision_mild": 0.0,
                "attention_bias_mean_vision": 0.1,
                "attention_bias_mean_distance": 0.0,
                "attention_bias_mean_ego": 0.0,
                "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
                "attention_bias_gate_mean_vision": 0.01,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.05,
                "attention_score_adapter_delta_norm_mean_distance": 0.04,
                "attention_score_adapter_delta_norm_mean_ego": 0.04,
                "attention_score_adapter_delta_norm_mean_route": 0.04,
                "gate_reconstruction_error_mean": 0.0,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "source_sign_score_anchor_error": 0.003,
                "source_clean_gate_preserve_error": 0.004,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "env_mode": "mild",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.1,
                "attention_score_mean_ego": 0.1,
                "attention_score_mean_route": 0.0,
                "attention_relative_gap_score_mean_vision_distance": 0.4,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.25,
                "attention_relative_gap_norm_mean": 0.6,
                "attention_mean_vision": 0.5,
                "attention_mean_distance": 0.18,
                "attention_mean_ego": 0.17,
                "attention_mean_route": 0.15,
                "targetlogit_shapley_mean_vision": 0.1,
                "reliability_mean_vision": 0.8,
                "reliability_mean_distance": 0.7,
                "reliability_mean_ego": 0.7,
                "reliability_mean_route": 0.7,
                "logit_mean_vision": 0.3,
                "logit_mean_vision_clean": 0.0,
                "logit_mean_vision_mild": 0.0,
                "attention_bias_mean_vision": 0.2,
                "attention_bias_mean_distance": 0.0,
                "attention_bias_mean_ego": 0.0,
                "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.2,
                "attention_bias_abs_mean": 0.2,
                "attention_bias_gate_mean_vision": 0.09,
                "attention_bias_gate_mean_distance": 0.02,
                "attention_bias_gate_mean_ego": 0.02,
                "attention_bias_gate_mean_route": 0.02,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.06,
                "attention_score_adapter_delta_norm_mean_vision": 0.06,
                "attention_score_adapter_delta_norm_mean_distance": 0.05,
                "attention_score_adapter_delta_norm_mean_ego": 0.05,
                "attention_score_adapter_delta_norm_mean_route": 0.05,
                "gate_reconstruction_error_mean": 0.0,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "source_sign_score_anchor_error": 0.003,
                "source_clean_gate_preserve_error": 0.004,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
        ]
    )
    metrics = compute_v3_7f1_source_gate_distilled_residual_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics.iloc[0]
    assert row["vision_source_gate_clean_mean"] == 0.04
    assert row["vision_source_gate_mild_mean"] == 0.03
    assert row["vision_gate_residual_mild_mean"] == 0.06
    assert row["gate_reconstruction_error_mean"] == 0.0


def test_forward_sequence_emits_attention_bias_gates() -> None:
    model = _build_stage1_model()
    model.fusion_module.use_relative_score_head = True
    model.fusion_module.use_reliability_bias_head = True
    model.fusion_module.use_reliability_bias_gate = True
    model.fusion_module.relative_gap_residual_scale = 0.2
    model.fusion_module.reliability_bias_residual_scale = 0.2
    obs_seq = {
        "vision": torch.randn(1, 2, 8),
        "distance": torch.randn(1, 2, 9),
        "ego": torch.randn(1, 2, 5),
        "route": torch.randn(1, 2, 5),
    }
    outputs = model.forward_sequence(
        obs_seq=obs_seq,
        episode_starts=torch.zeros(1, 2, dtype=torch.bool),
        initial_hidden=model.initial_hidden(1, torch.device("cpu")),
        actions=None,
        deterministic=False,
    )
    assert "attention_bias_gates" in outputs
    gates = outputs["attention_bias_gates"]
    assert gates.shape == (1, 2, 4)
    assert torch.isfinite(gates).all()
    assert torch.isfinite(gates.mean())


def test_forward_sequence_emits_adapter_gate_hidden_norms() -> None:
    model = _build_stage1_model()
    fusion = model.fusion_module
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_logit_v2"
    fusion.gate_input_mode = "source_residual_adapter_delta_aware_v2"
    fusion.adapter_gate_input_mode = "dedicated_v1"
    obs_seq = {
        "vision": torch.randn(1, 2, 8),
        "distance": torch.randn(1, 2, 9),
        "ego": torch.randn(1, 2, 5),
        "route": torch.randn(1, 2, 5),
    }
    zeros = torch.zeros(1, 2, 4)
    context = {
        "clean_teacher_scores": zeros,
        "sign_teacher_scores": zeros,
        "source_gate_scores": torch.full((1, 2, 4), 0.05),
        "clean_teacher_adapter_delta": torch.zeros(1, 2, 64),
        "sign_teacher_adapter_delta": torch.zeros(1, 2, 64),
    }
    outputs = model.forward_sequence(
        obs_seq=obs_seq,
        episode_starts=torch.zeros(1, 2, dtype=torch.bool),
        initial_hidden=model.initial_hidden(1, torch.device("cpu")),
        actions=None,
        deterministic=False,
        gate_feature_context=context,
    )
    assert "attention_bias_gate_adapter_hidden" in outputs
    assert "attention_bias_gate_adapter_hidden_norms" in outputs
    assert outputs["attention_bias_gate_adapter_hidden"].shape[:3] == (1, 2, 4)
    assert outputs["attention_bias_gate_adapter_hidden_norms"].shape == (1, 2, 4)


def test_projection_norm_floor_loss_positive_below_floor() -> None:
    projection = torch.tensor([[0.1, 0.0]], dtype=torch.float32)
    loss = compute_projection_norm_floor_loss(projection=projection, min_norm=0.5)
    assert loss.item() > 0.0


def test_source_residual_bias_aware_v1_supports_standalone_no_context() -> None:
    model = _build_stage1_model()
    fusion = model.fusion_module
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_logit_v2"
    fusion.gate_input_mode = "source_residual_bias_aware_v1"
    fusion.gate_use_source_gate_feature = True
    fusion.gate_use_teacher_delta_feature = True
    fusion.gate_use_score_delta_feature = True

    fused, aux = model.encode_and_fuse(_sample_obs(), gate_feature_context=None)

    assert fused.shape == (1, 64)
    assert "attention_bias_gates" in aux
    assert aux["attention_bias_gates"].shape == (1, len(MODALITY_NAMES))


def test_source_residual_adapter_delta_aware_v2_supports_standalone_no_context() -> None:
    model = _build_stage1_model()
    fusion = model.fusion_module
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_logit_v2"
    fusion.gate_input_mode = "source_residual_adapter_delta_aware_v2"
    fusion.gate_use_source_gate_feature = True
    fusion.gate_use_teacher_delta_feature = True
    fusion.gate_use_adapter_teacher_delta_feature = True
    fusion.gate_use_score_delta_feature = True
    fusion.adapter_gate_input_mode = "dedicated_v1"

    fused, aux = model.encode_and_fuse(_sample_obs(), gate_feature_context=None)

    assert fused.shape == (1, 64)
    assert "attention_bias_gates" in aux
    assert aux["attention_bias_gates"].shape == (1, len(MODALITY_NAMES))


def test_stage2_reference_corrupt_gate_context_uses_teacher_prebias_scores_and_source_gate() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.use_gate_feature_context = True
    trainer.raw_config = {"losses": {"proof_teacher_mode": "reference_corrupt", "teacher_target_shift_margin": 0.0}}
    trainer.device = torch.device("cpu")
    clean_outputs = {
        "attention_relative_scores_before_bias": torch.full((1, 2, 4), 0.6),
        "attention_scores": torch.full((1, 2, 4), 0.3),
        "attention_source_bias_gates": torch.full((1, 2, 4), 0.15),
        "attention_bias_gates": torch.full((1, 2, 4), 0.2),
        "attention_score_adapter_delta": torch.full((1, 2, 64), 0.1),
    }
    corrupt_outputs = {
        "attention_relative_scores_before_bias": torch.full((1, 2, 4), 0.2),
        "attention_scores": torch.full((1, 2, 4), 0.1),
        "attention_source_bias_gates": torch.full((1, 2, 4), 0.04),
        "attention_bias_gates": torch.full((1, 2, 4), 0.05),
        "attention_score_adapter_delta": torch.full((1, 2, 64), 0.4),
    }
    context = trainer._build_sequence_gate_feature_context(
        reference_clean_outputs=clean_outputs,
        target_mask=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupted_mask=torch.ones(1, 2, dtype=torch.float32),
        sign_teacher_outputs=corrupt_outputs,
    )
    assert torch.allclose(context["clean_teacher_scores"], clean_outputs["attention_relative_scores_before_bias"])
    assert torch.allclose(context["sign_teacher_scores"], corrupt_outputs["attention_relative_scores_before_bias"])
    assert torch.allclose(context["source_gate_scores"], corrupt_outputs["attention_source_bias_gates"])
    assert torch.allclose(context["clean_teacher_adapter_delta"], clean_outputs["attention_score_adapter_delta"])
    assert torch.allclose(context["sign_teacher_adapter_delta"], corrupt_outputs["attention_score_adapter_delta"])


def test_source_checkpoint_teacher_preserves_loaded_fusion_settings() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.raw_config = {
        "teacher_reference_model_mode": "source_checkpoint",
        "teacher_reference_checkpoint_root": "/tmp/teacher",
        "fusion_gate": {
            "use_relative_score_head": True,
            "reliability_bias_residual_scale": 0.05,
        },
        "losses": {},
    }
    trainer.anchor_reference_model = _build_stage1_model()
    teacher_model = _build_stage1_model()
    teacher_model.fusion_module.use_relative_score_head = False
    teacher_model.fusion_module.reliability_bias_residual_scale = 0.33
    trainer._build_model_from_checkpoint_root = lambda checkpoint_root, snapshot_path_override=None: teacher_model
    model = trainer._build_teacher_reference_model({})
    assert model is teacher_model
    assert model.fusion_module.use_relative_score_head is False
    assert model.fusion_module.reliability_bias_residual_scale == pytest.approx(0.33)


def test_source_checkpoint_teacher_candidate_name_resolves_candidate_subdir(tmp_path: Path) -> None:
    teacher_root = tmp_path / "teacher_root"
    candidate_root = teacher_root / "strong_teacher"
    candidate_root.mkdir(parents=True)
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.raw_config = {
        "teacher_reference_model_mode": "source_checkpoint",
        "teacher_reference_checkpoint_root": str(teacher_root),
        "teacher_reference_candidate_name": "strong_teacher",
        "losses": {},
    }
    trainer.anchor_reference_model = _build_stage1_model()
    teacher_model = _build_stage1_model()
    seen_root: dict[str, Path] = {}

    def _fake_build(checkpoint_root, snapshot_path_override=None):
        seen_root["path"] = checkpoint_root
        return teacher_model

    trainer._build_model_from_checkpoint_root = _fake_build
    model = trainer._build_teacher_reference_model({})
    assert model is teacher_model
    assert seen_root["path"] == candidate_root


def test_stage2_teacher_distilled_gate_losses_and_preserve_indices() -> None:
    assert select_stage2_gate_teacher_preserve_indices(True) == [1, 2, 3]
    (
        clean_gate_cap_loss,
        mild_gate_floor_loss,
        gate_gap_margin_loss,
        source_gate_distill_loss,
        source_clean_gate_preserve_loss,
        clean_score_anchor_loss,
        clean_action_anchor_loss,
        source_gate_residual_zero_loss,
        vision_final_gate_target_loss,
        nonvision_gate_stability_loss,
    ) = compute_stage2_teacher_distilled_gate_losses(
        clean_gate_scores=torch.tensor([[[0.05, 0.03, 0.03, 0.03]]], dtype=torch.float32),
        corrupt_gate_scores=torch.tensor([[[0.02, 0.08, 0.07, 0.07]]], dtype=torch.float32),
        clean_teacher_gate_scores=torch.tensor([[[0.04, 0.01, 0.01, 0.01]]], dtype=torch.float32),
        corrupt_teacher_gate_scores=torch.tensor([[[0.01, 0.02, 0.02, 0.02]]], dtype=torch.float32),
        clean_gate_residual_logits=torch.tensor([[[0.02, 0.03, 0.03, 0.03]]], dtype=torch.float32),
        clean_attention_scores_standalone=torch.tensor([[[0.30, 0.10, 0.10, 0.10]]], dtype=torch.float32),
        reference_clean_attention_scores=torch.tensor([[[0.25, 0.10, 0.10, 0.10]]], dtype=torch.float32),
        clean_action_means_standalone=torch.tensor([[[0.20, -0.20]]], dtype=torch.float32),
        reference_clean_action_means=torch.tensor([[[0.10, -0.10]]], dtype=torch.float32),
        target_mask=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupted_mask=torch.ones(1, 1, dtype=torch.float32),
        valid_mask=torch.ones(1, 1, dtype=torch.float32),
        target_modality="vision",
        clean_gate_max=0.02,
        mild_gate_min=0.08,
        gate_gap_margin=0.05,
        disable_vision_final_gate_teacher_preserve=True,
    )
    assert clean_gate_cap_loss.item() > 0.0
    assert mild_gate_floor_loss.item() > 0.0
    assert gate_gap_margin_loss.item() > 0.0
    assert source_gate_distill_loss.item() > 0.0
    assert source_clean_gate_preserve_loss.item() > 0.0
    assert clean_score_anchor_loss.item() > 0.0
    assert clean_action_anchor_loss.item() > 0.0
    assert source_gate_residual_zero_loss.item() > 0.0
    assert vision_final_gate_target_loss.item() > 0.0
    assert nonvision_gate_stability_loss.item() > 0.0


def test_attention_bias_gate_effect_helper_and_losses() -> None:
    clean_effects = compute_attention_bias_gate_effects(
        attention_bias_scores=torch.tensor([[[0.5, 0.1, 0.1, 0.1]]], dtype=torch.float32),
        attention_bias_gates=torch.tensor([[[0.4, 0.2, 0.2, 0.2]]], dtype=torch.float32),
        scale=0.05,
        attention_temperature=2.0,
    )
    corrupt_effects = compute_attention_bias_gate_effects(
        attention_bias_scores=torch.tensor([[[0.1, 0.1, 0.1, 0.1]]], dtype=torch.float32),
        attention_bias_gates=torch.tensor([[[0.1, 0.2, 0.2, 0.2]]], dtype=torch.float32),
        scale=0.05,
        attention_temperature=2.0,
    )
    assert clean_effects.shape == (1, 1, 4)
    assert clean_effects[0, 0, 0].item() == pytest.approx(0.005)
    assert clean_effects[0, 0, 0].item() > corrupt_effects[0, 0, 0].item()
    (
        target_gate_effect_down_loss,
        target_attention_down_loss,
        target_final_score_down_loss,
        target_bias_score_down_loss,
        nontarget_rescue_loss,
        reliability_gate_effect_alignment_ratio,
        gate_effect_attention_alignment_ratio,
        gate_effect_score_alignment_ratio,
        gate_effect_score_alignment_loss,
    ) = compute_stage2_fusion_shift_gate_effect_losses(
        clean_attention_weights=torch.tensor([[[0.50, 0.16, 0.16, 0.18]]], dtype=torch.float32),
        corrupt_attention_weights=torch.tensor([[[0.20, 0.20, 0.20, 0.40]]], dtype=torch.float32),
        clean_final_scores=torch.tensor([[[0.25, 0.05, 0.05, 0.05]]], dtype=torch.float32),
        corrupt_final_scores=torch.tensor([[[0.05, 0.06, 0.06, 0.08]]], dtype=torch.float32),
        clean_gate_effects=clean_effects,
        corrupt_gate_effects=corrupt_effects,
        clean_bias_scores=torch.tensor([[[0.50, 0.10, 0.10, 0.10]]], dtype=torch.float32),
        corrupt_bias_scores=torch.tensor([[[0.10, 0.10, 0.10, 0.10]]], dtype=torch.float32),
        clean_reliability_scores=torch.tensor([[[0.85, 0.80, 0.80, 0.80]]], dtype=torch.float32),
        corrupt_reliability_scores=torch.tensor([[[0.50, 0.80, 0.80, 0.80]]], dtype=torch.float32),
        target_mask=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupted_mask=torch.ones(1, 1, dtype=torch.float32),
        valid_mask=torch.ones(1, 1, dtype=torch.float32),
        target_modality="vision",
        gate_effect_down_margin=0.015,
        attention_down_margin=0.05,
        final_score_down_margin=0.05,
        bias_score_down_margin=0.45,
        nontarget_rescue_margin=0.0,
        alignment_margin=0.001,
        score_alignment_margin=0.10,
    )
    assert target_gate_effect_down_loss.item() > 0.0
    assert target_attention_down_loss.item() == pytest.approx(0.0)
    assert target_final_score_down_loss.item() == pytest.approx(0.0)
    assert target_bias_score_down_loss.item() > 0.0
    assert nontarget_rescue_loss.item() == pytest.approx(0.0)
    assert reliability_gate_effect_alignment_ratio.item() == pytest.approx(1.0)
    assert gate_effect_attention_alignment_ratio.item() == pytest.approx(1.0)
    assert gate_effect_score_alignment_ratio.item() == pytest.approx(1.0)
    assert gate_effect_score_alignment_loss.item() == pytest.approx(0.0)


def test_stage2_fusion_shift_prebias_losses() -> None:
    (
        target_prebias_score_down_loss,
        prebias_score_final_score_alignment_ratio,
        prebias_score_alignment_loss,
    ) = compute_stage2_fusion_shift_prebias_losses(
        clean_prebias_scores=torch.tensor([[[0.40, 0.05, 0.05, 0.05]]], dtype=torch.float32),
        corrupt_prebias_scores=torch.tensor([[[0.10, 0.05, 0.05, 0.05]]], dtype=torch.float32),
        clean_final_scores=torch.tensor([[[0.25, 0.05, 0.05, 0.05]]], dtype=torch.float32),
        corrupt_final_scores=torch.tensor([[[0.10, 0.05, 0.05, 0.05]]], dtype=torch.float32),
        target_mask=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupted_mask=torch.ones(1, 1, dtype=torch.float32),
        valid_mask=torch.ones(1, 1, dtype=torch.float32),
        target_modality="vision",
        prebias_score_down_margin=0.35,
        alignment_margin=0.01,
        score_alignment_margin=0.10,
    )
    assert target_prebias_score_down_loss.item() > 0.0
    assert prebias_score_final_score_alignment_ratio.item() == pytest.approx(1.0)
    assert prebias_score_alignment_loss.item() == pytest.approx(0.0)


def test_stage2_teacher_gate_effect_losses() -> None:
    (
        teacher_nonvision_gate_effect_preserve_loss,
        clean_attention_weight_anchor_loss,
        clean_reliability_anchor_loss,
        clean_bias_score_anchor_loss,
        clean_prebias_score_anchor_loss,
    ) = compute_stage2_teacher_gate_effect_losses(
        clean_gate_effects=torch.tensor([[[0.03, 0.01, 0.01, 0.01]]], dtype=torch.float32),
        corrupt_gate_effects=torch.tensor([[[-0.01, 0.03, 0.02, 0.02]]], dtype=torch.float32),
        clean_teacher_gate_effects=torch.tensor([[[0.02, 0.00, 0.00, 0.00]]], dtype=torch.float32),
        corrupt_teacher_gate_effects=torch.tensor([[[-0.02, 0.01, 0.01, 0.01]]], dtype=torch.float32),
        clean_attention_weights_standalone=torch.tensor([[[0.20, 0.15, 0.15, 0.50]]], dtype=torch.float32),
        anchor_clean_attention_weights=torch.tensor([[[0.18, 0.15, 0.15, 0.52]]], dtype=torch.float32),
        clean_reliability_scores_standalone=torch.tensor([[[0.80, 0.75, 0.70, 0.85]]], dtype=torch.float32),
        anchor_clean_reliability_scores=torch.tensor([[[0.82, 0.75, 0.72, 0.84]]], dtype=torch.float32),
        clean_bias_scores_standalone=torch.tensor([[[0.20, 0.10, 0.10, 0.05]]], dtype=torch.float32),
        anchor_clean_bias_scores=torch.tensor([[[0.18, 0.10, 0.10, 0.05]]], dtype=torch.float32),
        clean_prebias_scores_standalone=torch.tensor([[[0.30, 0.10, 0.10, 0.05]]], dtype=torch.float32),
        anchor_clean_prebias_scores=torch.tensor([[[0.28, 0.10, 0.10, 0.05]]], dtype=torch.float32),
        target_mask=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupted_mask=torch.ones(1, 1, dtype=torch.float32),
        valid_mask=torch.ones(1, 1, dtype=torch.float32),
        target_modality="vision",
        disable_vision_final_gate_teacher_preserve=True,
        disable_teacher_target_distill=True,
        teacher_effect_available=True,
    )
    assert teacher_nonvision_gate_effect_preserve_loss.item() > 0.0
    assert clean_attention_weight_anchor_loss.item() > 0.0
    assert clean_reliability_anchor_loss.item() > 0.0
    assert clean_bias_score_anchor_loss.item() > 0.0
    assert clean_prebias_score_anchor_loss.item() > 0.0


def test_stage2_teacher_preserve_signal_absmax_uses_non_target_indices() -> None:
    signal = compute_stage2_teacher_preserve_signal_absmax(
        clean_teacher_gate_effects=torch.tensor([[[0.0, 0.02, 0.03, 0.04]]], dtype=torch.float32),
        corrupt_teacher_gate_effects=torch.tensor([[[0.5, 0.01, 0.01, 0.01]]], dtype=torch.float32),
        target_modality="vision",
        disable_teacher_target_distill=True,
        disable_vision_final_gate_teacher_preserve=True,
    )
    assert signal.item() == pytest.approx(0.04)


def test_stage2_teacher_preserve_signal_absmax_ignores_padded_timesteps() -> None:
    signal = compute_stage2_teacher_preserve_signal_absmax(
        clean_teacher_gate_effects=torch.tensor(
            [[[0.0, 0.02, 0.03, 0.04], [0.0, 9.0, 9.0, 9.0]]], dtype=torch.float32
        ),
        corrupt_teacher_gate_effects=torch.tensor(
            [[[0.5, 0.01, 0.01, 0.01], [0.0, 8.0, 8.0, 8.0]]], dtype=torch.float32
        ),
        target_modality="vision",
        disable_teacher_target_distill=True,
        disable_vision_final_gate_teacher_preserve=True,
        valid_mask=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )
    assert signal.item() == pytest.approx(0.04)


def test_stage2_teacher_actuation_regime_state_flags_mismatch_when_teacher_is_flat() -> None:
    (
        teacher_gate_effect_absmax,
        teacher_bias_score_absmax,
        teacher_actuation_regime_mismatch_flag,
        teacher_preserve_active_flag,
    ) = compute_stage2_teacher_actuation_regime_state(
        clean_current_gate_effects=torch.tensor([[[0.0, 0.04, 0.03, 0.02]]], dtype=torch.float32),
        corrupt_current_gate_effects=torch.tensor([[[0.0, 0.01, 0.01, 0.01]]], dtype=torch.float32),
        clean_teacher_gate_effects=torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupt_teacher_gate_effects=torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        clean_current_bias_scores=torch.tensor([[[0.2, 0.2, 0.2, 0.2]]], dtype=torch.float32),
        corrupt_current_bias_scores=torch.tensor([[[0.1, 0.1, 0.1, 0.1]]], dtype=torch.float32),
        clean_teacher_bias_scores=torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        corrupt_teacher_bias_scores=torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        target_modality="vision",
        disable_teacher_target_distill=True,
        disable_vision_final_gate_teacher_preserve=True,
        valid_mask=torch.ones(1, 1, dtype=torch.float32),
    )
    assert teacher_gate_effect_absmax.item() == pytest.approx(0.0)
    assert teacher_bias_score_absmax.item() == pytest.approx(0.0)
    assert teacher_actuation_regime_mismatch_flag.item() == pytest.approx(1.0)
    assert teacher_preserve_active_flag.item() == pytest.approx(0.0)


def test_stage2_candidate_effective_config_supports_anchor_and_distilled_prebias_candidates() -> None:
    config = {
        "losses": {
            "teacher_nonvision_gate_effect_preserve_loss_coef": 0.0,
            "clean_score_anchor_loss_coef": 0.05,
            "clean_action_anchor_loss_coef": 0.05,
        },
        "candidates": {
            "vision_prebias_output_anchor_scale005": {
                "checkpoint_root": "/tmp/a",
                "overrides": {
                    "fusion_gate": {"reliability_bias_residual_scale": 0.05},
                    "losses": {"teacher_nonvision_gate_effect_preserve_loss_coef": 0.0},
                },
            },
            "vision_prebias_output_distilled_scale010": {
                "checkpoint_root": "/tmp/b",
                "overrides": {
                    "fusion_gate": {"reliability_bias_residual_scale": 0.10},
                    "losses": {"teacher_nonvision_gate_effect_preserve_loss_coef": 0.10},
                },
            },
        },
    }
    anchor = build_stage2_candidate_effective_config(config, "vision_prebias_output_anchor_scale005")
    distilled = build_stage2_candidate_effective_config(config, "vision_prebias_output_distilled_scale010")
    assert anchor["losses"]["teacher_nonvision_gate_effect_preserve_loss_coef"] == pytest.approx(0.0)
    assert distilled["losses"]["teacher_nonvision_gate_effect_preserve_loss_coef"] == pytest.approx(0.10)
    assert anchor["fusion_gate"]["reliability_bias_residual_scale"] == pytest.approx(0.05)
    assert distilled["fusion_gate"]["reliability_bias_residual_scale"] == pytest.approx(0.10)


def test_apply_attention_finetune_freeze_supports_threshold_shift_mode() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.model = _build_stage1_model()
    trainer._apply_attention_finetune_freeze(
        SimpleNamespace(
            attention_finetune_mode="source_gate_threshold_shift_micro_unfreeze",
            train_gate_input_adapter=True,
        )
    )
    trainable = {name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    assert any(
        name.startswith("fusion_module.reliability_bias_gate_threshold_shift_head.")
        for name in trainable
    )


def test_apply_attention_finetune_freeze_supports_threshold_shift_bias_mode() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.model = _build_stage1_model()
    trainer._apply_attention_finetune_freeze(
        SimpleNamespace(
            attention_finetune_mode="source_gate_threshold_shift_bias_micro_unfreeze",
            train_gate_input_adapter=True,
        )
    )
    trainable = {name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("fusion_module.reliability_bias_head.") for name in trainable)
    assert any(
        name.startswith("fusion_module.reliability_bias_gate_threshold_shift_head.")
        for name in trainable
    )
    assert not any(name.startswith("relative_gap_head.") for name in trainable)


def test_apply_attention_finetune_freeze_supports_bias_adapter_output_scale_mode() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.model = _build_stage1_model()
    configure_trainable_gate_actuation_parameters(
        trainer.model,
        {
            "gate_residual_alpha_mode": "learnable",
            "gate_residual_alpha_init": [1.0, 1.0, 1.0, 1.0],
            "gate_logit_offset_mode": "learnable",
            "gate_logit_offset_init": [0.0, 0.0, 0.0, 0.0],
            "adapter_gate_alpha_mode": "learnable",
            "adapter_gate_alpha_init": [1.0, 1.0, 1.0, 1.0],
            "gate_output_scale_mode": "learnable",
            "gate_output_scale_init": [1.0, 1.0, 1.0, 1.0],
            "gate_source_logit_bias_mode": "learnable",
            "gate_source_logit_bias_init": [0.0, 0.0, 0.0, 0.0],
        },
    )
    trainer._apply_attention_finetune_freeze(
        SimpleNamespace(
            attention_finetune_mode="source_gate_bias_adapter_output_scale_micro_unfreeze",
            train_gate_input_adapter=True,
        )
    )
    trainable = {name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("fusion_module.reliability_bias_head.") for name in trainable)
    assert any(name.startswith("fusion_module.score_adapter.") for name in trainable)
    assert any(name.startswith("fusion_module.gate_output_scale") for name in trainable)
    assert any(name.startswith("fusion_module.gate_source_logit_bias") for name in trainable)
    assert not any(name.startswith("fusion_module.relative_gap_head.") for name in trainable)


def test_materialize_gate_actuation_parameters_uses_fusion_gate_overrides() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.model = _build_stage1_model()
    trainer.raw_config = {
        "losses": {},
        "fusion_gate": {
            "score_adapter_residual_scale": 0.05,
            "gate_residual_alpha_mode": "learnable",
            "gate_residual_alpha_init": [1.0, 1.0, 1.0, 1.0],
            "gate_logit_offset_mode": "learnable",
            "gate_logit_offset_init": [0.0, 0.0, 0.0, 0.0],
            "adapter_gate_alpha_mode": "learnable",
            "adapter_gate_alpha_init": [1.0, 1.0, 1.0, 1.0],
            "gate_output_scale_mode": "learnable",
            "gate_output_scale_init": [1.0, 1.0, 1.0, 1.0],
            "gate_source_logit_bias_mode": "learnable",
            "gate_source_logit_bias_init": [0.0, 0.0, 0.0, 0.0],
        },
    }
    trainer._materialize_gate_actuation_parameters(trainer.model, include_fusion_gate=True)
    fusion = trainer.model.fusion_module
    assert fusion.score_adapter_residual_scale == pytest.approx(0.05)
    assert isinstance(fusion.gate_output_scale, torch.nn.Parameter)
    assert isinstance(fusion.gate_source_logit_bias, torch.nn.Parameter)


def test_v3_7f2_7_metrics_and_dashboard() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "normal_eval",
                "env_mode": "normal",
                "attention_bias_gate_mean_vision": 0.01,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.4,
                "attention_score_adapter_delta_norm_mean_vision": 0.02,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "clean_vs_mild_gate_feature_auc": 0.99,
                "clean_vs_mild_gate_feature_delta_norm": 1.0,
                "source_gate_feature_onoff_auc_delta": 0.1,
                "teacher_delta_feature_onoff_auc_delta": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "eval_success_rate": 0.9,
                "eval_return": 1.0,
                "eval_avg_progress_to_goal": 0.8,
                "same_state_action_diff_clean_vs_corrupt": 0.1,
                "reference_same_state_action_diff_clean_vs_corrupt": 0.1,
                "targetlogit_shapley_mean_vision": 0.2,
                "targetlogit_shapley_mean_ego": 0.0,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.2,
                "attention_relative_gap_score_mean_vision_distance": 0.3,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.2,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.1,
                "logit_mean_ego_clean": 0.0,
                "reference_logit_mean_vision_clean": 0.1,
                "reference_logit_mean_ego_clean": 0.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "source_sign_score_anchor_error": 0.0,
                "source_clean_gate_preserve_error": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_gate_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "mild_vision",
                "env_mode": "mild",
                "attention_bias_gate_mean_vision": 0.02,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_source_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.02,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.5,
                "attention_score_adapter_delta_norm_mean_vision": 0.03,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "clean_vs_mild_gate_feature_auc": 0.99,
                "clean_vs_mild_gate_feature_delta_norm": 1.0,
                "source_gate_feature_onoff_auc_delta": 0.1,
                "teacher_delta_feature_onoff_auc_delta": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "eval_success_rate": 0.8,
                "eval_return": 0.8,
                "eval_avg_progress_to_goal": 0.7,
                "same_state_action_diff_clean_vs_corrupt": 0.1,
                "reference_same_state_action_diff_clean_vs_corrupt": 0.1,
                "targetlogit_shapley_mean_vision": 0.1,
                "targetlogit_shapley_mean_ego": 0.0,
                "attention_mean_vision": 0.3,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.25,
                "attention_mean_route": 0.25,
                "attention_score_mean_vision": 0.8,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.2,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "reliability_mean_vision": 0.8,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_mild": 0.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "source_sign_score_anchor_error": 0.0,
                "source_clean_gate_preserve_error": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_gate_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "env_mode": "normal",
                "attention_bias_gate_mean_vision": 0.01,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.4,
                "attention_score_adapter_delta_norm_mean_vision": 0.02,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "clean_vs_mild_gate_feature_auc": 0.99,
                "clean_vs_mild_gate_feature_delta_norm": 1.0,
                "source_gate_feature_onoff_auc_delta": 0.1,
                "teacher_delta_feature_onoff_auc_delta": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "eval_success_rate": 0.9,
                "eval_return": 1.0,
                "eval_avg_progress_to_goal": 0.8,
                "same_state_action_diff_clean_vs_corrupt": 0.1,
                "reference_same_state_action_diff_clean_vs_corrupt": 0.1,
                "targetlogit_shapley_mean_vision": 0.2,
                "targetlogit_shapley_mean_ego": 0.0,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.2,
                "attention_relative_gap_score_mean_vision_distance": 0.3,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.2,
                "attention_relative_gap_norm_mean": 0.4,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.1,
                "logit_mean_ego_clean": 0.0,
                "reference_logit_mean_vision_clean": 0.1,
                "reference_logit_mean_ego_clean": 0.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "source_sign_score_anchor_error": 0.0,
                "source_clean_gate_preserve_error": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_gate_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "env_mode": "mild",
                "attention_bias_gate_mean_vision": 0.013,
                "attention_bias_gate_mean_distance": 0.01,
                "attention_bias_gate_mean_ego": 0.01,
                "attention_bias_gate_mean_route": 0.01,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.03,
                "attention_bias_gate_adapter_logit_mean_vision": 0.06,
                "attention_bias_gate_adapter_logit_mean_distance": 0.001,
                "attention_bias_gate_adapter_logit_mean_ego": 0.001,
                "attention_bias_gate_adapter_logit_mean_route": 0.001,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.8,
                "attention_score_adapter_delta_norm_mean_vision": 0.10,
                "attention_score_adapter_delta_norm_mean_distance": 0.001,
                "attention_score_adapter_delta_norm_mean_ego": 0.001,
                "attention_score_adapter_delta_norm_mean_route": 0.001,
                "clean_vs_mild_gate_feature_auc": 0.99,
                "clean_vs_mild_gate_feature_delta_norm": 1.0,
                "source_gate_feature_onoff_auc_delta": 0.1,
                "teacher_delta_feature_onoff_auc_delta": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "eval_success_rate": 0.8,
                "eval_return": 0.8,
                "eval_avg_progress_to_goal": 0.7,
                "same_state_action_diff_clean_vs_corrupt": 0.1,
                "reference_same_state_action_diff_clean_vs_corrupt": 0.1,
                "targetlogit_shapley_mean_vision": 0.1,
                "targetlogit_shapley_mean_ego": 0.0,
                "attention_mean_vision": 0.3,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.25,
                "attention_mean_route": 0.25,
                "attention_score_mean_vision": 0.8,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.2,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "reliability_mean_vision": 0.8,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_mild": 0.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "source_sign_score_anchor_error": 0.0,
                "source_clean_gate_preserve_error": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_gate_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
            },
        ]
    )
    for column in [
        "attention_bias_mean_vision",
        "attention_bias_mean_distance",
        "attention_bias_mean_ego",
        "attention_bias_mean_route",
        "attention_bias_norm_mean",
        "attention_bias_gate_norm_mean",
        "attention_bias_abs_mean",
        "logit_mean_vision_clean",
        "logit_mean_vision_mild",
        "logit_mean_ego_clean",
        "logit_mean_ego_mild",
    ]:
        if column not in df.columns:
            df[column] = 0.0
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        **df.iloc[0].to_dict(),
                        "candidate_name": "reference_v3_6_medium_locked",
                        "corruption_type": "normal_eval",
                    },
                    {
                        **df.iloc[0].to_dict(),
                        "candidate_name": "reference_v3_7c_medium_locked",
                        "corruption_type": "mild_vision",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    thresholds = {
        "cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05},
        "reference_v3_6_medium_locked": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05},
        "reference_v3_7c_medium_locked": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05},
    }
    gate_df = compute_v3_7f2_7_adapter_readout_metrics(
        df,
        candidate_thresholds=thresholds,
        clean_reference_candidate="reference_v3_7c_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    row = gate_df[gate_df["candidate_name"] == "cand"].iloc[0]
    assert row["vision_adapter_gate_logit_mild_minus_clean"] == pytest.approx(0.05)
    assert row["vision_adapter_gate_readout_compression_ratio"] > 0.30
    dashboard = build_v3_7f2_7_pass_distance_dashboard(
        gate_df,
        baseline_candidate="reference_v3_7c_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_adapter_readout_logit" in dash_row.index


def test_calibration_only_tagging_and_report(tmp_path: Path) -> None:
    losses_cfg = {
        "actor_loss_coef": 0.0,
        "value_loss_coef": 0.0,
        "entropy_coef": 0.0,
        "reliability_loss_coef": 0.0,
        "consistency_loss_coef": 0.0,
        "shapley_loss_coef": 0.0,
        "risk_loss_coef": 0.0,
    }
    assert is_calibration_only(losses_cfg)
    assert infer_experiment_mode(losses_cfg) == "calibration_only"

    score_df = pd.DataFrame(
        [{"candidate_name": "a", "pass_fail": "WARN", "calibration_only_flag": 1.0}]
    )
    tiny = pd.DataFrame([{"candidate_name": "a"}])
    out = tmp_path / "report.md"
    generate_v3_7_report(
        score_df=score_df,
        degradation_df=tiny,
        tiebreak_df=tiny,
        consistency_df=tiny,
        targetlogit_df=tiny,
        bias_df=tiny,
        output_path=out,
        recommended="a",
        secondary="a",
        title="test",
    )


def test_f28_metrics_and_dashboard_include_output_scale_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.3,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "attention_bias_gate_mean_vision": 0.05,
                "attention_bias_gate_mean_distance": 0.02,
                "attention_bias_gate_mean_ego": 0.02,
                "attention_bias_gate_mean_route": 0.02,
                "attention_source_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_logit_mean_vision": 0.01,
                "attention_bias_gate_residual_logit_mean_distance": 0.0,
                "attention_bias_gate_residual_logit_mean_ego": 0.0,
                "attention_bias_gate_residual_logit_mean_route": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_delta_logit_mean_vision": 0.02,
                "attention_bias_gate_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_delta_logit_mean_route": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.04,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_gate_output_scale_mean_vision": 4.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.10,
                "attention_score_adapter_delta_norm_mean_distance": 0.01,
                "attention_score_adapter_delta_norm_mean_ego": 0.01,
                "attention_score_adapter_delta_norm_mean_route": 0.01,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.5,
                "attention_bias_gate_adapter_hidden_norm_mean_distance": 0.1,
                "attention_bias_gate_adapter_hidden_norm_mean_ego": 0.1,
                "attention_bias_gate_adapter_hidden_norm_mean_route": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "corruption_kind": "clean",
            },
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.3,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "attention_bias_gate_mean_vision": 0.05,
                "attention_bias_gate_mean_distance": 0.02,
                "attention_bias_gate_mean_ego": 0.02,
                "attention_bias_gate_mean_route": 0.02,
                "attention_source_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_logit_mean_vision": 0.02,
                "attention_bias_gate_residual_logit_mean_distance": 0.0,
                "attention_bias_gate_residual_logit_mean_ego": 0.0,
                "attention_bias_gate_residual_logit_mean_route": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.04,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_delta_logit_mean_vision": 0.03,
                "attention_bias_gate_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_delta_logit_mean_route": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.05,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_gate_output_scale_mean_vision": 4.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.05,
                "attention_score_adapter_delta_norm_mean_distance": 0.01,
                "attention_score_adapter_delta_norm_mean_ego": 0.01,
                "attention_score_adapter_delta_norm_mean_route": 0.01,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.6,
                "attention_bias_gate_adapter_hidden_norm_mean_distance": 0.1,
                "attention_bias_gate_adapter_hidden_norm_mean_ego": 0.1,
                "attention_bias_gate_adapter_hidden_norm_mean_route": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "corruption_kind": "clean",
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "attention_score_mean_vision": 0.7,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.3,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "attention_bias_gate_mean_vision": 0.053,
                "attention_bias_gate_mean_distance": 0.02,
                "attention_bias_gate_mean_ego": 0.02,
                "attention_bias_gate_mean_route": 0.02,
                "attention_source_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_distance": 0.02,
                "attention_source_bias_gate_mean_ego": 0.02,
                "attention_source_bias_gate_mean_route": 0.02,
                "attention_bias_gate_residual_logit_mean_vision": 0.03,
                "attention_bias_gate_residual_logit_mean_distance": 0.0,
                "attention_bias_gate_residual_logit_mean_ego": 0.0,
                "attention_bias_gate_residual_logit_mean_route": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.08,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_delta_logit_mean_vision": 0.06,
                "attention_bias_gate_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_delta_logit_mean_route": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.18,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_gate_output_scale_mean_vision": 6.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.15,
                "attention_score_adapter_delta_norm_mean_distance": 0.01,
                "attention_score_adapter_delta_norm_mean_ego": 0.01,
                "attention_score_adapter_delta_norm_mean_route": 0.01,
                "attention_bias_gate_adapter_hidden_norm_mean_vision": 0.9,
                "attention_bias_gate_adapter_hidden_norm_mean_distance": 0.1,
                "attention_bias_gate_adapter_hidden_norm_mean_ego": 0.1,
                "attention_bias_gate_adapter_hidden_norm_mean_route": 0.1,
                "gate_reconstruction_error_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "adapter_gate_alpha_mean": 1.0,
                "gate_logit_offset_mean": 0.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "corruption_kind": "single",
            },
        ]
    )
    for column in [
        "attention_bias_mean_vision",
        "attention_bias_mean_distance",
        "attention_bias_mean_ego",
        "attention_bias_mean_route",
        "attention_bias_norm_mean",
        "attention_bias_gate_norm_mean",
        "attention_bias_abs_mean",
        "logit_mean_vision_clean",
        "logit_mean_vision_mild",
        "logit_mean_ego_clean",
        "logit_mean_ego_mild",
    ]:
        if column not in df.columns:
            df[column] = 0.0
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        **df.iloc[0].to_dict(),
                        "candidate_name": "reference_v3_6_medium_locked",
                        "corruption_type": "normal_eval",
                    },
                    {
                        **df.iloc[0].to_dict(),
                        "candidate_name": "reference_v3_7c_medium_locked",
                        "corruption_type": "mild_vision",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    thresholds = {
        "cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05},
        "reference_v3_6_medium_locked": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05},
        "reference_v3_7c_medium_locked": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05},
    }
    gate_df = compute_v3_7f2_8_output_scale_metrics(
        df,
        candidate_thresholds=thresholds,
        clean_reference_candidate="reference_v3_7c_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    row = gate_df[gate_df["candidate_name"] == "cand"].iloc[0]
    assert row["vision_scaled_delta_logit_mild_minus_clean"] == pytest.approx(0.13)
    assert row["vision_gate_output_scale"] == pytest.approx(6.0)
    dashboard = build_v3_7f2_8_pass_distance_dashboard(
        gate_df,
        baseline_candidate="reference_v3_7c_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_scaled_delta_logit" in dash_row.index


def test_smoke_output_root_is_isolated(tmp_path: Path) -> None:
    smoke_root = resolve_experiment_output_root(tmp_path / "pilot", smoke=True)
    normal_root = resolve_experiment_output_root(tmp_path / "pilot", smoke=False)
    assert smoke_root.name.endswith("_smoke")
    assert normal_root.name == "pilot"


def test_validate_run_completeness_rejects_missing_seed_or_candidate() -> None:
    df = pd.DataFrame(
        [
            {"candidate_name": "a", "seed": 0, "corruption_type": "normal_eval"},
            {"candidate_name": "a", "seed": 0, "corruption_type": "mild_vision"},
        ]
    )
    try:
        validate_run_completeness(
            df,
            expected_candidates=["a", "b"],
            expected_seeds=[0, 1],
            expected_probe_names=["normal_eval", "mild_vision"],
        )
    except RuntimeError as exc:
        assert "Incomplete" in str(exc)
    else:
        raise AssertionError("Expected completeness validation to fail")


def test_teacher_reproduction_guard_raises_on_mismatch() -> None:
    try:
        validate_teacher_reproduction_metrics(
            observed={"vision_gate_clean_mean": 0.01},
            reference={"vision_gate_clean_mean": 0.03},
            keys=["vision_gate_clean_mean"],
            config={"teacher_reproduction_tolerance_default": 1.0e-4},
        )
    except RuntimeError as exc:
        assert "Teacher reproduction check failed" in str(exc)
    else:
        raise AssertionError("Expected teacher reproduction mismatch to raise")


def test_source_residual_logit_v2_moves_gate() -> None:
    model = _build_stage1_model()
    fusion = model.fusion_module
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_logit_v2"
    fusion.reliability_bias_residual_scale = 0.22
    fusion.gate_residual_alpha = 1.0
    fusion.reliability_bias_gate_residual_head = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with torch.no_grad():
        fusion.reliability_bias_gate_residual_head[0].weight.fill_(1.0)
        fusion.reliability_bias_gate_residual_head[0].bias.zero_()
    stacked = torch.randn(1, 4, 64)
    bias_inputs = torch.randn(1, 4, 66)
    gate_inputs = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.2, 0.2], [0.2, 0.2]]], dtype=torch.float32)
    source_gate_scores = torch.full((1, 4), 0.05, dtype=torch.float32)
    _, _, _, _, _, _, gate_scores, source_gate_outputs, gate_residuals, _gate_residual_logits, _gate_adapter_logits, _, _, *_ = fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
        source_gate_scores=source_gate_scores,
    )
    expected = torch.sigmoid(torch.logit(source_gate_scores.clamp(1.0e-4, 1.0 - 1.0e-4)) + gate_residuals)
    assert torch.allclose(source_gate_outputs, source_gate_scores)
    assert torch.allclose(gate_scores, expected, atol=1.0e-6)
    assert gate_scores[0, 2] > gate_scores[0, 0]


def test_paired_selective_gate_losses() -> None:
    losses = compute_paired_selective_gate_losses(
        torch.tensor(0.04),
        torch.tensor(0.03),
        clean_gate_max=0.02,
        mild_gate_min=0.08,
        gate_gap_margin=0.05,
        residual_clean=torch.tensor(0.0),
        residual_mild=torch.tensor(0.005),
        residual_margin=0.01,
    )
    assert torch.isclose(losses["clean_gate_cap"], torch.tensor(0.02))
    assert torch.isclose(losses["mild_gate_floor"], torch.tensor(0.05))
    assert torch.isclose(losses["gate_gap_margin"], torch.tensor(0.06))
    assert torch.isclose(losses["residual_gap_margin"], torch.tensor(0.005))


def test_disable_vision_teacher_preserve_indices() -> None:
    assert select_gate_teacher_preserve_indices(True) == [1, 2, 3]
    assert select_gate_teacher_preserve_indices(False) == [0, 1, 2, 3]


def test_merge_teacher_loss_overrides_prefers_snapshot() -> None:
    merged, diffs = merge_teacher_loss_overrides(
        {"reliability_bias_gate_temperature": 8.0, "gate_input_mode": "legacy"},
        {"reliability_bias_gate_temperature": 12.0, "gate_input_mode": "source_residual_bias_aware_v1"},
        snapshot_precedence=True,
    )
    assert merged["reliability_bias_gate_temperature"] == 8.0
    assert merged["gate_input_mode"] == "legacy"
    assert diffs["reliability_bias_gate_temperature"] == (12.0, 8.0)


def test_compute_teacher_reproduction_metrics_from_rows() -> None:
    rows = [
        {
            "corruption_type": "normal_eval",
            "attention_mean_vision": 0.4,
            "targetlogit_shapley_mean_vision": 0.2,
            "attention_score_mean_vision": 1.0,
            "attention_score_mean_route": 0.2,
            "attention_score_mean_distance": 0.3,
            "attention_bias_gate_mean_vision": 0.05,
        },
        {
            "corruption_type": "mild_vision",
            "attention_mean_vision": 0.3,
            "targetlogit_shapley_mean_vision": 0.1,
            "attention_score_mean_vision": 0.8,
            "attention_score_mean_route": 0.1,
            "attention_score_mean_distance": 0.2,
            "attention_bias_gate_mean_vision": 0.04,
        },
    ]
    metrics = compute_teacher_reproduction_metrics_from_rows(rows)
    assert metrics["vision_attention_targetlogit_sign_match_flag"] == 1.0
    assert metrics["vision_attention_relative_score_clean_minus_mild"] == pytest.approx(0.2)
    assert metrics["vision_gate_clean_mean"] == pytest.approx(0.05)


def test_probe_logging_fields_preserve_mixed_strengths() -> None:
    fields = _probe_logging_fields(
        "mild_vision_route",
        {"kind": "mixed", "strengths": [0.12, 0.05], "modalities": ["vision", "route"]},
    )
    assert fields["corruption_kind"] == "mixed"
    assert fields["corruption_strength"] == pytest.approx(0.12)
    assert fields["corruption_strength_scalar"] == pytest.approx(0.12)
    assert fields["corruption_strengths_json"] == "[0.12, 0.05]"


def test_sample_train_corruption_keeps_paired_vision_inside_mild_budget() -> None:
    trainer = Stage2TransitionTrainer.__new__(Stage2TransitionTrainer)
    trainer.raw_config = {
        "train": {"total_steps": 100, "rollout_steps": 10},
        "losses": {
            "paired_selective_gate_supervision": True,
            "paired_clean_mild_vision_prob": 0.5,
        },
        "train_corruption": {
            "clean_probability": 0.2,
            "mild_single_probability": 0.8,
            "single_modality": [
                {"name": "vision", "strength": 0.1, "weight": 1.0},
                {"name": "route", "strength": 0.1, "weight": 1.0},
            ],
            "mixed": [],
        },
    }
    trainer._current_update_idx = 0
    obs = {
        "vision": np.zeros(8, dtype=np.float32),
        "distance": np.zeros(9, dtype=np.float32),
        "ego": np.zeros(5, dtype=np.float32),
        "route": np.zeros(5, dtype=np.float32),
    }
    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    total = 4000
    for _ in range(total):
        _policy_obs, _severity, _mask, name = trainer._sample_train_corruption(obs, rng)
        counts[name] = counts.get(name, 0) + 1
    clean_ratio = counts.get("clean", 0) / total
    mild_ratio = 1.0 - clean_ratio
    paired_ratio = counts.get("mild_vision_paired", 0) / total
    assert clean_ratio == pytest.approx(0.2, abs=0.03)
    assert mild_ratio == pytest.approx(0.8, abs=0.03)
    assert 0.35 <= paired_ratio <= 0.45


def test_f2_metric_prefers_residual_logit_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "env_mode": "normal",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.15,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.3,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.2,
                "attention_relative_gap_norm_mean": 0.5,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.9,
                    "reliability_mean_distance": 0.7,
                    "reliability_mean_ego": 0.7,
                    "reliability_mean_route": 0.7,
                        "logit_mean_vision_clean": 0.4,
                        "logit_mean_vision_mild": 0.0,
                        "attention_bias_mean_vision": 0.1,
                    "attention_bias_mean_distance": 0.0,
                    "attention_bias_mean_ego": 0.0,
                    "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.1,
                "attention_bias_abs_mean": 0.1,
                "attention_bias_gate_mean_vision": 0.01,
                    "attention_source_bias_gate_mean_vision": 0.04,
                    "attention_bias_gate_residual_mean_vision": -0.9,
                    "attention_bias_gate_residual_logit_mean_vision": -0.2,
                    "attention_score_adapter_delta_norm_mean_vision": 0.0,
                    "attention_score_adapter_delta_norm_mean_distance": 0.0,
                    "attention_score_adapter_delta_norm_mean_ego": 0.0,
                    "attention_score_adapter_delta_norm_mean_route": 0.0,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "source_sign_score_anchor_error": 0.003,
                "source_clean_gate_preserve_error": 0.004,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "env_mode": "mild",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.1,
                "attention_score_mean_ego": 0.1,
                "attention_score_mean_route": 0.0,
                "attention_relative_gap_score_mean_vision_distance": 0.4,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.25,
                "attention_relative_gap_norm_mean": 0.6,
                "attention_mean_vision": 0.5,
                "attention_mean_distance": 0.18,
                "attention_mean_ego": 0.17,
                "attention_mean_route": 0.15,
                "targetlogit_shapley_mean_vision": 0.1,
                "reliability_mean_vision": 0.8,
                    "reliability_mean_distance": 0.7,
                    "reliability_mean_ego": 0.7,
                    "reliability_mean_route": 0.7,
                        "logit_mean_vision_clean": 0.4,
                        "logit_mean_vision_mild": 0.3,
                        "attention_bias_mean_vision": 0.2,
                    "attention_bias_mean_distance": 0.0,
                    "attention_bias_mean_ego": 0.0,
                    "attention_bias_mean_route": 0.0,
                "attention_bias_norm_mean": 0.2,
                "attention_bias_abs_mean": 0.2,
                "attention_bias_gate_mean_vision": 0.09,
                    "attention_source_bias_gate_mean_vision": 0.03,
                    "attention_bias_gate_residual_mean_vision": 0.9,
                    "attention_bias_gate_residual_logit_mean_vision": 0.7,
                    "attention_score_adapter_delta_norm_mean_vision": 0.0,
                    "attention_score_adapter_delta_norm_mean_distance": 0.0,
                    "attention_score_adapter_delta_norm_mean_ego": 0.0,
                    "attention_score_adapter_delta_norm_mean_route": 0.0,
                "clean_score_anchor_error": 0.001,
                "clean_action_anchor_error": 0.002,
                "source_sign_score_anchor_error": 0.003,
                "source_clean_gate_preserve_error": 0.004,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
                "normal_to_mild_return_drop": 0.05,
                "saturation_warning_flag": 0.0,
                "route_distance_side_effect_flag": 0.0,
            },
        ]
    )
    metrics = compute_v3_7f1_source_gate_distilled_residual_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics.iloc[0]
    assert row["vision_residual_logit_clean_mean"] == pytest.approx(-0.2)
    assert row["vision_residual_logit_mild_mean"] == pytest.approx(0.7)


def test_configure_trainable_gate_residual_alpha_registers_parameter() -> None:
    model = _build_stage1_model()
    configure_trainable_gate_residual_alpha(
        model,
        {
            "residual_alpha_mode": "learnable",
            "residual_alpha_init": 2.5,
            "residual_alpha_max": 6.0,
            "residual_alpha_l2_coef": 0.001,
        },
    )
    model.freeze_attention_source_gate_residual_only(train_gate_input_adapter=False)
    assert isinstance(model.fusion_module.gate_residual_alpha, torch.nn.Parameter)
    assert model.fusion_module.gate_residual_alpha.requires_grad
    assert float(model.fusion_module.gate_residual_alpha.detach().cpu().item()) == pytest.approx(2.5)


def test_configure_trainable_gate_actuation_parameters_registers_vector_params() -> None:
    model = _build_stage1_model()
    configure_trainable_gate_actuation_parameters(
        model,
        {
            "residual_alpha_mode": "learnable",
            "residual_alpha_init": [4.0, 1.0, 1.0, 1.0],
            "gate_logit_offset_mode": "learnable",
            "gate_logit_offset_init": [0.1, 0.0, 0.0, 0.0],
        },
    )
    model.freeze_attention_source_gate_residual_only(train_gate_input_adapter=True)
    assert isinstance(model.fusion_module.gate_residual_alpha, torch.nn.Parameter)
    assert isinstance(model.fusion_module.gate_logit_offset, torch.nn.Parameter)
    assert model.fusion_module.gate_residual_alpha.requires_grad
    assert model.fusion_module.gate_logit_offset.requires_grad
    assert model.fusion_module.gate_residual_alpha.shape == (4,)
    assert model.fusion_module.gate_logit_offset.shape == (4,)
    assert float(model.fusion_module.gate_residual_alpha.detach().cpu()[0]) == pytest.approx(4.0)
    assert float(model.fusion_module.gate_logit_offset.detach().cpu()[0]) == pytest.approx(0.1)


def test_residual_logit_target_loss_zero_at_target() -> None:
    alpha = torch.tensor(2.0)
    clean_gate_max = 0.02
    mild_gate_min = 0.08
    source_clean_gate = torch.tensor(0.05)
    source_mild_gate = torch.tensor(0.05)
    target_clean = (torch.logit(torch.tensor(clean_gate_max)) - torch.logit(source_clean_gate)) / alpha
    target_mild = (torch.logit(torch.tensor(mild_gate_min)) - torch.logit(source_mild_gate)) / alpha
    loss = compute_residual_logit_target_loss(
        clean_residual_logit=target_clean,
        mild_residual_logit=target_mild,
        source_clean_gate=source_clean_gate,
        source_mild_gate=source_mild_gate,
        residual_alpha=alpha,
        clean_gate_max=clean_gate_max,
        mild_gate_min=mild_gate_min,
        clip=8.0,
        loss_type="huber",
    )
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


def test_final_gate_target_loss_zero_at_target() -> None:
    loss = compute_final_gate_target_loss(
        clean_gate=torch.tensor(0.02),
        mild_gate=torch.tensor(0.08),
        clean_gate_max=0.02,
        mild_gate_min=0.08,
        loss_type="huber",
    )
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


def test_gate_feature_separability_audit_detects_signal() -> None:
    clean = np.zeros((8, 4), dtype=np.float64)
    mild = np.zeros((8, 4), dtype=np.float64)
    clean[:, 0] = -1.0
    mild[:, 0] = 1.0
    clean[:, 1] = -0.5
    mild[:, 1] = 0.5
    audit = compute_gate_feature_separability_audit(
        clean,
        mild,
        {
            "source_gate_scores": (0, 1),
            "clean_teacher_delta": (1, 2),
            "sign_teacher_delta": (2, 3),
        },
    )
    assert audit["clean_vs_mild_gate_feature_auc"] > 0.95
    assert audit["clean_vs_mild_gate_feature_delta_norm"] > 0.0
    assert audit["source_gate_feature_onoff_auc_delta"] >= 0.0
    assert audit["teacher_delta_feature_onoff_auc_delta"] >= 0.0


def test_pass_distance_dashboard_builds_expected_columns() -> None:
    score_df = pd.DataFrame(
        [
            {
                "candidate_name": "reference_v3_6_medium_locked",
                "vision_residual_logit_mild_minus_clean": 0.0,
                "vision_final_gate_mild_minus_clean": 0.0,
                "vision_gate_mild_above_floor_ratio": 0.0,
                "vision_gate_gap_margin_satisfied_ratio": 0.0,
                "vision_attention_relative_score_clean_minus_mild": -0.02,
                "normal_success_rate": 0.9,
                "mild_success_rate": 0.8,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "vision_residual_logit_mild_minus_clean": 0.0,
                "vision_final_gate_mild_minus_clean": 0.0,
                "vision_gate_mild_above_floor_ratio": 0.0,
                "vision_gate_gap_margin_satisfied_ratio": 0.0,
                "vision_attention_relative_score_clean_minus_mild": -0.01,
                "normal_success_rate": 0.88,
                "mild_success_rate": 0.79,
            },
            {
                "candidate_name": "cand",
                "vision_residual_logit_mild_minus_clean": 0.02,
                "vision_final_gate_mild_minus_clean": 0.03,
                "vision_gate_mild_above_floor_ratio": 0.25,
                "vision_gate_gap_margin_satisfied_ratio": 0.2,
                "vision_attention_relative_score_clean_minus_mild": -0.005,
                "normal_success_rate": 0.91,
                "mild_success_rate": 0.82,
            },
        ]
    )
    dashboard = build_v3_7f2_2_pass_distance_dashboard(
        score_df,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert row["distance_to_activation_residual_logit"] == pytest.approx(0.01)
    assert row["distance_to_promotion_final_gate"] == pytest.approx(0.01)


def test_normalize_tiebreak_weights_fills_missing_keys() -> None:
    weights = normalize_tiebreak_weights({"normal_success": 1.0})
    assert weights["normal_success"] == pytest.approx(1.0)
    assert "vision_sign_match" in weights
    assert weights["vision_sign_match"] == pytest.approx(0.0)
    assert weights["clean_score_anchor"] == pytest.approx(0.0)
    assert weights["clean_action_anchor"] == pytest.approx(0.0)


def test_freeze_attention_source_gate_relative_micro_unfreeze_scope() -> None:
    model = _build_stage1_model()
    fusion = model.fusion_module
    fusion.reliability_bias_gate_input_adapter = torch.nn.Linear(4, 4)
    configure_trainable_gate_actuation_parameters(
        model,
        {
            "residual_alpha_mode": "learnable",
            "residual_alpha_init": [4.0, 1.0, 1.0, 1.0],
            "gate_logit_offset_mode": "learnable",
            "gate_logit_offset_init": [0.0, 0.0, 0.0, 0.0],
        },
    )
    model.freeze_attention_source_gate_relative_micro_unfreeze(train_gate_input_adapter=True)
    assert all(parameter.requires_grad for parameter in fusion.relative_gap_head.parameters())
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_residual_head.parameters())
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_input_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.score_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.base_score_head.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.reliability_bias_head.parameters())


def test_freeze_attention_source_gate_score_adapter_micro_unfreeze_scope() -> None:
    model = _build_stage1_model()
    fusion = model.fusion_module
    fusion.reliability_bias_gate_input_adapter = torch.nn.Linear(4, 4)
    configure_trainable_gate_actuation_parameters(
        model,
        {
            "residual_alpha_mode": "learnable",
            "residual_alpha_init": [4.0, 1.0, 1.0, 1.0],
            "gate_logit_offset_mode": "learnable",
            "gate_logit_offset_init": [0.0, 0.0, 0.0, 0.0],
        },
    )
    model.freeze_attention_source_gate_score_adapter_micro_unfreeze(train_gate_input_adapter=True)
    assert all(parameter.requires_grad for parameter in fusion.score_adapter.parameters())
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_residual_head.parameters())
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_input_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.relative_gap_head.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.base_score_head.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.reliability_bias_head.parameters())


def test_relative_gap_loss_helpers() -> None:
    current = torch.tensor([0.2, 0.1, 0.0])
    teacher = torch.tensor([0.2, 0.1, 0.0])
    assert compute_relative_gap_anchor_loss(
        current_relative_gap=current,
        teacher_relative_gap=teacher,
        loss_type="mse",
    ) == pytest.approx(0.0)
    margin_loss = compute_relative_gap_margin_loss(
        clean_relative_gap=torch.tensor([0.3, 0.2, 0.1]),
        mild_relative_gap=torch.tensor([0.1, 0.1, 0.0]),
        margin=0.25,
    )
    assert margin_loss == pytest.approx(0.11666667)


def test_score_adapter_loss_helpers() -> None:
    current = torch.tensor([[0.2, 0.1]])
    teacher = torch.tensor([[0.2, 0.1]])
    assert compute_score_adapter_anchor_loss(
        current_delta=current,
        teacher_delta=teacher,
        loss_type="mse",
    ) == pytest.approx(0.0)
    margin_loss = compute_adapter_delta_margin_loss(
        clean_delta=torch.tensor([[0.1, 0.0]]),
        mild_delta=torch.tensor([[0.4, 0.0]]),
        margin=0.2,
    )
    assert margin_loss == pytest.approx(0.0)


def test_forward_sequence_emits_raw_score_adapter_delta() -> None:
    model = _build_stage1_model()
    obs_seq = {key: value.unsqueeze(1).repeat(1, 3, 1) for key, value in _sample_obs().items()}
    outputs = model.forward_sequence(
        obs_seq=obs_seq,
        episode_starts=torch.zeros(1, 3),
        initial_hidden=torch.zeros(1, model.gru_cell.hidden_size),
        actions=None,
        deterministic=False,
    )
    assert "attention_score_adapter_delta" in outputs
    assert "attention_score_features" in outputs
    assert outputs["attention_score_adapter_delta"].shape[:3] == (1, 3, 4)


def test_f25_dashboard_includes_adapter_columns() -> None:
    dashboard_input = pd.DataFrame(
        [
            {
                "candidate_name": "reference_v3_6_medium_locked",
                "vision_residual_logit_mild_minus_clean": 0.0,
                "vision_final_gate_mild_minus_clean": 0.0,
                "vision_gate_mild_above_floor_ratio": 0.0,
                "vision_gate_gap_margin_satisfied_ratio": 0.0,
                "vision_attention_relative_score_clean_minus_mild": -0.02,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
                "vision_adapter_delta_margin": 0.0,
                "score_adapter_delta_anchor_error_clean": 0.0,
                "score_adapter_delta_anchor_error_mild": 0.0,
                "score_adapter_delta_norm_nonvision_drift_mean": 0.0,
                "score_adapter_delta_mild_minus_clean": 0.0,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "vision_residual_logit_mild_minus_clean": 0.0,
                "vision_final_gate_mild_minus_clean": 0.0,
                "vision_gate_mild_above_floor_ratio": 0.0,
                "vision_gate_gap_margin_satisfied_ratio": 0.0,
                "vision_attention_relative_score_clean_minus_mild": -0.01,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
                "vision_adapter_delta_margin": 0.0,
                "score_adapter_delta_anchor_error_clean": 0.0,
                "score_adapter_delta_anchor_error_mild": 0.0,
                "score_adapter_delta_norm_nonvision_drift_mean": 0.0,
                "score_adapter_delta_mild_minus_clean": 0.0,
            },
            {
                "candidate_name": "cand",
                "vision_residual_logit_mild_minus_clean": 0.02,
                "vision_final_gate_mild_minus_clean": 0.03,
                "vision_gate_mild_above_floor_ratio": 0.25,
                "vision_gate_gap_margin_satisfied_ratio": 0.2,
                "vision_attention_relative_score_clean_minus_mild": -0.005,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
                "vision_adapter_delta_margin": 0.03,
                "score_adapter_delta_anchor_error_clean": 0.01,
                "score_adapter_delta_anchor_error_mild": 0.015,
                "score_adapter_delta_norm_nonvision_drift_mean": 0.005,
                "score_adapter_delta_mild_minus_clean": 0.03,
            },
        ]
    )
    dashboard = build_v3_7f2_5_pass_distance_dashboard(
        dashboard_input,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert dash_row["distance_to_adapter_delta_margin_positive"] == pytest.approx(0.03)
    assert dash_row["distance_to_adapter_anchor_clean"] == pytest.approx(0.01)


def test_freeze_attention_source_gate_adapter_coupled_micro_unfreeze_scope() -> None:
    model = _build_stage1_model()
    model.fusion_module.reliability_bias_gate_input_adapter = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model.freeze_attention_source_gate_adapter_coupled_micro_unfreeze(
        train_gate_input_adapter=True,
        train_score_adapter=False,
    )
    fusion = model.fusion_module
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_residual_head.parameters())
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_adapter_coupler.parameters())
    assert all(parameter.requires_grad for parameter in fusion.reliability_bias_gate_input_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.score_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.relative_gap_head.parameters())
    assert not any(parameter.requires_grad for parameter in fusion.base_score_head.parameters())


def test_adapter_gate_loss_helpers() -> None:
    current_clean = torch.tensor([[-1.0]])
    current_mild = torch.tensor([[0.5]])
    loss = compute_adapter_gate_target_loss(
        clean_adapter_gate_logit=current_clean,
        mild_adapter_gate_logit=current_mild,
        source_clean_gate=torch.full((1, 1), 0.05),
        source_mild_gate=torch.full((1, 1), 0.05),
        clean_gate_max=0.02,
        mild_gate_min=0.08,
        target_fraction=0.5,
        loss_type="huber",
    )
    assert float(loss) > 0.0
    margin_loss = compute_adapter_gate_margin_loss(
        clean_adapter_gate_logit=torch.tensor([[-0.3]]),
        mild_adapter_gate_logit=torch.tensor([[0.1]]),
        margin=0.6,
    )
    assert margin_loss == pytest.approx(0.2)


def test_compute_scores_emits_adapter_gate_logits() -> None:
    fusion = AttentionFusion(feature_dim=64, hidden_dim=32)
    fusion.use_relative_score_head = True
    fusion.use_reliability_bias_head = True
    fusion.use_reliability_bias_gate = True
    fusion.gate_mode = "source_residual_logit_v2"
    stacked = torch.randn(1, 4, 64)
    bias_inputs = torch.randn(1, 4, 66)
    gate_inputs = torch.randn(1, 4, 70)
    adapter_gate_inputs = torch.randn(1, 4, 12)
    source_gate_scores = torch.full((1, 4), 0.05)
    outputs = fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
        source_gate_scores=source_gate_scores,
        adapter_gate_inputs=adapter_gate_inputs,
    )
    assert len(outputs) == 24
    assert outputs[10].shape == (1, 4)
    assert outputs[11].shape == (1, 4)
    assert outputs[12].shape == (1, 4)
    assert outputs[-2].shape == (1, 4)
    assert outputs[-1].shape == (1, 4)
    assert outputs[-3].shape == (1, 4)
    assert outputs[-4].shape == (1, 4)


def test_gate_output_scale_tensor_clamps() -> None:
    fusion = AttentionFusion(feature_dim=64, hidden_dim=32)
    fusion.gate_output_scale = torch.tensor([100.0, -1.0, 2.0, 3.0])
    fusion.gate_output_scale_max = 8.0
    scale = fusion.get_gate_output_scale_tensor(device=torch.device("cpu"), dtype=torch.float32)
    assert torch.all(scale >= 0.0)
    assert float(scale[0]) == pytest.approx(8.0)
    assert float(scale[1]) == pytest.approx(0.0)


def test_f26_metrics_and_dashboard_include_adapter_gate_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "reference_v3_6_medium_locked",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.3,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.2,
                "attention_bias_gate_mean_vision": 0.02,
                "attention_source_bias_gate_mean_vision": 0.02,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "adapter_gate_alpha_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 0.95,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.19,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.39,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.28,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.18,
                "attention_bias_gate_mean_vision": 0.03,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "adapter_gate_alpha_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "mild_vision",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.18,
                "attention_relative_gap_score_mean_vision_ego": 0.09,
                "attention_relative_gap_score_mean_vision_route": 0.09,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.36,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.85,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_mild": 0.14,
                "attention_bias_gate_mean_vision": 0.03,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.03,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "adapter_gate_alpha_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 0.95,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.19,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.39,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.28,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.18,
                "attention_bias_gate_mean_vision": 0.03,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": -0.02,
                "attention_bias_gate_residual_logit_mean_vision": -0.02,
                "attention_bias_gate_adapter_logit_mean_vision": -0.04,
                "attention_bias_gate_adapter_logit_mean_distance": 0.001,
                "attention_bias_gate_adapter_logit_mean_ego": 0.001,
                "attention_bias_gate_adapter_logit_mean_route": 0.001,
                "attention_score_adapter_delta_norm_mean_vision": 0.05,
                "attention_score_adapter_delta_norm_mean_distance": 0.001,
                "attention_score_adapter_delta_norm_mean_ego": 0.001,
                "attention_score_adapter_delta_norm_mean_route": 0.001,
                "adapter_gate_alpha_mean": 3.0,
                "residual_alpha_mean": 2.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.18,
                "attention_relative_gap_score_mean_vision_ego": 0.09,
                "attention_relative_gap_score_mean_vision_route": 0.09,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.36,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.85,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_mild": 0.14,
                "attention_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.01,
                "attention_bias_gate_residual_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_vision": 0.24,
                "attention_bias_gate_adapter_logit_mean_distance": 0.002,
                "attention_bias_gate_adapter_logit_mean_ego": 0.002,
                "attention_bias_gate_adapter_logit_mean_route": 0.002,
                "attention_score_adapter_delta_norm_mean_vision": 0.12,
                "attention_score_adapter_delta_norm_mean_distance": 0.002,
                "attention_score_adapter_delta_norm_mean_ego": 0.002,
                "attention_score_adapter_delta_norm_mean_route": 0.002,
                "adapter_gate_alpha_mean": 3.0,
                "residual_alpha_mean": 2.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
        ]
    )
    for col, value in {
        "attention_bias_mean_vision": 0.0,
        "attention_bias_mean_distance": 0.0,
        "attention_bias_mean_ego": 0.0,
        "attention_bias_mean_route": 0.0,
        "attention_bias_norm_mean": 0.0,
        "attention_bias_abs_mean": 0.0,
    }.items():
        df[col] = value
    metrics = compute_v3_7f2_6_adapter_gate_coupling_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics[metrics["candidate_name"] == "cand"].iloc[0]
    assert row["vision_adapter_gate_logit_mild_minus_clean"] == pytest.approx(0.28)
    assert row["adapter_gate_nonvision_drift_mean"] <= 0.002
    dashboard = build_v3_7f2_6_pass_distance_dashboard(
        metrics,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_activation_adapter_gate_logit" in dashboard.columns
    assert dash_row["distance_to_transfer_efficiency_min"] == pytest.approx(
        row["adapter_to_gate_transfer_efficiency"] - 0.03
    )


def test_f210_metrics_and_dashboard_include_cancel_contrast_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "reference_v3_6_medium_locked",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 1.0,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.2,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.3,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.2,
                "attention_bias_gate_mean_vision": 0.02,
                "attention_source_bias_gate_mean_vision": 0.02,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_vision": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "adapter_gate_alpha_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 0.95,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.19,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.39,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.28,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.18,
                "attention_bias_gate_mean_vision": 0.03,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.01,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": 0.01,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -0.5,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_vision": -0.2,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.01,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "adapter_gate_alpha_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "corruption_type": "mild_vision",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.18,
                "attention_relative_gap_score_mean_vision_ego": 0.09,
                "attention_relative_gap_score_mean_vision_route": 0.09,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.36,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.85,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_mild": 0.14,
                "attention_bias_gate_mean_vision": 0.03,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.0,
                "attention_bias_gate_residual_logit_mean_vision": 0.0,
                "attention_bias_gate_adapter_logit_mean_vision": 0.03,
                "attention_bias_gate_adapter_logit_mean_distance": 0.0,
                "attention_bias_gate_adapter_logit_mean_ego": 0.0,
                "attention_bias_gate_adapter_logit_mean_route": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.02,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.01,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": 0.03,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -0.45,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_vision": -0.1,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.02,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.0,
                "attention_score_adapter_delta_norm_mean_vision": 0.0,
                "attention_score_adapter_delta_norm_mean_distance": 0.0,
                "attention_score_adapter_delta_norm_mean_ego": 0.0,
                "attention_score_adapter_delta_norm_mean_route": 0.0,
                "adapter_gate_alpha_mean": 0.0,
                "residual_alpha_mean": 1.0,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "attention_score_mean_vision": 0.95,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.19,
                "attention_relative_gap_score_mean_vision_ego": 0.1,
                "attention_relative_gap_score_mean_vision_route": 0.1,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.39,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.28,
                "reliability_mean_vision": 0.9,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_clean": 0.18,
                "attention_bias_gate_mean_vision": 0.12,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": -0.02,
                "attention_bias_gate_residual_logit_mean_vision": -0.02,
                "attention_bias_gate_adapter_logit_mean_vision": -0.04,
                "attention_bias_gate_adapter_logit_mean_distance": 0.001,
                "attention_bias_gate_adapter_logit_mean_ego": 0.001,
                "attention_bias_gate_adapter_logit_mean_route": 0.001,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.30,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.02,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": 0.30,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -1.20,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_vision": -0.40,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.01,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.001,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.001,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.001,
                "attention_score_adapter_delta_norm_mean_vision": 0.05,
                "attention_score_adapter_delta_norm_mean_distance": 0.001,
                "attention_score_adapter_delta_norm_mean_ego": 0.001,
                "attention_score_adapter_delta_norm_mean_route": 0.001,
                "adapter_gate_alpha_mean": 3.0,
                "residual_alpha_mean": 1.5,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "attention_score_mean_vision": 0.9,
                "attention_score_mean_distance": 0.2,
                "attention_score_mean_ego": 0.2,
                "attention_score_mean_route": 0.1,
                "attention_relative_gap_score_mean_vision_distance": 0.18,
                "attention_relative_gap_score_mean_vision_ego": 0.09,
                "attention_relative_gap_score_mean_vision_route": 0.09,
                "attention_relative_gap_norm_mean": 0.3,
                "attention_mean_vision": 0.36,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "targetlogit_shapley_mean_vision": 0.2,
                "reliability_mean_vision": 0.85,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "logit_mean_vision_mild": 0.14,
                "attention_bias_gate_mean_vision": 0.14,
                "attention_source_bias_gate_mean_vision": 0.03,
                "attention_bias_gate_residual_mean_vision": 0.01,
                "attention_bias_gate_residual_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_vision": 0.24,
                "attention_bias_gate_adapter_logit_mean_distance": 0.002,
                "attention_bias_gate_adapter_logit_mean_ego": 0.002,
                "attention_bias_gate_adapter_logit_mean_route": 0.002,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.34,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.03,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": 0.35,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -1.10,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_vision": -0.20,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.08,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.002,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.002,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.002,
                "attention_score_adapter_delta_norm_mean_vision": 0.12,
                "attention_score_adapter_delta_norm_mean_distance": 0.002,
                "attention_score_adapter_delta_norm_mean_ego": 0.002,
                "attention_score_adapter_delta_norm_mean_route": 0.002,
                "adapter_gate_alpha_mean": 3.0,
                "residual_alpha_mean": 1.5,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
                "normal_success_rate": 1.0,
                "mild_success_rate": 1.0,
            },
        ]
    )
    for col, value in {
        "attention_bias_mean_vision": 0.0,
        "attention_bias_mean_distance": 0.0,
        "attention_bias_mean_ego": 0.0,
        "attention_bias_mean_route": 0.0,
        "attention_bias_norm_mean": 0.0,
        "attention_bias_abs_mean": 0.0,
        "attention_gate_output_scale_mean_vision": 2.0,
        "attention_gate_output_scale_mean_distance": 1.0,
        "attention_gate_output_scale_mean_ego": 1.0,
        "attention_gate_output_scale_mean_route": 1.0,
        "attention_gate_source_logit_bias_mean_vision": 0.0,
        "attention_gate_source_logit_bias_mean_distance": 0.0,
        "attention_gate_source_logit_bias_mean_ego": 0.0,
        "attention_gate_source_logit_bias_mean_route": 0.0,
    }.items():
        df[col] = value
    metrics = compute_v3_7f2_10_cancel_contrast_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics[metrics["candidate_name"] == "cand"].iloc[0]
    assert row["vision_common_cancel_logit_clean_mean"] == pytest.approx(-1.20)
    assert row["vision_contrast_shift_logit_mild_minus_clean"] == pytest.approx(0.07)
    assert row["common_cancel_nonvision_drift_mean"] <= 0.002
    dashboard = build_v3_7f2_10_pass_distance_dashboard(
        metrics,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_common_cancel_logit" in dashboard.columns
    assert dash_row["distance_to_contrast_shift_logit"] == pytest.approx(0.05)


def test_f211_metrics_and_dashboard_include_cancel_rescue_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "gate_reconstruction_error_mean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_final_gate_mild_minus_clean": 0.006,
                "vision_gate_clean_mean": 0.07,
                "vision_gate_mild_mean": 0.076,
                "vision_gate_clean_below_cap_ratio": 0.4,
                "vision_gate_mild_above_floor_ratio": 0.1,
                "vision_gate_gap_margin_satisfied_ratio": 0.05,
                "attention_bias_gate_mean_vision": 0.07,
                "attention_source_bias_gate_mean_vision": 0.05,
                "attention_bias_gate_common_cancel_logit_mean_vision": -0.45,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_vision": -2.9,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_vision": 0.1,
                "attention_bias_gate_mild_rescue_gate_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_value_mean_vision": 0.5,
                "attention_bias_gate_mild_rescue_value_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_value_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_value_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_vision": 0.01,
                "attention_bias_gate_mild_rescue_logit_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_route": 0.0,
                "normal_success_rate": 1.0,
                "eval_return": 1.0,
                "eval_success_rate": 1.0,
                "eval_collision_rate": 0.0,
                "eval_timeout_rate": 0.0,
                "eval_avg_progress_to_goal": 1.0,
                "consistency_loss": 0.0,
                "consistency_gap_mean": 0.0,
                "same_state_action_diff_clean_vs_corrupt": 0.0,
                "reference_same_state_action_diff_clean_vs_corrupt": 0.0,
                "reliability_mean_vision": 0.85,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "attention_mean_vision": 0.4,
                "attention_mean_distance": 0.2,
                "attention_mean_ego": 0.2,
                "attention_mean_route": 0.2,
                "attention_score_mean_vision": 0.18,
                "attention_score_mean_distance": 0.05,
                "attention_score_mean_ego": 0.03,
                "attention_score_mean_route": 0.01,
                "targetlogit_shapley_mean_vision": 0.2,
                "logit_mean_vision_mild": np.nan,
                "attention_score_adapter_delta_norm_mean_vision": 0.12,
                "attention_score_adapter_delta_norm_mean_distance": 0.002,
                "attention_score_adapter_delta_norm_mean_ego": 0.002,
                "attention_score_adapter_delta_norm_mean_route": 0.002,
                "attention_bias_gate_residual_mean_vision": 0.01,
                "attention_bias_gate_residual_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_vision": 0.24,
                "attention_bias_gate_adapter_logit_mean_distance": 0.002,
                "attention_bias_gate_adapter_logit_mean_ego": 0.002,
                "attention_bias_gate_adapter_logit_mean_route": 0.002,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.34,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": -2.9,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "adapter_gate_alpha_mean": 3.0,
                "residual_alpha_mean": 1.5,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "gate_reconstruction_error_mean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_final_gate_mild_minus_clean": 0.006,
                "vision_gate_clean_mean": 0.07,
                "vision_gate_mild_mean": 0.076,
                "vision_gate_clean_below_cap_ratio": 0.4,
                "vision_gate_mild_above_floor_ratio": 0.1,
                "vision_gate_gap_margin_satisfied_ratio": 0.05,
                "attention_bias_gate_mean_vision": 0.076,
                "attention_source_bias_gate_mean_vision": 0.05,
                "attention_bias_gate_common_cancel_logit_mean_vision": -0.40,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_vision": -2.85,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_contrast_shift_logit_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_vision": 0.3,
                "attention_bias_gate_mild_rescue_gate_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_value_mean_vision": 0.5,
                "attention_bias_gate_mild_rescue_value_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_value_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_value_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_vision": 0.06,
                "attention_bias_gate_mild_rescue_logit_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_route": 0.0,
                "normal_success_rate": 1.0,
                "eval_return": 1.0,
                "eval_success_rate": 1.0,
                "eval_collision_rate": 0.0,
                "eval_timeout_rate": 0.0,
                "eval_avg_progress_to_goal": 1.0,
                "consistency_loss": 0.0,
                "consistency_gap_mean": 0.0,
                "same_state_action_diff_clean_vs_corrupt": 0.0,
                "reference_same_state_action_diff_clean_vs_corrupt": 0.0,
                "reliability_mean_vision": 0.8,
                "reliability_mean_distance": 0.8,
                "reliability_mean_ego": 0.8,
                "reliability_mean_route": 0.8,
                "attention_mean_vision": 0.42,
                "attention_mean_distance": 0.19,
                "attention_mean_ego": 0.19,
                "attention_mean_route": 0.20,
                "attention_score_mean_vision": 0.14,
                "attention_score_mean_distance": 0.05,
                "attention_score_mean_ego": 0.03,
                "attention_score_mean_route": 0.01,
                "targetlogit_shapley_mean_vision": 0.2,
                "logit_mean_vision_mild": 0.14,
                "attention_score_adapter_delta_norm_mean_vision": 0.12,
                "attention_score_adapter_delta_norm_mean_distance": 0.002,
                "attention_score_adapter_delta_norm_mean_ego": 0.002,
                "attention_score_adapter_delta_norm_mean_route": 0.002,
                "attention_bias_gate_residual_mean_vision": 0.01,
                "attention_bias_gate_residual_logit_mean_vision": 0.01,
                "attention_bias_gate_adapter_logit_mean_vision": 0.24,
                "attention_bias_gate_adapter_logit_mean_distance": 0.002,
                "attention_bias_gate_adapter_logit_mean_ego": 0.002,
                "attention_bias_gate_adapter_logit_mean_route": 0.002,
                "attention_bias_gate_scaled_delta_logit_mean_vision": 0.34,
                "attention_bias_gate_scaled_delta_logit_mean_distance": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_ego": 0.0,
                "attention_bias_gate_scaled_delta_logit_mean_route": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_vision": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_threshold_shift_logit_mean_route": 0.0,
                "attention_bias_gate_post_shift_logit_mean_vision": -2.85,
                "attention_bias_gate_post_shift_logit_mean_distance": 0.0,
                "attention_bias_gate_post_shift_logit_mean_ego": 0.0,
                "attention_bias_gate_post_shift_logit_mean_route": 0.0,
                "adapter_gate_alpha_mean": 3.0,
                "residual_alpha_mean": 1.5,
                "clean_score_anchor_error": 0.0,
                "clean_action_anchor_error": 0.0,
            },
        ]
    )
    for col, value in {
        "attention_relative_gap_score_mean_vision_distance": 0.01,
        "attention_relative_gap_score_mean_vision_ego": 0.01,
        "attention_relative_gap_score_mean_vision_route": 0.01,
        "attention_relative_gap_norm_mean": 0.01,
        "attention_bias_mean_vision": 0.0,
        "attention_bias_mean_distance": 0.0,
        "attention_bias_mean_ego": 0.0,
        "attention_bias_mean_route": 0.0,
        "attention_bias_norm_mean": 0.0,
        "attention_bias_abs_mean": 0.0,
        "attention_gate_output_scale_mean_vision": 2.0,
        "attention_gate_output_scale_mean_distance": 1.0,
        "attention_gate_output_scale_mean_ego": 1.0,
        "attention_gate_output_scale_mean_route": 1.0,
        "attention_gate_source_logit_bias_mean_vision": 0.0,
        "attention_gate_source_logit_bias_mean_distance": 0.0,
        "attention_gate_source_logit_bias_mean_ego": 0.0,
        "attention_gate_source_logit_bias_mean_route": 0.0,
        "attention_gate_common_cancel_alpha_mean_vision": 3.5,
        "attention_gate_common_cancel_alpha_mean_distance": 0.0,
        "attention_gate_common_cancel_alpha_mean_ego": 0.0,
        "attention_gate_common_cancel_alpha_mean_route": 0.0,
        "attention_gate_contrast_shift_alpha_mean_vision": 0.0,
        "attention_gate_contrast_shift_alpha_mean_distance": 0.0,
        "attention_gate_contrast_shift_alpha_mean_ego": 0.0,
        "attention_gate_contrast_shift_alpha_mean_route": 0.0,
        "attention_gate_mild_rescue_alpha_mean_vision": 2.5,
        "attention_gate_mild_rescue_alpha_mean_distance": 0.0,
        "attention_gate_mild_rescue_alpha_mean_ego": 0.0,
        "attention_gate_mild_rescue_alpha_mean_route": 0.0,
        "logit_mean_vision_clean": 0.12,
        "logit_mean_ego_clean": 0.05,
        "current_clean_logit_mean_vision": 0.12,
        "current_clean_logit_mean_ego": 0.05,
        "reference_logit_mean_vision_clean": 0.12,
        "reference_logit_mean_ego_clean": 0.05,
    }.items():
        df[col] = value
    reference_rows = []
    for col, value in {
        "attention_mean_vision": 0.4,
        "attention_mean_distance": 0.2,
        "attention_mean_ego": 0.2,
        "attention_mean_route": 0.2,
        "reliability_mean_vision": 0.85,
        "reliability_mean_distance": 0.8,
        "reliability_mean_ego": 0.8,
        "reliability_mean_route": 0.8,
        "eval_return": 1.0,
        "eval_success_rate": 1.0,
        "eval_collision_rate": 0.0,
        "eval_timeout_rate": 0.0,
        "eval_avg_progress_to_goal": 1.0,
        "consistency_loss": 0.0,
        "consistency_gap_mean": 0.0,
        "same_state_action_diff_clean_vs_corrupt": 0.0,
        "reference_same_state_action_diff_clean_vs_corrupt": 0.0,
        "targetlogit_shapley_mean_vision": 0.2,
        "logit_mean_vision_clean": 0.12,
        "logit_mean_vision_mild": 0.14,
        "current_clean_logit_mean_vision": 0.12,
        "reference_logit_mean_vision_clean": 0.12,
        "attention_score_adapter_delta_norm_mean_vision": 0.12,
        "attention_score_adapter_delta_norm_mean_distance": 0.002,
        "attention_score_adapter_delta_norm_mean_ego": 0.002,
        "attention_score_adapter_delta_norm_mean_route": 0.002,
        "attention_relative_gap_score_mean_vision_distance": 0.01,
        "attention_relative_gap_score_mean_vision_ego": 0.01,
        "attention_relative_gap_score_mean_vision_route": 0.01,
        "attention_relative_gap_norm_mean": 0.01,
        "attention_bias_mean_vision": 0.0,
        "attention_bias_mean_distance": 0.0,
        "attention_bias_mean_ego": 0.0,
        "attention_bias_mean_route": 0.0,
        "attention_bias_norm_mean": 0.0,
        "attention_bias_abs_mean": 0.0,
    }.items():
        df[col] = value
    ref_normal = df.iloc[0].copy()
    ref_normal["candidate_name"] = "reference_v3_6_medium_locked"
    ref_normal["corruption_type"] = "normal_eval"
    reference_rows.append(ref_normal)
    source_normal = df.iloc[0].copy()
    source_normal["candidate_name"] = "reference_v3_7c_medium_locked"
    source_normal["corruption_type"] = "normal_eval"
    reference_rows.append(source_normal)
    source_mild = df.iloc[1].copy()
    source_mild["candidate_name"] = "reference_v3_7c_medium_locked"
    source_mild["corruption_type"] = "mild_vision"
    reference_rows.append(source_mild)
    df = pd.concat([df, pd.DataFrame(reference_rows)], ignore_index=True)
    metrics = compute_v3_7f2_11_cancel_rescue_metrics(
        df,
        candidate_thresholds={"cand": {"clean_gate_max": 0.02, "mild_gate_min": 0.08, "gate_gap_margin": 0.05}},
    )
    row = metrics[metrics["candidate_name"] == "cand"].iloc[0]
    assert row["vision_mild_rescue_logit_mild_minus_clean"] == pytest.approx(0.05)
    assert row["vision_gate_clean_near_source_ratio"] == pytest.approx(0.0)
    dashboard = build_v3_7f2_11_pass_distance_dashboard(
        metrics,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_mild_rescue_gap" in dashboard.columns
    assert dash_row["distance_to_mild_rescue_gap"] == pytest.approx(0.02)


def test_f212_metrics_and_dashboard_include_selective_rescue_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "gate_reconstruction_error_mean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_final_gate_mild_minus_clean": 0.006,
                "vision_gate_clean_mean": 0.05,
                "vision_gate_mild_mean": 0.056,
                "vision_gate_clean_below_cap_ratio": 0.8,
                "vision_gate_mild_above_floor_ratio": 0.1,
                "vision_gate_gap_margin_satisfied_ratio": 0.05,
                "attention_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_bias_gate_post_shift_excess_mean_vision": 0.03,
                "attention_bias_gate_post_shift_excess_mean_distance": 0.0,
                "attention_bias_gate_post_shift_excess_mean_ego": 0.0,
                "attention_bias_gate_post_shift_excess_mean_route": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_vision": 0.4,
                "attention_bias_gate_common_cancel_gate_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -0.15,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_vision": -2.8,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_vision": 0.05,
                "attention_bias_gate_mild_rescue_gate_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_vision": 0.01,
                "attention_bias_gate_mild_rescue_logit_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_route": 0.0,
                "normal_success_rate": 1.0,
                "attention_score_mean_vision": 0.18,
                "attention_score_mean_distance": 0.05,
                "attention_score_mean_ego": 0.03,
                "attention_score_mean_route": 0.01,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "gate_reconstruction_error_mean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_final_gate_mild_minus_clean": 0.006,
                "vision_gate_clean_mean": 0.05,
                "vision_gate_mild_mean": 0.056,
                "vision_gate_clean_below_cap_ratio": 0.8,
                "vision_gate_mild_above_floor_ratio": 0.1,
                "vision_gate_gap_margin_satisfied_ratio": 0.05,
                "attention_bias_gate_mean_vision": 0.056,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_bias_gate_post_shift_excess_mean_vision": 0.02,
                "attention_bias_gate_post_shift_excess_mean_distance": 0.0,
                "attention_bias_gate_post_shift_excess_mean_ego": 0.0,
                "attention_bias_gate_post_shift_excess_mean_route": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_vision": 0.3,
                "attention_bias_gate_common_cancel_gate_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -0.10,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.001,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.001,
                "attention_bias_gate_post_cancel_logit_mean_vision": -2.75,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_vision": 0.30,
                "attention_bias_gate_mild_rescue_gate_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_gate_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_vision": 0.06,
                "attention_bias_gate_mild_rescue_logit_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_route": 0.0,
                "normal_success_rate": 1.0,
                "attention_score_mean_vision": 0.14,
                "attention_score_mean_distance": 0.05,
                "attention_score_mean_ego": 0.03,
                "attention_score_mean_route": 0.01,
            },
        ]
    )
    for col, value in {
        "attention_mean_vision": 0.4,
        "attention_mean_distance": 0.2,
        "attention_mean_ego": 0.2,
        "attention_mean_route": 0.2,
        "reliability_mean_vision": 0.85,
        "reliability_mean_distance": 0.8,
        "reliability_mean_ego": 0.8,
        "reliability_mean_route": 0.8,
        "eval_return": 1.0,
        "eval_success_rate": 1.0,
        "eval_collision_rate": 0.0,
        "eval_timeout_rate": 0.0,
        "eval_avg_progress_to_goal": 1.0,
        "consistency_loss": 0.0,
        "consistency_gap_mean": 0.0,
        "same_state_action_diff_clean_vs_corrupt": 0.0,
        "reference_same_state_action_diff_clean_vs_corrupt": 0.0,
        "targetlogit_shapley_mean_vision": 0.2,
        "logit_mean_vision_clean": 0.12,
        "logit_mean_vision_mild": 0.14,
        "current_clean_logit_mean_vision": 0.12,
        "reference_logit_mean_vision_clean": 0.12,
        "attention_score_adapter_delta_norm_mean_vision": 0.12,
        "attention_score_adapter_delta_norm_mean_distance": 0.002,
        "attention_score_adapter_delta_norm_mean_ego": 0.002,
        "attention_score_adapter_delta_norm_mean_route": 0.002,
        "attention_relative_gap_score_mean_vision_distance": 0.01,
        "attention_relative_gap_score_mean_vision_ego": 0.01,
        "attention_relative_gap_score_mean_vision_route": 0.01,
        "attention_relative_gap_norm_mean": 0.01,
        "attention_bias_mean_vision": 0.0,
        "attention_bias_mean_distance": 0.0,
        "attention_bias_mean_ego": 0.0,
        "attention_bias_mean_route": 0.0,
        "attention_bias_norm_mean": 0.0,
        "attention_bias_abs_mean": 0.0,
    }.items():
        df[col] = value
    reference_rows = []
    ref_normal = df.iloc[0].copy()
    ref_normal["candidate_name"] = "reference_v3_6_medium_locked"
    reference_rows.append(ref_normal)
    source_normal = df.iloc[0].copy()
    source_normal["candidate_name"] = "reference_v3_7c_medium_locked"
    reference_rows.append(source_normal)
    source_mild = df.iloc[1].copy()
    source_mild["candidate_name"] = "reference_v3_7c_medium_locked"
    reference_rows.append(source_mild)
    df = pd.concat([df, pd.DataFrame(reference_rows)], ignore_index=True)
    metrics = compute_v3_7f2_12_excess_cancel_selective_rescue_metrics(
        df,
        candidate_thresholds={"cand": {"clean_rescue_gate_max": 0.10, "clean_gate_max": 0.02, "mild_gate_min": 0.08}},
    )
    row = metrics[metrics["candidate_name"] == "cand"].iloc[0]
    assert row["vision_mild_rescue_gate_mild_minus_clean"] == pytest.approx(0.25)
    assert row["vision_post_shift_excess_clean_mean"] == pytest.approx(0.03)
    dashboard = build_v3_7f2_12_pass_distance_dashboard(
        metrics,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_rescue_gate_selectivity" in dashboard.columns
    assert dash_row["distance_to_rescue_gate_selectivity"] == pytest.approx(0.10)


def _f213_loss_overrides(
    *,
    cancel_temperature: float = 4.0,
    selector_temperature: float = 4.0,
    rescue_selector_floor: float = 0.55,
    mild_gate_min: float = 0.08,
) -> dict[str, object]:
    return {
        "use_relative_score_head": True,
        "use_reliability_bias_head": True,
        "use_reliability_bias_gate": True,
        "gate_mode": "source_residual_logit_v2",
        "gate_input_mode": "source_residual_adapter_delta_aware_v2",
        "adapter_gate_input_mode": "dedicated_v1",
        "gate_output_scale_mode": "fixed",
        "gate_output_scale_init": [2.0, 1.0, 1.0, 1.0],
        "gate_common_cancel_alpha_mode": "learnable",
        "gate_common_cancel_alpha_init": [1.0, 0.0, 0.0, 0.0],
        "gate_mild_rescue_alpha_mode": "learnable",
        "gate_mild_rescue_alpha_init": [2.0, 0.0, 0.0, 0.0],
        "cancel_temperature": cancel_temperature,
        "selector_temperature": selector_temperature,
        "rescue_selector_floor": rescue_selector_floor,
        "clean_gate_max": 0.02,
        "mild_gate_min": mild_gate_min,
        "gate_gap_margin": 0.05,
        "common_cancel_anchor_gate": 0.05,
    }


def _compute_f213_test_outputs(
    fusion: AttentionFusion,
    *,
    zero_modules: bool = False,
) -> tuple[torch.Tensor, ...]:
    feature_dim = int(getattr(fusion.base_score_head[0], "in_features", 64))
    bias_input_dim = int(getattr(fusion.reliability_bias_head[0], "in_features", feature_dim + 2))
    stacked = torch.randn(1, 4, feature_dim)
    bias_inputs = torch.randn(1, 4, bias_input_dim)
    gate_inputs = torch.randn(1, 4, 6)
    adapter_gate_inputs = torch.randn(1, 4, 16)
    mild_rescue_inputs = torch.randn(1, 4, 16)
    source_gate_scores = torch.full((1, 4), 0.01)
    fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
        adapter_gate_inputs=adapter_gate_inputs,
        source_gate_scores=source_gate_scores,
        mild_rescue_inputs=mild_rescue_inputs,
    )
    if zero_modules:
        with torch.no_grad():
            for module in (
                fusion.reliability_bias_gate_residual_head,
                fusion.reliability_bias_gate_adapter_coupler,
                fusion.reliability_bias_gate_common_cancel_head,
                fusion.reliability_bias_gate_mild_evidence_input_adapter,
                fusion.reliability_bias_gate_mild_value_input_adapter,
                fusion.reliability_bias_gate_mild_rescue_gate_head,
                fusion.reliability_bias_gate_mild_rescue_value_head,
            ):
                for parameter in module.parameters():
                    parameter.zero_()
    return fusion.compute_scores(
        stacked,
        bias_inputs=bias_inputs,
        gate_inputs=gate_inputs,
        adapter_gate_inputs=adapter_gate_inputs,
        source_gate_scores=source_gate_scores,
        mild_rescue_inputs=mild_rescue_inputs,
    )


def test_anchor_cancel_selector_routing_properties() -> None:
    model = _build_stage1_model()
    apply_attention_loss_overrides_to_model(model, _f213_loss_overrides(rescue_selector_floor=0.55))
    fusion = model.fusion_module
    outputs = _compute_f213_test_outputs(fusion, zero_modules=True)
    gate_post_shift_excess = outputs[15]
    gate_common_cancel_logits = outputs[17]
    selector_probs = outputs[20]
    selector_actives = outputs[21]
    rescue_logits = outputs[23]
    assert torch.allclose(gate_post_shift_excess, torch.zeros_like(gate_post_shift_excess))
    assert torch.allclose(gate_common_cancel_logits, torch.zeros_like(gate_common_cancel_logits))
    assert torch.all(selector_probs <= fusion.rescue_selector_floor)
    assert torch.allclose(selector_actives, torch.zeros_like(selector_actives))
    assert torch.all(rescue_logits >= 0.0)


def test_apply_attention_loss_overrides_propagates_selector_routing_knobs() -> None:
    model = _build_stage1_model()
    apply_attention_loss_overrides_to_model(
        model,
        _f213_loss_overrides(
            cancel_temperature=5.0,
            selector_temperature=6.0,
            rescue_selector_floor=0.55,
            mild_gate_min=0.10,
        ),
    )
    fusion = model.fusion_module
    assert fusion.cancel_temperature == pytest.approx(5.0)
    assert fusion.selector_temperature == pytest.approx(6.0)
    assert fusion.rescue_selector_floor == pytest.approx(0.55)
    assert fusion.mild_gate_min == pytest.approx(0.10)
    assert fusion.clean_gate_max == pytest.approx(0.02)
    assert fusion.gate_gap_margin == pytest.approx(0.05)


def test_selector_floor_routing_uses_config_applied_floor() -> None:
    model = _build_stage1_model()
    apply_attention_loss_overrides_to_model(model, _f213_loss_overrides(rescue_selector_floor=0.55))
    fusion = model.fusion_module
    outputs = _compute_f213_test_outputs(fusion, zero_modules=True)
    selector_probs = outputs[20]
    selector_actives = outputs[21]
    assert torch.all(selector_probs <= fusion.rescue_selector_floor)
    assert torch.allclose(selector_actives, torch.zeros_like(selector_actives))


def test_mild_gate_min_propagation_changes_rescue_need() -> None:
    model_08 = _build_stage1_model()
    apply_attention_loss_overrides_to_model(model_08, _f213_loss_overrides(mild_gate_min=0.08))
    model_10 = _build_stage1_model()
    apply_attention_loss_overrides_to_model(model_10, _f213_loss_overrides(mild_gate_min=0.10))
    post_cancel_logit = torch.tensor([-2.5], dtype=torch.float32)
    mild_need_08 = torch.relu(torch.logit(torch.tensor(model_08.fusion_module.mild_gate_min)) - post_cancel_logit)
    mild_need_10 = torch.relu(torch.logit(torch.tensor(model_10.fusion_module.mild_gate_min)) - post_cancel_logit)
    assert mild_need_10.item() > mild_need_08.item()


def test_anchor_cancel_selector_freeze_helper_includes_value_adapter() -> None:
    model = _build_stage1_model()
    model.freeze_attention_source_gate_anchor_cancel_selector_rescue_micro_unfreeze(
        train_gate_input_adapter=True
    )
    assert any(
        parameter.requires_grad
        for parameter in model.fusion_module.reliability_bias_gate_mild_value_input_adapter.parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.fusion_module.score_adapter.parameters())
    assert not any(parameter.requires_grad for parameter in model.fusion_module.relative_gap_head.parameters())


def test_cancel_and_selector_helper_losses() -> None:
    cancel_loss = compute_cancel_gate_floor_loss(
        clean_common_cancel_prob=torch.tensor([0.6, 0.9]),
        clean_excess=torch.tensor([1.2, 0.3]),
        trigger=0.5,
        min_prob=0.8,
    )
    selector_margin = compute_selector_pair_margin_loss(
        clean_selector_prob=torch.tensor([0.1]),
        mild_selector_prob=torch.tensor([0.2]),
        margin=0.25,
    )
    assert cancel_loss.item() > 0.0
    assert selector_margin.item() == pytest.approx(0.15)


def test_f213_metrics_and_dashboard_include_selector_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "corruption_type": "normal_eval",
                "gate_reconstruction_error_mean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_final_gate_mild_minus_clean": 0.007,
                "vision_gate_clean_mean": 0.05,
                "vision_gate_mild_mean": 0.06,
                "vision_gate_clean_below_cap_ratio": 0.9,
                "vision_gate_mild_above_floor_ratio": 0.2,
                "vision_gate_gap_margin_satisfied_ratio": 0.1,
                "attention_bias_gate_mean_vision": 0.05,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_bias_gate_post_shift_excess_mean_vision": 1.3,
                "attention_bias_gate_post_shift_excess_mean_distance": 0.0,
                "attention_bias_gate_post_shift_excess_mean_ego": 0.0,
                "attention_bias_gate_post_shift_excess_mean_route": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_vision": 0.86,
                "attention_bias_gate_common_cancel_gate_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -1.2,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_vision": -2.7,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_selector_prob_mean_vision": 0.04,
                "attention_bias_gate_selector_prob_mean_distance": 0.0,
                "attention_bias_gate_selector_prob_mean_ego": 0.0,
                "attention_bias_gate_selector_prob_mean_route": 0.0,
                "attention_bias_gate_selector_active_mean_vision": 0.0,
                "attention_bias_gate_selector_active_mean_distance": 0.0,
                "attention_bias_gate_selector_active_mean_ego": 0.0,
                "attention_bias_gate_selector_active_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_vision": 0.01,
                "attention_bias_gate_mild_rescue_logit_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_route": 0.0,
                "normal_success_rate": 1.0,
                "attention_score_mean_vision": 0.18,
                "attention_score_mean_distance": 0.05,
                "attention_score_mean_ego": 0.03,
                "attention_score_mean_route": 0.01,
            },
            {
                "candidate_name": "cand",
                "corruption_type": "mild_vision",
                "gate_reconstruction_error_mean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_final_gate_mild_minus_clean": 0.007,
                "vision_gate_clean_mean": 0.05,
                "vision_gate_mild_mean": 0.06,
                "vision_gate_clean_below_cap_ratio": 0.9,
                "vision_gate_mild_above_floor_ratio": 0.2,
                "vision_gate_gap_margin_satisfied_ratio": 0.1,
                "attention_bias_gate_mean_vision": 0.06,
                "attention_source_bias_gate_mean_vision": 0.04,
                "attention_bias_gate_post_shift_excess_mean_vision": 1.1,
                "attention_bias_gate_post_shift_excess_mean_distance": 0.0,
                "attention_bias_gate_post_shift_excess_mean_ego": 0.0,
                "attention_bias_gate_post_shift_excess_mean_route": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_vision": 0.80,
                "attention_bias_gate_common_cancel_gate_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_gate_mean_route": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_vision": -1.0,
                "attention_bias_gate_common_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_common_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_vision": -2.3,
                "attention_bias_gate_post_cancel_logit_mean_distance": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_ego": 0.0,
                "attention_bias_gate_post_cancel_logit_mean_route": 0.0,
                "attention_bias_gate_selector_prob_mean_vision": 0.40,
                "attention_bias_gate_selector_prob_mean_distance": 0.0,
                "attention_bias_gate_selector_prob_mean_ego": 0.0,
                "attention_bias_gate_selector_prob_mean_route": 0.0,
                "attention_bias_gate_selector_active_mean_vision": 0.12,
                "attention_bias_gate_selector_active_mean_distance": 0.0,
                "attention_bias_gate_selector_active_mean_ego": 0.0,
                "attention_bias_gate_selector_active_mean_route": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_vision": 0.07,
                "attention_bias_gate_mild_rescue_logit_mean_distance": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_ego": 0.0,
                "attention_bias_gate_mild_rescue_logit_mean_route": 0.0,
                "normal_success_rate": 1.0,
                "attention_score_mean_vision": 0.14,
                "attention_score_mean_distance": 0.05,
                "attention_score_mean_ego": 0.03,
                "attention_score_mean_route": 0.01,
            },
        ]
    )
    for col, value in {
        "attention_mean_vision": 0.4,
        "attention_mean_distance": 0.2,
        "attention_mean_ego": 0.2,
        "attention_mean_route": 0.2,
        "reliability_mean_vision": 0.85,
        "reliability_mean_distance": 0.8,
        "reliability_mean_ego": 0.8,
        "reliability_mean_route": 0.8,
        "eval_return": 1.0,
        "eval_success_rate": 1.0,
        "eval_collision_rate": 0.0,
        "eval_timeout_rate": 0.0,
        "eval_avg_progress_to_goal": 1.0,
        "consistency_loss": 0.0,
        "consistency_gap_mean": 0.0,
        "same_state_action_diff_clean_vs_corrupt": 0.0,
        "reference_same_state_action_diff_clean_vs_corrupt": 0.0,
        "targetlogit_shapley_mean_vision": 0.2,
        "logit_mean_vision_clean": 0.12,
        "logit_mean_vision_mild": 0.14,
        "current_clean_logit_mean_vision": 0.12,
        "reference_logit_mean_vision_clean": 0.12,
        "attention_score_adapter_delta_norm_mean_vision": 0.12,
        "attention_score_adapter_delta_norm_mean_distance": 0.002,
        "attention_score_adapter_delta_norm_mean_ego": 0.002,
        "attention_score_adapter_delta_norm_mean_route": 0.002,
        "attention_relative_gap_score_mean_vision_distance": 0.01,
        "attention_relative_gap_score_mean_vision_ego": 0.01,
        "attention_relative_gap_score_mean_vision_route": 0.01,
        "attention_relative_gap_norm_mean": 0.01,
        "attention_bias_mean_vision": 0.0,
        "attention_bias_mean_distance": 0.0,
        "attention_bias_mean_ego": 0.0,
        "attention_bias_mean_route": 0.0,
        "attention_bias_norm_mean": 0.0,
        "attention_bias_abs_mean": 0.0,
    }.items():
        df[col] = value
    reference_rows = []
    ref_normal = df.iloc[0].copy()
    ref_normal["candidate_name"] = "reference_v3_6_medium_locked"
    reference_rows.append(ref_normal)
    source_normal = df.iloc[0].copy()
    source_normal["candidate_name"] = "reference_v3_7c_medium_locked"
    reference_rows.append(source_normal)
    source_mild = df.iloc[1].copy()
    source_mild["candidate_name"] = "reference_v3_7c_medium_locked"
    reference_rows.append(source_mild)
    df = pd.concat([df, pd.DataFrame(reference_rows)], ignore_index=True)
    metrics = compute_v3_7f2_13_anchor_cancel_selector_rescue_metrics(
        df,
        candidate_thresholds={"cand": {"clean_selector_max": 0.10, "clean_gate_max": 0.02, "mild_gate_min": 0.08}},
    )
    row = metrics[metrics["candidate_name"] == "cand"].iloc[0]
    assert row["vision_selector_prob_mild_minus_clean"] == pytest.approx(0.36)
    assert row["vision_common_cancel_prob_clean_mean"] == pytest.approx(0.86)
    assert row["vision_gate_clean_source_margin"] == pytest.approx(0.005)
    dashboard = build_v3_7f2_13_pass_distance_dashboard(
        metrics,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
    )
    dash_row = dashboard[dashboard["candidate_name"] == "cand"].iloc[0]
    assert "distance_to_rescue_gate_selectivity" in dashboard.columns
    assert "distance_to_clean_source_margin" in dashboard.columns
    assert dash_row["distance_to_clean_source_margin"] == pytest.approx(0.005)
    assert dash_row["distance_to_rescue_gate_selectivity"] == pytest.approx(0.11)


def test_f2_13_summary_includes_clean_source_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    base_score_df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "gate_reconstruction_error_mean": 0.0,
                "vision_post_shift_excess_clean_mean": 1.2,
                "vision_common_cancel_prob_clean_mean": 0.9,
                "vision_post_cancel_logit_clean_mean": -2.5,
                "vision_selector_prob_clean_mean": 0.05,
                "vision_selector_prob_mild_minus_clean": 0.3,
                "vision_mild_rescue_logit_mild_minus_clean": 0.04,
                "vision_final_gate_mild_minus_clean": 0.007,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_gate_mild_above_floor_ratio": 0.1,
                "vision_gate_gap_margin_satisfied_ratio": 0.1,
                "tie_break_score": 0.5,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "gate_reconstruction_error_mean": 0.0,
                "vision_post_shift_excess_clean_mean": 0.0,
                "vision_common_cancel_prob_clean_mean": 0.0,
                "vision_post_cancel_logit_clean_mean": 0.0,
                "vision_selector_prob_clean_mean": 0.0,
                "vision_selector_prob_mild_minus_clean": 0.0,
                "vision_mild_rescue_logit_mild_minus_clean": 0.0,
                "vision_final_gate_mild_minus_clean": 0.0,
                "vision_attention_targetlogit_sign_match_flag": 1.0,
                "vision_attention_relative_score_clean_minus_mild": 0.01,
                "vision_attention_clean_minus_mild": 0.01,
                "vision_gate_mild_above_floor_ratio": 0.0,
                "vision_gate_gap_margin_satisfied_ratio": 0.0,
                "tie_break_score": 0.0,
            },
        ]
    )
    monkeypatch.setattr(
        v3_helpers,
        "build_v3_7f2_11_candidate_score_summary",
        lambda **kwargs: base_score_df.copy(),
    )
    gate_df = pd.DataFrame(
        [
            {
                "candidate_name": "cand",
                "vision_gate_clean_source_margin": 0.02,
                "vision_common_cancel_prob_clean_mean": 0.9,
                "vision_common_cancel_prob_mild_mean": 0.8,
                "vision_post_shift_excess_clean_mean": 1.2,
                "vision_post_shift_excess_mild_mean": 1.0,
                "vision_post_cancel_logit_clean_mean": -2.5,
                "vision_post_cancel_logit_mild_mean": -2.2,
                "vision_selector_prob_clean_mean": 0.05,
                "vision_selector_prob_mild_mean": 0.35,
                "vision_selector_prob_mild_minus_clean": 0.30,
                "vision_selector_active_clean_mean": 0.0,
                "vision_selector_active_mild_mean": 0.10,
                "vision_mild_rescue_logit_clean_mean": 0.0,
                "vision_mild_rescue_logit_mild_mean": 0.04,
                "vision_mild_rescue_logit_mild_minus_clean": 0.04,
                "attention_gate_cancel_temperature": 4.0,
                "attention_gate_selector_temperature": 4.0,
                "attention_gate_rescue_selector_floor": 0.45,
                "attention_gate_mild_gate_min": 0.08,
                "cancel_midrange_stall_warning_flag": 0.0,
                "selector_constant_mode_warning_flag": 0.0,
                "clean_selector_leak_warning_flag": 0.0,
                "common_cancel_nonvision_drift_mean": 0.0,
                "selector_nonvision_drift_mean": 0.0,
            },
            {
                "candidate_name": "reference_v3_7c_medium_locked",
                "vision_gate_clean_source_margin": 0.0,
                "vision_common_cancel_prob_clean_mean": 0.0,
                "vision_common_cancel_prob_mild_mean": 0.0,
                "vision_post_shift_excess_clean_mean": 0.0,
                "vision_post_shift_excess_mild_mean": 0.0,
                "vision_post_cancel_logit_clean_mean": 0.0,
                "vision_post_cancel_logit_mild_mean": 0.0,
                "vision_selector_prob_clean_mean": 0.0,
                "vision_selector_prob_mild_mean": 0.0,
                "vision_selector_prob_mild_minus_clean": 0.0,
                "vision_selector_active_clean_mean": 0.0,
                "vision_selector_active_mild_mean": 0.0,
                "vision_mild_rescue_logit_clean_mean": 0.0,
                "vision_mild_rescue_logit_mild_mean": 0.0,
                "vision_mild_rescue_logit_mild_minus_clean": 0.0,
                "attention_gate_cancel_temperature": 4.0,
                "attention_gate_selector_temperature": 4.0,
                "attention_gate_rescue_selector_floor": 0.45,
                "attention_gate_mild_gate_min": 0.08,
                "cancel_midrange_stall_warning_flag": 0.0,
                "selector_constant_mode_warning_flag": 0.0,
                "clean_selector_leak_warning_flag": 0.0,
                "common_cancel_nonvision_drift_mean": 0.0,
                "selector_nonvision_drift_mean": 0.0,
            },
        ]
    )
    score_df = build_v3_7f2_13_candidate_score_summary(
        df=pd.DataFrame(),
        degradation_df=pd.DataFrame(),
        consistency_df=pd.DataFrame(),
        tiebreak_df=pd.DataFrame(),
        targetlogit_df=pd.DataFrame(),
        gate_df=gate_df,
        baseline_candidate="reference_v3_6_medium_locked",
        source_candidate="reference_v3_7c_medium_locked",
        tiebreak_weights={},
    )
    row = score_df[score_df["candidate_name"] == "cand"].iloc[0]
    assert "vision_gate_clean_source_margin" in score_df.columns
    assert row["vision_gate_clean_source_margin"] == pytest.approx(0.02)


def test_f2_13_eval_rows_include_effective_selector_config() -> None:
    model = _build_stage1_model()
    apply_attention_loss_overrides_to_model(
        model,
        _f213_loss_overrides(
            cancel_temperature=5.0,
            selector_temperature=6.0,
            rescue_selector_floor=0.55,
            mild_gate_min=0.10,
        ),
    )
    teacher_scores = torch.zeros(1, 4)
    teacher_adapter_delta = torch.zeros(1, 4, 64)
    _fused, aux = model.encode_and_fuse(
        _sample_obs(),
        gate_feature_context={
            "clean_teacher_scores": teacher_scores,
            "sign_teacher_scores": teacher_scores,
            "source_gate_scores": torch.full((1, 4), 0.01),
            "clean_teacher_adapter_delta": teacher_adapter_delta,
            "sign_teacher_adapter_delta": teacher_adapter_delta,
        },
    )
    assert aux["attention_gate_cancel_temperature"].item() == pytest.approx(5.0)
    assert aux["attention_gate_selector_temperature"].item() == pytest.approx(6.0)
    assert aux["attention_gate_rescue_selector_floor"].item() == pytest.approx(0.55)
    assert aux["attention_gate_mild_gate_min"].item() == pytest.approx(0.10)
