from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from rcs_ac_v2.data.schema import make_synthetic_privileged_batch
from rcs_ac_v2.evaluation.carla_adapter import load_stage2_policy_runner
from rcs_ac_v2.evaluation.errors import CleanEvalRuntimeFailure
from rcs_ac_v2.evaluation.evaluate_clean import make_local_smoke_summary
from rcs_ac_v2.privileged_stream.policy import (
    PrivilegedPolicy,
    PrivilegedPolicyConfig,
    imagine_policy_rollout,
)
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.training.train_stage1_privileged_policy import run_privileged_policy_smoke


def test_privileged_policy_acts_from_world_model_latent() -> None:
    wm_cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    world_model = PrivilegedWorldModel(wm_cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=4,
        height=8,
        width=8,
        ego_dim=8,
        seed=14,
    )
    latent = world_model(batch).rssm_sequence.latent
    policy = PrivilegedPolicy(PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=32))

    output = policy(latent)
    action, log_prob = policy.act(latent, deterministic=True)

    assert output.logits.shape == (2, 4, wm_cfg.action_dim)
    assert output.value.shape == (2, 4)
    assert action.shape == (2, 4)
    assert log_prob.shape == (2, 4)
    assert int(action.min()) >= 0
    assert int(action.max()) < wm_cfg.action_dim


def test_privileged_policy_imagines_in_world_model_rollout() -> None:
    wm_cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    world_model = PrivilegedWorldModel(wm_cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=4,
        height=8,
        width=8,
        ego_dim=8,
        seed=16,
    )
    initial_state = world_model(batch).rssm_sequence.final_state()
    policy = PrivilegedPolicy(PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=32))

    rollout = imagine_policy_rollout(
        world_model=world_model,
        policy=policy,
        initial_state=initial_state,
        horizon=3,
        deterministic=True,
    )

    assert rollout.latent.shape == (2, 3, world_model.rssm.latent_dim)
    assert rollout.logits.shape == (2, 3, wm_cfg.action_dim)
    assert rollout.action.shape == (2, 3)
    assert rollout.reward.shape == (2, 3)
    assert rollout.cont.shape == (2, 3)
    assert torch.isfinite(rollout.reward).all()
    assert torch.logical_and(rollout.cont >= 0.0, rollout.cont <= 1.0).all()


def test_deterministic_policy_rollout_is_repeatable() -> None:
    wm_cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    torch.manual_seed(17)
    world_model = PrivilegedWorldModel(wm_cfg)
    batch = make_synthetic_privileged_batch(
        batch_size=2,
        sequence_length=4,
        height=8,
        width=8,
        ego_dim=8,
        seed=17,
    )
    initial_state = world_model(batch).rssm_sequence.final_state()
    policy = PrivilegedPolicy(PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=32))

    first = imagine_policy_rollout(
        world_model=world_model,
        policy=policy,
        initial_state=initial_state,
        horizon=3,
        deterministic=True,
    )
    second = imagine_policy_rollout(
        world_model=world_model,
        policy=policy,
        initial_state=initial_state,
        horizon=3,
        deterministic=True,
    )

    assert torch.equal(first.action, second.action)
    assert torch.allclose(first.reward, second.reward)
    assert torch.allclose(first.cont, second.cont)


def test_privileged_policy_smoke_has_stable_finite_metrics() -> None:
    result = run_privileged_policy_smoke(steps=18, seed=15)
    assert result.valid_action_ids
    assert torch.isfinite(torch.tensor(result.final_total_loss))
    assert torch.isfinite(torch.tensor(result.final_critic_loss))
    assert torch.isfinite(torch.tensor(result.rollout_reward_mean))
    assert 0.0 <= result.rollout_continue_mean <= 1.0
    assert result.final_entropy > 0
    assert result.final_critic_loss <= result.initial_critic_loss * 2.0


def test_privileged_policy_smoke_exports_stage2_checkpoints(tmp_path) -> None:
    result = run_privileged_policy_smoke(
        steps=4,
        seed=19,
        bev_height=16,
        bev_width=16,
        ego_dim=8,
        hidden_dim=32,
        output_dir=tmp_path,
        export_checkpoints=True,
    )

    assert result.world_model_checkpoint_path is not None
    assert result.policy_checkpoint_path is not None
    assert result.manifest_path is not None
    assert result.world_model_checkpoint_path.exists()
    assert result.policy_checkpoint_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["clean_gate_evidence"] is False
    assert manifest["world_model"]["sha256"]
    assert manifest["policy"]["sha256"]

    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["device"] = "cpu"
    config["checkpoints"]["world_model"].update(manifest["clean_eval_config_patch"]["checkpoints"]["world_model"])
    config["checkpoints"]["policy"].update(manifest["clean_eval_config_patch"]["checkpoints"]["policy"])
    config["checkpoint_contract"]["require_trained_provenance_for_clean_gate"] = False
    runner = load_stage2_policy_runner(config)
    output = runner.act(
        {
            "privileged_bev": torch.zeros(1, 1, 43, 16, 16),
            "privileged_ego": torch.zeros(1, 1, 8),
        },
        {},
    )
    assert output["logits"].shape == (1, 1, 39)
    assert torch.isfinite(output["logits"]).all()


def test_synthetic_smoke_checkpoints_are_rejected_for_clean_gate(tmp_path) -> None:
    result = run_privileged_policy_smoke(
        steps=4,
        seed=23,
        bev_height=16,
        bev_width=16,
        ego_dim=8,
        hidden_dim=32,
        output_dir=tmp_path,
        export_checkpoints=True,
    )
    manifest = json.loads(result.manifest_path.read_text())
    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["device"] = "cpu"
    config["checkpoints"]["world_model"].update(manifest["clean_eval_config_patch"]["checkpoints"]["world_model"])
    config["checkpoints"]["policy"].update(manifest["clean_eval_config_patch"]["checkpoints"]["policy"])

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)

    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "synthetic_checkpoint_not_clean_evidence"


def test_local_clean_summary_is_not_carla_evidence() -> None:
    summary = make_local_smoke_summary(horizon=128, route_progress=0.0)
    assert summary.safety_pass
    assert summary.source == "local_smoke_not_carla"
