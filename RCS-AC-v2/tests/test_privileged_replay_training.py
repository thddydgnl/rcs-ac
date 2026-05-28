from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rcs_ac_v2.evaluation.carla_adapter import load_stage2_policy_runner
from rcs_ac_v2.evaluation.checkpoint_action_probe import (
    _is_forward_steer_action,
    build_checkpoint_filled_config,
    probe_checkpoint_actions,
)
from rcs_ac_v2.evaluation.strict_actor_collapse_probe import probe_strict_actor_collapse
from rcs_ac_v2.evaluation.candidate_cross_replay_gate import evaluate_candidate_cross_replay_gate
from rcs_ac_v2.evaluation.candidate_decision_summary import (
    summarize_candidate_decision,
    summarize_missing_gate,
)
from rcs_ac_v2.env.discrete_actions import get_discrete_action
from rcs_ac_v2.evaluation.privileged_observation import (
    EGO8_SCHEMA_VERSION,
    PRIVILEGED_OBSERVATION_SOURCE,
    SEMANTIC43_CHANNEL_NAMES_SHA256,
    SEMANTIC43_SCHEMA_VERSION,
)
from rcs_ac_v2.training.collect_stage2_privileged_replay import (
    collect_episode,
    select_privileged_replay_action,
)
from rcs_ac_v2.training.privileged_replay import (
    REPLAY_SOURCE,
    PrivilegedReplaySampler,
    load_privileged_replay_episode,
    save_privileged_replay_episode,
    write_privileged_replay_manifest,
)
from rcs_ac_v2.training.train_stage2_privileged_from_replay import (
    _action_mass_cap_loss,
    _make_component_sampler_stats,
    _make_imitation_class_weights,
    _make_episode_sample_weights,
    _moving_steer_action_mask,
    _moving_steer_binary_loss,
    _moving_steer_false_positive_loss,
    _moving_steer_nonmoving_margin_loss,
    _moving_steer_action_ids,
    _parse_action_mass_cap_args,
    _parse_component_sample_weight_args,
    _route_action_distribution_loss,
    _route_action_override_loss,
    _route_late_moving_steer_margin_loss,
    _route_late_right_steer_margin_loss,
    _right_moving_steer_action_mask,
    _route_speed_action_mass_cap_loss,
    _route_throttle_ceiling_loss,
    _single_replay_bev_hw,
    train_stage2_privileged_from_replay,
)


def _replay_arrays(length: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "privileged_bev": (rng.random((length, 43, 16, 16)) > 0.9).astype(np.float32),
        "privileged_ego": rng.normal(size=(length, 8)).astype(np.float32),
        "action": np.asarray([(7 + 3 * index) % 39 for index in range(length)], dtype=np.int64),
        "reward": np.linspace(0.0, 0.5, length, dtype=np.float32),
        "cont": np.ones(length, dtype=np.float32),
    }


def test_privileged_replay_roundtrip_accepts_32x32_bev_and_records_shapes(tmp_path) -> None:
    arrays = _replay_arrays(length=5)
    arrays["privileged_bev"] = np.zeros((5, 43, 32, 32), dtype=np.float32)
    path = save_privileged_replay_episode(tmp_path / "episode_32_0000.npz", **arrays)

    episode = load_privileged_replay_episode(path)

    assert episode.privileged_bev.shape == (5, 43, 32, 32)
    assert episode.metadata["privileged_bev_shape"] == [5, 43, 32, 32]
    assert episode.metadata["privileged_ego_shape"] == [5, 8]


def test_replay_training_uses_32x32_replay_shape_for_world_model(tmp_path) -> None:
    replay_arrays = _replay_arrays(length=8)
    replay_arrays["privileged_bev"] = np.zeros((8, 43, 32, 32), dtype=np.float32)
    replay_path = save_privileged_replay_episode(
        tmp_path / "episode_32wm_0000.npz",
        **replay_arrays,
    )
    replay_config_path = tmp_path / "clean_128_32.yaml"
    replay_config_path.write_text("stage: stage2_privileged_policy_clean\n")
    replay_manifest_path = write_privileged_replay_manifest(
        tmp_path / "stage2_privileged_replay_manifest.json",
        config_path=replay_config_path,
        episode_paths=[replay_path],
        extra={"git_commit": "abc1234"},
    )

    result = train_stage2_privileged_from_replay(
        replay_paths=[replay_path],
        replay_manifest_path=replay_manifest_path,
        output_dir=tmp_path / "checkpoints",
        seed=31,
        sequence_length=4,
        batch_size=2,
        world_model_steps=1,
        policy_steps=1,
        imitation_weight=1.0,
        behavior_weight=0.0,
        hidden_dim=32,
        device="cpu",
    )
    manifest = json.loads(result.manifest_path.read_text())

    assert manifest["world_model"]["config"]["bev_height"] == 32
    assert manifest["world_model"]["config"]["bev_width"] == 32
    assert manifest["replay_manifest"]["privileged_bev_shapes"] == [[8, 43, 32, 32]]


def test_replay_training_rejects_mixed_bev_hw_shapes(tmp_path) -> None:
    replay16 = load_privileged_replay_episode(
        save_privileged_replay_episode(
            tmp_path / "episode_16_0000.npz",
            **_replay_arrays(length=4),
        )
    )
    replay_arrays32 = _replay_arrays(length=4)
    replay_arrays32["privileged_bev"] = np.zeros((4, 43, 32, 32), dtype=np.float32)
    replay32 = load_privileged_replay_episode(
        save_privileged_replay_episode(
            tmp_path / "episode_32_0000.npz",
            **replay_arrays32,
        )
    )

    with pytest.raises(ValueError, match="one BEV H/W shape"):
        _single_replay_bev_hw([replay16, replay32])


def test_privileged_replay_roundtrip_and_sampler(tmp_path) -> None:
    arrays = _replay_arrays()
    path = save_privileged_replay_episode(tmp_path / "episode_0000.npz", **arrays)

    episode = load_privileged_replay_episode(path)
    sampler = PrivilegedReplaySampler([episode], sequence_length=4, seed=3)
    batch = sampler.sample(batch_size=2)

    assert episode.metadata["source"] == REPLAY_SOURCE
    assert episode.privileged_bev.shape == (6, 43, 16, 16)
    assert batch.privileged_bev.shape == (2, 4, 43, 16, 16)
    assert batch.privileged_ego.shape == (2, 4, 8)
    assert batch.action.dtype == torch.int64


def test_privileged_replay_sampler_can_prioritize_steering_sequences(tmp_path) -> None:
    arrays = _replay_arrays(length=20)
    arrays["action"] = np.full(20, 7, dtype=np.int64)
    arrays["action"][10] = 4
    path = save_privileged_replay_episode(tmp_path / "episode_0000.npz", **arrays)
    episode = load_privileged_replay_episode(path)
    sampler = PrivilegedReplaySampler(
        [episode],
        sequence_length=4,
        seed=3,
        priority_action_ids=_moving_steer_action_ids(),
        priority_sample_fraction=1.0,
    )
    batch = sampler.sample(batch_size=12)
    stats = sampler.priority_stats()

    assert (batch.action == 4).any(dim=1).all()
    assert stats["total_window_count"] == 17
    assert stats["priority_window_count"] == 4
    assert stats["priority_window_fraction"] == pytest.approx(4 / 17)
    assert stats["effective_priority_sample_fraction"] == 1.0


def test_privileged_replay_sampler_can_prioritize_route_windows(tmp_path) -> None:
    arrays = _replay_arrays(length=20)
    arrays["privileged_ego"][:, 1] = np.linspace(0.0, 0.95, 20, dtype=np.float32)
    path = save_privileged_replay_episode(tmp_path / "episode_0000.npz", **arrays)
    episode = load_privileged_replay_episode(path)
    sampler = PrivilegedReplaySampler(
        [episode],
        sequence_length=4,
        seed=3,
        priority_sample_fraction=1.0,
        priority_min_route_progress=0.65,
        priority_max_route_progress=0.75,
    )
    batch = sampler.sample(batch_size=12)
    stats = sampler.priority_stats()
    route = batch.privileged_ego[:, :, 1]

    assert (((route >= 0.65) & (route <= 0.75)).any(dim=1)).all()
    assert stats["priority_window_count"] == 6
    assert stats["priority_window_fraction"] == pytest.approx(6 / 17)
    assert stats["priority_action_ids"] == []
    assert stats["priority_min_route_progress"] == 0.65
    assert stats["priority_max_route_progress"] == 0.75


def test_privileged_replay_sampler_limits_priority_to_allowed_episodes(tmp_path) -> None:
    primary_arrays = _replay_arrays(length=20)
    primary_arrays["privileged_ego"][:, 1] = np.linspace(0.0, 0.95, 20, dtype=np.float32)
    primary_path = save_privileged_replay_episode(
        tmp_path / "episode_target22_0000.npz",
        **primary_arrays,
    )
    late_arrays = _replay_arrays(length=20)
    late_arrays["privileged_ego"][:, 1] = np.linspace(0.0, 0.95, 20, dtype=np.float32)
    late_path = save_privileged_replay_episode(
        tmp_path / "episode_corr048_0000.npz",
        **late_arrays,
    )
    primary_episode = load_privileged_replay_episode(primary_path)
    late_episode = load_privileged_replay_episode(late_path)

    sampler = PrivilegedReplaySampler(
        [primary_episode, late_episode],
        sequence_length=4,
        seed=3,
        priority_sample_fraction=1.0,
        priority_min_route_progress=0.65,
        priority_max_route_progress=0.75,
        priority_episode_mask=[False, True],
    )
    batch = sampler.sample(batch_size=12)
    stats = sampler.priority_stats()
    route = batch.privileged_ego[:, :, 1]

    assert (((route >= 0.65) & (route <= 0.75)).any(dim=1)).all()
    assert stats["included_episode_count"] == 2
    assert stats["priority_episode_count"] == 1
    assert stats["priority_window_count"] == 6
    assert [entry["priority_allowed"] for entry in stats["included_episodes"]] == [False, True]


def test_privileged_replay_sampler_supports_episode_sample_weights(tmp_path) -> None:
    primary_arrays = _replay_arrays(length=20)
    primary_path = save_privileged_replay_episode(
        tmp_path / "episode_target22_0000.npz",
        **primary_arrays,
    )
    late_arrays = _replay_arrays(length=20)
    late_path = save_privileged_replay_episode(
        tmp_path / "episode_late068warm1_0000.npz",
        **late_arrays,
    )
    primary_episode = load_privileged_replay_episode(primary_path)
    late_episode = load_privileged_replay_episode(late_path)

    component_weights = _parse_component_sample_weight_args(["target22=1,late068warm1=0.25"])
    episode_weights = _make_episode_sample_weights(
        [primary_episode, late_episode],
        component_sample_weights=component_weights,
    )
    sampler = PrivilegedReplaySampler(
        [primary_episode, late_episode],
        sequence_length=4,
        seed=3,
        episode_sample_weights=episode_weights,
    )
    stats = sampler.priority_stats()

    assert episode_weights == [1.0, 0.25]
    assert stats["episode_sample_probabilities"] == pytest.approx([0.8, 0.2])
    assert stats["included_episode_count"] == 2
    assert stats["dropped_short_episode_count"] == 0


def test_component_sampler_stats_rejects_positive_weight_for_dropped_component(tmp_path) -> None:
    target_path = save_privileged_replay_episode(
        tmp_path / "episode_target22_0000.npz",
        **_replay_arrays(length=6),
    )
    corr_path = save_privileged_replay_episode(
        tmp_path / "episode_corr050_0000.npz",
        **_replay_arrays(length=3),
    )
    episodes = [load_privileged_replay_episode(target_path), load_privileged_replay_episode(corr_path)]

    stats = _make_component_sampler_stats(
        episodes,
        sequence_length=4,
        component_sample_weights={"corr050": 0.0},
    )
    assert stats["components"]["target22"]["included_episodes"] == 1
    assert stats["components"]["corr050"]["dropped_short_episodes"] == 1
    assert stats["configured_episode_sample_prob_by_component"]["corr050"] == 0.0

    with pytest.raises(ValueError, match="corr050.*sequence_length=4"):
        _make_component_sampler_stats(
            episodes,
            sequence_length=4,
            component_sample_weights={"corr050": 0.25},
        )


def test_candidate_cross_replay_gate_requires_action_and_latent_margins(tmp_path) -> None:
    for replay_name in ("target24", "steer2"):
        replay_dir = tmp_path / replay_name
        replay_dir.mkdir()
        (replay_dir / "stage2_checkpoint_action_probe.json").write_text(
            json.dumps(
                {
                    "recommended_for_clean_rerun": True,
                    "action_counts": {"7": 8, "10": 4},
                    "moving_steer_fraction": 0.1,
                    "dominant_action_fraction": 0.66,
                    "nonzero_throttle_fraction": 1.0,
                    "reverse_action_count": 0,
                }
            )
        )
        (replay_dir / "stage2_replay_latent_probe.json").write_text(
            json.dumps(
                {
                    "majority_action_fraction": 0.68,
                    "linear_probe_train_accuracy": 0.93,
                    "linear_probe_test_accuracy": 0.91,
                }
            )
        )

    result = evaluate_candidate_cross_replay_gate(
        output_dir=tmp_path,
        replay_names=["target24", "steer2"],
        output_path=tmp_path / "stage2_candidate_cross_replay_gate.json",
    )

    assert result["ready_for_clean_eval"] is True
    assert result["clean_gate_evidence"] is False
    assert result["replays"]["target24"]["latent_probe"]["accuracy_margin_over_majority"] == pytest.approx(0.23)
    assert (tmp_path / "stage2_candidate_cross_replay_gate.json").exists()


def test_candidate_cross_replay_gate_rejects_weak_latent_probe(tmp_path) -> None:
    replay_dir = tmp_path / "target24"
    replay_dir.mkdir()
    (replay_dir / "stage2_checkpoint_action_probe.json").write_text(
        json.dumps({"recommended_for_clean_rerun": True})
    )
    (replay_dir / "stage2_replay_latent_probe.json").write_text(
        json.dumps({"majority_action_fraction": 0.68, "linear_probe_test_accuracy": 0.73})
    )

    result = evaluate_candidate_cross_replay_gate(output_dir=tmp_path, replay_names=["target24"])

    assert result["ready_for_clean_eval"] is False
    assert "latent probe" in result["failures"][0]


def test_candidate_cross_replay_gate_uses_headroom_normalized_latent_margin(tmp_path) -> None:
    replay_dir = tmp_path / "signp"
    replay_dir.mkdir()
    (replay_dir / "stage2_checkpoint_action_probe.json").write_text(
        json.dumps(
            {
                "recommended_for_clean_rerun": True,
                "action_counts": {"7": 798, "4": 69, "15": 111},
                "moving_steer_fraction": 0.07,
                "dominant_action_fraction": 0.816,
                "nonzero_throttle_fraction": 1.0,
                "reverse_action_count": 0,
            }
        )
    )
    (replay_dir / "stage2_replay_latent_probe.json").write_text(
        json.dumps(
            {
                "majority_action_fraction": 0.8558282209,
                "linear_probe_train_accuracy": 0.9693,
                "linear_probe_test_accuracy": 0.9846938848,
            }
        )
    )

    result = evaluate_candidate_cross_replay_gate(output_dir=tmp_path, replay_names=["signp"])

    latent = result["replays"]["signp"]["latent_probe"]
    assert result["ready_for_clean_eval"] is True
    assert latent["accuracy_headroom_over_majority"] == pytest.approx(0.1441717791)
    assert latent["accuracy_margin_over_majority"] == pytest.approx(0.1288656639)
    assert latent["accuracy_margin_fraction_of_headroom"] == pytest.approx(0.8938341796)


def test_candidate_cross_replay_gate_reports_route_window_supplement(tmp_path) -> None:
    replay_dir = tmp_path / "signp"
    replay_dir.mkdir()
    (replay_dir / "stage2_checkpoint_action_probe.json").write_text(
        json.dumps(
            {
                "recommended_for_clean_rerun": False,
                "action_counts": {"7": 945, "10": 33},
                "moving_steer_fraction": 0.0337423313,
                "dominant_action_fraction": 0.9662576687,
                "nonzero_throttle_fraction": 1.0,
                "reverse_action_count": 0,
                "zero_throttle_steer_fraction": 0.0,
                "route_progress_bins": {
                    "0.00-0.50": {"count": 750, "action_counts": {"7": 750}},
                    "0.50-0.55": {"count": 72, "action_counts": {"7": 72}},
                    "0.55-0.60": {"count": 75, "action_counts": {"7": 75}},
                    "0.60-0.64": {"count": 36, "action_counts": {"7": 36}},
                    "0.64-0.66": {"count": 39, "action_counts": {"7": 12, "10": 27}},
                    "0.66-0.70": {"count": 6, "action_counts": {"10": 6}},
                },
            }
        )
    )
    (replay_dir / "stage2_replay_latent_probe.json").write_text(
        json.dumps(
            {
                "majority_action_fraction": 0.8558282209,
                "linear_probe_train_accuracy": 0.959,
                "linear_probe_test_accuracy": 0.9642857313,
            }
        )
    )

    result = evaluate_candidate_cross_replay_gate(
        output_dir=tmp_path,
        replay_names=["signp"],
        route_window_action_ids=[10],
        route_window_required_replay_names=["signp"],
        route_window_min_route_progress=0.64,
        route_window_max_route_progress=0.70,
        route_window_min_action_fraction=0.50,
        route_window_early_max_route_progress=0.60,
        route_window_max_early_action_fraction=0.02,
    )

    assert result["ready_for_clean_eval"] is False
    assert result["route_window_gate"]["ready_for_route_window_clean_eval"] is True
    assert result["route_window_gate"]["clean_gate_evidence"] is False
    signp = result["route_window_gate"]["replays"]["signp"]
    assert signp["window_count"] == 45
    assert signp["window_action_count"] == 33
    assert signp["window_action_fraction"] == pytest.approx(33 / 45)
    assert signp["early_action_fraction"] == 0.0


def test_candidate_cross_replay_route_window_gate_rejects_early_leakage(tmp_path) -> None:
    replay_dir = tmp_path / "signp"
    replay_dir.mkdir()
    (replay_dir / "stage2_checkpoint_action_probe.json").write_text(
        json.dumps(
            {
                "recommended_for_clean_rerun": True,
                "action_counts": {"7": 80, "10": 20},
                "nonzero_throttle_fraction": 1.0,
                "reverse_action_count": 0,
                "zero_throttle_steer_fraction": 0.0,
                "route_progress_bins": {
                    "0.00-0.50": {"count": 80, "action_counts": {"7": 70, "10": 10}},
                    "0.64-0.66": {"count": 20, "action_counts": {"10": 20}},
                },
            }
        )
    )
    (replay_dir / "stage2_replay_latent_probe.json").write_text(
        json.dumps({"majority_action_fraction": 0.50, "linear_probe_test_accuracy": 0.90})
    )

    result = evaluate_candidate_cross_replay_gate(
        output_dir=tmp_path,
        replay_names=["signp"],
        route_window_action_ids=[10],
        route_window_required_replay_names=["signp"],
    )

    route_gate = result["route_window_gate"]
    assert route_gate["ready_for_route_window_clean_eval"] is False
    assert any("early route action fraction" in failure for failure in route_gate["failures"])


def test_candidate_decision_summary_promotes_only_to_clean_eval_candidate(tmp_path) -> None:
    gate_path = tmp_path / "stage2_candidate_cross_replay_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "clean_gate_evidence": False,
                "ready_for_clean_eval": True,
                "failures": [],
                "replays": {
                    "target24": {
                        "action_probe": {"path": "target24/stage2_checkpoint_action_probe.json"},
                        "latent_probe": {"path": "target24/stage2_replay_latent_probe.json"},
                    },
                    "steer2": {
                        "action_probe": {"path": "steer2/stage2_checkpoint_action_probe.json"},
                        "latent_probe": {"path": "steer2/stage2_replay_latent_probe.json"},
                    },
                },
            }
        )
    )

    result = summarize_candidate_decision(
        gate_path=gate_path,
        candidate_name="target24",
        checkpoint_manifest_path="checkpoints/stage2_privileged_replay_checkpoint_manifest.json",
    )

    assert result["decision"] == "run_official_512_clean_eval"
    assert result["clean_gate_evidence"] is False
    assert result["ready_for_clean_eval"] is True
    assert any("target24/stage2_checkpoint_action_probe.json" in path for path in result["evidence_files_to_pull"])
    assert "clean PASS" in result["warning"]


def test_candidate_decision_summary_rejects_to_fallback(tmp_path) -> None:
    gate_path = tmp_path / "stage2_candidate_cross_replay_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "clean_gate_evidence": False,
                "ready_for_clean_eval": False,
                "failures": ["target24: action probe recommended_for_clean_rerun is not true"],
                "replays": {},
            }
        )
    )

    result = summarize_candidate_decision(gate_path=gate_path, candidate_name="target24")

    assert result["decision"] == "run_target22_fallback"
    assert result["ready_for_clean_eval"] is False
    assert "target24: action probe" in result["failures"][0]


def test_candidate_decision_summary_waits_for_missing_gate(tmp_path) -> None:
    result = summarize_missing_gate(gate_path=tmp_path / "missing.json", candidate_name="target24")

    assert result["decision"] == "wait_for_cross_replay_gate"
    assert result["clean_gate_evidence"] is False
    assert "missing gate file" in result["failures"][0]


def test_route_late_moving_steer_margin_loss_does_not_leak_early() -> None:
    moving_mask = _moving_steer_action_mask(action_dim=39, device=torch.device("cpu"))
    logits = torch.zeros(1, 3, 39)
    logits[..., 7] = 3.0
    logits[..., 10] = 0.0
    action = torch.full((1, 3), 7, dtype=torch.long)
    ego = torch.zeros(1, 3, 8)
    ego[..., 0] = 0.8
    ego[..., 1] = torch.tensor([0.60, 0.65, 0.70])

    early_only = _route_late_moving_steer_margin_loss(
        logits=logits[:, :2],
        privileged_ego=ego[:, :2],
        action=action[:, :2],
        moving_steer_mask=moving_mask,
        margin=0.25,
        min_speed_norm=0.0,
        min_route_progress=0.666,
        max_route_progress=0.75,
    )
    late = _route_late_moving_steer_margin_loss(
        logits=logits,
        privileged_ego=ego,
        action=action,
        moving_steer_mask=moving_mask,
        margin=0.25,
        min_speed_norm=0.0,
        min_route_progress=0.666,
        max_route_progress=0.75,
    )

    assert early_only.item() == pytest.approx(0.0)
    assert late.item() > 0.0


class _FakeCollectorEnv:
    def __init__(self) -> None:
        self.step_count = 0

    def reset(self) -> dict:
        return {
            "obs": self._obs(),
            "info": {"speed_kmh": 0.0, "heading_error": 0.0, "lane_offset": 0.0, **_simulator_privileged_info()},
        }

    def step(self, control: dict, policy_diagnostics: dict | None = None) -> dict:
        del control, policy_diagnostics
        self.step_count += 1
        return {
            "obs": self._obs(),
            "reward": 0.1,
            "done": self.step_count >= 3,
            "truncated": False,
            "info": {
                "speed_kmh": 8.0,
                "heading_error": 1.0,
                "lane_offset": 0.1,
                "route_progress": float(self.step_count) * 0.25,
                "done_reason": "done" if self.step_count >= 3 else "running",
                **_simulator_privileged_info(),
            },
        }

    @staticmethod
    def _obs() -> dict:
        bev = torch.zeros(1, 1, 43, 16, 16)
        bev[:, :, 0] = 1.0
        bev[:, :, 4, 8, 8] = 1.0
        return {
            "privileged_bev": bev,
            "privileged_ego": torch.zeros(1, 1, 8),
        }


def test_collect_episode_writes_valid_replay_fields() -> None:
    action_id = select_privileged_replay_action(
        {"speed_kmh": 0.0, "heading_error": 3.0, "lane_offset": 0.2}
    )
    episode = collect_episode(_FakeCollectorEnv(), max_steps=5)

    assert 0 <= action_id < 39
    assert episode["privileged_bev"].shape == (3, 43, 16, 16)
    assert episode["privileged_ego"].shape == (3, 8)
    assert episode["action"].shape == (3,)
    assert episode["cont"][-1] == 0.0


def test_collect_episode_can_record_only_after_route_progress() -> None:
    episode = collect_episode(
        _FakeCollectorEnv(),
        max_steps=5,
        record_after_route_progress=0.25,
        min_recorded_steps=2,
    )

    assert episode["privileged_bev"].shape == (2, 43, 16, 16)
    assert episode["action"].shape == (2,)
    assert episode["metadata"]["warmup_steps"] == 1
    assert episode["metadata"]["record_after_route_progress"] == 0.25


def test_replay_collector_explicit_target_speed_overrides_env_default() -> None:
    action_id = select_privileged_replay_action(
        {
            "speed_kmh": 21.0,
            "target_speed_kmh": 18.0,
            "heading_error": 0.0,
            "lane_offset": 0.0,
        },
        target_speed_kmh=35.0,
    )

    assert get_discrete_action(action_id).throttle > 0.0
    assert action_id != 34


def test_replay_collector_waypoint_distance_does_not_force_zero_throttle() -> None:
    action_id = select_privileged_replay_action(
        {
            "speed_kmh": 5.0,
            "target_distance": 1.0,
            "heading_error": 0.0,
            "lane_offset": 0.0,
        },
        target_speed_kmh=35.0,
    )

    assert get_discrete_action(action_id).throttle > 0.0


def test_replay_collector_can_force_minimum_recorded_throttle() -> None:
    action_id = select_privileged_replay_action(
        {
            "speed_kmh": 31.0,
            "heading_error": 0.0,
            "lane_offset": -3.0,
        },
        target_speed_kmh=24.0,
        min_throttle=0.3,
    )

    action = get_discrete_action(action_id)
    assert action.throttle >= 0.3
    assert action_id != 34


def test_collect_episode_records_minimum_throttle_metadata() -> None:
    episode = collect_episode(
        _FakeCollectorEnv(),
        max_steps=5,
        record_after_route_progress=0.25,
        min_recorded_steps=2,
        min_recorded_throttle=0.3,
    )

    assert episode["metadata"]["min_recorded_throttle"] == 0.3


def test_replay_training_exports_clean_eval_loadable_checkpoints(tmp_path) -> None:
    replay_arrays = _replay_arrays(length=8)
    replay_arrays["privileged_ego"][:, 1] = np.linspace(0.0, 0.875, 8, dtype=np.float32)
    replay_path = save_privileged_replay_episode(
        tmp_path / "episode_target22_0000.npz",
        **replay_arrays,
    )
    replay_config_path = tmp_path / "clean_128.yaml"
    replay_config_path.write_text("stage: stage2_privileged_policy_clean\n")
    replay_manifest_path = write_privileged_replay_manifest(
        tmp_path / "stage2_privileged_replay_manifest.json",
        config_path=replay_config_path,
        episode_paths=[replay_path],
        extra={"git_commit": "abc1234"},
    )
    result = train_stage2_privileged_from_replay(
        replay_paths=[replay_path],
        replay_manifest_path=replay_manifest_path,
        output_dir=tmp_path / "checkpoints",
        seed=11,
        sequence_length=4,
        batch_size=2,
        world_model_steps=2,
        policy_steps=2,
        imitation_weight=1.5,
        behavior_weight=0.5,
        imitation_class_balance=True,
        route_priority_sample_fraction=0.25,
        route_priority_action_ids=(10,),
        route_priority_components=("target22",),
        route_priority_min_route_progress=0.25,
        route_priority_max_route_progress=0.75,
        moving_steer_false_positive_weight=0.25,
        moving_steer_binary_weight=0.5,
        moving_steer_nonmoving_margin_weight=0.75,
        moving_steer_nonmoving_margin=0.1,
        world_model_action_aux_weight=0.25,
        component_sample_weights={"target22": 0.5},
        action_mass_cap_weight=0.2,
        action_mass_caps={7: 0.7, 15: 0.55},
        route_speed_action_mass_cap_weight=0.3,
        route_speed_action_ids=(7, 15),
        route_speed_action_mass_cap=0.4,
        route_speed_min_speed_norm=0.75,
        route_speed_min_route_progress=0.55,
        route_speed_max_route_progress=0.75,
        route_action_mass_cap_weight=0.35,
        route_action_mass_cap_ids=(13, 16),
        route_action_mass_cap=0.15,
        route_action_mass_cap_min_speed_norm=0.5,
        route_action_mass_cap_min_route_progress=0.55,
        route_action_mass_cap_max_route_progress=0.66,
        route_action_distribution_weight=0.42,
        route_action_distribution_min_speed_norm=0.5,
        route_action_distribution_min_route_progress=0.0,
        route_action_distribution_max_route_progress=0.55,
        route_late_action_logit_cap_weight=0.44,
        route_late_action_logit_cap_ids=(8, 25),
        route_late_action_logit_cap_margin=0.05,
        route_late_action_logit_cap_min_speed_norm=0.0,
        route_late_action_logit_cap_min_route_progress=0.74,
        route_late_action_logit_cap_max_route_progress=0.80,
        route_throttle_ceiling_weight=0.4,
        route_throttle_ceiling=0.35,
        route_throttle_min_speed_norm=0.7,
        route_throttle_min_route_progress=0.56,
        route_throttle_max_route_progress=0.60,
        route_late_throttle_ceiling_weight=0.45,
        route_late_throttle_ceiling=0.2,
        route_late_throttle_min_speed_norm=0.0,
        route_late_throttle_min_route_progress=0.64,
        route_late_throttle_max_route_progress=0.70,
        route_action_override_weight=0.6,
        route_action_override_id=10,
        route_action_override_min_speed_norm=0.5,
        route_action_override_min_route_progress=0.64,
        route_action_override_max_route_progress=0.70,
        hidden_dim=32,
        device="cpu",
    )
    manifest = json.loads(result.manifest_path.read_text())
    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["device"] = "cpu"
    config["checkpoints"]["world_model"].update(manifest["clean_eval_config_patch"]["checkpoints"]["world_model"])
    config["checkpoints"]["policy"].update(manifest["clean_eval_config_patch"]["checkpoints"]["policy"])

    runner = load_stage2_policy_runner(config)
    output = runner.act(
        {
            "privileged_bev": torch.zeros(1, 1, 43, 16, 16),
            "privileged_ego": torch.zeros(1, 1, 8),
        },
        {},
    )

    assert manifest["source"] == REPLAY_SOURCE
    assert manifest["replay_manifest"]["sha256"]
    assert manifest["replay_manifest"]["privileged_bev_shapes"] == [[8, 43, 16, 16]]
    assert manifest["replay_manifest"]["privileged_ego_shapes"] == [[8, 8]]
    assert manifest["policy"]["metrics"]["final_imitation"] >= 0.0
    assert manifest["imitation_weight"] == 1.5
    assert manifest["behavior_weight"] == 0.5
    assert manifest["imitation_class_balance"] is True
    assert manifest["imitation_balance_power"] == 1.0
    assert manifest["steer_sequence_fraction"] == 0.0
    assert manifest["route_priority_sample_fraction"] == 0.25
    assert manifest["route_priority_action_ids"] == [10]
    assert manifest["route_priority_components"] == ["target22"]
    assert manifest["route_priority_min_route_progress"] == 0.25
    assert manifest["route_priority_max_route_progress"] == 0.75
    assert manifest["moving_steer_false_positive_weight"] == 0.25
    assert manifest["moving_steer_binary_weight"] == 0.5
    assert manifest["moving_steer_nonmoving_margin_weight"] == 0.75
    assert manifest["moving_steer_nonmoving_margin"] == 0.1
    assert manifest["world_model_action_aux_weight"] == 0.25
    assert manifest["component_sample_weights"] == {"target22": 0.5}
    assert manifest["component_sampler_stats"]["components"]["target22"]["included_episodes"] == 1
    assert manifest["component_sampler_stats"]["configured_episode_sample_prob_by_component"] == {
        "target22": 1.0
    }
    assert manifest["action_mass_cap_weight"] == 0.2
    assert manifest["action_mass_caps"] == {"7": 0.7, "15": 0.55}
    assert manifest["route_speed_action_mass_cap_weight"] == 0.3
    assert manifest["route_speed_action_ids"] == [7, 15]
    assert manifest["route_speed_action_mass_cap"] == 0.4
    assert manifest["route_speed_min_speed_norm"] == 0.75
    assert manifest["route_speed_min_route_progress"] == 0.55
    assert manifest["route_speed_max_route_progress"] == 0.75
    assert manifest["route_action_mass_cap_weight"] == 0.35
    assert manifest["route_action_mass_cap_ids"] == [13, 16]
    assert manifest["route_action_mass_cap"] == 0.15
    assert manifest["route_action_mass_cap_min_speed_norm"] == 0.5
    assert manifest["route_action_mass_cap_min_route_progress"] == 0.55
    assert manifest["route_action_mass_cap_max_route_progress"] == 0.66
    assert manifest["route_action_distribution_weight"] == 0.42
    assert manifest["route_action_distribution_min_speed_norm"] == 0.5
    assert manifest["route_action_distribution_min_route_progress"] == 0.0
    assert manifest["route_action_distribution_max_route_progress"] == 0.55
    assert manifest["route_late_action_logit_cap_weight"] == 0.44
    assert manifest["route_late_action_logit_cap_ids"] == [8, 25]
    assert manifest["route_late_action_logit_cap_margin"] == 0.05
    assert manifest["route_late_action_logit_cap_min_speed_norm"] == 0.0
    assert manifest["route_late_action_logit_cap_min_route_progress"] == 0.74
    assert manifest["route_late_action_logit_cap_max_route_progress"] == 0.8
    assert manifest["route_throttle_ceiling_weight"] == 0.4
    assert manifest["route_throttle_ceiling"] == 0.35
    assert manifest["route_throttle_min_speed_norm"] == 0.7
    assert manifest["route_throttle_min_route_progress"] == 0.56
    assert manifest["route_throttle_max_route_progress"] == 0.6
    assert manifest["route_late_throttle_ceiling_weight"] == 0.45
    assert manifest["route_late_throttle_ceiling"] == 0.2
    assert manifest["route_late_throttle_min_speed_norm"] == 0.0
    assert manifest["route_late_throttle_min_route_progress"] == 0.64
    assert manifest["route_late_throttle_max_route_progress"] == 0.7
    assert manifest["route_action_override_weight"] == 0.6
    assert manifest["route_action_override_id"] == 10
    assert manifest["route_action_override_min_speed_norm"] == 0.5
    assert manifest["route_action_override_min_route_progress"] == 0.64
    assert manifest["route_action_override_max_route_progress"] == 0.7
    assert manifest["world_model"]["metrics"]["final_action_aux"] >= 0.0
    assert manifest["policy"]["metrics"]["final_moving_steer_false_positive"] >= 0.0
    assert manifest["policy"]["metrics"]["final_moving_steer_binary"] >= 0.0
    assert manifest["policy"]["metrics"]["final_moving_steer_nonmoving_margin"] >= 0.0
    assert manifest["policy"]["metrics"]["final_action_mass_cap"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_speed_action_mass_cap"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_action_mass_cap"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_action_distribution"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_late_action_logit_cap"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_throttle_ceiling"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_late_throttle_ceiling"] >= 0.0
    assert manifest["policy"]["metrics"]["final_route_action_override"] >= 0.0
    assert manifest["sampler"]["requested_priority_sample_fraction"] == 0.25
    assert "priority_window_count" in manifest["sampler"]
    assert manifest["sampler"]["priority_action_ids"] == [10]
    assert manifest["sampler"]["priority_episode_count"] == 1
    assert manifest["sampler"]["priority_min_route_progress"] == 0.25
    assert manifest["sampler"]["priority_max_route_progress"] == 0.75
    assert manifest["sampler"]["episode_sample_probabilities"] == [1.0]
    assert output["logits"].shape == (1, 1, 39)
    assert torch.isfinite(output["logits"]).all()

    promoted_config_path = tmp_path / "clean_128_with_checkpoints.yaml"
    promoted = build_checkpoint_filled_config(
        base_config_path=source_config,
        checkpoint_manifest_path=result.manifest_path,
        output_config_path=promoted_config_path,
        device="cuda",
        forbidden_action_ids=(16, 19),
    )
    action_probe = probe_checkpoint_actions(
        base_config_path=source_config,
        checkpoint_manifest_path=result.manifest_path,
        project_root=tmp_path,
        max_steps_per_file=4,
        device="cpu",
        output_path=tmp_path / "stage2_checkpoint_action_probe.json",
    )

    assert promoted_config_path.exists()
    assert promoted["dry_run"]["allow_null_checkpoints"] is False
    assert promoted["device"] == "cuda"
    assert promoted["carla"]["forbidden_action_ids"] == [16, 19]
    assert action_probe["clean_gate_evidence"] is False
    assert action_probe["probed_actions"] == 4
    assert action_probe["reverse_action_count"] == 0
    assert "nonzero_throttle_fraction" in action_probe
    assert action_probe["minimum_nonzero_throttle_fraction"] == 0.5
    assert "nonzero_steer_fraction" in action_probe
    assert "moving_steer_fraction" in action_probe
    assert "zero_throttle_steer_fraction" in action_probe
    assert "label_to_predicted_action_counts" in action_probe
    assert "route_progress_bins" in action_probe
    first_route_bin = next(iter(action_probe["route_progress_bins"].values()))
    assert "action_counts" in first_route_bin
    assert "label_action_counts" in first_route_bin
    assert action_probe["minimum_nonzero_steer_fraction"] == 0.02
    assert action_probe["minimum_nonzero_steer_count"] == 12
    assert action_probe["maximum_nonzero_steer_fraction"] == 0.5
    assert action_probe["maximum_dominant_action_fraction"] == 0.85
    assert "dominant_action_fraction" in action_probe
    assert "previous_stuck_action8_fraction" in action_probe
    assert Path(tmp_path / "stage2_checkpoint_action_probe.json").exists()


def test_pdf_stage1_strict_replay_training_allows_only_actor_critic_path(tmp_path) -> None:
    replay_path = save_privileged_replay_episode(
        tmp_path / "episode_pdfstrict_0000.npz",
        **_replay_arrays(length=8),
    )
    replay_config_path = tmp_path / "clean_128.yaml"
    replay_config_path.write_text("stage: stage2_privileged_policy_clean\n")
    replay_manifest_path = write_privileged_replay_manifest(
        tmp_path / "stage2_privileged_replay_manifest.json",
        config_path=replay_config_path,
        episode_paths=[replay_path],
        extra={"git_commit": "abc1234"},
    )

    result = train_stage2_privileged_from_replay(
        replay_paths=[replay_path],
        replay_manifest_path=replay_manifest_path,
        output_dir=tmp_path / "checkpoints",
        seed=12,
        sequence_length=4,
        batch_size=2,
        world_model_steps=2,
        policy_steps=2,
        pdf_stage1_strict=True,
        behavior_weight=1.0,
        imitation_weight=0.0,
        hidden_dim=32,
        device="cpu",
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["pdf_stage1_strict"] is True
    assert manifest["behavior_weight"] == 1.0
    assert manifest["behavior_gamma"] == 0.99
    assert manifest["behavior_lambda"] == 1.0
    assert manifest["behavior_actor_weight"] == 0.1
    assert manifest["behavior_critic_weight"] == 1.0
    assert manifest["behavior_entropy_weight"] == 0.001
    assert manifest["imitation_weight"] == 0.0
    assert manifest["world_model_action_aux_weight"] == 0.0
    assert manifest["component_sample_weights"] == {}
    assert manifest["route_action_override_weight"] == 0.0
    assert manifest["policy"]["metrics"]["valid_action_ids"] is True


def test_strict_actor_collapse_probe_reports_counterfactual_returns(tmp_path) -> None:
    replay_path = save_privileged_replay_episode(
        tmp_path / "episode_actorprobe_0000.npz",
        **_replay_arrays(length=8),
    )
    replay_config_path = tmp_path / "clean_128.yaml"
    replay_config_path.write_text("stage: stage2_privileged_policy_clean\n")
    replay_manifest_path = write_privileged_replay_manifest(
        tmp_path / "stage2_privileged_replay_manifest.json",
        config_path=replay_config_path,
        episode_paths=[replay_path],
        extra={"git_commit": "abc1234"},
    )
    result = train_stage2_privileged_from_replay(
        replay_paths=[replay_path],
        replay_manifest_path=replay_manifest_path,
        output_dir=tmp_path / "checkpoints",
        seed=13,
        sequence_length=4,
        batch_size=2,
        world_model_steps=2,
        policy_steps=2,
        pdf_stage1_strict=True,
        hidden_dim=32,
        device="cpu",
    )
    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    output_path = tmp_path / "strict_actor_probe.json"

    probe = probe_strict_actor_collapse(
        base_config_path=source_config,
        checkpoint_manifest_path=result.manifest_path,
        replay_manifest_path=replay_manifest_path,
        project_root=tmp_path,
        max_steps_per_file=3,
        rollout_horizon=2,
        device="cpu",
        output_path=output_path,
    )

    assert probe["source"] == "stage2_strict_actor_collapse_probe"
    assert probe["clean_gate_evidence"] is False
    assert probe["samples"] == 3
    assert sum(probe["actor_action_counts"].values()) == 3
    assert sum(probe["counterfactual_best_action_counts"].values()) == 3
    assert "actor_equals_counterfactual_best_fraction" in probe
    assert "actor_return_rank_mean" in probe
    assert "imagined_lambda_return_mean" in probe
    assert output_path.exists()


def test_pdf_stage1_strict_rejects_non_pdf_diagnostic_policy_settings(tmp_path) -> None:
    with pytest.raises(ValueError, match="pdf_stage1_strict forbids"):
        train_stage2_privileged_from_replay(
            replay_paths=[tmp_path / "missing.npz"],
            replay_manifest_path=tmp_path / "missing_manifest.json",
            output_dir=tmp_path / "checkpoints",
            pdf_stage1_strict=True,
            imitation_weight=1.0,
            route_priority_sample_fraction=0.5,
            route_priority_components=("target22",),
            route_action_distribution_weight=0.5,
            route_action_override_weight=0.5,
            route_action_override_id=10,
            device="cpu",
        )


def test_action_mass_cap_loss_penalizes_over_cap_action_mass() -> None:
    caps = _parse_action_mass_cap_args(["7=0.25,15=0.5"])
    logits = torch.zeros(2, 3, 39)
    logits[..., 7] = 8.0

    high_loss = _action_mass_cap_loss(logits=logits, action_mass_caps=caps)
    low_loss = _action_mass_cap_loss(logits=torch.zeros(2, 3, 39), action_mass_caps=caps)
    empty_loss = _action_mass_cap_loss(logits=logits, action_mass_caps={})

    assert caps == {7: 0.25, 15: 0.5}
    assert high_loss > low_loss
    assert empty_loss.item() == pytest.approx(0.0)


def test_route_action_distribution_loss_penalizes_distribution_collapse() -> None:
    action = torch.tensor([[7, 7, 10, 13]], dtype=torch.long)
    ego = torch.zeros(1, 4, 8)
    ego[..., 0] = 0.5
    ego[..., 1] = torch.tensor([[0.1, 0.2, 0.3, 0.6]])
    collapsed = torch.zeros(1, 4, 39)
    collapsed[..., 10] = 8.0
    matched = torch.full((1, 4, 39), -8.0)
    matched[..., 7] = torch.log(torch.tensor(0.5))
    matched[..., 10] = torch.log(torch.tensor(0.25))
    matched[..., 13] = torch.log(torch.tensor(0.25))

    high_loss = _route_action_distribution_loss(
        logits=collapsed,
        privileged_ego=ego,
        action=action,
        action_dim=39,
        min_speed_norm=0.0,
        min_route_progress=0.0,
        max_route_progress=0.55,
    )
    low_loss = _route_action_distribution_loss(
        logits=matched,
        privileged_ego=ego,
        action=action,
        action_dim=39,
        min_speed_norm=0.0,
        min_route_progress=0.0,
        max_route_progress=0.55,
    )
    inactive_loss = _route_action_distribution_loss(
        logits=collapsed,
        privileged_ego=ego,
        action=action,
        action_dim=39,
        min_speed_norm=0.0,
        min_route_progress=0.7,
        max_route_progress=0.9,
    )

    assert high_loss > low_loss
    assert inactive_loss.item() == pytest.approx(0.0)


def test_route_speed_action_mass_cap_loss_is_conditional_on_ego_window() -> None:
    logits = torch.zeros(1, 3, 39)
    logits[..., 15] = 8.0
    ego = torch.zeros(1, 3, 8)
    ego[..., 0] = torch.tensor([[0.9, 0.4, 0.9]])
    ego[..., 1] = torch.tensor([[0.6, 0.6, 0.9]])

    active_loss = _route_speed_action_mass_cap_loss(
        logits=logits,
        privileged_ego=ego,
        action_ids=(15,),
        cap=0.2,
        min_speed_norm=0.8,
        min_route_progress=0.55,
        max_route_progress=0.75,
    )
    inactive_loss = _route_speed_action_mass_cap_loss(
        logits=logits,
        privileged_ego=ego,
        action_ids=(15,),
        cap=0.2,
        min_speed_norm=1.0,
        min_route_progress=0.55,
        max_route_progress=0.75,
    )

    assert active_loss.item() > 0.0
    assert inactive_loss.item() == pytest.approx(0.0)


def test_route_throttle_ceiling_loss_penalizes_expected_throttle_in_window() -> None:
    logits = torch.zeros(1, 3, 39)
    logits[..., 15] = 8.0
    ego = torch.zeros(1, 3, 8)
    ego[..., 0] = torch.tensor([[0.9, 0.4, 0.9]])
    ego[..., 1] = torch.tensor([[0.58, 0.58, 0.9]])

    active_loss = _route_throttle_ceiling_loss(
        logits=logits,
        privileged_ego=ego,
        ceiling=0.35,
        min_speed_norm=0.8,
        min_route_progress=0.56,
        max_route_progress=0.60,
    )
    inactive_loss = _route_throttle_ceiling_loss(
        logits=logits,
        privileged_ego=ego,
        ceiling=0.35,
        min_speed_norm=1.0,
        min_route_progress=0.56,
        max_route_progress=0.60,
    )

    assert active_loss.item() > 0.0
    assert inactive_loss.item() == pytest.approx(0.0)


def test_route_action_override_loss_is_conditional_on_ego_window() -> None:
    logits = torch.zeros(1, 3, 39)
    logits[..., 7] = 8.0
    ego = torch.zeros(1, 3, 8)
    ego[..., 0] = torch.tensor([[0.9, 0.4, 0.9]])
    ego[..., 1] = torch.tensor([[0.66, 0.66, 0.9]])

    active_loss = _route_action_override_loss(
        logits=logits,
        privileged_ego=ego,
        action_id=10,
        min_speed_norm=0.8,
        min_route_progress=0.64,
        max_route_progress=0.70,
    )
    inactive_loss = _route_action_override_loss(
        logits=logits,
        privileged_ego=ego,
        action_id=10,
        min_speed_norm=1.0,
        min_route_progress=0.64,
        max_route_progress=0.70,
    )
    disabled_loss = _route_action_override_loss(
        logits=logits,
        privileged_ego=ego,
        action_id=-1,
        min_speed_norm=0.0,
        min_route_progress=0.0,
        max_route_progress=1.0,
    )

    assert active_loss.item() > 1.0
    assert inactive_loss.item() == pytest.approx(0.0)
    assert disabled_loss.item() == pytest.approx(0.0)


def test_moving_steer_false_positive_loss_penalizes_nonmoving_labels() -> None:
    mask = _moving_steer_action_mask(action_dim=39, device=torch.device("cpu"))
    logits = torch.zeros(1, 2, 39)
    logits[0, 0, 10] = 6.0
    logits[0, 1, 7] = 6.0
    actions = torch.tensor([[7, 10]], dtype=torch.long)

    loss = _moving_steer_false_positive_loss(logits=logits, action=actions, moving_steer_mask=mask)

    assert loss.item() > 1.0


def test_moving_steer_binary_loss_penalizes_wrong_moving_state() -> None:
    mask = _moving_steer_action_mask(action_dim=39, device=torch.device("cpu"))
    logits = torch.zeros(1, 2, 39)
    logits[0, 0, 10] = 6.0
    logits[0, 1, 7] = 6.0
    actions = torch.tensor([[7, 10]], dtype=torch.long)

    loss = _moving_steer_binary_loss(logits=logits, action=actions, moving_steer_mask=mask)

    assert loss.item() > 1.0


def test_moving_steer_nonmoving_margin_loss_penalizes_argmax_risk() -> None:
    mask = _moving_steer_action_mask(action_dim=39, device=torch.device("cpu"))
    logits = torch.zeros(1, 2, 39)
    logits[0, 0, 10] = 6.0
    logits[0, 0, 7] = 5.95
    logits[0, 1, 10] = 4.0
    logits[0, 1, 7] = 6.0
    actions = torch.tensor([[7, 10]], dtype=torch.long)

    loss = _moving_steer_nonmoving_margin_loss(
        logits=logits,
        action=actions,
        moving_steer_mask=mask,
        margin=0.25,
    )

    assert loss.item() == pytest.approx(0.30, abs=1e-5)


def test_route_late_right_steer_margin_loss_is_route_windowed_and_directional() -> None:
    mask = _right_moving_steer_action_mask(action_dim=39, device=torch.device("cpu"))
    logits = torch.zeros(1, 3, 39)
    logits[0, 0, 7] = 4.0
    logits[0, 0, 13] = 3.0
    logits[0, 1, 7] = 4.0
    logits[0, 1, 13] = 3.0
    logits[0, 2, 13] = 4.0
    logits[0, 2, 7] = 3.0
    ego = torch.zeros(1, 3, 8)
    ego[0, :, 0] = 0.5
    ego[0, 0, 1] = 0.64
    ego[0, 1, 1] = 0.67
    ego[0, 2, 1] = 0.67

    loss = _route_late_right_steer_margin_loss(
        logits=logits,
        privileged_ego=ego,
        right_moving_steer_mask=mask,
        margin=0.25,
        min_speed_norm=0.0,
        min_route_progress=0.666,
        max_route_progress=0.70,
    )

    assert loss.item() == pytest.approx(0.625, abs=1e-5)


def test_imitation_class_weights_favor_rare_actions(tmp_path) -> None:
    arrays = _replay_arrays(length=8)
    arrays["action"] = np.asarray([7, 7, 7, 7, 7, 7, 7, 4], dtype=np.int64)
    path = save_privileged_replay_episode(tmp_path / "episode_0000.npz", **arrays)
    episode = load_privileged_replay_episode(path)

    weights = _make_imitation_class_weights([episode], action_dim=39, device=torch.device("cpu"))

    assert weights[4] > weights[7]
    assert weights[7] > 0
    assert weights[0] == 0


def test_checkpoint_action_probe_counts_only_forward_steering() -> None:
    assert _is_forward_steer_action(get_discrete_action(4)) is True
    assert _is_forward_steer_action(get_discrete_action(8)) is False
    assert _is_forward_steer_action(get_discrete_action(17)) is False


def test_late_replay_collector_supports_policy_warmup_static_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    collector = (root / "code" / "rcs_ac_v2" / "training" / "collect_stage2_privileged_replay.py").read_text()
    wrapper = (root / "scripts" / "windows" / "collect_stage2_privileged_replay.ps1").read_text()

    assert "warmup_policy_runner: Any | None = None" in collector
    assert "load_stage2_policy_runner" in collector
    assert "warmup_policy_config" in collector
    assert "max_route_progress" in collector
    assert "failed_attempts" in collector
    assert "--skip-failed-episodes" in collector
    assert "--max-attempts" in collector
    assert "--min-recorded-throttle" in collector
    assert "_select_warmup_policy_action" in collector
    assert "warmup_policy_runner.observe_action(action_id)" in collector
    assert '[string]$WarmupPolicyConfig = ""' in wrapper
    assert "[int]$MaxAttempts = 0" in wrapper
    assert "[switch]$SkipFailedEpisodes" in wrapper
    assert "[double]$MinRecordedThrottle = 0.0" in wrapper
    assert '"--warmup-policy-config", $WarmupPolicyConfig' in wrapper
    assert '"--max-attempts", $MaxAttempts' in wrapper
    assert '"--min-recorded-throttle", $MinRecordedThrottle' in wrapper
    assert '"--skip-failed-episodes"' in wrapper


def _simulator_privileged_info() -> dict:
    return {
        "privileged_observation_clean_gate_eligible": True,
        "privileged_observation_source": PRIVILEGED_OBSERVATION_SOURCE,
        "privileged_observation_schema": SEMANTIC43_SCHEMA_VERSION,
        "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
        "privileged_ego_schema": EGO8_SCHEMA_VERSION,
    }
