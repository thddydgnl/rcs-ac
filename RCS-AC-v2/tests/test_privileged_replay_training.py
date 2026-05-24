from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from rcs_ac_v2.evaluation.carla_adapter import load_stage2_policy_runner
from rcs_ac_v2.evaluation.checkpoint_action_probe import (
    _is_forward_steer_action,
    build_checkpoint_filled_config,
    probe_checkpoint_actions,
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
    _make_imitation_class_weights,
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


def test_replay_training_exports_clean_eval_loadable_checkpoints(tmp_path) -> None:
    replay_path = save_privileged_replay_episode(tmp_path / "episode_0000.npz", **_replay_arrays(length=8))
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
    assert manifest["policy"]["metrics"]["final_imitation"] >= 0.0
    assert manifest["imitation_weight"] == 1.5
    assert manifest["behavior_weight"] == 0.5
    assert manifest["imitation_class_balance"] is True
    assert output["logits"].shape == (1, 1, 39)
    assert torch.isfinite(output["logits"]).all()

    promoted_config_path = tmp_path / "clean_128_with_checkpoints.yaml"
    promoted = build_checkpoint_filled_config(
        base_config_path=source_config,
        checkpoint_manifest_path=result.manifest_path,
        output_config_path=promoted_config_path,
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
    assert action_probe["clean_gate_evidence"] is False
    assert action_probe["probed_actions"] == 4
    assert action_probe["reverse_action_count"] == 0
    assert "nonzero_throttle_fraction" in action_probe
    assert action_probe["minimum_nonzero_throttle_fraction"] == 0.5
    assert "nonzero_steer_fraction" in action_probe
    assert "moving_steer_fraction" in action_probe
    assert "zero_throttle_steer_fraction" in action_probe
    assert action_probe["minimum_nonzero_steer_fraction"] == 0.02
    assert action_probe["minimum_nonzero_steer_count"] == 12
    assert "previous_stuck_action8_fraction" in action_probe
    assert Path(tmp_path / "stage2_checkpoint_action_probe.json").exists()


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


def _simulator_privileged_info() -> dict:
    return {
        "privileged_observation_clean_gate_eligible": True,
        "privileged_observation_source": PRIVILEGED_OBSERVATION_SOURCE,
        "privileged_observation_schema": SEMANTIC43_SCHEMA_VERSION,
        "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
        "privileged_ego_schema": EGO8_SCHEMA_VERSION,
    }
