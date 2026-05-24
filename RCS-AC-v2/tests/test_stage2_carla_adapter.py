from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from rcs_ac_v2.evaluation.carla_adapter import (
    SharedCoreCarlaEnvAdapter,
    Stage2CarlaObservationAdapter,
    create_stage2_adapter_bundle,
    load_stage2_policy_runner,
)
from rcs_ac_v2.evaluation.evaluate_clean import CleanEvalRuntimeFailure, run_clean_eval_loop, run_real_carla_eval
from rcs_ac_v2.evaluation.probe_stage2_windows_env import probe_stage2_env_once, write_probe_result
from rcs_ac_v2.privileged_stream.policy import PrivilegedPolicy, PrivilegedPolicyConfig
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.utils.checkpoint import save_checkpoint


def _base_config(tmp_path: Path, *, horizon: int = 2) -> dict:
    source = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source.read_text())
    output_dir = tmp_path / "windows_128"
    config["horizon"] = horizon
    config["gate"]["pass_criteria"]["steps"] = horizon
    config["output_dir"] = str(output_dir)
    config["summary_path"] = str(output_dir / "summary.json")
    config["metrics_path"] = str(output_dir / "metrics.csv")
    config["device"] = "cpu"
    return config


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "clean_128.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stage2_checkpoints(tmp_path: Path, config: dict) -> None:
    wm_cfg = PrivilegedWorldModelConfig(bev_height=16, bev_width=16, ego_dim=8, hidden_dim=32)
    world_model = PrivilegedWorldModel(wm_cfg)
    policy = PrivilegedPolicy(PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=32))

    world_model_path = save_checkpoint(
        tmp_path / "world_model.pt",
        {
            "model_state_dict": world_model.state_dict(),
            "config": wm_cfg.__dict__,
        },
    )
    policy_path = save_checkpoint(
        tmp_path / "policy.pt",
        {
            "model_state_dict": policy.state_dict(),
            "config": policy.cfg.__dict__,
        },
    )
    config["checkpoints"]["world_model"]["path"] = str(world_model_path)
    config["checkpoints"]["world_model"]["sha256"] = _sha256(world_model_path)
    config["checkpoints"]["policy"]["path"] = str(policy_path)
    config["checkpoints"]["policy"]["sha256"] = _sha256(policy_path)


class FakeRawEnv:
    def __init__(self) -> None:
        self.actions: list[object] = []
        self.closed = False
        self.step_count = 0

    def reset(self) -> dict:
        return {"obs": self._obs(), "info": {"route_progress": 0.0, "speed_kmh": 0.0}}

    def step(self, action: object, policy_diagnostics: dict | None = None) -> dict:
        del policy_diagnostics
        self.actions.append(action)
        self.step_count += 1
        return {
            "obs": self._obs(),
            "reward": 0.25,
            "done": False,
            "truncated": False,
            "info": {
                "route_progress": 0.1 * self.step_count,
                "speed_kmh": 12.0,
                "collision": False,
                "offroad": False,
                "stuck": False,
                "done_reason": "running",
                "reward_route_progress": 0.1,
                "reward_collision": 0.0,
                "reward_offroad": 0.0,
                "reward_stuck": 0.0,
            },
        }

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _obs() -> dict:
        return {
            "privileged_bev": torch.zeros(43, 16, 16),
            "privileged_ego": torch.zeros(8),
        }


class FakeRawEnvWithoutPrivilegedObs(FakeRawEnv):
    @staticmethod
    def _obs() -> dict:
        return {
            "rgb": torch.zeros(3, 64, 64),
            "speed_kmh": 0.0,
        }


def test_observation_adapter_accepts_stage2_privileged_shapes(tmp_path) -> None:
    config = _base_config(tmp_path)
    adapter = Stage2CarlaObservationAdapter(config)
    obs = adapter.adapt(FakeRawEnv._obs(), {})

    assert tuple(obs["privileged_bev"].shape) == (1, 1, 43, 16, 16)
    assert tuple(obs["privileged_ego"].shape) == (1, 1, 8)


def test_policy_runner_loads_checkpoints_and_returns_logits(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    runner = load_stage2_policy_runner(config)

    output = runner.act(
        {
            "privileged_bev": torch.zeros(1, 1, 43, 16, 16),
            "privileged_ego": torch.zeros(1, 1, 8),
        },
        {},
    )
    runner.observe_action(7)
    second = runner.act(
        {
            "privileged_bev": torch.zeros(1, 1, 43, 16, 16),
            "privileged_ego": torch.zeros(1, 1, 8),
        },
        {},
    )

    assert tuple(output["logits"].shape) == (1, 1, 39)
    assert tuple(second["logits"].shape) == (1, 1, 39)
    assert torch.isfinite(second["logits"]).all()


def test_policy_runner_rejects_bad_checkpoint_sha(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    config["checkpoints"]["policy"]["sha256"] = "0" * 64

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.done_reason == "checkpoint_sha256_mismatch"


def test_policy_runner_requires_checkpoint_sha(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    config["checkpoints"]["policy"]["sha256"] = None

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "policy_checkpoint_sha256_missing"


def test_policy_runner_rejects_missing_payload_config(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    policy_path = save_checkpoint(
        tmp_path / "policy_without_config.pt",
        {
            "model_state_dict": PrivilegedPolicy(
                PrivilegedPolicyConfig(latent_dim=160, hidden_dim=32)
            ).state_dict(),
        },
    )
    config["checkpoints"]["policy"]["path"] = str(policy_path)
    config["checkpoints"]["policy"]["sha256"] = _sha256(policy_path)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "policy_config_missing"


def test_policy_runner_rejects_missing_state_dict(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    policy_path = save_checkpoint(
        tmp_path / "policy_without_state.pt",
        {
            "config": {"latent_dim": 160, "hidden_dim": 32},
        },
    )
    config["checkpoints"]["policy"]["path"] = str(policy_path)
    config["checkpoints"]["policy"]["sha256"] = _sha256(policy_path)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "policy_state_dict_missing"


def test_policy_runner_rejects_unreadable_checkpoint_path(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    config["checkpoints"]["world_model"]["path"] = str(tmp_path)
    config["checkpoints"]["world_model"]["sha256"] = "0" * 64

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "checkpoint_file_unreadable"


def test_policy_runner_rejects_invalid_policy_config(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    policy_path = save_checkpoint(
        tmp_path / "policy_invalid_config.pt",
        {
            "model_state_dict": {},
            "config": ["not", "a", "mapping"],
        },
    )
    config["checkpoints"]["policy"]["path"] = str(policy_path)
    config["checkpoints"]["policy"]["sha256"] = _sha256(policy_path)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "policy_config_invalid"


def test_policy_runner_rejects_policy_latent_dim_mismatch(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    policy_cfg = PrivilegedPolicyConfig(latent_dim=999, hidden_dim=32)
    policy = PrivilegedPolicy(policy_cfg)
    policy_path = save_checkpoint(
        tmp_path / "policy_latent_mismatch.pt",
        {
            "model_state_dict": policy.state_dict(),
            "config": policy_cfg.__dict__,
        },
    )
    config["checkpoints"]["policy"]["path"] = str(policy_path)
    config["checkpoints"]["policy"]["sha256"] = _sha256(policy_path)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "policy_latent_dim_mismatch"


def test_policy_runner_rejects_bad_policy_state_dict_load(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    wm_cfg = PrivilegedWorldModelConfig(bev_height=16, bev_width=16, ego_dim=8, hidden_dim=32)
    world_model = PrivilegedWorldModel(wm_cfg)
    policy_cfg = PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=32)
    policy_path = save_checkpoint(
        tmp_path / "policy_bad_state_dict.pt",
        {
            "model_state_dict": {"unexpected.weight": torch.zeros(1)},
            "config": policy_cfg.__dict__,
        },
    )
    config["checkpoints"]["policy"]["path"] = str(policy_path)
    config["checkpoints"]["policy"]["sha256"] = _sha256(policy_path)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "policy_state_dict_load_failed"


def test_policy_runner_rejects_world_model_dim_mismatch(tmp_path) -> None:
    config = _base_config(tmp_path)
    _write_stage2_checkpoints(tmp_path, config)
    bad_cfg = PrivilegedWorldModelConfig(bev_height=8, bev_width=8, ego_dim=8, hidden_dim=32)
    bad_world_model = PrivilegedWorldModel(bad_cfg)
    bad_world_model_path = save_checkpoint(
        tmp_path / "world_model_8x8.pt",
        {
            "model_state_dict": bad_world_model.state_dict(),
            "config": bad_cfg.__dict__,
        },
    )
    config["checkpoints"]["world_model"]["path"] = str(bad_world_model_path)
    config["checkpoints"]["world_model"]["sha256"] = _sha256(bad_world_model_path)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        load_stage2_policy_runner(config)
    assert excinfo.value.failure_class == "model"
    assert excinfo.value.done_reason == "world_model_config_mismatch"


def test_stage2_adapter_bundle_runs_fake_raw_env_through_clean_loop(tmp_path) -> None:
    config = _base_config(tmp_path, horizon=2)
    config["carla"]["env_action_format"] = "raw2drive_control_dict"
    _write_stage2_checkpoints(tmp_path, config)
    raw_env = FakeRawEnv()
    env, policy = create_stage2_adapter_bundle(config=config, raw_env=raw_env)
    config_path = _write_config(tmp_path, config)

    summary = run_clean_eval_loop(
        env=env,
        policy=policy,
        config=config,
        config_path=config_path,
        metadata={
            "commit": "abc123",
            "branch": "codex/rcs-ac-v2-raw2drive",
            "windows_command": "powershell -File run_stage2_clean_eval.ps1",
            "carla_version": "0.9.16",
        },
    )

    assert summary.status == "PASS"
    assert summary.clean_gate_pass
    assert raw_env.closed
    assert raw_env.actions
    assert isinstance(raw_env.actions[0], dict)
    data = json.loads(Path(config["summary_path"]).read_text())
    assert data["metrics_path"] == config["metrics_path"]


def test_direct_3d_adapter_rejects_reverse_without_silent_loss(tmp_path) -> None:
    config = _base_config(tmp_path)
    config["carla"]["env_action_format"] = "direct_3d"
    config["carla"]["allow_reverse_direct_3d"] = False
    adapter = SharedCoreCarlaEnvAdapter(FakeRawEnv(), config)

    with pytest.raises(CleanEvalRuntimeFailure) as excinfo:
        adapter.step({"steer": 0.0, "throttle": 0.5, "brake": 0.0, "reverse": True})
    assert excinfo.value.done_reason == "reverse_action_not_supported_by_direct_3d"


def test_stage4_v2_adapter_collapses_raw2drive_control_for_v11_shared_core(tmp_path) -> None:
    config = _base_config(tmp_path)
    config["carla"]["env_action_format"] = "stage4_v2"
    config["carla"]["allow_reverse_direct_3d"] = False
    adapter = SharedCoreCarlaEnvAdapter(FakeRawEnv(), config)

    env_action = adapter._control_to_env_action(
        {"steer": -0.25, "throttle": 0.6, "brake": 0.2, "reverse": False}
    )

    assert tuple(env_action.shape) == (2,)
    assert env_action.tolist() == pytest.approx([-0.25, 0.4])


def test_privileged_obs_probe_is_not_clean_gate_evidence(tmp_path) -> None:
    config = _base_config(tmp_path)
    config["carla"]["env_action_format"] = "stage4_v2"
    raw_env = FakeRawEnv()

    result = probe_stage2_env_once(raw_env=raw_env, config=config, step_neutral=True)
    output_path = write_probe_result(tmp_path / "privileged_obs_probe.json", result)
    data = json.loads(output_path.read_text())

    assert data["status"] == "PROBE_PASS"
    assert data["clean_gate_evidence"] is False
    assert data["source"] == "windows_carla_privileged_obs_probe"
    assert data["privileged_bev_shape"] == [1, 1, 43, 16, 16]
    assert data["privileged_ego_shape"] == [1, 1, 8]
    assert data["privileged_bev_finite"] is True
    assert data["privileged_ego_finite"] is True
    assert data["step_neutral"] is True
    assert data["neutral_step_done"] is False
    assert raw_env.closed


def test_privileged_obs_probe_records_raw_keys_on_adapter_failure(tmp_path) -> None:
    config = _base_config(tmp_path)
    raw_env = FakeRawEnvWithoutPrivilegedObs()

    result = probe_stage2_env_once(raw_env=raw_env, config=config)

    assert result["status"] == "PROBE_FAIL"
    assert result["failure_class"] == "adapter"
    assert result["done_reason"] == "privileged_bev_missing"
    assert result["raw_reset_type"] == "dict"
    assert result["raw_reset_keys"] == ["info", "obs"]
    assert result["raw_obs_keys"] == ["rgb", "speed_kmh"]
    assert result["raw_obs_summary"]["rgb"]["shape"] == [3, 64, 64]
    assert raw_env.closed


def test_run_carla_with_default_factory_classifies_missing_checkpoint_as_model_failure(tmp_path) -> None:
    config = _base_config(tmp_path)
    config_path = _write_config(tmp_path, config)
    summary = run_real_carla_eval(
        config_path,
        metadata={
            "commit": "abc123",
            "branch": "codex/rcs-ac-v2-raw2drive",
            "windows_command": "powershell -File run_stage2_clean_eval.ps1",
            "carla_version": "0.9.16",
        },
    )

    assert summary.status == "FAIL"
    assert summary.failure_class == "model"
    assert summary.done_reason == "world_model_checkpoint_missing"
    assert Path(summary.metrics_path).exists()
