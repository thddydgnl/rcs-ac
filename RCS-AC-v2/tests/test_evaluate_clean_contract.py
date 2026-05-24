from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from rcs_ac_v2.env.discrete_actions import get_discrete_action, to_carla_control
from rcs_ac_v2.evaluation.evaluate_clean import (
    CleanEvalSummary,
    load_clean_eval_config,
    main,
    run_clean_eval_loop,
    run_dry_run,
    run_real_carla_eval,
    summary_from_dict,
    summary_to_dict,
    validate_clean_summary,
    write_step_metrics,
)
from rcs_ac_v2.evaluation.privileged_observation import (
    EGO8_SCHEMA_VERSION,
    PRIVILEGED_OBSERVATION_SOURCE,
    SEMANTIC43_CHANNEL_NAMES_SHA256,
    SEMANTIC43_SCHEMA_VERSION,
)


def _test_config(tmp_path: Path, *, horizon: int = 3) -> dict:
    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source_config.read_text())
    output_dir = tmp_path / "windows_128"
    config["horizon"] = horizon
    config["gate"]["pass_criteria"]["steps"] = horizon
    config["output_dir"] = str(output_dir)
    config["summary_path"] = str(output_dir / "summary.json")
    config["metrics_path"] = str(output_dir / "metrics.csv")
    return config


def _write_temp_config(tmp_path: Path, *, horizon: int = 3) -> Path:
    config = _test_config(tmp_path, horizon=horizon)
    config_path = tmp_path / "clean_128.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _metadata() -> dict[str, str]:
    return {
        "commit": "abc123",
        "branch": "codex/rcs-ac-v2-raw2drive",
        "windows_command": "powershell -File run_stage2_clean_eval.ps1",
        "carla_version": "0.9.16",
    }


def _valid_step_rows(steps: int, *, action_id: int = 37, collision_step: int | None = None) -> list[dict]:
    control = to_carla_control(get_discrete_action(action_id))
    rows: list[dict] = []
    for step in range(steps):
        collision = collision_step == step
        rows.append(
            {
                "step": step,
                "action_id": action_id,
                "throttle": control["throttle"],
                "brake": control["brake"],
                "steer": control["steer"],
                "reverse": control["reverse"],
                "reward_total": 0.1,
                "reward_route_progress": 0.01,
                "reward_collision": -1.0 if collision else 0.0,
                "reward_offroad": 0.0,
                "reward_stuck": 0.0,
                "continue": 1.0,
                "route_progress": float(step + 1) * 0.01,
                "collision": collision,
                "offroad": False,
                "stuck": False,
                "speed_kmh": 15.0,
                "done": False,
                "truncated": False,
                "adapter_error": "",
                "runtime_error": "",
                "done_reason": "running",
                "failure_class": "none",
                "policy_logits_finite": True,
                "privileged_bev_shape": "(1, 1, 43, 16, 16)",
                "privileged_ego_shape": "(1, 1, 8)",
                "obs_finite": True,
                "action_valid": True,
                "control_valid": True,
                "reward_finite": True,
                "route_progress_delta": 0.01,
            }
        )
    return rows


class FakePolicy:
    def __init__(self, *, action_id: int = 37, non_finite_logits: bool = False) -> None:
        self.action_id = action_id
        self.non_finite_logits = non_finite_logits

    def act(self, obs: dict, info: dict) -> dict:
        del obs, info
        logits = torch.zeros(1, 1, 39)
        logits[0, 0, self.action_id] = 10.0
        if self.non_finite_logits:
            logits[0, 0, 0] = float("nan")
        return {"logits": logits}


class FakeEnv:
    def __init__(
        self,
        *,
        done_step: int | None = None,
        collision_step: int | None = None,
        continue_value: float | None = None,
        missing_privileged_obs: bool = False,
        privileged_clean_gate_eligible: bool = True,
    ) -> None:
        self.step_count = 0
        self.done_step = done_step
        self.collision_step = collision_step
        self.continue_value = continue_value
        self.missing_privileged_obs = missing_privileged_obs
        self.privileged_clean_gate_eligible = privileged_clean_gate_eligible
        self.received_controls: list[dict] = []
        self.closed = False

    def reset(self) -> dict:
        return {"obs": self._obs(), "info": {"route_progress": 0.0, "speed_kmh": 0.0, **self._privileged_info()}}

    def step(self, control: dict, policy_diagnostics: dict | None = None) -> dict:
        del policy_diagnostics
        self.received_controls.append(dict(control))
        self.step_count += 1
        collision = self.collision_step == self.step_count
        done = self.done_step == self.step_count or collision
        info = {
            "route_progress": float(self.step_count) * 0.01,
            "speed_kmh": 15.0,
            "collision": collision,
            "offroad": False,
            "stuck": False,
            "done_reason": "collision" if collision else ("done" if done else "running"),
            "reward_route_progress": 0.01,
            "reward_collision": -1.0 if collision else 0.0,
            "reward_offroad": 0.0,
            "reward_stuck": 0.0,
            **self._privileged_info(),
        }
        if self.continue_value is not None:
            info["continue"] = self.continue_value
        return {
            "obs": self._obs(),
            "reward": 0.1,
            "done": done,
            "truncated": False,
            "info": info,
        }

    def close(self) -> None:
        self.closed = True

    def _obs(self) -> dict:
        if self.missing_privileged_obs:
            return {}
        return {
            "privileged_bev": torch.zeros(1, 1, 43, 16, 16),
            "privileged_ego": torch.zeros(1, 1, 8),
        }

    def _privileged_info(self) -> dict:
        return _privileged_info(clean_gate_eligible=self.privileged_clean_gate_eligible)


def _privileged_info(*, clean_gate_eligible: bool) -> dict:
    return {
        "privileged_observation_clean_gate_eligible": clean_gate_eligible,
        "privileged_observation_source": PRIVILEGED_OBSERVATION_SOURCE,
        "privileged_observation_schema": SEMANTIC43_SCHEMA_VERSION,
        "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
        "privileged_ego_schema": EGO8_SCHEMA_VERSION,
    }

class FakeEnvWithDerivedDebugPrivileged(FakeEnv):
    @staticmethod
    def _privileged_info() -> dict:
        return {
            "privileged_observation_clean_gate_eligible": False,
            "privileged_observation_source": "derived_debug_from_raw_v1",
            "privileged_observation_schema": "derived_debug_from_raw_v1",
            "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
            "privileged_ego_schema": EGO8_SCHEMA_VERSION,
        }


def test_clean_128_config_loads_and_pins_contracts() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = load_clean_eval_config(config_path)

    assert config["stage"] == "stage2_privileged_policy_clean"
    assert config["mode"] == "windows_carla_clean_128"
    assert config["horizon"] == 128
    assert config["town"] == "Town04"
    assert config["weather"] == "ClearNoon"
    assert config["fixed_delta_seconds"] == 0.05
    assert config["synchronous_mode"] is True
    assert config["action_table_version"] == "raw2drive_table12_row_major_v1"
    assert config["reward_contract_version"] == "stage0_route_progress_penalty_v1"
    assert "none" in config["failure_classes"]
    assert config["gate"]["source_required_for_pass"] == "windows_carla"
    assert config["carla"]["privileged_observation"]["mode"] == "simulator_state_semantic43"
    assert config["observation_contract"]["require_true_privileged_for_clean_gate"] is True
    assert config["observation_contract"]["privileged_observation_source"] == PRIVILEGED_OBSERVATION_SOURCE
    assert config["observation_contract"]["privileged_bev_schema_version"] == SEMANTIC43_SCHEMA_VERSION
    assert config["observation_contract"]["privileged_bev_channel_names_sha256"] == SEMANTIC43_CHANNEL_NAMES_SHA256
    assert config["checkpoint_contract"]["require_trained_provenance_for_clean_gate"] is True
    assert config["checkpoint_contract"]["allowed_sources"] == ["windows_carla_privileged_replay"]
    assert "action_id" in config["per_step_log_fields"]
    assert "reward_total" in config["per_step_log_fields"]


def test_dry_run_writes_local_smoke_only_summary(tmp_path) -> None:
    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["output_dir"] = str(tmp_path / "dry_run")
    config_path = tmp_path / "clean_128.yaml"
    config_path.write_text(yaml.safe_dump(config))

    summary_path = run_dry_run(config_path)
    data = json.loads(summary_path.read_text())
    summary = summary_from_dict(data)

    assert summary.status == "LOCAL_SMOKE_ONLY"
    assert summary.source == "local_smoke_not_carla"
    assert not summary.clean_gate_pass
    assert summary.horizon == 128
    assert data["config"] == str(config_path)
    validate_clean_summary(summary)


def test_pass_requires_windows_carla_source_and_metadata() -> None:
    local_pass = CleanEvalSummary(
        horizon=128,
        steps=128,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.5,
        source="local_smoke_not_carla",
        status="PASS",
    )
    with pytest.raises(ValueError, match="windows_carla"):
        validate_clean_summary(local_pass)

    missing_metadata = CleanEvalSummary(
        horizon=128,
        steps=128,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.5,
        source="windows_carla",
        status="PASS",
        done_reason="horizon_reached",
    )
    with pytest.raises(ValueError, match="missing official metadata"):
        validate_clean_summary(missing_metadata)


def test_windows_carla_pass_rejects_safety_failures_and_partial_steps(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=128)
    metrics_path = write_step_metrics(config["metrics_path"], _valid_step_rows(128), config)
    base = {
        "horizon": 128,
        "steps": 128,
        "collision_count": 0,
        "offroad_count": 0,
        "stuck_count": 0,
        "route_progress": 1.28,
        "episode_return": 12.8,
        "source": "windows_carla",
        "status": "PASS",
        "failure_class": "none",
        "done_reason": "horizon_reached",
        "commit": "abc123",
        "branch": "codex/rcs-ac-v2-raw2drive",
        "windows_command": "powershell -File run_stage2_clean_eval.ps1",
        "config": "RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml",
        "carla_version": "0.9.16",
        "carla_town": "Town04",
        "output_path": "RCS-AC-v2/experiments/02_privileged_policy_clean/windows_128",
        "metrics_path": str(metrics_path),
        "pulled_files": ("summary.json", "metrics.csv"),
        "next_action": "Run 512 primary subset.",
    }
    with pytest.raises(ValueError, match="zero collision|collision count"):
        validate_clean_summary(CleanEvalSummary(**{**base, "collision_count": 1}))
    partial_metrics_path = write_step_metrics(
        tmp_path / "partial" / "metrics.csv",
        _valid_step_rows(127),
        {**config, "metrics_path": str(tmp_path / "partial" / "metrics.csv")},
    )
    with pytest.raises(ValueError, match="row count|steps"):
        validate_clean_summary(CleanEvalSummary(**{**base, "steps": 127, "metrics_path": str(partial_metrics_path)}))
    with pytest.raises(ValueError, match="pulled_files"):
        validate_clean_summary(CleanEvalSummary(**{**base, "pulled_files": ()}))
    with pytest.raises(ValueError, match="failure_class none"):
        validate_clean_summary(CleanEvalSummary(**{**base, "failure_class": "driving"}))
    with pytest.raises(ValueError, match="done_reason"):
        validate_clean_summary(CleanEvalSummary(**{**base, "done_reason": "ended_before_horizon"}))


def test_summary_rejects_non_finite_metrics_and_bad_failure_class() -> None:
    base = {
        "horizon": 128,
        "steps": 10,
        "collision_count": 0,
        "offroad_count": 0,
        "stuck_count": 0,
        "route_progress": 0.0,
        "source": "windows_carla",
        "status": "FAIL",
        "failure_class": "model",
    }
    with pytest.raises(ValueError, match="route_progress"):
        validate_clean_summary(CleanEvalSummary(**{**base, "route_progress": float("nan")}))
    with pytest.raises(ValueError, match="episode_return"):
        validate_clean_summary(CleanEvalSummary(**{**base, "episode_return": float("inf")}))
    with pytest.raises(ValueError, match="failure_class"):
        validate_clean_summary(CleanEvalSummary(**{**base, "failure_class": "mystery"}))


def test_windows_carla_pass_summary_roundtrip_validates(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=128)
    metrics_path = write_step_metrics(config["metrics_path"], _valid_step_rows(128), config)
    summary = CleanEvalSummary(
        horizon=128,
        steps=128,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=1.28,
        episode_return=12.8,
        source="windows_carla",
        status="PASS",
        failure_class="none",
        done_reason="horizon_reached",
        commit="abc123",
        branch="codex/rcs-ac-v2-raw2drive",
        windows_command="powershell -File run_stage2_clean_eval.ps1",
        config="RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml",
        carla_version="0.9.16",
        carla_town="Town04",
        output_path=str(Path(config["output_dir"])),
        metrics_path=str(metrics_path),
        pulled_files=("summary.json", "metrics.csv"),
        next_action="Run 512 primary subset.",
    )
    data = summary_to_dict(summary)
    restored = summary_from_dict(data)
    validate_clean_summary(restored)
    assert restored.clean_gate_pass


def test_clean_gate_pass_property_requires_full_gate_conditions() -> None:
    summary = CleanEvalSummary(
        horizon=128,
        steps=127,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.7,
        source="windows_carla",
        status="PASS",
        failure_class="none",
        done_reason="horizon_reached",
    )
    assert not summary.clean_gate_pass
    assert not CleanEvalSummary(
        horizon=128,
        steps=128,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.7,
        source="windows_carla",
        status="PASS",
        failure_class="driving",
        done_reason="horizon_reached",
    ).clean_gate_pass
    assert not CleanEvalSummary(
        horizon=128,
        steps=128,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.7,
        source="windows_carla",
        status="PASS",
        failure_class="none",
        done_reason="ended_before_horizon",
    ).clean_gate_pass


def test_windows_runner_dry_run_skips_carla_guard_static() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "run_stage2_clean_eval.ps1"
    text = runner.read_text()
    dry_run_index = text.index("if ($DryRun)")
    dependency_index = text.index("check_stage2_windows_dependencies.ps1")
    guard_index = text.index("-File $EnsureCarlaScript")
    assert dry_run_index < dependency_index
    assert dry_run_index < guard_index
    assert "--run-carla" in text
    assert "not implemented" not in text.lower()
    assert "goal_achieved_round_0002_execution\\support_code" in text
    assert "$WorkspaceRoot" in text
    assert "$SharedCorePath" in text
    assert "RCS-AC-v2\\scripts\\windows\\ensure_carla_ready.ps1" in text
    assert "RCS-AC-v2\\scripts\\windows\\start_carla_interactive.ps1" in text
    assert "$CheckDependenciesOnly" in text


def test_windows_runner_can_mirror_validated_artifacts_static() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "run_stage2_clean_eval.ps1"
    text = runner.read_text()
    validation_index = text.index("--validate-summary $SummaryPath")
    mirror_index = text.index("STAGE2_RUN_OUTPUT_DIR=$ResolvedRunOutputDir")
    assert "[string]$RunOutputDir" in text
    assert "Copy-Stage2RunArtifact" in text
    assert "ARTIFACT_COPY_SKIPPED_SAME_PATH" in text
    assert "Remove-Item $SummaryPath, $MetricsPath -ErrorAction SilentlyContinue" in text
    assert 'Join-Path $ResolvedRunOutputDir "summary.json"' in text
    assert 'Join-Path $ResolvedRunOutputDir "metrics.csv"' in text
    assert "Split-Path -Leaf $ConfigPath" in text
    assert "$WindowsCommand = \"$WindowsCommand -RunOutputDir $RunOutputDir\"" in text
    assert "--print-output-paths" in text
    assert "SUMMARY_PATH=*" in text
    assert "METRICS_PATH=*" in text
    assert '$SummaryPath = ([string]$SummaryPathLine).Substring("SUMMARY_PATH=".Length)' in text
    assert '$MetricsPath = ([string]$MetricsPathLine).Substring("METRICS_PATH=".Length)' in text
    assert '$SummaryPath = [string]$ConfiguredPaths[0]' not in text
    assert '$MetricsPath = [string]$ConfiguredPaths[1]' not in text
    assert '$SummaryPath = "RCS-AC-v2\\experiments\\02_privileged_policy_clean\\windows_128\\summary.json"' not in text
    assert validation_index < mirror_index


def test_detached_replay_training_wrapper_static_contract() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "start_stage2_replay_training_detached.ps1"
    text = script.read_text()
    assert "train_stage2_privileged_from_replay.ps1" in text
    assert "train_stage2_privileged_from_replay.detached.ps1" in text
    assert "$ResolvedOutputDir = (Resolve-Path $OutputDir).Path" in text
    assert "$ResolvedLogPath = Join-Path $ResolvedLogPath (Split-Path -Leaf $LogPath)" in text
    assert "New-ScheduledTaskAction" in text
    assert "New-ScheduledTaskPrincipal" in text
    assert "Start-ScheduledTask" in text
    assert "ImitationClassBalance" in text
    assert "TRAIN_LOG_PATH=$ResolvedLogPath" in text
    assert "TRAIN_SCRIPT_PATH=$InnerScriptPath" in text
    assert "TRAIN_TASK_NAME=$TaskName" in text
    assert "TRAIN_TASK_STARTED=True" in text
    assert "DETACHED_TRAIN_STARTED=`$(Get-Date -Format o)" in text
    assert "TRAIN_EXCEPTION=" in text
    assert "TRAIN_EXIT_CODE=`$Code" in text
    assert "[switch]$NoStart" in text
    assert "TRAIN_DETACHED_DRY_RUN=True" in text


def test_detached_replay_training_check_static_contract() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "check_stage2_replay_training_detached.ps1"
    text = script.read_text()
    assert "stage2_privileged_replay_checkpoint_manifest.json" in text
    assert "TRAIN_LOG_EXISTS=$LogExists" in text
    assert "TRAIN_MANIFEST_EXISTS=$ManifestExists" in text
    assert "TRAIN_RUNNING=$Running" in text
    assert "TRAIN_TASK_NAME=$TaskName" in text
    assert "TRAIN_TASK_STATE=$TaskState" in text
    assert '[System.IO.File]::ReadAllText((Resolve-Path $LogPath).Path) -replace "`0", ""' in text
    assert "TRAIN_EXIT_CODE=$ExitCode" in text
    assert "TRAIN_READY_FOR_PROBE=$ReadyForProbe" in text
    assert "TRAIN_LOG_TAIL_BEGIN" in text
    assert "exit 3" in text
    assert "exit 2" in text


def test_windows_dependency_probe_static_contract() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "check_stage2_windows_dependencies.ps1"
    text = script.read_text()
    assert "STAGE2_DEPENDENCY_$($entry.Name)" in text
    assert "V2_CODE" in text
    assert "V11_SUPPORT_CODE" in text
    assert "V11_CODE" in text
    assert "ENV_CONFIG" in text
    assert "SHARED_CORE" in text
    assert "STAGE2_WINDOWS_DEPENDENCIES_OK=False" in text
    assert "STAGE2_WINDOWS_DEPENDENCIES_OK=True" in text
    assert "rcs_ac_v2.evaluation.carla_adapter" in text
    assert "shared_core.env.carla_env" in text
    assert "SKIPPED:not configured" in text
    assert "exit 2" in text
    assert "exit 3" in text


def test_v2_local_carla_readiness_scripts_are_tracked_contract() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts" / "windows"
    ensure = scripts_dir / "ensure_carla_ready.ps1"
    start = scripts_dir / "start_carla_interactive.ps1"
    assert ensure.exists()
    assert start.exists()
    ensure_text = ensure.read_text()
    start_text = start.read_text()
    assert "RCS-AC-v2\\scripts\\windows\\start_carla_interactive.ps1" in ensure_text
    assert "CARLA_READY=True" in ensure_text
    assert "RCSV2CarlaStart" in start_text
    assert "CarlaUE4.exe" in start_text


def test_stage2_privileged_obs_probe_runner_is_not_clean_eval_static() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts" / "windows" / "run_stage2_privileged_obs_probe.ps1"
    text = runner.read_text()
    assert "check_stage2_windows_dependencies.ps1" in text
    assert "ensure_carla_ready.ps1" in text
    assert "rcs_ac_v2.evaluation.probe_stage2_windows_env" in text
    assert "privileged_obs_probe.json" in text
    assert "evaluate_clean" not in text
    assert "summary.json" not in text
    assert "clean PASS" not in text


def test_fake_adapter_loop_can_produce_windows_carla_pass(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    env = FakeEnv()
    summary = run_clean_eval_loop(
        env=env,
        policy=FakePolicy(action_id=37),
        config=config,
        config_path=tmp_path / "clean_128.yaml",
        metadata=_metadata(),
    )

    assert summary.status == "PASS"
    assert summary.clean_gate_pass
    assert summary.steps == 3
    assert summary.failure_class == "none"
    assert env.closed
    assert env.received_controls[0] == to_carla_control(get_discrete_action(37))

    data = json.loads(Path(config["summary_path"]).read_text())
    validate_clean_summary(summary_from_dict(data))
    metrics_lines = Path(config["metrics_path"]).read_text().strip().splitlines()
    assert len(metrics_lines) == 4


def test_fake_adapter_loop_512_pass_points_to_2048_next(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=512)
    env = FakeEnv()
    summary = run_clean_eval_loop(
        env=env,
        policy=FakePolicy(action_id=37),
        config=config,
        config_path=tmp_path / "clean_512.yaml",
        metadata=_metadata(),
    )

    assert summary.status == "PASS"
    assert summary.clean_gate_pass
    assert summary.horizon == 512
    assert summary.steps == 512
    assert summary.next_action == "Run 2048-step subset."
    validate_clean_summary(summary)


def test_fake_adapter_loop_classifies_collision_as_driving_failure(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    env = FakeEnv(collision_step=1)
    summary = run_clean_eval_loop(
        env=env,
        policy=FakePolicy(action_id=37),
        config=config,
        config_path=tmp_path / "clean_128.yaml",
        metadata=_metadata(),
    )

    assert summary.status == "FAIL"
    assert summary.failure_class == "driving"
    assert summary.collision_count == 1
    assert not summary.clean_gate_pass
    validate_clean_summary(summary)


def test_fake_adapter_loop_classifies_adapter_and_contract_failures(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    missing_obs_env = FakeEnv(missing_privileged_obs=True)
    adapter_summary = run_clean_eval_loop(
        env=missing_obs_env,
        policy=FakePolicy(action_id=37),
        config=config,
        config_path=tmp_path / "clean_128.yaml",
        metadata=_metadata(),
    )
    assert adapter_summary.status == "FAIL"
    assert adapter_summary.failure_class == "adapter"
    assert missing_obs_env.closed
    validate_clean_summary(adapter_summary)

    contract_config = _test_config(tmp_path / "contract", horizon=3)
    bad_continue_env = FakeEnv(continue_value=0.5)
    contract_summary = run_clean_eval_loop(
        env=bad_continue_env,
        policy=FakePolicy(action_id=37),
        config=contract_config,
        config_path=tmp_path / "contract" / "clean_128.yaml",
        metadata=_metadata(),
    )
    assert contract_summary.status == "FAIL"
    assert contract_summary.failure_class == "contract"
    validate_clean_summary(contract_summary)


def test_clean_loop_rejects_derived_debug_privileged_obs(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    env = FakeEnvWithDerivedDebugPrivileged()
    summary = run_clean_eval_loop(
        env=env,
        policy=FakePolicy(action_id=37),
        config=config,
        config_path=tmp_path / "clean_128.yaml",
        metadata=_metadata(),
    )

    assert summary.status == "FAIL"
    assert summary.failure_class == "adapter"
    assert summary.done_reason == "privileged_observation_provenance_missing"
    assert not summary.clean_gate_pass
    assert env.closed


def test_fake_adapter_loop_classifies_non_finite_logits_as_model_failure(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    env = FakeEnv()
    summary = run_clean_eval_loop(
        env=env,
        policy=FakePolicy(action_id=37, non_finite_logits=True),
        config=config,
        config_path=tmp_path / "clean_128.yaml",
        metadata=_metadata(),
    )

    assert summary.status == "FAIL"
    assert summary.failure_class == "model"
    assert env.closed
    validate_clean_summary(summary)


def test_run_carla_without_adapter_writes_adapter_failure_summary(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    config["adapter_factory"] = None
    config_path = tmp_path / "clean_128.yaml"
    config_path.write_text(yaml.safe_dump(config))
    summary = run_real_carla_eval(config_path, metadata=_metadata())

    assert summary.status == "FAIL"
    assert summary.failure_class == "adapter"
    assert summary.done_reason == "adapter_factory_missing"
    assert Path(summary.output_path, "summary.json").exists()
    assert Path(summary.metrics_path).exists()
    assert main(
        [
            "--validate-summary",
            str(Path(summary.output_path, "summary.json")),
        ]
    ) == 0


def test_metrics_csv_rejects_summary_only_or_short_pass(tmp_path) -> None:
    config = _test_config(tmp_path, horizon=3)
    metrics_path = write_step_metrics(config["metrics_path"], _valid_step_rows(2), config)
    summary = CleanEvalSummary(
        horizon=3,
        steps=2,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.02,
        episode_return=0.2,
        source="windows_carla",
        status="PASS",
        failure_class="none",
        done_reason="horizon_reached",
        commit="abc123",
        branch="codex/rcs-ac-v2-raw2drive",
        windows_command="powershell -File run_stage2_clean_eval.ps1",
        config="RCS-AC-v2/configs/eval/clean_128_stage2_privileged.yaml",
        carla_version="0.9.16",
        carla_town="Town04",
        output_path=str(Path(config["output_dir"])),
        metrics_path=str(metrics_path),
        pulled_files=("summary.json", "metrics.csv"),
        next_action="Run 512 primary subset.",
    )
    with pytest.raises(ValueError, match="row count|steps"):
        validate_clean_summary(summary)


def test_cli_dry_run_and_validate_summary(tmp_path, capsys) -> None:
    source_config = Path(__file__).resolve().parents[1] / "configs" / "eval" / "clean_128_stage2_privileged.yaml"
    config = yaml.safe_load(source_config.read_text())
    config["output_dir"] = str(tmp_path / "dry_run")
    config_path = tmp_path / "clean_128.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert main(["--config", str(config_path), "--dry-run"]) == 0
    summary_path = tmp_path / "dry_run" / "summary.json"
    assert summary_path.exists()
    assert main(["--validate-summary", str(summary_path)]) == 0
    captured = capsys.readouterr()
    assert "DRY_RUN_SUMMARY=" in captured.out
    assert "SUMMARY_VALID=" in captured.out


def test_cli_print_output_paths_uses_configured_paths(tmp_path, capsys) -> None:
    config_path = _write_temp_config(tmp_path, horizon=512)

    assert main(["--config", str(config_path), "--print-output-paths"]) == 0
    captured = capsys.readouterr()
    assert f"SUMMARY_PATH={tmp_path / 'windows_128' / 'summary.json'}" in captured.out
    assert f"METRICS_PATH={tmp_path / 'windows_128' / 'metrics.csv'}" in captured.out
