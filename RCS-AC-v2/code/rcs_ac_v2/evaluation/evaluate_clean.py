"""Clean-eval contracts, adapter loop, and CLI entry points.

Real closed-loop clean evaluation must run on Windows CARLA through the remote
workflow. Local tests exercise the adapter-loop contract with fake envs so the
Windows run cannot accidentally promote local smoke or summary-only evidence to
clean PASS.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from rcs_ac_v2.env.discrete_actions import (
    ACTION_TABLE_SHA256,
    ACTION_TABLE_VERSION,
    NUM_RAW2DRIVE_ACTIONS,
    get_discrete_action,
    to_carla_control,
)
from rcs_ac_v2.evaluation.errors import CleanEvalRuntimeFailure


ALLOWED_FAILURE_CLASSES = {
    "infrastructure",
    "CARLA",
    "adapter",
    "contract",
    "model",
    "driving",
    "none",
}
PASS_STATUSES = {"PASS"}
REQUIRED_STEP_LOG_FIELDS = (
    "step",
    "action_id",
    "throttle",
    "brake",
    "steer",
    "reverse",
    "reward_total",
    "reward_route_progress",
    "reward_collision",
    "reward_offroad",
    "reward_stuck",
    "continue",
    "route_progress",
    "collision",
    "offroad",
    "stuck",
    "speed_kmh",
    "done",
    "truncated",
    "adapter_error",
    "runtime_error",
)
EXTRA_STEP_LOG_FIELDS = (
    "done_reason",
    "failure_class",
    "policy_logits_finite",
    "privileged_bev_shape",
    "privileged_ego_shape",
    "obs_finite",
    "action_valid",
    "control_valid",
    "reward_finite",
    "route_progress_delta",
)


def _metadata_value(metadata: dict[str, Any], name: str, default: str = "UNSET") -> str:
    value = metadata.get(name, default)
    return default if value is None or str(value) == "" else str(value)


@dataclass(frozen=True)
class CleanEvalSummary:
    horizon: int
    steps: int
    collision_count: int
    offroad_count: int
    stuck_count: int
    route_progress: float
    source: str
    stage: str = "stage2_privileged_policy_clean"
    status: str = "PENDING"
    failure_class: str = "none"
    done_reason: str = "not_run"
    episode_return: float = 0.0
    commit: str = "UNSET"
    branch: str = "UNSET"
    windows_command: str = "UNSET"
    config: str = "UNSET"
    carla_version: str = "UNSET"
    carla_town: str = "UNSET"
    output_path: str = "UNSET"
    metrics_path: str = "UNSET"
    pulled_files: tuple[str, ...] = ()
    next_action: str = "UNSET"
    failure_message: str = ""
    action_table_version: str = ACTION_TABLE_VERSION
    action_table_sha256: str = ACTION_TABLE_SHA256
    reward_contract_version: str = "stage0_route_progress_penalty_v1"

    @property
    def safety_pass(self) -> bool:
        return self.collision_count == 0 and self.offroad_count == 0 and self.stuck_count == 0

    @property
    def is_carla_evidence(self) -> bool:
        return self.source == "windows_carla"

    @property
    def clean_gate_pass(self) -> bool:
        return (
            self.is_carla_evidence
            and self.status == "PASS"
            and self.steps == self.horizon
            and self.done_reason == "horizon_reached"
            and self.failure_class == "none"
            and self.safety_pass
        )


def make_local_smoke_summary(*, horizon: int, route_progress: float = 0.0) -> CleanEvalSummary:
    return CleanEvalSummary(
        horizon=horizon,
        steps=0,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=route_progress,
        source="local_smoke_not_carla",
        status="LOCAL_SMOKE_ONLY",
    )


def summary_to_dict(summary: CleanEvalSummary) -> dict[str, Any]:
    return {
        "stage": summary.stage,
        "status": summary.status,
        "source": summary.source,
        "horizon": summary.horizon,
        "steps": summary.steps,
        "collision_count": summary.collision_count,
        "offroad_count": summary.offroad_count,
        "stuck_count": summary.stuck_count,
        "route_progress": summary.route_progress,
        "episode_return": summary.episode_return,
        "commit": summary.commit,
        "branch": summary.branch,
        "windows_command": summary.windows_command,
        "config": summary.config,
        "carla_version": summary.carla_version,
        "carla_town": summary.carla_town,
        "output_path": summary.output_path,
        "metrics_path": summary.metrics_path,
        "pulled_files": list(summary.pulled_files),
        "next_action": summary.next_action,
        "done_reason": summary.done_reason,
        "failure_class": summary.failure_class,
        "failure_message": summary.failure_message,
        "safety_pass": summary.safety_pass,
        "clean_gate_pass": summary.clean_gate_pass,
        "action_table_version": summary.action_table_version,
        "action_table_sha256": summary.action_table_sha256,
        "reward_contract_version": summary.reward_contract_version,
    }


def summary_from_dict(data: dict[str, Any]) -> CleanEvalSummary:
    return CleanEvalSummary(
        stage=str(data.get("stage", "stage2_privileged_policy_clean")),
        status=str(data["status"]),
        source=str(data["source"]),
        horizon=int(data["horizon"]),
        steps=int(data.get("steps", data["horizon"])),
        collision_count=int(data["collision_count"]),
        offroad_count=int(data["offroad_count"]),
        stuck_count=int(data["stuck_count"]),
        route_progress=float(data["route_progress"]),
        episode_return=float(data.get("episode_return", 0.0)),
        commit=str(data.get("commit", "UNSET")),
        branch=str(data.get("branch", "UNSET")),
        windows_command=str(data.get("windows_command", "UNSET")),
        config=str(data.get("config", "UNSET")),
        carla_version=str(data.get("carla_version", "UNSET")),
        carla_town=str(data.get("carla_town", "UNSET")),
        output_path=str(data.get("output_path", "UNSET")),
        metrics_path=str(data.get("metrics_path", "UNSET")),
        pulled_files=tuple(data.get("pulled_files", ())),
        next_action=str(data.get("next_action", "UNSET")),
        done_reason=str(data.get("done_reason", "unknown")),
        failure_class=str(data.get("failure_class", "none")),
        failure_message=str(data.get("failure_message", "")),
        action_table_version=str(data.get("action_table_version", ACTION_TABLE_VERSION)),
        action_table_sha256=str(data.get("action_table_sha256", ACTION_TABLE_SHA256)),
        reward_contract_version=str(
            data.get("reward_contract_version", "stage0_route_progress_penalty_v1")
        ),
    )


def validate_clean_summary(summary: CleanEvalSummary) -> None:
    if summary.horizon <= 0:
        raise ValueError("horizon must be positive")
    if summary.steps < 0:
        raise ValueError("steps must be non-negative")
    if summary.steps > summary.horizon:
        raise ValueError("steps cannot exceed horizon")
    if not math.isfinite(summary.route_progress):
        raise ValueError("route_progress must be finite")
    if not math.isfinite(summary.episode_return):
        raise ValueError("episode_return must be finite")
    if summary.failure_class not in ALLOWED_FAILURE_CLASSES:
        raise ValueError(f"failure_class must be one of {sorted(ALLOWED_FAILURE_CLASSES)}")
    for name in ("collision_count", "offroad_count", "stuck_count"):
        if getattr(summary, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if summary.action_table_version != ACTION_TABLE_VERSION:
        raise ValueError("action_table_version mismatch")
    if summary.action_table_sha256 != ACTION_TABLE_SHA256:
        raise ValueError("action_table_sha256 mismatch")
    if summary.clean_gate_pass and not summary.is_carla_evidence:
        raise ValueError("clean gate PASS requires windows_carla evidence")
    if summary.status == "PASS" and not summary.is_carla_evidence:
        raise ValueError("PASS status requires windows_carla source")
    if summary.is_carla_evidence and summary.metrics_path != "UNSET":
        validate_metrics_csv(summary.metrics_path, summary=summary)
    if summary.status == "PASS":
        if not summary.safety_pass:
            raise ValueError("PASS summary requires zero collision/offroad/stuck counts")
        if summary.steps < summary.horizon:
            raise ValueError("PASS summary requires steps to reach horizon")
        if summary.failure_class != "none":
            raise ValueError("PASS summary requires failure_class none")
        if summary.done_reason != "horizon_reached":
            raise ValueError("PASS summary requires done_reason horizon_reached")
        required_metadata = {
            "commit": summary.commit,
            "branch": summary.branch,
            "windows_command": summary.windows_command,
            "config": summary.config,
            "carla_version": summary.carla_version,
            "carla_town": summary.carla_town,
            "output_path": summary.output_path,
            "next_action": summary.next_action,
        }
        missing = sorted(name for name, value in required_metadata.items() if value == "UNSET")
        if missing:
            raise ValueError(f"PASS summary missing official metadata: {missing}")
        if not summary.pulled_files:
            raise ValueError("PASS summary requires pulled_files evidence")
        if summary.metrics_path == "UNSET":
            raise ValueError("PASS summary requires metrics_path evidence")


def load_clean_eval_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("clean eval config must be a mapping")
    required = {
        "stage",
        "mode",
        "horizon",
        "town",
        "seed",
        "weather",
        "fixed_delta_seconds",
        "synchronous_mode",
        "checkpoints",
        "action_table_version",
        "action_table_sha256",
        "reward_contract_version",
        "output_dir",
        "summary_path",
        "metrics_path",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    if config["action_table_version"] != ACTION_TABLE_VERSION:
        raise ValueError("config action_table_version mismatch")
    if config["action_table_sha256"] != ACTION_TABLE_SHA256:
        raise ValueError("config action_table_sha256 mismatch")
    if config["stage"] != "stage2_privileged_policy_clean":
        raise ValueError("config stage must be stage2_privileged_policy_clean")
    if config["mode"] != "windows_carla_clean_128":
        raise ValueError("config mode must be windows_carla_clean_128")
    if config.get("action_space") != "raw2drive_discrete_39":
        raise ValueError("config action_space must be raw2drive_discrete_39")
    if int(config["horizon"]) <= 0:
        raise ValueError("config horizon must be positive")
    if float(config["fixed_delta_seconds"]) <= 0:
        raise ValueError("config fixed_delta_seconds must be positive")
    if config["synchronous_mode"] is not True:
        raise ValueError("config synchronous_mode must be true")
    checkpoints = config.get("checkpoints", {})
    dry_run = config.get("dry_run", {})
    if not dry_run.get("allow_null_checkpoints", False):
        for key in ("world_model", "policy"):
            entry = checkpoints.get(key)
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
                raise ValueError(f"official config requires checkpoints.{key}.path and sha256")
    continue_contract = config.get("continue_contract", {})
    if continue_contract.get("terminal_done") != 0.0 or continue_contract.get("running") != 1.0:
        raise ValueError("config continue_contract mismatch")
    failure_classes = set(config.get("failure_classes", ()))
    required_failure_classes = {"infrastructure", "CARLA", "adapter", "contract", "model", "driving", "none"}
    if not required_failure_classes.issubset(failure_classes):
        raise ValueError("config failure_classes incomplete")
    gate = config.get("gate", {})
    if gate.get("full_clean_pass_requires_source") != "windows_carla":
        raise ValueError("config gate must require windows_carla source")
    if gate.get("source_required_for_pass") != "windows_carla":
        raise ValueError("config gate source_required_for_pass must be windows_carla")
    pass_criteria = gate.get("pass_criteria", {})
    if int(pass_criteria.get("steps", config["horizon"])) != int(config["horizon"]):
        raise ValueError("config gate pass_criteria.steps must equal horizon")
    return config


def validate_metrics_csv(
    path: str | Path,
    *,
    summary: CleanEvalSummary,
    required_fields: tuple[str, ...] = REQUIRED_STEP_LOG_FIELDS,
) -> None:
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise ValueError(f"metrics CSV does not exist: {metrics_path}")
    expected_from_config = _expected_metrics_fields(summary)
    expected_fields = expected_from_config or required_fields
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(set(expected_fields).difference(fieldnames))
        if missing:
            raise ValueError(f"metrics CSV missing required fields: {missing}")
        if summary.status == "PASS" and expected_from_config is not None and tuple(fieldnames) != tuple(expected_fields):
            raise ValueError("PASS metrics CSV header must exactly match config per_step_log_fields")
        rows = list(reader)

    if len(rows) != summary.steps:
        raise ValueError(f"metrics CSV row count {len(rows)} does not match summary steps {summary.steps}")
    if summary.status == "PASS" and len(rows) != summary.horizon:
        raise ValueError("PASS metrics CSV row count must equal horizon")

    collision_count = 0
    offroad_count = 0
    stuck_count = 0
    route_progress = summary.route_progress if not rows else 0.0
    episode_return = 0.0
    for expected_step, row in enumerate(rows):
        if int(row["step"]) != expected_step:
            raise ValueError("metrics CSV step column must be contiguous from zero")
        action_id = int(row["action_id"])
        if action_id < 0 or action_id >= NUM_RAW2DRIVE_ACTIONS:
            raise ValueError("metrics CSV contains invalid action_id")
        for name in (
            "throttle",
            "brake",
            "steer",
            "reward_total",
            "reward_route_progress",
            "reward_collision",
            "reward_offroad",
            "reward_stuck",
            "continue",
            "route_progress",
            "speed_kmh",
            "route_progress_delta",
        ):
            if name in row and not math.isfinite(float(row[name])):
                raise ValueError(f"metrics CSV field {name} must be finite")
        cont = float(row["continue"])
        if cont not in (0.0, 1.0):
            raise ValueError("metrics CSV continue field must be 0 or 1")
        collision_count += int(_truthy(row["collision"]))
        offroad_count += int(_truthy(row["offroad"]))
        stuck_count += int(_truthy(row["stuck"]))
        route_progress = float(row["route_progress"])
        episode_return += float(row["reward_total"])
        if summary.status == "PASS":
            if _truthy(row["collision"]) or _truthy(row["offroad"]) or _truthy(row["stuck"]):
                raise ValueError("PASS metrics CSV must have no collision/offroad/stuck rows")
            if _truthy(row["adapter_error"]) or _truthy(row["runtime_error"]):
                raise ValueError("PASS metrics CSV must have no adapter/runtime error rows")
            for name in (
                "policy_logits_finite",
                "obs_finite",
                "action_valid",
                "control_valid",
                "reward_finite",
            ):
                if name in row and not _truthy(row[name]):
                    raise ValueError(f"PASS metrics CSV requires {name}=true for every row")
            if row.get("failure_class", "none") != "none":
                raise ValueError("PASS metrics CSV requires failure_class none for every row")
    if collision_count != summary.collision_count:
        raise ValueError("metrics CSV collision count does not match summary")
    if offroad_count != summary.offroad_count:
        raise ValueError("metrics CSV offroad count does not match summary")
    if stuck_count != summary.stuck_count:
        raise ValueError("metrics CSV stuck count does not match summary")
    if rows and not math.isclose(route_progress, summary.route_progress, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError("metrics CSV final route_progress does not match summary")
    if rows and not math.isclose(episode_return, summary.episode_return, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError("metrics CSV episode_return does not match summary")


def write_summary(path: str | Path, summary: CleanEvalSummary) -> Path:
    validate_clean_summary(summary)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True) + "\n")
    return output_path


def write_step_metrics(path: str | Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(
            list(config.get("per_step_log_fields", REQUIRED_STEP_LOG_FIELDS)) + list(EXTRA_STEP_LOG_FIELDS)
        )
    )
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return metrics_path


def run_dry_run(config_path: str | Path) -> Path:
    config = load_clean_eval_config(config_path)
    output_dir = Path(config["output_dir"])
    summary = CleanEvalSummary(
        horizon=int(config["horizon"]),
        steps=0,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=0.0,
        source="local_smoke_not_carla",
        status="LOCAL_SMOKE_ONLY",
        done_reason="dry_run_not_carla",
        config=str(config_path),
        carla_town=str(config.get("town", "UNSET")),
        output_path=str(output_dir),
        next_action="Provide real Windows CARLA adapter factory and checkpoints.",
    )
    return write_summary(output_dir / "summary.json", summary)


def run_real_carla_eval(
    config_path: str | Path,
    *,
    adapter_factory: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CleanEvalSummary:
    config = load_clean_eval_config(config_path)
    metadata = metadata or {}
    spec = adapter_factory or config.get("adapter_factory")
    if spec is None or str(spec).strip() == "":
        write_step_metrics(config["metrics_path"], [], config)
        summary = _make_failure_summary(
            config=config,
            config_path=config_path,
            metadata=metadata,
            failure_class="adapter",
            done_reason="adapter_factory_missing",
            failure_message="No adapter_factory was provided for Windows CARLA clean eval.",
            steps=0,
            episode_return=0.0,
            route_progress=0.0,
        )
        write_summary(config["summary_path"], summary)
        return summary

    try:
        env, policy = _instantiate_adapter_bundle(str(spec), config)
    except CleanEvalRuntimeFailure as exc:
        write_step_metrics(config["metrics_path"], [], config)
        summary = _make_failure_summary(
            config=config,
            config_path=config_path,
            metadata=metadata,
            failure_class=exc.failure_class,
            done_reason=exc.done_reason,
            failure_message=str(exc),
            steps=0,
            episode_return=0.0,
            route_progress=0.0,
        )
        write_summary(config["summary_path"], summary)
        return summary
    except Exception as exc:
        write_step_metrics(config["metrics_path"], [], config)
        summary = _make_failure_summary(
            config=config,
            config_path=config_path,
            metadata=metadata,
            failure_class="adapter",
            done_reason="adapter_factory_import_failed",
            failure_message=f"{type(exc).__name__}: {exc}",
            steps=0,
            episode_return=0.0,
            route_progress=0.0,
        )
        write_summary(config["summary_path"], summary)
        return summary

    return run_clean_eval_loop(
        env=env,
        policy=policy,
        config=config,
        config_path=config_path,
        metadata=metadata,
    )


def run_clean_eval_loop(
    *,
    env: Any,
    policy: Any,
    config: dict[str, Any],
    config_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> CleanEvalSummary:
    metadata = metadata or {}
    horizon = int(config["horizon"])
    deterministic = bool(config.get("deterministic_policy", True))
    rows: list[dict[str, Any]] = []
    episode_return = 0.0
    previous_route_progress = 0.0
    failure: CleanEvalRuntimeFailure | None = None
    last_info: dict[str, Any] = {}

    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = _split_reset(env.reset())
        last_info = info
        previous_route_progress = _finite_float(info.get("route_progress", 0.0), "route_progress")
        obs_shape = _validate_stage2_obs(obs, config)

        for step in range(horizon):
            action_id, logits_finite = _select_action_id(policy, obs, info, deterministic=deterministic)
            action = get_discrete_action(action_id)
            control = to_carla_control(action)
            _validate_control(control)
            if hasattr(policy, "observe_action"):
                policy.observe_action(action_id)

            result = _step_env(env, control, action_id=action_id)
            next_obs, reward, done, truncated, next_info = _split_step(result)
            reward = _finite_float(reward, "reward")
            route_progress = _finite_float(next_info.get("route_progress", previous_route_progress), "route_progress")
            speed_kmh = _finite_float(next_info.get("speed_kmh", 0.0), "speed_kmh")
            cont = _continue_value(done=done, truncated=truncated, info=next_info, config=config)
            next_obs_shape = _validate_stage2_obs(next_obs, config)
            route_progress_delta = route_progress - previous_route_progress
            row = {
                "step": step,
                "action_id": action_id,
                "throttle": control["throttle"],
                "brake": control["brake"],
                "steer": control["steer"],
                "reverse": control["reverse"],
                "reward_total": reward,
                "reward_route_progress": _finite_float(
                    next_info.get("reward_route_progress", route_progress_delta),
                    "reward_route_progress",
                ),
                "reward_collision": _finite_float(next_info.get("reward_collision", 0.0), "reward_collision"),
                "reward_offroad": _finite_float(next_info.get("reward_offroad", 0.0), "reward_offroad"),
                "reward_stuck": _finite_float(next_info.get("reward_stuck", 0.0), "reward_stuck"),
                "continue": cont,
                "route_progress": route_progress,
                "collision": bool(next_info.get("collision", False)),
                "offroad": bool(next_info.get("offroad", False)),
                "stuck": bool(next_info.get("stuck", False)),
                "speed_kmh": speed_kmh,
                "done": bool(done),
                "truncated": bool(truncated),
                "adapter_error": "",
                "runtime_error": "",
                "done_reason": str(next_info.get("done_reason", "running")),
                "failure_class": "none",
                "policy_logits_finite": logits_finite,
                "privileged_bev_shape": next_obs_shape["privileged_bev_shape"] or obs_shape["privileged_bev_shape"],
                "privileged_ego_shape": next_obs_shape["privileged_ego_shape"] or obs_shape["privileged_ego_shape"],
                "obs_finite": True,
                "action_valid": True,
                "control_valid": True,
                "reward_finite": True,
                "route_progress_delta": route_progress_delta,
            }
            rows.append(row)
            episode_return += reward
            previous_route_progress = route_progress
            obs, info = next_obs, next_info
            last_info = next_info
            if done:
                break
    except CleanEvalRuntimeFailure as exc:
        failure = exc
    except Exception as exc:  # pragma: no cover - real CARLA/adapter boundary.
        failure = CleanEvalRuntimeFailure(
            _classify_unhandled_failure(exc),
            "runtime_exception",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        finally:
            write_step_metrics(config["metrics_path"], rows, config)

    summary = _summary_from_loop_result(
        config=config,
        config_path=config_path,
        metadata=metadata,
        rows=rows,
        episode_return=episode_return,
        route_progress=previous_route_progress,
        failure=failure,
        last_info=last_info,
    )
    write_summary(config["summary_path"], summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RCS-AC v2 clean eval contract CLI")
    parser.add_argument("--config", required=False)
    parser.add_argument("--dry-run", action="store_true", help="write LOCAL_SMOKE_ONLY summary")
    parser.add_argument("--run-carla", action="store_true", help="run Windows CARLA adapter loop")
    parser.add_argument("--adapter-factory", type=str, default=None)
    parser.add_argument("--commit", type=str, default="UNSET")
    parser.add_argument("--branch", type=str, default="UNSET")
    parser.add_argument("--windows-command", type=str, default="UNSET")
    parser.add_argument("--carla-version", type=str, default="UNSET")
    parser.add_argument("--validate-summary", type=str, default=None)
    args = parser.parse_args(argv)

    if args.validate_summary:
        data = json.loads(Path(args.validate_summary).read_text())
        validate_clean_summary(summary_from_dict(data))
        print(f"SUMMARY_VALID={args.validate_summary}")
        return 0

    if args.dry_run:
        if args.config is None:
            raise SystemExit("--config is required for --dry-run")
        output = run_dry_run(args.config)
        print(f"DRY_RUN_SUMMARY={output}")
        return 0

    if args.run_carla:
        if args.config is None:
            raise SystemExit("--config is required for --run-carla")
        summary = run_real_carla_eval(
            args.config,
            adapter_factory=args.adapter_factory,
            metadata={
                "commit": args.commit,
                "branch": args.branch,
                "windows_command": args.windows_command,
                "carla_version": args.carla_version,
            },
        )
        print(f"CARLA_SUMMARY={Path(summary.output_path) / 'summary.json'}")
        return 0 if summary.clean_gate_pass else 2

    raise SystemExit("Use --dry-run, --run-carla, or --validate-summary.")


def _make_failure_summary(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    metadata: dict[str, Any],
    failure_class: str,
    done_reason: str,
    failure_message: str,
    steps: int,
    episode_return: float,
    route_progress: float,
) -> CleanEvalSummary:
    return CleanEvalSummary(
        horizon=int(config["horizon"]),
        steps=steps,
        collision_count=0,
        offroad_count=0,
        stuck_count=0,
        route_progress=route_progress,
        source="windows_carla",
        status="FAIL",
        failure_class=failure_class,
        done_reason=done_reason,
        episode_return=episode_return,
        commit=_metadata_value(metadata, "commit"),
        branch=_metadata_value(metadata, "branch"),
        windows_command=_metadata_value(metadata, "windows_command"),
        config=str(config_path),
        carla_version=_metadata_value(metadata, "carla_version"),
        carla_town=str(config.get("town", "UNSET")),
        output_path=str(config["output_dir"]),
        metrics_path=str(config["metrics_path"]),
        pulled_files=(str(config["summary_path"]), str(config["metrics_path"])),
        next_action="Fix the classified failure, then rerun the same Windows clean gate.",
        failure_message=failure_message,
    )


def _summary_from_loop_result(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    episode_return: float,
    route_progress: float,
    failure: CleanEvalRuntimeFailure | None,
    last_info: dict[str, Any],
) -> CleanEvalSummary:
    collision_count = sum(1 for row in rows if bool(row["collision"]))
    offroad_count = sum(1 for row in rows if bool(row["offroad"]))
    stuck_count = sum(1 for row in rows if bool(row["stuck"]))
    steps = len(rows)
    horizon = int(config["horizon"])
    if failure is not None:
        status = "FAIL"
        failure_class = failure.failure_class
        done_reason = failure.done_reason
        failure_message = str(failure)
    elif steps == horizon and collision_count == 0 and offroad_count == 0 and stuck_count == 0:
        status = "PASS"
        failure_class = "none"
        done_reason = "horizon_reached"
        failure_message = ""
    else:
        status = "FAIL"
        failure_class = "driving"
        done_reason = str(last_info.get("done_reason", "ended_before_horizon"))
        failure_message = "Valid runtime ended before horizon or had clean-driving violations."

    next_action = (
        "Run 512-step primary subset."
        if status == "PASS"
        else "Classify the failure, patch the smallest cause, then rerun 128-step clean."
    )
    return CleanEvalSummary(
        horizon=horizon,
        steps=steps,
        collision_count=collision_count,
        offroad_count=offroad_count,
        stuck_count=stuck_count,
        route_progress=route_progress,
        source="windows_carla",
        status=status,
        failure_class=failure_class,
        done_reason=done_reason,
        episode_return=episode_return,
        commit=_metadata_value(metadata, "commit"),
        branch=_metadata_value(metadata, "branch"),
        windows_command=_metadata_value(metadata, "windows_command"),
        config=str(config_path),
        carla_version=_metadata_value(metadata, "carla_version"),
        carla_town=str(config.get("town", "UNSET")),
        output_path=str(config["output_dir"]),
        metrics_path=str(config["metrics_path"]),
        pulled_files=(str(config["summary_path"]), str(config["metrics_path"])),
        next_action=next_action,
        failure_message=failure_message,
    )


def _instantiate_adapter_bundle(spec: str, config: dict[str, Any]) -> tuple[Any, Any]:
    module_name, sep, attr_name = spec.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError("adapter_factory must have form 'module:function'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    bundle = factory(config)
    if isinstance(bundle, dict):
        return bundle["env"], bundle["policy"]
    if isinstance(bundle, tuple) and len(bundle) == 2:
        return bundle[0], bundle[1]
    env = getattr(bundle, "env", None)
    policy = getattr(bundle, "policy", None)
    if env is None or policy is None:
        raise ValueError("adapter_factory must return (env, policy), dict, or object with env/policy")
    return env, policy


def _split_reset(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(result, dict):
        obs = result.get("obs")
        info = result.get("info", {})
    elif isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        raise CleanEvalRuntimeFailure("adapter", "reset_contract_invalid", "reset must return dict or (obs, info)")
    if not isinstance(obs, dict):
        raise CleanEvalRuntimeFailure("adapter", "reset_obs_invalid", "reset obs must be a dict")
    if not isinstance(info, dict):
        raise CleanEvalRuntimeFailure("adapter", "reset_info_invalid", "reset info must be a dict")
    return obs, info


def _split_step(result: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    if isinstance(result, dict):
        obs = result.get("obs")
        reward = result.get("reward", result.get("info", {}).get("reward", 0.0))
        done = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        info = result.get("info", {})
    elif isinstance(result, tuple) and len(result) == 5:
        obs, reward, done, truncated, info = result
    else:
        raise CleanEvalRuntimeFailure("adapter", "step_contract_invalid", "step must return dict or 5-tuple")
    if not isinstance(obs, dict):
        raise CleanEvalRuntimeFailure("adapter", "step_obs_invalid", "step obs must be a dict")
    if not isinstance(info, dict):
        raise CleanEvalRuntimeFailure("adapter", "step_info_invalid", "step info must be a dict")
    return obs, float(reward), bool(done), bool(truncated), info


def _step_env(env: Any, control: dict[str, float | bool], *, action_id: int) -> Any:
    try:
        return env.step(control, policy_diagnostics={"rcs_ac_v2": True, "action_id": action_id})
    except TypeError:
        return env.step(control)


def _validate_stage2_obs(obs: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    contract = config.get("observation_contract", {"require_privileged": True})
    if not contract.get("require_privileged", True):
        return {"privileged_bev_shape": "", "privileged_ego_shape": ""}
    if "privileged_bev" not in obs or "privileged_ego" not in obs:
        raise CleanEvalRuntimeFailure(
            "adapter",
            "privileged_obs_missing",
            "obs must include privileged_bev and privileged_ego for Stage 2 clean eval",
        )
    try:
        import torch

        bev = torch.as_tensor(obs["privileged_bev"], dtype=torch.float32)
        ego = torch.as_tensor(obs["privileged_ego"], dtype=torch.float32)
    except Exception as exc:
        raise CleanEvalRuntimeFailure("adapter", "privileged_obs_tensor_invalid", str(exc)) from exc
    if bev.ndim != 5 or tuple(bev.shape[:2]) != (1, 1) or int(bev.shape[2]) != 43:
        raise CleanEvalRuntimeFailure("adapter", "privileged_bev_shape_invalid", f"got {tuple(bev.shape)}")
    if ego.ndim != 3 or tuple(ego.shape[:2]) != (1, 1):
        raise CleanEvalRuntimeFailure("adapter", "privileged_ego_shape_invalid", f"got {tuple(ego.shape)}")
    if not torch.isfinite(bev).all() or not torch.isfinite(ego).all():
        raise CleanEvalRuntimeFailure("adapter", "privileged_obs_non_finite", "privileged obs contains NaN/inf")
    expected_bev_shape = tuple(contract.get("privileged_bev_shape", ()))
    expected_ego_shape = tuple(contract.get("privileged_ego_shape", ()))
    if expected_bev_shape and tuple(bev.shape) != expected_bev_shape:
        raise CleanEvalRuntimeFailure("adapter", "privileged_bev_shape_invalid", f"got {tuple(bev.shape)}")
    if expected_ego_shape and tuple(ego.shape) != expected_ego_shape:
        raise CleanEvalRuntimeFailure("adapter", "privileged_ego_shape_invalid", f"got {tuple(ego.shape)}")
    return {"privileged_bev_shape": str(tuple(bev.shape)), "privileged_ego_shape": str(tuple(ego.shape))}


def _select_action_id(policy: Any, obs: dict[str, Any], info: dict[str, Any], *, deterministic: bool) -> tuple[int, bool]:
    if hasattr(policy, "act"):
        output = policy.act(obs, info)
    elif callable(policy):
        output = policy(obs, info)
    else:
        raise CleanEvalRuntimeFailure("model", "policy_not_callable", "policy must be callable or expose act()")

    if not isinstance(output, dict) or "logits" not in output:
        raise CleanEvalRuntimeFailure("model", "policy_logits_missing", "policy output must include logits")
    try:
        import torch

        logits = torch.as_tensor(output["logits"], dtype=torch.float32)
    except Exception as exc:
        raise CleanEvalRuntimeFailure("model", "policy_logits_invalid", str(exc)) from exc
    if tuple(logits.shape) != (1, 1, NUM_RAW2DRIVE_ACTIONS):
        raise CleanEvalRuntimeFailure("model", "policy_logits_shape_invalid", f"got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise CleanEvalRuntimeFailure("model", "policy_logits_non_finite", "policy logits contain NaN/inf")
    if deterministic:
        action_id = int(torch.argmax(logits, dim=-1).item())
    else:
        action_id = int(torch.distributions.Categorical(logits=logits).sample().item())
    if action_id < 0 or action_id >= NUM_RAW2DRIVE_ACTIONS:
        raise CleanEvalRuntimeFailure("contract", "action_id_out_of_range", f"got {action_id}")
    return action_id, True


def _validate_control(control: dict[str, float | bool]) -> None:
    throttle = float(control["throttle"])
    brake = float(control["brake"])
    steer = float(control["steer"])
    reverse = control["reverse"]
    if not 0.0 <= throttle <= 1.0 or not 0.0 <= brake <= 1.0 or not -1.0 <= steer <= 1.0:
        raise CleanEvalRuntimeFailure("contract", "control_range_invalid", f"control={control}")
    if throttle > 0.0 and brake > 0.0:
        raise CleanEvalRuntimeFailure("contract", "control_throttle_brake_overlap", f"control={control}")
    if not isinstance(reverse, bool):
        raise CleanEvalRuntimeFailure("contract", "control_reverse_invalid", "reverse must be bool")


def _continue_value(*, done: bool, truncated: bool, info: dict[str, Any], config: dict[str, Any]) -> float:
    if "continue" in info:
        cont = _finite_float(info["continue"], "continue")
    elif done and not truncated:
        cont = float(config["continue_contract"]["terminal_done"])
    else:
        cont = float(config["continue_contract"]["running"])
    if cont not in (0.0, 1.0):
        raise CleanEvalRuntimeFailure("contract", "continue_contract_invalid", f"continue={cont}")
    return cont


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CleanEvalRuntimeFailure("contract", f"{name}_non_finite", f"{name} must be finite")
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _expected_metrics_fields(summary: CleanEvalSummary) -> tuple[str, ...] | None:
    config_path = Path(summary.config)
    if summary.config == "UNSET" or not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text())
    except OSError:
        return None
    if not isinstance(data, dict):
        return None
    fields = data.get("per_step_log_fields")
    if not isinstance(fields, list) or not fields:
        return None
    return tuple(str(field) for field in fields)


def _classify_unhandled_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "carla" in text or "rpc" in text or "timeout" in text or "unreal" in text:
        return "CARLA"
    return "adapter"


if __name__ == "__main__":
    raise SystemExit(main())
