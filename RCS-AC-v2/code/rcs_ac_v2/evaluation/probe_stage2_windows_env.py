"""Shape-only Windows CARLA probe for the Stage 2 privileged clean gate.

This probe is deliberately not a clean-eval summary. It writes
``privileged_obs_probe.json`` with ``clean_gate_evidence=false`` so plumbing
evidence cannot be mistaken for clean performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rcs_ac_v2.evaluation.carla_adapter import (
    SharedCoreCarlaEnvAdapter,
    Stage2CarlaObservationAdapter,
    _build_shared_core_env,
)
from rcs_ac_v2.evaluation.errors import CleanEvalRuntimeFailure
from rcs_ac_v2.evaluation.evaluate_clean import load_clean_eval_config


NEUTRAL_CONTROL = {"steer": 0.0, "throttle": 0.0, "brake": 0.0, "reverse": False}


def probe_stage2_env_once(
    *,
    raw_env: Any,
    config: dict[str, Any],
    step_neutral: bool = False,
) -> dict[str, Any]:
    observation_adapter = Stage2CarlaObservationAdapter(config)
    probe: dict[str, Any] = {
        "stage": "stage2_privileged_policy_clean",
        "source": "windows_carla_privileged_obs_probe",
        "clean_gate_evidence": False,
        "status": "PROBE_FAIL",
        "failure_class": "none",
        "done_reason": "UNSET",
        "raw_obs_keys": [],
        "raw_info_keys": [],
        "privileged_bev_shape": [],
        "privileged_ego_shape": [],
        "privileged_bev_finite": False,
        "privileged_ego_finite": False,
        "step_neutral": bool(step_neutral),
        "env_action_format": str(config.get("carla", {}).get("env_action_format", "UNSET")),
    }
    try:
        raw_reset = raw_env.reset()
        obs, info = _split_reset_result(raw_reset)
        probe.update(_summarize_raw_reset(raw_reset=raw_reset, obs=obs, info=info))
        adapted = observation_adapter.adapt(obs, info)
        bev = adapted["privileged_bev"]
        ego = adapted["privileged_ego"]
        probe.update(
            {
                "privileged_bev_shape": list(bev.shape),
                "privileged_ego_shape": list(ego.shape),
                "privileged_bev_finite": _is_finite(bev),
                "privileged_ego_finite": _is_finite(ego),
                "route_progress": _optional_float(info.get("route_progress")),
                "speed_kmh": _optional_float(info.get("speed_kmh")),
                "collision": bool(info.get("collision", False)),
                "offroad": bool(info.get("offroad", False)),
                "stuck": bool(info.get("stuck", False)),
            }
        )
        if step_neutral:
            wrapped = SharedCoreCarlaEnvAdapter(raw_env, config)
            step = wrapped.step(NEUTRAL_CONTROL, policy_diagnostics={"probe": "stage2_privileged_obs"})
            step_info = dict(step.get("info", {}))
            probe.update(
                {
                    "neutral_step_reward": _optional_float(step.get("reward")),
                    "neutral_step_done": bool(step.get("done", False)),
                    "neutral_step_truncated": bool(step.get("truncated", False)),
                    "neutral_step_info_keys": sorted(map(str, step_info.keys())),
                    "neutral_step_route_progress": _optional_float(step_info.get("route_progress")),
                    "neutral_step_speed_kmh": _optional_float(step_info.get("speed_kmh")),
                    "neutral_step_done_reason": str(step_info.get("done_reason", "UNSET")),
                    "neutral_step_collision": bool(step_info.get("collision", False)),
                    "neutral_step_offroad": bool(step_info.get("offroad", False)),
                    "neutral_step_stuck": bool(step_info.get("stuck", False)),
                }
            )
        probe["status"] = "PROBE_PASS"
        probe["done_reason"] = "privileged_obs_probe_pass"
        return probe
    except CleanEvalRuntimeFailure as exc:
        probe["failure_class"] = exc.failure_class
        probe["done_reason"] = exc.done_reason
        probe["failure_message"] = str(exc)
        return probe
    except Exception as exc:  # pragma: no cover - real CARLA boundary.
        probe["failure_class"] = "infrastructure"
        probe["done_reason"] = "privileged_obs_probe_exception"
        probe["failure_message"] = f"{type(exc).__name__}: {exc}"
        return probe
    finally:
        if hasattr(raw_env, "close"):
            raw_env.close()


def run_windows_probe(
    *,
    config_path: str | Path,
    output_path: str | Path | None = None,
    step_neutral: bool = False,
) -> dict[str, Any]:
    config = load_clean_eval_config(config_path)
    raw_env = _build_shared_core_env(config)
    result = probe_stage2_env_once(raw_env=raw_env, config=config, step_neutral=step_neutral)
    path = Path(output_path) if output_path is not None else Path(config["output_dir"]) / "privileged_obs_probe.json"
    write_probe_result(path, result)
    return result


def write_probe_result(path: str | Path, result: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output_path


def _split_reset_result(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(result, dict) and "obs" in result:
        obs = result["obs"]
        info = result.get("info", {})
        if not isinstance(obs, dict) or not isinstance(info, dict):
            raise CleanEvalRuntimeFailure("adapter", "probe_reset_schema_invalid", "reset obs/info must be mappings")
        return obs, info
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        if not isinstance(obs, dict) or not isinstance(info, dict):
            raise CleanEvalRuntimeFailure("adapter", "probe_reset_schema_invalid", "reset tuple obs/info must be mappings")
        return obs, info
    if isinstance(result, dict):
        return result, {}
    raise CleanEvalRuntimeFailure("adapter", "probe_reset_schema_invalid", f"unsupported reset result {type(result).__name__}")


def _summarize_raw_reset(*, raw_reset: Any, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_reset_type": type(raw_reset).__name__,
        "raw_reset_keys": sorted(map(str, raw_reset.keys())) if isinstance(raw_reset, dict) else [],
        "raw_obs_keys": sorted(map(str, obs.keys())),
        "raw_info_keys": sorted(map(str, info.keys())),
        "raw_obs_summary": {str(key): _summarize_value(value) for key, value in obs.items()},
        "raw_info_summary": {str(key): _summarize_value(value) for key, value in info.items()},
    }


def _summarize_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            summary["shape"] = list(shape)
        except TypeError:
            summary["shape"] = str(shape)
    if isinstance(value, (list, tuple)):
        summary["len"] = len(value)
    if isinstance(value, dict):
        summary["keys"] = sorted(map(str, value.keys()))
    return summary


def _is_finite(value: Any) -> bool:
    tensor = torch.as_tensor(value) if not isinstance(value, np.ndarray) else torch.from_numpy(value)
    return bool(torch.isfinite(tensor).all())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe Stage 2 privileged CARLA observations without clean PASS evidence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--step-neutral", action="store_true")
    args = parser.parse_args(argv)
    result = run_windows_probe(
        config_path=args.config,
        output_path=args.output or None,
        step_neutral=args.step_neutral,
    )
    output_path = args.output or str(Path(load_clean_eval_config(args.config)["output_dir"]) / "privileged_obs_probe.json")
    print(f"PRIVILEGED_OBS_PROBE={output_path}")
    if result.get("status") != "PROBE_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
