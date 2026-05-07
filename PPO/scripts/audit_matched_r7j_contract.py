from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "RCS-AC-v1.1" / "code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from PPO.ppo_r7j.models import PPOMatchedR7JPolicy
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config


EXPECTED_FORBIDDEN_FLAGS = {
    "rcs_ac_reliability": False,
    "rcs_ac_source_fusion": False,
    "rcs_ac_shapley": False,
    "rcs_ac_risk_context": False,
}


def main() -> None:
    args = parse_args()
    report = audit_contract(args.ppo_config)
    out_path = Path(args.output)
    if args.output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["pass"]):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PPO matched-r7j fairness contract.")
    parser.add_argument("--ppo-config", default="PPO/configs/matched_r7j/ppo_matched_r7j_train_seed7.yaml")
    parser.add_argument("--output", default="PPO/results/contract_audit/ppo_matched_r7j_contract_audit.json")
    return parser.parse_args()


def audit_contract(ppo_config_path: str | Path) -> dict[str, Any]:
    config_path = Path(ppo_config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    runtime_path = _cfg_path(config_path, cfg["runtime_config"])
    runtime_config = load_v11_runtime_config(runtime_path)
    model = PPOMatchedR7JPolicy(runtime_config)
    controller = runtime_config.controller_config
    model_cfg = dict(cfg.get("model", {}))
    fairness = dict(cfg.get("fairness_contract", {}))

    checks: dict[str, bool] = {
        "runtime_config_exists": runtime_path.exists(),
        "runtime_is_r7j": runtime_path.name == "full_bev_smoke_shared_agile_midbalanced_route_geometry_cc4_r7j_target_y_lowpass_light.yaml",
        "obs_total_dim_matches_model": runtime_config.observation_spec.total_dim > 0,
        "rgb_shape_matched": tuple(runtime_config.policy_config.rgb_shape) == (3, 90, 160),
        "bev_spec_matched": (
            runtime_config.bev_spec.height == 128
            and runtime_config.bev_spec.width == 128
            and abs(runtime_config.bev_spec.resolution_m - 0.375) < 1.0e-9
            and abs(runtime_config.bev_spec.front_m - 40.0) < 1.0e-9
            and abs(runtime_config.bev_spec.rear_m - 8.0) < 1.0e-9
            and abs(runtime_config.bev_spec.left_m - 24.0) < 1.0e-9
            and abs(runtime_config.bev_spec.right_m - 24.0) < 1.0e-9
        ),
        "ego_route_health_dims_matched": (
            runtime_config.policy_config.ego_scalar_dim == 6
            and runtime_config.policy_config.route_scalar_dim == 20
            and runtime_config.policy_config.sensor_health_dim == 15
        ),
        "policy_output_waypoint_speed": model.action_dim == 9 and runtime_config.policy_config.waypoint_count == 4,
        "action_format_direct_3d": str(cfg.get("action_format", "")) == "direct_3d",
        "controller_type_matched": str(runtime_config.raw["controller"].get("type")) == "speed_adaptive_pure_pursuit",
        "target_y_lowpass_tau_s_matched": abs(float(controller.target_y_lowpass_tau_s) - 0.06) < 1.0e-9,
        "steer_lowpass_tau_s_matched": abs(float(controller.steer_lowpass_tau_s) - 0.12) < 1.0e-9,
        "max_steer_rate_matched": abs(float(controller.max_steer_rate_per_s) - 2.8) < 1.0e-9,
        "target_speed_cap_matched": (
            abs(float(runtime_config.policy_config.min_target_speed_kmh) - 0.0) < 1.0e-9
            and abs(float(runtime_config.policy_config.max_target_speed_kmh) - 34.0) < 1.0e-9
        ),
        "curvature_speed_cap_enabled": bool(controller.curvature_speed_cap_enabled),
        "no_rcs_specific_modules": all(model_cfg.get(key, expected) is expected for key, expected in EXPECTED_FORBIDDEN_FLAGS.items()),
        "no_runtime_steering_override": not bool(fairness.get("runtime_steering_override", True)),
        "no_privileged_inference_signals": not bool(fairness.get("privileged_inference_signals", True)),
        "ppo_result_folder": str(cfg.get("output_dir", "")).startswith("PPO/"),
    }
    return {
        "pass": all(checks.values()),
        "ppo_config": str(config_path),
        "runtime_config": str(runtime_path),
        "checks": checks,
        "controller": {
            "target_y_lowpass_tau_s": float(controller.target_y_lowpass_tau_s),
            "steer_lowpass_tau_s": float(controller.steer_lowpass_tau_s),
            "max_steer_rate_per_s": float(controller.max_steer_rate_per_s),
            "curvature_speed_cap_enabled": bool(controller.curvature_speed_cap_enabled),
            "curvature_speed_mode": str(controller.curvature_speed_mode),
        },
        "policy": {
            "observation_total_dim": int(runtime_config.observation_spec.total_dim),
            "waypoint_count": int(runtime_config.policy_config.waypoint_count),
            "ppo_raw_action_dim": int(model.action_dim),
        },
    }


def _cfg_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (config_path.parent / path).resolve()
    return candidate if candidate.exists() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
