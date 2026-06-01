from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

from rcs_ac_first.rl.ppo_agent import PPOAgent, PPOConfig
from rcs_ac_first.rl.rollout_buffer import RolloutBuffer
from rcs_ac_first.runtime.action import STAGE4_ACTION_V2, decode_stage4_action_v2
from rcs_ac_first.runtime.auxiliary import build_auxiliary_observation_spec, flatten_auxiliary_observation_dict
from rcs_ac_first.runtime.config import Phase1CoreConfig, load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec, flatten_observation_dict
from rcs_ac_first.runtime.phase2_targets import build_phase2_targets_from_info


TRAINING_LOG_FIELDS: tuple[str, ...] = (
    "timestamp",
    "run_name",
    "model_type",
    "model_variant",
    "training_stage",
    "record_type",
    "total_steps",
    "update",
    "episode_index",
    "episode_reward",
    "episode_length",
    "average_reward",
    "average_route_progress",
    "route_progress",
    "max_route_progress_in_episode",
    "done_reason",
    "reached_goal",
    "goal_count",
    "collision",
    "offroad",
    "stuck",
    "actor_loss",
    "critic_loss",
    "entropy",
    "total_loss",
    "grad_norm",
    "aux_lane_offset_loss",
    "aux_heading_error_loss",
    "aux_lane_offset_mae",
    "aux_heading_error_mae",
    "aux_road_mask_loss",
    "aux_lane_edge_mask_loss",
    "aux_road_mask_mae",
    "aux_lane_edge_mask_mae",
    "aux_road_mask_iou",
    "aux_lane_edge_mask_iou",
    "aux_road_mask_positive_ratio",
    "aux_lane_edge_positive_ratio",
    "aux_mask_available_rate",
    "aux_mask_targets_available",
    "aux_mask_target_source",
    "reliability_loss",
    "risk_supervision_loss",
    "risk_regularization",
    "shapley_alignment_loss",
    "dropout_consistency_loss",
    "reliability_rgb",
    "reliability_lidar",
    "reliability_ego",
    "reliability_route",
    "reliability_target_rgb",
    "reliability_target_lidar",
    "reliability_target_ego",
    "reliability_target_route",
    "risk_scalar",
    "risk_target",
    "contribution_rgb",
    "contribution_lidar",
    "contribution_ego",
    "contribution_route",
    "shapley_raw_rgb",
    "shapley_raw_lidar",
    "shapley_raw_ego",
    "shapley_raw_route",
    "contribution_entropy",
    "shapley_estimated",
    "dropout_action_l1",
    "dropout_value_l1",
    "dropout_mask_rgb_rate",
    "dropout_mask_lidar_rate",
    "dropout_mask_ego_rate",
    "dropout_mask_route_rate",
    "shapley_subset_count",
    "corruption_level",
    "drop_rgb",
    "drop_lidar",
    "drop_ego",
    "drop_route",
    "policy_std_mean",
    "policy_std_dim0",
    "policy_std_dim1",
    "fusion_rgb",
    "fusion_lidar",
    "fusion_ego",
    "fusion_route",
    "fusion_weight_rgb",
    "fusion_weight_lidar",
    "fusion_weight_ego",
    "fusion_weight_route",
    "fusion_entropy",
    "rgb_feature_l1_mean",
    "rgb_feature_map_l1_mean",
    "compact_visual_features_enabled",
    "compact_visual_feature_dim",
    "compact_visual_feature_l1_mean",
    "road_compact_feature_l1_mean",
    "lane_compact_feature_l1_mean",
    "compact_visual_actor_input_l1_mean",
    "visual_actor_delta_l1_mean",
    "lidar_feature_l1_mean",
    "ego_feature_l1_mean",
    "route_feature_l1_mean",
    "fused_feature_l1_mean",
    "recurrent_hidden_l1_mean",
    "steer",
    "longitudinal",
    "throttle",
    "brake",
    "speed_kmh",
    "lane_offset",
    "heading_error",
    "reward",
    "checkpoint_path",
    "init_checkpoint_path",
    "init_checkpoint_loaded",
    "init_checkpoint_strict",
    "init_load_optimizer",
    "init_missing_keys",
    "init_unexpected_keys",
    "init_loaded_key_count",
    "init_skipped_key_count",
    "init_dummy_forward_ok",
    "std_override_enabled",
    "std_override_source",
    "log_std_dim0",
    "log_std_dim1",
    "deterministic_rollout",
    "update_success",
    "nan_detected",
)


def _default_config_path() -> Path:
    root_config = PROJECT_ROOT / "configs" / "phase1_core_smoke.yaml"
    if root_config.exists():
        return root_config
    return PROJECT_ROOT / "rcs_ac_first" / "configs" / "phase1_core_smoke.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCS-AC-first Phase 1 core in CARLA.")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--init-strict", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--load-init-optimizer", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--override-log-std", type=float, nargs=2, default=None)
    return parser.parse_args()


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _bootstrap_carla() -> None:
    try:
        import carla  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    search_roots: list[Path] = []
    if os.environ.get("CARLA_ROOT"):
        search_roots.append(Path(os.environ["CARLA_ROOT"]))
    search_roots.extend([Path.home() / "Downloads" / "CARLA_0.9.16", Path("C:/CARLA_0.9.16")])

    for root in search_roots:
        dist_dir = root / "PythonAPI" / "carla" / "dist"
        if not dist_dir.exists():
            continue
        matching = sorted(dist_dir.glob(f"carla-*-{py_tag}-{py_tag}-win_amd64.whl"))
        if matching:
            sys.path.append(str(matching[-1]))
            import carla  # noqa: F401
            return
        available = sorted(dist_dir.glob("carla-*.whl"))
        if available:
            versions = ", ".join(item.name for item in available)
            raise RuntimeError(
                "Found CARLA Python API, but no wheel matches this Python version. "
                f"Current Python is {sys.version.split()[0]}; available wheels: {versions}."
            )
    raise RuntimeError("CARLA Python API was not found. Set CARLA_ROOT or install the matching wheel.")


def _resolve_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate.resolve()
    workspace_candidate = WORKSPACE_ROOT / candidate
    if workspace_candidate.exists():
        return workspace_candidate.resolve()
    return project_candidate.resolve()


def _project_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _filter_dataclass_kwargs(dataclass_type: type[Any], raw: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in raw.items() if key in names}


def _build_env(config: Phase1CoreConfig, *, host: str, port: int, output_dir: Path, run_name: str) -> Any:
    _bootstrap_carla()
    from shared_core.env.carla_env import CarlaEnv
    from shared_core.env.reward import RewardConfig
    from shared_core.env.state_builder import ObservationConfig

    env_raw = dict(config.raw.get("env") or {})
    corruption_raw = env_raw.get("corruption")
    if isinstance(corruption_raw, dict) and str(corruption_raw.get("enabled", "")).strip().lower() == "auto":
        corruption_raw = dict(corruption_raw)
        lower_run_name = str(run_name).lower()
        corruption_raw["enabled"] = "corruption" in lower_run_name or "mild" in lower_run_name
    reward_raw = dict(config.raw.get("reward") or {})
    observation_config = ObservationConfig(
        route_representation=str(env_raw.get("route_representation", "single_target_point_xy")),
        route_lookahead_count=int(env_raw.get("route_lookahead_count", 3)),
    )
    reward_config = RewardConfig(**_filter_dataclass_kwargs(RewardConfig, reward_raw))
    return CarlaEnv(
        host=host,
        port=int(port),
        town=str(config.train.town),
        weather=str(env_raw.get("weather", "default")),
        traffic_mode=str(env_raw.get("traffic_mode", "none")),
        traffic_vehicle_count=int(env_raw.get("traffic_vehicle_count", 0)),
        traffic_seed=int(env_raw.get("traffic_seed", config.train.seed)),
        output_dir=str(output_dir),
        image_every=int(env_raw.get("image_every", 50)),
        client_timeout=float(env_raw.get("client_timeout", 30.0)),
        route_length=int(env_raw.get("route_length", 40)),
        route_spacing=float(env_raw.get("route_spacing", 3.0)),
        max_steps=int(config.train.max_episode_steps),
        observation_config=observation_config,
        reward_config=reward_config,
        stuck_speed_threshold_mps=float(env_raw.get("stuck_speed_threshold_mps", 0.3)),
        stuck_min_steps=int(env_raw.get("stuck_min_steps", 60)),
        offroad_min_steps=int(env_raw.get("offroad_min_steps", 5)),
        offroad_grace_steps=int(env_raw.get("offroad_grace_steps", 8)),
        offroad_lane_offset_threshold=env_raw.get("offroad_lane_offset_threshold"),
        goal_radius=float(env_raw.get("goal_radius", 3.0)),
        action_contract_version=STAGE4_ACTION_V2,
        suppress_conflicting_pedals=bool(env_raw.get("suppress_conflicting_pedals", True)),
        low_speed_brake_suppression=bool(env_raw.get("low_speed_brake_suppression", True)),
        low_speed_brake_speed_threshold_kmh=float(env_raw.get("low_speed_brake_speed_threshold_kmh", 3.0)),
        low_speed_brake_max=float(env_raw.get("low_speed_brake_max", 0.05)),
        longitudinal_control=env_raw.get("longitudinal_control"),
        action_smoothing=env_raw.get("action_smoothing"),
        lateral_recovery=env_raw.get("lateral_recovery"),
        rgb_sensor=env_raw.get("rgb_sensor"),
        lidar_sensor=env_raw.get("lidar_sensor"),
        semantic_aux_camera_enabled=bool(env_raw.get("semantic_aux_camera_enabled", False)),
        semantic_sensor=env_raw.get("semantic_sensor"),
        aux_mask_width=int(env_raw.get("aux_mask_width", 80)),
        aux_mask_height=int(env_raw.get("aux_mask_height", 45)),
        maneuver_command_enabled=bool(env_raw.get("maneuver_command_enabled", False)),
        maneuver_command_mode=str(env_raw.get("maneuver_command_mode", "cycle")),
        maneuver_command_values=env_raw.get("maneuver_command_values"),
        corruption_config=corruption_raw,
    )


def _normalize_aux_target(value: float, *, divisor: float, clip_abs: float | None) -> float:
    normalized = float(value) / float(divisor)
    if clip_abs is not None:
        normalized = float(np.clip(normalized, -float(clip_abs), float(clip_abs)))
    return normalized


def _aux_normalization(config: Phase1CoreConfig) -> dict[str, float | None]:
    raw = dict(config.raw.get("aux_target_normalization") or {})
    return {
        "lane_offset_divisor_m": float(raw.get("lane_offset_divisor_m", 2.5)),
        "lane_offset_clip_abs": raw.get("lane_offset_clip_abs", 1.0),
        "heading_error_divisor_deg": float(raw.get("heading_error_divisor_deg", 30.0)),
        "heading_error_clip_abs": raw.get("heading_error_clip_abs", 1.0),
    }


def _mask_positive_ratio(value: Any) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return 0.0
    return float(np.clip(array, 0.0, 1.0).mean())


def _load_init_checkpoint(
    agent: PPOAgent,
    checkpoint_path: Path,
    *,
    strict: bool,
    load_optimizer: bool,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Init checkpoint does not contain model_state_dict: {checkpoint_path}")

    model_state = checkpoint["model_state_dict"]
    if not isinstance(model_state, dict):
        raise ValueError("Init checkpoint model_state_dict must be a mapping.")

    metadata: dict[str, Any] = {
        "init_checkpoint_path": str(checkpoint_path),
        "init_checkpoint_loaded": False,
        "init_checkpoint_strict": bool(strict),
        "init_load_optimizer": bool(load_optimizer),
        "init_missing_keys": "",
        "init_unexpected_keys": "",
        "init_loaded_key_count": 0,
        "init_skipped_key_count": 0,
        "init_dummy_forward_ok": False,
    }

    if strict:
        incompatible = agent.model.load_state_dict(model_state, strict=True)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        loaded_count = len(model_state)
        skipped_count = 0
    else:
        current_state = agent.model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in model_state.items():
            if key in current_state and tuple(current_state[key].shape) == tuple(value.shape):
                filtered_state[key] = value
            else:
                skipped.append(str(key))
        incompatible = agent.model.load_state_dict(filtered_state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        loaded_count = len(filtered_state)
        skipped_count = len(skipped)
        if skipped:
            unexpected = sorted(set(unexpected + skipped))

    if load_optimizer:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("load_init_optimizer=True but checkpoint has no optimizer_state_dict.")
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    agent.reset_hidden()
    metadata.update(
        {
            "init_checkpoint_loaded": True,
            "init_missing_keys": "|".join(missing),
            "init_unexpected_keys": "|".join(unexpected),
            "init_loaded_key_count": int(loaded_count),
            "init_skipped_key_count": int(skipped_count),
        }
    )
    return metadata


def _override_model_log_std(
    agent: PPOAgent,
    values: list[float] | tuple[float, ...],
    *,
    source: str,
) -> dict[str, Any]:
    log_std = torch.as_tensor(values, dtype=torch.float32, device=agent.device).reshape(-1)
    expected = int(agent.config.model.action_dim)
    if int(log_std.numel()) != expected:
        raise ValueError(f"Expected {expected} log_std values, got {int(log_std.numel())}")
    if not torch.isfinite(log_std).all():
        raise ValueError(f"log_std override contains non-finite values: {values!r}")
    with torch.no_grad():
        agent.model.recurrent_actor_critic.log_std.copy_(log_std)
    std = agent.model.get_std().detach().cpu()
    return {
        "std_override_enabled": True,
        "std_override_source": str(source),
        "log_std_dim0": float(log_std[0].detach().cpu().item()),
        "log_std_dim1": float(log_std[1].detach().cpu().item()) if expected > 1 else "",
        "policy_std_dim0": float(std[0].item()),
        "policy_std_dim1": float(std[1].item()) if expected > 1 else "",
    }


def _ppo_trainable_module_names(config: Phase1CoreConfig) -> list[str]:
    ppo_raw = config.raw.get("ppo") if isinstance(config.raw, dict) else None
    if not isinstance(ppo_raw, dict):
        return []
    raw = ppo_raw.get("trainable_module_names", [])
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _ppo_deterministic_rollout(config: Phase1CoreConfig) -> bool:
    ppo_raw = config.raw.get("ppo") if isinstance(config.raw, dict) else None
    if not isinstance(ppo_raw, dict):
        return False
    return bool(ppo_raw.get("deterministic_rollout", False))


def _select_trainable_parameters(
    agent: PPOAgent,
    module_names: list[str],
) -> list[torch.nn.Parameter]:
    if not module_names:
        return [parameter for parameter in agent.model.parameters() if parameter.requires_grad]
    prefixes = tuple(str(name).strip() for name in module_names if str(name).strip())
    if not prefixes:
        return [parameter for parameter in agent.model.parameters() if parameter.requires_grad]
    selected: list[torch.nn.Parameter] = []
    for name, parameter in agent.model.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"ppo.trainable_module_names selected no parameters: {module_names}")
    selected_count = sum(int(parameter.numel()) for parameter in selected)
    total_count = sum(int(parameter.numel()) for parameter in agent.model.parameters())
    print(
        "[trainable] "
        f"prefixes={list(prefixes)} selected_params={selected_count} total_params={total_count}"
    )
    return selected


def _resolve_log_std_override(config: Phase1CoreConfig, args: argparse.Namespace) -> tuple[list[float] | None, str]:
    if args.override_log_std is not None:
        return [float(item) for item in args.override_log_std], "cli"
    ppo_raw = config.raw.get("ppo") if isinstance(config.raw, dict) else None
    if not isinstance(ppo_raw, dict):
        return None, ""
    if not bool(ppo_raw.get("override_loaded_log_std", False)):
        return None, ""
    raw_values = ppo_raw.get("loaded_log_std_by_dim", config.model.log_std_init_by_dim)
    if not isinstance(raw_values, (list, tuple)):
        raise ValueError("ppo.loaded_log_std_by_dim must be a list when override_loaded_log_std=true")
    return [float(item) for item in raw_values], "config"


def _verify_init_forward(
    agent: PPOAgent,
    *,
    obs_dim: int,
    auxiliary_dim: int,
) -> None:
    with torch.no_grad():
        obs = torch.zeros((1, int(obs_dim)), dtype=torch.float32, device=agent.device)
        auxiliary = torch.zeros((1, int(auxiliary_dim)), dtype=torch.float32, device=agent.device)
        outputs = agent.model.forward_policy_outputs(
            obs,
            auxiliary=auxiliary,
            hidden_state=None,
            track_diagnostics=False,
        )
        action_mean = outputs["action_mean"]
        value = outputs["value"]
        fusion_weights = outputs["fusion_weights"]
        checks = {
            "action_mean": (action_mean, (1, 2)),
            "value": (value, (1,)),
            "fusion_weights": (fusion_weights, (1, 4)),
        }
        for name, (tensor, expected_shape) in checks.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Init checkpoint dummy forward missing {name}.")
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Init checkpoint dummy forward {name} shape mismatch: "
                    f"expected {expected_shape}, got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Init checkpoint dummy forward produced non-finite {name}.")
    agent.reset_hidden()


def _write_training_log_row(
    writer: csv.DictWriter,
    log_file: Any,
    *,
    config: Phase1CoreConfig,
    run_name: str,
    record_type: str,
    total_steps: int,
    update: int,
    episode_index: int,
    episode_reward: float,
    episode_length: int,
    average_reward: float,
    average_route_progress: float,
    max_route_progress_in_episode: float,
    info: dict[str, Any],
    diagnostics: dict[str, float],
    metrics: dict[str, float] | None = None,
    checkpoint_path: str = "",
    init_metadata: dict[str, Any] | None = None,
    deterministic_rollout: bool = False,
    update_success: bool = True,
    nan_detected: bool = False,
) -> None:
    metrics = metrics or {}
    init_metadata = init_metadata or {}
    action = info.get("policy_action", {})
    control = info.get("control", {})
    decoded = decode_stage4_action_v2(
        [
            float(action.get("steer", control.get("steer", 0.0))),
            float(action.get("longitudinal", 0.0)),
        ]
    )
    actual_steer = float(control.get("steer", decoded["steer"])) if isinstance(control, dict) else float(decoded["steer"])
    actual_throttle = (
        float(control.get("throttle", decoded["throttle"])) if isinstance(control, dict) else float(decoded["throttle"])
    )
    actual_brake = float(control.get("brake", decoded["brake"])) if isinstance(control, dict) else float(decoded["brake"])
    row: dict[str, Any] = {field: "" for field in TRAINING_LOG_FIELDS}
    row.update(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_name,
            "model_type": config.train.model_type,
            "model_variant": config.train.model_variant,
            "training_stage": config.train.training_stage,
            "record_type": record_type,
            "total_steps": int(total_steps),
            "update": int(update),
            "episode_index": int(episode_index),
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "average_reward": float(average_reward),
            "average_route_progress": float(average_route_progress),
            "route_progress": float(info.get("route_progress", 0.0)),
            "max_route_progress_in_episode": float(max_route_progress_in_episode),
            "done_reason": str(info.get("done_reason", "running")),
            "reached_goal": bool(info.get("reached_goal", False)),
            "goal_count": int(bool(info.get("reached_goal", False))),
            "collision": bool(info.get("collision", False)),
            "offroad": bool(info.get("offroad", False)),
            "stuck": bool(info.get("stuck", False)),
            "steer": actual_steer,
            "longitudinal": float(decoded["longitudinal"]),
            "throttle": actual_throttle,
            "brake": actual_brake,
            "speed_kmh": float(info.get("speed_kmh", 0.0)),
            "lane_offset": float(info.get("lane_offset", 0.0)),
            "heading_error": float(info.get("heading_error", 0.0)),
            "reward": float(info.get("reward", 0.0)),
            "checkpoint_path": checkpoint_path,
            "deterministic_rollout": bool(deterministic_rollout),
            "update_success": bool(update_success),
            "nan_detected": bool(nan_detected),
            "aux_mask_targets_available": bool(info.get("aux_mask_targets_available", False)),
            "aux_mask_target_source": str(info.get("aux_mask_target_source", "")),
            "aux_road_mask_positive_ratio": _mask_positive_ratio(info.get("aux_road_mask_target")),
            "aux_lane_edge_positive_ratio": _mask_positive_ratio(info.get("aux_lane_edge_target")),
            "corruption_level": float(info.get("corruption_level", 0.0)),
            "drop_rgb": float(info.get("drop_rgb", 0.0)),
            "drop_lidar": float(info.get("drop_lidar", 0.0)),
            "drop_ego": float(info.get("drop_ego", 0.0)),
            "drop_route": float(info.get("drop_route", 0.0)),
        }
    )
    for key, value in init_metadata.items():
        if key in row:
            row[key] = value
    if "log_std_dim0" in row and row["log_std_dim0"] == "":
        try:
            log_std = diagnostics.get("log_std_dim0")
            if log_std is not None:
                row["log_std_dim0"] = float(log_std)
        except (TypeError, ValueError):
            pass
    if "log_std_dim1" in row and row["log_std_dim1"] == "":
        try:
            log_std = diagnostics.get("log_std_dim1")
            if log_std is not None:
                row["log_std_dim1"] = float(log_std)
        except (TypeError, ValueError):
            pass
    for key, value in metrics.items():
        if key in row:
            row[key] = float(value)
    for key, value in diagnostics.items():
        if key in row:
            row[key] = float(value)
    for key in ("rgb", "lidar", "ego", "route"):
        row[f"fusion_{key}"] = float(diagnostics.get(f"fusion_weight_{key}", 0.0))
    writer.writerow(row)
    log_file.flush()


def main() -> int:
    args = parse_args()
    config_path = _resolve_config_path(str(args.config))
    config = load_phase1_config(config_path)
    if args.total_timesteps is not None:
        config.train.total_timesteps = int(args.total_timesteps)
    if args.device is not None:
        config.train.device = str(args.device)
    run_base_name = args.run_name or config.train.run_name
    run_name = f"{run_base_name}_{_timestamp()}"
    host = args.host or config.train.host
    port = int(args.port if args.port is not None else config.train.port)
    device = config.train.resolve_device(args.device)
    _set_seed(config.train.seed)

    logs_dir = _project_path(config.train.logs_dir) / run_name
    checkpoints_dir = _project_path(config.train.checkpoints_dir) / run_name
    output_dir = _project_path(config.train.output_root) / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = logs_dir / "training_log.csv"

    obs_spec = build_policy_observation_spec(
        rgb_shape=config.model.rgb_shape,
        lidar_shape=config.model.lidar_shape,
        ego_dim=config.model.ego_dim,
        route_dim=config.model.route_dim,
    )
    aux_spec = build_auxiliary_observation_spec(
        quality_stats_dim=config.model.quality_stats_dim,
        risk_stats_dim=config.model.risk_stats_dim,
        route_commitment_stats_dim=config.model.route_commitment_stats_dim,
        sensor_dropout_dim=config.model.sensor_dropout_dim,
    )
    agent = PPOAgent(PPOConfig.from_phase1_config(config, device_override=device))
    init_metadata: dict[str, Any] = {
        "init_checkpoint_path": "",
        "init_checkpoint_loaded": False,
        "init_checkpoint_strict": bool(args.init_strict),
        "init_load_optimizer": bool(args.load_init_optimizer),
        "init_missing_keys": "",
        "init_unexpected_keys": "",
        "init_loaded_key_count": 0,
        "init_skipped_key_count": 0,
        "init_dummy_forward_ok": False,
        "std_override_enabled": False,
        "std_override_source": "",
        "log_std_dim0": "",
        "log_std_dim1": "",
    }
    if args.init_checkpoint:
        init_checkpoint_path = _project_path(args.init_checkpoint)
        init_metadata = _load_init_checkpoint(
            agent,
            init_checkpoint_path,
            strict=bool(args.init_strict),
            load_optimizer=bool(args.load_init_optimizer),
        )
        log_std_override, log_std_override_source = _resolve_log_std_override(config, args)
        if log_std_override is not None:
            init_metadata.update(
                _override_model_log_std(
                    agent,
                    log_std_override,
                    source=log_std_override_source,
                )
            )
        _verify_init_forward(agent, obs_dim=obs_spec.total_dim, auxiliary_dim=aux_spec.total_dim)
        init_metadata["init_dummy_forward_ok"] = True
        print(
            "[init] "
            f"checkpoint={init_checkpoint_path} strict={bool(args.init_strict)} "
            f"load_optimizer={bool(args.load_init_optimizer)} "
            f"loaded_keys={init_metadata['init_loaded_key_count']} "
            f"skipped_keys={init_metadata['init_skipped_key_count']} "
            f"std_override={init_metadata.get('std_override_enabled', False)}"
        )
    trainable_module_names = _ppo_trainable_module_names(config)
    deterministic_rollout = _ppo_deterministic_rollout(config)
    if trainable_module_names:
        if bool(args.load_init_optimizer):
            raise ValueError("ppo.trainable_module_names cannot be combined with --load-init-optimizer")
        trainable_parameters = _select_trainable_parameters(agent, trainable_module_names)
        agent.optimizer = torch.optim.Adam(trainable_parameters, lr=float(config.train.learning_rate))
    if deterministic_rollout:
        print("[rollout] deterministic_action=true")
    buffer = RolloutBuffer(
        capacity=config.train.rollout_steps,
        obs_dim=obs_spec.total_dim,
        action_dim=config.model.action_dim,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        aux_mask_shape=(1, config.model.aux_mask_height, config.model.aux_mask_width),
        auxiliary_dim=aux_spec.total_dim,
        hidden_state_shape=(config.model.recurrent_layers, config.model.recurrent_hidden_dim),
    )
    aux_norm = _aux_normalization(config)
    env = None
    log_file = training_log_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(log_file, fieldnames=TRAINING_LOG_FIELDS)
    writer.writeheader()

    try:
        env = _build_env(config, host=host, port=port, output_dir=output_dir, run_name=run_base_name)
        reset_result = env.reset()
        observation = flatten_observation_dict(reset_result["obs"], obs_spec)
        auxiliary_observation = flatten_auxiliary_observation_dict(reset_result["obs"], aux_spec)
        last_info = dict(reset_result.get("info", {}))
        last_done_reason = str(last_info.get("done_reason", "running"))
        total_steps = 0
        update = 0
        episode_index = 0
        episode_reward = 0.0
        episode_length = 0
        max_route_progress_in_episode = 0.0
        completed_rewards: list[float] = []
        completed_progresses: list[float] = []

        while total_steps < int(config.train.total_timesteps):
            buffer.reset()
            for _ in range(int(config.train.rollout_steps)):
                current_info = dict(last_info)
                current_auxiliary = auxiliary_observation
                reliability_target, risk_target, corruption_level = build_phase2_targets_from_info(
                    current_info,
                    current_auxiliary,
                    spec=aux_spec,
                )
                dropout_slice = aux_spec.item("sensor_dropout_mask")
                dropout_mask = current_auxiliary[dropout_slice.start : dropout_slice.stop]
                action, log_prob, value = agent.act(
                    observation,
                    auxiliary_observation=current_auxiliary,
                    deterministic=deterministic_rollout,
                )
                diagnostics = agent.get_diagnostics()
                step_result = env.step(action, policy_diagnostics=diagnostics)
                info = dict(step_result["info"])
                reward = float(step_result["reward"])
                done = bool(step_result["done"])
                if not np.isfinite(reward):
                    raise RuntimeError(f"Encountered non-finite reward: {reward}")
                if not np.all(np.isfinite(action)):
                    raise RuntimeError(f"Encountered non-finite action: {action}")
                next_observation = flatten_observation_dict(step_result["obs"], obs_spec)
                next_auxiliary_observation = flatten_auxiliary_observation_dict(step_result["obs"], aux_spec)
                buffer.add(
                    observation=observation,
                    auxiliary_observation=current_auxiliary,
                    action=action,
                    hidden_state_before_action=agent.hidden_state_before_action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    aux_lane_offset_target=_normalize_aux_target(
                        float(info.get("lane_offset", 0.0)),
                        divisor=float(aux_norm["lane_offset_divisor_m"]),
                        clip_abs=None if aux_norm["lane_offset_clip_abs"] is None else float(aux_norm["lane_offset_clip_abs"]),
                    ),
                    aux_heading_error_target=_normalize_aux_target(
                        float(info.get("heading_error", 0.0)),
                        divisor=float(aux_norm["heading_error_divisor_deg"]),
                        clip_abs=(
                            None
                            if aux_norm["heading_error_clip_abs"] is None
                            else float(aux_norm["heading_error_clip_abs"])
                        ),
                    ),
                    aux_road_mask_target=info.get("aux_road_mask_target"),
                    aux_lane_edge_target=info.get("aux_lane_edge_target"),
                    aux_mask_targets_available=bool(info.get("aux_mask_targets_available", False)),
                    reliability_target=reliability_target,
                    risk_target=risk_target,
                    corruption_level=corruption_level,
                    dropout_mask=dropout_mask,
                    route_progress=float(info.get("route_progress", 0.0)),
                    speed_kmh=float(info.get("speed_kmh", 0.0)),
                )

                total_steps += 1
                episode_reward += reward
                episode_length += 1
                max_route_progress_in_episode = max(
                    max_route_progress_in_episode,
                    float(info.get("route_progress", 0.0)),
                )
                last_info = info
                last_done_reason = str(info.get("done_reason", "running"))
                observation = next_observation
                auxiliary_observation = next_auxiliary_observation

                if done:
                    completed_rewards.append(float(episode_reward))
                    completed_progresses.append(float(info.get("route_progress", 0.0)))
                    _write_training_log_row(
                        writer,
                        log_file,
                        config=config,
                        run_name=run_name,
                        record_type="episode",
                        total_steps=total_steps,
                        update=update,
                        episode_index=episode_index,
                        episode_reward=episode_reward,
                        episode_length=episode_length,
                        average_reward=float(np.mean(completed_rewards)),
                        average_route_progress=float(np.mean(completed_progresses)),
                        max_route_progress_in_episode=max_route_progress_in_episode,
                        info=info,
                        diagnostics=agent.get_diagnostics(),
                        init_metadata=init_metadata,
                        deterministic_rollout=deterministic_rollout,
                    )
                    print(
                        f"[episode] idx={episode_index} total_steps={total_steps} "
                        f"reward={episode_reward:.3f} length={episode_length} "
                        f"done_reason={last_done_reason} route_progress={float(info.get('route_progress', 0.0)):.3f}"
                    )
                    episode_index += 1
                    episode_reward = 0.0
                    episode_length = 0
                    max_route_progress_in_episode = 0.0
                    agent.reset_hidden()
                    reset_result = env.reset()
                    observation = flatten_observation_dict(reset_result["obs"], obs_spec)
                    auxiliary_observation = flatten_auxiliary_observation_dict(reset_result["obs"], aux_spec)
                    last_info = dict(reset_result.get("info", {}))

                if total_steps >= int(config.train.total_timesteps):
                    break

            if buffer.ptr == 0:
                break
            if buffer.dones[buffer.ptr - 1] >= 1.0:
                last_value = 0.0
                last_done = True
            else:
                last_value = agent.estimate_value(
                    observation,
                    auxiliary_observation=auxiliary_observation,
                    hidden_state=agent.hidden_state,
                )
                last_done = False
            buffer.compute_returns_and_advantages(last_value=last_value, last_done=last_done)
            metrics = agent.update(buffer.get_training_batch(agent.device))
            metrics.update(agent.get_diagnostics())
            update += 1
            checkpoint_path = ""
            if update % int(config.train.save_every_updates) == 0:
                checkpoint_path = str(checkpoints_dir / f"update_{update:04d}.pt")
                agent.save(
                    checkpoint_path,
                    extra_state={"update": update, "total_steps": total_steps, "episode_index": episode_index},
                )
            _write_training_log_row(
                writer,
                log_file,
                config=config,
                run_name=run_name,
                record_type="update",
                total_steps=total_steps,
                update=update,
                episode_index=episode_index,
                episode_reward=episode_reward,
                episode_length=episode_length,
                average_reward=float(np.mean(completed_rewards)) if completed_rewards else 0.0,
                average_route_progress=float(np.mean(completed_progresses)) if completed_progresses else 0.0,
                max_route_progress_in_episode=max_route_progress_in_episode,
                info=last_info,
                diagnostics=agent.get_diagnostics(),
                metrics=metrics,
                checkpoint_path=checkpoint_path,
                init_metadata=init_metadata,
                deterministic_rollout=deterministic_rollout,
            )
            print(
                f"[update] idx={update} total_steps={total_steps} "
                f"actor_loss={metrics.get('actor_loss', 0.0):.4f} "
                f"critic_loss={metrics.get('critic_loss', 0.0):.4f} "
                f"entropy={metrics.get('entropy', 0.0):.4f}"
            )

        final_checkpoint = str(checkpoints_dir / "last.pt")
        agent.save(
            final_checkpoint,
            extra_state={"update": update, "total_steps": total_steps, "episode_index": episode_index},
        )
        print(f"[done] logs={training_log_path} checkpoint={final_checkpoint}")
        return 0
    finally:
        if env is not None:
            env.close()
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
