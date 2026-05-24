"""Windows CARLA adapter factory for Stage 2 privileged clean eval.

The factory is intentionally lazy: importing this module must not import CARLA
or shared_core. Windows-only dependencies are imported only when the factory is
called by the runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import sys

import numpy as np
import torch

from rcs_ac_v2.evaluation.errors import CleanEvalRuntimeFailure
from rcs_ac_v2.evaluation.privileged_observation import (
    EGO8_SCHEMA_VERSION,
    SEMANTIC43_CHANNEL_NAMES_SHA256,
    SEMANTIC43_SCHEMA_VERSION,
    build_carla_simulator_semantic43_observation,
)
from rcs_ac_v2.privileged_stream.policy import PrivilegedPolicy, PrivilegedPolicyConfig
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.utils.checkpoint import load_checkpoint


DEFAULT_BEV_KEYS = (
    "privileged_bev",
    "birdview",
    "birdview_bev",
    "bev",
    "gt_bev",
    "semantic_bev",
)
DEFAULT_EGO_KEYS = (
    "privileged_ego",
    "ego",
    "ego_state",
    "state",
)


class Stage2CarlaObservationAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        obs_config = config.get("carla", {}).get("privileged_observation", {})
        self.mode = str(obs_config.get("mode", "existing_keys"))
        self.bev_keys = tuple(obs_config.get("bev_keys", DEFAULT_BEV_KEYS))
        self.ego_keys = tuple(obs_config.get("ego_keys", DEFAULT_EGO_KEYS))
        contract = config.get("observation_contract", {})
        self.expected_bev_shape = tuple(contract.get("privileged_bev_shape", (1, 1, 43, 16, 16)))
        self.expected_ego_shape = tuple(contract.get("privileged_ego_shape", (1, 1, 8)))
        self.last_diagnostics: dict[str, Any] = {}

    def adapt(self, obs: dict[str, Any], info: dict[str, Any], *, simulator_context: Any | None = None) -> dict[str, Any]:
        bev_value = _first_present(obs, info, self.bev_keys)
        ego_value = _first_present(obs, info, self.ego_keys)
        if bev_value is None or ego_value is None:
            simulator_modes = {"auto_existing_or_simulator_state_semantic43", "simulator_state_semantic43"}
            if self.mode in simulator_modes:
                try:
                    simulator_obs = build_carla_simulator_semantic43_observation(
                        raw_env=simulator_context,
                        expected_bev_shape=self.expected_bev_shape,
                        expected_ego_shape=self.expected_ego_shape,
                    )
                except Exception as exc:
                    missing = "privileged_bev_missing" if bev_value is None else "privileged_ego_missing"
                    raise CleanEvalRuntimeFailure(
                        "adapter",
                        "privileged_simulator_state_observation_failed",
                        f"{missing}; simulator schema {SEMANTIC43_SCHEMA_VERSION} failed: {type(exc).__name__}: {exc}",
                    ) from exc
                adapted = dict(obs)
                adapted["privileged_bev"] = simulator_obs.privileged_bev
                adapted["privileged_ego"] = simulator_obs.privileged_ego
                self.last_diagnostics = dict(simulator_obs.diagnostics)
                return adapted
            if bev_value is None:
                raise CleanEvalRuntimeFailure(
                    "adapter",
                    "privileged_bev_missing",
                    f"CARLA obs/info missing privileged BEV keys {self.bev_keys}",
                )
            raise CleanEvalRuntimeFailure(
                "adapter",
                "privileged_ego_missing",
                f"CARLA obs/info missing privileged ego keys {self.ego_keys}",
            )
        bev = _as_stage2_bev(bev_value, expected_shape=self.expected_bev_shape)
        ego = _as_stage2_ego(ego_value, expected_shape=self.expected_ego_shape)
        adapted = dict(obs)
        adapted["privileged_bev"] = bev
        adapted["privileged_ego"] = ego
        contract = self.config.get("observation_contract", {})
        self.last_diagnostics = {
            "privileged_observation_schema": str(contract.get("privileged_bev_schema_version", SEMANTIC43_SCHEMA_VERSION)),
            "privileged_observation_mode": self.mode,
            "privileged_observation_source": "existing_env_keys",
            "privileged_observation_clean_gate_eligible": True,
            "privileged_bev_channel_names_sha256": str(
                contract.get("privileged_bev_channel_names_sha256", SEMANTIC43_CHANNEL_NAMES_SHA256)
            ),
            "privileged_ego_schema": str(contract.get("privileged_ego_schema_version", EGO8_SCHEMA_VERSION)),
        }
        return adapted


class SharedCoreCarlaEnvAdapter:
    def __init__(self, raw_env: Any, config: dict[str, Any]) -> None:
        self.raw_env = raw_env
        self.config = config
        self.observation_adapter = Stage2CarlaObservationAdapter(config)
        carla_config = config.get("carla", {})
        self.env_action_format = str(carla_config.get("env_action_format", "direct_3d"))
        self.allow_reverse_direct_3d = bool(carla_config.get("allow_reverse_direct_3d", False))

    def reset(self) -> dict[str, Any]:
        result = self.raw_env.reset()
        obs, info = _split_env_result(result, context="reset")
        adapted = self.observation_adapter.adapt(obs, info, simulator_context=self.raw_env)
        info = dict(info)
        info.update(self.observation_adapter.last_diagnostics)
        return {"obs": adapted, "info": info}

    def step(self, control: dict[str, float | bool], policy_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
        env_action = self._control_to_env_action(control)
        try:
            result = self.raw_env.step(env_action, policy_diagnostics=policy_diagnostics or {})
        except TypeError:
            result = self.raw_env.step(env_action)
        obs, info, reward, done, truncated = _split_step_result(result)
        adapted = self.observation_adapter.adapt(obs, info, simulator_context=self.raw_env)
        info = dict(info)
        info.update(self.observation_adapter.last_diagnostics)
        return {
            "obs": adapted,
            "info": info,
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }

    def close(self) -> None:
        if hasattr(self.raw_env, "close"):
            self.raw_env.close()

    def _control_to_env_action(self, control: dict[str, float | bool]) -> Any:
        if self.env_action_format == "raw2drive_control_dict":
            return dict(control)
        if self.env_action_format == "direct_3d":
            return _control_to_direct_3d(control, allow_reverse=self.allow_reverse_direct_3d)
        if self.env_action_format == "stage4_v2":
            direct = _control_to_direct_3d(control, allow_reverse=self.allow_reverse_direct_3d)
            return np.asarray([direct[0], direct[1] - direct[2]], dtype=np.float32)
        raise CleanEvalRuntimeFailure(
            "contract",
            "unsupported_env_action_format",
            f"unsupported env_action_format={self.env_action_format!r}",
        )


class Stage2PrivilegedPolicyRunner:
    def __init__(self, world_model: PrivilegedWorldModel, policy: PrivilegedPolicy, *, device: torch.device) -> None:
        self.world_model = world_model.to(device).eval()
        self.policy = policy.to(device).eval()
        self.device = device
        self._embeds: list[torch.Tensor] = []
        self._previous_actions: list[int] = []

    def reset(self) -> None:
        self._embeds.clear()
        self._previous_actions.clear()

    def act(self, obs: dict[str, Any], info: dict[str, Any]) -> dict[str, torch.Tensor]:
        del info
        bev = torch.as_tensor(obs["privileged_bev"], dtype=torch.float32, device=self.device)
        ego = torch.as_tensor(obs["privileged_ego"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embed = self.world_model.encoder(bev, ego)[:, 0]
            self._embeds.append(embed.detach())
            embeds = torch.stack(self._embeds, dim=1)
            action_ids = list(self._previous_actions) + [0]
            actions = torch.as_tensor(action_ids, dtype=torch.int64, device=self.device).reshape(1, -1)
            sequence = self.world_model.rssm.observe(embeds, actions)
            latent = sequence.latent[:, -1:].detach()
            output = self.policy(latent)
        return {"logits": output.logits.detach().cpu()}

    def observe_action(self, action_id: int) -> None:
        self._previous_actions.append(int(action_id))


def create_stage2_carla_adapter(config: dict[str, Any]) -> tuple[SharedCoreCarlaEnvAdapter, Stage2PrivilegedPolicyRunner]:
    """Factory used by evaluate_clean --run-carla on Windows."""

    policy = load_stage2_policy_runner(config)
    raw_env = _build_shared_core_env(config)
    return create_stage2_adapter_bundle(config=config, raw_env=raw_env, policy=policy)


def create_stage2_adapter_bundle(
    *,
    config: dict[str, Any],
    raw_env: Any,
    policy: Stage2PrivilegedPolicyRunner | None = None,
) -> tuple[SharedCoreCarlaEnvAdapter, Stage2PrivilegedPolicyRunner]:
    return SharedCoreCarlaEnvAdapter(raw_env, config), policy or load_stage2_policy_runner(config)


def load_stage2_policy_runner(config: dict[str, Any]) -> Stage2PrivilegedPolicyRunner:
    checkpoints = config.get("checkpoints", {})
    world_model_entry = _require_checkpoint_entry(checkpoints, "world_model")
    policy_entry = _require_checkpoint_entry(checkpoints, "policy")
    _require_sha256_entry(world_model_entry, "world_model")
    _require_sha256_entry(policy_entry, "policy")
    _verify_sha256(world_model_entry)
    _verify_sha256(policy_entry)

    device = _resolve_device(str(config.get("device", "auto")))
    world_model_payload = _load_checkpoint_checked(world_model_entry["path"], map_location=device, role="world_model")
    world_model_cfg_raw = _require_payload_mapping(world_model_payload, "config", role="world_model")
    world_model_cfg = PrivilegedWorldModelConfig(**world_model_cfg_raw)
    _validate_world_model_config(world_model_cfg, config)
    world_model = PrivilegedWorldModel(world_model_cfg)
    _load_state_dict_checked(
        world_model,
        _state_dict_from_payload(world_model_payload, preferred="model_state_dict", role="world_model"),
        role="world_model",
    )

    policy_payload = _load_checkpoint_checked(policy_entry["path"], map_location=device, role="policy")
    policy_cfg_raw = policy_payload.get("config") or policy_payload.get("policy_config")
    if policy_cfg_raw is None:
        raise CleanEvalRuntimeFailure("model", "policy_config_missing", "policy checkpoint missing config")
    if not isinstance(policy_cfg_raw, dict):
        raise CleanEvalRuntimeFailure("model", "policy_config_invalid", "policy config must be a mapping")
    policy_cfg = PrivilegedPolicyConfig(**policy_cfg_raw)
    if policy_cfg.latent_dim != world_model.rssm.latent_dim:
        raise CleanEvalRuntimeFailure(
            "model",
            "policy_latent_dim_mismatch",
            f"policy latent_dim={policy_cfg.latent_dim}, world_model latent_dim={world_model.rssm.latent_dim}",
        )
    policy = PrivilegedPolicy(policy_cfg)
    _load_state_dict_checked(
        policy,
        _state_dict_from_payload(policy_payload, preferred="model_state_dict", role="policy"),
        role="policy",
    )
    _validate_checkpoint_provenance(
        world_model_payload=world_model_payload,
        policy_payload=policy_payload,
        config=config,
    )
    return Stage2PrivilegedPolicyRunner(world_model, policy, device=device)


def _build_shared_core_env(config: dict[str, Any]) -> Any:
    carla_config = config.get("carla", {})
    _extend_pythonpath(carla_config.get("pythonpath_entries", ()))
    env_config_path = str(carla_config.get("env_config_path", "")).strip()
    if not env_config_path:
        raise CleanEvalRuntimeFailure("adapter", "carla_env_config_missing", "carla.env_config_path is required")

    try:
        from rcs_ac_first.runtime.config import load_phase1_config
        from rcs_ac_first.scripts.train import _build_env, _resolve_config_path
    except Exception as exc:  # pragma: no cover - Windows dependency boundary.
        raise CleanEvalRuntimeFailure(
            "infrastructure",
            "rcs_ac_first_import_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        env_config = load_phase1_config(_resolve_config_path(env_config_path))
        output_dir = Path(config["output_dir"])
        run_name = str(carla_config.get("run_name", "rcs_ac_v2_stage2_clean_128"))
        env = _build_env(
            env_config,
            host=str(carla_config.get("host", "127.0.0.1")),
            port=int(carla_config.get("port", 2000)),
            output_dir=output_dir,
            run_name=run_name,
        )
    except CleanEvalRuntimeFailure:
        raise
    except Exception as exc:  # pragma: no cover - Windows dependency boundary.
        raise CleanEvalRuntimeFailure(
            "CARLA",
            "shared_core_env_build_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    fixed_delta_seconds = carla_config.get("fixed_delta_seconds", config.get("fixed_delta_seconds"))
    if fixed_delta_seconds is not None:
        setattr(env, "fixed_delta_seconds", float(fixed_delta_seconds))
    return env


def _extend_pythonpath(entries: Any) -> None:
    if not isinstance(entries, (list, tuple)):
        return
    for entry in reversed(entries):
        path = Path(str(entry))
        if not path.is_absolute():
            path = Path.cwd() / path
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _require_checkpoint_entry(checkpoints: dict[str, Any], name: str) -> dict[str, Any]:
    entry = checkpoints.get(name)
    if not isinstance(entry, dict) or not entry.get("path"):
        raise CleanEvalRuntimeFailure("model", f"{name}_checkpoint_missing", f"checkpoints.{name}.path is required")
    return entry


def _require_sha256_entry(entry: dict[str, Any], name: str) -> None:
    expected = entry.get("sha256")
    if expected in (None, "", "UNSET"):
        raise CleanEvalRuntimeFailure("model", f"{name}_checkpoint_sha256_missing", f"checkpoints.{name}.sha256 is required")


def _verify_sha256(entry: dict[str, Any]) -> None:
    expected = entry.get("sha256")
    path = Path(entry["path"])
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CleanEvalRuntimeFailure(
            "model",
            "checkpoint_file_unreadable",
            f"{type(exc).__name__}: {path}",
        ) from exc
    if actual != str(expected):
        raise CleanEvalRuntimeFailure("model", "checkpoint_sha256_mismatch", f"{path}: {actual} != {expected}")


def _load_checkpoint_checked(path: str | Path, *, map_location: torch.device, role: str) -> dict[str, Any]:
    try:
        return load_checkpoint(path, map_location=map_location)
    except CleanEvalRuntimeFailure:
        raise
    except Exception as exc:
        raise CleanEvalRuntimeFailure(
            "model",
            f"{role}_checkpoint_load_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _require_payload_mapping(payload: dict[str, Any], key: str, *, role: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise CleanEvalRuntimeFailure("model", f"{role}_{key}_missing", f"{role} checkpoint missing mapping {key!r}")
    return value


def _state_dict_from_payload(payload: dict[str, Any], *, preferred: str, role: str) -> dict[str, Any]:
    for key in (preferred, "policy_state_dict", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    raise CleanEvalRuntimeFailure("model", f"{role}_state_dict_missing", f"{role} checkpoint missing model state dict")


def _load_state_dict_checked(module: torch.nn.Module, state_dict: dict[str, Any], *, role: str) -> None:
    try:
        module.load_state_dict(state_dict)
    except Exception as exc:
        raise CleanEvalRuntimeFailure(
            "model",
            f"{role}_state_dict_load_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _validate_checkpoint_provenance(
    *,
    world_model_payload: dict[str, Any],
    policy_payload: dict[str, Any],
    config: dict[str, Any],
) -> None:
    contract = config.get("checkpoint_contract", {})
    if not contract.get("require_trained_provenance_for_clean_gate", False):
        return
    allowed_sources = {
        str(item)
        for item in contract.get("allowed_sources", ("windows_carla_privileged_replay",))
    }
    rejected_sources = {
        str(item)
        for item in contract.get("rejected_sources", ("local_synthetic_smoke_not_carla",))
    }
    for role, payload in (("world_model", world_model_payload), ("policy", policy_payload)):
        source = str(payload.get("source", "UNSET"))
        if source in rejected_sources:
            raise CleanEvalRuntimeFailure(
                "model",
                "synthetic_checkpoint_not_clean_evidence",
                f"{role} checkpoint source {source!r} is not eligible for clean eval.",
            )
        if source not in allowed_sources:
            raise CleanEvalRuntimeFailure(
                "model",
                f"{role}_checkpoint_provenance_invalid",
                f"{role} checkpoint source {source!r} must be one of {sorted(allowed_sources)}.",
            )
        if str(payload.get("stage", "UNSET")) != "stage2_privileged_policy_clean":
            raise CleanEvalRuntimeFailure(
                "model",
                f"{role}_checkpoint_stage_invalid",
                f"{role} checkpoint stage must be stage2_privileged_policy_clean.",
            )


def _validate_world_model_config(cfg: PrivilegedWorldModelConfig, config: dict[str, Any]) -> None:
    expected_bev_shape = tuple(config.get("observation_contract", {}).get("privileged_bev_shape", (1, 1, 43, 16, 16)))
    expected_ego_shape = tuple(config.get("observation_contract", {}).get("privileged_ego_shape", (1, 1, 8)))
    expected_channels = int(expected_bev_shape[2]) if len(expected_bev_shape) == 5 else 43
    expected_height = int(expected_bev_shape[3]) if len(expected_bev_shape) == 5 else 16
    expected_width = int(expected_bev_shape[4]) if len(expected_bev_shape) == 5 else 16
    expected_ego_dim = int(expected_ego_shape[2]) if len(expected_ego_shape) == 3 else 8
    expected_action_dim = int(config.get("action_dim", 39))
    mismatches = {
        "bev_channels": (cfg.bev_channels, expected_channels),
        "bev_height": (cfg.bev_height, expected_height),
        "bev_width": (cfg.bev_width, expected_width),
        "ego_dim": (cfg.ego_dim, expected_ego_dim),
        "action_dim": (cfg.action_dim, expected_action_dim),
    }
    bad = {name: values for name, values in mismatches.items() if int(values[0]) != int(values[1])}
    if bad:
        raise CleanEvalRuntimeFailure("model", "world_model_config_mismatch", f"world model config mismatch: {bad}")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _first_present(obs: dict[str, Any], info: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in obs:
            return obs[key]
        if key in info:
            return info[key]
    return None


def _as_stage2_bev(value: Any, *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != expected_shape:
        raise CleanEvalRuntimeFailure(
            "adapter",
            "privileged_bev_shape_invalid",
            f"expected {expected_shape}, got {tuple(tensor.shape)}",
        )
    if not torch.isfinite(tensor).all():
        raise CleanEvalRuntimeFailure("adapter", "privileged_bev_non_finite", "privileged_bev contains NaN/inf")
    return tensor.cpu()


def _as_stage2_ego(value: Any, *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != expected_shape:
        raise CleanEvalRuntimeFailure(
            "adapter",
            "privileged_ego_shape_invalid",
            f"expected {expected_shape}, got {tuple(tensor.shape)}",
        )
    if not torch.isfinite(tensor).all():
        raise CleanEvalRuntimeFailure("adapter", "privileged_ego_non_finite", "privileged_ego contains NaN/inf")
    return tensor.cpu()


def _split_env_result(result: Any, *, context: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(result, dict):
        obs = result.get("obs", {})
        info = result.get("info", {})
    elif isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        raise CleanEvalRuntimeFailure("adapter", f"{context}_return_invalid", f"{context} returned {type(result)}")
    if not isinstance(obs, dict) or not isinstance(info, dict):
        raise CleanEvalRuntimeFailure("adapter", f"{context}_obs_info_invalid", "obs/info must be dicts")
    return obs, info


def _split_step_result(result: Any) -> tuple[dict[str, Any], dict[str, Any], float, bool, bool]:
    if isinstance(result, dict):
        obs = result.get("obs", {})
        info = result.get("info", {})
        reward = float(result.get("reward", info.get("reward", 0.0)))
        done = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
    elif isinstance(result, tuple) and len(result) == 5:
        obs, reward, done, truncated, info = result
        reward = float(reward)
        done = bool(done)
        truncated = bool(truncated)
    else:
        raise CleanEvalRuntimeFailure("adapter", "step_return_invalid", f"step returned {type(result)}")
    if not isinstance(obs, dict) or not isinstance(info, dict):
        raise CleanEvalRuntimeFailure("adapter", "step_obs_info_invalid", "step obs/info must be dicts")
    return obs, info, reward, done, truncated


def _control_to_direct_3d(control: dict[str, float | bool], *, allow_reverse: bool) -> np.ndarray:
    if bool(control.get("reverse", False)) and not allow_reverse:
        raise CleanEvalRuntimeFailure(
            "contract",
            "reverse_action_not_supported_by_direct_3d",
            "shared direct_3d action format cannot preserve Raw2Drive reverse=True",
        )
    return np.asarray(
        [
            float(control["steer"]),
            float(control["throttle"]),
            float(control["brake"]),
        ],
        dtype=np.float32,
    )
