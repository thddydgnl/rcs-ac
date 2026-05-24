"""Privileged CARLA replay IO for Stage 2 training.

These files are training artifacts, not clean-eval evidence. They carry the
same simulator-state semantic43/ego8 provenance that the clean runner requires
so synthetic smoke checkpoints cannot be promoted by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from rcs_ac_v2.data.schema import PrivilegedSequenceBatch, SchemaConfig, validate_privileged_batch
from rcs_ac_v2.evaluation.privileged_observation import (
    EGO8_FIELD_NAMES,
    EGO8_SCHEMA_VERSION,
    PRIVILEGED_OBSERVATION_SOURCE,
    SEMANTIC43_CHANNEL_NAMES_SHA256,
    SEMANTIC43_SCHEMA_VERSION,
)
from rcs_ac_v2.utils.checkpoint import sha256_file


REPLAY_SOURCE = "windows_carla_privileged_replay"
REPLAY_FORMAT_VERSION = "stage2_privileged_replay_npz_v1"
REPLAY_COLLECTOR_NAME = "collect_stage2_privileged_replay"


@dataclass(frozen=True)
class PrivilegedReplayEpisode:
    path: Path
    privileged_bev: np.ndarray
    privileged_ego: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    cont: np.ndarray
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.action.shape[0])


def default_replay_metadata(*, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        "format_version": REPLAY_FORMAT_VERSION,
        "source": REPLAY_SOURCE,
        "clean_gate_evidence": False,
        "privileged_observation_schema": SEMANTIC43_SCHEMA_VERSION,
        "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
        "privileged_ego_schema": EGO8_SCHEMA_VERSION,
        "privileged_ego_fields": list(EGO8_FIELD_NAMES),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_privileged_replay_episode(
    path: str | Path,
    *,
    privileged_bev: np.ndarray,
    privileged_ego: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    cont: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    episode = PrivilegedReplayEpisode(
        path=target,
        privileged_bev=np.asarray(privileged_bev, dtype=np.float32),
        privileged_ego=np.asarray(privileged_ego, dtype=np.float32),
        action=np.asarray(action, dtype=np.int64),
        reward=np.asarray(reward, dtype=np.float32),
        cont=np.asarray(cont, dtype=np.float32),
        metadata=default_replay_metadata(extra=metadata or {}),
    )
    validate_replay_episode(episode)
    np.savez_compressed(
        target,
        privileged_bev=episode.privileged_bev,
        privileged_ego=episode.privileged_ego,
        action=episode.action,
        reward=episode.reward,
        cont=episode.cont,
        metadata_json=json.dumps(episode.metadata, sort_keys=True),
    )
    return target


def load_privileged_replay_episode(path: str | Path) -> PrivilegedReplayEpisode:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"]
        metadata_text = str(metadata_raw.item() if metadata_raw.shape == () else metadata_raw.tolist())
        episode = PrivilegedReplayEpisode(
            path=source,
            privileged_bev=np.asarray(data["privileged_bev"], dtype=np.float32),
            privileged_ego=np.asarray(data["privileged_ego"], dtype=np.float32),
            action=np.asarray(data["action"], dtype=np.int64),
            reward=np.asarray(data["reward"], dtype=np.float32),
            cont=np.asarray(data["cont"], dtype=np.float32),
            metadata=json.loads(metadata_text),
        )
    validate_replay_episode(episode)
    return episode


def validate_replay_episode(episode: PrivilegedReplayEpisode) -> None:
    bev = episode.privileged_bev
    ego = episode.privileged_ego
    action = episode.action
    reward = episode.reward
    cont = episode.cont
    if bev.ndim != 4:
        raise ValueError(f"privileged_bev must be [T,43,H,W], got {bev.shape}")
    if bev.shape[1] != 43:
        raise ValueError(f"privileged_bev must have 43 channels, got {bev.shape[1]}")
    if ego.ndim != 2 or ego.shape[1] != 8:
        raise ValueError(f"privileged_ego must be [T,8], got {ego.shape}")
    length = int(bev.shape[0])
    for name, array in (("privileged_ego", ego), ("action", action), ("reward", reward), ("cont", cont)):
        if int(array.shape[0]) != length:
            raise ValueError(f"{name} length must match privileged_bev length")
    if action.ndim != 1 or action.dtype != np.int64:
        raise ValueError("action must be int64 [T]")
    if reward.ndim != 1 or cont.ndim != 1:
        raise ValueError("reward and cont must be [T]")
    if not np.isfinite(bev).all() or not np.isfinite(ego).all():
        raise ValueError("privileged replay contains non-finite observation values")
    if not np.isfinite(reward).all() or not np.isfinite(cont).all():
        raise ValueError("privileged replay contains non-finite reward/continue values")
    if action.size > 0 and (int(action.min()) < 0 or int(action.max()) >= 39):
        raise ValueError("action ids must be in [0,38]")
    if not np.logical_or(cont == 0.0, cont == 1.0).all():
        raise ValueError("cont must contain only 0 or 1")
    metadata = episode.metadata
    if metadata.get("source") != REPLAY_SOURCE:
        raise ValueError("privileged replay source must be windows_carla_privileged_replay")
    if metadata.get("privileged_observation_schema") != SEMANTIC43_SCHEMA_VERSION:
        raise ValueError("privileged replay semantic schema mismatch")
    if metadata.get("privileged_bev_channel_names_sha256") != SEMANTIC43_CHANNEL_NAMES_SHA256:
        raise ValueError("privileged replay semantic channel hash mismatch")


def find_replay_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.npz")))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def load_replay_episodes(paths: Iterable[str | Path]) -> list[PrivilegedReplayEpisode]:
    files = find_replay_files(paths)
    if not files:
        raise ValueError("no privileged replay .npz files found")
    return [load_privileged_replay_episode(path) for path in files]


def replay_file_manifest(files: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in files
    ]


def write_privileged_replay_manifest(
    path: str | Path,
    *,
    config_path: str | Path,
    episode_paths: Iterable[Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    episodes = [load_privileged_replay_episode(item) for item in episode_paths]
    target = Path(path)
    payload: dict[str, Any] = {
        "source": REPLAY_SOURCE,
        "clean_gate_evidence": False,
        "collector": REPLAY_COLLECTOR_NAME,
        "privileged_observation_source": PRIVILEGED_OBSERVATION_SOURCE,
        "privileged_observation_schema": SEMANTIC43_SCHEMA_VERSION,
        "privileged_bev_channel_names_sha256": SEMANTIC43_CHANNEL_NAMES_SHA256,
        "privileged_ego_schema": EGO8_SCHEMA_VERSION,
        "config": str(config_path),
        "config_sha256": sha256_file(Path(config_path)),
        "episodes": len(episodes),
        "episode_lengths": [episode.length for episode in episodes],
        "total_steps": sum(episode.length for episode in episodes),
        "files": replay_file_manifest([episode.path for episode in episodes]),
    }
    if extra:
        payload.update(extra)
    validate_replay_manifest_payload(payload, manifest_dir=target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def load_replay_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("privileged replay manifest must be a JSON object")
    validate_replay_manifest_payload(payload, manifest_dir=manifest_path.parent)
    return payload


def validate_replay_manifest(
    path: str | Path,
    *,
    replay_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = load_replay_manifest(manifest_path)
    manifest_files = {_resolve_manifest_file(entry, manifest_path.parent).resolve() for entry in payload["files"]}
    if replay_paths is not None:
        requested_files = {item.resolve() for item in find_replay_files(replay_paths)}
        if manifest_files != requested_files:
            raise ValueError("privileged replay manifest files do not match requested replay paths")
    for entry in payload["files"]:
        load_privileged_replay_episode(_resolve_manifest_file(entry, manifest_path.parent))
    return payload


def validate_replay_manifest_payload(payload: dict[str, Any], *, manifest_dir: Path) -> None:
    if payload.get("source") != REPLAY_SOURCE:
        raise ValueError("privileged replay manifest source must be windows_carla_privileged_replay")
    if payload.get("clean_gate_evidence") is not False:
        raise ValueError("privileged replay manifest must not be clean gate evidence")
    if payload.get("collector") != REPLAY_COLLECTOR_NAME:
        raise ValueError("privileged replay manifest must be written by collect_stage2_privileged_replay")
    if payload.get("privileged_observation_source") != PRIVILEGED_OBSERVATION_SOURCE:
        raise ValueError("privileged replay manifest must use carla_simulator_state observations")
    if payload.get("privileged_observation_schema") != SEMANTIC43_SCHEMA_VERSION:
        raise ValueError("privileged replay manifest semantic schema mismatch")
    if payload.get("privileged_bev_channel_names_sha256") != SEMANTIC43_CHANNEL_NAMES_SHA256:
        raise ValueError("privileged replay manifest semantic channel hash mismatch")
    if payload.get("privileged_ego_schema") != EGO8_SCHEMA_VERSION:
        raise ValueError("privileged replay manifest ego schema mismatch")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("privileged replay manifest must list replay files")
    episode_lengths = payload.get("episode_lengths")
    if not isinstance(episode_lengths, list) or len(episode_lengths) != len(files):
        raise ValueError("privileged replay manifest episode_lengths must match files")
    if int(payload.get("episodes", -1)) != len(files):
        raise ValueError("privileged replay manifest episode count mismatch")
    if int(payload.get("total_steps", -1)) != sum(int(item) for item in episode_lengths):
        raise ValueError("privileged replay manifest total_steps mismatch")
    config_sha = str(payload.get("config_sha256", ""))
    if len(config_sha) != 64:
        raise ValueError("privileged replay manifest must include config_sha256")
    git_commit = str(payload.get("git_commit", ""))
    if len(git_commit) < 7 or git_commit == "UNSET":
        raise ValueError("privileged replay manifest must include git_commit")
    for entry in files:
        replay_path = _resolve_manifest_file(entry, manifest_dir)
        if not replay_path.is_file():
            raise ValueError(f"privileged replay file missing: {replay_path}")
        if int(entry.get("bytes", -1)) != int(replay_path.stat().st_size):
            raise ValueError(f"privileged replay file byte mismatch: {replay_path}")
        if str(entry.get("sha256", "")) != sha256_file(replay_path):
            raise ValueError(f"privileged replay file sha256 mismatch: {replay_path}")


def _resolve_manifest_file(entry: dict[str, Any], manifest_dir: Path) -> Path:
    if not isinstance(entry, dict):
        raise ValueError("privileged replay manifest file entry must be a mapping")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("privileged replay manifest file entry missing path")
    path = Path(raw_path)
    if path.exists():
        return path
    sibling = manifest_dir / path.name
    if sibling.exists():
        return sibling
    return path


class PrivilegedReplaySampler:
    def __init__(
        self,
        episodes: list[PrivilegedReplayEpisode],
        *,
        sequence_length: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.episodes = [episode for episode in episodes if episode.length >= sequence_length]
        if not self.episodes:
            raise ValueError("no replay episodes are long enough for the requested sequence_length")
        self.sequence_length = int(sequence_length)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

    def sample(self, batch_size: int) -> PrivilegedSequenceBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        bev_batch: list[np.ndarray] = []
        ego_batch: list[np.ndarray] = []
        action_batch: list[np.ndarray] = []
        reward_batch: list[np.ndarray] = []
        cont_batch: list[np.ndarray] = []
        for _ in range(batch_size):
            episode = self.episodes[int(self.rng.integers(0, len(self.episodes)))]
            start = int(self.rng.integers(0, episode.length - self.sequence_length + 1))
            end = start + self.sequence_length
            bev_batch.append(episode.privileged_bev[start:end])
            ego_batch.append(episode.privileged_ego[start:end])
            action_batch.append(episode.action[start:end])
            reward_batch.append(episode.reward[start:end])
            cont_batch.append(episode.cont[start:end])

        batch = PrivilegedSequenceBatch(
            privileged_bev=torch.as_tensor(np.stack(bev_batch), dtype=torch.float32, device=self.device),
            privileged_ego=torch.as_tensor(np.stack(ego_batch), dtype=torch.float32, device=self.device),
            action=torch.as_tensor(np.stack(action_batch), dtype=torch.int64, device=self.device),
            reward=torch.as_tensor(np.stack(reward_batch), dtype=torch.float32, device=self.device),
            cont=torch.as_tensor(np.stack(cont_batch), dtype=torch.float32, device=self.device),
        )
        validate_privileged_batch(batch, SchemaConfig(ego_dim=8))
        return batch
