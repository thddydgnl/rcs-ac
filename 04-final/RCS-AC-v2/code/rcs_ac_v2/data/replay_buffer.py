"""Simple in-memory sequence replay buffer for Stage 0 contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import torch


@dataclass
class EpisodeData:
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    raw_images: torch.Tensor
    raw_imu: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    done: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.action.shape[0])


@dataclass
class EpisodeBatch:
    privileged_bev: torch.Tensor
    privileged_ego: torch.Tensor
    raw_images: torch.Tensor
    raw_imu: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    done: torch.Tensor
    metadata: list[dict[str, Any]]


class SequenceReplayBuffer:
    def __init__(self) -> None:
        self._episodes: list[EpisodeData] = []

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def episodes(self) -> tuple[EpisodeData, ...]:
        return tuple(self._episodes)

    def add_episode(self, episode: EpisodeData) -> None:
        _validate_episode(episode)
        self._episodes.append(episode)

    def sample(self, batch_size: int, sequence_length: int, *, seed: int | None = None) -> EpisodeBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        candidates = [
            (episode_index, start)
            for episode_index, episode in enumerate(self._episodes)
            if episode.length >= sequence_length
            for start in range(episode.length - sequence_length + 1)
        ]
        if not candidates:
            raise ValueError("no episodes are long enough for the requested sequence_length")

        rng = random.Random(seed)
        selected = [rng.choice(candidates) for _ in range(batch_size)]
        slices: list[EpisodeData] = []
        metadata: list[dict[str, Any]] = []
        for episode_index, start in selected:
            episode = self._episodes[episode_index]
            end = start + sequence_length
            slices.append(_slice_episode(episode, start, end))
            metadata.append(
                {
                    **episode.metadata,
                    "episode_index": episode_index,
                    "sequence_start": start,
                    "sequence_end": end,
                }
            )

        return EpisodeBatch(
            privileged_bev=torch.stack([item.privileged_bev for item in slices], dim=0),
            privileged_ego=torch.stack([item.privileged_ego for item in slices], dim=0),
            raw_images=torch.stack([item.raw_images for item in slices], dim=0),
            raw_imu=torch.stack([item.raw_imu for item in slices], dim=0),
            action=torch.stack([item.action for item in slices], dim=0),
            reward=torch.stack([item.reward for item in slices], dim=0),
            cont=torch.stack([item.cont for item in slices], dim=0),
            done=torch.stack([item.done for item in slices], dim=0),
            metadata=metadata,
        )


def _validate_episode(episode: EpisodeData) -> None:
    tensors = {
        "privileged_bev": episode.privileged_bev,
        "privileged_ego": episode.privileged_ego,
        "raw_images": episode.raw_images,
        "raw_imu": episode.raw_imu,
        "action": episode.action,
        "reward": episode.reward,
        "cont": episode.cont,
        "done": episode.done,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.shape[0] != episode.length:
            raise ValueError(f"{name} length {tensor.shape[0]} does not match action length {episode.length}")
    if episode.action.dtype != torch.int64:
        raise ValueError("action must be torch.int64")
    if episode.done.dtype != torch.bool:
        raise ValueError("done must be torch.bool")
    for name in ("reward", "cont"):
        tensor = tensors[name]
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be floating point")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    if not torch.logical_or(episode.cont == 0.0, episode.cont == 1.0).all():
        raise ValueError("cont must contain only 0 or 1")
    done_indices = torch.nonzero(episode.done, as_tuple=False).flatten()
    if done_indices.numel() > 0:
        first_done = int(done_indices[0])
        if not (episode.cont[first_done:] == 0.0).all():
            raise ValueError("cont must remain 0 from the first done timestep onward")


def _slice_episode(episode: EpisodeData, start: int, end: int) -> EpisodeData:
    return EpisodeData(
        privileged_bev=episode.privileged_bev[start:end],
        privileged_ego=episode.privileged_ego[start:end],
        raw_images=episode.raw_images[start:end],
        raw_imu=episode.raw_imu[start:end],
        action=episode.action[start:end],
        reward=episode.reward[start:end],
        cont=episode.cont[start:end],
        done=episode.done[start:end],
        metadata=dict(episode.metadata),
    )
