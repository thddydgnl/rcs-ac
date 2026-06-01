"""Sequence sampling helpers."""

from __future__ import annotations

from rcs_ac_v2.data.replay_buffer import EpisodeBatch, SequenceReplayBuffer


def sample_sequence(
    buffer: SequenceReplayBuffer,
    *,
    batch_size: int,
    sequence_length: int,
    seed: int | None = None,
) -> EpisodeBatch:
    return buffer.sample(batch_size=batch_size, sequence_length=sequence_length, seed=seed)
