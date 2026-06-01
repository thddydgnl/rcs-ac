from __future__ import annotations

import pytest
import torch

from rcs_ac_v2.data.replay_buffer import EpisodeData, SequenceReplayBuffer


def make_episode(length: int, *, episode_id: str = "ep", done_at: int | None = None) -> EpisodeData:
    done = torch.zeros(length, dtype=torch.bool)
    cont = torch.ones(length, dtype=torch.float32)
    if done_at is not None:
        done[done_at:] = True
        cont[done_at:] = 0.0
    action = torch.arange(length, dtype=torch.int64) % 39
    return EpisodeData(
        privileged_bev=torch.zeros(length, 43, 8, 8),
        privileged_ego=torch.zeros(length, 5),
        raw_images=torch.zeros(length, 2, 3, 8, 8),
        raw_imu=torch.zeros(length, 6),
        action=action,
        reward=torch.arange(length, dtype=torch.float32),
        cont=cont,
        done=done,
        metadata={"episode_id": episode_id},
    )


def test_add_episode_and_len() -> None:
    buffer = SequenceReplayBuffer()
    buffer.add_episode(make_episode(5))
    assert len(buffer) == 1


def test_short_episode_is_excluded_from_sampling_candidates() -> None:
    buffer = SequenceReplayBuffer()
    buffer.add_episode(make_episode(2, episode_id="short"))
    buffer.add_episode(make_episode(5, episode_id="long"))
    batch = buffer.sample(batch_size=4, sequence_length=3, seed=0)
    assert batch.action.shape == (4, 3)
    assert {item["episode_id"] for item in batch.metadata} == {"long"}


def test_sampling_is_deterministic_with_seed() -> None:
    buffer = SequenceReplayBuffer()
    buffer.add_episode(make_episode(8, episode_id="a"))
    buffer.add_episode(make_episode(8, episode_id="b"))
    first = buffer.sample(batch_size=6, sequence_length=4, seed=123)
    second = buffer.sample(batch_size=6, sequence_length=4, seed=123)
    assert torch.equal(first.action, second.action)
    assert first.metadata == second.metadata


def test_sample_does_not_cross_episode_boundaries() -> None:
    buffer = SequenceReplayBuffer()
    buffer.add_episode(make_episode(5, episode_id="a"))
    buffer.add_episode(make_episode(5, episode_id="b"))
    batch = buffer.sample(batch_size=20, sequence_length=5, seed=7)
    for metadata, action in zip(batch.metadata, batch.action, strict=True):
        assert metadata["sequence_start"] == 0
        assert metadata["sequence_end"] == 5
        assert action.tolist() == [0, 1, 2, 3, 4]


def test_done_requires_continue_to_remain_zero() -> None:
    episode = make_episode(5, done_at=2)
    episode.cont[4] = 1.0
    buffer = SequenceReplayBuffer()
    with pytest.raises(ValueError, match="cont must remain 0"):
        buffer.add_episode(episode)


def test_no_candidate_raises() -> None:
    buffer = SequenceReplayBuffer()
    buffer.add_episode(make_episode(2))
    with pytest.raises(ValueError, match="no episodes"):
        buffer.sample(batch_size=1, sequence_length=3)
