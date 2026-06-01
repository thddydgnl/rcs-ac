from __future__ import annotations

import numpy as np

from rcs_ac_first.data.expert_dataset import save_expert_npz
from rcs_ac_first.scripts.combine_expert_datasets import _combine
from rcs_ac_first.scripts.mix_expert_datasets import _concat_parts, _slice_part


def _arrays(n: int, *, episode_base: int = 0) -> dict[str, np.ndarray]:
    return {
        "obs_flat": np.zeros((n, 47311), dtype=np.float32),
        "auxiliary": np.zeros((n, 30), dtype=np.float32),
        "expert_action": np.zeros((n, 2), dtype=np.float32),
        "expert_control": np.zeros((n, 3), dtype=np.float32),
        "reward": np.zeros((n,), dtype=np.float32),
        "done": np.zeros((n,), dtype=np.bool_),
        "route_progress": np.linspace(0.0, 0.1, n, dtype=np.float32),
        "episode_id": np.full((n,), episode_base, dtype=np.int32),
        "step_in_episode": np.arange(n, dtype=np.int32),
        "bad_transition_mask": np.zeros((n,), dtype=np.bool_),
    }


def test_mix_expert_datasets_writes_row_level_source_metadata() -> None:
    arrays = _arrays(5, episode_base=7)
    indices = np.array([1, 3, 4], dtype=np.int64)

    part, summary, next_episode_offset = _slice_part(
        arrays,
        indices,
        label="early_left",
        dataset_index=2,
        dataset_path="data/expert/source.npz",
        episode_offset=10,
    )

    assert summary["label"] == "early_left"
    assert summary["dataset_index"] == 2
    assert summary["dataset_path"] == "data/expert/source.npz"
    assert next_episode_offset == 18
    np.testing.assert_array_equal(part["episode_id"], np.array([17, 17, 17], dtype=np.int32))
    np.testing.assert_array_equal(part["source_episode_id"], np.array([7, 7, 7], dtype=np.int32))
    np.testing.assert_array_equal(part["source_step_in_episode"], np.array([1, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(part["source_row_index"], indices.astype(np.int32))
    np.testing.assert_array_equal(part["source_dataset_index"], np.full((3,), 2, dtype=np.int32))
    assert part["source_label"].tolist() == ["early_left", "early_left", "early_left"]
    assert part["source_dataset"].tolist() == ["data/expert/source.npz"] * 3


def test_mix_expert_datasets_shuffles_source_metadata_with_rows() -> None:
    arrays = _arrays(6)
    first, _, _ = _slice_part(
        arrays,
        np.array([0, 1, 2], dtype=np.int64),
        label="a",
        dataset_index=0,
        dataset_path="a.npz",
        episode_offset=0,
    )
    second, _, _ = _slice_part(
        arrays,
        np.array([3, 4, 5], dtype=np.int64),
        label="b",
        dataset_index=1,
        dataset_path="b.npz",
        episode_offset=1,
    )

    mixed = _concat_parts([first, second], shuffle_output=True, seed=3)

    for row_index, label, dataset_index in zip(
        mixed["source_row_index"], mixed["source_label"], mixed["source_dataset_index"], strict=True
    ):
        if int(dataset_index) == 0:
            assert str(label) == "a"
            assert int(row_index) in {0, 1, 2}
        else:
            assert str(label) == "b"
            assert int(row_index) in {3, 4, 5}


def test_combine_expert_datasets_fills_missing_source_metadata_before_episode_reindex(tmp_path) -> None:
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    first = _arrays(2, episode_base=9)
    second = _arrays(3, episode_base=9)
    save_expert_npz(first_path, first)
    save_expert_npz(second_path, second)

    combined = _combine([first_path, second_path])

    assert combined["source_label"].tolist() == ["first", "first", "second", "second", "second"]
    assert combined["source_dataset_index"].tolist() == [0, 0, 1, 1, 1]
    assert combined["source_row_index"].tolist() == [0, 1, 0, 1, 2]
    assert combined["source_episode_id"].tolist() == [9, 9, 9, 9, 9]
    assert combined["source_step_in_episode"].tolist() == [0, 1, 0, 1, 2]
    assert combined["episode_id"].tolist() == [0, 0, 1, 1, 1]
    assert all(str(path).endswith((".npz",)) for path in combined["source_dataset"])
