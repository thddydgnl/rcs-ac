import numpy as np

from rcs_ac_first.data.expert_dataset import SequenceExpertDataset, save_expert_npz


def test_sequence_dataset_breaks_across_filtered_bad_transition(tmp_path) -> None:
    path = tmp_path / "expert.npz"
    n = 6
    arrays = {
        "obs_flat": np.arange(n, dtype=np.float32).reshape(n, 1).repeat(47311, axis=1),
        "auxiliary": np.zeros((n, 30), dtype=np.float32),
        "expert_action": np.zeros((n, 2), dtype=np.float32),
        "expert_control": np.zeros((n, 3), dtype=np.float32),
        "reward": np.zeros((n,), dtype=np.float32),
        "done": np.zeros((n,), dtype=np.bool_),
        "route_progress": np.arange(n, dtype=np.float32),
        "episode_id": np.array([0, 0, 0, 0, 0, 1], dtype=np.int32),
        "step_in_episode": np.array([0, 1, 2, 3, 4, 0], dtype=np.int32),
        "bad_transition_mask": np.array([False, False, True, False, False, False], dtype=np.bool_),
    }
    save_expert_npz(path, arrays)

    dataset = SequenceExpertDataset(
        path,
        split="all",
        sequence_length=3,
        stride=1,
        include_bad_transitions=False,
    )

    item = dataset[0]

    np.testing.assert_allclose(item["valid_mask"].numpy(), np.array([1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(item["route_progress"].numpy(), np.array([0.0, 1.0, 0.0], dtype=np.float32))
