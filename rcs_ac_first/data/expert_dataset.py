from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_EXPERT_KEYS: tuple[str, ...] = (
    "obs_flat",
    "auxiliary",
    "expert_action",
    "expert_control",
    "reward",
    "done",
    "route_progress",
    "episode_id",
    "step_in_episode",
    "bad_transition_mask",
)

OPTIONAL_MASK_KEYS: tuple[str, ...] = (
    "aux_road_mask_target",
    "aux_lane_edge_target",
    "aux_mask_targets_available",
)


@dataclass(frozen=True)
class ExpertDatasetSummary:
    num_transitions: int
    obs_dim: int
    auxiliary_dim: int
    action_dim: int
    bad_transition_count: int
    done_count: int
    route_progress_min: float
    route_progress_max: float
    route_progress_mean: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "num_transitions": int(self.num_transitions),
            "obs_dim": int(self.obs_dim),
            "auxiliary_dim": int(self.auxiliary_dim),
            "action_dim": int(self.action_dim),
            "bad_transition_count": int(self.bad_transition_count),
            "done_count": int(self.done_count),
            "route_progress_min": float(self.route_progress_min),
            "route_progress_max": float(self.route_progress_max),
            "route_progress_mean": float(self.route_progress_mean),
        }


def _as_array(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _require_finite(name: str, array: np.ndarray) -> None:
    if array.size and np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"Expert dataset key {name!r} contains NaN or inf values.")


def validate_expert_arrays(
    arrays: dict[str, Any],
    *,
    expected_obs_dim: int = 47311,
    expected_auxiliary_dim: int = 30,
    expected_action_dim: int = 2,
) -> dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_EXPERT_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"Expert dataset missing required keys: {missing}")

    result: dict[str, np.ndarray] = {}
    result["obs_flat"] = _as_array(arrays["obs_flat"], dtype=np.float32)
    result["auxiliary"] = _as_array(arrays["auxiliary"], dtype=np.float32)
    result["expert_action"] = _as_array(arrays["expert_action"], dtype=np.float32)
    result["expert_control"] = _as_array(arrays["expert_control"], dtype=np.float32)
    result["reward"] = _as_array(arrays["reward"], dtype=np.float32).reshape(-1)
    result["done"] = _as_array(arrays["done"], dtype=np.bool_).reshape(-1)
    result["route_progress"] = _as_array(arrays["route_progress"], dtype=np.float32).reshape(-1)
    result["episode_id"] = _as_array(arrays["episode_id"], dtype=np.int32).reshape(-1)
    result["step_in_episode"] = _as_array(arrays["step_in_episode"], dtype=np.int32).reshape(-1)
    result["bad_transition_mask"] = _as_array(arrays["bad_transition_mask"], dtype=np.bool_).reshape(-1)

    if result["obs_flat"].ndim != 2 or int(result["obs_flat"].shape[1]) != int(expected_obs_dim):
        raise ValueError(
            f"obs_flat must have shape [N, {expected_obs_dim}], got {tuple(result['obs_flat'].shape)}"
        )
    if result["auxiliary"].ndim != 2 or int(result["auxiliary"].shape[1]) != int(expected_auxiliary_dim):
        raise ValueError(
            f"auxiliary must have shape [N, {expected_auxiliary_dim}], got {tuple(result['auxiliary'].shape)}"
        )
    if result["expert_action"].ndim != 2 or int(result["expert_action"].shape[1]) != int(expected_action_dim):
        raise ValueError(
            f"expert_action must have shape [N, {expected_action_dim}], got {tuple(result['expert_action'].shape)}"
        )
    if result["expert_control"].ndim != 2 or int(result["expert_control"].shape[1]) != 3:
        raise ValueError(f"expert_control must have shape [N, 3], got {tuple(result['expert_control'].shape)}")

    n = int(result["obs_flat"].shape[0])
    if n <= 0:
        raise ValueError("Expert dataset is empty.")
    for key, value in result.items():
        if int(value.shape[0]) != n:
            raise ValueError(f"Expert dataset key {key!r} has length {int(value.shape[0])}, expected {n}.")
        _require_finite(key, value)

    if not np.all((result["expert_action"] >= -1.0001) & (result["expert_action"] <= 1.0001)):
        raise ValueError("expert_action contains values outside [-1, 1].")

    for key, value in arrays.items():
        if key in result:
            continue
        optional = np.asarray(value)
        if optional.shape[:1] == (n,):
            result[key] = optional.copy()
    if (
        "target_corrective_delta" not in result
        and "target_corrected_steer" in result
        and "policy_action_steer" in result
    ):
        result["target_corrective_delta"] = (
            np.asarray(result["target_corrected_steer"], dtype=np.float32).reshape(-1)
            - np.asarray(result["policy_action_steer"], dtype=np.float32).reshape(-1)
        ).astype(np.float32, copy=False)
        result["target_corrective_delta_available"] = np.ones((n,), dtype=np.float32)
    _validate_optional_mask_arrays(result, n)
    return result


def _validate_optional_mask_arrays(arrays: dict[str, np.ndarray], n: int) -> None:
    road = arrays.get("aux_road_mask_target")
    lane = arrays.get("aux_lane_edge_target")
    available = arrays.get("aux_mask_targets_available")
    if road is None and lane is None and available is None:
        return
    if road is None or lane is None or available is None:
        raise KeyError(
            "Optional mask targets must include aux_road_mask_target, "
            "aux_lane_edge_target, and aux_mask_targets_available together."
        )
    if tuple(road.shape) != tuple(lane.shape):
        raise ValueError(f"Mask target shape mismatch: road={tuple(road.shape)} lane={tuple(lane.shape)}")
    if road.ndim != 4 or int(road.shape[0]) != int(n) or int(road.shape[1]) != 1:
        raise ValueError(f"Mask targets must have shape [N, 1, H, W], got {tuple(road.shape)}")
    if int(np.asarray(available).reshape(-1).shape[0]) != int(n):
        raise ValueError("aux_mask_targets_available length mismatch.")
    arrays["aux_road_mask_target"] = road.astype(np.float32, copy=False)
    arrays["aux_lane_edge_target"] = lane.astype(np.float32, copy=False)
    arrays["aux_mask_targets_available"] = np.asarray(available, dtype=np.float32).reshape(-1)
    _require_finite("aux_road_mask_target", arrays["aux_road_mask_target"])
    _require_finite("aux_lane_edge_target", arrays["aux_lane_edge_target"])
    _require_finite("aux_mask_targets_available", arrays["aux_mask_targets_available"])


def _optional_mask_item(arrays: dict[str, np.ndarray], row: int) -> dict[str, torch.Tensor]:
    road = arrays.get("aux_road_mask_target")
    lane = arrays.get("aux_lane_edge_target")
    available = arrays.get("aux_mask_targets_available")
    if road is None or lane is None or available is None:
        road_tensor = torch.zeros((1, 45, 80), dtype=torch.float32)
        lane_tensor = torch.zeros((1, 45, 80), dtype=torch.float32)
        available_tensor = torch.tensor(0.0, dtype=torch.float32)
    else:
        road_tensor = torch.from_numpy(road[row].astype(np.float32, copy=False))
        lane_tensor = torch.from_numpy(lane[row].astype(np.float32, copy=False))
        available_tensor = torch.tensor(float(available[row]), dtype=torch.float32)
    return {
        "aux_road_mask_target": road_tensor,
        "aux_lane_edge_target": lane_tensor,
        "aux_mask_targets_available": available_tensor,
    }


def _optional_scalar_item(arrays: dict[str, np.ndarray], key: str, row: int) -> torch.Tensor:
    value = arrays.get(key)
    if value is None:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.tensor(float(np.asarray(value).reshape(-1)[row]), dtype=torch.float32)


def _optional_scalar_available_item(arrays: dict[str, np.ndarray], key: str, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = arrays.get(key)
    if value is None:
        return torch.tensor(0.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    available = arrays.get(f"{key}_available")
    if available is None:
        available_value = 1.0
    else:
        available_value = float(np.asarray(available).reshape(-1)[row])
    return (
        torch.tensor(float(np.asarray(value).reshape(-1)[row]), dtype=torch.float32),
        torch.tensor(available_value, dtype=torch.float32),
    )


def summarize_expert_arrays(arrays: dict[str, Any]) -> ExpertDatasetSummary:
    validated = validate_expert_arrays(arrays)
    route_progress = validated["route_progress"]
    return ExpertDatasetSummary(
        num_transitions=int(validated["obs_flat"].shape[0]),
        obs_dim=int(validated["obs_flat"].shape[1]),
        auxiliary_dim=int(validated["auxiliary"].shape[1]),
        action_dim=int(validated["expert_action"].shape[1]),
        bad_transition_count=int(np.count_nonzero(validated["bad_transition_mask"])),
        done_count=int(np.count_nonzero(validated["done"])),
        route_progress_min=float(np.min(route_progress)),
        route_progress_max=float(np.max(route_progress)),
        route_progress_mean=float(np.mean(route_progress)),
    )


def save_expert_npz(path: str | Path, arrays: dict[str, Any]) -> ExpertDatasetSummary:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_expert_arrays(arrays)
    np.savez_compressed(output_path, **validated)
    return summarize_expert_arrays(validated)


def load_expert_npz(path: str | Path) -> dict[str, np.ndarray]:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return validate_expert_arrays(arrays)


def train_val_indices(
    length: int,
    *,
    val_fraction: float = 0.1,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(length)
    if n <= 1:
        raise ValueError("Need at least two samples for train/val split.")
    val_count = max(1, int(round(n * float(val_fraction))))
    val_count = min(val_count, n - 1)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


class ExpertDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 7,
        include_bad_transitions: bool = True,
    ) -> None:
        arrays = load_expert_npz(path)
        if not include_bad_transitions:
            keep = ~arrays["bad_transition_mask"].astype(np.bool_)
            arrays = {key: value[keep] if value.shape[:1] == keep.shape else value for key, value in arrays.items()}
        train_idx, val_idx = train_val_indices(int(arrays["obs_flat"].shape[0]), val_fraction=val_fraction, seed=seed)
        if split == "train":
            self.indices = train_idx
        elif split in {"val", "validation"}:
            self.indices = val_idx
        elif split == "all":
            self.indices = np.arange(int(arrays["obs_flat"].shape[0]), dtype=np.int64)
        else:
            raise ValueError("split must be one of: train, val, validation, all")
        self.arrays = arrays
        self.path = Path(path)
        self.split = split

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[int(index)])
        target_delta, target_delta_available = _optional_scalar_available_item(
            self.arrays,
            "target_corrective_delta",
            row,
        )
        item = {
            "obs": torch.from_numpy(self.arrays["obs_flat"][row].astype(np.float32, copy=False)),
            "auxiliary": torch.from_numpy(self.arrays["auxiliary"][row].astype(np.float32, copy=False)),
            "expert_action": torch.from_numpy(self.arrays["expert_action"][row].astype(np.float32, copy=False)),
            "route_progress": torch.tensor(float(self.arrays["route_progress"][row]), dtype=torch.float32),
            "lane_offset": _optional_scalar_item(self.arrays, "lane_offset", row),
            "heading_error": _optional_scalar_item(self.arrays, "heading_error", row),
            "policy_action_steer": _optional_scalar_item(self.arrays, "policy_action_steer", row),
            "target_corrective_delta": target_delta,
            "target_corrective_delta_available": target_delta_available,
            "done": torch.tensor(bool(self.arrays["done"][row]), dtype=torch.bool),
            "bad_transition": torch.tensor(bool(self.arrays["bad_transition_mask"][row]), dtype=torch.bool),
        }
        item.update(_optional_mask_item(self.arrays, row))
        return item


def train_val_episode_ids(
    episode_ids: np.ndarray,
    *,
    val_fraction: float = 0.1,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(np.asarray(episode_ids, dtype=np.int32).reshape(-1))
    if int(unique.shape[0]) <= 1:
        raise ValueError("Need at least two episodes for sequence train/val split.")
    val_count = max(1, int(round(int(unique.shape[0]) * float(val_fraction))))
    val_count = min(val_count, int(unique.shape[0]) - 1)
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    val_ids = np.sort(shuffled[:val_count])
    train_ids = np.sort(shuffled[val_count:])
    return train_ids.astype(np.int32, copy=False), val_ids.astype(np.int32, copy=False)


class SequenceExpertDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 7,
        sequence_length: int = 32,
        stride: int = 8,
        include_bad_transitions: bool = True,
    ) -> None:
        arrays = load_expert_npz(path)
        source_indices = np.arange(int(arrays["obs_flat"].shape[0]), dtype=np.int64)
        if not include_bad_transitions:
            keep = ~arrays["bad_transition_mask"].astype(np.bool_)
            arrays = {key: value[keep] if value.shape[:1] == keep.shape else value for key, value in arrays.items()}
            source_indices = source_indices[keep]
        self.arrays = arrays
        self.source_indices = source_indices
        self.path = Path(path)
        self.split = split
        self.sequence_length = int(sequence_length)
        self.stride = max(1, int(stride))
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")

        train_ids, val_ids = train_val_episode_ids(
            arrays["episode_id"],
            val_fraction=val_fraction,
            seed=seed,
        )
        if split == "train":
            selected_ids = set(int(item) for item in train_ids.tolist())
        elif split in {"val", "validation"}:
            selected_ids = set(int(item) for item in val_ids.tolist())
        elif split == "all":
            selected_ids = set(int(item) for item in np.unique(arrays["episode_id"]).tolist())
        else:
            raise ValueError("split must be one of: train, val, validation, all")

        starts: list[int] = []
        episode_ids = arrays["episode_id"].astype(np.int32)
        for episode_id in sorted(selected_ids):
            indices = np.flatnonzero(episode_ids == int(episode_id))
            if indices.size == 0:
                continue
            for offset in range(0, int(indices.size), self.stride):
                starts.append(int(indices[offset]))
        if not starts:
            raise ValueError(f"No sequence starts available for split={split!r}")
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[int(index)])
        episode_id = int(self.arrays["episode_id"][start])
        length = int(self.sequence_length)
        obs_dim = int(self.arrays["obs_flat"].shape[1])
        aux_dim = int(self.arrays["auxiliary"].shape[1])
        action_dim = int(self.arrays["expert_action"].shape[1])

        obs = np.zeros((length, obs_dim), dtype=np.float32)
        auxiliary = np.zeros((length, aux_dim), dtype=np.float32)
        expert_action = np.zeros((length, action_dim), dtype=np.float32)
        route_progress = np.zeros((length,), dtype=np.float32)
        lane_offset = np.zeros((length,), dtype=np.float32)
        heading_error = np.zeros((length,), dtype=np.float32)
        policy_action_steer = np.zeros((length,), dtype=np.float32)
        target_corrective_delta = np.zeros((length,), dtype=np.float32)
        target_corrective_delta_available = np.zeros((length,), dtype=np.float32)
        done = np.zeros((length,), dtype=np.bool_)
        bad = np.zeros((length,), dtype=np.bool_)
        valid = np.zeros((length,), dtype=np.float32)
        road_mask = np.zeros((length, 1, 45, 80), dtype=np.float32)
        lane_mask = np.zeros((length, 1, 45, 80), dtype=np.float32)
        mask_available = np.zeros((length,), dtype=np.float32)

        n = int(self.arrays["obs_flat"].shape[0])
        for step in range(length):
            row = start + step
            if row >= n or int(self.arrays["episode_id"][row]) != episode_id:
                break
            if step > 0:
                previous_row = row - 1
                source_gap = int(self.source_indices[row]) - int(self.source_indices[previous_row])
                step_gap = int(self.arrays["step_in_episode"][row]) - int(self.arrays["step_in_episode"][previous_row])
                if source_gap != 1 or step_gap != 1:
                    break
            obs[step] = self.arrays["obs_flat"][row].astype(np.float32, copy=False)
            auxiliary[step] = self.arrays["auxiliary"][row].astype(np.float32, copy=False)
            expert_action[step] = self.arrays["expert_action"][row].astype(np.float32, copy=False)
            route_progress[step] = float(self.arrays["route_progress"][row])
            lane_offset[step] = float(_optional_scalar_item(self.arrays, "lane_offset", row).item())
            heading_error[step] = float(_optional_scalar_item(self.arrays, "heading_error", row).item())
            policy_action_steer[step] = float(_optional_scalar_item(self.arrays, "policy_action_steer", row).item())
            target_delta, target_delta_available = _optional_scalar_available_item(
                self.arrays,
                "target_corrective_delta",
                row,
            )
            target_corrective_delta[step] = float(target_delta.item())
            target_corrective_delta_available[step] = float(target_delta_available.item())
            done[step] = bool(self.arrays["done"][row])
            bad[step] = bool(self.arrays["bad_transition_mask"][row])
            valid[step] = 1.0
            mask_item = _optional_mask_item(self.arrays, row)
            road_mask[step] = mask_item["aux_road_mask_target"].numpy()
            lane_mask[step] = mask_item["aux_lane_edge_target"].numpy()
            mask_available[step] = float(mask_item["aux_mask_targets_available"].item())
            if done[step]:
                break

        return {
            "obs": torch.from_numpy(obs),
            "auxiliary": torch.from_numpy(auxiliary),
            "expert_action": torch.from_numpy(expert_action),
            "route_progress": torch.from_numpy(route_progress),
            "lane_offset": torch.from_numpy(lane_offset),
            "heading_error": torch.from_numpy(heading_error),
            "policy_action_steer": torch.from_numpy(policy_action_steer),
            "target_corrective_delta": torch.from_numpy(target_corrective_delta),
            "target_corrective_delta_available": torch.from_numpy(target_corrective_delta_available),
            "done": torch.from_numpy(done),
            "bad_transition": torch.from_numpy(bad),
            "valid_mask": torch.from_numpy(valid),
            "aux_road_mask_target": torch.from_numpy(road_mask),
            "aux_lane_edge_target": torch.from_numpy(lane_mask),
            "aux_mask_targets_available": torch.from_numpy(mask_available),
            "episode_id": torch.tensor(episode_id, dtype=torch.int32),
            "start_index": torch.tensor(start, dtype=torch.int64),
        }
