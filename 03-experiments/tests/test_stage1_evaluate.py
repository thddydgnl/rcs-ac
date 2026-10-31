from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcs_ac_v1.model import MODALITY_NAMES
from stage1_warmup.evaluate import aux_to_reliability_targets, evaluate_stage1_policy


def _make_obs(aux_values: list[float]) -> dict[str, np.ndarray]:
    aux = np.asarray(aux_values, dtype=np.float32)
    return {
        "vision": np.asarray([aux[0]], dtype=np.float32),
        "distance": np.asarray([aux[1]], dtype=np.float32),
        "ego": np.asarray([aux[2]], dtype=np.float32),
        "route": np.asarray([aux[3]], dtype=np.float32),
        "aux": aux,
    }


class MockStage1EvalEnv:
    def __init__(self, initial_obs: dict[str, np.ndarray], steps: list[dict[str, object]]):
        self.initial_obs = initial_obs
        self.steps = steps
        self.step_idx = 0
        self.total_step_calls = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del seed
        self.step_idx = 0
        return self.initial_obs, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        assert action.shape == (2,)
        transition = self.steps[self.step_idx]
        self.step_idx += 1
        self.total_step_calls += 1
        return (
            transition["next_obs"],
            float(transition["reward"]),
            bool(transition["terminated"]),
            bool(transition["truncated"]),
            dict(transition["info"]),
        )


class MockStage1EvalModel:
    def __init__(
        self,
        reliabilities: list[np.ndarray],
        attentions: list[np.ndarray] | None = None,
    ) -> None:
        self.reliabilities = [np.asarray(item, dtype=np.float32) for item in reliabilities]
        self.attentions = [
            np.asarray(item, dtype=np.float32) for item in (attentions or [np.full(4, 0.25, dtype=np.float32)] * len(reliabilities))
        ]
        self.step_calls = 0
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        assert batch_size == 1
        self.step_calls = 0
        return torch.zeros(batch_size, 1, dtype=torch.float32, device=device)

    def step(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool,
    ) -> SimpleNamespace:
        del obs_dict, episode_start, deterministic
        idx = self.step_calls
        self.step_calls += 1
        return SimpleNamespace(
            hidden_state=hidden_state + 1.0,
            action=torch.zeros((1, 2), dtype=torch.float32, device=hidden_state.device),
            aux_outputs={
                "hidden_norm": torch.full((1,), float(idx + 1), dtype=torch.float32, device=hidden_state.device),
                "attention_weights": torch.as_tensor(self.attentions[idx], dtype=torch.float32, device=hidden_state.device).unsqueeze(0),
                "reliability_scores": torch.as_tensor(
                    self.reliabilities[idx],
                    dtype=torch.float32,
                    device=hidden_state.device,
                ).unsqueeze(0),
            },
        )


def test_stage1_eval_uses_current_observation_for_targets() -> None:
    initial_obs = _make_obs([0.2, 0.7, 0.1, 0.4])
    next_obs = _make_obs([0.9, 0.1, 0.8, 0.2])
    model = MockStage1EvalModel(
        reliabilities=[aux_to_reliability_targets(initial_obs["aux"])],
    )
    env = MockStage1EvalEnv(
        initial_obs=initial_obs,
        steps=[
            {
                "next_obs": next_obs,
                "reward": 1.25,
                "terminated": True,
                "truncated": False,
                "info": {
                    "progress_delta": 0.3,
                    "goal_reached": True,
                    "collision": False,
                    "timeout": False,
                },
            }
        ],
    )

    metrics = evaluate_stage1_policy(env, model, torch.device("cpu"), num_episodes=1)

    assert model.eval_called
    assert metrics["eval/reliability_mse"] == pytest.approx(0.0)
    assert metrics["eval/return_mean"] == pytest.approx(1.25)
    assert metrics["eval/success_rate"] == pytest.approx(1.0)


def test_stage1_eval_rejects_next_observation_target_alignment() -> None:
    initial_obs = _make_obs([0.2, 0.7, 0.1, 0.4])
    next_obs = _make_obs([0.9, 0.1, 0.8, 0.2])
    model = MockStage1EvalModel(
        reliabilities=[aux_to_reliability_targets(next_obs["aux"])],
    )
    env = MockStage1EvalEnv(
        initial_obs=initial_obs,
        steps=[
            {
                "next_obs": next_obs,
                "reward": 0.5,
                "terminated": True,
                "truncated": False,
                "info": {
                    "progress_delta": 0.1,
                    "goal_reached": False,
                    "collision": True,
                    "timeout": False,
                },
            }
        ],
    )

    metrics = evaluate_stage1_policy(env, model, torch.device("cpu"), num_episodes=1)
    expected_mse = float(
        np.mean(
            (
                aux_to_reliability_targets(next_obs["aux"])
                - aux_to_reliability_targets(initial_obs["aux"])
            )
            ** 2
        )
    )

    assert metrics["eval/reliability_mse"] == pytest.approx(expected_mse)
    assert metrics["eval/reliability_mse"] > 0.0
    assert metrics["eval/collision_rate"] == pytest.approx(1.0)


def test_stage1_eval_multistep_alignment_and_truncation() -> None:
    obs_0 = _make_obs([0.1, 0.2, 0.3, 0.4])
    obs_1 = _make_obs([0.3, 0.4, 0.2, 0.1])
    obs_2 = _make_obs([0.8, 0.6, 0.4, 0.2])
    obs_3 = _make_obs([0.5, 0.5, 0.5, 0.5])
    model = MockStage1EvalModel(
        reliabilities=[
            aux_to_reliability_targets(obs_0["aux"]),
            aux_to_reliability_targets(obs_1["aux"]),
            aux_to_reliability_targets(obs_2["aux"]),
        ],
        attentions=[
            np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
            np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        ],
    )
    env = MockStage1EvalEnv(
        initial_obs=obs_0,
        steps=[
            {
                "next_obs": obs_1,
                "reward": 0.2,
                "terminated": False,
                "truncated": False,
                "info": {"progress_delta": 0.05},
            },
            {
                "next_obs": obs_2,
                "reward": 0.3,
                "terminated": False,
                "truncated": False,
                "info": {"progress_delta": 0.08},
            },
            {
                "next_obs": obs_3,
                "reward": -0.1,
                "terminated": False,
                "truncated": True,
                "info": {
                    "progress_delta": -0.02,
                    "goal_reached": False,
                    "collision": False,
                    "timeout": True,
                },
            },
        ],
    )

    metrics = evaluate_stage1_policy(env, model, torch.device("cpu"), num_episodes=1)

    assert model.step_calls == 3
    assert env.total_step_calls == 3
    assert metrics["eval/reliability_mse"] == pytest.approx(0.0)
    assert metrics["eval/timeout_rate"] == pytest.approx(1.0)
    for name in MODALITY_NAMES:
        assert f"eval_attention_mean/{name}" in metrics
        assert f"eval_reliability_mean/{name}" in metrics
