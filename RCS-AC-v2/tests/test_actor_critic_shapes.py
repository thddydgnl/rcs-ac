from __future__ import annotations

import pytest
import torch

from rcs_ac_v2.env.discrete_actions import NUM_RAW2DRIVE_ACTIONS
from rcs_ac_v2.env.discrete_actions import get_discrete_action, to_carla_control
from rcs_ac_v2.policy.actor_critic import ActorCritic, ActorCriticConfig


def test_actor_critic_forward_shapes() -> None:
    policy = ActorCritic(ActorCriticConfig(latent_dim=20, hidden_dim=32))
    latent = torch.randn(3, 4, 20)

    output = policy(latent)

    assert output.logits.shape == (3, 4, NUM_RAW2DRIVE_ACTIONS)
    assert output.value.shape == (3, 4)
    assert output.entropy.shape == (3, 4)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.value).all()
    assert torch.isfinite(output.entropy).all()


def test_actor_critic_actions_are_valid_ids() -> None:
    policy = ActorCritic(ActorCriticConfig(latent_dim=20, hidden_dim=32))
    latent = torch.randn(3, 4, 20)

    action, log_prob = policy.act(latent, deterministic=False)
    deterministic_action, deterministic_log_prob = policy.act(latent, deterministic=True)

    for candidate in (action, deterministic_action):
        assert candidate.dtype == torch.int64
        assert candidate.shape == (3, 4)
        assert int(candidate.min()) >= 0
        assert int(candidate.max()) < NUM_RAW2DRIVE_ACTIONS
    assert log_prob.shape == (3, 4)
    assert deterministic_log_prob.shape == (3, 4)


def test_deterministic_action_is_argmax_and_maps_to_carla_control() -> None:
    policy = ActorCritic(ActorCriticConfig(latent_dim=20, hidden_dim=32))
    latent = torch.randn(2, 3, 20)
    output = policy(latent)
    action, _ = policy.act(latent, deterministic=True)

    assert torch.equal(action, torch.argmax(output.logits, dim=-1))
    control = to_carla_control(get_discrete_action(int(action[0, 0])))
    assert set(control) == {"throttle", "brake", "steer", "reverse"}


def test_sampled_actions_are_reproducible_with_manual_seed() -> None:
    policy = ActorCritic(ActorCriticConfig(latent_dim=20, hidden_dim=32))
    latent = torch.randn(2, 3, 20)

    torch.manual_seed(22)
    first, _ = policy.act(latent, deterministic=False)
    torch.manual_seed(22)
    second, _ = policy.act(latent, deterministic=False)

    assert torch.equal(first, second)


def test_actor_critic_rejects_wrong_latent_shape() -> None:
    policy = ActorCritic(ActorCriticConfig(latent_dim=20, hidden_dim=32))
    with pytest.raises(ValueError, match="latent must"):
        policy(torch.randn(3, 20))
    with pytest.raises(ValueError, match="latent dim"):
        policy(torch.randn(3, 4, 21))
