from __future__ import annotations

import pytest
import torch

from rcs_ac_v2.policy.behavior_loss import BehaviorLossConfig, compute_behavior_loss, lambda_returns


def test_lambda_returns_shape_and_terminal_continuation() -> None:
    reward = torch.tensor([[1.0, 2.0, 3.0]])
    cont = torch.tensor([[1.0, 0.0, 1.0]])
    value = torch.tensor([[10.0, 20.0, 30.0]])
    bootstrap = torch.tensor([99.0])

    returns = lambda_returns(
        reward=reward,
        cont=cont,
        value=value,
        bootstrap=bootstrap,
        gamma=1.0,
        lambda_=1.0,
    )

    assert returns.shape == (1, 3)
    assert returns[0, 1].item() == 2.0
    assert returns[0, 2].item() == 102.0
    assert returns[0, 0].item() == 3.0


def test_lambda_returns_lambda_zero_uses_next_value() -> None:
    reward = torch.tensor([[1.0, 2.0]])
    cont = torch.tensor([[1.0, 1.0]])
    value = torch.tensor([[10.0, 20.0]])
    bootstrap = torch.tensor([30.0])

    returns = lambda_returns(
        reward=reward,
        cont=cont,
        value=value,
        bootstrap=bootstrap,
        gamma=1.0,
        lambda_=0.0,
    )

    assert returns.tolist() == [[21.0, 32.0]]


def test_behavior_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 39, requires_grad=True)
    value = torch.randn(2, 4, requires_grad=True)
    action = torch.randint(0, 39, (2, 4), dtype=torch.int64)
    reward = torch.randn(2, 4)
    cont = torch.ones(2, 4)

    losses = compute_behavior_loss(
        logits=logits,
        value=value,
        action=action,
        reward=reward,
        cont=cont,
        cfg=BehaviorLossConfig(entropy_weight=1e-3),
    )
    losses.total.backward()

    for metric in losses.detached_scalars().values():
        assert torch.isfinite(torch.tensor(metric))
    assert logits.grad is not None
    assert value.grad is not None
    assert losses.entropy > 0
    assert losses.lambda_return.shape == (2, 4)


def test_behavior_loss_validates_action_dtype() -> None:
    logits = torch.randn(2, 3, 39)
    value = torch.randn(2, 3)
    action = torch.zeros(2, 3)
    reward = torch.randn(2, 3)
    cont = torch.ones(2, 3)
    with pytest.raises(ValueError, match="int64"):
        compute_behavior_loss(logits=logits, value=value, action=action, reward=reward, cont=cont)


def test_lambda_return_validates_shapes() -> None:
    with pytest.raises(ValueError, match="share"):
        lambda_returns(
            reward=torch.zeros(2, 3),
            cont=torch.zeros(2, 4),
            value=torch.zeros(2, 3),
            bootstrap=torch.zeros(2),
        )


def test_lambda_return_validates_gamma_and_lambda() -> None:
    kwargs = {
        "reward": torch.zeros(2, 3),
        "cont": torch.ones(2, 3),
        "value": torch.zeros(2, 3),
        "bootstrap": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="lambda"):
        lambda_returns(**kwargs, lambda_=1.1)
    with pytest.raises(ValueError, match="gamma"):
        lambda_returns(**kwargs, gamma=1.1)


def test_entropy_weight_lowers_total_loss_for_fixed_tensors() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 39)
    value = torch.randn(2, 4)
    action = torch.randint(0, 39, (2, 4), dtype=torch.int64)
    reward = torch.randn(2, 4)
    cont = torch.ones(2, 4)

    low_entropy = compute_behavior_loss(
        logits=logits,
        value=value,
        action=action,
        reward=reward,
        cont=cont,
        cfg=BehaviorLossConfig(entropy_weight=0.0),
    )
    high_entropy = compute_behavior_loss(
        logits=logits,
        value=value,
        action=action,
        reward=reward,
        cont=cont,
        cfg=BehaviorLossConfig(entropy_weight=1.0),
    )
    assert high_entropy.total < low_entropy.total
