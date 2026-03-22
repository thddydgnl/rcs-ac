"""Smoke test for RCS-AC v1 recurrent PPO."""

from __future__ import annotations

from pathlib import Path

import torch

from rcs_ac_v1.config import load_experiment_config
from rcs_ac_v1.model import RecurrentAttentionActorCritic
from rcs_ac_v1.trainer import RecurrentPPOTrainer


def test_hidden_reset() -> None:
    model = RecurrentAttentionActorCritic(
        encoder_hidden_dims=[16],
        encoder_output_dim=8,
        attention_hidden_dim=8,
        recurrent_hidden_dim=12,
        attention_temperature=1.5,
        use_feature_layer_norm=True,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        action_dim=2,
        action_log_std_init=-0.5,
    )
    hidden = torch.ones(3, 12)
    done_mask = torch.tensor([0.0, 1.0, 0.0])
    reset_hidden = model.reset_hidden(hidden, done_mask)
    assert torch.allclose(reset_hidden[0], torch.ones(12))
    assert torch.allclose(reset_hidden[1], torch.zeros(12))
    assert torch.allclose(reset_hidden[2], torch.ones(12))


def run_trainer_smoke() -> None:
    config = load_experiment_config("/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/rcs_ac_v1.yaml")
    config.ppo.total_steps = 256
    config.ppo.rollout_steps = 128
    config.ppo.sequence_length = 16
    config.ppo.update_epochs = 2
    config.ppo.minibatch_num_sequences = 2
    config.ppo.num_eval_episodes = 2
    config.logging.eval_interval_updates = 1
    config.logging.save_interval_updates = 100
    config.logging.output_dir = "/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/runs/smoke_rcs_ac_v1"

    trainer = RecurrentPPOTrainer(config)
    trainer.train()

    run_dir = Path(config.logging.output_dir)
    assert (run_dir / "train_metrics.jsonl").exists()
    assert (run_dir / "eval_metrics.jsonl").exists()
    assert (run_dir / "run_status.json").exists()


if __name__ == "__main__":
    test_hidden_reset()
    run_trainer_smoke()
    print("RCS-AC v1 smoke test passed.")
