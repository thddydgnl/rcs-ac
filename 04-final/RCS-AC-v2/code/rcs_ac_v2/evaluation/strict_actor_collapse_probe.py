"""Offline PDF Stage I actor-collapse diagnostics for privileged checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rcs_ac_v2.evaluation.checkpoint_action_probe import (
    _resolve_path,
    build_checkpoint_filled_config,
)
from rcs_ac_v2.evaluation.carla_adapter import load_stage2_policy_runner
from rcs_ac_v2.policy.behavior_loss import BehaviorLossConfig, lambda_returns
from rcs_ac_v2.privileged_stream.policy import imagine_policy_rollout
from rcs_ac_v2.training.privileged_replay import validate_replay_manifest
from rcs_ac_v2.utils.checkpoint import sha256_file
from rcs_ac_v2.world_model.rssm import RSSMState


def probe_strict_actor_collapse(
    *,
    base_config_path: str | Path,
    checkpoint_manifest_path: str | Path,
    replay_manifest_path: str | Path,
    project_root: str | Path | None = None,
    max_steps_per_file: int = 128,
    rollout_horizon: int = 16,
    gamma: float = 0.99,
    lambda_: float = 1.0,
    device: str = "cuda",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_steps_per_file <= 0:
        raise ValueError("max_steps_per_file must be positive")
    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda must be in [0, 1]")

    root = Path(project_root) if project_root is not None else Path.cwd()
    manifest_path = Path(checkpoint_manifest_path)
    replay_manifest_resolved = _resolve_path(
        replay_manifest_path,
        root=root,
        fallback_dir=manifest_path.parent,
    )
    replay_manifest = validate_replay_manifest(replay_manifest_resolved)
    config = build_checkpoint_filled_config(
        base_config_path=base_config_path,
        checkpoint_manifest_path=manifest_path,
        device=device,
    )
    runner = load_stage2_policy_runner(config)
    runner.world_model.eval()
    runner.policy.eval()

    actor_counts: Counter[int] = Counter()
    return_counts: Counter[int] = Counter()
    label_counts: Counter[int] = Counter()
    actor_equals_return_best = 0
    actor_equals_label = 0
    samples = 0
    logit_entropies: list[float] = []
    top2_margins: list[float] = []
    logit_variances: list[float] = []
    actor_return_ranks: list[int] = []
    actor_return_gaps: list[float] = []
    label_return_ranks: list[int] = []
    label_return_gaps: list[float] = []
    imagined_reward_means: list[float] = []
    imagined_continue_means: list[float] = []
    imagined_value_means: list[float] = []
    imagined_lambda_return_means: list[float] = []
    imagined_advantage_means: list[float] = []
    probed_files: list[dict[str, Any]] = []

    for file_entry in replay_manifest["files"]:
        replay_path = _resolve_path(
            file_entry["path"],
            root=root,
            fallback_dir=replay_manifest_resolved.parent,
        )
        with np.load(replay_path, allow_pickle=False) as data:
            bev_np = np.asarray(data["privileged_bev"], dtype=np.float32)
            ego_np = np.asarray(data["privileged_ego"], dtype=np.float32)
            action_np = np.asarray(data["action"], dtype=np.int64)
        steps = min(max_steps_per_file, int(bev_np.shape[0]))
        if steps <= 0:
            continue
        bev = torch.as_tensor(bev_np[:steps][None, ...], dtype=torch.float32, device=runner.device)
        ego = torch.as_tensor(ego_np[:steps][None, ...], dtype=torch.float32, device=runner.device)
        replay_actions = torch.as_tensor(action_np[:steps][None, ...], dtype=torch.int64, device=runner.device)
        with torch.no_grad():
            embeds = runner.world_model.encoder(bev, ego)
            sequence = runner.world_model.rssm.observe(embeds, replay_actions)
            for index in range(steps):
                state = _state_at(sequence, index)
                latent = sequence.latent[:, index : index + 1]
                policy_output = runner.policy(latent)
                logits = policy_output.logits[:, 0]
                action_id = int(torch.argmax(logits, dim=-1).item())
                label_id = int(replay_actions[0, index].item())
                counterfactual = _one_step_counterfactual_returns(
                    runner=runner,
                    state=state,
                    gamma=float(gamma),
                )
                return_scores = counterfactual["one_step_return"]
                best_return_action = int(torch.argmax(return_scores).item())
                sorted_actions = torch.argsort(return_scores, descending=True)
                actor_rank = _rank_of_action(sorted_actions, action_id)
                label_rank = _rank_of_action(sorted_actions, label_id)
                best_score = float(return_scores[best_return_action].detach().cpu().item())
                actor_score = float(return_scores[action_id].detach().cpu().item())
                label_score = float(return_scores[label_id].detach().cpu().item())

                rollout = imagine_policy_rollout(
                    world_model=runner.world_model,
                    policy=runner.policy,
                    initial_state=state,
                    horizon=int(rollout_horizon),
                    deterministic=False,
                )
                returns = lambda_returns(
                    reward=rollout.reward,
                    cont=rollout.cont,
                    value=rollout.value.detach(),
                    bootstrap=rollout.value[:, -1].detach(),
                    gamma=float(gamma),
                    lambda_=float(lambda_),
                )
                advantage = returns - rollout.value

                actor_counts[action_id] += 1
                return_counts[best_return_action] += 1
                label_counts[label_id] += 1
                actor_equals_return_best += int(action_id == best_return_action)
                actor_equals_label += int(action_id == label_id)
                samples += 1
                logit_entropies.append(float(policy_output.entropy[:, 0].mean().detach().cpu().item()))
                top2 = torch.topk(logits, k=2, dim=-1).values[0]
                top2_margins.append(float((top2[0] - top2[1]).detach().cpu().item()))
                logit_variances.append(float(torch.var(logits, unbiased=False).detach().cpu().item()))
                actor_return_ranks.append(actor_rank)
                actor_return_gaps.append(best_score - actor_score)
                label_return_ranks.append(label_rank)
                label_return_gaps.append(best_score - label_score)
                imagined_reward_means.append(float(rollout.reward.mean().detach().cpu().item()))
                imagined_continue_means.append(float(rollout.cont.mean().detach().cpu().item()))
                imagined_value_means.append(float(rollout.value.mean().detach().cpu().item()))
                imagined_lambda_return_means.append(float(returns.mean().detach().cpu().item()))
                imagined_advantage_means.append(float(advantage.mean().detach().cpu().item()))
        probed_files.append({"path": str(replay_path), "steps": int(steps)})

    result = {
        "stage": "stage2_privileged_policy_clean",
        "source": "stage2_strict_actor_collapse_probe",
        "clean_gate_evidence": False,
        "checkpoint_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "replay_manifest": {
            "path": str(replay_manifest_resolved),
            "sha256": sha256_file(replay_manifest_resolved),
        },
        "probed_files": probed_files,
        "samples": int(samples),
        "rollout_horizon": int(rollout_horizon),
        "gamma": float(gamma),
        "lambda": float(lambda_),
        "actor_action_counts": _string_counts(actor_counts),
        "counterfactual_best_action_counts": _string_counts(return_counts),
        "label_action_counts": _string_counts(label_counts),
        "actor_dominant_action_fraction": _dominant_fraction(actor_counts),
        "counterfactual_dominant_action_fraction": _dominant_fraction(return_counts),
        "actor_equals_counterfactual_best_fraction": _fraction(actor_equals_return_best, samples),
        "actor_equals_label_fraction": _fraction(actor_equals_label, samples),
        "actor_return_rank_mean": _mean(actor_return_ranks),
        "actor_return_gap_mean": _mean(actor_return_gaps),
        "label_return_rank_mean": _mean(label_return_ranks),
        "label_return_gap_mean": _mean(label_return_gaps),
        "logit_entropy_mean": _mean(logit_entropies),
        "logit_top2_margin_mean": _mean(top2_margins),
        "logit_variance_mean": _mean(logit_variances),
        "imagined_reward_mean": _mean(imagined_reward_means),
        "imagined_continue_mean": _mean(imagined_continue_means),
        "imagined_value_mean": _mean(imagined_value_means),
        "imagined_lambda_return_mean": _mean(imagined_lambda_return_means),
        "imagined_advantage_mean": _mean(imagined_advantage_means),
        "interpretation": (
            "If counterfactual_best_action_counts also collapse and actor matches the best "
            "counterfactual action, prioritize world-model reward/continue/action-conditioned "
            "imagination. If counterfactual returns are state-dependent but actor_action_counts "
            "collapse, prioritize PDF Stage I actor-critic tuning."
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _state_at(sequence: Any, index: int) -> RSSMState:
    return RSSMState(
        deter=sequence.deter[:, index],
        stoch=sequence.stoch[:, index],
        prior_mean=sequence.prior_mean[:, index],
        prior_std=sequence.prior_std[:, index],
        post_mean=sequence.post_mean[:, index],
        post_std=sequence.post_std[:, index],
    )


def _one_step_counterfactual_returns(*, runner: Any, state: RSSMState, gamma: float) -> dict[str, torch.Tensor]:
    action_dim = 39
    repeated_state = RSSMState(
        deter=state.deter.repeat(action_dim, 1),
        stoch=state.stoch.repeat(action_dim, 1),
        prior_mean=state.prior_mean.repeat(action_dim, 1),
        prior_std=state.prior_std.repeat(action_dim, 1),
        post_mean=state.post_mean.repeat(action_dim, 1),
        post_std=state.post_std.repeat(action_dim, 1),
    )
    actions = torch.arange(action_dim, dtype=torch.int64, device=runner.device).reshape(action_dim, 1)
    next_sequence = runner.world_model.rssm.imagine(repeated_state, actions)
    predictions = runner.world_model.heads(next_sequence.latent)
    reward = predictions["reward"][:, 0]
    cont = torch.sigmoid(predictions["continue_logits"][:, 0])
    value = runner.policy(next_sequence.latent).value[:, 0]
    one_step_return = reward + float(gamma) * cont * value
    return {
        "reward": reward,
        "continue": cont,
        "value": value,
        "one_step_return": one_step_return,
    }


def _rank_of_action(sorted_actions: torch.Tensor, action_id: int) -> int:
    matches = (sorted_actions == int(action_id)).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return -1
    return int(matches[0, 0].item()) + 1


def _string_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(key): int(counts[key]) for key in sorted(counts)}


def _dominant_fraction(counts: Counter[int]) -> float:
    total = sum(counts.values())
    return _fraction(max(counts.values()) if counts else 0, total)


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe strict Stage I actor collapse on privileged replay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--project-root", default="")
    parser.add_argument("--max-steps-per-file", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    result = probe_strict_actor_collapse(
        base_config_path=args.config,
        checkpoint_manifest_path=args.checkpoint_manifest,
        replay_manifest_path=args.replay_manifest,
        project_root=args.project_root or None,
        max_steps_per_file=args.max_steps_per_file,
        rollout_horizon=args.rollout_horizon,
        gamma=args.gamma,
        lambda_=args.lambda_,
        device=args.device,
        output_path=args.output or None,
    )
    print(f"STAGE2_STRICT_ACTOR_COLLAPSE_PROBE={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
