"""Train Stage 2 privileged checkpoints from CARLA privileged replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rcs_ac_v2.env.discrete_actions import RAW2DRIVE_ACTIONS
from rcs_ac_v2.policy.behavior_loss import BehaviorLossConfig, compute_behavior_loss
from rcs_ac_v2.privileged_stream.policy import PrivilegedPolicy, PrivilegedPolicyConfig, imagine_policy_rollout
from rcs_ac_v2.privileged_stream.world_model import PrivilegedWorldModel, PrivilegedWorldModelConfig
from rcs_ac_v2.training.privileged_replay import (
    REPLAY_SOURCE,
    load_replay_episodes,
    PrivilegedReplaySampler,
    validate_replay_manifest,
)
from rcs_ac_v2.utils.checkpoint import save_checkpoint, sha256_file
from rcs_ac_v2.utils.seed import seed_everything
from rcs_ac_v2.world_model.losses import WorldModelLossConfig, compute_world_model_loss


@dataclass(frozen=True)
class ReplayTrainingResult:
    output_dir: Path
    world_model_checkpoint_path: Path
    policy_checkpoint_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def train_stage2_privileged_from_replay(
    *,
    replay_paths: list[str | Path],
    replay_manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 20260524,
    sequence_length: int = 16,
    batch_size: int = 8,
    world_model_steps: int = 200,
    policy_steps: int = 200,
    learning_rate: float = 3e-4,
    policy_learning_rate: float | None = None,
    pdf_stage1_strict: bool = False,
    imitation_weight: float = 0.0,
    behavior_weight: float = 1.0,
    behavior_gamma: float = 0.99,
    behavior_lambda: float = 1.0,
    behavior_actor_weight: float = 0.1,
    behavior_critic_weight: float = 1.0,
    behavior_entropy_weight: float = 1e-3,
    imitation_class_balance: bool = False,
    imitation_balance_power: float = 1.0,
    steer_sequence_fraction: float = 0.0,
    moving_steer_false_positive_weight: float = 0.0,
    moving_steer_binary_weight: float = 0.0,
    moving_steer_nonmoving_margin_weight: float = 0.0,
    moving_steer_nonmoving_margin: float = 0.25,
    route_late_moving_steer_margin_weight: float = 0.0,
    route_late_moving_steer_margin: float = 0.25,
    route_late_moving_steer_min_speed_norm: float = 0.0,
    route_late_moving_steer_min_route_progress: float = 0.0,
    route_late_moving_steer_max_route_progress: float = 1.0,
    route_late_right_steer_margin_weight: float = 0.0,
    route_late_right_steer_margin: float = 0.25,
    route_late_right_steer_min_speed_norm: float = 0.0,
    route_late_right_steer_min_route_progress: float = 0.0,
    route_late_right_steer_max_route_progress: float = 1.0,
    world_model_action_aux_weight: float = 0.0,
    component_sample_weights: dict[str, float] | None = None,
    action_mass_cap_weight: float = 0.0,
    action_mass_caps: dict[int, float] | None = None,
    route_speed_action_mass_cap_weight: float = 0.0,
    route_speed_action_ids: tuple[int, ...] = (),
    route_speed_action_mass_cap: float = 1.0,
    route_speed_min_speed_norm: float = 0.0,
    route_speed_min_route_progress: float = 0.0,
    route_speed_max_route_progress: float = 1.0,
    route_action_mass_cap_weight: float = 0.0,
    route_action_mass_cap_ids: tuple[int, ...] = (),
    route_action_mass_cap: float = 1.0,
    route_action_mass_cap_min_speed_norm: float = 0.0,
    route_action_mass_cap_min_route_progress: float = 0.0,
    route_action_mass_cap_max_route_progress: float = 1.0,
    route_action_logit_cap_weight: float = 0.0,
    route_action_logit_cap_ids: tuple[int, ...] = (),
    route_action_logit_cap_margin: float = 0.0,
    route_action_logit_cap_min_speed_norm: float = 0.0,
    route_action_logit_cap_min_route_progress: float = 0.0,
    route_action_logit_cap_max_route_progress: float = 1.0,
    route_late_action_logit_cap_weight: float = 0.0,
    route_late_action_logit_cap_ids: tuple[int, ...] = (),
    route_late_action_logit_cap_margin: float = 0.0,
    route_late_action_logit_cap_min_speed_norm: float = 0.0,
    route_late_action_logit_cap_min_route_progress: float = 0.0,
    route_late_action_logit_cap_max_route_progress: float = 1.0,
    route_throttle_ceiling_weight: float = 0.0,
    route_throttle_ceiling: float = 1.0,
    route_throttle_min_speed_norm: float = 0.0,
    route_throttle_min_route_progress: float = 0.0,
    route_throttle_max_route_progress: float = 1.0,
    route_late_throttle_ceiling_weight: float = 0.0,
    route_late_throttle_ceiling: float = 1.0,
    route_late_throttle_min_speed_norm: float = 0.0,
    route_late_throttle_min_route_progress: float = 0.0,
    route_late_throttle_max_route_progress: float = 1.0,
    route_action_override_weight: float = 0.0,
    route_action_override_id: int = -1,
    route_action_override_min_speed_norm: float = 0.0,
    route_action_override_min_route_progress: float = 0.0,
    route_action_override_max_route_progress: float = 1.0,
    hidden_dim: int = 128,
    device: str = "cuda",
) -> ReplayTrainingResult:
    if world_model_steps <= 0 or policy_steps <= 0:
        raise ValueError("world_model_steps and policy_steps must be positive")
    if pdf_stage1_strict:
        _validate_pdf_stage1_strict_config(
            imitation_weight=imitation_weight,
            behavior_weight=behavior_weight,
            behavior_gamma=behavior_gamma,
            behavior_lambda=behavior_lambda,
            behavior_actor_weight=behavior_actor_weight,
            behavior_critic_weight=behavior_critic_weight,
            behavior_entropy_weight=behavior_entropy_weight,
            imitation_class_balance=imitation_class_balance,
            steer_sequence_fraction=steer_sequence_fraction,
            moving_steer_false_positive_weight=moving_steer_false_positive_weight,
            moving_steer_binary_weight=moving_steer_binary_weight,
            moving_steer_nonmoving_margin_weight=moving_steer_nonmoving_margin_weight,
            route_late_moving_steer_margin_weight=route_late_moving_steer_margin_weight,
            route_late_right_steer_margin_weight=route_late_right_steer_margin_weight,
            world_model_action_aux_weight=world_model_action_aux_weight,
            component_sample_weights=component_sample_weights,
            action_mass_cap_weight=action_mass_cap_weight,
            action_mass_caps=action_mass_caps,
            route_speed_action_mass_cap_weight=route_speed_action_mass_cap_weight,
            route_speed_action_ids=route_speed_action_ids,
            route_action_mass_cap_weight=route_action_mass_cap_weight,
            route_action_mass_cap_ids=route_action_mass_cap_ids,
            route_action_logit_cap_weight=route_action_logit_cap_weight,
            route_action_logit_cap_ids=route_action_logit_cap_ids,
            route_late_action_logit_cap_weight=route_late_action_logit_cap_weight,
            route_late_action_logit_cap_ids=route_late_action_logit_cap_ids,
            route_throttle_ceiling_weight=route_throttle_ceiling_weight,
            route_late_throttle_ceiling_weight=route_late_throttle_ceiling_weight,
            route_action_override_weight=route_action_override_weight,
            route_action_override_id=route_action_override_id,
        )
    if world_model_action_aux_weight < 0.0:
        raise ValueError("world_model_action_aux_weight must be non-negative")
    if not 0.0 <= float(behavior_gamma) <= 1.0:
        raise ValueError("behavior_gamma must be in [0, 1]")
    if not 0.0 <= float(behavior_lambda) <= 1.0:
        raise ValueError("behavior_lambda must be in [0, 1]")
    if behavior_actor_weight < 0.0:
        raise ValueError("behavior_actor_weight must be non-negative")
    if behavior_critic_weight < 0.0:
        raise ValueError("behavior_critic_weight must be non-negative")
    if behavior_entropy_weight < 0.0:
        raise ValueError("behavior_entropy_weight must be non-negative")
    if moving_steer_false_positive_weight < 0.0:
        raise ValueError("moving_steer_false_positive_weight must be non-negative")
    if moving_steer_binary_weight < 0.0:
        raise ValueError("moving_steer_binary_weight must be non-negative")
    if moving_steer_nonmoving_margin_weight < 0.0:
        raise ValueError("moving_steer_nonmoving_margin_weight must be non-negative")
    if moving_steer_nonmoving_margin < 0.0:
        raise ValueError("moving_steer_nonmoving_margin must be non-negative")
    if route_late_moving_steer_margin_weight < 0.0:
        raise ValueError("route_late_moving_steer_margin_weight must be non-negative")
    if route_late_moving_steer_margin < 0.0:
        raise ValueError("route_late_moving_steer_margin must be non-negative")
    if not 0.0 <= float(route_late_moving_steer_min_speed_norm) <= 1.0:
        raise ValueError("route_late_moving_steer_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_late_moving_steer_min_route_progress) <= 1.0:
        raise ValueError("route_late_moving_steer_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_late_moving_steer_max_route_progress) <= 1.0:
        raise ValueError("route_late_moving_steer_max_route_progress must be in [0, 1]")
    if float(route_late_moving_steer_min_route_progress) > float(
        route_late_moving_steer_max_route_progress
    ):
        raise ValueError(
            "route_late_moving_steer_min_route_progress must be <= "
            "route_late_moving_steer_max_route_progress"
        )
    if route_late_right_steer_margin_weight < 0.0:
        raise ValueError("route_late_right_steer_margin_weight must be non-negative")
    if route_late_right_steer_margin < 0.0:
        raise ValueError("route_late_right_steer_margin must be non-negative")
    if not 0.0 <= float(route_late_right_steer_min_speed_norm) <= 1.0:
        raise ValueError("route_late_right_steer_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_late_right_steer_min_route_progress) <= 1.0:
        raise ValueError("route_late_right_steer_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_late_right_steer_max_route_progress) <= 1.0:
        raise ValueError("route_late_right_steer_max_route_progress must be in [0, 1]")
    if float(route_late_right_steer_min_route_progress) > float(
        route_late_right_steer_max_route_progress
    ):
        raise ValueError(
            "route_late_right_steer_min_route_progress must be <= "
            "route_late_right_steer_max_route_progress"
        )
    if action_mass_cap_weight < 0.0:
        raise ValueError("action_mass_cap_weight must be non-negative")
    if route_speed_action_mass_cap_weight < 0.0:
        raise ValueError("route_speed_action_mass_cap_weight must be non-negative")
    resolved_route_speed_action_ids = _normalize_action_id_tuple(route_speed_action_ids)
    if not 0.0 <= float(route_speed_action_mass_cap) <= 1.0:
        raise ValueError("route_speed_action_mass_cap must be in [0, 1]")
    if not 0.0 <= float(route_speed_min_speed_norm) <= 1.0:
        raise ValueError("route_speed_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_speed_min_route_progress) <= 1.0:
        raise ValueError("route_speed_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_speed_max_route_progress) <= 1.0:
        raise ValueError("route_speed_max_route_progress must be in [0, 1]")
    if float(route_speed_min_route_progress) > float(route_speed_max_route_progress):
        raise ValueError("route_speed_min_route_progress must be <= route_speed_max_route_progress")
    if route_action_mass_cap_weight < 0.0:
        raise ValueError("route_action_mass_cap_weight must be non-negative")
    resolved_route_action_mass_cap_ids = _normalize_action_id_tuple(route_action_mass_cap_ids)
    if not 0.0 <= float(route_action_mass_cap) <= 1.0:
        raise ValueError("route_action_mass_cap must be in [0, 1]")
    if not 0.0 <= float(route_action_mass_cap_min_speed_norm) <= 1.0:
        raise ValueError("route_action_mass_cap_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_action_mass_cap_min_route_progress) <= 1.0:
        raise ValueError("route_action_mass_cap_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_action_mass_cap_max_route_progress) <= 1.0:
        raise ValueError("route_action_mass_cap_max_route_progress must be in [0, 1]")
    if float(route_action_mass_cap_min_route_progress) > float(
        route_action_mass_cap_max_route_progress
    ):
        raise ValueError(
            "route_action_mass_cap_min_route_progress must be <= "
            "route_action_mass_cap_max_route_progress"
        )
    if route_action_logit_cap_weight < 0.0:
        raise ValueError("route_action_logit_cap_weight must be non-negative")
    resolved_route_action_logit_cap_ids = _normalize_action_id_tuple(
        route_action_logit_cap_ids
    )
    if route_action_logit_cap_margin < 0.0:
        raise ValueError("route_action_logit_cap_margin must be non-negative")
    if not 0.0 <= float(route_action_logit_cap_min_speed_norm) <= 1.0:
        raise ValueError("route_action_logit_cap_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_action_logit_cap_min_route_progress) <= 1.0:
        raise ValueError("route_action_logit_cap_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_action_logit_cap_max_route_progress) <= 1.0:
        raise ValueError("route_action_logit_cap_max_route_progress must be in [0, 1]")
    if float(route_action_logit_cap_min_route_progress) > float(
        route_action_logit_cap_max_route_progress
    ):
        raise ValueError(
            "route_action_logit_cap_min_route_progress must be <= "
            "route_action_logit_cap_max_route_progress"
        )
    if route_late_action_logit_cap_weight < 0.0:
        raise ValueError("route_late_action_logit_cap_weight must be non-negative")
    resolved_route_late_action_logit_cap_ids = _normalize_action_id_tuple(
        route_late_action_logit_cap_ids
    )
    if route_late_action_logit_cap_margin < 0.0:
        raise ValueError("route_late_action_logit_cap_margin must be non-negative")
    if not 0.0 <= float(route_late_action_logit_cap_min_speed_norm) <= 1.0:
        raise ValueError("route_late_action_logit_cap_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_late_action_logit_cap_min_route_progress) <= 1.0:
        raise ValueError("route_late_action_logit_cap_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_late_action_logit_cap_max_route_progress) <= 1.0:
        raise ValueError("route_late_action_logit_cap_max_route_progress must be in [0, 1]")
    if float(route_late_action_logit_cap_min_route_progress) > float(
        route_late_action_logit_cap_max_route_progress
    ):
        raise ValueError(
            "route_late_action_logit_cap_min_route_progress must be <= "
            "route_late_action_logit_cap_max_route_progress"
        )
    if route_throttle_ceiling_weight < 0.0:
        raise ValueError("route_throttle_ceiling_weight must be non-negative")
    if not 0.0 <= float(route_throttle_ceiling) <= 1.0:
        raise ValueError("route_throttle_ceiling must be in [0, 1]")
    if not 0.0 <= float(route_throttle_min_speed_norm) <= 1.0:
        raise ValueError("route_throttle_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_throttle_min_route_progress) <= 1.0:
        raise ValueError("route_throttle_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_throttle_max_route_progress) <= 1.0:
        raise ValueError("route_throttle_max_route_progress must be in [0, 1]")
    if float(route_throttle_min_route_progress) > float(route_throttle_max_route_progress):
        raise ValueError("route_throttle_min_route_progress must be <= route_throttle_max_route_progress")
    if route_late_throttle_ceiling_weight < 0.0:
        raise ValueError("route_late_throttle_ceiling_weight must be non-negative")
    if not 0.0 <= float(route_late_throttle_ceiling) <= 1.0:
        raise ValueError("route_late_throttle_ceiling must be in [0, 1]")
    if not 0.0 <= float(route_late_throttle_min_speed_norm) <= 1.0:
        raise ValueError("route_late_throttle_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_late_throttle_min_route_progress) <= 1.0:
        raise ValueError("route_late_throttle_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_late_throttle_max_route_progress) <= 1.0:
        raise ValueError("route_late_throttle_max_route_progress must be in [0, 1]")
    if float(route_late_throttle_min_route_progress) > float(
        route_late_throttle_max_route_progress
    ):
        raise ValueError(
            "route_late_throttle_min_route_progress must be <= "
            "route_late_throttle_max_route_progress"
        )
    if route_action_override_weight < 0.0:
        raise ValueError("route_action_override_weight must be non-negative")
    if route_action_override_id != -1:
        _normalize_action_id_tuple((route_action_override_id,))
    if not 0.0 <= float(route_action_override_min_speed_norm) <= 1.0:
        raise ValueError("route_action_override_min_speed_norm must be in [0, 1]")
    if not 0.0 <= float(route_action_override_min_route_progress) <= 1.0:
        raise ValueError("route_action_override_min_route_progress must be in [0, 1]")
    if not 0.0 <= float(route_action_override_max_route_progress) <= 1.0:
        raise ValueError("route_action_override_max_route_progress must be in [0, 1]")
    if float(route_action_override_min_route_progress) > float(route_action_override_max_route_progress):
        raise ValueError("route_action_override_min_route_progress must be <= route_action_override_max_route_progress")
    seed_everything(seed)
    resolved_device = _resolve_device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay_manifest = validate_replay_manifest(replay_manifest_path, replay_paths=replay_paths)
    replay_files = replay_manifest["files"]
    episodes = load_replay_episodes(replay_paths)
    resolved_component_weights = _normalize_component_sample_weights(component_sample_weights)
    resolved_action_mass_caps = _normalize_action_mass_caps(action_mass_caps)
    episode_sample_weights = _make_episode_sample_weights(
        episodes,
        component_sample_weights=resolved_component_weights,
    )
    component_sampler_stats = _make_component_sampler_stats(
        episodes,
        sequence_length=sequence_length,
        component_sample_weights=resolved_component_weights,
    )
    imitation_class_weights = (
        _make_imitation_class_weights(
            episodes,
            action_dim=39,
            device=resolved_device,
            power=imitation_balance_power,
        )
        if imitation_class_balance
        else None
    )
    sampler = PrivilegedReplaySampler(
        episodes,
        sequence_length=sequence_length,
        seed=seed,
        device=resolved_device,
        priority_action_ids=_moving_steer_action_ids() if steer_sequence_fraction > 0.0 else None,
        priority_sample_fraction=steer_sequence_fraction,
        episode_sample_weights=episode_sample_weights,
    )
    sampler_stats = sampler.priority_stats()

    wm_cfg = PrivilegedWorldModelConfig(bev_height=16, bev_width=16, ego_dim=8, hidden_dim=hidden_dim)
    world_model = PrivilegedWorldModel(wm_cfg).to(resolved_device)
    action_aux_head = torch.nn.Linear(world_model.rssm.latent_dim, wm_cfg.action_dim).to(resolved_device)
    wm_train_parameters = list(world_model.parameters())
    if world_model_action_aux_weight > 0.0:
        wm_train_parameters += list(action_aux_head.parameters())
    wm_optimizer = torch.optim.AdamW(wm_train_parameters, lr=learning_rate)
    wm_loss_cfg = WorldModelLossConfig()
    wm_history: list[dict[str, float]] = []
    for _step in range(world_model_steps):
        batch = sampler.sample(batch_size)
        wm_optimizer.zero_grad(set_to_none=True)
        wm_output = world_model(batch)
        losses = compute_world_model_loss(
            predictions=wm_output.predictions,
            rssm_sequence=wm_output.rssm_sequence,
            batch=batch,
            cfg=wm_loss_cfg,
        )
        action_aux_loss = _world_model_action_auxiliary_loss(
            latent=wm_output.rssm_sequence.latent,
            action=batch.action,
            action_head=action_aux_head,
            class_weights=imitation_class_weights,
        )
        total_wm_loss = losses.total + float(world_model_action_aux_weight) * action_aux_loss
        total_wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm_train_parameters, max_norm=10.0)
        wm_optimizer.step()
        scalars = losses.detached_scalars()
        scalars["action_aux"] = float(action_aux_loss.detach().cpu().item())
        scalars["combined_total"] = float(total_wm_loss.detach().cpu().item())
        wm_history.append(scalars)

    world_model.eval()
    for parameter in world_model.parameters():
        parameter.requires_grad_(False)

    policy_lr = learning_rate if policy_learning_rate is None else policy_learning_rate
    policy_cfg = PrivilegedPolicyConfig(latent_dim=world_model.rssm.latent_dim, hidden_dim=hidden_dim)
    policy = PrivilegedPolicy(policy_cfg).to(resolved_device)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=policy_lr)
    behavior_cfg = BehaviorLossConfig(
        gamma=behavior_gamma,
        lambda_=behavior_lambda,
        actor_weight=behavior_actor_weight,
        critic_weight=behavior_critic_weight,
        entropy_weight=behavior_entropy_weight,
    )
    moving_steer_mask = _moving_steer_action_mask(action_dim=wm_cfg.action_dim, device=resolved_device)
    right_moving_steer_mask = _right_moving_steer_action_mask(
        action_dim=wm_cfg.action_dim,
        device=resolved_device,
    )
    policy_history: list[dict[str, float]] = []
    valid_action_ids = False
    for _step in range(policy_steps):
        batch = sampler.sample(batch_size)
        with torch.no_grad():
            world_model_output = world_model(batch)
            initial_state = world_model_output.rssm_sequence.final_state()
            replay_latent = world_model_output.rssm_sequence.latent.detach()
        policy_optimizer.zero_grad(set_to_none=True)
        rollout = imagine_policy_rollout(
            world_model=world_model,
            policy=policy,
            initial_state=initial_state,
            horizon=sequence_length,
            deterministic=False,
        )
        losses = compute_behavior_loss(
            logits=rollout.logits,
            value=rollout.value,
            action=rollout.action,
            reward=rollout.reward,
            cont=rollout.cont,
            cfg=behavior_cfg,
        )
        replay_policy = policy(replay_latent)
        imitation_loss = F.cross_entropy(
            replay_policy.logits.reshape(-1, wm_cfg.action_dim),
            batch.action.reshape(-1),
            weight=imitation_class_weights,
        )
        moving_steer_false_positive_loss = _moving_steer_false_positive_loss(
            logits=replay_policy.logits,
            action=batch.action,
            moving_steer_mask=moving_steer_mask,
        )
        moving_steer_binary_loss = _moving_steer_binary_loss(
            logits=replay_policy.logits,
            action=batch.action,
            moving_steer_mask=moving_steer_mask,
        )
        moving_steer_nonmoving_margin_loss = _moving_steer_nonmoving_margin_loss(
            logits=replay_policy.logits,
            action=batch.action,
            moving_steer_mask=moving_steer_mask,
            margin=moving_steer_nonmoving_margin,
        )
        route_late_moving_steer_margin_loss = _route_late_moving_steer_margin_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            action=batch.action,
            moving_steer_mask=moving_steer_mask,
            margin=route_late_moving_steer_margin,
            min_speed_norm=route_late_moving_steer_min_speed_norm,
            min_route_progress=route_late_moving_steer_min_route_progress,
            max_route_progress=route_late_moving_steer_max_route_progress,
        )
        route_late_right_steer_margin_loss = _route_late_right_steer_margin_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            right_moving_steer_mask=right_moving_steer_mask,
            margin=route_late_right_steer_margin,
            min_speed_norm=route_late_right_steer_min_speed_norm,
            min_route_progress=route_late_right_steer_min_route_progress,
            max_route_progress=route_late_right_steer_max_route_progress,
        )
        action_mass_cap_loss = _action_mass_cap_loss(
            logits=replay_policy.logits,
            action_mass_caps=resolved_action_mass_caps,
        )
        route_speed_action_mass_cap_loss = _route_speed_action_mass_cap_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            action_ids=resolved_route_speed_action_ids,
            cap=route_speed_action_mass_cap,
            min_speed_norm=route_speed_min_speed_norm,
            min_route_progress=route_speed_min_route_progress,
            max_route_progress=route_speed_max_route_progress,
        )
        route_action_mass_cap_loss = _route_speed_action_mass_cap_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            action_ids=resolved_route_action_mass_cap_ids,
            cap=route_action_mass_cap,
            min_speed_norm=route_action_mass_cap_min_speed_norm,
            min_route_progress=route_action_mass_cap_min_route_progress,
            max_route_progress=route_action_mass_cap_max_route_progress,
        )
        route_action_logit_cap_loss = _route_action_logit_cap_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            action_ids=resolved_route_action_logit_cap_ids,
            margin=route_action_logit_cap_margin,
            min_speed_norm=route_action_logit_cap_min_speed_norm,
            min_route_progress=route_action_logit_cap_min_route_progress,
            max_route_progress=route_action_logit_cap_max_route_progress,
        )
        route_late_action_logit_cap_loss = _route_action_logit_cap_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            action_ids=resolved_route_late_action_logit_cap_ids,
            margin=route_late_action_logit_cap_margin,
            min_speed_norm=route_late_action_logit_cap_min_speed_norm,
            min_route_progress=route_late_action_logit_cap_min_route_progress,
            max_route_progress=route_late_action_logit_cap_max_route_progress,
        )
        route_throttle_ceiling_loss = _route_throttle_ceiling_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            ceiling=route_throttle_ceiling,
            min_speed_norm=route_throttle_min_speed_norm,
            min_route_progress=route_throttle_min_route_progress,
            max_route_progress=route_throttle_max_route_progress,
        )
        route_late_throttle_ceiling_loss = _route_throttle_ceiling_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            ceiling=route_late_throttle_ceiling,
            min_speed_norm=route_late_throttle_min_speed_norm,
            min_route_progress=route_late_throttle_min_route_progress,
            max_route_progress=route_late_throttle_max_route_progress,
        )
        route_action_override_loss = _route_action_override_loss(
            logits=replay_policy.logits,
            privileged_ego=batch.privileged_ego,
            action_id=route_action_override_id,
            min_speed_norm=route_action_override_min_speed_norm,
            min_route_progress=route_action_override_min_route_progress,
            max_route_progress=route_action_override_max_route_progress,
        )
        total_loss = (
            float(behavior_weight) * losses.total
            + float(imitation_weight) * imitation_loss
            + float(moving_steer_false_positive_weight) * moving_steer_false_positive_loss
            + float(moving_steer_binary_weight) * moving_steer_binary_loss
            + float(moving_steer_nonmoving_margin_weight) * moving_steer_nonmoving_margin_loss
            + float(route_late_moving_steer_margin_weight) * route_late_moving_steer_margin_loss
            + float(route_late_right_steer_margin_weight) * route_late_right_steer_margin_loss
            + float(action_mass_cap_weight) * action_mass_cap_loss
            + float(route_speed_action_mass_cap_weight) * route_speed_action_mass_cap_loss
            + float(route_action_mass_cap_weight) * route_action_mass_cap_loss
            + float(route_action_logit_cap_weight) * route_action_logit_cap_loss
            + float(route_late_action_logit_cap_weight) * route_late_action_logit_cap_loss
            + float(route_throttle_ceiling_weight) * route_throttle_ceiling_loss
            + float(route_late_throttle_ceiling_weight) * route_late_throttle_ceiling_loss
            + float(route_action_override_weight) * route_action_override_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        policy_optimizer.step()
        scalars = losses.detached_scalars()
        scalars["imitation"] = float(imitation_loss.detach().cpu().item())
        scalars["moving_steer_false_positive"] = float(
            moving_steer_false_positive_loss.detach().cpu().item()
        )
        scalars["moving_steer_binary"] = float(moving_steer_binary_loss.detach().cpu().item())
        scalars["moving_steer_nonmoving_margin"] = float(
            moving_steer_nonmoving_margin_loss.detach().cpu().item()
        )
        scalars["route_late_moving_steer_margin"] = float(
            route_late_moving_steer_margin_loss.detach().cpu().item()
        )
        scalars["route_late_right_steer_margin"] = float(
            route_late_right_steer_margin_loss.detach().cpu().item()
        )
        scalars["action_mass_cap"] = float(action_mass_cap_loss.detach().cpu().item())
        scalars["route_speed_action_mass_cap"] = float(
            route_speed_action_mass_cap_loss.detach().cpu().item()
        )
        scalars["route_action_mass_cap"] = float(route_action_mass_cap_loss.detach().cpu().item())
        scalars["route_action_logit_cap"] = float(
            route_action_logit_cap_loss.detach().cpu().item()
        )
        scalars["route_late_action_logit_cap"] = float(
            route_late_action_logit_cap_loss.detach().cpu().item()
        )
        scalars["route_throttle_ceiling"] = float(route_throttle_ceiling_loss.detach().cpu().item())
        scalars["route_late_throttle_ceiling"] = float(
            route_late_throttle_ceiling_loss.detach().cpu().item()
        )
        scalars["route_action_override"] = float(route_action_override_loss.detach().cpu().item())
        scalars["combined_total"] = float(total_loss.detach().cpu().item())
        policy_history.append(scalars)
        valid_action_ids = bool(
            (rollout.action.detach() >= 0).all()
            and (rollout.action.detach() < wm_cfg.action_dim).all()
            and torch.isfinite(rollout.logits.detach()).all()
        )

    checkpoint_common = {
        "stage": "stage2_privileged_policy_clean",
        "source": REPLAY_SOURCE,
        "clean_gate_evidence": False,
        "trainer": "train_stage2_privileged_from_replay",
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "batch_size": int(batch_size),
        "world_model_steps": int(world_model_steps),
        "policy_steps": int(policy_steps),
        "pdf_stage1_strict": bool(pdf_stage1_strict),
        "imitation_weight": float(imitation_weight),
        "behavior_weight": float(behavior_weight),
        "behavior_gamma": float(behavior_gamma),
        "behavior_lambda": float(behavior_lambda),
        "behavior_actor_weight": float(behavior_actor_weight),
        "behavior_critic_weight": float(behavior_critic_weight),
        "behavior_entropy_weight": float(behavior_entropy_weight),
        "imitation_class_balance": bool(imitation_class_balance),
        "imitation_balance_power": float(imitation_balance_power),
        "steer_sequence_fraction": float(steer_sequence_fraction),
        "moving_steer_false_positive_weight": float(moving_steer_false_positive_weight),
        "moving_steer_binary_weight": float(moving_steer_binary_weight),
        "moving_steer_nonmoving_margin_weight": float(moving_steer_nonmoving_margin_weight),
        "moving_steer_nonmoving_margin": float(moving_steer_nonmoving_margin),
        "route_late_moving_steer_margin_weight": float(route_late_moving_steer_margin_weight),
        "route_late_moving_steer_margin": float(route_late_moving_steer_margin),
        "route_late_moving_steer_min_speed_norm": float(route_late_moving_steer_min_speed_norm),
        "route_late_moving_steer_min_route_progress": float(
            route_late_moving_steer_min_route_progress
        ),
        "route_late_moving_steer_max_route_progress": float(
            route_late_moving_steer_max_route_progress
        ),
        "route_late_right_steer_margin_weight": float(route_late_right_steer_margin_weight),
        "route_late_right_steer_margin": float(route_late_right_steer_margin),
        "route_late_right_steer_min_speed_norm": float(route_late_right_steer_min_speed_norm),
        "route_late_right_steer_min_route_progress": float(
            route_late_right_steer_min_route_progress
        ),
        "route_late_right_steer_max_route_progress": float(
            route_late_right_steer_max_route_progress
        ),
        "world_model_action_aux_weight": float(world_model_action_aux_weight),
        "component_sample_weights": resolved_component_weights,
        "component_sampler_stats": component_sampler_stats,
        "action_mass_cap_weight": float(action_mass_cap_weight),
        "action_mass_caps": {str(key): value for key, value in resolved_action_mass_caps.items()},
        "route_speed_action_mass_cap_weight": float(route_speed_action_mass_cap_weight),
        "route_speed_action_ids": list(resolved_route_speed_action_ids),
        "route_speed_action_mass_cap": float(route_speed_action_mass_cap),
        "route_speed_min_speed_norm": float(route_speed_min_speed_norm),
        "route_speed_min_route_progress": float(route_speed_min_route_progress),
        "route_speed_max_route_progress": float(route_speed_max_route_progress),
        "route_action_mass_cap_weight": float(route_action_mass_cap_weight),
        "route_action_mass_cap_ids": list(resolved_route_action_mass_cap_ids),
        "route_action_mass_cap": float(route_action_mass_cap),
        "route_action_mass_cap_min_speed_norm": float(route_action_mass_cap_min_speed_norm),
        "route_action_mass_cap_min_route_progress": float(
            route_action_mass_cap_min_route_progress
        ),
        "route_action_mass_cap_max_route_progress": float(
            route_action_mass_cap_max_route_progress
        ),
        "route_action_logit_cap_weight": float(route_action_logit_cap_weight),
        "route_action_logit_cap_ids": list(resolved_route_action_logit_cap_ids),
        "route_action_logit_cap_margin": float(route_action_logit_cap_margin),
        "route_action_logit_cap_min_speed_norm": float(route_action_logit_cap_min_speed_norm),
        "route_action_logit_cap_min_route_progress": float(
            route_action_logit_cap_min_route_progress
        ),
        "route_action_logit_cap_max_route_progress": float(
            route_action_logit_cap_max_route_progress
        ),
        "route_late_action_logit_cap_weight": float(route_late_action_logit_cap_weight),
        "route_late_action_logit_cap_ids": list(resolved_route_late_action_logit_cap_ids),
        "route_late_action_logit_cap_margin": float(route_late_action_logit_cap_margin),
        "route_late_action_logit_cap_min_speed_norm": float(
            route_late_action_logit_cap_min_speed_norm
        ),
        "route_late_action_logit_cap_min_route_progress": float(
            route_late_action_logit_cap_min_route_progress
        ),
        "route_late_action_logit_cap_max_route_progress": float(
            route_late_action_logit_cap_max_route_progress
        ),
        "route_throttle_ceiling_weight": float(route_throttle_ceiling_weight),
        "route_throttle_ceiling": float(route_throttle_ceiling),
        "route_throttle_min_speed_norm": float(route_throttle_min_speed_norm),
        "route_throttle_min_route_progress": float(route_throttle_min_route_progress),
        "route_throttle_max_route_progress": float(route_throttle_max_route_progress),
        "route_late_throttle_ceiling_weight": float(route_late_throttle_ceiling_weight),
        "route_late_throttle_ceiling": float(route_late_throttle_ceiling),
        "route_late_throttle_min_speed_norm": float(route_late_throttle_min_speed_norm),
        "route_late_throttle_min_route_progress": float(
            route_late_throttle_min_route_progress
        ),
        "route_late_throttle_max_route_progress": float(
            route_late_throttle_max_route_progress
        ),
        "route_action_override_weight": float(route_action_override_weight),
        "route_action_override_id": int(route_action_override_id),
        "route_action_override_min_speed_norm": float(route_action_override_min_speed_norm),
        "route_action_override_min_route_progress": float(route_action_override_min_route_progress),
        "route_action_override_max_route_progress": float(route_action_override_max_route_progress),
        "sampler": sampler_stats,
        "replay_manifest": {
            "path": str(replay_manifest_path),
            "sha256": sha256_file(Path(replay_manifest_path)),
        },
        "data_files": replay_files,
    }
    world_model_checkpoint_path = save_checkpoint(
        output / "stage2_privileged_world_model_replay.pt",
        {
            **checkpoint_common,
            "checkpoint_role": "world_model",
            "model_state_dict": world_model.state_dict(),
            "config": wm_cfg.__dict__,
            "metrics": {
                "initial_total": wm_history[0]["combined_total"],
                "final_total": wm_history[-1]["combined_total"],
                "final_base_total": wm_history[-1]["total"],
                "final_action_aux": wm_history[-1]["action_aux"],
                "final_decoder": wm_history[-1]["decoder"],
                "final_reward": wm_history[-1]["reward"],
                "final_continue": wm_history[-1]["continue"],
            },
        },
    )
    policy_checkpoint_path = save_checkpoint(
        output / "stage2_privileged_policy_replay.pt",
        {
            **checkpoint_common,
            "checkpoint_role": "policy",
            "model_state_dict": policy.state_dict(),
            "config": policy_cfg.__dict__,
            "metrics": {
                "initial_total": policy_history[0]["combined_total"],
                "final_total": policy_history[-1]["combined_total"],
                "final_behavior_total": policy_history[-1]["total"],
                "final_imitation": policy_history[-1]["imitation"],
                "final_moving_steer_false_positive": policy_history[-1][
                    "moving_steer_false_positive"
                ],
                "final_moving_steer_binary": policy_history[-1]["moving_steer_binary"],
                "final_moving_steer_nonmoving_margin": policy_history[-1][
                    "moving_steer_nonmoving_margin"
                ],
                "final_route_late_moving_steer_margin": policy_history[-1][
                    "route_late_moving_steer_margin"
                ],
                "final_route_late_right_steer_margin": policy_history[-1][
                    "route_late_right_steer_margin"
                ],
                "final_action_mass_cap": policy_history[-1]["action_mass_cap"],
                "final_route_speed_action_mass_cap": policy_history[-1][
                    "route_speed_action_mass_cap"
                ],
                "final_route_action_mass_cap": policy_history[-1]["route_action_mass_cap"],
                "final_route_action_logit_cap": policy_history[-1][
                    "route_action_logit_cap"
                ],
                "final_route_late_action_logit_cap": policy_history[-1][
                    "route_late_action_logit_cap"
                ],
                "final_route_throttle_ceiling": policy_history[-1]["route_throttle_ceiling"],
                "final_route_late_throttle_ceiling": policy_history[-1][
                    "route_late_throttle_ceiling"
                ],
                "final_route_action_override": policy_history[-1]["route_action_override"],
                "final_actor": policy_history[-1]["actor"],
                "final_critic": policy_history[-1]["critic"],
                "final_entropy": policy_history[-1]["entropy"],
                "valid_action_ids": valid_action_ids,
            },
        },
    )
    manifest = {
        **checkpoint_common,
        "world_model": {
            "path": str(world_model_checkpoint_path),
            "sha256": sha256_file(world_model_checkpoint_path),
            "config": wm_cfg.__dict__,
            "metrics": {
                "initial_total": wm_history[0]["combined_total"],
                "final_total": wm_history[-1]["combined_total"],
                "final_base_total": wm_history[-1]["total"],
                "final_action_aux": wm_history[-1]["action_aux"],
            },
        },
        "policy": {
            "path": str(policy_checkpoint_path),
            "sha256": sha256_file(policy_checkpoint_path),
            "config": policy_cfg.__dict__,
            "metrics": {
                "initial_total": policy_history[0]["combined_total"],
                "final_total": policy_history[-1]["combined_total"],
                "final_behavior_total": policy_history[-1]["total"],
                "final_imitation": policy_history[-1]["imitation"],
                "final_moving_steer_false_positive": policy_history[-1][
                    "moving_steer_false_positive"
                ],
                "final_moving_steer_binary": policy_history[-1]["moving_steer_binary"],
                "final_moving_steer_nonmoving_margin": policy_history[-1][
                    "moving_steer_nonmoving_margin"
                ],
                "final_route_late_moving_steer_margin": policy_history[-1][
                    "route_late_moving_steer_margin"
                ],
                "final_route_late_right_steer_margin": policy_history[-1][
                    "route_late_right_steer_margin"
                ],
                "final_action_mass_cap": policy_history[-1]["action_mass_cap"],
                "final_route_speed_action_mass_cap": policy_history[-1][
                    "route_speed_action_mass_cap"
                ],
                "final_route_action_mass_cap": policy_history[-1]["route_action_mass_cap"],
                "final_route_action_logit_cap": policy_history[-1][
                    "route_action_logit_cap"
                ],
                "final_route_late_action_logit_cap": policy_history[-1][
                    "route_late_action_logit_cap"
                ],
                "final_route_throttle_ceiling": policy_history[-1]["route_throttle_ceiling"],
                "final_route_late_throttle_ceiling": policy_history[-1][
                    "route_late_throttle_ceiling"
                ],
                "final_route_action_override": policy_history[-1]["route_action_override"],
                "valid_action_ids": valid_action_ids,
            },
        },
        "clean_eval_config_patch": {
            "checkpoints": {
                "world_model": {
                    "path": str(world_model_checkpoint_path),
                    "sha256": sha256_file(world_model_checkpoint_path),
                    "commit": None,
                },
                "policy": {
                    "path": str(policy_checkpoint_path),
                    "sha256": sha256_file(policy_checkpoint_path),
                    "commit": None,
                },
            }
        },
        "warning": "Replay-trained checkpoints are inputs for clean eval, not clean PASS evidence.",
    }
    manifest_path = output / "stage2_privileged_replay_checkpoint_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return ReplayTrainingResult(
        output_dir=output,
        world_model_checkpoint_path=world_model_checkpoint_path,
        policy_checkpoint_path=policy_checkpoint_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 privileged checkpoints from replay")
    parser.add_argument("--replay", action="append", required=True, help="Replay .npz file or directory; repeatable")
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--world-model-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-learning-rate", type=float, default=None)
    parser.add_argument("--pdf-stage1-strict", action="store_true")
    parser.add_argument("--imitation-weight", type=float, default=0.0)
    parser.add_argument("--behavior-weight", type=float, default=1.0)
    parser.add_argument("--behavior-gamma", type=float, default=0.99)
    parser.add_argument("--behavior-lambda", type=float, default=1.0)
    parser.add_argument("--behavior-actor-weight", type=float, default=0.1)
    parser.add_argument("--behavior-critic-weight", type=float, default=1.0)
    parser.add_argument("--behavior-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--imitation-class-balance", action="store_true")
    parser.add_argument("--imitation-balance-power", type=float, default=1.0)
    parser.add_argument("--steer-sequence-fraction", type=float, default=0.0)
    parser.add_argument("--moving-steer-false-positive-weight", type=float, default=0.0)
    parser.add_argument("--moving-steer-binary-weight", type=float, default=0.0)
    parser.add_argument("--moving-steer-nonmoving-margin-weight", type=float, default=0.0)
    parser.add_argument("--moving-steer-nonmoving-margin", type=float, default=0.25)
    parser.add_argument("--route-late-moving-steer-margin-weight", type=float, default=0.0)
    parser.add_argument("--route-late-moving-steer-margin", type=float, default=0.25)
    parser.add_argument("--route-late-moving-steer-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-late-moving-steer-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-late-moving-steer-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-late-right-steer-margin-weight", type=float, default=0.0)
    parser.add_argument("--route-late-right-steer-margin", type=float, default=0.25)
    parser.add_argument("--route-late-right-steer-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-late-right-steer-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-late-right-steer-max-route-progress", type=float, default=1.0)
    parser.add_argument("--world-model-action-aux-weight", type=float, default=0.0)
    parser.add_argument("--component-sample-weight", action="append", default=[])
    parser.add_argument("--action-mass-cap-weight", type=float, default=0.0)
    parser.add_argument("--action-mass-cap", action="append", default=[])
    parser.add_argument("--route-speed-action-mass-cap-weight", type=float, default=0.0)
    parser.add_argument("--route-speed-action-ids", default="")
    parser.add_argument("--route-speed-action-mass-cap", type=float, default=1.0)
    parser.add_argument("--route-speed-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-speed-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-speed-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-action-mass-cap-weight", type=float, default=0.0)
    parser.add_argument("--route-action-mass-cap-ids", default="")
    parser.add_argument("--route-action-mass-cap", type=float, default=1.0)
    parser.add_argument("--route-action-mass-cap-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-action-mass-cap-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-action-mass-cap-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-action-logit-cap-weight", type=float, default=0.0)
    parser.add_argument("--route-action-logit-cap-ids", default="")
    parser.add_argument("--route-action-logit-cap-margin", type=float, default=0.0)
    parser.add_argument("--route-action-logit-cap-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-action-logit-cap-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-action-logit-cap-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-late-action-logit-cap-weight", type=float, default=0.0)
    parser.add_argument("--route-late-action-logit-cap-ids", default="")
    parser.add_argument("--route-late-action-logit-cap-margin", type=float, default=0.0)
    parser.add_argument("--route-late-action-logit-cap-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-late-action-logit-cap-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-late-action-logit-cap-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-throttle-ceiling-weight", type=float, default=0.0)
    parser.add_argument("--route-throttle-ceiling", type=float, default=1.0)
    parser.add_argument("--route-throttle-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-throttle-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-throttle-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-late-throttle-ceiling-weight", type=float, default=0.0)
    parser.add_argument("--route-late-throttle-ceiling", type=float, default=1.0)
    parser.add_argument("--route-late-throttle-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-late-throttle-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-late-throttle-max-route-progress", type=float, default=1.0)
    parser.add_argument("--route-action-override-weight", type=float, default=0.0)
    parser.add_argument("--route-action-override-id", type=int, default=-1)
    parser.add_argument("--route-action-override-min-speed-norm", type=float, default=0.0)
    parser.add_argument("--route-action-override-min-route-progress", type=float, default=0.0)
    parser.add_argument("--route-action-override-max-route-progress", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = train_stage2_privileged_from_replay(
        replay_paths=[Path(item) for item in args.replay],
        replay_manifest_path=args.replay_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        world_model_steps=args.world_model_steps,
        policy_steps=args.policy_steps,
        learning_rate=args.learning_rate,
        policy_learning_rate=args.policy_learning_rate,
        pdf_stage1_strict=args.pdf_stage1_strict,
        imitation_weight=args.imitation_weight,
        behavior_weight=args.behavior_weight,
        behavior_gamma=args.behavior_gamma,
        behavior_lambda=args.behavior_lambda,
        behavior_actor_weight=args.behavior_actor_weight,
        behavior_critic_weight=args.behavior_critic_weight,
        behavior_entropy_weight=args.behavior_entropy_weight,
        imitation_class_balance=args.imitation_class_balance,
        imitation_balance_power=args.imitation_balance_power,
        steer_sequence_fraction=args.steer_sequence_fraction,
        moving_steer_false_positive_weight=args.moving_steer_false_positive_weight,
        moving_steer_binary_weight=args.moving_steer_binary_weight,
        moving_steer_nonmoving_margin_weight=args.moving_steer_nonmoving_margin_weight,
        moving_steer_nonmoving_margin=args.moving_steer_nonmoving_margin,
        route_late_moving_steer_margin_weight=args.route_late_moving_steer_margin_weight,
        route_late_moving_steer_margin=args.route_late_moving_steer_margin,
        route_late_moving_steer_min_speed_norm=args.route_late_moving_steer_min_speed_norm,
        route_late_moving_steer_min_route_progress=args.route_late_moving_steer_min_route_progress,
        route_late_moving_steer_max_route_progress=args.route_late_moving_steer_max_route_progress,
        route_late_right_steer_margin_weight=args.route_late_right_steer_margin_weight,
        route_late_right_steer_margin=args.route_late_right_steer_margin,
        route_late_right_steer_min_speed_norm=args.route_late_right_steer_min_speed_norm,
        route_late_right_steer_min_route_progress=args.route_late_right_steer_min_route_progress,
        route_late_right_steer_max_route_progress=args.route_late_right_steer_max_route_progress,
        world_model_action_aux_weight=args.world_model_action_aux_weight,
        component_sample_weights=_parse_component_sample_weight_args(args.component_sample_weight),
        action_mass_cap_weight=args.action_mass_cap_weight,
        action_mass_caps=_parse_action_mass_cap_args(args.action_mass_cap),
        route_speed_action_mass_cap_weight=args.route_speed_action_mass_cap_weight,
        route_speed_action_ids=_parse_action_id_list(args.route_speed_action_ids),
        route_speed_action_mass_cap=args.route_speed_action_mass_cap,
        route_speed_min_speed_norm=args.route_speed_min_speed_norm,
        route_speed_min_route_progress=args.route_speed_min_route_progress,
        route_speed_max_route_progress=args.route_speed_max_route_progress,
        route_action_mass_cap_weight=args.route_action_mass_cap_weight,
        route_action_mass_cap_ids=_parse_action_id_list(args.route_action_mass_cap_ids),
        route_action_mass_cap=args.route_action_mass_cap,
        route_action_mass_cap_min_speed_norm=args.route_action_mass_cap_min_speed_norm,
        route_action_mass_cap_min_route_progress=args.route_action_mass_cap_min_route_progress,
        route_action_mass_cap_max_route_progress=args.route_action_mass_cap_max_route_progress,
        route_action_logit_cap_weight=args.route_action_logit_cap_weight,
        route_action_logit_cap_ids=_parse_action_id_list(args.route_action_logit_cap_ids),
        route_action_logit_cap_margin=args.route_action_logit_cap_margin,
        route_action_logit_cap_min_speed_norm=args.route_action_logit_cap_min_speed_norm,
        route_action_logit_cap_min_route_progress=args.route_action_logit_cap_min_route_progress,
        route_action_logit_cap_max_route_progress=args.route_action_logit_cap_max_route_progress,
        route_late_action_logit_cap_weight=args.route_late_action_logit_cap_weight,
        route_late_action_logit_cap_ids=_parse_action_id_list(args.route_late_action_logit_cap_ids),
        route_late_action_logit_cap_margin=args.route_late_action_logit_cap_margin,
        route_late_action_logit_cap_min_speed_norm=args.route_late_action_logit_cap_min_speed_norm,
        route_late_action_logit_cap_min_route_progress=(
            args.route_late_action_logit_cap_min_route_progress
        ),
        route_late_action_logit_cap_max_route_progress=(
            args.route_late_action_logit_cap_max_route_progress
        ),
        route_throttle_ceiling_weight=args.route_throttle_ceiling_weight,
        route_throttle_ceiling=args.route_throttle_ceiling,
        route_throttle_min_speed_norm=args.route_throttle_min_speed_norm,
        route_throttle_min_route_progress=args.route_throttle_min_route_progress,
        route_throttle_max_route_progress=args.route_throttle_max_route_progress,
        route_late_throttle_ceiling_weight=args.route_late_throttle_ceiling_weight,
        route_late_throttle_ceiling=args.route_late_throttle_ceiling,
        route_late_throttle_min_speed_norm=args.route_late_throttle_min_speed_norm,
        route_late_throttle_min_route_progress=args.route_late_throttle_min_route_progress,
        route_late_throttle_max_route_progress=args.route_late_throttle_max_route_progress,
        route_action_override_weight=args.route_action_override_weight,
        route_action_override_id=args.route_action_override_id,
        route_action_override_min_speed_norm=args.route_action_override_min_speed_norm,
        route_action_override_min_route_progress=args.route_action_override_min_route_progress,
        route_action_override_max_route_progress=args.route_action_override_max_route_progress,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"STAGE2_REPLAY_CHECKPOINT_MANIFEST={result.manifest_path}")
    print(f"STAGE2_REPLAY_WORLD_MODEL={result.world_model_checkpoint_path}")
    print(f"STAGE2_REPLAY_POLICY={result.policy_checkpoint_path}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_pdf_stage1_strict_config(
    *,
    imitation_weight: float,
    behavior_weight: float,
    behavior_gamma: float,
    behavior_lambda: float,
    behavior_actor_weight: float,
    behavior_critic_weight: float,
    behavior_entropy_weight: float,
    imitation_class_balance: bool,
    steer_sequence_fraction: float,
    moving_steer_false_positive_weight: float,
    moving_steer_binary_weight: float,
    moving_steer_nonmoving_margin_weight: float,
    route_late_moving_steer_margin_weight: float,
    route_late_right_steer_margin_weight: float,
    world_model_action_aux_weight: float,
    component_sample_weights: dict[str, float] | None,
    action_mass_cap_weight: float,
    action_mass_caps: dict[int, float] | None,
    route_speed_action_mass_cap_weight: float,
    route_speed_action_ids: tuple[int, ...],
    route_action_mass_cap_weight: float,
    route_action_mass_cap_ids: tuple[int, ...],
    route_action_logit_cap_weight: float,
    route_action_logit_cap_ids: tuple[int, ...],
    route_late_action_logit_cap_weight: float,
    route_late_action_logit_cap_ids: tuple[int, ...],
    route_throttle_ceiling_weight: float,
    route_late_throttle_ceiling_weight: float,
    route_action_override_weight: float,
    route_action_override_id: int,
) -> None:
    """Reject diagnostic losses that are outside Raw2Drive Algorithm 1 Stage I."""
    violations: list[str] = []
    if float(behavior_weight) <= 0.0:
        violations.append("behavior_weight must be positive")
    if not 0.0 <= float(behavior_gamma) <= 1.0:
        violations.append("behavior_gamma must be in [0, 1]")
    if not 0.0 <= float(behavior_lambda) <= 1.0:
        violations.append("behavior_lambda must be in [0, 1]")
    for name, value in {
        "behavior_actor_weight": behavior_actor_weight,
        "behavior_critic_weight": behavior_critic_weight,
        "behavior_entropy_weight": behavior_entropy_weight,
    }.items():
        if float(value) < 0.0:
            violations.append(f"{name} must be non-negative")
    if float(imitation_weight) != 0.0:
        violations.append("imitation_weight must be 0")
    if imitation_class_balance:
        violations.append("imitation_class_balance must be disabled")
    if float(steer_sequence_fraction) != 0.0:
        violations.append("steer_sequence_fraction must be 0")
    for name, value in {
        "moving_steer_false_positive_weight": moving_steer_false_positive_weight,
        "moving_steer_binary_weight": moving_steer_binary_weight,
        "moving_steer_nonmoving_margin_weight": moving_steer_nonmoving_margin_weight,
        "route_late_moving_steer_margin_weight": route_late_moving_steer_margin_weight,
        "route_late_right_steer_margin_weight": route_late_right_steer_margin_weight,
        "world_model_action_aux_weight": world_model_action_aux_weight,
        "action_mass_cap_weight": action_mass_cap_weight,
        "route_speed_action_mass_cap_weight": route_speed_action_mass_cap_weight,
        "route_action_mass_cap_weight": route_action_mass_cap_weight,
        "route_action_logit_cap_weight": route_action_logit_cap_weight,
        "route_late_action_logit_cap_weight": route_late_action_logit_cap_weight,
        "route_throttle_ceiling_weight": route_throttle_ceiling_weight,
        "route_late_throttle_ceiling_weight": route_late_throttle_ceiling_weight,
        "route_action_override_weight": route_action_override_weight,
    }.items():
        if float(value) != 0.0:
            violations.append(f"{name} must be 0")
    if component_sample_weights:
        violations.append("component_sample_weights must be empty")
    if action_mass_caps:
        violations.append("action_mass_caps must be empty")
    if route_speed_action_ids:
        violations.append("route_speed_action_ids must be empty")
    if route_action_mass_cap_ids:
        violations.append("route_action_mass_cap_ids must be empty")
    if route_action_logit_cap_ids:
        violations.append("route_action_logit_cap_ids must be empty")
    if route_late_action_logit_cap_ids:
        violations.append("route_late_action_logit_cap_ids must be empty")
    if int(route_action_override_id) != -1:
        violations.append("route_action_override_id must be -1")
    if violations:
        joined = "; ".join(violations)
        raise ValueError(f"pdf_stage1_strict forbids non-PDF Stage I settings: {joined}")


def _make_imitation_class_weights(
    episodes: list[Any],
    *,
    action_dim: int,
    device: torch.device,
    power: float = 1.0,
) -> torch.Tensor:
    if not 0.0 <= float(power) <= 1.0:
        raise ValueError("power must be in [0, 1]")
    counts = torch.zeros(action_dim, dtype=torch.float32)
    for episode in episodes:
        actions = torch.as_tensor(episode.action, dtype=torch.long)
        counts += torch.bincount(actions.clamp(0, action_dim - 1), minlength=action_dim).to(torch.float32)
    present = counts > 0
    weights = torch.zeros(action_dim, dtype=torch.float32)
    if present.any():
        present_counts = counts[present]
        weights[present] = (present_counts.mean() / present_counts).pow(float(power))
    return weights.to(device)


def _parse_component_sample_weight_args(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw_value in values:
        for item in str(raw_value).split(","):
            text = item.strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError("component sample weights must use name=weight")
            name, raw_weight = text.split("=", 1)
            component = name.strip()
            if not component:
                raise ValueError("component sample weight name must not be empty")
            weights[component] = float(raw_weight)
    return _normalize_component_sample_weights(weights)


def _normalize_component_sample_weights(weights: dict[str, float] | None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in (weights or {}).items():
        component = str(key).strip()
        if not component:
            raise ValueError("component sample weight name must not be empty")
        weight = float(value)
        if not torch.isfinite(torch.tensor(weight)):
            raise ValueError("component sample weights must be finite")
        if weight < 0.0:
            raise ValueError("component sample weights must be non-negative")
        normalized[component] = weight
    return normalized


def _parse_action_mass_cap_args(values: list[str]) -> dict[int, float]:
    caps: dict[int, float] = {}
    for raw_value in values:
        for item in str(raw_value).split(","):
            text = item.strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError("action mass caps must use action_id=max_probability")
            raw_action_id, raw_cap = text.split("=", 1)
            try:
                action_id = int(raw_action_id.strip())
            except ValueError as exc:
                raise ValueError("action mass cap action_id must be an integer") from exc
            caps[action_id] = float(raw_cap)
    return _normalize_action_mass_caps(caps)


def _normalize_action_mass_caps(caps: dict[int, float] | None) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for key, value in (caps or {}).items():
        action_id = int(key)
        if action_id < 0 or action_id >= len(RAW2DRIVE_ACTIONS):
            raise ValueError(f"action mass cap action_id must be in [0, {len(RAW2DRIVE_ACTIONS) - 1}]")
        cap = float(value)
        if not torch.isfinite(torch.tensor(cap)):
            raise ValueError("action mass caps must be finite")
        if not 0.0 <= cap <= 1.0:
            raise ValueError("action mass caps must be in [0, 1]")
        normalized[action_id] = cap
    return normalized


def _parse_action_id_list(value: str) -> tuple[int, ...]:
    if str(value).strip() == "":
        return ()
    return _normalize_action_id_tuple(tuple(int(item.strip()) for item in str(value).split(",") if item.strip()))


def _normalize_action_id_tuple(action_ids: tuple[int, ...]) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw_action_id in action_ids:
        action_id = int(raw_action_id)
        if action_id < 0 or action_id >= len(RAW2DRIVE_ACTIONS):
            raise ValueError(f"action_id must be in [0, {len(RAW2DRIVE_ACTIONS) - 1}]")
        if action_id not in normalized:
            normalized.append(action_id)
    return tuple(normalized)


def _make_episode_sample_weights(
    episodes: list[Any],
    *,
    component_sample_weights: dict[str, float],
) -> list[float]:
    weights: list[float] = []
    for episode in episodes:
        component = _episode_component_name(Path(episode.path))
        weights.append(float(component_sample_weights.get(component, 1.0)))
    if weights and sum(weights) <= 0.0:
        raise ValueError("at least one episode sample weight must be positive")
    return weights


def _make_component_sampler_stats(
    episodes: list[Any],
    *,
    sequence_length: int,
    component_sample_weights: dict[str, float],
) -> dict[str, Any]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    components: dict[str, dict[str, Any]] = {}
    episode_entries: list[dict[str, Any]] = []
    for episode in episodes:
        component = _episode_component_name(Path(episode.path))
        length = int(episode.length)
        weight = float(component_sample_weights.get(component, 1.0))
        included = length >= int(sequence_length)
        window_count = max(0, length - int(sequence_length) + 1)
        stats = components.setdefault(
            component,
            {
                "episodes": 0,
                "included_episodes": 0,
                "dropped_short_episodes": 0,
                "total_steps": 0,
                "included_steps": 0,
                "window_count": 0,
                "raw_weight_sum": 0.0,
                "included_weight_sum": 0.0,
                "requested_weight": weight,
            },
        )
        stats["episodes"] += 1
        stats["total_steps"] += length
        stats["raw_weight_sum"] += weight
        if included:
            stats["included_episodes"] += 1
            stats["included_steps"] += length
            stats["window_count"] += window_count
            stats["included_weight_sum"] += weight
        else:
            stats["dropped_short_episodes"] += 1
        episode_entries.append(
            {
                "path": str(episode.path),
                "component": component,
                "length": length,
                "included": bool(included),
                "window_count": int(window_count),
                "requested_weight": weight,
            }
        )

    for component, weight in component_sample_weights.items():
        component_stats = components.get(component)
        included_count = int(component_stats["included_episodes"]) if component_stats else 0
        if float(weight) > 0.0 and included_count == 0:
            raise ValueError(
                f"component {component!r} has positive sample weight but no episodes "
                f"long enough for sequence_length={int(sequence_length)}"
            )

    total_included_weight = sum(float(stats["included_weight_sum"]) for stats in components.values())
    configured_probabilities = {
        component: (
            float(stats["included_weight_sum"]) / float(total_included_weight)
            if total_included_weight > 0.0
            else 0.0
        )
        for component, stats in components.items()
    }
    return {
        "sequence_length": int(sequence_length),
        "components": components,
        "configured_episode_sample_prob_by_component": configured_probabilities,
        "episodes": episode_entries,
    }


def _episode_component_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("episode_"):
        suffix = stem.removeprefix("episode_")
        maybe_name, separator, maybe_index = suffix.rpartition("_")
        if separator and maybe_name and maybe_index.isdigit():
            return maybe_name
    return path.parent.name


def _moving_steer_action_ids() -> tuple[int, ...]:
    return tuple(
        action.action_id
        for action in RAW2DRIVE_ACTIONS
        if float(action.throttle) > 0.0 and abs(float(action.steer)) > 0.0 and not action.reverse
    )


def _right_moving_steer_action_ids() -> tuple[int, ...]:
    return tuple(
        action.action_id
        for action in RAW2DRIVE_ACTIONS
        if float(action.throttle) > 0.0 and float(action.steer) > 0.0 and not action.reverse
    )


def _moving_steer_action_mask(*, action_dim: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(action_dim, dtype=torch.bool, device=device)
    for action_id in _moving_steer_action_ids():
        mask[action_id] = True
    return mask


def _right_moving_steer_action_mask(*, action_dim: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(action_dim, dtype=torch.bool, device=device)
    for action_id in _right_moving_steer_action_ids():
        mask[action_id] = True
    return mask


def _moving_steer_false_positive_loss(
    *,
    logits: torch.Tensor,
    action: torch.Tensor,
    moving_steer_mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_action = action.reshape(-1)
    label_is_moving_steer = moving_steer_mask[flat_action]
    nonmoving_logits = flat_logits[~label_is_moving_steer]
    if int(nonmoving_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    moving_mass = torch.softmax(nonmoving_logits, dim=-1)[:, moving_steer_mask].sum(dim=-1)
    return -torch.log1p(-moving_mass.clamp(max=1.0 - 1e-6)).mean()


def _moving_steer_binary_loss(
    *,
    logits: torch.Tensor,
    action: torch.Tensor,
    moving_steer_mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_action = action.reshape(-1)
    target = moving_steer_mask[flat_action].to(flat_logits.dtype)
    moving_mass = torch.softmax(flat_logits, dim=-1)[:, moving_steer_mask].sum(dim=-1)
    moving_mass = moving_mass.clamp(min=1e-6, max=1.0 - 1e-6)
    return F.binary_cross_entropy(moving_mass, target)


def _moving_steer_nonmoving_margin_loss(
    *,
    logits: torch.Tensor,
    action: torch.Tensor,
    moving_steer_mask: torch.Tensor,
    margin: float = 0.25,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_action = action.reshape(-1)
    label_is_moving_steer = moving_steer_mask[flat_action]
    nonmoving_logits = flat_logits[~label_is_moving_steer]
    if int(nonmoving_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    moving_top = nonmoving_logits[:, moving_steer_mask].max(dim=-1).values
    nonmoving_top = nonmoving_logits[:, ~moving_steer_mask].max(dim=-1).values
    return F.relu(moving_top - nonmoving_top + float(margin)).mean()


def _route_late_moving_steer_margin_loss(
    *,
    logits: torch.Tensor,
    privileged_ego: torch.Tensor,
    action: torch.Tensor,
    moving_steer_mask: torch.Tensor,
    margin: float,
    min_speed_norm: float,
    min_route_progress: float,
    max_route_progress: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_action = action.reshape(-1)
    flat_ego = privileged_ego.reshape(-1, privileged_ego.shape[-1])
    speed_norm = flat_ego[:, 0]
    route_progress = flat_ego[:, 1]
    label_is_moving_steer = moving_steer_mask[flat_action]
    late_nonmoving = (
        ~label_is_moving_steer
        & (speed_norm >= float(min_speed_norm))
        & (route_progress >= float(min_route_progress))
        & (route_progress <= float(max_route_progress))
    )
    selected_logits = flat_logits[late_nonmoving]
    if int(selected_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    moving_top = selected_logits[:, moving_steer_mask].max(dim=-1).values
    nonmoving_top = selected_logits[:, ~moving_steer_mask].max(dim=-1).values
    return F.relu(nonmoving_top - moving_top + float(margin)).mean()


def _route_late_right_steer_margin_loss(
    *,
    logits: torch.Tensor,
    privileged_ego: torch.Tensor,
    right_moving_steer_mask: torch.Tensor,
    margin: float,
    min_speed_norm: float,
    min_route_progress: float,
    max_route_progress: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_ego = privileged_ego.reshape(-1, privileged_ego.shape[-1])
    speed_norm = flat_ego[:, 0]
    route_progress = flat_ego[:, 1]
    late_route = (
        (speed_norm >= float(min_speed_norm))
        & (route_progress >= float(min_route_progress))
        & (route_progress <= float(max_route_progress))
    )
    selected_logits = flat_logits[late_route]
    if int(selected_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    right_top = selected_logits[:, right_moving_steer_mask].max(dim=-1).values
    nonright_top = selected_logits[:, ~right_moving_steer_mask].max(dim=-1).values
    return F.relu(nonright_top - right_top + float(margin)).mean()


def _action_mass_cap_loss(
    *,
    logits: torch.Tensor,
    action_mass_caps: dict[int, float],
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    if not action_mass_caps:
        return flat_logits.sum() * 0.0
    action_mass = torch.softmax(flat_logits, dim=-1).mean(dim=0)
    penalties = [
        F.relu(action_mass[action_id] - float(cap)).pow(2)
        for action_id, cap in action_mass_caps.items()
    ]
    return torch.stack(penalties).mean()


def _route_speed_action_mass_cap_loss(
    *,
    logits: torch.Tensor,
    privileged_ego: torch.Tensor,
    action_ids: tuple[int, ...],
    cap: float,
    min_speed_norm: float,
    min_route_progress: float,
    max_route_progress: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    if not action_ids:
        return flat_logits.sum() * 0.0
    flat_ego = privileged_ego.reshape(-1, privileged_ego.shape[-1])
    speed_norm = flat_ego[:, 0]
    route_progress = flat_ego[:, 1]
    mask = (
        (speed_norm >= float(min_speed_norm))
        & (route_progress >= float(min_route_progress))
        & (route_progress <= float(max_route_progress))
    )
    selected_logits = flat_logits[mask]
    if int(selected_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    probs = torch.softmax(selected_logits, dim=-1)
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=logits.device)
    selected_mass = probs.index_select(dim=-1, index=ids).sum(dim=-1)
    return F.relu(selected_mass - float(cap)).pow(2).mean()


def _route_throttle_ceiling_loss(
    *,
    logits: torch.Tensor,
    privileged_ego: torch.Tensor,
    ceiling: float,
    min_speed_norm: float,
    min_route_progress: float,
    max_route_progress: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_ego = privileged_ego.reshape(-1, privileged_ego.shape[-1])
    speed_norm = flat_ego[:, 0]
    route_progress = flat_ego[:, 1]
    mask = (
        (speed_norm >= float(min_speed_norm))
        & (route_progress >= float(min_route_progress))
        & (route_progress <= float(max_route_progress))
    )
    selected_logits = flat_logits[mask]
    if int(selected_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    throttle = torch.as_tensor(
        [float(action.throttle) for action in RAW2DRIVE_ACTIONS],
        dtype=selected_logits.dtype,
        device=selected_logits.device,
    )
    expected_throttle = torch.softmax(selected_logits, dim=-1).matmul(throttle)
    return F.relu(expected_throttle - float(ceiling)).pow(2).mean()


def _route_action_logit_cap_loss(
    *,
    logits: torch.Tensor,
    privileged_ego: torch.Tensor,
    action_ids: tuple[int, ...],
    margin: float,
    min_speed_norm: float,
    min_route_progress: float,
    max_route_progress: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    if not action_ids:
        return flat_logits.sum() * 0.0
    flat_ego = privileged_ego.reshape(-1, privileged_ego.shape[-1])
    speed_norm = flat_ego[:, 0]
    route_progress = flat_ego[:, 1]
    mask = (
        (speed_norm >= float(min_speed_norm))
        & (route_progress >= float(min_route_progress))
        & (route_progress <= float(max_route_progress))
    )
    selected_logits = flat_logits[mask]
    if int(selected_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    capped_ids = torch.as_tensor(action_ids, dtype=torch.long, device=logits.device)
    capped = selected_logits.index_select(dim=-1, index=capped_ids).max(dim=-1).values
    keep_mask = torch.ones(selected_logits.shape[-1], dtype=torch.bool, device=logits.device)
    keep_mask[capped_ids] = False
    if not bool(keep_mask.any()):
        return flat_logits.sum() * 0.0
    keep_top = selected_logits[:, keep_mask].max(dim=-1).values
    return F.relu(capped - keep_top + float(margin)).mean()


def _route_action_override_loss(
    *,
    logits: torch.Tensor,
    privileged_ego: torch.Tensor,
    action_id: int,
    min_speed_norm: float,
    min_route_progress: float,
    max_route_progress: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    if int(action_id) < 0:
        return flat_logits.sum() * 0.0
    if int(action_id) >= flat_logits.shape[-1]:
        raise ValueError("route action override action_id is outside logits action dimension")
    flat_ego = privileged_ego.reshape(-1, privileged_ego.shape[-1])
    speed_norm = flat_ego[:, 0]
    route_progress = flat_ego[:, 1]
    mask = (
        (speed_norm >= float(min_speed_norm))
        & (route_progress >= float(min_route_progress))
        & (route_progress <= float(max_route_progress))
    )
    selected_logits = flat_logits[mask]
    if int(selected_logits.shape[0]) == 0:
        return flat_logits.sum() * 0.0
    target = torch.full(
        (selected_logits.shape[0],),
        int(action_id),
        dtype=torch.long,
        device=selected_logits.device,
    )
    return F.cross_entropy(selected_logits, target)


def _world_model_action_auxiliary_loss(
    *,
    latent: torch.Tensor,
    action: torch.Tensor,
    action_head: torch.nn.Linear,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    logits = action_head(latent)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        action.reshape(-1),
        weight=class_weights,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()
