"""Calibration-only vision/ego logit-triplet sweep with frozen policy backbone."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
import yaml

from rcs_ac_v1.model import MODALITY_NAMES
from rcs_toy_nav_env import RCSToyNavEnv
from run_stage1_targeted_monotonicity_calibration import deep_apply_overrides
from run_very_weak_probe import ensure_dir, latest_checkpoint, save_yaml
from stage1_warmup.config import ExperimentConfig, load_experiment_config
from stage1_warmup.corruption import apply_shared_corruption
from stage1_warmup.model import Stage1WarmupActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-backbone vision/ego logit-triplet calibration sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage1_vision_ego_logit_triplet_calibration.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def obs_to_tensor_batch(obs_list: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(np.stack([obs[key] for obs in obs_list], axis=0), dtype=torch.float32, device=device)
        for key in MODALITY_NAMES
    }


def build_model_and_cfg(config_path: str, checkpoint_path: Path, seed: int, device: str, overrides: dict[str, Any]) -> tuple[ExperimentConfig, Stage1WarmupActorCritic]:
    cfg = load_experiment_config(config_path)
    cfg.seed = seed
    cfg.device = device
    deep_apply_overrides(cfg, overrides)
    model = Stage1WarmupActorCritic(
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        encoder_output_dim=cfg.model.encoder_output_dim,
        attention_hidden_dim=cfg.model.attention_hidden_dim,
        recurrent_hidden_dim=cfg.model.recurrent_hidden_dim,
        attention_temperature=cfg.model.attention_temperature,
        use_feature_layer_norm=cfg.model.use_feature_layer_norm,
        actor_hidden_dims=cfg.model.actor_hidden_dims,
        critic_hidden_dims=cfg.model.critic_hidden_dims,
        reliability_hidden_dims=cfg.model.reliability_hidden_dims,
        reliability_input_mode=cfg.model.reliability_input_mode,
        reliability_raw_adapter_dim=cfg.model.reliability_raw_adapter_dim,
        reliability_ego_residual_enabled=cfg.model.reliability_ego_residual_enabled,
        reliability_ego_raw_adapter_dim=cfg.model.reliability_ego_raw_adapter_dim,
        reliability_ego_residual_input_mode=cfg.model.reliability_ego_residual_input_mode,
        reliability_ego_residual_apply_to_provisional=cfg.model.reliability_ego_residual_apply_to_provisional,
        action_dim=2,
        action_log_std_init=cfg.model.action_log_std_init,
    ).to(torch.device(device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    current = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        if key not in current:
            continue
        current_value = current[key]
        if isinstance(current_value, (UninitializedParameter, UninitializedBuffer)):
            compatible[key] = value
            continue
        if tuple(current_value.shape) == tuple(value.shape):
            compatible[key] = value
    current.update(compatible)
    model.load_state_dict(current)
    return cfg, model


def freeze_policy_backbone(model: Stage1WarmupActorCritic) -> None:
    model.freeze_policy_backbone()
    model.unfreeze_reliability_head()


def collect_base_states(model: Stage1WarmupActorCritic, device: torch.device, seed: int, num_states: int) -> list[dict[str, np.ndarray]]:
    env = RCSToyNavEnv()
    env.set_corruption_mode("clean")
    obs, info = env.reset(seed=seed)
    del info
    hidden = model.initial_hidden(batch_size=1, device=device)
    episode_start = torch.ones(1, device=device)
    states: list[dict[str, np.ndarray]] = []
    while len(states) < num_states:
        states.append({key: np.asarray(obs[key], dtype=np.float32).copy() for key in [*MODALITY_NAMES, "aux"]})
        with torch.no_grad():
            output = model.step(
                obs_dict={key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0) for key in MODALITY_NAMES},
                hidden_state=hidden,
                episode_start=episode_start,
                deterministic=True,
            )
        action = output.action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        del reward, info
        hidden = output.hidden_state
        episode_start = torch.tensor([float(terminated or truncated)], device=device)
        hidden = model.reset_hidden(hidden, episode_start)
        if terminated or truncated:
            obs, info = env.reset()
            del info
    return states


def build_paired_calibration_batches(
    base_states: list[dict[str, np.ndarray]],
    device: torch.device,
    rng: np.random.Generator,
    cfg: ExperimentConfig,
    batch_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    batch_states = [base_states[int(idx)] for idx in rng.integers(0, len(base_states), size=batch_size)]
    variants: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in ["clean", "weak_vision", "mild_vision", "weak_ego", "mild_ego"]}
    for state_idx, state in enumerate(batch_states):
        base_obs = {key: state[key] for key in [*MODALITY_NAMES, "aux"]}
        variants["clean"].append({key: np.asarray(base_obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES})
        for probe_name, corruption_type, strength in [
            ("weak_vision", "vision", cfg.warmup.calibration_only_vision_weak_strength),
            ("mild_vision", "vision", cfg.warmup.calibration_only_vision_mild_strength),
            ("weak_ego", "ego", cfg.warmup.calibration_only_ego_weak_strength),
            ("mild_ego", "ego", cfg.warmup.calibration_only_ego_mild_strength),
        ]:
            local_rng = np.random.default_rng(rng.integers(0, 1_000_000_000) + state_idx)
            corrupted_obs, _severity, _metadata = apply_shared_corruption(
                obs=base_obs,
                corruption_type=corruption_type,
                corruption_strength=float(strength),
                rng=local_rng,
                corruption_params={"corruption_type": corruption_type, "corruption_strength": float(strength)},
                probe_name=probe_name,
            )
            variants[probe_name].append(corrupted_obs)
    return {name: obs_to_tensor_batch(obs_list, device) for name, obs_list in variants.items()}


def add_vision_logit_triplet_loss(outputs: dict[str, dict[str, torch.Tensor]], cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    idx = MODALITY_NAMES.index("vision")
    clean = outputs["clean"]["reliability_logits"][:, idx]
    weak = outputs["weak_vision"]["reliability_logits"][:, idx]
    mild = outputs["mild_vision"]["reliability_logits"][:, idx]
    weak_loss = torch.relu(float(cfg.warmup.weak_margin_vision) - (clean - weak))
    mild_loss = torch.relu(float(cfg.warmup.mild_margin_vision) - (weak - mild))
    return weak_loss.mean(), mild_loss.mean()


def add_ego_logit_triplet_loss(outputs: dict[str, dict[str, torch.Tensor]], cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    idx = MODALITY_NAMES.index("ego")
    clean = outputs["clean"]["reliability_logits"][:, idx]
    weak = outputs["weak_ego"]["reliability_logits"][:, idx]
    mild = outputs["mild_ego"]["reliability_logits"][:, idx]
    weak_loss = torch.relu(float(cfg.warmup.weak_margin_ego) - (clean - weak))
    mild_loss = torch.relu(float(cfg.warmup.mild_margin_ego) - (weak - mild))
    return weak_loss.mean(), mild_loss.mean()


def ensure_vision_ego_pair_batching(base_states: list[dict[str, np.ndarray]], device: torch.device, rng: np.random.Generator, cfg: ExperimentConfig, batch_size: int) -> dict[str, dict[str, torch.Tensor]]:
    return build_paired_calibration_batches(base_states, device, rng, cfg, batch_size)


def fine_tune_reference_calibration(candidate_name: str, candidate_spec: dict[str, Any], config: dict[str, Any]) -> None:
    if candidate_spec.get("use_existing_checkpoint", False):
        return
    output_root = ensure_dir(Path(config["output_root"]) / candidate_name)
    base_checkpoint_root = Path(config["base_checkpoint_root"])
    device = config.get("device", "cpu")
    for seed in config["seeds"]:
        run_dir = ensure_dir(output_root / f"seed_{seed}")
        if (run_dir / "run_status.json").exists():
            continue
        cfg, model = build_model_and_cfg(
            config["base_stage1_config_path"],
            latest_checkpoint(base_checkpoint_root / f"seed_{seed}"),
            seed=seed,
            device=device,
            overrides=config.get("base_config_overrides", {}),
        )
        deep_apply_overrides(cfg, candidate_spec.get("overrides", {}))
        freeze_policy_backbone(model)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["train"]["learning_rate"]),
        )
        base_states = collect_base_states(
            model=model,
            device=torch.device(device),
            seed=seed * 1000 + 7,
            num_states=int(config["train"]["num_base_states"]),
        )
        rng = np.random.default_rng(seed + 12345)
        metrics_rows = []
        for update_idx in range(int(config["train"]["num_updates"])):
            batches = ensure_vision_ego_pair_batching(
                base_states=base_states,
                device=torch.device(device),
                rng=rng,
                cfg=cfg,
                batch_size=int(config["train"]["batch_size"]),
            )
            outputs = {name: model.forward_reliability(obs_dict) for name, obs_dict in batches.items()}
            vision_cw, vision_wm = add_vision_logit_triplet_loss(outputs, cfg)
            ego_cw, ego_wm = add_ego_logit_triplet_loss(outputs, cfg)
            directional_loss = (
                torch.relu(-(outputs["clean"]["reliability_logits"][:, MODALITY_NAMES.index("vision")] - outputs["weak_vision"]["reliability_logits"][:, MODALITY_NAMES.index("vision")])).mean()
                + torch.relu(-(outputs["weak_vision"]["reliability_logits"][:, MODALITY_NAMES.index("vision")] - outputs["mild_vision"]["reliability_logits"][:, MODALITY_NAMES.index("vision")])).mean()
                + torch.relu(-(outputs["clean"]["reliability_logits"][:, MODALITY_NAMES.index("ego")] - outputs["weak_ego"]["reliability_logits"][:, MODALITY_NAMES.index("ego")])).mean()
                + torch.relu(-(outputs["weak_ego"]["reliability_logits"][:, MODALITY_NAMES.index("ego")] - outputs["mild_ego"]["reliability_logits"][:, MODALITY_NAMES.index("ego")])).mean()
            ) / 4.0
            loss = float(cfg.warmup.logit_triplet_loss_coef) * (vision_cw + vision_wm + ego_cw + ego_wm)
            loss = loss + float(cfg.warmup.directional_monotonicity_loss_coef) * directional_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            metrics_rows.append(
                {
                    "update": update_idx + 1,
                    "loss/logit_triplet_total": float(loss.item()),
                    "loss/vision_clean_weak": float(vision_cw.item()),
                    "loss/vision_weak_mild": float(vision_wm.item()),
                    "loss/ego_clean_weak": float(ego_cw.item()),
                    "loss/ego_weak_mild": float(ego_wm.item()),
                    "loss/directional_helper": float(directional_loss.item()),
                    "loss/grad_norm": grad_norm,
                }
            )
        torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint_final.pt")
        (run_dir / "candidate_config_snapshot.json").write_text(json.dumps(asdict(cfg), ensure_ascii=True, indent=2), encoding="utf-8")
        pd.DataFrame(metrics_rows).to_json(run_dir / "calibration_metrics.json", orient="records", indent=2)
        (run_dir / "run_status.json").write_text(json.dumps({"finished_flag": 1, "nan_detected": 0, "crashed_flag": 0}, ensure_ascii=True, indent=2), encoding="utf-8")


@torch.no_grad()
def run_probe_suite(candidate_name: str, candidate_spec: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = ensure_dir(Path(config["output_root"]) / candidate_name)
    checkpoint_root = Path(config["base_checkpoint_root"]) if candidate_spec.get("use_existing_checkpoint", False) else output_root
    rows: list[dict[str, Any]] = []
    probes = dict(config["probes"])
    device = torch.device(config.get("device", "cpu"))
    for seed in config["seeds"]:
        checkpoint = latest_checkpoint(checkpoint_root / f"seed_{seed}") if candidate_spec.get("use_existing_checkpoint", False) else (checkpoint_root / f"seed_{seed}" / "checkpoint_final.pt")
        cfg, model = build_model_and_cfg(
            config["base_stage1_config_path"],
            checkpoint,
            seed=seed,
            device=str(device),
            overrides=config.get("base_config_overrides", {}),
        )
        deep_apply_overrides(cfg, candidate_spec.get("overrides", {}))
        env = RCSToyNavEnv()
        env.set_corruption_mode("clean")
        for probe_name, probe_cfg in probes.items():
            returns = []
            successes = []
            collisions = []
            timeouts = []
            progress_values = []
            attention_values = []
            reliability_values = []
            logit_values = []
            rng = np.random.default_rng(seed + 5000)
            for episode_idx in range(int(config["num_eval_episodes"])):
                obs, info = env.reset(seed=seed * 100 + episode_idx)
                del info
                hidden = model.initial_hidden(batch_size=1, device=device)
                episode_start = torch.ones(1, device=device)
                episode_return = 0.0
                episode_progress = []
                while True:
                    if probe_name == "clean":
                        policy_obs = {key: np.asarray(obs[key], dtype=np.float32).copy() for key in MODALITY_NAMES}
                    else:
                        policy_obs, _severity, _metadata = apply_shared_corruption(
                            obs=obs,
                            corruption_type=str(probe_cfg["corruption_type"]),
                            corruption_strength=float(probe_cfg["corruption_strength"]),
                            rng=rng,
                            corruption_params=probe_cfg,
                            probe_name=probe_name,
                        )
                    output = model.step(
                        obs_dict={key: torch.as_tensor(policy_obs[key], dtype=torch.float32, device=device).unsqueeze(0) for key in MODALITY_NAMES},
                        hidden_state=hidden,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    hidden = output.hidden_state
                    action = output.action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_progress.append(float(info.get("progress_delta", 0.0)))
                    attention_values.append(output.aux_outputs["attention_weights"].squeeze(0).detach().cpu().numpy())
                    reliability_values.append(output.aux_outputs["reliability_scores"].squeeze(0).detach().cpu().numpy())
                    logit_values.append(output.aux_outputs["reliability_logits"].squeeze(0).detach().cpu().numpy())
                    episode_start = torch.tensor([float(terminated or truncated)], device=device)
                    if terminated or truncated:
                        returns.append(episode_return)
                        successes.append(float(info["goal_reached"]))
                        collisions.append(float(info["collision"]))
                        timeouts.append(float(info["timeout"]))
                        progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                        break
            att = np.stack(attention_values, axis=0)
            rel = np.stack(reliability_values, axis=0)
            logits = np.stack(logit_values, axis=0)
            row = {
                "candidate_name": candidate_name,
                "seed": seed,
                "corruption_type": probe_name,
                "corruption_strength": float(probe_cfg.get("corruption_strength", 0.0)),
                "eval_return": float(np.mean(returns)),
                "eval_success_rate": float(np.mean(successes)),
                "eval_collision_rate": float(np.mean(collisions)),
                "eval_timeout_rate": float(np.mean(timeouts)),
                "eval_avg_progress_to_goal": float(np.mean(progress_values)),
                "reliability_nan_detected": float(not np.all(np.isfinite(rel))),
                "route_attention_dominance_ratio": float(att[:, 3].mean() / max(np.delete(att.mean(axis=0), 3).mean(), 1e-6)),
            }
            for idx, name in enumerate(MODALITY_NAMES):
                row[f"attention_mean_{name}"] = float(att[:, idx].mean())
                row[f"attention_std_{name}"] = float(att[:, idx].std())
                row[f"reliability_mean_{name}"] = float(rel[:, idx].mean())
                row[f"reliability_std_{name}"] = float(rel[:, idx].std())
                if name in ("vision", "ego"):
                    row[f"logit_mean_{name}_{probe_name.split('_')[0] if probe_name != 'clean' else 'clean'}"] = float(logits[:, idx].mean())
            rows.append(row)
    return rows


def compute_positive_drop_metrics(df: pd.DataFrame, modality: str) -> dict[str, float]:
    clean = float(df[df["corruption_type"] == "clean"][f"reliability_mean_{modality}"].mean())
    weak = float(df[df["corruption_type"] == f"weak_{modality}"][f"reliability_mean_{modality}"].mean())
    mild = float(df[df["corruption_type"] == f"mild_{modality}"][f"reliability_mean_{modality}"].mean())
    return {
        "clean": clean,
        "weak": weak,
        "mild": mild,
        "clean_minus_weak": clean - weak,
        "clean_minus_mild": clean - mild,
        "weak_minus_mild": weak - mild,
        "positive_flag": float((clean - weak) > 0.0 or (clean - mild) > 0.0),
        "directional_flag": float((clean - mild) > (clean - weak)),
    }


def compute_logit_metrics(df: pd.DataFrame, modality: str) -> dict[str, float]:
    clean = float(df[df["corruption_type"] == "clean"][f"logit_mean_{modality}_clean"].mean())
    weak = float(df[df["corruption_type"] == f"weak_{modality}"][f"logit_mean_{modality}_weak"].mean())
    mild = float(df[df["corruption_type"] == f"mild_{modality}"][f"logit_mean_{modality}_mild"].mean())
    return {
        "clean": clean,
        "weak": weak,
        "mild": mild,
        "clean_minus_weak": clean - weak,
        "clean_minus_mild": clean - mild,
        "weak_minus_mild": weak - mild,
    }


def aggregate_candidate_scores(candidate_name: str, df: pd.DataFrame, reference_clean: dict[str, float]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    vision_rel = compute_positive_drop_metrics(df, "vision")
    ego_rel = compute_positive_drop_metrics(df, "ego")
    vision_logit = compute_logit_metrics(df, "vision")
    ego_logit = compute_logit_metrics(df, "ego")
    clean_success = float(df[df["corruption_type"] == "clean"]["eval_success_rate"].mean())
    clean_return = float(df[df["corruption_type"] == "clean"]["eval_return"].mean())
    clean_progress = float(df[df["corruption_type"] == "clean"]["eval_avg_progress_to_goal"].mean())
    clean_rel_max = float(df[df["corruption_type"] == "clean"][[f"reliability_mean_{m}" for m in MODALITY_NAMES]].mean().max())
    saturation_warning_flag = float(clean_rel_max >= 0.98)
    summary = {
        "candidate_name": candidate_name,
        "vision_positive_ratio": vision_rel["positive_flag"],
        "ego_positive_ratio": ego_rel["positive_flag"],
        "target_positive_ratio": float((vision_rel["positive_flag"] + ego_rel["positive_flag"]) / 2.0),
        "weak_to_mild_directional_ratio": float((vision_rel["directional_flag"] + ego_rel["directional_flag"]) / 2.0),
        "vision_directional_monotonicity_ratio": vision_rel["directional_flag"],
        "ego_directional_monotonicity_ratio": ego_rel["directional_flag"],
        "clean_success_rate": clean_success,
        "clean_return": clean_return,
        "clean_progress": clean_progress,
        "clean_success_drop_vs_reference": max(0.0, reference_clean["eval_success_rate"] - clean_success),
        "clean_return_drop_vs_reference": max(0.0, reference_clean["eval_return"] - clean_return),
        "clean_progress_drop_vs_reference": max(0.0, reference_clean["eval_avg_progress_to_goal"] - clean_progress),
        "saturation_warning_flag": saturation_warning_flag,
        "nan_or_crash": float(df["reliability_nan_detected"].max() > 0),
    }
    summary["pass_fail"] = evaluate_pass_fail(summary)
    summary["calibration_score"] = (
        5.0 * summary["vision_positive_ratio"]
        + 5.0 * summary["ego_positive_ratio"]
        + 3.0 * summary["weak_to_mild_directional_ratio"]
        + 2.0 * summary["vision_directional_monotonicity_ratio"]
        + 2.0 * summary["ego_directional_monotonicity_ratio"]
        - 2.0 * summary["clean_success_drop_vs_reference"]
        - 1.0 * summary["clean_return_drop_vs_reference"]
        - 1.0 * summary["saturation_warning_flag"]
    )
    rel_rows = []
    logit_rows = []
    for modality, rel_metrics, logit_metrics in [("vision", vision_rel, vision_logit), ("ego", ego_rel, ego_logit)]:
        rel_rows.append(
            {
                "candidate_name": candidate_name,
                "modality": modality,
                "clean_mean": rel_metrics["clean"],
                "weak_mean": rel_metrics["weak"],
                "mild_mean": rel_metrics["mild"],
                "clean_weak_delta": rel_metrics["clean_minus_weak"],
                "clean_mild_delta": rel_metrics["clean_minus_mild"],
                "weak_mild_delta": rel_metrics["weak_minus_mild"],
            }
        )
        logit_rows.append(
            {
                "candidate_name": candidate_name,
                "modality": modality,
                "clean_logit": logit_metrics["clean"],
                "weak_logit": logit_metrics["weak"],
                "mild_logit": logit_metrics["mild"],
                "clean_weak_delta": logit_metrics["clean_minus_weak"],
                "clean_mild_delta": logit_metrics["clean_minus_mild"],
                "weak_mild_delta": logit_metrics["weak_minus_mild"],
            }
        )
    return summary, pd.DataFrame(rel_rows), pd.DataFrame(logit_rows)


def evaluate_pass_fail(summary: dict[str, Any]) -> str:
    if summary["nan_or_crash"] > 0:
        return "FAIL"
    if summary["vision_positive_ratio"] == 0.0 and summary["ego_positive_ratio"] == 0.0:
        return "FAIL"
    if summary["weak_to_mild_directional_ratio"] == 0.0:
        return "FAIL"
    if (
        (summary["vision_positive_ratio"] > 0.0 or summary["ego_positive_ratio"] > 0.0)
        and summary["weak_to_mild_directional_ratio"] > 0.0
        and (summary["vision_directional_monotonicity_ratio"] > 0.0 or summary["ego_directional_monotonicity_ratio"] > 0.0)
        and summary["clean_success_drop_vs_reference"] <= 0.10
    ):
        return "PASS"
    return "WARN"


def generate_final_recommendation_report(score_df: pd.DataFrame, output_path: Path, recommended: str, secondary: str) -> None:
    lines = [
        "# Stage 1 Vision/Ego Logit Triplet Calibration Report",
        "",
        f"- Recommended: **{recommended}**",
        f"- Secondary: **{secondary}**",
        "",
        score_df.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "vision_ego_logit_triplet_config_snapshot.yaml", config)

    all_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    reliability_comparison_rows: list[dict[str, Any]] = []
    logit_comparison_rows: list[dict[str, Any]] = []

    reference_name = config["reference_candidate_name"]
    candidates = dict(config["candidates"])
    ordered_candidates = list(config.get("candidate_order", candidates.keys()))
    if reference_name not in ordered_candidates:
        ordered_candidates = [reference_name, *ordered_candidates]

    reference_rows = run_probe_suite(reference_name, candidates[reference_name], config)
    all_rows.extend(reference_rows)
    reference_df = pd.DataFrame(reference_rows)
    reference_clean = reference_df[reference_df["corruption_type"] == "clean"][["eval_success_rate", "eval_return", "eval_avg_progress_to_goal"]].mean().to_dict()

    for candidate_name in ordered_candidates:
        candidate_spec = candidates[candidate_name]
        fine_tune_reference_calibration(candidate_name, candidate_spec, config)
        rows = reference_rows if candidate_name == reference_name else run_probe_suite(candidate_name, candidate_spec, config)
        if candidate_name != reference_name:
            all_rows.extend(rows)
        candidate_df = pd.DataFrame(rows)
        summary, rel_cmp_df, logit_cmp_df = aggregate_candidate_scores(candidate_name, candidate_df, reference_clean)
        summary["round"] = candidate_spec.get("round", "")
        score_rows.append(summary)
        reliability_comparison_rows.extend(rel_cmp_df.to_dict("records"))
        logit_comparison_rows.extend(logit_cmp_df.to_dict("records"))

    all_df = pd.DataFrame(all_rows).sort_values(["candidate_name", "corruption_type", "seed"]).reset_index(drop=True)
    score_df = pd.DataFrame(score_rows).sort_values(
        ["pass_fail", "vision_positive_ratio", "ego_positive_ratio", "weak_to_mild_directional_ratio", "calibration_score"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    order_map = {"PASS": 0, "WARN": 1, "FAIL": 2}
    score_df["_order"] = score_df["pass_fail"].map(order_map).fillna(9)
    score_df = score_df.sort_values(
        ["_order", "vision_positive_ratio", "ego_positive_ratio", "weak_to_mild_directional_ratio", "vision_directional_monotonicity_ratio", "ego_directional_monotonicity_ratio", "calibration_score"],
        ascending=[True, False, False, False, False, False, False],
    ).drop(columns="_order").reset_index(drop=True)

    all_df.to_csv(summary_dir / "vision_ego_logit_triplet_sweep_metrics.csv", index=False)
    all_df.to_json(summary_dir / "vision_ego_logit_triplet_sweep_metrics.json", orient="records", indent=2)
    score_df.to_csv(summary_dir / "vision_ego_logit_triplet_score_summary.csv", index=False)
    pd.DataFrame(reliability_comparison_rows).to_csv(summary_dir / "vision_ego_logit_triplet_attention_reliability_comparison.csv", index=False)
    pd.DataFrame(logit_comparison_rows).to_csv(summary_dir / "vision_ego_logit_triplet_logit_comparison.csv", index=False)
    (summary_dir / "vision_ego_logit_triplet_pass_fail_report.md").write_text(
        "# Vision/Ego Logit Triplet PASS/WARN/FAIL Report\n\n" + score_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    recommended = score_df.iloc[0]["candidate_name"]
    secondary = score_df.iloc[1]["candidate_name"] if len(score_df) > 1 else recommended
    generate_final_recommendation_report(score_df, summary_dir / "vision_ego_logit_triplet_final_recommendation.md", recommended, secondary)
    print("Recommended:", recommended)
    print("Secondary:", secondary)


if __name__ == "__main__":
    main()
