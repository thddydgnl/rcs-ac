"""Very weak corruption probe for Stage 1 reliability sensitivity."""

from __future__ import annotations

import argparse
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
from stage1_warmup.config import (
    EnvConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    PPOConfig,
    WarmupConfig,
    load_experiment_config,
)
from stage1_warmup.corruption import apply_shared_corruption
from stage1_warmup.model import Stage1WarmupActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run very weak corruption probe for Stage 1 reliability.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/very_weak_probe.yaml",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


def build_very_weak_corruption_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(config["probes"])


def apply_corruption(obs: dict[str, np.ndarray], probe_cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
    corrupted, _severity, _metadata = apply_shared_corruption(
        obs=obs,
        corruption_type=str(probe_cfg.get("corruption_type", "clean")),
        corruption_strength=float(probe_cfg.get("corruption_strength", 0.0)),
        rng=rng,
        corruption_params=probe_cfg,
        probe_name=str(probe_cfg.get("probe_name", probe_cfg.get("corruption_type", "clean"))),
    )
    return corrupted


def build_model(stage1_config_source: str | dict[str, Any], checkpoint_path: Path, device: torch.device) -> Stage1WarmupActorCritic:
    if isinstance(stage1_config_source, dict):
        cfg = ExperimentConfig(
            seed=stage1_config_source["seed"],
            device=stage1_config_source["device"],
            env=EnvConfig(**stage1_config_source["env"]),
            model=ModelConfig(**stage1_config_source["model"]),
            ppo=PPOConfig(**stage1_config_source["ppo"]),
            warmup=WarmupConfig(**stage1_config_source["warmup"]),
            logging=LoggingConfig(**stage1_config_source["logging"]),
        )
    else:
        cfg = load_experiment_config(stage1_config_source)
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
        reliability_local_residual_enabled_modalities=cfg.model.reliability_local_residual_enabled_modalities,
        reliability_local_residual_adapter_dim=cfg.model.reliability_local_residual_adapter_dim,
        reliability_local_residual_input_mode=cfg.model.reliability_local_residual_input_mode,
        reliability_local_residual_apply_to_provisional=cfg.model.reliability_local_residual_apply_to_provisional,
        action_dim=2,
        action_log_std_init=cfg.model.action_log_std_init,
    ).to(device)
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
    model.eval()
    return model


def latest_checkpoint(seed_dir: Path) -> Path:
    checkpoints = sorted(seed_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
    return checkpoints[-1]


@torch.no_grad()
def run_clean_eval(
    model: Stage1WarmupActorCritic,
    device: torch.device,
    seed: int,
    num_eval_episodes: int,
    run_mode: str = "full",
) -> dict[str, Any]:
    return run_weak_corruption_eval(
        model=model,
        device=device,
        seed=seed,
        probe_name="clean",
        probe_cfg={"corruption_type": "clean", "corruption_strength": 0.0},
        num_eval_episodes=num_eval_episodes,
        run_mode=run_mode,
    )


@torch.no_grad()
def run_weak_corruption_eval(
    model: Stage1WarmupActorCritic,
    device: torch.device,
    seed: int,
    probe_name: str,
    probe_cfg: dict[str, Any],
    num_eval_episodes: int,
    run_mode: str = "full",
) -> dict[str, Any]:
    env = RCSToyNavEnv()
    env.set_corruption_mode("clean")
    returns = []
    successes = []
    collisions = []
    timeouts = []
    progress_values = []
    attention_values = []
    reliability_values = []
    success_attention = []
    failure_attention = []
    success_reliability = []
    failure_reliability = []
    vision_residual_delta_values = []
    provisional_vision_residual_delta_values = []
    ego_residual_delta_values = []
    provisional_ego_residual_delta_values = []
    rng = np.random.default_rng(seed + 1000)

    for episode_idx in range(num_eval_episodes):
        obs, info = env.reset(seed=seed * 100 + episode_idx)
        del info
        hidden = model.initial_hidden(batch_size=1, device=device)
        episode_start = torch.ones(1, device=device)
        episode_return = 0.0
        episode_progress = []
        episode_attention = []
        episode_reliability = []

        while True:
            corrupted_obs = apply_corruption(obs, probe_cfg, rng)
            output = model.step(
                obs_dict=obs_to_tensor(corrupted_obs, device),
                hidden_state=hidden,
                episode_start=episode_start,
                deterministic=True,
            )
            hidden = output.hidden_state
            action = output.action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_progress.append(float(info.get("progress_delta", 0.0)))
            attention = output.aux_outputs["attention_weights"].squeeze(0).cpu().numpy()
            reliability = output.aux_outputs["reliability_scores"].squeeze(0).cpu().numpy()
            local_residual_deltas = output.aux_outputs.get(
                "reliability_local_residual_deltas",
                output.aux_outputs.get("reliability_ego_residual_deltas"),
            )
            if local_residual_deltas is not None:
                squeezed_local = local_residual_deltas.squeeze(0)
                vision_residual_delta_values.append(
                    float(squeezed_local[MODALITY_NAMES.index("vision")].cpu().item())
                )
                ego_residual_delta_values.append(float(squeezed_local[MODALITY_NAMES.index("ego")].cpu().item()))
            provisional_local_residual_deltas = output.aux_outputs.get(
                "provisional_reliability_local_residual_deltas",
                output.aux_outputs.get("provisional_reliability_ego_residual_deltas"),
            )
            if provisional_local_residual_deltas is not None:
                squeezed_provisional = provisional_local_residual_deltas.squeeze(0)
                provisional_vision_residual_delta_values.append(
                    float(squeezed_provisional[MODALITY_NAMES.index("vision")].cpu().item())
                )
                provisional_ego_residual_delta_values.append(
                    float(squeezed_provisional[MODALITY_NAMES.index("ego")].cpu().item())
                )
            attention_values.append(attention)
            reliability_values.append(reliability)
            episode_attention.append(attention)
            episode_reliability.append(reliability)
            episode_start = torch.tensor([float(terminated or truncated)], device=device)

            if terminated or truncated:
                returns.append(episode_return)
                successes.append(float(info["goal_reached"]))
                collisions.append(float(info["collision"]))
                timeouts.append(float(info["timeout"]))
                progress_values.append(float(np.mean(episode_progress)) if episode_progress else 0.0)
                if episode_attention:
                    mean_att = np.mean(np.stack(episode_attention, axis=0), axis=0)
                    mean_rel = np.mean(np.stack(episode_reliability, axis=0), axis=0)
                    if info["goal_reached"]:
                        success_attention.append(mean_att)
                        success_reliability.append(mean_rel)
                    else:
                        failure_attention.append(mean_att)
                        failure_reliability.append(mean_rel)
                break

    metrics = {
        "model_name": "standard_v1_stage1_warmup",
        "config_name": "standard_v1",
        "env_mode": "normal",
        "probe_name": probe_name,
        "requested_probe_name": probe_name,
        "resolved_corruption_type": probe_cfg.get("corruption_type", probe_name),
        "resolved_corruption_strength": float(probe_cfg.get("corruption_strength", 0.0)),
        "corruption_type": probe_name,
        "base_corruption_type": probe_cfg.get("corruption_type", probe_name),
        "corruption_strength": float(probe_cfg.get("corruption_strength", 0.0)),
        "seed": seed,
        "num_eval_episodes": int(num_eval_episodes),
        "run_mode": str(run_mode),
        "eval_return": float(np.mean(returns)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_timeout_rate": float(np.mean(timeouts)),
        "eval_avg_progress_to_goal": float(np.mean(progress_values)),
        "reliability_nan_detected": float(0.0 if not reliability_values else float(not np.all(np.isfinite(np.stack(reliability_values, axis=0))))),
    }

    if attention_values:
        att = np.stack(attention_values, axis=0)
        att_mean = att.mean(axis=0)
        att_std = att.std(axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"attention_mean_{name}"] = float(att_mean[idx])
            metrics[f"attention_std_{name}"] = float(att_std[idx])
        route_idx = MODALITY_NAMES.index("route")
        other_mean = np.mean(np.delete(att_mean, route_idx))
        metrics["route_attention_dominance_ratio"] = float(att_mean[route_idx] / max(other_mean, 1e-6))

    if reliability_values:
        rel = np.stack(reliability_values, axis=0)
        rel_mean = rel.mean(axis=0)
        rel_std = rel.std(axis=0)
        for idx, name in enumerate(MODALITY_NAMES):
            metrics[f"reliability_mean_{name}"] = float(rel_mean[idx])
            metrics[f"reliability_std_{name}"] = float(rel_std[idx])
    metrics["reliability_vision_residual_delta_mean"] = (
        float(np.mean(vision_residual_delta_values)) if vision_residual_delta_values else 0.0
    )
    metrics["reliability_vision_residual_delta_std"] = (
        float(np.std(vision_residual_delta_values)) if vision_residual_delta_values else 0.0
    )
    metrics["reliability_ego_residual_delta_mean"] = (
        float(np.mean(ego_residual_delta_values)) if ego_residual_delta_values else 0.0
    )
    metrics["reliability_ego_residual_delta_std"] = (
        float(np.std(ego_residual_delta_values)) if ego_residual_delta_values else 0.0
    )
    metrics["provisional_reliability_vision_residual_delta_mean"] = (
        float(np.mean(provisional_vision_residual_delta_values)) if provisional_vision_residual_delta_values else 0.0
    )
    metrics["provisional_reliability_vision_residual_delta_std"] = (
        float(np.std(provisional_vision_residual_delta_values)) if provisional_vision_residual_delta_values else 0.0
    )
    metrics["provisional_reliability_ego_residual_delta_mean"] = (
        float(np.mean(provisional_ego_residual_delta_values)) if provisional_ego_residual_delta_values else 0.0
    )
    metrics["provisional_reliability_ego_residual_delta_std"] = (
        float(np.std(provisional_ego_residual_delta_values)) if provisional_ego_residual_delta_values else 0.0
    )

    for prefix, values in (("success_episode_attention_summary", success_attention), ("failure_episode_attention_summary", failure_attention)):
        if values:
            stack = np.mean(np.stack(values, axis=0), axis=0)
            for idx, name in enumerate(MODALITY_NAMES):
                metrics[f"{prefix}_{name}"] = float(stack[idx])
    for prefix, values in (("success_episode_reliability_summary", success_reliability), ("failure_episode_reliability_summary", failure_reliability)):
        if values:
            stack = np.mean(np.stack(values, axis=0), axis=0)
            for idx, name in enumerate(MODALITY_NAMES):
                metrics[f"{prefix}_{name}"] = float(stack[idx])

    return metrics


def aggregate_probe_results(rows: list[dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["corruption_type", "seed"]).reset_index(drop=True)
    df.to_csv(summary_dir / "weak_probe_metrics.csv", index=False)
    df.to_json(summary_dir / "weak_probe_metrics.json", orient="records", indent=2)
    return df


def _weighted_mean(values: pd.Series, weights: pd.Series | None) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.empty:
        return 0.0
    if weights is None:
        return float(numeric_values.mean())
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = numeric_values.notna() & numeric_weights.gt(0.0)
    if not valid.any():
        return float(numeric_values.mean())
    return float(np.average(numeric_values[valid].to_numpy(dtype=np.float64), weights=numeric_weights[valid].to_numpy(dtype=np.float64)))


def compare_clean_vs_corrupted_reliability(df: pd.DataFrame) -> pd.DataFrame:
    mods = list(MODALITY_NAMES)
    key_col = "probe_name" if "probe_name" in df.columns else "corruption_type"
    clean = df[df[key_col] == "clean"]
    rows = []
    clean_weights = clean["num_eval_episodes"] if "num_eval_episodes" in clean.columns else None
    clean_means = {m: _weighted_mean(clean[f"reliability_mean_{m}"], clean_weights) for m in mods}
    clean_stds = {m: _weighted_mean(clean[f"reliability_std_{m}"], clean_weights) for m in mods}
    for corruption_type in [x for x in df[key_col].unique() if x != "clean"]:
        sub = df[df[key_col] == corruption_type]
        sub_weights = sub["num_eval_episodes"] if "num_eval_episodes" in sub.columns else None
        for m in mods:
            weak_mean = _weighted_mean(sub[f"reliability_mean_{m}"], sub_weights)
            clean_mean = clean_means[m]
            delta = clean_mean - weak_mean
            rows.append(
                {
                    "corruption_type": corruption_type,
                    "modality": m,
                    "clean_mean": clean_mean,
                    "weak_mean": weak_mean,
                    "delta_clean_minus_weak": delta,
                    "weak_std": _weighted_mean(sub[f"reliability_std_{m}"], sub_weights),
                    "clean_std": clean_stds[m],
                    "relative_drop_ratio": float(delta / max(clean_mean, 1e-6)),
                }
            )
    return pd.DataFrame(rows).sort_values(["corruption_type", "delta_clean_minus_weak"], ascending=[True, False]).reset_index(drop=True)


def compare_clean_vs_corrupted_attention(df: pd.DataFrame) -> pd.DataFrame:
    mods = list(MODALITY_NAMES)
    key_col = "probe_name" if "probe_name" in df.columns else "corruption_type"
    clean = df[df[key_col] == "clean"]
    rows = []
    clean_weights = clean["num_eval_episodes"] if "num_eval_episodes" in clean.columns else None
    clean_means = {m: _weighted_mean(clean[f"attention_mean_{m}"], clean_weights) for m in mods}
    clean_stds = {m: _weighted_mean(clean[f"attention_std_{m}"], clean_weights) for m in mods}
    for corruption_type in [x for x in df[key_col].unique() if x != "clean"]:
        sub = df[df[key_col] == corruption_type]
        sub_weights = sub["num_eval_episodes"] if "num_eval_episodes" in sub.columns else None
        for m in mods:
            weak_mean = _weighted_mean(sub[f"attention_mean_{m}"], sub_weights)
            clean_mean = clean_means[m]
            rows.append(
                {
                    "corruption_type": corruption_type,
                    "modality": m,
                    "clean_mean": clean_mean,
                    "weak_mean": weak_mean,
                    "delta_clean_minus_weak": float(clean_mean - weak_mean),
                    "weak_std": _weighted_mean(sub[f"attention_std_{m}"], sub_weights),
                    "clean_std": clean_stds[m],
                }
            )
    return pd.DataFrame(rows).sort_values(["corruption_type", "modality"]).reset_index(drop=True)


def sensitivity_ranking(reliability_cmp: pd.DataFrame) -> pd.DataFrame:
    target_map = {
        "weak_vision": "vision",
        "weak_distance": "distance",
        "weak_ego": "ego",
        "weak_route": "route",
    }
    rows = []
    for corruption_type, target in target_map.items():
        sub = reliability_cmp[reliability_cmp["corruption_type"] == target]
        if sub.empty:
            sub = reliability_cmp[reliability_cmp["corruption_type"] == corruption_type]
        if sub.empty:
            continue
        target_row = sub[sub["modality"] == target].iloc[0]
        rows.append(
            {
                "corruption_type": corruption_type,
                "target_modality": target,
                "target_delta_clean_minus_weak": float(target_row["delta_clean_minus_weak"]),
                "target_relative_drop_ratio": float(target_row["relative_drop_ratio"]),
                "rank_among_modalities": int(sub["delta_clean_minus_weak"].rank(ascending=False, method="min")[sub["modality"] == target].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_delta_clean_minus_weak"], ascending=False).reset_index(drop=True)


def evaluate_probe_pass_fail(df: pd.DataFrame, reliability_cmp: pd.DataFrame, ranking_df: pd.DataFrame) -> tuple[str, str]:
    if df["reliability_nan_detected"].max() > 0:
        return "FAIL", "reliability produced NaN during probe evaluation"

    clean = df[df["corruption_type"] == "clean"]
    target_map = {"weak_vision": "vision", "weak_distance": "distance", "weak_ego": "ego", "weak_route": "route"}
    responsive = 0
    specific = 0
    severe_perf_drop = 0
    dead_head = 0

    for corruption_type, target in target_map.items():
        sub = df[df["corruption_type"] == corruption_type]
        if sub.empty:
            continue
        rel_row = reliability_cmp[(reliability_cmp["corruption_type"] == corruption_type) & (reliability_cmp["modality"] == target)]
        if rel_row.empty:
            continue
        delta = float(rel_row.iloc[0]["delta_clean_minus_weak"])
        if delta > 0.005:
            responsive += 1
        sub_rel = reliability_cmp[reliability_cmp["corruption_type"] == corruption_type]
        if not sub_rel.empty:
            target_rank = int(sub_rel["delta_clean_minus_weak"].rank(ascending=False, method="min")[sub_rel["modality"] == target].iloc[0])
            if target_rank <= 2:
                specific += 1
        success_drop = float(clean["eval_success_rate"].mean() - sub["eval_success_rate"].mean())
        if success_drop > 0.20:
            severe_perf_drop += 1
        att_delta = abs(float((clean[f"attention_mean_{target}"].mean() - sub[f"attention_mean_{target}"].mean())))
        if delta < 0.003 and att_delta > 0.01:
            dead_head += 1

    if responsive >= 3 and specific >= 3 and severe_perf_drop == 0 and dead_head == 0:
        return "PASS", "targeted weak corruption causes small but directional reliability drops without major performance collapse"
    if responsive >= 2 and severe_perf_drop <= 1:
        return "WARN", "reliability reacts somewhat, but sensitivity or modality specificity is still modest"
    return "FAIL", "reliability shows little directional response or behaves like a dead head under weak corruption"


def generate_weak_probe_report(
    perf_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    verdict: str,
    reason: str,
    output_path: Path,
) -> None:
    lines = [
        "# Very Weak Corruption Probe Report",
        "",
        f"- Verdict: **{verdict}**",
        f"- Reason: {reason}",
        "",
        "## Performance Summary",
        "",
        perf_df.to_markdown(index=False),
        "",
        "## Reliability Sensitivity",
        "",
        reliability_df.to_markdown(index=False),
        "",
        "## Attention Change",
        "",
        attention_df.to_markdown(index=False),
        "",
        "## Sensitivity Ranking",
        "",
        ranking_df.to_markdown(index=False),
        "",
        f"- Stage 2 readiness: **{'Stage 2 진행 가능' if verdict == 'PASS' else ('약한 quality variation을 더 주며 재확인 필요' if verdict == 'WARN' else 'reliability head 반응성 재점검 필요')}**",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_probe_summary(perf_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(perf_df["corruption_type"], perf_df["eval_success_rate"], color="#4472C4")
    axes[0].set_title("Eval Success Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(perf_df["corruption_type"], perf_df["eval_return"], color="#70AD47")
    axes[1].set_title("Eval Return")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "weak_probe_config_snapshot.yaml", config)

    device = torch.device(config.get("device", "cpu"))
    probes = build_very_weak_corruption_configs(config)
    rows = []
    seeds = list(config["seeds"])

    for probe_name, probe_cfg in probes.items():
        probe_dir = ensure_dir(output_root / probe_name)
        for seed in seeds:
            seed_dir = ensure_dir(probe_dir / f"seed_{seed}")
            checkpoint = latest_checkpoint(Path(config["stage1_checkpoint_root"]) / f"seed_{seed}")
            model = build_model(config["stage1_config_path"], checkpoint, device)
            if probe_name == "clean":
                row = run_clean_eval(model, device, seed, int(config["num_eval_episodes"]))
            else:
                row = run_weak_corruption_eval(
                    model=model,
                    device=device,
                    seed=seed,
                    probe_name=probe_name,
                    probe_cfg=probe_cfg,
                    num_eval_episodes=int(config["num_eval_episodes"]),
                )
            row["probe_name"] = probe_name
            (seed_dir / "eval_summary.json").write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")
            rows.append(row)

    df = aggregate_probe_results(rows, summary_dir)
    perf_df = (
        df.groupby("corruption_type", as_index=False)[
            ["eval_return", "eval_success_rate", "eval_collision_rate", "eval_timeout_rate", "eval_avg_progress_to_goal"]
        ]
        .mean()
        .sort_values("corruption_type")
        .reset_index(drop=True)
    )

    reliability_df = compare_clean_vs_corrupted_reliability(df)
    attention_df = compare_clean_vs_corrupted_attention(df)
    ranking_df = sensitivity_ranking(reliability_df)

    clean = df[df["corruption_type"] == "clean"]
    for mod in MODALITY_NAMES:
        clean_mean_rel = float(clean[f"reliability_mean_{mod}"].mean())
        clean_mean_att = float(clean[f"attention_mean_{mod}"].mean())
        reliability_df[f"clean_minus_corrupted_reliability_{mod}"] = np.where(
            reliability_df["modality"] == mod,
            reliability_df["delta_clean_minus_weak"],
            np.nan,
        )
        attention_df[f"clean_minus_corrupted_attention_{mod}"] = np.where(
            attention_df["modality"] == mod,
            attention_df["delta_clean_minus_weak"],
            np.nan,
        )

    verdict, reason = evaluate_probe_pass_fail(df, reliability_df, ranking_df)

    perf_df.to_csv(summary_dir / "clean_vs_weak_performance_summary.csv", index=False)
    reliability_df.to_csv(summary_dir / "clean_vs_weak_reliability_comparison.csv", index=False)
    attention_df.to_csv(summary_dir / "clean_vs_weak_attention_comparison.csv", index=False)
    ranking_df.to_csv(summary_dir / "corruption_sensitivity_ranking.csv", index=False)
    df.to_csv(summary_dir / "weak_probe_attention_reliability_summary.csv", index=False)
    plot_probe_summary(perf_df, summary_dir / "weak_probe_learning_curve_or_eval_summary.png")
    generate_weak_probe_report(
        perf_df=perf_df,
        reliability_df=reliability_df,
        attention_df=attention_df,
        ranking_df=ranking_df,
        verdict=verdict,
        reason=reason,
        output_path=summary_dir / "weak_probe_pass_fail_report.md",
    )
    print(f"Probe verdict: {verdict}")
    print(reason)
    print("Stage 2로 바로 넘어가도 되는가:", "예" if verdict == "PASS" else "아직 재확인 필요")


if __name__ == "__main__":
    main()
