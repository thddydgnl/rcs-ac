"""Debug corruption propagation in Stage 1 weak/mild evaluation probes.

This script verifies that probe configs are propagated, corruption is actually
applied to observations, and train/eval corruption paths are consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from rcs_ac_v1.model import MODALITY_NAMES
from rcs_toy_nav_env import RCSToyNavEnv
from run_very_weak_probe import apply_corruption, build_model, load_yaml
from stage1_warmup.corruption import apply_targeted_corruption


DEBUG_STEPS = 4
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug corruption propagation in eval probes.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/corruption_debug.yaml",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def env_mode_to_corruption_mode(env_mode: str) -> str:
    if env_mode == "normal":
        return "clean"
    if env_mode in ("mild", "adverse"):
        return "corrupted"
    raise ValueError(f"Unsupported env_mode: {env_mode}")


def latest_checkpoint(seed_dir: Path) -> Path:
    checkpoints = sorted(seed_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
    return checkpoints[-1]


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def trace_probe_config_to_env(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    env_mode = config.get("env_mode", "normal")
    requested_corruption_mode = env_mode_to_corruption_mode(env_mode)
    for probe_name, probe_cfg in config["probes"].items():
        env = RCSToyNavEnv()
        env.set_corruption_mode(requested_corruption_mode)
        rows.append(
            {
                "requested_probe_name": probe_name,
                "resolved_corruption_type": probe_cfg.get("corruption_type", probe_name),
                "resolved_corruption_strength": float(probe_cfg.get("corruption_strength", 0.0)),
                "env_mode": env_mode,
                "env_corruption_mode": requested_corruption_mode,
                "wrapper_attached_flag": int(probe_name != "clean"),
                "corruption_applied_flag": int(probe_name != "clean"),
                "env_instance_id": id(env),
                "corruption_metadata": json.dumps(
                    {
                        "probe_name": probe_name,
                        "probe_cfg": probe_cfg,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _fixed_action_sequence(num_steps: int) -> list[np.ndarray]:
    # Keep world trajectory identical across probes.
    return [np.zeros(2, dtype=np.float32) for _ in range(num_steps)]


def dump_raw_observations_for_probe(
    probe_name: str,
    probe_cfg: dict[str, Any],
    seed: int,
    num_steps: int,
) -> dict[str, Any]:
    env = RCSToyNavEnv()
    env.set_corruption_mode("clean")
    obs, info = env.reset(seed=seed)
    del info
    rng = np.random.default_rng(seed + 1000)
    action_sequence = _fixed_action_sequence(num_steps)
    steps = []

    for step_idx in range(num_steps):
        corrupted = apply_corruption(obs, probe_cfg, rng)
        steps.append(
            {
                "step": step_idx,
                "clean_obs": {k: np.asarray(obs[k], dtype=np.float32).copy() for k in MODALITY_NAMES},
                "probe_obs": {k: np.asarray(corrupted[k], dtype=np.float32).copy() for k in MODALITY_NAMES},
                "corruption_metadata": {
                    "requested_probe_name": probe_name,
                    "resolved_corruption_type": probe_cfg.get("corruption_type", probe_name),
                    "resolved_corruption_strength": float(probe_cfg.get("corruption_strength", 0.0)),
                },
            }
        )
        obs, _reward, terminated, truncated, _info = env.step(action_sequence[step_idx])
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + step_idx + 1)
    return {
        "probe_name": probe_name,
        "seed": seed,
        "num_steps": num_steps,
        "steps": sanitize_for_json(steps),
    }


def compare_clean_vs_probe_observations(
    clean_dump: dict[str, Any],
    probe_dump: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for clean_step, probe_step in zip(clean_dump["steps"], probe_dump["steps"], strict=True):
        for modality in MODALITY_NAMES:
            clean_obs = np.asarray(clean_step["probe_obs"][modality], dtype=np.float32)
            probe_obs = np.asarray(probe_step["probe_obs"][modality], dtype=np.float32)
            abs_diff = np.abs(clean_obs - probe_obs)
            rows.append(
                {
                    "requested_probe_name": probe_dump["probe_name"],
                    "step": int(probe_step["step"]),
                    "modality": modality,
                    "per_modality_obs_diff_l2": float(np.linalg.norm(probe_obs - clean_obs)),
                    "per_modality_obs_diff_max": float(abs_diff.max()) if abs_diff.size else 0.0,
                    "per_modality_changed_count": int((abs_diff > 1e-9).sum()),
                    "exact_equal_flag": int(np.array_equal(clean_obs, probe_obs)),
                    "near_equal_flag": int(float(np.linalg.norm(probe_obs - clean_obs)) < 1e-8),
                }
            )
    return pd.DataFrame(rows)


def validate_corruption_metadata_consistency(
    config_trace_df: pd.DataFrame,
    obs_diff_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    target_modality_map = {
        "weak_vision": "vision",
        "weak_distance": "distance",
        "weak_ego": "ego",
        "weak_route": "route",
        "mild_vision": "vision",
        "mild_distance": "distance",
        "mild_route": "route",
    }
    for probe_name, probe_rows in obs_diff_df.groupby("requested_probe_name"):
        if probe_name == "clean":
            continue
        trace_row = config_trace_df[config_trace_df["requested_probe_name"] == probe_name].iloc[0]
        target_modality = target_modality_map.get(probe_name, trace_row["resolved_corruption_type"])
        target_rows = probe_rows[probe_rows["modality"] == target_modality]
        other_rows = probe_rows[probe_rows["modality"] != target_modality]
        target_changed = int((target_rows["per_modality_changed_count"] > 0).any())
        any_changed = int((probe_rows["per_modality_changed_count"] > 0).any())
        max_target_l2 = float(target_rows["per_modality_obs_diff_l2"].max()) if not target_rows.empty else 0.0
        max_other_l2 = float(other_rows["per_modality_obs_diff_l2"].max()) if not other_rows.empty else 0.0
        probe_not_applied = int(trace_row["corruption_applied_flag"] == 1 and any_changed == 0)
        wrong_modality = int(trace_row["corruption_applied_flag"] == 1 and target_changed == 0 and any_changed == 1)
        rows.append(
            {
                "requested_probe_name": probe_name,
                "resolved_corruption_type": trace_row["resolved_corruption_type"],
                "resolved_corruption_strength": trace_row["resolved_corruption_strength"],
                "corruption_metadata_on_flag": int(trace_row["corruption_applied_flag"]),
                "actual_obs_changed_flag": any_changed,
                "target_modality": target_modality,
                "target_modality_changed_flag": target_changed,
                "target_max_l2": max_target_l2,
                "other_max_l2": max_other_l2,
                "probe_not_applied_flag": probe_not_applied,
                "wrong_modality_changed_flag": wrong_modality,
            }
        )
    return pd.DataFrame(rows)


def detect_probe_not_applied(consistency_df: pd.DataFrame) -> pd.DataFrame:
    flagged = consistency_df.copy()
    flagged["probe_not_applied_reason"] = np.where(
        flagged["probe_not_applied_flag"] == 1,
        "corruption metadata on but actual observation diff is zero",
        np.where(
            flagged["wrong_modality_changed_flag"] == 1,
            "non-target modality changed while target modality stayed unchanged",
            "",
        ),
    )
    return flagged


def compare_train_vs_eval_corruption_path(config: dict[str, Any], seed: int) -> pd.DataFrame:
    env = RCSToyNavEnv()
    env.set_corruption_mode("clean")
    obs, _ = env.reset(seed=seed)
    rows = []
    for probe_name, probe_cfg in config["probes"].items():
        if probe_name == "clean":
            continue
        rng_eval = np.random.default_rng(seed + 1000)
        rng_train = np.random.default_rng(seed + 1000)
        eval_obs = apply_corruption(obs, probe_cfg, rng_eval)
        train_obs, _severity = apply_targeted_corruption(
            obs=obs,
            corruption_type=probe_cfg["corruption_type"],
            corruption_strength=float(probe_cfg.get("corruption_strength", 0.0)),
            rng=rng_train,
        )
        for modality in MODALITY_NAMES:
            diff = np.abs(np.asarray(eval_obs[modality]) - np.asarray(train_obs[modality]))
            rows.append(
                {
                    "requested_probe_name": probe_name,
                    "modality": modality,
                    "train_eval_path_diff_l2": float(np.linalg.norm(diff)),
                    "train_eval_path_diff_max": float(diff.max()) if diff.size else 0.0,
                    "train_eval_changed_count": int((diff > 1e-9).sum()),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def _run_first_episode_metric(
    model,
    probe_cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    env = RCSToyNavEnv()
    env.set_corruption_mode("clean")
    obs, _info = env.reset(seed=seed)
    hidden = model.initial_hidden(batch_size=1, device=device)
    episode_start = torch.ones(1, device=device)
    rng = np.random.default_rng(seed + 1000)
    episode_return = 0.0
    progress_values = []
    attention_values = []
    reliability_values = []

    while True:
        corrupted = apply_corruption(obs, probe_cfg, rng)
        output = model.step(
            obs_dict=obs_to_tensor(corrupted, device),
            hidden_state=hidden,
            episode_start=episode_start,
            deterministic=True,
        )
        hidden = output.hidden_state
        action = output.action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        progress_values.append(float(info.get("progress_delta", 0.0)))
        attention_values.append(output.aux_outputs["attention_weights"].squeeze(0).cpu().numpy())
        reliability_values.append(output.aux_outputs["reliability_scores"].squeeze(0).cpu().numpy())
        episode_start = torch.tensor([float(terminated or truncated)], device=device)
        if terminated or truncated:
            break

    attention_stack = np.stack(attention_values, axis=0)
    reliability_stack = np.stack(reliability_values, axis=0)
    result = {
        "eval_return": float(episode_return),
        "eval_success_rate": float(info["goal_reached"]),
        "eval_collision_rate": float(info["collision"]),
        "eval_timeout_rate": float(info["timeout"]),
        "eval_avg_progress_to_goal": float(np.mean(progress_values)) if progress_values else 0.0,
    }
    for idx, name in enumerate(MODALITY_NAMES):
        result[f"attention_mean_{name}"] = float(attention_stack[:, idx].mean())
        result[f"reliability_mean_{name}"] = float(reliability_stack[:, idx].mean())
    return result


def first_episode_metric_sanity(
    config: dict[str, Any],
    config_trace_df: pd.DataFrame,
) -> pd.DataFrame:
    stage1_checkpoint_root = Path(config["stage1_checkpoint_root"])
    stage1_config_path = config["stage1_config_path"]
    device = torch.device(config.get("device", "cpu"))
    seed = int(config["seeds"][0])
    checkpoint = latest_checkpoint(stage1_checkpoint_root / f"seed_{seed}")
    model = build_model(stage1_config_path, checkpoint, device)

    clean_metrics = _run_first_episode_metric(model, {"corruption_type": "clean", "corruption_strength": 0.0}, seed, device)
    rows = []
    for probe_name, probe_cfg in config["probes"].items():
        metrics = clean_metrics if probe_name == "clean" else _run_first_episode_metric(model, probe_cfg, seed, device)
        identical_metrics_warning = 0
        if probe_name != "clean":
            deltas = []
            for key in ["eval_return", "eval_success_rate", "eval_collision_rate", "eval_timeout_rate", "eval_avg_progress_to_goal"]:
                deltas.append(abs(float(clean_metrics[key]) - float(metrics[key])))
            for name in MODALITY_NAMES:
                deltas.append(abs(float(clean_metrics[f"attention_mean_{name}"]) - float(metrics[f"attention_mean_{name}"])))
                deltas.append(abs(float(clean_metrics[f"reliability_mean_{name}"]) - float(metrics[f"reliability_mean_{name}"])))
            identical_metrics_warning = int(max(deltas) < 1e-6)
        rows.append(
            {
                "requested_probe_name": probe_name,
                "resolved_corruption_type": config_trace_df.loc[
                    config_trace_df["requested_probe_name"] == probe_name, "resolved_corruption_type"
                ].iloc[0],
                "resolved_corruption_strength": float(config_trace_df.loc[
                    config_trace_df["requested_probe_name"] == probe_name, "resolved_corruption_strength"
                ].iloc[0]),
                "env_mode": config.get("env_mode", "normal"),
                "identical_metrics_warning": identical_metrics_warning,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def root_cause_suspicion_ranking(
    consistency_df: pd.DataFrame,
    train_eval_df: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    probe_not_applied_count = int(consistency_df["probe_not_applied_flag"].sum()) if not consistency_df.empty else 0
    wrong_modality_count = int(consistency_df["wrong_modality_changed_flag"].sum()) if not consistency_df.empty else 0
    identical_metrics_count = int(metric_df["identical_metrics_warning"].sum()) if "identical_metrics_warning" in metric_df else 0
    train_eval_mismatch = float(train_eval_df["train_eval_path_diff_l2"].mean()) if not train_eval_df.empty else 0.0

    candidates = [
        ("A/C/E eval config/env propagation bug", probe_not_applied_count * 10.0 + identical_metrics_count * 3.0),
        ("D metadata only, observation unchanged", probe_not_applied_count * 10.0),
        ("F target modality bypass or wrong modality changed", wrong_modality_count * 10.0),
        ("B train/eval corruption path mismatch", train_eval_mismatch * 100.0),
        ("Probe too weak / metrics too coarse", max(0.0, 5.0 - probe_not_applied_count * 5.0 - wrong_modality_count * 5.0) + identical_metrics_count),
    ]
    for rank, (name, score) in enumerate(sorted(candidates, key=lambda x: x[1], reverse=True), start=1):
        rows.append(
            {
                "rank": rank,
                "root_cause_candidate": name,
                "suspicion_score": float(score),
            }
        )
    return pd.DataFrame(rows)


def evaluate_corruption_debug_pass_fail(
    consistency_df: pd.DataFrame,
    train_eval_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    train_eval_l2_tolerance: float = 1e-6,
) -> tuple[str, str, int]:
    probe_not_applied = int(consistency_df["probe_not_applied_flag"].sum()) if not consistency_df.empty else 0
    wrong_modality = int(consistency_df["wrong_modality_changed_flag"].sum()) if not consistency_df.empty else 0
    mean_train_eval_l2 = float(train_eval_df["train_eval_path_diff_l2"].mean()) if not train_eval_df.empty else 0.0
    identical_metrics = int(metric_df["identical_metrics_warning"].sum()) if "identical_metrics_warning" in metric_df else 0

    if probe_not_applied > 0 or wrong_modality > 0:
        return "FAIL", "Corruption metadata and raw observation changes are inconsistent.", 0
    if mean_train_eval_l2 > train_eval_l2_tolerance:
        return "WARN", "Eval corruption is applied, but train/eval corruption paths are inconsistent. Unify corruption helper before more calibration sweeps.", 0
    if identical_metrics > 0:
        return "WARN", "Observation corruption is applied, but first-episode metrics remain nearly identical. Probe may be too weak or metrics too coarse.", 1
    return "PASS", "Requested probe parameters match actual targeted observation changes and train/eval paths are consistent.", 1


def plot_obs_diff_summary(obs_diff_df: pd.DataFrame, output_path: Path) -> None:
    agg = (
        obs_diff_df.groupby(["requested_probe_name", "modality"], as_index=False)["per_modality_obs_diff_l2"]
        .mean()
        .sort_values(["requested_probe_name", "modality"])
    )
    probes = [p for p in agg["requested_probe_name"].unique() if p != "clean"]
    fig, axes = plt.subplots(len(probes), 1, figsize=(10, 2.5 * max(len(probes), 1)), squeeze=False)
    for ax, probe_name in zip(axes[:, 0], probes, strict=True):
        sub = agg[agg["requested_probe_name"] == probe_name]
        ax.bar(sub["modality"], sub["per_modality_obs_diff_l2"], color="#4472C4")
        ax.set_title(probe_name)
        ax.set_ylabel("mean L2 diff")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def generate_corruption_debug_report(
    config_trace_df: pd.DataFrame,
    obs_diff_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    train_eval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Corruption Debug Root Cause Report",
        "",
        "## Table 1. Requested probe vs actual env corruption parameters",
        "",
        config_trace_df.to_markdown(index=False),
        "",
        "## Table 2. Clean vs probe raw observation diff summary",
        "",
        (
            obs_diff_df.groupby(["requested_probe_name", "modality"], as_index=False)[
                ["per_modality_obs_diff_l2", "per_modality_obs_diff_max", "per_modality_changed_count"]
            ]
            .mean()
            .to_markdown(index=False)
        ),
        "",
        "## Table 3. Corruption metadata vs actual observation changed 여부",
        "",
        consistency_df.to_markdown(index=False),
        "",
        "## Table 4. Clean vs probe first-episode metrics sanity check",
        "",
        metric_df.to_markdown(index=False),
        "",
        "## Table 5. Root-cause suspicion ranking",
        "",
        ranking_df.to_markdown(index=False),
        "",
        "## Train vs Eval Corruption Path",
        "",
        (
            train_eval_df.groupby(["requested_probe_name", "modality"], as_index=False)[
                ["train_eval_path_diff_l2", "train_eval_path_diff_max", "train_eval_changed_count"]
            ]
            .mean()
            .to_markdown(index=False)
        ),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "corruption_debug_config_snapshot.yaml", config)

    config_trace_df = trace_probe_config_to_env(config)

    raw_samples = {}
    clean_probe_name = "clean"
    clean_dump = dump_raw_observations_for_probe(
        probe_name=clean_probe_name,
        probe_cfg=config["probes"][clean_probe_name],
        seed=int(config["debug_seed"]),
        num_steps=int(config.get("debug_steps", DEBUG_STEPS)),
    )
    raw_samples[clean_probe_name] = clean_dump

    obs_diff_frames = []
    for probe_name, probe_cfg in config["probes"].items():
        probe_dir = ensure_dir(output_root / probe_name)
        probe_dump = dump_raw_observations_for_probe(
            probe_name=probe_name,
            probe_cfg=probe_cfg,
            seed=int(config["debug_seed"]),
            num_steps=int(config.get("debug_steps", DEBUG_STEPS)),
        )
        raw_samples[probe_name] = probe_dump
        if probe_name != "clean":
            diff_df = compare_clean_vs_probe_observations(clean_dump, probe_dump)
            diff_df.to_csv(probe_dir / "raw_obs_diff.csv", index=False)
            obs_diff_frames.append(diff_df)

    obs_diff_df = pd.concat(obs_diff_frames, ignore_index=True) if obs_diff_frames else pd.DataFrame()
    consistency_df = validate_corruption_metadata_consistency(config_trace_df, obs_diff_df)
    consistency_df = detect_probe_not_applied(consistency_df)
    metric_df = first_episode_metric_sanity(config, config_trace_df)
    train_eval_df = compare_train_vs_eval_corruption_path(config, int(config["debug_seed"]))
    ranking_df = root_cause_suspicion_ranking(consistency_df, train_eval_df, metric_df)

    config_trace_df.to_csv(summary_dir / "corruption_debug_config_trace.csv", index=False)
    obs_diff_df.to_csv(summary_dir / "corruption_debug_obs_diff_summary.csv", index=False)
    metric_df.to_csv(summary_dir / "corruption_debug_metric_sanity.csv", index=False)
    train_eval_df.to_csv(summary_dir / "corruption_debug_train_eval_path.csv", index=False)
    (summary_dir / "corruption_debug_raw_obs_samples.json").write_text(
        json.dumps(sanitize_for_json(raw_samples), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    generate_corruption_debug_report(
        config_trace_df=config_trace_df,
        obs_diff_df=obs_diff_df,
        consistency_df=consistency_df,
        metric_df=metric_df,
        train_eval_df=train_eval_df,
        ranking_df=ranking_df,
        output_path=summary_dir / "corruption_debug_root_cause_report.md",
    )
    plot_obs_diff_summary(obs_diff_df, summary_dir / "corruption_debug_obs_diff_overview.png")

    verdict, recommendation, go_no_go_for_calibration = evaluate_corruption_debug_pass_fail(
        consistency_df=consistency_df,
        train_eval_df=train_eval_df,
        metric_df=metric_df,
        train_eval_l2_tolerance=float(config.get("train_eval_l2_tolerance", 1e-6)),
    )
    config_trace_df["go_no_go_for_calibration"] = go_no_go_for_calibration
    consistency_df["go_no_go_for_calibration"] = go_no_go_for_calibration
    metric_df["go_no_go_for_calibration"] = go_no_go_for_calibration
    ranking_df["go_no_go_for_calibration"] = go_no_go_for_calibration
    config_trace_df.to_csv(summary_dir / "corruption_debug_config_trace.csv", index=False)
    obs_diff_df.to_csv(summary_dir / "corruption_debug_obs_diff_summary.csv", index=False)
    metric_df.to_csv(summary_dir / "corruption_debug_metric_sanity.csv", index=False)
    train_eval_df.to_csv(summary_dir / "corruption_debug_train_eval_path.csv", index=False)

    final_lines = [
        "# Corruption Debug Final Recommendation",
        "",
        f"- Verdict: **{verdict}**",
        f"- Primary recommendation: {recommendation}",
        f"- go_no_go_for_calibration: **{go_no_go_for_calibration}**",
        "",
        "## Next Actions",
        "",
    ]
    if verdict == "FAIL":
        final_lines.extend(
            [
                "1. Fix corruption path bug before any calibration or loss tuning.",
                "2. Re-run this corruption debug script on clean/weak/mild probes.",
                "3. Only after PASS, rerun Stage 1 calibration sweep.",
            ]
        )
    elif verdict == "WARN":
        final_lines.extend(
            [
                "1. Unify train/eval corruption helper or strengthen probe slightly.",
                "2. Re-run this corruption debug script and verify first-episode metric deltas are non-trivial.",
                "3. Then rerun the calibration sweep; do not keep tuning losses first.",
            ]
        )
    else:
        final_lines.extend(
            [
                "1. Corruption path is not the primary blocker.",
                "2. Move back to calibration supervision and loss balancing.",
                "3. Keep this script as a regression check before future sweeps.",
            ]
        )
    (summary_dir / "corruption_debug_final_recommendation.md").write_text(
        "\n".join(final_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Corruption debug verdict: {verdict}")
    print(recommendation)


if __name__ == "__main__":
    main()
