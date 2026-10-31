"""Feature separability debug for vision/ego reliability bottleneck analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from rcs_ac_v1.model import MODALITY_NAMES
from rcs_toy_nav_env import RCSToyNavEnv
from run_very_weak_probe import build_model, ensure_dir, latest_checkpoint, save_yaml
from stage1_warmup.corruption import apply_shared_corruption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 vision/ego feature separability debug.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/stage1_vision_ego_feature_debug.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
        for key in MODALITY_NAMES
    }


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def _variant_output(
    model,
    device: torch.device,
    obs: dict[str, np.ndarray],
    hidden: torch.Tensor,
    episode_start: torch.Tensor,
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        output = model.step(
            obs_dict=obs_to_tensor(obs, device),
            hidden_state=hidden,
            episode_start=episode_start,
            deterministic=True,
        )
    aux = output.aux_outputs
    return {
        "action": output.action.squeeze(0).detach().cpu().numpy(),
        "attention": aux["attention_weights"].squeeze(0).detach().cpu().numpy(),
        "reliability": aux["reliability_scores"].squeeze(0).detach().cpu().numpy(),
        "reliability_logits": aux["reliability_logits"].squeeze(0).detach().cpu().numpy(),
        "encoder_features": aux["encoder_features"].squeeze(0).detach().cpu().numpy(),
        "reliability_inputs": aux["reliability_inputs"].squeeze(0).detach().cpu().numpy(),
        "fused_feature": aux["fused_feature"].squeeze(0).detach().cpu().numpy(),
        "hidden_state": output.hidden_state,
    }


def run_feature_debug(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    output_root = ensure_dir(config["output_root"])
    summary_dir = ensure_dir(output_root / "summary")
    save_yaml(summary_dir / "vision_ego_feature_debug_config_snapshot.yaml", config)
    device = torch.device(config.get("device", "cpu"))

    probes = dict(config["probes"])
    trace_rows: list[dict[str, Any]] = []
    trace_json: list[dict[str, Any]] = []

    step_limit = int(config.get("trace_steps", 4))
    num_episodes = int(config.get("num_eval_episodes", 3))
    feature_diff_eps = float(config.get("feature_diff_epsilon", 1e-6))
    logit_diff_eps = float(config.get("logit_diff_epsilon", 1e-6))
    reliability_diff_eps = float(config.get("reliability_diff_epsilon", 1e-6))

    for seed in config["seeds"]:
        checkpoint = latest_checkpoint(Path(config["checkpoint_root"]) / f"seed_{seed}")
        model = build_model(config["stage1_config_path"], checkpoint, device)
        env = RCSToyNavEnv()
        env.set_corruption_mode("clean")

        for episode_idx in range(num_episodes):
            obs, info = env.reset(seed=seed * 1000 + episode_idx)
            del info
            hidden = model.initial_hidden(batch_size=1, device=device)
            episode_start = torch.ones(1, device=device)

            for step_idx in range(step_limit):
                clean_variant = _variant_output(model, device, obs, hidden, episode_start)
                clean_hidden = clean_variant["hidden_state"]
                clean_action = clean_variant["action"]

                trace_entry: dict[str, Any] = {
                    "seed": seed,
                    "episode_idx": episode_idx,
                    "step_idx": step_idx,
                    "clean": {
                        "vision": obs["vision"].tolist(),
                        "distance": obs["distance"].tolist(),
                        "ego": obs["ego"].tolist(),
                        "route": obs["route"].tolist(),
                        "attention": clean_variant["attention"].tolist(),
                        "reliability": clean_variant["reliability"].tolist(),
                        "reliability_logits": clean_variant["reliability_logits"].tolist(),
                    },
                    "probes": {},
                }

                for probe_name, probe_cfg in probes.items():
                    if probe_name == "clean":
                        continue
                    rng = np.random.default_rng(seed * 1_000_000 + episode_idx * 1_000 + step_idx * 10 + hash(probe_name) % 997)
                    corrupted_obs, severity, metadata = apply_shared_corruption(
                        obs=obs,
                        corruption_type=str(probe_cfg["corruption_type"]),
                        corruption_strength=float(probe_cfg["corruption_strength"]),
                        rng=rng,
                        corruption_params=probe_cfg,
                        probe_name=probe_name,
                    )
                    probe_variant = _variant_output(model, device, corrupted_obs, hidden, episode_start)
                    target_modality = probe_name.split("_", 1)[-1]
                    target_idx = MODALITY_NAMES.index(target_modality)

                    raw_clean = np.asarray(obs[target_modality], dtype=np.float32)
                    raw_probe = np.asarray(corrupted_obs[target_modality], dtype=np.float32)
                    enc_clean = clean_variant["encoder_features"][target_idx]
                    enc_probe = probe_variant["encoder_features"][target_idx]
                    head_clean = clean_variant["reliability_inputs"][target_idx]
                    head_probe = probe_variant["reliability_inputs"][target_idx]
                    logit_clean = float(clean_variant["reliability_logits"][target_idx])
                    logit_probe = float(probe_variant["reliability_logits"][target_idx])
                    rel_clean = float(clean_variant["reliability"][target_idx])
                    rel_probe = float(probe_variant["reliability"][target_idx])

                    row = {
                        "seed": seed,
                        "episode_idx": episode_idx,
                        "step_idx": step_idx,
                        "probe_name": probe_name,
                        "target_modality": target_modality,
                        "corruption_strength": float(probe_cfg["corruption_strength"]),
                        "raw_obs_diff_l2": float(np.linalg.norm(raw_clean - raw_probe)),
                        "raw_obs_diff_max": float(np.max(np.abs(raw_clean - raw_probe))),
                        "encoder_feature_l2": float(np.linalg.norm(enc_clean - enc_probe)),
                        "encoder_feature_cosine_distance": _cosine_distance(enc_clean, enc_probe),
                        "head_input_l2": float(np.linalg.norm(head_clean - head_probe)),
                        "head_input_cosine_distance": _cosine_distance(head_clean, head_probe),
                        "target_logit_delta_clean_minus_probe": float(logit_clean - logit_probe),
                        "target_reliability_delta_clean_minus_probe": float(rel_clean - rel_probe),
                        "feature_diff_positive_flag": int(float(np.linalg.norm(enc_clean - enc_probe)) > feature_diff_eps),
                        "head_input_diff_positive_flag": int(float(np.linalg.norm(head_clean - head_probe)) > feature_diff_eps),
                        "head_logit_diff_positive_flag": int(abs(logit_clean - logit_probe) > logit_diff_eps),
                        "reliability_directional_positive_flag": int((rel_clean - rel_probe) > reliability_diff_eps),
                        "reliability_directional_negative_flag": int((rel_clean - rel_probe) < -reliability_diff_eps),
                    }
                    trace_rows.append(row)
                    trace_entry["probes"][probe_name] = {
                        "target_modality": target_modality,
                        "corruption_strength": float(probe_cfg["corruption_strength"]),
                        "corruption_metadata": metadata,
                        "severity": np.asarray(severity, dtype=np.float32).tolist(),
                        "target_modality_raw_obs": raw_probe.tolist(),
                        "attention": probe_variant["attention"].tolist(),
                        "reliability": probe_variant["reliability"].tolist(),
                        "reliability_logits": probe_variant["reliability_logits"].tolist(),
                        "raw_obs_diff_l2": row["raw_obs_diff_l2"],
                        "encoder_feature_l2": row["encoder_feature_l2"],
                        "head_input_l2": row["head_input_l2"],
                        "target_logit_delta_clean_minus_probe": row["target_logit_delta_clean_minus_probe"],
                        "target_reliability_delta_clean_minus_probe": row["target_reliability_delta_clean_minus_probe"],
                    }

                trace_json.append(trace_entry)
                obs, reward, terminated, truncated, info = env.step(clean_action)
                del reward, info
                hidden = clean_hidden
                episode_start = torch.tensor([float(terminated or truncated)], device=device)
                if terminated or truncated:
                    break

    trace_df = pd.DataFrame(trace_rows)
    if trace_df.empty:
        raise RuntimeError("Feature separability debug produced no rows.")

    summary_df = (
        trace_df.groupby("target_modality", as_index=False)
        .agg(
            vision_ego_probe_count=("probe_name", "count"),
            mean_raw_obs_diff_l2=("raw_obs_diff_l2", "mean"),
            mean_encoder_feature_l2=("encoder_feature_l2", "mean"),
            mean_encoder_feature_cosine_distance=("encoder_feature_cosine_distance", "mean"),
            mean_head_input_l2=("head_input_l2", "mean"),
            mean_head_input_cosine_distance=("head_input_cosine_distance", "mean"),
            mean_target_logit_delta_clean_minus_probe=("target_logit_delta_clean_minus_probe", "mean"),
            mean_target_reliability_delta_clean_minus_probe=("target_reliability_delta_clean_minus_probe", "mean"),
            feature_diff_positive_ratio=("feature_diff_positive_flag", "mean"),
            head_input_diff_positive_ratio=("head_input_diff_positive_flag", "mean"),
            head_logit_diff_positive_ratio=("head_logit_diff_positive_flag", "mean"),
            reliability_directional_positive_ratio=("reliability_directional_positive_flag", "mean"),
            reliability_directional_negative_ratio=("reliability_directional_negative_flag", "mean"),
        )
        .sort_values("target_modality")
        .reset_index(drop=True)
    )

    head_input_df = trace_df[
        [
            "seed",
            "episode_idx",
            "step_idx",
            "probe_name",
            "target_modality",
            "encoder_feature_l2",
            "encoder_feature_cosine_distance",
            "head_input_l2",
            "head_input_cosine_distance",
            "target_logit_delta_clean_minus_probe",
            "target_reliability_delta_clean_minus_probe",
        ]
    ].copy()

    summary_map = {row["target_modality"]: row for row in summary_df.to_dict("records")}
    vision_row = summary_map.get("vision", {})
    ego_row = summary_map.get("ego", {})
    feature_path = "B"
    reason = "feature and head-input diffs exist, so the next bottleneck is likely objective/head calibration."
    if (
        vision_row.get("feature_diff_positive_ratio", 0.0) < 0.8
        or ego_row.get("feature_diff_positive_ratio", 0.0) < 0.8
        or vision_row.get("head_input_diff_positive_ratio", 0.0) < 0.8
        or ego_row.get("head_input_diff_positive_ratio", 0.0) < 0.8
    ):
        feature_path = "A"
        reason = "vision/ego corruption is not reaching encoder/head input reliably enough; head-input path should be fixed first."

    decision = {
        "selected_path": feature_path,
        "reason": reason,
        "vision_feature_diff_positive_ratio": float(vision_row.get("feature_diff_positive_ratio", 0.0)),
        "ego_feature_diff_positive_ratio": float(ego_row.get("feature_diff_positive_ratio", 0.0)),
        "vision_head_input_diff_positive_ratio": float(vision_row.get("head_input_diff_positive_ratio", 0.0)),
        "ego_head_input_diff_positive_ratio": float(ego_row.get("head_input_diff_positive_ratio", 0.0)),
        "vision_head_logit_diff_positive_ratio": float(vision_row.get("head_logit_diff_positive_ratio", 0.0)),
        "ego_head_logit_diff_positive_ratio": float(ego_row.get("head_logit_diff_positive_ratio", 0.0)),
        "vision_reliability_directional_positive_ratio": float(vision_row.get("reliability_directional_positive_ratio", 0.0)),
        "ego_reliability_directional_positive_ratio": float(ego_row.get("reliability_directional_positive_ratio", 0.0)),
    }

    trace_df.to_csv(summary_dir / "vision_ego_feature_separability.csv", index=False)
    head_input_df.to_csv(summary_dir / "vision_ego_head_input_diff.csv", index=False)
    (summary_dir / "vision_ego_feature_trace.json").write_text(
        json.dumps(trace_json, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (summary_dir / "vision_ego_separability_decision.json").write_text(
        json.dumps(decision, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Vision/Ego Feature Separability Report",
        "",
        f"- Selected path: **{feature_path}**",
        f"- Reason: {reason}",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    (summary_dir / "vision_ego_separability_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary_df, head_input_df, trace_json, decision


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_feature_debug(config)


if __name__ == "__main__":
    main()
