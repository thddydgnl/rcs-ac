from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from rcs_ac_first.rl.ppo_agent import PPOAgent, PPOConfig
from rcs_ac_first.rl.shapley import ShapleyEstimatorConfig, estimate_exact_shapley_current_critic
from rcs_ac_first.runtime.auxiliary import build_auxiliary_observation_spec, flatten_auxiliary_observation_dict
from rcs_ac_first.runtime.config import load_phase1_config
from rcs_ac_first.runtime.observation import build_policy_observation_spec, flatten_observation_dict
from rcs_ac_first.scripts.train import _build_env, _resolve_config_path, _set_seed


FIELDS = (
    "step",
    "route_progress",
    "risk_scalar",
    "corruption_level",
    "fusion_rgb",
    "fusion_lidar",
    "fusion_ego",
    "fusion_route",
    "contribution_rgb",
    "contribution_lidar",
    "contribution_ego",
    "contribution_route",
    "reliability_rgb",
    "reliability_lidar",
    "reliability_ego",
    "reliability_route",
    "value_full",
    "value_empty",
    "done_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-ish Shapley evaluation for RCS-AC-first.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default="phase3_shapley_eval")
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def main() -> int:
    args = parse_args()
    config_path = _resolve_config_path(str(args.config))
    config = load_phase1_config(config_path)
    device = config.train.resolve_device(args.device)
    _set_seed(config.train.seed)
    run_name = f"{args.run_name}_{_timestamp()}"
    output_dir = PROJECT_ROOT / "outputs" / "phase3_shapley_eval" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_spec = build_policy_observation_spec(
        rgb_shape=config.model.rgb_shape,
        lidar_shape=config.model.lidar_shape,
        ego_dim=config.model.ego_dim,
        route_dim=config.model.route_dim,
    )
    aux_spec = build_auxiliary_observation_spec(
        quality_stats_dim=config.model.quality_stats_dim,
        risk_stats_dim=config.model.risk_stats_dim,
        route_commitment_stats_dim=config.model.route_commitment_stats_dim,
        sensor_dropout_dim=config.model.sensor_dropout_dim,
    )
    agent = PPOAgent(PPOConfig.from_phase1_config(config, device_override=device))
    agent.load(str(Path(args.checkpoint)), map_location=agent.device)
    env = _build_env(
        config,
        host=args.host or config.train.host,
        port=int(args.port if args.port is not None else config.train.port),
        output_dir=output_dir,
        run_name=run_name,
    )
    timeseries_path = output_dir / "contribution_timeseries.csv"
    fusion_path = output_dir / "fusion_vs_contribution.csv"
    summary_path = output_dir / "scenario_summary.json"
    rows: list[dict[str, float | int | str]] = []
    try:
        reset = env.reset()
        observation = flatten_observation_dict(reset["obs"], obs_spec)
        auxiliary = flatten_auxiliary_observation_dict(reset["obs"], aux_spec)
        for step in range(int(args.num_steps)):
            action, _, _ = agent.act(observation, auxiliary_observation=auxiliary, deterministic=True)
            with torch.no_grad():
                obs_t = torch.as_tensor(observation, dtype=torch.float32, device=agent.device).reshape(1, -1)
                aux_t = torch.as_tensor(auxiliary, dtype=torch.float32, device=agent.device).reshape(1, -1)
                hidden = agent.hidden_state_before_action
                base = agent.model.forward_policy_outputs(
                    obs_t,
                    auxiliary=aux_t,
                    hidden_state=hidden,
                    return_shapley_context=True,
                    track_diagnostics=False,
                )
                estimate = estimate_exact_shapley_current_critic(
                    agent.model,
                    projected_feature_stack=base["projected_feature_stack"],  # type: ignore[arg-type]
                    fusion_weights=base["fusion_weights"],  # type: ignore[arg-type]
                    hidden_state=hidden,
                    config=ShapleyEstimatorConfig(
                        contribution_temperature=float(config.shapley.contribution_temperature),
                        min_contribution=float(config.shapley.min_contribution),
                    ),
                )
                final = agent.model.forward_policy_outputs(
                    obs_t,
                    auxiliary=aux_t,
                    hidden_state=hidden,
                    contribution_override=estimate.contribution,
                    track_diagnostics=False,
                )
                empty_mask = torch.zeros((1, 4), dtype=torch.float32, device=agent.device)
                value_empty = agent.model.evaluate_masked_values(
                    final["projected_feature_stack"] if isinstance(final["projected_feature_stack"], torch.Tensor) else base["projected_feature_stack"],  # type: ignore[arg-type]
                    final["fusion_weights"],  # type: ignore[arg-type]
                    empty_mask,
                    hidden_state=hidden,
                )
            result = env.step(action, policy_diagnostics=agent.get_diagnostics())
            info = dict(result["info"])
            row = {
                "step": step,
                "route_progress": float(info.get("route_progress", 0.0)),
                "risk_scalar": float(final["risk_scalar"].detach().mean().cpu().item()),  # type: ignore[union-attr]
                "corruption_level": float(info.get("corruption_level", 0.0)),
                "value_full": float(final["value"].detach().mean().cpu().item()),  # type: ignore[union-attr]
                "value_empty": float(value_empty.detach().mean().cpu().item()),
                "done_reason": str(info.get("done_reason", "running")),
            }
            for index, key in enumerate(("rgb", "lidar", "ego", "route")):
                row[f"fusion_{key}"] = float(final["fusion_weights"][:, index].detach().mean().cpu().item())  # type: ignore[index,union-attr]
                row[f"contribution_{key}"] = float(estimate.contribution[:, index].detach().mean().cpu().item())
                row[f"reliability_{key}"] = float(final["reliability"][:, index].detach().mean().cpu().item())  # type: ignore[index,union-attr]
            rows.append(row)
            observation = flatten_observation_dict(result["obs"], obs_spec)
            auxiliary = flatten_auxiliary_observation_dict(result["obs"], aux_spec)
            if bool(result["done"]):
                reset = env.reset()
                observation = flatten_observation_dict(reset["obs"], obs_spec)
                auxiliary = flatten_auxiliary_observation_dict(reset["obs"], aux_spec)
                agent.reset_hidden()
        with timeseries_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with fusion_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "run_name": run_name,
            "steps": len(rows),
            "timeseries": str(timeseries_path),
            "fusion_vs_contribution": str(fusion_path),
            "max_corruption": max((float(row["corruption_level"]) for row in rows), default=0.0),
            "final_route_progress": float(rows[-1]["route_progress"]) if rows else 0.0,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
