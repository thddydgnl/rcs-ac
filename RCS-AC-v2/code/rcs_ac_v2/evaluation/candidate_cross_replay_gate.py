"""Gate Stage 2 candidate promotion using cross-replay offline probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_candidate_cross_replay_gate(
    *,
    output_dir: str | Path,
    replay_names: list[str],
    min_latent_accuracy_margin: float = 0.15,
    min_latent_test_accuracy: float = 0.80,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not replay_names:
        raise ValueError("at least one replay name is required")
    root = Path(output_dir)
    replay_results: dict[str, Any] = {}
    failures: list[str] = []
    for replay_name in replay_names:
        replay_dir = root / replay_name
        action_path = replay_dir / "stage2_checkpoint_action_probe.json"
        latent_path = replay_dir / "stage2_replay_latent_probe.json"
        if not action_path.exists():
            failures.append(f"{replay_name}: missing action probe {action_path}")
            continue
        if not latent_path.exists():
            failures.append(f"{replay_name}: missing latent probe {latent_path}")
            continue
        action_probe = _read_json(action_path)
        latent_probe = _read_json(latent_path)
        action_ready = bool(action_probe.get("recommended_for_clean_rerun") is True)
        latent_test = float(latent_probe.get("linear_probe_test_accuracy", 0.0))
        latent_majority = float(latent_probe.get("majority_action_fraction", 0.0))
        latent_margin = latent_test - latent_majority
        latent_ready = latent_test >= min_latent_test_accuracy and latent_margin >= min_latent_accuracy_margin
        if not action_ready:
            failures.append(f"{replay_name}: action probe recommended_for_clean_rerun is not true")
        if not latent_ready:
            failures.append(
                f"{replay_name}: latent probe test={latent_test:.6f}, majority={latent_majority:.6f}, "
                f"margin={latent_margin:.6f}"
            )
        replay_results[replay_name] = {
            "action_probe": {
                "path": str(action_path),
                "recommended_for_clean_rerun": action_ready,
                "action_counts": action_probe.get("action_counts", {}),
                "moving_steer_fraction": action_probe.get("moving_steer_fraction"),
                "dominant_action_fraction": action_probe.get("dominant_action_fraction"),
                "nonzero_throttle_fraction": action_probe.get("nonzero_throttle_fraction"),
                "reverse_action_count": action_probe.get("reverse_action_count"),
            },
            "latent_probe": {
                "path": str(latent_path),
                "majority_action_fraction": latent_majority,
                "linear_probe_test_accuracy": latent_test,
                "linear_probe_train_accuracy": latent_probe.get("linear_probe_train_accuracy"),
                "accuracy_margin_over_majority": latent_margin,
            },
            "ready": action_ready and latent_ready,
        }
    ready_for_clean_eval = not failures and len(replay_results) == len(replay_names)
    result = {
        "stage": "stage2_privileged_policy_clean",
        "source": "stage2_candidate_cross_replay_gate",
        "clean_gate_evidence": False,
        "output_dir": str(root),
        "replay_names": replay_names,
        "min_latent_accuracy_margin": float(min_latent_accuracy_margin),
        "min_latent_test_accuracy": float(min_latent_test_accuracy),
        "ready_for_clean_eval": ready_for_clean_eval,
        "failures": failures,
        "replays": replay_results,
        "warning": "Offline gate only; this is not clean PASS evidence.",
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate Stage 2 candidate promotion from cross-replay probes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-name", action="append", required=True)
    parser.add_argument("--min-latent-accuracy-margin", type=float, default=0.15)
    parser.add_argument("--min-latent-test-accuracy", type=float, default=0.80)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    result = evaluate_candidate_cross_replay_gate(
        output_dir=args.output_dir,
        replay_names=list(args.replay_name),
        min_latent_accuracy_margin=args.min_latent_accuracy_margin,
        min_latent_test_accuracy=args.min_latent_test_accuracy,
        output_path=args.output or None,
    )
    print(f"STAGE2_CANDIDATE_CROSS_REPLAY_GATE={json.dumps(result, sort_keys=True)}")
    if not result["ready_for_clean_eval"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
