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
    route_window_action_ids: list[int] | None = None,
    route_window_min_route_progress: float = 0.64,
    route_window_max_route_progress: float = 0.70,
    route_window_min_action_fraction: float = 0.50,
    route_window_early_max_route_progress: float = 0.60,
    route_window_max_early_action_fraction: float = 0.02,
    route_window_required_replay_names: list[str] | None = None,
    route_window_max_zero_throttle_steer_fraction: float = 0.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not replay_names:
        raise ValueError("at least one replay name is required")
    if min_latent_accuracy_margin < 0.0:
        raise ValueError("min_latent_accuracy_margin must be non-negative")
    route_action_ids = tuple(int(action_id) for action_id in route_window_action_ids or ())
    route_required_replays = tuple(route_window_required_replay_names or ())
    if route_action_ids:
        if route_window_min_route_progress >= route_window_max_route_progress:
            raise ValueError("route_window_min_route_progress must be < route_window_max_route_progress")
        if not 0.0 <= route_window_min_action_fraction <= 1.0:
            raise ValueError("route_window_min_action_fraction must be in [0, 1]")
        if not 0.0 <= route_window_max_early_action_fraction <= 1.0:
            raise ValueError("route_window_max_early_action_fraction must be in [0, 1]")
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
        latent_headroom = max(0.0, 1.0 - latent_majority)
        latent_margin_fraction = latent_margin / latent_headroom if latent_headroom > 0.0 else 0.0
        latent_ready = (
            latent_test >= min_latent_test_accuracy
            and latent_margin_fraction >= min_latent_accuracy_margin
        )
        if not action_ready:
            failures.append(f"{replay_name}: action probe recommended_for_clean_rerun is not true")
        if not latent_ready:
            failures.append(
                f"{replay_name}: latent probe test={latent_test:.6f}, majority={latent_majority:.6f}, "
                f"margin={latent_margin:.6f}, headroom_fraction={latent_margin_fraction:.6f}"
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
                "accuracy_headroom_over_majority": latent_headroom,
                "accuracy_margin_fraction_of_headroom": latent_margin_fraction,
            },
            "ready": action_ready and latent_ready,
        }
    route_window_gate = None
    if route_action_ids:
        route_window_gate = _evaluate_route_window_gate(
            replay_results=replay_results,
            route_action_ids=route_action_ids,
            min_route_progress=float(route_window_min_route_progress),
            max_route_progress=float(route_window_max_route_progress),
            min_action_fraction=float(route_window_min_action_fraction),
            early_max_route_progress=float(route_window_early_max_route_progress),
            max_early_action_fraction=float(route_window_max_early_action_fraction),
            required_replay_names=route_required_replays,
            max_zero_throttle_steer_fraction=float(route_window_max_zero_throttle_steer_fraction),
        )
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
        "route_window_gate": route_window_gate,
        "failures": failures,
        "replays": replay_results,
        "warning": "Offline gate only; this is not clean PASS evidence.",
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _evaluate_route_window_gate(
    *,
    replay_results: dict[str, Any],
    route_action_ids: tuple[int, ...],
    min_route_progress: float,
    max_route_progress: float,
    min_action_fraction: float,
    early_max_route_progress: float,
    max_early_action_fraction: float,
    required_replay_names: tuple[str, ...],
    max_zero_throttle_steer_fraction: float,
) -> dict[str, Any]:
    action_id_strings = {str(action_id) for action_id in route_action_ids}
    failures: list[str] = []
    replays: dict[str, Any] = {}
    required_names = set(required_replay_names)
    for replay_name, replay_result in replay_results.items():
        action_probe = _read_json(Path(replay_result["action_probe"]["path"]))
        bins = action_probe.get("route_progress_bins", {})
        if not isinstance(bins, dict):
            failures.append(f"{replay_name}: action probe missing route_progress_bins")
            continue
        window_count = 0
        window_action_count = 0
        early_count = 0
        early_action_count = 0
        for bin_name, bin_payload in bins.items():
            if not isinstance(bin_payload, dict):
                continue
            bin_lower, bin_upper = _parse_route_bin(str(bin_name))
            count = int(bin_payload.get("count", 0))
            action_counts = bin_payload.get("action_counts", {})
            if not isinstance(action_counts, dict):
                action_counts = {}
            matching_actions = sum(int(action_counts.get(action_id, 0)) for action_id in action_id_strings)
            if _overlaps(bin_lower, bin_upper, min_route_progress, max_route_progress):
                window_count += count
                window_action_count += matching_actions
            if bin_upper <= early_max_route_progress:
                early_count += count
                early_action_count += matching_actions
        window_action_fraction = float(window_action_count) / float(window_count) if window_count else 0.0
        early_action_fraction = float(early_action_count) / float(early_count) if early_count else 0.0
        zero_throttle_steer_fraction = float(action_probe.get("zero_throttle_steer_fraction", 0.0))
        nonzero_throttle_fraction = float(action_probe.get("nonzero_throttle_fraction", 0.0))
        reverse_action_count = int(action_probe.get("reverse_action_count", 0))
        replay_required = replay_name in required_names or (not required_names and window_count > 0)
        if replay_required and window_count == 0:
            failures.append(f"{replay_name}: no route-window bins in {min_route_progress:.3f}-{max_route_progress:.3f}")
        if replay_required and window_action_fraction < min_action_fraction:
            failures.append(
                f"{replay_name}: route-window action fraction {window_action_fraction:.6f} "
                f"< {min_action_fraction:.6f}"
            )
        if early_action_fraction > max_early_action_fraction:
            failures.append(
                f"{replay_name}: early route action fraction {early_action_fraction:.6f} "
                f"> {max_early_action_fraction:.6f}"
            )
        if zero_throttle_steer_fraction > max_zero_throttle_steer_fraction:
            failures.append(
                f"{replay_name}: zero-throttle steer fraction {zero_throttle_steer_fraction:.6f} "
                f"> {max_zero_throttle_steer_fraction:.6f}"
            )
        if nonzero_throttle_fraction < 0.5:
            failures.append(f"{replay_name}: nonzero throttle fraction {nonzero_throttle_fraction:.6f} < 0.500000")
        if reverse_action_count:
            failures.append(f"{replay_name}: reverse action count {reverse_action_count} > 0")
        replays[replay_name] = {
            "required_for_window": replay_required,
            "window_count": int(window_count),
            "window_action_count": int(window_action_count),
            "window_action_fraction": window_action_fraction,
            "early_count": int(early_count),
            "early_action_count": int(early_action_count),
            "early_action_fraction": early_action_fraction,
            "zero_throttle_steer_fraction": zero_throttle_steer_fraction,
            "nonzero_throttle_fraction": nonzero_throttle_fraction,
            "reverse_action_count": reverse_action_count,
        }
    if required_names:
        missing = sorted(required_names.difference(replay_results))
        failures.extend(f"{name}: required route-window replay missing" for name in missing)
    ready = not failures and bool(replay_results)
    return {
        "source": "route_window_cross_replay_gate",
        "clean_gate_evidence": False,
        "ready_for_route_window_clean_eval": ready,
        "action_ids": list(route_action_ids),
        "min_route_progress": float(min_route_progress),
        "max_route_progress": float(max_route_progress),
        "min_action_fraction": float(min_action_fraction),
        "early_max_route_progress": float(early_max_route_progress),
        "max_early_action_fraction": float(max_early_action_fraction),
        "max_zero_throttle_steer_fraction": float(max_zero_throttle_steer_fraction),
        "required_replay_names": list(required_replay_names),
        "failures": failures,
        "replays": replays,
        "warning": (
            "Route-window gate is H3 clean-stability fill only; it does not replace "
            "official CARLA clean evidence or PDF-explicit core results."
        ),
    }


def _parse_route_bin(bin_name: str) -> tuple[float, float]:
    if "-" not in bin_name:
        raise ValueError(f"invalid route bin name: {bin_name}")
    lower_raw, upper_raw = bin_name.split("-", 1)
    return float(lower_raw), float(upper_raw)


def _overlaps(left_a: float, right_a: float, left_b: float, right_b: float) -> bool:
    return left_a < right_b and left_b < right_a


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
    parser.add_argument("--route-window-action-ids", default="")
    parser.add_argument("--route-window-min-route-progress", type=float, default=0.64)
    parser.add_argument("--route-window-max-route-progress", type=float, default=0.70)
    parser.add_argument("--route-window-min-action-fraction", type=float, default=0.50)
    parser.add_argument("--route-window-early-max-route-progress", type=float, default=0.60)
    parser.add_argument("--route-window-max-early-action-fraction", type=float, default=0.02)
    parser.add_argument("--route-window-required-replay-name", action="append", default=[])
    parser.add_argument("--route-window-max-zero-throttle-steer-fraction", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    route_window_action_ids = _parse_action_ids(args.route_window_action_ids)
    result = evaluate_candidate_cross_replay_gate(
        output_dir=args.output_dir,
        replay_names=list(args.replay_name),
        min_latent_accuracy_margin=args.min_latent_accuracy_margin,
        min_latent_test_accuracy=args.min_latent_test_accuracy,
        route_window_action_ids=route_window_action_ids,
        route_window_min_route_progress=args.route_window_min_route_progress,
        route_window_max_route_progress=args.route_window_max_route_progress,
        route_window_min_action_fraction=args.route_window_min_action_fraction,
        route_window_early_max_route_progress=args.route_window_early_max_route_progress,
        route_window_max_early_action_fraction=args.route_window_max_early_action_fraction,
        route_window_required_replay_names=list(args.route_window_required_replay_name),
        route_window_max_zero_throttle_steer_fraction=args.route_window_max_zero_throttle_steer_fraction,
        output_path=args.output or None,
    )
    print(f"STAGE2_CANDIDATE_CROSS_REPLAY_GATE={json.dumps(result, sort_keys=True)}")
    if not result["ready_for_clean_eval"]:
        raise SystemExit(4)


def _parse_action_ids(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    main()
