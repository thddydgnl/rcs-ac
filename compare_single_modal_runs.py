from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple single-modal training logs.")
    parser.add_argument("training_logs", nargs="+", help="Paths to training_log.csv files")
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(path: str) -> dict[str, float | str | dict[str, int]]:
    rows = read_rows(path)
    episode_rows = [row for row in rows if row.get("record_type") == "episode"]
    done_reason_distribution: dict[str, int] = {}
    for row in episode_rows:
        key = row.get("done_reason", "")
        done_reason_distribution[key] = done_reason_distribution.get(key, 0) + 1

    total = max(1, len(episode_rows))
    return {
        "path": str(Path(path).resolve()),
        "episodes": len(episode_rows),
        "done_reason_distribution": done_reason_distribution,
        "avg_episode_length": average([float(row.get("episode_length", 0.0)) for row in episode_rows]),
        "avg_route_progress": average([float(row.get("route_progress", 0.0)) for row in episode_rows]),
        "max_route_progress": max([float(row.get("route_progress", 0.0)) for row in episode_rows], default=0.0),
        "avg_episode_speed_kmh": average([float(row.get("average_speed_kmh", 0.0)) for row in episode_rows]),
        "avg_recent_throttle_mean": average([float(row.get("recent_throttle_mean", 0.0)) for row in episode_rows]),
        "avg_recent_brake_mean": average([float(row.get("recent_brake_mean", 0.0)) for row in episode_rows]),
        "avg_recent_steer_mean": average([float(row.get("recent_steer_mean", 0.0)) for row in episode_rows]),
        "stuck_ratio": sum(1 for row in episode_rows if row.get("done_reason") == "stuck") / total,
        "offroad_ratio": sum(1 for row in episode_rows if row.get("done_reason") == "offroad") / total,
        "max_steps_ratio": sum(1 for row in episode_rows if row.get("done_reason") == "max_steps") / total,
    }


def main() -> int:
    args = parse_args()
    for path in args.training_logs:
        summary = summarize(path)
        print(f"[RUN] {summary['path']}")
        print(f"  episodes={summary['episodes']}")
        print(f"  done_reason_distribution={summary['done_reason_distribution']}")
        print(f"  avg_episode_length={summary['avg_episode_length']:.3f}")
        print(f"  avg_route_progress={summary['avg_route_progress']:.6f}")
        print(f"  max_route_progress={summary['max_route_progress']:.6f}")
        print(f"  avg_episode_speed_kmh={summary['avg_episode_speed_kmh']:.6f}")
        print(f"  avg_recent_throttle_mean={summary['avg_recent_throttle_mean']:.6f}")
        print(f"  avg_recent_brake_mean={summary['avg_recent_brake_mean']:.6f}")
        print(f"  avg_recent_steer_mean={summary['avg_recent_steer_mean']:.6f}")
        print(f"  stuck_ratio={summary['stuck_ratio']:.3f}")
        print(f"  offroad_ratio={summary['offroad_ratio']:.3f}")
        print(f"  max_steps_ratio={summary['max_steps_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
