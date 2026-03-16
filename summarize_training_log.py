from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize single-modal training_log.csv.")
    parser.add_argument("training_log", help="Path to training_log.csv")
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> int:
    args = parse_args()
    log_path = Path(args.training_log)
    rows = read_rows(str(log_path))
    episode_rows = [row for row in rows if row.get("record_type") == "episode"]
    update_rows = [row for row in rows if row.get("record_type") == "update"]

    done_reason_counts: dict[str, int] = {}
    for row in episode_rows:
        key = row.get("done_reason", "")
        done_reason_counts[key] = done_reason_counts.get(key, 0) + 1

    route_progress_values = [float(row.get("route_progress", 0.0)) for row in episode_rows]
    episode_length_values = [float(row.get("episode_length", 0.0)) for row in episode_rows]
    avg_speed_values = [float(row.get("average_speed_kmh", 0.0)) for row in episode_rows]

    print(f"[INFO] log={log_path.resolve()}")
    print(f"[INFO] episode_rows={len(episode_rows)} update_rows={len(update_rows)}")
    print(f"[INFO] done_reason_distribution={done_reason_counts}")
    print(f"[INFO] avg_episode_length={average(episode_length_values):.3f}")
    print(f"[INFO] avg_route_progress={average(route_progress_values):.6f}")
    print(f"[INFO] max_route_progress={max(route_progress_values) if route_progress_values else 0.0:.6f}")
    print(f"[INFO] avg_episode_speed_kmh={average(avg_speed_values):.6f}")

    if episode_rows:
        best_row = max(episode_rows, key=lambda row: float(row.get("route_progress", 0.0)))
        print(
            "[INFO] best_episode="
            f"episode_idx={best_row.get('episode_idx', '')} "
            f"route_progress={best_row.get('route_progress', '')} "
            f"done_reason={best_row.get('done_reason', '')} "
            f"episode_length={best_row.get('episode_length', '')} "
            f"average_speed_kmh={best_row.get('average_speed_kmh', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
