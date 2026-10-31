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
    throttle_values = [float(row.get("recent_throttle_mean", 0.0)) for row in episode_rows]
    brake_values = [float(row.get("recent_brake_mean", 0.0)) for row in episode_rows]
    steer_values = [float(row.get("recent_steer_mean", 0.0)) for row in episode_rows]
    min_target_distances = [float(row.get("min_target_distance", row.get("target_distance", 0.0))) for row in episode_rows]
    final_target_distances = [float(row.get("final_target_distance", row.get("target_distance", 0.0))) for row in episode_rows]
    total_episodes = max(1, len(episode_rows))
    goal_count = sum(1 for row in episode_rows if row.get("done_reason") == "goal")
    stuck_count = sum(1 for row in episode_rows if row.get("done_reason") == "stuck")
    offroad_count = sum(1 for row in episode_rows if row.get("done_reason") == "offroad")
    max_steps_count = sum(1 for row in episode_rows if row.get("done_reason") == "max_steps")
    throttle_gt_brake = sum(1 for row in episode_rows if float(row.get("recent_throttle_mean", 0.0)) > float(row.get("recent_brake_mean", 0.0)))
    brake_gt_throttle = sum(1 for row in episode_rows if float(row.get("recent_brake_mean", 0.0)) > float(row.get("recent_throttle_mean", 0.0)))
    low_speed_brake_suppressed = sum(
        1 for row in episode_rows if str(row.get("low_speed_brake_suppressed", "")).lower() == "true"
    )
    goal_check_hits = sum(1 for row in episode_rows if str(row.get("goal_check_hit", "")).lower() == "true")
    near_goal_max_steps = sum(
        1
        for row in episode_rows
        if row.get("done_reason") == "max_steps" and float(row.get("route_progress_at_done", row.get("route_progress", 0.0))) >= 0.8
    )

    print(f"[INFO] log={log_path.resolve()}")
    print(f"[INFO] episode_rows={len(episode_rows)} update_rows={len(update_rows)}")
    print(f"[INFO] done_reason_distribution={done_reason_counts}")
    print(f"[INFO] avg_episode_length={average(episode_length_values):.3f}")
    print(f"[INFO] avg_route_progress={average(route_progress_values):.6f}")
    print(f"[INFO] max_route_progress={max(route_progress_values) if route_progress_values else 0.0:.6f}")
    print(f"[INFO] avg_episode_speed_kmh={average(avg_speed_values):.6f}")
    print(f"[INFO] avg_recent_throttle_mean={average(throttle_values):.6f}")
    print(f"[INFO] avg_recent_brake_mean={average(brake_values):.6f}")
    print(f"[INFO] avg_recent_steer_mean={average(steer_values):.6f}")
    print(f"[INFO] min_target_distance_min={min(min_target_distances) if min_target_distances else 0.0:.6f}")
    print(f"[INFO] avg_final_target_distance={average(final_target_distances):.6f}")
    print(f"[INFO] goal_count={goal_count}")
    print(f"[INFO] goal_ratio={goal_count / total_episodes:.3f}")
    print(f"[INFO] goal_check_hit_count={goal_check_hits}")
    print(f"[INFO] stuck_ratio={stuck_count / total_episodes:.3f}")
    print(f"[INFO] offroad_ratio={offroad_count / total_episodes:.3f}")
    print(f"[INFO] max_steps_ratio={max_steps_count / total_episodes:.3f}")
    print(f"[INFO] near_goal_max_steps_count={near_goal_max_steps}")
    print(f"[INFO] throttle_gt_brake_ratio={throttle_gt_brake / total_episodes:.3f}")
    print(f"[INFO] brake_gt_throttle_ratio={brake_gt_throttle / total_episodes:.3f}")
    print(f"[INFO] low_speed_brake_suppressed_ratio={low_speed_brake_suppressed / total_episodes:.3f}")

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
