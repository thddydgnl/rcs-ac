from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stuck episodes from single-modal training logs.")
    parser.add_argument("training_log", help="Path to training_log.csv")
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    log_path = Path(args.training_log)
    rows = read_rows(str(log_path))
    stuck_rows = [row for row in rows if row.get("record_type") == "episode" and row.get("done_reason") == "stuck"]

    print(f"[INFO] log={log_path.resolve()}")
    print(f"[INFO] stuck_episode_count={len(stuck_rows)}")
    if not stuck_rows:
        print("[INFO] No stuck episodes found.")
        return 0

    for row in stuck_rows[:10]:
        print(
            "[STUCK] "
            f"episode_idx={row.get('episode_idx', '')} "
            f"episode_length={row.get('episode_length', '')} "
            f"route_progress={row.get('route_progress', '')} "
            f"average_speed_kmh={row.get('average_speed_kmh', '')} "
            f"low_speed_step_count={row.get('low_speed_step_count', '')} "
            f"stuck_trigger_step={row.get('stuck_trigger_step', '')} "
            f"final_lane_offset={row.get('lane_offset', '')} "
            f"final_heading_error={row.get('heading_error', '')} "
            f"final_action=({row.get('final_action_steer', '')}, {row.get('final_action_throttle', '')}, {row.get('final_action_brake', '')}) "
            f"recent_ctrl=window:{row.get('recent_control_window', '')} "
            f"steer:{row.get('recent_steer_mean', '')} "
            f"throttle:{row.get('recent_throttle_mean', '')} "
            f"brake:{row.get('recent_brake_mean', '')} "
            f"pedal_overlap_suppressed={row.get('pedal_overlap_suppressed', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
