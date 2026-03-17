from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze early/mid/late policy shift from training_log.csv.")
    parser.add_argument("training_log", help="Path to training_log.csv")
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def phase_slices(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    episode_rows = [row for row in rows if row.get("record_type") == "episode"]
    if not episode_rows:
        return {"early": [], "mid": [], "late": []}

    count = len(episode_rows)
    quarter = max(1, count // 4)
    mid_start = max(0, (count // 2) - (quarter // 2))
    mid_end = min(count, mid_start + quarter)
    return {
        "early": episode_rows[:quarter],
        "mid": episode_rows[mid_start:mid_end],
        "late": episode_rows[-quarter:],
    }


def summarize_phase(rows: list[dict[str, str]]) -> dict[str, float | dict[str, int]]:
    done_counts: dict[str, int] = {}
    for row in rows:
        key = row.get("done_reason", "")
        done_counts[key] = done_counts.get(key, 0) + 1

    throttle_values = [float(row.get("recent_throttle_mean", 0.0)) for row in rows]
    brake_values = [float(row.get("recent_brake_mean", 0.0)) for row in rows]
    steer_values = [float(row.get("recent_steer_mean", 0.0)) for row in rows]
    speed_values = [float(row.get("average_speed_kmh", 0.0)) for row in rows]
    route_values = [float(row.get("route_progress", 0.0)) for row in rows]
    length_values = [float(row.get("episode_length", 0.0)) for row in rows]

    throttle_gt_brake = sum(1 for row in rows if float(row.get("recent_throttle_mean", 0.0)) > float(row.get("recent_brake_mean", 0.0)))
    brake_gt_throttle = sum(1 for row in rows if float(row.get("recent_brake_mean", 0.0)) > float(row.get("recent_throttle_mean", 0.0)))
    stuck_count = sum(1 for row in rows if row.get("done_reason") == "stuck")
    offroad_count = sum(1 for row in rows if row.get("done_reason") == "offroad")
    max_steps_count = sum(1 for row in rows if row.get("done_reason") == "max_steps")
    overlap_count = sum(1 for row in rows if str(row.get("pedal_overlap_suppressed", "")).lower() == "true")
    low_speed_brake_count = sum(1 for row in rows if str(row.get("low_speed_brake_suppressed", "")).lower() == "true")

    total = max(1, len(rows))
    return {
        "episodes": len(rows),
        "avg_episode_speed_kmh": average(speed_values),
        "avg_route_progress": average(route_values),
        "max_route_progress": max(route_values) if route_values else 0.0,
        "avg_episode_length": average(length_values),
        "avg_recent_throttle_mean": average(throttle_values),
        "avg_recent_brake_mean": average(brake_values),
        "avg_recent_steer_mean": average(steer_values),
        "stuck_ratio": stuck_count / total,
        "offroad_ratio": offroad_count / total,
        "max_steps_ratio": max_steps_count / total,
        "throttle_gt_brake_ratio": throttle_gt_brake / total,
        "brake_gt_throttle_ratio": brake_gt_throttle / total,
        "pedal_overlap_suppressed_ratio": overlap_count / total,
        "low_speed_brake_suppressed_ratio": low_speed_brake_count / total,
        "done_reason_distribution": done_counts,
    }


def print_phase(name: str, summary: dict[str, float | dict[str, int]]) -> None:
    print(
        f"[PHASE] {name} "
        f"episodes={summary['episodes']} "
        f"avg_speed={summary['avg_episode_speed_kmh']:.6f} "
        f"avg_route_progress={summary['avg_route_progress']:.6f} "
        f"max_route_progress={summary['max_route_progress']:.6f} "
        f"avg_episode_length={summary['avg_episode_length']:.3f} "
        f"avg_throttle={summary['avg_recent_throttle_mean']:.6f} "
        f"avg_brake={summary['avg_recent_brake_mean']:.6f} "
        f"avg_steer={summary['avg_recent_steer_mean']:.6f} "
        f"stuck_ratio={summary['stuck_ratio']:.3f} "
        f"offroad_ratio={summary['offroad_ratio']:.3f} "
        f"max_steps_ratio={summary['max_steps_ratio']:.3f} "
        f"throttle_gt_brake_ratio={summary['throttle_gt_brake_ratio']:.3f} "
        f"brake_gt_throttle_ratio={summary['brake_gt_throttle_ratio']:.3f} "
        f"pedal_overlap_suppressed_ratio={summary['pedal_overlap_suppressed_ratio']:.3f} "
        f"low_speed_brake_suppressed_ratio={summary['low_speed_brake_suppressed_ratio']:.3f}"
    )
    print(f"[PHASE-DONE] {name} {summary['done_reason_distribution']}")


def rule_based_judgement(early: dict[str, float | dict[str, int]], late: dict[str, float | dict[str, int]]) -> list[str]:
    messages: list[str] = []

    if (
        float(late["avg_recent_throttle_mean"]) < float(early["avg_recent_throttle_mean"])
        and float(late["avg_recent_brake_mean"]) > float(early["avg_recent_brake_mean"])
        and float(late["avg_episode_speed_kmh"]) < float(early["avg_episode_speed_kmh"])
        and float(late["stuck_ratio"]) > float(early["stuck_ratio"])
    ):
        messages.append("throttle 감소 + brake 증가 + speed 감소 + stuck 증가 -> 정책이 정지/보수 방향으로 수렴하는 신호가 강합니다.")

    if (
        float(late["offroad_ratio"]) <= float(early["offroad_ratio"])
        and float(late["stuck_ratio"]) > float(early["stuck_ratio"])
        and float(late["avg_route_progress"]) <= float(early["avg_route_progress"])
    ):
        messages.append("offroad는 줄거나 비슷한데 stuck가 늘고 route_progress가 정체 -> '안 벗어나는 대신 안 가는' 정책일 가능성이 큽니다.")

    if abs(float(late["avg_recent_steer_mean"])) > abs(float(early["avg_recent_steer_mean"])) and float(late["avg_episode_speed_kmh"]) < 2.0:
        messages.append("steer 편향은 일부 보이지만 speed가 매우 낮아 1차 원인은 조향보다 저속 정체일 가능성이 높습니다.")

    if not messages:
        messages.append("phase shift는 일부 보이지만 단일 원인으로 확정하긴 이릅니다. 추가 run 또는 speed 로그 비교가 더 필요합니다.")
    return messages


def main() -> int:
    args = parse_args()
    log_path = Path(args.training_log)
    rows = read_rows(str(log_path))
    phases = phase_slices(rows)
    early_summary = summarize_phase(phases["early"])
    mid_summary = summarize_phase(phases["mid"])
    late_summary = summarize_phase(phases["late"])

    print(f"[INFO] log={log_path.resolve()}")
    print_phase("early", early_summary)
    print_phase("mid", mid_summary)
    print_phase("late", late_summary)

    print("[JUDGEMENT]")
    for message in rule_based_judgement(early_summary, late_summary):
        print(f"- {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
