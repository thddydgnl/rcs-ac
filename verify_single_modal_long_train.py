from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a long single-modal training verification and print summaries.")
    parser.add_argument("--config", default=os.path.join("configs", "train_single_modal.yaml"), help="Config path.")
    parser.add_argument("--run-name", default="verify_single_baseline_long", help="Run name override.")
    parser.add_argument("--total-timesteps", default="8192", help="Total timesteps override.")
    return parser.parse_args()


def latest_matching_file(pattern: str) -> str:
    matches = sorted(Path().glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return str(matches[0])


def read_csv_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def summarize_distribution(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    run_name = args.run_name
    command = [
        sys.executable,
        "train.py",
        "--config",
        args.config,
        "--model-type",
        "single_modal",
        "--total-timesteps",
        str(args.total_timesteps),
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)

    if completed.returncode != 0:
        print(f"[FAIL] single-modal long train failed: returncode={completed.returncode}")
        return 1

    training_log_path = latest_matching_file(f"logs/{run_name}_*/training_log.csv")
    checkpoint_path = latest_matching_file(f"checkpoints/{run_name}_*/last.pt")
    env_log_path = latest_matching_file(f"outputs/ppo_single_modal_env/{run_name}_*/logs.csv")

    _, training_rows = read_csv_rows(training_log_path)
    _, env_rows = read_csv_rows(env_log_path)
    episode_rows = [row for row in training_rows if row.get("record_type") == "episode"]
    update_rows = [row for row in training_rows if row.get("record_type") == "update"]
    offroad_episodes = [row for row in episode_rows if row.get("done_reason") == "offroad"]

    avg_episode_length = (
        sum(float(row["episode_length"]) for row in episode_rows) / len(episode_rows)
        if episode_rows
        else 0.0
    )
    avg_route_progress = (
        sum(float(row["route_progress"]) for row in episode_rows) / len(episode_rows)
        if episode_rows
        else 0.0
    )
    avg_initial_lane_offset = (
        sum(abs(float(row["initial_lane_offset"])) for row in episode_rows) / len(episode_rows)
        if episode_rows
        else 0.0
    )
    avg_initial_heading_error = (
        sum(abs(float(row["initial_heading_error"])) for row in episode_rows) / len(episode_rows)
        if episode_rows
        else 0.0
    )
    avg_first_offroad_step = (
        sum(float(row["first_offroad_step"]) for row in offroad_episodes if float(row["first_offroad_step"]) >= 0.0)
        / max(1, len([row for row in offroad_episodes if float(row["first_offroad_step"]) >= 0.0]))
        if offroad_episodes
        else -1.0
    )
    done_reason_dist = summarize_distribution(episode_rows, "done_reason")

    print(f"[INFO] training_log: {os.path.abspath(training_log_path)}")
    print(f"[INFO] checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"[INFO] env_log: {os.path.abspath(env_log_path)}")
    print(f"[INFO] episode_rows={len(episode_rows)} update_rows={len(update_rows)}")
    print(f"[INFO] done_reason_distribution={done_reason_dist}")
    print(f"[INFO] avg_episode_length={avg_episode_length:.3f}")
    print(f"[INFO] avg_route_progress={avg_route_progress:.6f}")
    print(f"[INFO] avg_initial_lane_offset={avg_initial_lane_offset:.6f}")
    print(f"[INFO] avg_initial_heading_error={avg_initial_heading_error:.6f}")
    print(f"[INFO] avg_first_offroad_step={avg_first_offroad_step:.3f}")

    initial_env_row = env_rows[0] if env_rows else {}
    print(
        "[INFO] initial_env="
        f"spawn_point_index={initial_env_row.get('spawn_point_index', '')} "
        f"lane_offset={initial_env_row.get('lane_offset', '')} "
        f"heading_error={initial_env_row.get('heading_error', '')} "
        f"target_distance={initial_env_row.get('target_distance', '')} "
        f"offroad_candidate={initial_env_row.get('offroad_candidate', '')} "
        f"offroad_grace_active={initial_env_row.get('offroad_grace_active', '')}"
    )

    offroad_env_rows = [row for row in env_rows if row.get("done_reason") == "offroad"]
    if offroad_env_rows:
        sample = offroad_env_rows[0]
        print(
            "[INFO] offroad_sample="
            f"step={sample.get('step', '')} "
            f"lane_offset={sample.get('lane_offset', '')} "
            f"heading_error={sample.get('heading_error', '')} "
            f"target_distance={sample.get('target_distance', '')} "
            f"offroad_steps={sample.get('offroad_steps', '')} "
            f"first_offroad_step={sample.get('first_offroad_step', '')} "
            f"steer={sample.get('steer', '')} "
            f"throttle={sample.get('throttle', '')} "
            f"brake={sample.get('brake', '')}"
        )

    for script_name in ("summarize_training_log.py", "analyze_stuck_debug.py", "analyze_training_phase_shift.py"):
        summary_run = subprocess.run(
            [sys.executable, script_name, training_log_path],
            capture_output=True,
            text=True,
            check=False,
        )
        print(summary_run.stdout)
        if summary_run.stderr.strip():
            print(summary_run.stderr)

    if not episode_rows or not update_rows:
        print("[RESULT] FAIL: episode/update rows are missing.")
        return 1
    if avg_episode_length < 8.0:
        print("[RESULT] FAIL: average episode length is still too short for meaningful analysis.")
        return 1
    if avg_route_progress <= 0.0:
        print("[RESULT] CONDITIONAL PASS: long training runs, but route_progress is still nearly flat.")
        return 0

    print("[RESULT] PASS: long training is stable enough and offroad diagnostics are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
