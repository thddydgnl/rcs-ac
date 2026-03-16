from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def latest_matching_file(pattern: str) -> str:
    matches = sorted(Path().glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return str(matches[0])


def main() -> int:
    run_name = "verify_single_4096"
    command = [
        sys.executable,
        "train.py",
        "--config",
        os.path.join("configs", "train_single_modal.yaml"),
        "--model-type",
        "single_modal",
        "--total-timesteps",
        "4096",
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)
    if completed.returncode != 0:
        print(f"[FAIL] single-modal 4096 run failed: returncode={completed.returncode}")
        return 1

    training_log_path = latest_matching_file(f"logs/{run_name}_*/training_log.csv")
    checkpoint_path = latest_matching_file(f"checkpoints/{run_name}_*/last.pt")

    print(f"[INFO] training_log={Path(training_log_path).resolve()}")
    print(f"[INFO] checkpoint={Path(checkpoint_path).resolve()}")

    summarize = subprocess.run(
        [sys.executable, "summarize_training_log.py", training_log_path],
        capture_output=True,
        text=True,
        check=False,
    )
    print(summarize.stdout)
    if summarize.stderr.strip():
        print(summarize.stderr)

    analyze = subprocess.run(
        [sys.executable, "analyze_stuck_debug.py", training_log_path],
        capture_output=True,
        text=True,
        check=False,
    )
    print(analyze.stdout)
    if analyze.stderr.strip():
        print(analyze.stderr)

    return 0 if summarize.returncode == 0 and analyze.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
