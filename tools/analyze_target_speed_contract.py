from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _float(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name, "")
        if value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bin_label(value: float, width: float) -> str:
    start = int(value / width) * width
    end = start + width
    return f"{start:.2f}-{end:.2f}"


def _summarize(rows: list[dict[str, str]], *, bin_width: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        progress = _float(row, "route_progress", default=-1.0)
        if progress < 0.0:
            continue
        groups[_bin_label(progress, bin_width)].append(row)

    result: dict[str, Any] = {}
    for label, group in sorted(groups.items()):
        values: dict[str, list[float]] = defaultdict(list)
        done_counts: dict[str, int] = defaultdict(int)
        for row in group:
            for output_key, names in {
                "speed_kmh": ("speed_kmh", "speed"),
                "target_speed_kmh": ("target_speed_kmh",),
                "speed_error_kmh": ("speed_error_kmh",),
                "actual_throttle": ("applied_throttle", "throttle"),
                "actual_brake": ("applied_brake", "brake"),
                "policy_longitudinal": ("longitudinal", "raw_longitudinal", "policy_action_longitudinal"),
                "lane_offset": ("lane_offset",),
                "heading_error": ("heading_error",),
                "fusion_rgb": ("fusion_rgb", "fusion_weight_rgb"),
                "fusion_lidar": ("fusion_lidar", "fusion_weight_lidar"),
                "fusion_ego": ("fusion_ego", "fusion_weight_ego"),
                "fusion_route": ("fusion_route", "fusion_weight_route"),
            }.items():
                values[output_key].append(_float(row, *names))
            if str(row.get("done", "")).lower() in {"true", "1"}:
                done_reason = str(row.get("done_reason", "unknown") or "unknown")
                done_counts[done_reason] += 1

        summary: dict[str, Any] = {"rows": len(group)}
        for key, vals in values.items():
            if vals:
                summary[f"{key}_mean"] = mean(vals)
                summary[f"{key}_min"] = min(vals)
                summary[f"{key}_max"] = max(vals)
        if done_counts:
            summary["done_reason_counts"] = dict(done_counts)
        result[label] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, Any] = {}
    for path in args.csv_paths:
        rows = _load_rows(path)
        output[str(path)] = _summarize(rows, bin_width=float(args.bin_width))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
