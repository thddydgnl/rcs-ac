"""Helpers for saving CARLA episode artifacts that Codex can inspect."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_rgb_frame(frame: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB frame, got shape {rgb.shape!r}")

    try:
        from PIL import Image
    except ImportError:
        ppm_path = path.with_suffix(".ppm")
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
        ppm_path.write_bytes(header + rgb.tobytes())
        return ppm_path

    Image.fromarray(rgb, mode="RGB").save(path)
    return path


@dataclass(slots=True)
class CodexArtifactRecorder:
    root: Path
    step_records: list[dict[str, Any]] = field(default_factory=list)
    summary_path: Path = field(init=False)
    errors_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    steps_path: Path = field(init=False)
    frame_path: Path = field(init=False)

    def __post_init__(self) -> None:
        ensure_dir(self.root)
        self.summary_path = self.root / "scene_summary.json"
        self.errors_path = self.root / "errors.json"
        self.metrics_path = self.root / "metrics.json"
        self.steps_path = self.root / "step_trace.json"
        self.frame_path = self.root / "latest_frame.png"

    def record_frame(self, frame: np.ndarray) -> str:
        saved_path = save_rgb_frame(frame, self.frame_path)
        return saved_path.name

    def append_step(self, payload: dict[str, Any]) -> None:
        self.step_records.append(payload)
        write_json(self.steps_path, self.step_records)

    def finalize(self, summary: dict[str, Any], metrics: dict[str, Any], errors: list[dict[str, Any]]) -> None:
        write_json(self.summary_path, summary)
        write_json(self.metrics_path, metrics)
        write_json(self.errors_path, errors)
