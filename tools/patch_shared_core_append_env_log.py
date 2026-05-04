from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "env" / "carla_env.py"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"pattern not found:\n{old[:400]}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """        self._close_log_file()
        self.csv_file = open(self.logs_path, "w", newline="", encoding="utf-8")
""",
        """        self._close_log_file()
        log_exists = os.path.exists(self.logs_path) and os.path.getsize(self.logs_path) > 0
        self.csv_file = open(self.logs_path, "a" if log_exists else "w", newline="", encoding="utf-8")
""",
    )
    text = replace_once(
        text,
        """        self.csv_writer.writeheader()
""",
        """        if not log_exists:
            self.csv_writer.writeheader()
""",
    )
    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
