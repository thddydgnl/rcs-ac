from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
MD_PATH = ROOT / "docs" / "gpt_consultation" / "rcs_ac_diagnosis_package_2026-05-02.md"
PDF_PATH = ROOT / "docs" / "gpt_consultation" / "rcs_ac_diagnosis_package_2026-05-02.pdf"
FONT_PATH = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")


PAGE_W = 8.27
PAGE_H = 11.69
LEFT = 0.62
RIGHT = 0.62
TOP = 0.62
BOTTOM = 0.62
BODY_SIZE = 9.2
CODE_SIZE = 7.4
LINE_H = 0.158
CODE_LINE_H = 0.132


def _strip_md_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for part in text.splitlines():
        if not part:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            part,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines


def _prepare_blocks(markdown: str) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    in_code = False
    code_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                blocks.append({"kind": "code", "lines": code_lines[:]})
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("# "):
            blocks.append({"kind": "h1", "text": _strip_md_inline(line[2:].strip())})
        elif line.startswith("## "):
            blocks.append({"kind": "h2", "text": _strip_md_inline(line[3:].strip())})
        elif line.startswith("### "):
            blocks.append({"kind": "h3", "text": _strip_md_inline(line[4:].strip())})
        elif line == "---":
            blocks.append({"kind": "space", "height": 0.14})
        elif line.strip() == "":
            blocks.append({"kind": "space", "height": 0.08})
        else:
            blocks.append({"kind": "p", "text": _strip_md_inline(line)})
    if code_lines:
        blocks.append({"kind": "code", "lines": code_lines[:]})
    return blocks


def _new_page(pdf: PdfPages, page_num: int) -> tuple[plt.Figure, plt.Axes, float]:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    return fig, ax, PAGE_H - TOP


def _save_page(pdf: PdfPages, fig: plt.Figure, page_num: int, font: FontProperties) -> None:
    fig.text(
        0.5,
        0.028,
        f"RCS-AC 안정화 진단 패키지 | {page_num}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontproperties=font,
        color="#666666",
    )
    pdf.savefig(fig)
    plt.close(fig)


def render() -> None:
    markdown = MD_PATH.read_text(encoding="utf-8")
    blocks = _prepare_blocks(markdown)
    font = FontProperties(fname=str(FONT_PATH))
    mono = font
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    page_num = 1
    with PdfPages(PDF_PATH) as pdf:
        fig, ax, y = _new_page(pdf, page_num)

        def ensure(space: float) -> None:
            nonlocal fig, ax, y, page_num
            if y - space < BOTTOM:
                _save_page(pdf, fig, page_num, font)
                page_num += 1
                fig, ax, y = _new_page(pdf, page_num)

        for block in blocks:
            kind = str(block["kind"])
            if kind == "space":
                y -= float(block.get("height", 0.08))
                continue
            if kind in {"h1", "h2", "h3"}:
                size = {"h1": 18.0, "h2": 13.0, "h3": 10.7}[kind]
                leading = {"h1": 0.34, "h2": 0.25, "h3": 0.20}[kind]
                after = {"h1": 0.18, "h2": 0.11, "h3": 0.08}[kind]
                lines = _wrap(str(block["text"]), 46 if kind == "h1" else 70)
                ensure(leading + len(lines) * LINE_H + after)
                y -= leading
                for line in lines:
                    fig.text(LEFT, y / PAGE_H, line, ha="left", va="top", fontsize=size, fontproperties=font, weight="bold")
                    y -= LINE_H * (size / BODY_SIZE)
                y -= after
            elif kind == "code":
                raw_lines = [item.expandtabs(2) for item in block.get("lines", [])]  # type: ignore[union-attr]
                code_lines: list[str] = []
                for line in raw_lines:
                    code_lines.extend(_wrap(line, 96))
                ensure(0.10 + len(code_lines) * CODE_LINE_H + 0.08)
                y -= 0.05
                for line in code_lines:
                    fig.text(LEFT + 0.05, y / PAGE_H, line, ha="left", va="top", fontsize=CODE_SIZE, fontproperties=mono, color="#222222")
                    y -= CODE_LINE_H
                y -= 0.05
            else:
                text = str(block["text"])
                if text.startswith("- "):
                    bullet = "• "
                    text = bullet + text[2:]
                    indent = LEFT + 0.10
                    wrap_width = 82
                else:
                    indent = LEFT
                    wrap_width = 88
                lines = _wrap(text, wrap_width)
                ensure(len(lines) * LINE_H + 0.03)
                for line in lines:
                    fig.text(indent, y / PAGE_H, line, ha="left", va="top", fontsize=BODY_SIZE, fontproperties=font, color="#111111")
                    y -= LINE_H
                y -= 0.015

        _save_page(pdf, fig, page_num, font)

    print(PDF_PATH)


if __name__ == "__main__":
    render()
