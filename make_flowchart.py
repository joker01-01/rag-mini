# -*- coding: utf-8 -*-
"""生成文章配图：最小 RAG 系统数据流示意图（PNG）。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
OUT = ROOT / "article" / "rag_flowchart.png"

STEPS = [
    "任意文件\nPDF/Word/PPT/Excel",
    "MarkItDown\n转 Markdown",
    "按标题切分\nchunk",
    "bge-m3\n向量化",
    "Chroma\n向量库",
    "检索 top-5\n相似度过滤",
    "DeepSeek\n生成回答",
    "带来源引用\n的答案",
]


def main() -> None:
    for font_path in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"]:
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(13.5, 3.2), dpi=200)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    n = len(STEPS)
    box_w, box_h = 1.42, 1.7
    gap = 0.24
    x = 0.18
    y = 0.75
    for i, label in enumerate(STEPS):
        is_gen = (i == n - 2)
        is_out = (i == n - 1)
        if is_out:
            fc, ec = "#e8f5e9", "#2e7d32"
        elif is_gen:
            fc, ec = "#e3f2fd", "#1565c0"
        else:
            fc, ec = "#fff8e1", "#f9a825"
        box = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.06",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#212121",
            linespacing=1.5,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap, y + box_h / 2),
                xytext=(x + box_w, y + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#616161", lw=1.5),
            )
        x += box_w + gap

    ax.text(6.75, 2.85, "最小 RAG 系统数据流", ha="center", fontsize=14, color="#37474f")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"示意图已生成: {OUT}")


if __name__ == "__main__":
    main()
