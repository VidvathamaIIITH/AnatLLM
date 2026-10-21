"""Shared matplotlib styling so every figure in the report looks like a set."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Colour-blind-safe qualitative palette (Okabe-Ito), used everywhere.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple-pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C564B",  # brown
]

GRID_KW = dict(color="#D8DEE4", linewidth=0.7, alpha=0.9)


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#5A6672",
        "axes.grid": True,
        "grid.color": GRID_KW["color"],
        "grid.linewidth": GRID_KW["linewidth"],
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.color": "#3C4753",
        "ytick.color": "#3C4753",
        "axes.labelcolor": "#22282F",
        "text.color": "#22282F",
        "lines.linewidth": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def save(fig, path: str | Path, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def annotate_bars(ax, bars, values, fmt="{:.2f}", dy=0.01, fontsize=7.5) -> None:
    """Put a value label above each bar."""
    top = max(values) if len(values) else 1.0
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dy * top,
                fmt.format(v), ha="center", va="bottom", fontsize=fontsize)
