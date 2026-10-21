"""Figures for Part 3: decoding-strategy quality, diversity and beam-search cost."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..plotting import annotate_bars, color, save, use_style
from ..utils import configure_stdout, load_json, repo_root, work_dir

FAMILY_COLOR = {"greedy": 0, "beam": 1, "top_k": 2, "top_p": 3}


def _colors(results: list[dict]) -> list[str]:
    return [color(FAMILY_COLOR.get(m["strategy"], 4)) for m in results]


def plot_metrics(results: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [m["label"] for m in results]
    x = np.arange(len(results))
    cols = _colors(results)

    panels = [
        ("accuracy", "Positional token accuracy", "{:.3f}"),
        ("precision", "Precision (unigram overlap)", "{:.3f}"),
        ("recall", "Recall (unigram overlap)", "{:.3f}"),
        ("f1", "F1 (unigram overlap)", "{:.3f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2))
    for ax, (key, title, fmt) in zip(axes.flat, panels):
        vals = [m[key] for m in results]
        b = ax.bar(x, vals, color=cols, width=0.68)
        annotate_bars(ax, b, vals, fmt, fontsize=6.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7.5)
        ax.set_axisbelow(True)
    fig.suptitle("Part 3 - ROCStories continuation quality by decoding strategy "
                 "(Pythia-160M, 1000 stories)", y=1.02, fontsize=11)
    save(fig, out / "part3_metrics.png")


def plot_quality_vs_diversity(results: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    cols = _colors(results)

    axes[0].scatter([m["distinct_2"] for m in results], [m["f1"] for m in results],
                    c=cols, s=64, zorder=3)
    for m in results:
        axes[0].annotate(m["label"], (m["distinct_2"], m["f1"]), fontsize=6.5,
                         xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel("distinct-2 (diversity)")
    axes[0].set_ylabel("F1 (overlap with reference)")
    axes[0].set_title("Quality against diversity")

    x = np.arange(len(results))
    vals = [m["gen_perplexity"] for m in results]
    b = axes[1].bar(x, vals, color=cols, width=0.68)
    annotate_bars(axes[1], b, vals, "{:.1f}", fontsize=6.5)
    axes[1].set_title("Perplexity of the model's own output")
    axes[1].set_ylabel("perplexity")

    vals = [m["repetition_4gram"] for m in results]
    b = axes[2].bar(x, vals, color=cols, width=0.68)
    annotate_bars(axes[2], b, vals, "{:.3f}", fontsize=6.5)
    axes[2].set_title("Repeated 4-grams within a generation")
    axes[2].set_ylabel("fraction repeated")

    for ax in axes[1:]:
        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in results], rotation=32, ha="right", fontsize=7)
        ax.set_axisbelow(True)
    fig.suptitle("Part 3 - the quality / diversity trade-off", y=1.03, fontsize=11)
    save(fig, out / "part3_quality_diversity.png")


def plot_beam_timing(timing: dict, results: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = timing["rows"]
    widths = np.array([r["beam_width"] for r in rows], dtype=float)
    secs = np.array([r["seconds_best"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(widths, secs, marker="o", markersize=6, color=color(1), label="measured")
    axes[0].plot(widths, secs[0] * widths, ls="--", color="#5A6672",
                 label="linear in beam width")
    axes[0].set_xlabel("beam width K")
    axes[0].set_ylabel(f"seconds for {rows[0]['n_examples']} prompts")
    axes[0].set_title("Beam-search wall-clock cost")
    axes[0].set_xticks(widths)
    axes[0].legend()

    ratio = secs / secs[0] / widths
    b = axes[1].bar(range(len(widths)), ratio, color=color(1), width=0.6)
    annotate_bars(axes[1], b, list(ratio), "{:.2f}")
    axes[1].axhline(1.0, ls="--", color="#5A6672", lw=1.2)
    axes[1].set_xticks(range(len(widths)))
    axes[1].set_xticklabels([f"K={int(w)}" for w in widths])
    axes[1].set_ylabel("measured / linear prediction")
    axes[1].set_title("Sub-linear scaling from better GPU utilisation")
    axes[1].set_axisbelow(True)

    fig.suptitle("Part 3 - beam search cost against beam width", y=1.03, fontsize=11)
    save(fig, out / "part3_beam_timing.png")

    # Quality against cost, using the main run's beam entries.
    beams = [m for m in results if m["strategy"] == "beam"]
    if len(beams) >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot([m["wall_seconds"] for m in beams], [m["f1"] for m in beams],
                marker="o", markersize=7, color=color(1))
        for m in beams:
            ax.annotate(f"K={m['beam_width']}", (m["wall_seconds"], m["f1"]),
                        fontsize=8, xytext=(6, 3), textcoords="offset points")
        ax.set_xlabel("wall-clock seconds for the full test set")
        ax.set_ylabel("F1")
        ax.set_title("Does a wider beam buy quality?")
        save(fig, out / "part3_beam_quality_vs_cost.png")


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    use_style()
    ap = argparse.ArgumentParser(description="Generate Part 3 figures")
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    base = Path(args.results) if args.results else work_dir() / "part3"
    out = Path(args.out) if args.out else repo_root() / "assets"

    rpath = base / "results.json"
    if not rpath.exists():
        raise SystemExit(f"no Part 3 results at {rpath}")
    payload = load_json(rpath)
    results = payload["results"]

    plot_metrics(results, out)
    plot_quality_vs_diversity(results, out)

    tpath = base / "beam_timing.json"
    if tpath.exists():
        plot_beam_timing(load_json(tpath), results, out)


if __name__ == "__main__":  # pragma: no cover
    main()
