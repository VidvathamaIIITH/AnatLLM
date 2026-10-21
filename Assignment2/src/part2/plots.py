"""Figures for Part 2: optimizer training dynamics, in the style of Figure 2 of
*Fantastic Pretraining Optimizers and Where to Find Them*."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..plotting import annotate_bars, color, save, use_style
from ..utils import configure_stdout, load_json, read_jsonl, repo_root, work_dir
from .optimizers import OPTIMIZER_LABEL, OPTIMIZERS

ORDER = ["adamw", "mars", "lion", "muon", "sophia"]


def _load(runs_dir: Path):
    summaries, logs = {}, {}
    for name in ORDER:
        s = runs_dir / name / "summary.json"
        l = runs_dir / name / "train_log.jsonl"
        if s.exists():
            summaries[name] = load_json(s)
        if l.exists():
            logs[name] = read_jsonl(l)
    return summaries, logs


def plot_dynamics(summaries: dict, out: Path) -> None:
    """Validation loss and test BLEU against tokens seen - the Figure-2 view."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for i, (name, s) in enumerate(summaries.items()):
        curve = s["curve"]
        if not curve:
            continue
        x = np.array([c["tokens"] for c in curve]) / 1e6
        axes[0].plot(x, [c["val_loss"] for c in curve], marker="o", markersize=4,
                     color=color(i), label=OPTIMIZER_LABEL[name])
        bleu = [c["test_bleu"] for c in curve]
        if not all(np.isnan(bleu)):
            axes[1].plot(x, bleu, marker="s", markersize=4, color=color(i),
                         label=OPTIMIZER_LABEL[name])

    axes[0].set_ylabel("validation loss")
    axes[0].set_title("Validation loss vs tokens")
    axes[1].set_ylabel("test BLEU")
    axes[1].set_title("Test BLEU vs tokens")
    for ax in axes:
        ax.set_xlabel("tokens seen (millions)")
        ax.legend()

    # A second x-axis in units of the dataset, which is what the assignment asks for.
    any_s = next(iter(summaries.values()))
    total = any_s["dataset_tokens"] / 1e6
    for ax in axes:
        sec = ax.secondary_xaxis("top", functions=(lambda v: v / total, lambda v: v * total))
        sec.set_xlabel("fraction of the dataset (1.0x = one epoch)")
    fig.suptitle("Part 2 - training dynamics by optimizer (identical data order and budget)",
                 y=1.06, fontsize=11)
    save(fig, out / "part2_dynamics.png")


def plot_final_comparison(summaries: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(summaries)
    labels = [OPTIMIZER_LABEL[n].split(" (")[0] for n in names]
    x = np.arange(len(names))
    val = [summaries[n]["final_val_loss"] for n in names]
    bleu = [summaries[n]["final_test_bleu"] for n in names]
    state = [summaries[n]["optimizer_state_ratio"] for n in names]
    tps = [summaries[n]["tokens_per_s"] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    for ax, vals, title, fmt in (
        (axes[0], val, "Final validation loss", "{:.4f}"),
        (axes[1], bleu, "Final test BLEU", "{:.2f}"),
        (axes[2], state, "Optimizer state\n(x model fp32 size)", "{:.2f}"),
        (axes[3], tps, "Throughput (tokens/s)", "{:,.0f}"),
    ):
        vals = [0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in vals]
        b = ax.bar(x, vals, color=[color(i) for i in range(len(names))], width=0.62)
        annotate_bars(ax, b, vals, fmt)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_axisbelow(True)
    axes[0].set_ylim(min(val) - 0.05, max(val) + 0.05)
    fig.suptitle("Part 2 - final results at 1.0x the dataset", y=1.05, fontsize=11)
    save(fig, out / "part2_final_comparison.png")


def plot_train_loss(logs: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (name, rows) in enumerate(logs.items()):
        tr = [(r["tokens"], r["loss"]) for r in rows if r["event"] == "train"]
        if not tr:
            continue
        x, y = zip(*tr)
        ax.plot(np.array(x) / 1e6, y, color=color(i), label=OPTIMIZER_LABEL[name], alpha=0.9)
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("training loss")
    ax.set_title("Part 2 - training loss by optimizer")
    ax.legend()
    save(fig, out / "part2_train_loss.png")


def plot_sweep(sweep_dir: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    path = sweep_dir / "sweep_results.json"
    if not path.exists():
        return
    data = load_json(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (name, r) in enumerate(data.items()):
        lrs = [t["lr"] for t in r["trials"]]
        loss = [t["val_loss"] for t in r["trials"]]
        ax.plot(lrs, loss, marker="o", markersize=5, color=color(ORDER.index(name)
                if name in ORDER else i), label=OPTIMIZER_LABEL.get(name, name))
        best = r["best_lr"]
        bl = min(loss)
        ax.scatter([best], [bl], s=110, facecolors="none",
                   edgecolors=color(ORDER.index(name) if name in ORDER else i), linewidths=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("validation loss")
    frac = next(iter(data.values()))["fraction"] if data else 0.1
    ax.set_title(f"Part 2 - learning-rate sweep at {frac:.1f}x the dataset\n"
                 "(circled = selected for the full run)")
    ax.legend()
    save(fig, out / "part2_lr_sweep.png")


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    use_style()
    ap = argparse.ArgumentParser(description="Generate Part 2 figures")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs) if args.runs else work_dir() / "part2" / "runs"
    sweep_dir = Path(args.sweep) if args.sweep else work_dir() / "part2" / "sweep"
    out = Path(args.out) if args.out else repo_root() / "assets"

    summaries, logs = _load(runs_dir)
    if summaries:
        plot_dynamics(summaries, out)
        plot_final_comparison(summaries, out)
    if logs:
        plot_train_loss(logs, out)
    plot_sweep(sweep_dir, out)
    if not summaries and not logs:
        raise SystemExit(f"no Part 2 results under {runs_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
