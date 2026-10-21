"""Figures for Part 1: training curves, quality-vs-parameters, expert usage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..plotting import annotate_bars, color, save, use_style
from ..utils import configure_stdout, load_json, read_jsonl, repo_root, work_dir
from .model import VARIANTS, VARIANT_LABELS

SHORT = {
    "dense": "Dense MLP",
    "moe_4e_top1": "MoE 4e top-1",
    "moe_4e_top2": "MoE 4e top-2",
    "moe_1s_3r_top1": "MoE 1s+3r top-1",
    "moe_4e_top2_wide": "MoE 4e top-2 wide",
}


def _load(runs_dir: Path):
    evals, logs = {}, {}
    for v in VARIANTS:
        ev = runs_dir / v / "eval.json"
        lg = runs_dir / v / "train_log.jsonl"
        if ev.exists():
            evals[v] = load_json(ev)
        if lg.exists():
            logs[v] = read_jsonl(lg)
    return evals, logs


def plot_training_curves(logs: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for i, (v, rows) in enumerate(logs.items()):
        tr = [(r["tokens"], r["lm_loss"]) for r in rows if r["event"] == "train"]
        ev = [(r["tokens"], r["loss_target"]) for r in rows if r["event"] == "eval"]
        if tr:
            x, y = zip(*tr)
            axes[0].plot(np.array(x) / 1e6, y, color=color(i), label=SHORT[v])
        if ev:
            x, y = zip(*ev)
            axes[1].plot(np.array(x) / 1e6, y, color=color(i), marker="o",
                         markersize=3, label=SHORT[v])
    axes[0].set_title("Training loss (all predicted tokens)")
    axes[1].set_title("Validation loss (English target tokens)")
    for ax in axes:
        ax.set_xlabel("training tokens (millions)")
        ax.set_ylabel("cross-entropy")
    axes[0].legend()
    fig.suptitle("Part 1 - identical token budget across FFN variants", y=1.03, fontsize=11)
    save(fig, out / "part1_training_curves.png")


def plot_quality_vs_params(evals: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    order = [v for v in VARIANTS if v in evals]
    labels = [SHORT[v] for v in order]
    ppl = [evals[v]["perplexity"]["ppl_target"] for v in order]
    bleu = [evals[v]["bleu_beam"]["all"]["bleu"] for v in order]
    total = [evals[v]["params_total"] / 1e6 for v in order]
    active = [evals[v]["params_active"] / 1e6 for v in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    x = np.arange(len(order))

    b = axes[0].bar(x, ppl, color=[color(i) for i in range(len(order))], width=0.62)
    annotate_bars(axes[0], b, ppl, "{:.1f}")
    axes[0].set_title("Test perplexity (target tokens)")
    axes[0].set_ylabel("perplexity (lower is better)")

    b = axes[1].bar(x, bleu, color=[color(i) for i in range(len(order))], width=0.62)
    annotate_bars(axes[1], b, bleu, "{:.2f}")
    axes[1].set_title("Test BLEU (beam-4)")
    axes[1].set_ylabel("BLEU (higher is better)")

    w = 0.36
    b1 = axes[2].bar(x - w / 2, total, width=w, color="#5A6672", label="total")
    b2 = axes[2].bar(x + w / 2, active, width=w, color=color(0), label="active / token")
    annotate_bars(axes[2], b1, total, "{:.1f}")
    annotate_bars(axes[2], b2, active, "{:.1f}")
    axes[2].set_title("Parameters (millions)")
    axes[2].set_ylabel("millions")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_axisbelow(True)
    fig.suptitle("Part 1 - translation quality against the parameter budget",
                 y=1.04, fontsize=11)
    save(fig, out / "part1_quality_vs_params.png")


def plot_per_language(evals: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    order = [v for v in VARIANTS if v in evals]
    labels = [SHORT[v] for v in order]
    x = np.arange(len(order))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    vi = [evals[v]["bleu_beam"]["vi-en"]["bleu"] for v in order]
    ja = [evals[v]["bleu_beam"]["ja-en"]["bleu"] for v in order]
    b1 = axes[0].bar(x - w / 2, vi, width=w, color=color(0), label="vi -> en")
    b2 = axes[0].bar(x + w / 2, ja, width=w, color=color(1), label="ja -> en")
    annotate_bars(axes[0], b1, vi, "{:.1f}")
    annotate_bars(axes[0], b2, ja, "{:.1f}")
    axes[0].set_title("BLEU by translation direction")
    axes[0].set_ylabel("BLEU")
    axes[0].legend()

    vi = [evals[v]["perplexity"]["per_lang"]["vi-en"]["ppl_target"] for v in order]
    ja = [evals[v]["perplexity"]["per_lang"]["ja-en"]["ppl_target"] for v in order]
    b1 = axes[1].bar(x - w / 2, vi, width=w, color=color(0), label="vi -> en")
    b2 = axes[1].bar(x + w / 2, ja, width=w, color=color(1), label="ja -> en")
    annotate_bars(axes[1], b1, vi, "{:.0f}")
    annotate_bars(axes[1], b2, ja, "{:.0f}")
    axes[1].set_title("Perplexity by translation direction")
    axes[1].set_ylabel("perplexity")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_axisbelow(True)
    save(fig, out / "part1_per_language.png")


def plot_expert_heatmaps(evals: dict, out: Path) -> None:
    """One panel per MoE variant: expert usage share per layer, per language."""
    import matplotlib.pyplot as plt

    moe = [v for v in VARIANTS if v in evals and evals[v].get("expert_usage")]
    if not moe:
        return

    for v in moe:
        u = evals[v]["expert_usage"]
        frac = np.asarray(u["fractions"])         # (L, E, C)
        cats = u["categories"]
        L, E, C = frac.shape

        fig, axes = plt.subplots(1, C, figsize=(3.0 * C + 1.2, 0.42 * L + 2.2),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for c in range(C):
            ax = axes[c]
            im = ax.imshow(frac[:, :, c], cmap="magma", vmin=0.0,
                           vmax=max(0.5, float(frac.max())), aspect="auto")
            ax.set_title(cats[c], fontsize=9)
            ax.set_xticks(range(E))
            ax.set_xticklabels([f"E{e}" for e in range(E)])
            ax.set_xlabel("expert")
            ax.grid(False)
            for l in range(L):
                for e in range(E):
                    val = frac[l, e, c]
                    ax.text(e, l, f"{val*100:.0f}", ha="center", va="center",
                            fontsize=6.5, color="white" if val < 0.55 else "black")
        axes[0].set_yticks(range(L))
        axes[0].set_yticklabels([f"L{l}" for l in range(L)])
        axes[0].set_ylabel("layer")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="share of routed tokens")
        fig.suptitle(f"{VARIANT_LABELS[v]} - expert usage by token category "
                     f"(top-{u['top_k']} of {E} routed"
                     + (f", +{u['n_shared']} shared" if u["n_shared"] else "") + ")",
                     y=1.02, fontsize=10.5)
        save(fig, out / f"part1_experts_{v}.png")


def plot_language_specialisation(evals: dict, out: Path) -> None:
    """How differently are vi and ja source tokens routed, layer by layer?"""
    import matplotlib.pyplot as plt

    moe = [v for v in VARIANTS if v in evals and evals[v].get("expert_usage")]
    if not moe:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for i, v in enumerate(moe):
        u = evals[v]["expert_usage"]
        frac = np.asarray(u["fractions"])
        # Total variation distance between the vi-source and ja-source
        # expert distributions: 0 = identical routing, 1 = disjoint experts.
        tvd = 0.5 * np.abs(frac[:, :, 0] - frac[:, :, 1]).sum(axis=1)
        axes[0].plot(range(frac.shape[0]), tvd, marker="o", markersize=4,
                     color=color(i), label=SHORT[v])
        ent = np.asarray(u["entropy_bits"]) / u["max_entropy_bits"]
        axes[1].plot(range(frac.shape[0]), ent, marker="s", markersize=4,
                     color=color(i), label=SHORT[v])

    axes[0].set_title("Language specialisation of the router\n(vi vs ja source tokens)")
    axes[0].set_ylabel("total variation distance")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Load balance across experts")
    axes[1].set_ylabel("routing entropy / max entropy")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend()
    save(fig, out / "part1_expert_specialisation.png")


def plot_throughput(evals: dict, runs_dir: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    order = [v for v in VARIANTS if v in evals]
    tps, labels = [], []
    for v in order:
        s = runs_dir / v / "summary.json"
        if s.exists():
            tps.append(load_json(s)["tokens_per_s"])
            labels.append(SHORT[v])
    if not tps:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    b = ax.bar(range(len(tps)), tps, color=[color(i) for i in range(len(tps))], width=0.62)
    annotate_bars(ax, b, tps, "{:,.0f}")
    ax.set_xticks(range(len(tps)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("training tokens / second")
    ax.set_title("Training throughput on one RTX 3050 (4 GiB)")
    ax.set_axisbelow(True)
    save(fig, out / "part1_throughput.png")


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    use_style()
    ap = argparse.ArgumentParser(description="Generate Part 1 figures")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs) if args.runs else work_dir() / "part1" / "runs"
    out = Path(args.out) if args.out else repo_root() / "assets"
    evals, logs = _load(runs_dir)
    if not evals and not logs:
        raise SystemExit(f"no Part 1 results under {runs_dir}")

    if logs:
        plot_training_curves(logs, out)
    if evals:
        plot_quality_vs_params(evals, out)
        plot_per_language(evals, out)
        plot_expert_heatmaps(evals, out)
        plot_language_specialisation(evals, out)
        plot_throughput(evals, runs_dir, out)


if __name__ == "__main__":  # pragma: no cover
    main()
