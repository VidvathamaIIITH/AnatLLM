"""Collect every experimental result into ready-to-use tables.

Writes ``report_data/`` containing:

* ``metrics.json``  - every number produced by the three parts, in one place
* ``tables.md``     - the same numbers as Markdown tables
* ``tables.tex``    - the same numbers as LaTeX ``tabular`` environments
* ``figures.tex``   - ``\\includegraphics`` blocks for every figure in assets/

This module only *formats* results; it contains no analysis or prose.

    python -m src.report_data
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .part1.model import VARIANT_LABELS, VARIANTS
from .part2.optimizers import OPTIMIZER_CATEGORY, OPTIMIZER_LABEL
from .utils import configure_stdout, load_json, repo_root, work_dir

P2_ORDER = ["adamw", "mars", "lion", "muon", "sophia"]


# --------------------------------------------------------------------------- #
# Table rendering
# --------------------------------------------------------------------------- #


class Table:
    def __init__(self, title: str, header: Sequence[str], rows: Sequence[Sequence[Any]],
                 label: str, note: str = ""):
        self.title = title
        self.header = list(header)
        self.rows = [list(r) for r in rows]
        self.label = label
        self.note = note

    def to_markdown(self) -> str:
        out = [f"### {self.title}", ""]
        if self.note:
            out += [self.note, ""]
        out.append("| " + " | ".join(self.header) + " |")
        out.append("|" + "|".join("---" for _ in self.header) + "|")
        for r in self.rows:
            out.append("| " + " | ".join(str(c) for c in r) + " |")
        out.append("")
        return "\n".join(out)

    def to_latex(self) -> str:
        def esc(s: Any) -> str:
            s = str(s)
            for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                         ("&", r"\&"), ("#", r"\#")):
                s = s.replace(a, b)
            return s

        cols = "l" + "r" * (len(self.header) - 1)
        lines = [r"\begin{table}[htbp]", r"  \centering",
                 f"  \\caption{{{esc(self.title)}}}",
                 f"  \\label{{tab:{self.label}}}",
                 f"  \\begin{{tabular}}{{{cols}}}", r"    \toprule",
                 "    " + " & ".join(esc(h) for h in self.header) + r" \\",
                 r"    \midrule"]
        for r in self.rows:
            lines.append("    " + " & ".join(esc(c) for c in r) + r" \\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
        return "\n".join(lines)


def fmt(v: Any, spec: str = ".2f") -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and v != v:      # NaN
        return "-"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Part 1
# --------------------------------------------------------------------------- #


def part1_tables(runs: Path) -> tuple[list[Table], dict]:
    evals, summaries = {}, {}
    for v in VARIANTS:
        if (runs / v / "eval.json").exists():
            evals[v] = load_json(runs / v / "eval.json")
        if (runs / v / "summary.json").exists():
            summaries[v] = load_json(runs / v / "summary.json")
    if not evals:
        return [], {}

    rows = []
    for v in VARIANTS:
        if v not in evals:
            continue
        e = evals[v]
        s = summaries.get(v, {})
        rows.append([
            VARIANT_LABELS[v],
            fmt(e["params_total"] / 1e6), fmt(e["params_active"] / 1e6),
            fmt(e["perplexity"]["ppl_all"]), fmt(e["perplexity"]["ppl_target"]),
            fmt(e["bleu_greedy"]["all"]["bleu"]), fmt(e["bleu_beam"]["all"]["bleu"]),
            fmt(e["bleu_beam"]["all"]["chrf2"]),
            fmt(s.get("tokens_per_s"), ",.0f"),
            fmt(s.get("train_seconds", 0) / 60, ".1f"),
        ])
    main = Table(
        "Part 1: FFN ablations (identical 30M-token budget)",
        ["Variant", "Total (M)", "Active (M)", "PPL all", "PPL target",
         "BLEU greedy", "BLEU beam-4", "chrF2", "tok/s", "train (min)"],
        rows, "part1_main",
        note="PPL target = perplexity over the English target tokens only.",
    )

    lang_rows = []
    for v in VARIANTS:
        if v not in evals:
            continue
        e = evals[v]
        lang_rows.append([
            VARIANT_LABELS[v],
            fmt(e["perplexity"]["per_lang"]["vi-en"]["ppl_target"]),
            fmt(e["bleu_beam"]["vi-en"]["bleu"]),
            fmt(e["perplexity"]["per_lang"]["ja-en"]["ppl_target"]),
            fmt(e["bleu_beam"]["ja-en"]["bleu"]),
        ])
    lang = Table("Part 1: results by translation direction",
                 ["Variant", "vi-en PPL", "vi-en BLEU", "ja-en PPL", "ja-en BLEU"],
                 lang_rows, "part1_lang")

    usage_rows = []
    for v in VARIANTS:
        u = evals.get(v, {}).get("expert_usage")
        if not u:
            continue
        import numpy as np

        frac = np.asarray(u["fractions"])
        tvd = 0.5 * np.abs(frac[:, :, 0] - frac[:, :, 1]).sum(axis=1)
        ent = np.asarray(u["entropy_bits"]) / u["max_entropy_bits"]
        share = np.asarray(u["per_layer_share"])
        usage_rows.append([
            VARIANT_LABELS[v], f"{u['top_k']}/{u['n_experts']}", u["n_shared"],
            fmt(float(ent.mean()), ".3f"), fmt(float(ent.min()), ".3f"),
            fmt(float(tvd.mean()), ".3f"), fmt(float(tvd.max()), ".3f"),
            fmt(float(share.max()), ".3f"),
        ])
    usage = Table(
        "Part 1: routing statistics (test set)",
        ["Variant", "top-k / experts", "shared", "mean bal.", "min bal.",
         "mean TVD(vi,ja)", "max TVD(vi,ja)", "max expert share"],
        usage_rows, "part1_routing",
        note=("bal. = routing entropy / log2(E), 1.0 is a perfectly even load. "
              "TVD = total variation distance between the expert distributions "
              "of Vietnamese and Japanese source tokens, per layer."),
    )
    return [main, lang, usage], {"evals": evals, "summaries": summaries}


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #


def part2_tables(runs: Path, sweep_dir: Path) -> tuple[list[Table], dict]:
    summaries = {n: load_json(runs / n / "summary.json")
                 for n in P2_ORDER if (runs / n / "summary.json").exists()}
    if not summaries:
        return [], {}

    rows = []
    for n, s in summaries.items():
        rows.append([
            OPTIMIZER_LABEL[n], OPTIMIZER_CATEGORY[n],
            fmt(s["hyperparams"].get("lr"), ".2e"),
            fmt(s["final_val_loss"], ".4f"),
            fmt(math.exp(min(s["final_val_loss"], 50)), ".2f"),
            fmt(s["final_test_bleu"]),
            fmt(s["optimizer_state_ratio"], ".2f") + "x",
            fmt(s["tokens_per_s"], ",.0f"),
            fmt(s["train_seconds"] / 60, ".1f"),
        ])
    main = Table(
        "Part 2: optimizers at 1.0x the dataset",
        ["Optimizer", "Category", "LR", "Val loss", "Val PPL", "Test BLEU",
         "State", "tok/s", "train (min)"],
        rows, "part2_main",
        note=("State = optimizer state size as a multiple of the fp32 model. "
              "All runs see the same data in the same order."),
    )

    any_s = next(iter(summaries.values()))
    fractions = [c["fraction"] for c in any_s["curve"]]
    curve_rows = []
    for n, s in summaries.items():
        by_frac = {c["fraction"]: c for c in s["curve"]}
        curve_rows.append([OPTIMIZER_LABEL[n].split(" (")[0]] +
                          [fmt(by_frac[f]["val_loss"], ".4f") if f in by_frac else "-"
                           for f in fractions])
    curve = Table(
        "Part 2: validation loss against the fraction of the dataset consumed",
        ["Optimizer"] + [f"{f:g}x" for f in fractions],
        curve_rows, "part2_curve",
        note=f"1.0x = {any_s['dataset_tokens']:,} tokens = "
             f"{any_s['dataset_tokens']/any_s['chinchilla_tokens']:.3f}x Chinchilla "
             f"for this {any_s['params']/1e6:.2f}M-parameter model.",
    )

    bleu_rows = []
    for n, s in summaries.items():
        by_frac = {c["fraction"]: c for c in s["curve"]}
        bleu_rows.append([OPTIMIZER_LABEL[n].split(" (")[0]] +
                         [fmt(by_frac[f]["test_bleu"]) if f in by_frac else "-"
                          for f in fractions])
    bleu = Table("Part 2: test BLEU against the fraction of the dataset consumed",
                 ["Optimizer"] + [f"{f:g}x" for f in fractions],
                 bleu_rows, "part2_bleu")

    tables = [main, curve, bleu]

    spath = sweep_dir / "sweep_results.json"
    sweep_data = {}
    if spath.exists():
        sweep_data = load_json(spath)
        srows = []
        for n, r in sweep_data.items():
            grid = "  ".join(f"{t['lr']:g}: {t['val_loss']:.4f}" for t in r["trials"])
            srows.append([OPTIMIZER_LABEL.get(n, n), fmt(r["best_lr"], ".2e"),
                          fmt(r["best_val_loss"], ".4f"), grid])
        tables.append(Table(
            f"Part 2 (bonus): learning-rate sweep at "
            f"{next(iter(sweep_data.values()))['fraction']:g}x the dataset",
            ["Optimizer", "Selected LR", "Val loss", "Grid (lr: val loss)"],
            srows, "part2_sweep"))

    return tables, {"summaries": summaries, "sweep": sweep_data}


# --------------------------------------------------------------------------- #
# Part 3
# --------------------------------------------------------------------------- #


def part3_tables(base: Path) -> tuple[list[Table], dict]:
    rpath = base / "results.json"
    if not rpath.exists():
        return [], {}
    payload = load_json(rpath)
    res = payload["results"]

    rows = []
    for m in res:
        rows.append([
            m["label"], fmt(m["accuracy"], ".4f"), fmt(m["precision"], ".4f"),
            fmt(m["recall"], ".4f"), fmt(m["f1"], ".4f"),
            fmt(m["gen_perplexity"]), fmt(m["bleu"]), fmt(m["chrf2"]),
            fmt(m["wall_seconds"], ".1f"),
        ])
    main = Table(
        "Part 3: decoding strategies on ROCStories (Pythia-160M, "
        f"{payload['n_examples']} stories)",
        ["Strategy", "Accuracy", "Precision", "Recall", "F1",
         "PPL (own output)", "BLEU", "chrF2", "seconds"],
        rows, "part3_main",
        note=("Precision/recall/F1 are unigram multiset overlap with the reference "
              "continuation; accuracy is positional token agreement. Reference-"
              f"continuation perplexity (strategy independent) is "
              f"{payload['reference_perplexity']['ppl']:.2f}."),
    )

    div_rows = []
    for m in res:
        div_rows.append([
            m["label"], fmt(m["distinct_1"], ".4f"), fmt(m["distinct_2"], ".4f"),
            fmt(m["repetition_4gram"], ".4f"), fmt(m["mean_hyp_words"], ".1f"),
            fmt(m["mean_gen_tokens"], ".1f"), fmt(m["f1_macro"], ".4f"),
        ])
    div = Table("Part 3: diversity and length by strategy",
                ["Strategy", "distinct-1", "distinct-2", "repeated 4-grams",
                 "words/gen", "tokens/gen", "macro F1"],
                div_rows, "part3_diversity")

    tables = [main, div]
    timing = {}
    tpath = base / "beam_timing.json"
    if tpath.exists():
        timing = load_json(tpath)
        trows = []
        for r in timing["rows"]:
            trows.append([
                r["beam_width"], fmt(r["seconds_best"], ".2f"),
                fmt(r["seconds_per_example"] * 1000, ".1f"),
                fmt(r["speed_ratio_vs_width1"], ".2f") + "x",
                fmt(r["ratio_over_linear"], ".2f"),
                fmt(r["peak_mib"], ".0f"),
            ])
        tables.append(Table(
            f"Part 3: beam-search cost vs beam width "
            f"({timing['rows'][0]['n_examples']} prompts, batch {timing['batch_size']})",
            ["Beam width K", "seconds", "ms/example", "vs K=1", "measured/linear",
             "peak MiB"],
            trows, "part3_timing",
            note="A ratio of 1.00 in 'measured/linear' would mean cost exactly "
                 "proportional to K."))

    return tables, {"payload": payload, "timing": timing}


# --------------------------------------------------------------------------- #
# Stand-alone LaTeX fragments that report.tex \input directly
# --------------------------------------------------------------------------- #


def write_benchmark_table(path: Path) -> bool:
    """Controlled optimizer-overhead table (src/part2/benchmark.py)."""
    src = work_dir() / "part2" / "benchmark.json"
    if not src.exists():
        return False
    d = load_json(src)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Controlled optimizer overhead: " + str(d["rounds"]) +
        r" interleaved rounds of " + str(d["steps"]) + r" timed steps each, best "
        r"round reported. The round-to-round spread is shown because it is the "
        r"same size as most of the between-optimizer differences.}",
        r"\label{tab:p2-bench}", r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrl}", r"\toprule",
        r"Optimizer & tok/s & ms/step & vs AdamW & per-round tok/s \\", r"\midrule",
    ]
    for r in d["rows"]:
        rounds = ", ".join(f"{x:,.0f}" for x in r["all_rounds_tokens_per_s"])
        name = r["label"].split(" (")[0]
        lines.append(f"{name} & {r['tokens_per_s']:,.0f} & {r['ms_per_step']:.1f} & "
                     f"{r['relative_to_adamw']:.2f}$\\times$ & {{\\scriptsize {rounds}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def write_part3_tables(path: Path) -> bool:
    """The two Part 3 result tables."""
    src = work_dir() / "part3" / "results.json"
    if not src.exists():
        return False
    p = load_json(src)
    res = p["results"]
    ref = p["reference_perplexity"]["ppl"]

    def best_of(key: str, lo: bool = False):
        vals = [m[key] for m in res]
        return min(vals) if lo else max(vals)

    hi_f1, hi_acc = best_of("f1"), best_of("accuracy")
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Decoding strategies on ROCStories (Pythia-160M, " +
        f"{p['n_examples']}" + r" stories). Precision/recall/F1 are unigram "
        r"multiset overlap with the reference; accuracy is positional token "
        r"agreement. Perplexity of the \emph{gold} continuation, which does not "
        r"depend on the strategy, is " + f"{ref:.2f}" + r".}",
        r"\label{tab:p3-main}", r"\small",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Strategy & PPL(own) & Accuracy & Precision & Recall & F1 & BLEU & chrF2 \\",
        r"\midrule",
    ]
    for m in res:
        f1 = (r"\best{" + f"{m['f1']:.4f}" + "}") if m["f1"] == hi_f1 else f"{m['f1']:.4f}"
        acc = ((r"\best{" + f"{m['accuracy']:.4f}" + "}")
               if m["accuracy"] == hi_acc else f"{m['accuracy']:.4f}")
        lines.append(f"{_tex(m['label'])} & {m['gen_perplexity']:.2f} & {acc} & "
                     f"{m['precision']:.4f} & {m['recall']:.4f} & {f1} & "
                     f"{m['bleu']:.2f} & {m['chrf2']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

    # The diversity table goes to its own file so the report can include it or not.
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Diversity and length. ``Repeated 4-grams'' is the share of "
        r"4-grams that recur \emph{within the same generation} -- the signature of "
        r"degenerate decoding.}",
        r"\label{tab:p3-div}", r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Strategy & distinct-1 & distinct-2 & repeated 4-grams & words/gen & macro F1 \\",
        r"\midrule",
    ]
    for m in res:
        lines.append(f"{_tex(m['label'])} & {m['distinct_1']:.4f} & {m['distinct_2']:.4f} & "
                     f"{m['repetition_4gram']:.4f} & {m['mean_hyp_words']:.1f} & "
                     f"{m['f1_macro']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    path.with_name("part3_diversity_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return True


def _tex(s: str) -> str:
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Collect results into report tables")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else repo_root() / "report_data"
    out.mkdir(parents=True, exist_ok=True)

    tables: list[Table] = []
    metrics: dict[str, Any] = {}

    t, m = part1_tables(work_dir() / "part1" / "runs")
    tables += t
    metrics["part1"] = m
    t, m = part2_tables(work_dir() / "part2" / "runs", work_dir() / "part2" / "sweep")
    tables += t
    metrics["part2"] = m
    t, m = part3_tables(work_dir() / "part3")
    tables += t
    metrics["part3"] = m

    if not tables:
        raise SystemExit("no results found - run the experiments first")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (out / "tables.md").write_text(
        "# Assignment 2 - result tables\n\n"
        "Generated by `python -m src.report_data`. Numbers only; no analysis.\n\n"
        + "\n".join(t.to_markdown() for t in tables), encoding="utf-8")
    (out / "tables.tex").write_text(
        "% Generated by `python -m src.report_data`.\n"
        "% Requires \\usepackage{booktabs} in the preamble.\n\n"
        + "\n".join(t.to_latex() for t in tables), encoding="utf-8")

    assets = repo_root() / "assets"
    figs = sorted(assets.glob("*.png")) if assets.exists() else []
    lines = ["% Generated by `python -m src.report_data`.",
             "% Requires \\usepackage{graphicx} in the preamble.", ""]
    for f in figs:
        lines += [r"\begin{figure}[htbp]", r"  \centering",
                  f"  \\includegraphics[width=\\linewidth]{{assets/{f.name}}}",
                  f"  \\caption{{{f.stem.replace('_', ' ')}}}",
                  f"  \\label{{fig:{f.stem}}}", r"\end{figure}", ""]
    (out / "figures.tex").write_text("\n".join(lines), encoding="utf-8")

    # Fragments that report.tex \inputs directly.
    if write_benchmark_table(out / "benchmark_table.tex"):
        print("[report_data] wrote benchmark_table.tex")
    if write_part3_tables(out / "part3_tables.tex"):
        print("[report_data] wrote part3_tables.tex")

    print(f"[report_data] {len(tables)} tables, {len(figs)} figures -> {out}")
    for t in tables:
        print(f"  - {t.title}")


if __name__ == "__main__":  # pragma: no cover
    main()
