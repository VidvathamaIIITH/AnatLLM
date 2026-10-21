"""Metrics, plots, logging helpers and benchmarking utilities.

Every metric required by Section 4 of the brief is implemented here from
scratch (no ``nltk`` / ``rouge_score`` / ``python-Levenshtein`` dependency):

* **Bit-Level Accuracy** -- fraction of matching bits.  If the target strings
  are literal ``0``/``1`` bit-strings, each character *is* a bit; otherwise the
  strings are UTF-8 encoded and compared bit-by-bit.  Length mismatches are
  penalised: the denominator is ``max(len(pred), len(ref))``.
* **Sequence Accuracy** -- exact string match.
* **Levenshtein Distance** -- character edit distance (raw mean + normalised).
* **BLEU** -- corpus BLEU-4 with brevity penalty and add-1 smoothing on n>1.
* **ROUGE** -- ROUGE-1 / ROUGE-2 / ROUGE-L F1, averaged over samples.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "set_seed", "count_parameters", "human_int", "save_json", "load_json",
    "levenshtein", "detect_binary_mode", "to_bits",
    "bit_accuracy", "sequence_accuracy", "levenshtein_stats",
    "corpus_bleu", "rouge_scores", "compute_metrics",
    "Timer", "GpuMemory", "WandbLogger",
    "plot_training_curves", "plot_ablation_bars", "plot_efficiency",
    "plot_attention_map", "results_markdown_table",
]


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def human_int(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else f"{int(n)}"
        n /= 1000.0
    return f"{n:.2f}T"


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# edit distance
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance (two-row DP, prefix/suffix trimmed)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # trim the common prefix / suffix -- they never contribute to the distance
    start = 0
    n_min = min(len(a), len(b))
    while start < n_min and a[start] == b[start]:
        start += 1
    end = 0
    while end < n_min - start and a[len(a) - 1 - end] == b[len(b) - 1 - end]:
        end += 1
    a = a[start : len(a) - end]
    b = b[start : len(b) - end]
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur[j] = ins if ins < dele else dele
            if sub < cur[j]:
                cur[j] = sub
        prev, cur = cur, prev
    return prev[len(b)]


# --------------------------------------------------------------------------- #
# bit-level metrics
# --------------------------------------------------------------------------- #
_BINARY_RE = re.compile(r"^[01\s]+$")


def detect_binary_mode(refs: Sequence[str], sample: int = 256) -> bool:
    """True when the references are literal bit-strings (so 1 char == 1 bit)."""
    checked = 0
    for r in refs:
        if not r:
            continue
        if not _BINARY_RE.match(r):
            return False
        checked += 1
        if checked >= sample:
            break
    return checked > 0


def to_bits(s: str, binary_mode: bool) -> List[int]:
    if binary_mode:
        return [1 if c == "1" else 0 for c in s if c in "01"]
    out: List[int] = []
    for byte in s.encode("utf-8", errors="replace"):
        for k in range(7, -1, -1):
            out.append((byte >> k) & 1)
    return out


def bit_accuracy(preds: Sequence[str], refs: Sequence[str], binary_mode: bool) -> float:
    """Matching bits / max(len_pred, len_ref) bits, aggregated over the corpus."""
    matched = 0
    total = 0
    for p, r in zip(preds, refs):
        bp = to_bits(p, binary_mode)
        br = to_bits(r, binary_mode)
        n = min(len(bp), len(br))
        matched += sum(1 for i in range(n) if bp[i] == br[i])
        total += max(len(bp), len(br))
    return 100.0 * matched / total if total else 0.0


def sequence_accuracy(preds: Sequence[str], refs: Sequence[str]) -> float:
    if not refs:
        return 0.0
    return 100.0 * sum(1 for p, r in zip(preds, refs) if p == r) / len(refs)


def levenshtein_stats(preds: Sequence[str], refs: Sequence[str]) -> Tuple[float, float]:
    """Returns ``(mean_distance, mean_normalised_distance)``."""
    if not refs:
        return 0.0, 0.0
    dists, norms = [], []
    for p, r in zip(preds, refs):
        d = levenshtein(p, r)
        dists.append(d)
        denom = max(len(p), len(r), 1)
        norms.append(d / denom)
    return float(np.mean(dists)), float(np.mean(norms))


# --------------------------------------------------------------------------- #
# n-gram metrics
# --------------------------------------------------------------------------- #
def _tokenize_for_ngrams(s: str, word_level: bool) -> List[str]:
    return s.split() if word_level else list(s)


def _use_word_level(refs: Sequence[str], sample: int = 256) -> bool:
    for i, r in enumerate(refs):
        if i >= sample:
            break
        if " " in r or "\t" in r:
            return True
    return False


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    preds: Sequence[str], refs: Sequence[str], max_n: int = 4, word_level: Optional[bool] = None
) -> float:
    """Corpus BLEU-``max_n`` (%) with brevity penalty and add-1 smoothing for n>1."""
    if not refs:
        return 0.0
    if word_level is None:
        word_level = _use_word_level(refs)
    clipped = [0] * max_n
    total = [0] * max_n
    pred_len = ref_len = 0
    for p, r in zip(preds, refs):
        pt = _tokenize_for_ngrams(p, word_level)
        rt = _tokenize_for_ngrams(r, word_level)
        pred_len += len(pt)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            pc = _ngrams(pt, n)
            rc = _ngrams(rt, n)
            total[n - 1] += max(0, len(pt) - n + 1)
            clipped[n - 1] += sum(min(c, rc[g]) for g, c in pc.items())

    precisions = []
    for n in range(max_n):
        if total[n] == 0:
            precisions.append(0.0)
        elif n == 0:
            precisions.append(clipped[n] / total[n])
        else:  # add-1 smoothing (Chen & Cherry method 2) for higher-order n-grams
            precisions.append((clipped[n] + 1.0) / (total[n] + 1.0))
    if any(p <= 0.0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))
    return 100.0 * bp * math.exp(log_avg)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[len(b)]


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if overlap == 0 or n_pred == 0 or n_ref == 0:
        return 0.0
    p = overlap / n_pred
    r = overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_scores(
    preds: Sequence[str], refs: Sequence[str], word_level: Optional[bool] = None
) -> Dict[str, float]:
    """Sample-averaged ROUGE-1 / ROUGE-2 / ROUGE-L F1 (%)."""
    if not refs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if word_level is None:
        word_level = _use_word_level(refs)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        pt = _tokenize_for_ngrams(p, word_level)
        rt = _tokenize_for_ngrams(r, word_level)
        for n, bucket in ((1, r1), (2, r2)):
            pc, rc = _ngrams(pt, n), _ngrams(rt, n)
            overlap = sum(min(c, rc[g]) for g, c in pc.items())
            bucket.append(_f1(overlap, max(0, len(pt) - n + 1), max(0, len(rt) - n + 1)))
        rl.append(_f1(_lcs_length(pt, rt), len(pt), len(rt)))
    return {
        "rouge1": 100.0 * float(np.mean(r1)),
        "rouge2": 100.0 * float(np.mean(r2)),
        "rougeL": 100.0 * float(np.mean(rl)),
    }


def compute_metrics(
    preds: Sequence[str], refs: Sequence[str], compute_ngram: bool = True
) -> Dict[str, float]:
    """All Section-4 metrics in one call."""
    binary_mode = detect_binary_mode(refs)
    word_level = _use_word_level(refs)
    # Levenshtein is by far the most expensive metric here, so it is computed
    # once and reused for the raw, normalised and character-accuracy variants.
    dists = [levenshtein(p, r) for p, r in zip(preds, refs)]
    denoms = [max(len(p), len(r), 1) for p, r in zip(preds, refs)]
    norms = [d / n for d, n in zip(dists, denoms)]
    out = {
        "bit_accuracy": bit_accuracy(preds, refs, binary_mode),
        "sequence_accuracy": sequence_accuracy(preds, refs),
        "levenshtein": float(np.mean(dists)) if dists else 0.0,
        "levenshtein_normalized": float(np.mean(norms)) if norms else 0.0,
        "char_accuracy": 100.0 * float(np.mean([1.0 - min(1.0, x) for x in norms]))
        if norms else 0.0,
        "_binary_mode": float(binary_mode),
        "_word_level_ngrams": float(word_level),
    }
    if compute_ngram:
        out["bleu"] = corpus_bleu(preds, refs, word_level=word_level)
        out.update(rouge_scores(preds, refs, word_level=word_level))
    return out


def results_markdown_table(results: Dict[str, Dict[str, float]], columns: Sequence[str],
                           headers: Optional[Sequence[str]] = None) -> str:
    """Render ``{config: {metric: value}}`` as a GitHub-flavoured markdown table."""
    headers = list(headers or columns)
    lines = ["| Config | " + " | ".join(headers) + " |",
             "|" + "---|" * (len(headers) + 1)]
    for name in sorted(results):
        row = [name]
        for c in columns:
            v = results[name].get(c)
            row.append("n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# benchmarking
# --------------------------------------------------------------------------- #
class Timer:
    """Wall-clock timer that also syncs CUDA so GPU work is actually counted."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.elapsed = 0.0
        self._t0 = None

    def _sync(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.elapsed = time.perf_counter() - self._t0
        return False


class GpuMemory:
    """Peak-memory probe.  Returns 0 on CPU-only machines (and says so)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.available = device.type == "cuda"

    def reset(self):
        if self.available:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

    def peak_mb(self) -> float:
        if not self.available:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def peak_reserved_mb(self) -> float:
        if not self.available:
            return 0.0
        return torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class WandbLogger:
    """Weights & Biases wrapper that degrades gracefully.

    ``mode`` is one of ``online`` / ``offline`` / ``disabled``.  Whatever
    happens, every logged dict is also appended to ``<out_dir>/metrics_log.jsonl``
    so the run is never lost because of a network hiccup.
    """

    def __init__(
        self,
        mode: str = "online",
        project: str = "anlp-a1-transformers",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        out_dir: str = "outputs",
        tags: Optional[Sequence[str]] = None,
    ):
        self.mode = mode
        self.run = None
        self.wandb = None
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, "metrics_log.jsonl")
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")

        if mode == "disabled":
            print("[wandb] disabled (--wandb disabled)", flush=True)
            return
        try:
            import wandb  # noqa: WPS433

            self.wandb = wandb
            if mode == "online" and not (
                os.environ.get("WANDB_API_KEY") or os.path.exists(
                    os.path.expanduser("~/.netrc")
                )
            ):
                print("[wandb] no API key found -> falling back to offline mode", flush=True)
                mode = "offline"
            self.run = wandb.init(
                project=project, entity=entity, name=name, config=config or {},
                mode=mode, tags=list(tags or []), reinit=True,
            )
            self.mode = mode
            print(f"[wandb] logging {mode}: {getattr(self.run, 'url', '(offline)')}", flush=True)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[wandb] unavailable ({exc}); logging to {self.jsonl_path} only", flush=True)
            self.run = None

    def log(self, data: dict, step: Optional[int] = None):
        payload = {k: v for k, v in data.items() if not k.startswith("_")}
        rec = dict(payload)
        if step is not None:
            rec["step"] = step
        self._fh.write(json.dumps(rec, default=float) + "\n")
        self._fh.flush()
        if self.run is not None:
            try:
                self.run.log(payload, step=step)
            except Exception:
                pass

    def summary(self, data: dict):
        self._fh.write(json.dumps({"summary": data}, default=float) + "\n")
        self._fh.flush()
        if self.run is not None:
            try:
                for k, v in data.items():
                    self.run.summary[k] = v
            except Exception:
                pass

    def url(self) -> str:
        try:
            return self.run.url if self.run is not None else ""
        except Exception:
            return ""

    def finish(self):
        try:
            self._fh.close()
        except Exception:
            pass
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(history: Dict[str, List[float]], path: str, title: str = "") -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = history.get("epoch") or list(range(1, len(history.get("train_loss", [])) + 1))

    axes[0].plot(epochs, history.get("train_loss", []), marker="o", label="train")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], marker="s", label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    # train uses label smoothing + dropout, val uses neither -- so train can sit
    # *above* val without that meaning anything is wrong.
    axes[0].text(
        0.5, -0.28, "train: with label smoothing & dropout   |   val: without",
        transform=axes[0].transAxes, ha="center", va="top", fontsize=7.5, color="0.35",
    )

    if history.get("val_ppl"):
        axes[1].plot(epochs, history["val_ppl"], marker="s", color="tab:red", label="val ppl")
    if history.get("val_token_acc"):
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["val_token_acc"], marker="^", color="tab:green",
                 label="val token acc (%)")
        ax2.set_ylabel("token accuracy (%)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("perplexity")
    axes[1].set_title("Validation"); axes[1].grid(alpha=0.3)

    fig.suptitle(title or "Training curves")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation_bars(
    results: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
    path: str,
    title: str = "Ablation results",
    pretty: Optional[Dict[str, str]] = None,
) -> None:
    plt = _plt()
    names = sorted(results)
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    for ax, metric in zip(axes[0], metrics):
        vals = [results[k].get(metric, float("nan")) for k in names]
        bars = ax.bar(names, vals, color=colors[: len(names)])
        ax.set_title((pretty or {}).get(metric, metric))
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_efficiency(results: Dict[str, Dict[str, float]], path: str) -> None:
    """Params / time-per-epoch / peak memory -- the C1-vs-C5 benchmarking plot."""
    plt = _plt()
    names = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    series = [
        ("params_millions", "Parameters (M)", "#4C72B0"),
        ("sec_per_epoch", "Train time / epoch (s)", "#DD8452"),
        ("peak_gpu_mem_mb", "Peak GPU memory (MB)", "#55A868"),
    ]
    for ax, (key, label, color) in zip(axes, series):
        vals = [results[k].get(key, 0.0) for k in names]
        bars = ax.bar(names, vals, color=color)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Computational cost of each configuration")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_map(weights: np.ndarray, path: str, title: str = "Attention") -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_xlabel("key position (source)")
    ax.set_ylabel("query position (target)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
