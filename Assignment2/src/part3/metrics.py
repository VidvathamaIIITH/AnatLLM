"""Metrics for Part 3.

Open-ended generation has no single "correct" output, so accuracy / precision /
recall / F1 have to be given a precise operational meaning.  The definitions
used throughout this assignment are stated here once and applied identically to
every decoding strategy:

* **Precision** - of the word tokens the model produced, what share also occurs
  in the reference continuation (counted as a *multiset* intersection, so
  repeating a word three times only pays off if the reference repeats it too).
* **Recall** - of the reference's word tokens, what share the model produced.
* **F1** - the harmonic mean of the two.  This is exactly unigram-overlap F1,
  i.e. ROUGE-1 F.
* **Accuracy** - positional token accuracy: aligning hypothesis and reference at
  position 0 and comparing token by token, the share of positions that agree,
  normalised by the *longer* of the two so that neither truncation nor padding
  can inflate it.

Both micro (pool every token in the corpus) and macro (average the per-story
score) variants are reported, since they answer slightly different questions.

Two perplexities are reported as well:

* **reference perplexity** - the model's perplexity on the *gold* continuation.
  It does not depend on the decoding strategy and acts as a fixed anchor.
* **generation perplexity** - the model's perplexity on *its own* output,
  ``exp(-sum log p / n_tokens)``.  This one does vary by strategy and measures
  how "safe" the sampled text is.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _WORD_RE.findall(text.lower())


def prf_counts(hyp_tokens: Sequence[str], ref_tokens: Sequence[str]) -> tuple[int, int, int]:
    """Return ``(overlap, n_hyp, n_ref)`` using multiset intersection."""
    overlap = sum((Counter(hyp_tokens) & Counter(ref_tokens)).values())
    return overlap, len(hyp_tokens), len(ref_tokens)


def prf(overlap: int, n_hyp: int, n_ref: int) -> tuple[float, float, float]:
    p = overlap / n_hyp if n_hyp else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def positional_accuracy(hyp_tokens: Sequence[str], ref_tokens: Sequence[str]) -> tuple[int, int]:
    """Return ``(matches, denominator)`` for positional token accuracy."""
    n = max(len(hyp_tokens), len(ref_tokens))
    if n == 0:
        return 0, 0
    matches = sum(1 for a, b in zip(hyp_tokens, ref_tokens) if a == b)
    return matches, n


def distinct_n(token_lists: Sequence[Sequence[str]], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams (diversity)."""
    total = 0
    seen = set()
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i:i + n]))
            total += 1
    return len(seen) / total if total else 0.0


def repetition_rate(token_lists: Sequence[Sequence[str]], n: int = 4) -> float:
    """Share of n-grams that are repeated *within the same generation*.

    This is the quantity that makes greedy and low-temperature decoding look
    degenerate: it climbs sharply when the model falls into a loop.
    """
    rates = []
    for toks in token_lists:
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        repeated = sum(c - 1 for c in counts.values())
        rates.append(repeated / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def evaluate_generations(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    """Compute every text-level metric for one decoding configuration."""
    import sacrebleu

    hyp_toks = [normalise(h) for h in hyps]
    ref_toks = [normalise(r) for r in refs]

    tot_overlap = tot_hyp = tot_ref = 0
    tot_match = tot_pos = 0
    macro_p, macro_r, macro_f, macro_acc = [], [], [], []

    for h, r in zip(hyp_toks, ref_toks):
        o, nh, nr = prf_counts(h, r)
        tot_overlap += o
        tot_hyp += nh
        tot_ref += nr
        p, rec, f = prf(o, nh, nr)
        macro_p.append(p)
        macro_r.append(rec)
        macro_f.append(f)
        m, d = positional_accuracy(h, r)
        tot_match += m
        tot_pos += d
        macro_acc.append(m / d if d else 0.0)

    micro_p, micro_r, micro_f = prf(tot_overlap, tot_hyp, tot_ref)
    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(list(hyps), [list(refs)])
    chrf = sacrebleu.corpus_chrf(list(hyps), [list(refs)])

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(hyps),
        "accuracy": tot_match / tot_pos if tot_pos else 0.0,
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f,
        "accuracy_macro": mean(macro_acc),
        "precision_macro": mean(macro_p),
        "recall_macro": mean(macro_r),
        "f1_macro": mean(macro_f),
        "bleu": float(bleu.score),
        "chrf2": float(chrf.score),
        "distinct_1": distinct_n(hyp_toks, 1),
        "distinct_2": distinct_n(hyp_toks, 2),
        "repetition_4gram": repetition_rate(hyp_toks, 4),
        "mean_hyp_words": mean([float(len(h)) for h in hyp_toks]),
        "mean_ref_words": mean([float(len(r)) for r in ref_toks]),
    }


def perplexity_from_logprobs(sum_logprob: Sequence[float], n_tokens: Sequence[int]) -> float:
    """Corpus perplexity from per-sequence total log-probabilities."""
    total_lp = float(sum(sum_logprob))
    total_n = int(sum(n_tokens))
    if total_n == 0:
        return float("nan")
    return math.exp(min(-total_lp / total_n, 50.0))
