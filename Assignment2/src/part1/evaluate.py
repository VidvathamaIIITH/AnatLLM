"""Evaluation for Part 1: perplexity, BLEU and expert-usage statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..part3.decoding import GenerationConfig, LocalBackend, generate, strip_after_eos
from ..tokenization import BPETokenizer
from .data import LANG_NAME, BinarisedSplit, build_batches, collate, prompts_for_generation
from .model import TransformerLM

#: token categories used for the expert-specialisation heatmaps
CATEGORIES = ("vi (source)", "ja (source)", "en (target | vi)", "en (target | ja)")


# --------------------------------------------------------------------------- #
# Perplexity
# --------------------------------------------------------------------------- #


@dataclass
class PerplexityResult:
    loss_all: float
    ppl_all: float
    loss_target: float
    ppl_target: float
    tokens_all: int
    tokens_target: int
    per_lang: dict[str, dict[str, float]]

    def as_dict(self) -> dict:
        return {
            "loss_all": self.loss_all, "ppl_all": self.ppl_all,
            "loss_target": self.loss_target, "ppl_target": self.ppl_target,
            "tokens_all": self.tokens_all, "tokens_target": self.tokens_target,
            "per_lang": self.per_lang,
        }


@torch.no_grad()
def evaluate_perplexity(
    model: TransformerLM,
    split: BinarisedSplit,
    pad_id: int,
    device: torch.device,
    max_tokens: int = 2048,
    max_examples: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
    seed: int = 0,
) -> PerplexityResult:
    """Token-weighted perplexity over (a deterministic subset of) a split.

    Two numbers are reported: ``all`` covers every predicted token (source side
    included, matching the training objective) while ``target`` covers only the
    English tokens the model is actually asked to produce - the latter is the
    translation perplexity quoted in the report.
    """
    was_training = model.training
    model.eval()

    indices = None
    if max_examples is not None and max_examples < len(split):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(split), size=max_examples, replace=False))

    nll_all = torch.zeros((), dtype=torch.float64)
    nll_tgt = torch.zeros((), dtype=torch.float64)
    n_all = 0
    n_tgt = 0
    per_lang_nll = {0: 0.0, 1: 0.0}
    per_lang_n = {0: 0, 1: 0}

    for batch_idx in build_batches(split.lengths, max_tokens, 256, indices):
        batch = collate(split, batch_idx, pad_id, device)
        ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with ctx:
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        B, T, V = logits.shape
        tok_nll = F.cross_entropy(
            logits.float().view(-1, V), batch["targets"].view(-1), reduction="none"
        ).view(B, T)

        loss_mask = batch["loss_mask"].bool()
        tgt_mask = batch["target_mask"].bool()
        nll_all += tok_nll[loss_mask].sum().double().cpu()
        n_all += int(loss_mask.sum())
        nll_tgt += tok_nll[tgt_mask].sum().double().cpu()
        n_tgt += int(tgt_mask.sum())

        for lang in (0, 1):
            rows = (batch["lang"] == lang).view(-1, 1)
            m = tgt_mask & rows
            per_lang_nll[lang] += float(tok_nll[m].sum().double().cpu())
            per_lang_n[lang] += int(m.sum())

    if was_training:
        model.train()

    loss_all = float(nll_all) / max(1, n_all)
    loss_tgt = float(nll_tgt) / max(1, n_tgt)
    per_lang = {}
    for lang in (0, 1):
        l = per_lang_nll[lang] / max(1, per_lang_n[lang])
        per_lang[LANG_NAME[lang]] = {
            "loss_target": l,
            "ppl_target": math.exp(min(l, 50.0)),
            "tokens_target": per_lang_n[lang],
        }

    return PerplexityResult(
        loss_all=loss_all, ppl_all=math.exp(min(loss_all, 50.0)),
        loss_target=loss_tgt, ppl_target=math.exp(min(loss_tgt, 50.0)),
        tokens_all=n_all, tokens_target=n_tgt, per_lang=per_lang,
    )


# --------------------------------------------------------------------------- #
# Translation + BLEU
# --------------------------------------------------------------------------- #


@torch.no_grad()
def translate(
    model: TransformerLM,
    split: BinarisedSplit,
    tok: BPETokenizer,
    indices: Sequence[int],
    device: torch.device,
    cfg: GenerationConfig,
    batch_size: int = 32,
    amp_dtype: Optional[torch.dtype] = None,
) -> tuple[list[str], list[str], list[int], float]:
    """Decode English hypotheses for the given examples.

    Prompts are sorted by length before batching so that a batch contains
    similarly sized prompts, which keeps left padding (and therefore wasted
    compute) small.  Returns ``(hypotheses, references, langs, seconds)``.
    """
    was_training = model.training
    model.eval()

    idx = np.asarray(indices, dtype=np.int64)
    order = idx[np.argsort(split.prompt_len[idx], kind="stable")]

    hyps: dict[int, str] = {}
    refs: dict[int, str] = {}
    langs: dict[int, int] = {}
    seconds = 0.0

    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        prompts, mask, ref_ids, lang_ids = prompts_for_generation(split, chunk, tok.pad_id)
        backend = LocalBackend(model)
        ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with ctx:
            res = generate(backend, prompts.to(device), mask.to(device), cfg)
        seconds += res.seconds
        for j, ex in enumerate(chunk):
            hyps[int(ex)] = tok.decode(strip_after_eos(res.tokens[j], cfg.eos_id))
            refs[int(ex)] = tok.decode(strip_after_eos(ref_ids[j], cfg.eos_id))
            langs[int(ex)] = lang_ids[j]

    if was_training:
        model.train()

    keys = [int(i) for i in idx]
    return ([hyps[k] for k in keys], [refs[k] for k in keys],
            [langs[k] for k in keys], seconds)


def corpus_bleu(hyps: Sequence[str], refs: Sequence[str]) -> dict[str, float]:
    """sacreBLEU corpus BLEU (13a tokenisation) plus chrF2 as a secondary view."""
    import sacrebleu

    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(list(hyps), [list(refs)])
    chrf = sacrebleu.corpus_chrf(list(hyps), [list(refs)])
    return {
        "bleu": float(bleu.score),
        "bleu_signature": str(metric.get_signature()),
        "precisions": [float(p) for p in bleu.precisions],
        "brevity_penalty": float(bleu.bp),
        "sys_len": int(bleu.sys_len),
        "ref_len": int(bleu.ref_len),
        "chrf2": float(chrf.score),
    }


def bleu_by_language(hyps: Sequence[str], refs: Sequence[str],
                     langs: Sequence[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {"all": corpus_bleu(hyps, refs)}
    for lang in (0, 1):
        sel = [i for i, l in enumerate(langs) if l == lang]
        if sel:
            out[LANG_NAME[lang]] = corpus_bleu([hyps[i] for i in sel], [refs[i] for i in sel])
    return out


# --------------------------------------------------------------------------- #
# Expert usage
# --------------------------------------------------------------------------- #


@torch.no_grad()
def expert_usage(
    model: TransformerLM,
    split: BinarisedSplit,
    pad_id: int,
    device: torch.device,
    max_examples: int = 8000,
    max_tokens: int = 2048,
    amp_dtype: Optional[torch.dtype] = None,
    seed: int = 0,
) -> dict:
    """Count how often each expert is selected, split by token category.

    Returns ``counts`` of shape ``(n_layers, n_routed_experts, n_categories)``
    where the categories are :data:`CATEGORIES`.  Only routed experts are
    counted; shared experts are by construction active for every token.
    """
    if not model.is_moe:
        return {}

    was_training = model.training
    model.eval()
    model.set_capture_routing(True)

    cfg = model.config
    L, E = cfg.n_layers, cfg.n_routed_experts
    counts = np.zeros((L, E, len(CATEGORIES)), dtype=np.int64)
    # Mean router probability mass, a softer view than the hard top-k counts.
    prob_mass = np.zeros((L, E, len(CATEGORIES)), dtype=np.float64)

    rng = np.random.default_rng(seed)
    if max_examples < len(split):
        indices = np.sort(rng.choice(len(split), size=max_examples, replace=False))
    else:
        indices = None

    for batch_idx in build_batches(split.lengths, max_tokens, 256, indices):
        batch = collate(split, batch_idx, pad_id, device)
        ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with ctx:
            out = model(batch["input_ids"], attention_mask=batch["attention_mask"])

        B, T = batch["input_ids"].shape
        pos = torch.arange(T, device=device).view(1, -1)
        plen = batch["prompt_len"].view(-1, 1)
        real = batch["attention_mask"].bool()
        is_src = (pos >= 2) & (pos < plen - 1) & real     # source-language tokens
        is_tgt = (pos >= plen) & real                     # English target tokens
        is_vi = (batch["lang"] == 0).view(-1, 1)
        cat_masks = [is_src & is_vi, is_src & ~is_vi, is_tgt & is_vi, is_tgt & ~is_vi]

        for layer, stats in enumerate(out.router_stats):
            topi = stats.topk_idx                          # (B, T, k)
            topw = stats.topk_weight.float()               # (B, T, k)
            for c, cmask in enumerate(cat_masks):
                m3 = cmask.unsqueeze(-1)
                for e in range(E):
                    hit = (topi == e) & m3
                    counts[layer, e, c] += int(hit.sum())
                    prob_mass[layer, e, c] += float((topw * hit).sum())

    model.set_capture_routing(False)
    if was_training:
        model.train()

    totals = counts.sum(axis=1, keepdims=True)             # per (layer, category)
    fractions = np.divide(counts, np.maximum(totals, 1))

    # Entropy of the expert distribution per layer (bits); log2(E) == perfectly
    # balanced, 0 == a single expert takes everything.
    per_layer = counts.sum(axis=2)
    p = per_layer / np.maximum(per_layer.sum(axis=1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)

    return {
        "categories": list(CATEGORIES),
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "prob_mass": prob_mass.tolist(),
        "per_layer_share": p.tolist(),
        "entropy_bits": ent.tolist(),
        "max_entropy_bits": float(np.log2(E)),
        "n_layers": L,
        "n_experts": E,
        "n_shared": cfg.n_shared_experts,
        "top_k": cfg.n_active_experts,
    }
