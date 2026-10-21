"""Evaluation for Part 2: validation loss and test BLEU as a function of tokens seen."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..part1.model import TransformerLM
from ..part3.decoding import GenerationConfig, LocalBackend, generate
from ..tokenization import BPETokenizer
from .data import PackedSplit, make_batch


@torch.no_grad()
def evaluate_loss(
    model: TransformerLM,
    split: PackedSplit,
    device: torch.device,
    batch_blocks: int = 8,
    max_blocks: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> dict[str, float]:
    """Token-weighted cross-entropy over a deterministic prefix of the split."""
    was_training = model.training
    model.eval()

    n = split.n_blocks if max_blocks is None else min(max_blocks, split.n_blocks)
    total_nll = 0.0
    total_tok = 0
    for s in range(0, n, batch_blocks):
        ids = np.arange(s, min(s + batch_blocks, n))
        batch = make_batch(split, ids, device)
        ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with ctx:
            logits = model(batch["input_ids"]).logits
        nll = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            batch["targets"].view(-1), reduction="sum",
        )
        total_nll += float(nll)
        total_tok += int(batch["targets"].numel())

    if was_training:
        model.train()
    loss = total_nll / max(1, total_tok)
    return {"loss": loss, "ppl": math.exp(min(loss, 50.0)), "tokens": total_tok}


@torch.no_grad()
def evaluate_bleu(
    model: TransformerLM,
    split: PackedSplit,
    tok: BPETokenizer,
    device: torch.device,
    n_prompts: int = 250,
    prompt_tokens: int = 128,
    gen_tokens: int = 48,
    batch_size: int = 25,
    amp_dtype: Optional[torch.dtype] = None,
    strategy: str = "greedy",
) -> dict[str, float]:
    """BLEU of generated continuations against the true continuation.

    Each test block is split into a ``prompt_tokens`` prefix and a
    ``gen_tokens`` reference continuation; the model free-runs from the prefix
    and the two are compared with sacreBLEU.  This is a blunt instrument for a
    pure language-modelling task - continuations are legitimately open-ended -
    but it is a consistent yardstick across optimizers and training budgets,
    which is what the comparison needs.
    """
    import sacrebleu

    was_training = model.training
    model.eval()

    need = prompt_tokens + gen_tokens
    if split.n_ctx + 1 < need:
        prompt_tokens = max(8, (split.n_ctx + 1) - gen_tokens)
        need = prompt_tokens + gen_tokens

    n = min(n_prompts, split.n_blocks)
    hyps: list[str] = []
    refs: list[str] = []
    seconds = 0.0

    cfg = GenerationConfig(strategy=strategy, max_new_tokens=gen_tokens,
                           eos_id=None, pad_id=tok.pad_id)

    for s in range(0, n, batch_size):
        ids = np.arange(s, min(s + batch_size, n))
        rows = np.stack([split.block(int(i))[:need] for i in ids]).astype(np.int64)
        prompts = torch.from_numpy(rows[:, :prompt_tokens]).to(device)
        mask = torch.ones_like(prompts)
        backend = LocalBackend(model)
        ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None and device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with ctx:
            res = generate(backend, prompts, mask, cfg)
        seconds += res.seconds
        for j in range(len(ids)):
            hyps.append(tok.decode(res.tokens[j]))
            refs.append(tok.decode(rows[j, prompt_tokens:].tolist()))

    if was_training:
        model.train()

    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {
        "bleu": float(bleu.score),
        "chrf2": float(chrf.score),
        "n_prompts": len(hyps),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "seconds": seconds,
        "sample_hyp": hyps[0] if hyps else "",
        "sample_ref": refs[0] if refs else "",
    }
