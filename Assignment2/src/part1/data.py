"""Data pipeline for Part 1: vi->en and ja->en translation.

``belumind/en-vi-ja-curated-500k-triplets`` gives aligned (en, vi, ja) triplets.
Each triplet yields *two* decoder-only training examples:

    <bos> <src_vi> {vietnamese} <tgt_en> {english} <eos>
    <bos> <src_ja> {japanese}   <tgt_en> {english} <eos>

Examples are tokenised once and cached as flat ``uint16`` arrays plus offsets,
which keeps start-up cheap and lets every ablation consume *byte-for-byte the
same batches in the same order* - the token budget is therefore identical
across variants, as the assignment requires.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from ..tokenization import (
    BPETokenizer,
    SRC_JA,
    SRC_VI,
    TGT_EN,
    TRANSLATION_SPECIALS,
    train_bpe,
)
from ..utils import hf_cache_dir, save_json, work_dir

DATASET_ID = "belumind/en-vi-ja-curated-500k-triplets"
SPLITS = ("train", "validation", "test")

#: integer code -> (direction label, source-tag token)
LANGS: dict[int, tuple[str, str]] = {0: ("vi-en", SRC_VI), 1: ("ja-en", SRC_JA)}
LANG_NAME = {0: "vi-en", 1: "ja-en"}


def part1_dir() -> Path:
    d = work_dir() / "part1"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_raw(split: str, max_rows: int | None = None):
    """Load one split of the triplet corpus (cached under the work directory)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, cache_dir=hf_cache_dir())
    if max_rows is not None and max_rows < len(ds):
        ds = ds.select(range(max_rows))
    return ds


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def tokenizer_path() -> Path:
    return part1_dir() / "tokenizer.json"


def build_tokenizer(vocab_size: int = 32000, sample_rows: int = 250_000,
                    force: bool = False) -> BPETokenizer:
    """Train (once) the shared en/vi/ja byte-level BPE tokenizer."""
    path = tokenizer_path()
    if path.exists() and not force:
        return BPETokenizer(path)

    ds = load_raw("train", max_rows=sample_rows)

    def corpus() -> Iterator[str]:
        # Interleave the three languages so the merge table is not dominated by
        # whichever language happens to come first.
        for col in ("en", "vi", "ja"):
            for chunk in ds.select_columns([col]).iter(batch_size=2000):
                for text in chunk[col]:
                    if text:
                        yield text

    print(f"[part1] training BPE (vocab={vocab_size}) on {len(ds):,} triplets ...")
    train_bpe(corpus(), vocab_size=vocab_size, out_path=path,
              specials=TRANSLATION_SPECIALS)
    return BPETokenizer(path)


# --------------------------------------------------------------------------- #
# Binarisation
# --------------------------------------------------------------------------- #


@dataclass
class BinarisedSplit:
    tokens: np.ndarray     # (total_tokens,) uint16
    offsets: np.ndarray    # (N + 1,) int64
    prompt_len: np.ndarray  # (N,) int32 - tokens before the first target token
    lang: np.ndarray       # (N,) uint8  - 0 = vi-en, 1 = ja-en

    def __len__(self) -> int:
        return len(self.prompt_len)

    @property
    def lengths(self) -> np.ndarray:
        return (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)

    def example(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]

    def total_tokens(self) -> int:
        return int(self.offsets[-1])


def _paths(split: str) -> dict[str, Path]:
    d = part1_dir() / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return {k: d / f"{split}.{k}.npy" for k in ("tokens", "offsets", "prompt_len", "lang")}


def binarise_split(
    split: str,
    tok: BPETokenizer,
    n_ctx: int,
    max_rows: int | None = None,
    min_quality: float = 0.0,
    force: bool = False,
) -> BinarisedSplit:
    """Tokenise one split into flat arrays, caching the result on disk."""
    paths = _paths(split)
    if all(p.exists() for p in paths.values()) and not force:
        return BinarisedSplit(**{k: np.load(p) for k, p in paths.items()})

    ds = load_raw(split, max_rows=max_rows)
    bos = tok.bos_id
    eos = tok.eos_id
    tgt_tag = tok.special_id(TGT_EN)
    src_tags = {0: tok.special_id(SRC_VI), 1: tok.special_id(SRC_JA)}

    tokens_chunks: list[np.ndarray] = []
    lengths: list[int] = []
    prompts: list[int] = []
    langs: list[int] = []
    dropped = 0

    print(f"[part1] tokenising split={split} ({len(ds):,} rows) ...")
    for batch in ds.iter(batch_size=2000):
        keep = range(len(batch["en"]))
        if min_quality > 0.0 and "quality_score" in batch:
            keep = [i for i in keep if (batch["quality_score"][i] or 0.0) >= min_quality]
        en_ids = tok.encode_batch([batch["en"][i] for i in keep])
        vi_ids = tok.encode_batch([batch["vi"][i] for i in keep])
        ja_ids = tok.encode_batch([batch["ja"][i] for i in keep])

        for pos in range(len(en_ids)):
            en = en_ids[pos]
            if not en:
                dropped += 2
                continue
            for lang, src in ((0, vi_ids[pos]), (1, ja_ids[pos])):
                if not src:
                    dropped += 1
                    continue
                seq = [bos, src_tags[lang]] + src + [tgt_tag] + en + [eos]
                if len(seq) > n_ctx:
                    dropped += 1
                    continue
                tokens_chunks.append(np.asarray(seq, dtype=np.uint16))
                lengths.append(len(seq))
                prompts.append(2 + len(src) + 1)
                langs.append(lang)

    tokens = np.concatenate(tokens_chunks) if tokens_chunks else np.zeros(0, np.uint16)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
    out = BinarisedSplit(
        tokens=tokens,
        offsets=offsets,
        prompt_len=np.asarray(prompts, dtype=np.int32),
        lang=np.asarray(langs, dtype=np.uint8),
    )
    for key, path in paths.items():
        np.save(path, getattr(out, key))
    print(f"[part1]   -> {len(out):,} examples, {out.total_tokens():,} tokens, "
          f"{dropped:,} dropped (empty or longer than n_ctx={n_ctx})")
    return out


def prepare(
    vocab_size: int = 32000,
    n_ctx: int = 256,
    max_train_rows: int | None = None,
    min_quality: float = 0.0,
    force: bool = False,
    tokenizer_rows: int = 250_000,
) -> tuple[BPETokenizer, dict[str, BinarisedSplit]]:
    """Train the tokenizer (if needed) and binarise all three splits."""
    tok = build_tokenizer(vocab_size=vocab_size, sample_rows=tokenizer_rows, force=force)
    splits = {}
    for split in SPLITS:
        rows = max_train_rows if split == "train" else None
        splits[split] = binarise_split(split, tok, n_ctx, max_rows=rows,
                                       min_quality=min_quality, force=force)
    stats = {
        "vocab_size": tok.vocab_size,
        "n_ctx": n_ctx,
        "splits": {
            s: {
                "examples": len(b),
                "tokens": b.total_tokens(),
                "mean_len": float(b.lengths.mean()) if len(b) else 0.0,
                "p95_len": float(np.percentile(b.lengths, 95)) if len(b) else 0.0,
                "vi_en": int((b.lang == 0).sum()),
                "ja_en": int((b.lang == 1).sum()),
            }
            for s, b in splits.items()
        },
    }
    save_json(stats, part1_dir() / "data_stats.json")
    return tok, splits


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def build_batches(
    lengths: np.ndarray,
    max_tokens: int,
    max_seqs: int = 256,
    indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Length-grouped batches, each holding at most ``max_tokens`` padded tokens.

    Grouping by length keeps padding waste in the low single-digit percent,
    which matters because the training budget is measured in *real* tokens.
    """
    if indices is None:
        indices = np.arange(len(lengths))
    order = indices[np.argsort(lengths[indices], kind="stable")]
    batches: list[np.ndarray] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        L = int(lengths[i])
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_seqs):
            batches.append(np.asarray(cur, dtype=np.int64))
            cur, cur_max = [], 0
            new_max = L
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(np.asarray(cur, dtype=np.int64))
    return batches


def plan_training_batches(
    split: BinarisedSplit,
    token_budget: int,
    max_tokens: int,
    seed: int,
    max_seqs: int = 256,
) -> tuple[list[np.ndarray], int]:
    """Deterministically enumerate every batch of the run, up to the budget.

    Returns the batch list and the exact number of real (non-padding) tokens it
    covers.  Because the plan depends only on the data and the seed, all five
    ablations train on an identical token stream.
    """
    lengths = split.lengths
    plan: list[np.ndarray] = []
    consumed = 0
    epoch = 0
    while consumed < token_budget:
        rng = np.random.default_rng(seed + epoch)
        batches = build_batches(lengths, max_tokens, max_seqs)
        rng.shuffle(batches)
        for b in batches:
            plan.append(b)
            consumed += int(lengths[b].sum())
            if consumed >= token_budget:
                break
        epoch += 1
        if epoch > 1000:  # pragma: no cover - guards against a pathological budget
            raise RuntimeError("token budget unreachable")
    return plan, consumed


def collate(
    split: BinarisedSplit,
    idx: Sequence[int],
    pad_id: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a right-padded batch.

    Right padding plus causal attention means padded positions can never leak
    into a real token's context, so no attention mask is needed at train time;
    the mask is still returned because the MoE load-balancing loss and the
    expert-usage statistics must ignore padding.
    """
    idx = np.asarray(idx, dtype=np.int64)
    lens = (split.offsets[idx + 1] - split.offsets[idx]).astype(np.int64)
    T = int(lens.max())
    B = len(idx)

    buf = np.full((B, T), pad_id, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.int64)
    tgt_start = np.zeros((B,), dtype=np.int64)
    for r, i in enumerate(idx):
        seq = split.tokens[split.offsets[i]:split.offsets[i + 1]].astype(np.int64)
        buf[r, : len(seq)] = seq
        mask[r, : len(seq)] = 1
        tgt_start[r] = int(split.prompt_len[i])

    tokens = torch.from_numpy(buf)
    mask_t = torch.from_numpy(mask)
    input_ids = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()
    attn_mask = mask_t[:, :-1].contiguous()          # keys the model may attend to
    loss_mask = mask_t[:, 1:].contiguous()           # every real predicted token

    # Positions whose *target* lies in the English side (used for target-only
    # perplexity): target j corresponds to absolute position j + 1.
    pos = torch.arange(1, T, dtype=torch.long).view(1, -1)
    target_mask = (pos >= torch.from_numpy(tgt_start).view(-1, 1)) & loss_mask.bool()

    out = {
        "input_ids": input_ids,
        "targets": targets,
        "attention_mask": attn_mask,
        "loss_mask": loss_mask,
        "target_mask": target_mask.long(),
        "prompt_len": torch.from_numpy(tgt_start),
        "lang": torch.from_numpy(split.lang[idx].astype(np.int64)),
        "n_tokens": torch.tensor(int(lens.sum())),
    }
    if str(device) != "cpu":
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    return out


def iter_eval_batches(
    split: BinarisedSplit,
    pad_id: int,
    max_tokens: int,
    device: torch.device | str = "cpu",
    indices: np.ndarray | None = None,
    max_seqs: int = 256,
) -> Iterator[dict[str, torch.Tensor]]:
    """Deterministic length-grouped pass over (a subset of) a split."""
    for b in build_batches(split.lengths, max_tokens, max_seqs, indices):
        yield collate(split, b, pad_id, device)


def prompts_for_generation(
    split: BinarisedSplit,
    indices: Sequence[int],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[int]]:
    """Left-padded prompts (through ``<tgt_en>``) plus the reference targets.

    Left padding is what beam search and batched sampling need: every sequence
    then ends at the same column, so a single "next token" step is well defined.
    """
    idx = np.asarray(indices, dtype=np.int64)
    prompts = [split.example(int(i))[: int(split.prompt_len[i])].astype(np.int64).tolist()
               for i in idx]
    refs = [split.example(int(i))[int(split.prompt_len[i]):].astype(np.int64).tolist()
            for i in idx]
    T = max(len(p) for p in prompts)
    buf = np.full((len(prompts), T), pad_id, dtype=np.int64)
    mask = np.zeros((len(prompts), T), dtype=np.int64)
    for r, p in enumerate(prompts):
        buf[r, T - len(p):] = p
        mask[r, T - len(p):] = 1
    return (torch.from_numpy(buf), torch.from_numpy(mask), refs,
            [int(split.lang[i]) for i in idx])


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import argparse

    ap = argparse.ArgumentParser(description="Prepare the Part 1 translation data")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--min-quality", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", hf_cache_dir())
    prepare(vocab_size=args.vocab_size, n_ctx=args.n_ctx,
            max_train_rows=args.max_train_rows, min_quality=args.min_quality,
            force=args.force)
