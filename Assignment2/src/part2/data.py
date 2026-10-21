"""Data pipeline for Part 2: next-token pre-training on the human-AI parallel corpus.

``browndw/human-ai-parallel-corpus`` ships a single ``train`` split of 66,320
documents whose ids look like ``acad_0001@gpt-4o-2024-08-06``.  The part before
the ``@`` identifies the *source document* and the part after it identifies the
writer - either a human chunk or one of several LLMs.  Because the same source
document appears once per writer, splitting rows at random would put near
-duplicate texts on both sides of the train/test boundary.  Splits are therefore
made on the **base document id**, so every version of a document lands in the
same split.

Tokens are packed into a flat stream and cut into windows of ``n_ctx + 1`` with
stride ``n_ctx``, so that across one epoch every token is predicted exactly once
- which is what makes "1x the dataset" a well-defined training budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from ..tokenization import BASE_SPECIALS, BPETokenizer, train_bpe
from ..utils import hf_cache_dir, save_json, work_dir

DATASET_ID = "browndw/human-ai-parallel-corpus"
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.90, "validation": 0.05, "test": 0.05}


def part2_dir() -> Path:
    d = work_dir() / "part2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_doc_id(doc_id: str) -> str:
    """``acad_0001@gpt-4o`` -> ``acad_0001`` (the shared source document)."""
    return doc_id.split("@", 1)[0]


def assign_split(base_id: str) -> str:
    """Deterministic, stable hash split on the base document id."""
    h = hashlib.blake2b(base_id.encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, "big") / float(1 << 64)
    if u < SPLIT_FRACTIONS["train"]:
        return "train"
    if u < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]:
        return "validation"
    return "test"


def load_raw():
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split="train", cache_dir=hf_cache_dir())


def split_texts(split: str) -> Iterator[str]:
    ds = load_raw()
    for batch in ds.iter(batch_size=1000):
        for doc_id, text in zip(batch["doc_id"], batch["text"]):
            if text and assign_split(base_doc_id(doc_id)) == split:
                yield text


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def tokenizer_path() -> Path:
    return part2_dir() / "tokenizer.json"


def build_tokenizer(vocab_size: int = 16000, force: bool = False) -> BPETokenizer:
    """Train (once) an English byte-level BPE on the *training* split only."""
    path = tokenizer_path()
    if path.exists() and not force:
        return BPETokenizer(path)
    print(f"[part2] training BPE (vocab={vocab_size}) on the training split ...")
    train_bpe(split_texts("train"), vocab_size=vocab_size, out_path=path,
              specials=BASE_SPECIALS)
    return BPETokenizer(path)


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


@dataclass
class PackedSplit:
    """A flat token stream cut into windows of ``n_ctx + 1``."""

    tokens: np.ndarray   # (T,) uint16
    n_ctx: int

    @property
    def n_blocks(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.n_ctx)

    @property
    def n_tokens(self) -> int:
        """Tokens actually predicted in one epoch."""
        return self.n_blocks * self.n_ctx

    def block(self, i: int) -> np.ndarray:
        s = i * self.n_ctx
        return self.tokens[s:s + self.n_ctx + 1]


def _tokens_path(split: str) -> Path:
    d = part2_dir() / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{split}.tokens.npy"


def binarise_split(split: str, tok: BPETokenizer, n_ctx: int,
                   force: bool = False) -> PackedSplit:
    path = _tokens_path(split)
    if path.exists() and not force:
        return PackedSplit(np.load(path), n_ctx)

    print(f"[part2] tokenising split={split} ...")
    chunks: list[np.ndarray] = []
    n_docs = 0
    buf: list[str] = []

    def flush(buf: list[str]) -> None:
        nonlocal n_docs
        if not buf:
            return
        for ids in tok.encode_batch(buf):
            chunks.append(np.asarray(ids + [tok.eos_id], dtype=np.uint16))
            n_docs += 1

    for text in split_texts(split):
        buf.append(text)
        if len(buf) >= 500:
            flush(buf)
            buf = []
    flush(buf)

    tokens = np.concatenate(chunks) if chunks else np.zeros(0, np.uint16)
    np.save(path, tokens)
    out = PackedSplit(tokens, n_ctx)
    print(f"[part2]   -> {n_docs:,} documents, {len(tokens):,} tokens, "
          f"{out.n_blocks:,} blocks of {n_ctx}")
    return out


def prepare(vocab_size: int = 16000, n_ctx: int = 256,
            force: bool = False) -> tuple[BPETokenizer, dict[str, PackedSplit]]:
    tok = build_tokenizer(vocab_size=vocab_size, force=force)
    splits = {s: binarise_split(s, tok, n_ctx, force=force) for s in SPLITS}
    stats = {
        "vocab_size": tok.vocab_size,
        "n_ctx": n_ctx,
        "splits": {s: {"tokens": int(len(b.tokens)), "blocks": b.n_blocks,
                       "epoch_tokens": b.n_tokens}
                   for s, b in splits.items()},
    }
    save_json(stats, part2_dir() / "data_stats.json")
    return tok, splits


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def epoch_order(split: PackedSplit, seed: int) -> np.ndarray:
    """A deterministic permutation of every block in the split."""
    rng = np.random.default_rng(seed)
    order = np.arange(split.n_blocks)
    rng.shuffle(order)
    return order


def make_batch(split: PackedSplit, block_ids: np.ndarray,
               device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """All blocks have the same length, so no padding or masking is needed."""
    rows = np.stack([split.block(int(i)) for i in block_ids]).astype(np.int64)
    t = torch.from_numpy(rows)
    out = {
        "input_ids": t[:, :-1].contiguous(),
        "targets": t[:, 1:].contiguous(),
        "n_tokens": torch.tensor(len(block_ids) * split.n_ctx),
    }
    if str(device) != "cpu":
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    return out


def iter_batches(split: PackedSplit, batch_blocks: int, device, order: np.ndarray | None = None,
                 max_blocks: int | None = None) -> Iterator[dict[str, torch.Tensor]]:
    order = np.arange(split.n_blocks) if order is None else order
    if max_blocks is not None:
        order = order[:max_blocks]
    for s in range(0, len(order), batch_blocks):
        yield make_batch(split, order[s:s + batch_blocks], device)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from ..utils import configure_stdout

    configure_stdout()
    ap = argparse.ArgumentParser(description="Prepare the Part 2 pre-training data")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare(vocab_size=args.vocab_size, n_ctx=args.n_ctx, force=args.force)
