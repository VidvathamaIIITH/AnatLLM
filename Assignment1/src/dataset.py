"""Dataset discovery, loading and batching -- tokenised *and* token-free.

Three input layouts are supported:

1. **Parallel files** -- two line-aligned text files, e.g. ``brown_cipher.txt``
   and ``brown_plain.txt``.  Line *k* of one corresponds to line *k* of the
   other.  Detected automatically from the file names, or given explicitly with
   ``--src-file`` / ``--tgt-file``.
2. **Tabular** -- ``.csv``, ``.tsv``, ``.json``, ``.jsonl``, ``.parquet`` with a
   source column and a target column (auto-detected by name).
3. **Paired lines** -- a ``.txt`` where each line is ``source<TAB>target``.

Splits: files whose name contains ``train`` / ``val|valid|dev`` / ``test`` are
used as-is; otherwise the corpus is split 80/10/10 with a fixed seed.

Chunking
--------
``chunk_target_chars`` cuts every long pair into aligned segments *after* the
train/val/test split, so no document ever straddles two splits.  The source is
cut proportionally: if ``len(source)`` is an exact multiple ``r`` of
``len(target)``, segment *i* is ``target[i:i+N]`` against
``source[r*i : r*(i+len(segment))]``.

This matters for the assignment dataset, where the cipher is exactly 8 bits per
plaintext character under a repeating 8-character key that restarts on every
line.  Chunking at a multiple of 8 characters keeps every segment starting at
the same key phase, so the mapping stays exactly as learnable as before -- while
turning 5k document-length pairs into ~45k short, trainable ones.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

__all__ = [
    "DataBundle",
    "discover_dataset",
    "load_pairs_from_file",
    "load_parallel_files",
    "chunk_pairs",
    "Seq2SeqDataset",
    "make_collate_fn",
    "length_percentile",
]

SRC_HINTS = [
    "encrypted", "encryption", "ciphertext", "cipher", "binary", "bits", "bitstring",
    "bit_string", "encoded", "code", "input", "source", "src", "x",
]
TGT_HINTS = [
    "plaintext", "plain_text", "plain", "decrypted", "decoded", "message", "text",
    "target", "tgt", "output", "label", "y",
]
# tokens used to decide the *role* of a whole file in a parallel pair
SRC_ROLE_TOKENS = {
    "cipher", "ciphertext", "encrypted", "encoded", "binary", "bits", "bitstring",
    "src", "source", "input", "enc",
}
TGT_ROLE_TOKENS = {
    "plain", "plaintext", "decrypted", "decoded", "text", "target", "tgt",
    "output", "label", "dec",
}
DATA_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".txt")


# --------------------------------------------------------------------------- #
# tabular readers
# --------------------------------------------------------------------------- #
def _read_table(path: str) -> List[Dict[str, str]]:
    """Read any supported tabular file into a list of ``{column: value}`` dicts."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        import pandas as pd  # parquet genuinely needs pandas + pyarrow

        return pd.read_parquet(path).astype(str).to_dict("records")

    if ext in {".csv", ".tsv"}:
        delim = "\t" if ext == ".tsv" else ","
        try:
            import pandas as pd

            df = pd.read_csv(path, sep=delim, dtype=str, keep_default_na=False)
            return df.to_dict("records")
        except ImportError:
            import csv

            with open(path, "r", encoding="utf-8", newline="") as f:
                return [dict(r) for r in csv.DictReader(f, delimiter=delim)]

    if ext in {".jsonl", ".ndjson"}:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2:
                return [{"source": str(r[0]), "target": str(r[1])} for r in obj]
            return list(obj)
        if isinstance(obj, dict):
            if all(isinstance(v, list) for v in obj.values()) and obj:
                keys = list(obj)
                n = min(len(obj[k]) for k in keys)
                return [{k: obj[k][i] for k in keys} for i in range(n)]
            for key in ("data", "records", "rows", "examples", "pairs"):
                if key in obj and isinstance(obj[key], list):
                    return list(obj[key])
        raise ValueError(f"Unrecognised JSON structure in {path}")

    if ext == ".txt":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                for sep in ("\t", "|||", " ||| ", "\x01"):
                    if sep in line:
                        a, b = line.split(sep, 1)
                        rows.append({"source": a.strip(), "target": b.strip()})
                        break
        if not rows:
            raise ValueError(
                f"{path}: not a table and not tab/'|||' separated pairs. If this is "
                "one half of a parallel corpus, pass --src-file and --tgt-file."
            )
        return rows

    raise ValueError(f"Unsupported file type: {path}")


def _score_column(name: str, hints: Sequence[str]) -> int:
    n = name.strip().lower().replace(" ", "_")
    for rank, h in enumerate(hints):
        if n == h:
            return 1000 - rank
    for rank, h in enumerate(hints):
        if h in n:
            return 500 - rank
    return -1


def _pick_columns(
    rows: List[Dict[str, str]], src_col: Optional[str], tgt_col: Optional[str]
) -> Tuple[str, str]:
    if not rows:
        raise ValueError("Empty dataset file")
    cols = [c for c in rows[0].keys()]
    for provided in (src_col, tgt_col):
        if provided and provided not in cols:
            raise ValueError(f"Column '{provided}' not in file (available: {cols})")
    if src_col and tgt_col:
        return src_col, tgt_col

    src_scores = {c: _score_column(c, SRC_HINTS) for c in cols}
    tgt_scores = {c: _score_column(c, TGT_HINTS) for c in cols}

    chosen_src = src_col or max(src_scores, key=lambda c: src_scores[c])
    if src_scores.get(chosen_src, -1) < 0 and src_col is None:
        chosen_src = None
    chosen_tgt = tgt_col or max(tgt_scores, key=lambda c: tgt_scores[c])
    if tgt_scores.get(chosen_tgt, -1) < 0 and tgt_col is None:
        chosen_tgt = None
    if chosen_src is not None and chosen_src == chosen_tgt:
        if src_scores[chosen_src] >= tgt_scores[chosen_tgt]:
            chosen_tgt = None
        else:
            chosen_src = None

    remaining = [c for c in cols if c not in {chosen_src, chosen_tgt}]
    if chosen_src is None:
        chosen_src = remaining.pop(0) if remaining else cols[0]
    if chosen_tgt is None:
        remaining = [c for c in cols if c not in {chosen_src, chosen_tgt}]
        if not remaining:
            raise ValueError(
                f"Could not find two distinct text columns in {cols}. "
                "Pass --src-col and --tgt-col explicitly."
            )
        chosen_tgt = remaining[0]
    return chosen_src, chosen_tgt


def load_pairs_from_file(
    path: str,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
    swap: bool = False,
) -> Tuple[List[Tuple[str, str]], str, str]:
    rows = _read_table(path)
    s, t = _pick_columns(rows, src_col, tgt_col)
    if swap:
        s, t = t, s
    pairs: List[Tuple[str, str]] = []
    for r in rows:
        a, b = r.get(s, ""), r.get(t, "")
        if a is None or b is None:
            continue
        a, b = str(a), str(b)
        if a == "" or b == "":
            continue
        pairs.append((a, b))
    if not pairs:
        raise ValueError(f"No non-empty pairs read from {path} using columns ({s}, {t})")
    return pairs, s, t


# --------------------------------------------------------------------------- #
# parallel files
# --------------------------------------------------------------------------- #
def _name_parts(path: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return [p for p in re.split(r"[^a-z0-9]+", stem) if p]


def _file_role(path: str) -> Optional[str]:
    parts = set(_name_parts(path))
    src_hit = bool(parts & SRC_ROLE_TOKENS)
    tgt_hit = bool(parts & TGT_ROLE_TOKENS)
    if src_hit and not tgt_hit:
        return "src"
    if tgt_hit and not src_hit:
        return "tgt"
    return None


def _pair_key(path: str) -> str:
    """File identity ignoring the role token, so ``a_cipher``/``a_plain`` pair up."""
    parts = [p for p in _name_parts(path) if p not in SRC_ROLE_TOKENS | TGT_ROLE_TOKENS]
    return "_".join(parts)


def load_parallel_files(
    src_path: str, tgt_path: str, swap: bool = False, verbose: bool = True
) -> List[Tuple[str, str]]:
    """Read two line-aligned text files into ``(source, target)`` pairs."""

    def _lines(p: str) -> List[str]:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return [ln.rstrip("\n").rstrip("\r") for ln in f]

    src_lines = _lines(src_path)
    tgt_lines = _lines(tgt_path)
    # tolerate trailing blank lines on either side
    while src_lines and not src_lines[-1].strip():
        src_lines.pop()
    while tgt_lines and not tgt_lines[-1].strip():
        tgt_lines.pop()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Parallel files are not line-aligned: {src_path} has {len(src_lines)} "
            f"lines but {tgt_path} has {len(tgt_lines)}."
        )
    if swap:
        src_lines, tgt_lines = tgt_lines, src_lines

    pairs = [
        (s.strip(), t.strip())
        for s, t in zip(src_lines, tgt_lines)
        if s.strip() and t.strip()
    ]
    if not pairs:
        raise ValueError(f"No non-empty aligned lines in {src_path} / {tgt_path}")
    if verbose:
        print(
            f"  [data] parallel: {os.path.basename(src_path)} + "
            f"{os.path.basename(tgt_path)} -> {len(pairs)} aligned lines",
            flush=True,
        )
    return pairs


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def chunk_pairs(
    pairs: Sequence[Tuple[str, str]],
    target_chars: int,
    verbose: bool = True,
    label: str = "",
) -> List[Tuple[str, str]]:
    """Cut long pairs into aligned segments of ``target_chars`` target characters.

    A pair is only chunked when ``len(source)`` is an exact multiple of
    ``len(target)`` -- that integer ratio is what makes the source cut point
    unambiguous. Pairs failing the test are passed through untouched.
    """
    if not target_chars or target_chars <= 0:
        return list(pairs)

    out: List[Tuple[str, str]] = []
    n_chunked = 0
    n_passthrough = 0
    for s, t in pairs:
        if not t or len(t) <= target_chars or len(s) % len(t) != 0:
            if t and len(t) > target_chars:
                n_passthrough += 1
            out.append((s, t))
            continue
        ratio = len(s) // len(t)
        if ratio == 0:
            out.append((s, t))
            continue
        n_chunked += 1
        for i in range(0, len(t), target_chars):
            seg_t = t[i : i + target_chars]
            seg_s = s[ratio * i : ratio * (i + len(seg_t))]
            if seg_t and seg_s:
                out.append((seg_s, seg_t))
    if verbose and (n_chunked or n_passthrough):
        tag = f"{label} " if label else ""
        msg = (
            f"  [data] {tag}chunked {n_chunked} pairs at {target_chars} target "
            f"chars -> {len(out)} segments"
        )
        if n_passthrough:
            msg += (
                f" ({n_passthrough} left whole: source/target length ratio "
                f"is not an integer)"
            )
        print(msg, flush=True)
    return out


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    train: List[Tuple[str, str]] = field(default_factory=list)
    val: List[Tuple[str, str]] = field(default_factory=list)
    test: List[Tuple[str, str]] = field(default_factory=list)
    src_col: str = "source"
    tgt_col: str = "target"
    files: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"train={len(self.train)} val={len(self.val)} test={len(self.test)} "
            f"| source='{self.src_col}' target='{self.tgt_col}'"
        )


def _classify(fname: str) -> Optional[str]:
    parts = set(_name_parts(fname))
    if "train" in parts:
        return "train"
    if parts & {"valid", "val", "dev", "validation"}:
        return "val"
    if "test" in parts:
        return "test"
    return None


def _find_parallel_pairs(files: Sequence[str]) -> List[Tuple[str, str]]:
    """Group files into ``(src_path, tgt_path)`` pairs by name."""
    buckets: Dict[str, Dict[str, str]] = {}
    for f in files:
        role = _file_role(f)
        if role is None:
            continue
        buckets.setdefault(_pair_key(f), {})[role] = f
    return [(v["src"], v["tgt"]) for v in buckets.values() if "src" in v and "tgt" in v]


def discover_dataset(
    data_dir: str,
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    test_file: Optional[str] = None,
    src_file: Optional[str] = None,
    tgt_file: Optional[str] = None,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
    swap: bool = False,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    chunk_target_chars: int = 0,
    verbose: bool = True,
) -> DataBundle:
    """Find the data, load it, split it, then chunk each split independently."""
    bundle = DataBundle()
    col_s, col_t = src_col, tgt_col
    eff_s, eff_t = None, None
    parallel_pairs: List[Tuple[str, str]] = []
    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": [], "unknown": []}

    # ---- 1. explicit parallel files --------------------------------------
    if src_file or tgt_file:
        if not (src_file and tgt_file):
            raise ValueError("--src-file and --tgt-file must be given together")
        for p in (src_file, tgt_file):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Parallel file not found: {p}")
        parallel_pairs = [(src_file, tgt_file)]

    # ---- 2. explicit tabular split files ---------------------------------
    elif any((train_file, val_file, test_file)):
        for split, path in (("train", train_file), ("val", val_file), ("test", test_file)):
            if path:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"--{split}-file not found: {path}")
                buckets[split].append(path)

    # ---- 3. scan the directory -------------------------------------------
    else:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory '{data_dir}' does not exist.\n"
                "Download the assignment dataset into it, or run\n"
                "    python -m src.make_synthetic_data --out-dir data_synth\n"
                "to generate a stand-in dataset for a pipeline smoke test."
            )
        found: List[str] = []
        for root, _dirs, files in os.walk(data_dir):
            for f in sorted(files):
                if f.lower().endswith(DATA_EXTS) and f.lower() != "readme.md":
                    found.append(os.path.join(root, f))
        if not found:
            raise FileNotFoundError(
                f"No data files ({', '.join(DATA_EXTS)}) under '{data_dir}'.\n"
                "Place the downloaded dataset there, or generate a synthetic one with\n"
                "    python -m src.make_synthetic_data --out-dir data_synth"
            )
        parallel_pairs = _find_parallel_pairs(found)
        if parallel_pairs:
            used = {p for pair in parallel_pairs for p in pair}
            leftover_files = [f for f in found if f not in used]
            if verbose and leftover_files:
                print(
                    f"  [data] ignoring unpaired files: "
                    f"{[os.path.basename(x) for x in leftover_files]}",
                    flush=True,
                )
        else:
            for f in found:
                buckets[_classify(f) or "unknown"].append(f)

    # ---- load ------------------------------------------------------------
    def _load_tabular(paths: List[str]) -> List[Tuple[str, str]]:
        nonlocal col_s, col_t, eff_s, eff_t
        out: List[Tuple[str, str]] = []
        for p in paths:
            pairs, s, t = load_pairs_from_file(p, col_s, col_t, swap)
            if eff_s is None:
                eff_s, eff_t = s, t
            if col_s is None or col_t is None:
                col_s, col_t = (t, s) if swap else (s, t)
            out.extend(pairs)
            if verbose:
                print(f"  [data] {p}: {len(pairs)} pairs", flush=True)
        return out

    if parallel_pairs:
        by_split: Dict[str, List[Tuple[str, str]]] = {
            "train": [], "val": [], "test": [], "unknown": [],
        }
        for sp, tp in parallel_pairs:
            pairs = load_parallel_files(sp, tp, swap=swap, verbose=verbose)
            split = _classify(sp) or _classify(tp) or "unknown"
            by_split[split].extend(pairs)
            if eff_s is None:
                a, b = os.path.basename(sp), os.path.basename(tp)
                eff_s, eff_t = (b, a) if swap else (a, b)
        bundle.train = by_split["train"]
        bundle.val = by_split["val"]
        bundle.test = by_split["test"]
        leftover = by_split["unknown"]
        buckets = {"parallel": [p for pair in parallel_pairs for p in pair]}
    else:
        bundle.train = _load_tabular(buckets["train"])
        bundle.val = _load_tabular(buckets["val"])
        bundle.test = _load_tabular(buckets["test"])
        leftover = _load_tabular(buckets["unknown"])

    if leftover and not bundle.train:
        rng = random.Random(seed)
        rng.shuffle(leftover)
        n = len(leftover)
        n_test = max(1, int(round(test_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        bundle.test = leftover[:n_test]
        bundle.val = leftover[n_test : n_test + n_val]
        bundle.train = leftover[n_test + n_val :]
        if verbose:
            print(f"  [data] single corpus -> auto split 80/10/10 (seed={seed})", flush=True)
    elif leftover:
        bundle.train.extend(leftover)

    # carve out any split that is still missing
    if not bundle.val or not bundle.test:
        rng = random.Random(seed)
        pool = list(bundle.train)
        rng.shuffle(pool)
        if not bundle.test:
            k = max(1, int(round(test_frac * len(pool))))
            bundle.test = pool[:k]
            pool = pool[k:]
        if not bundle.val:
            k = max(1, int(round(val_frac * len(pool))))
            bundle.val = pool[:k]
            pool = pool[k:]
        bundle.train = pool
        if verbose:
            print("  [data] missing split(s) carved out of train", flush=True)

    bundle.src_col = eff_s or "source"
    bundle.tgt_col = eff_t or "target"
    if verbose:
        print(f"  [data] documents: {bundle.summary()}", flush=True)

    # ---- chunk AFTER splitting so no document straddles two splits --------
    if chunk_target_chars and chunk_target_chars > 0:
        bundle.train = chunk_pairs(bundle.train, chunk_target_chars, verbose, "train")
        bundle.val = chunk_pairs(bundle.val, chunk_target_chars, verbose, "val")
        bundle.test = chunk_pairs(bundle.test, chunk_target_chars, verbose, "test")

    bundle.files = {k: v for k, v in buckets.items() if v}
    if verbose:
        print(f"  [data] examples:  {bundle.summary()}", flush=True)
        if bundle.train:
            s, t = bundle.train[0]
            print(f"  [data] example source[:80]: {s[:80]!r}", flush=True)
            print(f"  [data] example target[:80]: {t[:80]!r}", flush=True)
    return bundle


# --------------------------------------------------------------------------- #
# torch datasets
# --------------------------------------------------------------------------- #
def length_percentile(lengths: Sequence[int], q: float = 0.99) -> int:
    if not lengths:
        return 1
    xs = sorted(lengths)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


class Seq2SeqDataset(Dataset):
    """Pre-encodes string pairs into id tensors.

    Works with :class:`~src.tokenizer.BPETokenizer` (C1-C4) and
    :class:`~src.tokenizer.ByteTokenizer` (C5) interchangeably.

    Layout per item::

        src    = [BOS] src_ids [EOS]
        tgt_in = [BOS] tgt_ids            (decoder input)
        tgt_out=       tgt_ids [EOS]      (labels, shifted by one)

    ``tgt_text`` is the *decoded truncated* target, i.e. exactly the string the
    model is trained to emit.  Comparing greedy output against the untruncated
    original would penalise the model for text it was never shown, making the
    generation metrics meaningless whenever truncation is in play.
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_tokenizer,
        tgt_tokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        keep_text: bool = True,
    ):
        self.pairs = list(pairs)
        self.max_src_len = int(max_src_len)
        self.max_tgt_len = int(max_tgt_len)
        self.keep_text = keep_text
        self.pad_id = int(getattr(src_tokenizer, "pad_id", 0))
        self.bos_id = int(getattr(src_tokenizer, "bos_id", 1))
        self.eos_id = int(getattr(src_tokenizer, "eos_id", 2))

        self.src_ids: List[List[int]] = []
        self.tgt_ids: List[List[int]] = []
        self.ref_texts: List[str] = []
        self.n_src_truncated = 0
        self.n_tgt_truncated = 0

        for s, t in self.pairs:
            si = src_tokenizer.encode(s)
            ti = tgt_tokenizer.encode(t)
            if len(si) > self.max_src_len - 2:
                si = si[: self.max_src_len - 2]
                self.n_src_truncated += 1
            if len(ti) > self.max_tgt_len - 2:
                ti = ti[: self.max_tgt_len - 2]
                self.n_tgt_truncated += 1
                ref = tgt_tokenizer.decode(ti)
            else:
                ref = t
            self.src_ids.append([self.bos_id] + si + [self.eos_id])
            self.tgt_ids.append([self.bos_id] + ti + [self.eos_id])
            if keep_text:
                self.ref_texts.append(ref)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        full = self.tgt_ids[i]
        item = {
            "src": torch.tensor(self.src_ids[i], dtype=torch.long),
            "tgt_in": torch.tensor(full[:-1], dtype=torch.long),
            "tgt_out": torch.tensor(full[1:], dtype=torch.long),
        }
        if self.keep_text:
            item["src_text"] = self.pairs[i][0]
            item["tgt_text"] = self.ref_texts[i]
        return item

    def raw_lengths(self) -> Tuple[List[int], List[int]]:
        return [len(x) for x in self.src_ids], [len(x) for x in self.tgt_ids]


def make_collate_fn(pad_id: int = 0):
    """Pad a list of items into a batch (right padding)."""

    def collate(batch: List[dict]) -> dict:
        def pad(key: str) -> torch.Tensor:
            seqs = [b[key] for b in batch]
            n = max(s.size(0) for s in seqs)
            out = torch.full((len(seqs), n), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : s.size(0)] = s
            return out

        src = pad("src")
        tgt_in = pad("tgt_in")
        tgt_out = pad("tgt_out")
        out = {
            "src": src,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "src_pad_mask": src.ne(pad_id),
            "tgt_pad_mask": tgt_in.ne(pad_id),
        }
        if "src_text" in batch[0]:
            out["src_text"] = [b["src_text"] for b in batch]
            out["tgt_text"] = [b["tgt_text"] for b in batch]
        return out

    return collate
