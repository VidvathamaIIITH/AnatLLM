"""Tokenizers written from scratch.

Two strategies, matching the two branches of the ablation:

* :class:`BPETokenizer` -- a **byte-level Byte-Pair-Encoding** subword tokenizer
  (the "Standard Subword" column, configs C1-C4).  Trained on the corpus with
  the classic Sennrich et al. (2016) merge algorithm, accelerated with a
  pair-count index and a lazy-deletion heap so a 4k-merge vocabulary trains in
  seconds instead of minutes.

* :class:`ByteTokenizer` -- the token-free byte alphabet used by BLT (config
  C5).  No training, no vocabulary file, 259 ids (3 specials + 256 bytes).

Why *byte-level* BPE?  Encoding goes ``str -> utf-8 bytes -> one printable
unicode char per byte -> merges``.  Every possible input is representable, so
``decode(encode(x)) == x`` exactly -- which matters here because Sequence
Accuracy is an exact-string metric and any lossy round-trip would silently cap
every model's score.
"""

from __future__ import annotations

import heapq
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "UNK_ID",
    "SPECIAL_TOKENS",
    "BPETokenizer",
    "ByteTokenizer",
    "build_tokenizer",
]

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# whitespace runs and non-whitespace runs; concatenation reconstructs the input
_PRETOKEN_RE = re.compile(r"\s+|[^\s]+")


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's reversible byte <-> printable-unicode map (no byte is ever lost)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


class BPETokenizer:
    """Byte-level BPE.

    Args:
        max_pretoken_len: pre-tokens longer than this are cut into fixed-size
            chunks *before* merges are counted.  Essential for this dataset:
            an encrypted bit-string is one enormous whitespace-free "word", and
            without chunking the merge statistics would be dominated by a
            handful of gigantic symbols.
    """

    def __init__(self, max_pretoken_len: int = 32):
        self.max_pretoken_len = int(max_pretoken_len)
        self.vocab: List[str] = list(SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.merges: List[Tuple[str, str]] = []
        self.ranks: Dict[Tuple[str, str], int] = {}
        self._cache: Dict[str, Tuple[str, ...]] = {}
        self.trained = False

    # ------------------------------------------------------------------ props
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    pad_id, bos_id, eos_id, unk_id = PAD_ID, BOS_ID, EOS_ID, UNK_ID

    # ------------------------------------------------------------ pretokenise
    def _pretokenize(self, text: str) -> List[str]:
        out: List[str] = []
        for pt in _PRETOKEN_RE.findall(text):
            mapped = "".join(_BYTE_ENCODER[b] for b in pt.encode("utf-8"))
            n = self.max_pretoken_len
            if n and len(mapped) > n:
                out.extend(mapped[i : i + n] for i in range(0, len(mapped), n))
            else:
                out.append(mapped)
        return out

    # ----------------------------------------------------------------- train
    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """Learn merges until ``len(vocab) == vocab_size`` (or no pair repeats)."""
        min_vocab = len(SPECIAL_TOKENS) + 256
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size must be >= {min_vocab} for a byte-level BPE (got {vocab_size})"
            )

        word_freqs: Counter = Counter()
        n_texts = 0
        for text in texts:
            n_texts += 1
            for chunk in self._pretokenize(text):
                word_freqs[chunk] += 1

        # base alphabet = all 256 byte chars, so encoding never produces <unk>
        self.vocab = list(SPECIAL_TOKENS) + [_BYTE_ENCODER[b] for b in range(256)]

        words: List[List[str]] = [list(w) for w in word_freqs]
        freqs: List[int] = [word_freqs[w] for w in word_freqs]

        pair_counts: Counter = Counter()
        where: Dict[Tuple[str, str], set] = defaultdict(set)
        for idx, symbols in enumerate(words):
            f = freqs[idx]
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += f
                where[pair].add(idx)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        target_merges = vocab_size - len(self.vocab)
        self.merges = []
        while len(self.merges) < target_merges:
            best = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_count:
                    best = (pair, -neg_count)
                    break
            if best is None:
                break
            pair, count = best
            if count < min_frequency:
                break

            new_symbol = pair[0] + pair[1]
            self.merges.append(pair)
            self.vocab.append(new_symbol)

            touched = list(where.get(pair, ()))
            for idx in touched:
                symbols = words[idx]
                f = freqs[idx]
                merged: List[str] = []
                i = 0
                changed = False
                while i < len(symbols):
                    if (
                        i < len(symbols) - 1
                        and symbols[i] == pair[0]
                        and symbols[i + 1] == pair[1]
                    ):
                        merged.append(new_symbol)
                        i += 2
                        changed = True
                    else:
                        merged.append(symbols[i])
                        i += 1
                if not changed:
                    continue
                # Heap invariant: after *every* count change the heap must still
                # hold an entry matching the new count, otherwise the stale entry
                # is discarded on pop and the pair silently disappears from
                # consideration even though it is still frequent.
                for p in zip(symbols, symbols[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    else:
                        heapq.heappush(heap, (-pair_counts[p], p))
                    where[p].discard(idx)
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += f
                    where[p].add(idx)
                    heapq.heappush(heap, (-pair_counts[p], p))
                words[idx] = merged

            pair_counts.pop(pair, None)
            where.pop(pair, None)

            if verbose and len(self.merges) % 500 == 0:
                print(f"    [bpe] {len(self.merges)}/{target_merges} merges "
                      f"(vocab={len(self.vocab)})", flush=True)

        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self._cache = {}
        self.trained = True
        if verbose:
            print(
                f"    [bpe] trained on {n_texts} texts | {len(word_freqs)} unique pre-tokens "
                f"| vocab={len(self.vocab)} | merges={len(self.merges)}",
                flush=True,
            )
        return self

    # ---------------------------------------------------------------- encode
    def _bpe(self, token: str) -> Tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        symbols: List[str] = list(token)
        while len(symbols) > 1:
            best_rank, best_i = None, -1
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        out = tuple(symbols)
        if len(self._cache) < 500_000:
            self._cache[token] = out
        return out

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [BOS_ID] if add_bos else []
        for chunk in self._pretokenize(text):
            for piece in self._bpe(chunk):
                ids.append(self.token_to_id.get(piece, UNK_ID))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_batch(self, texts: Sequence[str], **kw) -> List[List[int]]:
        return [self.encode(t, **kw) for t in texts]

    # ---------------------------------------------------------- compliance
    def piece_length_stats(self, texts: Sequence[str], sample: int = 300) -> Dict[str, float]:
        """Character-length statistics of the pieces this tokenizer actually emits.

        Used as a runtime guardrail: genuine learned subword tokenization
        produces pieces of *varying* length, driven by corpus statistics. A
        tokenizer whose every emitted piece has the same length has degenerated
        into fixed-width chunking -- the exact scheme the assignment brief
        explicitly disallows ("8 bits = 1 token does not count as subword
        tokenization"). That degeneration is usually caused by setting
        ``max_pretoken_len`` too close to the alphabet's natural period (e.g. 8
        for an 8-bit-periodic binary cipher): BPE then just learns a fixed
        byte-lookup table and has nothing shorter left to merge.

        See ``build_tokenizers`` in ``train.py``, which calls this after every
        training/load and prints a loud warning if ``n_distinct <= 1``.
        """
        lens: List[int] = []
        for text in texts[:sample]:
            for chunk in self._pretokenize(text):
                for piece in self._bpe(chunk):
                    lens.append(len(piece))
        if not lens:
            return {"mean": 0.0, "min": 0, "max": 0, "stdev": 0.0, "n_distinct": 0}
        mean = sum(lens) / len(lens)
        var = sum((x - mean) ** 2 for x in lens) / len(lens)
        return {
            "mean": mean,
            "min": min(lens),
            "max": max(lens),
            "stdev": var ** 0.5,
            "n_distinct": len(set(lens)),
        }

    # ---------------------------------------------------------------- decode
    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        pieces: List[str] = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.vocab):
                continue
            if i < len(SPECIAL_TOKENS):
                if skip_special:
                    continue
                pieces.append(self.vocab[i])
                continue
            pieces.append(self.vocab[i])
        text = "".join(pieces)
        try:
            raw = bytes(_BYTE_DECODER[c] for c in text)
        except KeyError:
            raw = bytes(_BYTE_DECODER.get(c, ord("?")) for c in text)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- save/load
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "bpe",
                    "max_pretoken_len": self.max_pretoken_len,
                    "vocab": self.vocab,
                    "merges": [list(p) for p in self.merges],
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls(max_pretoken_len=obj.get("max_pretoken_len", 32))
        tok.vocab = list(obj["vocab"])
        tok.merges = [tuple(p) for p in obj["merges"]]
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.ranks = {p: i for i, p in enumerate(tok.merges)}
        tok.trained = True
        return tok


class ByteTokenizer:
    """Token-free byte tokenizer used by BLT (config C5).

    ids: ``0=<pad>``, ``1=<bos>``, ``2=<eos>``, and raw byte ``b -> b + 3``.

    Two input conventions
    ---------------------
    ``bit_packed=False`` (the target side): the string is ordinary text, so its
    UTF-8 bytes are the byte stream.

    ``bit_packed=True`` (the cipher source): the string is a *binary sequence*
    written as ASCII ``'0'``/``'1'`` characters. The underlying data is the
    binary sequence itself, not its text encoding, so every **8 bits are packed
    (MSB-first) into one byte value 0-255** before the model sees them. Treating
    each ``'0'``/``'1'`` character as a byte would feed the model the text
    encoding of the data and inflate the sequence 8-fold.

    A 256-char cipher segment therefore becomes 32 bytes, not 256.
    """

    pad_id, bos_id, eos_id = 0, 1, 2
    offset = 3
    vocab_size = 256 + 3
    trained = True

    def __init__(self, bit_packed: bool = False):
        self.bit_packed = bool(bit_packed)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _pack_bits(text: str) -> bytes:
        """``'0101...'`` -> packed bytes, MSB-first, zero-padded to a multiple of 8."""
        s = text.strip()
        if len(s) % 8:
            s = s + "0" * (8 - len(s) % 8)
        return bytes(int(s[i:i + 8], 2) for i in range(0, len(s), 8))

    @staticmethod
    def _unpack_bits(raw: bytes) -> str:
        return "".join(format(b, "08b") for b in raw)

    def _to_bytes(self, text: str) -> bytes:
        return self._pack_bits(text) if self.bit_packed else text.encode("utf-8")

    # -------------------------------------------------------------- public
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.bos_id] if add_bos else []
        ids.extend(b + self.offset for b in self._to_bytes(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: Sequence[str], **kw) -> List[List[int]]:
        return [self.encode(t, **kw) for t in texts]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        raw = bytearray()
        for i in ids:
            i = int(i)
            if i >= self.offset:
                raw.append(i - self.offset)
        if self.bit_packed:
            return self._unpack_bits(bytes(raw))
        return bytes(raw).decode("utf-8", errors="replace")

    def raw_bytes(self, text: str) -> List[int]:
        """The 0-255 byte values this tokenizer would emit (no specials).

        Used to fit the n-gram entropy model, which works on raw bytes.
        """
        return list(self._to_bytes(text))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte", "vocab_size": self.vocab_size,
                       "bit_packed": self.bit_packed}, f)

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        try:
            with open(path, encoding="utf-8") as f:
                return cls(bit_packed=bool(json.load(f).get("bit_packed", False)))
        except Exception:
            return cls()


def build_tokenizer(kind: str, **kwargs):
    """``kind in {"bpe", "subword", "byte", "blt"}``."""
    key = kind.lower()
    if key in {"bpe", "subword", "standard", "standard_subword"}:
        return BPETokenizer(**kwargs)
    if key in {"byte", "blt", "tokenfree", "token_free"}:
        return ByteTokenizer(bit_packed=bool(kwargs.get("bit_packed", False)))
    raise ValueError(f"Unknown tokenizer '{kind}'")
