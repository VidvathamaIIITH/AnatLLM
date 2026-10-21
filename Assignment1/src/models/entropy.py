"""Lightweight next-byte entropy estimation for BLT's dynamic patching (C5).

The Byte Latent Transformer segments a byte stream into *variable-length*
patches by monitoring the entropy of the next-byte distribution: a new patch
starts wherever the model is locally uncertain, so predictable runs get grouped
into long patches and surprising positions get their own short ones.

The original paper estimates that entropy with a small byte-level LM. That is
unnecessary at this scale, so this module uses an **order-2 byte n-gram (Markov)
model with add-alpha smoothing and stupid backoff** -- the "lightweight n-gram /
Markov" estimator the assignment clarification permits. Fitting is a single
counting pass over the training split and takes a couple of seconds.

Why a lookup table
------------------
Because the estimator is a fixed n-gram model, the conditional entropy
:math:`H(x_t \\mid x_{t-2}, x_{t-1})` depends only on the *context*, never on
:math:`x_t`. So every entropy the model will ever need can be precomputed into
three tables::

    H0                scalar          -- no context   (position 0)
    H1[256]           order-1         -- one byte of context (position 1)
    H2[256*256]       order-2         -- two bytes of context (positions >= 2)

Those tables are registered as tensors on the model, so patch boundaries are
computed with a couple of gathers on the GPU. Crucially this means training and
incremental generation run the *identical* boundary logic -- there is no second
CPU implementation to drift out of sync.

Threshold calibration
---------------------
A raw entropy threshold is not portable across data (cipher bytes are close to
uniform; English text is not). :meth:`calibrate_threshold` instead picks the
global threshold as the quantile of the observed training entropies that yields
a target mean patch length, which is what makes the same code behave sensibly on
both sides of this task.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["ByteNgramEntropyModel", "segment_from_entropy"]

VOCAB = 256


class ByteNgramEntropyModel:
    """Order-2 byte n-gram entropy estimator with add-alpha smoothing.

    Args:
        alpha: additive (Laplace/Lidstone) smoothing constant.
        min_count: a context must have been seen at least this many times to be
            used; otherwise the estimate backs off to the shorter context.
    """

    def __init__(self, alpha: float = 0.5, min_count: int = 8):
        self.alpha = float(alpha)
        self.min_count = int(min_count)
        self.H0: float = math.log2(VOCAB)
        self.H1: np.ndarray = np.full(VOCAB, math.log2(VOCAB), dtype=np.float32)
        self.H2: np.ndarray = np.full(VOCAB * VOCAB, math.log2(VOCAB), dtype=np.float32)
        self.fitted = False
        self.threshold: float = float("inf")
        self._stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, sequences: Iterable[Sequence[int]], verbose: bool = True) -> "ByteNgramEntropyModel":
        """Count unigram/bigram/trigram statistics over raw byte sequences.

        Args:
            sequences: iterable of sequences of **raw** byte values (0-255).
        """
        c0 = np.zeros(VOCAB, dtype=np.float64)
        c1 = np.zeros((VOCAB, VOCAB), dtype=np.float64)
        # order-2 counts are sparse; accumulate in a dict of arrays
        c2: Dict[int, np.ndarray] = {}

        n_seq = n_bytes = 0
        for seq in sequences:
            a = np.frombuffer(bytes(seq), dtype=np.uint8) if not isinstance(seq, np.ndarray) \
                else seq.astype(np.uint8)
            if a.size == 0:
                continue
            n_seq += 1
            n_bytes += a.size
            np.add.at(c0, a, 1.0)
            if a.size >= 2:
                np.add.at(c1, (a[:-1], a[1:]), 1.0)
            if a.size >= 3:
                ctx = a[:-2].astype(np.int32) * VOCAB + a[1:-1].astype(np.int32)
                nxt = a[2:]
                for cc, nn in zip(ctx, nxt):
                    arr = c2.get(int(cc))
                    if arr is None:
                        arr = np.zeros(VOCAB, dtype=np.float32)
                        c2[int(cc)] = arr
                    arr[nn] += 1.0

        # ---- order 0 -----------------------------------------------------
        self.H0 = float(_entropy(c0, self.alpha))

        # ---- order 1 -----------------------------------------------------
        tot1 = c1.sum(axis=1)
        for b in range(VOCAB):
            self.H1[b] = _entropy(c1[b], self.alpha) if tot1[b] >= self.min_count else self.H0

        # ---- order 2 (stupid backoff to order 1) -------------------------
        # ctx = prev2 * 256 + prev1, so prev1 is ctx % 256; default each order-2
        # context to its order-1 estimate, then overwrite the well-attested ones.
        prev1_of_ctx = np.arange(VOCAB * VOCAB, dtype=np.int64) % VOCAB
        self.H2 = self.H1[prev1_of_ctx].astype(np.float32).copy()
        n_ctx_used = 0
        for cc, arr in c2.items():
            if arr.sum() >= self.min_count:
                self.H2[cc] = _entropy(arr, self.alpha)
                n_ctx_used += 1

        self.fitted = True
        self._stats = {
            "sequences": n_seq, "bytes": n_bytes,
            "order2_contexts_seen": len(c2), "order2_contexts_used": n_ctx_used,
            "H0": self.H0, "H1_mean": float(self.H1.mean()), "H2_mean": float(self.H2.mean()),
        }
        if verbose:
            print(f"    [entropy] fit on {n_seq} seqs / {n_bytes:,} bytes | "
                  f"H0={self.H0:.3f} bits | order-2 contexts used "
                  f"{n_ctx_used:,}/{len(c2):,}", flush=True)
        return self

    # -------------------------------------------------------------- entropy
    def entropies(self, seq: Sequence[int]) -> np.ndarray:
        """Per-position next-byte entropy for one raw byte sequence."""
        a = np.asarray(seq, dtype=np.int64)
        n = a.size
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        out[0] = self.H0
        if n >= 2:
            out[1] = self.H1[a[0]]
        if n >= 3:
            ctx = a[:-2] * VOCAB + a[1:-1]
            out[2:] = self.H2[ctx]
        return out

    # ------------------------------------------------------------ calibrate
    def calibrate_threshold(
        self,
        sequences: Sequence[Sequence[int]],
        target_patch_len: float = 4.0,
        sample: int = 4000,
        verbose: bool = True,
    ) -> float:
        """Pick the global threshold that yields ``target_patch_len`` on average.

        A boundary is opened wherever ``H_t > threshold``; position 0 always
        opens one. To average one patch every ``L`` bytes we therefore want a
        fraction ``1/L`` of positions above the threshold, i.e. the
        ``(1 - 1/L)`` quantile of the entropy distribution.
        """
        vals: List[np.ndarray] = []
        for seq in list(sequences)[:sample]:
            if len(seq):
                vals.append(self.entropies(seq))
        if not vals:
            self.threshold = self.H0
            return self.threshold
        allv = np.concatenate(vals)
        q = max(0.0, min(1.0, 1.0 - 1.0 / max(1.0, float(target_patch_len))))
        self.threshold = float(np.quantile(allv, q))
        if verbose:
            frac = float((allv > self.threshold).mean())
            print(f"    [entropy] threshold={self.threshold:.4f} bits "
                  f"(q={q:.3f}) -> {frac*100:.1f}% of positions open a patch "
                  f"(~{1.0/max(frac,1e-9):.2f} bytes/patch)", flush=True)
        return self.threshold

    # --------------------------------------------------------------- tensors
    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return ``(H1 [256], H2 [65536], H0)`` for registration on a module."""
        return (
            torch.from_numpy(self.H1.astype(np.float32)),
            torch.from_numpy(self.H2.astype(np.float32)),
            float(self.H0),
        )

    # ------------------------------------------------------------ serialise
    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha, "min_count": self.min_count,
            "H0": self.H0, "H1": self.H1.tolist(), "H2": self.H2.tolist(),
            "threshold": self.threshold, "fitted": self.fitted, "stats": self._stats,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "ByteNgramEntropyModel":
        m = cls(alpha=d.get("alpha", 0.5), min_count=d.get("min_count", 8))
        m.H0 = float(d["H0"])
        m.H1 = np.asarray(d["H1"], dtype=np.float32)
        m.H2 = np.asarray(d["H2"], dtype=np.float32)
        m.threshold = float(d.get("threshold", float("inf")))
        m.fitted = bool(d.get("fitted", True))
        m._stats = d.get("stats", {})
        return m

    @property
    def stats(self) -> Dict[str, float]:
        return dict(self._stats)


def _entropy(counts: np.ndarray, alpha: float) -> float:
    """Shannon entropy (bits) of a count vector under add-alpha smoothing."""
    p = counts.astype(np.float64) + alpha
    p /= p.sum()
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


# --------------------------------------------------------------------------- #
# boundary -> segment ids
# --------------------------------------------------------------------------- #
def segment_from_entropy(
    ids: torch.Tensor,
    H1: torch.Tensor,
    H2: torch.Tensor,
    H0: float,
    threshold: float,
    offset: int,
    n_special: int,
    pad_mask: Optional[torch.Tensor] = None,
    max_patch: int = 0,
) -> torch.Tensor:
    """Entropy-based dynamic patch assignment for a batch of byte ids.

    Args:
        ids: ``[B, T]`` byte *token ids* (raw byte ``b`` is stored as ``b+offset``).
        H1/H2: entropy lookup tables, ``[256]`` and ``[65536]``.
        H0: entropy with no context.
        threshold: open a new patch where ``H_t > threshold``.
        offset: id offset applied to raw bytes.
        n_special: ids ``< n_special`` are specials (pad/bos/eos).
        pad_mask: ``[B, T]`` bool, ``True`` = real byte.
        max_patch: if > 0, force a boundary once a patch reaches this length.

    Returns:
        ``seg`` ``[B, T]`` int64 patch index per byte (0-based, contiguous and
        non-decreasing along T). Padded positions get ``-1``.
    """
    b, t = ids.shape
    device = ids.device
    raw = (ids - offset).clamp_(0, VOCAB - 1)
    is_special = ids < n_special

    # H_t depends only on the context (x_{t-2}, x_{t-1})
    H = torch.empty(b, t, dtype=torch.float32, device=device)
    if t >= 1:
        H[:, 0] = H0
    if t >= 2:
        H[:, 1] = H1[raw[:, 0]]
    if t >= 3:
        ctx = raw[:, :-2] * VOCAB + raw[:, 1:-1]
        H[:, 2:] = H2[ctx]

    boundary = H > float(threshold)
    boundary[:, 0] = True                      # position 0 always opens a patch
    # a special token (BOS/EOS) always starts its own patch: its "context" is
    # meaningless to the n-gram model, so never let it extend a real patch.
    boundary |= is_special

    if max_patch and max_patch > 0:
        # Force a boundary whenever the current patch reaches max_patch bytes.
        # T is small (tens of bytes), so a scan is cheaper than it looks and
        # keeps M bounded regardless of how flat the entropy happens to be.
        run = torch.zeros(b, dtype=torch.long, device=device)
        cols = []
        for i in range(t):
            bi = boundary[:, i] | (run >= max_patch)
            run = torch.where(bi, torch.ones_like(run), run + 1)
            cols.append(bi)
        boundary = torch.stack(cols, dim=1)
        boundary[:, 0] = True

    seg = boundary.long().cumsum(dim=1) - 1
    if pad_mask is not None:
        seg = seg.masked_fill(~pad_mask, -1)
    return seg
