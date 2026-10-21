"""Positional information: sinusoidal absolute encodings and RoPE.

* :class:`SinusoidalPositionalEncoding` -- the additive absolute encoding from
  "Attention Is All You Need".  Used by C1, C3, C4, C5.
* :class:`RotaryPositionalEmbedding` -- RoPE (Su et al., 2021).  Used by C2.

RoPE is *not* added to the embeddings; it rotates the query/key vectors inside
every self-attention block, which makes the attention logit depend only on the
relative offset ``m - n``.  Consequently, when ``pos_encoding == "rope"`` the
model adds **no** absolute encoding to the token embeddings.

Cross-attention (decoder -> encoder memory) is left un-rotated: query and key
live in two different sequences, so a "relative distance" between them is not
meaningful.  This is the standard choice in RoPE encoder-decoder models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "build_positional",
]


class SinusoidalPositionalEncoding(nn.Module):
    r"""Fixed sinusoidal absolute positional encoding.

    .. math::
        PE_{(pos, 2i)}   = \sin\!\left(pos / 10000^{2i/d}\right) \\
        PE_{(pos, 2i+1)} = \cos\!\left(pos / 10000^{2i/d}\right)

    The table is a non-trainable buffer, so it moves with ``.to(device)`` and is
    saved in the checkpoint (harmless, and keeps loading simple).
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.0, base: float = 10000.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal PE, got {d_model}")
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
        )  # [d/2]
        pe = torch.zeros(self.max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encodings to ``x``.

        Args:
            x: ``[B, T, d_model]``.
            offset: index of the first position of ``x`` inside the full
                sequence.  Non-zero during incremental (KV-cached) decoding.
        """
        seq_len = x.size(1)
        if offset + seq_len > self.max_len:
            raise ValueError(
                f"Sequence position {offset + seq_len} exceeds max_len={self.max_len}. "
                "Increase --max-src-len / --max-tgt-len headroom."
            )
        pos = self.pe[offset : offset + seq_len].to(dtype=x.dtype, device=x.device)
        return self.dropout(x + pos.unsqueeze(0))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: ``[x1, x2] -> [-x2, x1]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the rotary transform to ``x`` of shape ``[B, H, T, head_dim]``.

    ``cos`` / ``sin`` are ``[1, 1, T, head_dim]`` and already duplicated so that
    the two halves of ``head_dim`` share a frequency.
    """
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    r"""Rotary Position Embedding (RoPE).

    Each 2-dimensional slice of a head is rotated by an angle proportional to its
    absolute position; the dot product of two rotated vectors therefore depends
    only on their *relative* distance:

    .. math::
        \langle R_m q, R_n k \rangle = g(q, k, m - n)

    Args:
        head_dim: per-head width (must be even).
        max_len: largest position pre-computed in the cos/sin cache.
        base: the ``theta`` base (10000 in the original paper).
    """

    def __init__(self, head_dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_len = int(max_len)
        self.base = float(base)

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )  # [head_dim/2]
        t = torch.arange(self.max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [L, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(self, seq_len: int, offset: int, device, dtype):
        if offset + seq_len > self.max_len:
            raise ValueError(
                f"RoPE position {offset + seq_len} exceeds max_len={self.max_len}."
            )
        cos = self.cos_cached[offset : offset + seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(device=device, dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]  # [1, 1, T, head_dim]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys.

        Args:
            q: ``[B, H_q, T, head_dim]``
            k: ``[B, H_kv, T, head_dim]`` (same ``T`` and ``offset`` as ``q``;
               with a KV cache this is only the *new* slice, which is precisely
               why RoPE must be applied before concatenating the cache).
            offset: absolute index of the first position.
        """
        cos, sin = self.get_cos_sin(q.size(-2), offset, q.device, q.dtype)
        q_out = apply_rotary_pos_emb(q, cos, sin)
        if k.size(-2) == q.size(-2):
            k_out = apply_rotary_pos_emb(k, cos, sin)
        else:  # defensive: allow a differing key length
            cos_k, sin_k = self.get_cos_sin(k.size(-2), offset, k.device, k.dtype)
            k_out = apply_rotary_pos_emb(k, cos_k, sin_k)
        return q_out, k_out


def build_positional(kind: str, d_model: int, head_dim: int, max_len: int, dropout: float = 0.0):
    """Return ``(absolute_module_or_None, rope_module_or_None)`` for a config.

    Exactly one of the two is non-``None``:

    * ``"sinusoidal"`` -> additive absolute encoding on the embeddings.
    * ``"rope"``       -> rotary embedding applied inside self-attention.
    """
    key = kind.lower().replace("_", "").replace("-", "")
    if key in {"sinusoidal", "sin", "absolute", "sinusoidalabsolute"}:
        return SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout), None
    if key in {"rope", "rotary"}:
        return None, RotaryPositionalEmbedding(head_dim, max_len=max_len)
    raise ValueError(f"Unknown positional encoding '{kind}' (expected 'sinusoidal' or 'rope')")
