"""Attention written from scratch: SDPA, Multi-Head Attention and Grouped-Query Attention.

Nothing here uses ``nn.MultiheadAttention`` / ``nn.Transformer`` /
``F.scaled_dot_product_attention``; the whole thing is matmul + softmax.

Mask convention used everywhere in this project
-----------------------------------------------
Masks are **boolean** tensors broadcastable to ``[B, H, T_q, T_k]`` where
``True`` means *"this key may be attended"* and ``False`` means *"forbidden"*.
Forbidden logits are filled with ``finfo(dtype).min`` rather than ``-inf`` so
that a fully-masked row (which happens for padded *query* positions) produces a
uniform distribution instead of ``NaN``.  Those rows are discarded downstream by
the loss' ``ignore_index``.

Grouped-Query Attention
-----------------------
GQA keeps ``n_heads`` query heads but only ``n_kv_heads`` key/value heads; each
KV head is shared by ``n_heads // n_kv_heads`` query heads.  This shrinks the
K/V projections and, more importantly, the decoding KV-cache by the same factor
- the effect configuration **C3** is designed to measure.
``n_kv_heads == n_heads`` recovers exact MHA, ``n_kv_heads == 1`` is MQA.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "scaled_dot_product_attention",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "repeat_kv",
    "build_attention",
]


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The core operation of the assignment.

    .. math::
        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V

    Args:
        q: ``[..., T_q, d_k]``
        k: ``[..., T_k, d_k]``
        v: ``[..., T_k, d_v]``
        mask: boolean, broadcastable to ``[..., T_q, T_k]``; ``True`` = keep.
        dropout_p: dropout applied to the attention probabilities.
        training: whether dropout is active.

    Returns:
        ``(output [..., T_q, d_v], attn_probs [..., T_q, T_k])``
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)
    out = torch.matmul(attn, v)
    return out, attn


class ScaledDotProductAttention(nn.Module):
    """``nn.Module`` wrapper around :func:`scaled_dot_product_attention`."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = float(dropout)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        return scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p, training=self.training
        )


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``[B, H_kv, T, D]`` to ``[B, H_kv * n_rep, T, D]`` for GQA.

    Uses ``expand`` + ``reshape`` (a view where possible) so no extra memory is
    materialised until the matmul consumes it.
    """
    if n_rep == 1:
        return x
    b, h_kv, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, h_kv, n_rep, t, d)
    return x.reshape(b, h_kv * n_rep, t, d)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional grouped key/value heads and KV caching.

    Args:
        d_model: model width.
        n_heads: number of **query** heads.
        n_kv_heads: number of key/value heads.  ``None`` -> ``n_heads`` (plain MHA).
        dropout: dropout on the attention probabilities and on the output proj.
        bias: whether the four projections carry biases.
        is_cross: marks a cross-attention block.  Cross-attention keys/values
            depend only on the (static) encoder memory, so during incremental
            decoding they are computed once and reused verbatim.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        is_cross: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.is_cross = bool(is_cross)
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=bias)
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ utils
    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope=None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        offset: int = 0,
        need_weights: bool = False,
    ):
        """
        Args:
            x_q: queries ``[B, T_q, d_model]``.
            x_kv: keys/values source ``[B, T_k, d_model]``; ``None`` -> self-attention.
            attn_mask: boolean mask broadcastable to ``[B, H, T_q, T_k]`` (``True`` = keep).
            rope: optional :class:`RotaryPositionalEmbedding` applied to q/k.
            past_kv: ``(k, v)`` from a previous step, each ``[B, H_kv, T_past, D]``.
            use_cache: return the (possibly extended) ``(k, v)`` for reuse.
            offset: absolute position of ``x_q[:, 0]`` (matters for RoPE).
            need_weights: also return the attention probabilities.

        Returns:
            ``(output, present_kv, attn_weights)`` where the last two may be ``None``.
        """
        is_self = x_kv is None
        x_kv = x_q if is_self else x_kv

        q = self._shape_q(self.q_proj(x_q))  # [B, Hq, Tq, D]

        if self.is_cross and past_kv is not None:
            # Encoder memory never changes -> reuse the projected K/V verbatim.
            k, v = past_kv
        else:
            k = self._shape_kv(self.k_proj(x_kv))  # [B, Hkv, Tk_new, D]
            v = self._shape_kv(self.v_proj(x_kv))
            if rope is not None:
                # RoPE is applied to the *new* slice at its absolute positions,
                # before concatenating the cache (already-rotated) below.
                q, k = rope(q, k, offset=offset)
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)

        # Note: cross-attention is never rotated -- query and key live in two
        # different sequences, so a relative distance between them is meaningless
        # (see the rationale in positional.py).

        present = (k, v) if use_cache else None

        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        out, weights = self.attn(q, k_rep, v_rep, mask=attn_mask)
        b, _, t_q, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, t_q, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.out_proj(out))
        return out, present, (weights if need_weights else None)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, cross={self.is_cross}"
        )


class GroupedQueryAttention(MultiHeadAttention):
    """Grouped-Query Attention -- MHA with ``n_kv_heads < n_heads`` (config C3).

    Parameter / cache saving versus MHA::

        KV params  : 2 * d_model * n_kv_heads * head_dim   (instead of 2 * d_model^2)
        KV cache   : 2 * B * n_kv_heads * T * head_dim      (instead of 2 * B * n_heads * T * head_dim)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        is_cross: bool = False,
    ):
        if n_kv_heads is None:
            raise ValueError("GroupedQueryAttention requires an explicit n_kv_heads")
        if n_kv_heads > n_heads:
            raise ValueError(f"n_kv_heads ({n_kv_heads}) cannot exceed n_heads ({n_heads})")
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
            is_cross=is_cross,
        )


def build_attention(
    kind: str,
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int] = None,
    dropout: float = 0.0,
    bias: bool = False,
    is_cross: bool = False,
) -> MultiHeadAttention:
    """Factory used by the config system: ``kind in {"mha", "gqa"}``."""
    key = kind.lower().replace("-", "").replace("_", "")
    if key in {"mha", "multihead", "multiheadattention"}:
        return MultiHeadAttention(
            d_model, n_heads, n_kv_heads=None, dropout=dropout, bias=bias, is_cross=is_cross
        )
    if key in {"gqa", "groupedquery", "groupedqueryattention"}:
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)
        return GroupedQueryAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias, is_cross=is_cross
        )
    raise ValueError(f"Unknown attention '{kind}' (expected 'mha' or 'gqa')")
