"""Encoder-decoder Transformer assembled from the from-scratch blocks.

Layout is **Pre-LN** (normalise -> sublayer -> residual add), as required by the
brief ("Pre-Layer Normalization"), with a final norm at the top of each stack.
Pre-LN keeps the residual stream un-normalised, which is what lets these small
models train stably without a long warm-up.

    x = x + Dropout(SelfAttn(Norm(x)))
    x = x + Dropout(CrossAttn(Norm(x), memory))     # decoder only
    x = x + Dropout(FFN(Norm(x)))

The four swappable knobs of the ablation all live in the constructor:

===========  =========================================================
``pos``      ``"sinusoidal"`` (C1/C3/C4/C5) or ``"rope"`` (C2)
``attention````"mha"`` (C1/C2/C4/C5) or ``"gqa"`` (C3)
``norm``     ``"layernorm"`` (C1/C2/C3/C5) or ``"rmsnorm"`` (C4)
tokenisation handled outside the model (``dataset.py`` / ``blt.py``)
===========  =========================================================
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from .attention import build_attention
from .norm import build_norm
from .positional import build_positional

__all__ = [
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Seq2SeqTransformer",
    "causal_mask",
    "pad_mask_to_attn_mask",
]


# --------------------------------------------------------------------------- #
# mask helpers
# --------------------------------------------------------------------------- #
def causal_mask(q_len: int, kv_len: int, offset: int, device) -> torch.Tensor:
    """``[1, 1, q_len, kv_len]`` boolean mask, ``True`` where attention is allowed.

    Query ``i`` sits at absolute position ``offset + i`` and may attend every key
    ``j <= offset + i``.  With a KV cache ``offset`` is the number of cached
    steps and ``q_len`` is typically 1.
    """
    i = torch.arange(q_len, device=device).unsqueeze(1) + offset
    j = torch.arange(kv_len, device=device).unsqueeze(0)
    return (j <= i).unsqueeze(0).unsqueeze(0)


def pad_mask_to_attn_mask(pad_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """``[B, T]`` key-validity mask -> ``[B, 1, 1, T]`` for broadcasting."""
    if pad_mask is None:
        return None
    return pad_mask.bool().unsqueeze(1).unsqueeze(1)


# --------------------------------------------------------------------------- #
# sublayers
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    """Position-wise feed-forward network: ``W2 . act(W1 x)``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu",
                 bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act in {"silu", "swish"}:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class EncoderLayer(nn.Module):
    """Pre-LN encoder layer: self-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention: str = "mha",
        n_kv_heads: Optional[int] = None,
        norm: str = "layernorm",
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = build_norm(norm, d_model)
        self.self_attn = build_attention(
            attention, d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout
        )
        self.norm2 = build_norm(norm, d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, rope=None, need_weights: bool = False):
        h, _, w = self.self_attn(
            self.norm1(x), attn_mask=attn_mask, rope=rope, need_weights=need_weights
        )
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, w

#prelayer normalization

class DecoderLayer(nn.Module):
    """Pre-LN decoder layer: causal self-attention + cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention: str = "mha",
        n_kv_heads: Optional[int] = None,
        norm: str = "layernorm",
        activation: str = "gelu",
        d_memory: Optional[int] = None,
    ):
        super().__init__()
        d_memory = d_model if d_memory is None else d_memory
        self.norm1 = build_norm(norm, d_model)
        self.self_attn = build_attention(
            attention, d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout
        )
        self.norm2 = build_norm(norm, d_model)
        self.cross_attn = build_attention(
            attention, d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, is_cross=True
        )
        self.mem_proj = nn.Linear(d_memory, d_model) if d_memory != d_model else nn.Identity()
        self.norm3 = build_norm(norm, d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        rope=None,
        cache: Optional[dict] = None,
        use_cache: bool = False,
        offset: int = 0,
        need_weights: bool = False,
    ):
        past_self = cache.get("self") if cache is not None else None
        past_cross = cache.get("cross") if cache is not None else None

        h, present_self, _ = self.self_attn(
            self.norm1(x),
            attn_mask=self_mask,
            rope=rope,
            past_kv=past_self,
            use_cache=use_cache,
            offset=offset,
        )
        x = x + self.dropout(h)

        cross_w = None
        present_cross = None
        if memory is not None:
            # With a cross-attention cache the projected K/V are already stored,
            # so the memory projection can be skipped entirely.
            mem = memory if past_cross is not None else self.mem_proj(memory)
            h, present_cross, cross_w = self.cross_attn(
                self.norm2(x),
                x_kv=mem,
                attn_mask=cross_mask,
                past_kv=past_cross,
                use_cache=use_cache,
                need_weights=need_weights,
            )
            x = x + self.dropout(h)

        x = x + self.dropout(self.ffn(self.norm3(x)))

        new_cache = None
        if use_cache:
            new_cache = {"self": present_self, "cross": present_cross}
        return x, new_cache, cross_w


# --------------------------------------------------------------------------- #
# stacks
# --------------------------------------------------------------------------- #
class TransformerEncoder(nn.Module):
    """Stack of :class:`EncoderLayer` + final norm (Pre-LN convention)."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention: str = "mha",
        n_kv_heads: Optional[int] = None,
        norm: str = "layernorm",
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model, n_heads, d_ff, dropout, attention, n_kv_heads, norm, activation
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm, d_model)

    def forward(self, x, attn_mask=None, rope=None, need_weights: bool = False):
        weights = []
        for layer in self.layers:
            x, w = layer(x, attn_mask=attn_mask, rope=rope, need_weights=need_weights)
            if need_weights:
                weights.append(w)
        return self.final_norm(x), weights


class TransformerDecoder(nn.Module):
    """Stack of :class:`DecoderLayer` + final norm, with per-layer KV caches."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention: str = "mha",
        n_kv_heads: Optional[int] = None,
        norm: str = "layernorm",
        activation: str = "gelu",
        d_memory: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model, n_heads, d_ff, dropout, attention, n_kv_heads, norm,
                    activation, d_memory,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm, d_model)

    def forward(
        self,
        x,
        memory=None,
        self_mask=None,
        cross_mask=None,
        rope=None,
        caches: Optional[List[dict]] = None,
        use_cache: bool = False,
        offset: int = 0,
        need_weights: bool = False,
    ):
        new_caches: List[dict] = []
        cross_weights = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x, new_cache, cw = layer(
                x,
                memory=memory,
                self_mask=self_mask,
                cross_mask=cross_mask,
                rope=rope,
                cache=cache,
                use_cache=use_cache,
                offset=offset,
                need_weights=need_weights,
            )
            if use_cache:
                new_caches.append(new_cache)
            if need_weights:
                cross_weights.append(cw)
        return self.final_norm(x), (new_caches if use_cache else None), cross_weights


# --------------------------------------------------------------------------- #
# full seq2seq model (configs C1-C4)
# --------------------------------------------------------------------------- #
class Seq2SeqTransformer(nn.Module):
    """Tokenised encoder-decoder Transformer (configurations C1-C4).

    Args:
        src_vocab_size / tgt_vocab_size: vocabulary sizes of the two BPE tokenizers.
        d_model, n_encoder_layers, n_decoder_layers, n_heads, d_ff, dropout: geometry.
        n_kv_heads: only used when ``attention="gqa"``.
        max_len: capacity of the positional tables.
        pos_encoding: ``"sinusoidal"`` | ``"rope"``.
        attention: ``"mha"`` | ``"gqa"``.
        norm: ``"layernorm"`` | ``"rmsnorm"``.
        pad_id / bos_id / eos_id: special token ids (shared by both tokenizers).
        tie_embeddings: tie the decoder input embedding with the output projection.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_kv_heads: Optional[int] = None,
        max_len: int = 1024,
        pos_encoding: str = "sinusoidal",
        attention: str = "mha",
        norm: str = "layernorm",
        activation: str = "gelu",
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.pos_encoding = pos_encoding
        self.attention_kind = attention
        self.norm_kind = norm
        self.max_len = max_len

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        abs_pos, rope = build_positional(pos_encoding, d_model, self.head_dim, max_len, dropout=0.0)
        self.abs_pos = abs_pos          # None when using RoPE
        self.rope = rope                # None when using sinusoidal

        self.encoder = TransformerEncoder(
            n_encoder_layers, d_model, n_heads, d_ff, dropout, attention, n_kv_heads, norm, activation
        )
        self.decoder = TransformerDecoder(
            n_decoder_layers, d_model, n_heads, d_ff, dropout, attention, n_kv_heads, norm, activation
        )

        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings and tgt_vocab_size == self.tgt_embed.num_embeddings:
            self.lm_head.weight = self.tgt_embed.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ init
    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

    # -------------------------------------------------------------- encoding
    def _embed(self, ids: torch.Tensor, embed: nn.Embedding, offset: int = 0) -> torch.Tensor:
        x = embed(ids) * self.embed_scale
        if self.abs_pos is not None:
            x = self.abs_pos(x, offset=offset)
        return self.embed_dropout(x)

    def encode(self, src: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None,
               need_weights: bool = False):
        """``src``: ``[B, S]`` ids; ``src_pad_mask``: ``[B, S]`` bool, ``True`` = real token."""
        x = self._embed(src, self.src_embed)
        attn_mask = pad_mask_to_attn_mask(src_pad_mask)
        memory, weights = self.encoder(x, attn_mask=attn_mask, rope=self.rope,
                                       need_weights=need_weights)
        return (memory, weights) if need_weights else memory

    # -------------------------------------------------------------- decoding
    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
        caches: Optional[List[dict]] = None,
        use_cache: bool = False,
        offset: int = 0,
        need_weights: bool = False,
    ):
        """One decoder pass.  ``offset`` = number of already-cached steps."""
        x = self._embed(tgt_in, self.tgt_embed, offset=offset)

        q_len = tgt_in.size(1)
        kv_len = offset + q_len
        self_mask = causal_mask(q_len, kv_len, offset, tgt_in.device)
        if tgt_pad_mask is not None:
            self_mask = self_mask & tgt_pad_mask.bool().unsqueeze(1).unsqueeze(1)

        cross_mask = pad_mask_to_attn_mask(src_pad_mask)
        h, new_caches, cross_w = self.decoder(
            x,
            memory=memory,
            self_mask=self_mask,
            cross_mask=cross_mask,
            rope=self.rope,
            caches=caches,
            use_cache=use_cache,
            offset=offset,
            need_weights=need_weights,
        )
        logits = self.lm_head(h)
        return logits, new_caches, cross_w

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced training pass -> logits ``[B, T, tgt_vocab]``."""
        memory = self.encode(src, src_pad_mask)
        logits, _, _ = self.decode(tgt_in, memory, src_pad_mask, tgt_pad_mask)
        return logits

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Batched greedy decoding.  Returns ``[B, <=max_new_tokens+1]`` incl. BOS."""
        self.eval()
        device = src.device
        b = src.size(0)
        memory = self.encode(src, src_pad_mask)

        ys = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        caches = None

        for _ in range(max_new_tokens):
            if use_cache and caches is not None:
                step_in = ys[:, -1:]
                offset = ys.size(1) - 1
            else:
                step_in = ys
                offset = 0
            logits, new_caches, _ = self.decode(
                step_in,
                memory,
                src_pad_mask=src_pad_mask,
                caches=caches,
                use_cache=use_cache,
                offset=offset,
            )
            if use_cache:
                caches = new_caches
            next_tok = logits[:, -1].argmax(dim=-1)# greedy decode
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_id), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == self.eos_id)
            if bool(finished.all()):
                break
        return ys

    # ----------------------------------------------------------------- misc
    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))
