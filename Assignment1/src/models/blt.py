"""Simplified Byte Latent Transformer (configuration C5).

Pipeline (no vocabulary anywhere -- raw bytes in, raw bytes out)::

    source bytes --> LocalEncoder (windowed, bidirectional)
                 --> PatchPooler  (attention-pool each ENTROPY-DEFINED patch)
                 --> Global Transformer Encoder            |
                                                           | patch-level memory
    target bytes --> LocalEncoder (windowed, CAUSAL)       |
                 --> PatchPooler                           |
                 --> Global Transformer Decoder  <---------+
                 --> LocalDecoder (windowed causal + cross-attn to patches)
                 --> next-byte logits

The global transformer is byte-for-byte the same stack used by C1 (same
``d_model``, depth, heads, sinusoidal PE, LayerNorm, MHA) but it runs on a much
shorter patch sequence, which is the whole point of the comparison: BLT trades a
cheap byte-level periphery for a shorter global sequence.

Entropy-based dynamic patching
------------------------------
Patches are **variable length**, not fixed width. A new patch opens wherever the
next byte is hard to predict::

    H(x_t | x_{t-2}, x_{t-1}) > theta

The entropy comes from a lightweight order-2 byte n-gram model
(:mod:`src.models.entropy`) fit on the training split; no separate neural LM is
trained. :meth:`BLTSeq2Seq.install_entropy` registers its lookup tables as
buffers and switches ``dynamic_patching`` on. Until then the module falls back to
fixed patches of ``patch_size`` bytes, which is what the ``--blt-fixed-patches``
ablation and several unit tests exercise.

Because the n-gram entropy is conditioned on the *context* only, the boundary at
``t`` never depends on ``x_t`` itself. A byte's patch is therefore known before
that byte is generated, which is what makes dynamic patching legal inside an
autoregressive decoder (see the causality note below).

Note on the input bytes
-----------------------
For this task the source is a *binary sequence* written as ASCII ``'0'``/``'1'``
characters. The tokenizer packs every 8 bits into one byte value 0-255, so the
model sees the underlying binary data (32 bytes per 256-character segment) rather
than its 8x longer text encoding.

Causality (the part that is easy to get wrong)
----------------------------------------------
A learned **start patch** is prepended to the target patch sequence, so the
global decoder runs over ``[start, p_1 .. p_M]`` and emits ``[g_0, g_1 .. g_M]``:

* the target LocalEncoder is causal, so its state at byte ``t`` sees bytes ``<= t``;
* ``p_k`` pools only the bytes of patch ``k``, so it sees bytes ``<= e_k`` where
  ``e_k`` is that patch's last byte (fixed OR entropy-defined);
* the global decoder is causal over patches, so ``g_k`` sees bytes ``<= e_{k-1}``
  (and ``g_0`` sees *no* target byte at all -- only the source);
* byte ``t``, which lives in patch ``j``, may attend ``g_0 .. g_j``, and ``g_j``
  saw only bytes ``<= e_{j-1} < s_j <= t``.  Strictly in the past.

  With dynamic patching this still holds because ``j`` itself is decided by
  ``H(x_t | x_{t-2}, x_{t-1})``, a function of the context and never of ``x_t``,
  so a byte's patch id is fixed before that byte exists.

The start patch is what makes the *first* patch work.  It has already
cross-attended to the encoder memory, so those bytes are generated with full
source information but zero target leakage.  Two failure modes it avoids:
without any legal slot the first bytes' cross-attention rows would be fully
masked -- and a fully-masked softmax row is *uniform*, silently averaging over
future patches -- while a constant learned vector would leave the opening bytes
of every output unconditioned on the source.

``tests/test_modules.py`` verifies all of this numerically:
``test_blt_causality`` and ``test_blt_dynamic_patching_is_causal`` perturb future
target bytes and check earlier logits do not move (fixed and dynamic patching
respectively); ``test_patch_boundary_depends_only_on_past`` perturbs ``x_t`` and
checks its own patch id is unchanged; ``test_blt_source_reaches_first_byte``
checks the source does reach byte 0.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import build_norm
from .positional import SinusoidalPositionalEncoding
from .transformer import (
    DecoderLayer,
    EncoderLayer,
    TransformerDecoder,
    TransformerEncoder,
    pad_mask_to_attn_mask,
)

__all__ = [
    "BYTE_PAD_ID",
    "BYTE_BOS_ID",
    "BYTE_EOS_ID",
    "BYTE_OFFSET",
    "BYTE_VOCAB_SIZE",
    "local_window_mask",
    "LocalEncoder",
    "PatchPooler",
    "LocalDecoder",
    "BLTSeq2Seq",
]

# ---- byte "vocabulary": 3 specials + the 256 raw byte values -----------------
BYTE_PAD_ID = 0
BYTE_BOS_ID = 1
BYTE_EOS_ID = 2
BYTE_OFFSET = 3           # raw byte b  ->  token id  b + BYTE_OFFSET
BYTE_VOCAB_SIZE = 256 + BYTE_OFFSET   # 259


def local_window_mask(seq_len: int, window: int, causal: bool, device) -> torch.Tensor:
    """``[1, 1, T, T]`` boolean mask for *local* attention.

    Args:
        window: half-width.  ``causal`` -> query ``i`` sees ``[i-window+1, i]``;
            otherwise it sees ``[i-window+1, i+window-1]``.  ``window <= 0``
            disables the restriction (full attention).
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    if causal:
        m = j <= i
        if window and window > 0:
            m = m & ((i - j) < window)
    else:
        if window and window > 0:
            m = (i - j).abs() < window
        else:
            m = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return m.unsqueeze(0).unsqueeze(0)


def pad_to_multiple(
    ids: torch.Tensor, pad_mask: Optional[torch.Tensor], multiple: int, pad_id: int = BYTE_PAD_ID
):
    """Right-pad ``ids``/``pad_mask`` so the length is a multiple of ``multiple``."""
    t = ids.size(1)
    rem = (-t) % multiple
    if rem:
        ids = F.pad(ids, (0, rem), value=pad_id)
        if pad_mask is not None:
            pad_mask = F.pad(pad_mask, (0, rem), value=False)
    return ids, pad_mask, t


class LocalEncoder(nn.Module):
    """Byte-level local transformer: embeds bytes and mixes them inside a window.

    ``causal=True`` is used on the target side (so the pooled patch vectors stay
    causal), ``causal=False`` on the source side.
    """

    def __init__(
        self,
        d_local: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        window: int = 64,
        causal: bool = False,
        max_len: int = 2048,
        vocab_size: int = BYTE_VOCAB_SIZE,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.d_local = d_local
        self.window = window
        self.causal = causal
        self.embed = nn.Embedding(vocab_size, d_local, padding_idx=BYTE_PAD_ID)
        self.embed_scale = math.sqrt(d_local)
        self.pos = SinusoidalPositionalEncoding(d_local, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_local, n_heads, d_ff, dropout,
                    attention="mha", n_kv_heads=None, norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm, d_local)

    def forward(self, byte_ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        """``byte_ids``: ``[B, T]`` -> byte hidden states ``[B, T, d_local]``."""
        x = self.dropout(self.pos(self.embed(byte_ids) * self.embed_scale))
        mask = local_window_mask(byte_ids.size(1), self.window, self.causal, byte_ids.device)
        if pad_mask is not None:
            mask = mask & pad_mask_to_attn_mask(pad_mask)
        for layer in self.layers:
            x, _ = layer(x, attn_mask=mask)
        return self.final_norm(x)


class PatchPooler(nn.Module):
    """Pool every ``patch_size`` consecutive byte states into one patch vector.

    Uses learned attention pooling *inside* each patch (a scalar score per byte,
    softmax over the ``P`` bytes of that patch), then projects ``d_local ->
    d_model``.  Pooling never crosses a patch boundary, which is what keeps the
    causal argument in the module docstring valid.
    """

    def __init__(self, d_local: int, d_model: int, patch_size: int, norm: str = "layernorm"):
        super().__init__()
        self.patch_size = patch_size
        self.score = nn.Linear(d_local, 1)
        self.proj = nn.Linear(d_local, d_model)
        self.norm = build_norm(norm, d_model)

    def forward(
        self,
        h: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        seg: Optional[torch.Tensor] = None,
        n_patches: Optional[int] = None,
    ):
        """
        Args:
            h: ``[B, T, d_local]``.
            pad_mask: ``[B, T]`` bool, ``True`` = real byte.
            seg: ``[B, T]`` int patch index per byte (``-1`` = padding). When
                given, patches are **variable length** and defined by ``seg``
                (entropy-based dynamic patching). When ``None``, falls back to
                fixed patches of ``patch_size`` bytes.
            n_patches: ``M``; inferred from ``seg`` when omitted.

        Returns:
            ``(patches [B, M, d_model], patch_mask [B, M])``
        """
        b, t, d = h.shape

        if seg is None:
            # ---- fixed-width fallback (kept for unit tests / ablation) -----
            p = self.patch_size
            if t % p != 0:
                raise ValueError(f"sequence length {t} is not a multiple of patch_size {p}")
            m = t // p
            hp = h.view(b, m, p, d)
            scores = self.score(hp).squeeze(-1)
            if pad_mask is not None:
                mp = pad_mask.view(b, m, p)
                scores = scores.masked_fill(~mp, torch.finfo(scores.dtype).min)
                patch_mask = mp.any(dim=-1)
            else:
                patch_mask = torch.ones(b, m, dtype=torch.bool, device=h.device)
            weights = torch.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * hp).sum(dim=2)
            return self.norm(self.proj(pooled)), patch_mask

        # ---- entropy-based dynamic patching --------------------------------
        # Attention-pool over each patch's *members*, which are now variable in
        # number. `member` is [B, M, T]; T is a few tens of bytes here, so the
        # dense mask is far cheaper than a gather/scatter over ragged segments.
        m = int(n_patches if n_patches is not None else int(seg.max().item()) + 1)
        m = max(m, 1)
        ar = torch.arange(m, device=h.device).view(1, m, 1)
        member = seg.unsqueeze(1) == ar                       # [B, M, T]
        if pad_mask is not None:
            member = member & pad_mask.unsqueeze(1)

        scores = self.score(h).squeeze(-1)                    # [B, T]
        s = scores.unsqueeze(1).expand(b, m, t)
        s = s.masked_fill(~member, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=-1)
        # A patch with no members yields a uniform (meaningless) row; zeroing it
        # makes the pooled vector exactly 0 and `patch_mask` masks it downstream.
        w = w * member.to(w.dtype)
        pooled = torch.bmm(w, h)                              # [B, M, d_local]
        patch_mask = member.any(dim=-1)                       # [B, M]
        return self.norm(self.proj(pooled)), patch_mask


class LocalDecoder(nn.Module):
    """Byte-level decoder conditioned on (strictly past) global patch states.

    Each layer is a Pre-LN decoder layer whose self-attention is causal+windowed
    over bytes and whose cross-attention looks at the patch memory through the
    causal patch mask described in the module docstring.
    """

    def __init__(
        self,
        d_local: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        window: int = 64,
        patch_size: int = 4,
        vocab_size: int = BYTE_VOCAB_SIZE,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.window = window
        self.cond_proj = nn.Linear(d_model, d_local)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_local, n_heads, d_ff, dropout,
                    attention="mha", n_kv_heads=None, norm=norm, d_memory=d_model,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = build_norm(norm, d_local)
        self.head = nn.Linear(d_local, vocab_size, bias=False)

    def forward(
        self,
        byte_hidden: torch.Tensor,       # [B, T, d_local]   (causal over target bytes)
        patch_memory: torch.Tensor,      # [B, M+1, d_model] (g_0 = start patch)
        patch_mask: torch.Tensor,        # [B, M+1] bool
        seg: Optional[torch.Tensor] = None,   # [B, T] patch index per byte
    ) -> torch.Tensor:
        b, t, _ = byte_hidden.shape
        device = byte_hidden.device

        # Byte t living in patch j is conditioned on slot j of the global output
        # [g_0, g_1 .. g_M]: g_0 is the start patch (no target bytes at all) and
        # g_j pooled only patches 0..j-1, i.e. bytes strictly before patch j
        # begins -- so strictly before t. This is the newest state that is still
        # legally in this byte's past, for fixed OR variable-length patches.
        if seg is None:
            idx = (torch.arange(t, device=device) // self.patch_size).expand(b, t)
        else:
            idx = seg.clamp(min=0)                                      # [B, T]

        d_model = patch_memory.size(-1)
        cond = torch.gather(patch_memory, 1, idx.unsqueeze(-1).expand(b, t, d_model))
        x = byte_hidden + self.cond_proj(cond)

        self_mask = local_window_mask(t, self.window, causal=True, device=device)

        m_aug = patch_memory.size(1)
        slots = torch.arange(m_aug, device=device).view(1, 1, m_aug)    # [1, 1, M+1]
        allowed = slots <= idx.unsqueeze(-1)                            # [B, T, M+1]
        cross_mask = allowed.unsqueeze(1) & patch_mask[:, None, None, :]

        for layer in self.layers:
            x, _, _ = layer(
                x, memory=patch_memory, self_mask=self_mask, cross_mask=cross_mask
            )
        return self.head(self.final_norm(x))


class BLTSeq2Seq(nn.Module):
    """Token-free byte-level encoder-decoder (configuration **C5**).

    The ``global_*`` stacks are identical in shape to the C1 baseline so the
    ablation isolates the tokenisation change.

    Args:
        d_model / n_encoder_layers / n_decoder_layers / n_heads / d_ff: global
            transformer geometry (kept equal to the C1 baseline).
        d_local / n_local_encoder_layers / n_local_decoder_layers / n_local_heads:
            the cheap byte-level periphery.
        patch_size: bytes per patch (``P``).  Larger ``P`` -> shorter global
            sequence, cheaper, but a coarser bottleneck.
        window: half-width of the local attention window.
        max_bytes: capacity of the byte-level positional table.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        d_local: int = 128,
        n_local_encoder_layers: int = 1,
        n_local_decoder_layers: int = 2,
        n_local_heads: int = 4,
        d_ff_local: Optional[int] = None,
        patch_size: int = 4,
        window: int = 64,
        max_bytes: int = 2048,
        norm: str = "layernorm",
        vocab_size: int = BYTE_VOCAB_SIZE,
        max_patch_len: int = 0,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.d_model = d_model
        self.d_local = d_local
        self.max_bytes = max_bytes
        self.vocab_size = vocab_size
        self.pad_id, self.bos_id, self.eos_id = BYTE_PAD_ID, BYTE_BOS_ID, BYTE_EOS_ID
        d_ff_local = d_ff_local or (4 * d_local)
        max_patches = max_bytes + 2   # dynamic patches: worst case one byte each

        # Entropy tables for dynamic patching. They are registered HERE, with
        # neutral defaults, rather than inside install_entropy(): a buffer that
        # only appears after install_entropy() would be an "unexpected key" when
        # a freshly-built model loads a trained checkpoint, so --eval-only, --ckpt
        # and anything downloading from the Hub would break.
        # A threshold of +inf means "never open a patch", which is how
        # `dynamic_patching` reports False until real tables are installed.
        max_uniform = math.log2(256)
        for tag in ("src", "tgt"):
            self.register_buffer(f"{tag}_H1", torch.full((256,), max_uniform))
            self.register_buffer(f"{tag}_H2", torch.full((256 * 256,), max_uniform))
            self.register_buffer(f"{tag}_H0", torch.tensor(max_uniform))
            self.register_buffer(f"{tag}_thr", torch.tensor(float("inf")))
        self.max_patch_len = int(max_patch_len)

        # ---- source side -----------------------------------------------------
        self.src_local_encoder = LocalEncoder(
            d_local, n_local_encoder_layers, n_local_heads, d_ff_local, dropout,
            window=window, causal=False, max_len=max_bytes, vocab_size=vocab_size, norm=norm,
        )
        self.src_pooler = PatchPooler(d_local, d_model, self.patch_size, norm=norm)

        # ---- target side -----------------------------------------------------
        self.tgt_local_encoder = LocalEncoder(
            d_local, n_local_encoder_layers, n_local_heads, d_ff_local, dropout,
            window=window, causal=True, max_len=max_bytes, vocab_size=vocab_size, norm=norm,
        )
        self.tgt_pooler = PatchPooler(d_local, d_model, self.patch_size, norm=norm)

        # ---- global (patch-level) transformer -- same shape as the C1 model ---
        self.patch_pos = SinusoidalPositionalEncoding(d_model, max_len=max_patches)
        self.global_encoder = TransformerEncoder(
            n_encoder_layers, d_model, n_heads, d_ff, dropout,
            attention="mha", n_kv_heads=None, norm=norm,
        )
        self.global_decoder = TransformerDecoder(
            n_decoder_layers, d_model, n_heads, d_ff, dropout,
            attention="mha", n_kv_heads=None, norm=norm,
        )

        # ---- byte-level output decoder --------------------------------------
        self.local_decoder = LocalDecoder(
            d_local, d_model, n_local_decoder_layers, n_local_heads, d_ff_local, dropout,
            window=window, patch_size=self.patch_size, vocab_size=vocab_size, norm=norm,
        )

        # learned start patch, prepended to the target patch sequence (see decode)
        self.start_patch = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_patch, std=0.02)

        self.apply(self._init_weights)

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

    # ------------------------------------------------------------------ encode
    def encode(self, src_bytes: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None):
        """Bytes -> patch-level encoder memory.

        Returns ``(memory [B, Ms, d_model], memory_mask [B, Ms])``.
        """
        if src_pad_mask is None:
            src_pad_mask = src_bytes.ne(self.pad_id)

        if self.dynamic_patching:
            seg = self._segments(src_bytes, "src", pad_mask=src_pad_mask)
            m = int(seg.max().item()) + 1 if seg.numel() else 1
            h = self.src_local_encoder(src_bytes, src_pad_mask)
            patches, patch_mask = self.src_pooler(h, src_pad_mask, seg=seg, n_patches=m)
        else:
            src_bytes, src_pad_mask, _ = pad_to_multiple(
                src_bytes, src_pad_mask, self.patch_size
            )
            h = self.src_local_encoder(src_bytes, src_pad_mask)
            patches, patch_mask = self.src_pooler(h, src_pad_mask)

        patches = self.patch_pos(patches)
        memory, _ = self.global_encoder(patches, attn_mask=pad_mask_to_attn_mask(patch_mask))
        return memory, patch_mask

    # ------------------------------------------------------------------ decode
    def decode(
        self,
        tgt_in_bytes: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced byte logits for ``tgt_in_bytes`` ``[B, T]`` -> ``[B, T, V]``.

        Any right-padding needed to reach a multiple of ``patch_size`` is added
        internally and stripped from the returned logits.
        """
        t_orig = tgt_in_bytes.size(1)

        if self.dynamic_patching:
            # seg[t] is decided by H(x_t | x_{t-2}, x_{t-1}) -- a function of the
            # CONTEXT only, never of x_t itself. So the patch a byte belongs to
            # is known before that byte is generated, which is exactly what makes
            # dynamic patching legal in an autoregressive decoder.
            tgt_bytes = tgt_in_bytes
            seg = self._segments(tgt_bytes, "tgt", pad_mask=None)
            m = int(seg.max().item()) + 1 if seg.numel() else 1
            hb = self.tgt_local_encoder(tgt_bytes, pad_mask=None)
            patches, _ = self.tgt_pooler(hb, pad_mask=None, seg=seg, n_patches=m)
        else:
            seg = None
            tgt_bytes, _, _ = pad_to_multiple(tgt_in_bytes, None, self.patch_size)
            hb = self.tgt_local_encoder(tgt_bytes, pad_mask=None)
            patches, _ = self.tgt_pooler(hb, pad_mask=None)
        b, m, _ = patches.shape

        # Prepend a learned start patch, then run the global decoder over
        # [start, p_1..p_M].  The start slot cross-attends to the source but has
        # seen no target byte, so it is the legal conditioning signal for the
        # bytes of the very first patch -- they get full source information
        # without any leakage.
        start = self.start_patch.expand(b, 1, self.d_model).to(patches.dtype)
        patches = self.patch_pos(torch.cat([start, patches], dim=1))         # [B, M+1, d_model]

        m1 = m + 1
        self_mask = torch.tril(
            torch.ones(m1, m1, dtype=torch.bool, device=patches.device)
        ).unsqueeze(0).unsqueeze(0)
        g, _, _ = self.global_decoder(
            patches,
            memory=memory,
            self_mask=self_mask,
            cross_mask=pad_mask_to_attn_mask(memory_mask),
        )
        # g[:, 0]  -> source only (no target bytes)
        # g[:, k]  -> source + target bytes <= k*P - 1   (k >= 1)

        patch_mask = torch.ones(b, m + 1, dtype=torch.bool, device=g.device)
        logits = self.local_decoder(hb, g, patch_mask, seg=seg)              # [B, T_pad, V]
        return logits[:, :t_orig]

    def forward(
        self,
        src_bytes: torch.Tensor,
        tgt_in_bytes: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,   # unused; kept for API symmetry
    ) -> torch.Tensor:
        memory, memory_mask = self.encode(src_bytes, src_pad_mask)
        return self.decode(tgt_in_bytes, memory, memory_mask)

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def greedy_decode(
        self,
        src_bytes: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        use_cache: bool = False,   # accepted for API parity; BLT recomputes the prefix
    ) -> torch.Tensor:
        """Batched greedy byte-by-byte decoding.

        The encoder runs once; the decoder side is recomputed for the growing
        prefix at every step.  That is O(T) forward passes -- acceptable because
        the local stacks are small and windowed, and it keeps the notoriously
        fiddly patch-boundary cache logic out of the evaluation path.
        """
        self.eval()
        device = src_bytes.device
        b = src_bytes.size(0)
        memory, memory_mask = self.encode(src_bytes, src_pad_mask)

        ys = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        max_new_tokens = min(max_new_tokens, self.max_bytes - 2)
        for _ in range(max_new_tokens):
            logits = self.decode(ys, memory, memory_mask)
            next_tok = logits[:, -1].argmax(dim=-1)#greedy decode
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_id), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == self.eos_id)
            if bool(finished.all()):
                break
        return ys

    # ------------------------------------------------- entropy-based patching
    def install_entropy(
        self,
        src_model,
        tgt_model,
        max_patch_len: int = 0,
    ) -> None:
        """Switch on entropy-based dynamic patching (BLT's actual mechanism).

        Args:
            src_model / tgt_model: fitted
                :class:`~src.models.entropy.ByteNgramEntropyModel` for the source
                and target byte streams, each already calibrated
                (``.threshold`` set).
            max_patch_len: optional hard cap on patch length; 0 = uncapped.

        The lookup tables are registered as (non-trainable) buffers, so they move
        with ``.to(device)`` and are saved in the checkpoint -- which means a
        reloaded model reproduces exactly the same patch boundaries.
        """
        for tag, mdl in (("src", src_model), ("tgt", tgt_model)):
            H1, H2, H0 = mdl.as_tensors()
            dev = getattr(self, f"{tag}_H1").device
            getattr(self, f"{tag}_H1").copy_(H1.to(dev))
            getattr(self, f"{tag}_H2").copy_(H2.to(dev))
            getattr(self, f"{tag}_H0").fill_(float(H0))
            getattr(self, f"{tag}_thr").fill_(float(mdl.threshold))
        self.max_patch_len = int(max_patch_len)

    @property
    def dynamic_patching(self) -> bool:
        """True once real entropy tables are installed.

        Derived from the threshold rather than stored as a flag, so it survives a
        state_dict round-trip without adding a buffer key that older checkpoints
        would be missing.
        """
        return bool(torch.isfinite(self.src_thr).item())

    def _segments(self, ids: torch.Tensor, tag: str, pad_mask=None) -> torch.Tensor:
        """Entropy-based patch ids for ``ids`` ``[B, T]`` -> ``[B, T]``."""
        from .entropy import segment_from_entropy
        return segment_from_entropy(
            ids,
            H1=getattr(self, f"{tag}_H1"),
            H2=getattr(self, f"{tag}_H2"),
            H0=float(getattr(self, f"{tag}_H0").item()),
            threshold=float(getattr(self, f"{tag}_thr").item()),
            offset=BYTE_OFFSET,
            n_special=BYTE_OFFSET,
            pad_mask=pad_mask,
            max_patch=self.max_patch_len,
        )

    @torch.no_grad()
    def patch_stats(self, ids: torch.Tensor, tag: str, pad_mask=None) -> dict:
        """Patch-length statistics for a batch -- used for logging/reporting."""
        seg = self._segments(ids, tag, pad_mask=pad_mask)
        valid = seg >= 0
        n_patches = (seg.max(dim=1).values + 1).float()
        lens = []
        for i in range(seg.size(0)):
            s = seg[i][valid[i]]
            if s.numel():
                lens.append(torch.bincount(s).float())
        if not lens:
            return {}
        allv = torch.cat(lens)
        return {
            "patches_per_seq": float(n_patches.mean()),
            "mean_patch_len": float(allv.mean()),
            "max_patch_len": float(allv.max()),
            "min_patch_len": float(allv.min()),
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
