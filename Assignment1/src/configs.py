"""The five ablation configurations (Table 1 of the assignment).

C2-C5 each differ from the C1 baseline in **exactly one** component; everything
else (depth, width, heads, d_ff, dropout, optimiser, LR schedule, batch size,
epochs, seed) is shared and comes from the CLI defaults in ``train.py``.

===== ==================== ================= ============= ==============
Cfg   Positional encoding  Attention         Normalisation Tokenisation
===== ==================== ================= ============= ==============
C1    Sinusoidal absolute  Multi-Head        LayerNorm     Subword (BPE)
C2    **RoPE**             Multi-Head        LayerNorm     Subword (BPE)
C3    Sinusoidal absolute  **Grouped-Query** LayerNorm     Subword (BPE)
C4    Sinusoidal absolute  Multi-Head        **RMSNorm**   Subword (BPE)
C5    Sinusoidal absolute  Multi-Head        LayerNorm     **BLT (bytes)**
===== ==================== ================= ============= ==============
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

__all__ = ["AblationConfig", "CONFIGS", "get_config", "build_model"]


@dataclass
class AblationConfig:
    name: str
    changed: str
    pos_encoding: str       # "sinusoidal" | "rope"
    attention: str          # "mha" | "gqa"
    norm: str               # "layernorm" | "rmsnorm"
    tokenization: str       # "bpe" | "byte"
    description: str = ""

    @property
    def is_blt(self) -> bool:
        return self.tokenization in {"byte", "blt"}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["is_blt"] = self.is_blt
        return d


CONFIGS: Dict[str, AblationConfig] = {
    "C1": AblationConfig(
        name="C1", changed="none (base)",
        pos_encoding="sinusoidal", attention="mha", norm="layernorm", tokenization="bpe",
        description="Baseline encoder-decoder transformer.",
    ),
    "C2": AblationConfig(
        name="C2", changed="positional encoding -> RoPE",
        pos_encoding="rope", attention="mha", norm="layernorm", tokenization="bpe",
        description="Rotary position embeddings inside self-attention; no absolute PE.",
    ),
    "C3": AblationConfig(
        name="C3", changed="attention -> Grouped-Query",
        pos_encoding="sinusoidal", attention="gqa", norm="layernorm", tokenization="bpe",
        description="Grouped-Query Attention (fewer KV heads) in every attention block.",
    ),
    "C4": AblationConfig(
        name="C4", changed="normalization -> RMSNorm",
        pos_encoding="sinusoidal", attention="mha", norm="rmsnorm", tokenization="bpe",
        description="RMSNorm replaces LayerNorm at every Pre-LN site.",
    ),
    "C5": AblationConfig(
        name="C5", changed="tokenization -> BLT (token-free)",
        pos_encoding="sinusoidal", attention="mha", norm="layernorm", tokenization="byte",
        description="Byte Latent Transformer: local byte encoder/decoder around a "
                    "patch-level global transformer of identical shape to C1.",
    ),
}


def get_config(name: str) -> AblationConfig:
    key = name.strip().upper()
    if key not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from {sorted(CONFIGS)}")
    return CONFIGS[key]


def build_model(cfg: AblationConfig, args, src_vocab_size: int, tgt_vocab_size: int):
    """Instantiate the model for a configuration.

    C1-C4 -> :class:`~src.models.transformer.Seq2SeqTransformer`
    C5    -> :class:`~src.models.blt.BLTSeq2Seq`
    """
    from .models.blt import BLTSeq2Seq
    from .models.transformer import Seq2SeqTransformer

    if cfg.is_blt:
        return BLTSeq2Seq(
            d_model=args.d_model,
            n_encoder_layers=args.n_layers,
            n_decoder_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            d_local=args.blt_d_local,
            n_local_encoder_layers=args.blt_local_enc_layers,
            n_local_decoder_layers=args.blt_local_dec_layers,
            n_local_heads=args.blt_local_heads,
            patch_size=args.blt_patch_size,
            window=args.blt_window,
            max_bytes=max(args.max_src_len, args.max_tgt_len) + 8,
            max_patch_len=getattr(args, "blt_max_patch", 0),
            norm=cfg.norm,
        )

    return Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_kv_heads=args.n_kv_heads if cfg.attention == "gqa" else None,
        max_len=max(args.max_src_len, args.max_tgt_len) + 8,
        pos_encoding=cfg.pos_encoding,
        attention=cfg.attention,
        norm=cfg.norm,
        tie_embeddings=not args.no_tie_embeddings,
    )
