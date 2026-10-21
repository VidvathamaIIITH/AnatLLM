"""From-scratch transformer building blocks."""

from .norm import LayerNorm, RMSNorm, build_norm
from .positional import (
    SinusoidalPositionalEncoding,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from .attention import (
    scaled_dot_product_attention,
    ScaledDotProductAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    build_attention,
)
from .transformer import (
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Seq2SeqTransformer,
    causal_mask,
)
from .blt import LocalEncoder, LocalDecoder, PatchPooler, BLTSeq2Seq

__all__ = [
    "LayerNorm",
    "RMSNorm",
    "build_norm",
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "scaled_dot_product_attention",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "build_attention",
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Seq2SeqTransformer",
    "causal_mask",
    "LocalEncoder",
    "LocalDecoder",
    "PatchPooler",
    "BLTSeq2Seq",
]
