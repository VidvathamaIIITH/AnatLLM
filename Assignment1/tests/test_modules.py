"""Self-tests for every from-scratch component.

Run either way::

    python -m tests.test_modules      # plain runner, no pytest needed
    pytest tests/test_modules.py -q

These are the checks that actually matter for this assignment: that the custom
LayerNorm/RMSNorm match their mathematical definition, that GQA collapses to MHA
when ``n_kv_heads == n_heads``, that RoPE really is relative, that the decoder
cannot see the future, that KV-cached greedy decoding equals full recomputation,
that the BLT byte->patch->byte path is strictly causal, and that the tokenizer
round-trips losslessly.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import CONFIGS, build_model, get_config  # noqa: E402
from src.models.attention import (  # noqa: E402
    GroupedQueryAttention,
    MultiHeadAttention,
    scaled_dot_product_attention,
)
from src.models.blt import BLTSeq2Seq, BYTE_BOS_ID, BYTE_VOCAB_SIZE  # noqa: E402
from src.models.norm import LayerNorm, RMSNorm  # noqa: E402
from src.models.positional import (  # noqa: E402
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from src.models.transformer import Seq2SeqTransformer  # noqa: E402
from src.tokenizer import BPETokenizer, ByteTokenizer  # noqa: E402
from src.utils import (  # noqa: E402
    bit_accuracy,
    compute_metrics,
    corpus_bleu,
    detect_binary_mode,
    levenshtein,
    sequence_accuracy,
)

TOL = 2e-5


def _close(a, b, tol=TOL, msg=""):
    diff = (a - b).abs().max().item()
    assert diff < tol, f"{msg} max|diff|={diff:.3e} (tol {tol:.1e})"
    return diff


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def test_layernorm_matches_reference():
    torch.manual_seed(0)
    d = 32
    x = torch.randn(4, 7, d)
    ours = LayerNorm(d)
    ref = nn.LayerNorm(d)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    _close(ours(x), ref(x), msg="LayerNorm vs nn.LayerNorm:")

    ours.weight.data.uniform_(0.5, 1.5)
    ours.bias.data.uniform_(-0.5, 0.5)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    _close(ours(x), ref(x), msg="LayerNorm (affine) vs nn.LayerNorm:")


def test_rmsnorm_matches_definition():
    torch.manual_seed(0)
    d = 32
    x = torch.randn(3, 5, d)
    norm = RMSNorm(d, eps=1e-6)
    norm.weight.data.uniform_(0.5, 1.5)
    manual = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    _close(norm(x), manual, msg="RMSNorm vs manual:")
    # RMSNorm must NOT re-center: a constant shift changes the output
    assert not torch.allclose(norm(x), norm(x + 1.0), atol=1e-3), "RMSNorm looks like LayerNorm"


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
def test_sdpa_matches_torch_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 9, 16) for _ in range(3))
    out, attn = scaled_dot_product_attention(q, k, v)
    # reference only -- the model itself never calls this
    ref = F.scaled_dot_product_attention(q, k, v)
    _close(out, ref, msg="SDPA vs F.scaled_dot_product_attention:")
    _close(attn.sum(-1), torch.ones_like(attn.sum(-1)), msg="attention rows must sum to 1:")


def test_sdpa_mask_is_respected():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 5, 8) for _ in range(3))
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))[None, None]
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    upper = attn.masked_select(~mask.expand_as(attn))
    assert float(upper.abs().max()) < 1e-6, "masked positions received attention mass"


def test_gqa_equals_mha_when_kv_heads_match():
    torch.manual_seed(0)
    d, h = 64, 8
    mha = MultiHeadAttention(d, h).eval()
    gqa = GroupedQueryAttention(d, h, n_kv_heads=h).eval()
    gqa.load_state_dict(mha.state_dict())
    x = torch.randn(3, 11, d)
    a, _, _ = mha(x)
    b, _, _ = gqa(x)
    _close(a, b, msg="GQA(n_kv=n_heads) vs MHA:")


def test_gqa_shrinks_kv_projections():
    d, h = 64, 8
    mha = MultiHeadAttention(d, h)
    gqa = GroupedQueryAttention(d, h, n_kv_heads=2)
    assert gqa.k_proj.weight.shape[0] == mha.k_proj.weight.shape[0] // 4
    assert gqa.n_rep == 4
    x = torch.randn(2, 6, d)
    out, present, _ = gqa(x, use_cache=True)
    assert out.shape == (2, 6, d)
    assert present[0].shape == (2, 2, 6, d // h), "GQA cache must have n_kv_heads heads"
    n_mha = sum(p.numel() for p in mha.parameters())
    n_gqa = sum(p.numel() for p in gqa.parameters())
    assert n_gqa < n_mha, "GQA should have fewer parameters than MHA"


# --------------------------------------------------------------------------- #
# positional
# --------------------------------------------------------------------------- #
def test_sinusoidal_properties():
    pe = SinusoidalPositionalEncoding(32, max_len=100)
    x = torch.zeros(1, 10, 32)
    out = pe(x, offset=0)
    out_shift = pe(x, offset=3)
    _close(out[:, 3:], out_shift[:, :7], msg="sinusoidal offset must shift the table:")
    assert torch.isfinite(out).all()


def test_rope_is_relative():
    torch.manual_seed(0)
    head_dim = 16
    rope = RotaryPositionalEmbedding(head_dim, max_len=128)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def dot(m, n):
        qm, _ = rope(q, q, offset=m)
        _, kn = rope(k, k, offset=n)
        return float((qm * kn).sum())

    # same relative distance => same score, different absolute positions
    assert abs(dot(5, 3) - dot(20, 18)) < 1e-4, "RoPE score must depend only on m-n"
    assert abs(dot(5, 3) - dot(5, 4)) > 1e-5, "RoPE must actually depend on the distance"


def test_rope_preserves_norm():
    torch.manual_seed(0)
    rope = RotaryPositionalEmbedding(16, max_len=64)
    q = torch.randn(2, 3, 7, 16)
    q_rot, _ = rope(q, q, offset=2)
    _close(q_rot.norm(dim=-1), q.norm(dim=-1), tol=1e-4,
           msg="RoPE is a rotation and must preserve norms:")


# --------------------------------------------------------------------------- #
# seq2seq transformer
# --------------------------------------------------------------------------- #
def _tiny_model(**kw):
    defaults = dict(
        src_vocab_size=40, tgt_vocab_size=40, d_model=32, n_encoder_layers=2,
        n_decoder_layers=2, n_heads=4, d_ff=64, dropout=0.0, max_len=64,
    )
    defaults.update(kw)
    return Seq2SeqTransformer(**defaults).eval()


def test_forward_shapes_and_gradients():
    for pos in ("sinusoidal", "rope"):
        for attn, kv in (("mha", None), ("gqa", 2)):
            for norm in ("layernorm", "rmsnorm"):
                model = _tiny_model(pos_encoding=pos, attention=attn, n_kv_heads=kv, norm=norm)
                model.train()
                src = torch.randint(4, 40, (2, 9))
                tgt_in = torch.randint(4, 40, (2, 7))
                logits = model(src, tgt_in, src.ne(0), tgt_in.ne(0))
                assert logits.shape == (2, 7, 40), f"{pos}/{attn}/{norm}: {logits.shape}"
                logits.sum().backward()
                missing = [
                    n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is None
                ]
                # the sinusoidal table is a buffer, so every *parameter* must get a grad
                assert not missing, f"{pos}/{attn}/{norm}: no gradient for {missing[:3]}"


def test_decoder_is_causal():
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randint(4, 40, (2, 9))
    tgt = torch.randint(4, 40, (2, 10))
    with torch.no_grad():
        base = model(src, tgt, src.ne(0), tgt.ne(0))
        tgt2 = tgt.clone()
        tgt2[:, 6:] = torch.randint(4, 40, (2, 4))
        other = model(src, tgt2, src.ne(0), tgt2.ne(0))
    _close(base[:, :6], other[:, :6], msg="decoder leaked future target tokens:")
    assert (base[:, 6:] - other[:, 6:]).abs().max() > 1e-6, "positions 6+ should have changed"


def test_encoder_ignores_padding():
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randint(4, 40, (1, 6))
    padded = torch.cat([src, torch.zeros(1, 5, dtype=torch.long)], dim=1)
    with torch.no_grad():
        m1 = model.encode(src, src.ne(0))
        m2 = model.encode(padded, padded.ne(0))
    _close(m1, m2[:, :6], tol=1e-4, msg="padding changed the encoder output:")


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    for pos in ("sinusoidal", "rope"):
        for attn, kv in (("mha", None), ("gqa", 2)):
            model = _tiny_model(pos_encoding=pos, attention=attn, n_kv_heads=kv)
            src = torch.randint(4, 40, (3, 8))
            with torch.no_grad():
                cached = model.greedy_decode(src, src.ne(0), max_new_tokens=12, use_cache=True)
                full = model.greedy_decode(src, src.ne(0), max_new_tokens=12, use_cache=False)
            assert torch.equal(cached, full), (
                f"{pos}/{attn}: cached decoding diverged from full recomputation\n"
                f"{cached}\n{full}"
            )


def test_greedy_decode_stops_at_eos():
    torch.manual_seed(0)
    model = _tiny_model()
    src = torch.randint(4, 40, (2, 6))
    out = model.greedy_decode(src, src.ne(0), max_new_tokens=15)
    assert out.shape[0] == 2 and out.shape[1] <= 16
    assert int(out[0, 0]) == model.bos_id


# --------------------------------------------------------------------------- #
# BLT
# --------------------------------------------------------------------------- #
def _tiny_blt(patch_size=4):
    return BLTSeq2Seq(
        d_model=32, n_encoder_layers=2, n_decoder_layers=2, n_heads=4, d_ff=64,
        dropout=0.0, d_local=16, n_local_encoder_layers=1, n_local_decoder_layers=1,
        n_local_heads=2, patch_size=patch_size, window=8, max_bytes=128,
    ).eval()


def test_blt_shapes():
    model = _tiny_blt()
    src = torch.randint(3, BYTE_VOCAB_SIZE, (2, 13))
    tgt = torch.randint(3, BYTE_VOCAB_SIZE, (2, 11))
    logits = model(src, tgt, src.ne(0), tgt.ne(0))
    assert logits.shape == (2, 11, BYTE_VOCAB_SIZE), logits.shape
    logits.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient for {missing[:5]}"


def test_blt_causality():
    """Perturbing byte t must not change any logit at position < t."""
    torch.manual_seed(0)
    for patch_size in (1, 2, 4):
        model = _tiny_blt(patch_size)
        src = torch.randint(3, 200, (2, 16))
        tgt = torch.randint(3, 200, (2, 16))
        cut = 9
        with torch.no_grad():
            base = model(src, tgt, src.ne(0), tgt.ne(0))
            tgt2 = tgt.clone()
            tgt2[:, cut:] = torch.randint(3, 200, (2, 16 - cut))
            other = model(src, tgt2, src.ne(0), tgt2.ne(0))
        _close(base[:, :cut], other[:, :cut], tol=1e-5,
               msg=f"BLT (P={patch_size}) leaked future target bytes:")
        assert (base[:, cut:] - other[:, cut:]).abs().max() > 1e-6


def test_blt_length_invariance():
    """Logits for a prefix must not depend on how much comes after it.

    This is what makes step-by-step greedy decoding equal to a single
    teacher-forced pass, i.e. the generation loop is sound.
    """
    torch.manual_seed(0)
    model = _tiny_blt(patch_size=4)
    src = torch.randint(3, 200, (1, 12))
    tgt = torch.randint(3, 200, (1, 14))
    with torch.no_grad():
        full = model(src, tgt, src.ne(0), tgt.ne(0))
        short = model(src, tgt[:, :7], src.ne(0), tgt[:, :7].ne(0))
    _close(full[:, :7], short, tol=1e-5, msg="BLT prefix logits changed with sequence length:")


def test_blt_source_reaches_first_byte():
    """Bytes of the *first* patch must still be conditioned on the source.

    The start patch exists precisely so that byte 0 is not generated blind:
    changing the source must move the logits at every position, including 0.
    """
    torch.manual_seed(0)
    model = _tiny_blt(patch_size=4)
    tgt = torch.randint(3, 200, (1, 12))
    src_a = torch.randint(3, 200, (1, 16))
    src_b = torch.randint(3, 200, (1, 16))
    with torch.no_grad():
        out_a = model(src_a, tgt, src_a.ne(0), tgt.ne(0))
        out_b = model(src_b, tgt, src_b.ne(0), tgt.ne(0))
    for pos in range(4):  # every byte of the first patch
        delta = (out_a[:, pos] - out_b[:, pos]).abs().max().item()
        assert delta > 1e-6, (
            f"logits at byte {pos} do not depend on the source "
            f"(max|diff|={delta:.2e}) -- the first patch is decoding blind"
        )


def test_blt_greedy_decode_runs():
    model = _tiny_blt()
    src = torch.randint(3, 200, (2, 12))
    out = model.greedy_decode(src, src.ne(0), max_new_tokens=10)
    assert out.shape[0] == 2 and out.shape[1] <= 11
    assert int(out[0, 0]) == BYTE_BOS_ID


# --------------------------------------------------------------------------- #
# tokenizers
# --------------------------------------------------------------------------- #
def test_bpe_roundtrip_is_lossless():
    corpus = [
        "hello world this is a test",
        "0101110100011110101010101010111",
        "alpha bravo charlie  delta\tend",
        "unicode: café ünïcodé ✓ 漢字",
    ] * 20
    tok = BPETokenizer(max_pretoken_len=16).train(corpus, vocab_size=400, verbose=False)
    for text in corpus[:4]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round-trip failed: {text!r} -> {tok.decode(ids)!r}"
    ids = tok.encode("held-out ✓ 1010101", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "held-out ✓ 1010101"
    assert tok.vocab_size <= 400


def test_bpe_is_genuine_subword_not_fixed_width():
    """Regression guard for the assignment's explicit prohibition:

        "Fixed-width chunking (such as 8 bits = 1 token) does not count as
        subword tokenization."

    A BPE tokenizer whose ``max_pretoken_len`` sits too close to the corpus's
    natural period degenerates into exactly that forbidden scheme: it learns a
    fixed lookup table (one merged token per period-length window) and stops,
    because nothing shorter is ever cheaper. On an 8-bit-periodic binary
    cipher, ``max_pretoken_len=8`` reproduces the forbidden example verbatim --
    every emitted piece is exactly 8 characters, with zero length variation.

    This test pins two things: that the degenerate case is still detectable
    (so the guardrail in ``train.py``'s ``build_tokenizers`` has something to
    catch), and that the actually-recommended window size (32) does not
    degenerate -- it produces pieces of genuinely varying length, i.e. real
    corpus-driven subword merges.
    """
    import random

    rng = random.Random(0)
    # Mirror the real mechanism: a small plaintext alphabet (letters + space,
    # like natural-language bytes) XORed with a fixed repeating 8-byte key.
    # That bounds the corpus to a small set of distinct byte values per key
    # phase -- exactly what lets BPE fully saturate into a per-byte lookup
    # table when the pretoken window equals the key period.
    alphabet = [ord(c) for c in "abcdefghijklmnopqrstuvwxyz "]
    key = [rng.randrange(256) for _ in range(8)]
    corpus = []
    for _ in range(300):
        plain = [rng.choice(alphabet) for _ in range(512 // 8)]
        cipher_bits = []
        for i, p in enumerate(plain):
            b = p ^ key[i % 8]
            cipher_bits.append(f"{b:08b}")
        corpus.append("".join(cipher_bits))

    degenerate = BPETokenizer(max_pretoken_len=8).train(corpus, vocab_size=4096, verbose=False)
    healthy = BPETokenizer(max_pretoken_len=32).train(corpus, vocab_size=4096, verbose=False)

    deg_stats = degenerate.piece_length_stats(corpus)
    healthy_stats = healthy.piece_length_stats(corpus)

    assert deg_stats["n_distinct"] == 1 and deg_stats["min"] == deg_stats["max"] == 8, (
        f"expected the pretoken_len=8 case to degenerate to fixed 8-char pieces "
        f"(this is the known-bad case the guardrail must catch), got {deg_stats}"
    )
    assert healthy_stats["n_distinct"] > 1, (
        f"pretoken_len=32 should yield variable-length pieces (genuine learned "
        f"subword tokenization), got {healthy_stats}"
    )
    assert healthy_stats["stdev"] > 0.0, "healthy tokenizer must show length variation"


def test_bpe_matches_naive_reference():
    """The heap-accelerated trainer must learn exactly the naive merge sequence.

    ``BPETokenizer.train`` maintains an incremental pair-count index plus a
    lazy-deletion heap instead of recounting every pair after every merge. This
    pins that optimisation to the textbook O(V) recount it replaces -- the merge
    lists must be byte-identical, not merely similar.
    """
    import random
    from collections import Counter

    from src.tokenizer import SPECIAL_TOKENS

    def naive_merges(texts, vocab_size, max_pretoken_len, min_frequency=2):
        helper = BPETokenizer(max_pretoken_len=max_pretoken_len)
        word_freqs = Counter()
        for t in texts:
            for chunk in helper._pretokenize(t):
                word_freqs[chunk] += 1
        base = len(SPECIAL_TOKENS) + 256
        words = [list(w) for w in word_freqs]
        freqs = [word_freqs[w] for w in word_freqs]
        merges = []
        while len(merges) < vocab_size - base:
            counts = Counter()
            for sym, f in zip(words, freqs):
                for p in zip(sym, sym[1:]):
                    counts[p] += f
            if not counts:
                break
            top = max(counts.values())
            if top < min_frequency:
                break
            best = min(p for p, v in counts.items() if v == top)  # heap tie-break
            merges.append(best)
            new = best[0] + best[1]
            for i, sym in enumerate(words):
                out, j = [], 0
                while j < len(sym):
                    if j < len(sym) - 1 and sym[j] == best[0] and sym[j + 1] == best[1]:
                        out.append(new)
                        j += 2
                    else:
                        out.append(sym[j])
                        j += 1
                words[i] = out
        return merges

    rng = random.Random(0)
    vocab = ["alpha", "bravo", "charlie", "delta", "echo", "node", "token", "patch"]
    texts = [" ".join(rng.choice(vocab) for _ in range(rng.randint(2, 6))) for _ in range(150)]
    texts += ["".join(rng.choice("01") for _ in range(120)) for _ in range(150)]

    for vocab_size in (300, 340, 420):
        fast = BPETokenizer(max_pretoken_len=12).train(texts, vocab_size=vocab_size, verbose=False)
        slow = naive_merges(texts, vocab_size, 12)
        assert fast.merges == slow, (
            f"vocab_size={vocab_size}: heap trainer diverged from the naive recount "
            f"({len(fast.merges)} vs {len(slow)} merges)"
        )
        assert len(fast.merges) > 0, "no merges were learned"


def test_bpe_save_load(tmp_path=None):
    import tempfile

    corpus = ["0101010101", "1111000011110000", "abc abc abc"] * 30
    tok = BPETokenizer(max_pretoken_len=8).train(corpus, vocab_size=300, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tok.json")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
    assert tok2.vocab == tok.vocab and tok2.merges == tok.merges
    assert tok2.encode("0101010101") == tok.encode("0101010101")


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for text in ("hello", "0101", "café ✓", ""):
        assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == BYTE_VOCAB_SIZE


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("flaw", "lawn") == 2
    assert levenshtein("prefix_MIDDLE_suffix", "prefix_middle_suffix") == 6


def test_bit_and_sequence_accuracy():
    refs = ["0101", "1111"]
    assert detect_binary_mode(refs) is True
    assert abs(bit_accuracy(refs, refs, True) - 100.0) < 1e-9
    assert abs(bit_accuracy(["0100", "1111"], refs, True) - 87.5) < 1e-9
    assert abs(sequence_accuracy(["0101", "0000"], refs) - 50.0) < 1e-9
    # length mismatch is penalised through the max(len) denominator
    assert bit_accuracy(["01"], ["0101"], True) < 60.0


def test_bleu_and_full_metric_pack():
    refs = ["alpha bravo charlie delta", "echo foxtrot golf hotel"]
    assert corpus_bleu(refs, refs) > 99.0
    assert corpus_bleu(["x y z w", "q r s t"], refs) < 5.0
    m = compute_metrics(refs, refs)
    assert m["sequence_accuracy"] == 100.0
    assert m["bit_accuracy"] > 99.99
    assert m["levenshtein"] == 0.0
    assert m["rougeL"] > 99.0


# --------------------------------------------------------------------------- #
# configurations
# --------------------------------------------------------------------------- #
def test_all_five_configs_build_and_step():
    args = argparse.Namespace(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, dropout=0.0,
        max_src_len=32, max_tgt_len=32, no_tie_embeddings=False,
        blt_patch_size=4, blt_d_local=16, blt_local_enc_layers=1,
        blt_local_dec_layers=1, blt_local_heads=2, blt_window=8,
    )
    for name in sorted(CONFIGS):
        cfg = get_config(name)
        vocab = BYTE_VOCAB_SIZE if cfg.is_blt else 50
        model = build_model(cfg, args, vocab, vocab)
        src = torch.randint(3, vocab, (2, 12))
        tgt = torch.randint(3, vocab, (2, 9))
        logits = model(src, tgt, src.ne(0), tgt.ne(0))
        assert logits.shape == (2, 9, vocab), f"{name}: {logits.shape}"
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
        loss.backward()
        assert math.isfinite(float(loss.detach())), f"{name}: non-finite loss"
        out = model.greedy_decode(src, src.ne(0), max_new_tokens=6)
        assert out.size(0) == 2, f"{name}: bad generation shape"


def test_c3_has_fewer_params_than_c1():
    args = argparse.Namespace(
        d_model=64, n_layers=2, n_heads=8, n_kv_heads=2, d_ff=128, dropout=0.0,
        max_src_len=32, max_tgt_len=32, no_tie_embeddings=False,
        blt_patch_size=4, blt_d_local=16, blt_local_enc_layers=1,
        blt_local_dec_layers=1, blt_local_heads=2, blt_window=8,
    )
    c1 = build_model(get_config("C1"), args, 100, 100)
    c3 = build_model(get_config("C3"), args, 100, 100)
    n1 = sum(p.numel() for p in c1.parameters())
    n3 = sum(p.numel() for p in c3.parameters())
    assert n3 < n1, f"GQA should be smaller: C1={n1} C3={n3}"


def test_c4_uses_rmsnorm_and_c1_does_not():
    args = argparse.Namespace(
        d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64, dropout=0.0,
        max_src_len=32, max_tgt_len=32, no_tie_embeddings=False,
        blt_patch_size=4, blt_d_local=16, blt_local_enc_layers=1,
        blt_local_dec_layers=1, blt_local_heads=2, blt_window=8,
    )
    c1 = build_model(get_config("C1"), args, 50, 50)
    c4 = build_model(get_config("C4"), args, 50, 50)
    assert any(isinstance(m, RMSNorm) for m in c4.modules())
    assert not any(isinstance(m, RMSNorm) for m in c1.modules())
    assert any(isinstance(m, LayerNorm) for m in c1.modules())


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _all_tests():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_") and callable(g[n])]


def main() -> int:
    torch.manual_seed(0)
    failures = []
    tests = _all_tests()
    for i, (name, fn) in enumerate(tests, start=1):
        try:
            fn()
            print(f"  [{i:>2}/{len(tests)}] PASS  {name}", flush=True)
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"  [{i:>2}/{len(tests)}] FAIL  {name}\n        {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [{i:>2}/{len(tests)}] ERROR {name}\n        "
                  f"{type(exc).__name__}: {exc}", flush=True)
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests failed:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =========================================================================== #
# Entropy-based dynamic patching (C5, per the assignment clarification)
# =========================================================================== #
def _fit_entropy(seqs, target_patch=4.0):
    from src.models.entropy import ByteNgramEntropyModel
    m = ByteNgramEntropyModel(alpha=0.5, min_count=2).fit(seqs, verbose=False)
    m.calibrate_threshold(seqs, target_patch_len=target_patch, verbose=False)
    return m


def _english_like(n_seq=400, n=32):
    """Low-entropy byte streams (repeated words) so patching has structure."""
    import random
    random.seed(0)
    words = [b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"over "]
    out = []
    for _ in range(n_seq):
        s = b""
        while len(s) < n:
            s += random.choice(words)
        out.append(list(s[:n]))
    return out


def test_bit_packing_is_lossless_and_eightfold():
    """The cipher is binary data, not text: 8 bits must become ONE byte."""
    tok = ByteTokenizer(bit_packed=True)
    bits = "".join(random.choice("01") for _ in range(256))
    ids = tok.encode(bits, add_bos=True, add_eos=True)
    assert len(ids) == 256 // 8 + 2, f"expected 34 ids, got {len(ids)}"
    assert tok.decode(ids) == bits, "bit packing is not lossless"
    raw = tok.raw_bytes(bits)
    assert len(raw) == 32 and all(0 <= b <= 255 for b in raw)
    # and the text-encoding interpretation is 8x longer, which is what we avoid
    assert len(ByteTokenizer(bit_packed=False).raw_bytes(bits)) == 256


def test_entropy_model_matches_naive_reference():
    from src.models.entropy import ByteNgramEntropyModel
    seqs = _english_like(200)
    m = ByteNgramEntropyModel(alpha=0.5, min_count=1).fit(seqs, verbose=False)
    # recompute H(x_t | ctx) for one well-attested order-2 context by hand
    from collections import Counter
    ctx = (seqs[0][0], seqs[0][1])
    cnt = Counter()
    for s in seqs:
        for i in range(len(s) - 2):
            if (s[i], s[i + 1]) == ctx:
                cnt[s[i + 2]] += 1
    if sum(cnt.values()) >= 1:
        p = np.array([cnt.get(b, 0) for b in range(256)], dtype=np.float64) + 0.5
        p /= p.sum()
        ref = float(-(p * np.log2(p)).sum())
        got = float(m.H2[ctx[0] * 256 + ctx[1]])
        assert abs(got - ref) < 1e-4, f"entropy {got} != reference {ref}"


def test_dynamic_patches_are_variable_length():
    """Entropy patching must produce genuinely variable patches, not fixed ones."""
    from src.models.entropy import segment_from_entropy
    seqs = _english_like(400)
    m = _fit_entropy(seqs, target_patch=4.0)
    H1, H2, H0 = m.as_tensors()
    ids = torch.tensor([[b + 3 for b in s] for s in seqs[:64]])
    seg = segment_from_entropy(ids, H1, H2, H0, m.threshold, offset=3, n_special=3)
    lens = torch.cat([torch.bincount(seg[i]).float() for i in range(seg.size(0))])
    assert lens.min() != lens.max(), "patches are all the same length (not dynamic)"
    assert seg.min() >= 0 and (seg.diff(dim=1) >= 0).all(), "seg must be non-decreasing"
    # boundaries must actually track entropy, not position
    assert len(torch.unique(lens)) >= 3, f"only {len(torch.unique(lens))} distinct lengths"


def test_patch_boundary_depends_only_on_past():
    """seg[t] must be computable BEFORE x_t exists, or decoding is illegal.

    The boundary at t is H(x_t | x_{t-2}, x_{t-1}) > theta, a function of the
    context only. Changing x_t itself must not move its own patch assignment.
    """
    from src.models.entropy import segment_from_entropy
    seqs = _english_like(200)
    m = _fit_entropy(seqs)
    H1, H2, H0 = m.as_tensors()
    ids = torch.tensor([[b + 3 for b in s] for s in seqs[:16]])
    seg = segment_from_entropy(ids, H1, H2, H0, m.threshold, offset=3, n_special=3)
    t = 10
    ids2 = ids.clone()
    ids2[:, t] = (ids2[:, t] + 7 - 3) % 256 + 3          # perturb byte t only
    seg2 = segment_from_entropy(ids2, H1, H2, H0, m.threshold, offset=3, n_special=3)
    assert torch.equal(seg[:, : t + 1], seg2[:, : t + 1]), (
        "changing x_t changed its own patch id -- boundaries would leak the future"
    )


def test_blt_dynamic_patching_is_causal():
    """Same causality guarantee as fixed patching, now with variable patches."""
    torch.manual_seed(0)
    seqs = _english_like(400)
    ent = _fit_entropy(seqs, target_patch=3.0)
    model = _tiny_blt(4)
    model.install_entropy(ent, ent, max_patch_len=8)
    model.eval()
    src = torch.randint(3, 200, (2, 16))
    tgt = torch.tensor([[b + 3 for b in s[:16]] for s in seqs[:2]])
    cut = 9
    with torch.no_grad():
        base = model(src, tgt, src.ne(0), tgt.ne(0))
        tgt2 = tgt.clone()
        tgt2[:, cut:] = torch.randint(3, 200, (2, 16 - cut))
        other = model(src, tgt2, src.ne(0), tgt2.ne(0))
    _close(base[:, :cut], other[:, :cut], tol=1e-5,
           msg="BLT dynamic patching leaked future target bytes:")


def test_blt_dynamic_forward_and_generate():
    torch.manual_seed(0)
    seqs = _english_like(400)
    ent = _fit_entropy(seqs)
    model = _tiny_blt(4)
    model.install_entropy(ent, ent, max_patch_len=8)
    src = torch.randint(3, 200, (2, 24))
    tgt = torch.randint(3, 200, (2, 13))
    logits = model(src, tgt, src.ne(0), tgt.ne(0))
    assert logits.shape == (2, 13, BYTE_VOCAB_SIZE), logits.shape
    logits.sum().backward()
    out = model.greedy_decode(src, src.ne(0), max_new_tokens=8)
    assert out.shape[0] == 2 and out.shape[1] <= 9
    stats = model.patch_stats(src, "src", src.ne(0))
    assert stats["mean_patch_len"] > 0


def test_dynamic_patch_ids_are_prefix_stable():
    """Appending bytes must not renumber the patches of earlier bytes.

    Greedy decoding recomputes the segmentation of a growing prefix at every
    step. If a later byte could change an earlier byte's patch id, the patches
    seen at generation time would differ from those seen under teacher forcing
    and decoding would silently diverge from training.
    """
    from src.models.entropy import segment_from_entropy
    seqs = _english_like(500, n=40)
    m = _fit_entropy(seqs)
    H1, H2, H0 = m.as_tensors()
    full = torch.tensor([[BYTE_BOS_ID] + [b + 3 for b in s] for s in seqs[:32]])
    for cap in (0, 12):
        seg_full = segment_from_entropy(full, H1, H2, H0, m.threshold,
                                        offset=3, n_special=3, max_patch=cap)
        for cut in (5, 12, 20, 30):
            seg_pre = segment_from_entropy(full[:, :cut], H1, H2, H0, m.threshold,
                                           offset=3, n_special=3, max_patch=cap)
            assert torch.equal(seg_pre, seg_full[:, :cut]), (
                f"patch ids changed for the prefix when the sequence grew "
                f"(cap={cap}, cut={cut})"
            )


def test_blt_entropy_checkpoint_round_trips():
    """A trained C5 checkpoint must reload into a freshly built model.

    The entropy tables are registered in __init__ (not in install_entropy) so
    they are never "unexpected keys" at load time, and `dynamic_patching` is
    derived from the threshold so it survives the round-trip without adding a
    key that older checkpoints would lack. Without this, --eval-only, --ckpt and
    anything downloading the checkpoint from the Hub would fail.
    """
    torch.manual_seed(0)
    ent = _fit_entropy(_english_like(300, n=40))
    kw = dict(d_model=32, n_encoder_layers=1, n_decoder_layers=1, n_heads=2,
              d_ff=64, d_local=16, n_local_heads=2, max_bytes=64, max_patch_len=8)

    a = BLTSeq2Seq(**kw)
    assert a.dynamic_patching is False, "a fresh model must not claim dynamic patching"
    a.install_entropy(ent, ent, max_patch_len=8)
    assert a.dynamic_patching is True

    b = BLTSeq2Seq(**kw)            # build_model() passes max_patch_len from args
    b.load_state_dict(a.state_dict())          # strict: no missing/unexpected keys
    assert b.dynamic_patching is True, "dynamic patching did not survive reload"

    a.eval(); b.eval()
    src = torch.randint(3, 200, (2, 24))
    tgt = torch.randint(3, 200, (2, 13))
    with torch.no_grad():
        _close(a(src, tgt, src.ne(0), tgt.ne(0)),
               b(src, tgt, src.ne(0), tgt.ne(0)), tol=1e-6,
               msg="reloaded BLT does not reproduce the original logits:")
        assert torch.equal(a.greedy_decode(src, src.ne(0), max_new_tokens=8),
                           b.greedy_decode(src, src.ne(0), max_new_tokens=8))
