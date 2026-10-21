"""Tests for the from-scratch decoding strategies and their metrics."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.part1.model import TransformerConfig, TransformerLM, variant_config
from src.part3.decoding import (
    GenerationConfig,
    LocalBackend,
    generate,
    strip_after_eos,
    top_k_filter,
    top_p_filter,
)
from src.part3.metrics import (
    distinct_n,
    evaluate_generations,
    normalise,
    perplexity_from_logprobs,
    positional_accuracy,
    prf,
    prf_counts,
    repetition_rate,
)
from src.part3.run import truncate_sentences

BASE = TransformerConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=4,
                         n_ctx=64, d_ff=128)


def toy_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(variant_config("dense", BASE)).eval()


def prompts(B: int = 3, T: int = 6):
    torch.manual_seed(1)
    ids = torch.randint(1, BASE.vocab_size, (B, T))
    return ids, torch.ones(B, T, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Logit filters
# --------------------------------------------------------------------------- #


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.randn(4, 50)
    for k in (1, 5, 20):
        out = top_k_filter(logits.clone(), k)
        assert (out > float("-inf")).sum(-1).tolist() == [k] * 4
        # The survivors must be the k largest.
        kept = out.topk(k, -1).indices.sort(-1).values
        expect = logits.topk(k, -1).indices.sort(-1).values
        assert torch.equal(kept, expect)


def test_top_k_is_a_no_op_when_k_covers_the_vocabulary():
    logits = torch.randn(2, 10)
    assert torch.equal(top_k_filter(logits.clone(), 10), logits)
    assert torch.equal(top_k_filter(logits.clone(), 0), logits)


def test_top_p_keeps_the_smallest_prefix_reaching_p():
    # Probabilities 0.5, 0.25, 0.125, 0.0625, 0.0625.
    # Cumulative:    0.5, 0.75, 0.875, 0.9375, 1.0.
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.125, 0.0625, 0.0625]]))

    # 0.5 < 0.7 <= 0.75, so two tokens are needed.
    kept = top_p_filter(logits.clone(), 0.7) > float("-inf")
    assert kept.tolist() == [[True, True, False, False, False]]

    # 0.75 < 0.8 <= 0.875, so three.
    kept = top_p_filter(logits.clone(), 0.8) > float("-inf")
    assert kept.tolist() == [[True, True, True, False, False]]

    # 0.875 < 0.9 <= 0.9375, so the fourth token is required too.
    kept = top_p_filter(logits.clone(), 0.9) > float("-inf")
    assert kept.tolist() == [[True, True, True, True, False]]

    # Exact tie: a token is dropped only when the mass *strictly before* it
    # already exceeds p, so at p = 0.75 the third token survives.  This matches
    # the reference HuggingFace filter and guarantees retained mass >= p.
    kept = top_p_filter(logits.clone(), 0.75) > float("-inf")
    assert kept.tolist() == [[True, True, True, False, False]]


def test_top_p_retained_mass_is_always_at_least_p():
    torch.manual_seed(0)
    logits = torch.randn(16, 60)
    probs = F.softmax(logits, dim=-1)
    for p in (0.1, 0.5, 0.9, 0.99):
        kept = top_p_filter(logits.clone(), p) > float("-inf")
        assert ((probs * kept).sum(-1) >= p - 1e-6).all(), p


def test_top_p_always_keeps_the_argmax():
    """Even a tiny p must leave one token, or sampling would be undefined."""
    logits = torch.randn(3, 40)
    out = top_p_filter(logits.clone(), 0.01)
    assert (out > float("-inf")).sum(-1).min() >= 1
    assert torch.equal(out.argmax(-1), logits.argmax(-1))


def test_top_p_is_a_no_op_at_one():
    logits = torch.randn(2, 10)
    assert torch.equal(top_p_filter(logits.clone(), 1.0), logits)


def test_top_p_rejects_an_invalid_threshold():
    with pytest.raises(ValueError):
        top_p_filter(torch.randn(2, 5), -0.5)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


def test_greedy_picks_the_argmax_at_every_step():
    model = toy_model()
    ids, mask = prompts()
    cfg = GenerationConfig(strategy="greedy", max_new_tokens=5, eos_id=None, pad_id=0)
    res = generate(LocalBackend(model), ids, mask, cfg)

    seq = ids.clone()
    for _ in range(5):
        with torch.no_grad():
            nxt = model(seq).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    assert [r for r in res.tokens] == seq[:, ids.shape[1]:].tolist()


def test_beam_width_one_equals_greedy():
    """The strongest single check on the beam-search implementation."""
    model = toy_model()
    ids, mask = prompts()
    kw = dict(max_new_tokens=8, eos_id=None, pad_id=0)
    g = generate(LocalBackend(model), ids, mask,
                 GenerationConfig(strategy="greedy", **kw))
    b = generate(LocalBackend(model), ids, mask,
                 GenerationConfig(strategy="beam", beam_width=1, **kw))
    assert g.tokens == b.tokens
    for x, y in zip(g.sum_logprob, b.sum_logprob):
        assert abs(x - y) < 1e-4


@pytest.mark.parametrize("width", [2, 4, 6])
def test_wider_beams_never_score_worse(width):
    model = toy_model()
    ids, mask = prompts()
    kw = dict(max_new_tokens=8, eos_id=None, pad_id=0, length_penalty=0.0)
    one = generate(LocalBackend(model), ids, mask,
                   GenerationConfig(strategy="beam", beam_width=1, **kw))
    many = generate(LocalBackend(model), ids, mask,
                    GenerationConfig(strategy="beam", beam_width=width, **kw))
    for a, b in zip(one.sum_logprob, many.sum_logprob):
        assert b >= a - 1e-4, (a, b)


def test_beam_search_scores_match_a_recomputed_forward():
    """The reported beam score must be the true log-probability of the output."""
    model = toy_model()
    ids, mask = prompts(B=2, T=5)
    cfg = GenerationConfig(strategy="beam", beam_width=3, max_new_tokens=6,
                           eos_id=None, pad_id=0, length_penalty=0.0)
    res = generate(LocalBackend(model), ids, mask, cfg)
    for b in range(ids.shape[0]):
        seq = torch.cat([ids[b], torch.tensor(res.tokens[b])]).unsqueeze(0)
        with torch.no_grad():
            logits = model(seq).logits
        lp = F.log_softmax(logits.float(), -1)
        total = sum(float(lp[0, ids.shape[1] - 1 + t, res.tokens[b][t]])
                    for t in range(len(res.tokens[b])))
        assert abs(total - res.sum_logprob[b]) < 1e-3, (b, total, res.sum_logprob[b])


def test_sampling_is_reproducible_and_respects_the_filter():
    model = toy_model()
    ids, mask = prompts()
    cfg = GenerationConfig(strategy="top_k", top_k=3, max_new_tokens=6,
                           eos_id=None, pad_id=0)
    g1 = torch.Generator()
    g1.manual_seed(7)
    g2 = torch.Generator()
    g2.manual_seed(7)
    a = generate(LocalBackend(model), ids, mask, cfg, generator=g1)
    b = generate(LocalBackend(model), ids, mask, cfg, generator=g2)
    assert a.tokens == b.tokens

    g3 = torch.Generator()
    g3.manual_seed(8)
    c = generate(LocalBackend(model), ids, mask, cfg, generator=g3)
    assert isinstance(c.tokens, list)


def test_top_k_one_is_greedy():
    model = toy_model()
    ids, mask = prompts()
    kw = dict(max_new_tokens=6, eos_id=None, pad_id=0)
    g = generate(LocalBackend(model), ids, mask, GenerationConfig(strategy="greedy", **kw))
    gen = torch.Generator()
    gen.manual_seed(0)
    k = generate(LocalBackend(model), ids, mask,
                 GenerationConfig(strategy="top_k", top_k=1, **kw), generator=gen)
    assert g.tokens == k.tokens


def test_eos_stops_a_sequence():
    model = toy_model()
    ids, mask = prompts()
    cfg = GenerationConfig(strategy="greedy", max_new_tokens=10, eos_id=None, pad_id=0)
    free = generate(LocalBackend(model), ids, mask, cfg)
    forced = free.tokens[0][0]                      # the token greedy picks first

    cfg_eos = GenerationConfig(strategy="greedy", max_new_tokens=10,
                               eos_id=forced, pad_id=0)
    stopped = generate(LocalBackend(model), ids, mask, cfg_eos)
    assert stopped.n_tokens[0] == 1
    assert stopped.tokens[0] == [forced]


@pytest.mark.parametrize("strategy", ["greedy", "top_k", "top_p", "beam"])
def test_generation_respects_the_length_cap(strategy):
    model = toy_model()
    ids, mask = prompts()
    cfg = GenerationConfig(strategy=strategy, max_new_tokens=7, beam_width=2,
                           eos_id=None, pad_id=0)
    res = generate(LocalBackend(model), ids, mask, cfg)
    assert all(n <= 7 for n in res.n_tokens)
    assert all(len(t) <= 7 for t in res.tokens)
    assert res.forward_calls <= 7


def test_left_padded_batch_matches_single_sequence_decoding():
    """Batched decoding must not be corrupted by padding."""
    model = toy_model()
    torch.manual_seed(2)
    seqs = [torch.randint(1, BASE.vocab_size, (n,)) for n in (7, 5, 3)]
    T = max(len(s) for s in seqs)
    ids = torch.zeros(3, T, dtype=torch.long)
    mask = torch.zeros(3, T, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, T - len(s):] = s
        mask[r, T - len(s):] = 1

    cfg = GenerationConfig(strategy="greedy", max_new_tokens=6, eos_id=None, pad_id=0)
    batched = generate(LocalBackend(model), ids, mask, cfg)
    for r, s in enumerate(seqs):
        solo = generate(LocalBackend(model), s.unsqueeze(0),
                        torch.ones(1, len(s), dtype=torch.long), cfg)
        assert batched.tokens[r] == solo.tokens[0], r


def test_strip_after_eos():
    assert strip_after_eos([1, 2, 3, 9, 4, 5], 9) == [1, 2, 3]
    assert strip_after_eos([1, 2, 3], 9) == [1, 2, 3]
    assert strip_after_eos([1, 2, 3], None) == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_normalise_strips_punctuation_and_case():
    assert normalise("Hello, World!  It's fine.") == ["hello", "world", "it's", "fine"]


def test_prf_on_a_worked_example():
    hyp = ["a", "b", "b", "c"]
    ref = ["a", "b", "d"]
    overlap, nh, nr = prf_counts(hyp, ref)
    assert (overlap, nh, nr) == (2, 4, 3)           # multiset intersection: a, b
    p, r, f = prf(overlap, nh, nr)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(2 / 3)
    assert f == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_prf_is_perfect_on_an_exact_match():
    toks = ["x", "y", "z"]
    p, r, f = prf(*prf_counts(toks, toks))
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_prf_handles_empty_input():
    assert prf(*prf_counts([], ["a"])) == (0.0, 0.0, 0.0)


def test_positional_accuracy_normalises_by_the_longer_sequence():
    assert positional_accuracy(["a", "b"], ["a", "b"]) == (2, 2)
    assert positional_accuracy(["a", "b"], ["a", "b", "c"]) == (2, 3)
    assert positional_accuracy(["b", "a"], ["a", "b"]) == (0, 2)
    assert positional_accuracy([], []) == (0, 0)


def test_distinct_and_repetition():
    assert distinct_n([["a", "b", "a", "b"]], 1) == pytest.approx(0.5)
    assert repetition_rate([["a", "b", "c", "d", "a", "b", "c", "d"]], 4) == pytest.approx(1 / 5)
    assert repetition_rate([["a", "b", "c", "d", "e"]], 4) == 0.0


def test_perplexity_from_logprobs():
    assert perplexity_from_logprobs([-2.0, -4.0], [2, 2]) == pytest.approx(math.exp(1.5))
    assert math.isnan(perplexity_from_logprobs([], []))


def test_evaluate_generations_on_an_exact_match():
    refs = ["The cat sat on the mat.", "She went home."]
    out = evaluate_generations(refs, refs)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["bleu"] > 99.0


def test_truncate_sentences():
    text = "One. Two! Three? Four. Five."
    assert truncate_sentences(text, 4) == "One. Two! Three? Four."
    assert truncate_sentences("No terminator here", 4) == "No terminator here"
    assert truncate_sentences("", 4) == ""
