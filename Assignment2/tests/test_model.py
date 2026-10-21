"""Model-level tests: parameter parity, MoE routing, KV cache, padding."""

from __future__ import annotations

import math

import pytest
import torch

from src.part1.model import (
    VARIANTS,
    MoE,
    TransformerConfig,
    TransformerLM,
    variant_config,
)

BASE = TransformerConfig(vocab_size=512, d_model=64, n_layers=3, n_heads=4,
                         n_ctx=64, d_ff=256)


def build(name: str) -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(variant_config(name, BASE))


# --------------------------------------------------------------------------- #
# Parameter parity - the core constraint of the assignment
# --------------------------------------------------------------------------- #


def test_analytic_param_counts_match_the_real_model():
    for name in VARIANTS:
        m = build(name)
        budget = m.config.param_budget()
        assert budget["total"] == m.num_parameters(), name
        assert budget["active"] == m.num_active_parameters(), name


def _router_params(cfg: TransformerConfig) -> int:
    """The only parameters an MoE layer adds beyond its experts."""
    return 0 if cfg.ffn_type == "mlp" else cfg.d_model * cfg.n_routed_experts * cfg.n_layers


def test_first_four_variants_share_a_total_parameter_budget():
    """Variants 1-4 must have equal total parameters, up to the router matrix."""
    counts = {n: build(n).num_parameters() for n in VARIANTS[:4]}
    dense = counts["dense"]
    for name, n in counts.items():
        assert n == dense + _router_params(variant_config(name, BASE)), name


def test_router_overhead_is_negligible_at_the_real_model_scale():
    """At the size actually trained, the router is well under 0.1% of the model."""
    real = TransformerConfig(vocab_size=32000, d_model=384, n_layers=6, n_heads=6,
                             n_ctx=256, d_ff=1536)
    dense = variant_config("dense", real).param_budget()["total"]
    for name in VARIANTS[1:4]:
        total = variant_config(name, real).param_budget()["total"]
        assert 0 < (total - dense) / dense < 1e-3, name


def test_fifth_variant_matches_dense_active_parameters():
    dense = build("dense")
    wide = build("moe_4e_top2_wide")
    router = _router_params(wide.config)
    assert wide.num_active_parameters() == dense.num_active_parameters() + router
    # ... while holding twice the dense layer's expert parameters.
    dense_ffn = dense.config.param_budget()["ffn_total"]
    wide_ffn = wide.config.param_budget()["ffn_total"] - router
    assert wide_ffn == 2 * dense_ffn


def test_moe_variants_activate_fewer_parameters_than_they_store():
    for name in VARIANTS[1:4]:
        m = build(name)
        assert m.num_active_parameters() < m.num_parameters()


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", VARIANTS)
def test_forward_shapes_and_loss(name):
    m = build(name)
    ids = torch.randint(0, BASE.vocab_size, (4, 16))
    out = m(ids[:, :-1], targets=ids[:, 1:])
    assert out.logits.shape == (4, 15, BASE.vocab_size)
    assert torch.isfinite(out.loss)
    # Untrained cross-entropy should sit near log(vocab).
    assert abs(float(out.lm_loss.detach()) - math.log(BASE.vocab_size)) < 1.0


@pytest.mark.parametrize("name", VARIANTS)
def test_backward_reaches_every_parameter(name):
    m = build(name)
    ids = torch.randint(0, BASE.vocab_size, (4, 16))
    m(ids[:, :-1], targets=ids[:, 1:]).loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"


def test_loss_mask_excludes_padding():
    m = build("dense")
    ids = torch.randint(1, BASE.vocab_size, (3, 12))
    full = m(ids[:, :-1], targets=ids[:, 1:]).lm_loss
    mask = torch.ones(3, 11, dtype=torch.long)
    same = m(ids[:, :-1], targets=ids[:, 1:], loss_mask=mask).lm_loss
    assert torch.allclose(full, same)

    mask[:, -4:] = 0
    partial = m(ids[:, :-1], targets=ids[:, 1:], loss_mask=mask).lm_loss
    assert not torch.allclose(full, partial)


# --------------------------------------------------------------------------- #
# MoE routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", VARIANTS[1:])
def test_router_weights_are_normalised_and_top_k(name):
    m = build(name)
    m.set_capture_routing(True)
    ids = torch.randint(0, BASE.vocab_size, (2, 10))
    out = m(ids)
    cfg = m.config
    for stats in out.router_stats:
        assert stats.topk_idx.shape == (2, 10, cfg.n_active_experts)
        assert torch.allclose(stats.topk_weight.float().sum(-1),
                              torch.ones(2, 10), atol=1e-5)
        assert int(stats.topk_idx.max()) < cfg.n_routed_experts
        # A token must not be routed to the same expert twice.
        for b in range(2):
            for t in range(10):
                picked = stats.topk_idx[b, t].tolist()
                assert len(set(picked)) == len(picked)


@pytest.mark.parametrize("name", VARIANTS[1:])
def test_aux_loss_is_one_for_uniform_routing(name):
    """Switch-style aux loss equals 1.0 exactly when the load is balanced."""
    m = build(name)
    moe = next(mod for mod in m.modules() if isinstance(mod, MoE))
    torch.nn.init.zeros_(moe.router.weight)          # uniform probabilities
    x = torch.randn(2, 8, BASE.d_model)
    _, stats = moe(x)
    assert abs(float(stats.aux_loss.detach()) - 1.0) < 0.15
    assert float(stats.aux_loss.detach()) >= 1.0 - 1e-4       # 1.0 is the minimum


def test_moe_dispatch_matches_a_dense_reference():
    """The sparse gather/scatter dispatch must equal the naive weighted sum."""
    torch.manual_seed(3)
    cfg = variant_config("moe_4e_top2", BASE)
    moe = MoE(cfg).eval()
    x = torch.randn(2, 7, cfg.d_model)
    got, stats = moe(x)

    xf = x.reshape(-1, cfg.d_model)
    logits = moe.router(xf).float()
    probs = torch.softmax(logits, -1)
    topw, topi = torch.topk(probs, cfg.n_active_experts, -1)
    topw = (topw / topw.sum(-1, keepdim=True)).to(x.dtype)
    ref = torch.zeros_like(xf)
    for se in moe.shared_experts:
        ref = ref + se(xf)
    for slot in range(cfg.n_active_experts):
        for e in range(cfg.n_routed_experts):
            sel = topi[:, slot] == e
            if sel.any():
                ref[sel] += moe.experts[e](xf[sel]) * topw[sel, slot].unsqueeze(-1)
    assert torch.allclose(got.reshape_as(ref), ref, atol=1e-5)


def test_padding_is_excluded_from_the_load_balancing_loss():
    torch.manual_seed(5)
    cfg = variant_config("moe_4e_top1", BASE)
    moe = MoE(cfg).eval()
    x = torch.randn(2, 10, cfg.d_model)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[:, 6:] = 0
    _, with_mask = moe(x, mask)
    _, no_mask = moe(x, None)
    assert not torch.allclose(with_mask.aux_loss, no_mask.aux_loss)
    # Masking must agree with simply dropping the padded positions.
    _, trimmed = moe(x[:, :6], torch.ones(2, 6, dtype=torch.long))
    assert torch.allclose(with_mask.aux_loss, trimmed.aux_loss, atol=1e-5)


# --------------------------------------------------------------------------- #
# KV cache and padding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", VARIANTS)
def test_incremental_decoding_matches_a_full_forward(name):
    m = build(name).eval()
    ids = torch.randint(0, BASE.vocab_size, (3, 11))
    with torch.no_grad():
        full = m(ids).logits
        past, steps = None, []
        for t in range(ids.shape[1]):
            out = m(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            steps.append(out.logits)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-4)


def test_left_padding_does_not_change_the_result():
    """RoPE is relative, so a per-row position shift must be a no-op."""
    m = build("dense").eval()
    ids = torch.randint(1, BASE.vocab_size, (3, 12))
    lens = [12, 9, 4]
    T = 12
    buf = torch.zeros(3, T, dtype=torch.long)
    mask = torch.zeros(3, T, dtype=torch.long)
    for r, L in enumerate(lens):
        buf[r, T - L:] = ids[r, :L]
        mask[r, T - L:] = 1
    with torch.no_grad():
        padded = m(buf, attention_mask=mask).logits
        for r, L in enumerate(lens):
            solo = m(ids[r:r + 1, :L]).logits
            assert torch.allclose(padded[r, T - L:], solo[0], atol=1e-4), r


def test_checkpoint_round_trip(tmp_path):
    m = build("moe_1s_3r_top1")
    path = tmp_path / "ckpt.pt"
    m.save(str(path), extra={"tag": "x"})
    loaded, extra = TransformerLM.load(str(path))
    assert extra["tag"] == "x"
    assert loaded.config == m.config
    ids = torch.randint(0, BASE.vocab_size, (2, 9))
    m.eval(), loaded.eval()
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, loaded(ids).logits, atol=1e-6)


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        TransformerConfig(d_model=64, n_heads=5)
    with pytest.raises(ValueError):
        TransformerConfig(ffn_type="moe", n_routed_experts=2, n_active_experts=3)
    with pytest.raises(ValueError):
        variant_config("does_not_exist", BASE)
