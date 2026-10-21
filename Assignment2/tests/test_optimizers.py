"""Tests for the from-scratch Part 2 optimizers."""

from __future__ import annotations

import math

import pytest
import torch

from src.part2.optimizers import (
    OPTIMIZER_CATEGORY,
    OPTIMIZERS,
    AdamW,
    Lion,
    MARS,
    Muon,
    SophiaG,
    build_muon_param_groups,
    optimizer_state_bytes,
    zeropower_via_newtonschulz5,
)


def toy_problem(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.GELU(),
                                torch.nn.Linear(32, 5))
    X = torch.randn(256, 8)
    Y = X @ torch.randn(8, 5) + 0.05 * torch.randn(256, 5)
    return model, X, Y


# --------------------------------------------------------------------------- #
# AdamW must match the reference implementation exactly
# --------------------------------------------------------------------------- #


def test_adamw_matches_torch_adamw():
    torch.manual_seed(0)
    W = torch.randn(8, 5)
    b = torch.randn(5)
    ours = [torch.nn.Parameter(W.clone()), torch.nn.Parameter(b.clone())]
    ref = [torch.nn.Parameter(W.clone()), torch.nn.Parameter(b.clone())]

    o1 = AdamW([{"params": [ours[0]], "weight_decay": 0.1},
                {"params": [ours[1]], "weight_decay": 0.0}],
               lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    o2 = torch.optim.AdamW([{"params": [ref[0]], "weight_decay": 0.1},
                            {"params": [ref[1]], "weight_decay": 0.0}],
                           lr=1e-2, betas=(0.9, 0.95), eps=1e-8)

    X = torch.randn(64, 8)
    Y = torch.randn(64, 5)
    for _ in range(50):
        for params, opt in ((ours, o1), (ref, o2)):
            opt.zero_grad()
            ((X @ params[0] + params[1] - Y) ** 2).mean().backward()
            opt.step()
    assert torch.allclose(ours[0], ref[0], atol=1e-6)
    assert torch.allclose(ours[1], ref[1], atol=1e-6)


def test_adamw_decoupled_weight_decay_shrinks_an_unused_parameter():
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(p, torch.full((4,), 1.0 - 0.1 * 0.5), atol=1e-6)


# --------------------------------------------------------------------------- #
# Every optimizer optimises
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_optimizer_reduces_loss(name):
    model, X, Y = toy_problem()
    lrs = {"adamw": 3e-2, "mars": 3e-2, "lion": 3e-3, "muon": 2e-2, "sophia": 3e-2}
    if name == "muon":
        opt = Muon(build_muon_param_groups(model, muon_lr=lrs[name], adamw_lr=3e-2,
                                           weight_decay=0.0))
    else:
        opt = OPTIMIZERS[name](model.parameters(), lr=lrs[name], weight_decay=0.0)

    first = float(((model(X) - Y) ** 2).mean().detach())
    for step in range(200):
        opt.zero_grad()
        ((model(X) - Y) ** 2).mean().backward()
        if name == "sophia":
            opt.step(bs=256)
            if step % 10 == 0:
                opt.zero_grad()
                pred = model(X)
                ((pred - (pred.detach() + torch.randn_like(pred))) ** 2).mean().backward()
                opt.update_hessian()
                opt.zero_grad()
        else:
            opt.step()
    last = float(((model(X) - Y) ** 2).mean().detach())
    assert last < first * 0.2, f"{name}: {first:.4f} -> {last:.4f}"
    assert math.isfinite(last)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_category_is_covered(name):
    assert name in OPTIMIZER_CATEGORY


# --------------------------------------------------------------------------- #
# Per-optimizer properties
# --------------------------------------------------------------------------- #


def test_lion_uses_half_the_state_of_adamw():
    """Lion's memory efficiency is its defining property."""
    model, X, Y = toy_problem()
    sizes = {}
    for name, opt in (("adamw", AdamW(model.parameters(), lr=1e-3)),
                      ("lion", Lion(model.parameters(), lr=1e-4))):
        opt.zero_grad()
        ((model(X) - Y) ** 2).mean().backward()
        opt.step()
        sizes[name] = optimizer_state_bytes(opt)
    assert sizes["lion"] * 2 == sizes["adamw"]


def test_lion_takes_unit_magnitude_steps():
    p = torch.nn.Parameter(torch.zeros(100))
    opt = Lion([p], lr=0.1, weight_decay=0.0)
    p.grad = torch.randn(100)
    opt.step()
    assert torch.allclose(p.abs(), torch.full((100,), 0.1), atol=1e-6)


def test_mars_stores_the_previous_gradient():
    model, X, Y = toy_problem()
    opt = MARS(model.parameters(), lr=1e-3)
    opt.zero_grad()
    ((model(X) - Y) ** 2).mean().backward()
    opt.step()
    matrix = next(p for p in model.parameters() if p.dim() >= 2)
    vector = next(p for p in model.parameters() if p.dim() == 1)
    assert "prev_grad" in opt.state[matrix]
    assert "prev_grad" not in opt.state[vector], "1-D params fall back to AdamW"


def test_mars_reduces_to_adamw_when_gamma_is_zero():
    torch.manual_seed(0)
    W = torch.randn(6, 4)
    a = torch.nn.Parameter(W.clone())
    b = torch.nn.Parameter(W.clone())
    oa = MARS([a], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0, gamma=0.0,
              max_grad_norm=0.0)
    ob = AdamW([b], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    for _ in range(20):
        g = torch.randn(6, 4)
        a.grad, b.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_newton_schulz_returns_a_near_orthogonal_matrix():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 64), (48, 48)]:
        G = torch.randn(*shape)
        O = zeropower_via_newtonschulz5(G, steps=5).float()
        assert O.shape == G.shape
        s = torch.linalg.svdvals(O)
        # The quintic iteration deliberately trades exactness for speed; all
        # singular values should still be pulled close to 1.
        assert s.min() > 0.5 and s.max() < 1.5, (shape, s.min(), s.max())
        # It must preserve the direction of the true orthogonal factor.
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        cos = torch.nn.functional.cosine_similarity((U @ Vh).flatten(), O.flatten(), dim=0)
        assert cos > 0.95, (shape, cos)


def test_muon_splits_matrices_from_embeddings_and_vectors():
    model = torch.nn.Module()
    model.emb = torch.nn.Embedding(50, 8)
    model.lin = torch.nn.Linear(8, 8, bias=True)
    model.norm = torch.nn.LayerNorm(8)
    groups = build_muon_param_groups(model, muon_lr=0.02, adamw_lr=1e-3, weight_decay=0.1)
    muon = [g for g in groups if g["use_muon"]][0]
    assert all(p.dim() >= 2 for p in muon["params"])
    ids = {id(p) for p in muon["params"]}
    assert id(model.emb.weight) not in ids, "embeddings must not be orthogonalised"
    assert id(model.lin.weight) in ids
    covered = sum(len(g["params"]) for g in groups)
    assert covered == len(list(model.parameters()))


def test_muon_rejects_a_vector_in_a_muon_group():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = Muon([{"params": [p], "use_muon": True}], lr=0.02)
    p.grad = torch.randn(5)
    with pytest.raises(ValueError):
        opt.step()


def test_sophia_hessian_ema_and_clipping():
    p = torch.nn.Parameter(torch.zeros(10))
    opt = SophiaG([p], lr=0.1, betas=(0.9, 0.99), rho=0.04, weight_decay=0.0)
    # Without any curvature information the ratio saturates: a pure sign step.
    p.grad = torch.randn(10)
    opt.step(bs=1)
    assert torch.allclose(p.abs(), torch.full((10,), 0.1), atol=1e-6)

    # After a Hessian update the state exists and damps the step.
    p.grad = torch.full((10,), 5.0)
    opt.update_hessian()
    assert opt.state[p]["hessian"].abs().sum() > 0
    assert 0.0 <= opt.clipped_fraction() <= 1.0


def test_optimizers_reject_bad_hyperparameters():
    p = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):
        AdamW([p], lr=-1.0)
    with pytest.raises(ValueError):
        AdamW([p], lr=1e-3, betas=(1.2, 0.9))
    with pytest.raises(ValueError):
        Lion([p], lr=-1.0)
    with pytest.raises(ValueError):
        SophiaG([p], lr=1e-3, rho=0.0)
