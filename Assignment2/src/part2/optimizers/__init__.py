"""From-scratch optimizers for Part 2.

One optimizer is implemented per category of Table 1 in *Fantastic Pretraining
Optimizers and Where to Find Them* (Wen et al., 2025):

===========================  ==================  =======================
Category                     Implementation      Optimizer state / param
===========================  ==================  =======================
1. AdamW                     :class:`AdamW`      2 tensors
2. Variance-reduced AdamW    :class:`MARS`       3 tensors
3. Memory-efficient          :class:`Lion`       1 tensor
4. Matrix-based              :class:`Muon`       1 tensor (+AdamW aux)
5. Hessian-based (bonus)     :class:`SophiaG`    2 tensors
===========================  ==================  =======================

Only ``torch.optim.Optimizer`` is imported from torch, strictly as a base class.
"""

from __future__ import annotations

from typing import Any

import torch

from .adamw import AdamW
from .lion import Lion
from .mars import MARS
from .muon import Muon, build_muon_param_groups, zeropower_via_newtonschulz5
from .sophia import SophiaG

__all__ = [
    "AdamW", "Lion", "MARS", "Muon", "SophiaG",
    "build_muon_param_groups", "zeropower_via_newtonschulz5",
    "OPTIMIZERS", "OPTIMIZER_CATEGORY", "build_optimizer", "needs_hessian",
]

OPTIMIZERS = {
    "adamw": AdamW,
    "mars": MARS,
    "lion": Lion,
    "muon": Muon,
    "sophia": SophiaG,
}

OPTIMIZER_CATEGORY = {
    "adamw": "1. AdamW",
    "mars": "2. Variance-reduced AdamW variant",
    "lion": "3. Memory-efficient",
    "muon": "4. Matrix-based",
    "sophia": "5. Hessian-based (bonus)",
}

OPTIMIZER_LABEL = {
    "adamw": "AdamW",
    "mars": "MARS (variance-reduced)",
    "lion": "Lion (memory-efficient)",
    "muon": "Muon (matrix-based)",
    "sophia": "Sophia-G (Hessian-based)",
}


def split_decay_groups(model: torch.nn.Module) -> tuple[list, list]:
    """Matrices get weight decay, 1-D parameters (norms, biases) do not."""
    decay, plain = [], []
    seen: set[int] = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else plain).append(p)
    return decay, plain


def needs_hessian(name: str) -> bool:
    """Whether the optimizer requires the extra sampled-label backward pass."""
    return name == "sophia"


def build_optimizer(name: str, model: torch.nn.Module, hp: dict[str, Any]):
    """Instantiate an optimizer by name from a hyper-parameter dict."""
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")

    lr = float(hp["lr"])
    wd = float(hp.get("weight_decay", 0.0))

    if name == "muon":
        groups = build_muon_param_groups(
            model,
            muon_lr=lr,
            adamw_lr=float(hp.get("adamw_lr", 3e-3)),
            weight_decay=wd,
            adamw_betas=tuple(hp.get("adamw_betas", (0.9, 0.95))),
            adamw_eps=float(hp.get("eps", 1e-8)),
            momentum=float(hp.get("momentum", 0.95)),
            nesterov=bool(hp.get("nesterov", True)),
            ns_steps=int(hp.get("ns_steps", 5)),
        )
        return Muon(groups, lr=lr, momentum=float(hp.get("momentum", 0.95)),
                    nesterov=bool(hp.get("nesterov", True)),
                    ns_steps=int(hp.get("ns_steps", 5)), weight_decay=wd)

    decay, plain = split_decay_groups(model)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": plain, "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]

    if name == "adamw":
        return AdamW(groups, lr=lr, betas=tuple(hp.get("betas", (0.9, 0.95))),
                     eps=float(hp.get("eps", 1e-8)), weight_decay=wd)
    if name == "mars":
        return MARS(groups, lr=lr, betas=tuple(hp.get("betas", (0.95, 0.99))),
                    eps=float(hp.get("eps", 1e-8)), weight_decay=wd,
                    gamma=float(hp.get("gamma", 0.025)),
                    max_grad_norm=float(hp.get("mars_max_grad_norm", 1.0)),
                    correct_1d=bool(hp.get("correct_1d", False)))
    if name == "lion":
        return Lion(groups, lr=lr, betas=tuple(hp.get("betas", (0.9, 0.99))),
                    weight_decay=wd)
    if name == "sophia":
        return SophiaG(groups, lr=lr, betas=tuple(hp.get("betas", (0.965, 0.99))),
                       rho=float(hp.get("rho", 0.04)), weight_decay=wd,
                       eps=float(hp.get("eps", 1e-15)))
    raise AssertionError("unreachable")


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Bytes held in optimizer state - the memory-efficiency axis of Table 1."""
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total
