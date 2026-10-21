"""Muon - category 4, a *matrix-based* optimizer.

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (2024), listed in Table 1 of *Fantastic Pretraining Optimizers and
Where to Find Them*.

Muon treats a weight matrix as a matrix rather than a bag of scalars: it takes
the momentum buffer :math:`M` and replaces it with the nearest semi-orthogonal
matrix :math:`UV^\top` (from the SVD :math:`M = U\Sigma V^\top`), computed
cheaply by a quintic Newton-Schulz iteration that never forms an SVD.  The
resulting update spreads the step evenly across all singular directions instead
of letting the few dominant ones take over.

Only hidden weight *matrices* are suited to this.  Embeddings, the output head
and every 1-D parameter (norm gains, biases) keep an AdamW update, which is the
configuration used in the paper and in practice; :func:`build_muon_param_groups`
performs that split.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    r"""Approximate the orthogonal factor of ``G`` by a quintic Newton-Schulz iteration.

    Iterates :math:`X \leftarrow aX + (bA + cA^2)X` with :math:`A = XX^\top` and
    the coefficients tuned by Jordan so that the iteration converges very fast
    onto the interval of interest at the price of not converging to exactly 1 -
    which is fine, since only the *direction* of the update matters.

    The iteration runs in bfloat16: it is a fixed-point-style recursion whose
    accuracy requirements are low, and this keeps it cheap.
    """
    if G.ndim < 2:
        raise ValueError("Newton-Schulz orthogonalisation needs a matrix")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon for hidden matrices with an AdamW fallback for everything else.

    Parameter groups carry a ``use_muon`` flag.  Groups with ``use_muon=True``
    get the orthogonalised momentum update; the others get plain AdamW.  Build
    the groups with :func:`build_muon_param_groups`.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay,
                                      adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                                      use_muon=True))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", True):
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    # -- the matrix-based half ------------------------------------------------ #

    def _step_muon(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim < 2:
                raise ValueError(
                    "Muon groups must contain matrices; put 1-D parameters in an "
                    "AdamW group (see build_muon_param_groups)"
                )
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]

            buf.lerp_(g, 1.0 - momentum)                 # buf = m*buf + (1-m)*g
            update = g.lerp(buf, momentum) if group["nesterov"] else buf
            update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])

            # Keep the update's RMS comparable to an AdamW step regardless of
            # the matrix aspect ratio.
            scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update, alpha=-lr * scale)

    # -- the AdamW fallback --------------------------------------------------- #

    def _step_adamw(self, group: dict) -> None:
        beta1, beta2 = group["adamw_betas"]
        lr, eps, wd = group["lr"], group["adamw_eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            bias1 = 1.0 - beta1 ** t
            bias2 = 1.0 - beta2 ** t
            denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
            p.addcdiv_(m, denom, value=-lr / bias1)


def build_muon_param_groups(
    model: torch.nn.Module,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
) -> list[dict]:
    """Split a model into Muon (hidden matrices) and AdamW (everything else).

    Embeddings and the tied output head are excluded from Muon: their rows are
    per-token and orthogonalising across the vocabulary is not meaningful.
    """
    muon_params, adamw_decay, adamw_plain = [], [], []
    seen: set[int] = set()

    embedding_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)
    }
    head = getattr(model, "lm_head", None)
    if head is not None:
        embedding_ids.add(id(head.weight))

    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embedding_ids:
            adamw_decay.append(p)
        elif p.dim() >= 2:
            muon_params.append(p)
        else:
            adamw_plain.append(p)

    groups = [
        {"params": muon_params, "use_muon": True, "lr": muon_lr,
         "weight_decay": weight_decay, "momentum": momentum, "nesterov": nesterov,
         "ns_steps": ns_steps, "adamw_betas": adamw_betas, "adamw_eps": adamw_eps},
        {"params": adamw_decay, "use_muon": False, "lr": adamw_lr,
         "weight_decay": weight_decay, "momentum": momentum, "nesterov": nesterov,
         "ns_steps": ns_steps, "adamw_betas": adamw_betas, "adamw_eps": adamw_eps},
        {"params": adamw_plain, "use_muon": False, "lr": adamw_lr,
         "weight_decay": 0.0, "momentum": momentum, "nesterov": nesterov,
         "ns_steps": ns_steps, "adamw_betas": adamw_betas, "adamw_eps": adamw_eps},
    ]
    return [g for g in groups if g["params"]]
