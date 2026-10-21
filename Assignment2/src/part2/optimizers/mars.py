"""MARS - category 2, a *variance-reduced* AdamW variant.

Reference: Yuan et al., "MARS: Unleashing the Power of Variance Reduction for
Training Large Models" (2024), listed in Table 1 of *Fantastic Pretraining
Optimizers and Where to Find Them*.

The idea is STORM-style variance reduction.  Instead of feeding the raw
stochastic gradient into Adam's moments, MARS feeds a *corrected* gradient

.. math::
    c_t = g_t + \gamma \frac{\beta_1}{1-\beta_1}\,(g_t - g_{t-1})

which extrapolates along the gradient difference and cancels much of the
minibatch noise.  ``c_t`` is then scaled back to unit norm if it grew too large
(the paper's gradient clipping) and used exactly where ``g_t`` would sit in
AdamW.  The cost is one extra buffer holding the previous gradient.

Following the paper, the correction is applied to matrix-shaped parameters and
1-D parameters (norms, biases) fall back to plain AdamW.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class MARS(Optimizer):
    """MARS-AdamW.

    Optimizer state: three tensors per corrected parameter (m, v and the
    previous gradient), i.e. 3x the model size - the price of variance
    reduction.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.025,
        max_grad_norm: float = 1.0,
        correct_1d: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, gamma=gamma,
                                      max_grad_norm=max_grad_norm,
                                      correct_1d=correct_1d))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            gamma, max_norm = group["gamma"], group["max_grad_norm"]
            correct_1d = group["correct_1d"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("MARS does not support sparse gradients")

                use_vr = correct_1d or p.dim() >= 2
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if use_vr:
                        state["prev_grad"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                if use_vr:
                    prev = state["prev_grad"]
                    # c_t = g_t + gamma * b1/(1-b1) * (g_t - g_{t-1})
                    c = g.add(g - prev, alpha=gamma * beta1 / (1.0 - beta1))
                    prev.copy_(g)
                    if max_norm and max_norm > 0:
                        norm = c.norm()
                        if norm > max_norm:
                            c = c * (max_norm / (norm + 1e-12))
                else:
                    c = g

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(c, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(c, c, value=1.0 - beta2)

                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias1)

        return loss
