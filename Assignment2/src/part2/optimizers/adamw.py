"""AdamW - category 1 of Table 1 in *Fantastic Pretraining Optimizers*.

Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019).

Implemented from scratch; only ``torch.optim.Optimizer`` is used, as a base
class.  The implementation is deliberately literal so it can be checked against
``torch.optim.AdamW`` bit-for-bit (see ``tests/test_optimizers.py``).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    r"""Adam with decoupled weight decay.

    .. math::
        m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
        \hat m_t &= m_t / (1-\beta_1^t), \quad \hat v_t = v_t / (1-\beta_2^t) \\
        \theta_t &= \theta_{t-1} - \eta\left(
            \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_{t-1}\right)

    Optimizer state: two tensors per parameter (m and v), i.e. 2x the model size.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"invalid eps: {eps}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # Decoupled weight decay: applied to the weights, not the gradient.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias1)

        return loss
