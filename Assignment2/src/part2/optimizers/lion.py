"""Lion - category 3, a *memory-efficient* optimizer.

Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023),
listed in Table 1 of *Fantastic Pretraining Optimizers and Where to Find Them*.

Lion keeps a single momentum buffer and takes a pure ``sign`` step:

.. math::
    u_t &= \operatorname{sign}\big(\beta_1 m_{t-1} + (1-\beta_1) g_t\big) \\
    \theta_t &= \theta_{t-1} - \eta\,(u_t + \lambda\theta_{t-1}) \\
    m_t &= \beta_2 m_{t-1} + (1-\beta_2) g_t

Note the two different interpolations: the *update* uses :math:`\beta_1` while
the *state* is refreshed with :math:`\beta_2`, which makes the step direction a
slightly "ahead of momentum" Nesterov-like quantity.

Because there is no second-moment buffer, the optimizer state is 1x the model
size instead of AdamW's 2x - that is what puts Lion in the memory-efficient
family.  The update has unit magnitude everywhere, so the learning rate must be
roughly 3-10x smaller than AdamW's and the weight decay correspondingly larger.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """EvoLved Sign Momentum."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]

                # Decoupled weight decay.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Step direction from the *interpolated* momentum ...
                update = m.mul(beta1).add_(g, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # ... then the buffer is updated with the other coefficient.
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss
