"""Sophia (SophiaG) - category 5, the *Hessian-based* optimizer (bonus).

Reference: Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer for
Language Model Pre-training" (2023), the second-order entry in Table 1 of
*Fantastic Pretraining Optimizers and Where to Find Them*.

Sophia keeps an EMA of a cheap *diagonal Hessian* estimate and pre-conditions
the momentum with it, then clips the resulting step element-wise:

.. math::
    \theta_t = \theta_{t-1} - \eta \cdot
        \operatorname{clip}\!\left(\frac{m_t}{\max(\rho\,B\,h_t,\ \epsilon)},\ 1\right)

The diagonal Hessian uses the Gauss-Newton-Bartlett estimator: labels are
*resampled from the model's own predictive distribution*, the gradient of that
sampled loss is taken, and its square is an unbiased estimate of the GN
diagonal.  That costs one extra forward/backward, so it is refreshed only every
``k`` steps (``hessian_every`` in the training config), amortising the overhead
to a few percent.

Because the estimator needs a second backward pass, the training loop drives it
explicitly:

    loss.backward();      opt.step(bs=n_tokens);      opt.zero_grad()
    # every k steps:
    sampled_loss.backward(); opt.update_hessian();    opt.zero_grad()
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class SophiaG(Optimizer):
    """SophiaG with the Gauss-Newton-Bartlett diagonal Hessian estimator.

    Optimizer state: three tensors per parameter (m, h and the step counter),
    i.e. the same 2x-plus-a-bit footprint as AdamW.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        eps: float = 1e-15,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if rho <= 0.0:
            raise ValueError(f"invalid rho: {rho}")
        super().__init__(params, dict(lr=lr, betas=betas, rho=rho,
                                      weight_decay=weight_decay, eps=eps))

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["hessian"] = torch.zeros_like(p)
        return state

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Fold the current ``p.grad`` (of the *sampled* loss) into the Hessian EMA.

        Call this right after backpropagating the resampled-label loss; the
        squared gradient is the Gauss-Newton-Bartlett estimate of the diagonal.
        """
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(p)
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, bs: int = 1):
        """``bs`` is the number of predicted tokens in the minibatch.

        It rescales the Hessian estimate, which is computed as a *mean* over the
        batch while the theory calls for a sum.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, _ = group["betas"]
            lr, rho, wd, eps = group["lr"], group["rho"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients")
                state = self._init_state(p)
                state["step"] += 1
                m, h = state["exp_avg"], state["hessian"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)

                # Element-wise clipped, Hessian-preconditioned step.  Where the
                # curvature estimate is tiny the ratio saturates at 1, so the
                # update degenerates gracefully to a signSGD step.
                ratio = (m.abs() / (rho * bs * h + eps)).clamp(max=1.0)
                p.addcmul_(m.sign(), ratio, value=-lr)

        return loss

    # -- diagnostics ---------------------------------------------------------- #

    @torch.no_grad()
    def clipped_fraction(self) -> float:
        """Share of coordinates whose step hit the clipping bound.

        The Sophia paper tracks this: a healthy run sits well below 1, meaning
        the curvature estimate is actually doing the scaling.
        """
        clipped = 0
        total = 0
        for group in self.param_groups:
            rho, eps = group["rho"], group["eps"]
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                m, h = state["exp_avg"], state["hessian"]
                ratio = m.abs() / (rho * h + eps)
                clipped += int((ratio >= 1.0).sum())
                total += m.numel()
        return clipped / max(1, total)
