"""Normalization layers written from scratch.

Both modules are implemented with elementary tensor ops (mean / var / rsqrt) and
deliberately avoid ``torch.nn.LayerNorm`` / ``torch.nn.functional.layer_norm``.

Statistics are accumulated in float32 even when the surrounding block runs under
autocast (fp16/bf16); this is what every production implementation does, because
``x.pow(2).mean()`` on long sequences overflows/underflows badly in fp16.

Used by the ablation as:
    * C1, C2, C3, C5 -> :class:`LayerNorm`
    * C4             -> :class:`RMSNorm`
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LayerNorm", "RMSNorm", "build_norm"]


class LayerNorm(nn.Module):
    r"""Layer Normalization (Ba et al., 2016), from scratch.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \odot \gamma + \beta

    where the statistics are taken over the last (feature) dimension.

    Args:
        normalized_shape: size of the last dimension (``d_model``).
        eps: numerical-stability constant added to the variance.
        elementwise_affine: whether to learn ``gamma`` / ``beta``.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected last dim {self.normalized_shape}, got {tuple(x.shape)}"
            )
        in_dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(dim=-1, keepdim=True)
        # biased variance (unbiased=False) -- matches nn.LayerNorm
        var = x32.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x32 - mean) * torch.rsqrt(var + self.eps)
        x_hat = x_hat.to(in_dtype)
        if self.elementwise_affine:
            x_hat = x_hat * self.weight + self.bias
        return x_hat

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.normalized_shape}, eps={self.eps}, affine={self.elementwise_affine}"


class RMSNorm(nn.Module):
    r"""Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019), from scratch.

    .. math::
        y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma

    RMSNorm drops the mean-subtraction (re-centering) of LayerNorm and keeps only
    the re-scaling.  That removes one reduction and the bias parameter, which is
    exactly the trade-off configuration **C4** is meant to measure.

    Args:
        normalized_shape: size of the last dimension (``d_model``).
        eps: numerical-stability constant added inside the square root.
        elementwise_affine: whether to learn the ``gamma`` gain.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"RMSNorm expected last dim {self.normalized_shape}, got {tuple(x.shape)}"
            )
        in_dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_hat = (x32 * rms).to(in_dtype)
        if self.elementwise_affine:
            x_hat = x_hat * self.weight
        return x_hat

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.normalized_shape}, eps={self.eps}, affine={self.elementwise_affine}"


def build_norm(kind: str, d_model: int, eps: float | None = None) -> nn.Module:
    """Factory used by the config system.

    Args:
        kind: ``"layernorm"`` / ``"ln"`` or ``"rmsnorm"`` / ``"rms"``.
        d_model: feature width.
        eps: optional override of the per-norm default epsilon.
    """
    key = kind.lower().replace("_", "").replace("-", "")
    if key in {"layernorm", "ln", "layer"}:
        return LayerNorm(d_model, eps=1e-5 if eps is None else eps)
    if key in {"rmsnorm", "rms"}:
        return RMSNorm(d_model, eps=1e-6 if eps is None else eps)
    raise ValueError(f"Unknown normalization '{kind}' (expected 'layernorm' or 'rmsnorm')")
