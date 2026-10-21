"""Decoder-only transformer whose feed-forward block is a drop-in choice of a
dense MLP or one of several Mixture-of-Experts layers.

The five configurations required by Part 1 are produced purely from
:class:`TransformerConfig` (see :func:`variant_config`):

  1. ``dense``            - standard 2-layer MLP, d_ff = 4 * d_model
  2. ``moe_4e_top1``      - 4 routed experts, top-1      (total params == dense)
  3. ``moe_4e_top2``      - 4 routed experts, top-2      (total params == dense)
  4. ``moe_1s_3r_top1``   - 1 shared + 3 routed, top-1   (total params == dense)
  5. ``moe_4e_top2_wide`` - 4 routed experts, top-2, expert width doubled so
                            that the *active* parameter count matches ``dense``

Parameter book-keeping per layer (ignoring the tiny router matrix), with
``d = d_model`` and dense ``d_ff = 4d``:

  dense                  total 8d^2    active 8d^2
  4 experts of width d   total 8d^2    active 2*k*d^2
  4 experts of width 2d  total 16d^2   active 4*k*d^2
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_ctx: int = 256

    # Feed-forward network -------------------------------------------------- #
    ffn_type: str = "mlp"          # "mlp" | "moe"
    d_ff: int = 1536               # dense MLP hidden width
    n_routed_experts: int = 4      # experts the router chooses among
    n_active_experts: int = 1      # top-k among the routed experts
    n_shared_experts: int = 0      # always-on experts
    d_ff_expert: int = 384         # hidden width of a single expert

    # Router ---------------------------------------------------------------- #
    router_aux_coef: float = 0.01  # Switch-style load-balancing loss weight
    router_z_coef: float = 1e-3    # router logit magnitude regulariser
    router_noise_std: float = 0.0  # optional exploration noise during training

    # Misc ------------------------------------------------------------------ #
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        if self.ffn_type not in {"mlp", "moe"}:
            raise ValueError(f"unknown ffn_type {self.ffn_type!r}")
        if self.ffn_type == "moe":
            if self.n_routed_experts < 1:
                raise ValueError("MoE needs at least one routed expert")
            if not 1 <= self.n_active_experts <= self.n_routed_experts:
                raise ValueError(
                    f"n_active_experts={self.n_active_experts} must lie in "
                    f"[1, n_routed_experts={self.n_routed_experts}]"
                )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    # -- analytic parameter accounting -------------------------------------- #

    def ffn_params_per_layer(self) -> tuple[int, int]:
        """Return ``(total, active)`` feed-forward parameters for one layer."""
        d = self.d_model
        if self.ffn_type == "mlp":
            n = 2 * d * self.d_ff
            return n, n
        per_expert = 2 * d * self.d_ff_expert
        n_experts = self.n_routed_experts + self.n_shared_experts
        router = d * self.n_routed_experts
        total = n_experts * per_expert + router
        active = (self.n_active_experts + self.n_shared_experts) * per_expert + router
        return total, active

    def param_budget(self) -> dict[str, int]:
        """Analytic parameter counts, split into total and per-token active."""
        d, L = self.d_model, self.n_layers
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        attn = L * (4 * d * d)
        norms = L * 2 * d + d
        ffn_total, ffn_active = self.ffn_params_per_layer()
        return {
            "embedding": embed + head,
            "attention": attn,
            "norms": norms,
            "ffn_total": L * ffn_total,
            "ffn_active": L * ffn_active,
            "total": embed + head + attn + norms + L * ffn_total,
            "active": embed + head + attn + norms + L * ffn_active,
        }


#: The five feed-forward ablations required by Part 1.
VARIANTS = (
    "dense",
    "moe_4e_top1",
    "moe_4e_top2",
    "moe_1s_3r_top1",
    "moe_4e_top2_wide",
)

VARIANT_LABELS = {
    "dense": "Dense MLP (baseline)",
    "moe_4e_top1": "MoE 4 experts, top-1",
    "moe_4e_top2": "MoE 4 experts, top-2",
    "moe_1s_3r_top1": "MoE 1 shared + 3 routed, top-1",
    "moe_4e_top2_wide": "MoE 4 wide experts, top-2 (active == dense)",
}


def variant_config(name: str, base: TransformerConfig) -> TransformerConfig:
    """Derive one of the five required FFN configurations from a base config.

    ``base.d_model`` and ``base.d_ff`` define the dense reference point; every
    MoE variant is sized relative to it so the parameter-parity constraints in
    the assignment hold exactly, up to the router matrix (~0.01% of a layer).
    """
    dense_ff = base.d_ff
    if dense_ff % 4 != 0:
        raise ValueError(f"d_ff={dense_ff} must be divisible by 4 to size experts")
    # An expert of width `unit` costs a quarter of the dense MLP, so 4 such
    # experts match the dense layer in *total* parameters.
    unit = dense_ff // 4
    if name == "dense":
        return replace(base, ffn_type="mlp", d_ff=dense_ff)
    if name == "moe_4e_top1":
        return replace(base, ffn_type="moe", n_routed_experts=4, n_active_experts=1,
                       n_shared_experts=0, d_ff_expert=unit)
    if name == "moe_4e_top2":
        return replace(base, ffn_type="moe", n_routed_experts=4, n_active_experts=2,
                       n_shared_experts=0, d_ff_expert=unit)
    if name == "moe_1s_3r_top1":
        return replace(base, ffn_type="moe", n_routed_experts=3, n_active_experts=1,
                       n_shared_experts=1, d_ff_expert=unit)
    if name == "moe_4e_top2_wide":
        # top-2 of 4 experts, each twice as wide -> active params == dense
        return replace(base, ffn_type="moe", n_routed_experts=4, n_active_experts=2,
                       n_shared_experts=0, d_ff_expert=2 * unit)
    raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


def build_rope_cache(n_ctx: int, head_dim: int, theta: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(n_ctx, device=device).float()
    freqs = torch.outer(t, inv_freq)                     # (T, hd/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape (B, H, T, hd); ``cos``/``sin`` have shape (T, hd/2)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary position embeddings."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        S = k.shape[2]
        if key_padding_mask is None and T == S:
            # Fast path: plain causal attention over a fresh, unpadded sequence.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Additive mask combining causality over *absolute* positions with
            # padded keys.  Query i sits at absolute position S - T + i.
            qpos = torch.arange(S - T, S, device=x.device).view(T, 1)
            kpos = torch.arange(S, device=x.device).view(1, S)
            allowed = (kpos <= qpos).view(1, 1, T, S).expand(B, 1, T, S)
            if key_padding_mask is not None:
                allowed = allowed & key_padding_mask[:, :S].view(B, 1, 1, S).bool()
            # A fully masked row would produce NaNs, so let such queries attend
            # to themselves; their outputs are masked out downstream anyway.
            dead = ~allowed.any(dim=-1, keepdim=True)
            allowed = allowed | dead
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y), present


class MLP(nn.Module):
    """The dense baseline: a standard 2-layer feed-forward network."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        # Input shape: (B, N_ctx, D_model) -> output shape: (B, N_ctx, D_model)
        return self.drop(self.fc2(F.gelu(self.fc1(x)))), None


class Expert(nn.Module):
    """A single expert: architecturally identical to :class:`MLP`, narrower."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


@dataclass
class RouterStats:
    """Per-forward routing telemetry, used for the expert-usage analysis."""

    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    topk_idx: Optional[torch.Tensor] = None     # (B, T, k) long
    topk_weight: Optional[torch.Tensor] = None  # (B, T, k) float
    prob_mean: Optional[torch.Tensor] = None    # (E,) mean routing probability


class MoE(nn.Module):
    """Token-choice Mixture-of-Experts feed-forward layer.

    Routing is softmax top-k over ``n_routed_experts`` with the selected gate
    values renormalised to sum to one; ``n_shared_experts`` experts (if any)
    are applied to every token unconditionally.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.k = config.n_active_experts
        self.noise_std = config.router_noise_std

        self.router = nn.Linear(config.d_model, self.n_routed, bias=False)
        self.experts = nn.ModuleList(
            [Expert(config.d_model, config.d_ff_expert) for _ in range(self.n_routed)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(config.d_model, config.d_ff_expert) for _ in range(self.n_shared)]
        )
        self.drop = nn.Dropout(config.dropout)

        # Flipped on by the analysis scripts to retain per-token assignments.
        self.capture_routing = False

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        # Input shape: (B, N_ctx, D_model) -> output shape: (B, N_ctx, D_model)
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N = xf.shape[0]

        router_logits = self.router(xf).float()                       # (N, E)
        if self.training and self.noise_std > 0:
            router_logits = router_logits + self.noise_std * torch.randn_like(router_logits)
        probs = F.softmax(router_logits, dim=-1)
        topw, topi = torch.topk(probs, self.k, dim=-1)                # (N, k)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topw = topw.to(x.dtype)

        out = torch.zeros_like(xf)
        for se in self.shared_experts:
            out = out + se(xf)

        # Dispatch: each (token, expert) assignment is handled by exactly one
        # gather -> expert -> scatter-add.  With a handful of experts this is
        # much cheaper than materialising a dense dispatch tensor.
        flat_expert = topi.reshape(-1)                                # (N*k,)
        flat_weight = topw.reshape(-1)
        flat_token = torch.arange(N, device=x.device).repeat_interleave(self.k)
        for e, expert in enumerate(self.experts):
            sel = (flat_expert == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tok = flat_token[sel]
            y = expert(xf[tok]) * flat_weight[sel].unsqueeze(-1)
            out = out.index_add(0, tok, y.to(out.dtype))

        # --- auxiliary losses, over real (non-padded) tokens only ----------- #
        if token_mask is not None:
            valid = token_mask.reshape(-1).bool()
        else:
            valid = torch.ones(N, dtype=torch.bool, device=x.device)
        n_valid = valid.sum().clamp_min(1)

        p_mean = probs[valid].mean(dim=0)                                       # (E,)
        one_hot = F.one_hot(topi, num_classes=self.n_routed).sum(dim=1).float()  # (N, E)
        f_i = one_hot[valid].sum(dim=0) / (n_valid.float() * self.k)             # sums to 1
        aux_loss = self.n_routed * torch.sum(f_i * p_mean)
        z_loss = torch.logsumexp(router_logits[valid], dim=-1).pow(2).mean()

        stats = RouterStats(
            aux_loss=aux_loss,
            z_loss=z_loss,
            topk_idx=topi.view(B, T, self.k).detach() if self.capture_routing else None,
            topk_weight=topw.view(B, T, self.k).detach() if self.capture_routing else None,
            prob_mean=p_mean.detach(),
        )
        return self.drop(out.view(B, T, D)), stats


def build_ffn(config: TransformerConfig) -> nn.Module:
    """The drop-in switch between the dense and the MoE feed-forward network."""
    return MoE(config) if config.ffn_type == "moe" else MLP(config)


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = build_ffn(config)

    def forward(self, x, cos, sin, key_padding_mask=None, token_mask=None,
                past_kv=None, use_cache=False):
        h, present = self.attn(self.ln1(x), cos, sin, key_padding_mask, past_kv, use_cache)
        x = x + h
        f, stats = self.ffn(self.ln2(x), token_mask)
        x = x + f
        return x, stats, present


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None       # lm_loss + router regularisers
    lm_loss: Optional[torch.Tensor] = None    # pure cross-entropy
    aux_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    past_key_values: Optional[list] = None
    router_stats: list = field(default_factory=list)


class TransformerLM(nn.Module):
    """Decoder-only transformer language model with a swappable FFN."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(config.n_ctx, config.head_dim, config.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init on the residual output projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # -- routing telemetry --------------------------------------------------- #

    def set_capture_routing(self, flag: bool) -> None:
        for m in self.modules():
            if isinstance(m, MoE):
                m.capture_routing = flag

    @property
    def is_moe(self) -> bool:
        return self.config.ffn_type == "moe"

    # -- parameter accounting ------------------------------------------------ #

    def num_parameters(self) -> int:
        """Distinct parameter tensors (tied embedding/head counted once)."""
        seen: dict[int, int] = {}
        for p in self.parameters():
            seen[id(p)] = p.numel()
        return sum(seen.values())

    def num_active_parameters(self) -> int:
        """Parameters touched by a single token (idle experts excluded)."""
        total = self.num_parameters()
        if not self.is_moe:
            return total
        cfg = self.config
        per_expert = 2 * cfg.d_model * cfg.d_ff_expert
        idle = cfg.n_layers * (cfg.n_routed_experts - cfg.n_active_experts) * per_expert
        return total - idle

    # -- forward ------------------------------------------------------------- #

    def _rope(self, T: int, offset: int, device, dtype):
        if self.rope_cos.device != device:
            self.rope_cos = self.rope_cos.to(device)
            self.rope_sin = self.rope_sin.to(device)
        if offset + T > self.rope_cos.shape[0]:
            cos, sin = build_rope_cache(offset + T + 64, self.config.head_dim,
                                        self.config.rope_theta, device)
            self.rope_cos, self.rope_sin = cos, sin
        return (self.rope_cos[offset:offset + T].to(dtype),
                self.rope_sin[offset:offset + T].to(dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ) -> ModelOutput:
        """``input_ids``: (B, N_ctx) -> ``logits``: (B, N_ctx, N_vocab).

        ``attention_mask`` marks real (1) vs padded (0) *keys* over the full
        sequence, cache included.  ``loss_mask`` selects the positions whose
        prediction contributes to the loss.
        """
        B, T = input_ids.shape
        offset = 0
        if past_key_values is not None and past_key_values[0] is not None:
            offset = past_key_values[0][0].shape[2]

        x = self.drop(self.tok_emb(input_ids))
        cos, sin = self._rope(T, offset, x.device, torch.float32)

        token_mask = attention_mask[:, offset:offset + T] if attention_mask is not None else None
        presents: Optional[list] = [] if use_cache else None
        stats_all: list[RouterStats] = []

        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, stats, present = block(x, cos, sin, attention_mask, token_mask, past, use_cache)
            if stats is not None:
                stats_all.append(stats)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = lm_loss = aux = zl = None
        if targets is not None:
            flat_targets = targets.reshape(-1).clone()
            if loss_mask is not None:
                flat_targets[~loss_mask.reshape(-1).bool()] = ignore_index
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                flat_targets,
                ignore_index=ignore_index,
            )
            loss = lm_loss
            if stats_all:
                aux = torch.stack([s.aux_loss for s in stats_all]).mean()
                zl = torch.stack([s.z_loss for s in stats_all]).mean()
                loss = loss + self.config.router_aux_coef * aux + self.config.router_z_coef * zl

        return ModelOutput(
            logits=logits, loss=loss, lm_loss=lm_loss, aux_loss=aux, z_loss=zl,
            past_key_values=presents, router_stats=stats_all,
        )

    # -- checkpoint helpers -------------------------------------------------- #

    def save(self, path: str, extra: Optional[dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        torch.save(
            {"config": asdict(self.config), "state_dict": self.state_dict(), "extra": extra or {}},
            path,
        )

    @classmethod
    def load(cls, path: str, map_location="cpu") -> tuple["TransformerLM", dict]:
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(TransformerConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        return model, blob.get("extra", {})
