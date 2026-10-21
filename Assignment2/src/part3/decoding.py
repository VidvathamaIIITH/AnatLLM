"""From-scratch decoding strategies: greedy, top-k, top-p (nucleus) and beam search.

Nothing here calls ``model.generate()``.  Every strategy is built on repeated
``model.forward()`` calls behind a small :class:`LMBackend` abstraction, so the
exact same implementations drive both the pretrained HuggingFace checkpoint of
Part 3 and the from-scratch translation models of Parts 1 and 2.

Incremental decoding uses a key/value cache: the prompt is processed once
(``prefill``) and each subsequent call feeds a single token (``step``).  Beam
search additionally needs to permute the cache along the batch axis whenever
beams are re-ranked (``reorder``), and to fan the batch out by the beam width
at the start (``expand``).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class LMBackend(ABC):
    """Stateful wrapper around a causal LM's ``forward`` + KV cache."""

    @abstractmethod
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Consume the prompt; return next-token logits of shape (B, V)."""

    @abstractmethod
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Consume one token per sequence, shape (B, 1); return logits (B, V)."""

    @abstractmethod
    def reorder(self, index: torch.Tensor) -> None:
        """Permute/duplicate the batch axis of the cache by ``index``."""

    @abstractmethod
    def expand(self, repeats: int) -> None:
        """Repeat each batch element ``repeats`` times, contiguously."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...


class HFBackend(LMBackend):
    """Backend for a HuggingFace ``AutoModelForCausalLM`` (e.g. Pythia-160M)."""

    def __init__(self, model):
        self.model = model
        self._past = None
        self._mask: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        self._past = out.past_key_values
        self._mask = attention_mask
        return out.logits[:, -1, :].float()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(tokens.shape[0], 1, dtype=self._mask.dtype, device=self._mask.device)
        self._mask = torch.cat([self._mask, ones], dim=1)
        out = self.model(input_ids=tokens, attention_mask=self._mask,
                         past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        return out.logits[:, -1, :].float()

    def reorder(self, index: torch.Tensor) -> None:
        self._mask = self._mask.index_select(0, index)
        self._past = _reorder_cache(self._past, index)

    def expand(self, repeats: int) -> None:
        if repeats == 1:
            return
        B = self._mask.shape[0]
        index = torch.arange(B, device=self._mask.device).repeat_interleave(repeats)
        self.reorder(index)


class LocalBackend(LMBackend):
    """Backend for the from-scratch :class:`~src.part1.model.TransformerLM`."""

    def __init__(self, model):
        self.model = model
        self._past = None
        self._mask: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
        self._past = out.past_key_values
        self._mask = attention_mask
        return out.logits[:, -1, :].float()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(tokens.shape[0], 1, dtype=self._mask.dtype, device=self._mask.device)
        self._mask = torch.cat([self._mask, ones], dim=1)
        out = self.model(tokens, attention_mask=self._mask,
                         past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        return out.logits[:, -1, :].float()

    def reorder(self, index: torch.Tensor) -> None:
        self._mask = self._mask.index_select(0, index)
        self._past = [(k.index_select(0, index), v.index_select(0, index))
                      for (k, v) in self._past]

    def expand(self, repeats: int) -> None:
        if repeats == 1:
            return
        B = self._mask.shape[0]
        index = torch.arange(B, device=self._mask.device).repeat_interleave(repeats)
        self.reorder(index)


def _reorder_cache(past, index: torch.Tensor):
    """Permute a HuggingFace KV cache along the batch axis.

    ``transformers`` has changed its cache representation several times, so the
    supported shapes are tried in turn: the modern ``Cache`` objects expose
    ``batch_select_indices`` / ``reorder_cache``, while older releases hand back
    a plain tuple of ``(key, value)`` tensors.
    """
    if past is None:
        return None
    for name in ("batch_select_indices", "reorder_cache"):
        fn = getattr(past, name, None)
        if callable(fn):
            fn(index)
            return past
    if isinstance(past, (list, tuple)):
        return tuple(tuple(t.index_select(0, index) for t in layer) for layer in past)
    raise TypeError(f"do not know how to reorder a cache of type {type(past)!r}")


# --------------------------------------------------------------------------- #
# Config / result
# --------------------------------------------------------------------------- #


@dataclass
class GenerationConfig:
    strategy: str = "greedy"          # greedy | top_k | top_p | beam
    max_new_tokens: int = 64
    beam_width: int = 1
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0
    length_penalty: float = 1.0       # beam: score / length ** length_penalty
    eos_id: Optional[int] = None
    pad_id: int = 0
    seed: Optional[int] = None

    def label(self) -> str:
        if self.strategy == "beam":
            return f"beam-{self.beam_width}"
        if self.strategy == "top_k":
            return f"top-k={self.top_k}(T={self.temperature:g})"
        if self.strategy == "top_p":
            return f"top-p={self.top_p:g}(T={self.temperature:g})"
        return "greedy"


@dataclass
class GenerationResult:
    tokens: list[list[int]]                  # generated ids, EOS kept, padding stripped
    sum_logprob: list[float]                 # model log-prob of what was generated
    n_tokens: list[int]
    seconds: float = 0.0
    forward_calls: int = 0
    field_meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Logit filtering
# --------------------------------------------------------------------------- #


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature == 1.0:
        return logits
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the ``k`` highest-scoring tokens per row; mask the rest."""
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest prefix of the sorted distribution
    whose cumulative probability first reaches ``p``.

    The token that crosses the threshold is *kept*, so the retained mass is
    always >= p and at least one token always survives.
    """
    if p is None or p >= 1.0:
        return logits
    if not 0 < p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Drop everything strictly after the first index where cumsum >= p.
    remove = cumulative - probs > p
    remove[..., 0] = False                       # never drop the argmax
    sorted_logits = sorted_logits.masked_fill(remove, NEG_INF)
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


# --------------------------------------------------------------------------- #
# Sampling-family strategies (greedy / top-k / top-p)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_sampling(
    backend: LMBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: GenerationConfig,
    generator: Optional[torch.Generator] = None,
) -> GenerationResult:
    """Greedy / top-k / top-p decoding over a left-padded batch of prompts."""
    device = backend.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    B = input_ids.shape[0]

    t0 = time.perf_counter()
    logits = backend.prefill(input_ids, attention_mask)
    calls = 1

    out_tokens = torch.full((B, cfg.max_new_tokens), cfg.pad_id, dtype=torch.long, device=device)
    out_logprob = torch.zeros(B, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(cfg.max_new_tokens):
        scores = apply_temperature(logits, cfg.temperature)
        if cfg.strategy == "top_k":
            scores = top_k_filter(scores, cfg.top_k)
        elif cfg.strategy == "top_p":
            scores = top_p_filter(scores, cfg.top_p)
        elif cfg.strategy != "greedy":
            raise ValueError(f"{cfg.strategy!r} is not a sampling strategy")

        logprobs = F.log_softmax(scores, dim=-1)
        if cfg.strategy == "greedy":
            nxt = logprobs.argmax(dim=-1)
        else:
            probs = logprobs.exp()
            nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

        # Score the chosen token under the *unmodified* distribution so that
        # perplexities are comparable across strategies.
        true_lp = F.log_softmax(logits, dim=-1).gather(-1, nxt[:, None]).squeeze(-1)

        alive = ~finished
        out_tokens[:, t] = torch.where(alive, nxt, torch.full_like(nxt, cfg.pad_id))
        out_logprob += torch.where(alive, true_lp, torch.zeros_like(true_lp))
        lengths += alive.long()
        if cfg.eos_id is not None:
            finished = finished | (alive & (nxt == cfg.eos_id))
        if bool(finished.all()) or t == cfg.max_new_tokens - 1:
            break

        feed = torch.where(alive, nxt, torch.full_like(nxt, cfg.pad_id))
        logits = backend.step(feed[:, None])
        calls += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    toks, lens = out_tokens.tolist(), lengths.tolist()
    return GenerationResult(
        tokens=[toks[i][: lens[i]] for i in range(B)],
        sum_logprob=out_logprob.tolist(),
        n_tokens=lens,
        seconds=seconds,
        forward_calls=calls,
    )


# --------------------------------------------------------------------------- #
# Beam search
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_beam(
    backend: LMBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: GenerationConfig,
) -> GenerationResult:
    """Batched beam search.

    The batch axis is fanned out to ``B * K``.  A beam that emits EOS is frozen:
    its log-probability stops accumulating and it can only ever extend itself
    with padding, so it stays in the running until either every beam is done or
    the length cap is reached.  Beams are finally ranked by their
    length-normalised score, ``sum_logprob / length ** length_penalty``.
    """
    K = max(1, int(cfg.beam_width))
    device = backend.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    B = input_ids.shape[0]

    t0 = time.perf_counter()
    logits = backend.prefill(input_ids, attention_mask)          # (B, V)
    calls = 1
    V = logits.shape[-1]

    backend.expand(K)                                            # cache -> (B*K, ...)
    logits = logits.repeat_interleave(K, dim=0)                  # (B*K, V)

    # Only beam 0 of each item is live at t=0, otherwise all K beams would pick
    # the same K-best continuations of an identical prefix.
    beam_scores = torch.full((B, K), NEG_INF, device=device)
    beam_scores[:, 0] = 0.0

    tokens = torch.full((B, K, cfg.max_new_tokens), cfg.pad_id, dtype=torch.long, device=device)
    lengths = torch.zeros(B, K, dtype=torch.long, device=device)
    done = torch.zeros(B, K, dtype=torch.bool, device=device)

    for t in range(cfg.max_new_tokens):
        logprobs = F.log_softmax(logits.float(), dim=-1).view(B, K, V)

        # Frozen beams may only emit padding, at zero cost.
        if bool(done.any()):
            frozen = torch.full((1, 1, V), NEG_INF, device=device).expand(B, K, V).clone()
            frozen[:, :, cfg.pad_id] = 0.0
            logprobs = torch.where(done[:, :, None], frozen, logprobs)

        cand = beam_scores[:, :, None] + logprobs                # (B, K, V)
        flat = cand.view(B, K * V)
        top_scores, top_idx = torch.topk(flat, K, dim=-1)        # (B, K)
        beam_idx = torch.div(top_idx, V, rounding_mode="floor")  # which beam
        token_idx = top_idx % V                                  # which token

        # Re-index every per-beam tensor by the surviving parents.
        gather = beam_idx + torch.arange(B, device=device)[:, None] * K   # flat (B, K)
        flat_gather = gather.view(-1)
        tokens = tokens.view(B * K, -1).index_select(0, flat_gather).view(B, K, -1)
        lengths = lengths.view(-1).index_select(0, flat_gather).view(B, K)
        done = done.view(-1).index_select(0, flat_gather).view(B, K)

        was_done = done.clone()
        tokens[:, :, t] = torch.where(was_done, torch.full_like(token_idx, cfg.pad_id), token_idx)
        lengths = lengths + (~was_done).long()
        beam_scores = top_scores
        if cfg.eos_id is not None:
            done = was_done | (token_idx == cfg.eos_id)

        if bool(done.all()) or t == cfg.max_new_tokens - 1:
            break

        backend.reorder(flat_gather)
        feed = torch.where(was_done, torch.full_like(token_idx, cfg.pad_id), token_idx)
        logits = backend.step(feed.view(B * K, 1))
        calls += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    norm = lengths.clamp_min(1).float() ** cfg.length_penalty
    ranked = (beam_scores / norm).argmax(dim=-1)                 # (B,)

    best_tokens, best_lp, best_len = [], [], []
    tok_list = tokens.tolist()
    len_list = lengths.tolist()
    score_list = beam_scores.tolist()
    for b in range(B):
        j = int(ranked[b])
        L = int(len_list[b][j])
        best_tokens.append(tok_list[b][j][:L])
        best_lp.append(float(score_list[b][j]))
        best_len.append(L)

    return GenerationResult(
        tokens=best_tokens, sum_logprob=best_lp, n_tokens=best_len,
        seconds=seconds, forward_calls=calls,
        field_meta={"beam_width": K},
    )


def generate(
    backend: LMBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: GenerationConfig,
    generator: Optional[torch.Generator] = None,
) -> GenerationResult:
    """Dispatch to the requested decoding strategy."""
    if cfg.strategy == "beam":
        return generate_beam(backend, input_ids, attention_mask, cfg)
    return generate_sampling(backend, input_ids, attention_mask, cfg, generator)


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def strip_after_eos(ids: Sequence[int], eos_id: Optional[int]) -> list[int]:
    """Drop the EOS token and anything after it."""
    out = list(ids)
    if eos_id is None:
        return out
    if eos_id in out:
        out = out[: out.index(eos_id)]
    return out
