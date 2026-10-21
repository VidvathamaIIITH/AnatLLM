"""Part 3: compare decoding strategies on ROCStories with EleutherAI/Pythia-160M.

``model.generate()`` is never called - every strategy in ``decoding.py`` is built
on ``model.forward()`` plus an explicit KV cache.

    python -m src.part3.run --config configs/part3.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..utils import (
    autocast_dtype,
    configure_stdout,
    deep_update,
    device_report,
    get_device,
    init_wandb,
    load_yaml,
    parse_overrides,
    save_json,
    set_seed,
    work_dir,
)
from .data import Example, hf_cache_dir, load_examples
from .decoding import GenerationConfig, HFBackend, generate
from .metrics import evaluate_generations, perplexity_from_logprobs

MODEL_ID = "EleutherAI/pythia-160m"
_SENT_RE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")


def file_label(label: str) -> str:
    """``top-k=10(T=1)`` -> ``top-k=10_T=1``: a label that is safe as a filename."""
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", label).strip("_")


def write_generations(path: Path, examples: Sequence[Example], out: dict) -> None:
    """One JSON line per story: prompt, reference, raw and scored generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i, ex in enumerate(examples):
            fh.write(json.dumps({
                "id": i, "prompt": ex.prompt, "reference": ex.reference,
                "generation": out["hypotheses_raw"][i],
                "generation_scored": out["hypotheses"][i],
                "n_tokens": out["n_tokens"][i], "sum_logprob": out["sum_logprob"][i],
            }, ensure_ascii=False) + "\n")


def truncate_sentences(text: str, n: int = 4) -> str:
    """Keep at most the first ``n`` sentences.

    The reference continuation is always exactly four sentences, while the model
    simply runs until the token budget is exhausted.  Truncating puts the two on
    the same footing; ``mean_hyp_words`` in the metrics shows what survives.
    """
    text = text.strip()
    parts = [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]
    return " ".join(parts[:n]).strip() if parts else text


def load_model(device: torch.device, dtype: Optional[torch.dtype]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=hf_cache_dir())
    model.to(device=device, dtype=dtype or torch.float32)
    model.eval()
    return model, tok


def encode_prompts(tok, examples: Sequence[Example], device: torch.device):
    """Left-padded prompt batch.

    Left padding keeps every sequence's last real token in the final column, so
    a single 'next token' step is well defined for the whole batch.  Pythia uses
    rotary embeddings, which depend only on *relative* offsets, so the per-row
    shift introduced by padding does not change the result (checked in
    ``tests/test_decoding.py``).
    """
    enc = [tok(ex.prompt, add_special_tokens=False)["input_ids"] for ex in examples]
    T = max(len(e) for e in enc)
    pad = tok.pad_token_id
    ids = torch.full((len(enc), T), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), T), dtype=torch.long)
    for r, e in enumerate(enc):
        ids[r, T - len(e):] = torch.tensor(e, dtype=torch.long)
        mask[r, T - len(e):] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def reference_perplexity(model, tok, examples: Sequence[Example], device: torch.device,
                         batch_size: int = 8) -> dict:
    """Teacher-forced perplexity of the gold continuations.

    Independent of the decoding strategy, so it is reported once as an anchor.
    """
    total_nll = 0.0
    total_tok = 0
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        prompt_ids = [tok(ex.prompt, add_special_tokens=False)["input_ids"] for ex in chunk]
        cont_ids = [tok(" " + ex.reference, add_special_tokens=False)["input_ids"]
                    for ex in chunk]
        full = [p + c for p, c in zip(prompt_ids, cont_ids)]
        T = max(len(f) for f in full)
        pad = tok.pad_token_id
        ids = torch.full((len(full), T), pad, dtype=torch.long)
        mask = torch.zeros((len(full), T), dtype=torch.long)
        loss_mask = torch.zeros((len(full), T), dtype=torch.bool)
        for r, f in enumerate(full):
            ids[r, :len(f)] = torch.tensor(f, dtype=torch.long)
            mask[r, :len(f)] = 1
            loss_mask[r, len(prompt_ids[r]):len(f)] = True

        ids, mask, loss_mask = ids.to(device), mask.to(device), loss_mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1), reduction="none",
        ).view(ids.size(0), -1)
        sel = loss_mask[:, 1:]
        total_nll += float((nll * sel).sum())
        total_tok += int(sel.sum())

    loss = total_nll / max(1, total_tok)
    return {"loss": loss, "ppl": math.exp(min(loss, 50.0)), "tokens": total_tok}


def build_configs(cfg: dict, eos_id: Optional[int], pad_id: int) -> list[GenerationConfig]:
    """Expand the YAML into the list of decoding configurations to compare."""
    g = cfg["generate"]
    max_new = int(g["max_new_tokens"])
    out: list[GenerationConfig] = []

    if g.get("greedy", True):
        out.append(GenerationConfig(strategy="greedy", max_new_tokens=max_new,
                                    eos_id=eos_id, pad_id=pad_id))
    for w in g.get("beam_widths", []):
        out.append(GenerationConfig(strategy="beam", beam_width=int(w),
                                    max_new_tokens=max_new, eos_id=eos_id, pad_id=pad_id,
                                    length_penalty=float(g.get("length_penalty", 1.0))))
    for k in g.get("top_k", []):
        for t in g.get("temperatures", [1.0]):
            out.append(GenerationConfig(strategy="top_k", top_k=int(k), temperature=float(t),
                                        max_new_tokens=max_new, eos_id=eos_id, pad_id=pad_id))
    for p in g.get("top_p", []):
        for t in g.get("temperatures", [1.0]):
            out.append(GenerationConfig(strategy="top_p", top_p=float(p), temperature=float(t),
                                        max_new_tokens=max_new, eos_id=eos_id, pad_id=pad_id))
    return out


def run_strategy(model, tok, examples, device, gcfg: GenerationConfig,
                 batch_size: int, seed: int, n_sentences: int) -> dict:
    """Decode the whole test set with one configuration and score it."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    hyps_raw: list[str] = []
    sum_lp: list[float] = []
    n_tok: list[int] = []
    seconds = 0.0
    forwards = 0

    t_wall = time.perf_counter()
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        ids, mask = encode_prompts(tok, chunk, device)
        backend = HFBackend(model)
        res = generate(backend, ids, mask, gcfg, generator=generator)
        seconds += res.seconds
        forwards += res.forward_calls
        for j in range(len(chunk)):
            hyps_raw.append(tok.decode(res.tokens[j], skip_special_tokens=True))
        sum_lp.extend(res.sum_logprob)
        n_tok.extend(res.n_tokens)
    wall = time.perf_counter() - t_wall

    hyps = [truncate_sentences(h, n_sentences) for h in hyps_raw]
    refs = [ex.reference for ex in examples]
    metrics = evaluate_generations(hyps, refs)
    metrics.update({
        "strategy": gcfg.strategy,
        "label": gcfg.label(),
        "beam_width": gcfg.beam_width if gcfg.strategy == "beam" else None,
        "top_k": gcfg.top_k if gcfg.strategy == "top_k" else None,
        "top_p": gcfg.top_p if gcfg.strategy == "top_p" else None,
        "temperature": gcfg.temperature,
        "gen_perplexity": perplexity_from_logprobs(sum_lp, n_tok),
        "mean_gen_tokens": sum(n_tok) / max(1, len(n_tok)),
        "decode_seconds": seconds,
        "wall_seconds": wall,
        "seconds_per_example": wall / max(1, len(examples)),
        "forward_calls": forwards,
    })
    return {"metrics": metrics, "hypotheses": hyps, "hypotheses_raw": hyps_raw,
            "sum_logprob": sum_lp, "n_tokens": n_tok}


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Part 3 decoding-strategy comparison")
    ap.add_argument("--config", default="configs/part3.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    out_dir = Path(args.out) if args.out else work_dir() / "part3"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(cfg.get("device"))
    dtype = torch.float16 if (cfg.get("fp16", True) and device.type == "cuda") else torch.float32
    print(f"[part3] device: {device_report()}  dtype={dtype}")

    model, tok = load_model(device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[part3] {MODEL_ID}: {n_params/1e6:.1f}M parameters, vocab {len(tok)}")

    examples = load_examples(split=cfg["data"]["split"], n=int(cfg["data"]["n_examples"]),
                            seed=seed)
    print(f"[part3] {len(examples)} ROCStories examples "
          f"(prompt = sentence 1, reference = sentences 2-5)")

    ref_ppl = reference_perplexity(model, tok, examples, device,
                                   batch_size=int(cfg["eval"].get("ppl_batch_size", 8)))
    print(f"[part3] reference-continuation perplexity: {ref_ppl['ppl']:.2f} "
          f"({ref_ppl['tokens']:,} tokens)")

    wb = init_wandb(cfg, name="part3-decoding", group="part3-decoding", job_type="eval",
                    tags=["part3", "pythia-160m", "rocstories"], config=cfg,
                    notes=f"{MODEL_ID} on {len(examples)} ROCStories test stories")
    if wb is not None:
        wb.summary["reference_perplexity"] = ref_ppl["ppl"]

    eos_id = tok.eos_token_id if cfg["generate"].get("stop_at_eos", True) else None
    configs = build_configs(cfg, eos_id, tok.pad_token_id)
    batch_size = int(cfg["generate"].get("batch_size", 16))
    n_sentences = int(cfg["generate"].get("truncate_sentences", 4))

    results = []
    generations = {}
    for gcfg in configs:
        bs = batch_size
        if gcfg.strategy == "beam":
            # Beam search fans the batch out by the beam width, so shrink the
            # batch to keep the memory footprint comparable.
            bs = max(1, batch_size // max(1, gcfg.beam_width))
        print(f"[part3] decoding with {gcfg.label()} (batch {bs}) ...", flush=True)
        out = run_strategy(model, tok, examples, device, gcfg, bs, seed, n_sentences)
        m = out["metrics"]
        m["batch_size"] = bs
        results.append(m)
        generations[gcfg.label()] = out["hypotheses"]
        write_generations(out_dir / "generations" / f"{file_label(gcfg.label())}.jsonl",
                          examples, out)
        print(f"[part3]   {gcfg.label():16s} ppl(gen) {m['gen_perplexity']:.2f} "
              f"acc {m['accuracy']:.4f} F1 {m['f1']:.4f} BLEU {m['bleu']:.2f} "
              f"rep4 {m['repetition_4gram']:.3f} {m['wall_seconds']:.1f}s", flush=True)
        if wb is not None:
            wb.log({f"{gcfg.label()}/{k}": v for k, v in m.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})

    payload = {
        "model": MODEL_ID,
        "model_params": n_params,
        "n_examples": len(examples),
        "reference_perplexity": ref_ppl,
        "config": cfg,
        "results": results,
    }
    save_json(payload, out_dir / "results.json")

    samples = []
    for i in range(min(10, len(examples))):
        row = {"prompt": examples[i].prompt, "reference": examples[i].reference}
        for label, hyps in generations.items():
            row[label] = hyps[i]
        samples.append(row)
    save_json(samples, out_dir / "samples.json")
    save_json(generations, out_dir / "generations.json")

    if wb is not None:
        import wandb

        cols = ["label", "strategy", "gen_perplexity", "accuracy", "accuracy_macro",
                "precision", "recall", "f1", "bleu", "chrf2", "distinct_2",
                "repetition_4gram", "mean_gen_tokens", "wall_seconds"]
        wb.log({"results": wandb.Table(columns=cols,
                                       data=[[m[c] for c in cols] for m in results])})
        art = wandb.Artifact("part3-raw-generations", type="generations",
                             description="Raw generations of every decoding strategy, "
                                         "one JSONL file per configuration")
        art.add_dir(str(out_dir / "generations"))
        art.add_file(str(out_dir / "results.json"))
        wb.log_artifact(art)
        wb.finish()

    print(f"\n[part3] wrote {out_dir / 'results.json'} and {out_dir / 'generations'}")
    header = (f"{'strategy':18s} {'ppl(gen)':>9s} {'acc':>7s} {'F1':>7s} "
              f"{'BLEU':>6s} {'rep-4':>6s} {'sec':>7s}")
    print(header)
    print("-" * len(header))
    for m in results:
        print(f"{m['label']:18s} {m['gen_perplexity']:9.2f} {m['accuracy']:7.4f} "
              f"{m['f1']:7.4f} {m['bleu']:6.2f} {m['repetition_4gram']:6.3f} "
              f"{m['wall_seconds']:7.1f}")


if __name__ == "__main__":  # pragma: no cover
    main()
