"""Train one Part 1 ablation (a single FFN variant) on vi/ja -> en translation.

Every variant is trained on an identical, deterministically planned stream of
batches, so the token budget is exactly equal across the five configurations.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..utils import (
    JsonlLogger,
    autocast_dtype,
    configure_stdout,
    cosine_lr,
    deep_update,
    device_report,
    get_device,
    human,
    init_wandb,
    load_yaml,
    parse_overrides,
    save_json,
    set_seed,
    work_dir,
)
from .data import binarise_split, build_tokenizer, collate, plan_training_batches
from .evaluate import evaluate_perplexity
from .model import VARIANT_LABELS, VARIANTS, TransformerConfig, TransformerLM, variant_config


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                    betas: tuple[float, float], eps: float = 1e-8):
    """AdamW with decay applied only to matrix-shaped parameters.

    Part 1 is allowed to use the stock optimizer; the from-scratch optimizer
    implementations live in ``src/part2/optimizers`` as that section requires.
    """
    decay, no_decay = [], []
    seen: set[int] = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def run(cfg: dict, variant: str, out_root: Path) -> dict:
    configure_stdout()
    seed = int(cfg["seed"])
    set_seed(seed)

    device = get_device(cfg.get("device"))
    amp = autocast_dtype(device)
    print(f"[part1] device: {device_report()}")
    print(f"[part1] variant: {variant} - {VARIANT_LABELS[variant]}")

    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]

    tok = build_tokenizer(vocab_size=dcfg["vocab_size"])
    train_split = binarise_split("train", tok, dcfg["n_ctx"])
    val_split = binarise_split("validation", tok, dcfg["n_ctx"])

    base = TransformerConfig(vocab_size=tok.vocab_size, n_ctx=dcfg["n_ctx"], **mcfg)
    model_cfg = variant_config(variant, base)
    model = TransformerLM(model_cfg).to(device)

    n_total = model.num_parameters()
    n_active = model.num_active_parameters()
    print(f"[part1] parameters: total={human(n_total)} active/token={human(n_active)}")

    # ---- deterministic batch plan (identical for every variant) ------------ #
    budget = int(tcfg["token_budget"])
    plan, planned_tokens = plan_training_batches(
        train_split, budget, int(tcfg["micro_batch_tokens"]), seed=seed,
        max_seqs=int(tcfg.get("max_seqs", 256)),
    )
    accum = int(tcfg["grad_accum"])
    total_steps = len(plan) // accum
    plan = plan[: total_steps * accum]
    print(f"[part1] plan: {len(plan):,} micro-batches -> {total_steps:,} optimizer steps "
          f"covering {planned_tokens:,} tokens ({planned_tokens / len(train_split.tokens):.2f} epochs)")

    optimizer = build_optimizer(
        model, float(tcfg["lr"]), float(tcfg["weight_decay"]),
        tuple(tcfg.get("betas", (0.9, 0.95))),
    )

    out_dir = out_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"variant": variant, "config": cfg, "model_config": model_cfg,
                  "params_total": n_total, "params_active": n_active,
                  "param_budget": model_cfg.param_budget(),
                  "planned_tokens": planned_tokens, "total_steps": total_steps}
    save_json(run_config, out_dir / "run_config.json")
    wb = init_wandb(cfg, name=f"part1-{variant}", group="part1-moe", job_type="train",
                    tags=["part1", variant], config=run_config,
                    notes=VARIANT_LABELS[variant])
    logger = JsonlLogger(out_dir / "train_log.jsonl", wandb_run=wb)

    log_every = int(tcfg.get("log_every", 25))
    eval_every = int(tcfg.get("eval_every", 250))
    eval_examples = int(tcfg.get("eval_examples", 8000))
    warmup = int(tcfg.get("warmup_steps", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    min_ratio = float(tcfg.get("min_lr_ratio", 0.1))

    model.train()
    t_start = time.perf_counter()
    tokens_seen = 0
    running = {"lm": 0.0, "aux": 0.0, "n": 0}
    best_val = float("inf")
    history: list[dict] = []

    for step in range(total_steps):
        lr = cosine_lr(step, base_lr=float(tcfg["lr"]), warmup=warmup,
                       total=total_steps, min_ratio=min_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_tokens = 0
        for micro in range(accum):
            batch = collate(train_split, plan[step * accum + micro], tok.pad_id, device)
            ctx = (torch.autocast("cuda", dtype=amp) if amp is not None and device.type == "cuda"
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                out = model(
                    batch["input_ids"], targets=batch["targets"],
                    attention_mask=batch["attention_mask"], loss_mask=batch["loss_mask"],
                )
            (out.loss / accum).backward()
            step_tokens += int(batch["n_tokens"])
            running["lm"] += float(out.lm_loss.detach())
            running["aux"] += float(out.aux_loss.detach()) if out.aux_loss is not None else 0.0
            running["n"] += 1

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tokens_seen += step_tokens

        if (step + 1) % log_every == 0 or step == 0:
            n = max(1, running["n"])
            elapsed = time.perf_counter() - t_start
            rec = {
                "event": "train", "step": step + 1, "tokens": tokens_seen,
                "lm_loss": running["lm"] / n, "aux_loss": running["aux"] / n,
                "lr": lr, "grad_norm": float(gnorm), "elapsed_s": elapsed,
                "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
            }
            logger.log(**rec)
            print(f"[{variant}] step {step+1:>5}/{total_steps} "
                  f"loss {rec['lm_loss']:.4f} aux {rec['aux_loss']:.3f} "
                  f"lr {lr:.2e} |g| {rec['grad_norm']:.2f} "
                  f"{rec['tokens_per_s']:,.0f} tok/s "
                  f"({tokens_seen/budget*100:.1f}%)", flush=True)
            running = {"lm": 0.0, "aux": 0.0, "n": 0}

        is_last = step == total_steps - 1
        if (step + 1) % eval_every == 0 or is_last:
            res = evaluate_perplexity(
                model, val_split, tok.pad_id, device,
                max_tokens=int(tcfg.get("eval_micro_tokens", 2048)),
                max_examples=eval_examples, amp_dtype=amp, seed=seed,
            )
            rec = {"event": "eval", "step": step + 1, "tokens": tokens_seen, **res.as_dict()}
            logger.log(**rec)
            history.append(rec)
            print(f"[{variant}] eval @ step {step+1}: val loss(all) {res.loss_all:.4f} "
                  f"ppl(all) {res.ppl_all:.2f} | loss(tgt) {res.loss_target:.4f} "
                  f"ppl(tgt) {res.ppl_target:.2f}", flush=True)
            if res.loss_target < best_val:
                best_val = res.loss_target
                model.save(str(out_dir / "best.pt"),
                           extra={"variant": variant, "step": step + 1,
                                  "tokens": tokens_seen, "val": res.as_dict()})

    elapsed = time.perf_counter() - t_start
    model.save(str(out_dir / "final.pt"),
               extra={"variant": variant, "steps": total_steps, "tokens": tokens_seen})
    summary = {
        "variant": variant, "label": VARIANT_LABELS[variant],
        "params_total": n_total, "params_active": n_active,
        "tokens_trained": tokens_seen, "steps": total_steps,
        "train_seconds": elapsed, "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
        "best_val_loss_target": best_val, "history": history,
    }
    save_json(summary, out_dir / "summary.json")
    logger.close()
    if wb is not None:
        wb.summary.update({k: v for k, v in summary.items() if k != "history"})
        wb.finish()
    print(f"[part1] {variant} done in {elapsed/60:.1f} min; "
          f"best val target loss {best_val:.4f} (ppl {math.exp(min(best_val,50)):.2f})")
    return summary


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Train a Part 1 FFN ablation")
    ap.add_argument("--config", default="configs/part1.yaml")
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", default=None, help="output root (default: <work>/part1/runs)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. --set train.lr=3e-4")
    args = ap.parse_args(argv)

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    out_root = Path(args.out) if args.out else work_dir() / "part1" / "runs"
    run(cfg, args.variant, out_root)


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
