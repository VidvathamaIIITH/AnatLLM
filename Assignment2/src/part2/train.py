"""Train the dense Part 1 architecture with one from-scratch optimizer.

The training budget is one pass over ``browndw/human-ai-parallel-corpus``
("1x the dataset"), with validation loss and test BLEU measured every 0.1x so
the training dynamics can be compared the way Figure 2 of *Fantastic
Pretraining Optimizers and Where to Find Them* compares them.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..part1.model import TransformerConfig, TransformerLM, variant_config
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
from .data import binarise_split, build_tokenizer, epoch_order, make_batch
from .evaluate import evaluate_bleu, evaluate_loss
from .optimizers import (
    OPTIMIZER_CATEGORY,
    OPTIMIZER_LABEL,
    OPTIMIZERS,
    build_optimizer,
    needs_hessian,
    optimizer_state_bytes,
)

#: Chinchilla-optimal token count is ~20 tokens per parameter.
CHINCHILLA_TOKENS_PER_PARAM = 20


def hyperparams_for(cfg: dict, name: str) -> dict:
    """Merge the shared optimizer defaults with the per-optimizer overrides."""
    hp = dict(cfg["optim"].get("common", {}))
    hp.update(cfg["optim"]["per_optimizer"].get(name, {}))
    return hp


def run(cfg: dict, opt_name: str, out_root: Path) -> dict:
    configure_stdout()
    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(cfg.get("device"))
    amp = autocast_dtype(device)
    print(f"[part2] device: {device_report()}")
    print(f"[part2] optimizer: {opt_name} - {OPTIMIZER_CATEGORY[opt_name]}")

    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    tok = build_tokenizer(vocab_size=dcfg["vocab_size"])
    train_split = binarise_split("train", tok, dcfg["n_ctx"])
    val_split = binarise_split("validation", tok, dcfg["n_ctx"])
    test_split = binarise_split("test", tok, dcfg["n_ctx"])

    base = TransformerConfig(vocab_size=tok.vocab_size, n_ctx=dcfg["n_ctx"], **mcfg)
    model_cfg = variant_config("dense", base)          # same FFN as Part 1 config (1)
    model = TransformerLM(model_cfg).to(device)
    n_params = model.num_parameters()
    chinchilla = CHINCHILLA_TOKENS_PER_PARAM * n_params

    hp = hyperparams_for(cfg, opt_name)
    optimizer = build_optimizer(opt_name, model, hp)
    for g in optimizer.param_groups:                   # remember each group's base LR
        g["base_lr"] = g["lr"]

    micro_blocks = int(tcfg["micro_batch_blocks"])
    accum = int(tcfg["grad_accum"])
    blocks_per_step = micro_blocks * accum
    full_steps = train_split.n_blocks // blocks_per_step
    # `fraction` < 1 shortens the run; it is what the learning-rate sweep uses.
    fraction = float(tcfg.get("fraction", 1.0))
    total_steps = max(1, int(round(full_steps * fraction)))
    dataset_tokens = train_split.n_tokens
    budget_tokens = total_steps * blocks_per_step * dcfg["n_ctx"]

    print(f"[part2] params={human(n_params)}  dataset={dataset_tokens:,} tokens "
          f"({dataset_tokens / chinchilla:.3f}x Chinchilla for this model)")
    print(f"[part2] {total_steps:,} steps x {blocks_per_step} blocks x {dcfg['n_ctx']} "
          f"tokens = {budget_tokens:,} tokens ({fraction:.2f}x dataset)")

    order = epoch_order(train_split, seed)             # identical for every optimizer

    # Checkpoints sit at fixed fractions *of the whole dataset*, so a shortened
    # run lines up with the corresponding prefix of a full run.
    eval_steps: dict[int, float] = {}
    for i in range(1, 11):
        f = round(0.1 * i, 1)
        s = max(1, int(round(f * full_steps)))
        if s <= total_steps:
            eval_steps[s] = f
    eval_steps[total_steps] = round(total_steps / max(1, full_steps), 3)

    out_dir = out_root / opt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # The LR sweep passes its own name/group/tags so trials stay separate from
    # the final runs in the WandB project.
    wmeta = cfg.get("wandb_run", {})
    wb = init_wandb(cfg, name=wmeta.get("name", f"part2-{opt_name}"),
                    group=wmeta.get("group", "part2-optimizers"),
                    job_type=wmeta.get("job_type", "train"),
                    tags=wmeta.get("tags", ["part2", opt_name]),
                    config={"optimizer": opt_name, "category": OPTIMIZER_CATEGORY[opt_name],
                            "hyperparams": hp, "params": n_params,
                            "fraction": fraction, "total_steps": total_steps, **cfg},
                    notes=OPTIMIZER_LABEL[opt_name])
    logger = JsonlLogger(out_dir / "train_log.jsonl", wandb_run=wb)

    warmup = int(tcfg.get("warmup_steps", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    min_ratio = float(tcfg.get("min_lr_ratio", 0.1))
    log_every = int(tcfg.get("log_every", 50))
    hess_every = int(hp.get("hessian_every", 10))
    use_hessian = needs_hessian(opt_name)
    hess_blocks = min(int(hp.get("hessian_blocks", 4)), micro_blocks)
    hess_bs = hess_blocks * int(dcfg["n_ctx"])

    ecfg = cfg["eval"]
    model.train()
    t0 = time.perf_counter()
    tokens_seen = 0
    running = {"loss": 0.0, "n": 0}
    curve: list[dict] = []
    state_bytes_logged = False

    for step in range(total_steps):
        mult = cosine_lr(step, base_lr=1.0, warmup=warmup, total=total_steps,
                         min_ratio=min_ratio)
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * mult

        optimizer.zero_grad(set_to_none=True)
        step_tokens = 0
        last_batch = None
        for micro in range(accum):
            s = (step * accum + micro) * micro_blocks
            batch = make_batch(train_split, order[s:s + micro_blocks], device)
            last_batch = batch
            ctx = (torch.autocast("cuda", dtype=amp) if amp is not None and device.type == "cuda"
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                out = model(batch["input_ids"], targets=batch["targets"])
            (out.loss / accum).backward()
            step_tokens += int(batch["n_tokens"])
            running["loss"] += float(out.lm_loss.detach())
            running["n"] += 1

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_hessian:
            # `bs` converts Sophia's mean-over-batch curvature estimate back into
            # a sum, so it must match the batch the estimate was taken over.
            optimizer.step(bs=hess_bs)
        else:
            optimizer.step()
        tokens_seen += step_tokens

        # --- Sophia's Gauss-Newton-Bartlett curvature refresh ---------------- #
        if use_hessian and (step + 1) % hess_every == 0:
            optimizer.zero_grad(set_to_none=True)
            # The estimator is stochastic anyway, so a slice of the last batch is
            # enough; materialising float32 logits for the whole batch on top of
            # the training activations would not fit in 4 GiB.
            hb = min(hess_blocks, last_batch["input_ids"].shape[0])
            ctx = (torch.autocast("cuda", dtype=amp) if amp is not None and device.type == "cuda"
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                logits = model(last_batch["input_ids"][:hb]).logits
            flat = logits.float().view(-1, logits.size(-1))
            with torch.no_grad():
                # Resample labels from the model's own predictive distribution.
                sampled = torch.multinomial(F.softmax(flat, dim=-1), 1).squeeze(-1)
            F.cross_entropy(flat, sampled).backward()
            optimizer.update_hessian()
            optimizer.zero_grad(set_to_none=True)
            del logits, flat, sampled

        if not state_bytes_logged and step >= 1:
            state_bytes = optimizer_state_bytes(optimizer)
            print(f"[part2] optimizer state: {state_bytes / 1024**2:.1f} MiB "
                  f"({state_bytes / (n_params * 4):.2f}x model fp32 size)")
            state_bytes_logged = True

        if (step + 1) % log_every == 0 or step == 0:
            n = max(1, running["n"])
            elapsed = time.perf_counter() - t0
            rec = {"event": "train", "step": step + 1, "tokens": tokens_seen,
                   "loss": running["loss"] / n, "lr_mult": mult,
                   "grad_norm": float(gnorm), "elapsed_s": elapsed,
                   "tokens_per_s": tokens_seen / max(elapsed, 1e-9)}
            logger.log(**rec)
            print(f"[{opt_name}] step {step+1:>5}/{total_steps} loss {rec['loss']:.4f} "
                  f"lrx {mult:.3f} |g| {rec['grad_norm']:.2f} "
                  f"{rec['tokens_per_s']:,.0f} tok/s ({tokens_seen/budget_tokens*100:.0f}%)",
                  flush=True)
            running = {"loss": 0.0, "n": 0}

        if (step + 1) in eval_steps:
            frac = eval_steps[step + 1]
            val = evaluate_loss(model, val_split, device,
                                batch_blocks=int(ecfg.get("batch_blocks", 8)),
                                max_blocks=int(ecfg.get("val_blocks", 1200)), amp_dtype=amp)
            if ecfg.get("skip_bleu", False):
                bleu = {"bleu": float("nan"), "chrf2": float("nan")}
            else:
                bleu = evaluate_bleu(model, test_split, tok, device,
                                     n_prompts=int(ecfg.get("bleu_prompts", 250)),
                                     prompt_tokens=int(ecfg.get("prompt_tokens", 128)),
                                     gen_tokens=int(ecfg.get("gen_tokens", 48)),
                                     batch_size=int(ecfg.get("gen_batch_size", 25)),
                                     amp_dtype=amp)
            rec = {"event": "eval", "step": step + 1, "fraction": frac,
                   "tokens": tokens_seen,
                   "chinchilla_ratio": tokens_seen / chinchilla,
                   "val_loss": val["loss"], "val_ppl": val["ppl"],
                   "test_bleu": bleu["bleu"], "test_chrf2": bleu["chrf2"]}
            logger.log(**rec, bleu_detail=bleu)
            curve.append(rec)
            print(f"[{opt_name}] {frac:.1f}x dataset ({tokens_seen:,} tok, "
                  f"{tokens_seen/chinchilla:.3f}x Chinchilla): val loss {val['loss']:.4f} "
                  f"ppl {val['ppl']:.2f} | test BLEU {bleu['bleu']:.2f} "
                  f"chrF2 {bleu['chrf2']:.2f}", flush=True)

    elapsed = time.perf_counter() - t0
    if tcfg.get("save_checkpoint", True):
        model.save(str(out_dir / "final.pt"),
                   extra={"optimizer": opt_name, "steps": total_steps, "tokens": tokens_seen})
    summary = {
        "optimizer": opt_name,
        "label": OPTIMIZER_LABEL[opt_name],
        "category": OPTIMIZER_CATEGORY[opt_name],
        "hyperparams": hp,
        "params": n_params,
        "chinchilla_tokens": chinchilla,
        "dataset_tokens": dataset_tokens,
        "tokens_trained": tokens_seen,
        "steps": total_steps,
        "train_seconds": elapsed,
        "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "optimizer_state_ratio": optimizer_state_bytes(optimizer) / (n_params * 4),
        "curve": curve,
        "final_val_loss": curve[-1]["val_loss"] if curve else None,
        "final_test_bleu": curve[-1]["test_bleu"] if curve else None,
    }
    save_json(summary, out_dir / "summary.json")
    logger.close()
    if wb is not None:
        wb.summary.update({k: v for k, v in summary.items() if k != "curve"})
        wb.finish()
    print(f"[part2] {opt_name} done in {elapsed/60:.1f} min; "
          f"final val loss {summary['final_val_loss']:.4f}")
    return summary


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Train Part 2 with one optimizer")
    ap.add_argument("--config", default="configs/part2.yaml")
    ap.add_argument("--optimizer", required=True, choices=sorted(OPTIMIZERS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    out_root = Path(args.out) if args.out else work_dir() / "part2" / "runs"
    run(cfg, args.optimizer, out_root)


if __name__ == "__main__":  # pragma: no cover
    main()
