"""Controlled per-step cost measurement for the Part 2 optimizers.

The wall-clock time of a full training run is not a usable measure of optimizer
overhead: the runs are hours long and whatever else the machine happens to be
doing leaks straight into the number.  In our first pass that artefact was large
enough to rank Sophia as the *fastest* optimizer despite it taking an extra
forward/backward pass every ``hessian_every`` steps, which cannot be true.

This benchmark instead runs every optimizer back to back, in one process, on the
same batches, interleaving repeated short measurements so that any drift in
machine state is spread across all of them rather than landing on whichever
optimizer happened to run first.  Reported figures are the *best* of several
rounds, which is the standard way to estimate a floor on cost under noise.

    python -m src.part2.benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..part1.model import TransformerConfig, TransformerLM, variant_config
from ..utils import (
    autocast_dtype,
    configure_stdout,
    device_report,
    get_device,
    load_yaml,
    save_json,
    set_seed,
    work_dir,
)
from .data import binarise_split, build_tokenizer, epoch_order, make_batch
from .optimizers import (
    OPTIMIZER_LABEL,
    build_optimizer,
    needs_hessian,
    optimizer_state_bytes,
)
from .train import hyperparams_for

ORDER = ["adamw", "mars", "lion", "muon", "sophia"]


def measure(name: str, cfg: dict, split, order, device, amp, steps: int, warmup: int) -> dict:
    """Time ``steps`` optimizer steps after ``warmup`` untimed ones."""
    set_seed(int(cfg["seed"]))
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    base = TransformerConfig(vocab_size=int(dcfg["vocab_size"]), n_ctx=int(dcfg["n_ctx"]), **mcfg)
    model = TransformerLM(variant_config("dense", base)).to(device)
    model.train()

    hp = hyperparams_for(cfg, name)
    opt = build_optimizer(name, model, hp)
    micro = int(tcfg["micro_batch_blocks"])
    accum = int(tcfg["grad_accum"])
    use_hess = needs_hessian(name)
    hess_every = int(hp.get("hessian_every", 10))
    hess_blocks = min(int(hp.get("hessian_blocks", 4)), micro)
    hess_bs = hess_blocks * int(dcfg["n_ctx"])

    tokens = 0
    t0 = None
    for step in range(warmup + steps):
        opt.zero_grad(set_to_none=True)
        last = None
        for m in range(accum):
            s = ((step * accum + m) * micro) % max(1, len(order) - micro)
            batch = make_batch(split, order[s:s + micro], device)
            last = batch
            with torch.autocast("cuda", dtype=amp) if amp else torch.autocast("cpu", enabled=False):
                out = model(batch["input_ids"], targets=batch["targets"])
            (out.loss / accum).backward()
            if step >= warmup:
                tokens += int(batch["n_tokens"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(bs=hess_bs) if use_hess else opt.step()

        if use_hess and (step + 1) % hess_every == 0:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp) if amp else torch.autocast("cpu", enabled=False):
                logits = model(last["input_ids"][:hess_blocks]).logits
            flat = logits.float().view(-1, logits.size(-1))
            with torch.no_grad():
                sampled = torch.multinomial(F.softmax(flat, dim=-1), 1).squeeze(-1)
            F.cross_entropy(flat, sampled).backward()
            opt.update_hessian()
            opt.zero_grad(set_to_none=True)
            del logits, flat, sampled

        if step == warmup - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    state = optimizer_state_bytes(opt)
    peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"seconds": dt, "tokens": tokens, "tokens_per_s": tokens / dt,
            "ms_per_step": dt / steps * 1000, "state_bytes": state, "peak_mib": peak}


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Controlled optimizer overhead benchmark")
    ap.add_argument("--config", default="configs/part2.yaml")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    device = get_device(cfg.get("device"))
    amp = autocast_dtype(device)
    print(f"[bench] device: {device_report()}")
    print(f"[bench] {args.rounds} rounds x {args.steps} timed steps "
          f"({args.warmup} warmup) per optimizer, interleaved\n")

    tok = build_tokenizer(vocab_size=int(cfg["data"]["vocab_size"]))
    split = binarise_split("train", tok, int(cfg["data"]["n_ctx"]))
    order = epoch_order(split, int(cfg["seed"]))

    # Adopt the swept learning rates so each optimizer is benchmarked in the
    # configuration it was actually trained in.
    sweep_path = work_dir() / "part2" / "sweep" / "sweep_results.json"
    if sweep_path.exists():
        import json

        for n, r in json.loads(sweep_path.read_text(encoding="utf-8")).items():
            cfg["optim"]["per_optimizer"].setdefault(n, {})["lr"] = r["best_lr"]

    rounds: dict[str, list[dict]] = {n: [] for n in ORDER}
    for r in range(args.rounds):
        for name in ORDER:                       # interleaved, so drift is shared
            res = measure(name, cfg, split, order, device, amp, args.steps, args.warmup)
            rounds[name].append(res)
            print(f"[bench] round {r+1} {name:8s} {res['tokens_per_s']:9,.0f} tok/s "
                  f"{res['ms_per_step']:7.1f} ms/step", flush=True)
        print()

    best = {n: max(v, key=lambda d: d["tokens_per_s"]) for n, v in rounds.items()}
    baseline = best["adamw"]["ms_per_step"]
    out_rows = []
    for n in ORDER:
        b = best[n]
        out_rows.append({
            "optimizer": n, "label": OPTIMIZER_LABEL[n],
            "tokens_per_s": b["tokens_per_s"], "ms_per_step": b["ms_per_step"],
            "relative_to_adamw": b["ms_per_step"] / baseline,
            "state_bytes": b["state_bytes"], "peak_mib": b["peak_mib"],
            "all_rounds_tokens_per_s": [x["tokens_per_s"] for x in rounds[n]],
        })

    payload = {"device": device_report(), "steps": args.steps, "warmup": args.warmup,
               "rounds": args.rounds, "rows": out_rows}
    out = Path(args.out) if args.out else work_dir() / "part2" / "benchmark.json"
    save_json(payload, out)

    print(f"{'optimizer':10s} {'tok/s':>10s} {'ms/step':>9s} {'vs AdamW':>9s} "
          f"{'state':>8s} {'peak MiB':>9s}")
    print("-" * 60)
    for r in out_rows:
        print(f"{r['optimizer']:10s} {r['tokens_per_s']:10,.0f} {r['ms_per_step']:9.1f} "
              f"{r['relative_to_adamw']:8.2f}x {r['state_bytes']/1024**2:7.0f}M "
              f"{r['peak_mib']:9.0f}")
    print(f"\n[bench] wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
