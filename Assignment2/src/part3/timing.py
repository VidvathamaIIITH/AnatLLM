"""Beam-search cost as a function of beam width.

Expected complexity for a batch of ``B`` prompts, beam width ``K``, ``L``
generated steps, vocabulary ``V`` and a model whose single-sequence forward
costs ``C``:

* the decoder is run on ``B * K`` rows at every step   ->  ``O(B * K * L * C)``
* each step ranks ``B x (K * V)`` candidates with top-k ->  ``O(B * K * V * L)``
* the KV cache holds ``B * K`` sequences               ->  ``O(B * K * L)`` memory

So the whole thing is **linear in the beam width**, with a constant that is
below 1 in practice because widening the beam fills the GPU better than a
narrow one does - a batch of ``B`` rows may leave the device idle while
``B * K`` rows saturate it.  This script measures that constant.

To keep the comparison honest the *batch size is held fixed* across widths, so
the only thing that changes is how many rows the model sees per step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..utils import (
    configure_stdout,
    deep_update,
    device_report,
    get_device,
    load_yaml,
    parse_overrides,
    save_json,
    set_seed,
    work_dir,
)
from .data import load_examples
from .decoding import GenerationConfig, HFBackend, generate
from .run import encode_prompts, load_model


def time_beam_widths(cfg: dict, out_dir: Path) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(cfg.get("device"))
    dtype = torch.float16 if (cfg.get("fp16", True) and device.type == "cuda") else torch.float32
    model, tok = load_model(device, dtype)

    tcfg = cfg["timing"]
    n = int(tcfg.get("n_examples", 200))
    repeats = int(tcfg.get("repeats", 3))
    widths = [int(w) for w in tcfg.get("beam_widths", [1, 2, 4])]
    batch_size = int(tcfg.get("batch_size", cfg["generate"].get("batch_size", 32)))
    max_new = int(cfg["generate"]["max_new_tokens"])

    examples = load_examples(split=cfg["data"]["split"], n=n, seed=seed)
    print(f"[part3/timing] {len(examples)} prompts, batch {batch_size}, "
          f"max_new_tokens {max_new}, {repeats} repeats per width")

    rows = []
    for w in widths:
        gcfg = GenerationConfig(strategy="beam", beam_width=w, max_new_tokens=max_new,
                                eos_id=tok.eos_token_id, pad_id=tok.pad_token_id)
        times, peaks, calls = [], [], 0
        for rep in range(repeats):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            calls = 0
            for s in range(0, len(examples), batch_size):
                chunk = examples[s:s + batch_size]
                ids, mask = encode_prompts(tok, chunk, device)
                res = generate(HFBackend(model), ids, mask, gcfg)
                calls += res.forward_calls
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            peaks.append(torch.cuda.max_memory_allocated() / 1024**2
                         if device.type == "cuda" else 0.0)
        best = min(times)
        rows.append({
            "beam_width": w, "seconds_best": best, "seconds_mean": sum(times) / len(times),
            "seconds_all": times, "peak_mib": max(peaks) if peaks else 0.0,
            "forward_calls": calls, "n_examples": len(examples),
            "seconds_per_example": best / len(examples),
        })
        print(f"[part3/timing] width {w:2d}: {best:7.2f}s best "
              f"({best/len(examples)*1000:6.1f} ms/example), peak {rows[-1]['peak_mib']:.0f} MiB",
              flush=True)

    base = rows[0]["seconds_best"]
    for r in rows:
        r["speed_ratio_vs_width1"] = r["seconds_best"] / base
        r["ratio_over_linear"] = (r["seconds_best"] / base) / r["beam_width"]

    payload = {"device": device_report(), "batch_size": batch_size,
               "max_new_tokens": max_new, "rows": rows}
    save_json(payload, out_dir / "beam_timing.json")

    print(f"\n{'width':>6s} {'seconds':>9s} {'ms/ex':>8s} {'x width-1':>10s} {'/linear':>9s}")
    for r in rows:
        print(f"{r['beam_width']:6d} {r['seconds_best']:9.2f} "
              f"{r['seconds_per_example']*1000:8.1f} {r['speed_ratio_vs_width1']:10.2f} "
              f"{r['ratio_over_linear']:9.2f}")
    return payload


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Beam-search timing study")
    ap.add_argument("--config", default="configs/part3.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    out_dir = Path(args.out) if args.out else work_dir() / "part3"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[part3/timing] device: {device_report()}")
    time_beam_widths(cfg, out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
