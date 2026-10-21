"""Evaluate trained Part 1 checkpoints: test perplexity, BLEU, expert usage.

Run per variant, or over every variant found under the runs directory:

    python -m src.part1.run_eval --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..part3.decoding import GenerationConfig
from ..tokenization import TGT_EN
from ..utils import (
    autocast_dtype,
    configure_stdout,
    deep_update,
    device_report,
    get_device,
    human,
    load_json,
    load_yaml,
    parse_overrides,
    save_json,
    set_seed,
    work_dir,
)
from .data import LANG_NAME, binarise_split, build_tokenizer
from .evaluate import bleu_by_language, evaluate_perplexity, expert_usage, translate
from .model import VARIANT_LABELS, VARIANTS, TransformerLM


def evaluate_variant(variant: str, cfg: dict, runs_dir: Path, ckpt_name: str = "best.pt") -> dict:
    device = get_device(cfg.get("device"))
    amp = autocast_dtype(device)
    seed = int(cfg["seed"])
    set_seed(seed)

    ckpt = runs_dir / variant / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    model, extra = TransformerLM.load(str(ckpt), map_location="cpu")
    model.to(device).eval()

    dcfg, ecfg = cfg["data"], cfg["eval"]
    tok = build_tokenizer(vocab_size=dcfg["vocab_size"])
    test = binarise_split("test", tok, dcfg["n_ctx"])

    print(f"\n[part1] === {variant}: {VARIANT_LABELS[variant]} ===")
    print(f"[part1] params total={human(model.num_parameters())} "
          f"active={human(model.num_active_parameters())}")

    # ---- perplexity ------------------------------------------------------- #
    ppl = evaluate_perplexity(
        model, test, tok.pad_id, device,
        max_tokens=int(cfg["train"].get("eval_micro_tokens", 2048)),
        max_examples=ecfg.get("ppl_examples"), amp_dtype=amp, seed=seed,
    )
    print(f"[part1] test ppl(all)={ppl.ppl_all:.2f}  ppl(target)={ppl.ppl_target:.2f}  "
          f"[vi-en {ppl.per_lang['vi-en']['ppl_target']:.2f} | "
          f"ja-en {ppl.per_lang['ja-en']['ppl_target']:.2f}]")

    # ---- BLEU ------------------------------------------------------------- #
    n_bleu = int(ecfg["bleu_examples"])
    rng = np.random.default_rng(seed)
    # Stratify so both directions are equally represented.
    picks = []
    for lang in (0, 1):
        pool = np.flatnonzero(test.lang == lang)
        picks.append(rng.choice(pool, size=min(n_bleu // 2, len(pool)), replace=False))
    bleu_idx = np.sort(np.concatenate(picks))

    gcfg = GenerationConfig(
        strategy="beam", beam_width=int(ecfg.get("beam_width", 4)),
        max_new_tokens=int(ecfg.get("max_new_tokens", 128)),
        eos_id=tok.eos_id, pad_id=tok.pad_id, length_penalty=1.0,
    )
    hyps, refs, langs, gen_s = translate(
        model, test, tok, bleu_idx, device, gcfg,
        batch_size=int(ecfg.get("gen_batch_size", 48)), amp_dtype=amp,
    )
    bleu = bleu_by_language(hyps, refs, langs)
    print(f"[part1] BLEU(all)={bleu['all']['bleu']:.2f}  "
          f"vi-en={bleu['vi-en']['bleu']:.2f}  ja-en={bleu['ja-en']['bleu']:.2f}  "
          f"chrF2={bleu['all']['chrf2']:.2f}  ({gen_s:.0f}s for {len(hyps)} beams)")

    # ---- greedy BLEU for reference ---------------------------------------- #
    gcfg_greedy = GenerationConfig(
        strategy="greedy", max_new_tokens=gcfg.max_new_tokens,
        eos_id=tok.eos_id, pad_id=tok.pad_id,
    )
    g_hyps, g_refs, g_langs, g_s = translate(
        model, test, tok, bleu_idx, device, gcfg_greedy,
        batch_size=int(ecfg.get("gen_batch_size", 48)), amp_dtype=amp,
    )
    bleu_greedy = bleu_by_language(g_hyps, g_refs, g_langs)
    print(f"[part1] BLEU greedy(all)={bleu_greedy['all']['bleu']:.2f}  ({g_s:.0f}s)")

    # ---- expert usage ----------------------------------------------------- #
    usage = expert_usage(
        model, test, tok.pad_id, device,
        max_examples=int(ecfg.get("usage_examples", 8000)),
        max_tokens=int(cfg["train"].get("eval_micro_tokens", 2048)),
        amp_dtype=amp, seed=seed,
    )
    if usage:
        ent = np.asarray(usage["entropy_bits"])
        print(f"[part1] routing entropy per layer (max {usage['max_entropy_bits']:.2f} bits): "
              + " ".join(f"{e:.2f}" for e in ent))

    # ---- qualitative samples ---------------------------------------------- #
    samples = []
    for j in list(range(0, len(hyps), max(1, len(hyps) // 12)))[:12]:
        ex = int(bleu_idx[j])
        seq = test.example(ex)
        src = tok.decode(seq[2:int(test.prompt_len[ex]) - 1].tolist())
        samples.append({
            "direction": LANG_NAME[langs[j]], "source": src,
            "reference": refs[j], "beam": hyps[j], "greedy": g_hyps[j],
        })

    result = {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "checkpoint": str(ckpt),
        "checkpoint_extra": {k: v for k, v in extra.items() if k != "val"},
        "params_total": model.num_parameters(),
        "params_active": model.num_active_parameters(),
        "param_budget": model.config.param_budget(),
        "perplexity": ppl.as_dict(),
        "bleu_beam": bleu,
        "bleu_greedy": bleu_greedy,
        "beam_width": gcfg.beam_width,
        "gen_seconds_beam": gen_s,
        "gen_seconds_greedy": g_s,
        "n_bleu_examples": len(hyps),
        "expert_usage": usage,
        "samples": samples,
    }
    save_json(result, runs_dir / variant / "eval.json")

    # Full hypothesis dump, useful for the qualitative analysis in the report.
    save_json(
        [{"direction": LANG_NAME[l], "reference": r, "beam": h, "greedy": g}
         for l, r, h, g in zip(langs, refs, hyps, g_hyps)],
        runs_dir / variant / "generations.json",
    )
    del model
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Evaluate Part 1 checkpoints")
    ap.add_argument("--config", default="configs/part1.yaml")
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    runs_dir = Path(args.runs) if args.runs else work_dir() / "part1" / "runs"
    print(f"[part1] device: {device_report()}")

    targets = list(VARIANTS) if args.all else [args.variant]
    if not targets or targets == [None]:
        ap.error("pass --variant NAME or --all")

    results = {}
    for v in targets:
        if not (runs_dir / v / args.ckpt).exists():
            print(f"[part1] skipping {v}: no {args.ckpt}")
            continue
        results[v] = evaluate_variant(v, cfg, runs_dir, args.ckpt)

    if not results:
        raise SystemExit(f"[part1] no {args.ckpt} found under {runs_dir} - train first, "
                         f"or run `python main.py fetch` to download the released checkpoints")

    if results:
        # Merge, so evaluating one variant keeps the others' earlier results.
        merged_path = runs_dir / "eval_all.json"
        merged = load_json(merged_path) if merged_path.exists() else {}
        merged.update(results)
        save_json(merged, merged_path)
        print(f"\n[part1] wrote {runs_dir / 'eval_all.json'}")
        header = f"{'variant':22s} {'total':>10s} {'active':>10s} {'ppl(tgt)':>10s} {'BLEU':>7s} {'chrF2':>7s}"
        print(header)
        print("-" * len(header))
        for v, r in results.items():
            print(f"{v:22s} {human(r['params_total']):>10s} {human(r['params_active']):>10s} "
                  f"{r['perplexity']['ppl_target']:10.2f} {r['bleu_beam']['all']['bleu']:7.2f} "
                  f"{r['bleu_beam']['all']['chrf2']:7.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
