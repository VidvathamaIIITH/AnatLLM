"""Main training loop for one ablation configuration, with WandB logging.

Examples
--------
Train the baseline::

    python -m src.train --config C1 --data-dir data --epochs 15

Fast pipeline check (tiny model, 200 samples, 1 epoch, no WandB)::

    python -m src.train --config C5 --data-dir data --smoke --wandb disabled

Every configuration shares the same CLI defaults, so C1-C5 differ only by the
single component named in ``configs.py`` -- which is what makes the study
controlled.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import statistics
import sys
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):  # allow `python src/train.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from .configs import CONFIGS, build_model, get_config  # noqa: E402
from .dataset import (  # noqa: E402
    Seq2SeqDataset,
    discover_dataset,
    length_percentile,
    make_collate_fn,
)
from .evaluate import evaluate_generation, evaluate_loss  # noqa: E402
from .tokenizer import BPETokenizer, ByteTokenizer  # noqa: E402
from .utils import (  # noqa: E402
    GpuMemory,
    Timer,
    WandbLogger,
    count_parameters,
    human_int,
    plot_training_curves,
    save_json,
    set_seed,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train one ablation configuration (C1-C5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- what to run -------------------------------------------------------
    p.add_argument("--config", default="C1", choices=sorted(CONFIGS))
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--run-name", default="")

    # --- data --------------------------------------------------------------
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-file", default="")
    p.add_argument("--val-file", default="")
    p.add_argument("--test-file", default="")
    p.add_argument("--src-file", default="",
                   help="parallel corpus: file of source lines (use with --tgt-file)")
    p.add_argument("--tgt-file", default="",
                   help="parallel corpus: line-aligned file of target lines")
    p.add_argument("--chunk-chars", type=int, default=0,
                   help="split long pairs into aligned segments of N target characters "
                        "(0 = off). For the assignment cipher use a multiple of 8.")
    p.add_argument("--src-col", default="", help="source column name (auto-detected if empty)")
    p.add_argument("--tgt-col", default="", help="target column name (auto-detected if empty)")
    p.add_argument("--swap", action="store_true", help="swap source/target direction")
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-eval-samples", type=int, default=1000,
                   help="cap on greedy-decoded test samples (0 = all)")

    # --- tokenisation ------------------------------------------------------
    p.add_argument("--vocab-size", type=int, default=4096, help="BPE vocab (configs C1-C4)")
    p.add_argument("--max-pretoken-len", type=int, default=32,
                   help="BPE pre-token chunk size for the TARGET side")
    p.add_argument("--src-max-pretoken-len", type=int, default=0,
                   help="BPE pre-token chunk size for the SOURCE side (0 = same as "
                        "--max-pretoken-len, i.e. 32). Do NOT set this to a value close "
                        "to the underlying alphabet's period (e.g. 8 for this 8-bit-"
                        "periodic cipher) -- BPE then degenerates into a fixed byte-"
                        "lookup table (every token exactly 8 chars long), which is the "
                        "fixed-width chunking scheme the assignment brief explicitly "
                        "disallows as a substitute for learned subword tokenization. "
                        "build_tokenizers() prints a piece-length report and warns loudly "
                        "if this happens; heed it.")
    p.add_argument("--max-src-len", type=int, default=0, help="0 = 99th percentile of train")
    p.add_argument("--max-tgt-len", type=int, default=0, help="0 = 99th percentile of train")
    p.add_argument("--len-percentile", type=float, default=0.99)
    p.add_argument("--rebuild-tokenizer", action="store_true")

    # --- model geometry (identical across configs) -------------------------
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3, help="encoder layers == decoder layers")
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=2, help="C3 only (GQA)")
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-tie-embeddings", action="store_true")

    # --- BLT-specific (C5) --------------------------------------------------
    p.add_argument("--blt-patch-size", type=int, default=4)
    p.add_argument("--blt-d-local", type=int, default=128)
    p.add_argument("--blt-local-enc-layers", type=int, default=1)
    p.add_argument("--blt-local-dec-layers", type=int, default=2)
    p.add_argument("--blt-local-heads", type=int, default=4)
    p.add_argument("--blt-window", type=int, default=64)
    # ---- entropy-based dynamic patching (BLT's actual mechanism) ----
    p.add_argument("--blt-target-patch", type=float, default=4.0,
                   help="calibrate the entropy threshold to this mean patch length")
    p.add_argument("--blt-max-patch", type=int, default=12,
                   help="hard cap on patch length (0 = uncapped)")
    p.add_argument("--blt-entropy-alpha", type=float, default=0.5,
                   help="add-alpha smoothing for the n-gram entropy model")
    p.add_argument("--blt-entropy-fit-samples", type=int, default=0,
                   help="training pairs used to fit the entropy model (0 = all)")
    p.add_argument("--blt-fixed-patches", action="store_true",
                   help="ablation: use fixed-width patches instead of entropy-based")

    # --- optimisation -------------------------------------------------------
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98])
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "inverse_sqrt", "constant"])
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=5, help="early stopping (0 = off)")

    # --- runtime ------------------------------------------------------------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="tiny fast run to validate the pipeline")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", default="", help="checkpoint to load before training/eval")

    # --- logging ------------------------------------------------------------
    p.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="anlp-a1-transformers")
    p.add_argument("--wandb-entity", default="")
    return p


def apply_smoke(args: argparse.Namespace) -> None:
    args.epochs = min(args.epochs, 1)
    args.d_model = 64
    args.n_layers = 2
    args.n_heads = 4
    args.n_kv_heads = 2
    args.d_ff = 128
    args.vocab_size = 512
    args.blt_d_local = 32
    args.blt_local_heads = 2
    args.batch_size = 8
    args.eval_batch_size = 8
    args.max_train_samples = min(args.max_train_samples or 200, 200)
    args.max_eval_samples = min(args.max_eval_samples or 32, 32)
    args.max_src_len = args.max_src_len or 64
    args.max_tgt_len = args.max_tgt_len or 64
    args.warmup_steps = 5
    args.patience = 0
    args.num_workers = 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_amp_dtype(spec: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda" or spec == "off":
        return None
    if spec == "bf16":
        return torch.bfloat16
    if spec == "fp16":
        return torch.float16
    # auto
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def lr_at(step: int, args: argparse.Namespace, total_steps: int) -> float:
    """Warmup + decay, evaluated at (1-based) optimiser step."""
    warmup = max(1, args.warmup_steps)
    if step <= warmup:
        return args.lr * step / warmup
    if args.scheduler == "constant":
        return args.lr
    if args.scheduler == "inverse_sqrt":
        return max(args.min_lr, args.lr * math.sqrt(warmup / step))
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def build_tokenizers(cfg, args, bundle, run_dir: str, verbose: bool = True):
    """Train (or reload) the source/target tokenizers for this configuration."""
    if cfg.is_blt:
        # The cipher side is a *binary sequence written as ASCII 0/1 characters*.
        # The underlying data is the binary sequence itself, so pack every 8 bits
        # into one byte (0-255) rather than feeding the model the text encoding
        # -- a 256-character segment is 32 bytes of data, not 256.
        sample = [s for s, _ in bundle.train[:200]]
        binary_src = bool(sample) and all(
            set(s) <= {"0", "1"} and len(s) % 8 == 0 for s in sample
        )
        src_tok = ByteTokenizer(bit_packed=binary_src)
        tgt_tok = ByteTokenizer(bit_packed=False)
        if verbose:
            kind = "bit-packed (8 bits -> 1 byte)" if binary_src else "utf-8"
            print(f"  [tok] token-free byte tokenizer | vocab={src_tok.vocab_size} | "
                  f"source={kind}, target=utf-8", flush=True)
            if binary_src:
                ex = sample[0]
                print(f"  [tok] source: {len(ex)} bits -> "
                      f"{len(src_tok.raw_bytes(ex))} bytes", flush=True)
        src_tok.save(os.path.join(run_dir, "tokenizer_src.json"))
        tgt_tok.save(os.path.join(run_dir, "tokenizer_tgt.json"))
        return src_tok, tgt_tok, src_tok.vocab_size, tgt_tok.vocab_size

    # A checkpoint is only meaningful together with the tokenizer that produced
    # its embedding table, and that tokenizer travels with it. Always prefer it:
    # a retrained BPE can differ (different corpus slice, different vocab size,
    # a newer trainer) and every learned embedding would silently point at the
    # wrong token.
    if args.ckpt:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        src_path = os.path.join(ckpt_dir, "tokenizer_src.json")
        tgt_path = os.path.join(ckpt_dir, "tokenizer_tgt.json")
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            src_tok = BPETokenizer.load(src_path)
            tgt_tok = BPETokenizer.load(tgt_path)
            if verbose:
                print(f"  [tok] using the tokenizer shipped with {args.ckpt} "
                      f"({src_tok.vocab_size}/{tgt_tok.vocab_size})", flush=True)
            if os.path.abspath(run_dir) != ckpt_dir:
                src_tok.save(os.path.join(run_dir, "tokenizer_src.json"))
                tgt_tok.save(os.path.join(run_dir, "tokenizer_tgt.json"))
            return src_tok, tgt_tok, src_tok.vocab_size, tgt_tok.vocab_size
        print(
            f"  [warn] --ckpt given but no tokenizer_*.json next to {args.ckpt}; "
            f"falling back to training/loading a tokenizer from the corpus. If the "
            f"checkpoint was made with a different tokenizer its embeddings will be "
            f"misaligned and the metrics will be meaningless.",
            flush=True,
        )

    # BPE tokenizers are shared by C1-C4: train once, cache under out-dir/tokenizers.
    # The cache key includes a fingerprint of the training corpus, so pointing
    # --data-dir at a different dataset retrains instead of silently reusing a
    # vocabulary learned from the old one.
    cache_dir = os.path.join(args.out_dir, "tokenizers")
    os.makedirs(cache_dir, exist_ok=True)
    src_mpl = args.src_max_pretoken_len or args.max_pretoken_len
    fingerprint = hashlib.sha1(
        repr((len(bundle.train), bundle.src_col, bundle.tgt_col,
              bundle.train[:3], bundle.train[-3:])).encode("utf-8")
    ).hexdigest()[:10]
    tag = (f"v{args.vocab_size}_p{src_mpl}-{args.max_pretoken_len}"
           f"_s{args.seed}_{fingerprint}")
    src_path = os.path.join(cache_dir, f"src_{tag}.json")
    tgt_path = os.path.join(cache_dir, f"tgt_{tag}.json")

    if os.path.exists(src_path) and os.path.exists(tgt_path) and not args.rebuild_tokenizer:
        src_tok = BPETokenizer.load(src_path)
        tgt_tok = BPETokenizer.load(tgt_path)
        if verbose:
            print(f"  [tok] loaded cached BPE ({src_tok.vocab_size}/{tgt_tok.vocab_size})",
                  flush=True)
    else:
        if verbose:
            print(f"  [tok] training byte-level BPE (vocab={args.vocab_size}, "
                  f"pretoken src={src_mpl} tgt={args.max_pretoken_len}) ...", flush=True)
        src_tok = BPETokenizer(max_pretoken_len=src_mpl)
        src_tok.train([s for s, _ in bundle.train], vocab_size=args.vocab_size, verbose=verbose)
        tgt_tok = BPETokenizer(max_pretoken_len=args.max_pretoken_len)
        tgt_tok.train([t for _, t in bundle.train], vocab_size=args.vocab_size, verbose=verbose)
        src_tok.save(src_path)
        tgt_tok.save(tgt_path)

    # a copy travels with the checkpoint so evaluate.py can stand alone
    src_tok.save(os.path.join(run_dir, "tokenizer_src.json"))
    tgt_tok.save(os.path.join(run_dir, "tokenizer_tgt.json"))

    if verbose:
        _report_piece_length_stats(src_tok, [s for s, _ in bundle.train], "source")
        _report_piece_length_stats(tgt_tok, [t for _, t in bundle.train], "target")

    return src_tok, tgt_tok, src_tok.vocab_size, tgt_tok.vocab_size


def _report_piece_length_stats(tok: BPETokenizer, texts, label: str) -> None:
    """Print piece-length stats and loudly warn if the tokenizer degenerated
    into fixed-width chunking (assignment brief: "8 bits = 1 token does not
    count as subword tokenization"). See BPETokenizer.piece_length_stats.
    """
    stats = tok.piece_length_stats(texts)
    kind = "variable-length (genuine learned subword)" if stats["n_distinct"] > 1 else "FIXED-WIDTH"
    print(
        f"  [tok] {label} piece length: mean={stats['mean']:.2f} chars, "
        f"range=[{stats['min']},{stats['max']}], {stats['n_distinct']} distinct "
        f"lengths -> {kind}",
        flush=True,
    )
    if stats["n_distinct"] <= 1:
        print(
            "  " + "!" * 74 + "\n"
            f"  WARNING: every {label} token this tokenizer emits has the SAME length "
            f"({stats['min']} chars).\n"
            "  That is fixed-width chunking, not learned subword tokenization -- the\n"
            "  assignment brief explicitly disallows this. Increase --max-pretoken-len\n"
            "  (or --src-max-pretoken-len for the source) and retrain with\n"
            "  --rebuild-tokenizer. A pretoken window close to the alphabet's period\n"
            "  (e.g. 8 for this 8-bit-periodic cipher) makes BPE degenerate into a\n"
            "  fixed byte-lookup table; use a substantially larger window instead.\n"
            + "  " + "!" * 74,
            flush=True,
        )


def resolve_lengths(args, bundle, src_tok, tgt_tok, verbose=True):
    """Choose max source/target lengths from a sample of the training data."""
    if args.max_src_len and args.max_tgt_len:
        return args.max_src_len, args.max_tgt_len
    sample = bundle.train[:5000]
    src_lens = [len(src_tok.encode(s)) + 2 for s, _ in sample]
    tgt_lens = [len(tgt_tok.encode(t)) + 2 for _, t in sample]
    q = args.len_percentile
    max_src = args.max_src_len or max(8, length_percentile(src_lens, q))
    max_tgt = args.max_tgt_len or max(8, length_percentile(tgt_lens, q))
    if verbose:
        print(
            f"  [len] p{int(q*100)} source={max_src} target={max_tgt} "
            f"(max seen: {max(src_lens)}/{max(tgt_lens)})",
            flush=True,
        )
    return int(max_src), int(max_tgt)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.smoke:
        apply_smoke(args)

    cfg = get_config(args.config)
    run_name = args.run_name or f"{cfg.name}-{cfg.changed.split(' ->')[0].replace(' ', '_')}"
    run_dir = os.path.join(args.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)
    amp_dtype = pick_amp_dtype(args.amp, device)
    mem = GpuMemory(device)

    print("=" * 78)
    print(f"Configuration {cfg.name}  ({cfg.changed})")
    print(f"  positional={cfg.pos_encoding}  attention={cfg.attention}  "
          f"norm={cfg.norm}  tokenization={cfg.tokenization}")
    print(f"  device={device}  amp={amp_dtype}  torch={torch.__version__}")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------------- data
    bundle = discover_dataset(
        args.data_dir,
        train_file=args.train_file or None,
        val_file=args.val_file or None,
        test_file=args.test_file or None,
        src_file=args.src_file or None,
        tgt_file=args.tgt_file or None,
        src_col=args.src_col or None,
        tgt_col=args.tgt_col or None,
        swap=args.swap,
        seed=args.seed,
        chunk_target_chars=args.chunk_chars,
    )
    # Loud advisory: this corpus is document-length unless it has been chunked.
    if not args.chunk_chars and bundle.train:
        sample = sorted(len(t) for _, t in bundle.train[:5000])
        p99_tgt = sample[int(0.99 * (len(sample) - 1))]
        if p99_tgt > 256:
            print(
                "\n" + "!" * 78 + "\n"
                f"  Targets are long (p99 = {p99_tgt} characters) and --chunk-chars is off.\n"
                "  Training on whole documents is very slow and C5 (BLT) will most likely\n"
                "  run out of memory: its local attention is quadratic in BYTES.\n"
                "  For the assignment cipher, add:\n"
                "      --chunk-chars 64\n"
                "  (64 is a multiple of the 8-character key period, so every segment starts\n"
                "   at the same key phase and the mapping is preserved exactly. This only\n"
                "   affects how documents are split into training examples -- it does NOT\n"
                "   define token boundaries; --max-pretoken-len 32, the default, still\n"
                "   drives genuine learned BPE merges within each 512-char segment.)\n"
                + "!" * 78 + "\n",
                flush=True,
            )

    if args.max_train_samples:
        bundle.train = bundle.train[: args.max_train_samples]

    src_tok, tgt_tok, src_vocab, tgt_vocab = build_tokenizers(cfg, args, bundle, run_dir)
    args.max_src_len, args.max_tgt_len = resolve_lengths(args, bundle, src_tok, tgt_tok)

    if cfg.is_blt and max(args.max_src_len, args.max_tgt_len) > 512:
        # BLT's local attention is dense-masked, so its score matrix is O(T^2) in
        # *bytes* -- the exact overhead the C1-vs-C5 comparison is meant to expose.
        print(
            f"  [warn] BLT byte length is {max(args.max_src_len, args.max_tgt_len)}; "
            f"local attention memory grows with its square.\n"
            f"         If this OOMs, lower --batch-size for ALL configs (to keep the "
            f"ablation controlled) or raise --blt-patch-size.",
            flush=True,
        )

    train_ds = Seq2SeqDataset(bundle.train, src_tok, tgt_tok, args.max_src_len, args.max_tgt_len)
    val_ds = Seq2SeqDataset(bundle.val, src_tok, tgt_tok, args.max_src_len, args.max_tgt_len)
    test_ds = Seq2SeqDataset(bundle.test, src_tok, tgt_tok, args.max_src_len, args.max_tgt_len)
    print(
        f"  [data] encoded train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
        f" | truncated src={train_ds.n_src_truncated} tgt={train_ds.n_tgt_truncated}",
        flush=True,
    )

    collate = make_collate_fn(pad_id=0)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=pin,
    )

    # --------------------------------------------------------------- model
    model = build_model(cfg, args, src_vocab, tgt_vocab).to(device)

    # ---- BLT: fit the entropy estimator that drives dynamic patching -------
    # Fitted on the TRAINING SPLIT ONLY -- the byte statistics of val/test never
    # touch it, so patch boundaries carry no leakage.
    entropy_info = {}
    if cfg.is_blt and not args.blt_fixed_patches:
        from .models.entropy import ByteNgramEntropyModel
        print(f"  [blt] fitting order-2 byte n-gram entropy models "
              f"(target mean patch = {args.blt_target_patch} bytes) ...", flush=True)
        fit_sample = bundle.train[: args.blt_entropy_fit_samples or len(bundle.train)]
        src_ent = ByteNgramEntropyModel(alpha=args.blt_entropy_alpha).fit(
            (src_tok.raw_bytes(s) for s, _ in fit_sample))
        tgt_ent = ByteNgramEntropyModel(alpha=args.blt_entropy_alpha).fit(
            (tgt_tok.raw_bytes(t) for _, t in fit_sample))
        src_ent.calibrate_threshold(
            [src_tok.raw_bytes(s) for s, _ in fit_sample[:3000]],
            target_patch_len=args.blt_target_patch)
        tgt_ent.calibrate_threshold(
            [tgt_tok.raw_bytes(t) for _, t in fit_sample[:3000]],
            target_patch_len=args.blt_target_patch)
        model.install_entropy(src_ent, tgt_ent, max_patch_len=args.blt_max_patch)
        model.to(device)
        entropy_info = {
            "src_threshold": src_ent.threshold, "tgt_threshold": tgt_ent.threshold,
            "src_H0": src_ent.H0, "tgt_H0": tgt_ent.H0,
            "target_patch_len": args.blt_target_patch,
            "max_patch_len": args.blt_max_patch,
        }
        save_json({"src": src_ent.state_dict(), "tgt": tgt_ent.state_dict()},
                  os.path.join(run_dir, "entropy_model.json"))
    elif cfg.is_blt:
        print("  [blt] --blt-fixed-patches: using FIXED-width patches "
              f"(P={args.blt_patch_size}) instead of entropy-based patching", flush=True)

    n_params = count_parameters(model)
    print(f"  [model] {type(model).__name__} | trainable params = {human_int(n_params)}",
          flush=True)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"  [model] loaded weights from {args.ckpt}", flush=True)

    # ------------------------------------------------------------- logging
    wandb_config = dict(vars(args))
    wandb_config.update(cfg.as_dict())
    wandb_config.update(
        {"n_params": n_params, "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
         "device": str(device), "amp": str(amp_dtype), "host": platform.node()}
    )
    logger = WandbLogger(
        mode=args.wandb, project=args.wandb_project,
        entity=args.wandb_entity or None, name=run_name,
        config=wandb_config, out_dir=run_dir,
        tags=[cfg.name, cfg.pos_encoding, cfg.attention, cfg.norm, cfg.tokenization],
    )

    # ---------------------------------------------------------- optimiser
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=tuple(args.betas), eps=1e-9,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.accum_steps)))
    total_steps = steps_per_epoch * max(1, args.epochs)

    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_ppl": [],
        "val_token_acc": [], "sec_per_epoch": [], "lr": [],
    }
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    global_step = 0
    epoch_times: List[float] = []
    total_train_tokens = 0

    mem.reset()

    # ------------------------------------------------------------- training
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            running_tokens = 0
            t_epoch = Timer(device)
            t_epoch.__enter__()
            optimizer.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                src = batch["src"].to(device, non_blocking=True)
                tgt_in = batch["tgt_in"].to(device, non_blocking=True)
                tgt_out = batch["tgt_out"].to(device, non_blocking=True)
                src_pad = batch["src_pad_mask"].to(device, non_blocking=True)
                tgt_pad = batch["tgt_pad_mask"].to(device, non_blocking=True)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if amp_dtype is not None
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with autocast_ctx:
                    logits = model(src, tgt_in, src_pad, tgt_pad)
                    loss = criterion(
                        logits.float().reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
                    )
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    print(f"  [warn] non-finite loss at step {global_step}; skipping batch",
                          flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss / args.accum_steps).backward()

                if (it + 1) % args.accum_steps == 0 or (it + 1) == len(train_loader):
                    global_step += 1
                    lr_now = lr_at(global_step, args, total_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_now
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if global_step % args.log_every == 0:
                        logger.log(
                            {"train/loss_step": loss_value, "train/lr": lr_now,
                             "train/epoch": epoch},
                            step=global_step,
                        )

                n_tok = int(tgt_out.ne(0).sum())
                running += loss_value * n_tok
                running_tokens += n_tok
                total_train_tokens += n_tok

            t_epoch.__exit__()
            epoch_times.append(t_epoch.elapsed)
            train_loss = running / max(1, running_tokens)

            val_stats = evaluate_loss(model, val_loader, device, pad_id=0, amp_dtype=amp_dtype)
            lr_now = lr_at(max(1, global_step), args, total_steps)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_stats["loss"])
            history["val_ppl"].append(val_stats["ppl"])
            history["val_token_acc"].append(val_stats["token_acc"])
            history["sec_per_epoch"].append(t_epoch.elapsed)
            history["lr"].append(lr_now)

            logger.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_stats["loss"],
                    "val/ppl": val_stats["ppl"],
                    "val/token_accuracy": val_stats["token_acc"],
                    "perf/sec_per_epoch": t_epoch.elapsed,
                    "perf/tokens_per_sec": running_tokens / max(1e-9, t_epoch.elapsed),
                    "perf/peak_gpu_mem_mb": mem.peak_mb(),
                },
                step=global_step,
            )
            print(
                f"  epoch {epoch:>3}/{args.epochs} | train {train_loss:.4f} | "
                f"val {val_stats['loss']:.4f} (ppl {val_stats['ppl']:.2f}, "
                f"tok-acc {val_stats['token_acc']:.2f}%) | {t_epoch.elapsed:.1f}s",
                flush=True,
            )

            improved = val_stats["loss"] < best_val - 1e-5
            if improved:
                best_val = val_stats["loss"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg.as_dict(),
                        "args": vars(args),
                        "src_vocab_size": src_vocab,
                        "tgt_vocab_size": tgt_vocab,
                        "epoch": epoch,
                        "val_loss": best_val,
                    },
                    os.path.join(run_dir, "best.pt"),
                )
            else:
                bad_epochs += 1
                if args.patience and bad_epochs >= args.patience:
                    print(f"  [early stop] no improvement for {bad_epochs} epochs", flush=True)
                    break

        if best_epoch > 0:
            state = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu",
                               weights_only=False)
            model.load_state_dict(state["model_state"])
            model.to(device)
            print(f"  [model] restored best checkpoint (epoch {best_epoch}, "
                  f"val loss {best_val:.4f})", flush=True)

    peak_mem = mem.peak_mb()
    peak_reserved = mem.peak_reserved_mb()

    # ----------------------------------------------------------- evaluation
    print("  [eval] teacher-forced test loss ...", flush=True)
    test_loss_stats = evaluate_loss(model, test_loader, device, pad_id=0, amp_dtype=amp_dtype)

    print("  [eval] greedy decoding ...", flush=True)
    gen_timer = Timer(device)
    with gen_timer:
        gen_metrics = evaluate_generation(
            model, test_loader, device, tgt_tok,
            max_new_tokens=args.max_tgt_len,
            amp_dtype=amp_dtype,
            max_samples=args.max_eval_samples or None,
            use_cache=not cfg.is_blt,
            dump_path=os.path.join(run_dir, "predictions_test.json"),
        )

    results = {
        "config": cfg.name,
        "changed": cfg.changed,
        "pos_encoding": cfg.pos_encoding,
        "attention": cfg.attention,
        "norm": cfg.norm,
        "tokenization": cfg.tokenization,
        "params": n_params,
        "params_millions": n_params / 1e6,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        "epochs_run": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "test_loss": test_loss_stats["loss"],
        "test_ppl": test_loss_stats["ppl"],
        "test_token_accuracy": test_loss_stats["token_acc"],
        # Epoch time is reported as the MEDIAN, not the mean. These are
        # wall-clock measurements on a laptop, so a single epoch can absorb an
        # OS suspend/hibernate (one C4 epoch recorded 24,890 s of which ~127 s
        # was compute). The median is unaffected by such an outlier, whereas the
        # mean is destroyed by it; with no interruption the two agree to ~2%.
        "sec_per_epoch": statistics.median(epoch_times) if epoch_times else 0.0,
        "sec_per_epoch_mean": (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
        "total_train_seconds": sum(epoch_times),
        "train_tokens_per_sec": (
            (total_train_tokens / len(epoch_times)) / statistics.median(epoch_times)
            if epoch_times and statistics.median(epoch_times) > 0 else 0.0
        ),
        "peak_gpu_mem_mb": peak_mem,
        "peak_gpu_mem_reserved_mb": peak_reserved,
        "generation_seconds": gen_timer.elapsed,
        "device": str(device),
        "amp": str(amp_dtype),
    }
    results.update({k: v for k, v in gen_metrics.items()})
    if cfg.is_blt:
        # BLEU/ROUGE are defined by the brief for tokenised models; kept for
        # reference on C5 and flagged so the report table can mark them.
        results["ngram_metrics_reference_only"] = True

    save_json(results, os.path.join(run_dir, "metrics.json"))
    save_json(history, os.path.join(run_dir, "history.json"))
    save_json({"config": cfg.as_dict(), "args": vars(args)}, os.path.join(run_dir, "config.json"))

    if history["epoch"]:
        plot_training_curves(
            history, os.path.join(run_dir, f"{cfg.name}_curves.png"),
            title=f"{cfg.name} — {cfg.changed}",
        )

    logger.summary({f"test/{k}": v for k, v in results.items() if isinstance(v, (int, float))})
    wandb_url = logger.url()
    if wandb_url:
        results["wandb_url"] = wandb_url
        save_json(results, os.path.join(run_dir, "metrics.json"))
    logger.finish()

    print("\n" + "-" * 78)
    print(f"RESULTS - {cfg.name} ({cfg.changed})")
    for key in [
        "bit_accuracy", "sequence_accuracy", "levenshtein", "levenshtein_normalized",
        "bleu", "rouge1", "rouge2", "rougeL", "test_loss", "test_ppl",
        "test_token_accuracy", "params_millions", "sec_per_epoch", "peak_gpu_mem_mb",
        "generation_seconds",
    ]:
        if key in results and results[key] is not None:
            print(f"  {key:>24}: {results[key]:.4f}")
    print(f"  artifacts -> {run_dir}")
    print("-" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
