"""Evaluation: teacher-forced loss, greedy decoding and the Section-4 metrics.

Importable from ``train.py`` and runnable on its own::

    python -m src.evaluate --ckpt outputs/C1/best.pt --data-dir data
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):  # allow `python src/evaluate.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from .dataset import Seq2SeqDataset, discover_dataset, make_collate_fn  # noqa: E402
from .tokenizer import BPETokenizer, ByteTokenizer  # noqa: E402
from .utils import compute_metrics, save_json  # noqa: E402

__all__ = [
    "evaluate_loss",
    "ids_to_texts",
    "generate_predictions",
    "evaluate_generation",
]


def _autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
    amp_dtype: Optional[torch.dtype] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Teacher-forced cross-entropy + token accuracy (no label smoothing)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        src_pad = batch["src_pad_mask"].to(device, non_blocking=True)
        tgt_pad = batch["tgt_pad_mask"].to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            logits = model(src, tgt_in, src_pad, tgt_pad)
            loss = criterion(logits.float().reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        mask = tgt_out.ne(pad_id)
        n_tok = int(mask.sum())
        total_loss += float(loss)
        total_tokens += n_tok
        correct += int((logits.argmax(-1).eq(tgt_out) & mask).sum())
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "token_acc": 0.0}
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "ppl": float(torch.exp(torch.tensor(min(mean_loss, 20.0)))),
        "token_acc": 100.0 * correct / total_tokens,
    }


def ids_to_texts(ids: torch.Tensor, tokenizer, bos_id: int, eos_id: int, pad_id: int) -> List[str]:
    """Strip BOS, cut at the first EOS, drop PAD, then detokenise."""
    out: List[str] = []
    for row in ids.tolist():
        seq: List[int] = []
        for k, tok in enumerate(row):
            if k == 0 and tok == bos_id:
                continue
            if tok == eos_id:
                break
            if tok == pad_id:
                continue
            seq.append(tok)
        out.append(tokenizer.decode(seq))
    return out


@torch.no_grad()
def generate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tgt_tokenizer,
    max_new_tokens: int,
    amp_dtype: Optional[torch.dtype] = None,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    progress: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """Greedy-decode the whole loader.  Returns ``(preds, refs, sources)``."""
    model.eval()
    pad_id = int(getattr(model, "pad_id", 0))
    bos_id = int(getattr(model, "bos_id", 1))
    eos_id = int(getattr(model, "eos_id", 2))

    preds: List[str] = []
    refs: List[str] = []
    srcs: List[str] = []
    seen = 0
    n_batches = len(loader)
    for bi, batch in enumerate(loader):
        if max_samples is not None and seen >= max_samples:
            break
        src = batch["src"].to(device, non_blocking=True)
        src_pad = batch["src_pad_mask"].to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            gen = model.greedy_decode(
                src, src_pad, max_new_tokens=max_new_tokens, use_cache=use_cache
            )
        preds.extend(ids_to_texts(gen, tgt_tokenizer, bos_id, eos_id, pad_id))
        refs.extend(batch.get("tgt_text", []))
        srcs.extend(batch.get("src_text", []))
        seen += src.size(0)
        if progress and (bi % 10 == 0 or bi == n_batches - 1):
            print(f"    [gen] batch {bi + 1}/{n_batches} ({seen} samples)", flush=True)

    if max_samples is not None:
        preds, refs, srcs = preds[:max_samples], refs[:max_samples], srcs[:max_samples]
    return preds, refs, srcs


def evaluate_generation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tgt_tokenizer,
    max_new_tokens: int,
    amp_dtype: Optional[torch.dtype] = None,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    dump_path: Optional[str] = None,
    n_dump: int = 200,
) -> Dict[str, float]:
    """Greedy decoding + all Section-4 metrics, optionally dumping predictions."""
    preds, refs, srcs = generate_predictions(
        model, loader, device, tgt_tokenizer, max_new_tokens,
        amp_dtype=amp_dtype, max_samples=max_samples, use_cache=use_cache,
    )
    metrics = compute_metrics(preds, refs)
    metrics["n_eval_samples"] = float(len(preds))

    if dump_path:
        save_json(
            [
                {"source": s, "reference": r, "prediction": p, "exact": bool(p == r)}
                for s, r, p in zip(srcs[:n_dump], refs[:n_dump], preds[:n_dump])
            ],
            dump_path,
        )
    return metrics


# --------------------------------------------------------------------------- #
# standalone CLI
# --------------------------------------------------------------------------- #
def _load_checkpoint(ckpt_path: str, device: torch.device):
    from .configs import build_model, get_config as _get

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _get(ckpt["config"]["name"])
    args = argparse.Namespace(**ckpt["args"])
    model = build_model(cfg, args, ckpt["src_vocab_size"], ckpt["tgt_vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, cfg, args, ckpt


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    args_cli = ap.parse_args(argv)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args_cli.device == "auto"
        else args_cli.device
    )
    model, cfg, targs, ckpt = _load_checkpoint(args_cli.ckpt, device)
    ckpt_dir = os.path.dirname(os.path.abspath(args_cli.ckpt))

    if cfg.is_blt:
        src_tok = tgt_tok = ByteTokenizer()
    else:
        src_tok = BPETokenizer.load(os.path.join(ckpt_dir, "tokenizer_src.json"))
        tgt_tok = BPETokenizer.load(os.path.join(ckpt_dir, "tokenizer_tgt.json"))

    bundle = discover_dataset(
        args_cli.data_dir,
        src_file=getattr(targs, "src_file", "") or None,
        tgt_file=getattr(targs, "tgt_file", "") or None,
        src_col=getattr(targs, "src_col", None) or None,
        tgt_col=getattr(targs, "tgt_col", None) or None,
        swap=getattr(targs, "swap", False),
        seed=getattr(targs, "seed", 42),
        chunk_target_chars=getattr(targs, "chunk_chars", 0),
    )
    pairs = {"train": bundle.train, "val": bundle.val, "test": bundle.test}[args_cli.split]
    ds = Seq2SeqDataset(pairs, src_tok, tgt_tok, targs.max_src_len, targs.max_tgt_len)
    loader = DataLoader(
        ds, batch_size=args_cli.batch_size, shuffle=False, collate_fn=make_collate_fn(0)
    )

    metrics = evaluate_generation(
        model, loader, device, tgt_tok, targs.max_tgt_len,
        max_samples=args_cli.max_samples or None,
        use_cache=not cfg.is_blt,
        dump_path=os.path.join(ckpt_dir, f"predictions_{args_cli.split}.json"),
    )
    print(f"\n=== {cfg.name} / {args_cli.split} ===")
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"  {k:>24}: {v:.4f}")
    if args_cli.out:
        save_json(metrics, args_cli.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
