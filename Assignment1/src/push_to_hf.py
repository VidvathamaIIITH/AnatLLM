"""Upload the trained checkpoints to the Hugging Face Hub.

    huggingface-cli login          # or: export HF_TOKEN=hf_xxx
    python -m src.push_to_hf --repo-id <your-username>/anlp-a1-transformers

Uploads, for every configuration found under ``outputs/``:
``best.pt``, ``config.json``, ``metrics.json``, ``history.json``, the tokenizer
JSONs and the plots -- plus an auto-generated model card at the repo root.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from .utils import load_json  # noqa: E402

UPLOAD_NAMES = (
    "best.pt", "config.json", "metrics.json", "history.json",
    "tokenizer_src.json", "tokenizer_tgt.json", "predictions_test.json",
    "entropy_model.json",   # C5: the n-gram model that defines patch boundaries
)


def build_model_card(repo_id: str, results: Dict[str, dict]) -> str:
    rows = []
    for name in sorted(results):
        r = results[name]
        rows.append(
            f"| {name} | {r.get('changed','')} | {r.get('bit_accuracy', float('nan')):.2f} | "
            f"{r.get('sequence_accuracy', float('nan')):.2f} | "
            f"{r.get('levenshtein', float('nan')):.2f} | "
            f"{r.get('params_millions', float('nan')):.2f} | "
            f"{r.get('peak_gpu_mem_mb', float('nan')):.0f} |"
        )
    table = "\n".join(rows) if rows else "| - | - | - | - | - | - | - |"
    return f"""---
license: mit
tags:
  - transformer
  - from-scratch
  - byte-latent-transformer
  - ablation
  - seq2seq
---

# ANLP Assignment 1 — Custom Transformers & Byte Latent Transformers

Encoder–decoder Transformers built from scratch in PyTorch (no `nn.Transformer`,
no `nn.MultiheadAttention`) for the task of mapping **encrypted binary sequences
to plaintext**, plus a controlled five-way architectural ablation.

| Config | Change from base | Bit Acc (%) | Seq Acc (%) | Levenshtein | Params (M) | Peak GPU MB |
|---|---|---|---|---|---|---|
{table}

## Configurations

| Config | Positional | Attention | Normalization | Tokenization |
|---|---|---|---|---|
| C1 | Sinusoidal absolute | Multi-Head | LayerNorm | Subword (byte-level BPE) |
| C2 | **RoPE** | Multi-Head | LayerNorm | Subword (byte-level BPE) |
| C3 | Sinusoidal absolute | **Grouped-Query** | LayerNorm | Subword (byte-level BPE) |
| C4 | Sinusoidal absolute | Multi-Head | **RMSNorm** | Subword (byte-level BPE) |
| C5 | Sinusoidal absolute | Multi-Head | LayerNorm | **BLT (token-free, entropy-patched bytes)** |

## How C5 (BLT) works

* **Bytes, not the text of bytes.** The ciphertext is a binary sequence stored as
  ASCII `0`/`1` characters. Every **8 bits are packed into one byte value 0-255**,
  so a 256-character cipher segment is **32 bytes**, not 256.
* **Entropy-based dynamic patching.** Patches are variable length: a new patch
  opens where `H(x_t | x_{{t-2}}, x_{{t-1}}) > theta`. The entropy comes from a
  lightweight order-2 byte n-gram model fit on the training split only
  (`entropy_model.json`, also stored as buffers inside `best.pt`); no separate
  neural LM is trained. `theta` is calibrated to a target mean patch length,
  giving ~10 variable-length patches per 34-byte sequence.
* A standard learned 256-entry byte embedding represents the byte values.

## Repository layout

```
C1/best.pt            checkpoint (state_dict + args + config)
C1/config.json        exact CLI arguments used
C1/metrics.json       test metrics, timings, peak memory
C1/tokenizer_*.json   the from-scratch byte-level BPE vocab + merges
...                   idem for C2..C5
C5/entropy_model.json the n-gram entropy model defining patch boundaries
```

## Loading a checkpoint

```python
import torch
from src.configs import get_config, build_model
import argparse

ck   = torch.load("C1/best.pt", map_location="cpu", weights_only=False)
cfg  = get_config(ck["config"]["name"])
args = argparse.Namespace(**ck["args"])
model = build_model(cfg, args, ck["src_vocab_size"], ck["tgt_vocab_size"])
model.load_state_dict(ck["model_state"])
model.eval()
```

Training/evaluation code: see the accompanying assignment submission
(`src/train.py`, `src/run_all.py`).
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Push checkpoints to the Hugging Face Hub")
    ap.add_argument("--repo-id", required=True, help="e.g. username/anlp-a1-transformers")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--configs", default="C1,C2,C3,C4,C5")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    names = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    results: Dict[str, dict] = {}
    to_upload: List[tuple] = []
    for name in names:
        run_dir = os.path.join(args.out_dir, name)
        if not os.path.isdir(run_dir):
            print(f"[hf] {name}: no directory {run_dir}; skipping")
            continue
        m = os.path.join(run_dir, "metrics.json")
        if os.path.exists(m):
            results[name] = load_json(m)
        for fname in UPLOAD_NAMES:
            path = os.path.join(run_dir, fname)
            if os.path.exists(path):
                to_upload.append((path, f"{name}/{fname}"))
        for extra in os.listdir(run_dir):
            if extra.endswith(".png"):
                to_upload.append((os.path.join(run_dir, extra), f"{name}/{extra}"))

    for extra in ("all_results.json", "report_tables.md",
                  "ablation_quality.png", "ablation_efficiency.png"):
        path = os.path.join(args.out_dir, extra)
        if os.path.exists(path):
            to_upload.append((path, extra))

    card = build_model_card(args.repo_id, results)
    card_path = os.path.join(args.out_dir, "README_hf.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    to_upload.append((card_path, "README.md"))

    print(f"[hf] {len(to_upload)} files queued for {args.repo_id}")
    for _, dst in to_upload:
        print(f"      {dst}")
    if args.dry_run:
        print("[hf] --dry-run: nothing uploaded")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hf] huggingface_hub is not installed:  pip install huggingface_hub")
        return 1

    api = HfApi(token=args.token or None)
    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    for src, dst in to_upload:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id=args.repo_id, repo_type="model")
        print(f"[hf] uploaded {dst}")
    print(f"\n[hf] done -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
