---
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
| C1 | none (base) | 97.17 | 77.20 | 0.44 | 7.62 | 372 |
| C2 | positional encoding -> RoPE | 97.61 | 81.20 | 0.35 | 7.62 | 372 |
| C3 | attention -> Grouped-Query | 96.81 | 73.80 | 0.54 | 6.73 | 360 |
| C4 | normalization -> RMSNorm | 97.33 | 78.80 | 0.42 | 7.61 | 350 |
| C5 | tokenization -> BLT (token-free) | 97.02 | 70.00 | 1.00 | 6.71 | 523 |

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
  opens where `H(x_t | x_{t-2}, x_{t-1}) > theta`. The entropy comes from a
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
