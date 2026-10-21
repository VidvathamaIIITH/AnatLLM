# ANLP Assignment 1 — Transformers from Scratch, Architectural Variants & Byte Latent Transformers

Encoder–decoder Transformers built from **elementary PyTorch operations** — no
`nn.Transformer`, no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention` — applied
to decrypting XOR-enciphered binary sequences back into plaintext English, plus a controlled
five-way architectural ablation (C1–C5).

---

## Links

| Resource | URL |
|---|---|
| **Weights & Biases project** | <https://wandb.ai/vidvathama-babu-iiit-hyderabad/anlp-a1-blt-ablation> |
| **Hugging Face checkpoints** | <https://huggingface.co/Viv0605101/anlp-a1-transformer-blt-ablation> |
| **Kaggle mirror** | not used — `src/push_to_kaggle.py` is included and documented if you want one |

Per-config W&B run links are in [`outputs/wandb_runs.json`](outputs/wandb_runs.json);
all links are collected in [`outputs/links.json`](outputs/links.json).

---

## The five configurations

Each of C2–C5 changes **exactly one** component from the C1 baseline.

| Config | Change from base | Positional | Attention | Norm | Tokenization |
|---|---|---|---|---|---|
| **C1** | — (base) | Sinusoidal absolute | Multi-Head | LayerNorm | BPE subword |
| **C2** | Positional encoding | **RoPE** | Multi-Head | LayerNorm | BPE subword |
| **C3** | Attention | Sinusoidal absolute | **Grouped-Query** (8q/2kv) | LayerNorm | BPE subword |
| **C4** | Normalization | Sinusoidal absolute | Multi-Head | **RMSNorm** | BPE subword |
| **C5** | Tokenization | Sinusoidal absolute | Multi-Head | LayerNorm | **BLT (token-free)** |

All five share depth, width, heads, `d_ff`, dropout, optimizer, LR schedule, batch size,
epochs and seed — so any difference is attributable to the single component that moved.

---

## Directory structure

```
<rollnumber>_assignment1/
├── src/
│   ├── models/
│   │   ├── attention.py      # SDPA, MHA, GQA  (from scratch)
│   │   ├── positional.py     # Sinusoidal absolute + RoPE
│   │   ├── norm.py           # LayerNorm + RMSNorm (from scratch)
│   │   ├── blt.py            # Local encoder / patch pooler / local decoder
│   │   ├── entropy.py        # Byte n-gram entropy model for dynamic patching
│   │   └── transformer.py    # Encoder/decoder stack, KV cache, greedy decode
│   ├── dataset.py            # Tokenized vs token-free loaders, chunking, splits
│   ├── tokenizer.py          # Byte-level BPE (from scratch) + byte tokenizer
│   ├── configs.py            # The C1–C5 definitions
│   ├── train.py              # Training loop with W&B logging
│   ├── evaluate.py           # Greedy decoding + metric computation
│   ├── utils.py              # Metrics (bit acc, Levenshtein, BLEU, ROUGE), plots
│   ├── run_all.py            # Runs C1–C5, aggregates tables + plots
│   ├── repair_timings.py     # Robust (median) wall-clock timing
│   ├── push_to_hf.py         # Upload checkpoints to Hugging Face
│   └── push_to_kaggle.py     # Upload checkpoints to Kaggle (Models + Dataset)
├── notebooks/
│   ├── 01_train_local_rtx3050.ipynb    # End-to-end local training
│   └── 02_keys_wandb_hf_kaggle.ipynb   # Keys + publishing
├── report/
│   ├── Report.tex            # Overleaf source (12pt Times, 1in margins)
│   ├── refs.bib
│   └── make_tables.py        # Generates results_tables.tex from real results
├── tests/test_modules.py     # 38 unit tests for the from-scratch modules
├── outputs/                  # Metrics, logs, plots, checkpoints
├── run_ablation.py           # One-command full ablation
├── setup_keys.py             # Parse a key file -> .env, verify all three services
├── make_submission.py        # Build the submission ZIP (with a secret audit)
├── requirements.txt
├── README.md
└── Report.pdf
```

---

## Setup

### 1. Install the **CUDA** build of PyTorch

> ⚠️ **This is the step everyone gets wrong.** A plain `pip install torch` on Windows
> installs the **CPU-only** wheel, which silently ignores your GPU and runs ~30× slower.

```bash
python -m pip install --upgrade torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

Verify — this must print `True`:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.8.0+cu126 True
```

(`cu126` matches driver 571.96 / CUDA 12.8 on an RTX 3050. For other drivers see
<https://pytorch.org/get-started/locally/>.)

### 2. Everything else

```bash
python -m pip install -r requirements.txt
```

### 3. Data

`data/brown_cipher.txt` and `data/brown_plain.txt` — 5,000 line-aligned pairs.
Not shipped in the submission ZIP; download from the link in the assignment brief.

### 4. Credentials

```bash
cp .env.example .env      # then paste your keys into .env
python setup_keys.py      # verifies W&B + HF + Kaggle
```

`.env` is git-ignored **and** actively excluded by `make_submission.py`, which refuses to
build a ZIP containing anything that looks like a secret. Full walkthrough (where each key
comes from, the three places tools look for them, and the exact Windows paths) is in
[`notebooks/02_keys_wandb_hf_kaggle.ipynb`](notebooks/02_keys_wandb_hf_kaggle.ipynb).

---

## Running

### Everything, one command

```bash
python run_ablation.py
```

Trains C1→C5, evaluates each with greedy decoding, logs to W&B, and writes
`outputs/all_results.json`, `outputs/report_tables.md` and the summary plots.

**~5.5–6 hours** on an RTX 3050 Laptop (4 GB).

### Useful variants

```bash
python run_ablation.py --configs C1,C5      # a subset
python run_ablation.py --skip-existing      # resume after an interruption
python run_ablation.py --aggregate-only     # rebuild tables/plots, no training
```

### A single configuration

```bash
python -m src.train --config C3 \
    --src-file data/brown_cipher.txt --tgt-file data/brown_plain.txt \
    --chunk-chars 32 --epochs 25 --batch-size 64
```

### Two-minute sanity check

```bash
python -m src.train --config C1 --src-file data/brown_cipher.txt \
    --tgt-file data/brown_plain.txt --chunk-chars 32 --smoke --wandb disabled
```

### Unit tests

```bash
python -m pytest tests/test_modules.py -q     # 38 passed
```

These verify the claims the report makes: SDPA against an explicit reference, RoPE
producing genuinely *relative* logits, GQA reducing to MHA when `n_kv_heads == n_heads`,
decoder causality, KV-cache equivalence to full recomputation, BLT causality and
non-leakage, and that the learned BPE is real variable-length subword (not a degenerate
fixed-width code).

---

## Publishing

```bash
python -m src.push_to_hf     --repo-id <hf-username>/anlp-a1-transformer-blt-ablation
python -m src.push_to_kaggle --target both --public
python make_submission.py    --rollnumber <rollnumber>
```

---

## Results

<!-- RESULTS_TABLE_START -->
All five trained on one **RTX 3050 Laptop GPU (4 GB, bfloat16)**, identical hyperparameters,
seed 42, greedy decoding, 1,000 test segments. Each config trained in its **own process**, so
peak memory is a true per-model high-water mark.

### Quality

| Config | Bit Acc % | Seq Acc % | Levenshtein (lower better) | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| C1 (base) | 97.17 | 77.20 | 0.44 | 90.34 | 95.10 |
| **C2 (RoPE)** | **97.61** | **81.20** | **0.35** | **92.27** | **96.06** |
| C3 (GQA) | 96.81 | 73.80 | 0.54 | 88.54 | 94.21 |
| C4 (RMSNorm) | 97.33 | 78.80 | 0.42 | 90.91 | 95.42 |
| C5 (BLT) | 97.02 | 70.00 | 1.00 | 82.80 | 91.23 |

### Cost

| Config | Params (M) | s/epoch | Epochs | Peak GPU (MB) | Decode (s) |
|---|---|---|---|---|---|
| C1 | 7.62 | 143.0 | 25 | 372 | 6.6 |
| C2 | 7.62 | 153.8 | 25 | 372 | 6.6 |
| C3 | 6.73 | 147.2 | 25 | 360 | 6.1 |
| C4 | 7.61 | **131.9** | 25 | **350** | **5.7** |
| C5 | **6.71** | 275.7 | 25 | 523 | 27.6 |

### What each change did

- **C2 (RoPE)** — best tokenized config. **+4.0 pts sequence accuracy** for *zero* parameters
  and *zero* extra memory; costs 7.6% throughput. The cipher is periodic in position
  (period 8), so relative position is the better inductive bias.
- **C3 (GQA)** — the only strict quality regression (−3.4 pts), buying 11.6% fewer parameters
  and 7.6% faster decoding. Note it is **slower to train**: teacher forcing has no KV-cache,
  so GQA's benefit doesn't apply, leaving only the `expand`/`reshape` overhead. GQA is an
  *inference-time* optimization.
- **C4 (RMSNorm)** — a strict win: accuracy equal within noise, but 7.7% faster epochs,
  6.0% less memory, 14.9% faster decoding. Re-centering earns nothing here and costs real time.
- **C5 (BLT)** — the largest effect, and a **negative** one: −7.2 pts sequence accuracy at
  1.41× peak memory, 1.93× epoch time and 4.15× decode time, in exchange for the *fewest*
  parameters (6.71 M, no vocabulary).

**Why the token-free model loses here.** Note the contradiction: C5 has the **lowest**
teacher-forced perplexity of all five (1.10) yet the **worst** sequence accuracy. Given the
correct prefix the next byte is nearly determined, but under free-running greedy decoding it
is weakest, so its errors are about *where* a byte belongs, not *which* byte. The failures
show it directly:

| | error style | example |
|---|---|---|
| C1 (BPE) | local substitution | `Dynasty` becomes `Dylesty` |
| C5 (BLT) | positional / shift | `half of the Mi` becomes `off hald the M` |

After bit-packing, the task is rigidly aligned: output byte `i` depends on source byte `i`
and `i mod 8`, so the ideal model learns a fixed diagonal correspondence. Entropy patching is
**content-driven**, and the two sides get cut at unrelated places: cipher bytes are
near-uniform (H0 = 6.04 bits) so their boundaries are near-arbitrary, while plaintext is
English (H0 = 4.34 bits) and breaks at word-like positions. Neither respects the key period
of 8, nor aligns with the other, so the patch bottleneck must rebuild a strict one-to-one
correspondence across boundaries that do not correspond.

This is a property of the *interaction* between mechanism and task, not a defect of the
mechanism. On natural language, where entropy boundaries and linguistic structure coincide,
the ordering would be expected to reverse.

### How C5 (BLT) is implemented

Two details matter and are easy to get wrong:

1. **Bytes, not the text of bytes.** The ciphertext is a *binary sequence* stored as ASCII
   `0`/`1` characters. Every **8 bits are packed MSB-first into one byte value 0-255**
   before the model sees them, so a 256-character cipher segment is **32 bytes**, not 256.
   Treating each `0`/`1` character as a byte would feed the model the *text encoding* of
   the data and inflate the sequence eightfold. The plaintext side is real text, so its
   UTF-8 bytes are used directly.

2. **Entropy-based dynamic patching, not fixed width.** Patches are variable length. A new
   patch opens where the next byte is hard to predict:

   ```
   H(x_t | x_{t-2}, x_{t-1}) > theta
   ```

   The entropy comes from a lightweight **order-2 byte n-gram (Markov) model** with
   add-alpha smoothing and stupid backoff (`src/models/entropy.py`), fit by one counting
   pass over the **training split only** (no leakage). No separate neural LM is trained.
   Because the estimator is an n-gram, the conditional entropy depends only on the
   *context*, so it precomputes into three lookup tables (`H0`, `H1[256]`, `H2[65536]`)
   registered as model buffers. Training and incremental generation therefore run
   **identical** boundary logic, and the tables travel inside the checkpoint.

   `theta` is calibrated as the quantile of training entropies giving a target mean patch
   length, since a raw threshold does not transfer between data of different character
   (cipher bytes are near-uniform; English is not). Measured on this data:

   | side | bytes/seq | patches/seq | mean patch len | distinct lengths |
   |---|---|---|---|---|
   | source (cipher) | 34 | 10.3 | 3.30 | 32 |
   | target (English) | 34 | 10.0 | 3.40 | 17 |

   **Causality holds** because the boundary at `t` is a function of `x_{t-2}, x_{t-1}` and
   never of `x_t`, so a byte's patch is known *before* that byte is generated.
   `tests/test_modules.py` verifies this directly.

   `--blt-fixed-patches` switches back to fixed-width patches as an ablation.

> **Note on timing.** `sec_per_epoch` is the **median** over epochs, not the mean. These are
> wall-clock measurements: one C4 epoch absorbed an OS suspend and recorded 24,890 s against
> ~127 s of actual compute. The median is immune to that; on uninterrupted runs it agrees with
> the mean to within 2%. `src/repair_timings.py` normalizes this across all configs.
<!-- RESULTS_TABLE_END -->

---

## Notes on the data

Reading the cipher 8 bits at a time as a byte and XOR-ing against the aligned plaintext
character reveals a repeating 8-byte key (`ANLP2026`), restarting at each line:

```
cipher[8i:8i+8] = bits8( ord(plain[i]) XOR key[i mod 8] )
```

**No model is ever given this.** It matters for two reasons: the task is exactly learnable
(so any error is optimization or capacity, not noise), and the target at position `i`
depends on `i mod 8`, which makes the positional-encoding ablation genuinely meaningful
on this dataset.

Sequences are cut into aligned **32-character segments** *after* the 80/10/10
document-level split — so no document straddles two splits — giving
**76,823 / 9,461 / 9,544** train/val/test segments. 32 is a multiple of the key period 8,
so every segment starts at key phase 0.
