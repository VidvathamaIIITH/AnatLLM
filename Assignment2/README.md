# ANLP Assignment 2 — Mixture of Experts, Optimizers, Decoding Strategies

Monsoon 2026. Everything is implemented in PyTorch from scratch; HuggingFace is
used only for datasets, the `tokenizers` BPE trainer, and the pretrained
Pythia-160M checkpoint of Part 3.

| Part | Task | Dataset |
|---|---|---|
| 1 | Five feed-forward variants (dense MLP + 4 MoE layouts) in a decoder-only transformer | [`belumind/en-vi-ja-curated-500k-triplets`](https://huggingface.co/datasets/belumind/en-vi-ja-curated-500k-triplets) |
| 2 | Five optimizers implemented from scratch, one per category of Table 1 of *Fantastic Pretraining Optimizers* | [`browndw/human-ai-parallel-corpus`](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus) |
| 3 | Greedy / top-k / top-p / beam-search decoding, built on `model.forward()` only | [`hamishivi/ROCStories`](https://huggingface.co/datasets/hamishivi/ROCStories) with [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m) |

## Links

| What | Where |
|---|---|
| Part 1 checkpoints (5 FFN variants) | [huggingface.co/Viv0605101/anlp-a2-part1-moe](https://huggingface.co/Viv0605101/anlp-a2-part1-moe) |
| Part 2 checkpoints (5 optimizers) | [huggingface.co/Viv0605101/anlp-a2-part2-optimizers](https://huggingface.co/Viv0605101/anlp-a2-part2-optimizers) |
| Part 3 raw generations (10 decoding configs) | [huggingface.co/datasets/Viv0605101/anlp-a2-part3-generations](https://huggingface.co/datasets/Viv0605101/anlp-a2-part3-generations) |
| Training / evaluation logs (all runs) | [wandb.ai/vidvathama-babu-iiit-hyderabad/anlp-a2](https://wandb.ai/vidvathama-babu-iiit-hyderabad/anlp-a2) |

WandB runs are grouped by part: `part1-moe` (runs `part1-<variant>`),
`part2-optimizers` (runs `part2-<optimizer>`), `part2-lr-sweep` (25 sweep
trials) and `part3-decoding` (`part3-decoding`, `part3-beam-timing`).

---

## Quick start (for TAs)

Needs only [uv](https://docs.astral.sh/uv/getting-started/installation/); it
installs Python 3.11 and every dependency from `uv.lock`. A CUDA GPU is used
when present (CUDA 12.6 wheels on Linux/Windows); everything also runs on CPU,
more slowly. Run from the repository root:

```bash
uv sync                              # 1. create the environment
uv run python main.py test           # 2. 86 unit tests (~1 min)
uv run python main.py smoke          # 3. every stage of Parts 1-3 on a tiny budget (~10 min on a GPU)
uv run python main.py fetch          # 4. download the released checkpoints / generations
uv run python main.py part1 eval --variant dense   # 5. re-score a released checkpoint
```

* `smoke` trains, evaluates and plots every part end to end on toy budgets. It
  writes only to `work/smoke/` (figures to `work/smoke/assets/`), so the real
  results and `assets/` are never touched. Its LR sweep uses a two-point grid, so the
  "peaked at the edge of its grid" warning it prints is expected.
* `fetch` lays the Hub artefacts out exactly as training left them, so
  `part1 eval` re-scores the checkpoints (drop `--variant` for all five) and
  `part1 plots` / `part2 plots` / `part3 plots` redraw the report's figures,
  all without training.
* No command asks for a WandB login: logging is skipped automatically when no
  login is found.

## Setup

Large artefacts (checkpoints, tokenizers, binarised corpora, HuggingFace caches)
are written **outside** the repository, because the code.iiit.ac.in instance has a
strict quota. Point `ANLP_WORK_DIR` wherever you have space:

```bash
export ANLP_WORK_DIR=/path/with/space      # Windows: $env:ANLP_WORK_DIR="D:\anlp_work"
```

It defaults to `./work/`, which `.gitignore` excludes.

### Logging and uploads

Training (Parts 1–2, including the LR sweep) and Part 3 decoding log to WandB
project `anlp-a2` while they run; every event is also written to a local
`train_log.jsonl`. Logging happens only when a WandB login exists (`wandb login`
or `WANDB_API_KEY`); set `WANDB_MODE=disabled` or `wandb.enabled: false` in the
YAML to switch it off explicitly.

```bash
uv run python scripts/sync_wandb.py      # publish runs trained before live logging (idempotent)
uv run python scripts/upload_to_hf.py --user <hf-user> --wandb <wandb-project-url>
```

## Running everything

```bash
uv run python main.py all            # all three parts, start to finish (many GPU-hours)
```

or one part / one stage at a time:

```bash
uv run python main.py part1 all                        # prepare, train x5, eval, plots
uv run python main.py part1 train --variant dense
uv run python main.py part2 all                        # prepare, LR sweep, train x5, plots
uv run python main.py part2 train --optimizer muon
uv run python main.py part3 all                        # decode, beam timing, plots
uv run python -m src.report_data                       # collect every result into report_data/
```

Any config value can be overridden from the command line:

```bash
uv run python main.py part1 train --variant dense --set train.lr=3e-4 train.token_budget=5000000
```

## Layout

```
main.py                      single entry point (see --help)
configs/part{1,2,3}.yaml     all hyperparameters
src/
  utils.py                   seeding, config merge, LR schedule, JSONL + WandB logging
  tokenization.py            byte-level BPE training + runtime wrapper
  plotting.py                shared matplotlib style
  report_data.py             collects results into Markdown/LaTeX tables
  part1/
    model.py                 TransformerConfig, MLP, MoE, attention, TransformerLM
    data.py                  triplets -> two translation directions, binarisation, batching
    train.py                 training loop for one FFN variant
    evaluate.py              perplexity, BLEU, expert-usage statistics
    run_eval.py              evaluation driver
    plots.py                 figures
  part2/
    optimizers/              AdamW, MARS, Lion, Muon, SophiaG (from scratch)
    data.py                  leakage-free split, packing into fixed blocks
    train.py, evaluate.py, sweep.py, plots.py
  part3/
    decoding.py              greedy, top-k, top-p, beam search + KV-cache backends
    metrics.py               perplexity, accuracy, unigram P/R/F1, diversity
    run.py                   decodes the test set with every strategy, writes raw generations
    data.py, timing.py, plots.py
tests/                       86 unit tests (model, optimizers, decoding, metrics)
scripts/sync_wandb.py        publish finished runs' JSONL logs to WandB
scripts/upload_to_hf.py      push checkpoints and Part 3 generations to the Hub
assets/                      generated figures
report_data/                 generated tables + metrics.json
```

---

## Part 1 — Mixture of Experts

A decoder-only transformer (RMSNorm, RoPE, causal multi-head attention, tied
embeddings) whose feed-forward block is chosen by `TransformerConfig.ffn_type`,
exactly as in the assignment's appendix listing:

```python
self.ffn = MoE(config) if config.use_moe else MLP(config)
```

### The five variants and the parameter-parity constraint

With `d = d_model` and the dense width `d_ff = 4d`, a 2-layer FFN costs `8d²`
parameters per layer. An expert of width `d` costs `2d²`, so **four such experts
cost exactly as much as the dense layer**. That single fact makes the required
parity constraints exact:

| # | key | Layout | Total FFN / layer | Active FFN / layer |
|---|---|---|---|---|
| 1 | `dense` | 2-layer MLP, `d_ff = 4d` | `8d²` | `8d²` |
| 2 | `moe_4e_top1` | 4 experts of width `d`, top-1 | `8d²` | `2d²` |
| 3 | `moe_4e_top2` | 4 experts of width `d`, top-2 | `8d²` | `4d²` |
| 4 | `moe_1s_3r_top1` | 1 shared + 3 routed of width `d`, top-1 routed | `8d²` | `4d²` |
| 5 | `moe_4e_top2_wide` | 4 experts of width `2d`, top-2 | `16d²` | `8d²` |

Variants 1–4 therefore share a total parameter budget and variant 5 matches the
dense baseline's *active* count — the only difference is the router matrix
(`d × E` per layer, **+0.04 %** of the model at the trained size). `tests/test_model.py`
asserts all of this exactly.

Routing is softmax top-k with the selected gates renormalised to sum to 1, a
Switch-style load-balancing auxiliary loss and a router z-loss. Padding is
excluded from both the auxiliary loss and the expert-usage statistics.

### Data

Each `(en, vi, ja)` triplet yields **two** training examples, so the 446 k
training triplets become 892 k examples / 36.7 M tokens:

```
<bos> <src_vi> {vietnamese} <tgt_en> {english} <eos>
<bos> <src_ja> {japanese}   <tgt_en> {english} <eos>
```

The tokenizer is a 32 k **byte-level** BPE trained jointly on all three
languages — byte level because the Vietnamese is diacritic-heavy and the
Japanese is not romanised and has no whitespace boundaries, so no other
pre-tokenisation covers all three without emitting `<unk>`.

Batches are length-grouped to a fixed token budget (padding overhead ≈ 6 %) and
the whole batch sequence is **planned deterministically up front**, so all five
variants consume byte-for-byte the same tokens in the same order.

---

## Part 2 — Optimizers

One optimizer per category of Table 1 of Wen et al. (2025), each implemented
from scratch; only `torch.optim.Optimizer` is imported from torch, as a base class.

| Category | Implementation | Optimizer state | Idea |
|---|---|---|---|
| 1. AdamW | `AdamW` | 2× model | decoupled weight decay |
| 2. Variance-reduced AdamW | `MARS` | 3× model | STORM-style corrected gradient `g_t + γ·β₁/(1−β₁)·(g_t − g_{t−1})` |
| 3. Memory-efficient | `Lion` | 1× model | sign of interpolated momentum; no second moment |
| 4. Matrix-based | `Muon` | 1× model (+AdamW aux) | momentum orthogonalised by a quintic Newton–Schulz iteration |
| 5. Hessian-based (bonus) | `SophiaG` | 2× model | clipped step preconditioned by a Gauss–Newton–Bartlett diagonal Hessian |

`AdamW` is verified to match `torch.optim.AdamW` to within 1e-6 over 50 steps,
and `MARS` with `γ = 0` is verified to reduce exactly to `AdamW`.

The model is the dense Part 1 configuration; only the vocabulary differs (16 k,
English-only). Documents are split on the **base document id** so that the human
and LLM rewrites of the same source text can never straddle the train/test
boundary. Tokens are packed into windows of `n_ctx + 1` with stride `n_ctx`, so
one epoch predicts every token exactly once — which makes "1× the dataset" a
well-defined budget. Validation loss and test BLEU are recorded every 0.1×.

A learning-rate sweep (the bonus) runs each optimizer over a small grid at 0.1×
the dataset and adopts the winner for the full run — necessary because Lion's
sign update and Muon's orthogonalised update live on completely different
learning-rate scales from AdamW's.

---

## Part 3 — Decoding strategies

`src/part3/decoding.py` implements greedy, top-k, top-p and beam search on top
of repeated `model.forward()` calls with an explicit KV cache. `model.generate()`
is never called. A small `LMBackend` abstraction (`prefill` / `step` / `reorder`
/ `expand`) lets the same four implementations drive both the HuggingFace
Pythia-160M checkpoint and the from-scratch translation models of Parts 1–2.

Correctness checks that the test-suite enforces:

* beam width 1 reproduces greedy decoding **token for token**;
* a wider beam never returns a lower-scoring sequence;
* the reported beam score equals the log-probability recomputed by a fresh forward pass;
* top-k keeps exactly `k` tokens and top-p keeps the smallest prefix whose mass reaches `p`;
* a left-padded batch decodes identically to each sequence decoded alone.

### Metric definitions

The prompt is the story's first sentence and the reference is sentences 2–5.
The two required metrics, reported for every configuration:

* **Perplexity** — the model's perplexity on its *own* output,
  `exp(−Σ log p / n_tokens)`, with `log p` taken from the *unmodified* model
  distribution (before temperature / top-k / top-p truncation), so it is
  comparable across strategies. The perplexity of the *gold* continuation
  (25.10, strategy-independent) is reported once as an anchor.
* **Accuracy** — positional word-token agreement with the reference (lowercased,
  punctuation stripped), normalised by the longer of the two sequences so that
  neither truncation nor padding can inflate it.

Unigram precision / recall / F1 (ROUGE-1), BLEU, chrF2, distinct-1/2 and the
within-generation repeated-4-gram rate are reported alongside as supporting
metrics. The **raw generations** of every configuration are written to
`<work>/part3/generations/<strategy>.jsonl` (prompt, reference, raw output,
scored output, token count, log-probability) and published on the Hub; all ten
are also bundled in `assets/part3_generations.jsonl.gz` (one line per story with
`prompt`, `reference` and `generations[<strategy>]`).

---

## Results

Full analysis is in [`report.pdf`](report.pdf) (ACL format, 11pt, 8 pages). `report.tex` builds with
the official [ACL style files](https://github.com/acl-org/acl-style-files) on the TeX path:
`pdflatex report && pdflatex report` (the bibliography is inline, so no BibTeX step).
Every table is in `report_data/tables.md` and every figure in `assets/`; tables
are regenerated by `uv run python -m src.report_data` and figures by the `plots`
stages.

### Part 1 — all five variants trained on an identical 30,000,842-token stream

| Variant | Total | Active/token | Test PPL (target) | BLEU (beam-4) |
|---|---|---|---|---|
| Dense MLP | 22.91M | 22.91M | 9.39 | **31.04** |
| MoE 4e top-1 | 22.92M | 17.61M | 12.66 | 26.28 |
| MoE 4e top-2 | 22.92M | 19.38M | 9.56 | 30.34 |
| MoE 1 shared + 3 routed | 22.92M | 19.38M | 10.53 | 29.23 |
| MoE 4e top-2 wide | 30.00M | 22.92M | **9.12** | 30.95 |

Quality tracks **active** parameters, not total: at matched total parameters the
dense baseline beats every sparse variant, and MoE only wins once total capacity
is added at constant compute (variant 5). Routing specialises far more by
*role* — source vs. target tokens, TVD 0.35 — than by language (TVD 0.27), and
top-1 routing barely specialises at all because the load-balancing loss and
specialisation are in direct conflict when only one expert is chosen.

### Part 2 — one epoch of the corpus (0.116× Chinchilla), per-optimizer LR sweep

| Optimizer | Category | LR | Val loss | Optimizer state |
|---|---|---|---|---|
| AdamW | 1. AdamW | 1.2e-3 | 4.0925 | 2.00× |
| MARS | 2. Variance-reduced | 1.2e-3 | **4.0731** | 3.00× |
| Lion | 3. Memory-efficient | 2.0e-4 | 4.2090 | **1.00×** |
| Muon | 4. Matrix-based | 4.0e-2 | 4.1127 | 1.37× |
| Sophia-G | 5. Hessian-based (bonus) | 3.0e-4 | 4.2017 | 2.00× |

All five land within 0.14 nats. The ranking changes with budget — AdamW leads at
0.1×, MARS from 0.2× onward — which is the source paper's point about
short-horizon comparisons, visible inside a single epoch.

### Part 3 — 1000 ROCStories with Pythia-160M

Perplexity of the gold continuation: **25.10**.

| Strategy | Perplexity (own output) | Accuracy | Unigram F1 | Repeated 4-grams |
|---|---|---|---|---|
| Greedy | 1.94 | 0.0121 | 0.1168 | 0.490 |
| Beam, width 1 | 1.94 | 0.0121 | 0.1168 | 0.490 |
| Beam, width 2 | 1.80 | 0.0113 | 0.1125 | 0.437 |
| Beam, width 4 | 1.66 | 0.0092 | 0.1001 | 0.454 |
| Top-k, k=10 | 8.04 | 0.0091 | 0.1809 | 0.011 |
| Top-k, k=40 | 13.22 | 0.0078 | 0.1742 | 0.004 |
| Top-k, k=100 | 17.93 | 0.0077 | 0.1688 | 0.002 |
| Top-p, p=0.7 | 9.77 | 0.0095 | 0.1744 | 0.010 |
| Top-p, p=0.9 | 23.05 | 0.0073 | 0.1598 | 0.003 |
| Top-p, p=0.95 | 33.48 | 0.0069 | 0.1556 | 0.001 |

Beam width 1 reproduces greedy token for token. The beam-width timing study
(widths 1, 2, 4, 8) is in `report_data/tables.md` and `assets/part3_beam_timing.png`.

## Notes on the environment

Everything was developed and run on a single **NVIDIA RTX 3050 Laptop GPU
(4 GiB)**, which is what drives the batch sizes and the ~23 M parameter model
size. `micro_batch_tokens` is set so that peak allocation stays near 2.1 GiB;
raise it on a larger card. bf16 autocast is used throughout (fp16 on pre-Ampere
cards, fp32 on CPU) — `src/utils.py:autocast_dtype` picks automatically.
