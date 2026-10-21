# AnatLLM: The Anatomy of a Large Language Model

Every major component of a modern LLM, built from elementary PyTorch operations and studied through controlled ablations: attention, positional encodings, normalisation, tokenisation, feed-forward / Mixture-of-Experts layers, optimizers and decoding.

Work from the Advanced NLP course (Monsoon 2026, IIIT Hyderabad) by **Vidvathama R**.

| | Project | Components built from scratch | Report |
|---|---|---|---|
| 1 | [Transformers from Scratch, Architectural Variants & Byte Latent Transformers](Assignment1/) | Encoder–decoder Transformer, MHA / GQA, sinusoidal / RoPE, LayerNorm / RMSNorm, byte-level BPE, Byte Latent Transformer with entropy-based patching, KV-cache greedy decoding | [PDF](Assignment1/Report.pdf) |
| 2 | [Mixture of Experts, Optimizers, Decoding Strategies](Assignment2/) | Decoder-only LM with dense and 4 MoE FFN layouts (top-1, top-2, shared + routed, wide), AdamW, MARS, Lion, Muon, Sophia-G, greedy / top-k / top-p / beam search with KV cache | [PDF](Assignment2/report.pdf) |

## Assignment 1: architectural ablation (C1–C5)

A ~7.6M-parameter encoder–decoder Transformer decrypts XOR-enciphered binary sequences back into English text (Brown corpus). The model uses no `nn.Transformer`, `nn.MultiheadAttention` or `F.scaled_dot_product_attention`. Each of C2–C5 changes exactly one component of the C1 baseline.

| Config | Change | Bit Acc (%) | Seq Acc (%) | BLEU | s / epoch |
|---|---|---|---|---|---|
| C1 | baseline (sinusoidal, MHA, LayerNorm, BPE) | 97.17 | 77.2 | 90.34 | 143.0 |
| C2 | RoPE | **97.61** | **81.2** | **92.27** | 153.8 |
| C3 | Grouped-Query Attention (8q / 2kv) | 96.81 | 73.8 | 88.54 | 147.2 |
| C4 | RMSNorm | 97.33 | 78.8 | 90.91 | **131.9** |
| C5 | Byte Latent Transformer (token-free) | 97.02 | 70.0 | 82.80* | 275.7 |

\*C5 BLEU is byte-level and is included for reference only.

Details: [Assignment1/README.md](Assignment1/README.md) · [W&B](https://wandb.ai/vidvathama-babu-iiit-hyderabad/anlp-a1-blt-ablation) · [Hugging Face checkpoints](https://huggingface.co/Viv0605101/anlp-a1-transformer-blt-ablation)

## Assignment 2: MoE, optimizers, decoding

- **Part 1, Mixture of Experts.** A ~23M-parameter decoder-only Transformer (RMSNorm, RoPE, tied embeddings) translates vi/ja→en with a swappable FFN: dense MLP or 4 MoE layouts with load-balancing and router z-loss. Dense reaches **31.04 BLEU** (beam 4). MoE quality tracks *active* rather than total parameters: the wide MoE (30M total, 22.9M active) ties dense.
- **Part 2, Optimizers.** Five optimizers implemented from scratch, one from each category of *Fantastic Pretraining Optimizers*, with a 25-trial LR sweep. Final validation loss: **MARS 4.073** < AdamW 4.093 < Muon 4.113 < Sophia-G 4.202 < Lion 4.209. The ranking changes with the training budget.
- **Part 3, Decoding.** Greedy, top-k, top-p and beam search on Pythia-160M, built only on `model.forward()` with a KV cache. Top-k (k=10) raises unigram F1 to **0.181** from 0.117 for greedy (+55% relative). Greedy repeats 49% of 4-grams. Beam search lowers perplexity but also lowers F1.

Details: [Assignment2/README.md](Assignment2/README.md) · [W&B](https://wandb.ai/vidvathama-babu-iiit-hyderabad/anlp-a2) · Hugging Face: [MoE checkpoints](https://huggingface.co/Viv0605101/anlp-a2-part1-moe), [optimizer checkpoints](https://huggingface.co/Viv0605101/anlp-a2-part2-optimizers), [generations](https://huggingface.co/datasets/Viv0605101/anlp-a2-part3-generations)

## Repository layout

```
Assignment1/          code, data, trained checkpoints (outputs/C1-C5/best.pt), W&B logs, LaTeX report
Assignment2/          code, configs, tests, figures (assets/), report data, W&B logs, LaTeX report
SparseLM_bundle.zip   a packaged copy of Assignment 2 with a short guide (GUIDE.md)
```

Each assignment folder is self-contained; see its README for setup and run instructions. API keys are not included. Copy `Assignment1/.env.example` to `.env` and fill in your own keys.
