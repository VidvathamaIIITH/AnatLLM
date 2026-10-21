"""Upload the trained checkpoints and Part 3 generations to the Hugging Face Hub.

    hf auth login                  # or set HF_TOKEN in the environment
    python scripts/upload_to_hf.py --user <hf-username> --wandb <wandb-project-url>

It creates one repository per part, each pushed as a single commit:

    <user>/anlp-a2-part1-moe          model    five Part 1 FFN ablations
    <user>/anlp-a2-part2-optimizers   model    five Part 2 optimizer runs
    <user>/anlp-a2-part3-generations  dataset  raw generations of every decoding strategy

Model repositories get, per run, the ``.pt`` checkpoint(s), ``train_log.jsonl``,
the run config and the evaluation JSON, plus the shared tokenizer, the YAML
config and a generated model card.  ``--dry-run`` lists what would be uploaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.part1.model import VARIANT_LABELS, VARIANTS  # noqa: E402
from src.part2.optimizers import OPTIMIZER_CATEGORY, OPTIMIZER_LABEL  # noqa: E402
from src.utils import configure_stdout, work_dir  # noqa: E402

Files = list[tuple[Path, str]]


def _add(files: Files, path: Path, dest: str) -> None:
    if path.exists():
        files.append((path, dest))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Part 1
# --------------------------------------------------------------------------- #


def collect_part1(root: Path, ckpts: list[str]) -> Files:
    files: Files = []
    runs = root / "part1" / "runs"
    for v in VARIANTS:
        for name in (*ckpts, "run_config.json", "summary.json", "eval.json", "train_log.jsonl"):
            _add(files, runs / v / name, f"{v}/{name}")
    _add(files, root / "part1" / "tokenizer.json", "tokenizer.json")
    _add(files, REPO / "configs" / "part1.yaml", "part1.yaml")
    return files


def part1_card(root: Path, wandb_url: str | None, ckpts: list[str]) -> str:
    rows = []
    for v in VARIANTS:
        p = root / "part1" / "runs" / v / "eval.json"
        if not p.exists():
            continue
        e = _load(p)
        rows.append(f"| {VARIANT_LABELS[v]} | `{v}/` | {e['params_total']/1e6:.2f}M | "
                    f"{e['params_active']/1e6:.2f}M | {e['perplexity']['ppl_target']:.2f} | "
                    f"{e['bleu_beam']['all']['bleu']:.2f} |")
    table = "\n".join(rows) or "| (no evaluation results found) | | | | | |"
    ckpt_note = "Test metrics are for `best.pt`, the checkpoint with the lowest validation loss."
    if "final.pt" in ckpts:
        ckpt_note += " `final.pt` holds the weights after the last training step."
    logs = (f"Training curves for every run: [WandB]({wandb_url}) "
            f"(group `part1-moe`, runs `part1-<variant>`).\n" if wandb_url else "")
    return f"""---
license: mit
language: [en, vi, ja]
tags: [mixture-of-experts, translation, from-scratch, anlp]
datasets: [belumind/en-vi-ja-curated-500k-triplets]
---

# ANLP Assignment 2 - Part 1: Mixture-of-Experts feed-forward ablations

Five decoder-only transformers trained from scratch on
[belumind/en-vi-ja-curated-500k-triplets](https://huggingface.co/datasets/belumind/en-vi-ja-curated-500k-triplets)
for vi->en and ja->en translation, each on the same 30,000,842-token stream.
The runs differ **only** in the feed-forward block: variants 1-4 share a total
parameter budget and variant 5 matches the dense baseline's *active* parameter count.

| Variant | Folder | Total params | Active/token | Test PPL (target) | BLEU (beam-4) |
|---|---|---|---|---|---|
{table}

{ckpt_note}  Each folder also holds `train_log.jsonl`, `run_config.json`,
`summary.json` and `eval.json`.  All variants share `tokenizer.json`
(32k byte-level BPE) and `part1.yaml`.

{logs}
```python
from src.part1.model import TransformerLM
model, extra = TransformerLM.load("dense/best.pt", map_location="cpu")
```
"""


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #


def collect_part2(root: Path) -> Files:
    files: Files = []
    runs = root / "part2" / "runs"
    for d in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []:
        for name in ("final.pt", "summary.json", "train_log.jsonl"):
            _add(files, d / name, f"{d.name}/{name}")
    sweep = root / "part2" / "sweep"
    for p in sorted(sweep.glob("sweep_*.json")) if sweep.exists() else []:
        files.append((p, f"lr_sweep/{p.name}"))
    _add(files, root / "part2" / "tokenizer.json", "tokenizer.json")
    _add(files, REPO / "configs" / "part2.yaml", "part2.yaml")
    return files


def part2_card(root: Path, wandb_url: str | None) -> str:
    rows = []
    runs = root / "part2" / "runs"
    for d in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []:
        p = d / "summary.json"
        if not p.exists():
            continue
        s = _load(p)
        rows.append(f"| {OPTIMIZER_LABEL.get(d.name, d.name)} | {OPTIMIZER_CATEGORY.get(d.name, '')} | "
                    f"`{d.name}/` | {s['hyperparams'].get('lr', float('nan')):g} | "
                    f"{s['final_val_loss']:.4f} | {s['final_test_bleu']:.2f} | "
                    f"{s['optimizer_state_ratio']:.2f}x |")
    table = "\n".join(rows) or "| (no runs found) | | | | | | |"
    logs = (f"Training curves: [WandB]({wandb_url}) (group `part2-optimizers`; the "
            f"learning-rate sweep trials are in group `part2-lr-sweep`).\n" if wandb_url else "")
    return f"""---
license: mit
language: [en]
tags: [optimizers, pretraining, from-scratch, anlp]
datasets: [browndw/human-ai-parallel-corpus]
---

# ANLP Assignment 2 - Part 2: from-scratch optimizers

The dense Part 1 architecture pre-trained for next-token prediction on
[browndw/human-ai-parallel-corpus](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus)
for one full pass (1x the dataset), once per optimizer.  Every optimizer is
implemented from scratch (only `torch.optim.Optimizer` is used, as a base
class), one per category of Table 1 of *Fantastic Pretraining Optimizers and
Where to Find Them*; Sophia-G is the bonus Hessian-based optimizer.

| Optimizer | Category | Folder | LR | Final val loss | Final test BLEU | Optimizer state |
|---|---|---|---|---|---|---|
{table}

Each folder holds `final.pt` (weights after 1x the dataset), `summary.json`
(hyperparameters and the validation-loss / test-BLEU curve at every 0.1x of
the dataset) and `train_log.jsonl`.  `lr_sweep/` has the learning-rate sweep
results that chose each LR.  All runs share `tokenizer.json` (16k BPE) and `part2.yaml`.

{logs}
```python
from src.part1.model import TransformerLM
model, extra = TransformerLM.load("muon/final.pt", map_location="cpu")
```
"""


# --------------------------------------------------------------------------- #
# Part 3
# --------------------------------------------------------------------------- #


def collect_part3(root: Path) -> Files:
    files: Files = []
    gen = root / "part3" / "generations"
    for p in sorted(gen.glob("*.jsonl")) if gen.exists() else []:
        files.append((p, f"generations/{p.name}"))
    _add(files, root / "part3" / "results.json", "results.json")
    _add(files, root / "part3" / "beam_timing.json", "beam_timing.json")
    _add(files, REPO / "configs" / "part3.yaml", "part3.yaml")
    return files


def part3_card(root: Path, wandb_url: str | None) -> str:
    rows = []
    p = root / "part3" / "results.json"
    if p.exists():
        for m in _load(p)["results"]:
            rows.append(f"| {m['label']} | {m['gen_perplexity']:.2f} | {m['accuracy']:.4f} | "
                        f"{m['f1']:.4f} | {m['repetition_4gram']:.3f} |")
    table = "\n".join(rows) or "| (no results found) | | | | |"
    logs = f"WandB run: [part3-decoding]({wandb_url}).\n" if wandb_url else ""
    return f"""---
license: mit
language: [en]
tags: [decoding, beam-search, top-k, top-p, anlp]
configs:
- config_name: default
  data_files: generations/*.jsonl
---

# ANLP Assignment 2 - Part 3: raw generations of each decoding strategy

[EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) prompted
with the first sentence of 1000 [ROCStories](https://huggingface.co/datasets/hamishivi/ROCStories)
test stories; the reference is sentences 2-5.  Decoding uses only `model.forward()`.

`generations/<strategy>.jsonl` has one line per story: `prompt`, `reference`,
`generation` (raw model output), `generation_scored` (first four sentences,
which the metrics use), `n_tokens` and `sum_logprob`.

| Strategy | Perplexity (own output) | Accuracy | Unigram F1 | Repeated 4-grams |
|---|---|---|---|---|
{table}

{logs}"""


# --------------------------------------------------------------------------- #


def upload(repo_id: str, repo_type: str, files: Files, card: str,
           dry_run: bool, private: bool) -> None:
    """Upload ``files`` with one commit per top-level folder.

    Committing folder by folder means an interrupted upload keeps everything
    that finished, and files already on the Hub with the same size are skipped,
    so re-running the script resumes where it stopped.
    """
    print(f"\n=== {repo_id} ({repo_type}) ===")
    for p, dest in files:
        print(f"  {dest:44s} {p.stat().st_size/1024**2:8.2f} MiB")
    print(f"  total: {sum(p.stat().st_size for p, _ in files)/1024**2:.1f} MiB")
    if dry_run or not files:
        print("  (dry run - nothing uploaded)" if dry_run else "  nothing to upload, skipping")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=private)
    remote = {f.path: getattr(f, "size", None)
              for f in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True)}

    groups: dict[str, Files] = {}
    for p, dest in files:
        if remote.get(dest) == p.stat().st_size:
            continue
        groups.setdefault(dest.split("/")[0] if "/" in dest else ".", []).append((p, dest))
    # Small files first, so the repository is usable as early as possible.
    order = sorted(groups, key=lambda g: sum(p.stat().st_size for p, _ in groups[g]))
    for g in order:
        ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p))
               for p, dest in groups[g]]
        api.create_commit(repo_id, repo_type=repo_type, operations=ops,
                          commit_message=f"Upload {g if g != '.' else 'shared files'}")
        print(f"  committed {g} ({len(ops)} files)", flush=True)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type=repo_type, commit_message="Update card")
    prefix = "datasets/" if repo_type == "dataset" else ""
    print(f"  https://huggingface.co/{prefix}{repo_id}", flush=True)


def main() -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Upload checkpoints to the Hugging Face Hub")
    ap.add_argument("--user", required=True, help="Hugging Face username or org")
    ap.add_argument("--wandb", default=None, help="WandB project URL for the model cards")
    ap.add_argument("--part1-repo", default="anlp-a2-part1-moe")
    ap.add_argument("--part2-repo", default="anlp-a2-part2-optimizers")
    ap.add_argument("--part3-repo", default="anlp-a2-part3-generations")
    ap.add_argument("--ckpt", nargs="+", default=["best.pt", "final.pt"],
                    help="which Part 1 checkpoints to upload")
    ap.add_argument("--part", nargs="*", choices=["part1", "part2", "part3"],
                    default=["part1", "part2", "part3"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = work_dir()
    # Smallest first, so an interrupted run still leaves complete repositories.
    if "part3" in args.part:
        upload(f"{args.user}/{args.part3_repo}", "dataset", collect_part3(root),
               part3_card(root, args.wandb), args.dry_run, args.private)
    if "part2" in args.part:
        upload(f"{args.user}/{args.part2_repo}", "model", collect_part2(root),
               part2_card(root, args.wandb), args.dry_run, args.private)
    if "part1" in args.part:
        upload(f"{args.user}/{args.part1_repo}", "model", collect_part1(root, args.ckpt),
               part1_card(root, args.wandb, args.ckpt), args.dry_run, args.private)


if __name__ == "__main__":
    main()
