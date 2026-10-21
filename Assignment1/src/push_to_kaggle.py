"""Publish the trained C1-C5 checkpoints to Kaggle.

Two independent targets are supported; ``--target both`` (the default) does each
in turn and keeps going if one of them fails.

* **Kaggle Models** (``--target model``) -- the modern home for weights.  Uses
  ``kagglehub.model_upload`` and produces the handle
  ``<owner>/<slug>/pyTorch/<variation>``.
* **Kaggle Datasets** (``--target dataset``) -- a plain file bundle, handy when
  you want to attach the checkpoints to a Kaggle notebook.  Uses the official
  ``kaggle`` API and writes the required ``dataset-metadata.json`` for you.

Credentials are read from (in order):
  1. ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables,
  2. the project ``.env`` file,
  3. ``%USERPROFILE%\\.kaggle\\kaggle.json`` (what "Create New Token" downloads).

Examples::

    python -m src.push_to_kaggle --owner myuser
    python -m src.push_to_kaggle --owner myuser --target dataset --public
    python -m src.push_to_kaggle --owner myuser --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"

from .utils import load_json  # noqa: E402

CONFIG_NAMES = ["C1", "C2", "C3", "C4", "C5"]

# Per-config files worth publishing (small; the .pt is the only heavy one).
UPLOAD_NAMES = (
    "best.pt", "config.json", "metrics.json", "history.json",
    "tokenizer_src.json", "tokenizer_tgt.json", "predictions_test.json",
    "entropy_model.json",   # C5: the n-gram model that defines patch boundaries
)
# Repo-level extras
ROOT_EXTRAS = ("all_results.json", "report_tables.md",
               "ablation_quality.png", "ablation_efficiency.png")


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip() and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip()


def ensure_kaggle_credentials(repo_root: str) -> str:
    """Make sure the Kaggle SDK can authenticate; return the resolved username."""
    load_dotenv(os.path.join(repo_root, ".env"))

    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if not (user and key):
        # fall back to ~/.kaggle/kaggle.json
        home = os.path.expanduser("~")
        kj = os.path.join(home, ".kaggle", "kaggle.json")
        if os.path.exists(kj):
            try:
                blob = json.load(open(kj, encoding="utf-8"))
                user = user or str(blob.get("username", "")).strip()
                key = key or str(blob.get("key", "")).strip()
                os.environ["KAGGLE_USERNAME"] = user
                os.environ["KAGGLE_KEY"] = key
            except Exception as e:
                raise SystemExit(f"[kaggle] {kj} exists but could not be parsed: {e}")

    if not (user and key):
        raise SystemExit(
            "[kaggle] No credentials found.\n"
            "  Fix with ONE of:\n"
            "   (a) add KAGGLE_USERNAME=... and KAGGLE_KEY=... to your .env, or\n"
            "   (b) go to https://www.kaggle.com/settings -> API -> 'Create New Token',\n"
            "       then place the downloaded kaggle.json at "
            "%USERPROFILE%\\.kaggle\\kaggle.json"
        )
    return user


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
def collect_results(out_dir: str) -> Dict[str, dict]:
    results = {}
    for name in CONFIG_NAMES:
        p = os.path.join(out_dir, name, "metrics.json")
        if os.path.exists(p):
            results[name] = load_json(p)
    return results


def build_card(owner: str, slug: str, results: Dict[str, dict]) -> str:
    rows = []
    for n in sorted(results):
        r = results[n]
        rows.append(
            f"| {n} | {r.get('changed','')} | {r.get('bit_accuracy', float('nan')):.2f} | "
            f"{r.get('sequence_accuracy', float('nan')):.2f} | "
            f"{r.get('levenshtein', float('nan')):.2f} | "
            f"{r.get('params_millions', float('nan')):.2f} | "
            f"{r.get('peak_gpu_mem_mb', float('nan')):.0f} |"
        )
    table = "\n".join(rows) if rows else "| - | - | - | - | - | - | - |"
    return f"""# ANLP Assignment 1 — Custom Transformers & Byte Latent Transformers

Encoder–decoder Transformers implemented **from scratch** in PyTorch (no
`nn.Transformer`, no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`)
that decrypt XOR-enciphered binary sequences back into plaintext English, plus a
controlled five-way architectural ablation.

## Configurations

| Config | Change from base | Bit Acc (%) | Seq Acc (%) | Levenshtein | Params (M) | Peak GPU MB |
|---|---|---|---|---|---|---|
{table}

* **C1** — baseline: sinusoidal absolute PE, Multi-Head Attention, LayerNorm, BPE subwords
* **C2** — C1 but **RoPE** instead of sinusoidal PE
* **C3** — C1 but **Grouped-Query Attention** (8 query heads → 2 KV heads)
* **C4** — C1 but **RMSNorm** instead of LayerNorm
* **C5** — C1 but **token-free BLT** (raw bytes → local encoder → patches → global
  transformer → local byte decoder)

## Files

Each `C*/` folder holds `best.pt` (the checkpoint), `config.json` (the exact
architecture + hyperparameters needed to rebuild it), `metrics.json`,
`history.json`, the two tokenizer JSONs, and sample test predictions.

## Loading a checkpoint

```python
import torch, json
ckpt = torch.load("C1/best.pt", map_location="cpu")
cfg  = json.load(open("C1/config.json"))
# rebuild with src/configs.py:build_model(...) from the accompanying repo,
# then: model.load_state_dict(ckpt["model"])
```

Produced for the Advanced NLP course assignment; trained on a single
NVIDIA RTX 3050 Laptop GPU (4 GB).
"""


def stage(out_dir: str, staging: str, owner: str, slug: str,
          results: Dict[str, dict], verbose: bool = True) -> int:
    """Copy the publishable artefacts into a clean staging folder."""
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    n_files = 0
    for name in CONFIG_NAMES:
        src_dir = os.path.join(out_dir, name)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(staging, name)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in UPLOAD_NAMES:
            s = os.path.join(src_dir, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst_dir, fn))
                n_files += 1
        for fn in os.listdir(src_dir):          # curve plots
            if fn.endswith(".png"):
                shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
                n_files += 1

    for fn in ROOT_EXTRAS:
        s = os.path.join(out_dir, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(staging, fn))
            n_files += 1

    with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_card(owner, slug, results))
    n_files += 1

    if verbose:
        mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(staging) for f in fs
        ) / 1024**2
        print(f"[kaggle] staged {n_files} files ({mb:.1f} MB) in {staging}")
    return n_files


# --------------------------------------------------------------------------- #
# upload targets
# --------------------------------------------------------------------------- #
def push_model(staging: str, owner: str, slug: str, variation: str,
               notes: str, license_name: str) -> Optional[str]:
    try:
        import kagglehub
    except ImportError:
        print("[kaggle] kagglehub not installed -> pip install kagglehub")
        return None

    handle = f"{owner}/{slug}/pyTorch/{variation}"
    print(f"[kaggle] uploading MODEL -> {handle}")
    try:
        kagglehub.model_upload(
            handle=handle,
            local_model_dir=staging,
            license_name=license_name,
            version_notes=notes,
        )
    except Exception as e:
        print(f"[kaggle] model upload FAILED: {type(e).__name__}: {e}")
        return None
    url = f"https://www.kaggle.com/models/{owner}/{slug}"
    print(f"[kaggle] model page -> {url}")
    return url


def push_dataset(staging: str, owner: str, slug: str, title: str,
                 public: bool, notes: str) -> Optional[str]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[kaggle] kaggle SDK not installed -> pip install kaggle")
        return None

    meta = {
        "title": title[:50],
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(staging, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    api = KaggleApi()
    api.authenticate()

    print(f"[kaggle] uploading DATASET -> {owner}/{slug} ({'public' if public else 'private'})")
    try:
        # If it already exists this raises; fall back to a new version.
        api.dataset_create_new(
            folder=staging, public=public, quiet=False, dir_mode="zip"
        )
    except Exception as e:
        msg = str(e)
        if "already exists" in msg.lower() or "409" in msg:
            print("[kaggle] dataset exists -> pushing a new version")
            try:
                api.dataset_create_version(
                    folder=staging, version_notes=notes, quiet=False, dir_mode="zip"
                )
            except Exception as e2:
                print(f"[kaggle] dataset version FAILED: {type(e2).__name__}: {e2}")
                return None
        else:
            print(f"[kaggle] dataset upload FAILED: {type(e).__name__}: {e}")
            return None
    url = f"https://www.kaggle.com/datasets/{owner}/{slug}"
    print(f"[kaggle] dataset page -> {url}")
    return url


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(repo_root, "outputs"))
    ap.add_argument("--owner", default="", help="Kaggle username (defaults to KAGGLE_USERNAME)")
    ap.add_argument("--slug", default="anlp-a1-transformer-blt-ablation",
                    help="model / dataset slug (lowercase, hyphens)")
    ap.add_argument("--title", default="ANLP A1 - Transformer & BLT Ablation (C1-C5)")
    ap.add_argument("--variation", default="c1-c5", help="Kaggle Models variation slug")
    ap.add_argument("--target", default="both", choices=["model", "dataset", "both"])
    ap.add_argument("--license", default="CC0-1.0")
    ap.add_argument("--notes", default="Initial upload: C1-C5 ablation checkpoints.")
    ap.add_argument("--public", action="store_true", help="make the dataset public")
    ap.add_argument("--staging", default="", help="staging dir (default: <out-dir>/_kaggle_upload)")
    ap.add_argument("--dry-run", action="store_true", help="stage the files but do not upload")
    a = ap.parse_args(argv)

    user = ensure_kaggle_credentials(repo_root)
    owner = (a.owner or user).strip()

    results = collect_results(a.out_dir)
    if not results:
        raise SystemExit(
            f"[kaggle] no metrics.json under {a.out_dir} -- train the models first "
            "(python run_ablation.py)"
        )
    print(f"[kaggle] found configs: {', '.join(sorted(results))}")

    staging = a.staging or os.path.join(a.out_dir, "_kaggle_upload")
    stage(a.out_dir, staging, owner, a.slug, results)

    if a.dry_run:
        print("[kaggle] --dry-run: nothing uploaded.")
        return 0

    urls = {}
    if a.target in ("model", "both"):
        urls["model"] = push_model(staging, owner, a.slug, a.variation, a.notes, a.license)
    if a.target in ("dataset", "both"):
        urls["dataset"] = push_dataset(staging, owner, a.slug, a.title, a.public, a.notes)

    ok = [k for k, v in urls.items() if v]
    print("\n[kaggle] done. succeeded:", ", ".join(ok) if ok else "(none)")
    for k, v in urls.items():
        if v:
            print(f"  {k}: {v}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
