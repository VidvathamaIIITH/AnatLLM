"""Launch the full C1-C5 ablation with the exact hyperparameters used in the report.

    python run_ablation.py                 # all five configs
    python run_ablation.py --configs C1,C5 # a subset
    python run_ablation.py --aggregate-only

Every flag not listed here is forwarded to src/run_all.py -> src/train.py.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- load .env so WANDB_API_KEY / HF_TOKEN are visible to the child runs ----
env_path = os.path.join(HERE, ".env")
if os.path.exists(env_path):
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())

# ---- shared hyperparameters (identical for every configuration) ------------
SHARED = [
    "--src-file", "data/brown_cipher.txt",
    "--tgt-file", "data/brown_plain.txt",
    "--chunk-chars", "32",
    "--vocab-size", "4096",
    "--d-model", "256",
    "--n-layers", "3",
    "--n-heads", "8",
    "--n-kv-heads", "2",          # C3 only (GQA): 8 query heads -> 2 KV heads
    "--d-ff", "1024",
    "--dropout", "0.1",
    "--epochs", "25",
    "--batch-size", "64",
    "--eval-batch-size", "64",
    "--lr", "3e-4",
    "--min-lr", "1e-5",
    "--warmup-steps", "500",
    "--scheduler", "cosine",
    "--label-smoothing", "0.1",
    "--grad-clip", "1.0",
    "--patience", "5",
    "--max-eval-samples", "1000",
    # ---- C5 (BLT) only -------------------------------------------------
    # Patching is ENTROPY-BASED and dynamic: an order-2 byte n-gram model is fit
    # on the training split, its threshold calibrated so patches average
    # --blt-target-patch bytes. --blt-patch-size is only the fixed-width
    # fallback used by the --blt-fixed-patches ablation and the unit tests.
    "--blt-patch-size", "4",
    "--blt-target-patch", "4.0",
    "--blt-max-patch", "12",
    "--blt-entropy-alpha", "0.5",
    "--blt-d-local", "128",
    "--blt-local-enc-layers", "1",
    "--blt-local-dec-layers", "2",
    "--blt-local-heads", "4",
    "--blt-window", "64",
    "--seed", "42",
    "--amp", "auto",
    "--device", "auto",
    "--wandb", "online",
    "--wandb-project", os.environ.get("WANDB_PROJECT", "anlp-a1-blt-ablation"),
]
if os.environ.get("WANDB_ENTITY"):
    SHARED += ["--wandb-entity", os.environ["WANDB_ENTITY"]]

cmd = [sys.executable, "-m", "src.run_all", "--out-dir", "outputs"] + SHARED + sys.argv[1:]
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.run(cmd, cwd=HERE, env=os.environ).returncode)
