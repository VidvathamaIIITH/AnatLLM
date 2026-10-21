"""Package the submission ZIP exactly as the brief requires.

    python make_submission.py --rollnumber 2024XXXXXX

Produces ``<rollnumber>_assignment1.zip`` whose single top-level folder is
``<rollnumber>_assignment1/`` containing ``src/``, ``outputs/``, ``README.md``
and ``Report.pdf``.

Excluded from the ZIP: virtualenvs, ``__pycache__``, ``.git``, raw datasets,
``wandb/`` run directories, and (unless ``--include-checkpoints``) the ``.pt``
weights -- those live on the Hugging Face Hub, and Moodle upload limits are
usually the reason a submission fails at 23:58.
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from typing import List, Optional, Sequence

# Report_OUTLINE.md is a private scaffold for writing the report, not a
# deliverable, so it is deliberately left out of the ZIP.
INCLUDE_TOP = ["src", "outputs", "tests", "notebooks", "report",
               "README.md", "Report.pdf", "requirements.txt",
               "run_ablation.py", "setup_keys.py", ".env.example"]
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode",
                "wandb", ".ipynb_checkpoints", ".pytest_cache",
                "_kaggle_upload", "tokenizers"}
EXCLUDE_EXT = {".pyc", ".pyo"}

# Files that must NEVER leave the machine, matched on basename (case-insensitive).
# This is the last line of defence: the ZIP is uploaded publicly to Moodle.
#
# The patterns are deliberately NARROW. An over-broad rule silently drops real
# deliverables -- "token" would eat every tokenizer_*.json, and "_keys" would eat
# notebooks/02_keys_wandb_hf_kaggle.ipynb (which stores no secrets: it reads them
# with getpass). Dropping a graded file is as bad as leaking one, so match on
# exact names and unambiguous credential words only.
SECRET_NAMES = {".env", "kaggle.json", "_netrc", ".netrc",
                "anlp_keys_a1.txt", "keys.txt", "secrets.json",
                "credentials.json", "client_secret.json", "id_rsa"}
SECRET_SUBSTRINGS = ("api_key", "apikey", "api-key", "secret_key",
                     "credential", "password", "private_key", "passwd")

# Files whose names look alarming but are known-good deliverables.
SECRET_ALLOWLIST = {".env.example"}


def is_secret(path: str) -> bool:
    name = os.path.basename(path).lower()
    if name in SECRET_ALLOWLIST:
        return False
    if name in SECRET_NAMES:
        return True
    if name.startswith(".env"):         # .env.local, .env.txt, ...
        return True
    stem = os.path.splitext(name)[0]
    if any(s in stem for s in SECRET_SUBSTRINGS):
        return True
    # "<something>_keys.txt" / "my keys.txt" -- a pasted credential dump, but NOT
    # a .ipynb or .py whose name merely mentions keys.
    ext = os.path.splitext(name)[1]
    return ext in {".txt", ".json", ".csv"} and re.search(r"(^|[_\- ])keys?$", stem) is not None


def collect(root: str, include_checkpoints: bool) -> List[str]:
    files: List[str] = []
    for entry in INCLUDE_TOP:
        path = os.path.join(root, entry)
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            if not is_secret(path):
                files.append(path)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in EXCLUDE_EXT:
                    continue
                if ext == ".pt" and not include_checkpoints:
                    continue
                full = os.path.join(dirpath, fn)
                if is_secret(full):
                    print(f"[submit] EXCLUDED (looks like a secret): "
                          f"{os.path.relpath(full, root)}")
                    continue
                files.append(full)
    return sorted(files)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the submission ZIP")
    ap.add_argument("--rollnumber", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--include-checkpoints", action="store_true",
                    help="also embed the .pt weights (large; they are on HF already)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    folder = f"{args.rollnumber}_assignment1"
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(root))
    zip_path = os.path.join(out_dir, f"{folder}.zip")

    files = collect(root, args.include_checkpoints)
    if not files:
        print("[submit] nothing to package -- is --root correct?")
        return 1

    missing = [n for n in ("README.md", "Report.pdf") if not os.path.exists(os.path.join(root, n))]

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arc = os.path.join(folder, os.path.relpath(path, root)).replace("\\", "/")
            zf.write(path, arc)

    # Final audit: re-open the archive and prove nothing sensitive slipped in.
    leaked = [n for n in zipfile.ZipFile(zip_path).namelist() if is_secret(n)]
    if leaked:
        os.remove(zip_path)
        print("[submit] ABORTED - secrets found in the archive, ZIP deleted:")
        for n in leaked:
            print("   ", n)
        return 2

    size_mb = os.path.getsize(zip_path) / (1024 ** 2)
    print(f"[submit] wrote {zip_path}  ({len(files)} files, {size_mb:.1f} MB)")
    print("[submit] secret audit: clean")
    if not args.include_checkpoints:
        print("[submit] .pt checkpoints excluded -- make sure the HF link is in README.md")
    if missing:
        print(f"[submit] WARNING: missing from the submission: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
