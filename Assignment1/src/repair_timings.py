"""Recompute wall-clock timing metrics robustly, uniformly, across all configs.

Why this exists
---------------
``sec_per_epoch`` was originally the *mean* of per-epoch wall-clock times. Wall
clock on a laptop is not the same as compute time: if the host suspends mid-run,
the epoch that was in flight absorbs the entire sleep. That happened once here
(one C4 epoch recorded 24{,}890 s, of which roughly 127 s was actual compute),
which destroys the mean and every quantity derived from it.

The median is immune to a small number of such outliers, and when there is no
interruption it agrees with the mean to within ~2% (verified on C1--C3 below).
This script rewrites the timing fields of every ``metrics.json`` from the
per-epoch series in ``history.json`` so that all five configurations are
measured identically, regardless of which code version produced them.

It is idempotent and never touches accuracy metrics.

    python -m src.repair_timings                 # repair outputs/
    python -m src.repair_timings --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Optional, Sequence

CONFIGS = ["C1", "C2", "C3", "C4", "C5"]


def repair_one(cfg_dir: str, dry_run: bool = False) -> Optional[dict]:
    mpath = os.path.join(cfg_dir, "metrics.json")
    hpath = os.path.join(cfg_dir, "history.json")
    if not (os.path.exists(mpath) and os.path.exists(hpath)):
        return None

    metrics = json.load(open(mpath, encoding="utf-8"))
    history = json.load(open(hpath, encoding="utf-8"))
    times = [float(t) for t in history.get("sec_per_epoch", []) if t]
    if not times:
        return None

    median = statistics.median(times)
    mean = sum(times) / len(times)

    # Recover the total token count from whichever convention wrote this file.
    if "train_tokens_per_sec_raw_total" in metrics:
        total_tokens = metrics["train_tokens_per_sec_raw_total"]
    elif metrics.get("_timing_repaired"):
        total_tokens = metrics["train_tokens_per_sec"] * median * len(times)
    else:
        # original convention: tokens_per_sec = total_tokens / sum(times)
        total_tokens = metrics.get("train_tokens_per_sec", 0.0) * sum(times)

    before = metrics.get("sec_per_epoch")
    metrics["sec_per_epoch"] = median
    metrics["sec_per_epoch_mean"] = mean
    metrics["sec_per_epoch_max"] = max(times)
    metrics["total_train_seconds"] = sum(times)
    metrics["train_tokens_per_sec"] = (
        (total_tokens / len(times)) / median if median > 0 else 0.0
    )
    metrics["train_tokens_per_sec_raw_total"] = total_tokens
    metrics["_timing_repaired"] = True
    # How badly wall clock diverged from compute for this run:
    metrics["_timing_outlier_ratio"] = (mean / median) if median > 0 else 0.0

    if not dry_run:
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return {
        "n_epochs": len(times), "before": before, "median": median,
        "mean": mean, "max": max(times),
        "contaminated": mean > 1.10 * median,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    print(f"{'cfg':<5} {'n':>3} {'was':>10} {'median':>9} {'mean':>10} "
          f"{'max':>10}   status")
    any_fixed = False
    results = {}
    for c in CONFIGS:
        r = repair_one(os.path.join(a.out_dir, c), a.dry_run)
        if r is None:
            print(f"{c:<5}   -  (no metrics/history yet)")
            continue
        results[c] = r
        status = "CONTAMINATED -> repaired" if r["contaminated"] else "clean"
        any_fixed = any_fixed or r["contaminated"]
        print(f"{c:<5} {r['n_epochs']:>3} {r['before']:>10.1f} {r['median']:>9.1f} "
              f"{r['mean']:>10.1f} {r['max']:>10.1f}   {status}")

    # Refresh the aggregate so downstream tables pick the repaired values up.
    agg = os.path.join(a.out_dir, "all_results.json")
    if not a.dry_run and results:
        merged = {}
        for c in CONFIGS:
            p = os.path.join(a.out_dir, c, "metrics.json")
            if os.path.exists(p):
                merged[c] = json.load(open(p, encoding="utf-8"))
        if merged:
            with open(agg, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            print(f"\nrewrote {agg} ({len(merged)} configs)")

    if a.dry_run:
        print("\n--dry-run: nothing written")
    elif not any_fixed:
        print("\nAll runs were clean; values changed only from mean to median.")
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
