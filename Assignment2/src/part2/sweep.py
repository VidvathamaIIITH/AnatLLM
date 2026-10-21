"""Learning-rate sweep for the Part 2 optimizers (bonus deliverable).

Every optimizer family has its own natural learning-rate scale - Lion's sign
update and Muon's orthogonalised update are nothing like AdamW's - so comparing
them all at one shared learning rate would say more about the choice of that
constant than about the optimizers.  Each optimizer is therefore swept over a
small grid at a short budget (``sweep.fraction`` of the dataset) and the winner
by validation loss is the setting used for the full 1x run.

    python -m src.part2.sweep --optimizer adamw mars lion muon sophia
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from ..utils import (
    configure_stdout,
    deep_update,
    load_yaml,
    parse_overrides,
    save_json,
    work_dir,
)
from .optimizers import OPTIMIZERS
from .train import run


def sweep(cfg: dict, names: list[str], out_root: Path) -> dict:
    grid = cfg["sweep"]["grid"]
    fraction = float(cfg["sweep"].get("fraction", 0.1))
    results: dict[str, dict] = {}

    for name in names:
        lrs = grid.get(name)
        if not lrs:
            print(f"[sweep] no grid for {name}, skipping")
            continue
        trials = []
        for lr in lrs:
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg["train"]["fraction"] = fraction
            trial_cfg["train"]["save_checkpoint"] = False
            trial_cfg["eval"]["skip_bleu"] = True
            # Scale warmup with the shortened budget, otherwise a sweep run spends
            # half its steps warming up and its LR profile stops resembling the
            # full run it is meant to predict.
            full_warmup = int(cfg["train"].get("warmup_steps", 150))
            trial_cfg["train"]["warmup_steps"] = max(10, int(round(full_warmup * fraction)))
            trial_cfg["optim"]["per_optimizer"].setdefault(name, {})["lr"] = float(lr)
            tag = f"{name}_lr{lr:g}"
            trial_cfg["wandb_run"] = {"name": f"part2-sweep-{tag}", "group": "part2-lr-sweep",
                                      "job_type": "sweep", "tags": ["part2", "sweep", name]}
            print(f"\n[sweep] === {tag} ({fraction:.2f}x dataset) ===")
            summary = run(trial_cfg, name, out_root / "trials" / tag)
            trials.append({
                "lr": float(lr),
                "val_loss": summary["final_val_loss"],
                "tokens": summary["tokens_trained"],
                "seconds": summary["train_seconds"],
                "tokens_per_s": summary["tokens_per_s"],
                "state_ratio": summary["optimizer_state_ratio"],
            })
            save_json({"optimizer": name, "fraction": fraction, "trials": trials},
                      out_root / f"sweep_{name}.json")

        best = min(trials, key=lambda t: t["val_loss"])
        # A winner sitting on an endpoint means the grid never bracketed the
        # optimum: the result is only a lower bound on what this optimizer can
        # do, and since different optimizers would be handicapped by different
        # amounts, the cross-optimizer comparison would not be trustworthy.
        ladder = sorted(t["lr"] for t in trials)
        at_edge = len(ladder) > 1 and best["lr"] in (ladder[0], ladder[-1])
        results[name] = {"fraction": fraction, "trials": trials, "best_lr": best["lr"],
                         "best_val_loss": best["val_loss"],
                         "optimum_at_grid_edge": at_edge}
        if at_edge:
            print(f"[sweep] WARNING: {name} peaked at the edge of its grid "
                  f"({best['lr']:g}); widen sweep.grid.{name} and re-run.")
        print(f"[sweep] {name}: best lr = {best['lr']:g} "
              f"(val loss {best['val_loss']:.4f})")

    save_json(results, out_root / "sweep_results.json")
    print("\n[sweep] summary")
    print(f"{'optimizer':10s} {'best lr':>10s} {'val loss':>10s}  edge  grid")
    for name, r in results.items():
        grid_txt = "  ".join(f"{t['lr']:g}:{t['val_loss']:.4f}" for t in r["trials"])
        edge = "YES " if r.get("optimum_at_grid_edge") else " no "
        print(f"{name:10s} {r['best_lr']:10g} {r['best_val_loss']:10.4f}  {edge}  {grid_txt}")
    edged = [n for n, r in results.items() if r.get("optimum_at_grid_edge")]
    if edged:
        print(f"\n[sweep] {len(edged)} optimizer(s) peaked on a grid endpoint: "
              f"{', '.join(edged)}. Widen those grids before trusting the comparison.")
    return results


def main(argv: list[str] | None = None) -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description="Learning-rate sweep for Part 2")
    ap.add_argument("--config", default="configs/part2.yaml")
    ap.add_argument("--optimizer", nargs="+", default=sorted(OPTIMIZERS),
                    choices=sorted(OPTIMIZERS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    out_root = Path(args.out) if args.out else work_dir() / "part2" / "sweep"
    sweep(cfg, args.optimizer, out_root)


if __name__ == "__main__":  # pragma: no cover
    main()
