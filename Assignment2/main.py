"""Single entry point for ANLP Assignment 2.

    python main.py test                      # unit tests (~1 min, CPU is fine)
    python main.py smoke                     # every stage of all three parts on a tiny budget
    python main.py fetch                     # download the released checkpoints from the Hub
    python main.py part1 all                 # prepare -> train 5 variants -> eval -> plots
    python main.py part1 train --variant dense
    python main.py part1 eval                # evaluate checkpoints (e.g. after `fetch`)
    python main.py part2 all                 # prepare -> sweep -> train 5 optimizers -> plots
    python main.py part3 all                 # decode -> beam timing -> plots
    python main.py all                       # everything, in order (many GPU-hours)

Heavy artefacts (checkpoints, tokenizers, binarised corpora, HF caches) are
written to ``$ANLP_WORK_DIR`` (default ``./work``) - see the README.  Only
figures under ``assets/`` are written into the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.part1.model import VARIANTS  # noqa: E402
from src.part2.optimizers import OPTIMIZERS  # noqa: E402
from src.utils import configure_stdout, device_report, work_dir  # noqa: E402

CONFIGS = REPO / "configs"
HF_USER = "Viv0605101"


def _banner(text: str) -> None:
    print(f"\n{'=' * 78}\n== {text}\n{'=' * 78}", flush=True)


def _out(args: argparse.Namespace) -> list[str]:
    """``--out`` for the plot stages; empty means the repository's assets/."""
    out = getattr(args, "assets", None)
    return ["--out", str(out)] if out else []


# --------------------------------------------------------------------------- #
# Part 1
# --------------------------------------------------------------------------- #


def part1(args: argparse.Namespace) -> None:
    from src.part1 import data as p1data
    from src.part1 import plots as p1plots
    from src.part1 import run_eval as p1eval
    from src.part1 import train as p1train
    from src.utils import deep_update, load_yaml, parse_overrides

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    variants = args.variant or list(VARIANTS)
    runs = Path(args.runs) if args.runs else work_dir() / "part1" / "runs"
    stage = args.stage

    if stage in ("prepare", "all"):
        _banner("Part 1 - preparing data")
        p1data.prepare(vocab_size=cfg["data"]["vocab_size"], n_ctx=cfg["data"]["n_ctx"],
                       **getattr(args, "prepare_kwargs", {}))

    if stage in ("train", "all"):
        for v in variants:
            _banner(f"Part 1 - training {v}")
            p1train.run(cfg, v, runs)

    if stage in ("eval", "all"):
        _banner("Part 1 - evaluating")
        overrides = ["--set"] + args.set if args.set else []
        if args.variant:
            for v in args.variant:
                p1eval.main(["--config", str(args.config), "--variant", v,
                             "--runs", str(runs)] + overrides)
        else:
            p1eval.main(["--config", str(args.config), "--all", "--runs", str(runs)]
                        + overrides)

    if stage in ("plots", "all"):
        _banner("Part 1 - figures")
        p1plots.main(["--runs", str(runs)] + _out(args))


# --------------------------------------------------------------------------- #
# Part 2
# --------------------------------------------------------------------------- #


def part2(args: argparse.Namespace) -> None:
    from src.part2 import data as p2data
    from src.part2 import plots as p2plots
    from src.part2 import sweep as p2sweep
    from src.part2 import train as p2train
    from src.utils import deep_update, load_yaml, parse_overrides

    cfg = deep_update(load_yaml(args.config), parse_overrides(args.set))
    names = args.optimizer or ["adamw", "mars", "lion", "muon", "sophia"]
    runs = Path(args.runs) if args.runs else work_dir() / "part2" / "runs"
    sweep_dir = work_dir() / "part2" / "sweep"
    stage = args.stage

    if stage in ("prepare", "all"):
        _banner("Part 2 - preparing data")
        p2data.prepare(vocab_size=cfg["data"]["vocab_size"], n_ctx=cfg["data"]["n_ctx"])

    if stage in ("sweep", "all"):
        _banner("Part 2 - learning-rate sweep")
        results = p2sweep.sweep(cfg, getattr(args, "sweep_optimizers", None) or names, sweep_dir)
        # Adopt each winner for the full-length runs.
        for name, r in results.items():
            cfg["optim"]["per_optimizer"].setdefault(name, {})["lr"] = r["best_lr"]

    if stage in ("train", "all"):
        for name in names:
            _banner(f"Part 2 - training with {name}")
            p2train.run(cfg, name, runs)

    if stage in ("plots", "all"):
        _banner("Part 2 - figures")
        p2plots.main(["--runs", str(runs), "--sweep", str(sweep_dir)] + _out(args))


# --------------------------------------------------------------------------- #
# Part 3
# --------------------------------------------------------------------------- #


def part3(args: argparse.Namespace) -> None:
    from src.part3 import plots as p3plots
    from src.part3 import run as p3run
    from src.part3 import timing as p3timing

    stage = args.stage
    extra = ["--config", str(args.config)] + (["--set"] + args.set if args.set else [])

    if stage in ("run", "all"):
        _banner("Part 3 - decoding strategies")
        p3run.main(extra)

    if stage in ("timing", "all"):
        _banner("Part 3 - beam-search timing")
        p3timing.main(extra)

    if stage in ("plots", "all"):
        _banner("Part 3 - figures")
        p3plots.main(_out(args))


# --------------------------------------------------------------------------- #
# Smoke test: every stage of every part, on a tiny budget
# --------------------------------------------------------------------------- #

SMOKE_PART1 = [
    "data.vocab_size=2000", "train.token_budget=120000", "train.warmup_steps=5",
    "train.log_every=5", "train.eval_every=10", "train.eval_examples=200",
    "eval.ppl_examples=300", "eval.bleu_examples=16", "eval.usage_examples=200",
    "eval.max_new_tokens=24", "eval.gen_batch_size=16",
]
SMOKE_PART2 = [
    "data.vocab_size=2000", "train.fraction=0.01", "train.warmup_steps=5",
    "train.log_every=5", "eval.val_blocks=40", "eval.bleu_prompts=8",
    "eval.gen_tokens=8", "eval.gen_batch_size=8",
    "sweep.fraction=0.005", "sweep.grid.adamw=[6.0e-4, 1.2e-3]",
]
SMOKE_PART3 = [
    "data.n_examples=16", "generate.max_new_tokens=16", "generate.beam_widths=[1, 2]",
    "generate.top_k=[10]", "generate.top_p=[0.9]", "generate.batch_size=8",
    "timing.n_examples=8", "timing.beam_widths=[1, 2]", "timing.repeats=1",
    "timing.batch_size=4",
]


def smoke(_: argparse.Namespace) -> None:
    """Run the whole pipeline end to end in a few minutes.

    Everything goes to ``<work>/smoke`` (figures to ``<work>/smoke/assets``), so
    the real results and the repository's figures are never touched.  WandB is
    switched off.  Downloaded datasets are shared with the main work directory.
    """
    base = work_dir()
    smoke_dir = base / "smoke"
    os.environ["ANLP_HF_CACHE"] = str(base / "hf_cache")
    os.environ["ANLP_WORK_DIR"] = str(smoke_dir)
    os.environ["WANDB_MODE"] = "disabled"
    assets = smoke_dir / "assets"
    t0 = time.perf_counter()

    part1(argparse.Namespace(
        stage="all", config=CONFIGS / "part1.yaml", variant=None, runs=None,
        set=SMOKE_PART1, assets=assets,
        prepare_kwargs={"max_train_rows": 4000, "tokenizer_rows": 4000}))
    part2(argparse.Namespace(
        stage="all", config=CONFIGS / "part2.yaml", optimizer=None, runs=None,
        set=SMOKE_PART2, assets=assets, sweep_optimizers=["adamw"]))
    part3(argparse.Namespace(stage="all", config=CONFIGS / "part3.yaml",
                             set=SMOKE_PART3, assets=assets))

    _banner(f"Smoke test passed in {(time.perf_counter() - t0) / 60:.1f} min - "
            f"outputs in {smoke_dir}")


# --------------------------------------------------------------------------- #
# Fetch the released checkpoints
# --------------------------------------------------------------------------- #


def fetch(args: argparse.Namespace) -> None:
    """Download the released artefacts into the work-directory layout.

    Afterwards ``main.py part1 eval`` re-scores the Part 1 checkpoints and the
    ``plots`` stages redraw every figure, without any training.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    root = work_dir()
    user = args.user
    keep = ["*.json", "*.jsonl", "best.pt"] + (["final.pt"] if args.all_checkpoints else [])

    _banner("Part 1 - MoE checkpoints")
    hf_hub_download(f"{user}/anlp-a2-part1-moe", "tokenizer.json", local_dir=root / "part1")
    snapshot_download(f"{user}/anlp-a2-part1-moe", local_dir=root / "part1" / "runs",
                      allow_patterns=[f"{v}/{p}" for v in VARIANTS for p in keep])

    _banner("Part 2 - optimizer checkpoints")
    hf_hub_download(f"{user}/anlp-a2-part2-optimizers", "tokenizer.json",
                    local_dir=root / "part2")
    snapshot_download(f"{user}/anlp-a2-part2-optimizers", local_dir=root / "part2" / "runs",
                      allow_patterns=[f"{o}/{p}" for o in OPTIMIZERS
                                      for p in ("final.pt", "summary.json", "train_log.jsonl")])
    snapshot_download(f"{user}/anlp-a2-part2-optimizers", local_dir=root / "part2" / "sweep",
                      allow_patterns=["lr_sweep/*.json"])
    for p in (root / "part2" / "sweep" / "lr_sweep").glob("*.json"):
        p.replace(root / "part2" / "sweep" / p.name)

    _banner("Part 3 - raw generations and results")
    snapshot_download(f"{user}/anlp-a2-part3-generations", repo_type="dataset",
                      local_dir=root / "part3",
                      allow_patterns=["generations/*.jsonl", "results.json", "beam_timing.json"])

    print(f"\n[fetch] done - artefacts are under {root}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(
        description="ANLP Assignment 2 - MoE, optimizers, decoding strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="part", required=True)

    p1 = sub.add_parser("part1", help="Mixture of Experts")
    p1.add_argument("stage", choices=["prepare", "train", "eval", "plots", "all"])
    p1.add_argument("--config", default=str(CONFIGS / "part1.yaml"))
    p1.add_argument("--variant", nargs="*", choices=list(VARIANTS))
    p1.add_argument("--runs", default=None)
    p1.add_argument("--assets", default=None, help="figure directory (default: assets/)")
    p1.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p1.set_defaults(fn=part1)

    p2 = sub.add_parser("part2", help="Optimizers")
    p2.add_argument("stage", choices=["prepare", "sweep", "train", "plots", "all"])
    p2.add_argument("--config", default=str(CONFIGS / "part2.yaml"))
    p2.add_argument("--optimizer", nargs="*", choices=sorted(OPTIMIZERS))
    p2.add_argument("--runs", default=None)
    p2.add_argument("--assets", default=None, help="figure directory (default: assets/)")
    p2.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p2.set_defaults(fn=part2)

    p3 = sub.add_parser("part3", help="Decoding strategies")
    p3.add_argument("stage", choices=["run", "timing", "plots", "all"])
    p3.add_argument("--config", default=str(CONFIGS / "part3.yaml"))
    p3.add_argument("--assets", default=None, help="figure directory (default: assets/)")
    p3.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p3.set_defaults(fn=part3)

    sp = sub.add_parser("smoke", help="every stage of every part on a tiny budget")
    sp.set_defaults(fn=smoke)

    fp = sub.add_parser("fetch", help="download the released checkpoints from the Hub")
    fp.add_argument("--user", default=HF_USER)
    fp.add_argument("--all-checkpoints", action="store_true",
                    help="also download Part 1 final.pt (best.pt is what was evaluated)")
    fp.set_defaults(fn=fetch)

    allp = sub.add_parser("all", help="run every part end to end")
    allp.set_defaults(fn=None)

    tp = sub.add_parser("test", help="run the unit tests")
    tp.set_defaults(fn=None)

    args = ap.parse_args()
    print(f"[main] device: {device_report()}")
    print(f"[main] work dir: {work_dir()}")

    if args.part == "test":
        import pytest

        # pyproject.toml already adds -q; a second -q would hide the "N passed" summary.
        raise SystemExit(pytest.main([str(REPO / "tests")]))

    if args.part == "all":
        part1(argparse.Namespace(stage="all", config=CONFIGS / "part1.yaml",
                                 variant=None, runs=None, set=[]))
        part2(argparse.Namespace(stage="all", config=CONFIGS / "part2.yaml",
                                 optimizer=None, runs=None, set=[]))
        part3(argparse.Namespace(stage="all", config=CONFIGS / "part3.yaml", set=[]))
        return

    args.fn(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
