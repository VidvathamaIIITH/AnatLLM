"""Publish the logs of already-finished runs to Weights & Biases.

Training writes every train/eval event to ``train_log.jsonl`` next to the
checkpoint (and, with ``wandb.enabled: true``, mirrors it live).  This script
replays those JSONL logs for runs that were trained before live logging was
switched on, so every checkpoint has a matching WandB run:

    part1-<variant>           five Part 1 FFN ablations      (group part1-moe)
    part2-<optimizer>         five Part 2 optimizer runs     (group part2-optimizers)
    part2-sweep-<opt>_lr<lr>  25 learning-rate sweep trials  (group part2-lr-sweep)
    part3-beam-timing         beam-width timing study        (group part3-decoding)

Runs that already exist in the project are skipped, so it is safe to re-run.

    python scripts/sync_wandb.py [--project anlp-a2] [--only part1 part2 sweep part3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.part1.model import VARIANT_LABELS, VARIANTS  # noqa: E402
from src.part2.optimizers import OPTIMIZER_CATEGORY, OPTIMIZER_LABEL  # noqa: E402
from src.utils import (  # noqa: E402
    _scalars,
    configure_stdout,
    init_wandb,
    load_json,
    read_jsonl,
    wandb_log_record,
    work_dir,
)

NOTE = "Replayed from the train_log.jsonl written during training."


def last_segment(rows: list[dict]) -> list[dict]:
    """Keep only the final run when a log file holds several appended runs.

    ``JsonlLogger`` appends, so re-running a configuration into the same folder
    leaves the earlier attempt in front of it; a drop in ``step`` marks the seam.
    """
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["step"] < rows[i - 1]["step"]:
            start = i
    return rows[start:]


class Publisher:
    def __init__(self, project: str, entity: str | None):
        import wandb

        self.cfg = {"wandb": {"enabled": True, "project": project, "entity": entity}}
        api = wandb.Api()
        path = f"{entity or api.default_entity}/{project}"
        try:
            self.existing = {r.name for r in api.runs(path)}
        except (ValueError, wandb.errors.CommError):  # project not created yet
            self.existing = set()
        self.path = path

    def publish(self, name: str, group: str, job_type: str, tags: list[str], config: dict,
                log: Path | None, summary: dict, images: dict[str, Path] | None = None,
                notes: str = NOTE) -> None:
        if name in self.existing:
            print(f"[wandb] {name}: already in {self.path}, skipping")
            return
        import wandb

        run = init_wandb(self.cfg, name=name, group=group, job_type=job_type,
                         tags=tags, config=config, notes=notes)
        if run is None:
            raise SystemExit("[wandb] could not start a run - is `wandb login` done?")
        n = 0
        if log is not None and log.exists():
            for rec in last_segment(read_jsonl(log)):
                wandb_log_record(run, rec)
                n += 1
        for key, path in (images or {}).items():
            if path.exists():
                run.log({key: wandb.Image(str(path))})
        run.summary.update(summary)
        url = run.url
        run.finish()
        self.existing.add(name)
        print(f"[wandb] {name}: {n} records -> {url}")


def part1(pub: Publisher, root: Path) -> None:
    for v in VARIANTS:
        d = root / "part1" / "runs" / v
        if not (d / "train_log.jsonl").exists():
            print(f"[wandb] part1 {v}: no log, skipping")
            continue
        run_cfg = load_json(d / "run_config.json")
        summary = {k: val for k, val in load_json(d / "summary.json").items() if k != "history"}
        if (d / "eval.json").exists():
            summary.update(_scalars(load_json(d / "eval.json"), "test"))
        images = {"expert_usage": REPO / "assets" / f"part1_experts_{v}.png"}
        pub.publish(f"part1-{v}", "part1-moe", "train", ["part1", v], run_cfg,
                    d / "train_log.jsonl", summary, images,
                    notes=f"{VARIANT_LABELS[v]}. {NOTE}")


def part2(pub: Publisher, root: Path) -> None:
    for d in sorted((root / "part2" / "runs").iterdir()):
        if not (d / "train_log.jsonl").exists():
            continue
        s = load_json(d / "summary.json")
        cfg = {"optimizer": d.name, "category": OPTIMIZER_CATEGORY[d.name],
               "hyperparams": s["hyperparams"], "params": s["params"], "fraction": 1.0}
        summary = {k: v for k, v in s.items() if k != "curve"}
        pub.publish(f"part2-{d.name}", "part2-optimizers", "train", ["part2", d.name], cfg,
                    d / "train_log.jsonl", summary,
                    notes=f"{OPTIMIZER_LABEL[d.name]}. {NOTE}")


def sweep(pub: Publisher, root: Path) -> None:
    sweep_dir = root / "part2" / "sweep"
    results = load_json(sweep_dir / "sweep_results.json")
    for opt, r in results.items():
        for t in r["trials"]:
            tag = f"{opt}_lr{t['lr']:g}"
            log = sweep_dir / "trials" / tag / opt / "train_log.jsonl"
            summary = {**t, "best_in_grid": t["lr"] == r["best_lr"]}
            pub.publish(f"part2-sweep-{tag}", "part2-lr-sweep", "sweep",
                        ["part2", "sweep", opt], {"optimizer": opt, "lr": t["lr"],
                                                  "fraction": r["fraction"]},
                        log, summary)


def part3(pub: Publisher, root: Path) -> None:
    timing = root / "part3" / "beam_timing.json"
    if not timing.exists():
        print("[wandb] part3: no beam_timing.json, skipping")
        return
    import wandb

    t = load_json(timing)
    cols = ["beam_width", "seconds_best", "seconds_mean", "seconds_per_example",
            "peak_mib", "forward_calls", "speed_ratio_vs_width1"]
    summary = {"timing": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in t["rows"]]),
               "device": t["device"]}
    for r in t["rows"]:
        summary[f"beam{r['beam_width']}/seconds_best"] = r["seconds_best"]
        summary[f"beam{r['beam_width']}/peak_mib"] = r["peak_mib"]
    pub.publish("part3-beam-timing", "part3-decoding", "timing", ["part3", "timing"],
                {k: v for k, v in t.items() if k != "rows"}, None, summary,
                {"beam_timing": REPO / "assets" / "part3_beam_timing.png"},
                notes="Wall-clock and memory of beam search at widths 1, 2, 4, 8.")


def main() -> None:
    configure_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default="anlp-a2")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--only", nargs="+", default=["part1", "part2", "sweep", "part3"],
                    choices=["part1", "part2", "sweep", "part3"])
    args = ap.parse_args()

    pub = Publisher(args.project, args.entity)
    root = work_dir()
    for step in args.only:
        {"part1": part1, "part2": part2, "sweep": sweep, "part3": part3}[step](pub, root)
    print(f"[wandb] project: https://wandb.ai/{pub.path}")


if __name__ == "__main__":
    main()
