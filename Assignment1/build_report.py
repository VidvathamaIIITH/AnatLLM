"""Rebuild Report.pdf from the training outputs, in the one correct order.

    python build_report.py

The ordering matters and is easy to get wrong by hand -- doing these steps out of
sequence silently bakes stale numbers into the PDF:

  1. src.repair_timings      normalise wall-clock timings in every metrics.json
                             (must run AFTER training: train.py rewrites
                             metrics.json when a run finishes)
  2. src.run_all             rebuild all_results.json + summary plots from the
     --aggregate-only        repaired metrics.json files
  3. report/make_tables.py   regenerate results_tables.tex + results_macros.tex
  4. copy figures            outputs/*.png -> report/figures/
  5. pdflatex/bibtex/pdflatex/pdflatex

Finally the PDF is copied to ./Report.pdf, where make_submission.py expects it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(ROOT, "report")
FIGS = os.path.join(REPORT, "figures")
PY = sys.executable


def run(cmd, cwd=ROOT, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if not quiet and r.stdout:
        print(r.stdout.rstrip())
    if r.returncode != 0:
        print(f"  !! exit {r.returncode}: {' '.join(cmd[:4])}...")
        if r.stderr:
            print("  " + r.stderr.strip()[:500])
    return r.returncode


def main() -> int:
    print("=" * 70)
    print("[1/5] normalising timings")
    print("=" * 70)
    run([PY, "-m", "src.repair_timings"])

    print("\n" + "=" * 70)
    print("[2/5] aggregating results + summary plots")
    print("=" * 70)
    run([PY, "-m", "src.run_all", "--aggregate-only", "--out-dir", "outputs"],
        quiet=True)
    # repair rewrote metrics.json; aggregation re-reads them, so all_results.json
    # is now consistent. Re-run repair to be safe if aggregation reordered keys.
    run([PY, "-m", "src.repair_timings"], quiet=True)

    print("\n" + "=" * 70)
    print("[3/5] regenerating LaTeX tables and macros")
    print("=" * 70)
    run([PY, os.path.join("report", "make_tables.py")])

    print("\n" + "=" * 70)
    print("[4/5] copying figures")
    print("=" * 70)
    os.makedirs(FIGS, exist_ok=True)
    n = 0
    for name in ("ablation_quality.png", "ablation_efficiency.png"):
        src = os.path.join(ROOT, "outputs", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FIGS, name))
            print(f"  {name}")
            n += 1
    for cfg in ("C1", "C2", "C3", "C4", "C5"):
        src = os.path.join(ROOT, "outputs", cfg, f"{cfg}_curves.png")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FIGS, f"{cfg}_curves.png"))
            n += 1
    print(f"  {n} figures -> report/figures/")

    print("\n" + "=" * 70)
    print("[5/5] compiling Report.pdf")
    print("=" * 70)
    for step in (["pdflatex", "-interaction=nonstopmode", "Report.tex"],
                 ["bibtex", "Report"],
                 ["pdflatex", "-interaction=nonstopmode", "Report.tex"],
                 ["pdflatex", "-interaction=nonstopmode", "Report.tex"]):
        run(step, cwd=REPORT, quiet=True)

    pdf = os.path.join(REPORT, "Report.pdf")
    if not os.path.exists(pdf):
        print("  FAILED: no PDF produced. See report/Report.log")
        return 1

    shutil.copy2(pdf, os.path.join(ROOT, "Report.pdf"))

    # page count + a check that no macro leaked through unresolved
    log = open(os.path.join(REPORT, "Report.log"), encoding="utf-8",
               errors="replace").read()
    import re
    pages = re.search(r"Output written on Report\.pdf \((\d+) page", log)
    print(f"\n  pages          : {pages.group(1) if pages else '?'}")
    for pat, label in [(r"Citation .* undefined", "undefined citations"),
                       (r"Reference .* undefined", "undefined references"),
                       (r"Undefined control sequence", "undefined macros")]:
        hits = len(re.findall(pat, log))
        print(f"  {label:15s}: {hits}" + ("  <-- FIX THIS" if hits else ""))
    over = len(re.findall(r"Overfull \\hbox", log))
    print(f"  overfull hboxes: {over}")
    print(f"\n  wrote {os.path.join(ROOT, 'Report.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
