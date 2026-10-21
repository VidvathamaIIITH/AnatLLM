"""Data for Part 3: ROCStories continuations.

``hamishivi/ROCStories`` provides ``prompt`` (the opening sentence of a
five-sentence story), ``continuation`` (the remaining four sentences) and
``text`` (both concatenated).  The decoding strategies are asked to reproduce
the continuation from the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils import hf_cache_dir, work_dir

DATASET_ID = "hamishivi/ROCStories"


def part3_dir() -> Path:
    d = work_dir() / "part3"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Example:
    prompt: str
    reference: str


def load_examples(split: str = "test", n: int = 1000, seed: int = 1234) -> list[Example]:
    """Load a deterministic random subset of ``split`` holding exactly ``n`` usable stories.

    A fixed seed means every decoding strategy is scored on exactly the same
    stories, which is what makes the comparison meaningful.

    Rows with an empty prompt or continuation cannot be scored, so the sample is
    drawn from a shuffled permutation and consumed until ``n`` usable stories
    have been collected.  Filtering *after* taking the first ``n`` would quietly
    return fewer than requested - the assignment sets 1000 as a minimum, not a
    target to fall short of.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, cache_dir=hf_cache_dir())
    order = np.arange(len(ds))
    if n is not None:
        np.random.default_rng(seed).shuffle(order)

    out: list[Example] = []
    for i in order.tolist():
        row = ds[i]
        prompt = (row.get("prompt") or "").strip()
        cont = (row.get("continuation") or "").strip()
        if prompt and cont:
            out.append(Example(prompt=prompt, reference=cont))
            if n is not None and len(out) >= n:
                break
    if n is not None and len(out) < n:
        raise ValueError(
            f"{DATASET_ID}:{split} yielded only {len(out)} usable stories, fewer than "
            f"the {n} requested"
        )
    return out
