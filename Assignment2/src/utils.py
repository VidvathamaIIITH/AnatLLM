"""Shared helpers used across all three parts of the assignment.

Nothing here is task specific: seeding, device selection, config loading,
parameter accounting, checkpoint IO, timing and small logging utilities.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import yaml

# --------------------------------------------------------------------------- #
# Reproducibility / devices
# --------------------------------------------------------------------------- #


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console defaults to cp1252, which raises ``UnicodeEncodeError``
    the moment a Vietnamese or Japanese sample is printed.  Every entry point
    calls this first.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - detached stream
                pass


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so runs are comparable across ablations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 on Ampere+, fp16 on older CUDA cards, no autocast on CPU."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def device_report() -> str:
    if not torch.cuda.is_available():
        return "cpu only"
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return (
        f"{props.name} | {props.total_memory / 1024**3:.1f} GiB | "
        f"sm_{props.major}{props.minor} | torch {torch.__version__}"
    )


# --------------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------------- #


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict, other: dict) -> dict:
    """Recursively merge ``other`` into a copy of ``base``."""
    out = dict(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """Turn ``["train.lr=3e-4", "model.n_layers=8"]`` into a nested dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override {pair!r} is not of the form key=value")
        key, raw = pair.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.2f}T"


# --------------------------------------------------------------------------- #
# Timing / logging
# --------------------------------------------------------------------------- #


@contextmanager
def timer(name: str, sink: dict | None = None) -> Iterator[None]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if sink is not None:
        sink[name] = dt
    else:
        print(f"[timer] {name}: {dt:.3f}s")


def init_wandb(cfg: dict, *, name: str, group: str, job_type: str,
               tags: Iterable[str] = (), config: Any = None, notes: str | None = None,
               **init_kwargs: Any):
    """Start a WandB run if ``cfg["wandb"]["enabled"]``; otherwise return ``None``.

    Logging is strictly best-effort: a missing package or a failed login prints a
    warning and training carries on with the local JSONL log only.
    """
    wcfg = cfg.get("wandb") or {}
    mode = os.environ.get("WANDB_MODE", "")
    if not wcfg.get("enabled", False) or mode == "disabled":
        return None
    if mode != "offline" and not _wandb_logged_in():
        # wandb.init would otherwise stop and wait for an interactive login.
        print("[wandb] no WandB login found - logging to JSONL only "
              "(run `wandb login` or set WANDB_MODE=offline to enable)")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed - logging to JSONL only")
        return None
    try:
        run = wandb.init(
            project=wcfg.get("project", "anlp-a2"), entity=wcfg.get("entity"),
            name=name, group=group, job_type=job_type, tags=list(tags),
            notes=notes, config=to_jsonable(config if config is not None else cfg),
            reinit="finish_previous", **init_kwargs,
        )
    except Exception as exc:  # pragma: no cover - network / auth failures
        print(f"[wandb] init failed ({type(exc).__name__}: {exc}) - logging to JSONL only")
        return None
    # Plot every curve against tokens seen, which is what the analysis compares on.
    run.define_metric("tokens")
    for prefix in ("train/*", "eval/*"):
        run.define_metric(prefix, step_metric="tokens")
    return run


def _wandb_logged_in() -> bool:
    """True if an API key is available from the environment or a netrc file."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    import netrc

    for name in (".netrc", "_netrc"):
        path = Path(os.environ.get("NETRC", Path.home() / name))
        try:
            if path.exists() and netrc.netrc(str(path)).authenticators("api.wandb.ai"):
                return True
        except (netrc.NetrcParseError, OSError):
            continue
    return False


def _scalars(record: dict, prefix: str) -> dict[str, float]:
    """Flatten the numeric fields of ``record`` into ``prefix/key[/subkey]``."""
    out: dict[str, float] = {}
    for k, v in record.items():
        if k in ("event", "step", "tokens"):
            continue
        if isinstance(v, dict):
            out.update(_scalars(v, f"{prefix}/{k}"))
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[f"{prefix}/{k}"] = float(v)
    return out


def wandb_log_record(run: Any, record: dict) -> None:
    """Mirror one JSONL training/eval record to a WandB run."""
    payload = _scalars(record, record.get("event", "train"))
    if "tokens" in record:
        payload["tokens"] = float(record["tokens"])
    run.log(payload, step=int(record.get("step", 0)))


class JsonlLogger:
    """Append-only JSONL log; one dict per training/eval event.

    If a WandB run is attached, every numeric field is mirrored to it as
    ``<event>/<field>`` (e.g. ``train/lm_loss``, ``eval/val_loss``).
    """

    def __init__(self, path: str | Path, wandb_run: Any = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.wandb_run = wandb_run

    def log(self, **record: Any) -> None:
        self._fh.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
        self._fh.flush()
        if self.wandb_run is not None:
            wandb_log_record(self.wandb_run, record)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #


def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to ``min_ratio * base_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * coeff)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def work_dir() -> Path:
    """Heavy artefacts (checkpoints, tokenizers, binarised data) live outside
    the git repository, which has a strict storage quota."""
    env = os.environ.get("ANLP_WORK_DIR")
    root = Path(env) if env else repo_root() / "work"
    root.mkdir(parents=True, exist_ok=True)
    return root


def hf_cache_dir() -> str:
    """HuggingFace dataset/model cache; ``$ANLP_HF_CACHE`` overrides the default
    ``<work>/hf_cache`` so separate work directories can share one download."""
    env = os.environ.get("ANLP_HF_CACHE")
    d = Path(env) if env else work_dir() / "hf_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
