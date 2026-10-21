"""Byte-level BPE tokenizer training and a thin runtime wrapper.

Both Part 1 (en / vi / ja translation) and Part 2 (English pre-training) train
their own tokenizer with this module.  A *byte-level* BPE is used deliberately:
the Vietnamese text is diacritic-heavy and the Japanese text is not romanised
and has no whitespace token boundaries, so a byte-level alphabet is the only
pre-tokenisation that covers all three scripts without ever emitting ``<unk>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Special tokens shared by both parts.  Order fixes the ids, so keep it stable.
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
BASE_SPECIALS: tuple[str, ...] = (PAD, BOS, EOS, UNK)

# Part 1 language control tokens.
SRC_VI, SRC_JA, TGT_EN = "<src_vi>", "<src_ja>", "<tgt_en>"
TRANSLATION_SPECIALS: tuple[str, ...] = BASE_SPECIALS + (SRC_VI, SRC_JA, TGT_EN)


def train_bpe(
    corpus: Iterable[str],
    vocab_size: int,
    out_path: str | Path,
    specials: Sequence[str] = BASE_SPECIALS,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer and write it to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=list(specials),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(str(out_path))
    return tok


class BPETokenizer:
    """Runtime wrapper exposing the handful of operations the scripts need."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.tok = Tokenizer.from_file(self.path)
        self._ids: dict[str, int] = {}
        for name in TRANSLATION_SPECIALS:
            tid = self.tok.token_to_id(name)
            if tid is not None:
                self._ids[name] = tid

    # -- ids ----------------------------------------------------------------- #

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._ids[PAD]

    @property
    def bos_id(self) -> int:
        return self._ids[BOS]

    @property
    def eos_id(self) -> int:
        return self._ids[EOS]

    def special_id(self, token: str) -> int:
        return self._ids[token]

    # -- encode / decode ------------------------------------------------------ #

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def decode_batch(self, seqs: Sequence[Sequence[int]], skip_special: bool = True) -> list[str]:
        return self.tok.decode_batch([list(s) for s in seqs], skip_special_tokens=skip_special)


def batched(iterable: Iterable[str], size: int) -> Iterator[list[str]]:
    """Yield lists of at most ``size`` items (used to feed the BPE trainer)."""
    buf: list[str] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
