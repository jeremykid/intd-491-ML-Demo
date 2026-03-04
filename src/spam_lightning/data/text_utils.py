"""Text tokenization and vocabulary utilities."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+")


def regex_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize *text* into alphanumeric tokens (keeps contractions).

    Returns an empty list when the input contains no matchable tokens.
    """
    tokens = _TOKEN_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocab:
    """A minimal string ↔ index vocabulary with special tokens."""

    token_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_token: dict[int, str] = field(default_factory=dict)
    pad_index: int = 0
    unk_index: int = 1

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def lookup_index(self, token: str) -> int:
        """Return the index for *token*, falling back to ``unk_index``."""
        return self.token_to_idx.get(token, self.unk_index)

    def lookup_token(self, index: int) -> str:
        """Return the token for *index*, falling back to ``<unk>``."""
        return self.idx_to_token.get(index, UNK_TOKEN)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "token_to_idx": self.token_to_idx,
            "pad_index": self.pad_index,
            "unk_index": self.unk_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        token_to_idx: dict[str, int] = data["token_to_idx"]
        idx_to_token = {v: k for k, v in token_to_idx.items()}
        return cls(
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            pad_index=data.get("pad_index", 0),
            unk_index=data.get("unk_index", 1),
        )


def build_vocab(
    sequences: Sequence[list[str]],
    min_freq: int = 2,
    max_size: Optional[int] = None,
) -> Vocab:
    """Build a :class:`Vocab` from an iterable of token lists.

    Tokens appearing fewer than *min_freq* times are excluded.
    """
    counter: Counter[str] = Counter()
    for seq in sequences:
        counter.update(seq)

    # Start with special tokens
    token_to_idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    idx = 2

    # Sort by frequency (descending), then alphabetically for determinism
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if token in token_to_idx:
            continue
        token_to_idx[token] = idx
        idx += 1
        if max_size is not None and idx >= max_size:
            break

    idx_to_token = {v: k for k, v in token_to_idx.items()}
    return Vocab(
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        pad_index=0,
        unk_index=1,
    )


def save_vocab(vocab: Vocab, path: Path) -> None:
    """Persist a :class:`Vocab` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_dict(), indent=2), encoding="utf-8")


def load_vocab(path: Path) -> Vocab:
    """Load a :class:`Vocab` from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Vocab.from_dict(data)
