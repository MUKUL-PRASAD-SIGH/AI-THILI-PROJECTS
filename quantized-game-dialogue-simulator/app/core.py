"""Dialogue format, character tokenizer and train / validation split."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROLES = ("BARD", "BLACKSMITH", "ELDER", "FERRYMAN", "GUARD", "HEALER", "INNKEEPER", "MERCHANT")
PLAYER = "PLAYER"


@dataclass(frozen=True)
class Exchange:
    role: str
    player: str
    reply: str

    def text(self) -> str:
        return f"{PLAYER}: {self.player}\n{self.role}: {self.reply}\n"


def parse_corpus(text: str) -> list[Exchange]:
    """Parse blank-line separated blocks of ``PLAYER: ...`` + ``ROLE: ...``; malformed blocks are skipped."""
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) != 2:
            continue
        (who1, _, said1), (who2, _, said2) = (l.partition(":") for l in lines)
        if who1.strip() != PLAYER or not who2.strip() or not said1.strip() or not said2.strip():
            continue
        out.append(Exchange(who2.strip(), said1.strip(), said2.strip()))
    return out


def format_corpus(exchanges: list[Exchange]) -> str:
    return "\n".join(e.text() for e in exchanges)


def format_prompt(role: str, player_line: str) -> str:
    """What the model is given at generation time; it continues with the NPC's reply and a newline."""
    return f"{PLAYER}: {player_line.strip()}\n{role.upper()}:"


def split_exchanges(exchanges: list[Exchange], val_frac: float = 0.1, seed: int = 0) -> tuple[str, str]:
    """Shuffle whole exchanges (never cut one in half) and return (train_text, val_text)."""
    items = list(exchanges)
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * val_frac))
    return format_corpus(items[n_val:]), format_corpus(items[:n_val])


class CharTokenizer:
    """One token per character. Tiny vocabulary (~70), so the embedding table stays small."""

    def __init__(self, chars: list[str]):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted({c for c in text if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in the vocabulary: {''.join(unknown)!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])
