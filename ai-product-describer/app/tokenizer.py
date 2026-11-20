"""Milestone 2: a byte-level byte-pair-encoding (BPE) tokenizer, from scratch.

How it works
------------
* Text is turned into UTF-8 bytes, so the starting vocabulary is the 256
  possible byte values and *any* input can be encoded (no unknown token).
* Training repeatedly finds the most frequent adjacent pair of token ids in
  the corpus and replaces it with a brand-new id ("merge"). After enough
  merges, common pieces such as ``" candle"`` or ``"wood"`` become a single
  token.
* Like GPT-2, text is first pre-split into word-ish chunks with a regex, so
  merges never glue the end of one word to the start of the next. Counting
  unique chunks (with their frequency) keeps training fast in pure Python.
* Special tokens (``<|product|>`` etc.) are never split: they are cut out of
  the text before byte-encoding and get their own reserved ids.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.core import SPECIAL_TOKENS

# Contractions | words | numbers | punctuation runs | whitespace
PRE_SPLIT = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def _pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter:
    counts: Counter = Counter()
    for ids, freq in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts


def _merge(ids: tuple[int, ...] | list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 special_tokens: tuple[str, ...] = SPECIAL_TOKENS):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair in merges or []:
            self._add_merge(pair)
        self.special: dict[str, int] = {}
        for tok in special_tokens:
            self.special[tok] = 256 + len(self.merges) + len(self.special)
        self._special_re = re.compile("(" + "|".join(map(re.escape, self.special)) + ")") \
            if self.special else None
        self._cache: dict[str, list[int]] = {}

    # ----------------------------------------------------------------- train
    def _add_merge(self, pair: tuple[int, int]) -> int:
        new_id = 256 + len(self.merges)
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return new_id

    @classmethod
    def train(cls, text: str, vocab_size: int = 1024, verbose: bool = False,
              special_tokens: tuple[str, ...] = SPECIAL_TOKENS) -> "BPETokenizer":
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size too small for 256 bytes + special tokens")
        for tok in special_tokens:              # specials are not learned from
            text = text.replace(tok, "\n")
        words = Counter(PRE_SPLIT.findall(text))
        chunks = {tuple(w.encode("utf-8")): f for w, f in words.items()}

        tok = cls(special_tokens=())
        for step in range(n_merges):
            counts = _pair_counts(chunks)
            if not counts:
                break
            pair, freq = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if freq < 2:
                break
            new_id = tok._add_merge(pair)
            merged: dict[tuple[int, ...], int] = {}
            for ids, f in chunks.items():
                key = tuple(_merge(ids, pair, new_id)) if pair[0] in ids else ids
                merged[key] = merged.get(key, 0) + f
            chunks = merged
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}: {tok.vocab[new_id]!r} (x{freq})")
        return cls(list(tok.merges), special_tokens=special_tokens)

    # ---------------------------------------------------------- encode/decode
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge available, exactly like training
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 50_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        parts = self._special_re.split(text) if self._special_re else [text]
        out: list[int] = []
        for part in parts:
            if part in self.special:
                out.append(self.special[part])
            elif part:
                for chunk in PRE_SPLIT.findall(part):
                    out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.special.items()}
        buf, out = b"", []
        for i in ids:
            if i in inv:
                out.append(buf.decode("utf-8", errors="replace"))
                out.append(inv[i])
                buf = b""
            else:
                buf += self.vocab[i]
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def token_strings(self, text: str) -> list[str]:
        """Human-readable pieces, handy for seeing what BPE learned."""
        inv = {v: k for k, v in self.special.items()}
        return [inv.get(i) or self.vocab[i].decode("utf-8", errors="replace")
                for i in self.encode(text)]

    # ------------------------------------------------------------- save/load
    def save(self, path: str | Path) -> None:
        data = {"merges": [list(p) for p in self.merges], "special_tokens": list(self.special)}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(p) for p in data["merges"]], tuple(data["special_tokens"]))
