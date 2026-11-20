"""Shared product model and the text format the GPT is trained on.

Every training example (and every prompt at generation time) looks like:

    <|product|>name: Midnight Fig | notes: black fig, cedarwood | size: 10 oz | length: medium
    <|description|>Ripe black fig meets dry cedarwood ...<|end|>

The header conditions the model on the product details; the text after
``<|description|>`` is what it learns to write. ``<|end|>`` tells generation
where to stop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PRODUCT = "<|product|>"
DESCRIPTION = "<|description|>"
END = "<|end|>"
SPECIAL_TOKENS = (PRODUCT, DESCRIPTION, END)

LENGTHS = ("short", "medium", "long")

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


@dataclass
class Product:
    name: str
    notes: list[str] = field(default_factory=list)
    size: str = ""
    length: str = "medium"
    description: str = ""

    def header(self) -> str:
        parts = [f"name: {self.name.strip()}"]
        notes = [n.strip() for n in self.notes if n.strip()]
        if notes:
            parts.append("notes: " + ", ".join(notes))
        if self.size:
            parts.append(f"size: {self.size.strip()}")
        parts.append(f"length: {self.length}")
        return PRODUCT + " | ".join(parts)

    def prompt(self) -> str:
        """Text the model is asked to continue at generation time."""
        return f"{self.header()}\n{DESCRIPTION}"

    def to_record(self) -> str:
        """One full training example."""
        return f"{self.prompt()}{self.description.strip()}{END}"


def count_sentences(text: str) -> int:
    return max(1, len(_SENTENCE_END.findall(text.strip())))


def length_bucket(text: str) -> str:
    n = count_sentences(text)
    if n <= 2:
        return "short"
    if n <= 4:
        return "medium"
    return "long"


def parse_notes(text: str) -> list[str]:
    """'lavender, vanilla & musk' -> ['lavender', 'vanilla', 'musk']."""
    parts = re.split(r",|&|\band\b|;|/", text)
    return [p.strip().lower() for p in parts if p.strip()]


def parse_record(record: str) -> Product:
    """Inverse of Product.to_record."""
    header, _, rest = record.partition("\n")
    if not header.startswith(PRODUCT):
        raise ValueError("record does not start with the product token")
    fields = {}
    for part in header[len(PRODUCT):].split(" | "):
        key, _, value = part.partition(": ")
        fields[key.strip()] = value.strip()
    desc = rest.removeprefix(DESCRIPTION).removesuffix(END)
    return Product(
        name=fields.get("name", ""),
        notes=parse_notes(fields.get("notes", "")),
        size=fields.get("size", ""),
        length=fields.get("length", "medium"),
        description=desc,
    )


def finish_text(text: str) -> str:
    """Cut generated text at the first special token and trim it back to the
    last complete sentence, so the user never sees a half-finished one."""
    for tok in SPECIAL_TOKENS:
        text = text.split(tok)[0]
    text = re.sub(r"\s+", " ", text).strip()
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last > 0:
        text = text[: last + 1]
    return text
